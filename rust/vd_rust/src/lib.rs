//! Compiled kernels for Visual Dynamics.
//!
//! Each function here has a Python implementation it must agree with,
//! and the package calls the compiled one only when this module imports.
//! Every call releases the interpreter lock for its numeric work, so a
//! kernel uses the cores it is given and the window stays responsive.

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::{Mat, Side};
use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

/// The Cholesky factorization L Lᵀ of a sparse symmetric positive
/// definite matrix, factored once and solved many times.
///
/// Built from the matrix in compressed sparse column form (scipy's
/// `indptr`, `indices`, `data`); only its lower triangle is read. The
/// ordering and the supernodal factorization are faer's, and both the
/// factorization and the solves run on every core.
#[pyclass(module = "vd_rust", frozen)]
struct SparseCholesky {
    factor: Llt<usize, f64>,
    size: usize,
}

#[pymethods]
impl SparseCholesky {
    #[new]
    fn new(
        py: Python<'_>,
        size: usize,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let to_usize = |values: &[i64]| -> PyResult<Vec<usize>> {
            values
                .iter()
                .map(|&v| usize::try_from(v).map_err(|_| PyValueError::new_err("a negative index")))
                .collect()
        };
        let col_ptr = to_usize(indptr.as_slice()?)?;
        let row_idx = to_usize(indices.as_slice()?)?;
        let values = data.as_slice()?.to_vec();
        if col_ptr.len() != size + 1 || row_idx.len() != values.len() {
            return Err(PyValueError::new_err("the arrays are not a square CSC matrix"));
        }
        if row_idx.iter().any(|&r| r >= size) || col_ptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(PyValueError::new_err("the arrays are not a square CSC matrix"));
        }
        let factor = py
            .detach(|| -> Result<Llt<usize, f64>, String> {
                // faer's checked view requires each column's rows in
                // ascending order, which scipy guarantees only after
                // sort_indices(), so they are sorted here
                let mut row_idx = row_idx;
                let mut values = values;
                for c in 0..size {
                    let (lo, hi) = (col_ptr[c], col_ptr[c + 1]);
                    let mut pairs: Vec<(usize, f64)> =
                        row_idx[lo..hi].iter().copied().zip(values[lo..hi].iter().copied()).collect();
                    pairs.sort_by_key(|p| p.0);
                    for (k, (r, v)) in pairs.into_iter().enumerate() {
                        row_idx[lo + k] = r;
                        values[lo + k] = v;
                    }
                }
                let symbolic =
                    SymbolicSparseColMatRef::new_checked(size, size, &col_ptr, None, &row_idx);
                let matrix = SparseColMatRef::new(symbolic, &values);
                let pattern = SymbolicLlt::try_new(symbolic, Side::Lower).map_err(|e| format!("{e:?}"))?;
                Llt::try_new_with_symbolic(pattern, matrix, Side::Lower).map_err(|e| format!("{e:?}"))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("the matrix could not be factored: {e}")))?;
        Ok(Self { factor, size })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.size, self.size)
    }

    /// X with A X = `rhs`, for one right-hand side or a block of them,
    /// in the shape `rhs` came in.
    fn solve_block<'py>(&self, py: Python<'py>, rhs: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = rhs.as_array();
        let (rows, cols) = view.dim();
        if rows != self.size {
            return Err(PyValueError::new_err(format!(
                "a right-hand side of {rows} rows for a matrix of {}",
                self.size
            )));
        }
        let mut block = Mat::<f64>::from_fn(rows, cols, |i, j| view[[i, j]]);
        let block = py.detach(|| {
            self.factor.solve_in_place(block.as_mut());
            block
        });
        let out = Array2::from_shape_fn((rows, cols).f(), |(i, j)| block[(i, j)]);
        Ok(out.into_pyarray(py))
    }
}

/// The highest and lowest response of a bank of second-order filters
/// run over one record, one pair per filter.
///
/// `b` and `a` are (filters, 3) coefficients with `a[:, 0] == 1`, as
/// `srs.ramp_invariant` designs them. Each filter is run in the
/// transposed direct form II that scipy's `lfilter` uses, so the
/// responses agree with it to round-off, and only the running extremes
/// are kept. Both start at zero. The filters are independent, so they
/// run in parallel, one per core.
#[pyfunction]
fn srs_peaks<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x = signal.as_slice()?.to_vec();
    let (b, a) = (b.as_array(), a.as_array());
    if b.dim().1 != 3 || a.dim() != b.dim() {
        return Err(PyValueError::new_err("b and a must both be (filters, 3)"));
    }
    let coefficients: Vec<[f64; 5]> = (0..b.dim().0)
        .map(|k| [b[[k, 0]], b[[k, 1]], b[[k, 2]], a[[k, 1]], a[[k, 2]]])
        .collect();
    let peaks: Vec<(f64, f64)> = py.detach(|| {
        coefficients
            .par_iter()
            .map(|&[b0, b1, b2, a1, a2]| {
                let (mut z1, mut z2) = (0.0f64, 0.0f64);
                let (mut high, mut low) = (0.0f64, 0.0f64);
                for &sample in &x {
                    let y = b0 * sample + z1;
                    z1 = b1 * sample - a1 * y + z2;
                    z2 = b2 * sample - a2 * y;
                    // NaN propagates as numpy's max does: a record with a
                    // NaN sample has a NaN peak, never a finite one
                    if y.is_nan() {
                        high = f64::NAN;
                        low = f64::NAN;
                    } else {
                        high = high.max(y);
                        low = low.min(y);
                    }
                }
                (high, low)
            })
            .collect()
    });
    let (high, low): (Vec<f64>, Vec<f64>) = peaks.into_iter().unzip();
    Ok((high.into_pyarray(py), low.into_pyarray(py)))
}

/// The scalogram's magnitude, each scale reduced to its largest value in
/// each slice of `step` samples, one row per scale.
///
/// `spectrum` is the padded record's FFT and `omega` the angular
/// frequency of each of its bins, as `wavelet._bands` computes them. At
/// each scale the analytic Morlet's Gaussian, times two, weights the
/// positive bins; the inverse FFT (rustfft's, normalized by the length
/// as scipy's is) is taken in a buffer of the padded length, and its
/// first `samples` values are reduced to `kept` slice maxima as they are
/// read, so no (scales, length) array is ever held. Scales run in
/// parallel, as many at once as `budget` bytes of buffers allow, at
/// least one.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn scalogram_peaks<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, numpy::Complex64>,
    omega: PyReadonlyArray1<'py, f64>,
    scales: PyReadonlyArray1<'py, f64>,
    omega0: f64,
    samples: usize,
    step: usize,
    kept: usize,
    budget: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    use rustfft::FftPlanner;
    use rustfft::num_complex::Complex64;

    let spectrum: Vec<Complex64> = spectrum.as_slice()?.iter().map(|c| Complex64::new(c.re, c.im)).collect();
    let omega = omega.as_slice()?.to_vec();
    let scales = scales.as_slice()?.to_vec();
    let length = spectrum.len();
    if omega.len() != length || samples > length || step == 0 || kept * step < samples {
        return Err(PyValueError::new_err("inconsistent scalogram sizes"));
    }
    let positive: Vec<usize> = (0..length).filter(|&j| omega[j] > 0.0).collect();
    let per_scale = 2 * length * std::mem::size_of::<Complex64>();
    let threads = (budget / per_scale).clamp(1, rayon::current_num_threads());
    let rows: Vec<Vec<f64>> = py.detach(|| {
        let fft = FftPlanner::<f64>::new().plan_fft_inverse(length);
        let scale = 1.0 / length as f64;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("a thread pool");
        pool.install(|| {
            scales
                .par_iter()
                .map_init(
                    || (vec![Complex64::new(0.0, 0.0); length], vec![Complex64::new(0.0, 0.0); fft.get_inplace_scratch_len()]),
                    |(buffer, scratch), &s| {
                        buffer.iter_mut().for_each(|c| *c = Complex64::new(0.0, 0.0));
                        for &j in &positive {
                            let d = s * omega[j] - omega0;
                            buffer[j] = spectrum[j] * (2.0 * (-0.5 * d * d).exp());
                        }
                        fft.process_with_scratch(buffer, scratch);
                        let mut row = vec![0.0f64; kept];
                        for (i, c) in buffer[..samples].iter().enumerate() {
                            let m = (c * scale).norm();
                            let slot = &mut row[i / step];
                            if m > *slot || m.is_nan() {
                                *slot = m;
                            }
                        }
                        row
                    },
                )
                .collect()
        })
    });
    let mut out = Array2::<f64>::zeros((scales.len(), kept));
    for (r, row) in rows.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            out[[r, c]] = *v;
        }
    }
    Ok(out.into_pyarray(py))
}

/// Set how many threads faer's factorizations and solves use: one runs
/// them sequentially, more on rayon's pool. Returns the count set.
#[pyfunction]
fn set_threads(threads: usize) -> usize {
    let threads = threads.max(1);
    faer::set_global_parallelism(if threads == 1 {
        faer::Par::Seq
    } else {
        faer::Par::rayon(threads)
    });
    threads
}

#[pymodule]
fn vd_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseCholesky>()?;
    m.add_function(wrap_pyfunction!(srs_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(scalogram_peaks, m)?)?;
    Ok(())
}
