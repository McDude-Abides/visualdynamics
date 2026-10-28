# Benchmarks for the Rust port

Three scripts, run with the repository's own virtual environment:

- `inventory.py` walks the package's AST and writes `out/inventory.csv`
  (every function and method: layer, size, loops, numpy/scipy use,
  whether its module imports Qt) and `out/modules.csv`.
- `baseline.py` times the numeric kernels across input sizes and writes
  `out/baseline-<commit>[-large][-t<threads>].csv` and `.json`.
  `--quick` runs the smaller sizes, `--large` adds the sizes in `LARGE`;
  names given on the command line select cases.
- `eigensolve.py` breaks the sparse eigensolution into its stages
  across model sizes and mode counts.

Machine: 12th Gen Intel(R) Core(TM) i7-12700 (20 threads), Python 3.14.8,
numpy 2.5.4, scipy 1.18.1, visualdynamics 0.1.0a74.

## BLAS threads

numpy and scipy each load their own OpenBLAS, and each starts one
thread per logical CPU unless `OPENBLAS_NUM_THREADS` says otherwise.
Code that alternates between the two libraries' BLAS calls has both
pools contending for the cores. The kernels below barely notice; the
sparse eigensolver, which alternates on every Lanczos step, runs 5 to
20 times slower with the default. Its time on the 10,086-DOF plate for
200 modes, by thread count:

| OPENBLAS_NUM_THREADS | 2 | 4 | 6 | 10 | 12 | 14 | 16 | 20 (default) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seconds | 1.85 | 1.70 | 1.74 | 2.32 | 8.61 | 14.08 | 16.13 | 19.94 |

## Kernels

Minimum of repeated calls after one warm-up call, swept to the `--large`
sizes. The exponent is a log-log fit over the larger half of the sizes.

| Case | Function | Varies | Held fixed | Largest | 4 threads | default | Exponent |
|---|---|---|---|---:|---:|---:|---:|
| srs.samples | `core.srs.peaks` | samples | 105 frequencies | 16,000,000 | 53.47 s | 53.69 s | 1.00 |
| srs.frequencies | `core.srs.peaks` | frequencies | 16384 samples | 10,000 | 0.61 s | 0.61 s | 0.62 |
| psd.samples | `TimeHistory.compute_psds` | samples | 8 channels, 1024 frame, 50% overlap | 16,777,216 | 1.30 s | 1.30 s | 1.06 |
| psd.channels | `TimeHistory.compute_psds` | channels | 262144 samples | 256 | 0.54 s | 0.53 s | 1.04 |
| cpsd.channels | `TimeHistory.compute_cpsds` | channels | 65536 samples | 128 | 2.28 s | 2.29 s | 1.98 |
| frf.Hv.responses | `TimeHistory.compute_frfs(Hv)` | responses | 2 references, 65536 samples | 256 | 0.46 s | 0.41 s | 1.07 |
| frf.H1.responses | `TimeHistory.compute_frfs(H1)` | responses | 2 references, 65536 samples | 256 | 0.48 s | 0.45 s | 1.06 |
| frf.Hv.samples | `TimeHistory.compute_frfs(Hv)` | samples | 8 responses, 2 references | 16,777,216 | 3.61 s | 3.60 s | 1.10 |
| filter.samples | `core.filters.filtered` | samples | 8 channels, low-pass 400 Hz order 4 | 16,777,216 | 0.52 s | 0.55 s | 1.03 |
| resample.samples | `core.resample.resample` | samples | 8 channels, rate / 4 | 16,777,216 | 1.06 s | 1.05 s | 0.98 |
| scalogram.samples | `core.wavelet.scalogram` | samples | 5-1000 Hz | 1,048,576 | 1.19 s | 1.20 s | 1.00 |
| scalogram_peaks.samples | `core.wavelet.scalogram_peaks` | samples | 5-1000 Hz | 16,777,216 | 50.95 s | 50.81 s | 1.21 |
| kurtosis.frames | `core.kurtosis.frame_kurtosis` | frames | 1024 per frame | 65,536 | 1.05 s | 1.05 s | 1.05 |
| time_statistics.samples | `core.time_statistics.time_statistics` | samples | defaults | 16,777,216 | 1.30 s | 1.30 s | 1.06 |
| mac.modes | `core.shapes.mac_matrix` | modes | 3000 DOFs | 3,000 | 0.19 s | 0.15 s | 1.69 |
| mac.dofs | `core.shapes.mac_matrix` | dofs | 50 modes | 1,000,000 | 0.18 s | 0.20 s | 1.12 |
| synthesize_frf.pairs | `ShapeSet.synthesize_frf` | pairs | 40 modes, 2048 lines | 10,000 | 0.13 s | 0.16 s | 0.93 |
| octave.lines | `core.octave.resample` | lines | 1/6 octave bands | 10,000,000 | 2.60 s | 2.60 s | 1.11 |

## Sparse eigensolution

`Model.eigensolution(num_modes=..., solver='sparse')` on the
demonstration plate meshed finer, seconds per stage, exclusive (a
stage called inside another is counted in itself only). *solves* are
SuperLU's solves with the shifted factor, from the Lanczos and the
polish; *lanczos* is the block Lanczos' own work outside its calls,
chiefly the reorthogonalization against the whole basis; *ritz eigh*
is the dense eigensolve of the projected matrix, which the Lanczos
repeats in full after every block.

With `OPENBLAS_NUM_THREADS=4`:

| DOF | modes | total | assembly | scaling | factor | solves | lanczos | orthonormalize | ritz eigh | polish | rigid check | other |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,646 | 50 | **0.17** | 0.03 | 0.00 | 0.01 | 0.03 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.01 |
| 2,646 | 100 | **0.26** | 0.03 | 0.00 | 0.01 | 0.05 | 0.03 | 0.03 | 0.07 | 0.02 | 0.03 | 0.01 |
| 2,646 | 200 | **0.64** | 0.03 | 0.00 | 0.01 | 0.09 | 0.08 | 0.05 | 0.31 | 0.04 | 0.02 | 0.01 |
| 2,646 | 400 | **2.31** | 0.03 | 0.01 | 0.01 | 0.18 | 0.28 | 0.08 | 1.58 | 0.10 | 0.02 | 0.02 |
| 5,766 | 50 | **0.28** | 0.03 | 0.01 | 0.02 | 0.08 | 0.03 | 0.03 | 0.02 | 0.02 | 0.03 | 0.01 |
| 5,766 | 100 | **0.47** | 0.03 | 0.01 | 0.02 | 0.14 | 0.06 | 0.05 | 0.08 | 0.04 | 0.03 | 0.01 |
| 5,766 | 200 | **1.07** | 0.03 | 0.01 | 0.02 | 0.25 | 0.16 | 0.09 | 0.36 | 0.10 | 0.03 | 0.02 |
| 5,766 | 400 | **3.43** | 0.03 | 0.01 | 0.02 | 0.46 | 0.52 | 0.15 | 1.94 | 0.23 | 0.03 | 0.05 |
| 10,086 | 50 | **0.47** | 0.05 | 0.01 | 0.04 | 0.16 | 0.05 | 0.06 | 0.02 | 0.04 | 0.04 | 0.01 |
| 10,086 | 100 | **0.80** | 0.05 | 0.01 | 0.04 | 0.28 | 0.11 | 0.09 | 0.08 | 0.07 | 0.04 | 0.02 |
| 10,086 | 200 | **1.62** | 0.05 | 0.01 | 0.04 | 0.50 | 0.27 | 0.15 | 0.34 | 0.18 | 0.04 | 0.04 |
| 10,086 | 400 | **4.38** | 0.05 | 0.01 | 0.04 | 0.94 | 0.81 | 0.26 | 1.74 | 0.42 | 0.04 | 0.08 |
| 22,326 | 50 | **1.19** | 0.09 | 0.03 | 0.12 | 0.48 | 0.12 | 0.14 | 0.03 | 0.09 | 0.06 | 0.03 |
| 22,326 | 100 | **1.86** | 0.10 | 0.03 | 0.12 | 0.78 | 0.24 | 0.21 | 0.08 | 0.17 | 0.06 | 0.06 |
| 22,326 | 200 | **3.64** | 0.09 | 0.03 | 0.12 | 1.43 | 0.62 | 0.35 | 0.37 | 0.47 | 0.06 | 0.10 |
| 22,326 | 400 | **9.00** | 0.09 | 0.03 | 0.12 | 2.76 | 1.87 | 0.63 | 2.23 | 1.02 | 0.06 | 0.20 |
| 39,366 | 50 | **2.37** | 0.17 | 0.06 | 0.28 | 0.99 | 0.23 | 0.25 | 0.03 | 0.16 | 0.15 | 0.06 |
| 39,366 | 100 | **3.62** | 0.17 | 0.06 | 0.28 | 1.62 | 0.45 | 0.38 | 0.08 | 0.37 | 0.12 | 0.10 |
| 39,366 | 200 | **6.81** | 0.17 | 0.06 | 0.28 | 3.01 | 1.11 | 0.64 | 0.38 | 0.84 | 0.11 | 0.21 |
| 39,366 | 400 | **15.88** | 0.17 | 0.06 | 0.27 | 5.97 | 3.43 | 1.21 | 2.48 | 1.79 | 0.11 | 0.38 |
| 58,806 | 50 | **3.50** | 0.27 | 0.10 | 0.40 | 1.50 | 0.35 | 0.35 | 0.03 | 0.24 | 0.18 | 0.08 |
| 58,806 | 100 | **5.48** | 0.27 | 0.08 | 0.40 | 2.54 | 0.69 | 0.54 | 0.09 | 0.54 | 0.17 | 0.15 |
| 58,806 | 200 | **10.42** | 0.27 | 0.08 | 0.41 | 4.79 | 1.79 | 0.95 | 0.44 | 1.22 | 0.18 | 0.29 |
| 58,806 | 400 | **22.56** | 0.27 | 0.08 | 0.40 | 9.12 | 5.12 | 1.70 | 2.49 | 2.64 | 0.17 | 0.57 |

With the default thread count (20 per library), the same cases:

| DOF | modes | total | assembly | scaling | factor | solves | lanczos | orthonormalize | ritz eigh | polish | rigid check | other |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,646 | 50 | **2.04** | 0.03 | 0.00 | 0.01 | 0.44 | 0.27 | 0.04 | 0.85 | 0.27 | 0.10 | 0.03 |
| 2,646 | 100 | **4.95** | 0.03 | 0.00 | 0.01 | 0.82 | 0.83 | 0.07 | 2.70 | 0.38 | 0.04 | 0.07 |
| 2,646 | 200 | **12.78** | 0.03 | 0.00 | 0.01 | 2.03 | 3.61 | 0.23 | 6.24 | 0.59 | 0.03 | 0.01 |
| 5,766 | 50 | **3.61** | 0.03 | 0.01 | 0.02 | 0.77 | 0.53 | 0.29 | 1.32 | 0.54 | 0.04 | 0.06 |
| 5,766 | 100 | **8.77** | 0.03 | 0.01 | 0.02 | 1.94 | 1.71 | 0.42 | 3.81 | 0.78 | 0.03 | 0.01 |
| 5,766 | 200 | **18.25** | 0.03 | 0.01 | 0.02 | 2.90 | 5.58 | 1.38 | 7.53 | 0.75 | 0.03 | 0.03 |
| 10,086 | 50 | **6.42** | 0.05 | 0.01 | 0.04 | 1.37 | 1.34 | 1.62 | 1.21 | 0.71 | 0.05 | 0.02 |
| 10,086 | 100 | **12.21** | 0.05 | 0.01 | 0.04 | 2.04 | 3.04 | 2.53 | 3.70 | 0.74 | 0.04 | 0.02 |
| 10,086 | 200 | **24.30** | 0.05 | 0.02 | 0.04 | 3.06 | 6.12 | 5.31 | 8.78 | 0.85 | 0.04 | 0.05 |
| 22,326 | 50 | **7.75** | 0.09 | 0.03 | 0.12 | 1.95 | 1.28 | 1.92 | 1.62 | 0.63 | 0.06 | 0.04 |
| 22,326 | 100 | **13.60** | 0.09 | 0.04 | 0.12 | 2.66 | 3.57 | 1.71 | 4.45 | 0.85 | 0.06 | 0.05 |
| 22,326 | 200 | **29.81** | 0.09 | 0.03 | 0.12 | 4.68 | 7.58 | 4.98 | 11.07 | 1.10 | 0.06 | 0.10 |

Asking for every mode up to a frequency (`maximum_frequency`, which is
what `Project.solve_modes` passes from the app) costs more than asking
for the same number of modes. The solver cannot know the count in
advance: it asks for 24, and while the wanted modes fall short of the
frequency it doubles the count and runs the whole Lanczos again from a
new start. 10,086-DOF plate, 4 threads:

| modes to the frequency | num_modes | maximum_frequency | Lanczos runs (modes asked, with the guard) |
|---:|---:|---:|---|
| 100 (20,656 Hz) | 0.81 s | 2.35 s | 32, 56, 104, 200 |
| 200 (36,236 Hz) | 1.62 s | 6.15 s | 32, 56, 104, 200, 392 |
| 400 (59,828 Hz) | 4.66 s | 24.88 s | 32, 56, 104, 200, 392, 776 |

