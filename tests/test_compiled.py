"""The compiled kernels in `vd_rust` against the Python they stand in for.

Each kernel has a Python implementation the package falls back to when
the module is not installed, and the two must agree: the SRS filter
bank and the scalogram's peaks to round-off, and the faer factor's
solves and the modes they give to the accuracy the modal tests already
hold SuperLU to. The whole file is skipped without the module.
"""

from __future__ import annotations

import numpy as np
import pytest

vd_rust = pytest.importorskip('vd_rust')

from visualdynamics.core import srs, wavelet
from visualdynamics.core.sparse_solve import (
    BACKEND_VARIABLE,
    FaerFactor,
    SuperLUFactor,
    factor_symmetric,
)

RATE = 4096.0


def _python(monkeypatch, module):
    monkeypatch.setattr(module, '_compiled_peaks', None)


def test_the_compiled_srs_is_lfilters_to_round_off(monkeypatch):
    """A shock and noise through a twelfth-octave bank, the positive and
    negative peaks each within a part in 10^12 of the scipy filter's."""
    rng = np.random.default_rng(3)
    t = np.arange(20_000) / RATE
    record = rng.normal(size=t.size) + 40.0 * np.sin(2 * np.pi * 60 * t) * (t < 0.011)
    frequencies = srs.octave_frequencies(5.0, 2000.0)
    compiled = srs.peaks(record, frequencies, RATE)
    _python(monkeypatch, srs)
    python = srs.peaks(record, frequencies, RATE)
    for got, want in zip(compiled, python, strict=True):
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_a_nan_sample_makes_every_compiled_srs_peak_nan(monkeypatch):
    record = np.random.default_rng(4).normal(size=4000)
    record[1000] = np.nan
    frequencies = srs.octave_frequencies(20.0, 1000.0)
    highest, lowest = srs.peaks(record, frequencies, RATE)
    _python(monkeypatch, srs)
    want_high, want_low = srs.peaks(record, frequencies, RATE)
    assert np.isnan(want_high).all() and np.isnan(want_low).all()
    assert np.isnan(highest).all() and np.isnan(lowest).all()


def test_an_empty_record_has_zero_peaks_either_way():
    highest, lowest = srs.peaks(np.array([]), np.array([100.0, 200.0]), RATE)
    assert np.array_equal(highest, [0.0, 0.0]) and np.array_equal(lowest, [0.0, 0.0])


@pytest.mark.parametrize('samples', [10_000, 10_007])
def test_the_compiled_scalogram_peaks_are_the_pythons(samples, monkeypatch):
    """A chirp and noise, a record that divides into the columns and one
    whose last slice is ragged: the same times and, to round-off, the
    same magnitudes."""
    rng = np.random.default_rng(5)
    t = np.arange(samples) / RATE
    record = np.sin(2 * np.pi * (20 + 200 * t) * t) + 0.1 * rng.normal(size=samples)
    frequencies = wavelet.log_frequencies(10.0, 800.0)
    times, compiled = wavelet.scalogram_peaks(record, RATE, frequencies, columns=700)
    _python(monkeypatch, wavelet)
    want_times, python = wavelet.scalogram_peaks(record, RATE, frequencies, columns=700)
    assert np.array_equal(times, want_times)
    assert compiled.shape == python.shape
    assert np.allclose(compiled, python, rtol=1e-12, atol=1e-13 * python.max())


def test_a_nan_sample_reaches_the_compiled_scalogram_as_the_pythons(monkeypatch):
    record = np.random.default_rng(6).normal(size=3000)
    record[1500] = np.nan
    frequencies = wavelet.log_frequencies(20.0, 1000.0)
    _times, compiled = wavelet.scalogram_peaks(record, RATE, frequencies)
    _python(monkeypatch, wavelet)
    _times, python = wavelet.scalogram_peaks(record, RATE, frequencies)
    assert np.array_equal(np.isnan(compiled), np.isnan(python))
    assert np.isnan(compiled).any()


def _shifted_plate(mesh=16):
    from visualdynamics.demo import plate

    stiffness, mass, *_rest = plate.build(mesh).scaled_system()
    return (stiffness + (2.0 * np.pi) ** 2 * mass).tocsc()


def test_the_faer_factor_solves_as_superlu_does():
    matrix = _shifted_plate()
    rhs = np.random.default_rng(7).normal(size=(matrix.shape[0], 12))
    faer, superlu = FaerFactor(matrix), SuperLUFactor(matrix)
    assert faer.shape == matrix.shape
    got, want = faer.solve(rhs), superlu.solve(rhs)
    # each solver's residual is what conditioning allows, and the two
    # answers agree as a whole; entry by entry the small ones carry the
    # conditioning's share of round-off differently in each
    for x in (got, want):
        assert np.linalg.norm(matrix @ x - rhs) / np.linalg.norm(rhs) < 1e-6
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-6
    assert faer.solve(rhs[:, 0]).shape == (matrix.shape[0],), 'one column in, one out'


def test_an_indefinite_matrix_goes_past_faer_to_superlu(monkeypatch):
    """faer's is a Cholesky, so a matrix that is not positive definite is
    refused by it and factored by SuperLU instead."""
    monkeypatch.setenv(BACKEND_VARIABLE, 'faer')
    matrix = _shifted_plate()
    indefinite = (matrix - 1.5 * matrix.diagonal().max() * _identity(matrix)).tocsc()
    with pytest.raises(ValueError):
        FaerFactor(indefinite)
    assert factor_symmetric(indefinite).backend == 'superlu'
    assert factor_symmetric(matrix).backend == 'faer'


def _identity(matrix):
    import scipy.sparse as sp

    return sp.identity(matrix.shape[0], format='csc')


def test_the_modes_through_faer_are_the_modes_through_superlu(monkeypatch):
    from visualdynamics.demo import plate

    model = plate.build(20)
    monkeypatch.setenv(BACKEND_VARIABLE, 'superlu')
    superlu = model.eigensolution(num_modes=40, solver='sparse')
    monkeypatch.setenv(BACKEND_VARIABLE, 'faer')
    faer = model.eigensolution(num_modes=40, solver='sparse')
    assert np.array_equal(faer.frequency == 0.0, superlu.frequency == 0.0)
    assert np.allclose(faer.frequency, superlu.frequency, rtol=1e-8, atol=1e-6)
