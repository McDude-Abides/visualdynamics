"""The ASD-ratio transmissibility beside H1, and the random error of H1.

The ratio is the root of the response's autospectrum over the input's;
H1 is the cross-spectrum over the input's autospectrum. Pinned against
hand values of the random-error expression, against the identity that
ties the two estimates to the coherence, against a synthetic survey with
a known answer (a 300 Hz resonance at Q 20 driven by band-limited random,
with uncorrelated noise on the response channel), and against the
package's own FRF and coherence from the same frames. Then the project's
reading behind an FRF and the plot.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.transmissibility import (
    SMALL_ERROR,
    TransmissibilityPoint,
    frf_random_error,
    read_transmissibility,
)

G = 9.80665
FS = 5120.0
#: the resonance the survey passes through
FN, Q = 300.0, 20.0


def truth(f):
    """The base-excited resonator's transmissibility at `f`, the answer
    the survey is read against."""
    r = np.asarray(f, dtype=float) / FN
    z = 1.0 / (2.0 * Q)
    return (1.0 + 2j * z * r) / (1.0 - r ** 2 + 2j * z * r)


def survey(seconds=24.0, noise=0.6, overlap=0.5, force=False):
    """Band-limited random from 20 to 2000 Hz at 0.04 g²/Hz on the input,
    the response through the resonance plus `noise` g rms of noise the
    input did not cause, stored in m/s². The response is the input's
    spectrum times the resonance, on a record that repeats, so every
    frame holds the steady answer and `truth` is exact at every line.
    With `force` the input is a force in newtons, the response an
    acceleration over it."""
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum = np.fft.rfft(np.random.default_rng(1505).standard_normal(n))
    spectrum[(f < 20.0) | (f > 2000.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x *= np.sqrt(0.04 * 1980.0) / np.std(x)
    y = np.fft.irfft(np.fft.rfft(x) * truth(f), n)
    y += np.random.default_rng(1506).standard_normal(n) * noise
    t = np.arange(n) / FS
    history = TimeHistory(
        t, np.vstack([x, y]) * G, response_dof=['1Z+', '101Z+'],
        ordinate_dim=['force' if force else 'acceleration', 'acceleration'],
        ordinate_unit=['N' if force else 'g', 'g'])
    history.averaging = Averaging(frame_length=2048, overlap=overlap,
                                  window='hann').filled(n, FS)
    return history


@pytest.fixture(scope='module')
def history():
    return survey()


@pytest.fixture(scope='module')
def reading(history):
    return history.read_transmissibility('101Z+', '1Z+')


# ---- the error expression, by hand --------------------------------------

@pytest.mark.parametrize('coherence,averages,expected', [
    (0.9, 60, 0.0304), (0.5, 60, 0.0913), (0.99, 60, 0.00917),
    (0.9, 10, 0.0745)])
def test_the_random_error_matches_the_hand_values(coherence, averages,
                                                  expected):
    assert float(frf_random_error(coherence, averages)) == pytest.approx(
        expected, rel=2e-3)


def test_the_error_goes_as_one_over_the_root_of_the_averages():
    four = frf_random_error(0.8, 25.0)
    assert float(frf_random_error(0.8, 100.0)) == pytest.approx(four / 2.0)


def test_the_error_s_two_ends_are_zero_and_unbounded():
    error = frf_random_error(np.array([1.0, 0.0, np.nan]), 10)
    assert error[0] == 0.0
    assert np.isinf(error[1])
    assert np.isnan(error[2])


def test_the_error_refuses_what_is_not_a_coherence_or_an_average():
    with pytest.raises(ValueError, match='between 0 and 1'):
        frf_random_error(1.2, 10)
    with pytest.raises(ValueError, match='at least one average'):
        frf_random_error(0.5, 0)


# ---- the two estimates, tied by the coherence ----------------------------

def test_the_ratio_is_h1_over_the_root_of_the_coherence(reading):
    good = np.isfinite(reading.coherence) & (reading.coherence > 0.0)
    expected = reading.magnitude[good] / np.sqrt(reading.coherence[good])
    np.testing.assert_allclose(reading.asd_ratio[good], expected,
                               rtol=1e-12)
    over = reading.overstatement_db[good]
    np.testing.assert_allclose(over, -10.0 * np.log10(
        reading.coherence[good]), rtol=1e-9, atol=1e-9)


def test_the_ratio_never_reads_under_h1(reading):
    good = np.isfinite(reading.coherence) & (reading.coherence > 0.0)
    assert np.all(reading.asd_ratio[good]
                  >= reading.magnitude[good] * (1.0 - 1e-12))


def test_h1_is_the_package_s_h1_and_the_coherence_its_coherence(history,
                                                                reading):
    frf = history.compute_frfs(references=['1Z+'], method='H1')
    coherence = history.compute_multiple_coherence(references=['1Z+'])
    np.testing.assert_allclose(reading.h1, frf.ordinate[0], rtol=1e-12)
    np.testing.assert_allclose(reading.coherence, coherence.ordinate[0],
                               rtol=1e-12, atol=1e-15)


def test_the_ratio_is_the_root_of_the_two_psds(history, reading):
    psd = history.compute_psds()
    expected = np.sqrt(psd.ordinate[1] / psd.ordinate[0])
    np.testing.assert_allclose(reading.asd_ratio, expected, rtol=1e-12)


# ---- the survey: where they agree and where the ratio overstates --------

def test_where_the_coherence_is_high_the_two_agree_with_the_truth(reading):
    point = reading.at(100.0)
    assert point.coherence > 0.99
    assert point.asd_ratio == pytest.approx(point.magnitude, rel=0.005)
    assert point.magnitude == pytest.approx(abs(truth(point.frequency)),
                                            rel=0.02)


def test_where_the_coherence_falls_the_ratio_reads_high(reading):
    f = reading.frequency
    low = (f > 1200.0) & (f < 1900.0) & (reading.coherence < 0.5)
    assert low.sum() > 20
    over_truth = reading.asd_ratio[low] / np.abs(truth(f[low]))
    h1_over_truth = reading.magnitude[low] / np.abs(truth(f[low]))
    # the noise counted as response lifts the ratio by over 40 percent
    # there; H1 scatters about the truth
    assert np.median(over_truth) > 1.4
    assert np.median(h1_over_truth) == pytest.approx(1.0, abs=0.1)


def test_h1_lies_inside_its_band_about_as_often_as_one_error_says(reading):
    # away from the resonance, where the 2.5 Hz lines are too coarse for
    # the peak and the bias is the resolution's, not the noise's; and
    # where the error is small enough for its expression to hold
    f = reading.frequency
    lower, upper = reading.band(1.0)
    real = np.abs(truth(f))
    clean = ((f > 500.0) & (f < 1900.0)
             & (reading.error <= SMALL_ERROR))
    inside = (real >= lower) & (real <= upper)
    share = inside[clean].mean()
    assert clean.sum() > 300
    # one standard deviation holds about 68 percent of a normal error
    assert 0.55 < share < 0.80
    lower2, upper2 = reading.band(2.0)
    share2 = ((real >= lower2) & (real <= upper2))[clean].mean()
    assert 0.88 < share2 <= 1.0


def test_the_error_is_read_at_the_averages_the_frames_are_worth(history,
                                                                reading):
    worth = history.averaging.statistical_dof()
    assert reading.frames == history.averaging.frames == 119
    assert reading.averages == pytest.approx(worth.equivalent / 2.0)
    assert reading.averages < reading.frames
    np.testing.assert_allclose(
        reading.error, frf_random_error(reading.coherence, worth.equivalent
                                        / 2.0))
    assert any('worth' in note for note in reading.notes)


def test_frames_that_do_not_overlap_are_their_own_count():
    reading = survey(seconds=12.0, overlap=0.0).read_transmissibility(
        '101Z+', '1Z+')
    assert reading.averages == reading.frames
    assert not any('worth' in note for note in reading.notes)


def test_the_point_reads_the_nearest_line(reading):
    point = reading.at(1001.0)
    k = reading.line(1001.0)
    assert point.frequency == reading.frequency[k] == 1000.0
    assert point.h1 == reading.h1[k]
    assert point.asd_ratio == reading.asd_ratio[k]
    assert point.error == pytest.approx(reading.error[k])
    lower, upper = reading.band()
    assert point.lower == pytest.approx(lower[k])
    assert point.upper == pytest.approx(upper[k])


def test_h1_s_phase_swings_through_a_quarter_cycle_at_the_mode(reading):
    # the base-excited resonator's phase at its natural frequency is
    # minus the angle whose tangent is 1 / (2 zeta), just short of 90
    expected = np.degrees(np.angle(truth(FN)))
    assert reading.at(FN).phase == pytest.approx(expected, abs=4.0)
    assert reading.at(100.0).phase == pytest.approx(0.0, abs=2.0)


def test_the_band_has_no_lower_edge_where_it_reaches_zero():
    f = np.array([10.0, 20.0, 30.0])
    gxx = np.ones(3)
    gyy = np.ones(3)
    # coherences of 0.81, 0.01 and 1/9: the second's error is past one
    gxy = np.array([0.9, 0.1, 1.0 / 3.0]) + 0j
    reading = read_transmissibility(f, gxx, gyy, gxy, frames=4)
    lower, upper = reading.band(1.0)
    assert np.isfinite(lower[0]) and np.isnan(lower[1])
    assert np.isfinite(upper[1])
    assert reading.at(20.0).lower is None
    assert reading.at(10.0).lower == pytest.approx(lower[0])
    assert not reading.at(20.0).small
    assert any('poor rather than how poor' in note for note in reading.notes)


def test_a_line_with_no_input_has_no_reading():
    f = np.array([0.0, 10.0])
    reading = read_transmissibility(f, [0.0, 1.0], [1.0, 1.0],
                                    [0.0, 0.5 + 0j], frames=8)
    assert np.isnan(reading.h1[0]) and np.isnan(reading.asd_ratio[0])
    assert np.isnan(reading.coherence[0])
    assert reading.coherence[1] == pytest.approx(0.25)
    assert any('no input power' in note for note in reading.notes)


def test_bare_arrays_read_like_a_record(history, reading):
    psd = history.compute_psds()
    frequency = reading.frequency
    gxy = reading.h1 * psd.ordinate[0]
    again = read_transmissibility(frequency, psd.ordinate[0],
                                  psd.ordinate[1], gxy, frames=119,
                                  averages=reading.averages)
    np.testing.assert_allclose(again.asd_ratio, reading.asd_ratio,
                               rtol=1e-12)
    np.testing.assert_allclose(again.coherence, reading.coherence,
                               rtol=1e-9)


def test_what_is_not_an_estimate_is_refused():
    with pytest.raises(ValueError, match='two frames'):
        read_transmissibility([1.0], [1.0], [1.0], [1.0 + 0j], frames=1)
    with pytest.raises(ValueError, match='same length'):
        read_transmissibility([1.0, 2.0], [1.0], [1.0], [1.0 + 0j],
                              frames=4)
    with pytest.raises(ValueError, match='more than zero'):
        read_transmissibility([1.0], [1.0], [1.0], [1.0 + 0j], frames=4,
                              averages=0.0)


def test_a_reading_is_one_response_over_one_reference(history):
    with pytest.raises(ValueError, match='one reference'):
        history.read_transmissibility('101Z+')     # no channel's role is
    with pytest.raises(ValueError, match='not a response'):
        history.read_transmissibility('1Z+', '1Z+')
    with pytest.raises(ValueError, match='no channel matches'):
        history.read_transmissibility('101Z+', '7Z+')


def test_a_force_reference_gives_a_ratio_in_acceleration_over_force():
    reading = survey(seconds=6.0, force=True).read_transmissibility('101Z+')
    assert reading.reference == '1Z+'
    assert reading.dimension == 'acceleration/force'


def test_the_point_describes_itself_in_display_units(reading):
    text = reading.at(1000.0).describe('g/g', 1.0)
    assert text.startswith('1000 Hz: H1 ')
    assert 'coherence 0.' in text and 'dB over H1' in text
    assert TransmissibilityPoint(1.0, 0j, 0.0, 0.0, np.inf, 1.0,
                                 4.0).describe().count('unbounded') == 1


# ---- the project and the plot -------------------------------------------

def _project(method='H1'):
    from visualdynamics import Project

    project = Project()
    record = survey(seconds=8.0)
    record.roles = {('1Z+', 'acceleration'): 'reference'}
    project.add('Run', record)
    return project, project.compute_frfs('Run', method)


@pytest.mark.parametrize('method', ['H1', 'Hv'])
def test_the_project_reads_behind_an_frf_from_its_frames(method):
    project, name = _project(method)
    assert project.averaging_of(name) == project['Run'].averaging
    reading = project.read_transmissibility(name)
    direct = project['Run'].read_transmissibility('101Z+', '1Z+')
    np.testing.assert_allclose(reading.h1, direct.h1, rtol=1e-12)
    assert (reading.response, reading.reference) == ('101Z+', '1Z+')
    if method == 'H1':
        np.testing.assert_allclose(reading.h1, project[name].ordinate[0],
                                   rtol=1e-12)


def test_the_project_reads_with_the_averaging_the_frf_was_made_with():
    project, name = _project()
    made = project['Run'].averaging
    # moved since: the FRF is stale, and its reading is still of its own
    # frames
    project['Run'].averaging = Averaging(frame_length=512, overlap=0.5,
                                         window='hann').filled(
        len(project['Run'].abscissa), FS)
    reading = project.read_transmissibility(name)
    assert reading.frames == made.frames
    assert reading.frequency[1] == pytest.approx(FS / 2048)


def test_the_project_refuses_what_has_no_autospectra_behind_it():
    from visualdynamics.core.data import Frf

    project, name = _project()
    frf = project[name]
    # the same numbers brought in from a file: no history behind them
    project.add('Copy', Frf(frf.abscissa, frf.ordinate,
                            response_dof=list(frf.response_dof),
                            reference_dof=list(frf.reference_dof),
                            ordinate_dim=list(frf.ordinate_dim)))
    with pytest.raises(ValueError, match='not an FRF computed'):
        project.read_transmissibility('Copy')
    with pytest.raises(ValueError, match='not an FRF computed'):
        project.read_transmissibility('Run')


def test_the_project_refuses_an_frf_of_several_references():
    from visualdynamics import Project

    record = survey(seconds=8.0)
    third = TimeHistory(record.abscissa,
                        np.vstack([record.ordinate,
                                   np.roll(record.ordinate[0], 977)]),
                        response_dof=['1Z+', '101Z+', '2Z+'],
                        ordinate_dim=['acceleration'] * 3,
                        ordinate_unit=['g'] * 3)
    third.averaging = record.averaging
    third.roles = {('1Z+', 'acceleration'): 'reference',
                   ('2Z+', 'acceleration'): 'reference'}
    project = Project()
    project.add('Run', third)
    name = project.compute_frfs('Run', 'H1')
    with pytest.raises(ValueError, match='2 references'):
        project.read_transmissibility(name)


def test_the_plot_draws_both_estimates_told_apart_by_style(reading, qt_app,
                                                          tmp_path):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot import plot_transmissibility
    from visualdynamics.plot.transmissibility import (
        H1_NAME,
        RATIO_NAME,
        build_transmissibility,
    )

    layout = pg.GraphicsLayoutWidget()
    top, bottom = build_transmissibility(layout, reading, band=(20, 2000))
    styles = {item.name(): item.opts['pen'].style()
              for item in top.listDataItems() if item.name()}
    assert styles[H1_NAME] == Qt.PenStyle.SolidLine
    assert styles[RATIO_NAME] == Qt.PenStyle.DashLine
    band = next(name for name in styles if name.startswith('H1 ± 1 random'))
    assert styles[band] == Qt.PenStyle.DotLine
    assert '112.8 averages' in band
    assert bottom.getAxis('left').labelText.startswith('coherence')
    low, high = top.getViewBox().viewRange()[0]
    assert (low, high) == pytest.approx((np.log10(20), np.log10(2000)))
    path = plot_transmissibility(reading, path=str(tmp_path / 't.png'),
                                 show=False)
    assert (tmp_path / 't.png').stat().st_size > 1000
    assert str(path).endswith('t.png')


def test_the_plot_band_refuses_a_zero_start(reading, qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.transmissibility import build_transmissibility

    with pytest.raises(ValueError, match='above zero'):
        build_transmissibility(pg.GraphicsLayoutWidget(), reading,
                               band=(0, 100))
