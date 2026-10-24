"""The amplitude spectrum: a record's lines scaled so a sine reads its
amplitude in peak units, the window's correction applied and stated,
the frames' magnitudes averaged.

Pinned against numbers that do not depend on the implementation: the
coherent gain of a periodic Hann window is its mean, one half, so its
amplitude correction is 2; scipy's flat top's is its first coefficient,
0.21557895 (a correction of 4.639); a sine midway between two lines
reads 0.849 of itself under Hann (1.42 dB low, Harris's scalloping
loss), 2/pi under a rectangle (3.92 dB) and within 0.01 dB under the
flat top. The worked case is a 1 g peak tone at 500 Hz sampled at 5,120
a second in 1,024-sample frames (5 Hz lines), which reads 1.000 g peak
on its line under every window, and the same tone moved to 502.5 Hz,
halfway between lines. A tone given a fresh phase in each of 20
captures reads its amplitude here and averages away in
`compute_spectra`'s complex mean, which is left as it was. A random
floor's height goes as one over the root of the frame length, which is
why it is not a level.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.amplitude_spectrum import (
    AmplitudeSpectrum,
    amplitude_correction,
    amplitude_lines,
    amplitude_spectrum,
    channel_row,
    coherent_gain,
    scalloping,
)
from visualdynamics.core.averaging import WINDOWS, Averaging
from visualdynamics.core.data import TimeHistory

G = 9.80665
RATE = 5120.0
FRAME = 1024


def history(*rows, rate=RATE, dofs=None):
    """A time history in g, stored in SI as every record is."""
    rows = np.vstack(rows) * G
    return TimeHistory(np.arange(rows.shape[1]) / rate, rows,
                       response_dof=dofs or [f'{k + 1}Z+'
                                             for k in range(len(rows))],
                       ordinate_dim=['acceleration'] * len(rows),
                       ordinate_unit=['g'] * len(rows))


def tone(frequency, amplitude=1.0, seconds=12.0, phase=0.3, rate=RATE):
    t = np.arange(round(seconds * rate)) / rate
    return amplitude * np.sin(2.0 * np.pi * frequency * t + phase)


def framed(record, frames=119, window='boxcar'):
    """`record` with an averaging of 1,024-sample frames at half
    overlap; the window given is the averaging's, which the amplitude
    spectrum does not use."""
    record.averaging = Averaging(frame_length=FRAME, overlap=0.5,
                                 window=window, frames=frames)
    return record


# ---- the window's arithmetic ------------------------------------------

def test_the_coherent_gain_is_the_windows_mean():
    assert coherent_gain('hann') == pytest.approx(0.5, abs=1e-15)
    assert coherent_gain('boxcar') == 1.0
    assert coherent_gain('flattop') == pytest.approx(0.21557895, abs=1e-12)
    assert coherent_gain('hamming') == pytest.approx(0.54, abs=1e-15)


def test_the_amplitude_correction_is_its_reciprocal():
    assert amplitude_correction('hann') == pytest.approx(2.0, abs=1e-14)
    assert amplitude_correction('flattop') == pytest.approx(4.6387, abs=1e-4)
    assert amplitude_correction('rectangle') == 1.0


def test_the_arithmetic_takes_an_averaging():
    averaging = Averaging(frame_length=512, window='flattop')
    assert coherent_gain(averaging) == coherent_gain('flattop', 512)
    assert amplitude_correction(averaging) == pytest.approx(
        1.0 / 0.21557895, rel=1e-12)
    # the averaging's own frame length, which a Kaiser's mean depends on
    short = Averaging(frame_length=16, window='kaiser')
    assert coherent_gain(short) == coherent_gain('kaiser', 16, 14.0)
    assert coherent_gain(short) != coherent_gain('kaiser')


def test_the_scalloping_is_harriss():
    assert scalloping('hann') == pytest.approx(0.8488, abs=1e-4)
    assert -20.0 * np.log10(scalloping('hann')) == pytest.approx(1.42,
                                                                  abs=0.005)
    assert scalloping('boxcar') == pytest.approx(2.0 / np.pi, abs=1e-6)
    assert -20.0 * np.log10(scalloping('flattop')) < 0.01


# ---- a tone reads its amplitude ----------------------------------------

@pytest.mark.parametrize('window', WINDOWS)
def test_a_sine_on_a_line_reads_its_amplitude_under_every_window(window):
    record = framed(history(tone(500.0, 1.0)))
    spectrum = record.amplitude_spectrum(window=window)
    frequency, amplitude = spectrum.tallest(0)
    assert frequency == 500.0
    assert amplitude / G == pytest.approx(1.0, rel=1e-4)
    assert spectrum.window == window


def test_the_lines_are_a_sine_amplitude_not_its_transform():
    """The raw transform reads A*N/2 under a rectangle and A*sum(w)/2
    under a window; the scaling takes both to A."""
    x = tone(500.0, 2.5)[:FRAME]
    shape = np.hanning(FRAME + 1)[:-1]
    lines = amplitude_lines(x * shape, shape)
    raw = np.abs(np.fft.rfft(x * shape))
    assert raw[100] == pytest.approx(2.5 * shape.sum() / 2.0, rel=1e-12)
    assert lines[100] == pytest.approx(2.5, rel=1e-12)


def test_a_window_that_sums_to_nothing_is_refused():
    with pytest.raises(ValueError, match='sums to nothing'):
        amplitude_lines(np.ones((2, 8)), np.zeros(8))


@pytest.mark.parametrize('window,reads', [('hann', 0.8488),
                                          ('boxcar', 0.6366),
                                          ('flattop', 0.9989)])
def test_a_sine_midway_between_lines_reads_low_by_the_windows_scalloping(
        window, reads):
    record = framed(history(tone(502.5, 1.0)))
    spectrum = record.amplitude_spectrum(window=window)
    _frequency, amplitude = spectrum.tallest(0)
    assert amplitude / G == pytest.approx(reads, abs=2e-3)
    # and the reading states the loss it shows
    assert amplitude / G == pytest.approx(spectrum.scalloping, abs=2e-3)


def test_the_flat_top_reads_a_tone_wherever_it_falls():
    for frequency in (500.0, 501.25, 502.5, 503.75):
        spectrum = framed(history(tone(frequency, 1.0))).amplitude_spectrum(
            window='flattop')
        assert spectrum.tallest(0)[1] / G == pytest.approx(1.0, abs=2e-3)


def test_the_dc_and_nyquist_lines_are_not_doubled():
    n = 8 * FRAME
    record = framed(history(np.full(n, 0.75)), frames=15)
    assert record.amplitude_spectrum().amplitude[0, 0] / G == \
        pytest.approx(0.75, rel=1e-12)
    alternating = 0.4 * (-1.0) ** np.arange(n)
    record = framed(history(alternating), frames=15)
    spectrum = record.amplitude_spectrum(window='boxcar')
    assert spectrum.frequencies[-1] == RATE / 2.0
    assert spectrum.amplitude[0, -1] / G == pytest.approx(0.4, rel=1e-12)


# ---- magnitudes, not a complex mean -------------------------------------

def test_a_tone_with_a_fresh_phase_each_capture_keeps_its_amplitude():
    """20 captures of one channel, each its own frame, the tone's phase
    new in each: the magnitudes average to the tone, and
    `compute_spectra`'s complex mean, unchanged, shrinks it."""
    phases = np.random.default_rng(15).uniform(0.0, 2.0 * np.pi, 20)
    rows = [tone(500.0, 1.0, seconds=FRAME / RATE, phase=p) for p in phases]
    record = history(*rows, dofs=['1Z+'] * 20)
    spectrum = record.amplitude_spectrum(window='boxcar')
    assert spectrum.frames == (20,)
    assert not spectrum.from_averaging
    assert spectrum.amplitude.shape[0] == 1
    assert spectrum.tallest(0) == (500.0, pytest.approx(G, rel=1e-12))
    raw = record.compute_spectra()
    line = int(np.argmin(np.abs(raw.abscissa - 500.0)))
    complex_mean = abs(raw.ordinate[0, line])
    # the raw spectrum is still unscaled and a complex mean: one
    # capture alone reads A*N/2 there, and twenty at random phases far
    # less
    assert abs(history(rows[0]).compute_spectra().ordinate[0, line]) == \
        pytest.approx(G * FRAME / 2.0, rel=1e-12)
    assert complex_mean < 0.5 * G * FRAME / 2.0
    expected = abs(np.mean(np.exp(1j * (phases - np.pi / 2.0))))
    assert complex_mean == pytest.approx(G * FRAME / 2.0 * expected,
                                         rel=1e-9)


def test_a_random_floor_is_not_a_level():
    """The mean magnitude of a white floor goes as one over the root of
    the frame length: four times the lines, half the floor, while the
    tone on it reads the same."""
    noise = np.random.default_rng(8).standard_normal(round(RATE * 48.0))
    x = noise + tone(500.0, 2.0, seconds=48.0)
    floors, tones = [], []
    for block in (FRAME, 4 * FRAME):
        record = history(x)
        record.averaging = Averaging(frame_length=block, overlap=0.5,
                                     window='hann').filled(len(x), RATE)
        spectrum = record.amplitude_spectrum()
        f = spectrum.frequencies
        away = (f > 1000.0) & (f < 2000.0)
        floors.append(float(np.mean(spectrum.amplitude[0, away])) / G)
        tones.append(spectrum.tallest(0)[1] / G)
        # a Rayleigh magnitude's mean, through the window's sums:
        # 2*sqrt(pi/4)*sqrt(sum w^2)/sum w for unit variance
        w = np.hanning(block + 1)[:-1]
        expected = np.sqrt(np.pi) * np.sqrt(np.sum(w ** 2)) / np.sum(w)
        assert floors[-1] == pytest.approx(expected, rel=0.03)
    assert floors[0] / floors[1] == pytest.approx(2.0, rel=0.03)
    assert tones == [pytest.approx(2.0, abs=0.03)] * 2


# ---- the frames are the averaging's, the window is its own ----------------

def test_the_frames_follow_the_averaging_and_not_its_window():
    record = framed(history(tone(500.0)), frames=40, window='boxcar')
    spectrum = record.amplitude_spectrum()
    assert spectrum.window == 'hann'
    assert spectrum.frames == (40,)
    assert spectrum.frame_length == FRAME
    assert spectrum.spacing == 5.0
    assert spectrum.from_averaging
    assert len(spectrum.frequencies) == FRAME // 2 + 1
    # the record's own averaging is left as it was
    assert record.averaging.window == 'boxcar'


def test_an_averaging_given_overrides_the_records():
    record = framed(history(tone(500.0)), frames=40)
    other = Averaging(frame_length=4 * FRAME, overlap=0.0, frames=3)
    spectrum = record.amplitude_spectrum(other)
    assert spectrum.frames == (3,)
    assert spectrum.spacing == 1.25


def test_without_an_averaging_each_record_is_one_frame():
    record = history(tone(500.0, seconds=2.0))
    spectrum = record.amplitude_spectrum()
    assert spectrum.frames == (1,)
    assert not spectrum.from_averaging
    assert spectrum.spacing == pytest.approx(0.5)
    assert spectrum.describe().endswith('1 frame averaged as magnitudes')
    assert spectrum.tallest() == spectrum.tallest(0)


def test_captures_pool_into_their_channels_row():
    a, b = tone(500.0, 1.0), tone(800.0, 3.0)
    record = history(a, b, a, b, dofs=['1Z+', '2Z+', '1Z+', '2Z+'])
    framed(record, frames=10)
    spectrum = record.amplitude_spectrum()
    assert spectrum.frames == (20, 20)
    assert [channel_row(record, i) for i in range(4)] == [0, 1, 0, 1]
    assert spectrum.tallest(channel_row(record, 3))[0] == 800.0
    assert spectrum.tallest(1)[1] / G == pytest.approx(3.0, rel=1e-6)


def test_a_parameterized_window_carries_its_number():
    record = framed(history(tone(500.0)))
    assert record.amplitude_spectrum(window='tukey').window_parameter == 0.5
    spectrum = record.amplitude_spectrum(window='kaiser', parameter=6.0)
    assert spectrum.window_parameter == 6.0
    assert spectrum.correction_text() == (
        f'kaiser (6) window, amplitude correction '
        f'×{amplitude_correction("kaiser", FRAME, 6.0):.3f}')


def test_the_reading_states_its_window_and_correction():
    record = framed(history(tone(500.0, 1.0)))
    spectrum = amplitude_spectrum(record)
    assert isinstance(spectrum, AmplitudeSpectrum)
    assert spectrum.correction == pytest.approx(2.0)
    assert spectrum.coherent_gain == pytest.approx(0.5)
    assert spectrum.noise_bandwidth == pytest.approx(1.5, abs=1e-9)
    assert spectrum.scalloping_db == pytest.approx(1.4236, abs=1e-3)
    assert spectrum.correction_text() == (
        'hann window, amplitude correction ×2.000')
    assert spectrum.describe(0, 'g', 1.0 / G) == (
        '1Z+: tallest line 1 g peak at 500 Hz; hann window, amplitude '
        'correction ×2.000, 119 frames averaged as magnitudes')


def test_the_tallest_line_leaves_out_dc_and_keeps_to_its_band():
    record = framed(history(5.0 + tone(500.0, 1.0) + tone(1200.0, 0.5)))
    spectrum = record.amplitude_spectrum()
    assert spectrum.amplitude[0, 0] / G == pytest.approx(5.0, rel=1e-9)
    # the mean leaks into the next line up under Hann, as tall again;
    # the line is read with the frames' means taken out, so the leak is
    # not taken for a tone
    assert spectrum.amplitude[0, 1] / G == pytest.approx(5.0, rel=1e-6)
    assert spectrum.leveled[0, 1] / G < 1e-6
    assert spectrum.tallest(0)[0] == 500.0
    assert spectrum.tallest(0, band=(1000.0, 1500.0))[0] == 1200.0
    # the band's edges are in it
    assert spectrum.tallest(0, band=(1200.0, 1200.0))[0] == 1200.0


@pytest.mark.parametrize('window, frequency', [
    ('hann', 5.0), ('hann', 10.0), ('flattop', 15.0), ('boxcar', 5.0)])
@pytest.mark.parametrize('mean', [0.0, 5.0])
def test_a_low_tone_is_read_whatever_the_window_s_lobe(window, frequency,
                                                       mean):
    """A tone on line 1 under Hann or line 3 under the flat top, inside
    the main lobe about DC, reads its own line and amplitude, with or
    without a mean under it. At 5120 Hz and 1024-sample frames a 5 Hz
    tone once read (10 Hz, 0.50) under Hann and (25 Hz, 0.016) under
    the flat top (review, 2026-10-23). Line 3 is the flat top's lowest:
    below it a tone's own mirror at negative frequency, 2k lines away,
    is inside the main lobe's five lines and the two add, which no
    reading undoes."""
    record = framed(history(mean + tone(frequency, 1.0)))
    spectrum = record.amplitude_spectrum(window=window)
    line, amplitude = spectrum.tallest(0)
    assert line == frequency
    assert amplitude / G == pytest.approx(1.0, rel=1e-6)


def test_a_line_with_no_window_reads_its_own_magnitude_not_the_mean():
    """Leveled frames are the averaging's when it levels them: no second
    pass, and the reading is the drawn line."""
    record = framed(history(5.0 + tone(500.0, 1.0)))
    record.averaging = Averaging(frame_length=FRAME, overlap=0.5,
                                 window='hann', frames=119, detrend='mean')
    spectrum = record.amplitude_spectrum()
    assert np.array_equal(spectrum.leveled, spectrum.amplitude)
    assert spectrum.amplitude[0, 0] / G < 1e-9


def test_a_dead_channel_is_named_not_read():
    zero = framed(history(np.zeros(round(12.0 * RATE))))
    spectrum = zero.amplitude_spectrum()
    with pytest.raises(ValueError, match='zero throughout'):
        spectrum.tallest(0)
    with pytest.raises(ValueError, match='zero throughout'):
        spectrum.describe(0)
    lost = tone(500.0)
    lost[100] = np.nan
    spectrum = framed(history(lost)).amplitude_spectrum()
    with pytest.raises(ValueError, match='not finite'):
        spectrum.tallest(0)


def test_a_constant_channel_is_named_not_read():
    """A constant 0.3 g, its means taken out, leaves lines of 1e-17 g,
    which once read as the tallest line."""
    spectrum = framed(history(np.full(round(12.0 * RATE), 0.3))
                      ).amplitude_spectrum()
    with pytest.raises(ValueError, match='constant throughout'):
        spectrum.tallest(0)
    with pytest.raises(ValueError, match='constant throughout'):
        spectrum.describe(0, 'g', 1.0 / G)


@pytest.mark.parametrize('window, lowest', [
    ('hann', 1), ('boxcar', 1), ('flattop', 3)])
def test_the_lowest_line_read_is_the_window_s(window, lowest):
    spectrum = framed(history(tone(500.0))).amplitude_spectrum(window=window)
    assert spectrum.lowest_line == lowest


def test_a_tone_below_the_flat_top_s_lowest_line_is_flagged():
    """Under the flat top a 5 Hz tone on 5 Hz lines reads anywhere from
    (10 Hz, 0.77 g) to (5 Hz, 1.64 g) with its phase: the reading says
    so, and names the window that reads it."""
    low = framed(history(tone(5.0)))
    said = low.amplitude_spectrum(window='flattop').describe(0, 'g', 1 / G)
    assert ('below 15 Hz the flattop window mixes a tone with its mirror '
            'at negative frequency; read it under hann') in said
    assert 'mixes' not in low.amplitude_spectrum().describe(0, 'g', 1 / G)
    clear = framed(history(tone(15.0)))
    said = clear.amplitude_spectrum(window='flattop').describe(0, 'g', 1 / G)
    assert 'mixes' not in said


def test_a_band_with_no_line_says_so():
    spectrum = framed(history(tone(10.0, 0.5))).amplitude_spectrum()
    with pytest.raises(ValueError, match='no line'):
        spectrum.tallest(0, band=(3000.0, 4000.0))
    with pytest.raises(ValueError, match='no line'):
        spectrum.tallest(0, band=(1.0, 2.0))


def test_the_arithmetic_on_the_spectrum_is_the_module_s():
    record = framed(history(tone(500.0)))
    for window in ('hann', 'flattop', 'kaiser'):
        spectrum = record.amplitude_spectrum(window=window)
        assert spectrum.coherent_gain == coherent_gain(window, FRAME)
        assert spectrum.scalloping == scalloping(window, FRAME)


def test_the_plot_takes_a_record_of_the_history(qt_app, tmp_path):
    """`record` is the history's record, as every plot of a history
    takes it: the fourth record of a capture-saved history is its
    second channel, drawn as that channel's row."""
    a, b = tone(500.0, 1.0), tone(800.0, 3.0)
    record = history(a, b, a, b, dofs=['1Z+', '2Z+', '1Z+', '2Z+'])
    framed(record, frames=10)
    paths = []
    for index in (1, 3):
        path = tmp_path / f'record{index}.png'
        visualdynamics.plot.plot_amplitude_spectrum(
            record, index, path=str(path), show=False)
        paths.append(path.read_bytes())
    assert paths[0] == paths[1]
    with pytest.raises(ValueError, match='record 4'):
        visualdynamics.plot.plot_amplitude_spectrum(
            record, 4, path=str(tmp_path / 'never.png'), show=False)


def test_frames_that_do_not_fit_are_refused():
    record = framed(history(tone(500.0, seconds=1.0)), frames=119)
    with pytest.raises(ValueError, match='do not fit'):
        record.amplitude_spectrum()


def test_an_unknown_window_is_refused():
    with pytest.raises(ValueError, match='unknown window'):
        framed(history(tone(500.0))).amplitude_spectrum(window='triangle')


# ---- the view ----------------------------------------------------------

def test_the_view_names_the_window_and_marks_the_tallest_line(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.amplitude_spectrum import (
        build_amplitude_spectrum,
        title_text,
    )

    spectrum = framed(history(tone(502.5, 1.0))).amplitude_spectrum(
        window='flattop')
    widget = pg.GraphicsLayoutWidget()
    tallest = build_amplitude_spectrum(widget, spectrum, 0)
    plot = widget.getItem(0, 0)
    title = plot.titleLabel.text
    assert 'flattop window, amplitude correction ×4.639' in title
    assert '5 Hz lines, 119 frames averaged as magnitudes' in title
    assert '0.01 dB low' in title
    assert title_text(spectrum, 0) in title
    names = [label.text for _sample, label in plot.legend.items]
    assert names[0] == '1Z+, peak per line'
    assert names[1] == f'tallest line: {tallest[1] / G:.4g} g peak at 505 Hz'
    marker = [item for item in plot.listDataItems()
              if item.opts.get('symbol') == 't']
    assert len(marker) == 1
    assert plot.getViewBox().state['logMode'][1]
    assert 'g peak' in plot.getAxis('left').labelText
    widget.close()


def test_the_view_draws_to_a_file(qt_app, tmp_path):
    from visualdynamics.plot import plot_amplitude_spectrum

    path = tmp_path / 'amplitude.png'
    plot_amplitude_spectrum(framed(history(tone(500.0))), 0,
                            window='flattop', path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


# ---- in the app --------------------------------------------------------

def _amplitude(window, pump, record, name='Run'):
    name = window.add_object(name, record)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    pane = window.data_pane
    if not pane.amplitude_action.isChecked():
        pane.amplitude_action.trigger()
    pump()
    return pane


def test_a_time_history_offers_the_amplitude_reading(window, pump):
    pane = _amplitude(window, pump, framed(history(tone(502.5, 1.0))))
    assert pane.amplitude_action.isVisible()
    assert pane.showing('amplitude')
    panel = pane.amplitude_panel
    assert not panel.isHidden()
    rows = {key: label.text() for key, label in panel.derived.items()}
    assert rows['channel'] == '1Z+'
    assert rows['source'] == 'the averaging'
    assert rows['spacing'] == '5 Hz'
    assert rows['frames'] == '119'
    assert rows['gain'] == '0.5000'
    assert rows['correction'] == '×2.000'
    assert rows['bandwidth'] == '1.5 lines'
    assert rows['scalloping'] == '1.42 dB low'
    assert rows['frequency'] == '505 Hz'
    assert rows['amplitude'] == '0.8488 g peak'
    assert rows['rms'] == '0.6002 g rms'
    assert 'up to 1.42 dB low under hann' in panel.notes.text()
    assert 'not a level' in panel.notes.text()
    line = window.project.journal[-1]
    assert line == ("visualdynamics.plot.plot_amplitude_spectrum("
                    "project['Run'], 0, window='hann', "
                    "path='amplitude_spectrum.png', show=False)")


def test_the_flat_top_chosen_in_the_panel_reads_the_tone(window, pump):
    pane = _amplitude(window, pump, framed(history(tone(502.5, 1.0))))
    panel = pane.amplitude_panel
    panel.window_box.setCurrentIndex(panel.window_box.findData('flattop'))
    pump()
    assert panel.derived['correction'].text() == '×4.639'
    assert panel.derived['amplitude'].text() == '0.9989 g peak'
    assert panel.derived['scalloping'].text() == '0.01 dB low'
    assert 'between lines' not in panel.notes.text()
    assert "window='flattop'" in window.project.journal[-1]
    plot = pane.graphics.getItem(0, 0)
    assert 'flattop window, amplitude correction ×4.639' in \
        plot.titleLabel.text


def test_the_window_survives_a_redraw(window, pump):
    pane = _amplitude(window, pump, framed(history(tone(500.0))))
    panel = pane.amplitude_panel
    panel.window_box.setCurrentIndex(panel.window_box.findData('kaiser'))
    pump()
    assert not panel.parameter_box.isHidden()
    panel.parameter_box.setValue(6.0)
    pump()
    window.render_current()
    pump()
    assert panel.window() == 'kaiser'
    assert panel.settings() == {'window': 'kaiser', 'parameter': 6.0}
    assert "window='kaiser', parameter=6.0" in window.project.journal[-1]


def test_a_record_with_no_averaging_is_read_whole(window, pump):
    pane = _amplitude(window, pump, history(tone(500.0, 2.0, seconds=2.0)))
    panel = pane.amplitude_panel
    assert panel.derived['source'].text() == 'each record whole'
    assert panel.derived['frames'].text() == '1'
    assert panel.derived['spacing'].text() == '0.5 Hz'
    assert panel.derived['amplitude'].text() == '2 g peak'


def test_the_second_channel_selected_is_the_one_read(window, pump):
    record = framed(history(tone(500.0, 1.0), tone(800.0, 3.0)))
    name = window.add_object('Two', record)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    window._select_records(name, [1])
    window.data_pane.amplitude_action.trigger()
    pump()
    window.render_current()
    pump()
    panel = window.data_pane.amplitude_panel
    assert panel.derived['channel'].text() == '2Z+'
    assert panel.derived['frequency'].text() == '800 Hz'
    assert panel.derived['amplitude'].text() == '3 g peak'


def test_a_capture_is_journaled_as_its_record(window, pump):
    """The journaled call takes the history's record, which the plot
    maps to its channel's row: a capture-saved record replays."""
    a, b = tone(500.0, 1.0), tone(800.0, 3.0)
    record = framed(history(a, b, a, b, dofs=['1Z+', '2Z+', '1Z+', '2Z+']),
                    frames=10)
    name = window.add_object('Captures', record)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    window._select_records(name, [3])
    window.data_pane.amplitude_action.trigger()
    pump()
    window.render_current()
    pump()
    panel = window.data_pane.amplitude_panel
    assert panel.derived['channel'].text() == '2Z+'
    line = window.project.journal[-1]
    assert "project['Captures'], 3, window='hann'" in line


def test_a_dead_channel_is_named_in_the_pane(window, pump):
    pane = _amplitude(window, pump,
                      framed(history(np.zeros(round(12.0 * RATE)))))
    panel = pane.amplitude_panel
    assert panel.derived['amplitude'].text() == '—'
    assert 'zero throughout' in panel.notes.text()
    assert 'zero throughout' in window.statusBar().currentMessage()


def test_a_low_tone_under_the_flat_top_is_flagged_in_the_pane(window, pump):
    pane = _amplitude(window, pump, framed(history(tone(5.0))))
    panel = pane.amplitude_panel
    assert 'mixes' not in panel.notes.text()
    panel.window_box.setCurrentIndex(panel.window_box.findData('flattop'))
    pump()
    assert ('Below 15 Hz the flattop window mixes a tone with its mirror '
            'at negative frequency; read it under hann.') in \
        panel.notes.text()
    assert 'read it under hann' in window.statusBar().currentMessage()


def test_frames_that_do_not_fit_clear_the_reading(window, pump):
    record = framed(history(tone(500.0, seconds=12.0)))
    pane = _amplitude(window, pump, record)
    record.averaging = Averaging(frame_length=FRAME, overlap=0.5,
                                 frames=100000)
    window.render_current()
    pump()
    assert pane.amplitude_panel.derived['amplitude'].text() == '—'


def test_the_amplitude_reading_is_offered_where_the_averaging_is(window):
    pane = window.data_pane
    controls = {'map_wanted': None, 'diagonal': None, 'cmif': False,
                'complex_data': False, 'pair': False}
    pane.show_controls(**controls, averaging=False, shocks=True)
    assert not pane.amplitude_action.isVisible()
    pane.show_controls(**controls, averaging=True, shocks=True)
    assert pane.amplitude_action.isVisible()


def test_the_amplitude_reading_is_flat(window, pump):
    pane = _amplitude(window, pump, framed(history(tone(500.0))))
    assert not pane.waterfall_action.isVisible()
