"""The transmissibility reading in the window: on an FRF computed from a
time history, the ASD ratio drawn dashed beside H1 with the band H1's
error gives, a cursor, and the reading at its line in the pane.

A reading of an FRF that exists, so a toggle on the bar with no act (as
the tones are), offered for an FRF drawn as its magnitude whose history
is in the project, against one reference, and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Frf, TimeHistory

G = 9.80665
FS = 5120.0


def _record(seconds=8.0):
    """Random from 20 to 2000 Hz on 1Z+, through a 300 Hz resonance at
    Q 20 to 101Z+ with 0.6 g rms of noise the input did not cause,
    stored in m/s², 1Z+ the reference."""
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum = np.fft.rfft(np.random.default_rng(1505).standard_normal(n))
    spectrum[(f < 20.0) | (f > 2000.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x *= np.sqrt(0.04 * 1980.0) / np.std(x)
    r, z = f / 300.0, 1.0 / 40.0
    y = np.fft.irfft(np.fft.rfft(x) * (1 + 2j * z * r)
                     / (1 - r ** 2 + 2j * z * r), n)
    y += np.random.default_rng(1506).standard_normal(n) * 0.6
    history = TimeHistory(np.arange(n) / FS, np.vstack([x, y]) * G,
                          response_dof=['1Z+', '101Z+'],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['g'] * 2)
    history.roles = {('1Z+', 'acceleration'): 'reference'}
    history.averaging = Averaging(frame_length=2048, overlap=0.5,
                                  window='hann').filled(n, FS)
    return history


def _frf(window, pump, method='H1'):
    window.add_object('Run', _record())
    name = window.project.compute_frfs('Run', method)
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    return name, pane


def _reading_on(window, pump, method='H1'):
    name, pane = _frf(window, pump, method)
    pane.transmissibility_action.trigger()
    pump()
    return name, pane


def test_the_reading_is_offered_on_an_frf_with_a_history_behind_it(window,
                                                                   pump):
    name, pane = _frf(window, pump)
    assert pane.transmissibility_action.isVisible()
    # the same numbers imported: no autospectra behind them to divide
    frf = window.project[name]
    window.add_object('Imported', Frf(frf.abscissa, frf.ordinate,
                                      response_dof=list(frf.response_dof),
                                      reference_dof=list(frf.reference_dof),
                                      ordinate_dim=list(frf.ordinate_dim)))
    window.show_object('Imported')
    pump()
    window.render_current()
    pump()
    assert not pane.transmissibility_action.isVisible()
    # and a spectrum of the history is not an FRF at all
    window.show_object(window.project.compute_psds('Run'))
    pump()
    window.render_current()
    pump()
    assert not pane.transmissibility_action.isVisible()


def test_the_reading_is_offered_on_the_magnitude_alone(window, pump):
    _name, pane = _frf(window, pump)
    box = pane.component_box
    box.setCurrentIndex(box.findData('phase'))
    pump()
    window.render_current()
    pump()
    assert not pane.transmissibility_action.isVisible(), \
        'the band is on a magnitude; a phase has none'
    box.setCurrentIndex(box.findData('magnitude'))
    pump()
    window.render_current()
    pump()
    assert pane.transmissibility_action.isVisible()


def test_the_toggle_draws_the_ratio_dashed_and_the_band(window, pump):
    from PySide6.QtCore import Qt

    name, pane = _reading_on(window, pump)
    assert pane.transmissibility_panel.isVisible()
    state = window._transmissibility
    assert state is not None and state['name'] == name
    overlay, reading = state['overlay'], state['reading']
    assert overlay.ratio.opts['pen'].style() == Qt.PenStyle.DashLine
    assert overlay.upper.opts['pen'].style() == Qt.PenStyle.DotLine
    assert overlay.lower.opts['pen'].style() == Qt.PenStyle.DotLine
    # the FRF on the plot is H1, so the overlay does not draw it again
    assert not overlay.h1.isVisible()
    # g/g on the plot is the SI ratio, one for one
    _x, y = overlay.ratio.getOriginalDataset()
    np.testing.assert_allclose(y, reading.asd_ratio)
    words = [label.text for _sample, label in overlay.key.items]
    assert words[0] == 'H1 = Gxy/Gxx'
    assert words[1].startswith('ASD ratio')
    assert words[2].startswith('H1 ± 1 random error')


def test_the_cursor_starts_on_the_resonance(window, pump):
    _name, pane = _reading_on(window, pump)
    point = window._transmissibility['point']
    assert point.frequency == pytest.approx(300.0, abs=5.0)
    derived = pane.transmissibility_panel.derived
    assert derived['channel'].text() == '101Z+ over 1Z+'
    assert derived['estimator'].text() == 'H1'
    assert derived['h1'].text() == f'{point.magnitude:.4g} g/g'
    assert derived['coherence'].text() == f'{point.coherence:.3f}'


def test_moving_the_cursor_reads_the_line_under_it(window, pump):
    _name, pane = _reading_on(window, pump)
    state = window._transmissibility
    cursor = state['cursor']
    cursor.setValue(np.log10(1500.0) if state['log_x'] else 1500.0)
    pump()
    point = window._transmissibility['point']
    assert point.frequency == pytest.approx(1500.0, abs=2.5)
    derived = pane.transmissibility_panel.derived
    assert derived['frequency'].text() == f'{point.frequency:.4g} Hz'
    assert derived['over'].text() == f'{point.overstatement_db:+.2f} dB'
    assert point.overstatement_db > 1.0, 'the noise lifts the ratio here'
    assert derived['error'].text() == f'±{100 * point.error:.1f} %'
    marks = state['overlay']
    assert marks.ratio_mark.getOriginalDataset()[1][0] == pytest.approx(
        point.asd_ratio)
    assert marks.h1_mark.getOriginalDataset()[1][0] == pytest.approx(
        point.magnitude)
    assert window.statusBar().currentMessage() == \
        '101Z+ over 1Z+, ' + point.describe('g/g', 1.0)
    # and where the cursor was left is kept across a redraw
    window.render_current()
    pump()
    assert window._transmissibility['point'].frequency == point.frequency


def test_an_hv_frf_has_h1_drawn_beside_it_and_says_so(window, pump):
    _name, pane = _reading_on(window, pump, method='Hv')
    overlay = window._transmissibility['overlay']
    assert overlay.h1.isVisible()
    panel = pane.transmissibility_panel
    assert panel.derived['estimator'].text() == 'Hv'
    assert 'The FRF drawn is Hv' in panel.notes.text()


def test_the_stage_stands_down_while_the_reading_is_up(window, pump):
    _name, pane = _reading_on(window, pump)
    pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert window._transmissibility is not None
    assert not (pane._waterfall_page is not None
                and pane._waterfall_page.isVisible())


def test_putting_the_reading_away_takes_it_off(window, pump):
    _name, pane = _reading_on(window, pump)
    overlay = window._transmissibility['overlay']
    pane.transmissibility_action.trigger()
    pump()
    assert window._transmissibility is None
    assert not pane.transmissibility_panel.isVisible()
    assert overlay.ratio.getViewBox() is None


def test_the_reading_is_journaled_as_its_plot_call(window, pump):
    name, _pane = _reading_on(window, pump)
    lines = [line for line in window.project.journal if line.startswith(
        'visualdynamics.plot.plot_transmissibility(')]
    assert lines == [(
        'visualdynamics.plot.plot_transmissibility('
        f"project.read_transmissibility({name!r}, 0), "
        "path='transmissibility.png', show=False)")]
    window.render_current()
    pump()
    assert len([line for line in window.project.journal if line.startswith(
        'visualdynamics.plot.plot_transmissibility(')]) == 1
