"""Resampling with an anti-alias filter, and the reading that shows it.

Pinned against arithmetic that does not depend on the implementation.
Content at F sampled at fs lands at |n fs - F|: a 4,600 Hz tone on a
channel sampled at 5,120 Hz lands at 520 Hz, and its samples are those
of a 520 Hz tone with its sign reversed. The alias-free span is the
rate over 2.56, so 5,120 Hz keeps 2,000 Hz, and a 1,024-sample frame
keeps 400 of its 512 lines and drops 112. The lowest frequency that
folds into the span is the rate less the span, 3,120 Hz, 1.56 spans:
a Butterworth of design order N run forward and backward is
20 log10(1 / (1 + 1.56^(2N))) there, 61.8 dB down at eighth order,
54.1 at seventh and 46.4 at sixth, against the 50 dB test documents
commonly ask for; at its corner it is 6.02 dB down. A 1 g peak tone on
a line of a Hann-windowed autospectrum at 5 Hz lines stands at
0.5 / (1.5 x 5) = 0.0667 g^2/Hz. And a low-pass at the span run after
the decimation leaves the 520 Hz alias where it is.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import Filtering, filtered
from visualdynamics.core.resample import (
    ORDER,
    SPAN_RATIO,
    AliasWarning,
    alias_free_span,
    alias_frequency,
    anti_alias_filter,
    decimation_factor,
    gain_db,
    level_change,
    line_counts,
    ordinal,
    read_resample,
    resample,
)

G = 9.80665
#: the record's rate: sixteen times the 5,120 Hz channel, as the
#: voltage is described between the converter's samples
FINE = 81920.0
RATE = 5120.0


def record(tones=((4600.0, 1.0),), samples=16 * 4096, rate=FINE,
           noise=0.0, seed=0):
    """A record of tones, each (frequency, g peak), in m/s^2 shown in g,
    with a little white noise if asked."""
    t = np.arange(samples) / rate
    y = sum(peak * np.sin(2.0 * np.pi * f * t) for f, peak in tones)
    if noise:
        y = y + noise * np.random.default_rng(seed).standard_normal(samples)
    return TimeHistory(t, np.atleast_2d(y * G), response_dof=['1Z+'],
                       ordinate_dim=['acceleration'], ordinate_unit=['g'])


def zero_phase_butterworth_db(ratio, order):
    """The analog Butterworth run forward and backward, at `ratio`
    times its corner: the hand value the digital design approaches."""
    return 20.0 * np.log10(1.0 / (1.0 + ratio ** (2 * order)))


# ---- the arithmetic ---------------------------------------------------


def test_a_4600_hz_tone_sampled_at_5120_hz_lands_at_520_hz():
    assert alias_frequency(4600.0, 5120.0) == 520.0
    # the band 90 to 110 kHz folds into a 10 kHz band at 100 kS/s
    assert np.allclose(alias_frequency([90e3, 95e3, 105e3, 110e3], 100e3),
                       [10e3, 5e3, 5e3, 10e3])
    # below the Nyquist frequency a frequency is its own alias, and far
    # above it the nearest multiple of the rate is taken
    assert alias_frequency(1000.0, 5120.0) == 1000.0
    assert alias_frequency(3 * 5120.0 + 100.0, 5120.0) == pytest.approx(100.0)
    assert alias_frequency(2 * 5120.0 - 100.0, 5120.0) == pytest.approx(100.0)


def test_the_span_is_the_rate_over_2_56():
    assert SPAN_RATIO == 2.56
    assert alias_free_span(5120.0) == pytest.approx(2000.0)
    assert alias_free_span(51200.0) == pytest.approx(20000.0)
    with pytest.raises(ValueError, match='above 2'):
        alias_free_span(5120.0, 2.0)
    with pytest.raises(ValueError, match='above zero'):
        alias_free_span(0.0)


@pytest.mark.parametrize('frame,expected', [(1024, (512, 400, 112)),
                                            (2048, (1024, 800, 224)),
                                            (4096, (2048, 1600, 448)),
                                            (1000, (500, 390, 110))])
def test_a_frame_keeps_its_length_over_2_56_lines(frame, expected):
    assert line_counts(frame) == expected


def test_the_factor_is_a_whole_number():
    assert decimation_factor(51200.0, 5120.0) == 10
    assert decimation_factor(FINE, RATE) == 16
    # a rate read off an abscissa's step does not come out exact
    assert decimation_factor(51199.99999, 5120.0) == 10
    with pytest.raises(ValueError, match=r'5120 Hz \(every 10th sample\) '
                                         r'and 4654.55 Hz \(every 11th'):
        decimation_factor(51200.0, 5000.0)
    with pytest.raises(ValueError, match='never raises'):
        decimation_factor(5120.0, 5120.0)
    with pytest.raises(ValueError, match='never raises'):
        decimation_factor(5120.0, 10240.0)
    with pytest.raises(ValueError, match='above zero'):
        decimation_factor(5120.0, 0.0)


def test_the_filter_is_6_db_down_at_the_span_and_62_down_where_it_could_fold():
    """At the corner the zero-phase pair is -6.02 dB by definition; at
    the lowest frequency that folds into the span, 1.56 spans, the
    default eighth order is past 50 dB, and seventh is the least that
    is."""
    span = alias_free_span(RATE)
    fold = RATE - span
    assert fold / span == pytest.approx(1.56)
    assert anti_alias_filter(RATE) == Filtering(high=2000.0, order=8)
    assert gain_db(anti_alias_filter(RATE), FINE, span) == pytest.approx(
        -6.0206, abs=1e-3)
    for order, hand in ((8, -61.8), (7, -54.1), (6, -46.4)):
        assert zero_phase_butterworth_db(1.56, order) == pytest.approx(
            hand, abs=0.05)
        assert gain_db(anti_alias_filter(RATE, order), FINE, fold) == \
            pytest.approx(hand, abs=0.5)
    assert gain_db(anti_alias_filter(RATE, 7), FINE, fold) < -50.0
    assert gain_db(anti_alias_filter(RATE, 6), FINE, fold) > -50.0
    assert ORDER == 8
    # an array in, an array out
    assert gain_db(anti_alias_filter(RATE), FINE, [span, fold]).shape == (2,)


def test_the_words():
    assert level_change(-61.84) == '61.8 dB down'
    assert level_change(0.01) == 'unchanged'
    assert level_change(3.06) == '3.1 dB up'
    assert [ordinal(n) for n in (1, 2, 3, 4, 10, 11, 12, 13, 16, 21, 22,
                                 111, 123)] == [
        '1st', '2nd', '3rd', '4th', '10th', '11th', '12th', '13th', '16th',
        '21st', '22nd', '111th', '123rd']


# ---- resampling a record ------------------------------------------------


def test_with_no_filter_the_tone_is_a_520_hz_tone_with_its_sign_reversed():
    """Every 16th sample, bare: the samples are those of a 520 Hz tone,
    reversed, and nothing in them tells the two apart. It warns."""
    source = record()
    with pytest.warns(AliasWarning, match=r'5120 Hz with no anti-alias '
                                          r'filter: .* between 2560 and '
                                          r'40960 Hz'):
        bare = resample(source, RATE, anti_alias=False)
    t = bare.abscissa
    assert bare.sample_rate == pytest.approx(RATE)
    assert np.max(np.abs(bare.ordinate[0] / G
                         + np.sin(2.0 * np.pi * 520.0 * t))) < 1e-9


def test_the_warning_points_at_the_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        resample(record(), RATE, anti_alias=False)
    assert [w.category for w in caught] == [AliasWarning]
    assert caught[0].filename == __file__


def test_filtered_first_the_tone_is_gone_and_the_content_kept():
    """The 4,600 Hz tone is 116 dB down behind the default filter; a
    300 Hz tone, well inside the span, comes through at its height."""
    source = record(((4600.0, 1.0), (300.0, 0.1)))
    with warnings.catch_warnings():
        warnings.simplefilter('error', AliasWarning)
        slow = resample(source, RATE)
    t = slow.abscissa
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    expected = 0.1 * np.sin(2.0 * np.pi * 300.0 * t)
    assert np.max(np.abs(slow.ordinate[0, middle] / G
                         - expected[middle])) < 1e-4


def test_the_result_keeps_the_clock_and_the_records_and_not_the_marks():
    source = record(samples=16 * 4096 + 5)
    source.averaging = Averaging(frame_length=8192, frames=2)
    slow = resample(source, RATE)
    assert len(slow.abscissa) == 4097
    assert np.array_equal(slow.abscissa, source.abscissa[::16])
    assert slow.response_dof == ['1Z+']
    assert slow.ordinate_dim == ['acceleration']
    assert slow.ordinate_unit == ['g']
    assert slow.averaging is None
    assert not slow.shocks


def test_every_channel_is_resampled():
    one, two = record(((4600.0, 1.0),)), record(((300.0, 1.0),))
    both = TimeHistory(one.abscissa, np.vstack([one.ordinate, two.ordinate]),
                       response_dof=['1Z+', '2Z+'],
                       ordinate_dim=['acceleration', 'acceleration'])
    slow = resample(both, RATE)
    assert slow.ordinate.shape == (2, 4096)
    assert np.allclose(slow.ordinate[1], resample(two, RATE).ordinate[0])


def test_the_method_is_the_function():
    source = record()
    assert np.array_equal(source.resample(RATE).ordinate,
                          resample(source, RATE).ordinate)
    assert np.array_equal(source.resample(RATE, order=4).ordinate,
                          resample(source, RATE, order=4).ordinate)
    with pytest.warns(AliasWarning):
        source.resample(RATE, anti_alias=False)


def test_a_low_pass_after_decimating_leaves_the_alias_where_it_is():
    """Run on the bare 5,120 Hz record, a 2,000 Hz low-pass finds the
    520 Hz alias inside its pass band and leaves it."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AliasWarning)
        bare = resample(record(), RATE, anti_alias=False)
    after = filtered(bare, Filtering(high=2000.0, order=ORDER))
    middle = slice(1024, 3072)
    assert np.max(np.abs(after.ordinate[0, middle]
                         - bare.ordinate[0, middle])) / G < 1e-3


def test_an_unevenly_sampled_record_is_refused():
    source = record()
    source.abscissa[10] += 1e-6
    with pytest.raises(ValueError, match='evenly spaced'):
        resample(source, RATE)


# ---- the reading --------------------------------------------------------


def test_the_reading_of_the_worked_example():
    reading = read_resample(record(), RATE, frequencies=[4600.0])
    assert (reading.source_rate, reading.rate, reading.factor) == (
        pytest.approx(FINE), pytest.approx(RATE), 16)
    assert reading.nyquist == pytest.approx(2560.0)
    assert reading.span == reading.corner == pytest.approx(2000.0)
    assert (reading.frame_length, reading.lines, reading.kept,
            reading.dropped) == (1024, 512, 400, 112)
    assert reading.resolution == pytest.approx(5.0)
    assert reading.lowest_fold == pytest.approx(3120.0)
    assert reading.gain_at_span == pytest.approx(-6.02, abs=0.01)
    assert reading.gain_at_fold == pytest.approx(-61.8, abs=0.5)
    (fold,) = reading.folds
    assert (fold.frequency, fold.alias, fold.multiple, fold.in_span) == (
        4600.0, 520.0, 1, True)
    # the tone on a line: 0.5 g^2 over the Hann window's 1.5 lines of 5 Hz
    assert fold.source / G ** 2 == pytest.approx(0.0667, abs=1e-4)
    assert fold.bare / G ** 2 == pytest.approx(0.0667, abs=1e-4)
    assert fold.after_db == pytest.approx(0.0, abs=0.01)
    assert fold.filtered_db < -100.0
    assert reading.describe() == (
        '81920 Hz to 5120 Hz (every 16th sample) behind a low-pass at '
        '2000 Hz: Nyquist 2560 Hz, span 2000 Hz, 400 of 512 lines kept. '
        f'4600 Hz folds to 520 Hz, inside the span: '
        f'{level_change(fold.filtered_db)} filtered first, unchanged '
        'low-passed after decimating.')


def test_the_spectra_share_one_line_spacing():
    reading = read_resample(record(), RATE)
    spacings = {name: f[1] - f[0] for name, (f, _g) in reading.spectra.items()}
    assert set(spacings) == {'source', 'filtered', 'bare', 'after'}
    assert all(s == pytest.approx(5.0) for s in spacings.values())
    assert reading.spectra['source'][0][-1] == pytest.approx(FINE / 2.0)
    assert reading.spectra['bare'][0][-1] == pytest.approx(RATE / 2.0)
    assert reading.density_dim == 'acceleration**2/frequency'


def test_with_no_frequency_given_the_tallest_line_above_nyquist_is_folded():
    reading = read_resample(record(((4600.0, 1.0), (300.0, 2.0),
                                    (7000.0, 0.1))), RATE)
    assert [fold.frequency for fold in reading.folds] == [4600.0]


def test_a_frequency_that_folds_above_the_span_lands_among_the_dropped_lines():
    """2,900 Hz folds to 2,220 Hz, between the span and the Nyquist
    frequency: a line the analyzer drops, and the low-pass after the
    decimation takes it down."""
    reading = read_resample(record(((2900.0, 1.0),)), RATE,
                            frequencies=[2900.0, 1000.0, 2560.0])
    (fold,) = reading.folds
    assert fold.alias == pytest.approx(2220.0)
    assert not fold.in_span
    assert fold.after_db < -10.0
    assert 'among the dropped lines' in fold.describe()


def test_a_record_with_nothing_above_the_new_nyquist_folds_nothing():
    reading = read_resample(record(((300.0, 1.0),)), RATE)
    # a pure tone's leakage reaches above 2,560 Hz, so its tallest line
    # there is folded; the reading says where, at a level far down
    assert len(reading.folds) <= 1
    silent = TimeHistory(np.arange(16 * 4096) / FINE,
                         np.zeros((1, 16 * 4096)), response_dof=['1Z+'],
                         ordinate_dim=['acceleration'])
    assert read_resample(silent, RATE).folds == ()
    assert read_resample(silent, RATE).describe().endswith('lines kept.')


def test_the_reading_reads_the_channel_asked_for():
    one, two = record(((4600.0, 1.0),)), record(((3600.0, 1.0),))
    both = TimeHistory(one.abscissa, np.vstack([one.ordinate, two.ordinate]),
                       response_dof=['1Z+', '2Z+'],
                       ordinate_dim=['acceleration', 'acceleration'])
    reading = read_resample(both, RATE, 1)
    assert reading.record == 1
    assert reading.label.startswith('2Z+')
    assert [fold.frequency for fold in reading.folds] == [3600.0]
    assert reading.folds[0].alias == pytest.approx(1520.0)


def test_the_frame_can_be_chosen_and_must_fit():
    source = record()
    reading = read_resample(source, RATE, frame_length=2048)
    assert (reading.lines, reading.kept, reading.dropped) == (1024, 800, 224)
    assert reading.resolution == pytest.approx(2.5)
    with pytest.raises(ValueError, match='longer than the record'):
        read_resample(source, RATE, frame_length=8192)
    with pytest.raises(ValueError, match='two samples'):
        read_resample(source, RATE, frame_length=1)
    # a short record takes the longest power of two it holds
    short = record(samples=16 * 600)
    assert read_resample(short, RATE).frame_length == 512


def test_the_reading_refuses_what_it_cannot_read():
    source = record()
    with pytest.raises(ValueError, match='no channel 3'):
        read_resample(source, RATE, 3)
    with pytest.raises(ValueError, match='too few for a spectrum'):
        read_resample(record(samples=200), FINE / 16)
    with pytest.raises(ValueError, match='whole factor'):
        read_resample(source, 5000.0)


def test_the_reading_does_not_warn():
    """The bare decimation is the comparison the reading shows, not a
    resampling anyone asked for."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', AliasWarning)
        read_resample(record(), RATE)


# ---- the view -----------------------------------------------------------


def _plots(layout):
    import pyqtgraph as pg

    return [item for item in layout.ci.items
            if isinstance(item, pg.PlotItem)]


def test_the_view_marks_the_span_the_nyquist_frequency_and_the_dropped_lines(
        qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.resample import build_resample

    reading = read_resample(record(noise=0.01), RATE, frequencies=[4600.0])
    layout = pg.GraphicsLayoutWidget()
    try:
        (plot,) = _plots(build_resample(layout, reading) and layout)
        lines = {round(item.value()): item for item in plot.items
                 if isinstance(item, pg.InfiniteLine)}
        assert set(lines) == {2000, 2560}
        assert lines[2560].pen.style() == Qt.PenStyle.DashLine
        assert lines[2000].pen.style() == Qt.PenStyle.SolidLine
        assert lines[2560].label.format == 'Nyquist 2560 Hz'
        assert lines[2000].label.format == 'span 2000 Hz'
        (region,) = [item for item in plot.items
                     if isinstance(item, pg.LinearRegionItem)]
        assert region.getRegion() == pytest.approx((2000.0, 2560.0))
        assert region.brush.style() == Qt.BrushStyle.BDiagPattern
        names = [label.text for _sample, label in plot.legend.items]
        assert names[:4] == [
            'as recorded, 81920 Hz',
            'filtered at 2000 Hz, then decimated to 5120 Hz',
            'decimated with no filter',
            'decimated, then low-passed at 2000 Hz']
        assert 'dropped lines, 112 of 512' in names
        assert 'above the new Nyquist frequency, as recorded' in names
        assert 'where it lands, decimated with no filter' in names
        # the spectra told apart by their line, not their color
        curves = [item for item in plot.listDataItems()
                  if item.opts['pen'] is not None and item.name()
                  in names[:4]]
        styles = [curve.opts['pen'].style() for curve in curves]
        assert styles == [Qt.PenStyle.SolidLine, Qt.PenStyle.SolidLine,
                          Qt.PenStyle.DashLine, Qt.PenStyle.DotLine]
        assert curves[1].opts['pen'].widthF() > curves[0].opts['pen'].widthF()
        assert plot.getViewBox().state['logMode'] == [False, True]
        assert 'every 16th sample, 400 of 512 lines kept' in \
            plot.titleLabel.text
        assert '4600 Hz folds to 520 Hz, inside the span' in \
            plot.titleLabel.text
        # the axis runs past the fold, so the content that folds is seen
        low, high = plot.getViewBox().viewRange()[0]
        assert low == 0.0 and high > 4600.0
    finally:
        layout.close()


def test_the_view_joins_each_fold_to_where_it_lands(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.resample import build_resample

    reading = read_resample(record(), RATE, frequencies=[4600.0])
    layout = pg.GraphicsLayoutWidget()
    try:
        build_resample(layout, reading)
        (plot,) = _plots(layout)
        marks = [item for item in plot.listDataItems()
                 if item.opts['symbol'] == 't']
        hollow = [m for m in marks if m.opts['symbolBrush'].color().alpha() == 0]
        filled = [m for m in marks if m.opts['symbolBrush'].color().alpha() > 0]
        assert [float(m.xData[0]) for m in hollow] == [4600.0]
        assert [float(m.xData[0]) for m in filled] == [520.0]
        (link,) = [item for item in plot.listDataItems()
                   if item.xData is not None and len(item.xData) == 2
                   and item.opts['symbol'] is None]
        assert list(link.xData) == [4600.0, 520.0]
        (words,) = [item for item in plot.items
                    if isinstance(item, pg.TextItem)]
        assert words.toPlainText().splitlines() == [
            '4600 Hz folds to 520 Hz', 'in the span',
            'low-passed after: unchanged',
            f'filtered first: {level_change(reading.folds[0].filtered_db)}']
    finally:
        layout.close()


def test_in_the_view_the_low_pass_after_lies_on_the_bare_curve_at_the_alias(
        qt_app):
    """The picture of a low-pass that came too late: at 520 Hz the
    dotted curve stands where the dashed one does, and the filtered
    curve is far below both."""
    import pyqtgraph as pg

    from visualdynamics.plot.resample import build_resample

    reading = read_resample(record(noise=0.01), RATE, frequencies=[4600.0])
    layout = pg.GraphicsLayoutWidget()
    try:
        build_resample(layout, reading)
        (plot,) = _plots(layout)
        by_name = {item.name(): item for item in plot.listDataItems()
                   if item.name()}

        def at(name, frequency):
            item = by_name[name]
            k = int(np.argmin(np.abs(item.xData - frequency)))
            return float(item.yData[k])

        bare = at('decimated with no filter', 520.0)
        after = at('decimated, then low-passed at 2000 Hz', 520.0)
        first = at('filtered at 2000 Hz, then decimated to 5120 Hz', 520.0)
        assert after == pytest.approx(bare, rel=1e-3)
        assert first < bare * 1e-3
        # drawn in the window's units: g^2/Hz in the default system
        assert bare == pytest.approx(reading.folds[0].bare / G ** 2,
                                     rel=1e-6)
    finally:
        layout.close()


def test_the_view_draws_to_a_file(qt_app, tmp_path):
    from visualdynamics.plot import plot_resample

    path = tmp_path / 'resample.png'
    assert plot_resample(record(), RATE, frequencies=[4600.0],
                         path=str(path), show=False) == str(path)
    assert path.exists() and path.stat().st_size > 2000


def test_the_summary():
    from visualdynamics.plot.resample import summary

    reading = read_resample(record(), RATE, frequencies=[4600.0])
    assert summary(reading).startswith('1Z+')
    assert summary(reading).endswith(reading.describe())


# ---- the window --------------------------------------------------------


def _resample_reading(window, pump, history=None, factor=16):
    """A record in the window, shown, on the Resample reading at
    `factor`."""
    name = window.add_object('Record', history or record(noise=0.001))
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    pane = window.data_pane
    assert pane.resample_action.isVisible()
    if not pane.resample_action.isChecked():
        pane.resample_action.trigger()
    pump()
    pane.resample_panel.factor_box.setValue(factor)
    pump()
    return pane


def test_a_time_history_offers_the_resample_reading(window, pump):
    """Choosing it puts the settings beside the spectrum and the
    reading in the panel and on the status line."""
    pane = _resample_reading(window, pump)
    panel = pane.resample_panel
    assert pane.resample_action.text() == 'Resample'
    assert panel.isVisible()
    assert len(_plots(pane.graphics)) == 1
    values = {key: label.text() for key, label in panel.values.items()}
    assert values == {
        'source': '81920 Hz', 'rate': '5120 Hz', 'nyquist': '2560 Hz',
        'span': '2000 Hz', 'lines': '400 of 512', 'dropped': '112',
        'resolution': '5 Hz', 'frame': '1024', 'at_span': '-6.0 dB',
        'lowest': '3120 Hz', 'at_lowest': '-62.2 dB'}
    assert panel.verdict.text().startswith(
        '4600 Hz folds to 520 Hz, inside the span: ')
    assert panel.verdict.text().endswith(
        'unchanged low-passed after decimating.')
    assert window._status_text.startswith(
        '1Z+: 81920 Hz to 5120 Hz (every 16th sample)')
    assert pane.resample_settings['factor'] == 16


def test_the_settings_redraw_the_reading_and_are_kept(window, pump):
    pane = _resample_reading(window, pump)
    panel = pane.resample_panel
    panel.order_box.setValue(6)
    panel.frame_box.setCurrentIndex(panel.frame_box.findData(2048))
    pump()
    assert panel.values['at_lowest'].text() == '-46.7 dB'
    assert panel.values['lines'].text() == '800 of 1024'
    assert pane.resample_settings == {'factor': 16, 'order': 6,
                                      'frame_length': 2048,
                                      'frequencies': None}
    # typed frequencies: one folding among the dropped lines
    panel.frequencies_edit.setText('2900, 4600')
    panel.frequencies_edit.editingFinished.emit()
    pump()
    assert pane.resample_settings['frequencies'] == (2900.0, 4600.0)
    lines = panel.verdict.text().splitlines()
    assert lines[0].startswith('2900 Hz folds to 2220 Hz, among the '
                               'dropped lines')
    assert lines[1].startswith('4600 Hz folds to 520 Hz, inside the span')
    # text that is not a list of frequencies is put back
    panel.frequencies_edit.setText('fast')
    panel.frequencies_edit.editingFinished.emit()
    pump()
    assert panel.frequencies_edit.text() == '2900, 4600'
    # and the settings survive clicking away and back
    other = window.add_object('Other', record(((300.0, 1.0),)))
    window.show_object(other)
    pump()
    window.show_object('Record')
    pump()
    assert panel.order_box.value() == 6
    assert panel.frequencies_edit.text() == '2900, 4600'


def test_the_resample_reading_is_journaled_as_its_plot_call(window, pump):
    pane = _resample_reading(window, pump)
    pane.resample_panel.frequencies_edit.setText('4600')
    pane.resample_panel.frequencies_edit.editingFinished.emit()
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_resample(')]
    assert lines == [
        ("visualdynamics.plot.plot_resample(project['Record'], 5120.0, 0, "
         "frequencies=(4600.0,), order=8, frame_length=1024, "
         "path='resample.png', show=False)")]


def test_a_factor_that_leaves_too_few_samples_says_so(window, pump):
    pane = _resample_reading(window, pump, record(samples=4096), factor=2)
    pane.resample_panel.factor_box.setValue(1024)
    pump()
    assert 'too few for a spectrum' in window._status_text
    assert pane.resample_panel.values['rate'].text() == '—'


def test_a_frame_longer_than_the_record_gives_way(window, pump):
    pane = _resample_reading(window, pump, record(samples=16 * 600))
    assert pane.resample_panel.values['frame'].text() == '512'
    assert 'frame_length' not in window.project.journal[-1]


def test_the_resample_reading_stands_the_trace_marks_down(window, pump):
    """It replaces the trace, so the averaging panel is put away and
    the 3-D toggle is not offered; another reading puts it away."""
    pane = _resample_reading(window, pump)
    assert not pane.averaging_panel.isVisibleTo(pane)
    assert not pane.waterfall_action.isVisible()
    pane.averaging_action.trigger()
    pump()
    assert not pane.resample_action.isChecked()
    assert not pane.resample_panel.isVisibleTo(pane)


def test_a_fold_reads_the_spike_within_a_line_of_the_frequency_given():
    """A tone at 4,604 Hz asked about as 4,600 Hz: its spike stands on
    the 4,605 Hz line, and the fold reads that spike, not the 4,600 Hz
    line on its shoulder."""
    reading = read_resample(record(((4604.0, 1.0),)), RATE,
                            frequencies=[4600.0])
    (fold,) = reading.folds
    f, g = reading.spectra['source']
    near = (f > 4590.0) & (f < 4615.0)
    assert fold.source == g[near].max()
    assert fold.source > g[np.argmin(np.abs(f - 4600.0))] * 2.0
