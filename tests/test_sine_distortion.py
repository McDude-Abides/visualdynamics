"""The sine distortion check: the unfiltered RMS against the tracked tone.

The numbers are closed forms. A tone with a harmonic at h of its
amplitude has an RMS of sqrt(1 + h**2) times the tone's, so the ratio
reads 1.005 at h = 0.10, 1.020 at 0.20 and 1.050 at 0.32, and the
distortion sqrt(ratio**2 - 1) reads h itself. The record is a 2 m/s**2
sweep from 100 to 500 Hz at 100 Hz/s sampled at 4096 Hz, so its third
harmonic stays under 1500 Hz, inside the share of the band the
unfiltered RMS reads in full.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.sine import (
    SineLevel,
    SineLevelSet,
    SineSweepSpecification,
    SineTone,
    extract_sine,
)
from visualdynamics.core.sine_distortion import (
    DISTORTION_EXCESS,
    check_distortion,
    decibels,
    distortion,
    distortion_limit,
)

FS = 4096.0
AMPLITUDE = 2.0
CONTROL = '101Z+'
LIMIT = np.sqrt(1.05 ** 2 - 1.0)            # 0.3202


def _sweep(harmonic=0.0, band=None, clip=None, offset=0.0, noise=0.0,
           top=500.0, seconds=6.0, second_tone=False):
    """The sweep with a third harmonic at `harmonic` of the tone (only
    where the drive is inside `band`, Hz, when given), clipped at `clip`,
    on an `offset`, under white `noise`; with `second_tone`, a 1 m/s**2
    tone at 1000 Hz plays alongside it from 1 to 3 s of the environment."""
    up = SineTone('Up', 0.5, [100.0, top], [[AMPLITUDE]] * 2, [0], [100.0])
    tones = [up]
    if second_tone:
        tones.append(SineTone('High', 1.0, [1000.0, 1000.0 + 2.0],
                              [[1.0]] * 2, [0], [1.0]))
    spec = SineSweepSpecification(tones, [CONTROL], ordinate_unit='m/s**2')
    n = int(seconds * FS)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n) * noise + offset
    argument = up.argument(1.0 / FS)
    start = int(0.5 * FS)
    m = min(len(argument), n - start)
    drive = up.frequency_at(np.arange(m) / FS)
    wave = np.cos(argument[:m])
    third = harmonic * np.cos(3.0 * argument[:m])
    if band is not None:
        third = third * ((drive >= band[0]) & (drive <= band[1]))
    wave = AMPLITUDE * (wave + third)
    if clip is not None:
        wave = np.clip(wave, -clip, clip)
    x[start:start + m] += wave
    if second_tone:
        high = tones[1].argument(1.0 / FS)
        at = int(1.0 * FS)
        x[at:at + len(high)] += np.cos(high)
    history = TimeHistory(np.arange(n) / FS, x[None, :],
                          response_dof=[CONTROL],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['m/s**2'])
    return history, spec


def _checked(**kwargs):
    history, spec = _sweep(**kwargs)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    return check_distortion(history, spec, levels), history, spec, levels


def _inside(check, low=150.0, high=450.0):
    return (check.frequency > low) & (check.frequency < high)


# ---- the arithmetic ---------------------------------------------------------


def test_the_distortion_is_what_is_left_over_the_tone():
    assert float(distortion(1.05, 1.0)) == pytest.approx(0.32016, abs=1e-5)
    assert float(distortion(np.sqrt(1.01), 1.0)) == pytest.approx(0.1)
    assert float(distortion(2.0, 2.0)) == 0.0
    assert float(distortion(0.99, 1.0)) == 0.0, \
        'a ratio under one is the readings\' scatter, not a negative power'
    assert np.isnan(distortion(1.0, 0.0))
    assert distortion([1.0, 1.05], [1.0, 1.0]) == pytest.approx([0.0, LIMIT])


def test_a_tolerance_on_the_ratio_is_a_distortion_limit():
    assert DISTORTION_EXCESS == 0.05
    assert distortion_limit() == pytest.approx(0.32016, abs=1e-5)
    assert distortion_limit(0.10) == pytest.approx(np.sqrt(1.21 - 1.0))
    assert float(decibels(distortion_limit())) == pytest.approx(-9.892,
                                                                abs=1e-3)
    assert float(decibels(0.1)) == pytest.approx(-20.0)
    assert decibels(0.0) == -np.inf
    with pytest.raises(ValueError, match='more than zero'):
        distortion_limit(0.0)


# ---- the check on a sweep ---------------------------------------------------


@pytest.mark.parametrize('harmonic,ratio', [(0.10, 1.005), (0.20, 1.020),
                                            (0.32, 1.050)])
def test_a_harmonic_reads_as_its_own_share_of_the_tone(harmonic, ratio):
    check, *_rest = _checked(harmonic=harmonic)
    inside = _inside(check)
    assert check.ratio[0][inside] == pytest.approx(ratio, abs=6e-4)
    assert check.distortion[0][inside] == pytest.approx(harmonic, abs=4e-3)
    assert check.fundamental[0][inside] == pytest.approx(
        AMPLITUDE / np.sqrt(2.0), rel=1e-4), 'the tone, and only the tone'


def test_a_clean_sweep_reads_near_zero_and_passes():
    check, *_rest = _checked()
    assert check.distortion.max() < 0.02
    assert check.judged.all() and check.passed
    assert check.notes == ()
    assert check.cycles == 10.0, "read over the extraction's own smoothing"
    assert check.dofs == [CONTROL] and check.tone == 'Up'


def test_a_harmonic_in_one_band_is_flagged_there_and_nowhere_else():
    check, *_rest = _checked(harmonic=0.5, band=(250.0, 350.0))
    over = check.over[0]
    assert not check.passed
    assert over.any()
    assert check.frequency[over].min() > 240.0
    assert check.frequency[over].max() < 360.0
    clear = (check.frequency < 230.0) | (check.frequency > 370.0)
    assert check.distortion[0][clear].max() < 0.05
    frequency, worst = check.worst()
    assert 250.0 < frequency < 352.0
    assert worst == pytest.approx(0.5, abs=0.01)
    assert check.describe(CONTROL).startswith(
        f'Up at {CONTROL}: worst 50.0 % (-6.0 dB) at ')
    assert check.describe().endswith('over the 32.0 % limit')


def test_a_looser_tolerance_lets_the_same_band_through():
    check, *_rest = _checked(harmonic=0.5, band=(250.0, 350.0))
    looser = replace(check, excess=0.15)
    assert looser.limit == pytest.approx(np.sqrt(1.15 ** 2 - 1.0))
    assert looser.passed, 'a 50 % harmonic is under a 57 % limit'
    assert looser.describe().endswith('within the 56.8 % limit')


def test_clipping_hardly_moves_the_check():
    """A sine clipped at four fifths of its peak puts most of the lost
    power back into the fundamental: the ratio reads about 1.004."""
    check, *_rest = _checked(clip=0.8 * AMPLITUDE)
    inside = _inside(check, 150.0, 300.0)
    assert np.median(check.ratio[0][inside]) == pytest.approx(1.004,
                                                              abs=1e-3)
    assert check.passed


def test_an_offset_is_taken_out_before_the_rms():
    clean, *_rest = _checked()
    offset, *_rest = _checked(offset=0.7)
    assert offset.unfiltered == pytest.approx(clean.unfiltered, abs=1e-9)
    assert offset.passed


def test_noise_is_counted_as_everything_that_is_not_the_tone():
    """White noise of 0.2 RMS beside a tone of 1.414 RMS is a distortion
    of 0.2 / 1.414, 14 %, read in the band the reconstruction holds."""
    check, *_rest = _checked(noise=0.2)
    inside = _inside(check, 150.0, 300.0)
    assert np.median(check.distortion[0][inside]) == pytest.approx(
        0.2 / np.sqrt(2.0), abs=0.01)


def test_two_tones_at_once_are_both_the_fundamental():
    """With both tones extracted, the second's power joins the first's,
    and a clean record still reads clean where they overlap."""
    check, *_rest = _checked(second_tone=True)
    assert check.judged.all()
    overlap = (check.seconds > 1.6) & (check.seconds < 2.9)
    assert overlap.any()
    assert check.fundamental[0][overlap] == pytest.approx(
        np.sqrt(AMPLITUDE ** 2 + 1.0) / np.sqrt(2.0), rel=4e-3)
    # the beat between the tones over a short span and the second
    # tone's own scatter, a few percent; counted by the line's center
    # alone, the spans across the second tone's ends read 29 %
    assert np.median(check.distortion[0][overlap]) < 0.04
    assert check.distortion.max() < 0.15
    assert check.passed


def test_a_tone_the_levels_do_not_hold_leaves_its_lines_unjudged():
    history, spec = _sweep(second_tone=True)
    levels = extract_sine(history, spec, tones=['Up'], cycles=10.0,
                          workers=1)
    check = check_distortion(history, spec, levels)
    shared = ~check.judged[0]
    assert shared.any()
    # a span reaching the other tone at all is shared: ten cycles of
    # the sweep at 150 to 350 Hz is under 0.07 s, half each side
    assert 1.0 - 0.035 <= check.seconds[shared].min() < 1.0
    assert 3.0 < check.seconds[shared].max() <= 3.0 + 0.035,         'a span with only its edge in the other tone is shared too'
    assert check.passed, 'the unjudged lines carry the other tone'
    assert check.ratio[0][shared].max() > 1.1, \
        'they would have been flagged if judged'
    assert any('shared with a tone the levels do not hold' in note
               for note in check.notes)


def test_a_line_under_the_noise_floor_is_not_judged():
    history, spec = _sweep(harmonic=0.5)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    level = levels.levels[0]
    below = np.zeros(np.shape(level.ordinate), dtype=bool)
    below[0, :20] = True
    flagged = SineLevel(level.abscissa, level.ordinate,
                        response_dof=list(level.response_dof),
                        ordinate_dim=list(level.ordinate_dim),
                        ordinate_unit=list(level.ordinate_unit),
                        tone=level.tone, onset=level.onset,
                        seconds=level.seconds, floor=level.floor,
                        below_floor=below)
    check = check_distortion(history, spec,
                             SineLevelSet([flagged], cycles=levels.cycles))
    assert not check.judged[0, :20].any() and check.judged[0, 20:].all()
    assert int(check.over[0].sum()) == len(check.frequency) - 20
    assert check.notes == ('20 lines under the noise floor, not judged',)


def test_without_levels_the_extraction_is_run_with_the_records_setting():
    from visualdynamics.core.sine import SineExtraction

    history, spec = _sweep(harmonic=0.2)
    history.sine_extraction = SineExtraction(cycles=10.0)
    run = check_distortion(history, spec)
    given = check_distortion(history, spec,
                             extract_sine(history, spec, cycles=10.0))
    assert np.array_equal(run.frequency, given.frequency)
    assert run.distortion == pytest.approx(given.distortion, rel=1e-9)


def test_a_wider_span_reads_the_same_steady_harmonic():
    history, spec = _sweep(harmonic=0.2)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    wide = check_distortion(history, spec, levels, cycles=40.0)
    assert wide.cycles == 40.0
    assert wide.distortion[0][_inside(wide)] == pytest.approx(0.2, abs=4e-3)


def test_a_sweep_past_the_reconstructed_band_says_so():
    check, *_rest = _checked(top=800.0, seconds=9.0)
    reach = 0.8 * FS / 2.0 / 3.0
    note = (f'above {reach:.4g} Hz the third harmonic is past 80 % of the '
            'Nyquist frequency, where the unfiltered RMS under-reads it')
    assert check.notes == (note,)


def test_what_cannot_be_checked_is_refused():
    history, spec = _sweep()
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    with pytest.raises(ValueError, match='more than zero'):
        check_distortion(history, spec, levels, excess=0.0)
    with pytest.raises(ValueError, match='more than zero cycles'):
        check_distortion(history, spec, levels, cycles=0.0)
    with pytest.raises(ValueError, match="no tone 'Down'"):
        check_distortion(history, spec, levels, tone='Down')
    with pytest.raises(ValueError, match='no reading for'):
        check_distortion(history, spec, levels, channels=['999X+'])
    with pytest.raises(ValueError, match='do not say what smoothing'):
        check_distortion(history, spec, SineLevelSet(levels.levels))
    with pytest.raises(ValueError, match='less than the 5000'):
        check_distortion(history, spec, levels, cycles=5000.0)
    with pytest.raises(ValueError, match='no tone; extract them first'):
        check_distortion(history, spec, SineLevelSet([], cycles=10.0))


# ---- the plot -----------------------------------------------------------------


def test_the_plot_names_the_limit_and_the_verdict_in_words(qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.sine_distortion import SYMBOLS, build_distortion

    check, *_rest = _checked(harmonic=0.5, band=(250.0, 350.0))
    layout = pg.GraphicsLayoutWidget()
    plot = build_distortion(layout, check)
    names = [label.text for _sample, label in plot.legend.items]
    assert 'limit 32.0 % (-9.9 dB): unfiltered 5 % over the tone' in names
    over = int(check.over[0].sum())
    assert any(name.startswith(f'{CONTROL}: worst 50.0 % at ')
               and name.endswith(f'{over} lines over the limit')
               for name in names), names
    limit, = [item for item in plot.listDataItems()
              if item.name() and item.name().startswith('limit')]
    assert limit.opts['pen'].style() == Qt.DashLine, \
        'the limit reads by its line style, not its hue'
    assert limit.getData()[1].tolist() == pytest.approx([100.0 * LIMIT] * 2)
    marked = [item for item in plot.listDataItems()
              if item.opts['symbol'] is not None]
    assert {item.opts['symbol'] for item in marked} == {SYMBOLS[0]}
    filled = [item for item in marked if item.opts['symbolBrush'] is not None]
    assert len(filled) == 1 and len(filled[0].getData()[0]) == over, \
        'a line over the limit is the same marker, filled'
    assert 'over the limit' in plot.titleLabel.text


def test_the_plot_draws_to_a_file_with_no_window(tmp_path):
    from visualdynamics.plot import plot_sine_distortion

    history, spec = _sweep(harmonic=0.2)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    path = tmp_path / 'distortion.png'
    plot_sine_distortion(history, spec, levels, path=str(path), show=False)
    assert path.stat().st_size > 5_000


# ---- the readout in the window --------------------------------------------------


def _shown(window, pump, history, spec, extract=True):
    window.add_object('Record', history)
    window.add_object('Sweep', spec)
    if extract:
        from visualdynamics.core.sine import SineExtraction

        history.sine_extraction = SineExtraction(cycles=10.0)
        window.project.extract_sine('Record', 'Sweep')
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    pane = window.data_pane
    pane.sine_action.setChecked(True)
    pane._choose_sine(True)
    pump()
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    return pane.sine_panel.distortion


def test_the_readout_checks_the_extracted_levels_in_the_sine_view(window,
                                                                   pump):
    panel = _shown(window, pump, *_sweep(harmonic=0.5, band=(250.0, 350.0)))
    assert panel.isVisible()
    assert panel.channel_box.currentText() == CONTROL
    assert not panel.tone_box.isVisible(), 'one tone: no tone to choose'
    assert panel.derived['limit'].text() == '32.0 % (-9.9 dB)'
    assert panel.derived['worst'].text().startswith('50.0 % (-6.0 dB) at ')
    assert panel.derived['verdict'].text() == 'over the limit'
    assert panel.derived['over'].text().endswith(' Hz')
    assert panel.preview.items, 'the distortion against frequency is drawn'
    assert window.project.journal[-1] == (
        "visualdynamics.plot.plot_sine_distortion(project['Record'], "
        "project['Sweep'], project['Sine Levels'], tone='Up', "
        "channel='101Z+', excess=0.05, path='distortion.png', show=False)")


def test_the_readout_stays_away_until_the_levels_are_extracted(window, pump):
    panel = _shown(window, pump, *_sweep(), extract=False)
    assert window.data_pane.sine_panel.isVisible()
    assert not panel.isVisible(), 'nothing to check yet (principle 3)'


def test_a_tolerance_edit_reads_again_without_reading_the_record(window,
                                                                 pump):
    panel = _shown(window, pump, *_sweep(harmonic=0.5, band=(250.0, 350.0)))
    made = panel.check
    panel.excess_box.setValue(15.0)
    pump()
    assert panel.check.excess == pytest.approx(0.15)
    assert panel.check.unfiltered is made.unfiltered, \
        'the RMS readings are kept; only the judgement moved'
    assert panel.derived['verdict'].text() == 'within the limit'
    assert panel.derived['limit'].text() == '56.8 % (-4.9 dB)'
    assert 'excess=0.15,' in window.project.journal[-1]
