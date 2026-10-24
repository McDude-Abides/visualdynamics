"""A sine profile written in segments, its crossovers solved and checked.

The hand values are the minimum-integrity profile test documents give
for equipment of one class, worked in four segments: 5 to 14 Hz at
0.20 in double amplitude, 14 to 33 Hz at 2 g, 33 to 52.1 Hz at 0.036 in
double amplitude and 52.1 to 500 Hz at 5 g. With a = 0.0511 f**2 D (g,
Hz, in peak to peak) every crossover checks: 0.0511 x 196 x 0.20 =
2.00 g at 14 Hz, 0.0511 x 1089 x 0.036 = 2.00 g at 33 Hz and
0.0511 x 2714 x 0.036 = 4.99 g at 52.1 Hz. The profile asks for 0.26 g
at 5 Hz, 8.8 in/s at the first crossover and 0.6 in/s at 500 Hz. Read as
peak, the double amplitude moves the first crossover to 9.9 Hz.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.sine import LINEAR, LOG, SineSweepSpecification, SineTone
from visualdynamics.core.sine_profile import (
    ACCURACY,
    CROSSOVER_TOLERANCE,
    ProfileSegment,
    SineProfile,
    acceleration,
    convert,
    crossover,
    displacement,
    interpolated,
    read_profile,
    velocity,
)

G = 9.80665
INCH = 0.0254
#: the hand constant: g per (Hz**2 in peak to peak)
HAND = (2.0 * np.pi) ** 2 / 2.0 / (G / INCH)


def _segments(first='peak to peak', unit='in', middle=0.036):
    return [displacement(0.20, 'in', first, start=5.0, end=14.0),
            acceleration(2.0, 'g', start=14.0, end=33.0),
            displacement(middle, unit, 'peak to peak', start=33.0, end=52.1),
            acceleration(5.0, 'g', start=52.1, end=500.0)]


def _profile(**kwargs):
    return SineProfile(_segments(**kwargs))


# ---- the relation ---------------------------------------------------------


def test_the_hand_constant_is_the_relation_in_inches_and_g():
    assert HAND == pytest.approx(0.0511, abs=5e-5)
    # 0.20 in peak to peak at 14 Hz is 2.00 g
    a = convert(0.10 * INCH, 14.0, 'displacement')
    assert a / G == pytest.approx(HAND * 196.0 * 0.20, rel=1e-12)
    assert a / G == pytest.approx(2.004, abs=1e-3)
    # v = 61.45 a / f in in/s with a in g: 2 g at 14 Hz is 8.78 in/s
    v = convert(2.0 * G, 14.0, 'acceleration', 'velocity')
    assert v / INCH == pytest.approx(61.45 * 2.0 / 14.0, rel=1e-3)
    # D = 19.57 a / f**2 in peak to peak: and back again
    d = convert(2.0 * G, 14.0, 'acceleration', 'displacement')
    assert 2.0 * d / INCH == pytest.approx(19.57 * 2.0 / 196.0, rel=1e-3)
    assert convert(3.0, 20.0, 'velocity', 'velocity') == 3.0
    assert np.isnan(convert(np.nan, 10.0, 'velocity')), \
        'a level that is not a number stays not a number'


def test_the_relation_refuses_what_it_cannot_carry():
    with pytest.raises(ValueError, match='is not a quantity'):
        convert(1.0, 10.0, 'jerk')
    with pytest.raises(ValueError, match='more than zero'):
        convert(1.0, 0.0, 'velocity')
    with pytest.raises(ValueError, match='more than zero'):
        convert(1.0, np.nan, 'velocity')


def test_a_crossover_is_where_two_segments_ask_for_the_same():
    d = (('displacement', 0.10 * INCH))
    assert crossover(d, ('acceleration', 2.0 * G)) == pytest.approx(
        np.sqrt(2.0 / (HAND * 0.20)), rel=1e-12)
    assert crossover(d, ('acceleration', 2.0 * G)) == pytest.approx(13.99,
                                                                    abs=0.01)
    # either order, and velocity against both
    assert crossover(('acceleration', 2.0 * G), d) == pytest.approx(
        crossover(d, ('acceleration', 2.0 * G)))
    v = ('velocity', 0.2)
    assert crossover(v, ('acceleration', 20.0)) == pytest.approx(
        20.0 / (2.0 * np.pi * 0.2))
    assert crossover(('displacement', 0.01), v) == pytest.approx(
        0.2 / (2.0 * np.pi * 0.01))
    assert crossover(('acceleration', 1.0), ('acceleration', 2.0)) is None
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match='finite number more than zero'):
            crossover(('acceleration', bad), d)


# ---- writing segments -----------------------------------------------------


def test_a_segment_is_converted_from_the_document_unit_and_convention():
    pp = displacement(0.20, 'in', 'peak to peak')
    assert pp.level == pytest.approx(0.10 * INCH)
    assert displacement(0.20, 'in', 'peak').level == pytest.approx(
        0.20 * INCH)
    assert displacement(5.1, 'mm', 'peak to peak').level == pytest.approx(
        0.00255)
    assert velocity(10.0, 'in/s').level == pytest.approx(0.254)
    assert acceleration(2.0, 'g').level == pytest.approx(2.0 * G)
    assert pp.describe() == '0.2 in peak to peak displacement'


@pytest.mark.parametrize('build,match', [
    (lambda: displacement(0.2, 'in', 'double'), 'say which it is'),
    (lambda: displacement(0.2, 'g', 'peak'),
     "'g' is acceleration, not a length"),
    (lambda: velocity(10.0, 'in'), "'in' is length, not a velocity"),
    (lambda: acceleration(2.0, 'in/s'), 'not a acceleration'),
    (lambda: acceleration(2.0, 'blorbs'), 'no quantity it can read'),
    (lambda: acceleration(np.nan, 'g'), 'finite number more than zero'),
    (lambda: acceleration(-2.0, 'g'), 'finite number more than zero'),
    (lambda: acceleration(2.0, 'g', start=np.nan), 'finite number'),
    (lambda: acceleration(2.0, 'g', start=20.0, end=10.0),
     'starts below where it ends'),
    (lambda: ProfileSegment(quantity='jerk', level=1.0), 'not a quantity'),
])
def test_a_segment_that_cannot_be_read_is_refused_by_name(build, match):
    with pytest.raises(ValueError, match=match):
        build()


# ---- the profile and its check --------------------------------------------


def test_every_crossover_of_the_worked_profile_checks():
    profile = _profile()
    assert profile.checks and profile.continuous
    solved = [crossing.solved for crossing in profile.crossings]
    assert solved == pytest.approx([13.99, 32.96, 52.12], abs=0.01)
    assert [crossing.written for crossing in profile.crossings] == \
        [14.0, 33.0, 52.1]
    assert [round(100.0 * crossing.off, 2) for crossing in profile.crossings] \
        == [0.1, 0.11, -0.04]
    # the profile is drawn on the solved crossovers, so it is continuous
    assert profile.bounds == pytest.approx([5.0, *solved, 500.0])
    assert profile.describe()[-1] == 'every crossover checks'
    assert profile.crossings[0].describe() == \
        'crossover at 13.99 Hz, written 14 Hz: checks'


def test_the_profile_asks_for_what_the_hand_arithmetic_gives():
    profile = _profile()
    f = np.array([5.0, 14.0, 33.0, 52.1, 500.0])
    assert profile.level(f) / G == pytest.approx(
        [0.256, 2.0, 2.004, 4.996, 5.0], abs=1e-3)
    assert profile.level(f, 'velocity') / INCH == pytest.approx(
        [3.14, 8.78, 3.73, 5.89, 0.614], abs=0.01)
    assert profile.level([5.0, 10.0], 'displacement', 'peak to peak') / INCH \
        == pytest.approx([0.20, 0.20])
    assert profile.level([5.0], 'displacement')[0] / INCH == \
        pytest.approx(0.10), 'peak is half the double amplitude'
    assert np.isnan(profile.level([4.9, 500.1])).all(), \
        'outside the band the profile asks for nothing'
    most = profile.maxima()
    assert most['displacement'][0] == 5.0
    assert most['displacement'][1] / INCH == pytest.approx(0.20)
    assert most['velocity'][0] == pytest.approx(13.99, abs=0.01)
    assert most['velocity'][1] / INCH == pytest.approx(8.79, abs=0.01)
    assert most['acceleration'][1] / G == pytest.approx(5.0)


def test_a_frequency_on_a_boundary_belongs_to_the_segment_above():
    profile = _profile()
    edges = profile.bounds
    assert profile.segment_of(edges).tolist() == [0, 1, 2, 3, 3]
    assert profile.segment_of([4.0, 600.0]).tolist() == [-1, -1]


def test_a_double_amplitude_entered_as_peak_does_not_check():
    profile = _profile(first='peak')
    first = profile.crossings[0]
    assert first.solved == pytest.approx(9.89, abs=0.01)
    assert not first.checks and not profile.checks
    assert first.off == pytest.approx(14.0 / first.solved - 1.0)
    # the controller's step at the written breakpoint: 4 g down to 2 g
    assert first.step == pytest.approx(0.5, abs=0.002)
    assert first.describe() == ('crossover at 9.889 Hz, written 14 Hz '
                                '(+41.6 %): does not check, a step of '
                                '0.499 there')
    assert profile.describe()[-1] == '1 crossover does not check'
    assert all(crossing.checks for crossing in profile.crossings[1:])
    with pytest.raises(ValueError, match='the profile does not check: '
                                         'crossover at 9.889 Hz'):
        profile.tone('Up', 2.0)


def test_millimeters_read_as_inches_do_not_meet_inside_the_band():
    with pytest.raises(ValueError, match=r'segment 2 \(2 g acceleration\) '
                                         'would run from 13.99 to 6.557 Hz'):
        _profile(middle=0.91)
    # in millimeters it checks
    assert _profile(middle=0.91, unit='mm').checks


def test_a_looser_tolerance_lets_a_rounded_breakpoint_check():
    rounded = [displacement(0.20, 'in', 'peak to peak', start=5.0, end=14.5),
               acceleration(2.0, 'g', start=14.5, end=100.0)]
    assert not SineProfile(rounded).checks, '3.6 % off at 1 % does not check'
    assert SineProfile(rounded, tolerance=0.05).checks
    assert SineProfile(rounded).tolerance == CROSSOVER_TOLERANCE
    with pytest.raises(ValueError, match='zero or more'):
        SineProfile(rounded, tolerance=np.nan)


def test_crossovers_left_unwritten_are_solved():
    profile = SineProfile([displacement(0.20, 'in', 'peak to peak', start=5.0),
                           acceleration(2.0, 'g'),
                           displacement(0.036, 'in', 'peak to peak'),
                           acceleration(5.0, 'g', end=500.0)])
    assert [c.written for c in profile.crossings] == [None, None, None]
    assert profile.bounds[1:-1] == pytest.approx(
        [c.solved for c in _profile().crossings])
    assert profile.checks
    assert profile.crossings[0].describe() == 'crossover at 13.99 Hz, solved'


def test_two_segments_of_one_quantity_step_and_are_reported():
    profile = SineProfile([acceleration(2.0, 'g', start=10.0, end=50.0),
                           acceleration(4.0, 'g', end=100.0)])
    step = profile.crossings[0]
    assert step.kind == 'step' and step.solved is None
    assert step.step == pytest.approx(2.0)
    assert step.checks, 'a step has no crossover to miss'
    assert not profile.continuous
    assert step.describe() == ('step at 50 Hz: 2 times the acceleration '
                               'below it')
    with pytest.raises(ValueError, match='a sweep cannot jump'):
        profile.breakpoints()
    joined = SineProfile([acceleration(2.0, 'g', start=10.0, end=50.0),
                          acceleration(2.0, 'g', end=100.0)])
    assert joined.continuous
    assert joined.crossings[0].describe() == 'joined at 50 Hz'


@pytest.mark.parametrize('segments,match', [
    ([], 'at least one segment'),
    ([acceleration(2.0, 'g', end=100.0)], 'says where the profile starts'),
    ([acceleration(2.0, 'g', start=10.0)], 'says where the profile ends'),
    ([displacement(0.2, 'in', 'peak', start=5.0, end=14.0),
      acceleration(2.0, 'g', start=15.0, end=100.0)],
     'a boundary is one frequency'),
    ([acceleration(2.0, 'g', start=10.0),
      acceleration(4.0, 'g', end=100.0)],
     'write the frequency where one ends'),
])
def test_a_profile_that_cannot_be_laid_out_is_refused(segments, match):
    with pytest.raises(ValueError, match=match):
        SineProfile(segments)


def test_a_profile_takes_only_segments():
    with pytest.raises(TypeError, match='is not a ProfileSegment'):
        SineProfile([('acceleration', 2.0)])


# ---- as a controller runs it ----------------------------------------------


@pytest.mark.parametrize('law,kind', [('log', LOG), ('linear', LINEAR)])
def test_the_tone_follows_the_profile_within_its_accuracy(law, kind):
    profile = _profile()
    tone = profile.tone('Up', 2.0 if law == 'log' else 1.0, law=law)
    assert set(tone.segment_type.tolist()) == {kind}
    f = np.geomspace(5.0, 500.0, 5000)
    err = tone.target(f)[:, 0] / profile.level(f) - 1.0
    assert np.nanmax(np.abs(err)) <= ACCURACY
    assert np.nanmax(np.abs(err)) > 0.5 * ACCURACY, \
        'as few breakpoints as hold the accuracy, not more'
    # a tighter accuracy writes more breakpoints and holds it
    fine = profile.tone('Up', 2.0, law=law, accuracy=0.001)
    assert len(fine.frequency) > len(tone.frequency)
    err = fine.target(f)[:, 0] / profile.level(f) - 1.0
    assert np.nanmax(np.abs(err)) <= 0.001


def test_a_constant_acceleration_needs_only_its_two_ends():
    profile = SineProfile([acceleration(2.0, 'g', start=20.0, end=2000.0)])
    f, a = profile.breakpoints()
    assert f.tolist() == [20.0, 2000.0]
    assert a == pytest.approx([2.0 * G] * 2)
    # a constant velocity is a straight line on a linear sweep only
    held = SineProfile([velocity(10.0, 'in/s', start=20.0, end=200.0)])
    assert len(held.breakpoints('linear')[0]) == 2
    assert len(held.breakpoints('log')[0]) > 2


def test_the_tone_sweeps_down_when_asked_and_carries_its_channels():
    profile = _profile()
    down = profile.tone('Down', 2.0, descending=True, channels=2,
                        start_time=1.5)
    assert down.frequency[0] == 500.0 and down.frequency[-1] == 5.0
    assert (down.segment_rate < 0.0).all()
    assert down.amplitude.shape == (len(down.frequency), 2)
    assert down.start_time == 1.5
    # 6.64 octaves at 2 oct/min
    assert down.duration() == pytest.approx(np.log2(100.0) / 2.0 * 60.0)
    spec = profile.specification(['1Z+', '2Z+'], 2.0, comment='floor')
    assert spec.ordinate_dim == 'acceleration'
    assert spec.ordinate_unit == 'm/s**2'
    assert spec.response_dof == ['1Z+', '2Z+']
    for bad, match in ((0.0, 'more than zero'), (np.nan, 'more than zero'),
                       (-2.0, 'descending=True')):
        with pytest.raises(ValueError, match=match):
            profile.tone('Up', bad)
    with pytest.raises(ValueError, match='one control channel or more'):
        profile.tone('Up', 2.0, channels=0)
    with pytest.raises(ValueError, match='at least one channel'):
        profile.specification([], 2.0)
    with pytest.raises(ValueError, match='sweep law'):
        profile.breakpoints('exponential')
    with pytest.raises(ValueError, match='more than zero'):
        profile.breakpoints(accuracy=0.0)


# ---- read back from a tone ------------------------------------------------


def test_a_built_tone_reads_back_as_the_profile_it_was_built_from():
    profile = _profile()
    back = read_profile(profile.specification(['1Z+'], 2.0))
    assert [s.quantity for s in back.segments] == \
        [s.quantity for s in profile.segments]
    assert [s.level for s in back.segments] == pytest.approx(
        [s.level for s in profile.segments], rel=1e-9)
    assert back.bounds == pytest.approx(profile.bounds, rel=1e-9)
    assert back.checks
    assert back.segments[0].describe() == \
        '0.00508 m peak to peak displacement'


def _controller(levels_g, frequencies=(5.0, 14.0, 33.0, 52.1, 500.0),
                unit='m/s**2', dim='acceleration', scale=G):
    tone = SineTone('Cat', 0.0, list(frequencies),
                    [[a * scale] for a in levels_g],
                    [LOG] * (len(frequencies) - 1),
                    [2.0] * (len(frequencies) - 1))
    return SineSweepSpecification([tone], ['1Z+'], ordinate_dim=dim,
                                  ordinate_unit=unit)


def test_a_controllers_breakpoints_are_read_as_segments_and_checked():
    spec = _controller([0.26, 2.0, 2.0, 5.0, 5.0])
    profile = read_profile(spec)
    assert [s.quantity for s in profile.segments] == [
        'displacement', 'acceleration', 'displacement', 'acceleration']
    assert profile.checks
    # the breakpoints are the written boundaries
    assert [c.written for c in profile.crossings] == [14.0, 33.0, 52.1]
    # a breakpoint typed at the wrong frequency bends the stretch below
    # it off every quantity, and is refused by name
    typo = _controller([0.26, 2.0, 2.0, 5.0, 5.0],
                       (5.0, 14.0, 33.0, 60.0, 500.0))
    with pytest.raises(ValueError, match='from 33 to 60 Hz the level rises '
                                         'at 9.2 dB per octave'):
        read_profile(typo)


def test_a_slope_that_holds_no_quantity_is_refused_by_name():
    spec = _controller([1.0, 2.0], (10.0, 40.0))       # 3 dB per octave
    with pytest.raises(ValueError, match=r'rises at 3.0 dB per octave, which '
                                         'holds none'):
        read_profile(spec)
    assert len(read_profile(spec, slope_tolerance=3.1).segments) == 1
    with pytest.raises(ValueError, match='zero or more'):
        read_profile(spec, slope_tolerance=-0.1)


def test_a_specification_in_velocity_is_carried_to_acceleration():
    spec = _controller([0.25, 0.25], (10.0, 100.0), unit='m/s',
                       dim='velocity', scale=1.0)
    profile = read_profile(spec)
    assert [s.quantity for s in profile.segments] == ['velocity']
    assert profile.segments[0].level == pytest.approx(0.25)


@pytest.mark.parametrize('spec,kwargs,match', [
    (_controller([1.0, 1.0], (10.0, 20.0), unit=None), {}, 'not declared'),
    (_controller([1.0, 1.0], (10.0, 20.0), dim='force', unit='N',
                 scale=1.0), {}, 'holds force'),
    (_controller([1.0, 1.0], (10.0, 20.0)), {'tone': 'Other'},
     "no tone 'Other'"),
    (_controller([1.0, 1.0], (10.0, 20.0)), {'channel': '9X+'},
     'controls no 9X+'),
    (_controller([1.0, np.nan], (10.0, 20.0)), {}, 'a level of nan'),
    (_controller([1.0, 0.0], (10.0, 20.0)), {}, 'a level of 0'),
    (SineSweepSpecification([SineTone.dwell('Hold', 50.0, [G], 10.0)],
                            ['1Z+'], ordinate_unit='m/s**2'), {},
     'holds a frequency'),
    (SineSweepSpecification([SineTone('Back', 0.0, [10.0, 40.0, 20.0],
                                      [[G]] * 3, [LOG, LOG], [1.0, -1.0])],
                            ['1Z+'], ordinate_unit='m/s**2'), {},
     'turns back'),
])
def test_a_tone_that_is_no_profile_is_refused(spec, kwargs, match):
    with pytest.raises(ValueError, match=match):
        read_profile(spec, **kwargs)


def test_a_tone_down_reads_as_the_same_profile_up():
    up = _profile().specification(['1Z+'], 2.0)
    down = _profile().specification(['1Z+'], 2.0, descending=True)
    assert read_profile(down).bounds == pytest.approx(
        read_profile(up).bounds)


def test_breakpoints_far_apart_part_from_the_profile_between_them():
    spec = _controller([0.2556, 2.0, 2.0, 5.0, 5.0])
    profile = read_profile(spec)
    f, a = interpolated(spec, lines=2000)
    shown = convert(a, f, 'acceleration', 'displacement')
    want = profile.level(f, 'displacement')
    low = f < 14.0
    # a tone interpolates its level linearly in sweep time, so between
    # 5 and 14 Hz it asks for more than the constant displacement
    assert (shown[low][1:-1] > want[low][1:-1]).all()
    # the most the chord of a square law over a ratio r stands above it
    u = np.linspace(0.0, 1.0, 20001)
    chord = np.max((1.0 + (2.8 ** 2 - 1.0) * u) / 2.8 ** (2.0 * u))
    assert np.max(shown[low] / want[low]) == pytest.approx(chord, rel=2e-3)
    assert chord == pytest.approx(1.65, abs=0.01)
    built = _profile().specification(['1Z+'], 2.0)
    f, a = interpolated(built, lines=2000)
    assert np.max(np.abs(a / _profile().level(f) - 1.0)) <= ACCURACY
    with pytest.raises(ValueError, match='two lines or more'):
        interpolated(spec, lines=1)


# ---- the view -------------------------------------------------------------


def test_the_view_marks_each_crossover_by_shape_and_says_it(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.sine_profile import SYMBOLS, build_profile

    layout = pg.GraphicsLayoutWidget()
    plots = build_profile(layout, _profile(first='peak'))
    assert len(plots) == 3
    titles = [plot.titleLabel.text for plot in plots]
    assert titles[0] == 'displacement: most 0.4 in peak to peak at 5 Hz'
    assert titles[1].startswith('velocity: most 12.4 in/s at 9.889 Hz')
    assert titles[2] == 'acceleration: most 5 g at 52.12 Hz'
    names = [label.text for _sample, label in plots[-1].legend.items]
    assert names[0] == 'profile'
    assert ('crossover at 9.889 Hz, written 14 Hz (+41.6 %): does not '
            'check, a step of 0.499 there') in names
    marks = [item for item in plots[-1].listDataItems()
             if item.opts['symbol'] is not None]
    off = [item for item in marks if item.opts['symbol'] == SYMBOLS['off']]
    good = [item for item in marks if item.opts['symbol'] == SYMBOLS['checks']]
    assert len(off) == 1 and off[0].opts['symbolBrush'] is not None, \
        'the crossover that does not check is a filled cross'
    assert len(good) == 2 and all(item.opts['symbolBrush'] is None
                                  for item in good)
    lines = [item for item in plots[0].items
             if isinstance(item, pg.InfiniteLine)]
    # three crossovers, and the written breakpoint beside the one off
    assert sorted(round(10.0 ** line.value(), 2) for line in lines) == \
        [9.89, 14.0, 32.96, 52.12]


def test_the_view_draws_a_tones_own_level_dashed(qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.sine_profile import build_profile

    spec = _controller([0.2556, 2.0, 2.0, 5.0, 5.0])
    layout = pg.GraphicsLayoutWidget()
    plots = build_profile(layout, read_profile(spec),
                          interpolated=interpolated(spec))
    dashed = [item for item in plots[0].listDataItems()
              if item.opts['pen'] is not None
              and pg.mkPen(item.opts['pen']).style() == Qt.PenStyle.DashLine]
    assert len(dashed) == 1
    layout = pg.GraphicsLayoutWidget()
    alone = build_profile(layout, None, interpolated=interpolated(spec))
    assert alone[0].titleLabel.text == 'displacement'
    with pytest.raises(ValueError, match='nothing to draw'):
        build_profile(pg.GraphicsLayoutWidget(), None)


def test_the_plot_draws_to_a_file_with_no_window(tmp_path):
    from visualdynamics.plot import plot_sine_profile

    path = tmp_path / 'profile.png'
    plot_sine_profile(_profile(), path=str(path), show=False)
    assert path.stat().st_size > 5_000
    path = tmp_path / 'spec.png'
    plot_sine_profile(_controller([0.26, 2.0, 2.0, 5.0, 5.0]),
                      path=str(path), show=False, tolerance=0.02)
    assert path.stat().st_size > 5_000
    with pytest.raises(TypeError, match='draws a SineProfile'):
        plot_sine_profile([1, 2, 3], show=False)


# ---- the reading in the window ----------------------------------------------


def _shown(window, pump, spec):
    window.add_object('Sweep', spec)
    select_objects(window, pump, 'Sweep')
    window.render_current()
    pump()
    pane = window.data_pane
    assert pane.sine_profile_action.isVisible(), \
        'offered beside a sine specification'
    pane.sine_profile_action.trigger()
    pump()
    return pane, pane.sine_profile_panel


def test_the_reading_reads_a_specification_as_a_profile(window, pump):
    _pane, panel = _shown(window, pump, _controller([0.26, 2.0, 2.0, 5.0,
                                                     5.0]))
    assert panel.isVisible()
    assert panel.derived['segments'].text() == '4'
    assert panel.derived['crossovers'].text() == '3, every one checks'
    assert panel.derived['displacement'].text() == \
        '0.201 in peak to peak at 5 Hz'
    assert panel.derived['acceleration'].text().startswith('5 g at 52.1')
    assert 'crossover at 13.93 Hz, written 14 Hz: checks' in \
        panel.readout.text()
    assert window.project.journal[-1] == (
        "visualdynamics.plot.plot_sine_profile(project['Sweep'], "
        "tone='Cat', channel=None, slope_tolerance=0.3, interpolated=True, "
        "path='sine_profile.png', show=False)")
    assert 'Cat: 4 segments, every crossover checks' in \
        window.statusBar().currentMessage()


def test_a_slope_tolerance_edit_reads_again(window, pump):
    # the first breakpoint above 14 Hz: 11.5 dB per octave below it
    spec = _controller([0.26, 2.0, 2.0, 5.0, 5.0],
                       (5.0, 14.5, 33.0, 52.1, 500.0))
    _pane, panel = _shown(window, pump, spec)
    assert panel.derived['segments'].text() == '—'
    assert 'from 5 to 14.5 Hz the level rises at 11.5 dB per octave' in         panel.readout.text()
    panel.slope_box.setValue(1.0)
    pump()
    assert panel.derived['segments'].text() == '4'
    assert 'slope_tolerance=1.0,' in window.project.journal[-1]


def test_a_tone_of_no_segments_is_refused_in_the_panel_and_still_drawn(
        window, pump):
    pane, panel = _shown(window, pump, _controller([1.0, 2.0], (10.0, 40.0)))
    assert panel.isVisible()
    assert panel.derived['segments'].text() == '—'
    assert '3.0 dB per octave' in panel.readout.text()
    plots = [item for item in pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    assert len(plots) == 3 and all(plot.listDataItems() for plot in plots)


def test_the_two_readings_of_a_specification_take_turns(window, pump):
    pane, panel = _shown(window, pump, _controller([0.26, 2.0, 2.0, 5.0,
                                                    5.0]))
    pane.sweep_rate_action.trigger()
    pump()
    assert pane.showing('sweep_rate') and not pane.showing('sine_profile')
    assert not panel.isVisible() and pane.sweep_rate_panel.isVisible()
    pane.sine_profile_action.trigger()
    pump()
    assert pane.showing('sine_profile') and not pane.showing('sweep_rate')
    assert panel.isVisible() and not pane.sweep_rate_panel.isVisible()
