"""A shock response spectrum judged point by point against its tolerance.

A shock test is accepted on a band in dB around the required SRS and a
rule about how many of the spectrum's points lie inside it. These pin
that judgement by counts that are known from the construction of each
case (a spectrum that is its requirement, one shifted a known number of
decibels, one with a known number of points moved out), and pin the
case the RMS reading cannot see: a narrow, deep dip that leaves the
RMS deviation well inside its threshold while the points rule fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import compliance
from visualdynamics.core.compliance import (
    SRS_ERROR_DB,
    SRS_INNER_DB,
    srs_errors,
    srs_points,
    srs_tolerance,
    srs_tolerance_all,
)
from visualdynamics.core.data import ShockSpecification, Srs

#: twelfth-octave points from 10 Hz to the last at or below 2 kHz
POINTS = 92


def requirement(**bands):
    """A slope to a knee at 100 Hz and a plateau to 2 kHz, 10 to 100 g
    at Q = 10, on its twelfth-octave points."""
    return ShockSpecification.from_breakpoints(
        [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '101Z+',
        ordinate_unit='g', **bands)


def measured(spec, db=0.0, kind='maximax', q=None, dof='101Z+',
             block=None):
    """The requirement shifted by `db` at every point, or by an array
    of `db`, one per point, as an Srs."""
    values = spec.ordinate[0] * 10.0 ** (np.asarray(db, dtype=float) / 20.0)
    return Srs(spec.abscissa, values[None], response_dof=[dof],
               ordinate_dim=['acceleration'], kind=kind,
               q=spec.q if q is None else q, block=block)


# ---- the four cases the rule is written for ------------------------------


def test_a_spectrum_equal_to_its_requirement_is_wholly_inside():
    spec = requirement()
    result = srs_tolerance(spec, measured(spec))
    assert result['points'] == POINTS
    assert result['inside_count'] == POINTS
    assert result['inside_percent'] == 100.0
    assert result['passed'] is True
    assert result['worst_margin_db'] < 0.0       # inside, with room


def test_two_db_over_everywhere_is_inside_the_inner_band():
    """+2 dB is under the inner band's +3 dB ceiling at every point."""
    spec = requirement()
    result = srs_tolerance(spec, measured(spec, 2.0))
    assert result['inside_count'] == POINTS
    assert result['over_count'] == 0
    assert result['passed'] is True
    assert result['worst_db'] == pytest.approx(2.0)
    assert result['worst_margin_db'] == pytest.approx(-1.0)


def test_two_db_under_everywhere_is_out_at_every_point_and_fails():
    """-2 dB is past the inner band's -1.5 dB floor at every point."""
    spec = requirement()
    result = srs_tolerance(spec, measured(spec, -2.0))
    assert result['inside_count'] == 0
    assert result['under_count'] == POINTS
    assert result['under'].sum() == POINTS
    assert result['passed'] is False
    assert result['worst_db'] == pytest.approx(-2.0)
    assert result['worst_margin_db'] == pytest.approx(0.5)


def test_ten_percent_outside_passes_at_ninety_and_fails_at_ninety_five():
    """Forty points, four of them moved 2 dB under: exactly 90 percent
    inside, which meets a 90 percent rule and not a 95."""
    f = np.geomspace(10.0, 2000.0, 40)
    level = np.full(40, 100.0)
    shifted = level.copy()
    shifted[[3, 11, 22, 37]] *= 10.0 ** (-2.0 / 20.0)
    at_ninety = srs_points(f, shifted, f, level, percent=90.0)
    assert at_ninety['inside_count'] == 36
    assert at_ninety['inside_percent'] == pytest.approx(90.0)
    assert at_ninety['passed'] is True
    assert srs_points(f, shifted, f, level, percent=95.0)['passed'] is False
    # and one point fewer inside is under ninety
    shifted[5] *= 10.0 ** (-2.0 / 20.0)
    assert srs_points(f, shifted, f, level, percent=90.0)['passed'] is False


def test_a_narrow_deep_dip_passes_the_rms_reading_and_fails_the_points_rule():
    """Ten neighboring points 5 dB under, the rest on the requirement:
    the signed RMS is under 2 dB, inside the 3 dB the deviation bars
    shade at, while ten of 92 points out is 89 percent inside, under
    the 90 percent rule. The two readings answer different questions,
    and only the count is the tolerance check."""
    spec = requirement()
    db = np.zeros(POINTS)
    db[40:50] = -5.0
    dipped = measured(spec, db)
    (_dof, _event, rms_db), = srs_errors(spec, dipped)
    assert abs(rms_db) < SRS_ERROR_DB
    assert rms_db == pytest.approx(-np.sqrt(10 * 25.0 / POINTS))
    result = srs_tolerance(spec, dipped)
    assert result['inside_count'] == POINTS - 10
    assert result['passed'] is False
    assert result['worst_db'] == pytest.approx(-5.0)
    assert 40 <= np.flatnonzero(spec.abscissa == result['worst_frequency'])[0] < 50


# ---- the edges of the rule ------------------------------------------------


def test_a_point_on_a_bands_edge_is_inside():
    """+3 dB and -1.5 dB exactly, as a spectrum written as the
    requirement times the band's factor reads in floating point."""
    spec = requirement()
    for db in SRS_INNER_DB:
        assert srs_tolerance(spec, measured(spec, db))['inside_count'] == POINTS


def test_a_point_on_the_outer_bands_edge_is_inside_it():
    spec = requirement()
    db = np.zeros(POINTS)
    db[5], db[50] = -3.0, 6.0
    result = srs_tolerance(spec, measured(spec, db), outer=(-3.0, 6.0))
    assert result['outer_count'] == 0
    assert result['inside_count'] == POINTS - 2
    assert result['passed'] is True


def test_the_outer_band_holds_every_point_whatever_the_share():
    """91 of 92 inside the inner band, and the one outside past the
    outer band: the share is met and the spectrum still fails."""
    spec = requirement()
    db = np.zeros(POINTS)
    db[60] = 7.0
    result = srs_tolerance(spec, measured(spec, db), outer=(-3.0, 6.0))
    assert result['inside_count'] == POINTS - 1
    assert result['outer_count'] == 1
    assert result['beyond_outer'][60]
    assert result['passed'] is False
    assert srs_tolerance(spec, measured(spec, db))['passed'] is True


def test_the_above_lower_rule_puts_no_ceiling_on_the_band():
    """+5 dB everywhere is above the inner band and fails a share of
    it, yet every point is above the lower edge; one point under the
    lower edge fails that rule outright."""
    spec = requirement()
    high = measured(spec, 5.0)
    assert srs_tolerance(spec, high)['passed'] is False
    above = srs_tolerance(spec, high, rule='above_lower')
    assert above['passed'] is True
    assert above['inside_count'] == 0         # still not inside the band
    db = np.zeros(POINTS)
    db[0] = -1.6
    one_under = srs_tolerance(spec, measured(spec, db), rule='above_lower')
    assert one_under['passed'] is False
    assert one_under['worst_frequency'] == pytest.approx(10.0)
    assert one_under['worst_margin_db'] == pytest.approx(0.1)


def test_a_share_above_the_requirement_is_the_share_rule_from_zero_up():
    """Half the points above the requirement, read as the share rule
    with a band from 0 dB up."""
    spec = requirement()
    db = np.where(np.arange(POINTS) % 2 == 0, 0.5, -0.5)
    result = srs_tolerance(spec, measured(spec, db), inner=(0.0, np.inf),
                           percent=50.0)
    assert result['inside_count'] == POINTS // 2
    assert result['passed'] is True
    assert srs_tolerance(spec, measured(spec, -0.5), inner=(0.0, np.inf),
                         percent=50.0)['passed'] is False


def test_the_worst_point_is_the_farthest_outside_not_the_farthest_away():
    """Under -1.5/+3 dB, a point 2.5 dB over is inside and one 2 dB
    under is not; the worst is the one under."""
    spec = requirement()
    db = np.zeros(POINTS)
    db[10] = 2.5
    db[70] = -2.0
    result = srs_tolerance(spec, measured(spec, db))
    assert result['worst_frequency'] == pytest.approx(spec.abscissa[70])
    assert result['worst_db'] == pytest.approx(-2.0)


def test_only_points_inside_the_requirement_are_judged():
    """A measurement that starts a decade below the requirement and
    runs past its end is counted only where the requirement is
    written; the points outside are neither in nor out."""
    f = np.geomspace(1.0, 5000.0, 60)
    level = np.full(60, 100.0)
    level[f < 10.0] = 1e-3              # far under, but not required
    result = srs_points(f, level, [10.0, 2000.0], [100.0, 100.0])
    written = (f >= 10.0) & (f <= 2000.0)
    assert result['points'] == int(written.sum())
    assert np.array_equal(result['judged'], written)
    assert result['passed'] is True
    assert np.isnan(result['db'][~written]).all()


def test_a_point_measured_at_nothing_is_not_judged():
    """As the RMS reading skips it: a zero peak is no reading at all,
    not a point an infinite number of decibels under."""
    f = np.geomspace(10.0, 2000.0, 20)
    level = np.full(20, 100.0)
    got = level.copy()
    got[7] = 0.0
    result = srs_points(f, got, f, level)
    assert result['points'] == 19
    assert not result['judged'][7]
    assert result['under_count'] == 0


def test_the_whole_range_of_shares_can_be_asked_for():
    """Every point (100 percent) and none (0 percent) are both rules."""
    f = np.geomspace(10.0, 2000.0, 20)
    level = np.full(20, 100.0)
    got = level.copy()
    got[3] = 50.0
    assert srs_points(f, level, f, level, percent=100.0)['passed'] is True
    assert srs_points(f, got, f, level, percent=100.0)['passed'] is False
    assert srs_points(f, got / 10.0, f, level, percent=0.0)['passed'] is True


def test_nothing_to_judge_gives_no_verdict():
    result = srs_points([1.0, 2.0], [1.0, 1.0], [10.0, 20.0], [1.0, 1.0])
    assert result['points'] == 0
    assert result['passed'] is None
    assert np.isnan(result['worst_db'])


@pytest.mark.parametrize('kwargs, message', [
    ({'rule': 'majority'}, 'not a rule'),
    ({'inner': (3.0, -1.5)}, 'not a band'),
    ({'inner': (0.0, 0.0)}, 'not a band'),
    ({'percent': -1.0}, 'not a share'),
    ({'outer': (6.0,)}, 'dB below, dB above'),
    ({'percent': 120.0}, 'not a share'),
])
def test_a_rule_or_band_that_cannot_be_read_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        srs_points([10.0, 20.0], [1.0, 1.0], [10.0, 20.0], [1.0, 1.0],
                   **kwargs)


# ---- the objects ------------------------------------------------------------


def test_a_spectrum_at_another_q_is_not_judged():
    spec = requirement()
    result = srs_tolerance(spec, measured(spec, q=50.0))
    assert result['points'] == 0
    assert result['passed'] is None
    assert 'Q=50' in result['refused']


def test_each_polarity_is_judged_as_it_is():
    """A maximax requirement, a positive spectrum on it and a negative
    one 2 dB under: judged one at a time, each keeps its own kind."""
    spec = requirement()
    positive = srs_tolerance(spec, measured(spec, kind='positive'))
    negative = srs_tolerance(spec, measured(spec, -2.0, kind='negative'))
    assert (positive['kind'], positive['passed']) == ('positive', True)
    assert (negative['kind'], negative['passed']) == ('negative', False)


def test_every_channel_and_event_is_judged_as_the_rms_reading_pairs_them():
    spec = requirement()
    values = np.vstack([spec.ordinate[0], spec.ordinate[0] * 10 ** -0.1,
                        spec.ordinate[0]])
    run = Srs(spec.abscissa, values, response_dof=['101Z+', '101Z+', '999X+'],
              ordinate_dim='acceleration', block=['shock 1', 'shock 2',
                                                  'shock 1'])
    rows = srs_tolerance_all(spec, run)
    assert [(dof, event) for dof, event, _r in rows] == [
        ('101Z+', 'shock 1'), ('101Z+', 'shock 2')]
    assert [(dof, event) for dof, event, _db in srs_errors(spec, run)] == [
        (dof, event) for dof, event, _r in rows]
    assert [r['passed'] for _dof, _event, r in rows] == [True, False]
    assert rows[1][2]['event'] == 'shock 2'


def test_a_refused_pair_is_reported_rather_than_dropped():
    """A run at the wrong Q is a row that says why, not a channel that
    silently vanished from the judgement."""
    spec = requirement()
    (_dof, _event, result), = srs_tolerance_all(spec,
                                                measured(spec, q=50.0))
    assert 'refused' in result
    assert srs_tolerance_all(spec, measured(spec, dof='202X+')) == []


# ---- a requirement from breakpoints --------------------------------------


def test_a_requirement_is_read_onto_its_spacings_points():
    """Twelfth-octave points from the first breakpoint, the last at or
    under the last breakpoint, on the power law between breakpoints."""
    spec = requirement()
    f = spec.abscissa
    assert spec.num_records == 1
    assert f.size == POINTS
    assert f[0] == 10.0
    assert f[-1] <= 2000.0 < f[-1] * 2.0 ** (1.0 / 12.0)
    assert np.allclose(np.log2(f[1:] / f[:-1]), 1.0 / 12.0)
    g = 9.80665
    at = np.argmin(np.abs(f - 100.0))
    assert spec.ordinate[0, 0] == pytest.approx(10.0 * g)
    assert spec.ordinate[0, at] == pytest.approx(100.0 * g)
    # half a decade up the slope from 10 to 100 g over 10 to 100 Hz
    # is the geometric mean, 31.6 g at 31.6 Hz
    mid = 10.0 * 2.0 ** (20 / 12)
    assert spec.ordinate[0, 20] == pytest.approx(10.0 * g * mid / 10.0)


def test_a_band_of_whole_octaves_ends_on_its_last_breakpoint():
    """Seven octaves at twelve to the octave: 85 points, the last on
    2560 Hz itself and written, not a rounding past it."""
    spec = ShockSpecification.from_breakpoints([20.0, 2560.0], [1.0, 1.0],
                                               '1X+')
    assert spec.abscissa.size == 85
    assert spec.abscissa[-1] == 2560.0
    assert np.isfinite(spec.ordinate).all()


def test_a_requirement_carries_its_q_type_unit_and_spacing():
    spec = ShockSpecification.from_breakpoints(
        [20.0, 2000.0], [5.0, 5.0], ['1X+', '1Y+'], per_octave=6, q=20.0,
        kind='positive', ordinate_unit='g')
    assert (spec.q, spec.kind) == (20.0, 'positive')
    assert spec.ordinate_unit == ['g', 'g']
    assert spec.response_dof == ['1X+', '1Y+']
    assert np.array_equal(spec.ordinate[0], spec.ordinate[1])
    assert np.allclose(np.log2(spec.abscissa[1:] / spec.abscissa[:-1]), 1 / 6)
    assert 'Q=20' in spec.comment[0] and '6 to the octave' in spec.comment[0]


def test_a_requirements_bands_become_its_limits():
    spec = requirement(warning=(-1.5, 3.0), abort=(-3.0, 6.0))
    target = spec.ordinate
    for name, db in (('warning_lower', -1.5), ('warning_upper', 3.0),
                     ('abort_lower', -3.0), ('abort_upper', 6.0)):
        assert np.allclose(spec.limits[name], target * 10.0 ** (db / 20.0))
    assert requirement().limits == {}


@pytest.mark.parametrize('frequencies, levels, message', [
    ([10.0], [1.0], 'two breakpoints'),
    ([100.0, 10.0], [1.0, 1.0], 'increasing'),
    ([10.0, 10.0], [1.0, 1.0], 'increasing'),
    ([0.0, 10.0], [1.0, 1.0], 'positive and increasing'),
    ([10.0, 100.0], [1.0, 0.0], 'positive'),
    ([10.0, 100.0], [1.0, 1.0, 1.0], 'levels for'),
])
def test_a_requirement_that_cannot_be_drawn_is_refused(frequencies, levels,
                                                       message):
    with pytest.raises(ValueError, match=message):
        ShockSpecification.from_breakpoints(frequencies, levels, '1X+')


def test_the_judgement_lives_beside_the_rms_reading():
    """One module holds both readings of a shock against its
    requirement, and the RMS one is unchanged by the second."""
    assert compliance.srs_tolerance is srs_tolerance
    assert compliance.SRS_INNER_DB == (-1.5, 3.0)
    assert compliance.SRS_POINTS_PERCENT == 90.0

