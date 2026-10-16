"""Whether a sine sweep is slow enough for the modes it passes.

A mode needs about Q cycles at its own frequency to build to its full
response. A sweep does not hold it there: it gives the mode a short time
inside its half-power band (the band ``fn/Q`` wide around ``fn``, where
the response is within 3 dB of its peak) and moves on. When that time
holds fewer than about Q of the mode's cycles, the swept response reads
the peak low, places it late in the direction of the sweep, and spreads
it, so a Q read from its half-power width comes out low as well. All
three at once, and all three are properties of the rate, not of the
article.

The arithmetic is short. On a logarithmic sweep at ``R`` octaves per
minute the band, ``1/(Q ln 2)`` octaves wide, is crossed in
``60 / (ln 2 Q R)`` seconds, about ``86.6 / (Q R)``, whatever the mode's
frequency; the mode completes ``fn`` times as many cycles in that time.
Setting those cycles equal to the Q the mode needs gives the rate at
which a sweep only just lets it build:

    R = (60 / ln 2) fn / Q**2,  about 87 fn / Q**2 oct/min.

A linear sweep at ``r`` Hz/s crosses the band, ``fn/Q`` Hz wide, in
``fn / (Q r)`` seconds, and the same equation gives ``r = fn**2 / Q**2``
Hz/s. At the rule's rate a mode still reads only about 95 % of its
steady height, so the rule is a ceiling to stay well under, not a
target. That is the reason the rule here flags a rate above it rather
than near it, and the reason each check also reports what a mode of
that Q would actually read at that rate.

What a mode reads at a rate is found by simulating it: the one-mass
resonator (`srs.ramp_invariant`, the shock spectrum's filter) driven
through the sweep at the segment's rate and law, its response read once
per cycle of the drive as that cycle's rms reported as the peak of a
sine (the rms detector a controller uses), and the peak and the
half-power width of that swept curve read the way they would be read
off a plot. The simulation is made at a natural frequency of 1 Hz,
since a sweep's effect on a mode depends on the rate only through
``R / fn`` (log) or ``r / fn**2`` (linear), so its cost does not grow
with the mode's frequency.

`check_sweep_rate` takes a `SineSweepSpecification` (or one `SineTone`)
and a list of modes, each a natural frequency and a Q, and returns one
`SweepRateCheck` per pass of a tone through a mode's band: the time and
the cycles inside the band, the rule's rate for that mode, the rate as
a multiple of it, the swept reading, and whether it is too fast. A tone
swept up and back passes each mode twice and gives two rows. A mode
whose band no tone reaches gives none.

`swept_q` and `read_swept_q` answer the reverse question, for a Q read
off a level that was itself swept (a tracked sine level, the usual
source of a Q in a sine test): whether the rate was slow enough for the
Q it reads, and what Q a mode would need to read that low at that rate.

References
----------
The effect is a standard result of the response of a linear oscillator
to a tone of changing frequency; these are where it is set out, and
where the filter the simulation uses comes from.

1. Lewis, F. M. (1932). "Vibration During Acceleration Through a
   Critical Speed." *Transactions of the American Society of
   Mechanical Engineers*. The peak reached by an oscillator swept
   through its resonance, lower and later the faster it is swept.
2. Cronin, D. L. (1965). *Response of Linear, Viscous Damped Systems to
   Excitations Having Time-Varying Frequency.* Doctoral thesis. The
   same response worked out
   for linear and logarithmic sweeps, with the shift of the peak and
   the spread of its width.
3. Smallwood, D. O. (1981). "An Improved Recursive Formula for
   Calculating Shock Response Spectra." *Shock and Vibration
   Bulletin*. The ramp-invariant recursion the one-mass simulation is
   run through.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .sine import LINEAR, LOG, SineSweepSpecification, SineTone

#: 60 / ln 2, the seconds a log sweep at one octave per minute takes to
#: cross a band one Q-th of its frequency wide, times Q: the 86.6 in
#: ``86.6 / (Q R)`` and the 87 in the rule ``87 fn / Q**2``
RULE_CONSTANT = 60.0 / np.log(2.0)

#: the rate, as a multiple of the rule's, above which a check is
#: flagged too fast. At the rule's own rate a mode reads about 95 % of
#: its steady height, so 1 already allows a 5 % under-read
MARGIN = 1.0

#: below this share of the rule's rate the sweep reads the steady peak
#: to a fraction of a percent, and the reading is taken as steady
#: rather than simulated. A 22 Hz, Q 40 mode swept at 0.06 of the rule's
#: rate reads 99.85 % of its height and a Q 0.3 % low; the simulation's
#: length grows as the rate falls, so this is where it stops paying
#: (`tests/test_sweep_rate.py` pins both)
STEADY_RATIO = 0.05

#: drive samples per cycle of the mode in the simulation. The
#: ramp-invariant filter reads a steady tone low when it is sampled
#: under about twenty times per cycle; at 64 the steady height reads
#: 0.08 % low (pinned in the tests)
SAMPLES_PER_CYCLE = 64

#: half-power widths the simulated sweep runs before the band and after
#: it, at the least. The response starts from rest, and the transient
#: of starting decays by e**(-pi) every Q cycles, so the run-in is also
#: held to `RUN_IN_Q` times Q cycles; the run-out has to hold the
#: swept peak, which lags the mode by about twice the rate's multiple
#: of the rule in half-widths
WINDOW_WIDTHS = 20.0
RUN_IN_Q = 4.0


def _positive(value: Any, what: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'{what} must be a positive number; got {value}')
    return value


def _q(value: Any) -> float:
    q = _positive(value, 'Q')
    if q <= 0.5:
        # Q = 0.5 is critical damping: no resonance, no half-power band
        raise ValueError(f'a Q of {q} is at or past critical damping '
                         '(0.5); there is no resonance to sweep through')
    return q


def half_power_band(frequency: float, q: float) -> tuple[float, float]:
    """The mode's half-power band, ``fn (1 - 1/(2Q))`` to
    ``fn (1 + 1/(2Q))`` in Hz: ``fn/Q`` wide, centred on the mode.

    The band the sweep's time and cycles are counted in. It is the
    width the mode's Q defines, not the crossings of any measured
    curve, which a fast sweep moves.

    Parameters
    ----------
    frequency : float
        The natural frequency, Hz.
    q : float
        The mode's Q, ``1/(2 zeta)``.

    Returns
    -------
    tuple of float
        (lower, upper) in Hz.
    """
    frequency, q = _positive(frequency, 'a frequency'), _q(q)
    half = frequency / (2.0 * q)
    return frequency - half, frequency + half


def rule_rate(frequency: float, q: float,
              segment_type: int = LOG) -> float:
    """The rule's rate for a mode: the sweep rate at which the time in
    its half-power band holds just the Q cycles it needs to build.

    ``(60 / ln 2) fn / Q**2`` oct/min for a logarithmic sweep (about
    ``87 fn / Q**2``), and ``fn**2 / Q**2`` Hz/s for a linear one. A
    sweep should run well below it: at this rate a mode reads about 95 %
    of its steady height.

    Parameters
    ----------
    frequency : float
        The natural frequency, Hz.
    q : float
        The mode's Q.
    segment_type : int
        `sine.LOG` or `sine.LINEAR`: which unit the rate is given in.

    Returns
    -------
    float
        oct/min for LOG, Hz/s for LINEAR; always positive.
    """
    frequency, q = _positive(frequency, 'a frequency'), _q(q)
    if int(segment_type) == LINEAR:
        return frequency ** 2 / q ** 2
    return RULE_CONSTANT * frequency / q ** 2


def rate_unit(segment_type: int) -> str:
    """'oct/min' for a logarithmic segment, 'Hz/s' for a linear one: the
    units the specification writes each rate in."""
    return 'Hz/s' if int(segment_type) == LINEAR else 'oct/min'


# ---- the simulated reading ---------------------------------------------


@dataclass(frozen=True)
class SweptReading:
    """What a sweep at one rate makes of one mode: `simulate_sweep`'s
    result.

    `peak_frequency` is where the swept peak sits, Hz; `peak_share` is
    its height over the mode's steady height at ``fn``,
    ``Q sqrt(1 + 1/Q**2)`` (NaN where the sweep passed the mode in too
    few cycles to read a peak at all); `q_read` is the peak frequency over the
    swept curve's half-power width, None where the curve did not fall
    to 0.707 of its peak inside the simulated stretch. `simulated` is
    False where the rate was under `STEADY_RATIO` of the rule's and the
    steady values were taken instead.
    """

    frequency: float
    q: float
    peak_frequency: float
    peak_share: float
    q_read: float | None
    simulated: bool = True

    @property
    def under_read(self) -> float:
        """How far the swept peak is under the steady one, a fraction:
        0.19 for a peak at 81 % of its height."""
        return 1.0 - self.peak_share

    @property
    def q_bias(self) -> float | None:
        """How far the Q read is from the mode's, a fraction: -0.46 for
        a Q of 40 read as 21.6. None where no Q could be read."""
        return None if self.q_read is None else self.q_read / self.q - 1.0


def _read_peak(f: np.ndarray, level: np.ndarray,
               candidates: np.ndarray) -> tuple[float, float, float | None]:
    """(frequency, height, Q) of the tallest of `candidates` on a curve
    given in sweep order: the top placed by a parabola through the log
    of it and its two neighbors, the half-power crossings found by
    walking out along the sweep and interpolating in frequency. Q is
    None where a side never falls to 0.707 of the top."""
    # The half-power reading of the curve as the eye would make it. A
    # half-power reader of any curve is a separate contribution; this
    # one reads only the swept curves here, which are in sweep order
    # (not sorted by frequency), one reading per cycle.
    i = int(candidates[np.argmax(level[candidates])])
    if i == 0 or i == len(level) - 1:
        return float(f[i]), float(level[i]), None
    a, b, c = np.log(level[i - 1:i + 2])
    bend = a - 2.0 * b + c
    p = 0.5 * (a - c) / bend if bend < 0.0 else 0.0
    top = float(f[i] + p * (f[i + 1] - f[i - 1]) / 2.0)
    height = float(np.exp(b - 0.25 * (a - c) * p))
    half = height / np.sqrt(2.0)

    def crossing(step):
        k = i
        while 0 <= k + step < len(level):
            k += step
            if level[k] < half:
                back = k - step
                return float(np.interp(half, [level[k], level[back]],
                                       [f[k], f[back]]))
        return None

    before, after = crossing(-1), crossing(+1)
    if before is None or after is None or after == before:
        return top, height, None
    return top, height, abs(top / (after - before))


@lru_cache(maxsize=512)
def _simulated(q: float, normalized: float, segment_type: int,
               down: bool) -> tuple[float, float, float | None, bool]:
    """The sweep through a 1 Hz mode: (peak frequency, share, Q read,
    simulated). `normalized` is the rate over the mode's frequency
    (log, oct/min per Hz) or over its square (linear, Hz/s per Hz**2),
    so one result serves every mode at the same ratio. Cached, because
    the window redraws on every edit and asks again."""
    from scipy.signal import lfilter

    from .srs import damping_for, ramp_invariant

    linear = segment_type == LINEAR
    rule = 1.0 / q ** 2 if linear else RULE_CONSTANT / q ** 2
    ratio = normalized / rule
    steady = q * np.sqrt(1.0 + 1.0 / q ** 2)
    if ratio < STEADY_RATIO:
        return 1.0, 1.0, q, False
    half = 0.5 / q
    widths = max(WINDOW_WIDTHS, 8.0 * ratio)
    near, far = 1.0 - widths * half, 1.0 + widths * half
    # the run-in must also be long enough, in cycles, for the start
    # from rest to have rung away; at a fast rate twenty half-widths
    # pass in a handful of cycles
    if linear:
        reach = RUN_IN_Q * q * normalized        # cycles ~ df / rate
        before, after = 1.0 - reach, 1.0 + reach
    else:
        kappa = normalized * np.log(2.0) / 60.0  # per natural period
        reach = RUN_IN_Q * q * kappa             # ln f over those cycles
        before, after = np.exp(-reach), np.exp(reach)
    if down:
        far = max(far, after)
    else:
        near = min(near, before)
    near, far = max(near, 0.5), min(far, 2.0)
    start, end = (far, near) if down else (near, far)
    if linear:
        slope = normalized if not down else -normalized
        duration = (end - start) / slope
    else:
        duration = np.log(end / start) / (kappa if not down else -kappa)
    n = int(np.ceil(duration * SAMPLES_PER_CYCLE)) + 1
    tau = np.arange(n) / SAMPLES_PER_CYCLE
    if linear:
        f = start + slope * tau
        cycles = start * tau + slope * tau ** 2 / 2.0
    else:
        k = kappa if not down else -kappa
        f = start * np.exp(k * tau)
        cycles = start * np.expm1(k * tau) / k
    drive = np.sin(2.0 * np.pi * cycles)
    b, a = ramp_invariant([1.0], SAMPLES_PER_CYCLE, damping_for(q))
    response = lfilter(b[0], a[0], drive)
    # one reading per cycle of the drive, as each cycle's rms reported
    # as a sine's peak, against the cycle's mean frequency: the rms
    # detector, read once a cycle
    edges = np.r_[0, np.flatnonzero(np.diff(np.floor(cycles))) + 1]
    count = np.diff(np.r_[edges, n])
    level = np.sqrt(2.0 * np.add.reduceat(response ** 2, edges) / count)
    centers = np.add.reduceat(f, edges) / count
    # the last cycle is cut short by the end of the run; leave it out
    level, centers = level[:-1], centers[:-1]
    # the peak is looked for near the mode, never in the start's ringing
    candidates = np.flatnonzero(np.abs(centers - 1.0) < (widths - 1.0) * half)
    if candidates.size < 3:
        # a sweep through an octave either side in a few cycles leaves
        # nothing to read a peak from; NaN, since zero would read as a
        # mode that did not respond
        return 1.0, float('nan'), None, True
    top, height, q_read = _read_peak(centers, level, candidates)
    return (float(top), float(height / steady),
            None if q_read is None else float(q_read), True)


def simulate_sweep(frequency: float, q: float, rate: float,
                   segment_type: int = LOG) -> SweptReading:
    """What a mode reads when a sweep passes it at `rate`, by
    simulation of the one-mass resonator.

    The resonator at `frequency` with `q` is driven from rest by a unit
    tone swept through its band at the constant `rate` and law
    (`segment_type`); a negative rate sweeps down. Its response is read
    once per cycle of the drive, each cycle's rms reported as the peak
    of a sine, and the swept curve's peak and half-power width are read
    from those readings. A 22 Hz, Q 40 mode swept up at 1, 4 and
    8 oct/min reads 95, 81 and 71 % of its steady height, with a Q of
    about 33, 22 and 17.

    Parameters
    ----------
    frequency : float
        The natural frequency, Hz.
    q : float
        The mode's Q.
    rate : float
        oct/min for a LOG sweep, Hz/s for a LINEAR one; negative sweeps
        down.
    segment_type : int
        `sine.LOG` or `sine.LINEAR`.

    Returns
    -------
    SweptReading
        The peak's frequency, its height as a share of the steady
        height, and the Q read from its width.
    """
    frequency, q = _positive(frequency, 'a frequency'), _q(q)
    rate = float(rate)
    if not np.isfinite(rate) or rate == 0.0:
        raise ValueError(f'a sweep rate must be a nonzero number; got {rate}')
    linear = int(segment_type) == LINEAR
    normalized = abs(rate) / (frequency ** 2 if linear else frequency)
    # rounded so a rate typed twice is one cache entry, far finer than
    # anything the reading can tell apart
    top, share, q_read, simulated = _simulated(
        round(q, 9), float(f'{normalized:.9g}'),
        LINEAR if linear else LOG, rate < 0.0)
    return SweptReading(frequency=frequency, q=q,
                        peak_frequency=top * frequency, peak_share=share,
                        q_read=q_read, simulated=simulated)


# ---- the check against a specification ----------------------------------


@dataclass(frozen=True)
class SweepRateCheck:
    """One pass of one tone through one mode's half-power band:
    `check_sweep_rate`'s row.

    `rate` is the specification's rate on the segment at the mode, as
    written (oct/min for a log segment, Hz/s for a linear one, negative
    sweeping down); `limit` is the rule's rate for the mode in the same
    unit. `start` is when the sweep enters the band, in seconds from
    the tone's start; `seconds` and `cycles` are how long it stays and
    how many of the drive's cycles that is. `crossed` is False where the
    sweep starts, ends or turns inside the band, so the time counted is
    not a full crossing. `reading` is what a mode of this Q reads at
    this rate (`simulate_sweep`).
    """

    tone: str
    frequency: float
    q: float
    segment_type: int
    rate: float
    start: float
    seconds: float
    cycles: float
    crossed: bool
    limit: float
    reading: SweptReading
    margin: float = MARGIN

    @property
    def unit(self) -> str:
        """'oct/min' or 'Hz/s', the unit of `rate` and `limit`."""
        return rate_unit(self.segment_type)

    @property
    def direction(self) -> str:
        """'up' or 'down'."""
        return 'up' if self.rate > 0.0 else 'down'

    @property
    def ratio(self) -> float:
        """The rate as a multiple of the rule's rate for this mode."""
        return abs(self.rate) / self.limit

    @property
    def cycles_needed(self) -> float:
        """The cycles the mode needs to build: about Q."""
        return self.q

    @property
    def too_fast(self) -> bool:
        """Whether the rate is above `margin` times the rule's."""
        return bool(self.ratio > self.margin)

    @property
    def peak_share(self) -> float:
        """The swept peak's height over the steady height."""
        return self.reading.peak_share

    @property
    def q_read(self) -> float | None:
        """The Q a swept reading of this mode gives at this rate."""
        return self.reading.q_read

    @property
    def verdict(self) -> str:
        """'too fast' or 'builds', the word the table and the plot use."""
        return 'too fast' if self.too_fast else 'builds'

    def describe(self) -> str:
        """The check in a line, as the window's status bar gives it:
        '22 Hz, Q 40, up at 4 oct/min: 0.54 s and 12 cycles in the
        band (40 to build), 3.3 times the rule's 1.2 oct/min; reads 81 %
        of the peak and a Q of 21.6; too fast'."""
        q_read = ('no Q' if self.q_read is None
                  else f'a Q of {self.q_read:.3g}')
        peak = ('no peak' if not np.isfinite(self.peak_share)
                else f'{100.0 * self.peak_share:.0f} % of the peak')
        part = '' if self.crossed else ' (not a full crossing)'
        return (f'{self.frequency:.4g} Hz, Q {self.q:.3g}, '
                f'{self.direction} at {abs(self.rate):.3g} {self.unit}: '
                f'{self.seconds:.3g} s and {self.cycles:.3g} cycles in the '
                f'band{part} ({self.q:.3g} to build), {self.ratio:.2g} times '
                f'the rule\'s {self.limit:.3g} {self.unit}; reads '
                f'{peak} and {q_read}; {self.verdict}')


def _segment_table(tone: SineTone):
    """(start s, length s, f0, f1, type, rate) for each segment."""
    rows, start = [], 0.0
    for i, length in enumerate(tone.segment_seconds()):
        rows.append((start, float(length), float(tone.frequency[i]),
                     float(tone.frequency[i + 1]),
                     int(tone.segment_type[i]), float(tone.segment_rate[i])))
        start += float(length)
    return rows


def _time_at(segment, f):
    """When a segment's sweep is at frequency `f`, s from the tone's
    start: the inverse of its law."""
    start, length, f0, f1, kind, _rate = segment
    if kind == LINEAR:
        fraction = (f - f0) / (f1 - f0)
    else:
        fraction = np.log(f / f0) / np.log(f1 / f0)
    return start + length * float(fraction)


def _passages(tone: SineTone, low: float, high: float):
    """Each stretch of the tone inside [low, high] Hz, as
    (entry s, exit s, entry Hz, exit Hz, segments): the segments of
    one stretch are consecutive, the sweep staying in the band across
    the breakpoints between them."""
    out = []
    current = None
    for segment in _segment_table(tone):
        start, length, f0, f1, _kind, _rate = segment
        lo, hi = max(low, min(f0, f1)), min(high, max(f0, f1))
        if lo >= hi:
            current = None
            continue
        enter, leave = (lo, hi) if f1 > f0 else (hi, lo)
        t_in, t_out = _time_at(segment, enter), _time_at(segment, leave)
        joins = (current is not None
                 and abs(current[1] - start) <= 1e-9 * max(1.0, start)
                 and abs(t_in - start) <= 1e-9 * max(1.0, start))
        if joins:
            current[1] = t_out
            current[3] = leave
            current[4].append(segment)
        else:
            current = [t_in, t_out, enter, leave, [segment]]
            out.append(current)
        if abs(t_out - (start + length)) > 1e-9 * max(1.0, start + length):
            # left the band inside this segment: the next one starts
            # a new stretch even if it comes straight back
            current = None
    return [tuple(passage) for passage in out]


def _at_mode(segments, frequency):
    """The segment of a stretch whose span holds the mode; failing
    that (the sweep stops short of it), the one longest in the band."""
    for segment in segments:
        f0, f1 = segment[2], segment[3]
        if min(f0, f1) <= frequency <= max(f0, f1):
            return segment
    return max(segments, key=lambda segment: segment[1])


def _modes(modes: Any) -> list[tuple[float, float]]:
    """(frequency, Q) pairs from pairs or from a `ShapeSet` (its
    damping is a fraction of critical; rigid-body and undamped modes,
    which have no band, are left out)."""
    if hasattr(modes, 'frequency') and hasattr(modes, 'damping'):
        return [(float(f), 0.5 / float(z))
                for f, z in zip(modes.frequency, modes.damping)
                if f > 0.0 and z > 0.0]
    out = []
    for mode in modes:
        frequency, q = mode
        out.append((_positive(frequency, 'a frequency'), _q(q)))
    return out


def _tones(specification: Any, tone: str | None) -> list[SineTone]:
    if isinstance(specification, SineTone):
        if tone is not None and tone != specification.name:
            raise KeyError(f'the tone given is {specification.name!r}, '
                           f'not {tone!r}')
        return [specification]
    if not isinstance(specification, SineSweepSpecification):
        raise TypeError('a sweep rate is checked against a '
                        'SineSweepSpecification or a SineTone, not '
                        f'{type(specification).__name__}')
    return ([specification.tone(tone)] if tone is not None
            else list(specification.tones))


def check_sweep_rate(specification: SineSweepSpecification | SineTone,
                     modes: Iterable[tuple[float, float]] | Any, *,
                     tone: str | None = None,
                     margin: float = MARGIN) -> list[SweepRateCheck]:
    """Check a sine specification's sweep rate against a list of modes.

    For every tone (or the one named) and every mode, each pass of the
    sweep through the mode's half-power band (`half_power_band`) gives
    a row: when it enters, how long it stays, how many of the drive's
    cycles that is against the Q the mode needs, the rule's rate for
    the mode (`rule_rate`) and the specification's as a multiple of it,
    what a mode of that Q reads at that rate (`simulate_sweep`), and
    whether the rate is too fast. The rate checked is the one written on
    the segment holding the mode; a band that straddles a breakpoint is
    one pass, counted across both segments.

        checks = check_sweep_rate(spec, [(22.0, 40.0), (180.0, 20.0)])
        for check in checks:
            print(check.describe())

    Parameters
    ----------
    specification : SineSweepSpecification or SineTone
        The sweep.
    modes : iterable of (frequency, Q), or ShapeSet
        Natural frequencies in Hz with their Q; a `ShapeSet` gives its
        modes with Q from their damping.
    tone : str, optional
        Check only the tone of this name.
    margin : float
        The multiple of the rule's rate above which a pass is flagged
        too fast; 1 by default, where a mode reads about 95 %.

    Returns
    -------
    list of SweepRateCheck
        In tone order, then mode order, then pass order. A mode whose
        band the sweep never reaches has no row.
    """
    margin = _positive(margin, 'the margin')
    checks = []
    for each in _tones(specification, tone):
        for frequency, q in _modes(modes):
            low, high = half_power_band(frequency, q)
            for entry, leave, f_in, f_out, segments in _passages(
                    each, low, high):
                segment = _at_mode(segments, frequency)
                kind, rate = segment[4], segment[5]
                cycles = float(np.diff(each.phase_at([entry, leave]))[0]
                               / (2.0 * np.pi))
                crossed = {f_in, f_out} == {low, high}
                checks.append(SweepRateCheck(
                    tone=each.name, frequency=frequency, q=q,
                    segment_type=kind, rate=rate, start=entry,
                    seconds=leave - entry, cycles=cycles, crossed=crossed,
                    limit=rule_rate(frequency, q, kind),
                    reading=simulate_sweep(frequency, q, rate, kind),
                    margin=margin))
    return checks


# ---- a Q read from a swept level ------------------------------------------


@dataclass(frozen=True)
class SweptQ:
    """A Q read from a swept level, judged against the rate it was
    swept at: what `swept_q` and `read_swept_q` return.

    `frequency` is the peak's, Hz; `q_read` the Q read from its width;
    `rate` and `segment_type` the sweep's at the peak. `q_true` is the
    Q a mode would need to read `q_read` at this rate, by inverting
    `simulate_sweep`; None where no Q up to `Q_SEARCH` times the one
    read reads it, which happens when the rate alone sets the width. `height` is the peak's height in the level's own units
    where it was read from a level, else None.
    """

    frequency: float
    q_read: float
    rate: float
    segment_type: int
    q_true: float | None
    margin: float = MARGIN
    height: float | None = None

    @property
    def unit(self) -> str:
        """'oct/min' or 'Hz/s'."""
        return rate_unit(self.segment_type)

    @property
    def limit(self) -> float:
        """The rule's rate for the Q the reading implies (`q_true`), or
        for the Q read where no Q could be inferred."""
        q = self.q_true if self.q_true is not None else self.q_read
        return rule_rate(self.frequency, q, self.segment_type)

    @property
    def ratio(self) -> float:
        """The sweep's rate as a multiple of `limit`."""
        return abs(self.rate) / self.limit

    @property
    def flagged(self) -> bool:
        """Whether the Q read cannot be taken as the mode's: no Q could
        be inferred at all, the sweep was too fast for the Q it implies,
        or the Q it implies is more than `Q_TOLERANCE` above the one
        read. The last catches a sweep under the rule's rate that still
        spreads the width: at the rule's rate a peak reads 95 % of its
        height but its Q reads about 17 % low."""
        return bool(self.q_true is None or self.ratio > self.margin
                    or self.q_true > self.q_read * (1.0 + Q_TOLERANCE))

    def describe(self) -> str:
        """The judgement in a line: 'Q 21.6 read at 22.6 Hz from a sweep
        at 4 oct/min: reads low; a mode reads this low at a Q of about
        40, for which the rate is 3.4 times the rule's'."""
        head = (f'Q {self.q_read:.3g} read at {self.frequency:.4g} Hz from '
                f'a sweep at {abs(self.rate):.3g} {self.unit}')
        if self.q_true is None:
            return (f'{head}: too fast to read any Q; the rate sets the '
                    'width, and the mode\'s Q is higher than any this '
                    'reading can tell apart')
        if not self.flagged:
            return f'{head}: slow enough for it'
        return (f'{head}: reads low; a mode reads this low at a Q of '
                f'about {self.q_true:.3g}, for which the rate is '
                f'{self.ratio:.2g} times the rule\'s')


#: how far the Q a reading implies may stand above the Q read before
#: the reading is flagged as low, a fraction
Q_TOLERANCE = 0.05

#: how far above the Q read `swept_q` looks for the Q that reads it,
#: and the step it walks up in
Q_SEARCH = 64.0
Q_STEP = 1.25


def swept_q(frequency: float, q_read: float, rate: float,
            segment_type: int = LOG, *, margin: float = MARGIN,
            height: float | None = None) -> SweptQ:
    """Judge a Q read from a level swept at `rate`: was the rate slow
    enough for it, and what Q would read that low?

    A swept reading's Q comes out low (high only by the fraction of a
    percent its per-cycle readings scatter by), so the Q read is in
    effect a lower bound on the mode's and the rule's rate worked out
    from it an upper bound. The Q a mode needs to read `q_read` at this
    rate is found on `simulate_sweep`: the swept Q rises with the true Q
    towards a ceiling the rate sets, so past that ceiling no Q reads
    that low and none is returned. The reading is flagged when the rate
    is above `margin` times the rule's rate for that Q, or when that Q
    is more than `Q_TOLERANCE` above the one read.

    Parameters
    ----------
    frequency : float
        The peak's frequency, Hz.
    q_read : float
        The Q read from the swept level's half-power width.
    rate : float
        The sweep's rate at the peak, oct/min (LOG) or Hz/s (LINEAR).
    segment_type : int
        `sine.LOG` or `sine.LINEAR`.
    margin : float
        As for `check_sweep_rate`.
    height : float, optional
        The peak's height, carried through for display.

    Returns
    -------
    SweptQ
        The reading, the Q it implies (None where none can be
        inferred), and the flag.
    """
    frequency, q_read = _positive(frequency, 'a frequency'), _q(q_read)
    margin = _positive(margin, 'the margin')
    kind = LINEAR if int(segment_type) == LINEAR else LOG

    def reads(q):
        found = simulate_sweep(frequency, q, rate, kind).q_read
        return -np.inf if found is None else found

    q_true = None
    if reads(q_read) >= q_read:
        q_true = q_read                  # read at a rate the Q reads true at
    else:
        # The swept Q rises with the true one to a ceiling the rate sets
        # and then falls, the ringing of a very sharp mode beating with
        # the tone that has moved on (at 4 oct/min a 22 Hz mode reads
        # 21.6 at Q 40, 26.3 at 200 and 10.4 at 1000). So the rising
        # branch is walked up in steps until it passes the Q read,
        # and given up on where it turns over first.
        low, last = q_read, reads(q_read)
        high = low * Q_STEP
        while high <= q_read * Q_SEARCH:
            found = reads(high)
            if found >= q_read:
                break
            if found < last:
                high = None              # turned over below the Q read
                break
            low, last, high = high, found, high * Q_STEP
        else:
            high = None
        if high is not None:
            # bisection in log Q inside the step that passed it
            while high / low > 1.0 + 1e-4:
                middle = np.sqrt(low * high)
                if reads(middle) < q_read:
                    low = middle
                else:
                    high = middle
            q_true = float(np.sqrt(low * high))
    return SweptQ(frequency=frequency, q_read=q_read, rate=float(rate),
                  segment_type=kind, q_true=q_true, margin=margin,
                  height=None if height is None else float(height))


def read_swept_q(level: Any, specification: SineSweepSpecification | SineTone,
                 near: float, *, record: int = 0, within: float = 0.2,
                 margin: float = MARGIN) -> SweptQ | None:
    """Read the peak nearest `near` on a tracked sine level and judge its
    Q against the rate the level was swept at.

    The peak is the tallest reading within `within` (a fraction) of
    `near`, its top placed between readings and its half-power
    crossings found by walking out along the sweep; the rate is the
    tone's (`level.tone`) on the segment holding the peak. Returns None
    where the window holds no peak (no readings, or the tallest at its
    edge, the flank of a peak elsewhere) or a crossing is missing, since
    then there is no Q to judge.

    Parameters
    ----------
    level : SineLevel
        A tracked or extracted level, readings in sweep order.
    specification : SineSweepSpecification or SineTone
        The sweep the level was read from.
    near : float
        Hz, the mode looked for.
    record : int
        The level's record (channel) to read.
    within : float
        The window around `near`, as a fraction of it.
    margin : float
        As for `check_sweep_rate`.

    Returns
    -------
    SweptQ or None
        The judgement, with the peak's height in the level's units.
    """
    near = _positive(near, 'a frequency')
    tones = _tones(specification, None)
    name = getattr(level, 'tone', '')
    tone = next((t for t in tones if t.name == name), tones[0])
    f = np.asarray(level.abscissa, dtype=float)
    values = np.abs(np.asarray(level.ordinate)[record]).astype(float)
    keep = np.isfinite(values) & (values > 0.0)
    candidates = np.flatnonzero(keep & (np.abs(f - near) <= within * near))
    if candidates.size < 3:
        return None
    # the tallest reading at an end of the window is the flank of some
    # other peak, not a peak here (a 35 Hz window on a 22 Hz mode's
    # level read a Q of 4 off its skirt)
    tallest = candidates[np.argmax(np.where(keep, values, 0.0)[candidates])]
    if tallest in (candidates[0], candidates[-1]):
        return None
    top, height, q_read = _read_peak(f, np.where(keep, values, 0.0),
                                     candidates)
    if q_read is None or q_read <= 0.5:
        return None
    rows = _segment_table(tone)
    segment = next((row for row in rows
                    if min(row[2], row[3]) <= top <= max(row[2], row[3])),
                   None)
    if segment is None:
        return None
    return swept_q(top, q_read, segment[5], segment[4], margin=margin,
                   height=height)


def checks_text(checks: Sequence[SweepRateCheck]) -> str:
    """Every check's `describe`, one per line."""
    return '\n'.join(check.describe() for check in checks)
