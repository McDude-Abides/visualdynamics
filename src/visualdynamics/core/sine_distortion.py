"""The sine distortion check: what is on a channel besides the tone.

A sine controller reads its control channel through a narrow band that
follows the drive (`core.sine_tracking`), so the level it holds is the
fundamental and nothing else. Whatever else the channel carries is
outside the band: harmonics from a rattle, a loose fixture bolt, a joint
that claps or an amplifier at its limit, a fixture resonance excited at a
multiple of the drive, noise. The controller does not see it, and the
article feels it. The check compares the channel read unfiltered with
the tone alone. With x_u the unfiltered RMS and x_1 the RMS of the
fundamental, the distortion is

    D = sqrt((x_u / x_1)**2 - 1),

the RMS of everything that is not the tone over the RMS of the tone,
since the tone and the rest are orthogonal and their powers add
(`distortion`). It is reported as a fraction (a percentage on screen)
and in dB, 20 log10 D.

The tolerance is usually written on the ratio rather than on D: the
unfiltered RMS no more than some share above the filtered one. Five
percent is an example of what test documents ask for, and it is the
default here (`DISTORTION_EXCESS`). Five percent over is a distortion of
sqrt(1.05**2 - 1) = 0.320, 32 % or -9.9 dB (`distortion_limit`), which is
the line the check draws and the figure a line is flagged above.

The check is not sensitive, and it is not meant to be. A tone with a
third harmonic at 10 % of it reads 1.005 on the ratio, at 20 % 1.020,
and only at 32 % does it reach 1.05: it catches distortion large enough
to change the test. It hardly sees clipping at all. A sine clipped flat
at four fifths of its peak reads 1.004, because cutting the tops of a
sine puts most of the lost power back into the fundamental, so flat tops
are found in the time history and its crest factor, not here.

The fundamental is the extraction's (`core.sine.extract_sine`): each
tone's Vold-Kalman envelope on its own sweep, debiased, read at the
instants the extraction chose, which is the best estimate of the tone
the package has. Its smoothing is `SineLevelSet.cycles` cycles of the
tone's own frequency, and the unfiltered RMS is read over the same
stretch: centered on the instant each line was read, over that many
cycles of the tone's phase, exactly, on the band-limited waveform rather
than the samples alone (the span average `core.sine_tracking` reads its
RMS detector with). Read from the samples alone, a span rounded to whole
samples at a few samples a cycle reads a sine's RMS off by a few tenths
of a percent, and near a ratio of one that is several percent of D. A
span that would run off either end of the tone's stretch of the record is
moved inward until it fits.

The record's mean over the tone's stretch is taken out first: a sensor's
offset is not motion the article feels. Anything slower than the span
that is not constant is counted, as it should be, since it is not the
tone either.

Where several tones play at once, the unfiltered RMS holds all of them,
so the fundamental is the root-sum-square of every tone live in the
span, each from its own level (interpolated along its own lines) and
each counted for the share of the span it plays in, and D is what is
left over all of them. Over a short span two tones beat, and the beat
does not average away: on a clean record they read a few percent of
distortion between them, more over fewer cycles. A line whose span
reaches a tone of the specification the levels do not hold cannot be
judged, and neither can a line the extraction found under its noise
floor (`SineLevel.below_floor`); both are kept, counted in the notes and
left out of the verdict (`DistortionCheck.judged`).

On a clean recording the ratio reads within a few parts in ten thousand
of one, and D, being the square root of the difference, reads a few
percent from that alone: 1.0002 is 2 %. The figure means something well
above that, which is where the check's limit is.

References
----------
The definition of the distortion and the estimator of the tone it is
read against are set out in these.

1. IEC 60068-2-6:2007, *Environmental testing - Part 2-6: Tests - Test
   Fc: Vibration (sinusoidal)*. Distortion of a sinusoidal vibration as
   the RMS of the motion other than the fundamental over the RMS of the
   fundamental, read at the reference point.
2. Brandt, A. (2011). *Noise and Vibration Analysis: Signal Analysis and
   Experimental Procedures*. Wiley. Order tracking, the Vold-Kalman
   filter among it: a component on a known frequency history read as a
   slowly varying amplitude by least squares.
3. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The mean square of a sum of
   orthogonal components as the sum of their mean squares.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import _even_steps
from .sine import SineLevelSet, SineSweepSpecification, extract_sine
from .sine_tracking import _laid, _span_average

# Why the unfiltered RMS reads the band-limited waveform (2026-10-23):
# on a clean 100-800 Hz sweep sampled at 4096 Hz over ten cycles, the
# samples alone put the ratio up to 1.002 off one at the top, a D of
# 6 %, where the span average holds it under 1.0002.

#: the share the unfiltered RMS may stand above the fundamental's
#: before a line is flagged: 5 %, an example of what test documents ask
#: for, which is a distortion of 32 % (`distortion_limit`)
DISTORTION_EXCESS = 0.05

#: the share of the Nyquist frequency up to which the band-limited
#: reconstruction counts content in full; above it the reconstruction
#: filter rolls off and the unfiltered RMS under-reads what is there
#: (a third harmonic at 10 % read 0.100 to 0.82 of the Nyquist
#: frequency and 0.069 at 0.99, sampled at 4096 Hz)
RECONSTRUCTED_SHARE = 0.8

#: the harmonic a note is written for when it climbs past
#: `RECONSTRUCTED_SHARE`: the third, the first a symmetric clip or a
#: rattle puts on a channel
HARMONIC = 3
HARMONIC_WORD = 'third'


def distortion(unfiltered: Any, fundamental: Any) -> np.ndarray:
    """The distortion from the two readings: sqrt((unfiltered /
    fundamental)**2 - 1), the RMS of everything that is not the tone over
    the RMS of the tone.

    A ratio under one, which only the scatter of the two readings gives
    (the unfiltered record holds the tone and more), reads as no
    distortion rather than as the root of a negative number. A
    fundamental of zero reads as not a number.

    Parameters
    ----------
    unfiltered : float or array-like
        The channel's RMS, unfiltered.
    fundamental : float or array-like
        The tone's RMS, in the same units.

    Returns
    -------
    ndarray
        D as a fraction (0.32 is 32 %), the shape the two broadcast to.
    """
    unfiltered = np.asarray(unfiltered, dtype=float)
    fundamental = np.asarray(fundamental, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(fundamental > 0.0, unfiltered / fundamental, np.nan)
    return np.sqrt(np.maximum(ratio * ratio - 1.0, 0.0))


def distortion_limit(excess: float = DISTORTION_EXCESS) -> float:
    """The distortion a tolerance on the ratio allows: sqrt((1 +
    excess)**2 - 1), 0.320 for 5 %.

    Raises ValueError unless `excess` is more than zero.

    Parameters
    ----------
    excess : float, default `DISTORTION_EXCESS`
        The share the unfiltered RMS may stand above the fundamental's.

    Returns
    -------
    float
        D as a fraction.
    """
    excess = float(excess)
    if not excess > 0.0:
        raise ValueError(f'a tolerance of {excess:g} on the ratio must be '
                         'more than zero')
    return float(np.sqrt((1.0 + excess) ** 2 - 1.0))


def decibels(value: Any) -> np.ndarray:
    """A distortion in dB, 20 log10 D; minus infinity where it is zero.

    Parameters
    ----------
    value : float or array-like
        D as a fraction.

    Returns
    -------
    ndarray
        dB, the shape of `value`.
    """
    with np.errstate(divide='ignore'):
        return 20.0 * np.log10(np.asarray(value, dtype=float))


@dataclass(frozen=True, kw_only=True, eq=False)
class DistortionCheck:
    """One tone's distortion along its sweep, on each channel read: what
    `check_distortion` returns.

    `frequency` and `seconds` are the extraction's lines, ascending in
    frequency, each with the second it was read, from the recording's
    first sample. `unfiltered` and `fundamental` are channels x lines of
    RMS in SI, the channel read over `cycles` cycles about each line and
    the root-sum-square of every tone live there. `judged` is channels x
    lines, False where a line cannot be judged (the tone under its noise
    floor, or a tone the levels do not hold live beside it). `excess` is
    the tolerance on the ratio. `notes` says what was not judged and why,
    empty when every line was. Two of these compare equal only if they
    are the same object, since the arrays have no single truth value.
    """

    tone: str
    dofs: list[str]
    dimensions: list[str]
    units: list[str | None]
    frequency: np.ndarray
    seconds: np.ndarray
    unfiltered: np.ndarray
    fundamental: np.ndarray
    judged: np.ndarray
    excess: float
    cycles: float
    notes: tuple[str, ...] = field(default=())

    @property
    def ratio(self) -> np.ndarray:
        """Channels x lines: the unfiltered RMS over the fundamental's,
        the number a tolerance like 'within 5 %' is written on."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.fundamental > 0.0,
                            self.unfiltered / self.fundamental, np.nan)

    @property
    def distortion(self) -> np.ndarray:
        """Channels x lines: D as a fraction (`distortion`)."""
        return distortion(self.unfiltered, self.fundamental)

    @property
    def decibels(self) -> np.ndarray:
        """Channels x lines: D in dB (`decibels`)."""
        return decibels(self.distortion)

    @property
    def limit(self) -> float:
        """The distortion the tolerance allows, as a fraction
        (`distortion_limit`)."""
        return distortion_limit(self.excess)

    @property
    def over(self) -> np.ndarray:
        """Channels x lines, True where a judged line's distortion is
        above the limit."""
        return self.judged & (self.distortion > self.limit)

    @property
    def passed(self) -> bool:
        """Whether no judged line on any channel is over the limit."""
        return not bool(self.over.any())

    def row(self, dof: str | None = None) -> int:
        """The row of channel `dof`; the first when None. Raises
        ValueError when the check did not read it."""
        if dof is None:
            return 0
        if str(dof) not in self.dofs:
            raise ValueError(f'{dof} was not read; the check holds '
                             f'{self.dofs}')
        return self.dofs.index(str(dof))

    def worst(self, dof: str | None = None) -> tuple[float, float] | None:
        """The largest judged distortion and the frequency it is at, on
        channel `dof` or, when None, on any channel: (Hz, D as a
        fraction), or None when no line was judged.

        Parameters
        ----------
        dof : str, optional
            The channel; every channel read when omitted.

        Returns
        -------
        tuple of float, or None
            (frequency in Hz, distortion as a fraction).
        """
        rows = (range(len(self.dofs)) if dof is None else [self.row(dof)])
        best = None
        values = self.distortion
        for r in rows:
            seen = self.judged[r] & np.isfinite(values[r])
            if not seen.any():
                continue
            k = int(np.flatnonzero(seen)[np.argmax(values[r][seen])])
            if best is None or values[r, k] > best[1]:
                best = (float(self.frequency[k]), float(values[r, k]))
        return best

    def describe(self, dof: str | None = None) -> str:
        """The verdict in one line, such as 'Up at 101Z+: worst 12.3 %
        (-18.2 dB) at 412 Hz, within the 32.0 % limit'. Every channel's
        worst when `dof` is None."""
        where = self.tone + ('' if dof is None else f' at {dof}')
        worst = self.worst(dof)
        if worst is None:
            return f'{where}: no line could be judged'
        rows = (range(len(self.dofs)) if dof is None else [self.row(dof)])
        over = int(sum(self.over[r].sum() for r in rows))
        judged = int(sum(self.judged[r].sum() for r in rows))
        frequency, value = worst
        words = (f'{where}: worst {100.0 * value:.1f} % '
                 f'({float(decibels(value)):.1f} dB) at {frequency:.4g} Hz, ')
        if over:
            return words + (f'{over} of {judged} lines over the '
                            f'{100.0 * self.limit:.1f} % limit')
        return words + f'within the {100.0 * self.limit:.1f} % limit'


def _live_windows(specification: SineSweepSpecification, chosen: Any,
                  onset: float) -> dict[str, tuple[float, float]]:
    """Each tone's span in recording seconds, placed on the clock the
    read tone was found on: the environment starts at its onset less its
    own start time, and every tone starts its own start time after that."""
    origin = float(onset) - chosen.start_time
    return {tone.name: (origin + tone.start_time,
                        origin + tone.start_time + tone.duration())
            for tone in specification.tones}


def _span(phase: np.ndarray, k: int, cycles: float) -> tuple[int, float]:
    """The span of exactly `cycles` cycles of `phase` centered on sample
    `k`, moved inward where it would run off either end: (its last
    sample, the phase it starts at), as the span average takes it."""
    turn = 2.0 * np.pi * cycles
    last = len(phase) - 1
    end = min(int(np.searchsorted(phase, phase[k] + turn / 2.0)), last)
    if phase[end] - turn < phase[0]:
        end = min(int(np.searchsorted(phase, phase[0] + turn)), last)
    return end, float(phase[end] - turn)


def _overlap(t0: np.ndarray, t1: np.ndarray, lo: float,
             hi: float) -> np.ndarray:
    """The share of each span from `t0` to `t1` that lies between `lo`
    and `hi`."""
    inside = np.clip(np.minimum(t1, hi) - np.maximum(t0, lo), 0.0, None)
    return inside / np.maximum(t1 - t0, 1e-300)


def check_distortion(history: Any, specification: SineSweepSpecification,
                     levels: SineLevelSet | None = None, *,
                     tone: str | None = None,
                     channels: Sequence[str] | None = None,
                     excess: float = DISTORTION_EXCESS,
                     cycles: float | None = None) -> DistortionCheck:
    """Check one tone's distortion along its sweep: the channel's
    unfiltered RMS against the RMS of the tracked fundamental, at every
    line of the extraction.

    The fundamental comes from `levels`, a `SineLevelSet` that
    `extract_sine` (Extract Sine Levels) read from this recording against
    this specification; with none, the extraction is run here with the
    recording's own `sine_extraction` setting (or its suggestion), as the
    project verb runs it. At each line the channel is read unfiltered over
    `cycles` cycles of the tone (the extraction's smoothing by default)
    centered on the line's instant, its mean over the tone's stretch
    taken out, and compared with the root-sum-square of every tone live
    there. A line is flagged where the distortion is above the limit
    `excess` sets (`distortion_limit`).

    Raises ValueError when `excess` or `cycles` is not more than zero, when
    the recording is not evenly sampled, when the levels hold no such tone
    or channel, or when the recording holds less of the tone than one span.

    Parameters
    ----------
    history : TimeHistory
        The recording, evenly sampled.
    specification : SineSweepSpecification
        The sweep the drive followed.
    levels : SineLevelSet, optional
        The extracted levels; extracted here when omitted.
    tone : str, optional
        Which tone, by name; the first the levels hold by default.
    channels : sequence of str, optional
        Which control channels; every channel the levels hold by default.
    excess : float, default `DISTORTION_EXCESS`
        The share the unfiltered RMS may stand above the fundamental's;
        5 %, a distortion of 32 %, by default.
    cycles : float, optional
        The span the unfiltered RMS is read over, in cycles of the tone;
        the extraction's smoothing by default, so both readings cover the
        same stretch.

    Returns
    -------
    DistortionCheck
        The two RMS readings at every line on every channel read, which
        lines could be judged, and the tolerance.
    """
    distortion_limit(excess)                  # refuses a bad tolerance
    dt = _even_steps(np.asarray(history.abscissa, dtype=float),
                     'Checking sine distortion')
    if levels is None:
        setting = (history.sine_extraction
                   or history.suggest_sine_extraction())
        levels = extract_sine(history, specification, cycles=setting.cycles,
                              refine=setting.refine,
                              target_db=setting.target_db)
    if not len(levels):
        raise ValueError('the levels hold no tone; extract them first')
    if tone is not None and tone not in levels.tone_names:
        raise ValueError(f'the levels hold no tone {tone!r}; they hold '
                         f'{levels.tone_names}')
    level = levels.levels[0] if tone is None else levels.tone(tone)
    chosen = specification.tone(level.tone)
    if cycles is None:
        cycles = levels.cycles
    if cycles is None:
        raise ValueError('these levels do not say what smoothing they '
                         'were read with; give cycles=')
    cycles = float(cycles)
    if not cycles > 0.0:
        raise ValueError('the unfiltered RMS is read over more than zero '
                         'cycles')
    held = [str(dof) for dof in level.response_dof]
    wanted = held if channels is None else [str(dof) for dof in channels]
    missing = [dof for dof in wanted if dof not in held]
    if missing:
        raise ValueError(f'the levels hold no reading for {missing}; they '
                         f'hold {held}')
    picks = [held.index(dof) for dof in wanted]
    dofs = list(history.response_dof)
    absent = [dof for dof in wanted if dof not in dofs]
    if absent:
        raise ValueError(f'the recording carries no record for {absent}')
    rows = [dofs.index(dof) for dof in wanted]

    first, last, _frequency, phase = _laid(history, chosen, level.onset, dt)
    if phase[-1] - phase[0] < 2.0 * np.pi * cycles:
        raise ValueError(
            f'{chosen.name}: the recording holds '
            f'{(phase[-1] - phase[0]) / (2.0 * np.pi):.3g} cycles of the '
            f'tone, less than the {cycles:g} the RMS is read over')
    seconds = np.asarray(level.seconds, dtype=float)
    lines = np.clip(np.round(seconds / dt).astype(np.int64) - first, 0,
                    last - first - 1)

    spans = [_span(phase, int(k), cycles) for k in lines]
    # each span's ends in recording seconds, for the other tones' share
    t0 = (first + np.interp([start for _end, start in spans], phase,
                            np.arange(len(phase)))) * dt
    t1 = (first + np.array([end for end, _start in spans])) * dt

    # every other tone: its power counted for the share of the span it
    # plays in, at its own level interpolated along its lines; a tone
    # the levels do not hold leaves every span it reaches unjudged. A
    # line counted the other tone only where its center was inside it
    # and read 29 % at both of that tone's ends (2026-10-23)
    windows = _live_windows(specification, chosen, level.onset)
    extracted = set(levels.tone_names)
    shared = np.zeros(len(seconds), dtype=bool)
    for name, (lo, hi) in windows.items():
        if name not in extracted:
            shared |= _overlap(t0, t1, lo, hi) > 0.0

    amplitude = np.abs(np.asarray(level.ordinate))[picks]
    power = (amplitude / np.sqrt(2.0)) ** 2
    for other in levels.levels:
        if other.tone == level.tone or other.seconds is None:
            continue
        share = _overlap(t0, t1, *windows.get(other.tone, (0.0, 0.0)))
        if not share.any():
            continue
        held_other = [str(dof) for dof in other.response_dof]
        when = np.asarray(other.seconds, dtype=float)
        order = np.argsort(when)
        for c, dof in enumerate(wanted):
            reading = np.abs(np.asarray(other.ordinate))[
                held_other.index(dof)][order]
            power[c] += share * (np.interp(seconds, when[order], reading)
                                 / np.sqrt(2.0)) ** 2
    fundamental = np.sqrt(power)

    unfiltered = np.zeros((len(rows), len(seconds)))
    for c, row in enumerate(rows):
        x = np.real(np.asarray(history.ordinate[row, first:last],
                               dtype=float))
        x = x - np.mean(x)
        unfiltered[c] = [np.sqrt(max(_span_average(x, phase, end, start,
                                                   'rms'), 0.0))
                         for end, start in spans]

    floor = (np.zeros_like(unfiltered, dtype=bool)
             if level.below_floor is None
             else np.asarray(level.below_floor, dtype=bool)[picks])
    judged = ~floor & ~shared[None, :]
    notes = []
    if floor.any():
        notes.append(f'{int(floor.sum())} line{"s" * int(floor.sum() != 1)} '
                     'under the noise floor, not judged')
    if shared.any():
        notes.append(f'{int(shared.sum())} line{"s" * int(shared.sum() != 1)} '
                     'shared with a tone the levels do not hold, not judged')
    reach = RECONSTRUCTED_SHARE * 0.5 / dt / HARMONIC
    if float(np.max(level.abscissa)) > reach:
        notes.append(f'above {reach:.4g} Hz the {HARMONIC_WORD} harmonic is '
                     f'past {100.0 * RECONSTRUCTED_SHARE:g} % of the Nyquist '
                     'frequency, where the unfiltered RMS under-reads it')
    return DistortionCheck(
        tone=chosen.name, dofs=wanted,
        dimensions=[history.ordinate_dim[row] for row in rows],
        units=[history.ordinate_unit[row] for row in rows],
        frequency=np.asarray(level.abscissa, dtype=float), seconds=seconds,
        unfiltered=unfiltered, fundamental=fundamental, judged=judged,
        excess=float(excess), cycles=cycles, notes=tuple(notes))


__all__ = ['DISTORTION_EXCESS', 'DistortionCheck', 'check_distortion',
           'decibels', 'distortion', 'distortion_limit']
