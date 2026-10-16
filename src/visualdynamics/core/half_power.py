"""Q read from the half-power width of one peak.

A resonance on a magnitude curve (an FRF magnitude, a transmissibility,
a tracked sine level) has two numbers a test engineer reads off it: the
frequency of its peak and how sharp it is. The sharpness is Q, the peak
frequency over the half-power width, the width between the two
frequencies where the curve has fallen to 1/sqrt(2) of its peak (0.707,
3 dB down; half power because power goes with amplitude squared). The
damping ratio is 1/(2Q). For one mode well clear of its neighbors this
agrees with the mode's own damping to within a few times the damping
squared: read finely off a one-mass accelerance, Q comes out 1 % low at
a Q of 10, 0.25 % at 20 and 0.04 % at 50 (a receptance or a
transmissibility half that).

`read_half_power` takes the curve and a frequency near the peak and
returns a `HalfPower`: the peak's frequency and height, the two
crossings, the width, Q, the damping ratio and how many lines span the
width. The reading is the one the eye makes off a plot, put in numbers,
and it is not a modal fit: it reads where the curve is, not the
resonator that made it. Two modes closer than their widths read as one
wide peak, and a swept level read too fast reads wide and low; both are
properties of the curve, and the reading reports the curve faithfully.

Three things are said rather than hidden:

- A crossing that is not inside the curve. The curve stops before it
  falls to 0.707 of the peak on that side (the peak is near an end of
  the measured band, or rides a shoulder). The crossing is None, so are
  the width, Q and damping, and a `HalfPowerWarning` says which side.
  Borrowing the other side's width, as the modal fit's seed does, is a
  guess about a curve that was not measured; a seed may guess, a reading
  may not.
- Too few lines across the peak. A peak sampled at a line spacing near
  its own width is read low (the tallest line misses the top) and wide
  (the crossings are interpolated across straight segments of a curved
  flank), and so its Q comes out wrong, nearly always low. The reading
  is still returned, since it is what the curve says, but with a
  `HalfPowerWarning` when fewer than `MIN_LINES` lines span the width.
- A peak so much narrower than its lines that the top, placed between
  them, stands more than 3 dB over every line. No crossing can be
  placed at all, so neither is, and the warning says why.

References
----------
The half-power reading is a standard method; these are where it and the
peak interpolation are described.

1. Ewins, D. J. (2000). *Modal Testing: Theory, Practice and
   Application*, 2nd ed. Research Studies Press. The peak-amplitude
   (half-power point) method of reading a mode's damping from the width
   of its peak, and the error it carries when modes are close.
2. Smith, J. O., & Serra, X. (1987). "PARSHL: An Analysis/Synthesis
   Program for Non-Harmonic Sounds Based on a Sinusoidal
   Representation." *Proceedings of the International Computer Music
   Conference*. A parabola through the logarithm of a sampled peak and
   its two neighbors, to place the top between lines.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

#: the fewest lines across the half-power width that a reading is
#: trusted at without a warning. Measured on a one-mass accelerance and
#: transmissibility at Q 20 and Q 50, over twenty placements of the
#: lines against the peak: at 4 lines across the width the Q read is at
#: most 3 % low and the height under 1 %; at 2 lines Q is up to 14 % low,
#: and at 1 line up to 44 % low with the height 22 % low. From two
#: lines up it was low at every placement; at one and a half lines one
#: placement in forty read 2 % high (`tests/test_half_power.py` pins
#: the two-line and four-line figures). Without the
#: parabola through the top (`_top`) the same 4 lines read up to 7 %
#: low, which is why the top is placed between lines at all.
MIN_LINES = 4

#: 1/sqrt(2): the half-power level as a share of the peak
HALF_POWER = 1.0 / np.sqrt(2.0)


class HalfPowerWarning(UserWarning):
    """Warned when a half-power reading is missing a crossing or rests
    on too few lines. The reading is returned either way; the warning
    says what it cannot be trusted for, and `HalfPower.notes` carries
    the same words for a caller that shows them rather than warns."""


@dataclass(frozen=True)
class HalfPower:
    """One peak read by its half-power width: what `read_half_power`
    returns.

    Frequencies are in Hz and `height` is in the curve's own units.
    `low` and `high` are None where the curve ends before falling to the
    half-power level on that side, and then `width`, `q` and `damping`
    are None too. `lines` is the width over the line spacing at the
    peak: how many lines the reading rests on. `notes` holds the
    warnings the reading raised, as text, empty when it raised none.
    """

    frequency: float
    height: float
    low: float | None
    high: float | None
    spacing: float
    notes: tuple[str, ...] = ()

    @property
    def level(self) -> float:
        """The half-power level: 0.707 of the height."""
        return self.height * HALF_POWER

    @property
    def width(self) -> float | None:
        """The half-power width in Hz, `high` minus `low`."""
        if self.low is None or self.high is None:
            return None
        return self.high - self.low

    @property
    def q(self) -> float | None:
        """Q: the peak frequency over the half-power width."""
        width = self.width
        return self.frequency / width if width else None

    @property
    def damping(self) -> float | None:
        """The damping ratio, a fraction of critical: 1/(2Q)."""
        q = self.q
        return 0.5 / q if q else None

    @property
    def lines(self) -> float | None:
        """How many lines span the half-power width."""
        width = self.width
        return width / self.spacing if width is not None and self.spacing \
            else None

    @property
    def resolved(self) -> bool:
        """True when both crossings were found and at least `MIN_LINES`
        lines span the width."""
        lines = self.lines
        return lines is not None and lines >= MIN_LINES

    def describe(self) -> str:
        """The reading in one line, as the application shows it:
        'Q 19.9 at 179.9 Hz, half-power 175.3 to 184.3 Hz (9.03 Hz,
        14 lines)'."""
        where = f'{self.frequency:.4g} Hz'
        if self.low is None and self.high is None:
            return f'peak at {where}: no half-power width can be read'
        if self.q is None:
            side = ('below' if self.low is None else 'above')
            return (f'peak at {where}: no half-power crossing {side} it '
                    'inside the curve')
        return (f'Q {self.q:.3g} at {where}, half-power '
                f'{self.low:.4g} to {self.high:.4g} Hz '
                f'({self.width:.3g} Hz, {self.lines:.3g} lines)')


def crossings(frequencies: Any, magnitude: Any, index: int,
              level: float) -> tuple[float | None, float | None]:
    """Where the curve first falls below `level` on each side of line
    `index`, as (low, high) in Hz, interpolated linearly between the
    last line at or above the level and the first below it.

    A side that never falls below the level inside the curve gives
    None. The level must not be above line `index` itself, which is the
    top the crossings are measured down from; a ValueError says so. This
    is the one place the package finds half-power crossings: the reading
    here and the modal fit's damping seed both call it.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    m = np.asarray(magnitude, dtype=np.float64)
    if level > m[index]:
        raise ValueError(f'a level of {level:g} is above the line it is '
                         f'measured down from ({m[index]:g})')

    def side(step: int) -> float | None:
        i = index
        while 0 <= i + step < len(m):
            if m[i + step] < level:
                # m[i] >= level > m[i + step], so the step is never flat
                fraction = (m[i] - level) / (m[i] - m[i + step])
                return float(f[i] + fraction * (f[i + step] - f[i]))
            i += step
        return None

    return side(-1), side(+1)


def _climb(m: np.ndarray, index: int) -> int:
    """From line `index`, uphill to the top of the peak it sits on."""
    last = len(m) - 1
    while True:
        below = m[index - 1] if index > 0 else -np.inf
        above = m[index + 1] if index < last else -np.inf
        if below > m[index] and below >= above:
            index -= 1
        elif above > m[index]:
            index += 1
        else:
            return index


def _top(f: np.ndarray, m: np.ndarray, i: int) -> tuple[float, float]:
    """The peak's frequency and height between lines: the vertex of the
    parabola through the logarithms of line `i` and its two neighbors.

    In the logarithm because a resonance's top is closer to a parabola
    there than in the magnitude, which the three points over-sharpen.
    The vertex is held between the neighbors. The fit is in local
    frequency, about line `i`, so uneven spacing (a tracked level is
    read evenly in time, not in frequency) is honored rather than
    assumed away.
    """
    if i == 0 or i == len(m) - 1 or np.any(m[i - 1:i + 2] <= 0.0):
        return float(f[i]), float(m[i])
    x = f[i - 1:i + 2] - f[i]
    y = np.log(m[i - 1:i + 2])
    a, b, c = np.polyfit(x, y, 2)
    if not a < 0.0:
        return float(f[i]), float(m[i])
    vertex = float(np.clip(-b / (2.0 * a), x[0], x[2]))
    return float(f[i] + vertex), float(np.exp(np.polyval((a, b, c), vertex)))


def read_half_power(frequencies: Any, magnitude: Any, near: float, *,
                    within: tuple[float, float] | None = None,
                    min_lines: float = MIN_LINES,
                    warn: bool = True) -> HalfPower:
    """Read Q from the half-power width of the peak nearest `near`.

    The peak is the one the line nearest `near` sits on, found by
    climbing uphill from that line, so a frequency anywhere on a peak's
    flank reads that peak and not a taller one elsewhere. With `within`,
    a (low, high) band in Hz, the peak is instead the tallest line
    inside the band, which is the reading for a band that holds one
    mode. The top is placed between lines by a parabola through the
    logarithms of the tallest line and its two neighbors, and the two
    crossings of 0.707 of that height are interpolated linearly between
    lines. Q is the peak frequency over the width between the
    crossings, and the damping ratio is 1/(2Q).

    A `HalfPowerWarning` is warned, and its text kept in `notes`, when a
    crossing is not inside the curve and when fewer than `min_lines`
    lines span the width (see the module's notes on resolution). Pass
    `warn=False` to keep the notes without warning, as a display that
    reads on every mouse move does.

    Parameters
    ----------
    frequencies : array-like
        The curve's frequencies in Hz, ascending. They need not be
        evenly spaced.
    magnitude : array-like
        The curve, one value per frequency. Complex values are read by
        their magnitude.
    near : float
        A frequency on the peak to read, in Hz.
    within : tuple of float, optional
        A band (low, high) in Hz; the tallest line inside it is the
        peak, and `near` is not used to find it.
    min_lines : float, default `MIN_LINES`
        The fewest lines across the half-power width read without a
        warning.
    warn : bool, default True
        Warn the notes as `HalfPowerWarning` as well as keeping them.

    Returns
    -------
    HalfPower
        The peak's frequency and height, the two crossings, the line
        spacing at the peak, and any notes.

    Raises
    ------
    ValueError
        When the frequencies and the magnitude differ in length, hold
        fewer than three lines, are not ascending, or `within` holds no
        line of the curve.
    """
    f = np.asarray(frequencies, dtype=np.float64).ravel()
    m = np.abs(np.asarray(magnitude)).astype(np.float64).ravel()
    if len(f) != len(m):
        raise ValueError(f'{len(f)} frequencies for {len(m)} values')
    if len(f) < 3:
        raise ValueError('a peak needs at least three lines to be read')
    if not np.all(np.diff(f) > 0.0):
        raise ValueError('the frequencies must be ascending')
    # a line the source could not read (NaN) is no part of any peak
    m = np.where(np.isfinite(m), m, -np.inf)
    if within is not None:
        low, high = sorted((float(within[0]), float(within[1])))
        band = np.flatnonzero((f >= low) & (f <= high))
        if not len(band):
            raise ValueError(f'no line of the curve lies between {low:g} '
                             f'and {high:g} Hz')
        i = int(band[np.argmax(m[band])])
    else:
        i = _climb(m, int(np.argmin(np.abs(f - float(near)))))
    peak, height = _top(f, np.where(np.isfinite(m), m, 0.0), i)
    # the spacing the peak was sampled at: the mean of the two steps
    # about it, which is the one step on an even axis
    spacing = float((f[min(i + 1, len(f) - 1)] - f[max(i - 1, 0)])
                    / (min(i + 1, len(f) - 1) - max(i - 1, 0)))
    notes = []
    narrow = height * HALF_POWER > m[i]
    if narrow:
        # the top, placed between lines, stands more than 3 dB over the
        # tallest line: the half-power level is above every line, so no
        # crossing can be placed. Both crossings would land on the
        # tallest line, a width of zero and a Q of infinity.
        below = above = None
        notes.append(f'the peak at {peak:.4g} Hz is narrower than its '
                     f'lines, {spacing:.3g} Hz apart: its half-power '
                     'level is above every line, and no width or Q is read')
    else:
        below, above = crossings(f, m, i, height * HALF_POWER)
    for crossing, word, end in ((below, 'below', f[0]),
                                (above, 'above', f[-1])):
        if crossing is None and not narrow:
            notes.append(f'the curve ends at {end:.4g} Hz before falling '
                         f'to half power {word} the peak at {peak:.4g} Hz; '
                         'no width or Q is read')
    reading = HalfPower(peak, height, below, above, spacing)
    lines = reading.lines
    if lines is not None and lines < min_lines:
        notes.append(f'the half-power width of the peak at {peak:.4g} Hz '
                     f'is {reading.width:.3g} Hz, {lines:.2g} lines of '
                     f'{spacing:.3g} Hz; with fewer than {min_lines:g} '
                     'lines across it the top is missed and the flanks '
                     'are cut straight, and Q is biased, usually low')
    if warn:
        for note in notes:
            warnings.warn(note, HalfPowerWarning, stacklevel=2)
    return HalfPower(peak, height, below, above, spacing, tuple(notes))
