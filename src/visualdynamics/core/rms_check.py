"""The rms check: the rms a spectrum carries beside the rms of the record
it was computed from.

The area under an autospectral density is the mean square of the record
it came from (Parseval's theorem), so the square root of the sum of its
lines, each times the line spacing, should come back to the rms of the
time history. It is the cheapest check there is on a spectral
computation, and it catches the scaling errors no plot shows: the
amplitude correction of a window applied to a density (the window's
noise bandwidth too high, 1.76 dB on a Hann density), the one-sided
doubling left out or applied twice (3.01 dB either way), and one side
in g where the other is in m/s² (19.83 dB). A spectrum that cannot give
back the rms of its own record is wrong, whatever its plot looks like.

Two time-history numbers are read, because a window makes the check an
estimate rather than an identity. With no window and frames laid end to
end every sample is used once at the same weight and the two rms agree
to rounding. A tapering window weights the middle of each frame more
than its ends, so an averaged, windowed spectrum is an estimate of the
record's mean square, and on a long random record it lands within a few
hundredths of a decibel of it. So the check reads, beside the record's
own rms over the span the frames cover (`RmsCheck.time_rms`), the mean
square as the frames themselves see it: each frame leveled as the
averaging levels it, windowed, its mean square taken with the sum of
the window's squares, and the frames averaged (`RmsCheck.framed_rms`).
A correctly scaled spectrum computed with that averaging carries
exactly that, to rounding, whatever the window; so when the gap to the
record is over the threshold and the gap to the frames is not, the
window's weighting is the whole of it and the scaling is right.

The gap is the spectrum's rms over the record's, in decibels, positive
when the spectrum reads high, and it is judged against a threshold that
is an argument. Its default is a few hundredths of a decibel: wide
enough for a windowed average of a long random record, and an order
below the smallest scaling error the check exists to catch.

References
----------
The check is Parseval's theorem applied to the averaged periodogram;
the window's noise bandwidth is what names the commonest error.

1. Welch, P. D. (1967). "The Use of Fast Fourier Transform for the
   Estimation of Power Spectra: A Method Based on Time Averaging Over
   Short, Modified Periodograms." *IEEE Transactions on Audio and
   Electroacoustics*, 15(2), 70-73.
   [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
   The averaged, windowed periodogram, scaled by the sum of the
   window's squares, whose area is the frames' mean square.
2. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The one-sided autospectral
   density and its integral, the mean square value.
3. Harris, F. J. (1978). "On the Use of Windows for Harmonic Analysis
   with the Discrete Fourier Transform." *Proceedings of the IEEE*,
   66(1), 51-83.
   [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
   The equivalent noise bandwidth, the factor between a window's
   amplitude correction and its energy correction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from .averaging import Averaging
    from .data import Psd, TimeHistory

#: the gap, in dB, beyond which the spectrum and the record are said to
#: disagree. A few hundredths of a decibel: a windowed, overlapped
#: average of a long random record lands within that of the record's
#: own rms, and the smallest scaling error worth catching, a window's
#: amplitude correction on a density, is 1.76 dB for Hann. A procedure
#: that names its own figure passes it as `threshold_db`.
GAP_DB = 0.05
# From the worked example the check was asked for with (2026-10-23): a
# 12 s random record at 5120 samples a second, 1024-sample frames,
# reads 0.000 dB with no window, 0.021 dB under with Hann and 0.014 dB
# under with Hann at half overlap; the amplitude correction on the
# density reads 1.74 dB high. "More than a few hundredths of a
# decibel" is that example's own rule for a scaling error. Hann frames
# laid end to end use each frame's ends hardly at all, and on a shorter
# or narrower record they scatter further: a 16 s record flat from 20
# to 800 Hz, 1024-sample frames at 2048 a second, reads 0.052 dB low
# with no overlap and under 0.01 dB at half overlap. That is why the
# frames' own mean square is read: such a gap is flagged and said to be
# the window's, not passed by widening the threshold for everyone.

#: how close the spectrum must come to the frames' own mean square for
#: the window to be the whole of a gap. A correctly scaled spectrum of
#: those frames agrees to rounding (1e-12 dB); a thousandth of a decibel
#: leaves room for a spectrum stored in single precision and nothing
#: else.
FRAMED_DB = 1e-3

#: how near a gap must sit to a known scaling error, in dB, to be named
#: as it. The errors are 1.76, 3.01 and 19.83 dB apart from the truth,
#: and a windowed estimate scatters by hundredths, so a tenth tells
#: them apart without naming a gap that is merely large.
CAUSE_DB = 0.1

#: standard gravity, m/s² per g: the factor between a density in g²/Hz
#: and one in (m/s²)²/Hz is its square
STANDARD_GRAVITY = 9.80665


@dataclass(frozen=True)
class RmsCheck:
    """One record's rms check: the rms from the spectrum beside the rms
    of the time history, and the gap between them in dB.

    The rms are in the record's SI unit, as the data is stored (m/s²
    for an acceleration). `gap_db` is 20 log10 of the spectrum's rms
    over the record's, positive when the spectrum reads high; NaN when
    either is not a positive number. `framed_rms` is the mean square
    as the averaging's frames see it, square-rooted, when the averaging
    is known and has the spectrum's line spacing; None otherwise.
    """

    #: the channel, as the spectrum names it
    dof: str
    #: the square root of the area under the spectrum, every line times
    #: its bin's width
    spectrum_rms: float
    #: the record's own rms over the span the frames cover
    time_rms: float
    #: the frames' mean square, windowed and leveled as they were
    #: averaged, square-rooted; None when the averaging is not known
    framed_rms: float | None
    #: what the gap is judged against, in dB either way
    threshold_db: float
    #: how many lines were summed, and how far apart, in Hz
    lines: int
    spacing: float
    #: how many samples the record's rms was read over, every capture
    #: of the channel pooled
    samples: int
    #: the window the spectrum was averaged with, when known
    window: str | None = None
    #: the window's equivalent noise bandwidth in lines, when known
    noise_bandwidth: float | None = None
    #: the span read, (start, stop) in seconds on the record's clock,
    #: when the check was read off a record
    span: tuple[float, float] | None = None
    #: what else the reader should know, in words: a detrend, an
    #: averaging that is not the spectrum's, a capture count
    notes: tuple[str, ...] = ()

    @property
    def gap_db(self) -> float:
        """The spectrum's rms over the record's, in dB: positive when the
        spectrum reads high. NaN when either is not a positive number."""
        return _gap(self.spectrum_rms, self.time_rms)

    @property
    def framed_gap_db(self) -> float | None:
        """The spectrum's rms over the frames' own, in dB; None when the
        frames are not known. Near zero for a correctly scaled spectrum
        of those frames, whatever the window."""
        if self.framed_rms is None:
            return None
        return _gap(self.spectrum_rms, self.framed_rms)

    @property
    def window_gap_db(self) -> float | None:
        """The frames' rms over the record's, in dB: how much of the gap
        the window's weighting of the samples accounts for. None when
        the frames are not known."""
        if self.framed_rms is None:
            return None
        return _gap(self.framed_rms, self.time_rms)

    @property
    def passed(self) -> bool:
        """Whether the gap is within the threshold. False when there is
        no gap to judge (a record of zeros, a spectrum with no lines)."""
        gap = self.gap_db
        return bool(np.isfinite(gap) and abs(gap) <= self.threshold_db)

    @property
    def explained(self) -> bool:
        """Whether a gap over the threshold is the window's alone: the
        spectrum carries the frames' own mean square, so its scaling is
        right and the frames weighted the record unevenly."""
        framed = self.framed_gap_db
        return bool(not self.passed and framed is not None
                    and np.isfinite(framed) and abs(framed) <= FRAMED_DB
                    and np.isfinite(self.gap_db))

    @property
    def cause(self) -> str | None:
        """The scaling error a failed gap matches, in words, or None.

        Named only when the gap sits within `CAUSE_DB` of one and the
        window is not the whole of it: a window's amplitude correction
        applied to a density (the window's noise bandwidth, so only
        when the window is known and tapers), the one-sided doubling
        left out or applied twice, and a density in g²/Hz read as
        (m/s²)²/Hz or the other way.
        """
        gap = self.gap_db
        if self.passed or self.explained or not np.isfinite(gap):
            return None
        bandwidth = self.noise_bandwidth
        if bandwidth is not None and bandwidth > 1.0 + 1e-6:
            amplitude = 10.0 * np.log10(bandwidth)
            if abs(gap - amplitude) <= CAUSE_DB:
                return (f"the window's amplitude correction applied to a "
                        f'density: a {self.window} window passes '
                        f'{bandwidth:.3g} lines of noise, '
                        f'{amplitude:+.2f} dB, where a density takes the '
                        'sum of its squares')
        doubling = 10.0 * np.log10(2.0)
        if abs(gap - doubling) <= CAUSE_DB:
            return ('twice the mean square: the one-sided doubling applied '
                    'twice, or to a spectrum that was already one-sided')
        if abs(gap + doubling) <= CAUSE_DB:
            return ('half the mean square: a two-sided density, or the '
                    'one-sided doubling left out')
        gravity = 20.0 * np.log10(STANDARD_GRAVITY)
        if abs(abs(gap) - gravity) <= CAUSE_DB:
            return ('a factor of standard gravity: one side in g, the '
                    'other in m/s²')
        return None

    @property
    def verdict(self) -> str:
        """The judgement in words: agree, the window's gap, a named
        scaling error, or a gap to look for one behind."""
        gap = self.gap_db
        within = f'{self.threshold_db:g} dB'
        if not np.isfinite(gap):
            return ('no gap to judge: the record or the spectrum carries '
                    'no mean square')
        if self.passed:
            return (f'agree within {within}: the spectrum carries the '
                    "record's mean square")
        side = 'high' if gap > 0 else 'low'
        if self.explained:
            return (f'the spectrum reads {abs(gap):.3f} dB {side}, all of '
                    "it the window's: it carries the frames' mean square "
                    'exactly, so the scaling is right, and a longer record '
                    'or more overlap narrows the gap')
        cause = self.cause
        if cause is not None:
            return f'the spectrum reads {abs(gap):.2f} dB {side}: {cause}'
        return (f'the spectrum reads {abs(gap):.2f} dB {side}, outside '
                f'{within}: look for a scaling error before its plot is '
                'read')

    def describe(self, unit: str = '', scale: float = 1.0) -> str:
        """One line for a status bar: both rms, the gap and the verdict.
        `scale` converts the rms from SI into `unit`."""
        def rms(value: float) -> str:
            return f'{value * scale:.4g} {unit} rms'.replace('  ', ' ')

        gap = self.gap_db
        said = f'{gap:+.3f} dB' if np.isfinite(gap) else 'no gap'
        return (f'{rms(self.spectrum_rms)} from the spectrum, '
                f'{rms(self.time_rms)} from the time history, {said}: '
                f'{self.verdict}')


def _gap(over: float, under: float) -> float:
    """20 log10(over/under), NaN when either is not a positive number."""
    if not (np.isfinite(over) and np.isfinite(under)
            and over > 0.0 and under > 0.0):
        return float('nan')
    return float(20.0 * np.log10(over / under))


def _pieces(samples: Any) -> list[np.ndarray]:
    """`samples` as a list of arrays: one record, or the captures of a
    channel given as a list or tuple of arrays (of any lengths)."""
    if (isinstance(samples, (list, tuple)) and len(samples)
            and all(np.ndim(piece) >= 1 for piece in samples)):
        return [np.asarray(piece, dtype=float) for piece in samples]
    return [np.asarray(samples, dtype=float)]


def time_rms(samples: ArrayLike | Sequence[ArrayLike]) -> float:
    """The rms of a record's samples: the square root of their mean
    square, with nothing removed.

    `samples` is one array, or several (the captures of one channel),
    pooled sample for sample, so a long capture weighs more than a
    short one, as it does in the average. A sample that is not a finite
    number is refused, since the mean square of a record holding one is
    not a number either.
    """
    flat = np.concatenate([piece.ravel() for piece in _pieces(samples)])
    if flat.size == 0:
        raise ValueError('there are no samples to take an rms of')
    if not np.all(np.isfinite(flat)):
        raise ValueError('the record holds a sample that is not a finite '
                         'number; its rms is not a number either')
    return float(np.sqrt(np.mean(flat ** 2)))


def rms_check(frequencies: ArrayLike, values: ArrayLike,
              samples: ArrayLike | Sequence[ArrayLike], *,
              threshold_db: float = GAP_DB,
              framed_rms: float | None = None,
              window: Any = None, dof: str = '') -> RmsCheck:
    """Check an autospectrum against the record it came from: its rms
    from the area under it beside the record's own.

    The spectrum's rms is `compliance.rms`: every line times its bin's
    width, summed and square-rooted, which on evenly spaced lines is
    the sum of the lines times the spacing, exactly Parseval's total.
    The record's is `time_rms` of `samples`, which should be the span
    the spectrum's frames covered.

    Parameters
    ----------
    frequencies : array_like
        The spectrum's lines, in Hz, evenly spaced.
    values : array_like
        The one-sided density at each line, in the square of the
        record's unit per hertz.
    samples : array_like or sequence of array_like
        The record over the span the frames covered, or the captures of
        one channel, pooled.
    threshold_db : float, default `GAP_DB`
        The gap, in dB either way, beyond which the two disagree.
    framed_rms : float, optional
        The frames' own mean square, windowed and leveled as they were
        averaged, square-rooted, when the averaging is known
        (`check_record` works it out); it tells a window's gap from a
        scaling error.
    window : str or Averaging, optional
        The window the spectrum was averaged with, which names the
        amplitude-correction error by its noise bandwidth.
    dof : str, default ''
        The channel's name, carried on the result.

    Returns
    -------
    RmsCheck
        Both rms, in the record's unit, the gap and the verdict.

    Raises
    ------
    ValueError
        For a negative threshold, a spectrum on uneven lines or with
        fewer than two, or a record with no samples or one that is not
        a finite number.
    """
    from .compliance import rms as area_rms
    from .tone_readout import line_spacing, noise_bandwidth

    if not threshold_db >= 0.0:
        raise ValueError('the threshold is a gap in dB, zero or more')
    f = np.asarray(frequencies, dtype=float)
    if f.size < 2:
        raise ValueError('a spectrum needs two lines or more to have an '
                         'area')
    spacing = line_spacing(f)
    name = bandwidth = None
    if window is not None:
        from .averaging import Averaging, normalize_window
        name = (window.window if isinstance(window, Averaging)
                else normalize_window(window))
        bandwidth = float(noise_bandwidth(window))
    pooled = _pieces(samples)
    count = int(sum(piece.size for piece in pooled))
    return RmsCheck(
        dof=str(dof), spectrum_rms=area_rms(f, values),
        time_rms=time_rms(pooled), framed_rms=(
            None if framed_rms is None else float(framed_rms)),
        threshold_db=float(threshold_db), lines=int(f.size),
        spacing=float(spacing), samples=count, window=name,
        noise_bandwidth=bandwidth)


def _level(frame: np.ndarray, detrend: str) -> np.ndarray:
    """A frame leveled as the averaging levels it before its window:
    nothing, its mean, or its least-squares line taken out."""
    if detrend == 'mean':
        return frame - frame.mean()
    if detrend == 'linear':
        n = np.arange(len(frame), dtype=float)
        slope, intercept = np.polyfit(n, frame, 1)
        return frame - (slope * n + intercept)
    return frame


def matching_records(history: TimeHistory, dof: str,
                     dimension: str) -> list[int]:
    """The records of `history` that are the channel a spectrum's record
    is the autospectrum of: the same DOF, measuring the quantity whose
    square per hertz `dimension` is ('acceleration' for
    'acceleration**2/frequency'). A channel whose units are not declared
    matches a spectrum whose units are not declared, by its DOF alone.
    Every capture of the channel is returned, in order.
    """
    from ..units import UNKNOWN
    from .data import channel_quantities

    if dimension == UNKNOWN:
        wanted = UNKNOWN
    else:
        response, reference = channel_quantities(dimension)
        if response != reference:
            return []
        wanted = response
    return [i for i in range(len(history.response_dof))
            if str(history.response_dof[i]) == str(dof)
            and str(history.ordinate_dim[i]) == wanted]


def refusal(spectrum: Psd, record: int = 0) -> str | None:
    """Why record `record` of `spectrum` cannot be rms-checked, in words,
    or None when it can: an autospectrum on evenly spaced lines. A
    banded spectrum, one written at breakpoints and a cross spectrum
    are refused, since none of them has an area that is a mean square
    of one record."""
    from .data import has_phase
    from .tone_readout import line_spacing

    if spectrum.bandwidth is not None or spectrum.interpolation != 'bin':
        return ('the rms check sums the evenly spaced lines of an averaged '
                'spectrum; this one is banded or written at breakpoints')
    response = spectrum.response_dof[record]
    reference = (response if spectrum.reference_dof is None
                 else spectrum.reference_dof[record])
    if reference != response or has_phase(spectrum.ordinate[record]):
        return (f'record {record} ({response}/{reference}) is a cross '
                'spectrum; the rms check reads an autospectrum')
    if len(spectrum.abscissa) < 2:
        return 'a spectrum needs two lines or more to have an area'
    try:
        line_spacing(spectrum.abscissa)
    except ValueError as reason:
        return str(reason)
    return None


def check_record(spectrum: Psd, history: TimeHistory, record: int = 0, *,
                 averaging: Averaging | None = None,
                 threshold_db: float = GAP_DB) -> RmsCheck:
    """The rms check of record `record` of `spectrum` against the channel
    of `history` it is the autospectrum of.

    The record's rms is read over the span the averaging's frames cover,
    from its start to the end of its last frame, every capture of the
    channel pooled, as the average pools them. A frame leveled before
    its window (`Averaging.detrend`) has its mean or line taken out, so
    the span is leveled the same way before its rms is taken, and the
    notes say so: leveled frame by frame, the frames shed a little more
    of the lowest lines than the span does. The frames' own mean square
    (`RmsCheck.framed_rms`) is worked out when the averaging has the
    spectrum's line spacing; an averaging with another is not the one
    the spectrum was made with, and the notes say that too.

    Parameters
    ----------
    spectrum : Psd
        The autospectrum, computed or imported.
    history : TimeHistory
        The record it was computed from.
    record : int, default 0
        Which record of `spectrum`.
    averaging : Averaging, optional
        How the spectrum was averaged; `history`'s own when omitted, or
        each record whole under a rectangle when it has none, as
        `TimeHistory.compute_psds` falls back.
    threshold_db : float, default `GAP_DB`
        The gap, in dB either way, beyond which the two disagree.

    Returns
    -------
    RmsCheck

    Raises
    ------
    ValueError
        When the record cannot be checked (`refusal`), when `history`
        holds no such channel, or when the frames do not fit in it.
    """
    from .averaging import Averaging

    reason = refusal(spectrum, record)
    if reason is not None:
        raise ValueError(reason)
    dof = str(spectrum.response_dof[record])
    rows = matching_records(history, dof, str(spectrum.ordinate_dim[record]))
    if not rows:
        raise ValueError(f'the time history holds no {dof} channel measuring '
                         f'what {spectrum.ordinate_dim[record]} is the '
                         'density of')
    rate = history.sample_rate
    samples = len(history.abscissa)
    if averaging is None:
        averaging = history.averaging
    if averaging is None:
        averaging = Averaging.for_records(samples)
    if not averaging.fits(samples, rate):
        raise ValueError(
            f'{averaging.frames} frames of {averaging.frame_length} samples '
            f'from {averaging.start:g} s do not fit in {samples} samples')
    first = averaging.start_sample(rate)
    stop = first + averaging.span
    n = averaging.frame_length
    shape = averaging.shape()
    power = float(np.sum(shape ** 2))
    notes = []
    spans, framed = [], []
    for i in rows:
        x = np.asarray(np.real(history.ordinate[i]), dtype=float)
        spans.append(_level(x[first:stop], averaging.detrend))
        for k in range(averaging.frames):
            at = first + k * averaging.hop
            frame = _level(x[at:at + n], averaging.detrend) * shape
            framed.append(float(np.sum(frame ** 2)) / power)
    if averaging.detrend != 'none':
        notes.append(f"the frames were leveled ({averaging.detrend}) before "
                     "their window, so the record's span is leveled the "
                     'same way before its rms is taken')
    if len(rows) > 1:
        notes.append(f'{len(rows)} captures of the channel, pooled as the '
                     'average pools them')
    lines = rate / n
    spacing = float(np.mean(np.diff(np.asarray(spectrum.abscissa,
                                               dtype=float))))
    framed_rms: float | None = float(np.sqrt(np.mean(framed)))
    if abs(spacing - lines) > 1e-6 * max(spacing, lines):
        framed_rms = None
        notes.append(f"the averaging gives lines {lines:.4g} Hz apart and "
                     f'the spectrum has them {spacing:.4g} Hz apart, so it '
                     "was not averaged this way; the frames' own mean "
                     'square is not read')
    check = rms_check(spectrum.abscissa, spectrum.ordinate[record], spans,
                      threshold_db=threshold_db, framed_rms=framed_rms,
                      window=averaging if framed_rms is not None else None,
                      dof=dof)
    origin = float(history.abscissa[0])
    span = (origin + first / rate, origin + stop / rate)
    return replace(check, span=span, notes=tuple(notes))
