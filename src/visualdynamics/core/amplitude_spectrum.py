"""A record's amplitude spectrum: each line scaled so a sine reads its
amplitude, in peak units, under a window chosen for it.

`TimeHistory.compute_spectra` is the raw transform: a rectangular
window, no scaling (a sine of amplitude A on a line reads A·N/2) and a
complex mean across the averages, so content whose phase moves from one
frame to the next averages toward zero. Other code depends on it as it
is. This module is the scaled spectrum beside it, the one a tone is read
from in g peak.

The scaling, for a frame of N samples x_n windowed by w_n:

    A(f_k) = 2·|X(k)| / Σw,    X(k) = Σ w_n x_n e^(-i2πkn/N)

with the 2 left off at the DC line and, for an even frame, the Nyquist
line, which have no mirror twin at negative frequency to fold in. A
sine of amplitude A that sits on line k transforms to A·Σw/2 there, so
the line reads A whatever the window. Σw is the window's amplitude
correction: Σw/N, the window's mean, is its coherent gain (0.5 for
Hann), and dividing by Σw rather than N undoes it (a correction of
×2.000 for Hann). It is not the correction a density takes. A density is
mean square per hertz and is corrected by the sum of the window's
squares (`TimeHistory.compute_psds`); the two differ by the window's
noise bandwidth, and applying the amplitude correction to a density
reads every ASD 1.5 times too high under Hann.

The frames' magnitudes are averaged, not their complex values. A tone
reads its amplitude in every frame whatever its phase there, so the
average reads it too: a tone that is not locked to the frames, or a
record saved as captures with a fresh phase each, keeps its amplitude
where a complex mean would shrink it. A broadband floor reads its mean
magnitude per line, which is not a level: it moves with the line spacing
and the window, and a random record's level is read from its ASD. An
amplitude spectrum is for tones.

The window is chosen here, not taken from the record's averaging. The
frames are the averaging's (its frame length, overlap, start, frame
count and detrend), because those are what was set for the record; but
the window right for a density of random data and the window right for a
tone's amplitude are not the same window. Hann is the default. It reads
a tone on a line exactly and one midway between lines up to 1.42 dB low,
because the tallest line then sits half a line off the tone's frequency
(scalloping). The flat top's crown is flat across a line, so a tone
reads within about 0.01 dB of its amplitude wherever it falls, at the
cost of a main lobe several lines wide; it is the window for amplitude
accuracy. Its correction depends on the variant a program implements and
how it is normalized; the one here is scipy's `flattop`, periodic, whose
coherent gain is 0.2156 (a correction of ×4.639). Every reading states
its window and its correction, so a number read here is never carried to
another program without them.

References
----------
The amplitude correction is the window's coherent gain and the loss of
a tone between lines its scalloping loss, both Harris's; the one-sided
scaling and the doubling left off at DC and Nyquist are how the
autospectrum beside it is scaled.

1. Harris, F. J. (1978). "On the Use of Windows for Harmonic Analysis
   with the Discrete Fourier Transform." *Proceedings of the IEEE*,
   66(1), 51-83.
   [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
   Coherent gain, equivalent noise bandwidth and scalloping loss,
   window by window, and the flat-top window.
2. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The finite Fourier transform
   of a record and its one-sided spectra.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

#: the frame the window's arithmetic is worked out on when no frame
#: length is given. A periodic cosine-sum window's mean is its first
#: coefficient at any length that holds it, so the coherent gain does not
#: depend on this; it only has to be long enough for every window offered
GAIN_LENGTH = 4096


def coherent_gain(window: Any = 'hann', length: int | None = None,
                  parameter: float | None = None) -> float:
    """A window's coherent gain: its mean, Σw/N.

    The share of a sine's amplitude a line on the sine keeps through the
    window: 1 for a rectangle (scipy's 'boxcar'), 0.5 for Hann, 0.2156
    for scipy's flat top. Its reciprocal is the amplitude correction
    (`amplitude_correction`).

    Parameters
    ----------
    window : str or Averaging, default 'hann'
        A window by any name `core.averaging.normalize_window` takes, or
        an `Averaging`, whose window, parameter and frame length are
        used.
    length : int, optional
        The frame length in samples; `GAIN_LENGTH` when omitted. The
        window is periodic, as a spectral analysis uses it.
    parameter : float, optional
        The Tukey window's alpha or the Kaiser window's beta.

    Returns
    -------
    float
    """
    from .averaging import Averaging, window_shape

    if isinstance(window, Averaging):
        parameter = window.window_parameter
        length = window.frame_length if length is None else length
        window = window.window
    n = GAIN_LENGTH if length is None else int(length)
    shape = np.asarray(window_shape(window, n, parameter), dtype=np.float64)
    return float(np.mean(shape))


def amplitude_correction(window: Any = 'hann', length: int | None = None,
                         parameter: float | None = None) -> float:
    """The factor a windowed transform's lines are multiplied by so a
    sine reads its amplitude: N/Σw, one over the coherent gain.

    2.000 for Hann, 4.639 for scipy's flat top, 1 for a rectangle. The
    arguments are as for `coherent_gain`.
    """
    return 1.0 / coherent_gain(window, length, parameter)


def scalloping(window: Any = 'hann', length: int | None = None,
               parameter: float | None = None) -> float:
    """The share of its amplitude a sine midway between two lines reads
    at the taller of them: 0.849 for Hann (1.42 dB low), 0.637 for a
    rectangle (3.92 dB low), within a part in a thousand of 1 for the
    flat top.

    Harris's scalloping loss, worked out from the window's own shape
    (`core.tone_readout.leakage`), so it holds for every window offered.
    The arguments are as for `coherent_gain`.
    """
    from .tone_readout import leakage

    # the leakage kernel is in power, the share of a tone's mean square
    # a line receives; its root is the share of the amplitude
    return float(np.sqrt(leakage(window, length, parameter).scallop(0.5)))


def amplitude_lines(frames: Any, shape: Any) -> np.ndarray:
    """The amplitude spectrum of frames already windowed, their
    magnitudes averaged.

    Parameters
    ----------
    frames : array_like
        One windowed frame per row, every row the same length N.
    shape : array_like
        The window they were multiplied by, N samples: its sum is the
        amplitude correction.

    Returns
    -------
    numpy.ndarray
        N//2 + 1 lines, from DC up: 2·mean|X|/Σw, and mean|X|/Σw at DC
        and at the Nyquist line of an even frame.
    """
    import scipy.fft as scipy_fft

    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    total = float(np.sum(np.asarray(shape, dtype=np.float64)))
    if not total > 0.0:
        raise ValueError('the window sums to nothing, so no line can be '
                         'scaled to an amplitude')
    n = frames.shape[1]
    lines = np.mean(np.abs(scipy_fft.rfft(frames, axis=1, workers=-1)),
                    axis=0)
    scale = np.full(lines.shape, 2.0 / total)
    # no mirror twin at DC, nor at the Nyquist line of an even frame:
    # doubled, a record's mean would read twice itself
    scale[0] = 1.0 / total
    if n % 2 == 0:
        scale[-1] = 1.0 / total
    return lines * scale


@dataclass(frozen=True, eq=False)
class AmplitudeSpectrum:
    """What `amplitude_spectrum` returns: a record's amplitude spectrum,
    one row per channel, with the window and its correction stated.

    `amplitude` is in peak units, SI as the record is stored (m/s² for an
    acceleration). `frames` is how many frames each channel's
    magnitudes were averaged over. `coherent_gain` is the window's mean
    and `correction` its reciprocal, the factor its lines were
    multiplied by; `noise_bandwidth` is in lines and `scalloping` is the
    share of its amplitude a sine midway between lines reads.
    `main_lobe` is the window's main-lobe half-width in whole lines (2
    for Hann, 1 for a rectangle): a record's mean leaks into that many
    lines from DC up, which `tallest` therefore passes over.
    `from_averaging` says whether the frames are the record's averaging
    (False: each record whole as one frame, the record having none).
    """

    frequencies: np.ndarray
    amplitude: np.ndarray
    response_dof: list[str]
    ordinate_dim: list[str]
    dimension_hint: list[str | None]
    window: str
    window_parameter: float | None
    frame_length: int
    sample_rate: float
    frames: tuple[int, ...]
    coherent_gain: float
    noise_bandwidth: float
    scalloping: float
    main_lobe: int = 1
    from_averaging: bool = True
    labels: list[str] = field(default_factory=list)

    @property
    def correction(self) -> float:
        """The amplitude correction applied: N/Σw (2.000 for Hann)."""
        return 1.0 / self.coherent_gain

    @property
    def spacing(self) -> float:
        """The line spacing in Hz: the sample rate over the frame
        length."""
        return self.sample_rate / self.frame_length

    @property
    def scalloping_db(self) -> float:
        """How far low, in dB, a sine midway between lines reads: 1.42
        for Hann, as a positive number."""
        return float(-20.0 * np.log10(self.scalloping))

    def label(self, record: int) -> str:
        """The channel's name, as the record labels it."""
        if record < len(self.labels):
            return self.labels[record]
        return str(self.response_dof[record])

    def tallest(self, record: int = 0,
                band: tuple[float, float] | None = None
                ) -> tuple[float, float]:
        """(frequency, amplitude) of the tallest line of channel
        `record`, the DC line and the lines its main lobe reaches left
        out. A record's mean is not a tone, and it leaks: under Hann a
        mean of 5 g reads 5 g at DC and 5 g again on the line above
        (`main_lobe`).

        `band` is (low, high) in Hz to look within; the whole spectrum
        above DC when omitted. A line that is not a number is passed
        over. Raises ValueError when no line is left to read.
        """
        f = self.frequencies
        values = np.asarray(self.amplitude[record], dtype=np.float64)
        keep = (np.arange(len(f)) >= self.main_lobe) & np.isfinite(values)
        if band is not None:
            low, high = band
            keep &= (f >= low) & (f <= high)
        if not keep.any():
            raise ValueError('no line to read in that band')
        at = int(np.flatnonzero(keep)[np.argmax(values[keep])])
        return float(f[at]), float(values[at])

    def correction_text(self) -> str:
        """The window and its correction, in words, as the plot and the
        pane state them: 'hann window, amplitude correction ×2.000'."""
        name = self.window
        if self.window_parameter is not None:
            name = f'{name} ({self.window_parameter:g})'
        return f'{name} window, amplitude correction ×{self.correction:.3f}'

    def describe(self, record: int = 0, unit: str = '',
                 scale: float = 1.0) -> str:
        """One line for a status bar: the tallest line of channel
        `record` in peak units, then the window and its correction.
        `scale` converts the amplitude into `unit`."""
        frequency, amplitude = self.tallest(record)
        value = f'{amplitude * scale:.4g} {unit} peak'.replace('  ', ' ')
        return (f'{self.label(record)}: tallest line {value} at '
                f'{frequency:.4g} Hz; {self.correction_text()}, '
                f'{_frames_text(self.frames[record])} averaged as '
                'magnitudes')


def _frames_text(count: int) -> str:
    return '1 frame' if count == 1 else f'{count} frames'


def channel_row(history: Any, record: int) -> int:
    """The row of an amplitude spectrum of `history` that record
    `record` is averaged into.

    A row is a channel (`TimeHistory.channel_key`), in the order its
    first record comes, so a history with one record per channel has
    its rows in record order, and every capture of a channel saved as
    captures maps to that channel's one row.
    """
    keys: list = []
    for i in range(len(history.response_dof)):
        key = history.channel_key(i)
        if key not in keys:
            keys.append(key)
    return keys.index(history.channel_key(record))


def amplitude_spectrum(history: Any, averaging: Any = None, *,
                       window: Any = 'hann',
                       parameter: float | None = None) -> AmplitudeSpectrum:
    """The amplitude spectrum of every channel of a time history, each
    line scaled so a sine reads its amplitude, the frames' magnitudes
    averaged.

    The frames are the averaging's: `averaging` when given, else the
    record's own, else each record whole as one frame, as
    `TimeHistory.compute_psds` cuts them. Only the window is replaced:
    `window` and `parameter` are this spectrum's own (the module says
    why). Channels are grouped as a PSD groups them
    (`TimeHistory.channel_key`), so a record saved as captures averages
    every capture's frames together.

    Parameters
    ----------
    history : TimeHistory
        The record, evenly sampled.
    averaging : Averaging, optional
        How to cut it into frames.
    window : str, default 'hann'
        Any window `core.averaging.WINDOWS` names; 'flattop' for a
        tone's amplitude wherever it falls between lines.
    parameter : float, optional
        The Tukey window's alpha or the Kaiser window's beta.

    Returns
    -------
    AmplitudeSpectrum

    Raises
    ------
    ValueError
        If the samples are not evenly spaced, the frames do not fit in
        the record, or the window is not one offered.
    """
    from .averaging import normalize_window

    key = normalize_window(window)
    given = averaging if averaging is not None else history.averaging
    used = history._resolved_averaging(averaging)
    used = replace(used, window=key, window_parameter=(
        None if parameter is None else float(parameter)))
    # the framing every spectrum of a record shares, so these frames are
    # the PSD's frames exactly; its density scale is not this one
    frequencies, _density, groups = history._spectral_frame(used)
    shape = used.shape()
    rows, dofs, dims, hints, counts, labels = [], [], [], [], [], []
    first = {}
    for i in range(len(history.response_dof)):
        first.setdefault(history.channel_key(i), i)
    for (dof, dim, hint), windowed in groups.items():
        rows.append(amplitude_lines(windowed, shape))
        dofs.append(dof)
        dims.append(dim)
        hints.append(hint)
        counts.append(len(windowed))
        labels.append(history.record_label(first[(dof, dim, hint)]))
    from .tone_readout import leakage

    kernel = leakage(used)
    return AmplitudeSpectrum(
        frequencies=np.asarray(frequencies, dtype=np.float64),
        amplitude=np.asarray(rows, dtype=np.float64),
        response_dof=dofs, ordinate_dim=dims, dimension_hint=hints,
        window=key, window_parameter=used.window_parameter,
        frame_length=int(used.frame_length),
        sample_rate=float(history.sample_rate),
        frames=tuple(counts),
        coherent_gain=float(np.mean(shape)),
        noise_bandwidth=float(kernel.noise_bandwidth),
        scalloping=float(np.sqrt(kernel.scallop(0.5))),
        main_lobe=int(kernel.half_width),
        from_averaging=given is not None,
        labels=labels)
