"""A record's amplitude spectrum, drawn with its window and correction
named on it.

One channel of a `core.amplitude_spectrum.AmplitudeSpectrum`, in peak
units of the unit system, against frequency. The title names what the
lines were scaled by, because an amplitude read off a windowed transform
means nothing without it: the window, its amplitude correction, the
line spacing and the frames whose magnitudes were averaged, then how far
low this window reads a sine that falls midway between two lines. The
tallest line above DC is marked with a downward triangle, its amplitude
and frequency written beside it and in the legend.

Nothing is told apart by hue alone: there is one curve, the reading is a
triangle with its value in words, and every mark is named in the legend
under the plot. The amplitude axis is logarithmic by default, so a tone
and the floor it stands on are both seen; a line at zero, which a log
axis cannot place, is left out of the drawing and not of the reading.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _reading_plot

#: the triangle's size, pixels
MARKER_SIZE = 12


def display_scale(spectrum: Any, record: int, unit_system: Any
                  ) -> tuple[float, str]:
    """(factor, unit) that put channel `record`'s amplitudes, stored in
    SI, into `unit_system`'s units: 1/9.80665 and 'g' for an
    acceleration shown in g; 1 and '' for a channel whose units are not
    declared. An amplitude is a size, so it converts by the unit's scale
    alone, never with an offset."""
    from ..units import UNKNOWN

    dimension = spectrum.ordinate_dim[record]
    if dimension == UNKNOWN:
        return 1.0, ''
    scale = float(unit_system.transform(dimension)[0])
    return 1.0 / scale, unit_system.label_text(dimension)


def title_text(spectrum: Any, record: int = 0) -> str:
    """The plot's title: the channel and the window with its correction,
    then the spacing, the frames averaged and the window's loss between
    lines. Two lines, joined by an HTML break."""
    frames = spectrum.frames[record]
    counted = '1 frame' if frames == 1 else f'{frames} frames'
    return (f'{spectrum.label(record)}: amplitude spectrum, '
            f'{spectrum.correction_text()}<br>'
            f'{spectrum.spacing:.4g} Hz lines, {counted} averaged as '
            f'magnitudes; a sine midway between lines reads '
            f'{spectrum.scalloping_db:.2f} dB low')


def build_amplitude_spectrum(layout: Any, spectrum: Any, record: int = 0,
                             *, unit_system: Any = None, theme: Any = None,
                             log: bool = True,
                             band: tuple[float, float] | None = None
                             ) -> tuple[float, float] | None:
    """Draw channel `record` of `spectrum`, an `AmplitudeSpectrum`, into
    `layout`, with its tallest line marked.

    Amplitudes are drawn in `unit_system`'s units, on a logarithmic axis
    unless `log` is False. `band` is (low, high) in Hz for the tallest
    line to be looked for in, as `AmplitudeSpectrum.tallest` takes it.
    Returns that line as (frequency, amplitude in SI), or None when the
    band holds no line.
    """
    import pyqtgraph as pg

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM, UNKNOWN
    from . import legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    factor, unit = display_scale(spectrum, record, us)
    frequencies = np.asarray(spectrum.frequencies, dtype=np.float64)
    shown = np.asarray(spectrum.amplitude[record], dtype=np.float64) * factor

    plot = _reading_plot(layout, 0)
    legend_below(layout, plot, 0, colors)
    plot.setLogMode(False, bool(log))
    drawn = shown.copy()
    if log:
        # a log axis has no place for zero; those lines are gaps
        drawn[~(drawn > 0.0)] = np.nan
    plot.plot(frequencies, drawn, connect='finite',
              pen=pg.mkPen(colors['response_curve'], width=1.5),
              name=f'{spectrum.label(record)}, peak per line')
    try:
        tallest = spectrum.tallest(record, band)
    except ValueError:
        tallest = None
    if tallest is not None:
        frequency, amplitude = tallest
        top = amplitude * factor
        value = f'{top:.4g} {unit} peak'.replace('  ', ' ')
        plot.plot([frequency], [top], pen=None, symbol='t',
                  symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(foreground),
                  symbolBrush=pg.mkBrush(foreground),
                  name=f'tallest line: {value} at {frequency:.4g} Hz')
        words = pg.TextItem(value, color=foreground, anchor=(-0.08, 0.5))
        words.setPos(frequency,
                     float(np.log10(top)) if log and top > 0.0 else top)
        plot.getViewBox().addItem(words, ignoreBounds=True)
    dimension = spectrum.ordinate_dim[record]
    if dimension == UNKNOWN:
        hint = spectrum.dimension_hint[record]
        what = f'{hint} [units undefined]' if hint else 'units undefined'
        plot.setLabel('left', f'{what}, peak')
    else:
        plot.setLabel('left', f'amplitude [{us.label_html(dimension)} peak]')
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setTitle(title_text(spectrum, record), color=foreground, size='9pt')
    return tallest
