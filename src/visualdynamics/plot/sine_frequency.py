"""A sine tone's frequency error against frequency, with the band drawn.

The question this view answers is the frequency check's: where, if
anywhere, was the tone on the channel off the frequency the profile asks
for by more than the band allows. So the view puts the error
(`core.sine_frequency`, the tone's frequency less the specified, as a
share of the specified, in percent) against the specified frequency, on
the log frequency axis every sine level is read on, and draws the band
either side of zero. Everything is said by line style, marker shape and
words as well as by color, never by hue alone:

- the band, a dashed line above and below zero, named in the legend in
  words ('band ±2 %', or '±2 % or 1 Hz, whichever is greater', when a
  floor widens it at low frequency, where the dashed lines open out);
- each channel, a solid line with a hollow marker of its own shape every
  few lines, named with its worst reading and its verdict in words;
- a line outside the band, the same marker filled, as the distortion
  view marks a line over its limit.

A line the check could not judge is left out, a gap in the curve, and
the notes say how many. The error axis is linear and centered on zero,
because a tone runs off its frequency either way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the marker each channel wears, in the order the check holds them
SYMBOLS = ('o', 's', 'd', 't1', 'star')

#: about how many hollow markers name each curve along its length
MARKERS_PER_CURVE = 12

#: the markers' size, pixels
MARKER_SIZE = 8


def _rows(check: Any, dof: str | None) -> list[int]:
    """The check's rows drawn: one channel's, or all of them."""
    return list(range(len(check.dofs))) if dof is None else [check.row(dof)]


def channel_words(check: Any, row: int) -> str:
    """One channel in the legend's words: its name, its worst reading and
    its verdict, such as '101Z+: worst +1.5 % at 100 Hz, 79 lines outside
    the band'."""
    dof = check.dofs[row]
    worst = check.worst(dof)
    if worst is None:
        return f'{dof}: no line judged'
    frequency, _error, relative = worst
    over = int(check.over[row].sum())
    verdict = (f'{over} line{"s" * (over != 1)} outside the band' if over
               else 'within the band')
    return (f'{dof}: worst {100.0 * relative:+.3g} % at {frequency:.4g} Hz, '
            f'{verdict}')


def band_words(check: Any) -> str:
    """The band in the legend's words: 'band ±2 %'."""
    return f'band {check.band_words()}'


def readout(check: Any, dof: str | None = None) -> dict[str, str]:
    """The check's numbers as the window's readout words them, on
    channel `dof` (every channel when None): 'band', 'worst', 'over' and
    'verdict'. A number the check does not give reads as a dash; the
    notes say why."""
    worst = check.worst(dof)
    rows = _rows(check, dof)
    over = int(sum(check.over[r].sum() for r in rows))
    judged = int(sum(check.judged[r].sum() for r in rows))
    words = {'band': check.band_words()}
    if worst is None:
        words.update(worst='—', over='—', verdict='not judged')
        return words
    frequency, error, relative = worst
    words['worst'] = (f'{100.0 * relative:+.3g} % ({error:+.3g} Hz) at '
                      f'{frequency:.4g} Hz')
    if over:
        hit = np.zeros(len(check.specified), dtype=bool)
        for r in rows:
            hit |= check.over[r]
        span = check.specified[hit]
        words['over'] = (f'{over} of {judged} lines, {span.min():.4g} to '
                         f'{span.max():.4g} Hz')
        words['verdict'] = 'outside the band'
    else:
        words['over'] = f'none of {judged} lines'
        words['verdict'] = 'within the band'
    return words


def draw_frequency(plot: Any, check: Any, *, dof: str | None = None,
                   colors: Any = None, legend: bool = True) -> list[Any]:
    """Draw a `FrequencyCheck` onto a pyqtgraph plot item and return the
    items drawn.

    The band dashed above and below zero in the plot's foreground, then
    each channel (or only `dof`) as a solid curve in its color, in
    percent of the specified frequency, with a hollow marker of its own
    shape every few lines and the same marker filled on every line
    outside the band. With `legend`, the band and each channel are named
    for the plot's legend in words (`band_words`, `channel_words`); the
    panel's small preview passes False and words them in its readout
    instead. The plot is expected on a log frequency axis.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme
    from . import curve_color

    colors = resolve_theme(None) if colors is None else colors
    ink = colors['plot_foreground']

    def named(text):
        return {'name': text} if legend else {}

    items = []
    order = np.argsort(check.specified)
    f = check.specified[order]
    edge = 100.0 * check.allowed[order] / f
    dashed = pg.mkPen(ink, width=1.5, style=Qt.DashLine)
    items.append(plot.plot(f, edge, pen=dashed, **named(band_words(check))))
    items.append(plot.plot(f, -edge, pen=dashed))
    values = 100.0 * check.relative
    step = max(1, len(f) // MARKERS_PER_CURVE)
    for k, r in enumerate(_rows(check, dof)):
        color = curve_color(k, colors)
        symbol = SYMBOLS[k % len(SYMBOLS)]
        shown = np.where(check.judged[r], values[r], np.nan)[order]
        over = check.over[r][order]
        items.append(plot.plot(f, shown, pen=pg.mkPen(color, width=1.5),
                               connect='finite',
                               **named(channel_words(check, r))))
        marked = np.zeros(len(shown), dtype=bool)
        marked[::step] = True
        marked &= np.isfinite(shown) & ~over
        if marked.any():
            items.append(plot.plot(
                f[marked], shown[marked], pen=None, symbol=symbol,
                symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(color, width=1.2),
                symbolBrush=None))
        if over.any():
            items.append(plot.plot(
                f[over], shown[over], pen=None, symbol=symbol,
                symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(color),
                symbolBrush=pg.mkBrush(color)))
    return items


def error_range(check: Any, dof: str | None = None) -> float:
    """How far either side of zero the error axis reaches, percent: past
    the band and past the widest judged error, by a quarter and a tenth."""
    rows = _rows(check, dof)
    band = float(np.max(100.0 * check.allowed / check.specified))
    judged = np.where(check.judged[rows], np.abs(check.relative[rows]),
                      np.nan)
    top = float(np.nanmax(judged, initial=0.0)) * 100.0
    return max(1.25 * band, 1.1 * top)


def build_frequency(layout: Any, check: Any, *, dof: str | None = None,
                    theme: Any = None) -> Any:
    """Draw a frequency check into a graphics layout, its legend below
    it, and return the plot item.

    The title names the tone and the verdict in words; the notes the
    check carries, if any, go under it, since a verdict that left lines
    unjudged should not be read without them.
    """
    from ..theme import theme as resolve_theme
    from . import legend_below
    from .scalogram import _decade_ticks

    colors = resolve_theme(theme)
    plot = layout.addPlot(row=0, col=0)
    plot.setLogMode(x=True, y=False)
    plot.getAxis('bottom').setTicks([_decade_ticks(np.array(
        [float(np.min(check.specified)), float(np.max(check.specified))]))])
    plot.showGrid(x=True, y=True, alpha=0.2)
    legend_below(layout, plot, 0, colors)
    rows = _rows(check, dof)
    over = any(check.over[r].any() for r in rows)
    title = (f'{check.tone}: frequency error against frequency, '
             + ('outside the band' if over else 'within the band'))
    if check.notes:
        title += '<br>' + '<br>'.join(check.notes)
    plot.setTitle(title, color=colors['plot_foreground'], size='9pt')
    draw_frequency(plot, check, dof=dof, colors=colors)
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', 'frequency error [%]')
    reach = error_range(check, dof)
    plot.setYRange(-reach, reach, padding=0.0)
    return plot


__all__ = ['SYMBOLS', 'band_words', 'build_frequency', 'channel_words',
           'draw_frequency', 'error_range', 'readout']
