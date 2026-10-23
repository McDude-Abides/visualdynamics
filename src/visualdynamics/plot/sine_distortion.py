"""A sine tone's distortion against frequency, with the limit drawn.

The question this view answers is the one the distortion check asks
along a sweep: where, if anywhere, was there enough on the channel
besides the tone to change the test. So the view puts the distortion
(`core.sine_distortion`, the RMS of everything that is not the tone over
the RMS of the tone, in percent) against frequency, on the log frequency
axis every sine level is read on, and draws the limit across it.
Everything is said by line style, marker shape and words as well as by
color, never by hue alone:

- the limit, a dashed line, named in the legend with its percentage, its
  dB and the tolerance on the ratio it comes from ('limit 32.0 %
  (-9.9 dB): unfiltered 5 % over the tone');
- each channel, a solid line with a hollow marker of its own shape every
  few lines, named with its worst reading and its verdict in words;
- a line over the limit, the same marker filled, so being over reads as
  a change of fill on a shape the reader has already matched to the
  legend, as the SRS tolerance view marks a point outside its band.

A line the check could not judge (the tone under its noise floor, or a
tone the levels do not hold live beside it) is left out, a gap in the
curve, and the notes say how many. The distortion axis is linear and
starts at zero, because the limit is read as a height and a clean channel
reads near zero, where a log axis has no floor.
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
    its verdict, such as '101Z+: worst 12.3 % at 412 Hz, within the
    limit' or '101Z+: worst 41.0 % at 400 Hz, 12 lines over the limit'."""
    dof = check.dofs[row]
    worst = check.worst(dof)
    if worst is None:
        return f'{dof}: no line judged'
    frequency, value = worst
    over = int(check.over[row].sum())
    verdict = (f'{over} line{"s" * (over != 1)} over the limit' if over
               else 'within the limit')
    return f'{dof}: worst {100.0 * value:.1f} % at {frequency:.4g} Hz, {verdict}'


def limit_words(check: Any) -> str:
    """The limit in the legend's words: 'limit 32.0 % (-9.9 dB):
    unfiltered 5 % over the tone'."""
    from ..core.sine_distortion import decibels

    return (f'limit {100.0 * check.limit:.1f} % '
            f'({float(decibels(check.limit)):.1f} dB): unfiltered '
            f'{100.0 * check.excess:g} % over the tone')


def readout(check: Any, dof: str | None = None) -> dict[str, str]:
    """The check's numbers as the window's readout words them, on
    channel `dof` (every channel when None): 'limit', 'worst', 'over' and
    'verdict'. A number the check does not give reads as a dash; the
    notes say why."""
    from ..core.sine_distortion import decibels

    worst = check.worst(dof)
    rows = _rows(check, dof)
    over = int(sum(check.over[r].sum() for r in rows))
    judged = int(sum(check.judged[r].sum() for r in rows))
    words = {'limit': (f'{100.0 * check.limit:.1f} % '
                       f'({float(decibels(check.limit)):.1f} dB)')}
    if worst is None:
        words.update(worst='—', over='—', verdict='not judged')
        return words
    frequency, value = worst
    words['worst'] = (f'{100.0 * value:.1f} % '
                      f'({float(decibels(value)):.1f} dB) at '
                      f'{frequency:.4g} Hz')
    if over:
        hit = np.zeros(len(check.frequency), dtype=bool)
        for r in rows:
            hit |= check.over[r]
        span = check.frequency[hit]
        words['over'] = (f'{over} of {judged} lines, {span.min():.4g} to '
                         f'{span.max():.4g} Hz')
        words['verdict'] = 'over the limit'
    else:
        words['over'] = f'none of {judged} lines'
        words['verdict'] = 'within the limit'
    return words


def draw_distortion(plot: Any, check: Any, *, dof: str | None = None,
                    colors: Any = None, legend: bool = True) -> list[Any]:
    """Draw a `DistortionCheck` onto a pyqtgraph plot item and return the
    items drawn.

    The limit dashed in the plot's foreground, then each channel (or
    only `dof`) as a solid curve in its color, in percent, with a hollow
    marker of its own shape every few lines and the same marker filled on
    every line over the limit. With `legend`, the limit and each channel
    are named for the plot's legend in words (`limit_words`,
    `channel_words`); the panel's small preview passes False and words
    them in its readout instead. The plot is expected on a log frequency
    axis; the curves are given in Hz and pyqtgraph takes the log.
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
    low, high = float(check.frequency.min()), float(check.frequency.max())
    percent = 100.0 * check.limit
    items.append(plot.plot([low, high], [percent, percent],
                           pen=pg.mkPen(ink, width=1.5, style=Qt.DashLine),
                           **named(limit_words(check))))
    values = 100.0 * check.distortion
    step = max(1, len(check.frequency) // MARKERS_PER_CURVE)
    for k, r in enumerate(_rows(check, dof)):
        color = curve_color(k, colors)
        symbol = SYMBOLS[k % len(SYMBOLS)]
        shown = np.where(check.judged[r], values[r], np.nan)
        items.append(plot.plot(check.frequency, shown,
                               pen=pg.mkPen(color, width=1.5),
                               connect='finite',
                               **named(channel_words(check, r))))
        marked = np.zeros(len(shown), dtype=bool)
        marked[::step] = True
        marked &= np.isfinite(shown) & ~check.over[r]
        if marked.any():
            items.append(plot.plot(
                check.frequency[marked], shown[marked], pen=None,
                symbol=symbol, symbolSize=MARKER_SIZE,
                symbolPen=pg.mkPen(color, width=1.2), symbolBrush=None))
        over = check.over[r]
        if over.any():
            items.append(plot.plot(
                check.frequency[over], shown[over], pen=None,
                symbol=symbol, symbolSize=MARKER_SIZE,
                symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color)))
    return items


def build_distortion(layout: Any, check: Any, *, dof: str | None = None,
                     theme: Any = None) -> Any:
    """Draw a distortion check into a graphics layout, its legend below
    it, and return the plot item.

    The title names the tone and the verdict in words; the notes the
    check carries, if any, go under it, since a verdict that left lines
    unjudged should not be read without them.
    """
    from ..theme import theme as resolve_theme
    from . import legend_below

    colors = resolve_theme(theme)
    plot = layout.addPlot(row=0, col=0)
    plot.setLogMode(x=True, y=False)
    plot.showGrid(x=True, y=True, alpha=0.2)
    legend_below(layout, plot, 0, colors)
    rows = _rows(check, dof)
    over = any(check.over[r].any() for r in rows)
    title = (f'{check.tone}: distortion against frequency, '
             + ('over the limit' if over else 'within the limit'))
    if check.notes:
        title += '<br>' + '<br>'.join(check.notes)
    plot.setTitle(title, color=colors['plot_foreground'], size='9pt')
    draw_distortion(plot, check, dof=dof, colors=colors)
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', 'distortion [%]')
    top = np.nanmax(np.where(check.judged[rows], check.distortion[rows],
                             np.nan), initial=0.0)
    plot.setYRange(0.0, 100.0 * max(1.25 * check.limit, 1.1 * top),
                   padding=0.0)
    return plot
