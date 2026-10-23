"""A record's resampling drawn on its autospectrum: the Nyquist
frequency, the span, the dropped lines, and where content above the
new Nyquist frequency folds.

One plot, frequency across on a linear axis (folding is a reflection
about multiples of the sample rate, which a linear axis shows as one)
and density up on a log axis. Four spectra on one line spacing
(`core.resample.read_resample`):

- the record as recorded, thin, out past the new Nyquist frequency, so
  the content that will fold is seen where it is;
- the record filtered first and then decimated, the resampling done
  right, solid and heavy;
- decimated with no filter, dashed;
- decimated and then low-passed through the same filter, dotted.
  Inside the span it lies on the dashed curve wherever an alias is: a
  low-pass applied after sampling leaves the alias where it is, and the
  two curves coinciding at the alias is the picture of that.

The new Nyquist frequency is a dashed vertical line and the span's top
a solid one, each named; the lines between them, the anti-alias
filter's roll-off that an analyzer computes and drops, are hatched and
counted in the legend. Each folded frequency is a hollow downward
triangle on the record as recorded, joined by a dotted line to a filled
triangle where it lands on the bare decimation, with the fold and what
each order of filtering left there written beside it.

Nothing is told apart by hue alone: the spectra differ by line style
and weight, the lines by style and their labels, the dropped lines by
their hatching, and the folds by hollow against filled markers and by
words. Densities are drawn in the unit system's units.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _reading_plot

#: the fold markers' size, pixels
MARKER_SIZE = 11

#: how far below the tallest line the density axis reaches, in decades:
#: a filtered-first record is well over 100 dB down at an alias, and
#: what it falls to below that is rounding, not signal
DECADES = 13.0

#: how far past the new Nyquist frequency the frequency axis runs, as a
#: share of it, when no fold reaches further: room to see what is above
REACH = 1.25


def _view(value: float, log: bool) -> float:
    """A coordinate in the view box's own terms: the logarithm on a log
    axis, where a TextItem lives."""
    return float(np.log10(max(value, 1e-300))) if log else float(value)


def fold_label(fold: Any) -> str:
    """What is written beside a fold: where it lands, and what each
    order of filtering left there, in dB against the bare decimation."""
    from ..core.resample import level_change

    where = 'in the span' if fold.in_span else 'among the dropped lines'
    text = f'{fold.frequency:g} Hz folds to {fold.alias:g} Hz\n{where}'
    if np.isfinite(fold.after_db):
        text += f'\nlow-passed after: {level_change(fold.after_db)}'
    if np.isfinite(fold.filtered_db):
        text += f'\nfiltered first: {level_change(fold.filtered_db)}'
    return text


def build_resample(layout: Any, reading: Any, *, unit_system: Any = None,
                   theme: Any = None) -> Any:
    """Draw `reading`, a `core.resample.ResampleReading`, into `layout`
    and return the plot.

    Densities are drawn in `unit_system`'s units, and the frequency
    axis runs from zero to past the new Nyquist frequency, far enough to
    show every folded frequency.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor

    from ..core.resample import ordinal
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, curve_color, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    alarm = colors['limit_abort']
    scale, _offset = us.transform(reading.density_dim)

    def shown(values):
        # a density is a size, so it converts by the unit's scale alone;
        # a line of nothing has no place on a log axis
        values = np.asarray(values, dtype=float) / scale
        return np.where(values > 0.0, values, np.nan)

    plot = _reading_plot(layout, 0)
    legend = legend_below(layout, plot, 0, colors)
    plot.setLogMode(x=False, y=True)
    reach = max([REACH * reading.nyquist]
                + [1.08 * fold.frequency for fold in reading.folds])

    # ---- the four spectra ---------------------------------------------
    faint = QColor(foreground)
    faint.setAlpha(110)
    styles = (
        ('source', pg.mkPen(faint, width=1),
         f'as recorded, {reading.source_rate:g} Hz'),
        ('filtered', pg.mkPen(curve_color(0, colors), width=2.5),
         (f'filtered at {reading.corner:g} Hz, then decimated to '
          f'{reading.rate:g} Hz')),
        ('bare', pg.mkPen(alarm, width=1.5, style=Qt.PenStyle.DashLine),
         'decimated with no filter'),
        ('after', pg.mkPen(colors['specification_curve'], width=2,
                           style=Qt.PenStyle.DotLine),
         f'decimated, then low-passed at {reading.corner:g} Hz'),
    )
    top = 0.0
    for key, pen, name in styles:
        f, g = reading.spectra[key]
        inside = f <= reach
        values = shown(g[inside])
        if np.isfinite(values).any():
            top = max(top, float(np.nanmax(values)))
        plot.plot(f[inside], values, pen=pen, name=name)

    # ---- the span, the Nyquist frequency and the dropped lines --------
    hatch = QColor(foreground)
    hatch.setAlpha(120)
    dropped = pg.LinearRegionItem(
        values=(reading.span, reading.nyquist), movable=False,
        brush=QBrush(hatch, Qt.BrushStyle.BDiagPattern),
        pen=pg.mkPen(None))
    dropped.setZValue(-15)
    plot.addItem(dropped, ignoreBounds=True)
    legend.addItem(
        pg.PlotDataItem([0.0, 1.0], [1.0, 1.0], fillLevel=0.0,
                        brush=QBrush(hatch, Qt.BrushStyle.BDiagPattern),
                        pen=pg.mkPen(hatch)),
        f'dropped lines, {reading.dropped} of {reading.lines}')
    nyquist_pen = pg.mkPen(foreground, width=1.5, style=Qt.PenStyle.DashLine)
    span_pen = pg.mkPen(foreground, width=1.5)
    for at, pen, text in ((reading.nyquist, nyquist_pen,
                           f'Nyquist {reading.nyquist:g} Hz'),
                          (reading.span, span_pen,
                           f'span {reading.span:g} Hz')):
        plot.addItem(pg.InfiniteLine(
            pos=at, angle=90, pen=pen, movable=False, label=text,
            labelOpts={'position': 0.96, 'color': foreground,
                       'anchors': [(1, 0), (1, 0)]}), ignoreBounds=True)
        legend.addItem(pg.PlotDataItem([], [], pen=pen), text)

    # ---- the folds ------------------------------------------------------
    hollow = pg.mkBrush(0, 0, 0, 0)
    link = pg.mkPen(foreground, width=1, style=Qt.PenStyle.DotLine)
    for k, fold in enumerate(reading.folds):
        start = float(shown(fold.source))
        end = float(shown(fold.bare))
        if np.isfinite(start) and np.isfinite(end):
            plot.addItem(pg.PlotDataItem([fold.frequency, fold.alias],
                                         [start, end], pen=link))
        # the first fold's markers name the two shapes in the legend: a
        # legend entry drawn from an empty item lost its marker's fill
        if np.isfinite(start):
            plot.plot([fold.frequency], [start], pen=None, symbol='t',
                      symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(foreground),
                      symbolBrush=hollow,
                      name=(None if k else
                            'above the new Nyquist frequency, as recorded'))
        if np.isfinite(end):
            plot.plot([fold.alias], [end], pen=None, symbol='t',
                      symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(alarm),
                      symbolBrush=pg.mkBrush(alarm),
                      name=(None if k else
                            'where it lands, decimated with no filter'))
            words = pg.TextItem(fold_label(fold), color=foreground,
                                anchor=(-0.05, 1.0))
            words.setPos(fold.alias, _view(end, True))
            plot.addItem(words, ignoreBounds=True)

    plot.setXRange(0.0, reach, padding=0.0)
    if top > 0.0:
        # headroom above the tallest line for the words beside a fold
        plot.setYRange(np.log10(top) - DECADES, np.log10(top) + 1.5,
                       padding=0.0)
    plot.setLabel('left', axis_label(reading.density_dim, us))
    plot.setLabel('bottom', f'frequency [{us.label_html("frequency")}]')
    title = (f'{reading.label}: {reading.source_rate:g} Hz to '
             f'{reading.rate:g} Hz, every {ordinal(reading.factor)} sample, '
             f'{reading.kept} of {reading.lines} lines kept')
    if reading.folds:
        title += ''.join(f'<br>{fold.describe()}' for fold in reading.folds)
    plot.setTitle(title, color=foreground, size='9pt')
    return plot


def summary(reading: Any) -> str:
    """The reading in one line, for the status bar."""
    return f'{reading.label}: {reading.describe()}'
