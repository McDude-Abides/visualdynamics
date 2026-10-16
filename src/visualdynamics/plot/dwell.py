"""A dwell's level against time, with the shares it is read by marked.

A dwell's level against frequency is one point, so this view puts it
against time: the tracked level of each cycle, at the instant it stands
for with the detector's lag taken out, building from the dwell's start,
holding, and ringing down after the stop. Over it go
the readings `core.dwell.read_dwell` makes, each drawn by a line style
and a marker shape and named in words, never by hue alone:

- the steady level, a wide gray line, as a requirement is drawn behind
  a measurement elsewhere, since it is what the rest is read against;
- the share the build-up is read to (95 % by default), dashed, with a
  filled circle where the level first reaches it;
- its complement (5 %), dotted, with a filled triangle where the
  ring-down first falls to it;
- the dwell's start and stop, thin dashed verticals.

The legend names each with its number, so a figure printed in gray still
reads: '95 % after 28.6 cycles (1.43 s)'. The level axis is linear,
because the shares are shares of a linear level; on a log axis 95 % and
100 % would sit a hair apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the marker on the build-up's crossing, and on the ring-down's
BUILT_SYMBOL = 'o'
RUNG_SYMBOL = 't'

#: the markers' size, pixels
MARKER_SIZE = 10


def readout(reading: Any, unit_system: Any = None) -> dict[str, str]:
    """The reading's numbers as the window's readout words them, in the
    display units: 'frequency', 'steady', 'built' and 'rung' (each in
    cycles, then seconds) and 'settled'. A count the record does not
    give reads as a dash; the notes say why."""
    from ..units import DEFAULT_SYSTEM

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    unit = us.label_text(reading.dimension)
    steady = float(us.from_si(reading.steady, reading.dimension))

    def count(cycles, seconds):
        return ('—' if cycles is None
                else f'{cycles:.1f} cycles, {seconds:.3g} s')

    return {'frequency': f'{reading.frequency:.6g} Hz',
            'steady': f'{steady:.4g} {unit}'.strip(),
            'built': count(reading.built, reading.built_seconds),
            'rung': count(reading.rung, reading.rung_seconds),
            'settled': 'yes' if reading.settled else 'no'}


def draw_dwell(plot: Any, reading: Any, *, unit_system: Any = None,
               colors: Any = None, legend: bool = True) -> list[Any]:
    """Draw a `DwellReading` onto a pyqtgraph plot item and return the
    items drawn.

    The level against time in the first curve color, the steady level
    in the specification's gray, the share dashed and its complement
    dotted, the two crossings marked by a filled circle and a filled
    triangle, and the dwell's start and stop as thin dashed verticals.
    With `legend`, every item is named for the plot's legend with its
    number; the panel's small preview passes False and words them in its
    readout instead.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import STOOD_BACK_WIDTH, curve_color

    colors = resolve_theme(None) if colors is None else colors
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    dim = reading.dimension

    def shown(values):
        return us.from_si(np.asarray(values, dtype=float), dim)

    def named(text):
        return {'name': text} if legend else {}

    ink = colors['plot_foreground']
    gray = colors['specification_curve']
    steady = float(shown(reading.steady))
    top = reading.share * steady
    bottom = (1.0 - reading.share) * steady
    percent = f'{100.0 * reading.share:g} %'
    rest = f'{100.0 * (1.0 - reading.share):g} %'
    t0, t1 = float(reading.start), float(reading.seconds[-1])
    items = []
    # the steady level first, wide and gray behind the reading, as a
    # requirement stands behind its response everywhere else
    items.append(plot.plot([t0, t1], [steady, steady],
                           pen=pg.mkPen(gray, width=STOOD_BACK_WIDTH),
                           **named(f'steady {steady:.4g} '
                                   f'{us.label_text(dim)}'.strip())))
    items.append(plot.plot([t0, t1], [top, top],
                           pen=pg.mkPen(ink, width=1, style=Qt.DashLine),
                           **named(f'{percent} of steady')))
    items.append(plot.plot([t0, t1], [bottom, bottom],
                           pen=pg.mkPen(ink, width=1, style=Qt.DotLine),
                           **named(f'{rest} of steady')))
    # the reading itself, cycle by cycle, each at the instant it stands
    # for (`DwellReading.stands`), the clock the two counts are read on,
    # so the marks sit on the curve; nothing is drawn before the first
    x, y = reading.stands, shown(reading.level)
    items.append(plot.plot(x, y, pen=pg.mkPen(curve_color(0, colors),
                                              width=1.5),
                           **named(f'{reading.dof}, '
                                   f'{reading.setting.describe()}')))
    for when, word in ((reading.start, 'start'), (reading.stop, 'stop')):
        line = pg.InfiniteLine(
            pos=float(when), angle=90,
            pen=pg.mkPen(gray, width=1, style=Qt.DashLine),
            label=word, labelOpts={'position': 0.95, 'color': ink})
        plot.addItem(line)
        items.append(line)
    if reading.built is not None:
        items.append(plot.plot(
            [reading.start + reading.built_seconds], [top], pen=None,
            symbol=BUILT_SYMBOL, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(ink), symbolBrush=pg.mkBrush(ink),
            **named(f'{percent} after {reading.built:.1f} cycles '
                    f'({reading.built_seconds:.3g} s)')))
    if reading.rung is not None:
        items.append(plot.plot(
            [reading.stop + reading.rung_seconds], [bottom], pen=None,
            symbol=RUNG_SYMBOL, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(ink), symbolBrush=pg.mkBrush(ink),
            **named(f'{rest} {reading.rung:.1f} cycles after the stop '
                    f'({reading.rung_seconds:.3g} s)')))
    return items


def build_dwell(layout: Any, reading: Any, *, unit_system: Any = None,
                theme: Any = None) -> Any:
    """Draw a dwell reading into a graphics layout, its legend below it,
    and return the plot item.

    The title names the tone, the frequency and the channel; the notes
    the reading carries, if any, go under the title, since a count that
    rests on an unsettled level should not be read without them.
    """
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    plot = layout.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    legend_below(layout, plot, 0, colors)
    title = (f'{reading.tone}: {reading.frequency:.6g} Hz held, read at '
             f'{reading.dof}')
    if reading.notes:
        title += '<br>' + '<br>'.join(reading.notes)
    plot.setTitle(title, color=colors['plot_foreground'], size='9pt')
    draw_dwell(plot, reading, unit_system=us, colors=colors)
    plot.setLabel('bottom', 'time [s]')
    plot.setLabel('left', axis_label(reading.dimension, us))
    return plot
