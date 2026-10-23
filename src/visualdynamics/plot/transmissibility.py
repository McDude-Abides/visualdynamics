"""The two transmissibilities of one record drawn together, and the band
H1's coherence and averages put about it.

A `TransmissibilityReading` (`core.transmissibility`) holds H1 and the
ASD ratio at every line. This module draws them on one magnitude axis so
the reader sees where they agree and where the ratio lifts away, with
the band H1's random error gives, and tells each apart by line style and
words, never by hue alone:

- H1 solid, named 'H1 = Gxy/Gxx' in words;
- the ASD ratio dashed, named 'ASD ratio √(Gyy/Gxx), magnitude only':
  it has no phase, and it reads over H1 by one over the root of the
  coherence, which says the two disagree and not which is nearer the
  truth (`core.transmissibility`);
- the band dotted at both edges with a faint fill between, named with
  how many errors it reaches and the averages it was read at. Where the
  band reaches zero it has no lower edge, and the lower dotted line
  stops there.

`TransmissibilityOverlay` puts the ratio and the band on a plot that
already draws the FRF (the window's FRF view), with H1 drawn as well
only when the FRF on the plot is another estimator, and a key in the
view's corner that says which line is which. Its items go on the view
box and are kept out of its bounds, as the other readings' overlays
are: they are a reading of the data, so the plot's legend, its copy and
its zoom never take them in. `build_transmissibility` draws a reading
as a figure of its own: the magnitudes above, the coherence below.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: what each line is called, in the key and the legend
H1_NAME = 'H1 = Gxy/Gxx'
RATIO_NAME = 'ASD ratio √(Gyy/Gxx), magnitude only'

#: the markers the cursor puts on the two estimates at its line: a
#: circle on H1 and a square on the ratio, shape rather than color
H1_SYMBOL = 'o'
RATIO_SYMBOL = 's'

#: the markers' size, pixels
MARKER_SIZE = 9

#: how opaque the band's fill is, of 255: enough to see the band, not so
#: much that the curve inside it is covered
FILL_ALPHA = 40


def band_name(reading: Any, spread: float = 1.0) -> str:
    """The band's name in words: how many random errors of H1 it reaches
    either side, and the averages the error was read at."""
    reach = ('1 random error' if spread == 1.0
             else f'{spread:g} random errors')
    return f'H1 ± {reach}, {reading.averages:.4g} averages'


def display_scale(reading: Any, unit_system: Any = None) -> tuple[float, str]:
    """(scale, unit): the factor that converts the reading's SI
    magnitudes into `unit_system`'s units for its dimension, and that
    unit's name; one and an empty name where the units are undefined."""
    from ..units import DEFAULT_SYSTEM, UNKNOWN

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    if reading.dimension == UNKNOWN:
        return 1.0, ''
    return (float(us.from_si(1.0, reading.dimension)),
            us.label_text(reading.dimension))


def _fill_edges(lower: np.ndarray, upper: np.ndarray, x: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, lower, upper) for the band's fill: the lines where its upper
    edge is a number, with the lower edge put up to the upper where it
    has none, so the fill closes to nothing there rather than reaching
    down an axis that has no zero."""
    keep = np.isfinite(upper) & (upper > 0.0)
    low = np.where(np.isfinite(lower), lower, upper)
    return x[keep], low[keep], upper[keep]


class TransmissibilityOverlay:
    """The items that draw a transmissibility reading over an FRF's
    magnitude on a pyqtgraph plot.

    Made once per plot with `plot`, the PlotItem drawing the FRF, and
    `color`, a pyqtgraph color (the plot's foreground by default); then
    `show(reading, ...)` places the ratio, the band and the key, `mark`
    puts the cursor's markers on one line, and `remove` takes it all
    off.
    """

    def __init__(self, plot: Any, color: Any = None, *,
                 z: float = 40.0) -> None:
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        if color is None:
            color = plot.getAxis('left').pen().color()
        self.plot: Any = plot
        self.color: Any = pg.mkColor(color)
        faint = pg.mkColor(color)
        faint.setAlpha(FILL_ALPHA)
        #: H1 itself, drawn only when the FRF on the plot is not H1
        self.h1: Any = pg.PlotDataItem(pen=pg.mkPen(self.color, width=1.5))
        self.ratio: Any = pg.PlotDataItem(
            pen=pg.mkPen(self.color, width=1.5, style=Qt.PenStyle.DashLine))
        self.lower: Any = pg.PlotDataItem(
            pen=pg.mkPen(self.color, width=1, style=Qt.PenStyle.DotLine),
            connect='finite')
        self.upper: Any = pg.PlotDataItem(
            pen=pg.mkPen(self.color, width=1, style=Qt.PenStyle.DotLine),
            connect='finite')
        # the fill's own two edges, never shown: the drawn lower edge
        # stops where the band has none, and a fill bounded by it would
        # close across the gap
        self._fill_low: Any = pg.PlotDataItem(pen=None)
        self._fill_high: Any = pg.PlotDataItem(pen=None)
        self.fill: Any = pg.FillBetweenItem(self._fill_low, self._fill_high,
                                            brush=pg.mkBrush(faint))
        self.h1_mark: Any = pg.PlotDataItem(
            pen=None, symbol=H1_SYMBOL, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(self.color, width=1.5), symbolBrush=None)
        self.ratio_mark: Any = pg.PlotDataItem(
            pen=None, symbol=RATIO_SYMBOL, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(self.color, width=1.5), symbolBrush=None)
        box = plot.getViewBox()
        for k, item in enumerate(self.items()):
            item.setZValue(z + k * 0.1)
            box.addItem(item, ignoreBounds=True)
        # the key: which line is which, in words beside a sample of each
        # line's style, in the view's top right corner
        self.key: Any = pg.LegendItem(offset=(-10, 10),
                                      labelTextColor=self.color)
        self.key.setParentItem(box)
        self.key.setZValue(z + 10)
        self.reading: Any = None
        self.scale: float = 1.0

    def items(self) -> tuple[Any, ...]:
        """Every item drawn on the view box, the key aside."""
        return (self.fill, self._fill_low, self._fill_high, self.lower,
                self.upper, self.h1, self.ratio, self.h1_mark,
                self.ratio_mark)

    def show(self, reading: Any, *, scale: float = 1.0, spread: float = 1.0,
             draw_h1: bool = False) -> None:
        """Draw `reading`, a `TransmissibilityReading`: the ratio dashed,
        the band reaching `spread` errors, and with `draw_h1` H1 solid
        (for a plot whose FRF is another estimator). `scale` converts the
        reading's SI magnitudes into the plot's units."""
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        self.reading, self.scale = reading, float(scale)
        x = np.asarray(reading.frequency, dtype=np.float64)
        log_x, log_y = self.plot.getViewBox().state['logMode']
        for item in self.items():
            if isinstance(item, pg.PlotDataItem):
                item.setLogMode(log_x, log_y)
        lower, upper = reading.band(spread)
        lower, upper = lower * scale, upper * scale
        self.ratio.setData(x, reading.asd_ratio * scale)
        self.lower.setData(x, lower)
        self.upper.setData(x, upper)
        fx, flow, fhigh = _fill_edges(lower, upper, x)
        self._fill_low.setData(fx, flow)
        self._fill_high.setData(fx, fhigh)
        if draw_h1:
            self.h1.setData(x, reading.magnitude * scale)
        else:
            self.h1.setData([], [])
        self.h1.setVisible(draw_h1)
        self.key.clear()
        sample = pg.PlotDataItem(pen=pg.mkPen(self.color, width=1.5))
        dotted = pg.PlotDataItem(pen=pg.mkPen(
            self.color, width=1, style=Qt.PenStyle.DotLine))
        self.key.addItem(sample, H1_NAME)
        self.key.addItem(self.ratio, RATIO_NAME)
        self.key.addItem(dotted, band_name(reading, spread))
        for item in (self.fill, self.lower, self.upper, self.ratio):
            item.setVisible(True)
        self.key.setVisible(True)

    def mark(self, frequency: float | None) -> None:
        """Put the cursor's markers on the line nearest `frequency`: a
        circle on H1, a square on the ratio. None takes them off."""
        if frequency is None or self.reading is None:
            self.h1_mark.setData([], [])
            self.ratio_mark.setData([], [])
            return
        point = self.reading.at(frequency)
        self.h1_mark.setData([point.frequency],
                             [point.magnitude * self.scale])
        self.ratio_mark.setData([point.frequency],
                                [point.asd_ratio * self.scale])

    def remove(self) -> None:
        """Take the items and the key off the plot for good."""
        box = self.plot.getViewBox()
        for item in self.items():
            box.removeItem(item)
        # off the scene, which takes it from its parent too: a legend
        # set to no parent tries to anchor itself to nothing and raises
        scene = self.key.scene()
        if scene is not None:
            scene.removeItem(self.key)


def build_transmissibility(layout: Any, reading: Any, *,
                           spread: float = 1.0,
                           band: tuple[float, float] | None = None,
                           unit_system: Any = None,
                           theme: Any = None) -> tuple[Any, Any]:
    """Draw a transmissibility reading into a graphics layout and return
    its two plots, (magnitude, coherence).

    Above, on log axes, H1 solid, the ASD ratio dashed and the band
    dotted with a faint fill, each named in the legend below the plot.
    Beneath, the coherence the band was read from, on the same
    frequencies, named with the frames. `band` is the (low, high)
    frequencies the view shows, in Hz, the magnitude axis fitted to what
    is drawn inside it; the whole reading when omitted. The notes the
    reading carries go under the title.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import _reading_plot, axis_label, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    ink = colors['plot_foreground']
    scale, _unit = display_scale(reading, us)
    x = np.asarray(reading.frequency, dtype=np.float64)
    shown = np.isfinite(reading.magnitude) & (reading.magnitude > 0.0) & (x > 0.0)

    top = _reading_plot(layout, 0)
    top.setLogMode(x=True, y=True)
    legend_below(layout, top, 0, colors)
    title = reading.describe()
    if reading.notes:
        title += '<br>' + '<br>'.join(reading.notes)
    top.setTitle(title, color=ink, size='9pt')
    lower, upper = reading.band(spread)
    fx, flow, fhigh = _fill_edges(lower * scale, upper * scale, x)
    keep = fx > 0.0
    faint = pg.mkColor(ink)
    faint.setAlpha(FILL_ALPHA)
    low_edge = top.plot(fx[keep], flow[keep], pen=None)
    high_edge = top.plot(fx[keep], fhigh[keep], pen=None)
    top.addItem(pg.FillBetweenItem(low_edge, high_edge,
                                   brush=pg.mkBrush(faint)))
    dotted = pg.mkPen(ink, width=1, style=Qt.PenStyle.DotLine)
    top.plot(x[shown], upper[shown] * scale, pen=dotted, connect='finite',
             name=band_name(reading, spread))
    top.plot(x[shown], lower[shown] * scale, pen=dotted, connect='finite')
    top.plot(x[shown], reading.magnitude[shown] * scale,
             pen=pg.mkPen(ink, width=1.5), name=H1_NAME)
    top.plot(x[shown], reading.asd_ratio[shown] * scale,
             pen=pg.mkPen(ink, width=1.5, style=Qt.PenStyle.DashLine),
             name=RATIO_NAME)
    top.setLabel('left', axis_label(reading.dimension, us))

    bottom = _reading_plot(layout, 2)
    bottom.setLogMode(x=True, y=False)
    bottom.setXLink(top)
    bottom.plot(x[x > 0.0], reading.coherence[x > 0.0],
                pen=pg.mkPen(ink, width=1.5))
    bottom.setYRange(0.0, 1.05, padding=0.0)
    bottom.setLabel('left', f'coherence, {reading.frames} frames')
    bottom.setLabel('bottom', 'frequency [Hz]')
    # the magnitudes are the reading, the coherence its evidence: the
    # first gets the larger share of the height
    grid = getattr(layout, 'ci', layout).layout
    grid.setRowStretchFactor(0, 3)
    grid.setRowStretchFactor(2, 2)
    if band is not None:
        low, high = sorted(float(edge) for edge in band)
        if not low > 0.0:
            raise ValueError('a band on a log frequency axis starts above '
                             'zero')
        inside = (x >= low) & (x <= high)
        values = np.concatenate([reading.magnitude[inside],
                                 reading.asd_ratio[inside],
                                 upper[inside], lower[inside]]) * scale
        values = values[np.isfinite(values) & (values > 0.0)]
        top.setXRange(np.log10(low), np.log10(high), padding=0.0)
        if len(values):
            top.setYRange(np.log10(values.min()) - 0.1,
                          np.log10(values.max()) + 0.1, padding=0.0)
    return top, bottom
