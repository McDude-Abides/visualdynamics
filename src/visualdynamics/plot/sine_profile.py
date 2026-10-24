"""A sine profile drawn three ways: displacement, velocity and
acceleration against frequency, with the crossovers marked on all three.

A profile written in segments looks different in each quantity. A
constant displacement is a flat line on the displacement row and a ramp
of 12 dB per octave on the acceleration row; a constant acceleration is
flat on its own row and falls on the other two. Drawn all three ways on
log axes the segments read as straight lines and the profile's demands
on the machine stand out: the most displacement, the most velocity and
the most acceleration, each named in its row's title with where it
falls, which is what goes against a shaker's stroke, velocity and force
ratings.

Every crossover (`core.sine_profile.Crossing`) is a vertical line across
the three rows, and nothing is said by hue alone:

- a crossover that checks against the breakpoint the document wrote is
  a dashed line with a hollow circle where the segments meet on the
  acceleration row;
- one that does not check is a dashed line with a filled cross, and the
  breakpoint the document wrote is a dotted line beside it, so the gap
  between the two is the error read straight off the plot;
- a step between two segments of one quantity is a dotted line with a
  filled square;

and the legend says each in words with its check. Where the profile was
read back from a tone written as breakpoints (`read_profile`), the
tone's own level, interpolated between breakpoints as the tone runs it
(`SineTone.target`), can be drawn dashed over the profile: where the two
part, the tone's breakpoints are too far apart for the segment they
write.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the rows, top to bottom: the quantity, the dimension its unit is
#: read in, and the convention a displacement is drawn in
ROWS = (('displacement', 'length', 'peak to peak'),
        ('velocity', 'velocity', 'peak'),
        ('acceleration', 'acceleration', 'peak'))

#: how many points draw the profile, evenly in log frequency
POINTS = 400

#: the markers on the acceleration row: a crossover that checks, one that
#: does not, and a step
SYMBOLS = {'checks': 'o', 'off': 'x', 'step': 's'}

#: the markers' size, pixels
MARKER_SIZE = 10

DASHED, DOTTED = 2, 3        # Qt.PenStyle values


def crossing_words(crossing: Any) -> str:
    """One crossing in the legend's words (`Crossing.describe`)."""
    return crossing.describe()


def maximum_words(profile: Any, quantity: str, unit_system: Any = None) -> str:
    """The most the profile asks for in `quantity` and where, in words:
    'most 0.2 in peak to peak at 5 Hz'."""
    from ..core.stroke import _amount
    from ..units import DEFAULT_SYSTEM

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    frequency, value = profile.maxima()[quantity]
    dimension = 'length' if quantity == 'displacement' else quantity
    words = f'most {_amount(value, dimension, us)}'
    if quantity == 'displacement':
        words += ' peak to peak'
    return f'{words} at {frequency:.4g} Hz'


def build_profile(layout: Any, profile: Any = None, *,
                  interpolated: tuple[Any, Any] | None = None,
                  unit_system: Any = None, theme: Any = None,
                  name: str = 'profile') -> list[Any]:
    """Draw a `core.sine_profile.SineProfile` into `layout` as three rows
    sharing their log frequency axis, and return the three plots.

    `interpolated` is (Hz, m/s**2 peak), a tone's own level as it runs
    it (`core.sine_profile.interpolated`), drawn dashed over the profile
    on every row; with no profile (one that could not be read from the
    tone) it is drawn alone. `name` is what the profile is called in the
    legend.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.sine_profile import convert
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import _reading_plot, axis_label, curve_color, legend_below
    from .scalogram import _decade_ticks
    from .shock_check import _level_ticks

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    ink = colors['plot_foreground']
    trace = pg.mkPen(colors['response_curve'], width=2.0)
    tone_pen = pg.mkPen(curve_color(1, colors), width=1.5,
                        style=Qt.PenStyle(DASHED))
    if profile is not None:
        low, high = profile.band
    elif interpolated is not None:
        low, high = (float(np.min(interpolated[0])),
                     float(np.max(interpolated[0])))
    else:
        raise ValueError('nothing to draw: give a profile, or a tone\'s '
                         'interpolated level')
    frequencies = np.geomspace(low, high, POINTS)
    if profile is not None:
        # every boundary drawn exactly, so a corner is a corner
        frequencies = np.unique(np.concatenate([frequencies, profile.bounds]))

    plots = []
    for row, (quantity, dimension, convention) in enumerate(ROWS):
        plot = _reading_plot(layout, row)
        plot.setLogMode(x=True, y=True)
        if plots:
            plot.setXLink(plots[0])
        plots.append(plot)
        if row == len(ROWS) - 1:
            legend_below(layout, plot, len(ROWS), colors)
        drawn = []
        if profile is not None:
            values = us.from_si(profile.level(frequencies, quantity,
                                              convention), dimension)
            plot.plot(frequencies, values, pen=trace,
                      name=name if row == len(ROWS) - 1 else None)
            drawn.append(values)
        if interpolated is not None:
            f, a = (np.asarray(v, dtype=float) for v in interpolated)
            shown = convert(a, f, 'acceleration', quantity)
            if convention == 'peak to peak':
                shown = 2.0 * shown
            shown = us.from_si(shown, dimension)
            plot.plot(f, shown, pen=tone_pen,
                      name=('as the tone interpolates its breakpoints'
                            if row == len(ROWS) - 1 else None))
            drawn.append(shown)
        # round values on both log axes, as the shock and wavelet views
        # label theirs, rather than pyqtgraph's powers of ten
        plot.getAxis('bottom').setTicks([_decade_ticks(np.array([low, high]))])
        reached = np.concatenate([np.ravel(v) for v in drawn])
        reached = reached[np.isfinite(reached) & (reached > 0.0)]
        if reached.size:
            plot.getAxis('left').setTicks(
                [_level_ticks(float(reached.min()), float(reached.max()))])
        label = axis_label(dimension, us)
        if quantity == 'displacement':
            label += ' peak to peak'
        plot.setLabel('left', label)
        words = quantity
        if profile is not None:
            words += ': ' + maximum_words(profile, quantity, us)
        plot.setTitle(words, color=ink, size='9pt')
    plots[-1].setLabel('bottom', 'frequency [Hz]')
    if profile is not None:
        _mark_crossings(plots, profile, us, colors)
    return plots


def _mark_crossings(plots: list[Any], profile: Any, us: Any,
                    colors: Any) -> None:
    """Each crossing as a vertical line on every row, its marker on the
    acceleration row, and its words in the legend."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    ink = colors['plot_foreground']
    abort = colors['limit_abort']
    bottom = plots[-1]
    for crossing in profile.crossings:
        if crossing.solved is None:
            kind, color, style = 'step', ink, DOTTED
        elif crossing.checks:
            kind, color, style = 'checks', ink, DASHED
        else:
            kind, color, style = 'off', abort, DASHED
        pen = pg.mkPen(color, width=1.2, style=Qt.PenStyle(style))
        at = float(np.log10(crossing.used))
        for plot in plots:
            plot.addItem(pg.InfiniteLine(pos=at, angle=90, movable=False,
                                         pen=pen))
        if kind == 'off' and crossing.written is not None:
            written = pg.mkPen(color, width=1.2, style=Qt.PenStyle(DOTTED))
            for plot in plots:
                plot.addItem(pg.InfiniteLine(
                    pos=float(np.log10(crossing.written)), angle=90,
                    movable=False, pen=written))
        level = float(us.from_si(crossing.lower.acceleration(crossing.used),
                                 'acceleration'))
        filled = kind != 'checks'
        bottom.plot([crossing.used], [level], pen=None,
                    symbol=SYMBOLS[kind], symbolSize=MARKER_SIZE,
                    symbolPen=pg.mkPen(color, width=1.5),
                    symbolBrush=pg.mkBrush(color) if filled else None,
                    name=crossing_words(crossing))


__all__ = ['ROWS', 'SYMBOLS', 'build_profile', 'crossing_words',
           'maximum_words']
