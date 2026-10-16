"""Whether a shock pulse fits on the shaker: three rows on one clock.

The acceleration at the top, the velocity integrated from it raw in the
middle, and the displacement at the bottom, all from one
`core.stroke.StrokeReading`. Each rating is drawn on the row it limits:
the shock velocity either side of zero on the velocity, and the stops
on the displacement, half the stroke either side of the rest position,
where a centered armature has them. With the body counted the travel
relative to the body is drawn beside the table's displacement, since
that is the travel the stops see. Every row is titled with its reading
in words, so the plot says what it shows without the panel beside it.

The curves are told apart by line style as well as color (the table
solid, the travel with the body dashed, the ratings dotted) and named
in a legend, so the plot reads in gray and to a reader who cannot
separate the palette's colors.
"""

from __future__ import annotations

from typing import Any

#: the pen styles, so nothing is carried by hue alone: the table's own
#: motion solid, the travel with the body dashed, the ratings dotted
SOLID, DASHED, DOTTED = 1, 2, 3        # Qt.PenStyle values


def build_stroke(layout: Any, reading: Any, *, unit_system: Any = None,
                 theme: Any = None) -> None:
    """Draw `reading`, a `core.stroke.StrokeReading`, into `layout` as
    three rows sharing their time axis."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.stroke import number
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, curve_color, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    trace = colors['response_curve']
    limit = colors['limit_abort']
    time = reading.time
    shares = reading.shares()

    def amount(value, dimension):
        return (f'{number(float(us.from_si(value, dimension)))} '
                f'{us.label_text(dimension)}')

    def rating_pen():
        return pg.mkPen(limit, width=1.5, style=Qt.PenStyle(DOTTED))

    def rating_lines(plot, level, dimension):
        shown = float(us.from_si(level, dimension))
        for at in (-shown, shown):
            line = pg.InfiniteLine(pos=at, angle=0, movable=False,
                                   pen=rating_pen())
            plot.addItem(line, ignoreBounds=False)

    # the acceleration
    acceleration_plot = _plot(layout, 0)
    acceleration_plot.plot(time, us.from_si(reading.acceleration,
                                            'acceleration'),
                           pen=pg.mkPen(trace, width=1.5))
    acceleration_plot.setLabel('left', axis_label('acceleration', us))
    words = f'{reading.dof}: peak {amount(reading.peak_acceleration, "acceleration")}'
    if reading.force is not None:
        words += f', force {amount(reading.force, "force")}'
        if shares['force'] is not None:
            words += (f' against {amount(reading.force_rating, "force")}'
                      f' ({100.0 * shares["force"]:.0f} %)')
    acceleration_plot.setTitle(words, color=_tone(shares['force'], colors),
                               size='9pt')

    # the velocity, with the shock velocity either side of zero
    velocity_plot = _plot(layout, 1)
    velocity_plot.setXLink(acceleration_plot)
    velocity_plot.plot(time, us.from_si(reading.velocity, 'velocity'),
                       pen=pg.mkPen(trace, width=1.5))
    if reading.velocity_rating is not None:
        rating_lines(velocity_plot, reading.velocity_rating, 'velocity')
    velocity_plot.setLabel('left', axis_label('velocity', us))
    words = f'peak velocity {amount(reading.peak_velocity, "velocity")}'
    if shares['velocity'] is not None:
        words += (f' against {amount(reading.velocity_rating, "velocity")}'
                  f' ({100.0 * shares["velocity"]:.0f} %)')
    velocity_plot.setTitle(words, color=_tone(shares['velocity'], colors),
                           size='9pt')

    # the displacement, with the stops of a centered armature
    displacement_plot = _plot(layout, 2)
    displacement_plot.setXLink(acceleration_plot)
    legend = legend_below(layout, displacement_plot, 3, colors)
    table = pg.mkPen(trace, width=1.5)
    displacement_plot.plot(time, us.from_si(reading.displacement, 'length'),
                           pen=table, name='table')
    if reading.body_factor != 1.0:
        relative = pg.mkPen(curve_color(1, colors), width=1.5,
                            style=Qt.PenStyle(DASHED))
        displacement_plot.plot(
            time, us.from_si(reading.body_factor * reading.displacement,
                             'length'),
            pen=relative, name='relative to the body')
    if reading.stroke is not None:
        rating_lines(displacement_plot, 0.5 * reading.stroke, 'length')
        legend.addItem(pg.PlotDataItem([], [], pen=rating_pen()),
                       'stops, armature centered')
    displacement_plot.setLabel('left', axis_label('length', us))
    displacement_plot.setLabel('bottom',
                               f'time [{us.label_html("time")}]')
    travel = ('travel with the body' if reading.body_factor != 1.0
              else 'peak to peak')
    words = f'{travel} {amount(reading.travel, "length")}'
    if shares['stroke'] is not None:
        words += (f' against a {amount(reading.stroke, "length")} stroke'
                  f' ({100.0 * shares["stroke"]:.0f} %)')
    over = shares['stroke']
    if reading.fits_centered is False:
        words += ', past a stop from a centered armature'
        # within the stroke peak to peak and past a stop from the
        # middle is not a pass: titled in the abort color either way
        over = max(over or 0.0, 1.0 + 1e-9)
    # a sentence to a line: a title wider than the plot is cut off
    words += (f'<br>table {amount(reading.peak_to_peak, "length")} '
              f'integrated, {amount(reading.rule, "length")} by the '
              'half-sine rule')
    displacement_plot.setTitle(words, color=_tone(over, colors), size='9pt')


def _tone(share: float | None, colors: Any) -> str:
    """The title's color: the abort color over a rating, the
    foreground otherwise. The words carry the verdict as well."""
    if share is not None and share > 1.0:
        return colors['limit_abort']
    return colors['plot_foreground']


def _plot(layout: Any, row: int) -> Any:
    """A plot in `row` of the layout, gridded, with no SI scaling."""
    plot = layout.addPlot(row=row, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    # the units are in the labels; an automatic 'x0.001' beside an axis
    # labeled in seconds would say the same thing twice, and turned off
    # after drawing an axis kept its old numbers
    for side in ('left', 'bottom'):
        plot.getAxis(side).enableAutoSIPrefix(False)
    return plot


def summary(reading: Any, unit_system: Any = None) -> str:
    """The reading's first line and its verdict, for a status bar."""
    lines = reading.describe(unit_system)
    verdict = {True: 'fits', False: 'does not fit',
               None: 'no ratings given'}[reading.fits]
    if reading.fits and reading.fits_centered is False:
        verdict = 'fits peak to peak, not from a centered armature'
    return f'{reading.dof}: {lines[0]}; {verdict}'

