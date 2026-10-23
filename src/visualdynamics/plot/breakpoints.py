"""A random specification's breakpoint table read: each segment's slope
and area written on the specification, and under it the rms accumulated
from the low end. Two rows.

The question this view answers is where a specification's rms comes
from. On log-log axes a slope at the top of the band looks like a short
line at the end of the profile, and it is often the largest area of
all, because a hertz at 1,500 Hz counts as much as a hertz at 100 Hz.
So each segment carries its slope in decibels per octave and its area,
written on it, and the second row draws the running rms, the root of
the area from the lowest breakpoint up to each frequency: it climbs
steeply where the area is, and ends at the specification's rms.

**The two rows are told apart by line style and words**, never by hue
alone: the specification is solid, as it is drawn everywhere else
here, and the running rms is dashed, with its title saying what it is
and where it ends. The breakpoints wear hollow circles on both rows, so
a reader matches a bend in the specification to the step in the rms
under it. The rows share their frequency axis, panned and zoomed
together.

The same two pieces are added to the specification sheet's plot in the
app (`add_segment_labels`, `add_running_rms`), where the specification
row is the sheet's own preview with its bands; the standalone plot
(`plot.plot_breakpoints`) draws the specification itself, log-log.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from . import _reading_plot

#: the most segments that are each given a label; a specification on a
#: controller's lines has thousands, and a label on each would bury it.
#: The panel and the title still give every number.
MOST_LABELS = 16

#: points the running rms is drawn at across each segment, spaced in
#: log frequency, so its curvature reads on either axis
RUNNING_POINTS = 48

#: the marker the breakpoints wear on both rows, and its size
BREAKPOINT_SYMBOL = 'o'
BREAKPOINT_SIZE = 7


def _units(dim: str, unit_system: Any) -> tuple[Any, bool]:
    """(unit system, declared): the system the values are shown in, and
    whether the quantity has units to convert at all."""
    from ..units import DEFAULT_SYSTEM, UNKNOWN

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    return us, bool(dim) and dim != UNKNOWN


def _shown(values: Any, dimension: str, us: Any, declared: bool) -> np.ndarray:
    """Values in display units; undeclared values are drawn as they are."""
    values = np.asarray(values, dtype=float)
    if not declared:
        return values
    return np.asarray(us.from_si(values, dimension), dtype=float)


def _unit(dimension: str, us: Any, declared: bool) -> str:
    """The unit's text, or '' for a quantity with none declared."""
    return us.label_text(dimension) if declared else ''


def _minus(text: str) -> str:
    """A typeset minus sign, as every reading here writes one."""
    return text.replace('-', '−')


def slope_text(db_per_octave: float) -> str:
    """A slope in words: 'flat', or '+3.0 dB/oct' with a typeset sign."""
    if abs(db_per_octave) < 0.05:
        return 'flat'
    return _minus(f'{db_per_octave:+.1f} dB/oct')


def running_curve(reading: Any) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, rms): the running rms of a `core.breakpoints.
    BreakpointReading` at `RUNNING_POINTS` across every segment, spaced
    in log frequency, the breakpoints among them, in SI."""
    from ..core.breakpoints import running_rms

    f = reading.frequencies
    fine = np.unique(np.concatenate([
        np.geomspace(a, b, RUNNING_POINTS) for a, b in itertools.pairwise(f)]
        + [f]))
    return fine, running_rms(f, reading.levels, fine)


def readout(reading: Any, dim: str,
            unit_system: Any = None) -> list[dict[str, str]]:
    """The numbers of a reading, short enough for the app's panel, in
    display units: one row per segment with its band ('segment'), its
    slope, its area, its share of the whole area, and the running rms at
    its end, then a total row with the whole area and the rms.

    `dim` is the quantity the density is of ('acceleration'), so areas
    are in its square and rms in it. Each value is text."""
    us, declared = _units(dim, unit_system)
    area_unit = _unit(f'{dim}**2', us, declared)
    rms_unit = _unit(dim, us, declared)
    areas = _shown(reading.areas, f'{dim}**2', us, declared)
    running = _shown(reading.running, dim, us, declared)
    f = reading.frequencies
    rows = []
    for k, share in enumerate(reading.shares):
        rows.append({
            'segment': f'{f[k]:g}–{f[k + 1]:g} Hz',
            'slope': slope_text(float(reading.slopes[k])),
            'area': f'{areas[k]:.4g} {area_unit}'.rstrip(),
            'share': f'{100.0 * share:.0f} %',
            'running': f'{running[k]:.4g} {rms_unit}'.rstrip()})
    rows.append({
        'segment': 'total', 'slope': '',
        'area': f'{float(np.sum(areas)):.4g} {area_unit}'.rstrip(),
        'share': '100 %' if reading.total_area > 0 else '',
        'running': f'{float(_shown(reading.rms, dim, us, declared)):.4g} '
                   f'{rms_unit} rms'.replace('  ', ' ')})
    return rows


def add_segment_labels(plot: Any, reading: Any, dim: str,
                       unit_system: Any = None, theme: Any = None) -> list:
    """Write each segment's slope and area on `plot`, which draws the
    specification, above the middle of the segment in log frequency;
    return the labels. Nothing is written past `MOST_LABELS` segments.

    The labels are placed in the plot's own coordinates, so a plot on
    log axes gets logarithms (pyqtgraph moves its curves into log space
    itself and leaves every other item where it is put)."""
    import pyqtgraph as pg

    from ..theme import theme as resolve_theme

    if len(reading.areas) > MOST_LABELS:
        return []
    colors = resolve_theme(theme)
    us, declared = _units(dim, unit_system)
    area_unit = _unit(f'{dim}**2', us, declared)
    log_x, log_y = plot.getViewBox().state['logMode']
    f, w = reading.frequencies, reading.levels
    areas = _shown(reading.areas, f'{dim}**2', us, declared)
    labels = []
    for k in range(len(areas)):
        middle = float(np.sqrt(f[k] * f[k + 1]))
        n = np.log(w[k + 1] / w[k]) / np.log(f[k + 1] / f[k])
        level = float(w[k] * (middle / f[k]) ** n)
        x = float(_shown(middle, 'frequency', us, True))
        y = float(_shown(level, f'{dim}**2/frequency', us, declared))
        # on the side of a slope away from the line: above and to the
        # left of a rise, above and to the right of a fall, centered
        # over a flat segment
        slope = float(reading.slopes[k])
        anchor = ((1.05, 1.0) if slope >= 0.05 else
                  (-0.05, 1.0) if slope <= -0.05 else (0.5, 1.05))
        label = pg.TextItem(
            html=(f'{slope_text(slope)}<br>{areas[k]:.3g} {area_unit} '
                  f'({100.0 * reading.shares[k]:.0f} %)'),
            color=colors['plot_foreground'], anchor=anchor)
        label.setPos(np.log10(x) if log_x else x,
                     np.log10(y) if log_y else y)
        label.is_segment_label = True
        plot.addItem(label, ignoreBounds=True)
        labels.append(label)
    return labels


def add_running_rms(layout: Any, row: int, reading: Any, dim: str, *,
                    link: Any = None, unit_system: Any = None,
                    theme: Any = None) -> Any:
    """The running rms of `reading` as its own row of `layout` at
    `row`, dashed, the breakpoints circled and its end written in the
    title; return the plot.

    With `link`, the specification's plot, the row shares its frequency
    axis and its log or linear scale. The running rms is accumulated
    from the lowest breakpoint upward, in the quantity's display unit
    ('g' for an acceleration density in g^2/Hz)."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme

    colors = resolve_theme(theme)
    us, declared = _units(dim, unit_system)
    rms_unit = _unit(dim, us, declared)
    plot = _reading_plot(layout, row)
    plot.is_running_rms = True
    log_x = bool(link.getViewBox().state['logMode'][0]) if link is not None \
        else True
    plot.setLogMode(x=log_x, y=False)
    if link is not None:
        plot.setXLink(link)
    fine, rms = running_curve(reading)
    pen = pg.mkPen(colors['specification_curve'], width=2,
                   style=Qt.PenStyle.DashLine)
    curve = plot.plot(_shown(fine, 'frequency', us, True),
                      _shown(rms, dim, us, declared), pen=pen)
    curve.is_running_rms = True
    f = reading.frequencies
    at = _shown(np.concatenate([[0.0], reading.running]), dim, us, declared)
    x = _shown(f, 'frequency', us, True)
    plot.plot(x, at,
              pen=None, symbol=BREAKPOINT_SYMBOL, symbolSize=BREAKPOINT_SIZE,
              symbolPen=pg.mkPen(colors['specification_curve']),
              symbolBrush=pg.mkBrush(colors['plot_background']))
    # the value reached at each breakpoint after the first, written
    # beside its circle, so the steps can be read off as numbers
    if len(reading.areas) <= MOST_LABELS:
        for fx, value in zip(x[1:], at[1:]):
            label = pg.TextItem(f'{value:.3g}', color=colors['plot_foreground'],
                                anchor=(1.1, -0.1))
            label.setPos(np.log10(fx) if log_x else fx, value)
            plot.addItem(label, ignoreBounds=True)
    plot.setLabel('left', f'running rms [{us.label_html(dim)}]' if declared
                  else 'running rms [units undefined]')
    plot.setLabel('bottom', f'frequency [{us.label_html("frequency")}]')
    end = float(_shown(reading.rms, dim, us, declared))
    plot.setTitle(
        f'running rms, accumulated from {f[0]:g} Hz upward (dashed): '
        f'{end:.4g} {rms_unit} at {f[-1]:g} Hz'.replace('  ', ' '),
        color=colors['plot_foreground'], size='9pt')
    return plot


def build_breakpoints(layout: Any, reading: Any, dim: str, *,
                      name: str | None = None, unit_system: Any = None,
                      theme: Any = None) -> tuple[Any, Any]:
    """Draw a `core.breakpoints.BreakpointReading` into `layout`: the
    specification on log-log axes, solid, its breakpoints circled and
    each segment's slope and area written on it, above; the running rms
    beneath, dashed, sharing the frequency axis. `dim` is the quantity
    the density is of; `name` labels the channel in the title. Returns
    the two plots."""
    import pyqtgraph as pg

    from ..theme import theme as resolve_theme
    from . import legend_below

    colors = resolve_theme(theme)
    us, declared = _units(dim, unit_system)
    density = f'{dim}**2/frequency'
    plot = _reading_plot(layout, 0)
    legend = legend_below(layout, plot, 0, colors)
    plot.setLogMode(x=True, y=True)
    f = _shown(reading.frequencies, 'frequency', us, True)
    w = _shown(reading.levels, density, us, declared)
    # straight between breakpoints on log-log axes is the power law, so
    # the breakpoints alone draw the specification exactly
    pen = pg.mkPen(colors['specification_curve'], width=2)
    plot.plot(f, w, pen=pen)
    legend.addItem(pg.PlotDataItem([], [], pen=pen),
                   'specification (solid), a power law between breakpoints')
    marks = {'pen': None, 'symbol': BREAKPOINT_SYMBOL,
             'symbolSize': BREAKPOINT_SIZE,
             'symbolPen': pg.mkPen(colors['specification_curve']),
             'symbolBrush': pg.mkBrush(colors['plot_background'])}
    plot.plot(f, w, **marks)
    legend.addItem(pg.PlotDataItem([], [], **marks), 'breakpoints')
    add_segment_labels(plot, reading, dim, us, theme)
    plot.setLabel('left', f'{us.label_html(density)}' if declared
                  else 'units undefined')
    rms_unit = _unit(dim, us, declared)
    segments = len(reading.areas)
    said = (f'{segments} segment{"s" * (segments != 1)}, each labeled with '
            'its slope and area' if segments <= MOST_LABELS
            else f'{segments} segments, too many to label')
    rms = float(_shown(reading.rms, dim, us, declared))
    plot.setTitle(
        f'{name + ": " if name else ""}{rms:.4g} {rms_unit} rms over '
        f'{reading.frequencies[0]:g}–{reading.frequencies[-1]:g} Hz, '
        f'{said}'.replace('  ', ' '),
        color=colors['plot_foreground'], size='9pt')
    running = add_running_rms(layout, 2, reading, dim, link=plot,
                              unit_system=us, theme=theme)
    return plot, running
