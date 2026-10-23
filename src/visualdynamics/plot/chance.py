"""One channel of a comparison read as its miss from the target, line by
line in decibels, beside what chance alone would put outside its bands.

The question this view answers is whether the lines outside a band are
the test or the estimate: an averaged spectrum scatters about its true
level by an amount its degrees of freedom set (`core.chance`), so a few
lines outside a narrow band are expected on every average. The plot
draws each judged line's (or group's) level over the target in dB, the
target at zero, and the warning and abort edges about it. The lines
outside are marked, and the legend and the title give each count beside
the number chance alone puts there at the comparison's DOF, so a reader
of a printed figure can tell the two apart without counting markers.

With lines combined, the groups are drawn over the single lines, which
stand back in gray: the chance scatter shrinks in the groups, and an
excursion that holds across neighbors survives.

**Nothing here is carried by hue alone.** The warning edges are dashed
and the abort edges solid, each named in the legend. A line outside is
marked with a triangle pointing the way it is out, down for under the
band and up for over it: hollow outside the warning band, filled
outside the abort band.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.chance import PAIRS, expected_text

#: the size of the triangles marking the lines outside
MARK_SIZE = 11


def _record_pair(specification: Any, measured: Any,
                 channel: str | None) -> tuple[str, int, int]:
    """(label, spec record, measured record) of the channel named, or of
    the first channel the two have in common."""
    from ..core.compliance import matched_records

    rows = matched_records(specification, measured)
    if not rows:
        raise ValueError('the specification and the measurement have no '
                         'channel in common')
    if channel is None:
        return rows[0]
    for row in rows:
        if row[0] == str(channel):
            return row
    raise ValueError(f'no channel {channel} in common to compare')


def _steps(low: np.ndarray, high: np.ndarray,
           values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A step curve across each group's reach: flat over each, joined
    to the next where they touch and broken (NaN) across a gap."""
    xs: list[float] = []
    ys: list[float] = []
    for k in range(len(values)):
        if k and low[k] > high[k - 1] * (1.0 + 1e-9):
            xs.append(np.nan)
            ys.append(np.nan)
        xs.extend((float(low[k]), float(high[k])))
        ys.extend((float(values[k]), float(values[k])))
    return np.asarray(xs), np.asarray(ys)


def title_text(result: dict[str, Any], channel: str) -> str:
    """The plot's title: the channel, the DOF it was judged at, and each
    pair's count beside its expectation, as in '1Z+ at 120 DOF: 5 of
    389 lines outside warning, 3.3 by chance; 0 outside abort, 0.00039
    by chance'."""
    combine = result['combine']
    if combine == 1:
        head = f'{channel} at {result["dof"]:.4g} DOF'
        unit = 'lines'
    else:
        head = (f'{channel}, {combine} lines combined at '
                f'{result["judged_dof"]:.4g} DOF')
        unit = 'groups'
    parts = []
    for pair in PAIRS:
        if f'{pair}_lines' not in result:
            continue
        count = result[f'{pair}_lines']
        lead = (f'{count} of {result["lines"]} {unit} outside {pair}'
                if not parts else f'{count} outside {pair}')
        parts.append(f'{lead}, '
                     f'{expected_text(result[f"{pair}_expected"])} by chance')
    return f'{head}: ' + ('; '.join(parts) if parts else 'no bands to judge')


def build_chance(layout: Any, specification: Any, measured: Any, *,
                 channel: str | None = None, combine: int = 1,
                 dof: float | None = None, scale_db: float | None = None,
                 theme: Any = None) -> dict[str, Any]:
    """Draw one channel's miss from its target against the bands, with the
    lines outside marked, and return the reading
    (`core.chance.compare_chance`).

    `channel` is the DOF (or 'response/reference') to draw, the first the
    two have in common by default. `combine`, `dof` and `scale_db` are
    `compare_chance`'s; the scale is resolved the way every comparison
    resolves it when omitted. A pair that cannot be read is refused with
    the reason as a ValueError, so nothing is drawn that would read as a
    count.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.chance import compare_chance
    from ..theme import theme as resolve_theme
    from . import STOOD_BACK_WIDTH, _Extents, curve_color, legend_below

    colors = resolve_theme(theme)
    label, spec_index, measured_index = _record_pair(
        specification, measured, channel)
    result = compare_chance(specification, measured, spec_index,
                            measured_index, dof=dof, combine=combine,
                            scale_db=scale_db)
    if 'refused' in result:
        raise ValueError(result['refused'])
    if not result['lines']:
        raise ValueError(f'nothing of {label} to compare')
    groups = result['groups']
    low, high = groups['low'], groups['high']

    plot = layout.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLogMode(x=True, y=False)
    legend = legend_below(layout, plot, 0, colors)
    extents = _Extents()

    target = plot.plot([float(low[0]), float(high[-1])], [0.0, 0.0],
                       name='target',
                       pen=pg.mkPen(colors['specification_curve'],
                                    width=STOOD_BACK_WIDTH))
    target.setZValue(-2)

    # the edges, the warning dashed in the warning band's yellow and the
    # abort solid in the foreground: the style says which band, so the
    # colors need not be told apart (the zone's red and blue were tried
    # for abort, and the blue is the measured curve's own)
    styles = {'warning': (Qt.PenStyle.DashLine, 'warning band',
                          colors['limit_warning']),
              'abort': (Qt.PenStyle.SolidLine, 'abort band',
                        colors['plot_foreground'])}
    for pair in PAIRS:
        if f'{pair}_lines' not in result:
            continue
        style, name, hue = styles[pair]
        named = False
        for side in ('upper', 'lower'):
            edge = groups[f'{pair}_{side}_db']
            if not np.isfinite(edge).any():
                continue
            x, y = _steps(low, high, edge)
            plot.plot(x, y, connect='finite',
                      name=None if named else name,
                      pen=pg.mkPen(hue, width=1.5, style=style))
            named = True
            extents.add(x, y, False, True)

    if result['combine'] > 1:
        # the single lines behind the groups, stood back, so the scatter
        # the combination takes out can be seen beside what it leaves
        single = compare_chance(specification, measured, spec_index,
                                measured_index, dof=result['dof'],
                                combine=1, scale_db=result['scale_db'])
        lines = single['groups']
        x, y = _steps(lines['low'], lines['high'], lines['miss_db'])
        behind = plot.plot(x, y, connect='finite', name='each line',
                           pen=pg.mkPen(colors['specification_curve'],
                                        width=1))
        behind.setZValue(-1)
        extents.add(x, y, False, True)

    color = curve_color(0, colors)
    x, y = _steps(low, high, groups['miss_db'])
    name = ('each line' if result['combine'] == 1
            else f'{result["combine"]} lines combined')
    plot.plot(x, y, connect='finite', name=name,
              pen=pg.mkPen(color, width=2))
    extents.add(x, y, False, True)

    centers = np.sqrt(low * high)
    marks = ((('warning', 'below'), 't', False),
             (('warning', 'above'), 't1', False),
             (('abort', 'below'), 't', True),
             (('abort', 'above'), 't1', True))
    hollow = pg.mkBrush(0, 0, 0, 0)
    for (pair, side), symbol, filled in marks:
        key = f'{pair}_{side}'
        if key not in groups:
            continue
        out = groups[key]
        if pair == 'warning' and 'abort_below' in groups:
            # a line past abort is marked once, filled, not twice
            out = out & ~(groups['abort_below'] | groups['abort_above'])
        brush = pg.mkBrush(color) if filled else hollow
        if out.any():
            item = plot.plot(centers[out], groups['miss_db'][out], pen=None,
                             symbol=symbol, symbolSize=MARK_SIZE,
                             symbolPen=pg.mkPen(color, width=1.5),
                             symbolBrush=brush)
            item.is_outside_mark = True
        unit = 'line' if result['combine'] == 1 else 'group'
        legend.addItem(pg.PlotDataItem(
            [], [], pen=None, symbol=symbol, symbolSize=MARK_SIZE,
            symbolPen=pg.mkPen(color, width=1.5), symbolBrush=brush),
            f'{side} the {pair} band: {result[key]} '
            f'{unit}{"s" * (result[key] != 1)}, '
            f'{expected_text(result[f"{pair}_expected_{side}"])} by chance')

    plot.setTitle(title_text(result, label),
                  color=colors['plot_foreground'], size='9pt')
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', 'measured over target [dB]')
    extents.frame_y(plot)
    extents.lock(plot)
    return result
