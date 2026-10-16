"""A shock response spectrum against its tolerance band, point by point:
the requirement, its bands, each measured polarity, and the points
that fell outside.

The question this view answers is the one a shock test is accepted
on: how many of the spectrum's points lie inside the band, and which
do not. So the band is shaded the way a specification's zones are
shaded everywhere else here, the requirement stands back in gray, and
every point outside is marked on the curve it belongs to. The count
and the share inside are written in the legend beside each curve's
name, with whether it passed, because a reader of a printed figure
should not have to count markers.

The zones follow the random comparison's colors. Between the inner
band's edge and the outer band's is yellow, the ground a share rule
lets a few points stand in; past the outer band is red above and blue
below, where no point may be. With no outer band the yellow runs to
the edge of the plot.

**Each polarity wears a marker as well as a color**, for the reason
the sine tracking view gives: several curves of one quantity on one
channel are compared line for line here, and a reader who cannot tell
two of the palette's hues apart must still be able to tell the
positive spectrum from the negative. A few hollow markers along each
curve name it; a point outside the band is the same marker, filled,
so being out reads as a change of fill on a shape the reader has
already matched to the legend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.compliance import SRS_INNER_DB, SRS_POINTS_PERCENT

#: the marker each polarity wears, in the order the spectra are given
SYMBOLS = ('o', 's', 'd', 't1', 'star')


def _record(data: Any, dof: str, event: str | None) -> int | None:
    """The first record of `data` at `dof`, and at `event` when one is
    named; None when there is none."""
    for i, name in enumerate(data.response_dof):
        if str(name) != dof:
            continue
        block = None if data.block is None else data.block[i]
        if event is None or block == event:
            return i
    return None


def _shown(values: Any, dimension: str, unit_system: Any) -> np.ndarray:
    """Values in display units; a record whose units are undeclared is
    drawn as it is."""
    from ..units import UNKNOWN

    values = np.asarray(np.real(values), dtype=float)
    if dimension == UNKNOWN:
        return values
    return np.asarray(unit_system.from_si(values, dimension), dtype=float)


def band_text(inner: tuple[float, float],
              outer: tuple[float, float] | None) -> str:
    """The band in words, as the title and the legend give it:
    '−1.5/+3 dB', or '−1.5/+3 dB within −3/+6 dB' with an outer band."""
    def pair(band):
        below, above = band
        top = 'up' if np.isinf(above) else f'{above:+g}'
        return f'{below:+g}/{top} dB'.replace('-', '\u2212')

    text = pair(inner)
    return text if outer is None else f'{text} within {pair(outer)}'


def describe(result: dict[str, Any]) -> str:
    """One judged spectrum in a line: its polarity, the count and share
    of its points inside the band, and whether it passed, as in
    'maximax: 85/92 inside (92.4 %), passes'. Short, because it is a
    legend entry beside two or three others."""
    name = result.get('kind') or 'measured'
    points = result['points']
    if result['rule'] == 'above_lower':
        held = points - result['under_count']
        where = f'at or above {result["inner"][0]:+g} dB'.replace(
            '-', '\u2212')
        share = 100.0 * held / points if points else float('nan')
    else:
        held = result['inside_count']
        where = 'inside'
        share = result['inside_percent']
    verdict = {True: 'passes', False: 'fails', None: 'not judged'}[
        result['passed']]
    return f'{name}: {held}/{points} {where} ({share:.1f} %), {verdict}'


def outside_points(result: dict[str, Any]) -> np.ndarray:
    """Which points the plot marks: those the rule holds against the
    spectrum. Under 'share' every point outside the inner band, under
    'above_lower' every point under its lower edge, and under either
    every point outside the outer band."""
    out = result['under'] if result['rule'] == 'above_lower' else (
        result['judged'] & ~result['inside'])
    return out | result['beyond_outer']


def build_srs_tolerance(layout: Any, specification: Any,
                        measured: Any | Sequence[Any], *,
                        channel: str | None = None,
                        event: str | None = None,
                        inner: tuple[float, float] = SRS_INNER_DB,
                        outer: tuple[float, float] | None = None,
                        rule: str = 'share',
                        percent: float = SRS_POINTS_PERCENT,
                        unit_system: Any = None,
                        theme: Any = None) -> list[dict[str, Any]]:
    """Draw one channel's spectra against the requirement and its band,
    and return each spectrum's judgement (`compliance.srs_tolerance`).

    `measured` is one `Srs` or several, one per polarity; each is drawn
    at `channel` (the specification's first DOF by default) and at
    `event` (each spectrum's first record at that channel by default).
    """
    import pyqtgraph as pg

    from ..core.compliance import srs_tolerance
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import (
        STOOD_BACK_WIDTH,
        _Extents,
        _shade_limit_zones,
        axis_label,
        curve_color,
        legend_below,
        zone_colors,
    )
    from .sine_tracking import MARKER_SIZE, _marker_spots

    spectra = [measured] if hasattr(measured, 'ordinate') else list(measured)
    if not spectra:
        raise ValueError('no spectra to draw')
    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    dof = str(specification.response_dof[0] if channel is None else channel)
    j = _record(specification, dof, None)
    if j is None:
        raise ValueError(f'the requirement names no channel {dof}')
    picked = []
    for data in spectra:
        i = _record(data, dof, event)
        if i is None:
            where = dof if event is None else f'{dof}, {event}'
            raise ValueError(f'a spectrum has no record at {where}')
        result = srs_tolerance(specification, data, j, i, inner=inner,
                               outer=outer, rule=rule, percent=percent)
        if 'refused' in result:
            raise ValueError(result['refused'])
        picked.append((data, i, result))
    dimension = specification.ordinate_dim[j]

    plot = layout.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLogMode(x=True, y=True)
    legend = legend_below(layout, plot, 0, colors)
    extents = _Extents()

    x = np.asarray(specification.abscissa, dtype=float)
    required = _shown(specification.ordinate[j], dimension, us)
    low, high = picked[0][2]['inner']
    bands = {'warning_lower': required * 10.0 ** (low / 20.0),
             'warning_upper': required * 10.0 ** (high / 20.0)}
    if rule == 'above_lower':
        # that rule puts no ceiling on the inner band, so nothing above
        # it is shaded as ground to stay out of
        del bands['warning_upper']
    if outer is not None:
        reach = picked[0][2]['outer']
        bands['abort_lower'] = required * 10.0 ** (reach[0] / 20.0)
        bands['abort_upper'] = required * 10.0 ** (reach[1] / 20.0)
    zones = _shade_limit_zones(plot, x, bands, colors)
    for values in bands.values():
        # the bands set how far the view reaches, so an edge above
        # everything measured is not cropped off; an open edge (a band
        # from 0 dB up) has nowhere to reach
        extents.add(x, values, True, True)
    spec = plot.plot(x, required, connect='finite', name='requirement',
                     pen=pg.mkPen(colors['specification_curve'],
                                  width=STOOD_BACK_WIDTH))
    spec.setZValue(-1)
    extents.add(x, required, True, True)

    hollow = pg.mkBrush(0, 0, 0, 0)
    for k, (data, i, result) in enumerate(picked):
        color = curve_color(k, colors)
        symbol = SYMBOLS[k % len(SYMBOLS)]
        fx = np.asarray(data.abscissa, dtype=float)
        fy = _shown(data.ordinate[i], data.ordinate_dim[i], us)
        pen = pg.mkPen(color, width=1.5)
        plot.plot(fx, fy, pen=pen, connect='finite')
        spots = _marker_spots(fx, k, len(picked), True)
        plot.plot(fx[spots], fy[spots], pen=None, symbol=symbol,
                  symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(color),
                  symbolBrush=hollow)
        out = outside_points(result)
        if out.any():
            marks = plot.plot(fx[out], fy[out], pen=None, symbol=symbol,
                              symbolSize=MARKER_SIZE,
                              symbolPen=pg.mkPen(color),
                              symbolBrush=pg.mkBrush(color))
            marks.is_outside_mark = True
        extents.add(fx, fy, True, True)
        legend.addItem(pg.PlotDataItem(
            [], [], pen=pen, symbol=symbol, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(color), symbolBrush=hollow),
            describe(result))

    shades = zone_colors(colors)
    for zone, name in tolerance_key(zones, *_bands(picked), rule):
        swatch = pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0,
                                 pen=None, brush=pg.mkBrush(shades[zone]))
        swatch.is_zone_edge = True
        legend.addItem(swatch, name)
    legend.addItem(pg.PlotDataItem(
        [], [], pen=None, symbol=SYMBOLS[0], symbolSize=MARKER_SIZE,
        symbolPen=pg.mkPen(colors['plot_foreground']),
        symbolBrush=pg.mkBrush(colors['plot_foreground'])),
        'filled: a point the rule counts against')

    rule_text = (f'{picked[0][2]["percent"]:g} % of points inside'
                 if rule == 'share' else 'every point at or above the '
                 'lower edge')
    where = dof if event is None else f'{dof}, {event}'
    plot.setTitle(f'{where} against {band_text(*_bands(picked))}; '
                  f'{rule_text}', color=colors['plot_foreground'],
                  size='9pt')
    plot.setLabel('bottom', 'natural frequency [Hz]')
    plot.setLabel('left', axis_label(dimension, us))
    extents.frame_y(plot)
    extents.lock(plot)
    return [result for _data, _i, result in picked]


def _bands(picked):
    """(inner, outer) as the judgement read them."""
    first = picked[0][2]
    return first['inner'], first['outer']


def tolerance_key(zones: set, inner: tuple[float, float],
                  outer: tuple[float, float] | None,
                  rule: str = 'share') -> list[tuple[str, str]]:
    """The legend entries for the shading, (zone, name), each only when
    it was drawn: `zones` holds any of 'warning', 'above' and 'below',
    as `_shade_limit_zones` returns them. Under 'above_lower' only the
    inner band's lower edge is shaded, and the entry says so."""
    entries = []
    if 'warning' in zones:
        name = (f'below {inner[0]:+g} dB'.replace('-', '\u2212')
                if rule == 'above_lower'
                else f'outside {band_text(inner, None)}')
        entries.append(('warning', name))
    if outer is not None:
        if 'above' in zones:
            entries.append(('above', f'above {outer[1]:+g} dB'))
        if 'below' in zones:
            entries.append(('below', f'below {outer[0]:+g} dB'.replace(
                '-', '\u2212')))
    return entries
