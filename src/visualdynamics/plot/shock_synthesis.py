"""A synthesized shock: the waveform and what it asks of the shaker,
then its spectrum against the requirement's band, on one figure.

The two parts are the two questions a synthesis answers, each drawn by
the view that already answers it for a measured record. The top is the
stroke reading (`plot.stroke`): the acceleration, the velocity and the
displacement integrated raw on one clock, each rating on the row it
limits, so the travel to the stops is read as it is for any pulse.
Under it is the SRS tolerance view (`plot.srs_tolerance`): the record's
spectrum against the requirement, its band shaded, the count of points
inside in the legend. Nothing here draws a curve of its own; it stacks
the two, one channel at a time, in one grid rather than in panels
nested side by side, since the window's figure is cleared and redrawn
in place and a nested panel's plots do not survive that cleanly.
"""

from __future__ import annotations

from typing import Any

#: the row the spectrum is drawn in, in `legend_below`'s count (its plot
#: in the layout's row twice this): under the stroke reading's legend
SPECTRUM_ROW = 4


def channel_index(synthesis: Any, channel: str | None) -> int:
    """The index of `channel` among the synthesis's channels, the first
    when None, or a refusal naming the ones it has."""
    dofs = [c.dof for c in synthesis.channels]
    if channel is None:
        return 0
    if str(channel) not in dofs:
        raise ValueError(f'the synthesis has no channel {channel}; it has '
                         f'{", ".join(dofs)}')
    return dofs.index(str(channel))


def build_shock_synthesis(layout: Any, synthesis: Any,
                          channel: str | None = None, *,
                          unit_system: Any = None,
                          theme: Any = None) -> dict[str, Any]:
    """Draw one channel of `synthesis`, a
    `core.shock_synthesis.ShockSynthesis`, into `layout`: its stroke
    reading's three rows, then its spectrum against the requirement
    under them, given the height of two of them. Returns the
    spectrum's judgement (`compliance.srs_tolerance`)."""
    from .srs_tolerance import build_srs_tolerance
    from .stroke import build_stroke

    index = channel_index(synthesis, channel)
    chosen = synthesis.channels[index]
    build_stroke(layout, chosen.reading, unit_system=unit_system, theme=theme)
    # the stroke reading takes rows 0 to 2 and its legend row 7; the
    # spectrum goes in row 8 and its legend in row 9
    results = build_srs_tolerance(layout, synthesis.specification,
                                  synthesis.srs, channel=chosen.dof,
                                  unit_system=unit_system, theme=theme,
                                  row=SPECTRUM_ROW)
    grid = getattr(layout, 'ci', layout).layout
    grid.setRowStretchFactor(2 * SPECTRUM_ROW, 2)
    return results[0]


def summary(synthesis: Any, channel: str | None = None,
            unit_system: Any = None) -> str:
    """One channel's synthesis in a line, for a status bar: its
    wavelets, the band it met, and whether the shaker holds it."""
    chosen = synthesis.channels[channel_index(synthesis, channel)]
    verdict = {True: 'fits the shaker', False: 'does not fit',
               None: 'no ratings given'}[chosen.reading.fits]
    return (f'{chosen.dof}: SRS synthesis, {chosen.points.size} wavelets of '
            f'{chosen.half_cycles} half cycles, within '
            f'{chosen.worst_db:.2f} dB of the requirement; {verdict}')
