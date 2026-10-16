"""Check a shock record before its spectrum is computed from it.

A shock on a shaker starts and ends with the table at rest, so the
acceleration it records, integrated once, is a velocity that leaves
zero, swings, and comes back to zero. A record whose velocity does not
come back carries something the table did not do. The usual causes
are a zero shift, the offset a piezoelectric accelerometer can be left
with after a severe shock, and a DC offset anywhere in the signal
chain. Both are small beside the peak of the acceleration and easy to
miss on its plot, and both integrate to a ramp. An SRS computed from
such a record is wrong at its low end, because an oscillator far below
the shock's content responds to the step and to nothing else, and
draws a shelf the shock never had.

So the record is integrated raw here, by the trapezoid rule with
nothing removed: no mean, no drift high-pass. That is the opposite of
what `filters.integrate` does by default, and on purpose. Its 2 Hz
drift high-pass is the right default for turning a good record into a
displacement estimate, and the wrong one for this check, because the
drift it removes is the fault the check looks for: a record with a
zero shift, integrated with the default, ends at zero velocity as if
nothing were wrong. The integration is still `filters.integrate`,
called with ``drift_corner=None``, so there is one trapezoid rule in
the package and this is a second, explicit path through it rather
than a second rule.

Each window is read from its own start, so every event begins from
rest whatever the record did before it. The whole record is integrated
once, as `core.filters` integrates it, and each window's velocity is
the record's minus its value where the window opens (and its
displacement likewise, less the velocity it opened with times the time
since). The trapezoid rule is additive, so that is exactly the window
integrated on its own.

The reading, per channel and per window, is the velocity change (the
span from the lowest velocity to the highest, which for a one-signed
pulse with compensation on either side is the main pulse's change),
and where the velocity and displacement end as a share of their
largest excursion. A record whose velocity ends further from rest than
the `tolerance` share is flagged, and the words say which of two
things it did: held at a net velocity change, as a drop machine's
pulse does, or still changing when the record stops, as a zero shift
or an offset does.

References
----------
The check is a standard one; these set it out and say why a record
that fails it is not reported.

1. MIL-STD-810H (2019), *Environmental Engineering Considerations and
   Laboratory Tests*, Method 516.8, Shock. Every shock record is
   integrated to velocity before its peak or spectrum is read, and a
   record that shows a zero shift is discarded.
2. ISO 18431-4:2007, *Mechanical vibration and shock - Signal
   processing - Part 4: Shock-response spectrum analysis*. The
   spectrum the check is made before.
3. Smallwood, D. O. (1981). "An Improved Recursive Formula for
   Calculating Shock Response Spectra." *Shock and Vibration
   Bulletin*. The filter `core.srs` computes the spectrum with.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# The view was asked for on 2026-10-16, from a worked example in which
# the default integration ended a 4 g zero shift's velocity at zero
# where raw it ends at 9.05 m/s, and took the velocity change out of
# every classical pulse with the mean.

#: how far from rest a velocity may end, as a share of its largest
#: excursion, before the record is flagged. A tenth: a good record's
#: velocity does not end at exactly zero, because the sensor's noise is
#: integrated too and walks over a long record, and a zero shift worth
#: catching ramps the velocity well past its pulse. A procedure that
#: names its own figure passes it as `tolerance`.
RETURN_SHARE = 0.1
# Measured on the guide's example (2026-10-16): the ringing shock with
# a noise floor ends at 6.1 % of its peak velocity, so a twentieth
# would have flagged a good record; with its 4 g zero shift it ends at
# 100 %, still climbing.

#: the stretch at the end of a window, as a share of it, read to tell a
#: velocity that holds from one that is still changing
TAIL = 0.1


@dataclass(frozen=True, eq=False)
class ShockCheck:
    """One channel's velocity check over one window.

    The arrays are the raw velocity and displacement, in SI (m/s and
    m), sampled with the window and starting from rest at its first
    sample; `start` is that sample's time on the record's clock, in
    seconds. The numbers are read from them. `flagged` says the
    velocity did not come back within `tolerance` of rest, and
    `drifting` says it was still changing when the window closed.
    """

    #: the channel's DOF, as the record names it
    dof: str
    #: which window, counting from 0 in `TimeHistory.srs_windows` order
    window: int
    #: the window's first instant on the record's clock, in seconds
    start: float
    #: the samples per second the arrays are at
    sample_rate: float
    #: velocity through the window, m/s, from rest at its first sample
    velocity: np.ndarray
    #: displacement through the window, m, from rest at its first sample
    displacement: np.ndarray
    #: the share of the largest excursion an end may sit from rest
    tolerance: float = RETURN_SHARE

    @property
    def time(self) -> np.ndarray:
        """Each sample's time on the record's clock, in seconds."""
        return self.start + np.arange(self.velocity.size) / self.sample_rate

    @property
    def velocity_change(self) -> float:
        """The span from the lowest velocity to the highest, in m/s.

        For one pulse with nothing either side it is the area under the
        pulse, 2AT/π for a half-sine of peak A and duration T. For a
        one-signed pulse with compensation pulses before and after it,
        the velocity runs down through the first, up through the pulse
        and back through the second, and the span is still the main
        pulse's change, where the record's net change is zero by
        design.
        """
        return float(self.velocity.max() - self.velocity.min())

    @property
    def peak_velocity(self) -> float:
        """The largest velocity either way, in m/s."""
        return float(np.abs(self.velocity).max())

    @property
    def end_velocity(self) -> float:
        """The velocity at the window's last sample, in m/s, signed."""
        return float(self.velocity[-1])

    @property
    def peak_displacement(self) -> float:
        """The largest displacement either way, in m."""
        return float(np.abs(self.displacement).max())

    @property
    def end_displacement(self) -> float:
        """The displacement at the window's last sample, in m, signed."""
        return float(self.displacement[-1])

    @property
    def end_velocity_share(self) -> float:
        """How far from rest the velocity ends, as a share of its peak.

        Zero for a record that ends at rest, one for a pulse that ends
        at its full velocity change, more than one for a velocity that
        turned and kept going. Zero when the window never moved.
        """
        return _share(self.end_velocity, self.peak_velocity)

    @property
    def end_displacement_share(self) -> float:
        """How far from where it started the displacement ends, as a
        share of its peak; zero when the window never moved."""
        return _share(self.end_displacement, self.peak_displacement)

    @property
    def flagged(self) -> bool:
        """Whether the velocity ends further from rest than `tolerance`
        of its peak."""
        return bool(self.end_velocity_share > self.tolerance)

    @property
    def drifting(self) -> bool:
        """Whether the velocity is still changing when the window closes.

        The change across the window's last `TAIL` is carried across the
        whole window, which is the velocity the acceleration the record
        ends on would build on its own, and compared with the peak by
        the same `tolerance`. A zero shift or an offset leaves the
        record ending on an acceleration that is not zero, and this is
        what says so; a pulse that ends at a held velocity change ends
        on none.
        """
        samples = self.velocity.size
        tail = max(round(samples * TAIL), 1)
        if samples < 2 or self.peak_velocity == 0.0:
            return False
        change = self.velocity[-1] - self.velocity[-1 - tail]
        return bool(abs(change) * (samples - 1) / tail
                    > self.tolerance * self.peak_velocity)

    def describe(self) -> str:
        """The verdict in words, as the view and the shock panel state
        it."""
        ends = _percent(self.end_velocity_share)
        if not self.flagged:
            return f'Velocity returns to rest: it ends at {ends} of its peak.'
        if self.drifting:
            return (f'Velocity does not return to rest: it ends at {ends} '
                    'of its peak and is still changing when the record '
                    'stops. A zero shift or an offset does this, and the '
                    'spectrum is false at its low end.')
        return (f'Velocity does not return to rest: it ends at {ends} of '
                'its peak and holds there, a net velocity change. A '
                'shaker shock ends at rest, and a drop machine’s pulse '
                'does not.')


def _share(value: float, peak: float) -> float:
    return abs(value) / peak if peak > 0.0 else 0.0


def _percent(share: float) -> str:
    """A share as a percentage, to a decimal place below ten."""
    value = 100.0 * share
    return f'{value:.1f} %' if value < 9.95 else f'{value:.0f} %'


def check_shock(history: TimeHistory, *,
                shocks: Sequence[Any] | None = None,
                tolerance: float = RETURN_SHARE) -> list[ShockCheck]:
    """The velocity check of every acceleration channel over every
    window.

    The windows are the ones the SRS would be computed over
    (`TimeHistory.srs_windows`): the record's shocks, or its averaging
    frames, or the whole record as one window. Pass `shocks` to check
    other windows. Each acceleration channel is integrated raw, once to
    velocity and once more to displacement, and read from each
    window's start (the module docstring says why raw, and why from the
    window's start). Channels of any other quantity are left out.

    Parameters
    ----------
    history : TimeHistory
        The shock record.
    shocks : sequence of Shock, optional
        The windows to check in place of the record's own.
    tolerance : float, default `RETURN_SHARE`
        How far from rest the velocity may end, as a share of its peak,
        before the check is flagged.

    Returns
    -------
    list of ShockCheck
        One per acceleration channel per window, channel by channel in
        the record's order and window by window within each.

    Raises
    ------
    ValueError
        If the record holds no acceleration channel, or `tolerance` is
        not a positive share.
    """
    from .filters import integrate

    tolerance = float(tolerance)
    if not tolerance > 0.0:
        raise ValueError(f'the tolerance ({tolerance:g}) is a share of the '
                         'peak and has to be above zero')
    if not any(history.known_dim(i) == 'acceleration'
               for i in range(history.num_records)):
        raise ValueError('a shock record is checked by integrating its '
                         'acceleration, and this one holds none')
    rate = history.sample_rate
    # velocity channels integrate to displacement and are dropped here:
    # a velocity read as though it were one is the mistake this check
    # exists to catch, not to make
    moving = integrate(history, None)
    velocity = [k for k in range(moving.num_records)
                if moving.known_dim(k) == 'velocity']
    travel = integrate(moving, None)
    v_rows = np.real(np.asarray(moving.ordinate))[velocity]
    d_rows = np.real(np.asarray(travel.ordinate))
    origin = float(history.abscissa[0])
    dt = 1.0 / rate

    checks = []
    for row, k in enumerate(velocity):
        for index, window in enumerate(history.srs_windows(shocks)):
            first, last = window.bounds(rate)
            opened = v_rows[row, first]
            v = v_rows[row, first:last] - opened
            # the displacement less where it was and how fast it was
            # going when the window opened: the trapezoid rule is
            # additive, so this is the window integrated on its own
            d = (d_rows[row, first:last] - d_rows[row, first]
                 - opened * dt * np.arange(last - first))
            checks.append(ShockCheck(
                dof=moving.response_dof[k], window=index,
                start=origin + first * dt, sample_rate=rate,
                velocity=v, displacement=d, tolerance=tolerance))
    return checks


def summarize(checks: Sequence[ShockCheck], most: int = 3) -> str:
    """The checks of a record in a few lines, worst first.

    One line when every check returns to rest. Otherwise how many did
    not, and the verdict of up to `most` of them, the furthest from
    rest first, each named by its channel and, when the record has
    more than one window, its shock.
    """
    if not checks:
        return ''
    windows = len({check.window for check in checks}) > 1
    failed = sorted((check for check in checks if check.flagged),
                    key=lambda check: -check.end_velocity_share)
    if not failed:
        return ('Velocity returns to rest on every channel'
                + (' in every shock.' if windows else '.'))
    lines = [f'{len(failed)} of {len(checks)} do not return to rest.']
    for check in failed[:most]:
        where = (f'{check.dof}, shock {check.window + 1}' if windows
                 else check.dof)
        lines.append(f'{where}: {check.describe()}')
    if len(failed) > most:
        lines.append(f'and {len(failed) - most} more.')
    return '\n'.join(lines)
