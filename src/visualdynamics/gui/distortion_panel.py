"""The distortion readout: what a sine record carries besides its tone,
read in the sine view.

A sine controller holds the tone it reads through its tracking band and
sees nothing else, so the levels the extraction reads say nothing of
what else was on the channel. Once the levels have been extracted, this
readout sits under the extraction's settings and checks them
(`core.sine_distortion.check_distortion`): the RMS of the record less
the tracked tones against the RMS of the tones at every line of the
sweep, as a distortion in percent and dB, against the limit a tolerance
on the ratio sets (the unfiltered RMS 5 % over the tone by default, a distortion of
32 %). On a channel chosen here (a control channel, since those are the
ones the extraction reads), it gives the limit, the worst reading and
where, how many lines are over, and the verdict in words. The small plot
under the numbers is the distortion against frequency with the limit
dashed across it, the same drawing `plot_sine_distortion` makes
(`plot.sine_distortion.draw_distortion`).

It is a reading, not an act: nothing is made, so there is no button,
and an edit reads again at once. It reads the levels already extracted
and never extracts them itself, since an extraction is the long part,
so beside a record with no levels it stays away (principle 3), and
beside levels the project marks stale it says so and checks nothing
(`gui.levels_readout`, which the frequency readout shares).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtWidgets import QWidget

from ..core.sine_distortion import DISTORTION_EXCESS
from .editors import DoubleSpinBox, commit_on_enter
from .levels_readout import CheckPlot, LevelsReadout


class DistortionPlot(CheckPlot):
    """The distortion against frequency, small: frequency on the log axis
    every sine level is read on, the distortion linear from zero, since
    the limit is read as a height."""

    def draw(self, check: Any, dof: str | None) -> list[Any]:
        from ..plot.sine_distortion import draw_distortion

        return draw_distortion(self.plot, check, dof=dof, colors=self.colors,
                               legend=False)

    def span(self, check: Any, dof: str | None) -> tuple[float, float]:
        row = check.row(dof)
        top = max(1.25 * check.limit,
                  1.1 * float(max((value for value, judged
                                   in zip(check.distortion[row],
                                          check.judged[row]) if judged),
                                  default=0.0)))
        return 0.0, 100.0 * top


class DistortionPanel(LevelsReadout):
    """The tone, the channel and the limit, then what the check found."""

    TITLE = 'Distortion'
    PLOT = DistortionPlot
    DERIVED = (('limit', 'Distortion limit'), ('worst', 'Worst'),
               ('over', 'Over the limit'), ('verdict', 'Verdict'))

    def _setting_rows(self) -> list[tuple[str, str, QWidget]]:
        self.excess_box: DoubleSpinBox = DoubleSpinBox()
        self.excess_box.setRange(0.1, 50.0)
        self.excess_box.setDecimals(1)
        self.excess_box.setSingleStep(1.0)
        self.excess_box.setSuffix(' %')
        self.excess_box.setValue(100.0 * DISTORTION_EXCESS)
        self.excess_box.setToolTip(
            "How far the unfiltered RMS may stand above the tone's. Test "
            'documents often ask for 5 %, a distortion of 32 %')
        commit_on_enter(self.excess_box)
        self.excess_box.valueChanged.connect(self._edited)
        return [('excess', 'Unfiltered over tone', self.excess_box)]

    def show_distortion(self, history: Any, specification: Any,
                        levels: Any, stale: str | None = None) -> Any:
        """Check the extracted `levels` of `history` against
        `specification` and show it (`LevelsReadout.show_levels`)."""
        return self.show_levels(history, specification, levels, stale)

    def _own_settings(self) -> dict[str, Any]:
        """'excess', a fraction, as `plot_sine_distortion` takes it."""
        return {'excess': round(self.excess_box.value() / 100.0, 6)}

    def _make(self, settings: dict[str, Any]) -> Any:
        from ..core.sine_distortion import check_distortion

        return check_distortion(self._history, self._specification,
                                self._levels, tone=settings['tone'],
                                excess=settings['excess'])

    def _rejudge(self, check: Any, settings: dict[str, Any]) -> Any:
        # the RMS readings depend on the record, the levels and the tone
        # only, so a new limit is the check already made, judged again
        if check.excess == settings['excess']:
            return check
        return replace(check, excess=settings['excess'])

    def _words(self, check: Any, dof: str | None) -> dict[str, str]:
        from ..plot.sine_distortion import readout

        return readout(check, dof)


__all__ = ['DistortionPanel', 'DistortionPlot']
