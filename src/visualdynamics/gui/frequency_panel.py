"""The frequency readout: whether a sine record's tone was at the
frequency its profile asks for, read in the sine view.

A sine test is held to its frequency as well as its level, and the
levels the extraction reads are placed at the specified frequency
whether the tone was there or not. Once the levels have been extracted,
this readout sits under the extraction's settings, below the distortion
readout, and checks them (`core.sine_frequency.check_frequency`): the
tone's own frequency on the record against the specification's at every
line of the sweep, as an error in percent and hertz, against a band set
as a share of the frequency with a floor in hertz (2 % and no floor by
default; '2 % or 1 Hz, whichever is greater' is 2 % with a 1 Hz floor).
On a channel chosen here (a control channel, since those are the ones
the extraction reads), it gives the band, the worst reading and where,
how many lines are outside, and the verdict in words. The small plot
under the numbers is the error against frequency with the band dashed
either side of zero, the same drawing `plot_sine_frequency` makes
(`plot.sine_frequency.draw_frequency`).

It is a reading, not an act: nothing is made, so there is no button,
and an edit reads again at once. It reads the levels already extracted
and never extracts them itself, so beside a record with no levels it
stays away (principle 3), and beside levels the project marks stale it
says so and checks nothing (`gui.levels_readout`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PySide6.QtWidgets import QWidget

from ..core.sine_frequency import FREQUENCY_TOLERANCE
from .editors import DoubleSpinBox, commit_on_enter
from .levels_readout import CheckPlot, LevelsReadout


class FrequencyPlot(CheckPlot):
    """The frequency error against frequency, small: frequency on the
    log axis every sine level is read on, the error linear about zero,
    since a tone runs off either way."""

    def draw(self, check: Any, dof: str | None) -> list[Any]:
        from ..plot.sine_frequency import draw_frequency

        return draw_frequency(self.plot, check, dof=dof, colors=self.colors,
                              legend=False)

    def span(self, check: Any, dof: str | None) -> tuple[float, float]:
        from ..plot.sine_frequency import error_range

        reach = error_range(check, dof)
        return -reach, reach


class FrequencyPanel(LevelsReadout):
    """The tone, the channel and the band, then what the check found."""

    TITLE = 'Frequency'
    PLOT = FrequencyPlot
    DERIVED = (('band', 'Band'), ('worst', 'Worst'),
               ('over', 'Outside the band'), ('verdict', 'Verdict'))

    def _setting_rows(self) -> list[tuple[str, str, QWidget]]:
        self.tolerance_box: DoubleSpinBox = DoubleSpinBox()
        self.tolerance_box.setRange(0.01, 50.0)
        self.tolerance_box.setDecimals(2)
        self.tolerance_box.setSingleStep(0.25)
        self.tolerance_box.setSuffix(' %')
        self.tolerance_box.setValue(100.0 * FREQUENCY_TOLERANCE)
        self.tolerance_box.setToolTip(
            'How far the tone may stand from the specified frequency, as '
            'a share of it, either side. Test documents often ask for '
            '2 %')
        self.floor_box: DoubleSpinBox = DoubleSpinBox()
        self.floor_box.setRange(0.0, 100.0)
        self.floor_box.setDecimals(2)
        self.floor_box.setSingleStep(0.5)
        self.floor_box.setSuffix(' Hz')
        self.floor_box.setValue(0.0)
        self.floor_box.setToolTip(
            'The least the band may be, in hertz, for a band written as '
            '"2 % or 1 Hz, whichever is greater"; zero for none')
        commit_on_enter(self.tolerance_box, self.floor_box)
        self.tolerance_box.valueChanged.connect(self._edited)
        self.floor_box.valueChanged.connect(self._edited)
        return [('tolerance', 'Band', self.tolerance_box),
                ('floor', 'At least', self.floor_box)]

    def show_frequency(self, history: Any, specification: Any,
                       levels: Any, stale: str | None = None) -> Any:
        """Check the extracted `levels` of `history` against
        `specification` and show it (`LevelsReadout.show_levels`)."""
        return self.show_levels(history, specification, levels, stale)

    def _own_settings(self) -> dict[str, Any]:
        """'tolerance', a fraction, and 'floor', Hz, as
        `plot_sine_frequency` takes them."""
        return {'tolerance': round(self.tolerance_box.value() / 100.0, 8),
                'floor': round(self.floor_box.value(), 6)}

    def _make(self, settings: dict[str, Any]) -> Any:
        from ..core.sine_frequency import check_frequency

        return check_frequency(self._history, self._specification,
                               self._levels, tone=settings['tone'],
                               tolerance=settings['tolerance'],
                               floor=settings['floor'])

    def _rejudge(self, check: Any, settings: dict[str, Any]) -> Any:
        # the frequencies read depend on the record, the levels and the
        # tone only, so a new band is the check already made, judged again
        band = (settings['tolerance'], settings['floor'])
        if (check.tolerance, check.floor) == band:
            return check
        return replace(check, tolerance=band[0], floor=band[1])

    def _words(self, check: Any, dof: str | None) -> dict[str, str]:
        from ..plot.sine_frequency import readout

        return readout(check, dof)


__all__ = ['FrequencyPanel', 'FrequencyPlot']
