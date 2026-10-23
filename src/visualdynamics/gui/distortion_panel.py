"""The distortion readout: what a sine record carries besides its tone,
read in the sine view.

A sine controller holds the tone it reads through its tracking band and
sees nothing else, so the levels the extraction reads say nothing of
what else was on the channel. Once the levels have been extracted, this
readout sits under the extraction's settings and checks them
(`core.sine_distortion.check_distortion`): the channel's unfiltered RMS
against the RMS of the tracked tone at every line of the sweep, as a
distortion in percent and dB, against the limit a tolerance on the ratio
sets (the unfiltered RMS 5 % over the tone by default, a distortion of
32 %). On a channel chosen here (a control channel, since those are the
ones the extraction reads), it gives the limit, the worst reading and
where, how many lines are over, and the verdict in words. The small plot
under the numbers is the distortion against frequency with the limit
dashed across it, the same drawing `plot_sine_distortion` makes
(`plot.sine_distortion.draw_distortion`).

It is a reading, not an act: nothing is made, so there is no button,
and an edit reads again at once. It reads the levels already extracted
and never extracts them itself, since an extraction is the long part,
so beside a record with no levels it stays away (principle 3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QWidget,
)

from ..core.sine_distortion import DISTORTION_EXCESS
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import PreviewPlot, add_derived, add_keyed_settings


class DistortionPlot(PreviewPlot):
    """The distortion against frequency, small: frequency on the log axis
    every sine level is read on, the distortion linear from zero, since
    the limit is read as a height."""

    HEIGHT = 160
    LOG_Y = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items: list[Any] = []

    def show_check(self, check: Any, dof: str | None) -> None:
        """Draw channel `dof` of `check`, or clear the plot when the
        check is None."""
        from ..plot.sine_distortion import draw_distortion

        for item in self.items:
            self.plot.removeItem(item)
        self.items = []
        if check is None:
            return
        self.items = draw_distortion(self.plot, check, dof=dof,
                                     colors=self.colors, legend=False)
        row = check.row(dof)
        top = max(1.25 * check.limit,
                  1.1 * float(max((value for value, judged
                                   in zip(check.distortion[row],
                                          check.judged[row]) if judged),
                                  default=0.0)))
        self.plot.setYRange(0.0, 100.0 * top, padding=0.0)


class DistortionPanel(QWidget):
    """The tone, the channel and the limit, then what the check found."""

    #: the check was read: (history, specification, levels, check, dof),
    #: so the window can journal the call that draws it
    read = Signal(object, object, object, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(rule, 0, 0, 1, 2)
        self.title: QLabel = QLabel('Distortion')
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        grid.addWidget(self.title, 1, 0, 1, 2)
        #: what is being read, and the check last made of it
        self._history: Any = None
        self._specification: Any = None
        self._levels: Any = None
        self._key: tuple | None = None
        self.check: Any = None
        self._loading = False

        self.tone_box: QComboBox = QComboBox()
        self.tone_box.setToolTip('The tone checked, of those extracted')
        self.channel_box: QComboBox = QComboBox()
        self.channel_box.setToolTip(
            'The control channel checked: the extraction reads the '
            'channels the specification controls')
        self.excess_box: DoubleSpinBox = DoubleSpinBox()
        self.excess_box.setRange(0.1, 50.0)
        self.excess_box.setDecimals(1)
        self.excess_box.setSingleStep(1.0)
        self.excess_box.setSuffix(' %')
        self.excess_box.setValue(100.0 * DISTORTION_EXCESS)
        self.excess_box.setToolTip(
            'How far the unfiltered RMS may stand above the tone\'s. Test '
            'documents often ask for 5 %, a distortion of 32 %')
        self.names, self.editors = add_keyed_settings(grid, [
            ('tone', 'Tone', self.tone_box),
            ('channel', 'Channel', self.channel_box),
            ('excess', 'Unfiltered over tone', self.excess_box)],
            first_row=2)
        made = add_derived(grid, (('limit', 'Distortion limit'),
                                  ('worst', 'Worst'),
                                  ('over', 'Over the limit'),
                                  ('verdict', 'Verdict')), 5)
        self.derived: dict[str, QLabel] = {key: value for key, (_name, value)
                                           in made.items()}
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self.notes, 9, 0, 1, 2)
        self.preview: DistortionPlot = DistortionPlot()
        grid.addWidget(self.preview, 10, 0, 1, 2)
        commit_on_enter(self.excess_box)
        self.tone_box.currentIndexChanged.connect(self._edited)
        self.channel_box.currentIndexChanged.connect(self._edited)
        self.excess_box.valueChanged.connect(self._edited)

    # ---- what is being read -------------------------------------------------

    def show_distortion(self, history: Any, specification: Any,
                        levels: Any) -> Any:
        """Check the extracted `levels` of `history` against
        `specification` and show it; hide, and return None, when there are
        no levels to check (principle 3). The tone and channel boxes list
        what the levels hold and keep the ones chosen while the levels
        keep them; the tone row shows only when there is more than one.
        Returns the check, or None when there is none or it was refused
        (the refusal is shown)."""
        if levels is None or not len(levels):
            self.hide()
            self.check = None
            return None
        self._loading = True
        try:
            self._fill(self.tone_box, list(levels.tone_names))
            self._fill(self.channel_box,
                       [str(dof) for dof in levels.response_dof])
        finally:
            self._loading = False
        several = self.tone_box.count() > 1
        self.names['tone'].setVisible(several)
        self.tone_box.setVisible(several)
        self._history, self._specification = history, specification
        self._levels = levels
        self.show()
        self._read()
        return self.check

    @staticmethod
    def _fill(box: QComboBox, items: list[str]) -> None:
        """List `items` in `box`, keeping the choice while it is listed."""
        held = box.currentText()
        if [box.itemText(i) for i in range(box.count())] != items:
            box.clear()
            box.addItems(items)
        box.setCurrentIndex(items.index(held) if held in items else 0)

    def settings(self) -> dict[str, Any]:
        """The check's arguments as `plot_sine_distortion` takes them:
        'tone', 'channel' and 'excess' (a fraction)."""
        return {'tone': self.tone_box.currentText() or None,
                'channel': self.channel_box.currentText() or None,
                'excess': round(self.excess_box.value() / 100.0, 6)}

    # ---- the reading ------------------------------------------------------

    def _edited(self, *_args: Any) -> None:
        if not self._loading and self._history is not None:
            self._read()

    def _refused(self, refusal: Exception) -> None:
        self.check = None
        self._key = None
        for value in self.derived.values():
            value.setText('—')
        self.notes.setText(str(refusal))
        self.preview.show_check(None, None)

    def _read(self) -> None:
        """Check the levels with the settings shown, unless the same check
        is already held, and show it. The RMS readings depend on the
        record, the levels and the tone only, so a new channel or limit
        reads the check already made."""
        from ..core.sine_distortion import check_distortion
        from ..plot.sine_distortion import readout

        settings = self.settings()
        key = (id(self._history), id(self._history.ordinate),
               id(self._specification), id(self._levels), settings['tone'])
        try:
            if key != self._key:
                self.check = check_distortion(
                    self._history, self._specification, self._levels,
                    tone=settings['tone'], excess=settings['excess'])
                self._key = key
            elif self.check.excess != settings['excess']:
                self.check = replace(self.check, excess=settings['excess'])
        except (ValueError, KeyError) as refusal:
            self._refused(refusal)
            return
        dof = settings['channel']
        words = readout(self.check, dof)
        for name, value in self.derived.items():
            value.setText(words[name])
        self.notes.setText('\n'.join(self.check.notes))
        self.preview.show_check(self.check, dof)
        self.read.emit(self._history, self._specification, self._levels,
                       self.check, dof)

    def apply_theme(self, colors: Mapping[str, str]) -> None:
        """Redraw the small plot in a new theme's colors."""
        self.preview.apply_theme(colors)
        if self.check is not None:
            self.preview.show_check(self.check, self.settings()['channel'])


__all__ = ['DistortionPanel', 'DistortionPlot']
