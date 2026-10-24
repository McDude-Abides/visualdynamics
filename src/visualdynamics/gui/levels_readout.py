"""What the sine view's readouts of extracted levels share.

The distortion readout (`gui.distortion_panel`) and the frequency readout
(`gui.frequency_panel`) are one shape: a rule and a title under the
extraction's settings, the tone and the control channel to check, the
check's own settings, the numbers it found, its notes, and a small plot.
Each reads the levels already extracted and never extracts them, hides
beside a record with none (principle 3), says so and checks nothing
beside levels the project marks stale, and reads again at once on an
edit, keeping the expensive part of the check when only its judgement
moved. That behavior lives here once; each readout says what it checks,
what its settings are and how its numbers are worded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QWidget,
)

from .settings_panel import PreviewPlot, add_derived, add_keyed_settings


class CheckPlot(PreviewPlot):
    """A check against frequency, small: frequency on the log axis every
    sine level is read on, the check's own reading linear up the side.
    A readout's plot says how the check is drawn (`draw`) and how far up
    the side it reaches (`span`)."""

    HEIGHT = 160
    LOG_Y = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items: list[Any] = []

    def draw(self, check: Any, dof: str | None) -> list[Any]:
        """Draw `check` on channel `dof` and return the items drawn."""
        raise NotImplementedError

    def span(self, check: Any, dof: str | None) -> tuple[float, float]:
        """The range up the side, (bottom, top)."""
        raise NotImplementedError

    def show_check(self, check: Any, dof: str | None) -> None:
        """Draw channel `dof` of `check`, or clear the plot when the
        check is None."""
        for item in self.items:
            self.plot.removeItem(item)
        self.items = []
        if check is None:
            return
        self.items = self.draw(check, dof)
        self.plot.setYRange(*self.span(check, dof), padding=0.0)


class LevelsReadout(QWidget):
    """The tone, the channel and the check's settings, then what it
    found. A readout names its `TITLE`, its plot class `PLOT`, its
    derived rows `DERIVED` as (key, label), and fills in the methods
    below the line."""

    TITLE = ''
    PLOT: type[CheckPlot] = CheckPlot
    DERIVED: tuple[tuple[str, str], ...] = ()

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
        self.title: QLabel = QLabel(self.TITLE)
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
        #: why the levels shown are stale, when they are: nothing is
        #: checked against them until they are extracted again
        self._stale: str | None = None
        self._loading = False

        self.tone_box: QComboBox = QComboBox()
        self.tone_box.setToolTip('The tone checked, of those extracted')
        self.channel_box: QComboBox = QComboBox()
        self.channel_box.setToolTip(
            'The control channel checked: the extraction reads the '
            'channels the specification controls')
        rows = ([('tone', 'Tone', self.tone_box),
                 ('channel', 'Channel', self.channel_box)]
                + self._setting_rows())
        self.names, self.editors = add_keyed_settings(grid, rows,
                                                      first_row=2)
        first = 2 + len(rows)
        made = add_derived(grid, self.DERIVED, first)
        self.derived: dict[str, QLabel] = {key: value for key, (_name, value)
                                           in made.items()}
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self.notes, first + len(self.DERIVED), 0, 1, 2)
        self.preview: CheckPlot = self.PLOT()
        grid.addWidget(self.preview, first + len(self.DERIVED) + 1, 0, 1, 2)
        self.tone_box.currentIndexChanged.connect(self._edited)
        self.channel_box.currentIndexChanged.connect(self._edited)

    # ---- what is being read -------------------------------------------------

    def show_levels(self, history: Any, specification: Any, levels: Any,
                    stale: str | None = None) -> Any:
        """Check the extracted `levels` of `history` against
        `specification` and show it; hide, and return None, when there are
        no levels to check (principle 3). The tone and channel boxes list
        what the levels hold and keep the ones chosen while the levels
        keep them; the tone row shows only when there is more than one.
        `stale` is the project's reason the levels are stale, if they
        are: then nothing is checked and the reason is shown, since the
        levels describe the record as it was. Returns the check, or None
        when there is none or it was refused (the refusal is shown)."""
        if levels is None or not len(levels):
            self.hide()
            self.check = None
            self._stale = None
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
        self._stale = stale or None
        self.show()
        if stale:
            self._refused(f'These levels are stale ({stale}); extract them '
                          'again to check the record against them.')
            return None
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
        """The check's arguments as its plot function takes them: 'tone',
        'channel' and the readout's own (`_own_settings`)."""
        return {'tone': self.tone_box.currentText() or None,
                'channel': self.channel_box.currentText() or None,
                **self._own_settings()}

    # ---- the reading ------------------------------------------------------

    def _edited(self, *_args: Any) -> None:
        if (not self._loading and self._history is not None
                and self._stale is None):
            self._read()

    def _refused(self, refusal: Exception | str) -> None:
        self.check = None
        self._key = None
        for value in self.derived.values():
            value.setText('—')
        self.notes.setText(str(refusal))
        self.preview.show_check(None, None)

    def _read(self) -> None:
        """Check the levels with the settings shown, unless the same check
        is already held, and show it. What a check reads off the record
        depends on the record, the levels and the tone only, so a new
        channel or tolerance rejudges the check already made."""
        settings = self.settings()
        key = (id(self._history), id(self._history.ordinate),
               id(self._specification), id(self._levels), settings['tone'])
        try:
            if key != self._key:
                self.check = self._make(settings)
                self._key = key
            else:
                self.check = self._rejudge(self.check, settings)
        except (ValueError, KeyError) as refusal:
            self._refused(refusal)
            return
        dof = settings['channel']
        words = self._words(self.check, dof)
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

    # ---- what each readout says -------------------------------------------

    def _setting_rows(self) -> list[tuple[str, str, QWidget]]:
        """The check's own settings, as (key, label, editor), built and
        connected to `_edited`."""
        return []

    def _own_settings(self) -> dict[str, Any]:
        """The check's own settings, as its plot function takes them."""
        return {}

    def _make(self, settings: dict[str, Any]) -> Any:
        """Run the check on the levels held, with `settings`."""
        raise NotImplementedError

    def _rejudge(self, check: Any, settings: dict[str, Any]) -> Any:
        """`check` judged with `settings`, keeping what it read."""
        return check

    def _words(self, check: Any, dof: str | None) -> dict[str, str]:
        """The derived rows' words, by key."""
        raise NotImplementedError


__all__ = ['CheckPlot', 'LevelsReadout']
