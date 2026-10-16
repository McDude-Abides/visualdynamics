"""The dwell readout: a held tone's build-up and ring-down, read in the
sine view.

A dwell's level against frequency is one point, so the sine view's
levels say nothing of it. When the specification beside the record
holds a dwell, this readout sits under the extraction's settings and
reads the dwell against time (`core.dwell.read_dwell`): on a channel
chosen here (a response by default, since a control is held at its
level and does not build), with the peak over each cycle, the steady
level the dwell reached, the cycles it took to reach the share set here
(95 % by default), and the cycles after the stop for the level to fall
to the rest of it. The small plot under the numbers is the level against
time with the steady level and the two shares drawn, the same drawing
`plot_dwell` makes (`plot.dwell.draw_dwell`).

It is a reading, not an act: nothing is made, so there is no button,
and an edit reads again at once. Every number shown is the reading's.
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

from ..core.dwell import DWELL_SHARE
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import PreviewPlot, add_derived, add_keyed_settings


class DwellPlot(PreviewPlot):
    """The level against time, small: linear on both axes, since time
    and a share of a level are both read linearly."""

    HEIGHT = 160
    LOG_Y = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot.setLogMode(x=False, y=False)
        self.items: list[Any] = []

    def show_reading(self, reading: Any, unit_system: Any) -> None:
        """Draw `reading`, or clear the plot when it is None."""
        from ..plot.dwell import draw_dwell

        for item in self.items:
            self.plot.removeItem(item)
        self.items = []
        if reading is not None:
            self.items = draw_dwell(self.plot, reading,
                                    unit_system=unit_system,
                                    colors=self.colors, legend=False)


class DwellPanel(QWidget):
    """The channel and the share, then what the reading found."""

    #: the dwell was read: (history, specification, reading), so the
    #: window can journal the call that draws it
    read = Signal(object, object, object)

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
        self.title: QLabel = QLabel('Dwell')
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        grid.addWidget(self.title, 1, 0, 1, 2)
        #: what is being read, and the reading last made of it
        self._history: Any = None
        self._specification: Any = None
        self._unit_system: Any = None
        self._key: tuple | None = None
        self.reading: Any = None
        self._loading = False

        self.channel_box: QComboBox = QComboBox()
        self.channel_box.setToolTip(
            'The channel read. A response builds at a resonance; the '
            'control is held at its level by the controller')
        self.share_box: DoubleSpinBox = DoubleSpinBox()
        self.share_box.setRange(50.0, 99.9)
        self.share_box.setDecimals(1)
        self.share_box.setSingleStep(1.0)
        self.share_box.setSuffix(' %')
        self.share_box.setValue(100.0 * DWELL_SHARE)
        self.share_box.setToolTip(
            'The share of the steady level the build-up is read to; the '
            'ring-down is read down to the rest of it. A resonance '
            'reaches 95 % in about Q cycles')
        self.names, self.editors = add_keyed_settings(grid, [
            ('channel', 'Channel', self.channel_box),
            ('share', 'Share', self.share_box)], first_row=2)
        made = add_derived(grid, (('frequency', 'Frequency'),
                                  ('steady', 'Steady'),
                                  ('built', 'Build-up'),
                                  ('rung', 'Ring-down'),
                                  ('settled', 'Settled')), 4)
        self.derived: dict[str, QLabel] = {key: value for key, (_name, value)
                                           in made.items()}
        self.derived_names: dict[str, QLabel] = {
            key: name for key, (name, _value) in made.items()}
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self.notes, 9, 0, 1, 2)
        self.preview: DwellPlot = DwellPlot()
        grid.addWidget(self.preview, 10, 0, 1, 2)
        commit_on_enter(self.share_box)
        self.channel_box.currentIndexChanged.connect(self._edited)
        self.share_box.valueChanged.connect(self._edited)
        self._name_shares()

    # ---- what is being read -------------------------------------------------

    def show_dwell(self, history: Any, specification: Any,
                   unit_system: Any = None) -> Any:
        """Read the dwell `specification` holds on `history` and show it;
        hide, and return None, when no tone of it holds a dwell (principle
        3). The channel box lists the record's channels and keeps the one
        chosen while the record keeps it. Returns the reading, or None when
        there is none or it was refused (the refusal is shown)."""
        if not any(tone.dwell_segments() for tone in specification.tones):
            self.hide()
            self.reading = None
            return None
        dofs = [str(dof) for dof in history.response_dof]
        held = self.channel_box.currentText()
        self._loading = True
        try:
            if [self.channel_box.itemText(i)
                    for i in range(self.channel_box.count())] != dofs:
                self.channel_box.clear()
                self.channel_box.addItems(dofs)
                held = held if held in dofs else self._default(history,
                                                               specification)
            self.channel_box.setCurrentIndex(dofs.index(held) if held in dofs
                                             else 0)
        finally:
            self._loading = False
        self._history, self._specification = history, specification
        self._unit_system = unit_system
        self.show()
        self._read()
        return self.reading

    @staticmethod
    def _default(history: Any, specification: Any) -> str:
        """The channel `read_dwell` reads when none is named."""
        from ..core.dwell import default_channel

        try:
            return default_channel(history, specification)
        except ValueError:
            return str(history.response_dof[0])

    def settings(self) -> dict[str, Any]:
        """The reading's arguments as `read_dwell` takes them: 'channel'
        and 'share' (a fraction)."""
        return {'channel': self.channel_box.currentText() or None,
                'share': round(self.share_box.value() / 100.0, 6)}

    # ---- the reading ------------------------------------------------------

    def _name_shares(self) -> None:
        """Name the two counts by the share they are read to, so the
        rows say what was counted: 'To 95 %', 'Down to 5 %'."""
        share = self.share_box.value()
        self.derived_names['built'].setText(f'To {share:g} %')
        self.derived_names['rung'].setText(f'Down to {100.0 - share:g} %')

    def _edited(self, *_args: Any) -> None:
        self._name_shares()
        if not self._loading and self._history is not None:
            self._read()

    def _read(self) -> None:
        """Read the dwell with the settings shown, unless the same
        reading is already held, and show it."""
        from ..core.dwell import read_dwell
        from ..plot.dwell import readout

        settings = self.settings()
        key = (id(self._history), id(self._history.ordinate),
               id(self._specification), settings['channel'],
               settings['share'])
        if key != self._key:
            try:
                self.reading = read_dwell(self._history, self._specification,
                                          **settings)
            except ValueError as refusal:
                self.reading = None
                self._key = None
                for value in self.derived.values():
                    value.setText('—')
                self.notes.setText(str(refusal))
                self.preview.show_reading(None, self._unit_system)
                return
            self._key = key
        words = readout(self.reading, self._unit_system)
        for name, value in self.derived.items():
            value.setText(words[name])
        self.notes.setText('\n'.join(self.reading.notes))
        self.preview.show_reading(self.reading, self._unit_system)
        self.read.emit(self._history, self._specification, self.reading)

    def apply_theme(self, colors: Mapping[str, str]) -> None:
        """Redraw the small plot in a new theme's colors."""
        self.preview.apply_theme(colors)
        self.preview.show_reading(self.reading, self._unit_system)


__all__ = ['DwellPanel', 'DwellPlot']
