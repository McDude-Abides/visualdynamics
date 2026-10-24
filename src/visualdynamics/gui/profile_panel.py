"""A sine specification read as a profile, beside the profile drawn three
ways.

The Sine Profile reading of a sine specification reads one tone back as
segments of constant displacement, velocity and acceleration
(`core.sine_profile.read_profile`) and draws it as displacement,
velocity and acceleration against frequency with every crossover marked
(`plot.sine_profile`). This panel holds what the reading is made with:
how far a stretch between two breakpoints may slope off a quantity held
constant and still be read as it, in dB per octave, and whether the
tone's own level, as it interpolates its breakpoints, is drawn over the
profile. Under them are
what follows: the segments, the crossovers and how many check, and the
most the profile asks for in each quantity, which is what goes against a
shaker's stroke, velocity and force ratings; and then one line per
segment and per crossover. A tone that is not written in segments of the
three quantities (a slope of 3 dB per octave, a dwell) is refused by
name here, and its own level is still drawn: a breakpoint typed at the
wrong frequency or level bends the stretches beside it off all three,
so the refusal is where such a typo shows.

Every finished edit redraws. Nothing is made: the reading is of a
specification that already exists, as the sweep-rate check is, so there
is no Apply button. A script gets the same numbers from
`core.sine_profile.read_profile` and the same picture from
`plot_sine_profile`; a profile is written from its segments with
`core.sine_profile.SineProfile`.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QWidget

from ..core.sine_profile import SLOPE_TOLERANCE
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid


class ProfilePanel(QWidget):
    """The slope tolerance and whether the tone's own level is drawn,
    then what the profile asks for. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Sine Profile')
        grid = panel_grid(self, self.title)
        self._loading = False
        self.slope_box: DoubleSpinBox = DoubleSpinBox()
        self.slope_box.setRange(0.0, 6.0)
        self.slope_box.setDecimals(2)
        self.slope_box.setSingleStep(0.1)
        self.slope_box.setSuffix(' dB/oct')
        self.slope_box.setValue(SLOPE_TOLERANCE)
        self.slope_box.setToolTip(
            'How far a stretch between two breakpoints may slope off '
            'flat, 6 or 12 dB per octave and still be read as a constant '
            'acceleration, velocity or displacement: room for levels '
            'rounded to two figures')
        self.interpolated_box: QCheckBox = QCheckBox()
        self.interpolated_box.setChecked(True)
        self.interpolated_box.setToolTip(
            "Draw the tone's own level, its breakpoints interpolated as "
            'the tone runs them, dashed over the profile: where the two '
            'part, the breakpoints are too far apart for the segment they '
            'write')
        commit_on_enter(self.slope_box)
        self.names, self.editors = add_keyed_settings(grid, [
            ('slope_tolerance', 'Slope within', self.slope_box),
            ('interpolated', "Tone's own level", self.interpolated_box),
        ])
        add_rule(grid, 3)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, (('segments', 'Segments'),
                       ('crossovers', 'Crossovers'),
                       ('displacement', 'Most displacement'),
                       ('velocity', 'Most velocity'),
                       ('acceleration', 'Most acceleration')), 4).items()}
        # one line per segment and per crossover, as many as there are
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 9, 0, 1, 2)
        grid.setRowStretch(10, 1)
        self.slope_box.valueChanged.connect(self._edited)
        self.interpolated_box.toggled.connect(self._edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """The reading's settings: 'slope_tolerance', dB per octave, as
        `read_profile` takes it, and 'interpolated', as
        `plot_sine_profile` takes it."""
        return {'slope_tolerance': round(self.slope_box.value(), 6),
                'interpolated': self.interpolated_box.isChecked()}

    def set_settings(self, slope_tolerance: float | None = None,
                     interpolated: bool | None = None) -> None:
        """Set the fields without a redraw per field, then emit once."""
        self._loading = True
        try:
            if slope_tolerance is not None:
                self.slope_box.setValue(float(slope_tolerance))
            if interpolated is not None:
                self.interpolated_box.setChecked(bool(interpolated))
        finally:
            self._loading = False
        self._edited()

    def show_profile(self, profile: Any, refusal: str | None = None,
                     unit_system: Any = None) -> None:
        """Write what `profile` asks for, or, with none, the refusal that
        kept the tone from being read as one."""
        from ..plot.sine_profile import maximum_words

        if profile is None:
            for value in self.derived.values():
                value.setText('—')
            self.readout.setText(refusal or '')
            return
        self.derived['segments'].setText(str(len(profile.segments)))
        crossings = profile.crossings
        bad = sum(not crossing.checks for crossing in crossings)
        if not crossings:
            words = 'none, one segment'
        elif bad:
            words = f'{bad} of {len(crossings)} do not check'
        else:
            words = f'{len(crossings)}, every one checks'
        self.derived['crossovers'].setText(words)
        for quantity in ('displacement', 'velocity', 'acceleration'):
            self.derived[quantity].setText(
                maximum_words(profile, quantity, unit_system)
                .removeprefix('most '))
        self.readout.setText('\n'.join(profile.describe()[:-1]))


__all__ = ['ProfilePanel']
