"""The confidence band, beside the averaged autospectrum it is drawn on.

The Confidence reading of an autospectrum shades the chi-square band
the true spectrum lies in (`plot.confidence_preview`), and this panel
says what the band is: the confidence, the degrees of freedom it was
worked out at and where they came from, and the band's two edges in
decibels and as factors of the estimate. The numbers are the plot's
own (`Psd.confidence_reading`, the rule `Psd.confidence_band` draws
by), so a figure read here is the edge on the plot.

Two settings are made here. The confidence is 90 or 95 percent. The
degrees of freedom are the spectrum's own when it carries them, as a
PSD computed in the project does (`Psd.statistical_dof`, the
equivalent count discounted for overlap), and they can be typed: for a
spectrum read from a file, which carries none, or to read the band at
the count another system says it averaged to. Nothing is made, so
there is no Apply button: the band is a statement about a spectrum
that already exists.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from ..core.averaging import CONFIDENCE_LEVELS, db_text, dof_text
from .editors import DoubleSpinBox
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: the readings under the rule, in order: key and row name
ROWS = (('source', 'DOF from'), ('channel', 'Channel'),
        ('dof', 'Equivalent DOF'), ('below', 'Below'), ('above', 'Above'),
        ('factors', 'Factors'))

#: the largest count that can be typed; past a few thousand the band is
#: a tenth of a decibel wide and a larger number changes nothing seen
MOST_DOF = 1.0e6


class ConfidencePanel(QWidget):
    """The confidence, the count and the reading. Edits are emitted
    with no argument; `level` and `typed_dof` give the settings."""

    #: the confidence or the count was changed
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Confidence Band')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.level_box: QComboBox = QComboBox()
        for level in CONFIDENCE_LEVELS:
            self.level_box.addItem(f'{round(level * 100)} %', level)
        self.level_box.setToolTip(
            'How sure the band is: the share of lines whose true value '
            'lies inside it. 95 % is wider than 90 %')
        # zero is 'as averaged': the spectrum's own count, which a
        # computed PSD carries and an imported one does not
        self.dof_box: DoubleSpinBox = DoubleSpinBox()
        self.dof_box.setRange(0.0, MOST_DOF)
        self.dof_box.setDecimals(1)
        self.dof_box.setSingleStep(10.0)
        self.dof_box.setSpecialValueText('as averaged')
        self.dof_box.setToolTip(
            'The statistical degrees of freedom the band is worked out '
            "at. 'as averaged' takes the spectrum's own equivalent "
            'count; type one for a spectrum read from a file, from the '
            'system that averaged it. Not the measurement DOFs the '
            'channels are taken at')
        add_settings(grid, [('Confidence', self.level_box),
                            ('DOF', self.dof_box)])
        add_rule(grid, 3)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 4).items()}
        # the band in words under the numbers: what it means, and when
        # the records drawn were averaged to different counts
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 4 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(5 + len(ROWS), 1)
        self.level_box.currentIndexChanged.connect(self._edited)
        self.dof_box.valueChanged.connect(self._edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit()

    def level(self) -> float:
        """The confidence chosen, a fraction such as 0.9."""
        return float(self.level_box.currentData())

    def typed_dof(self) -> float | None:
        """The count typed, or None for the spectrum's own."""
        value = float(self.dof_box.value())
        return value if value > 0.0 else None

    def set_settings(self, level: float, dof: float | None) -> None:
        """Show `level` and `dof` (None for 'as averaged') without
        emitting."""
        self._loading = True
        try:
            at = self.level_box.findData(level)
            if at >= 0:
                self.level_box.setCurrentIndex(at)
            self.dof_box.setValue(0.0 if dof is None else float(dof))
        finally:
            self._loading = False

    def show_reading(self, band: Any, channel: str = '', source: str = '',
                     counts: Any = None, passed: Any = None) -> None:
        """Write `band`, the first banded record's `ConfidenceBand`, into
        the rows, with the `channel` it belongs to and the `source` of
        its count ('the averaging' or 'typed'). `counts` are every
        banded record's counts, for the note when they differ, and
        `passed` the channels drawn with no line to band. None clears
        the rows, and `source` then says why ('not known' for no count,
        'no lines' for nothing to band)."""
        values = dict.fromkeys(key for key, _name in ROWS)
        values['source'] = source or None
        notes = []
        if band is not None:
            values.update(
                channel=channel or '—',
                dof=dof_text(band.dof),
                below=db_text(band.lower_db),
                above=db_text(band.upper_db),
                factors=(f'×{band.lower_factor:.3g} to '
                         f'×{band.upper_factor:.3g}'))
            share = round(band.level * 100)
            notes.append(
                f'At {share} % confidence the true spectrum lies inside '
                f'the band at {share} lines in 100; a line outside it is '
                'not by itself a fault.')
            spread = sorted({dof_text(n) for n in (counts or [])},
                            key=float)
            if len(spread) > 1:
                notes.append(
                    f'The records drawn were averaged to {spread[0]} to '
                    f'{spread[-1]} DOF; each is banded at its own, and '
                    'the rows read the first.')
            if passed:
                notes.append(
                    f'No band on {", ".join(passed)}: no line there is '
                    'finite and above zero.')
        elif source == 'no lines':
            values['source'] = None
            notes.append(
                'No line drawn is finite and above zero, so there is no '
                'band to shade or read.')
        elif source == 'not known':
            notes.append(
                'This spectrum does not say what it was averaged over. '
                'Type the degrees of freedom it was averaged to, from '
                'the system that computed it.')
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        self.notes.setText(' '.join(notes))
