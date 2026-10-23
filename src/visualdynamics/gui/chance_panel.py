"""The lines outside a random comparison's bands beside what chance puts
there, next to the plot that draws them.

The Chance reading of a specification with a measured PSD against it
draws one channel's miss from its target in dB (`plot.chance`). This
panel holds the two settings that reading is made under and what came
of it. The settings: how many adjacent lines are judged together (one,
or the number a test document allows to be combined for judging a
tolerance), and the degrees of freedom of one line, read from the PSD
when it says how it was averaged and typed here when it does not. The
readings: the DOF per line and per group judged, the lines judged, and
for each pair of limits the count outside beside the number chance
alone puts there, below and above.

Every edit redraws the plot at once. Nothing is made: the reading is of
two objects that already exist, as the RMS bars are, so there is no
Apply button. The same call a script makes,
`core.chance.compare_chance`, gives every number shown here.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QWidget

from ..core.chance import PAIRS, describe_chance
from .editors import DoubleSpinBox, SpinBox, number_box
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: the most lines that may be combined into one group
MOST_COMBINED = 64

#: the readings under the rule, in order: key and row name
ROWS = (('dof', 'DOF per line'), ('judged_dof', 'DOF judged'),
        ('lines', 'Lines judged'))


class ChancePanel(QWidget):
    """The settings and the reading. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Lines by Chance')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.combine_box: SpinBox = SpinBox()
        self.combine_box.setRange(1, MOST_COMBINED)
        self.combine_box.setValue(1)
        self.combine_box.setToolTip(
            'How many adjacent lines are judged together, their powers '
            'added and set against the limits\' power over the same '
            'band. Some test documents allow this for judging a '
            'tolerance; each group is worth about this many times a '
            'line\'s degrees of freedom, so chance puts fewer outside')
        # zero stands for "the PSD's own count", said in words in the box
        self.dof_box: DoubleSpinBox = number_box(
            0.0, 1.0e6, value=0.0, decimals=1, step=10.0,
            tip='The statistical degrees of freedom of one line. Left '
                'at "from the PSD", the count the PSD was stamped with '
                'when it was averaged; a PSD that does not say needs '
                'one typed here')
        self.dof_box.setSpecialValueText('from the PSD')
        add_settings(grid, [('Combine lines', self.combine_box),
                            ('DOF', self.dof_box)])
        add_rule(grid, 3)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 4).items()}
        # a paragraph per pair of limits, as many as the specification
        # carries, so a label that wraps rather than a fixed set of rows
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 4 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(5 + len(ROWS), 1)
        self.combine_box.valueChanged.connect(self._edited)
        self.dof_box.valueChanged.connect(self._edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """The reading's arguments as `compare_chance` takes them:
        'combine' and 'dof' (None for the PSD's own count)."""
        dof = float(self.dof_box.value())
        return {'combine': int(self.combine_box.value()),
                'dof': dof if dof > 0.0 else None}

    def set_settings(self, combine: int | None = None,
                     dof: float | None = None) -> None:
        """Set the fields without a redraw per field, then emit once.
        `dof` None returns the count to the PSD's own."""
        self._loading = True
        try:
            if combine is not None:
                self.combine_box.setValue(int(combine))
            self.dof_box.setValue(0.0 if dof is None else float(dof))
        finally:
            self._loading = False
        self._edited()

    def show_result(self, result: dict[str, Any] | None,
                    reason: str = '') -> None:
        """Write a `compare_chance` result into the rows and a paragraph
        per pair of limits under them. None clears the rows and shows
        `reason` instead, the words for why nothing was read."""
        if result is None:
            for value in self.derived.values():
                value.setText('—')
            self.readout.setText(reason)
            return
        self.derived['dof'].setText(f'{result["dof"]:.4g}')
        self.derived['judged_dof'].setText(f'{result["judged_dof"]:.4g}')
        self.derived['lines'].setText(
            str(result['lines']) if result['combine'] == 1
            else f'{result["lines"]} groups of {result["combine"]}')
        paragraphs = [describe_chance(result, pair)
                      for pair in PAIRS if f'{pair}_lines' in result]
        self.readout.setText('\n\n'.join(paragraphs) if paragraphs
                             else 'The specification carries no bands.')
