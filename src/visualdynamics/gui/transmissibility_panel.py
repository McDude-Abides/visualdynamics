"""The transmissibility reading, beside the FRF it is read behind.

The Transmissibility toggle on an FRF's bar draws, over the FRF's
magnitude, the ASD-ratio transmissibility dashed and the band H1's
random error puts about H1 (`plot.transmissibility`), and puts a cursor
on the plot. Wherever the cursor stands this panel says, at that line,
what the two estimates read and why they differ: |H1| and its phase,
the ASD ratio and how far over H1 it reads, the coherence, the random
error of |H1| and the band it gives, and the frames and averages the
error was read at. The ratio over H1 is the coherence restated, since
the ratio is |H1| over the root of the coherence; it is shown in
decibels because that is how far apart the two curves stand on the
plot.

Nothing is set and nothing is made, so the panel is read-only: the
reading is of an FRF that exists, from the frames it was computed from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QLabel, QWidget

from .settings_panel import add_derived, panel_grid

#: the readings, in order: key and row name
ROWS = (('channel', 'Channel'), ('estimator', 'FRF drawn'),
        ('frequency', 'At'), ('h1', 'H1'), ('phase', 'H1 phase'),
        ('ratio', 'ASD ratio'), ('over', 'Ratio over H1'),
        ('coherence', 'Coherence'), ('error', 'Random error'),
        ('band', 'Band'), ('averages', 'Averages'))


class TransmissibilityPanel(QWidget):
    """The reading at the cursor's line, in rows, with the notes under
    them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Transmissibility')
        grid = panel_grid(self, self.title)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 1).items()}
        # what the reading should be read with, in words under the
        # numbers: the overlap discount, the lines past the small-error
        # limit of the expression
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 1 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(2 + len(ROWS), 1)

    def show_reading(self, reading: Any, point: Any = None, *,
                     estimator: str = '', scale: float = 1.0,
                     unit: str = '') -> None:
        """Write `point`, a `TransmissibilityPoint` of `reading`, into
        the rows, magnitudes converted by `scale` into `unit`, and the
        reading's notes under them, with the point's own when its error
        is past what its expression can size. `estimator` names the FRF
        on the plot. None clears them."""
        values = dict.fromkeys(key for key, _name in ROWS)
        notes: list[str] = []
        if reading is not None and point is not None:
            unit_text = f' {unit}' if unit else ''
            magnitude = point.magnitude * scale
            lower = point.lower
            if np.isfinite(point.error):
                error = f'±{100.0 * point.error:.1f} %'
            else:
                error = 'unbounded'
            band = (f'{lower * scale:.4g} to {point.upper * scale:.4g}'
                    if lower is not None and np.isfinite(point.upper)
                    else f'up to {point.upper * scale:.4g}'
                    if np.isfinite(point.upper) else 'none')
            values.update(
                channel=f'{reading.response} over {reading.reference}',
                estimator=estimator or '—',
                frequency=f'{point.frequency:.4g} Hz',
                h1=f'{magnitude:.4g}{unit_text}',
                phase=f'{point.phase:.1f}°',
                ratio=f'{point.asd_ratio * scale:.4g}{unit_text}',
                over=f'{point.overstatement_db:+.2f} dB',
                coherence=f'{point.coherence:.3f}',
                error=error,
                band=f'{band}{unit_text}' if band != 'none' else band,
                averages=(f'{reading.frames} frames, '
                          f'{reading.averages:.4g} independent'))
            notes = list(reading.notes)
            if not point.small:
                notes.insert(0, 'at this line the random error is past '
                                'what its expression can size: the '
                                'estimate is poor, by more than the band '
                                'says')
            if estimator and estimator != 'H1':
                notes.insert(0, f'the FRF drawn is {estimator}; the band '
                                'and the ratio are against H1 from the '
                                'same frames, drawn solid with them')
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        self.notes.setText('\n'.join(note[0].upper() + note[1:] + '.'
                                     for note in notes))
