"""The tone readout, beside the autospectrum it reads.

The Tones reading of a plain autospectrum puts a cursor on the plot.
Wherever it stands, the tone nearest it is read off the spectrum
(`core.tone_readout.read_tone`) and drawn on it (`plot.tone_readout`),
and this panel says what was read: the tone's amplitude and rms, the
spike's height beside it and the height it would have on a line, the
floor taken away, the line spacing, the window and its noise bandwidth,
and where the tone sits between lines. The spike and the amplitude are
shown together because a spike's height moves with the spacing and the
window while the amplitude does not, and the plot alone invites reading
one as the other.

One setting is made here, the window, because a spectrum does not carry
the one it was averaged with: a PSD computed in the project has it in
its provenance (`Project.averaging_of`), and the panel starts from that
and says so; an imported one is read as Hann, the common choice, and the
panel says that too. Nothing is made, so there is no Apply button: the
reading is of a spectrum that already exists, as the octave reading's
preview is before its button is pressed.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from ..core.averaging import WINDOWS
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: the readings under the rule, in order: key and row name
ROWS = (('source', 'Window from'), ('channel', 'Channel'),
        ('frequency', 'Tone at'), ('amplitude', 'Tone'), ('rms', 'Tone rms'),
        ('spike', 'Spike'), ('on_line', 'On a line'), ('floor', 'Floor'),
        ('spacing', 'Line spacing'), ('bandwidth', 'Bandwidth'),
        ('offset', 'Off its line'))


class TonePanel(QWidget):
    """The window setting and the reading. Edits are emitted as the
    window's name."""

    #: the window was changed; here is its name now
    changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Tone Readout')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.window_box: QComboBox = QComboBox()
        for name in WINDOWS:
            self.window_box.addItem(name, name)
        self.window_box.setToolTip(
            'The window the spectrum was averaged with. It sets how far '
            'a tone leaks and how tall its spike stands, so a reading '
            'under the wrong window reads the spike wrong, though the '
            'amplitude summed from the lines barely moves')
        add_settings(grid, [('Window', self.window_box)])
        add_rule(grid, 2)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 3).items()}
        # what the reading warns, in words, under the numbers: a tone
        # between lines, a floor that differs either side, no tone
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 3 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(4 + len(ROWS), 1)
        self.window_box.currentIndexChanged.connect(self._edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.derived['source'].setText('chosen')
            self.changed.emit(self.window())

    def window(self) -> str:
        """The window chosen, one of `core.averaging.WINDOWS`."""
        return str(self.window_box.currentData())

    def set_window(self, name: str, source: str) -> None:
        """Show `name` as the window without emitting, and say where it
        came from ('the averaging', 'chosen', or 'assumed')."""
        self._loading = True
        try:
            self.window_box.setCurrentIndex(self.window_box.findData(name))
        finally:
            self._loading = False
        self.derived['source'].setText(source)

    def show_reading(self, reading: Any, channel: str = '', *,
                     scale: float = 1.0, density_scale: float = 1.0,
                     unit: str = '') -> None:
        """Write `reading`, a `ToneReading`, into the rows: amplitudes
        converted by `scale` into `unit`, densities by `density_scale`.
        None clears them."""
        values = dict.fromkeys(key for key, _name in ROWS if key != 'source')
        if reading is not None:
            density = f'{unit}²/Hz' if unit else '/Hz'
            amplitude = ('none' if not reading.amplitude else
                         f'{reading.amplitude * scale:.4g} {unit} peak')
            values.update(
                channel=channel or '—',
                frequency=f'{reading.frequency:.4g} Hz',
                amplitude=amplitude.replace('  ', ' '),
                rms=f'{reading.rms * scale:.4g} {unit}'.strip(),
                spike=f'{reading.spike * density_scale:.4g} {density}',
                on_line=(f'{(reading.on_line + reading.floor) * density_scale:.4g}'
                         f' {density}'),
                floor=f'{reading.floor * density_scale:.4g} {density}',
                spacing=f'{reading.spacing:.4g} Hz',
                bandwidth=f'{reading.noise_bandwidth:.3g} lines',
                offset=f'{abs(reading.offset):.2f} of a line')
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        self.notes.setText('' if reading is None else '\n'.join(
            note[0].upper() + note[1:] + '.' for note in reading.notes))
