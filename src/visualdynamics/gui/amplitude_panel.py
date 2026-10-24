"""The amplitude spectrum's window, and what it read, beside the plot.

The Amplitude reading of a time history draws one channel's amplitude
spectrum (`plot.amplitude_spectrum`): each line scaled so a sine reads
its amplitude in peak units, the frames' magnitudes averaged. This
panel holds the one setting the reading is made with, above the rule:
the window, with its alpha or beta for a Tukey or Kaiser window. Hann
is the default, and flattop is the window for a tone's amplitude
wherever it falls between lines. The frames themselves are the record's
averaging, set in the Averaging reading, and the panel says so.

Under the rule are the reading's numbers: where the frames came from,
the line spacing and how many frames were averaged, the window's
coherent gain and the amplitude correction it gives, its noise
bandwidth, how far low it reads a sine midway between lines, and the
tallest line above DC in peak and in rms. Under them, in words, what an
amplitude spectrum's floor is not. Nothing is made: the reading is of a
record that already exists, as the time statistics are, so there is no
Apply button.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from ..core.averaging import WINDOW_PARAMETERS, WINDOWS
from .editors import DoubleSpinBox
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: the window the reading starts with
DEFAULT_WINDOW = 'hann'

#: the readings under the rule, in order: key and row name
ROWS = (('channel', 'Channel'), ('source', 'Frames from'),
        ('spacing', 'Line spacing'), ('frames', 'Frames averaged'),
        ('gain', 'Coherent gain'), ('correction', 'Correction'),
        ('bandwidth', 'Noise bandwidth'), ('scalloping', 'Between lines'),
        ('frequency', 'Tallest at'), ('amplitude', 'Amplitude'),
        ('rms', 'Amplitude rms'))

#: what is said under the numbers whatever the window: what the average
#: of magnitudes does to a tone, and what it does not make of a floor
FLOOR_NOTE = ('The frames\' magnitudes are averaged, so a tone reads its '
              'amplitude whatever its phase in each frame. The floor '
              'between tones is not a level: it moves with the line '
              'spacing and the window. A random level is read from the '
              'ASD.')


class AmplitudePanel(QWidget):
    """The window setting and the reading. Edits are emitted as
    `settings()`."""

    #: the window or its parameter moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Amplitude Spectrum')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.window_box: QComboBox = QComboBox()
        for name in WINDOWS:
            self.window_box.addItem(name, name)
        self.window_box.setCurrentIndex(self.window_box.findData(
            DEFAULT_WINDOW))
        self.window_box.setToolTip(
            'The window each frame is multiplied by. Its amplitude '
            'correction is applied, so a sine on a line reads its '
            'amplitude under any window; hann reads one midway between '
            'lines 1.42 dB low, flattop within 0.01 dB')
        # a Tukey's alpha or a Kaiser's beta, shown only while such a
        # window is chosen, as the averaging and tone panes show it
        self.parameter_box: DoubleSpinBox = DoubleSpinBox()
        self.parameter_box.setDecimals(2)
        self.parameter_box.setToolTip(
            "The window's own number: Tukey's alpha, the tapered "
            "fraction, or Kaiser's beta")
        names = add_settings(grid, [('Window', self.window_box),
                                    ('Alpha', self.parameter_box)])
        self._parameter_label: QLabel = names[1]
        add_rule(grid, 3)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 4).items()}
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 4 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(5 + len(ROWS), 1)
        self._restate_parameter(None)
        self.window_box.currentIndexChanged.connect(self._window_edited)
        self.parameter_box.valueChanged.connect(self._edited)

    def _restate_parameter(self, value: float | None) -> None:
        """Fit the parameter row to the window: named for its number and
        set to `value`, or its default when None; hidden for a window
        that takes none."""
        settings = WINDOW_PARAMETERS.get(self.window())
        self._parameter_label.setVisible(settings is not None)
        self.parameter_box.setVisible(settings is not None)
        if settings is None:
            return
        what, default, (lowest, highest) = settings
        was, self._loading = self._loading, True
        try:
            self._parameter_label.setText(what.capitalize())
            self.parameter_box.setRange(lowest, highest)
            self.parameter_box.setSingleStep((highest - lowest) / 20.0)
            self.parameter_box.setValue(default if value is None else value)
        finally:
            self._loading = was

    def _window_edited(self, *_args: Any) -> None:
        if not self._loading:
            self._restate_parameter(None)
        self._edited()

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def window(self) -> str:
        """The window chosen, one of `core.averaging.WINDOWS`."""
        return str(self.window_box.currentData())

    def settings(self) -> dict[str, Any]:
        """The reading's settings: 'window', and 'parameter', the Tukey
        window's alpha or the Kaiser window's beta (None for a window
        that takes none), as `TimeHistory.amplitude_spectrum` takes
        them."""
        parameter = (float(self.parameter_box.value())
                     if self.window() in WINDOW_PARAMETERS else None)
        return {'window': self.window(), 'parameter': parameter}

    def set_settings(self, settings: dict[str, Any] | None) -> None:
        """Set the fields from a `settings()` dict without emitting,
        since the caller is about to draw."""
        if not settings:
            return
        self._loading = True
        try:
            self.window_box.setCurrentIndex(self.window_box.findData(
                settings.get('window', DEFAULT_WINDOW)))
            self._restate_parameter(settings.get('parameter'))
        finally:
            self._loading = False

    def show_reading(self, spectrum: Any, record: int = 0,
                     tallest: tuple[float, float] | None = None, *,
                     scale: float = 1.0, unit: str = '',
                     unread: str | None = None) -> None:
        """Write channel `record` of `spectrum`, an `AmplitudeSpectrum`,
        into the rows, with `tallest` its tallest line as (frequency,
        amplitude in SI); amplitudes converted by `scale` into `unit`.
        `unread` is why no line could be read, said in the notes. None
        clears them."""
        values = dict.fromkeys(key for key, _name in ROWS)
        if spectrum is not None:
            frames = spectrum.frames[record]
            bandwidth = f'{spectrum.noise_bandwidth:.3g}'
            values.update(
                channel=spectrum.label(record),
                source=('the averaging' if spectrum.from_averaging
                        else 'each record whole'),
                spacing=f'{spectrum.spacing:.4g} Hz',
                frames=f'{frames}',
                gain=f'{spectrum.coherent_gain:.4f}',
                correction=f'×{spectrum.correction:.3f}',
                bandwidth=(f'{bandwidth} line' if bandwidth == '1'
                           else f'{bandwidth} lines'),
                scalloping=f'{spectrum.scalloping_db:.2f} dB low')
            if tallest is not None:
                frequency, amplitude = tallest
                values.update(
                    frequency=f'{frequency:.4g} Hz',
                    amplitude=f'{amplitude * scale:.4g} {unit} peak'
                    .replace('  ', ' '),
                    rms=f'{amplitude * scale / 2.0 ** 0.5:.4g} {unit} rms'
                    .replace('  ', ' '))
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        if spectrum is None:
            self.notes.setText('')
            return
        notes = [] if unread is None else [unread[:1].upper() + unread[1:]
                                           + '.']
        notes.append(FLOOR_NOTE)
        if spectrum.scalloping_db > 0.1:
            # the tallest line is the reading, and between lines it is
            # short by as much as the window's scalloping
            notes.append(f'A tone between lines reads up to '
                         f'{spectrum.scalloping_db:.2f} dB low under '
                         f'{spectrum.window}; flattop reads it within '
                         '0.01 dB.')
        self.notes.setText('\n'.join(notes))
