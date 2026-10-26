"""What a long verb's input gate keeps out, and where it is put.

While `MainWindow._run_long` holds the event loop, mouse presses, the
wheel, keys and shortcuts go nowhere except to the Cancel button and
Esc. The gate filters the shown windows and never the application: a
filter on the application is handed every object's events, and PySide
makes a Python object for each receiver to hand it over, the web
engine's internal objects included, which crashed the loop on Linux.
"""

from __future__ import annotations

import threading

import pytest


def _inside_the_loop(window, act):
    """Run `act` on this thread while a long verb holds the loop, and
    return what it gave back. The verb waits for `act` to finish, then
    ticks, so a cancel asked meanwhile is raised from the runner."""
    from PySide6.QtCore import QTimer

    done = threading.Event()
    kept = []

    def turn():
        try:
            kept.append(act())
        finally:
            done.set()

    def verb(progress=None):
        done.wait(30)
        progress(1, 1)

    QTimer.singleShot(0, turn)
    window._run_long('Waiting…', verb)
    return kept[0]


def _center(widget, window):
    """The middle of `widget`, in the window's (and its QWindow's)
    coordinates, with any layout still owed done first."""
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.sendPostedEvents()
    return widget.mapTo(window, widget.rect().center())


def test_the_gate_filters_the_windows_not_every_object(window, monkeypatch):
    """An event sent to an object outside the windows never reaches the
    gate, while the window's own events do."""
    from PySide6.QtCore import QCoreApplication, QEvent, QObject

    from visualdynamics.gui import main_window

    seen = []
    real = main_window._InputGate.eventFilter

    def watched(gate, target, event):
        seen.append(target)
        return real(gate, target, event)

    monkeypatch.setattr(main_window._InputGate, 'eventFilter', watched)
    probe = QObject()

    def send():
        QCoreApplication.sendEvent(probe, QEvent(QEvent.Type.User))
        return [target for target in seen if target is probe]

    assert _inside_the_loop(window, send) == []
    assert any(target is window for target in seen), \
        'the window was filtered meanwhile'


def test_a_shortcut_and_a_click_go_nowhere_while_a_verb_runs(window):
    """The shortcut and the click that act with no verb running, sent
    the way the platform sends them, through the QWindow, do nothing
    while one runs."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton

    handle = window.windowHandle()
    fired = []
    action = QAction('Probe', window)
    action.setShortcut(QKeySequence('Ctrl+F12'))
    action.triggered.connect(lambda: fired.append('shortcut'))
    window.addAction(action)
    button = QPushButton('Probe', window)
    button.setGeometry(40, 120, 120, 32)
    button.show()
    button.raise_()
    button.clicked.connect(lambda: fired.append('click'))

    def press():
        QTest.keyClick(handle, Qt.Key.Key_F12,
                       Qt.KeyboardModifier.ControlModifier)
        QTest.mouseClick(handle, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier,
                         _center(button, window))

    press()
    assert fired == ['shortcut', 'click'], 'both act with no verb running'
    fired.clear()
    _inside_the_loop(window, press)
    assert fired == [], 'neither acts while one runs'


@pytest.mark.parametrize('way', ['click', 'escape'])
def test_cancel_and_escape_reach_through_the_gate(window, way):
    """A click on Cancel and Esc, through the QWindow, stop the verb."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from visualdynamics.core.progress import Cancelled

    handle = window.windowHandle()

    def ask():
        if way == 'click':
            QTest.mouseClick(handle, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier,
                             _center(window._cancel_button, window))
        else:
            QTest.keyClick(handle, Qt.Key.Key_Escape)

    with pytest.raises(Cancelled):
        _inside_the_loop(window, ask)
