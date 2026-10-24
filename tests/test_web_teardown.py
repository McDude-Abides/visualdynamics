"""The test helpers' web-view teardown ends the page's render process
before the page goes (2026-10-02): a page torn down with a live
renderer could hang a gate worker for good, by delete or by discard.
"""

from __future__ import annotations

import os
import sys

import pytest
from conftest import web_close, web_read, web_view


def _alive(pid: int) -> bool:
    """Whether a process is still running. On POSIX signal 0 asks
    without sending anything; on Windows signal 0 is CTRL_C_EVENT, so
    the process handle's exit code is read instead."""
    if sys.platform != 'win32':
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    import ctypes

    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)  # query limited
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def test_closing_a_view_ends_its_render_process_first(qt_app):
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    view = web_view()
    view.setHtml('<html><body><p id="here">here</p></body></html>')
    view.show()
    web_read(view, "document.getElementById('here') ? 'yes' : null",
             ready=lambda v: v == 'yes')
    pid = int(view.page().renderProcessPid())
    assert pid > 0, 'a loaded page has a render process'
    assert _alive(pid), 'the render process runs before the close'
    web_close(view)
    assert not _alive(pid), 'the close ends the render process'
