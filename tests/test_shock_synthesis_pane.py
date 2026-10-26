"""SRS synthesis in the Generate Signal pane: the synthesis iterated off
the window's thread once the settings rest, held for them, drawn as
`plot_shock_synthesis` draws it, and made by Generate on the worker.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_shock_synthesis import RATE, _spec

import visualdynamics


@pytest.fixture
def held_back(monkeypatch):
    """The pane's rest before it iterates, long enough that a loaded
    machine's pumps never reach it: each test iterates when it says."""
    from visualdynamics.gui import main_window

    monkeypatch.setattr(main_window, 'SYNTHESIS_SETTLE_MS', 600_000)


def _open(window, pump, name):
    from conftest import select_objects

    select_objects(window, pump, name)
    window.generate_signal_act()
    pump()
    return window.data_pane.signal_panel


def _shown(panel):
    return {k for k, e in panel._editors.items() if e.isVisibleTo(panel)}


def _counting(window, monkeypatch):
    """The window's worker, counted: what ran on it, by status."""
    real = window._run_long
    seen = []

    def counted(status, verb, *args, **kwargs):
        seen.append(status)
        return real(status, verb, *args, **kwargs)

    monkeypatch.setattr(window, '_run_long', counted)
    return seen


def test_the_pane_iterates_the_synthesis_on_the_worker_once(window, pump,
                                                            monkeypatch, held_back):
    from conftest import bar_acts, select_objects

    window.add_object('Requirement', _spec())
    pump()
    select_objects(window, pump, 'Requirement')
    assert 'Generate Signal' in bar_acts(window.data_pane)
    seen = _counting(window, monkeypatch)
    panel = _open(window, pump, 'Requirement')
    assert [panel.kind_box.itemData(i)
            for i in range(panel.kind_box.count())] == ['srs']
    assert {'rate', 'half_cycles', 'stroke', 'velocity', 'force',
            'moving_mass', 'body_mass'} <= _shown(panel)
    assert not {'duration', 'seed', 'peak', 'frame'} & _shown(panel)
    assert panel.settings().sample_rate == 18432.0, '10 x 1796 Hz, rounded up'
    assert 'when the settings rest' in panel.note.text()
    assert window._synthesis_settle.isActive(), 'not iterated on a redraw'
    assert not seen
    window._signal_synthesize()
    pump()
    assert seen == ['Synthesizing a shock to Requirement…']
    assert panel._derived['cycles'][1].text() == '5'
    assert panel._derived['inside'][1].text() == '32/32'
    assert panel._derived['event'][1].text().endswith(' ms')
    assert '32/32 points inside the band' in panel.note.text()
    message = window.statusBar().currentMessage()
    assert ('1Z+: SRS synthesis, 32 wavelets of 5 half cycles' in message
            and 'no ratings given' in message)
    import pyqtgraph as pg

    graphics = window.data_pane.graphics
    plots = [row for row in range(10)
             if isinstance(graphics.getItem(row, 0), pg.PlotItem)]
    assert plots == [0, 1, 2, 8],         'the stroke reading three rows, the spectrum under its legend'
    assert graphics.getItem(8, 0).getViewBox().state['logMode'] == [True,
                                                                    True]
    line = next(e for e in window.project.journal
                if 'plot_shock_synthesis' in e)
    assert "project['Requirement'], sample_rate=18432.0" in line
    window.render_current()
    pump()
    window._signal_synthesize()
    assert seen == ['Synthesizing a shock to Requirement…'], \
        'held for the same settings: drawn again, not iterated again'


def test_a_rating_the_shaker_cannot_meet_is_said_in_the_pane(window, pump,
                                                             monkeypatch, held_back):
    window.add_object('Requirement', _spec())
    pump()
    panel = _open(window, pump, 'Requirement')
    assert panel._editors['stroke'].suffix() == ' in'
    panel._editors['stroke'].setValue(0.01)
    pump()
    assert panel.settings().stroke == pytest.approx(0.000254)
    assert window._synthesis_settle.isActive()
    window._signal_synthesize()
    pump()
    assert 'the stroke runs out' in panel.note.text()
    assert ('requirement from 50 to' in panel.note.text()
            or 'requirement at 50 Hz' in panel.note.text())
    assert 'the stroke runs out' in window.statusBar().currentMessage()
    assert panel._derived['cycles'][1].text() == '—'


def test_the_ratings_open_on_the_stroke_readings_shaker(window, pump,
                                                       held_back):
    window.data_pane.stroke_settings = {'stroke': 0.0508, 'velocity': 1.78,
                                        'force': None, 'moving_mass': 45.0,
                                        'body_mass': None}
    window.add_object('Requirement', _spec())
    pump()
    panel = _open(window, pump, 'Requirement')
    settings = panel.settings()
    assert settings.stroke == pytest.approx(0.0508)
    assert settings.velocity == 1.78 and settings.moving_mass == 45.0,         'kept as held, not read back through the rounding of the boxes'
    assert settings.force is None and settings.body_mass is None


def test_generate_makes_the_synthesis_on_the_worker_and_journals(
        window, pump, monkeypatch, held_back):
    window.add_object('Requirement', _spec())
    pump()
    seen = _counting(window, monkeypatch)
    panel = _open(window, pump, 'Requirement')
    panel._editors['rate'].setValue(RATE)
    pump()
    panel.apply_button.click()
    pump()
    assert seen == ['Synthesizing a shock to Requirement…']
    made = [n for n in window.objects if n.startswith('Requirement SRS')]
    assert made, 'the synthesis was made'
    record = window.project.provenance[made[0]]
    assert record['params']['kind'] == 'srs'
    assert record['params']['sample_rate'] == RATE
    line = next(e for e in window.project.journal if 'generate_signal' in e)
    assert "'srs'" in line and 'sample_rate=10240.0' in line
    assert 'progress' not in line
    room = {'visualdynamics': visualdynamics,
            'project': visualdynamics.Project()}
    room['project'].add('Requirement', _spec())
    exec(line, room)                                     # noqa: S102
    assert np.array_equal(room['project'][made[0]].ordinate,
                          window.objects[made[0]].ordinate)


def test_a_changed_requirement_is_not_drawn_from_the_one_held(window, pump,
                                                              held_back):
    window.add_object('Requirement', _spec())
    pump()
    panel = _open(window, pump, 'Requirement')
    window._signal_synthesize()
    pump()
    assert panel._derived['inside'][1].text() == '32/32'
    window.objects['Requirement'].ordinate *= 2.0
    window.render_current()
    pump()
    assert 'when the settings rest' in panel.note.text()
    assert window._synthesis_settle.isActive()
    assert panel._derived['inside'][1].text() == '—'
