"""Composing a figure of one's own from a script.

`compose`, `as_series`, `figure_plots`, `to_view` and `pin` are the
public way to draw what the package has no single view for: several
readings on one axis, marks and words over them, panels side by side,
written to a file with no window. Each test pins the thing a figure
script depends on: the size written, the theme's ground, the plot found
by its quantity, the log axis's decades, the range that opens.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from visualdynamics.core.data import Psd, TimeHistory, frequency_axis
from visualdynamics.plot import (
    as_series,
    build_plots,
    compose,
    figure_plots,
    pin,
    plot_series,
    to_view,
)
from visualdynamics.units import UnitError

G = 9.80665


def history():
    """Two records of different quantities: they draw on two plots."""
    t = np.arange(1000) / 1000.0
    return TimeHistory(t, np.vstack([np.sin(2 * np.pi * 5 * t),
                                     0.01 * np.cos(2 * np.pi * 5 * t)]),
                       response_dof=['1X+', '2X+'],
                       ordinate_dim=['acceleration', 'length'],
                       ordinate_unit=['m/s**2', 'm'])


def psd(level=0.01):
    """A flat ASD of `level` g^2/Hz from 20 to 2000 Hz: a log ordinate."""
    f = np.linspace(20.0, 2000.0, 100)
    return Psd(f, np.full((1, f.size), level * G ** 2), ['1Z+'],
               ordinate_dim=['acceleration**2/frequency'])


def drawn(build, **kwargs):
    """Run `build` through `compose` to a throwaway file and hand back
    what it saw, for the tests that read the layout rather than the
    picture."""
    seen = {}

    def keep(layout):
        seen['result'] = build(layout)
        seen['layout'] = layout

    compose(keep, show=False, **kwargs)
    return seen['result']


# ---- compose: the frame every standalone plot uses ----------------------


def test_a_composed_figure_is_written_at_its_size(qt_app, tmp_path):
    from PySide6.QtGui import QImage

    calls = []
    path = compose(lambda layout: calls.append(layout) or build_plots(
        layout, as_series([psd()])), path=tmp_path / 'a.png',
        size=(640, 360), show=False)
    assert path == str(tmp_path / 'a.png')
    image = QImage(path)
    assert (image.width(), image.height()) == (640, 360)
    # built once, into the layout widget the builders take
    assert len(calls) == 1 and hasattr(calls[0], 'ci')


@pytest.mark.parametrize('theme, ground', [('light', 0xFFFFFFFF),
                                           ('dark', 0xFF000000)])
def test_the_figure_wears_the_theme_it_is_given(qt_app, tmp_path, theme,
                                                ground):
    """The frame's ground follows `theme` under a plot drawn by hand;
    the builders are told the theme themselves, as in the app."""
    from PySide6.QtGui import QImage

    by_hand = compose(lambda layout: layout.addPlot(row=0, col=0),
                      path=tmp_path / f'{theme}-hand.png', theme=theme,
                      size=(400, 300), show=False)
    built = compose(lambda layout: build_plots(layout, as_series([psd()]),
                                               theme=theme),
                    path=tmp_path / f'{theme}.png', theme=theme,
                    size=(400, 300), show=False)
    # the corner is outside every axis: the ground itself
    assert QImage(by_hand).pixel(1, 1) == ground
    assert QImage(built).pixel(1, 1) == ground


def test_without_a_path_the_figure_is_the_apps_pane(qt_app):
    from visualdynamics.gui.panes import DataPane

    pane = compose(lambda layout: build_plots(layout, as_series([psd()])),
                   show=False, title='Composed')
    try:
        assert isinstance(pane, DataPane)
        assert pane.windowTitle() == 'Composed'
        assert len(figure_plots(pane.graphics)) == 1
    finally:
        pane.close()
        pane.deleteLater()


def test_a_figure_that_cannot_be_made_is_refused_before_it_is_drawn(
        qt_app, tmp_path):
    calls = []

    def build(layout):
        calls.append(layout)

    with pytest.raises(TypeError, match='build is a function'):
        compose('not a function', path=tmp_path / 'a.png')
    with pytest.raises(ValueError, match='two positive numbers'):
        compose(build, path=tmp_path / 'a.png', size=(0, 300))
    with pytest.raises(ValueError, match=r'\(width, height\)'):
        compose(build, path=tmp_path / 'a.png', size=500)
    with pytest.raises(ValueError, match="has .pdf"):
        compose(build, path=tmp_path / 'a.pdf')
    with pytest.raises(ValueError, match='has no extension'):
        compose(build, path=tmp_path / 'figure')
    assert calls == []
    assert not any(tmp_path.iterdir())


def test_another_image_type_qt_writes_is_still_written(qt_app, tmp_path):
    """The check refuses what nothing can write, not what is not a
    .png: a .jpg was written before and still is."""
    path = compose(lambda layout: build_plots(layout, as_series([psd()])),
                   path=tmp_path / 'a.jpg', size=(300, 200), show=False)
    assert os.path.getsize(path) > 1000


def test_the_frequency_axis_holds_for_the_figure_and_is_put_back(
        qt_app, tmp_path):
    before = frequency_axis()
    seen = []

    def build(layout):
        build_plots(layout, as_series([psd()]))
        seen.append(frequency_axis())
        seen.append(figure_plots(layout)[0].getViewBox().state['logMode'][0])

    compose(build, path=tmp_path / 'a.png', size=(300, 200),
            frequency_axis='log')
    assert seen == ['log', True]
    assert frequency_axis() == before


def test_the_frequency_axis_is_put_back_when_the_build_fails(qt_app,
                                                             tmp_path):
    before = frequency_axis()

    def build(layout):
        raise RuntimeError('the script broke')

    with pytest.raises(RuntimeError, match='the script broke'):
        compose(build, path=tmp_path / 'a.png', frequency_axis='log')
    assert frequency_axis() == before


def test_a_frequency_axis_that_cannot_hold_is_refused(qt_app, tmp_path):
    before = frequency_axis()
    calls = []
    with pytest.raises(ValueError, match='applies to a figure written'):
        compose(calls.append, frequency_axis='log', show=False)
    with pytest.raises(ValueError):
        compose(calls.append, path=tmp_path / 'a.png',
                frequency_axis='sideways')
    assert calls == []
    assert frequency_axis() == before


# ---- as_series: the items, as the builders take them --------------------


def test_items_are_read_in_any_mixture():
    th, spectrum = history(), psd()
    assert as_series([th, ('Spec', spectrum), ('Run', th, [1])]) == [
        (None, th, None), ('Spec', spectrum, None), ('Run', th, [1])]


def test_records_are_plain_indices_whatever_they_came_as():
    th = history()
    (_, _, records), = as_series([(None, th, np.array([1, 0]))])
    assert records == [1, 0] and all(type(i) is int for i in records)
    assert as_series([(None, th, np.int64(1))])[0][2] == [1]


@pytest.mark.parametrize('items, error, words', [
    ([], ValueError, 'nothing to draw'),
    ([('one',)], ValueError, 'tuple of 1'),
    ([('a', None, None, None)], ValueError, 'tuple of 4'),
    ([np.zeros(10)], TypeError, 'item 0 is a ndarray, not a data array'),
    ([psd(), ('x', 3.0)], TypeError, 'item 1 is a float'),
])
def test_items_that_are_not_data_are_refused(items, error, words):
    with pytest.raises(error, match=words):
        as_series(items)


def test_a_name_must_be_words():
    with pytest.raises(TypeError, match='named by a int'):
        as_series([(3, psd())])


@pytest.mark.parametrize('records, error, words', [
    ([], ValueError, 'picks no records'),
    ([2], ValueError, r'has 2 record\(s\), so record\(s\) \[2\]'),
    ([-1], ValueError, r'record\(s\) \[-1\] are not in it'),
    ([0.5], TypeError, 'not indices'),
])
def test_records_that_are_not_in_the_array_are_refused(records, error,
                                                       words):
    with pytest.raises(error, match=words):
        as_series([('Run', history(), records)])


def test_plot_series_reads_its_items_the_same_way(qt_app, tmp_path):
    with pytest.raises(ValueError, match='nothing to draw'):
        plot_series([], path=tmp_path / 'a.png', show=False)
    with pytest.raises(ValueError, match=r'record\(s\) \[5\]'):
        plot_series([('Run', history(), [5])], path=tmp_path / 'a.png',
                    show=False)


# ---- figure_plots: the plot a mark goes on ------------------------------


def test_the_plots_are_found_in_reading_order(qt_app):
    plots = drawn(lambda layout: (build_plots(layout, as_series([history()])),
                                  figure_plots(layout))[1])
    assert [p.series_key[1] for p in plots] == ['acceleration', 'length']


def test_a_plot_is_found_by_the_quantity_it_draws(qt_app):
    def build(layout):
        build_plots(layout, as_series([history(), psd()]))
        return (figure_plots(layout, 'length'),
                figure_plots(layout, 'acceleration**2 / frequency'),
                figure_plots(layout))

    length, density, every = drawn(build)
    assert len(every) == 3
    assert length == [every[1]]
    # written another way, the same dimension
    assert density == [every[2]]


def test_a_quantity_not_drawn_names_what_is(qt_app):
    def build(layout):
        build_plots(layout, as_series([history()]))
        with pytest.raises(ValueError, match='no plot in this figure draws '
                                             'force; it draws acceleration, '
                                             'length'):
            figure_plots(layout, 'force')
        with pytest.raises(UnitError, match='displacment'):
            figure_plots(layout, 'displacment')

    drawn(build)


def test_a_plot_drawn_by_hand_is_listed_but_has_no_quantity(qt_app):
    def build(layout):
        mine = layout.addPlot(row=0, col=0)
        assert figure_plots(layout) == [mine]
        with pytest.raises(ValueError, match='holds no plot drawn from data'):
            figure_plots(layout, 'acceleration')

    drawn(build)


def test_panels_side_by_side_are_read_where_they_sit(qt_app):
    def build(layout):
        right = layout.addLayout(row=0, col=1)
        left = layout.addLayout(row=0, col=0)
        build_plots(right, as_series([psd()]))
        build_plots(left, as_series([history()]))
        return figure_plots(layout), figure_plots(right)

    every, right = drawn(build)
    # added right first, read left first
    assert [p.series_key[1] for p in every] == [
        'acceleration', 'length', 'acceleration**2/frequency']
    assert right == [every[2]]


def test_a_figure_of_panels_with_legends_closes_cleanly(tmp_path):
    """Its own interpreter, because the failure was a crash: a legend in
    a nested panel, laid out once more while the figure closed, mapped
    through a layout already deleted and took the process down."""
    script = textwrap.dedent(f"""
        import numpy as np
        from visualdynamics.core.data import Psd, TimeHistory
        from visualdynamics.plot import as_series, build_plots, compose

        t = np.arange(500) / 500.0
        th = TimeHistory(t, np.vstack([np.sin(9 * t), np.cos(9 * t)]),
                         response_dof=['1X+', '2X+'],
                         ordinate_dim=['acceleration', 'length'])
        f = np.linspace(20.0, 2000.0, 50)
        asd = Psd(f, np.ones((1, 50)), ['1Z+'],
                  ordinate_dim=['acceleration**2/frequency'])

        def build(layout):
            build_plots(layout.addLayout(row=0, col=0), as_series([th]))
            build_plots(layout.addLayout(row=0, col=1), as_series([asd]))

        print(compose(build, path={str(tmp_path / 'panels.png')!r},
                      size=(900, 400)))
    """)
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    done = subprocess.run([sys.executable, '-c', script], env=env, check=False,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / 'panels.png').stat().st_size > 4000


# ---- to_view: data units into the plot's own coordinates ----------------


def test_a_log_axis_takes_decades_and_a_linear_one_the_value(qt_app):
    def build(layout):
        build_plots(layout, as_series([psd()]))
        plot = figure_plots(layout)[0]
        return (to_view(plot, x=100.0, y=0.01),
                to_view(plot, y=[1e-3, 1e-1]), to_view(plot))

    point, ys, nothing = drawn(build)
    assert point == (100.0, pytest.approx(-2.0))
    assert ys[0] is None
    np.testing.assert_allclose(ys[1], [-3.0, -1.0])
    assert nothing == (None, None)


def test_the_abscissa_takes_decades_when_it_is_drawn_in_them(qt_app,
                                                             tmp_path):
    seen = []

    def build(layout):
        build_plots(layout, as_series([psd()]))
        seen.append(to_view(figure_plots(layout)[0], x=1000.0)[0])

    compose(build, path=tmp_path / 'a.png', size=(300, 200),
            frequency_axis='log')
    assert seen == [pytest.approx(3.0)]


def test_a_point_with_no_place_on_the_plot_is_refused(qt_app):
    def build(layout):
        build_plots(layout, as_series([psd()]))
        plot = figure_plots(layout)[0]
        with pytest.raises(ValueError, match='1 value'):
            to_view(plot, x=[100.0, np.nan])
        with pytest.raises(ValueError, match=r'y = 0 has no logarithm'):
            to_view(plot, y=0.0)
        with pytest.raises(ValueError, match=r'y = -1 has no logarithm'):
            to_view(plot, y=[0.1, -1.0])
        # a linear axis has room for a negative value
        assert to_view(plot, x=-5.0)[0] == -5.0

    drawn(build)


# ---- pin: one scale for panels meant to be compared ---------------------


def test_a_pinned_range_opens_where_it_is_told(qt_app):
    def build(layout):
        build_plots(layout, as_series([psd()]))
        plot = figure_plots(layout)[0]
        pin(plot, x=(50.0, 1000.0), y=(1e-4, 1.0))
        view = plot.getViewBox()
        return (view.viewRange(), view.state['limits'],
                view.state['autoRange'])

    (x_range, y_range), limits, auto = drawn(build)
    np.testing.assert_allclose(x_range, [50.0, 1000.0])
    np.testing.assert_allclose(y_range, [-4.0, 0.0])
    np.testing.assert_allclose(limits['xLimits'], [50.0, 1000.0])
    np.testing.assert_allclose(limits['yLimits'], [-4.0, 0.0])
    assert auto == [False, False]


def test_a_range_wider_than_the_data_is_not_clamped_back(qt_app):
    """`build_plots` locks zoom to the data's own frame; a range set
    straight on the plot was clamped inside it, so two panels meant to
    share a scale did not."""
    def build(layout):
        build_plots(layout, as_series([psd(0.01)]))
        plot = figure_plots(layout)[0]
        pin(plot, y=(1e-6, 10.0))
        return plot.getViewBox().viewRange()[1]

    np.testing.assert_allclose(drawn(build), [-6.0, 1.0])


def test_one_axis_pinned_leaves_the_other_alone(qt_app):
    def build(layout):
        build_plots(layout, as_series([psd()]))
        plot = figure_plots(layout)[0]
        before = plot.getViewBox().state['limits']['yLimits']
        pin(plot, x=(100.0, 200.0))
        return before, plot.getViewBox().state

    before, state = drawn(build)
    assert state['limits']['yLimits'] == before
    assert state['autoRange'][0] is False


@pytest.mark.parametrize('span, words', [
    ((1.0, 0.1), r'0.1 is not below|1 is not below 0.1'),
    ((0.1, 0.1), 'is not below'),
    ((0.0, 1.0), 'y = 0 has no logarithm'),
    ((np.nan, 1.0), 'not finite'),
    ((1.0,), r'\(low, high\) range'),
    (None, None),
])
def test_a_range_that_cannot_be_pinned_is_refused(qt_app, span, words):
    def build(layout):
        build_plots(layout, as_series([psd()]))
        plot = figure_plots(layout)[0]
        before = plot.getViewBox().viewRange()
        if span is None:
            pin(plot)     # nothing asked, nothing changed
        else:
            with pytest.raises(ValueError, match=words):
                pin(plot, y=span)
        assert plot.getViewBox().viewRange() == before

    drawn(build)
