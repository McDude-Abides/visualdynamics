"""The chi-square band as a reading: each record's band stated with the
confidence and the count it was worked out at, and the legend of the
plot that shades it saying the same in words.

The counts are the ones the PSD was stamped with when it was averaged
(`Psd.statistical_dof`), never worked out again here. Hand values are
the printed chi-square table: at 120 degrees of freedom the 90 percent
band runs from 0.9 dB below to 1.0 dB above the estimate, at 16 from
2.2 below to 3.0 above, and at 2 from 4.8 below to 12.9 above.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.averaging import (
    Averaging,
    ConfidenceBand,
    chi_square_factors,
    confidence_bands,
    dof_text,
)
from visualdynamics.core.data import Psd, TimeHistory


def _averaged(frames=10, overlap=0.5, channels=3, length=256,
              rate=1024.0, seed=1503):
    """The PSD of white noise averaged over Hann frames, carrying the
    count its averaging gives."""
    averaging = Averaging(frame_length=length, overlap=overlap,
                          window='hann', frames=frames)
    rng = np.random.default_rng(seed)
    history = TimeHistory(np.arange(averaging.span) / rate,
                          rng.standard_normal((channels, averaging.span)),
                          response_dof=[f'{i + 1}X+' for i in
                                        range(channels)],
                          ordinate_dim='acceleration')
    history.averaging = averaging
    return history.compute_psds(), averaging


# ---- the band in words ---------------------------------------------------


@pytest.mark.parametrize('dof, level, words', [
    (120.0, 0.90, '90 % band at 120 DOF: -0.9 / +1.0 dB'),
    (120.0, 0.95, '95 % band at 120 DOF: -1.0 / +1.2 dB'),
    (16.0, 0.90, '90 % band at 16 DOF: -2.2 / +3.0 dB'),
    (2.0, 0.90, '90 % band at 2 DOF: -4.8 / +12.9 dB'),
])
def test_a_band_says_its_confidence_its_count_and_its_edges(dof, level,
                                                            words):
    band, = confidence_bands(1.0, dof, levels=[level])
    assert band.describe() == words


@pytest.mark.parametrize('dof, shown', [
    (120.0, '120'), (225.9437, '225.9'), (2.0, '2'), (17.1718, '17.17'),
    (1504.6, '1505'),
])
def test_a_count_is_shown_to_four_figures(dof, shown):
    assert dof_text(dof) == shown


# ---- each record's band, from the count it was stamped with -------------


def test_each_record_is_read_at_the_count_it_carries():
    psd, averaging = _averaged()
    readings = psd.confidence_reading(0.9)
    assert len(readings) == psd.num_records
    assert all(isinstance(band, ConfidenceBand) for band in readings)
    expected = averaging.statistical_dof().equivalent
    for i, band in enumerate(readings):
        assert band.level == 0.9
        assert band.dof == pytest.approx(expected)
        low, high = chi_square_factors(expected, 0.9)
        assert band.lower_factor == pytest.approx(low)
        assert band.upper_factor == pytest.approx(high)
        assert np.allclose(band.lower, np.real(psd.ordinate[i]) * low)
        assert np.allclose(band.upper, np.real(psd.ordinate[i]) * high)


def test_the_reading_is_the_band_the_plot_draws():
    """The two views of one band: the reading's edges are the PSDs
    `confidence_band` returns, record by record."""
    psd, _averaging = _averaged()
    for level in (0.9, 0.95):
        lower, upper = psd.confidence_band(level)
        for i, band in enumerate(psd.confidence_reading(level)):
            assert np.allclose(band.lower, lower.ordinate[i])
            assert np.allclose(band.upper, upper.ordinate[i])


def test_ninety_five_is_wider_than_ninety_both_ways():
    psd, _averaging = _averaged()
    ninety = psd.confidence_reading(0.90)[0]
    ninety_five = psd.confidence_reading(0.95)[0]
    assert ninety_five.lower_db < ninety.lower_db < 0.0
    assert ninety_five.upper_db > ninety.upper_db > 0.0


def test_a_typed_count_overrides_one_for_all_or_one_per_record():
    psd, _averaging = _averaged()
    every = psd.confidence_reading(0.9, dof=120.0)
    assert [band.dof for band in every] == [120.0] * psd.num_records
    assert every[0].describe() == '90 % band at 120 DOF: -0.9 / +1.0 dB'
    each = psd.confidence_reading(0.9, dof=[2.0, 16.0, 120.0])
    assert [band.dof for band in each] == [2.0, 16.0, 120.0]
    assert each[1].lower_factor == pytest.approx(chi_square_factors(16.0)[0])


def test_an_imported_spectrum_is_read_only_at_a_count_given():
    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    assert psd.statistical_dof is None
    with pytest.raises(ValueError, match='does not say what it was '
                                         'averaged over'):
        psd.confidence_reading()
    band = psd.confidence_reading(0.95, dof=40.0)[0]
    assert band.dof == 40.0
    assert band.lower_factor == pytest.approx(
        chi_square_factors(40.0, 0.95)[0])


def test_a_cross_term_is_not_an_autospectrum():
    lines = np.linspace(10.0, 100.0, 4)
    auto = Psd(lines, np.full((1, 4), 2.0), response_dof=['1X+'],
               ordinate_dim='acceleration**2/frequency')
    assert auto.is_autospectrum()
    cross = Psd(lines, np.ones((1, 4)), response_dof=['1X+'],
                reference_dof=['2X+'],
                ordinate_dim='acceleration*acceleration/frequency')
    assert not cross.is_autospectrum()
    # one DOF named twice, two quantities: still a cross term
    drive = Psd(lines, np.full((1, 4), 2.0 + 1.0j), response_dof=['1X+'],
                reference_dof=['1X+'],
                ordinate_dim='acceleration*force/frequency')
    assert not drive.is_autospectrum()
    for refused in (cross, drive):
        with pytest.raises(ValueError, match='cross terms'):
            refused.confidence_reading(dof=12.0)


# ---- the legend says what the band is worth ------------------------------


def _legend(psd, records, **kwargs):
    import pyqtgraph as pg

    from visualdynamics.plot import build_plot, confidence_preview
    from visualdynamics.theme import theme

    layout = pg.GraphicsLayoutWidget()
    build_plot(layout, psd, records=records)
    drawn = confidence_preview(layout.ci.items, psd, theme('light'),
                               records=records, **kwargs)
    plot = drawn[0][0]
    # the layout goes back with the drawing: dropped here, it would
    # take the items with it before the caller read them
    return [label.text for _sample, label in plot.legend.items],         (layout, drawn)


def test_the_legend_states_the_confidence_and_the_count(qt_app):
    psd, averaging = _averaged()
    names, _drawn = _legend(psd, [0], levels=[0.95])
    count = dof_text(averaging.statistical_dof().equivalent)
    assert f'95 % confidence, {count} DOF' in names
    assert not any(name.startswith('90 %') for name in names), \
        'only the level asked for'


def test_records_averaged_differently_are_given_as_a_range(qt_app):
    psd, _averaging = _averaged()
    psd.statistical_dof = np.array([18.0, 120.0, 40.0])
    names, _drawn = _legend(psd, [0, 1, 2], levels=[0.9])
    assert '90 % confidence, 18 to 120 DOF' in names
    # and only the records drawn count toward it
    names, _drawn = _legend(psd, [1, 2], levels=[0.9])
    assert '90 % confidence, 40 to 120 DOF' in names


def test_a_typed_count_draws_the_band_and_is_named(qt_app):
    """An imported spectrum carries no count; given one, the band is
    drawn at it, and the legend says the count it was given."""
    import pyqtgraph as pg

    from visualdynamics.plot import step_outline
    from visualdynamics.units import DEFAULT_SYSTEM

    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    names, (_layout, drawn) = _legend(psd, [0], levels=[0.9], dof=40.0)
    assert '90 % confidence, 40 DOF' in names
    edges = [item for _plot, item in drawn
             if isinstance(item, pg.PlotDataItem)]
    low, high = chi_square_factors(40.0, 0.9)
    shown = psd.display_ordinate(DEFAULT_SYSTEM, [0])[0]
    _x, steps = step_outline(psd.abscissa, shown)
    assert (steps > 0.0).all(), 'one run, so one pair of edges'
    assert len(edges) == 2
    assert np.allclose(edges[0].getData()[1], np.log10(steps * low))
    assert np.allclose(edges[1].getData()[1], np.log10(steps * high))


def test_plot_data_takes_a_count_for_the_band_alone(qt_app, tmp_path):
    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    path = tmp_path / 'band.png'
    visualdynamics.plot.plot_data(psd, marks='confidence', levels=[0.9],
                                  dof=40.0, records=[0], path=str(path),
                                  show=False)
    assert path.stat().st_size > 2000
    with pytest.raises(ValueError, match="dof is the count of "
                                         "marks='confidence'"):
        visualdynamics.plot.plot_data(psd, marks='octave', dof=40.0,
                                      path=str(tmp_path / 'never.png'),
                                      show=False)
