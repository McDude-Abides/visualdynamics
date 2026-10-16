"""The concept figures in the guide, drawn by the package itself.

Each figure is a small constructed object rendered through the same
plot calls the application uses, so the pictures cannot drift from
what the code actually draws. Regenerate after anything touching the
PSD reading rules:

    QT_QPA_PLATFORM=offscreen ./.venv/bin/python tools/make_concept_figures.py
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'docs' / 'guide' / 'images'


def psd_reading():
    """Three readings of 'the area is the picture': steps, bars, law."""
    import numpy as np

    from visualdynamics.core.data import Psd, Specification
    from visualdynamics.units import SI

    # ten coarse lines whose values sum to 10: the bins are the
    # midpoints, the edges land at 5 and 105 Hz, and the area is
    # exactly 100 (m/s^2)^2 -> 10 m/s^2 RMS, a number worth stating
    values = np.array([[0.5, 1.5, 0.8, 1.7, 1.2,
                        1.0, 0.9, 1.3, 0.6, 0.5]])
    narrowband = Psd(np.arange(10.0, 101.0, 10.0),
                     values, response_dof=['101Z+'],
                     ordinate_dim='acceleration**2/frequency',
                     ordinate_unit='m/s**2')
    narrowband.plot(path=str(OUT / 'psd-reading-steps.png'),
                    unit_system=SI, show=False)
    print(f'narrowband RMS {np.sqrt(narrowband.area()):.3f}')

    banded = narrowband.to_octave(3)
    banded.plot(path=str(OUT / 'psd-reading-octave.png'),
                unit_system=SI, show=False)
    print(f'octave RMS     {np.sqrt(banded.area()):.3f}')

    specification = Specification(
        np.array([20.0, 80.0, 350.0, 2000.0]),
        np.array([[0.01, 0.04, 0.04, 0.007]]),
        response_dof=['101Z+'],
        ordinate_dim='acceleration**2/frequency',
        ordinate_unit='m/s**2')
    specification.plot(path=str(OUT / 'psd-reading-law.png'),
                       unit_system=SI, show=False)
    print(f'specification RMS {np.sqrt(specification.area()):.3f}')


def _breakpoints():
    """The four-breakpoint specification every compliance figure judges
    against: a decade of flat top with power-law skirts, warning at
    ±3 dB and abort at ±6 dB."""
    import numpy as np

    from visualdynamics.core.data import Specification

    f = np.array([20.0, 80.0, 800.0, 2000.0])
    target = np.array([[0.01, 0.04, 0.04, 0.007]])
    limits = {name: target * 10 ** (db / 10)
              for name, db in (('warning_lower', -3), ('warning_upper', 3),
                               ('abort_lower', -6), ('abort_upper', 6))}
    return Specification(f, target, response_dof=['101Z+'],
                         ordinate_dim='acceleration**2/frequency',
                         ordinate_unit='m/s**2', **limits)


def _lines(specification, df=2.0, high=2000.0):
    """The same requirement as a controller writes it: a density per
    line, on lines `df` apart, NaN beyond `high` — stored to Nyquist
    and written only where it was controlled."""
    import numpy as np

    from visualdynamics.core.compliance import log_interpolate
    from visualdynamics.core.data import Specification

    f = np.arange(0.0, 2560.0 + df / 2, df)
    rows = {}
    for name, values in (('target', specification.ordinate[0]),
                         *specification.limits.items()):
        written = log_interpolate(f, specification.abscissa, np.real(values[0])
                                  if name != 'target' else np.real(values))
        written[f > high] = np.nan
        rows[name] = np.atleast_2d(written)
    out = Specification(f, rows.pop('target'), response_dof=['101Z+'],
                        ordinate_dim='acceleration**2/frequency',
                        ordinate_unit='m/s**2', **rows)
    out.interpolation = Specification.reading_of(f)
    return out


def _response(specification, df=1.0, high=2560.0, seed=7):
    """A measurement of the target with two departures: a resonance
    over the abort limit around 300 Hz and a notch under it around
    1200 Hz, on `df` lines to `high` — past the requirement's end."""
    import numpy as np

    from visualdynamics.core.compliance import log_interpolate
    from visualdynamics.core.data import Psd

    f = np.arange(0.0, high + df / 2, df)
    target = log_interpolate(f, specification.abscissa,
                             np.real(specification.ordinate[0]))
    rng = np.random.default_rng(seed)
    shape = np.exp(0.15 * rng.standard_normal(f.size))
    # wide enough to be out on a third-octave band too, not only on
    # the lines under it: a requirement on bands is on their power
    shape *= 1.0 + 6.0 * np.exp(-((f - 300.0) / 60.0) ** 2)
    shape *= 1.0 - 0.85 * np.exp(-((f - 1200.0) / 120.0) ** 2)
    values = np.where(np.isfinite(target), target, 0.002) * shape
    out = Psd(f, np.atleast_2d(values), response_dof=['101Z+'],
              ordinate_dim='acceleration**2/frequency',
              ordinate_unit='m/s**2')
    out.scale_db = 0
    return out


def compliance():
    """One figure per comparison there is, plus the two edge cases —
    each judged by the one cell rule (`compliance.judge`) and drawn
    by `plot_comparison`."""
    from visualdynamics.core.compliance import compare
    from visualdynamics.plot import plot_comparison
    from visualdynamics.units import SI

    breakpoints = _breakpoints()
    lines = _lines(breakpoints)
    octave_spec = lines.to_octave(3)
    narrow = _response(breakpoints)
    octave = narrow.to_octave(3)
    # the three comparisons there are (Brandon, 2026-09-19): bands
    # compare only with the same bands
    cases = {
        'lines-vs-breakpoints': (narrow, breakpoints),
        'lines-vs-lines': (narrow, lines),
        'octave-vs-octave': (octave, octave_spec),
    }
    for name, (measured, specification) in cases.items():
        plot_comparison(measured, specification, unit_system=SI, show=False,
                        path=str(OUT / f'compliance-{name}.png'))
        got = compare(specification, measured, scale_db=0)
        print(f'{name:>24}: {got["lines"]} cells, {got["abort_percent"]:.1f}% '
              f'of the band outside abort, RMS {got["difference_db"]:+.2f} dB')
    # the end of a requirement inside a band: the lines end at 1450 Hz
    # and the third-octave band holding 1450 reaches to 1778, whole.
    # The controller held the narrowband response to the requirement
    # and drove nothing past it, so the response falls away there
    # too — the case that happens (Brandon, 2026-09-20)
    import numpy as np

    short = _lines(breakpoints, high=1450.0).to_octave(3)
    controlled = _response(breakpoints, seed=7)
    beyond = np.asarray(controlled.abscissa) > 1450.0
    controlled.ordinate[:, beyond] *= 0.02
    plot_comparison(controlled.to_octave(3), short, unit_system=SI, show=False,
                    path=str(OUT / 'compliance-edge.png'))
    got = compare(short, controlled.to_octave(3), scale_db=0)
    print(f'{"edge":>24}: {got["lines"]} cells, judged to {got["band"][1]:.0f} Hz, '
          f'the whole last band; {got["abort_lines"]} band out')
    # a hole: a controller's notch, written as zero across 400-500 Hz

    notched = _lines(breakpoints)
    hole = (notched.abscissa >= 400.0) & (notched.abscissa <= 500.0)
    notched.ordinate[:, hole] = 0.0
    for values in notched.limits.values():
        values[:, hole] = 0.0
    plot_comparison(narrow, notched, unit_system=SI, show=False,
                    path=str(OUT / 'compliance-hole.png'))
    got = compare(notched, narrow, scale_db=0)
    print(f'{"hole":>24}: {got["lines"]} cells, nothing judged in the notch')


def sine_tracking():
    """One swept tone read the ways a controller reads it: a clipped
    drive's third harmonic through each detector and band, and a step
    in level through bands of different widths."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone
    from visualdynamics.core.sine_tracking import SineTracking, track_sine
    from visualdynamics.plot import plot_sine_tracking
    from visualdynamics.units import SI

    rate = 16384.0
    dof = '101Z+'

    def record(tone, envelope, harmonic=0.0, onset=0.0, tail=0.0):
        t = np.arange(int((onset + tone.duration() + tail) * rate)) / rate
        local = np.clip(t - onset, 0.0, None)
        inside = (t >= onset) & (local <= tone.duration())
        phase = tone.phase_at(local)
        wave = np.cos(phase) + harmonic * np.cos(3.0 * phase)
        x = np.where(inside, envelope(local) * wave, 0.0)
        return TimeHistory(t, x[None], response_dof=[dof],
                           ordinate_dim=['acceleration'])

    # a log sweep, 20 to 320 Hz in four seconds, at 10 m/s**2 with a
    # third harmonic of 30 % in phase: the clipped drive's signature
    tone = SineTone('Sine Tone 1', 0.5, [20.0, 320.0], [[10.0], [10.0]],
                    [LOG], [60.0])
    spec = SineSweepSpecification([tone], [dof], ordinate_unit='m/s**2')
    clipped = record(tone, lambda _t: 10.0, harmonic=0.3, onset=0.5,
                     tail=0.25)
    settings = [SineTracking(detector='peak'), SineTracking(detector='rms'),
                SineTracking(proportional=0.5),
                SineTracking(proportional=0.1)]
    plot_sine_tracking(clipped, spec, settings, unit_system=SI, show=False,
                       size=(1000, 600),
                       path=str(OUT / 'sine-tracking-harmonic.png'))
    for setting, level in zip(settings, track_sine(clipped, spec, settings)):
        print(f'{setting.describe():>34}: median '
              f'{np.median(np.abs(level.ordinate[0])):.3f} m/s^2')

    # a linear sweep at 100 Hz/s whose level doubles at 250 Hz: the
    # requirement steps over one hertz, the record at the same instant
    stepped = SineTone('Sine Tone 1', 0.0, [50.0, 250.0, 251.0, 450.0],
                       [[10.0], [10.0], [20.0], [20.0]], [0, 0, 0],
                       [100.0, 100.0, 100.0])
    spec = SineSweepSpecification([stepped], [dof], ordinate_unit='m/s**2')
    doubled = record(stepped, lambda t: np.where(t < 2.0, 10.0, 20.0))
    settings = [SineTracking(detector='peak'),
                SineTracking(proportional=0.5),
                SineTracking(proportional=0.1), SineTracking(fixed=5.0)]
    plot_sine_tracking(doubled, spec, settings, onset=0.0, lines=2000,
                       unit_system=SI, show=False, size=(1000, 600),
                       path=str(OUT / 'sine-tracking-step.png'))
    for setting, level in zip(settings, track_sine(doubled, spec, settings,
                                                   onset=0.0, lines=2000)):
        after = (level.seconds > 2.0) & (np.abs(level.ordinate[0]) >= 15.0)
        lag = level.seconds[np.flatnonzero(after)[0]] - 2.0
        print(f'{setting.describe():>34}: half way {1000 * lag:.0f} ms '
              'after the step')


def sine_tracking_band():
    """The tracking band drawn on the record it reads: the same clipped
    sweep, now under broadband noise too, through a 50 % proportional
    band and a 5 Hz fixed one, the cursor where the tone passes 100 Hz."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone
    from visualdynamics.core.sine_tracking import SineTracking
    from visualdynamics.plot import plot_tracking_filter
    from visualdynamics.units import SI

    rate = 16384.0
    dof = '101Z+'
    tone = SineTone('Sine Tone 1', 0.5, [20.0, 320.0], [[10.0], [10.0]],
                    [LOG], [60.0])
    spec = SineSweepSpecification([tone], [dof], ordinate_unit='m/s**2')
    t = np.arange(int((0.5 + tone.duration() + 0.25) * rate)) / rate
    local = np.clip(t - 0.5, 0.0, None)
    inside = (t >= 0.5) & (local <= tone.duration())
    phase = tone.phase_at(local)
    # the harmonic figure's record, with noise at a fifth of the tone
    # over the whole recording, lead-in and tail included
    x = np.where(inside, 10.0 * (np.cos(phase) + 0.3 * np.cos(3.0 * phase)),
                 0.0)
    x = x + 2.0 * np.random.default_rng(7).standard_normal(len(t))
    history = TimeHistory(t, x[None], response_dof=[dof],
                          ordinate_dim=['acceleration'])
    for setting, name in ((SineTracking(proportional=0.5), 'proportional'),
                          (SineTracking(fixed=5.0), 'fixed')):
        plot_tracking_filter(history, spec, setting, onset=0.5,
                             cursor_hz=100.0, unit_system=SI, show=False,
                             path=str(OUT / f'sine-tracking-band-{name}.png'))
        print(f'{setting.describe():>34}: drawn')


def srs_tolerance():
    """A shock spectrum judged point by point: a requirement written at
    three breakpoints, a maximax spectrum on it but for a narrow, deep
    dip, and a negative spectrum that falls away at the top. The dip
    leaves the RMS reading well inside 3 dB and fails the points rule."""
    import numpy as np

    from visualdynamics.core.compliance import srs_errors, srs_tolerance
    from visualdynamics.core.data import ShockSpecification, Srs
    from visualdynamics.plot import plot_srs_tolerance

    spec = ShockSpecification.from_breakpoints(
        [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '101Z+',
        ordinate_unit='g')
    f = spec.abscissa
    octaves = np.log2(f / 10.0)
    # a little ripple, as a synthesized pulse leaves, and a dip a third
    # of an octave wide and 5 dB deep near 400 Hz
    db = 0.6 * np.sin(2.0 * np.pi * octaves / 1.5)
    db -= 5.0 * np.exp(-0.5 * (np.log2(f / 400.0) / 0.12) ** 2)
    maximax = Srs(f, spec.ordinate * 10.0 ** (db / 20.0),
                  response_dof=['101Z+'], ordinate_dim=['acceleration'],
                  kind='maximax')
    # the negative spectrum on the requirement to the knee, then falling
    # 6 dB an octave from 400 Hz: a pulse whose other side is small
    fall = -0.4 - 6.0 * np.clip(np.log2(f / 400.0), 0.0, None)
    negative = Srs(f, spec.ordinate * 10.0 ** (fall / 20.0),
                   response_dof=['101Z+'], ordinate_dim=['acceleration'],
                   kind='negative')
    plot_srs_tolerance([maximax, negative], spec, outer=(-3.0, 6.0),
                       show=False, size=(1000, 640),
                       path=str(OUT / 'srs-tolerance.png'))
    for data in (maximax, negative):
        result = srs_tolerance(spec, data, outer=(-3.0, 6.0))
        (_dof, _event, rms_db), = srs_errors(spec, data)
        print(f'{data.kind:>9}: RMS {rms_db:+.2f} dB, '
              f'{result["inside_count"]}/{result["points"]} inside, '
              f'worst {result["worst_db"]:+.1f} dB at '
              f'{result["worst_frequency"]:.0f} Hz, passed {result["passed"]}')
def shock_check():
    """A shock record checked before its spectrum: a ringing shock with
    a 4 g zero shift from the moment it arrives, and a compensated
    half-sine whose two spectra part above the pulse."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.shock_check import check_shock
    from visualdynamics.plot import plot_shock_check
    from visualdynamics.units import IN_LBF_S

    g = 9.80665
    rate = 102400.0
    shown = IN_LBF_S.with_units(acceleration='g')
    t = np.arange(int(0.25 * rate)) / rate

    def record(values):
        return TimeHistory(t, values[None], response_dof=['101Z+'],
                           ordinate_dim=['acceleration'])

    # four decaying tones from 600 Hz to 6 kHz arriving at 20 ms, the
    # ringing a structure answers a pyrotechnic event with. Each is the
    # rate of change of a decaying sine velocity, so the motion it
    # describes starts and ends at rest, as a structure's does
    after = np.clip(t - 0.02, 0.0, None)

    def tone(f, level, decay):
        w = 2 * np.pi * f
        return level * np.exp(-after / decay) * (
            np.cos(w * after) - np.sin(w * after) / (w * decay))

    ringing = sum(tone(f, level, decay)
                  for f, level, decay in ((600.0, 120.0, 0.012),
                                          (1500.0, 200.0, 0.008),
                                          (3000.0, 160.0, 0.005),
                                          (6000.0, 110.0, 0.003)))
    ringing = np.where(t >= 0.02, ringing, 0.0) * g
    noise = 0.05 * g * np.random.default_rng(11).standard_normal(t.size)
    shift = np.where(t >= 0.02, 4.0 * g, 0.0)
    for name, values in (('good', ringing + noise),
                         ('shifted', ringing + noise + shift)):
        history = record(values)
        check, = check_shock(history)
        print(f'{name:>8}: ends at {check.end_velocity / 0.0254:7.2f} in/s, '
              f'{100 * check.end_velocity_share:5.1f} % of its peak, '
              f'{"flagged" if check.flagged else "passes"}')
        if name == 'shifted':
            plot_shock_check(history, low=10.0, high=10000.0,
                             unit_system=shown, show=False,
                             size=(1000, 900),
                             path=str(OUT / 'shock-check-shifted.png'))

    # a 50 g, 11 ms half-sine with a 2.5 g half-sine either side, each
    # carrying half its area: the shaker's compensated pulse
    duration, peak, depth = 0.011, 50.0 * g, 2.5 * g
    compensation = peak * duration / (2.0 * depth)
    opens = 0.125 - duration / 2.0

    def half_sine(start, height, length):
        inside = (t >= start) & (t <= start + length)
        return np.where(inside,
                        height * np.sin(np.pi * (t - start) / length), 0.0)

    pulse = (half_sine(opens, peak, duration)
             + half_sine(opens - compensation, -depth, compensation)
             + half_sine(opens + duration, -depth, compensation))
    history = record(pulse)
    check, = check_shock(history)
    print(f'compensated: change {check.velocity_change / 0.0254:.1f} in/s, '
          f'ends at {100 * check.end_velocity_share:.4f} % of its peak')
    plot_shock_check(history, low=10.0, high=10000.0, unit_system=shown,
                     show=False, size=(1000, 900),
                     path=str(OUT / 'shock-check-compensated.png'))
def averaging_confidence():
    """What an average is worth: a resonance under random excitation,
    averaged over sixteen Hann frames at three-quarter overlap, drawn
    with the chi-square band its equivalent degrees of freedom give."""
    import numpy as np
    from scipy.signal import iirpeak, lfilter

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.units import SI

    rate = 2048.0
    averaging = Averaging(frame_length=512, overlap=0.75, window='hann',
                          frames=16)
    noise = np.random.default_rng(11).standard_normal(averaging.span)
    # one resonance at 200 Hz over a flat floor, so the band is seen
    # riding a peak as well as a plateau
    b, a = iirpeak(200.0, 8.0, fs=rate)
    x = 0.2 * noise + lfilter(b, a, noise)
    history = TimeHistory(np.arange(averaging.span) / rate, x[None],
                          response_dof=['101Z+'],
                          ordinate_dim=['acceleration'])
    history.averaging = averaging
    psd = history.compute_psds()
    psd.plot(marks='confidence', unit_system=SI, show=False,
             path=str(OUT / 'averaging-confidence.png'))
    worth = averaging.statistical_dof()
    print(f'DOF {worth.nominal} nominal, {worth.equivalent:.1f} equivalent')


def tone_readout():
    """One composite read at 1 Hz and at 4 Hz lines: a 1 grms floor from
    5 to 500 Hz with tones of 0.12, 1.944, 2.5 and 1.5 g peak added. The
    spikes move with the spacing; the amplitudes read off the lines do
    not, and the small tone at the floor's edge is called uncertain."""
    import warnings

    import numpy as np

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.plot import plot_tone_readout

    fs, seconds, g = 2048.0, 60.0, 9.80665
    tones = ((4.86, 0.12), (19.44, 1.944), (38.88, 2.50), (58.32, 1.50))
    rng = np.random.default_rng(3304)
    n = int(fs * seconds)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum[(f < 5.0) | (f > 500.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x = x / np.sqrt(np.mean(x ** 2))
    t = np.arange(n) / fs
    for frequency, amplitude in tones:
        x = x + amplitude * np.sin(2.0 * np.pi * frequency * t + frequency)
    history = TimeHistory(t, x[None] * g, response_dof=['1Z+'],
                          ordinate_dim=['acceleration'], ordinate_unit=['g'])
    for block, name in ((2048, 'tone-readout-1hz.png'),
                         (512, 'tone-readout-4hz.png')):
        averaging = Averaging(frame_length=block, overlap=0.5,
                              window='hann').filled(n, fs)
        asd = history.compute_psds(averaging)
        asd.log_abscissa = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_tone_readout(asd, [4.86, 19.44, 58.32], window=averaging,
                              snap=False,
                              band=(2.0, 200.0), show=False,
                              size=(1000, 600), path=str(OUT / name))
            for frequency, amplitude in tones:
                reading = asd.read_tone(frequency, window=averaging,
                                        snap=False)
                state = ('uncertain' if reading.uncertain else
                         'between lines' if reading.between_lines else '')
                print(f'{asd.abscissa[1]:g} Hz lines, {frequency:5.2f} Hz '
                      f'({amplitude:.3f} g): {reading.amplitude / g:.3f} g '
                      f'peak, spike {reading.spike / g ** 2:.4f} g2/Hz, '
                      f'{reading.on_line / g ** 2:.4f} on a line, floor '
                      f'{reading.floor / g ** 2:.5f} {state}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    psd_reading()
    compliance()
    sine_tracking()
    sine_tracking_band()
    srs_tolerance()
    shock_check()
    averaging_confidence()
    tone_readout()
    print('done')


if __name__ == '__main__':
    main()
