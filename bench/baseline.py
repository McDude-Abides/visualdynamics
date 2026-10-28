"""Baseline timings of the numeric kernels, swept over input size.

Each case builds its input at a given size outside the timed region,
then times one call repeatedly. The minimum is reported as the
kernel's cost (the least disturbed run) with the median beside it, and
a log-log fit across the sizes gives the empirical scaling exponent:
about 1 for a linear kernel, about 2 for one that grows with the square
of its input. These are the numbers a Rust port is measured against.

    ./.venv/bin/python bench/baseline.py                 # every case
    ./.venv/bin/python bench/baseline.py srs psd         # names containing these
    ./.venv/bin/python bench/baseline.py --quick         # the smaller sizes only
    ./.venv/bin/python bench/baseline.py --large         # and larger ones (`LARGE`)

Writes ``bench/out/baseline-<commit>[-large][-t<threads>].csv`` and
``.json``. Set ``OPENBLAS_NUM_THREADS`` to compare thread counts: numpy
and scipy each load their own OpenBLAS with a thread per logical CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from visualdynamics.core import kurtosis, octave, shapes, srs, time_statistics, wavelet
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import Filtering, filtered
from visualdynamics.core.resample import resample
from visualdynamics.core.shapes import ShapeSet

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'bench' / 'out'
RATE = 4096.0

#: a case stops growing once one call takes longer than this, so the
#: sweep finds where a kernel becomes slow without waiting on it
SLOW_CALL = 20.0
#: each size is repeated until this much time has been spent on it, or
#: REPEATS calls, whichever comes first (and always at least twice)
BUDGET = 1.0
REPEATS = 15


@dataclass
class Case:
    name: str
    target: str            # the function being measured
    axis: str              # what `size` counts
    sizes: list[int]
    setup: Callable[[int], Callable[[], object]]
    fixed: str = ''        # what is held constant while size varies


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def history(channels: int, samples: int, *, drives: int = 0,
            averaging: bool = True) -> TimeHistory:
    """A random record: `channels` accelerations and `drives` forces,
    with responses mixed from the forces so FRFs have something to
    estimate."""
    rng = _rng()
    forces = rng.normal(size=(drives, samples))
    responses = rng.normal(scale=0.05, size=(channels, samples))
    if drives:
        responses += rng.normal(size=(channels, drives)) @ forces
    values = np.vstack([responses, forces])
    dofs = ([f'{100 + i}Z+' for i in range(channels)]
            + [f'{900 + k}Z+' for k in range(drives)])
    dims = ['acceleration'] * channels + ['force'] * drives
    th = TimeHistory(np.arange(samples) / RATE, values,
                     response_dof=dofs, ordinate_dim=dims)
    if averaging:
        frame = 1024
        frames = (samples - frame) // (frame // 2) + 1
        th.averaging = Averaging(frame_length=frame, overlap=0.5,
                                 window='hann', frames=frames)
    return th


def shape_set(modes: int, dofs: int) -> ShapeSet:
    rng = _rng()
    return ShapeSet(frequency=np.linspace(10.0, 1000.0, modes),
                    damping=np.full(modes, 0.02),
                    coordinate=[f'{1 + i // 3}{"XYZ"[i % 3]}+'
                                for i in range(dofs)],
                    shape_matrix=rng.normal(size=(modes, dofs)))


SRS_FREQUENCIES = srs.octave_frequencies(5.0, 2000.0)   # 1/12 octave


def _srs_samples(n):
    x = _rng().normal(size=n)
    return lambda: srs.peaks(x, SRS_FREQUENCIES, RATE)


def _srs_frequencies(n):
    x = _rng().normal(size=16384)
    f = np.geomspace(5.0, 2000.0, n)
    return lambda: srs.peaks(x, f, RATE)


def _psd_samples(n):
    th = history(8, n)
    return th.compute_psds


def _psd_channels(n):
    th = history(n, 1 << 18)
    return th.compute_psds


def _cpsd_channels(n):
    th = history(n, 1 << 16)
    return th.compute_cpsds


def _frf(method):
    def setup(n):
        th = history(n, 1 << 16, drives=2)
        return lambda: th.compute_frfs(method=method)
    return setup


def _frf_samples(n):
    th = history(8, n, drives=2)
    return lambda: th.compute_frfs(method='Hv')


def _filter_samples(n):
    th = history(8, n, averaging=False)
    return lambda: filtered(th, Filtering(high=400.0))


def _resample_samples(n):
    th = history(8, n, averaging=False)
    return lambda: resample(th, RATE / 4)


def _scalogram_samples(n):
    x = _rng().normal(size=n)
    f = wavelet.log_frequencies(5.0, 1000.0)
    return lambda: wavelet.scalogram(x, RATE, f)


def _scalogram_peaks_samples(n):
    x = _rng().normal(size=n)
    f = wavelet.log_frequencies(5.0, 1000.0)
    return lambda: wavelet.scalogram_peaks(x, RATE, f)


def _kurtosis_frames(n):
    frames = _rng().normal(size=(n, 1024))
    return lambda: kurtosis.frame_kurtosis(frames)


def _time_statistics_samples(n):
    x = _rng().normal(size=n)
    return lambda: time_statistics.time_statistics(x, RATE)


def _mac_modes(n):
    phi = _rng().normal(size=(n, 3000))
    return lambda: shapes.mac_matrix(phi)


def _mac_dofs(n):
    phi = _rng().normal(size=(50, n))
    return lambda: shapes.mac_matrix(phi)


def _synth_pairs(n):
    s = shape_set(40, 300)
    dofs = s.coordinate
    resp = [dofs[i % len(dofs)] for i in range(n)]
    ref = [dofs[0]] * n
    f = np.linspace(1.0, 1200.0, 2048)
    return lambda: s.synthesize_frf(f, resp, ref)


def _octave_lines(n):
    f = np.linspace(0.0, RATE / 2, n)
    v = _rng().random(n)
    bounds = octave.edges(5.0, RATE / 2 * 0.99)
    return lambda: octave.resample(f, v, bounds)


POW2 = [1 << k for k in range(12, 24, 2)]          # 4 Ki .. 4 Mi samples

CASES = [
    Case('srs.samples', 'core.srs.peaks', 'samples',
         [1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000],
         _srs_samples, f'{len(SRS_FREQUENCIES)} frequencies'),
    Case('srs.frequencies', 'core.srs.peaks', 'frequencies',
         [10, 30, 100, 300, 1000, 3000], _srs_frequencies, '16384 samples'),
    Case('psd.samples', 'TimeHistory.compute_psds', 'samples', POW2,
         _psd_samples, '8 channels, 1024 frame, 50% overlap'),
    Case('psd.channels', 'TimeHistory.compute_psds', 'channels',
         [1, 4, 16, 64, 128], _psd_channels, '262144 samples'),
    Case('cpsd.channels', 'TimeHistory.compute_cpsds', 'channels',
         [2, 4, 8, 16, 32, 64], _cpsd_channels, '65536 samples'),
    Case('frf.Hv.responses', 'TimeHistory.compute_frfs(Hv)', 'responses',
         [2, 4, 8, 16, 32, 64], _frf('Hv'), '2 references, 65536 samples'),
    Case('frf.H1.responses', 'TimeHistory.compute_frfs(H1)', 'responses',
         [2, 4, 8, 16, 32, 64], _frf('H1'), '2 references, 65536 samples'),
    Case('frf.Hv.samples', 'TimeHistory.compute_frfs(Hv)', 'samples', POW2,
         _frf_samples, '8 responses, 2 references'),
    Case('filter.samples', 'core.filters.filtered', 'samples', POW2,
         _filter_samples, '8 channels, low-pass 400 Hz order 4'),
    Case('resample.samples', 'core.resample.resample', 'samples', POW2,
         _resample_samples, '8 channels, rate / 4'),
    Case('scalogram.samples', 'core.wavelet.scalogram', 'samples',
         POW2[:5], _scalogram_samples, '5-1000 Hz'),
    Case('scalogram_peaks.samples', 'core.wavelet.scalogram_peaks',
         'samples', POW2, _scalogram_peaks_samples, '5-1000 Hz'),
    Case('kurtosis.frames', 'core.kurtosis.frame_kurtosis', 'frames',
         [16, 64, 256, 1024, 4096, 16384], _kurtosis_frames, '1024 per frame'),
    Case('time_statistics.samples', 'core.time_statistics.time_statistics',
         'samples', POW2, _time_statistics_samples, 'defaults'),
    Case('mac.modes', 'core.shapes.mac_matrix', 'modes',
         [10, 30, 100, 300, 1000], _mac_modes, '3000 DOFs'),
    Case('mac.dofs', 'core.shapes.mac_matrix', 'dofs',
         [100, 1000, 10_000, 100_000], _mac_dofs, '50 modes'),
    Case('synthesize_frf.pairs', 'ShapeSet.synthesize_frf', 'pairs',
         [1, 10, 100, 1000], _synth_pairs, '40 modes, 2048 lines'),
    Case('octave.lines', 'core.octave.resample', 'lines',
         [1_000, 10_000, 100_000, 1_000_000], _octave_lines,
         f'1/{octave.PER_OCTAVE} octave bands'),
]


#: the sizes `--large` adds past each sweep, to check that the trend
#: holds; kept within a few gigabytes per case. The full scalogram is
#: left out: at 4 Mi samples its result alone is about 6 GB.
LARGE = {
    'srs.samples': [4_000_000, 16_000_000],
    'srs.frequencies': [10_000],
    'psd.samples': [1 << 24],
    'psd.channels': [256],
    'cpsd.channels': [128],
    'frf.Hv.responses': [128, 256],
    'frf.H1.responses': [128, 256],
    'frf.Hv.samples': [1 << 24],
    'filter.samples': [1 << 24],
    'resample.samples': [1 << 24],
    'scalogram_peaks.samples': [1 << 24],
    'kurtosis.frames': [65_536],
    'time_statistics.samples': [1 << 24],
    'mac.modes': [3_000],
    'mac.dofs': [1_000_000],
    'synthesize_frf.pairs': [10_000],
    'octave.lines': [10_000_000],
}
#: with `--large` a case keeps growing until one call takes this long
LARGE_SLOW_CALL = 150.0


def measure(call: Callable[[], object]) -> list[float]:
    # one untimed call first: the first call of a kernel pays for lazy
    # imports, FFT plans and thread pools that every later call reuses
    call()
    times: list[float] = []
    spent = 0.0
    while len(times) < 2 or (spent < BUDGET and len(times) < REPEATS):
        start = time.perf_counter()
        call()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        spent += elapsed
        if elapsed > SLOW_CALL:
            break
    return times


def slope(sizes: list[int], seconds: list[float]) -> float | None:
    """The scaling exponent from the larger half of the sizes, where
    the fixed overhead of a call no longer hides how it grows."""
    if len(sizes) < 3:
        return None
    keep = slice(len(sizes) // 2 - (len(sizes) % 2 == 0), None)
    x, y = np.log(sizes[keep]), np.log(seconds[keep])
    return float(np.polyfit(x, y, 1)[0])


def environment() -> dict:
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    import scipy

    import visualdynamics
    cpu = platform.processor()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        'commit': git('rev-parse', '--short', 'HEAD'),
        'dirty': bool(git('status', '--porcelain', '--', 'src')),
        'visualdynamics': visualdynamics.__version__,
        'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'cpu': cpu, 'cores': os.cpu_count(),
        'platform': platform.platform(),
        'openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS', 'default'),
        'when': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('names', nargs='*',
                        help='run only cases whose name contains one of these')
    parser.add_argument('--quick', action='store_true',
                        help='the smaller half of each sweep')
    parser.add_argument('--large', action='store_true',
                        help='extend each sweep with the sizes in LARGE')
    args = parser.parse_args()
    global SLOW_CALL
    if args.large:
        SLOW_CALL = LARGE_SLOW_CALL
    warnings.simplefilter('ignore')

    env = environment()
    print(f"visualdynamics {env['visualdynamics']} @ {env['commit']}"
          f"{' (dirty)' if env['dirty'] else ''}, python {env['python']}, "
          f"numpy {env['numpy']}, scipy {env['scipy']}\n{env['cpu']}, "
          f"{env['cores']} threads\n")

    rows, summary = [], []
    for case in CASES:
        if args.names and not any(n in case.name for n in args.names):
            continue
        sizes = case.sizes[:max(2, len(case.sizes) // 2 + 1)] if args.quick \
            else case.sizes
        if args.large:
            sizes = sizes + LARGE.get(case.name, [])
        print(f'{case.name}  ({case.target}; {case.fixed})')
        done, best = [], []
        for size in sizes:
            call = case.setup(size)
            times = measure(call)
            lo, med = min(times), statistics.median(times)
            done.append(size)
            best.append(lo)
            rows.append({'case': case.name, 'target': case.target,
                         'axis': case.axis, 'size': size, 'fixed': case.fixed,
                         'calls': len(times), 'min_s': lo, 'median_s': med,
                         'per_unit_ns': lo / size * 1e9})
            print(f'  {case.axis:>12} {size:>10,}  min {lo * 1e3:>11.3f} ms'
                  f'  median {med * 1e3:>11.3f} ms  ({len(times)} calls)',
                  flush=True)
            if lo > SLOW_CALL:
                print('  stopped growing: one call is over '
                      f'{SLOW_CALL:.0f} s')
                break
        k = slope(done, best)
        summary.append({'case': case.name, 'target': case.target,
                        'axis': case.axis, 'fixed': case.fixed,
                        'exponent': k, 'largest': done[-1],
                        'largest_s': best[-1]})
        if k is not None:
            print(f'  scaling ~ {case.axis}^{k:.2f}')
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    threads = os.environ.get('OPENBLAS_NUM_THREADS')
    stem = OUT / (f"baseline-{env['commit']}" + ('-large' if args.large else '')
                  + (f'-t{threads}' if threads else ''))
    with open(f'{stem}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f'{stem}.json', 'w') as f:
        json.dump({'environment': env, 'summary': summary, 'rows': rows},
                  f, indent=1)
    print(f'wrote {stem}.csv and .json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
