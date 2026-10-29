"""Where the sparse eigensolution spends its time, by stage, across
model sizes and mode counts.

The model is the demonstration plate (`demo.plate.build`) at a range
of meshes, solved sparse for the lowest `num_modes`. Each stage's time
is exclusive: a stage called from inside another (the factor's solves
inside the polish, say) is counted once, in itself.

    ./.venv/bin/python bench/eigensolve.py
    ./.venv/bin/python bench/eigensolve.py --mesh 40 60 --modes 100 400
    OPENBLAS_NUM_THREADS=4 ./.venv/bin/python bench/eigensolve.py
    ./.venv/bin/python bench/eigensolve.py --by frequency

The BLAS thread count matters more than anything here: numpy and scipy
each load their own OpenBLAS, and each starts a thread per logical CPU
unless told otherwise, so the threads of the two libraries contend for
the cores. Run it with and without `OPENBLAS_NUM_THREADS` and compare.
`--by frequency` asks for every mode up to a frequency instead, the
way the app's Solve Modes does: the frequency of the `num_modes`-th
mode, found by an untimed solve first, so the two ways are timed on the
same modes. Each Lanczos run's mode count is recorded, since the solver
grows the count and starts again until the frequency is reached.
Writes ``bench/out/eigensolve-<commit>-<threads>[-frequency].csv``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import pathlib
import subprocess
import sys
import time
import warnings

import scipy.linalg

from visualdynamics.core import fem, sparse_solve
from visualdynamics.demo import plate

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'bench' / 'out'

#: (stage, owner, attribute): each is wrapped with a timer for the run
STAGES = [
    ('assembly', fem.Model, 'sparse_matrices'),
    ('scaling', fem.Model, 'scaled_system'),
    ('factor', sparse_solve, 'factor_symmetric'),
    ('solves', sparse_solve.SuperLUFactor, 'solve'),
    # a Mac's factor is Accelerate's; its solves are the same stage
    ('solves', sparse_solve.AccelerateFactor, 'solve'),
    ('lanczos', fem, 'block_lanczos_modes'),
    ('orthonormalize', fem, '_m_orthonormal'),
    ('ritz eigh', scipy.linalg, 'eigh'),
    ('polish', fem, 'polish_modes'),
    ('rigid check', fem.Model, '_strains_no_element'),
]


class Timers:
    """Exclusive time per stage: a stage's clock pauses while a stage
    it calls is running."""

    def __init__(self):
        self.spent = dict.fromkeys(
            dict.fromkeys(name for name, _owner, _attr in STAGES), 0.0)
        self.calls = dict.fromkeys(self.spent, 0)
        self.stack: list[list] = []
        #: the mode count each Lanczos run was asked for
        self.asked: list[int] = []

    def wrap(self, name, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            if name == 'lanczos':
                self.asked.append(int(args[3]))
            now = time.perf_counter()
            if self.stack:
                outer = self.stack[-1]
                self.spent[outer[0]] += now - outer[1]
            self.stack.append([name, now])
            try:
                return function(*args, **kwargs)
            finally:
                done = time.perf_counter()
                self.spent[name] += done - self.stack.pop()[1]
                self.calls[name] += 1
                if self.stack:
                    self.stack[-1][1] = done
        return timed


def run(mesh: int, modes: int, by: str = 'count') -> dict:
    model = plate.build(mesh)
    if by == 'frequency':
        top = float(model.eigensolution(num_modes=modes,
                                        solver='sparse').frequency[-1])
        ask = {'maximum_frequency': top}
    else:
        ask = {'num_modes': modes}
    timers = Timers()
    saved = []
    for name, owner, attr in STAGES:
        original = owner.__dict__[attr] if isinstance(owner, type) \
            else getattr(owner, attr)
        saved.append((owner, attr, original))
        setattr(owner, attr, timers.wrap(name, original))
    # every stage is looked up by name when it is called (a module
    # global, or an import inside the solver), so patching the owner's
    # attribute reaches it
    try:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            shapes = model.eigensolution(solver='sparse', **ask)
        total = time.perf_counter() - start
    finally:
        for owner, attr, original in saved:
            setattr(owner, attr, original)
    accounted = sum(timers.spent.values())
    return {
        'mesh': mesh, 'dof': model.num_dof, 'by': by, 'modes': modes,
        'found': len(shapes.frequency),
        'top_hz': float(shapes.frequency[-1]), 'total_s': total,
        **{f'{k}_s': v for k, v in timers.spent.items()},
        'other_s': total - accounted,
        'solve_calls': timers.calls['solves'],
        'eigh_calls': timers.calls['ritz eigh'],
        'fell_back': any('ARPACK' in str(w.message) for w in caught),
        'lanczos_asked': ' '.join(map(str, timers.asked)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--mesh', type=int, nargs='+',
                        default=[20, 30, 40, 60, 80, 98],
                        help='plate elements per side (at most 98)')
    parser.add_argument('--modes', type=int, nargs='+',
                        default=[50, 100, 200, 400])
    parser.add_argument('--by', choices=('count', 'frequency'),
                        default='count',
                        help='ask for num_modes, or for every mode up to '
                        'the frequency of that many')
    args = parser.parse_args()

    # OpenBLAS's setting, or Accelerate's on a Mac, so runs of one
    # machine at several thread counts write separate files
    threads = os.environ.get('OPENBLAS_NUM_THREADS') or (
        f"veclib{os.environ['VECLIB_MAXIMUM_THREADS']}"
        if os.environ.get('VECLIB_MAXIMUM_THREADS') else 'default')
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True,
                            check=False).stdout.strip()
    print(f'BLAS threads {threads}, {os.cpu_count()} logical CPUs, '
          f'asking by {args.by}\n')
    names = list(dict.fromkeys(name for name, _owner, _attr in STAGES)) + ['other']
    print(f"{'dof':>7} {'modes':>5} {'total':>8}  "
          + ' '.join(f'{n[:9]:>9}' for n in names))
    rows = []
    for mesh in args.mesh:
        for modes in args.modes:
            row = run(mesh, modes, args.by)
            rows.append({'threads': threads, **row})
            print(f"{row['dof']:>7} {modes:>5} {row['total_s']:>7.2f}s  "
                  + ' '.join(f"{row[f'{n}_s']:>9.2f}" for n in names)
                  + ('  (ARPACK)' if row['fell_back'] else '')
                  + (f"  runs {row['lanczos_asked']}" if args.by == 'frequency'
                     else ''), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f'eigensolve-{commit}-{threads}'
                  + ('-frequency' if args.by == 'frequency' else '') + '.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
