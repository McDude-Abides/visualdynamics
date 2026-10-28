"""Where the sparse eigensolution spends its time, by stage, across
model sizes and mode counts.

The model is the demonstration plate (`demo.plate.build`) at a range
of meshes, solved sparse for the lowest `num_modes`. Each stage's time
is exclusive: a stage called from inside another (the factor's solves
inside the polish, say) is counted once, in itself.

    ./.venv/bin/python bench/eigensolve.py
    ./.venv/bin/python bench/eigensolve.py --mesh 40 60 --modes 100 400
    OPENBLAS_NUM_THREADS=4 ./.venv/bin/python bench/eigensolve.py

The BLAS thread count matters more than anything here: numpy and scipy
each load their own OpenBLAS, and each starts a thread per logical CPU
unless told otherwise, so the threads of the two libraries contend for
the cores. Run it with and without `OPENBLAS_NUM_THREADS` and compare.
Writes ``bench/out/eigensolve-<commit>-<threads>.csv``.
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
        self.spent = {name: 0.0 for name, _owner, _attr in STAGES}
        self.calls = dict.fromkeys(self.spent, 0)
        self.stack: list[list] = []

    def wrap(self, name, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
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


def run(mesh: int, modes: int) -> dict:
    model = plate.build(mesh)
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
            shapes = model.eigensolution(num_modes=modes, solver='sparse')
        total = time.perf_counter() - start
    finally:
        for owner, attr, original in saved:
            setattr(owner, attr, original)
    accounted = sum(timers.spent.values())
    return {
        'mesh': mesh, 'dof': model.num_dof, 'modes': modes,
        'top_hz': float(shapes.frequency[-1]), 'total_s': total,
        **{f'{k}_s': v for k, v in timers.spent.items()},
        'other_s': total - accounted,
        'solve_calls': timers.calls['solves'],
        'eigh_calls': timers.calls['ritz eigh'],
        'fell_back': any('ARPACK' in str(w.message) for w in caught),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--mesh', type=int, nargs='+',
                        default=[20, 30, 40, 60, 80, 98],
                        help='plate elements per side (at most 98)')
    parser.add_argument('--modes', type=int, nargs='+',
                        default=[50, 100, 200, 400])
    args = parser.parse_args()

    threads = os.environ.get('OPENBLAS_NUM_THREADS', 'default')
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True,
                            check=False).stdout.strip()
    print(f'OPENBLAS_NUM_THREADS={threads}, {os.cpu_count()} logical CPUs\n')
    names = [name for name, _owner, _attr in STAGES] + ['other']
    print(f"{'dof':>7} {'modes':>5} {'total':>8}  "
          + ' '.join(f'{n[:9]:>9}' for n in names))
    rows = []
    for mesh in args.mesh:
        for modes in args.modes:
            row = run(mesh, modes)
            rows.append({'threads': threads, **row})
            print(f"{row['dof']:>7} {modes:>5} {row['total_s']:>7.2f}s  "
                  + ' '.join(f"{row[f'{n}_s']:>9.2f}" for n in names)
                  + ('  (ARPACK)' if row['fell_back'] else ''), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'eigensolve-{commit}-{threads}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
