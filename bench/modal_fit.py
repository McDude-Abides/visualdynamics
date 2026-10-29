"""Where the interactive modal fit spends its time, across FRF sizes and
mode counts.

The FRFs are synthesized from the demonstration plate's own modes
(`ShapeSet.synthesize_frf`, 2 % damping), `responses` Z channels
against two references, so the fit has true modes to find. Each
phase is timed as the app drives it: opening the session (which makes
the first suggestion), `confirm` then `suggest` for each mode, and one
`refine_residues` of all confirmed modes together.

    ./.venv/bin/python bench/modal_fit.py
    ./.venv/bin/python bench/modal_fit.py --responses 100 --modes 10 40

Writes ``bench/out/modal_fit-<commit>-<threads>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import time
import warnings

import numpy as np

from visualdynamics.core.data import Frf
from visualdynamics.core.modal_fit import ModalFitSession
from visualdynamics.demo import plate

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'bench' / 'out'
MESH = 24             # 625 nodes: enough Z channels for the largest sweep
LINES = 2048
TOP = 2500.0          # Hz, the span the FRFs cover


def frfs(responses: int, references: int = 2) -> Frf:
    model = plate.build(MESH)
    shapes = model.eigensolution(maximum_frequency=1.5 * TOP, damping=0.02)
    z = [dof for dof in shapes.coordinate if dof.endswith('Z+')]
    picked = [z[i] for i in np.linspace(0, len(z) - 1, responses).astype(int)]
    drives = [z[len(z) // 3], z[2 * len(z) // 3]][:references]
    response_dof = [r for r in picked for _ in drives]
    reference_dof = [d for _ in picked for d in drives]
    frequency = np.linspace(TOP / LINES, TOP, LINES)
    values = shapes.synthesize_frf(frequency, response_dof, reference_dof,
                                   power=2)
    return Frf(frequency, values, response_dof=response_dof,
               reference_dof=reference_dof,
               ordinate_dim='acceleration/force')


def run(data: Frf, modes: int) -> dict:
    start = time.perf_counter()
    session = ModalFitSession(data)
    opened = time.perf_counter() - start
    confirm = suggest = 0.0
    for _ in range(modes):
        t = time.perf_counter()
        session.confirm()
        confirm += time.perf_counter() - t
        t = time.perf_counter()
        session.suggest()
        suggest += time.perf_counter() - t
    t = time.perf_counter()
    session.refine_residues()
    refine = time.perf_counter() - t
    return {'open_s': opened, 'confirm_s': confirm, 'suggest_s': suggest,
            'refine_s': refine, 'total_s': opened + confirm + suggest + refine}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--responses', type=int, nargs='+',
                        default=[25, 100, 400])
    parser.add_argument('--modes', type=int, nargs='+',
                        default=[5, 10, 20, 40])
    args = parser.parse_args()
    warnings.simplefilter('ignore')
    threads = os.environ.get('OPENBLAS_NUM_THREADS', 'default')
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True,
                            check=False).stdout.strip()
    print(f'OPENBLAS_NUM_THREADS={threads}, {LINES} lines, 2 references\n')
    print(f"{'responses':>9} {'modes':>5} {'total':>8} {'open':>8} "
          f"{'confirm':>8} {'suggest':>8} {'refine':>8}")
    rows = []
    for responses in args.responses:
        data = frfs(responses)
        for modes in args.modes:
            row = {'threads': threads, 'responses': responses,
                   'records': data.num_records, 'lines': LINES,
                   'modes': modes, **run(data, modes)}
            rows.append(row)
            print(f"{responses:>9} {modes:>5} {row['total_s']:>7.2f}s "
                  + ' '.join(f"{row[k]:>8.2f}" for k in
                             ('open_s', 'confirm_s', 'suggest_s', 'refine_s')),
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'modal_fit-{commit}-{threads}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
