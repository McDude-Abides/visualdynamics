"""Inventory every function and class in the package by walking its AST.

Writes ``bench/out/inventory.csv`` (one row per function or method) and
``bench/out/modules.csv`` (one row per module), and prints a summary by
layer. The columns are the facts that decide whether a function is
worth porting to Rust: its size, whether it loops in Python, whether it
calls numpy or scipy, and whether it is tied to Qt.
"""

import ast
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'visualdynamics'
OUT = ROOT / 'bench' / 'out'

QT_ROOTS = {'PySide6', 'pyqtgraph', 'pyvistaqt', 'shiboken6'}
NUMERIC_ROOTS = {'numpy', 'scipy'}


def layer_of(rel: pathlib.PurePath) -> str:
    """The architecture layer a module belongs to (docs/architecture.md)."""
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = rel.stem
    if stem in {'deform', 'rotate', 'decimate'}:
        return 'motion'
    return stem.strip('_') or 'package'


def import_roots(tree: ast.AST) -> set[str]:
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split('.')[0])
    return roots


def numeric_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to numpy or scipy (np, sp, signal, linalg, ...)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] in NUMERIC_ROOTS:
                    names.add((a.asname or a.name).split('.')[0])
        elif (isinstance(node, ast.ImportFrom) and node.module
              and node.module.split('.')[0] in NUMERIC_ROOTS):
            names |= {a.asname or a.name for a in node.names}
    return names


def body_facts(fn: ast.AST, aliases: set[str]) -> dict:
    loops = nested = numeric = 0
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.While, ast.comprehension)):
            loops += 1
            for inner in ast.walk(node):
                if inner is not node and isinstance(
                        inner, (ast.For, ast.While, ast.comprehension)):
                    nested += 1
                    break
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in aliases:
                numeric += 1
        elif isinstance(node, ast.Name) and node.id in aliases:
            numeric += 1
    return {'loops': loops, 'nested_loops': nested, 'numeric_refs': numeric}


def walk_defs(tree, aliases, module, layer, qt):
    rows = []

    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f'{owner}.{child.name}' if owner else child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines = child.end_lineno - child.lineno + 1
                rows.append({
                    'module': module, 'layer': layer,
                    'qualname': f'{owner}.{child.name}' if owner else child.name,
                    'kind': 'method' if owner else 'function',
                    'public': not child.name.startswith('_')
                    or child.name.startswith('__') and child.name.endswith('__'),
                    'line': child.lineno, 'lines': lines,
                    'args': len(child.args.args) + len(child.args.kwonlyargs),
                    'module_qt': qt,
                    **body_facts(child, aliases),
                })
                # Nested functions are counted inside their parent only.

    visit(tree, '')
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    functions, modules = [], []
    for path in sorted(SRC.rglob('*.py')):
        rel = path.relative_to(SRC)
        module = 'visualdynamics.' + '.'.join(rel.with_suffix('').parts)
        module = module.removesuffix('.__init__')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        roots = import_roots(tree)
        qt = bool(roots & QT_ROOTS)
        layer = layer_of(rel)
        rows = walk_defs(tree, numeric_aliases(tree), module, layer, qt)
        functions += rows
        modules.append({
            'module': module, 'layer': layer,
            'lines': len(path.read_text(encoding='utf-8').splitlines()),
            'classes': sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
            'functions': len(rows), 'qt': qt,
            'numpy': 'numpy' in roots, 'scipy': 'scipy' in roots,
            'third_party': ' '.join(sorted(
                r for r in roots
                if r not in sys.stdlib_module_names and r != 'visualdynamics')),
        })

    for name, rows in (('inventory', functions), ('modules', modules)):
        with open(OUT / f'{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    by_layer = {}
    for m in modules:
        d = by_layer.setdefault(m['layer'], [0, 0, 0, False])
        d[0] += 1
        d[1] += m['lines']
        d[2] += m['functions']
        d[3] |= m['qt']
    print(f"{'layer':<10}{'modules':>8}{'lines':>9}{'defs':>7}  qt")
    for layer, (n, lines, defs, qt) in sorted(
            by_layer.items(), key=lambda kv: -kv[1][1]):
        print(f'{layer:<10}{n:>8}{lines:>9}{defs:>7}  {"yes" if qt else ""}')
    print(f"{'total':<10}{len(modules):>8}{sum(m['lines'] for m in modules):>9}"
          f'{len(functions):>7}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
