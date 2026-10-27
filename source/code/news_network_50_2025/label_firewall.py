"""Auditable inference/training boundary; evaluation labels stay unopened.

The process-wide audit hook is inert outside a scoped boundary. Training label
loading uses an explicit split allowlist before handing paths to native code.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import re
import sys

_ACTIVE = ContextVar("label_access_boundary", default=None)
LABEL_KEYS = {"y", "outcome", "outcomes", "residual", "residuals", "label", "labels",
              "target_return", "future_return", "future_volatility"}


def check_payload(value, *, development=False):
    if isinstance(value, dict):
        if LABEL_KEYS & {str(k).lower() for k in value}:
            raise ValueError("Label field at inference boundary")
        if development and value.get("split") not in (None, "train", "valid"):
            raise ValueError("Only train/valid inputs allowed in training boundary")
        for item in value.values():
            check_payload(item, development=development)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_payload(item, development=development)


def _audit(event, args):
    state = _ACTIVE.get()
    if state is None:
        return
    if event in {"subprocess.Popen", "os.system", "os.exec", "os.spawn"}:
        argv = args[1] if event == "subprocess.Popen" else ()
        if (not isinstance(argv, (list, tuple)) or Path(str(args[0])).stem.lower() != 'claude' or
                not {"--restricted", "--strict-mcp-config", "--tools", "--allowedTools",
                     '--no-session-persistence', '--disable-slash-commands'} <= set(map(str, argv))):
            raise ValueError("Subprocess escape from label boundary")
        # The restricted provider has only the checked Toolkit capability. Each
        # tool invocation installs its own boundary in the MCP Python process.
    if event != "open" or not args or not isinstance(args[0], (str, bytes, Path)):
        return
    name = str(args[0]).replace("\\", "/").lower()
    base = name.rsplit("/", 1)[-1]
    label_file = (not base.endswith(('.py', '.pyc', '.pyo', '.pyd')) and
                  bool(re.search(r"(?:label|outcome|residual|target)(?:s)?(?:[_.-]|$)", base)))
    if label_file:
        permitted = state["training"] and base in {"labels_train.parquet", "labels_valid.parquet"}
        state["accesses"].append(dict(file=base, allowed=permitted))
        if not permitted:
            raise ValueError("Label file blocked by runtime firewall: " + base)


sys.addaudithook(_audit)


@contextmanager
def boundary(*, training=False):
    parent = _ACTIVE.get()
    state = dict(training=training, accesses=parent["accesses"] if parent else [])
    if parent is not None and training and not parent["training"]:
        raise ValueError("Cannot widen inference boundary to training")
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


def current_accesses():
    state = _ACTIVE.get()
    return list(state["accesses"]) if state else []


def in_boundary():
    return _ACTIVE.get() is not None


def development_labels(root, split):
    if split not in ("train", "valid"):
        raise ValueError("Evaluation labels are sealed")
    state = _ACTIVE.get()
    if state is not None and not state["training"]:
        raise ValueError("No label loader in inference boundary")
    import pandas as pd
    path = Path(root) / ("labels_" + split + ".parquet")
    if path.is_symlink():
        raise ValueError("Label loader refuses symlink aliases")
    if state is not None:
        state["accesses"].append(dict(file=path.name, allowed=True))
    return pd.read_parquet(path)


def guarded_case(function):
    @wraps(function)
    def wrapped(self, case, *args, **kwargs):
        if not self.enable_graph_router:
            return function(self, case, *args, **kwargs)
        check_payload(case)
        with boundary():
            return function(self, case, *args, **kwargs)
    return wrapped


def static_verify(root=None):
    """Check the bounded router/retrieval sources, not the final evaluator.

    Ban executable label subscripts/get/attributes and dynamic code in these
    modules. Runtime checks cover dynamic filenames and nested input payloads.
    This is a scoped access guarantee, not a proof about arbitrary plugins.
    """
    root = Path(root or Path(__file__).parent)
    names = ("graph_projection.py", "graph_router.py", "graph_skills.py", "domain_signals.py",
             "runner_receipts.py", "toolkit.py", "workflow.py", "context_selection.py")
    failures, checked = [], []
    for name in names:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        if name == 'context_selection.py':
            tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != 'compile_context_program']
        checked.append(name)
        for node in ast.walk(tree):
            key = None
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                key = node.slice.value
            if isinstance(node, ast.Attribute):
                key = node.attr
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "pop"} and node.args and isinstance(node.args[0], ast.Constant):
                    key = node.args[0].value
                if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
                    failures.append(f"{name}:{node.lineno}: dynamic code")
            if isinstance(key, str) and key.lower() in LABEL_KEYS:
                failures.append(f"{name}:{node.lineno}: label access {key}")
    if failures:
        raise ValueError("Static label firewall failed: " + "; ".join(failures))
    # The optimizer's sole label loader is a literal train/valid iteration;
    # context search rejects evaluation rows before inspecting their labels.
    optimizer = ast.parse((root / 'optimize.py').read_text(encoding='utf-8'))
    for node in ast.walk(optimizer):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'read_parquet':
            raise ValueError('Optimizer bypasses the development label loader')
    loaders = [n for n in ast.walk(optimizer) if isinstance(n, ast.ListComp) and
               isinstance(n.elt, ast.Call) and isinstance(n.elt.func, ast.Name) and n.elt.func.id == 'development_labels']
    if len(loaders) != 1 or ast.literal_eval(loaders[0].generators[0].iter) != ('train', 'valid'):
        raise ValueError('Optimizer must load literal train/valid splits only')
    context = ast.parse((root / 'context_selection.py').read_text(encoding='utf-8'))
    compile_fn = next(n for n in context.body if isinstance(n, ast.FunctionDef) and n.name == 'compile_context_program')
    first = compile_fn.body[1]  # docstring, then split guard, before any label inspection
    expected = ast.parse("not rows or any(row.get('split') not in {'train', 'valid'} for row in rows)", mode='eval').body
    if not isinstance(first, ast.If) or ast.dump(first.test) != ast.dump(expected) or not isinstance(first.body[0], ast.Raise):
        raise ValueError('Context optimizer must reject evaluation rows before reading labels')
    execution = ast.parse((root / 'run.py').read_text(encoding='utf-8'))
    execute_fn = next(n for n in execution.body if isinstance(n, ast.FunctionDef) and n.name == '_execute')
    evaluation = [n for n in ast.walk(execute_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == 'matched' and any(isinstance(a, ast.Constant) and a.value == 'test' for a in n.args)]
    seals = [n for n in ast.walk(execute_fn) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
             and n.test.id == 'enable_graph_router' and isinstance(n.body[-1], ast.Return)]
    if len(evaluation) != 1 or not any(n.lineno < evaluation[0].lineno for n in seals):
        raise ValueError('Enabled runner must return before final evaluation')
    checked += ['optimize.py:development_labels-only', 'context_selection.py:train/valid split guard (runtime tested)',
                'run.py:gate4 return before test evaluation']
    return dict(verified=True, modules=checked, scope="router, retrieval, selection packet; final evaluator excluded")
