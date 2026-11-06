"""Schemes: one Python module per scheme.

A scheme file holds a docstring, a KNOBS table and top-level functions:

    KNOBS = {"name": (default, low, high, "int" | "log" | "linear"), ...}

    def build(vectors, ctx): ...             offline, once per dataset
    def layout(index, vectors, ctx): ...     offline: what goes where in HBF, HBM and SRAM
    def serve(state, events, ctx): ...       online, timed: called at every wake
    def helper(...): ...                     any other top-level function, called by name

The functions call each other by name in one shared namespace (torch, F and math are provided). Knobs are read only as
ctx.knobs["name"] with a literal name, and a knob that is read must be declared. `build` and `layout` results are cached on
the functions each one reaches and the knobs those read, so a knob that only `serve` reads never rebuilds an index.
See contract.py for the interface of the three entry functions.
"""
import ast
import copy
import hashlib
import math
import re

from .policy import BUILTINS, RESERVED, Rejected, check_function

ENTRIES = {"build": 2, "layout": 3, "serve": 3}  # name -> number of positional arguments
LIMITS = {"functions": 60, "lines": 80, "chars": 160_000}
NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")
KNOB = re.compile(r"[a-z][a-z0-9_]{0,39}")
SCALES = ("log", "linear", "int")


def _strip(fn):
    """The function without docstrings: reformatting, comments and docstrings never change its identity."""
    fn = copy.deepcopy(fn)
    for node in ast.walk(fn):
        body = getattr(node, "body", None)
        if isinstance(node, ast.FunctionDef) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return fn


def function_hash(fn):
    return hashlib.sha256(ast.dump(_strip(fn), annotate_fields=True, include_attributes=False).encode()).hexdigest()


def knob_references(fn):
    """Names read as <expr>.knobs["name"]; any other use of .knobs is rejected."""
    used, parents = set(), {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "knobs":
            parent = parents.get(id(node))
            if not (isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.slice, ast.Constant)
                    and isinstance(parent.slice.value, str) and isinstance(parent.ctx, ast.Load)):
                raise Rejected(fn.name + ': knobs are read only as ctx.knobs["name"] with a literal name')
            used.add(parent.slice.value)
    return used


def _knob(name, spec):
    if not isinstance(name, str) or not KNOB.fullmatch(name):
        raise Rejected("Invalid knob name " + repr(name)[:60])
    if not isinstance(spec, tuple) or len(spec) != 4 or spec[3] not in SCALES:
        raise Rejected("Knob " + name + ": (default, low, high, scale) with scale log|linear|int")
    default, low, high, scale = spec
    values = (low, default, high)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or not low <= default <= high or low == high:
        raise Rejected("Knob " + name + ": need finite low <= default <= high with low < high")
    if scale == "log" and low <= 0:
        raise Rejected("Knob " + name + ": a log-scale knob needs low > 0")
    if scale == "int" and any(float(v) != int(v) for v in values):
        raise Rejected("Knob " + name + ": an int knob needs integer bounds and default")
    return {"default": default, "low": low, "high": high, "scale": scale}


def reachable(calls, entries):
    seen, stack = set(), sorted(entries)
    while stack:
        name = stack.pop()
        if name not in seen:
            seen.add(name)
            stack.extend(sorted(calls[name] - seen))
    return seen


class Scheme:
    """A parsed and checked scheme: functions, knobs, the call graph and the knobs each function reads."""

    def __init__(self, source, name="scheme"):
        self.name, self.source = name, source
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise Rejected(name + ": syntax error: " + str(exc)[:200]) from None
        self.doc = ast.get_docstring(tree) or ""
        self.functions, table = {}, None
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id == "KNOBS" and table is None:
                try:
                    table = ast.literal_eval(node.value)
                except ValueError:
                    raise Rejected("KNOBS must be a literal dict of (default, low, high, scale)") from None
            elif isinstance(node, ast.FunctionDef):
                if not NAME.fullmatch(node.name) or node.name in RESERVED:
                    raise Rejected("Invalid function name " + repr(node.name)[:60])
                if node.name in self.functions:
                    raise Rejected("Function " + node.name + " is defined twice")
                if node.end_lineno - node.lineno + 1 > LIMITS["lines"]:
                    raise Rejected(node.name + ": longer than " + str(LIMITS["lines"]) + " lines; split it")
                self.functions[node.name] = node
            else:
                raise Rejected("Only a docstring, KNOBS and top-level functions are allowed in a scheme file")
        if not isinstance(table, dict):
            raise Rejected("A scheme file declares KNOBS = {...}")
        if len(self.functions) > LIMITS["functions"] or len(source) > LIMITS["chars"]:
            raise Rejected("Scheme too large")
        self.knobs = {n: _knob(n, s) for n, s in table.items()}
        for entry, arity in ENTRIES.items():
            fn = self.functions.get(entry)
            if fn is None:
                raise Rejected("Missing entry function " + entry)
            a = fn.args
            if len(a.args) != arity or a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg or a.defaults:
                raise Rejected("Entry " + entry + " takes exactly " + str(arity) + " positional arguments")
        names = set(self.functions)
        self.calls = {n: check_function(n, fn, names) for n, fn in self.functions.items()}
        self.refs = {n: knob_references(fn) for n, fn in self.functions.items()}
        self.live = reachable(self.calls, ENTRIES)
        used = set().union(*(self.refs[n] for n in self.live))
        if used - set(self.knobs):
            raise Rejected("Undeclared knobs: " + ", ".join(sorted(used - set(self.knobs))))

    def namespace(self):
        """The functions, executed in a space with torch, F and math and the allowed builtins only."""
        import builtins
        import torch
        import torch.nn.functional as F
        allowed = {name: getattr(builtins, name) for name in BUILTINS}
        allowed["__import__"] = builtins.__import__  # torch's C code imports through the calling frame's builtins; schemes cannot name it
        space = {"__builtins__": allowed, "torch": torch, "F": F, "math": math}
        exec(compile(self.source, "<scheme %s>" % self.name, "exec"), space)
        return space

    def knob_values(self, overrides=None):
        """Declared knobs at their defaults or the given values, clamped to their range."""
        values = {}
        for name, spec in self.knobs.items():
            v = overrides[name] if overrides and name in overrides else spec["default"]
            v = min(max(v, spec["low"]), spec["high"])
            values[name] = int(round(v)) if spec["scale"] == "int" else float(v)
        return values

    def stage_keys(self, knobs):
        """What build and layout depend on: the functions each reaches and the knobs those read."""
        out = {}
        for entry in ("build", "layout"):
            fns = reachable(self.calls, {entry: 0})
            used = set().union(*(self.refs[f] for f in fns))
            out[entry] = {"units": {f: function_hash(self.functions[f]) for f in sorted(fns)},
                          "knobs": {k: knobs[k] for k in sorted(used)}}
        return out


def load(path):
    with open(path) as f:
        return Scheme(f.read(), name=str(path).rsplit("/", 1)[-1].removesuffix(".py"))
