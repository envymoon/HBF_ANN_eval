"""Static code policy for scheme functions (AST only; no scheme code runs here).

Defence in depth, not a security boundary: run schemes you do not trust in a container. The policy keeps a scheme inside a
small, deterministic, side-effect-free language, so that two runs with one seed give one result and a scheme cannot reach
the evaluator's state:

- no import / global / nonlocal / class / try / raise / async / yield / decorators / match; no names or attributes starting
  with "_";
- builtins only from BUILTINS; host-injected modules only torch, F (torch.nn.functional) and math, always used as the base
  of an attribute chain that resolves (on the real modules) to a function, dtype, number or an allowed context-manager
  class, never to a module or other object;
- no attribute assignment or deletion (no hidden state on function objects); default arguments must be immutable constants;
- randomness only through ctx.uniform: global-RNG functions and tensor methods are denied, as are file IO, uninitialised
  memory, global settings and escape hatches (storage, numpy, hooks, str.format).
"""
import ast
import math


class Rejected(ValueError):
    """The scheme breaks the code policy or the interface."""


BUILTINS = ("abs", "min", "max", "range", "len", "float", "int", "bool", "enumerate", "zip", "sum", "any", "all",
            "isinstance", "tuple", "list", "dict", "round", "sorted", "reversed", "pow", "map", "filter")
MODULE_NAMES = ("torch", "F", "math")
SUBMODULES = ("linalg", "special", "fft", "autograd")
ALLOWED_CLASSES = ("no_grad", "enable_grad")
DENY = {
    # file IO, serialisation, compilation, devices, distributed, global state
    "load", "save", "from_file", "hub", "ops", "utils", "jit", "compile", "cuda", "backends", "distributed",
    "multiprocessing", "library", "classes", "package", "onnx", "profiler", "fx", "export", "overrides",
    "testing", "nn", "optim", "distributions", "random", "Generator", "default_generator", "use_deterministic_algorithms",
    "compiler", "mps", "xpu", "cpu", "futures", "monitor", "quantization", "sparse", "storage", "untyped_storage",
    # global random number generation (randomness must come from ctx.uniform)
    "manual_seed", "seed", "initial_seed", "get_rng_state", "rand", "randn", "randint", "randperm", "rand_like",
    "randn_like", "randint_like", "normal", "bernoulli", "multinomial", "poisson", "uniform_", "normal_",
    "bernoulli_", "random_", "exponential_", "geometric_", "cauchy_", "log_normal_", "dropout", "dropout1d",
    "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout", "rrelu", "rrelu_", "gumbel_softmax",
    "fractional_max_pool2d", "fractional_max_pool3d", "binomial", "standard_gamma", "sample_dirichlet",
    # uninitialised memory, raw memory, hooks, interop, formatting escapes
    "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "resize_", "resize_as_",
    "data_ptr", "share_memory_", "pin_memory", "numpy", "ctypes", "register_hook", "register_prehook",
    "register_post_accumulate_grad_hook", "apply_", "map_", "map2_", "format", "format_map",
    "to_dlpack", "from_dlpack", "frombuffer", "asarray", "from_numpy", "cpp",
}
# torch.<name> must be a tensor operation (also a Tensor method) or a creation/combination function.
CREATION = ("zeros", "ones", "full", "arange", "linspace", "logspace", "eye", "tensor", "as_tensor", "zeros_like",
            "ones_like", "full_like", "stack", "cat", "concat", "hstack", "vstack", "dstack", "einsum", "cdist",
            "meshgrid", "broadcast_tensors", "broadcast_shapes", "cartesian_prod", "block_diag", "tril_indices",
            "triu_indices", "no_grad", "enable_grad", "is_tensor", "numel", "result_type", "promote_types",
            "float32", "float64", "float16", "bfloat16", "int64", "int32", "int16", "int8", "uint8", "bool", "long",
            "float", "double", "int", "inf", "nan", "pi", "e", "clamp", "vmap", "searchsorted", "bucketize",
            "atleast_1d", "atleast_2d", "atleast_3d", "column_stack", "row_stack", "tensordot", "chain_matmul",
            "scatter", "scatter_add", "index_select", "gather", "where", "lerp", "logcumsumexp", "cummax", "cummin",
            "unique_consecutive", "complex", "polar", "kron", "diag_embed", "rot90", "roll", "flip", "flatten")
_TORCH_ALLOWED = None


def torch_allowed():
    global _TORCH_ALLOWED
    if _TORCH_ALLOWED is None:
        import torch
        methods = {n for n in dir(torch.Tensor) if not n.startswith("_")}
        _TORCH_ALLOWED = ((methods & set(dir(torch))) | set(CREATION) | set(SUBMODULES)) - DENY
    return _TORCH_ALLOWED
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                   ast.Await, ast.Yield, ast.YieldFrom, ast.Try, ast.Raise, ast.AsyncFor, ast.AsyncWith, ast.Match,
                   ast.Delete)
if hasattr(ast, "TryStar"): FORBIDDEN_NODES += (ast.TryStar,)
RESERVED = set(BUILTINS) | set(MODULE_NAMES) | {"True", "False", "None"}

_MODULES = None


def modules():
    global _MODULES
    if _MODULES is None:
        import torch
        import torch.nn.functional as F
        _MODULES = {"torch": torch, "F": F, "math": math}
    return _MODULES


def _immutable_constant(node):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, str, bool, type(None)))
    if isinstance(node, ast.Tuple):
        return all(_immutable_constant(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _immutable_constant(node.operand)
    return False


def _check_chain(name_node, parents):
    """name_node is a Name in MODULE_NAMES; resolve its attribute chain on the real modules."""
    import torch
    obj, node, path = modules()[name_node.id], name_node, [name_node.id]
    while isinstance(obj, type(math)):
        parent = parents.get(id(node))
        if not isinstance(parent, ast.Attribute) or parent.value is not node:
            raise Rejected("Module " + ".".join(path) + " may only be used as torch.<function>, F.<function> or math.<function>")
        if not isinstance(parent.ctx, ast.Load):
            raise Rejected("Assigning to " + ".".join(path + [parent.attr]) + " is not allowed")
        attr = parent.attr
        if attr in DENY or attr.startswith("set_") or (obj is torch and attr not in torch_allowed()):
            raise Rejected("Denied: " + ".".join(path + [attr]))
        if isinstance(getattr(obj, attr, None), type(math)):
            if not (obj is torch and attr in SUBMODULES):
                raise Rejected("Module " + ".".join(path + [attr]) + " is not available")
        elif not hasattr(obj, attr):
            raise Rejected("Unknown " + ".".join(path + [attr]))
        obj, node = getattr(obj, attr), parent
        path.append(attr)
    if isinstance(obj, type):
        if not (path[0] == "torch" and len(path) == 2 and path[1] in ALLOWED_CLASSES):
            raise Rejected("Classes are not available: " + ".".join(path))
    elif not (callable(obj) or isinstance(obj, (int, float, torch.dtype))):
        raise Rejected("Not a function, dtype or number: " + ".".join(path))
    if path[:2] == ["torch", "autograd"] and path[2:] not in (["grad"],):
        raise Rejected("Only torch.autograd.grad is available")


def bound_names(fn):
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef) and node is not fn:
            names.add(node.name)
    return names


def check_function(name, fn, function_names):
    """Policy check of one parsed top-level function; returns the names of the scheme functions it calls."""
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    local = bound_names(fn)
    clash = local & (RESERVED | set(function_names))
    if clash:
        raise Rejected(name + ": must not rebind reserved names or other function names: " + ", ".join(sorted(clash)))
    calls = set()
    for node in ast.walk(fn):
        if isinstance(node, FORBIDDEN_NODES):
            raise Rejected(name + ": " + type(node).__name__ + " is not allowed")
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            if isinstance(node, ast.FunctionDef) and node.decorator_list:
                raise Rejected(name + ": decorators are not allowed")
            args = node.args
            if any(not _immutable_constant(d) for d in args.defaults + [d for d in args.kw_defaults if d is not None]):
                raise Rejected(name + ": default arguments must be immutable constants")
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
                raise Rejected(name + ": names must not start with an underscore")
        if isinstance(node, ast.arg) and node.arg.startswith("_") and node.arg != "_":
            raise Rejected(name + ": names must not start with an underscore")
        if isinstance(node, ast.keyword) and node.arg and node.arg.startswith("_"):
            raise Rejected(name + ": keywords must not start with an underscore")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise Rejected(name + ": attributes must not start with an underscore")
            if node.attr in DENY or node.attr.startswith("set_"):
                raise Rejected(name + ": ." + node.attr + " is denied")
            if not isinstance(node.ctx, ast.Load):
                raise Rejected(name + ": attribute assignment is not allowed")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") and node.id != "_":
                raise Rejected(name + ": names must not start with an underscore")
            if isinstance(node.ctx, ast.Load):
                if node.id in MODULE_NAMES:
                    _check_chain(node, parents)
                elif node.id in function_names:
                    calls.add(node.id)
                elif node.id not in BUILTINS and node.id not in local:
                    raise Rejected(name + ": undefined name " + node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > 2000:
            raise Rejected(name + ": constant too long")
    return calls
