"""GPU compute time.

Every torch operation a scheme runs inside `serve` is one kernel on the GPU and is billed here, deterministically, from the
shapes alone: launch + max(flops / F, bytes / B), the "launch overhead + roofline" form of Characterizing HBF §3.2. Views
and metadata are free; reading a value back to the host (.item(), bool(), int(), .tolist()) costs a sync. There is no
uncharged computation in `serve`, so heavy work cannot hide outside the charged interface.

On-chip billing (the default table; form B is kept for comparison): each wake's work runs as one fused kernel, so a wake pays one host round (step_us) and its kernels pay no launch of
their own (form C); a kernel's operands are the queries' working state and data already brought in through ctx (every HBF,
HBM and SRAM read is billed there), which are treated as registers and shared memory, so their bytes move at
on-chip bandwidth; a value read back to the host is a data-dependent branch inside that kernel (sync_us) plus its bytes
over the host link (host_bytes_per_us), so state cannot be pulled to the host for free Python arithmetic. The batched
brute-force bound (scoring.brute) is billed with the same table. Form B ("graph": a CUDA-graph node per kernel, 200 GB/s)
and form A ("eager") are kept for comparison runs only.

"""
import math

VERSION = "placeholder-4060-public-specs-v2-onchip-step1.2"
GRAPH = {  # form B, comparison runs only
    "step_us": 5.,  # one host round per wake: notice completions, replay the graph, synchronise
    "launch_us": .5,  # per kernel: a node boundary inside a replayed CUDA graph
    "sync_us": 5.,  # a value read back to the host
    "flops_per_us": 7.5e6,  # 7.5 TFLOPS
    "bytes_per_us": 2.0e5,  # 200 GB/s of on-GPU memory traffic for the kernel's own operands
    "host_bytes_per_us": 2.0e5,  # read-back bytes (billed like any operand)
}
TABLE = {  # on-chip billing: the default table
    "step_us": 1.2,  # per wake, a GPU-resident loop: two grid syncs, 0.616 us each at the worst measured 4060 config
    "launch_us": 0.,  # kernels inside the fused wake kernel pay no launch of their own (form C)
    "sync_us": .5,  # a data-dependent branch inside the fused kernel
    "flops_per_us": 7.5e6,  # 7.5 TFLOPS
    "bytes_per_us": 1.0e7,  # 10 TB/s: operands are working state and data already brought in through ctx (on chip)
    "host_bytes_per_us": 1.6e4,  # 16 GB/s, PCIe 4.0 x8: bytes of a value read back to the host
    # ctx.lanes: a lane is one thread block on one SM, so its branches stay on the GPU
    "lane_sms": 24,  # SMs of the 4060: one lane runs at 1/24 of the GPU's compute and on-chip bandwidth
    "lane_sync_us": .02,  # a data-dependent branch inside a block, ~30 cycles
    "lane_phase_us": .6,  # one grid barrier into and out of the per-lane phase (half a step: step_us is two grid syncs)
}
DEFAULT = "onchip"
TABLES = {"onchip": TABLE, "graph": GRAPH, "eager": dict(GRAPH, launch_us=2.)}
for _t in (GRAPH, TABLES["eager"]):
    _t.update(lane_sms=24, lane_sync_us=.02, lane_phase_us=.6)


def lane_table(table):
    """The billing inside ctx.lanes: a branch costs lane_sync_us and its bytes move at one SM's share of on-chip bandwidth."""
    return dict(table, sync_us=table["lane_sync_us"], host_bytes_per_us=table["bytes_per_us"] / table["lane_sms"])

FREE = {"__get__", "size", "dim", "ndimension", "numel", "nelement", "stride", "element_size", "is_contiguous",
        "is_floating_point", "is_complex", "storage_offset", "__len__", "dtype", "shape", "device", "requires_grad",
        "is_tensor", "result_type", "promote_types", "broadcast_shapes", "__iter__"}
SYNC = {"item", "tolist", "__bool__", "__int__", "__float__", "__index__", "__format__", "__repr__", "__str__"}
GATHER = {"__getitem__", "gather", "index_select", "take", "take_along_dim", "embedding", "masked_select", "narrow_copy"}
WRITE_INTO = {"__setitem__", "scatter_", "scatter_add_", "scatter_reduce_", "index_put_", "index_add_", "index_copy_",
              "index_fill_", "masked_fill_", "masked_scatter_", "put_", "copy_", "fill_", "zero_"}
MATMUL = {"matmul", "mm", "bmm", "__matmul__", "__rmatmul__", "mv", "dot", "inner", "linear", "vdot"}
ADD_MATMUL = {"addmm", "baddbmm", "addbmm", "addmv"}
INPLACE_DUNDER = {"__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__", "__iand__",
                  "__ior__", "__ixor__", "__ilshift__", "__irshift__", "__imatmul__"}
SORT = {"sort", "argsort", "topk", "kthvalue", "msort", "unique", "median", "nanmedian", "mode", "unique_consecutive"}


_TENSOR = None


def tensor_type():
    global _TENSOR
    if _TENSOR is None:
        import torch
        _TENSOR = torch.Tensor
    return _TENSOR


def tensors(x, out):
    """Tensors among call arguments: top level and one level of lists, tuples and dicts (all torch calls need)."""
    T = tensor_type()
    for a in (x.values() if isinstance(x, dict) else x):
        if isinstance(a, T):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(b for b in a if isinstance(b, T))
    return out


def logical(t):
    return t.numel() * t.element_size()


def einsum_flops(args):
    eq = args[0] if args and isinstance(args[0], str) else None
    ops = tensors(args[1:], [])
    if eq is None or not ops:
        return sum(t.numel() for t in ops)
    sizes = {}
    for spec, t in zip(eq.split("->")[0].replace(" ", "").split(","), ops):
        letters = [c for c in spec if c.isalpha()]
        for c, n in zip(letters[-t.dim():] if "..." in spec else letters, t.shape[-len(letters):] if letters else []):
            sizes[c] = max(sizes.get(c, 1), int(n))
    return 2 * math.prod(sizes.values()) if sizes else 0


def cost(name, args, kwargs, out, table=TABLE):
    """(microseconds, kind) of one call; kind is free, sync or kernel. (Written for speed: it runs on every torch call.)"""
    if name in FREE:
        return 0., "free"
    T = _TENSOR or tensor_type()
    ins = []
    for a in args:
        if isinstance(a, T):
            ins.append(a)
        elif isinstance(a, (list, tuple)):
            ins.extend([b for b in a if isinstance(b, T)])
    if kwargs:
        for a in kwargs.values():
            if isinstance(a, T):
                ins.append(a)
            elif isinstance(a, (list, tuple)):
                ins.extend([b for b in a if isinstance(b, T)])
    if isinstance(out, T):
        outs = [out]
    elif isinstance(out, (list, tuple)):
        outs = [o for o in out if isinstance(o, T)]
    else:
        outs = []
    if name in SYNC or (out is not None and not outs and ins and name not in WRITE_INTO):  # a Python value comes back
        return table["sync_us"] + (ins[0].nbytes if ins else 0) / table["host_bytes_per_us"], "sync"
    inplace = (name[-1:] == "_" and name[-2:] != "__") or name in WRITE_INTO or name in INPLACE_DUNDER
    if not inplace and outs:
        for o in outs:
            if not o._is_view():
                for t in ins:
                    if o is t:
                        break
                else:
                    break
        else:
            return 0., "free"  # a view (reshape, slice, transpose, expand, detach) or an input handed back (.to(same dtype))
    if name in GATHER:
        moved = 2 * sum([o.nbytes for o in outs]) + sum([t.nbytes for t in ins[1:]])
    elif name in WRITE_INTO:
        moved = 2 * sum([t.nbytes for t in ins[1:]]) if len(ins) > 1 else 2 * sum([t.nbytes for t in ins])
    else:  # an expanded view reads its storage, a slice reads itself
        moved = sum([o.nbytes for o in outs])
        for t in ins:
            moved += min(t.nbytes, t.untyped_storage().nbytes()) if t._is_view() else t.nbytes
    o = outs[0] if outs else None
    if name in MATMUL and ins and o is not None:
        flops = 2 * o.numel() * (ins[0].shape[-1] if ins[0].dim() else 1)
    elif name in ADD_MATMUL and len(ins) > 1 and o is not None:
        flops = 2 * o.numel() * ins[1].shape[-1]
    elif name == "einsum":
        flops = einsum_flops(args)
    elif name == "cdist" and len(ins) > 1 and o is not None:
        flops = 3 * o.numel() * ins[0].shape[-1]
    elif name == "tensordot" and len(ins) > 1 and o is not None:
        flops = 2 * math.sqrt(max(ins[0].numel() * ins[1].numel() * max(o.numel(), 1), 0))
    elif name in SORT and ins:
        n = ins[0].numel()
        flops = 4 * n * max(1., math.log2(max(ins[0].shape[-1] if ins[0].dim() else 1, 2)))
    else:
        flops = max([t.numel() for t in ins] + [t.numel() for t in outs] or [0])
    return table["launch_us"] + max(flops / table["flops_per_us"], moved / table["bytes_per_us"]), "kernel"


class Meter:
    """Bills torch calls made by scheme code while active. The evaluator pauses it (ctx methods) with `off()`."""

    def __init__(self, charge, table=TABLE, check=None, every=2000):
        from torch.overrides import TorchFunctionMode

        meter = self

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if meter.paused:
                    return out
                us, kind = cost(getattr(func, "__name__", ""), args, kwargs, out, meter.table)
                if us:
                    meter.kind = kind
                    meter.charge(us)
                    meter.time += us
                meter.counts[kind] += 1
                meter.calls += 1
                if meter.check is not None and meter.calls % meter.every == 0:
                    meter.check()
                return out

        self.mode, self.charge, self.table, self.check, self.every = Mode(), charge, table, check, every
        self.paused, self.time, self.calls, self.counts = 0, 0., 0, {"free": 0, "sync": 0, "kernel": 0}
        self.kind = "kernel"  # the kind of the call being charged (ctx.lanes bills syncs and kernels apart)

    def __enter__(self):
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self.mode.__exit__(*exc)

    def off(self):
        meter = self

        class Off:
            def __enter__(self):
                meter.paused += 1

            def __exit__(self, *exc):
                meter.paused -= 1

        return Off()
