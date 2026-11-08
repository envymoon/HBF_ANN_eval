"""The charged interfaces handed to scheme code.

Scheme code can reach only the public methods below: the static policy forbids names starting with "_", and every piece of
evaluator state lives behind one underscore slot. Everything handed out is a fresh copy.
"""
import math


class SchemeError(Exception):
    """The scheme broke the contract; the evaluation fails."""


def _int_tensor(x, name, dims=1):
    import torch
    if not isinstance(x, torch.Tensor) or x.dtype != torch.int64 or x.dim() != dims:
        raise SchemeError("%s must be an int64 tensor with %d dimension(s)" % (name, dims))
    return x


class BuildCtx:
    """For build (sees the data only) and layout (also sees the device's nominal figures)."""
    __slots__ = ("_s",)

    def __init__(self, knobs, generator, spec):
        object.__setattr__(self, "_s", {"knobs": dict(knobs), "g": generator, "spec": dict(spec)})

    def __setattr__(self, name, value):
        raise SchemeError("ctx is read-only")

    @property
    def knobs(self):
        return dict(self._s["knobs"])

    @property
    def effort(self):
        raise SchemeError("ctx.effort exists only in serve: build and layout are cached across effort levels")

    def uniform(self, *shape):
        import torch
        return torch.rand(*shape, generator=self._s["g"]).clamp_(1e-6, 1 - 1e-6)

    def spec(self):
        return dict(self._s["spec"])


class ServeCtx:
    """For serve: the online, timed side. All methods bill simulated time through the session."""
    __slots__ = ("_s",)

    def __init__(self, session):
        object.__setattr__(self, "_s", session)

    def __setattr__(self, name, value):
        raise SchemeError("ctx is read-only")

    @property
    def knobs(self):
        return dict(self._s.knobs)

    @property
    def effort(self):
        return self._s.effort

    def spec(self):
        return dict(self._s.spec)

    def now(self):
        return self._s.lane_now()

    def lanes(self, n):
        """for i in ctx.lanes(n): per-lane code. The lanes run in parallel (one thread block each); a branch on a tensor
        value inside a lane stays on the GPU (cheap), unlike one outside. The loop must run to its end."""
        return self._s.lanes(n)

    def uniform(self, *shape):
        return self._s.uniform(shape)

    def submit(self, pages, offsets, nbytes, tags, batch=False):
        """Issue HBF reads now; they complete later and come back in events. Byte offsets and sizes are multiples of 64,
        inside one 4 KiB page. nbytes may be one int for all. batch=True sets the Batch Read hint."""
        return self._s.submit(pages, offsets, nbytes, tags, batch)

    def hbm(self, name, rows=None):
        """Rows of an HBM tensor placed by layout (all of it with rows=None); waits for the data."""
        return self._s.read_tier("hbm", name, rows)

    def sram(self, name, rows=None):
        """Rows of a scratchpad tensor placed by layout (row r lives on channel r mod channels); waits for the data."""
        return self._s.read_tier("sram", name, rows)

    def hbm_write(self, name, rows, values):
        """Overwrite rows of an HBM tensor placed by layout (rows=None: all of it) with values of its dtype and row shape;
        rows distinct. Billed as a transfer of the same bytes over the HBM link. The size and shape stay as layout placed."""
        return self._s.write_tier("hbm", name, rows, values)

    def sram_write(self, name, rows, values):
        """Overwrite rows of a scratchpad tensor placed by layout (row r lives on channel r mod channels); billed on the
        channel links like a read of the same rows."""
        return self._s.write_tier("sram", name, rows, values)

    def finish(self, qids, ids):
        """Answer queries: ids int64[len(qids), k]. Each answered client sends its next query at once."""
        return self._s.finish(qids, ids)

    def wait(self, count=1, timeout_us=math.inf):
        """After serve returns, wake when `count` events (completions plus new queries) wait, or after `timeout_us`,
        or when nothing is in flight. Applies to the next wake only."""
        return self._s.set_wait(count, timeout_us)

    def mapping(self, pages):
        """int64[r, 3]: channel, die, bank of each page (what the host reads from the configuration registers)."""
        return self._s.mapping(pages)

    def pending(self):
        """int64 tags of the reads issued and not yet seen complete, in issue order."""
        return self._s.pending()

    def debt(self):
        """float64[dies]: pages of refresh work each die owes (the most any of its planes owes); die id = channel *
        dies_per_channel + die, as ctx.mapping gives them."""
        return self._s.debt()

    def refresh(self, dies, counts=1):
        """Ask for refresh operations now: counts[i] on die dies[i] (int64 tensors; counts may be one int). An operation
        rewrites one page on each plane of the die that owes one (or, where programs hold only a plane, on the plane that
        owes most); it waits for the die's reads in progress and holds the die for tR + program time."""
        return self._s.refresh(dies, counts)
