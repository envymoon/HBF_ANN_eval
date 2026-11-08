"""The closed-loop online run : N clients, each sending its next query the moment the last one is answered.

The host is one GPU stream: `serve` runs at each wake, every torch op and every ctx call moves the host's clock (the cursor)
forward, and reads issued with ctx.submit run on the device model meanwhile. When serve returns, the host sleeps until its
wake condition holds (ctx.wait), then gets the completions and the new queries as `events`.
"""
import math
import time

import numpy as np

from .compute_table import TABLE, Meter
from .ctx import SchemeError, ServeCtx, _int_tensor
from .device import ALIGN, PAGE, DeviceError

HBM_SECTOR, SRAM_SECTOR = 32, 64
MAX_PER_CALL, MAX_OUTSTANDING = 1 << 20, 1 << 21
IDLE_STEPS = 1000  # consecutive wakes with nothing in flight and nothing delivered before we call it a deadlock


class TimeLimit(Exception):
    pass


class Column:
    """A growing int64 array indexed by request id (ids are issued 0, 1, 2, ... in submission order)."""

    def __init__(self):
        self.a, self.n = np.zeros(1024, np.int64), 0

    def extend(self, x):
        k = x.shape[0]
        if self.n + k > self.a.shape[0]:
            grown = np.zeros(max(2 * self.a.shape[0], self.n + k), np.int64)
            grown[:self.n] = self.a[:self.n]
            self.a = grown
        self.a[self.n:self.n + k] = x
        self.n += k


def state_bytes(x, seen, depth=0):
    import torch
    if depth > 8:
        raise SchemeError("state nested too deeply")
    if isinstance(x, torch.Tensor):
        key = x.untyped_storage().data_ptr()
        if key in seen:
            return 0
        seen.add(key)
        return x.untyped_storage().nbytes()
    if x is None or isinstance(x, (bool, int, float)):
        return 8
    if isinstance(x, str):
        if len(x) > 64:
            raise SchemeError("strings in state are at most 64 characters")
        return len(x)
    if isinstance(x, (list, tuple)):
        return 8 * len(x) + sum(state_bytes(y, seen, depth + 1) for y in x)
    if isinstance(x, dict):
        if not all(isinstance(k, str) for k in x):
            raise SchemeError("state dict keys must be strings")
        return 8 * len(x) + sum(state_bytes(y, seen, depth + 1) for y in x.values())
    raise SchemeError("state may hold tensors, numbers, short strings, lists, tuples and dicts, not " + type(x).__name__)


class Session:
    def __init__(self, *, device, plan, queries, k, n, clients, spec, knobs, effort, generator, state_cap,
                 deadline=math.inf, table=TABLE):
        self.device, self.k, self.n, self.N, self.table = device, k, n, clients, table
        self.pages = plan["pages"]
        # serve may rewrite rows of what layout placed (ctx.hbm_write / ctx.sram_write), so each run works on its own copy:
        # the cached plan and the other effort levels never see a run's writes
        self.tiers = {tier: {kk: v.clone() for kk, v in plan[tier].items()} for tier in ("hbm", "sram")}
        self.written = {"hbm": 0, "sram": 0}
        self.queries, self.total = queries, queries.shape[0]
        self.spec, self.knobs, self.effort, self.g = spec, knobs, effort, generator
        self.state_cap, self.deadline = state_cap, deadline
        self.cursor = 0.
        self.meter = Meter(self.charge, table, check=self.check_time)
        self.wait_count, self.wait_timeout = 1, math.inf
        self.req_page, self.req_off, self.req_bytes = Column(), Column(), Column()
        self.arrivals, self.open = [], set()
        self.arrive_t, self.finish_t = [math.nan] * self.total, [math.nan] * self.total
        self.results = [None] * self.total
        self.issued = 0
        self.ctx_us, self.steps, self.max_state = 0., 0, 0

    # ---- billing ----
    def charge(self, us):
        self.cursor += us

    def kernel(self, us=0.):
        us += self.table["launch_us"]
        self.cursor += us
        self.ctx_us += us

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise TimeLimit("wall-clock limit for this task exceeded")

    # ---- ctx methods ----
    def uniform(self, shape):
        import torch
        with self.meter.off():
            out = torch.rand(*shape, generator=self.g).clamp_(1e-6, 1 - 1e-6)
            self.kernel(out.numel() * 4 / self.table["bytes_per_us"])
        return out

    def submit(self, pages, offsets, nbytes, tags, batch):
        import torch
        with self.meter.off():
            pages, offsets, tags = (_int_tensor(x, n) for x, n in ((pages, "pages"), (offsets, "offsets"), (tags, "tags")))
            r = pages.shape[0]
            if isinstance(nbytes, int) and not isinstance(nbytes, bool):
                nbytes = torch.full((r,), nbytes, dtype=torch.int64)
            nbytes = _int_tensor(nbytes, "nbytes")
            if not (offsets.shape[0] == r == nbytes.shape[0] == tags.shape[0]):
                raise SchemeError("submit: pages, offsets, nbytes and tags must have one length")
            if r == 0:
                return None
            if r > MAX_PER_CALL or self.device.outstanding + r > MAX_OUTSTANDING:
                raise SchemeError("submit: too many reads in flight")
            if int(pages.min()) < 0 or int(pages.max()) >= self.pages.shape[0]:
                raise SchemeError("submit: page out of range (layout wrote %d pages)" % self.pages.shape[0])
            p, o, b = pages.contiguous().numpy(), offsets.contiguous().numpy(), nbytes.contiguous().numpy()
            try:
                self.device.submit(self.cursor, p, o, b, tags.contiguous().numpy(), bool(batch))
            except DeviceError as exc:
                raise SchemeError("submit: " + str(exc)) from None
            self.req_page.extend(p)
            self.req_off.extend(o)
            self.req_bytes.extend(b)
            self.kernel()
        return None

    def tier_tensor(self, tier, name, what):
        store = self.tiers[tier]
        if not isinstance(name, str) or name not in store:
            raise SchemeError("ctx.%s: no tensor %r was placed there by layout" % (what, name))
        t = store[name]
        row_bytes = t.element_size() * (t[0].numel() if t.dim() else 1)
        sector = HBM_SECTOR if tier == "hbm" else SRAM_SECTOR
        return t, -(-row_bytes // sector) * sector

    def tier_rows(self, t, rows, what):
        import torch
        if not isinstance(rows, torch.Tensor) or rows.dtype != torch.int64:
            raise SchemeError("ctx.%s: rows must be an int64 tensor" % what)
        if t.dim() == 0:
            raise SchemeError("ctx.%s: this tensor is a scalar; use rows=None" % what)
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= t.shape[0]):
            raise SchemeError("ctx.%s: row out of range" % what)

    def tier_bill(self, tier, count, unit, index):
        """Move the cursor past one synchronous transfer of `count` rows of `unit` bytes (a read waits for its data; a write
        is billed the same: it takes the same link for the same bytes and returns when the tier has it)."""
        import torch
        self.kernel()
        if tier == "hbm":
            self.cursor = self.device.hbm_read(self.cursor, count * unit)
        else:
            C = self.device.C
            if index is None:
                per = {c: (count // C + (1 if c < count % C else 0)) * unit for c in range(min(C, count))}
            else:
                per = {c: int(v) * unit for c, v in enumerate(torch.bincount(index.reshape(-1) % C, minlength=C).tolist()) if v}
            self.cursor = self.device.sram_read(self.cursor, per)

    def read_tier(self, tier, name, rows):
        with self.meter.off():
            t, unit = self.tier_tensor(tier, name, tier)
            if rows is None:
                out, index = t.clone(), None
                count = t.shape[0] if t.dim() else 1
            else:
                self.tier_rows(t, rows, tier)
                out, index, count = t[rows], rows, rows.numel()
            self.tier_bill(tier, count, unit, index)
        return out

    def write_tier(self, tier, name, rows, values):
        import torch
        what = tier + "_write"
        with self.meter.off():
            t, unit = self.tier_tensor(tier, name, what)
            if not isinstance(values, torch.Tensor) or values.dtype != t.dtype:
                raise SchemeError("ctx.%s: values must be a %s tensor (the dtype layout placed)" % (what, t.dtype))
            if rows is None:
                if values.shape != t.shape:
                    raise SchemeError("ctx.%s: values must have the tensor's shape %s" % (what, list(t.shape)))
                t.copy_(values)
                index, count = None, t.shape[0] if t.dim() else 1
            else:
                self.tier_rows(t, rows, what)
                if rows.dim() != 1:
                    raise SchemeError("ctx.%s: rows must be one-dimensional" % what)
                if values.shape != (rows.shape[0], *t.shape[1:]):
                    raise SchemeError("ctx.%s: values must be [len(rows), %s]" % (what, ", ".join(map(str, t.shape[1:]))))
                if rows.numel() != torch.unique(rows).numel():
                    raise SchemeError("ctx.%s: rows must be distinct" % what)
                t[rows] = values
                index, count = rows, rows.numel()
            self.written[tier] += count * unit
            self.tier_bill(tier, count, unit, index)
        return None

    def finish(self, qids, ids):
        import torch
        with self.meter.off():
            qids = _int_tensor(qids, "qids")
            ids = _int_tensor(ids, "ids", 2)
            if ids.shape != (qids.shape[0], self.k):
                raise SchemeError("finish: ids must be int64[len(qids), k=%d]" % self.k)
            if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.n):
                raise SchemeError("finish: result id out of range")
            q = qids.tolist()
            if len(set(q)) != len(q) or not set(q) <= self.open:
                raise SchemeError("finish: every qid must be an open query, once")
            for i, qid in enumerate(q):
                self.open.discard(qid)
                self.finish_t[qid] = self.cursor
                self.results[qid] = ids[i].clone()
                self.arrive_next()
            self.kernel()
        return None

    def arrive_next(self):
        if self.issued < self.total:
            qid = self.issued
            self.issued += 1
            self.arrive_t[qid] = self.cursor
            self.arrivals.append(qid)

    def set_wait(self, count, timeout_us):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise SchemeError("wait: count is an int >= 1")
        if not isinstance(timeout_us, (int, float)) or not timeout_us >= 0:
            raise SchemeError("wait: timeout_us is a number >= 0")
        self.wait_count, self.wait_timeout = count, float(timeout_us)
        return None

    def mapping(self, pages):
        import torch
        with self.meter.off():
            pages = _int_tensor(pages, "pages")
            d = self.device
            g, C = d.g, d.C
            channel = (pages // g) % C
            local = g * (pages // (g * C)) + pages % g
            out = torch.stack([channel, (local // d.banks) % d.dies, local % d.banks], dim=1)
        return out

    def debt(self):
        import torch
        with self.meter.off():
            self.kernel()  # a register read (OCP §11.5: the host may query the block read counts)
            out = torch.tensor(self.device.debt_per_die(self.cursor), dtype=torch.float64)
        return out

    def refresh(self, dies, counts):
        import torch
        with self.meter.off():
            dies = _int_tensor(dies, "refresh: dies")
            if isinstance(counts, int) and not isinstance(counts, bool):
                counts = torch.full((dies.shape[0],), counts, dtype=torch.int64)
            counts = _int_tensor(counts, "refresh: counts")
            if counts.shape != dies.shape:
                raise SchemeError("refresh: dies and counts must have one length")
            total = self.device.C * self.device.dies
            if dies.numel() and (int(dies.min()) < 0 or int(dies.max()) >= total):
                raise SchemeError("refresh: die out of range (%d dies)" % total)
            if counts.numel() and int(counts.min()) < 0:
                raise SchemeError("refresh: counts must be >= 0")
            if int(counts.sum()) > MAX_PER_CALL:
                raise SchemeError("refresh: too many operations in one call")
            for die, n in zip(dies.tolist(), counts.tolist()):
                if n:
                    self.device.refresh(self.cursor, die, n)
            self.kernel()
        return None

    def pending(self):
        import torch
        with self.meter.off():
            out = torch.tensor(self.device.pending_tags(), dtype=torch.int64)
        return out

    # ---- the loop ----
    def events(self):
        import torch
        _, reqs, tags = self.device.take_done_arrays()
        a, self.arrivals = self.arrivals, []
        ev = {"qids": torch.tensor(a, dtype=torch.int64), "queries": self.queries[a].clone() if a else self.queries[:0].clone()}
        nbytes = self.req_bytes.a[reqs]
        ev["tags"] = torch.from_numpy(tags)
        ev["nbytes"] = torch.from_numpy(nbytes)
        if reqs.shape[0]:
            width = int(nbytes.max())
            page = torch.from_numpy(self.req_page.a[reqs])
            off = torch.from_numpy(self.req_off.a[reqs])
            col = torch.arange(width, dtype=torch.int64)
            idx = (off[:, None] + col[None, :]).clamp_(max=PAGE - 1)
            data = self.pages[page[:, None], idx]
            data[col[None, :] >= ev["nbytes"][:, None]] = 0
            ev["data"] = data
        else:
            ev["data"] = torch.zeros(0, ALIGN, dtype=torch.uint8)
        return ev, int(reqs.shape[0])

    def run(self, serve):
        ctx = ServeCtx(self)
        for _ in range(min(self.N, self.total)):
            self.arrive_next()
            self.open.add(self.issued - 1)
        state, wake, idle = {}, 0., 0
        while True:
            ev, delivered = self.events()
            self.open.update(ev["qids"].tolist())
            self.cursor = wake + self.table["step_us"]
            self.wait_count, self.wait_timeout = 1, math.inf
            with self.meter:
                state = serve(state, ev, ctx)
            if not isinstance(state, dict):
                raise SchemeError("serve must return its state as a dict")
            size = state_bytes(state, set())
            if size > self.state_cap:
                raise SchemeError("state too large: %d bytes, cap %d" % (size, self.state_cap))
            self.max_state = max(self.max_state, size)
            self.steps += 1
            self.check_time()
            if not self.open and not self.arrivals and self.issued >= self.total:
                break
            nothing = self.device.outstanding == 0 and self.device.done_count() == 0 and not self.arrivals
            idle = idle + 1 if nothing and not delivered and not len(ev["qids"]) else 0
            if idle > IDLE_STEPS:
                raise SchemeError("deadlock: %d queries open, nothing in flight, serve makes no progress" % len(self.open))
            wake = self.device.next_wake(self.cursor, self.wait_count, self.wait_timeout, extra=len(self.arrivals))
        return self.summary()

    def summary(self):
        return {"arrive": self.arrive_t, "finish": self.finish_t, "results": self.results, "end": self.cursor,
                "steps": self.steps, "step_us": self.table["step_us"], "compute_us": self.meter.time, "ctx_us": self.ctx_us, "ops": dict(self.meter.counts),
                "max_state": self.max_state,
                "hbm_written": self.written["hbm"], "sram_written": self.written["sram"]}
