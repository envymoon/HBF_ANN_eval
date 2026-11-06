"""The C device model (device_c.c) behind the same interface as device.Device.

The shared library is built once per source version into _build/ (named by the source hash), on first use. Without a C compiler, or if loading fails, the evaluator falls back to
the Python reference (same results, slower); `implementation()` says which one runs.
"""
import ctypes
import hashlib
import math
import os
import subprocess
import tempfile

import numpy as np

from .device import ALIGN, BANKS_PER_DIE, PAGE, POLICIES, SCOPES, REFRESH_POLICIES, Device as PyDevice, DeviceError, check_config

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "device_c.c")
BUILD = os.path.join(HERE, "_build")
FLAGS = ["-O2", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math", "-std=c99"]
_LIB = None
_TRIED = False

P_DEV = ctypes.c_void_p
I64, F64 = ctypes.c_int64, ctypes.c_double
P_I64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
P_F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def library_path():
    with open(SOURCE, "rb") as f:
        digest = hashlib.sha256(f.read() + " ".join(FLAGS).encode()).hexdigest()[:16]
    return os.path.join(BUILD, "device-%s.so" % digest)


def build():
    """Compile device_c.c if this version is not built yet; returns the library path or None."""
    path = library_path()
    if os.path.exists(path):
        return path
    try:
        os.makedirs(BUILD, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".so", dir=BUILD)
        os.close(fd)
        subprocess.run([os.environ.get("CC", "cc")] + FLAGS + ["-o", tmp, SOURCE, "-lm"], check=True, capture_output=True)
        os.replace(tmp, path)
        return path
    except (OSError, subprocess.CalledProcessError):
        return None


def load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    path = library_path()
    if not os.path.exists(path) and os.access(HERE, os.W_OK):
        path = build()
    if not path or not os.path.exists(path):
        return None
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    lib.dev_new.restype = P_DEV
    lib.dev_new.argtypes = [P_I64, P_F64, ctypes.c_uint64]
    lib.dev_free.argtypes = [P_DEV]
    lib.dev_submit.restype = I64
    lib.dev_submit.argtypes = [P_DEV, F64, I64, P_I64, P_I64, P_I64, P_I64, I64]
    lib.dev_run_until.argtypes = [P_DEV, F64]
    lib.dev_next_wake.restype = F64
    lib.dev_next_wake.argtypes = [P_DEV, F64, I64, F64, I64]
    lib.dev_hbm_read.restype = F64
    lib.dev_hbm_read.argtypes = [P_DEV, F64, I64]
    lib.dev_sram_read.restype = F64
    lib.dev_sram_read.argtypes = [P_DEV, F64, I64, P_I64, P_I64]
    lib.dev_refresh.restype = I64
    lib.dev_refresh.argtypes = [P_DEV, F64, I64, I64]
    lib.dev_debt.argtypes = [P_DEV, F64, P_F64]
    lib.dev_settle.restype = F64
    lib.dev_settle.argtypes = [P_DEV]
    for name in ("dev_ndone", "dev_outstanding", "dev_nreq", "dev_ncdc"):
        getattr(lib, name).restype = I64
        getattr(lib, name).argtypes = [P_DEV]
    lib.dev_failed.restype = ctypes.c_int
    lib.dev_failed.argtypes = [P_DEV]
    lib.dev_take_done.restype = I64
    lib.dev_take_done.argtypes = [P_DEV, P_F64, P_I64, P_I64]
    lib.dev_pending.restype = I64
    lib.dev_pending.argtypes = [P_DEV, P_I64, I64]
    lib.dev_tag.restype = I64
    lib.dev_tag.argtypes = [P_DEV, I64]
    lib.dev_counters.argtypes = [P_DEV, P_I64]
    lib.dev_floats.argtypes = [P_DEV, P_F64]
    lib.dev_arrays.argtypes = [P_DEV, P_F64, P_I64, P_F64, P_I64, P_F64]
    _LIB = lib
    return lib


def implementation():
    return "c" if load() is not None else "python"


def ints(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.int64).reshape(-1))


class Tags:
    """d.r_tag[r] for the tests, read from the C side."""

    def __init__(self, dev):
        self.dev = dev

    def __getitem__(self, r):
        return int(self.dev.lib.dev_tag(self.dev.h, int(r)))


class CDevice:
    def __init__(self, *, channels, units, tR, channel_bw, policy="lru", jitter=0., tmr=0., mocs=1 << 30,
                 interleave=1, disturb_us=0., pages_per_block=256, hbm_bw=0., hbm_latency=0., sram_latency=0., seed=0,
                 banks_per_die=BANKS_PER_DIE, channels_per_stack=0, power_w=0., power_depth=0., e_sense=0., e_prog=0.,
                 coredie=0, coredie_bw=0., refresh_x=0., t_prog=0., scope="die", d_max=math.inf, retry_p=0.,
                 refresh_policy="default", hits=True, buffers=2, hit_us=0.):
        check_config(units, banks_per_die, channels, channels_per_stack, coredie, scope, refresh_policy, policy, buffers)
        self.lib = load()
        if self.lib is None:
            raise RuntimeError("the C device model is not available")
        self.C, self.U, self.g = int(channels), int(units), int(interleave)
        self.banks = int(banks_per_die)
        self.dies = self.U // self.banks
        ip = np.array([self.C, self.U, POLICIES.index(policy), int(mocs), self.g, int(pages_per_block), self.banks,
                       int(channels_per_stack), int(coredie), SCOPES.index(scope), REFRESH_POLICIES.index(refresh_policy),
                       int(bool(hits)), int(buffers)],
                      dtype=np.int64)
        dp = np.array([tR, channel_bw, jitter, tmr, disturb_us, hbm_bw, hbm_latency, sram_latency, power_w, power_depth,
                       e_sense, e_prog, coredie_bw, refresh_x, t_prog, d_max, retry_p, hit_us],
                      dtype=np.float64)
        self.h = self.lib.dev_new(ip, dp, int(seed) & ((1 << 64) - 1))
        if not self.h:
            raise MemoryError("device model: out of memory")
        self.r_tag = Tags(self)

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.dev_free(self.h)
            self.h = None

    def check(self):
        if self.lib.dev_failed(self.h):
            raise MemoryError("device model: out of memory")

    # ---- the interface of device.Device ----
    def locate(self, page):
        g, C = self.g, self.C
        return (page // g) % C, g * (page // (g * C)) + page % g

    def mapping(self, page):
        channel, local = self.locate(page)
        return channel, (local // self.banks) % self.dies, local % self.banks

    def submit(self, t, pages, offsets, nbytes, tags, batch):
        p, o, b, g = ints(pages), ints(offsets), ints(nbytes), ints(tags)
        n = p.shape[0]
        first = self.lib.dev_submit(self.h, float(t), n, p, o, b, g, 1 if batch else 0)
        if first == -1:
            bad = next(i for i in range(n) if b[i] <= 0 or b[i] % ALIGN or o[i] % ALIGN or o[i] < 0 or o[i] + b[i] > PAGE or p[i] < 0)
            raise DeviceError("a read is 64 B to 4 KiB, 64 B aligned, inside one 4 KiB page (offset %d, %d bytes)" % (o[bad], b[bad]))
        if first == -2:
            raise RuntimeError("device time went backwards")
        if first < 0:
            raise MemoryError("device model: out of memory")
        return range(first, first + n)

    def hbm_read(self, t, nbytes):
        out = self.lib.dev_hbm_read(self.h, float(t), int(nbytes))
        if math.isnan(out):
            raise DeviceError("no HBM in this configuration")
        return out

    def sram_read(self, t, per_channel):
        items = sorted(per_channel.items())
        ch, nb = ints([c for c, _ in items]), ints([n for _, n in items])
        out = self.lib.dev_sram_read(self.h, float(t), len(items), ch, nb)
        self.check()
        return out

    def run_until(self, t):
        self.lib.dev_run_until(self.h, float(t))
        self.check()

    def next_wake(self, cursor, count=1, timeout=math.inf, extra=0):
        out = self.lib.dev_next_wake(self.h, float(cursor), int(min(count, 1 << 62)), float(timeout), int(extra))
        self.check()
        if math.isnan(out):
            raise RuntimeError("device stalled with %d requests outstanding" % self.outstanding)
        return out

    def take_done_arrays(self):
        n = self.lib.dev_ndone(self.h)
        t, r, tag = np.empty(n, np.float64), np.empty(n, np.int64), np.empty(n, np.int64)
        self.lib.dev_take_done(self.h, t, r, tag)
        return t, r, tag

    def take_done(self):
        t, r, _ = self.take_done_arrays()
        return list(zip(t.tolist(), r.tolist()))

    def done_count(self):
        return int(self.lib.dev_ndone(self.h))

    @property
    def outstanding(self):
        return int(self.lib.dev_outstanding(self.h))

    def pending_tags(self):
        n = self.lib.dev_pending(self.h, np.empty(0, np.int64), 0)
        out = np.empty(n, np.int64)
        self.lib.dev_pending(self.h, out, n)
        return out.tolist()

    def refresh(self, t, die, n):
        out = self.lib.dev_refresh(self.h, float(t), int(die), int(n))
        if out == -2:
            raise RuntimeError("device time went backwards")
        if out == -1:
            raise DeviceError("refresh: die must be in [0, %d) and n >= 0" % (self.C * self.dies))
        self.check()

    def debt_per_die(self, t):
        out = np.zeros(self.C * self.dies, np.float64)
        self.lib.dev_debt(self.h, float(t), out)
        self.check()
        return out.tolist()

    def settle(self):
        return float(self.lib.dev_settle(self.h))

    def counters(self):
        out = np.zeros(7, np.int64)
        self.lib.dev_counters(self.h, out)
        return out.tolist()

    def floats(self):
        out = np.zeros(6, np.float64)
        self.lib.dev_floats(self.h, out)
        return out.tolist()

    @property
    def senses(self):
        return self.counters()[1]

    @property
    def hits(self):
        return self.counters()[0]

    @property
    def retries(self):
        return self.counters()[5]

    @property
    def ref_ops(self):
        return self.counters()[6]

    def arrays(self):
        CU, K = self.C * self.U, int(self.lib.dev_ncdc(self.h))
        ub, us, lt, lb = np.zeros(CU, np.float64), np.zeros(CU, np.int64), np.zeros(self.C, np.float64), np.zeros(self.C, np.int64)
        ct = np.zeros(max(K, 1), np.float64)
        self.lib.dev_arrays(self.h, ub, us, lt, lb, ct)
        return ub.tolist(), us.tolist(), lt.tolist(), lb.tolist(), ct[:K].tolist()

    @property
    def unit_busy(self):
        return self.arrays()[0]

    def stats(self, span):
        hits, senses, sram_bytes, hbm_bytes, hottest, retries, ref_ops = self.counters()
        ref_pages, ref_unit_us, max_debt, energy, power_wait, settle = self.floats()
        loads, _, link_time, link_bytes, ctime = self.arrays()
        CU = self.C * self.U
        mean = sum(loads) / CU if CU else 0.
        busiest = max(loads) if loads else 0.
        link_mean = sum(link_time) / self.C if self.C else 0.
        K = len(ctime)
        cdc_mean = sum(ctime) / K if K else 0.
        return {"senses": senses, "hits": hits, "requests": int(self.lib.dev_nreq(self.h)),
                "bytes": sum(link_bytes), "sram_bytes": sram_bytes, "hbm_bytes": hbm_bytes,
                "unit_rho": busiest / mean if mean > 0 else 0., "busiest_unit_util": busiest / span if span > 0 else 0.,
                "unit_util": mean / span if span > 0 else 0.,
                "channel_rho": max(link_time) / link_mean if link_mean > 0 else 0.,
                "channel_util": link_mean / span if span > 0 else 0., "hottest_block_reads": hottest,
                "coredie_rho": max(ctime) / cdc_mean if cdc_mean > 0 else 0.,
                "coredie_util": cdc_mean / span if span > 0 else 0.,
                "retries": retries, "refresh_ops": ref_ops, "refresh_pages": ref_pages,
                "refresh_unit_share": ref_unit_us / (CU * span) if span > 0 and CU else 0.,
                "max_debt": max_debt, "energy_uj": energy, "power_wait_us": power_wait, "settle_us": settle}


def Device(**kw):
    """The C model when it is available, else the Python reference: identical results."""
    return CDevice(**kw) if load() is not None else PyDevice(**kw)
