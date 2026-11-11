"""Evaluate one scheme on one task: build and layout (cached), sweep ctx.effort, run the closed loop, score.

evaluate(scheme, knobs, task, seed, job, deadline) -> one record per task (curve, speedups, bounds, diagnostics).
job = {"scale": ..., "data_root": ..., "cache": ...}; scale = {"cpp", "pool", "k", "efforts", "total_min", "total_max",
"total_per_client", "n"} and, for the speedup score, "mode": "speedup" and "levels": [recall levels]. Optional:
"tail_floor" (an operating point counts only if the mean recall of its worst 5 % of queries reaches it), "refine" (bisection
steps on the effort interval that crosses each recall level) and "graded_floor" (a level not reached scores below
the floor by the recall shortfall).
"""
import hashlib
import json
import math
import os
import sys
import time


from . import data as datasets
from . import fastdevice
from .arch import POSITIONS, REFRESH, Geometry
from .compute_table import DEFAULT as DEFAULT_TABLE, TABLES, VERSION as TABLE_VERSION
from .ctx import BuildCtx, SchemeError
from .device import PAGE
from .scoring import SPEEDUP_LEVEL, area, bounds, speedup
from .sim import Session, TimeLimit
from .tasks import LEVELS, clients, draw_weather, query_order, total_queries

CAP_ERROR = "too many reads in flight"
EVALUATOR = "hbf_ann.runner.v5"  # v5: tail-recall floor, effort bisection at each recall level, graded floor;
# report: offline cost (build / layout seconds, index / HBF / HBM / SRAM bytes), latency ratios, dependent HBF read rounds
# v4: serve may rewrite HBM / SRAM rows (ctx.hbm_write, ctx.sram_write) and run per-lane control flow on the GPU (ctx.lanes)
# v3: buffer hits take the base-die SRAM access time
# The build stage's own version: it is part of the cache key and of the build's random stream. Bump it only when build's
# inputs or semantics change.
BUILD_VERSION = "hbf_ann.runner.v1"
STATE_PER_CLIENT = 64 * 1024  # cap on the state a client may keep
ONCHIP_PER_STACK = 32 << 20  # on-chip state per stack (position) of the GPU, x 8 positions, scaled like the channels
INDEX_CAP = 16  # an index may take at most 16x the data (cache size guard)
DEPTH_QUERIES = 8  # queries of the one-client read-rounds measurement
_MEMO = {}
_SECONDS = {}  # cache key -> wall seconds of the stage that produced it (report only)


class Cache:
    """Build and layout results: in memory for this process, on disk under <root>/index when a cache directory is bound."""

    def __init__(self, root):
        self.root = os.path.join(root, "index") if root else None

    def get(self, key):
        if key in _MEMO:
            return _MEMO[key]
        if not self.root:
            return None
        path = os.path.join(self.root, key + ".pt")
        if not os.path.exists(path):
            return None
        import torch
        try:
            saved = torch.load(path, weights_only=True)
        except Exception:
            return None
        if saved.get("sum") != digest(saved.get("value")):
            return None
        _MEMO[key] = saved["value"]
        if "seconds" in saved:
            _SECONDS[key] = saved["seconds"]
        return saved["value"]

    def seconds(self, key):
        """Wall-clock seconds the stage took when it was computed (None if unknown, e.g. an entry written before v5)."""
        return _SECONDS.get(key)

    def put(self, key, value, nbytes, seconds=None):
        if len(_MEMO) > 8:
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[key] = value
        if seconds is not None:
            _SECONDS[key] = round(seconds, 3)
        if self.root and nbytes < (2 << 30):
            import torch
            os.makedirs(self.root, exist_ok=True)
            path = os.path.join(self.root, key + ".pt")
            atomic_save({"value": value, "sum": digest(value), "seconds": _SECONDS.get(key)}, path)


def fingerprint(value):
    """Content hash of JSON-able data (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def atomic_save(obj, path):
    """Write to a private temporary file, then rename: parallel workers writing the same key cannot collide, and a
    reader never sees half a file. Losing a race only costs the cache entry, never the evaluation."""
    import tempfile
    import torch
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(value):
    import torch
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, torch.Tensor):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.contiguous().view(-1).view(torch.uint8).numpy().tobytes() if x.numel() else b"")
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(repr(k).encode())
                walk(x[k])
        else:
            h.update(repr(x).encode())

    walk(value)
    return h.hexdigest()


def tensor_dict(x, what):
    import torch
    if not isinstance(x, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in x.items()):
        raise SchemeError(what + " must be {name: tensor}")
    return x


def nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors.values())


def check_plan(plan, weather, data_bytes):
    import torch
    if not isinstance(plan, dict) or not {"pages", "hbm", "sram", "interleave"} <= set(plan) \
            or not set(plan) <= {"pages", "hbm", "sram", "interleave", "refresh"}:
        raise SchemeError("layout must return {pages, hbm, sram, interleave} and optionally refresh "
                             "(m is the platform's: read it from ctx.spec())")
    pages = plan["pages"]
    if not isinstance(pages, torch.Tensor) or pages.dtype != torch.uint8 or pages.dim() != 2 or pages.shape[1] != PAGE \
            or pages.shape[0] < 1:
        raise SchemeError("plan pages must be uint8[P, 4096] with P >= 1")
    g = plan["interleave"]
    if not isinstance(g, int) or isinstance(g, bool) or not 1 <= g <= (1 << 20):
        raise SchemeError("plan interleave must be an int >= 1 (pages)")
    if plan.get("refresh", "default") not in REFRESH:
        raise SchemeError("plan refresh must be 'default' or 'scheme'")
    geo = Geometry(weather, data_bytes)
    m = geo.m
    if pages.shape[0] > geo.hbf_pages:
        raise SchemeError("HBF capacity: %d pages, cap %d at m=%d" % (pages.shape[0], geo.hbf_pages, m))
    hbm, sram = tensor_dict(plan["hbm"], "plan hbm"), tensor_dict(plan["sram"], "plan sram")
    if nbytes(hbm) > geo.hbm_budget:
        raise SchemeError("HBM budget: %d bytes, cap %d at m=%d" % (nbytes(hbm), geo.hbm_budget, m))
    per_channel = sum(math.ceil((t.shape[0] if t.dim() else 1) / geo.channels) * t.element_size() * (t[0].numel() if t.dim() else 1)
                      for t in sram.values())
    if per_channel > geo.sram_per_channel:
        raise SchemeError("SRAM: %d bytes per channel, cap %d" % (per_channel, geo.sram_per_channel))
    return geo


def stage(space, name, key, args, cache, deadline):
    """(value, cached, seconds): seconds is the wall time of computing the stage, now or when it was cached."""
    value = cache.get(key)
    if value is None:
        t0 = time.monotonic()
        value = space[name](*args)
        if time.monotonic() > deadline:
            raise TimeLimit(name + " exceeded the time limit")
        return value, False, time.monotonic() - t0
    return value, True, cache.seconds(key)


def evaluate(scheme, knobs, task, seed, job, deadline, space=None):
    import torch
    space = space or scheme.namespace()
    spec, started = task["spec"], time.monotonic()
    record = {"task": task["name"], "norm": None, "final": None, "failure": None, "seconds": None}
    try:
        scale, k = job["scale"], job["scale"]["k"]
        d = datasets.load(spec["dataset"], spec["n"], scale["pool"], k, job.get("data_root"), job.get("cache"))
        n, dim, vb = d["base"].shape[0], d["dim"], d["vector_bytes"]
        data_bytes = n * vb
        weather = draw_weather(seed, spec, spec.get("split", "main"), scale["cpp"])
        keys = scheme.stage_keys(knobs)
        data_spec = {"n": n, "dim": dim, "metric": d["metric"], "k": k, "vector_bytes": vb, "dtype": d["dtype"]}
        cache = Cache(job.get("cache"))
        bkey = fingerprint({"stage": "build", "evaluator": BUILD_VERSION, "data": d["id"], **keys["build"]})
        bctx = BuildCtx(knobs, torch.Generator().manual_seed(datasets.stream_seed("build", bkey)), data_spec)
        index, hit_b, build_s = stage(space, "build", bkey, (d["base"].clone(), bctx), cache, deadline)
        tensor_dict(index, "build's index")
        if nbytes(index) > INDEX_CAP * data_bytes:
            raise SchemeError("index too large: %d bytes, cap %d" % (nbytes(index), INDEX_CAP * data_bytes))
        if not hit_b:
            cache.put(bkey, index, nbytes(index), build_s)
        layout_spec = dict(data_spec, **Geometry.options(weather, data_bytes))
        lkey = fingerprint({"stage": "layout", "build": bkey, "spec": layout_spec, **keys["layout"]})
        lctx = BuildCtx(knobs, torch.Generator().manual_seed(datasets.stream_seed("layout", lkey)), layout_spec)
        plan, hit_l, layout_s = stage(space, "layout", lkey, ({kk: v.clone() for kk, v in index.items()}, d["base"].clone(), lctx), cache, deadline)
        geo = check_plan(plan, weather, data_bytes)
        if not hit_l:
            cache.put(lkey, plan, plan["pages"].numel() + nbytes(plan["hbm"]) + nbytes(plan["sram"]), layout_s)
        N = clients(spec, weather)
        total = total_queries(spec, N, scale)
        order = query_order(seed, spec, d["queries"].shape[0], total, d["queries"])
        queries, truth = d["queries"][order].contiguous(), d["truth"][order]
        total = queries.shape[0]
        cap = min(N * STATE_PER_CLIENT, int(POSITIONS * ONCHIP_PER_STACK * scale["cpp"] / 16))
        serve_spec = dict(layout_spec, channels=geo.channels, pages=int(plan["pages"].shape[0]),
                          interleave=plan["interleave"], refresh=plan.get("refresh", "default"), clients=N, state_bytes=cap,
                          hbm={kk: list(v.shape) for kk, v in plan["hbm"].items()},
                          sram={kk: list(v.shape) for kk, v in plan["sram"].items()})
        table = TABLES[job.get("compute", DEFAULT_TABLE)]
        bound = bounds(weather, data_bytes, n, dim, k, vb, N, table)
        points, curve, diag_at, point_efforts = [], [], None, []
        efforts, compact = scale["efforts"], scale.get("mode") == "speedup"
        tail = scale.get("tail_floor")  # an operating point counts only if the mean recall of its worst 5 % reaches this

        def run_at(effort):
            device = geo.device(weather, plan["interleave"], datasets.stream_seed("device", seed, task["name"]),
                                refresh=plan.get("refresh", "default"))
            g = torch.Generator().manual_seed(datasets.stream_seed("serve", seed, task["name"]))
            session = Session(device=device, plan=plan, queries=queries, k=k, n=n, clients=N, spec=serve_spec, knobs=knobs,
                              effort=effort, generator=g, state_cap=cap, deadline=deadline, table=table)
            out = session.run(space["serve"])
            row = measure(out, truth, k, device, weather, N)
            curve.append([round(effort, 4)] + [row[x] for x in ("recall", "qps", "p99_us", "p50_us", "reads_per_query", "senses_per_query")])
            points.append(row)
            point_efforts.append(effort)
            return row

        # the speedup score reads the curve at 0.9 (or at each of scale["levels"]); the full suite up to LEVELS[-1]
        targets = sorted(scale["levels"]) if compact and scale.get("levels") else [SPEEDUP_LEVEL if compact else LEVELS[-1]]
        ok = lambda row, level: row["recall"] >= level and tail_ok(row, tail)
        # An effort whose reads overflow the in-flight cap ends the sweep: the scheme cannot go higher, the points below it
        # stand, and a level it had not reached scores as not reached (record["capped_at"] names that effort).
        capped = None

        def run_or_cap(effort):
            nonlocal capped
            try:
                return run_at(effort)
            except SchemeError as e:
                if CAP_ERROR not in str(e) or not points:
                    raise
                capped = effort if capped is None else min(capped, effort)
                return None

        for i in range(efforts):
            effort = i / (efforts - 1) if efforts > 1 else 1.
            row = run_or_cap(effort)
            if row is None or ok(row, targets[-1]):
                break
        # bisect the effort interval that crosses each level, so the crossing is read from nearby points instead of
        # interpolated across a whole effort step
        for level in targets:
            for _ in range(scale.get("refine", 0)):
                fail = [e for e, r in zip(point_efforts, points) if not ok(r, level)]
                good = [e for e, r in zip(point_efforts, points) if ok(r, level) and (not fail or e > max(fail))]
                if not fail or not good:
                    break
                if run_or_cap((max(fail) + min(good)) / 2) is None:
                    break
        curve.sort(key=lambda c: c[0])
        for row in points:
            if diag_at is None or abs(row["recall"] - .95) < abs(diag_at["recall"] - .95):
                diag_at = row

        # Dependent HBF read rounds per query (report only): one client, DEPTH_QUERIES queries at the effort of the
        # diagnostics point; the wakes in which serve issues HBF reads are the query's chain of dependent read rounds (a
        # graph walk: about one per hop; a cluster scan: one or two).
        def read_rounds():
            try:
                q = min(DEPTH_QUERIES, queries.shape[0])
                device = geo.device(weather, plan["interleave"], datasets.stream_seed("depth", seed, task["name"]),
                                    refresh=plan.get("refresh", "default"))
                g = torch.Generator().manual_seed(datasets.stream_seed("depth", seed, task["name"]))
                cap1 = min(STATE_PER_CLIENT, cap)
                session = Session(device=device, plan=plan, queries=queries[:q], k=k, n=n, clients=1,
                                  spec=dict(serve_spec, clients=1, state_bytes=cap1), knobs=knobs,
                                  effort=point_efforts[points.index(diag_at)], generator=g, state_cap=cap1, deadline=deadline,
                                  table=table)
                return round(session.run(space["serve"])["read_wakes"] / q, 3)
            except TimeLimit:
                raise
            except Exception:
                return None
        rounds = read_rounds()
        if spec["score"] == "p99":
            pts, lo, hi = [(p["recall"], 1e6 / p["p99_us"]) for p in points if p["p99_us"] > 0 and tail_ok(p, tail)], 1e6 / bound["lat_lo"], 1e6 / bound["lat_hi"]
        else:
            pts, lo, hi = [(p["recall"], p["qps"]) for p in points if tail_ok(p, tail)], bound["qps_lo"], bound["qps_hi"]
        if tail is not None:
            record["tail_floor"] = tail
            record["tail_failed"] = sum(not tail_ok(p, tail) for p in points)  # operating points dropped by the floor
        if capped is not None:
            record["capped_at"] = round(capped, 4)  # first effort whose reads overflowed the in-flight cap
        norm, levels = area(pts, lo, hi)
        if compact:
            record["area"] = round(norm, 6)
            graded = scale.get("graded_floor", False)
            norm = speedup(pts, lo, graded=graded)
            if scale.get("levels"):  # the same score read at other recall levels
                record["speedups"] = {"%g" % lv: round(speedup(pts, lo, lv, graded), 6) for lv in scale["levels"]}
        record.update(norm=round(norm, 6), final=round(max(p["recall"] for p in points), 6), levels=[round(x, 4) for x in levels],
                      curve=curve, bounds={kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in bound.items()},
                      diag=dict(diag_at, **latency_ratios(diag_at, bound, geo, table), **({"read_rounds_per_query": rounds} if rounds is not None else {})), m=geo.m,
                      cost=cost_report(build_s, layout_s, index, plan, geo, data_bytes), profile=weather["profile"], clients=N, queries=total, cached=[hit_b, hit_l], synthetic=d["synthetic"],
                      weather={kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in weather.items()})
    except (SchemeError, TimeLimit) as exc:
        record["failure"] = type(exc).__name__ + ": " + str(exc)[:300]
    except Exception as exc:  # anything else the scheme's code raised
        record["failure"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
    record["seconds"] = round(time.monotonic() - started, 4)
    return record


def tail_ok(row, floor):
    """Tail-recall floor: the mean recall of the worst 5 % of queries must reach `floor` (None: no floor)."""
    return floor is None or row["recall_worst5"] >= floor


def deterministic_part(record):
    """The record without wall-clock fields (run seconds, cache hits, offline stage seconds): what two runs must agree on."""
    out = {k: v for k, v in record.items() if k not in ("seconds", "cached")}
    if isinstance(out.get("cost"), dict):
        out["cost"] = {k: v for k, v in out["cost"].items() if not k.endswith("_seconds")}
    return out


def cost_report(build_s, layout_s, index, plan, geo, data_bytes):
    """Report only (not scored): what the offline stages cost and how much of each memory the index takes. Seconds are wall
    time on the machine that built the entry, so they vary between machines; bytes are exact."""
    hbm = nbytes(plan["hbm"])
    return {"build_seconds": None if build_s is None else round(build_s, 3),
            "layout_seconds": None if layout_s is None else round(layout_s, 3),
            "index_bytes": nbytes(index), "hbf_bytes": int(plan["pages"].shape[0]) * PAGE, "hbm_bytes": hbm,
            "sram_bytes": nbytes(plan["sram"]), "data_bytes": data_bytes,
            "hbm_share_of_budget": round(hbm / max(geo.hbm_budget, 1), 4), "hbm_over_data": round(hbm / max(data_bytes, 1), 4),
            "hbf_over_data": round(int(plan["pages"].shape[0]) * PAGE / max(data_bytes, 1), 4)}


def latency_ratios(row, bound, geo, table):
    """Report only: the median latency in units of the fastest possible query on HBF (one read: the roof's latency) and of
    one HBM access (wake step + launch + HBM latency); the second says how close to an all-HBM query the scheme gets."""
    hbm_floor = table["step_us"] + table["launch_us"] + geo.hbm_latency
    return {"p50_over_hbf_floor": round(row["p50_us"] / bound["lat_hi"], 3),
            "p50_over_hbm_floor": round(row["p50_us"] / hbm_floor, 3)}


def measure(out, truth, k, device, weather, clients):
    import torch
    if any(r is None for r in out["results"]):
        raise SchemeError("not every query was answered")
    res = torch.stack(out["results"])
    hit = (truth[:, :, None] == res[:, None, :]).any(dim=2).sum(dim=1).double() / k
    recall = float(hit.mean())
    worst = float(hit.sort().values[: max(1, hit.numel() // 20)].mean())
    finish = torch.tensor(out["finish"], dtype=torch.float64)
    arrive = torch.tensor(out["arrive"], dtype=torch.float64)
    q = finish.numel()
    order = finish.sort().values
    lo, hi = q // 10, q - 1 - q // 10  # drop the first and last 10% (warm-up and drain)
    span = float(order[hi] - order[lo]) if hi > lo else float(order[-1])
    # The window is a steady state only when every client has run at least 2 queries; with fewer (a single
    # batch: all queries start at once, many finish in the same wake) it can shrink to nothing and report absurd qps
    # (PageANN at N = 512 with 400 queries: 980k qps one effort step after 4.8k). Then: queries / makespan.
    steady = q >= 2 * clients and hi > lo and span > 0
    # Refresh work still owed at the end is paid after it, so the run is stretched by the time that
    # takes: deferring refresh past the window buys no throughput.
    settle = device.settle()
    stretch = 1. + settle / max(out["end"], 1e-9)
    qps = (hi - lo) / (span * stretch) * 1e6 if steady else q / (max(float(order[-1]), 1e-9) + settle) * 1e6
    inside = (finish >= order[lo]) & (finish <= order[hi])
    lat = (finish - arrive)[inside] if bool(inside.any()) else finish - arrive
    p50, p99 = float(torch.quantile(lat, .5)), float(torch.quantile(lat, .99))
    total_span = max(out["end"], 1e-9)
    st = device.stats(total_span)
    senses = st["senses"]
    return {"recall": round(recall, 6), "recall_worst5": round(worst, 6), "qps": round(qps, 3), "steady": steady,
            "queries_per_client": round(q / max(1, clients), 3), "p50_us": round(p50, 4),
            "p99_us": round(p99, 4), "reads_per_query": round(st["requests"] / q, 3), "senses_per_query": round(senses / q, 3), "hit_rate": round(st["hits"] / max(st["requests"], 1), 4),
            "sense_efficiency": round(st["bytes"] / max(senses * PAGE, 1), 4), "bytes_per_query": round(st["bytes"] / q, 1),
            "unit_rho": round(st["unit_rho"], 3), "channel_rho": round(st["channel_rho"], 3),
            "busiest_unit_util": round(st["busiest_unit_util"], 4), "unit_util": round(st["unit_util"], 4),
            "channel_util": round(st["channel_util"], 4), "compute_share": round((out["compute_us"] + out["ctx_us"]) / total_span, 4),
            "wake_share": round(out["steps"] * out["step_us"] / total_span, 4),
            "steps": out["steps"], "syncs": out["ops"].get("sync", 0), "lane_syncs": out["ops"].get("lane_sync", 0),
            "lane_phases": out.get("lane_phases", 0), "lanes_per_phase": round(out.get("lanes", 0) / max(out.get("lane_phases", 0), 1), 2), "kernels": out["ops"].get("kernel", 0),
            "hbm_bytes_per_query": round(st["hbm_bytes"] / q, 1), "sram_bytes_per_query": round(st["sram_bytes"] / q, 1),
            "hbm_written_per_query": round(out.get("hbm_written", 0) / q, 1), "sram_written_per_query": round(out.get("sram_written", 0) / q, 1),
            "max_state": out["max_state"], "hottest_block_reads": st["hottest_block_reads"],
            "coredie_rho": round(st["coredie_rho"], 3), "coredie_util": round(st["coredie_util"], 4),
            "power_wait_share": round(st["power_wait_us"] / max(senses, 1) / max(weather["tR"], 1e-9), 4),
            "energy_uj_per_query": round(st["energy_uj"] / q, 3), "retries_per_query": round(st["retries"] / q, 4),
            "refresh_ops": st["refresh_ops"], "refresh_unit_share": round(st["refresh_unit_share"], 4),
            "max_debt": round(st["max_debt"], 3), "settle_us": round(settle, 2)}
