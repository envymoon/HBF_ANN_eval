"""Run reference schemes (or your own) on the compact suite and print the speedup table.

    python3 -m hbf_eval.bench list
    python3 -m hbf_eval.bench run --scale 100k --data clustered --workers 4
    python3 -m hbf_eval.bench run --schemes baseline,haven --scale 100k --data flat --wide
    python3 -m hbf_eval.bench check path/to/my_scheme.py
    python3 -m hbf_eval.bench run --schemes path/to/my_scheme.py --set nprobe_hi=96 --set nlist=2048 --mem-gb 8
    python3 -m hbf_eval.bench report
    python3 -m hbf_eval.bench compare results/results.jsonl reference/results_100k.jsonl

A result is log2(throughput of the scheme at recall R / throughput of batched brute force on the same device) per task, read at
R = 0.90, 0.95 and 0.99. An operating point counts only if the worst 5 % of its queries reach recall 0.3; the effort
interval around each level's crossing is bisected twice; a level the scheme does not reach scores below the floor (-8) by its
recall shortfall (in (-16, -8]). The score of a scheme is the mean over the four
tasks (high_load, low_load_4, power_bound, data_deep) at one seed. Data variants: "clustered" (SIFT-like and DEEP-like
stand-ins), "flat" (the same shapes, weakly clustered), and a suffix _mK sets the number of HBM positions of the platform
(e.g. clustered_m1). "--wide" opens every *_hi knob (the upper end of an effort range) to its limit.
Your own scheme: --schemes path/to/my_scheme.py (see scheme.py for the format).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = [.9, .95, .99]
BASE = {"cpp": 1, "pool": 3072, "k": 10, "efforts": 9, "total_min": 400, "total_max": 1 << 20, "total_per_client": 2,
        "mode": "speedup", "n": 100000, "levels": LEVELS,
        # an operating point counts only if the worst 5 % of its queries reach recall 0.3; two bisection steps around each
        # level's crossing; a level not reached scores below -8 by the recall shortfall
        "tail_floor": .3, "refine": 2, "graded_floor": True}
SCALES = {"100k": BASE, "1m": dict(BASE, n=1000000, cpp=2)}
DATA = {"clustered": {}, "flat": {"sift": "sift_flat", "deep": "deep_flat"}}
SEED = 2000000
TIME_LIMIT = {"100k": 3600, "1m": 7200}
ADDRESS_SPACE = {"100k": 12 << 30, "1m": 14 << 30}
RESIDENT = {"100k": None, "1m": 10 << 30}  # resident-memory budget per scheme process; over it the run stops as "memory over budget"
PUBLISHED = ("diskann", "starling", "pipeann", "pageann", "haven")


def find(name):
    from . import schemes
    return name if name.endswith(".py") else str(schemes.path(name))


def label_of(name):
    return os.path.basename(name)[:-3] if name.endswith(".py") else name


def parse_sets(items):
    out = {}
    for item in items or []:
        name, eq, value = item.partition("=")
        if not eq:
            sys.exit("--set expects name=value, got %r" % item)
        out[name.strip()] = float(value)
    return out


def tag(name, wide, sets, device=None):
    return (label_of(name) + ("@wide" if wide else "") + "".join("@%s=%g" % kv for kv in sorted(sets.items()))
            + ("@fixed" + device if device else ""))


def job(name, data, scale_name, cache, wide=False, data_root=None, sets=None, mem_gb=None, device=None):
    limit = int(mem_gb * (1 << 30)) if mem_gb else ADDRESS_SPACE[scale_name]
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    import torch
    torch.set_num_threads(1)
    from . import runner, scheme
    from .tasks import suite
    sch = scheme.load(find(name))
    over = {n: s["high"] for n, s in sch.knobs.items() if n.endswith("_hi")} if wide else {}
    unknown = sorted(set(sets or {}) - set(sch.knobs))
    if unknown:
        sys.exit("unknown knob(s) %s; declared: %s" % (unknown, sorted(sch.knobs)))
    over.update(sets or {})
    label = tag(name, wide, sets or {}, device)
    knobs, space = sch.knob_values(over), sch.namespace()
    scale = SCALES[scale_name]
    base, _, mm = data.partition("_m")
    tasks = []
    for t in suite("main", scale):
        t["spec"]["split"] = "main"
        t["spec"]["dataset"] = DATA[base].get(t["spec"]["dataset"], t["spec"]["dataset"])
        if mm:
            t["spec"]["m"] = int(mm)
        if device:  # fixed device: the firmware values of tasks.FIXED_FIRMWARE, no jitter, profile `device` unless the task pins one
            t["spec"]["weather"] = dict({"profiles": [device]}, **t["spec"]["weather"], fixed=True)
        rec = runner.evaluate(sch, knobs, t, SEED, {"scale": scale, "data_root": data_root, "cache": cache},
                              time.monotonic() + TIME_LIMIT[scale_name], space)
        keep = {k: rec.get(k) for k in ("task", "failure", "norm", "speedups", "final", "curve", "bounds", "profile", "seconds",
                                         "cost", "tail_failed")}
        keep["diag"] = {k: rec["diag"].get(k) for k in ("recall", "qps", "p50_us", "reads_per_query", "senses_per_query", "hit_rate",
                                                        "compute_share", "wake_share", "unit_util", "power_wait_share",
                                                        "recall_worst5", "read_rounds_per_query", "p50_over_hbf_floor",
                                                        "p50_over_hbm_floor")} if rec.get("diag") else None
        keep["dataset"] = t["spec"]["dataset"]
        tasks.append(keep)
        print(json.dumps({"label": label, "data": data, "scale": scale_name, "task": t["name"],
                          "failure": rec["failure"], "speedups": rec.get("speedups"), "seconds": rec["seconds"]}), file=sys.stderr, flush=True)
    return {"label": label, "data": data, "scale": scale_name, "tasks": tasks}


def check(a):
    """Load a scheme file through the same checks a run applies and print what the evaluator sees. Runs nothing."""
    from . import scheme
    sch = scheme.load(find(a.new))
    print("ok:", a.new)
    print("functions:", ", ".join(sorted(sch.functions)))
    print("knobs (default, low, high, scale):")
    for n, k in sch.knobs.items():
        print("  %-18s %-12g %-12g %-12g %s" % (n, k["default"], k["low"], k["high"], k["scale"]))
    for entry, key in sch.stage_keys(sch.knob_values()).items():
        print("%s reads knobs: %s" % (entry, ", ".join(key["knobs"]) or "-"))


def resident(pid):
    """Resident bytes of `pid` and its descendants (Linux /proc; 0 where /proc is missing)."""
    children, rss = {}, {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry) as f:
                fields = f.read().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(entry))
            rss[int(entry)] = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            continue
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += children.get(p, [])
    return total


def failed(key, status, failure):
    return json.dumps({"label": key[0], "data": key[1], "scale": key[2], "status": status, "failure": failure, "tasks": []})


def results(out):
    path = os.path.join(out, "results.jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def run(a):
    os.makedirs(a.out, exist_ok=True)
    done = {(r["label"], r["data"], r["scale"]) for r in results(a.out)}
    names = a.schemes.split(",") if a.schemes else ["baseline", *PUBLISHED]
    sets = parse_sets(a.set)
    a.fixed_device = getattr(a, "fixed_device", None)
    from . import scheme
    for n in names:  # fail before any process starts: bad file, bad knob name
        known = scheme.load(find(n)).knobs
        if set(sets) - set(known):
            sys.exit("%s: unknown knob(s) %s; declared: %s" % (n, sorted(set(sets) - set(known)), sorted(known)))
    todo = [(n, d, s) for s in a.scale.split(",") for d in a.data.split(",") for n in names if (tag(n, a.wide, sets, a.fixed_device), d, s) not in done]
    procs, log = [], open(os.path.join(a.out, "driver.log"), "a")
    while todo or procs:
        while todo and len(procs) < a.workers:
            n, d, s = todo.pop(0)
            err = open(os.path.join(a.out, "stderr_%s_%s_%s.log" % (tag(n, a.wide, sets, a.fixed_device), d, s)), "w")
            cmd = [sys.executable, "-m", "hbf_eval.bench", "one", "--scheme", n, "--data", d, "--scale", s, "--cache", a.cache]
            cmd += ["--data-root", a.data_root] if a.data_root else []
            cmd += ["--mem-gb", str(a.mem_gb)] if a.mem_gb else []
            cmd += ["--fixed-device", a.fixed_device] if a.fixed_device else []
            for item in a.set or []:
                cmd += ["--set", item]
            p = subprocess.Popen(cmd + (["--wide"] if a.wide else []), cwd=ROOT, stdout=subprocess.PIPE, stderr=err, text=True)
            budget = int(a.rss_gb * (1 << 30)) if a.rss_gb else RESIDENT[s]
            procs.append(((tag(n, a.wide, sets, a.fixed_device), d, s), p, time.time(), budget, {"peak": 0, "over": False}))
            print(time.strftime("%F %T"), "start", *procs[-1][0], file=log, flush=True)
        time.sleep(1)
        for item in list(procs):
            key, p, t0, budget, mem = item
            if p.poll() is None:
                if budget:
                    used = resident(p.pid)
                    mem["peak"] = max(mem["peak"], used)
                    if used > budget:
                        mem["over"] = True
                        p.kill()
                        p.wait()
                if p.poll() is None:
                    continue
            procs.remove(item)
            lines = p.stdout.read().strip().splitlines()
            if mem["over"]:
                line = failed(key, p.returncode, "memory over budget: resident memory went over %.1f GB; the run was stopped" % (budget / 2 ** 30))
            elif p.returncode in (-9, 137):
                line = failed(key, p.returncode, "killed by the system (exit %d), most likely out of memory: peak resident memory "
                                                  "seen %.1f GB" % (p.returncode, mem["peak"] / 2 ** 30))
            elif p.returncode == 0 and lines:
                line = lines[-1]
            else:
                line = failed(key, p.returncode, None)
            with open(os.path.join(a.out, "results.jsonl"), "a") as f:
                f.write(line + "\n")
            print(time.strftime("%F %T"), "done", *key, p.returncode, round(time.time() - t0), "s", file=log, flush=True)


def table(rows):
    out = {}
    for r in rows:
        if len(r.get("tasks", [])) != 4:
            continue
        per = {"tasks": {t["task"]: t.get("speedups") for t in r["tasks"]}}
        for lv in LEVELS:
            vals = [t["speedups"]["%g" % lv] if t.get("speedups") else -8. for t in r["tasks"]]
            per["%g" % lv] = round(sum(vals) / 4, 3)
        out.setdefault("%s|%s" % (r["data"], r["scale"]), {})[r["label"]] = per
    return out


def report(a):
    for key, t in sorted(table(results(a.out)).items()):
        print("==", key)
        for l, v in sorted(t.items(), key=lambda x: -x[1]["0.9"]):
            print("  %-16s  0.90 %7.3f  0.95 %7.3f  0.99 %7.3f" % (l, v["0.9"], v["0.95"], v["0.99"]))


def compare(a):
    """Two result files: every shared (scheme, data, scale, task) must agree exactly on speedups, curve and bounds."""
    load = lambda p: {(r["label"], r["data"], r["scale"]): r for r in map(json.loads, open(p)) if r.get("tasks")}
    new, old = load(a.new), load(a.old)
    bad = n = 0
    for key in sorted(set(new) & set(old)):
        for tn, to in zip(new[key]["tasks"], old[key]["tasks"]):
            for field in ("task", "speedups", "norm", "final", "curve", "bounds", "profile", "failure"):
                n += 1
                if tn.get(field) != to.get(field):
                    bad += 1
                    print("DIFF", key, tn["task"], field, "new", str(tn.get(field))[:140], "old", str(to.get(field))[:140])
    print("compared %d schemes, %d fields, %d differences" % (len(set(new) & set(old)), n, bad))
    sys.exit(1 if bad or not n else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "one", "report", "list", "compare", "check"))
    ap.add_argument("new", nargs="?")
    ap.add_argument("old", nargs="?")
    ap.add_argument("--scheme")
    ap.add_argument("--schemes")
    ap.add_argument("--data", default="clustered")
    ap.add_argument("--scale", default="100k")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--set", action="append", metavar="KNOB=VALUE", help="override a knob (repeatable); clamped to its declared range")
    ap.add_argument("--mem-gb", type=float, default=None, help="address-space limit per scheme process in GiB (default 12 at 100k, 14 at 1m)")
    ap.add_argument("--rss-gb", type=float, default=None,
                    help="resident-memory budget per scheme process in GiB; over it the run stops and is recorded as "
                         "'memory over budget' (default: none at 100k, 10 at 1m)")
    ap.add_argument("--fixed-device", choices=("A", "B", "C", "D"), default=None,
                    help="fixed device: firmware values of tasks.FIXED_FIRMWARE instead of a per-seed draw, no tR jitter, and this "
                         "product profile for every task that does not pin its own (power_bound keeps C)")
    ap.add_argument("--data-root", default=None, help="folder of prepared datasets (see data.py); default: synthetic stand-ins")
    a = ap.parse_args()
    if a.cmd == "one":
        print(json.dumps(job(a.scheme, a.data, a.scale, a.cache, a.wide, a.data_root, parse_sets(a.set), a.mem_gb, a.fixed_device)))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "list":
        from . import schemes
        print("\n".join(schemes.NAMES))
    elif a.cmd == "compare":
        compare(a)
    elif a.cmd == "check":
        check(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
