"""HBF checks of the device model: the profiles are self-consistent, the five assumptions SSD-era ANN designs make fail on
HBF the way they should, and the model reproduces what other HBF models report. Synthetic access sequences on the device
alone (no index, no scheme): seconds to run.

    python3 -m hbf_eval.hbfcheck --out runs/hbfcheck.json

Each check returns {"name", "measured", "expected", "holds", ...}. Model against model and model against spec, not against
silicon: what they lower is the risk that our model behaves unlike the HBF the literature describes.
"""
import argparse
import json
import math
import random

from .arch import Geometry
from .device import PAGE
from .tasks import PROFILES, profile_weather

FIRMWARE = {"policy": "lru", "tmr": 0., "mocs": 1 << 30, "sram_per_channel": 0, "disturb_threshold": 0., "hbm_fraction": .1}


def device(name, stacks=1, refresh="default", seed=1, **over):
    """One or more full stacks (16 channels each) of a profile, firmware fixed, no jitter, refresh off unless asked."""
    fw = dict(FIRMWARE)
    fw.update({k: v for k, v in over.items() if k in fw})
    w = profile_weather(name, 16, 8 - stacks, jitter=0., **fw)
    w.update({k: v for k, v in over.items() if k not in fw})
    g = Geometry(w, PAGE * 1024)
    return g.device(w, 1, seed, refresh=refresh), g, w


def one_channel(name, refresh="default", seed=1, **over):
    """A single channel (a 1/16 stack, power scaled with it), as at S scale."""
    fw = dict(FIRMWARE)
    fw.update({k: v for k, v in over.items() if k in fw})
    w = profile_weather(name, 1, 7, jitter=0., **fw)
    w.update({k: v for k, v in over.items() if k not in fw})
    g = Geometry(w, PAGE * 1024)
    return g.device(w, 1, seed, refresh=refresh), g, w


def drain(d, t):
    t = d.next_wake(t, count=1 << 60)
    return dict((r, time) for time, r in d.take_done())


def check(name, measured, expected, holds, **extra):
    return dict(name=name, measured=measured, expected=expected, holds=bool(holds), **extra)


# ---- §9.1: the profiles are self-consistent ----
def rates(name):
    """Full-page senses per us per channel allowed by the structure, the power budget, the core-die channels, the link."""
    p = PROFILES[name]
    bw = p["channel_GBps"] * 1e3
    return {"structure": p["dies"] * p["banks_per_die"] / p["tR"], "power": 30. / 16 / p["e_sense"],
            "coredie": bw / PAGE, "link": bw / PAGE}


def sustained(name, rounds=32):
    """Sequential full-page reads over one stack (4 KiB x every unit x rounds, all issued at once, Batch Read): the
    bandwidth after the power burst, in GB/s per stack."""
    d, g, _ = device(name)
    n = g.channels * g.units * rounds
    d.submit(0., list(range(n)), [0] * n, [PAGE] * n, list(range(n)), True)
    done = sorted(drain(d, 0.).values())
    lo, hi = n // 2, n - 1  # skip the burst the bucket allows (profile C: about 7.7 rounds of every unit)
    return (hi - lo) * PAGE / (done[hi] - done[lo]) / 1e3


def profiles():
    out = []
    for name in sorted(PROFILES):
        r = rates(name)
        bound = min(r.values()) * PAGE * 16 / 1e3
        measured = sustained(name)
        binds = min(r, key=r.get)
        out.append(check("profile %s: sustained sequential bandwidth = the binding limit (%s)" % (name, binds),
                         round(measured, 1), round(bound, 1), abs(measured / bound - 1) < .05, unit="GB/s per stack",
                         rates={k: round(v, 2) for k, v in r.items()}))
    r = rates("C")
    out.append(check("profile C: power binds first, about half the banks can sense at once", round(r["power"] / r["structure"], 3),
                     "0.4-0.5", .4 <= r["power"] / r["structure"] <= .5))
    for name in ("A", "D"):
        r = rates(name)
        spread = max(r.values()) / min(r.values())
        out.append(check("profile %s: structure, power and link within 2x" % name, round(spread, 3), "<= 2", spread <= 2))
    return out


def ocp_13_1():
    """OCP §13.1: sequential 4 KiB x N x 16 (one page per unit of a stack) completes in about one read latency on a
    profile whose power budget allows it (D)."""
    d, g, w = device("D")
    n = g.channels * g.units
    d.submit(0., list(range(n)), [0] * n, [PAGE] * n, list(range(n)), True)
    end = max(drain(d, 0.).values())
    floor = w["tR"] + PAGE / g.coredie_bw + PAGE / g.channel_bw  # one sense, one core-die move, one transfer
    per_unit_transfer = g.units * PAGE / g.channel_bw  # the link must still carry a page per unit
    return check("OCP §13.1: one page per unit of a stack (profile D) in about tR + the link time", round(end, 3),
                 round(w["tR"] + per_unit_transfer, 3), end <= 1.1 * (w["tR"] + per_unit_transfer) and end >= floor,
                 unit="us")


def haven():
    """HAVEN: 460 GB/s sustainable per stack at 30 W. Profile C (its energy) read sequentially must land on that."""
    measured = sustained("C")
    return check("HAVEN: profile C sustains about 460 GB/s per stack (power-bound)", round(measured, 1), 460.,
                 abs(measured / 460. - 1) < .05, unit="GB/s per stack")


# ---- §9.2: the SSD assumptions fail on HBF ----
def birthday(w, banks):
    p = 1.
    for i in range(w):
        p *= 1. - i / banks
    return 1. - p


def a1(name="C", widths=(8, 16), trials=2000):
    """A1 (SSD: a round of W reads costs one read latency). On HBF a round of W random reads finishes when the busiest
    bank does: with W = 8 over 64 banks, 37% of the rounds collide."""
    out = []
    for W in widths:
        d, g, w = one_channel(name)
        rng, t, slow = random.Random(W), 0., 0
        for _ in range(trials):
            pages = rng.sample(range(g.units * 400), W)
            d.submit(t, pages, [0] * W, [64] * W, pages, True)
            end = max(drain(d, t).values())
            slow += end - t > 1.5 * w["tR"]
            t = end + 10 * w["tR"]
        p = birthday(W, g.units)
        out.append(check("A1: W = %d random reads over %d banks: rounds that take two tR or more" % (W, g.units),
                         round(slow / trials, 3), round(p, 3), abs(slow / trials - p) < .04))
    return out


def a2(name="C"):
    """A2 (SSD: a page read twice hits the cache). On HBF the 2 page buffers of a bank keep a page only until 2 other
    pages of that bank are read in between."""
    out = []
    for between in (0, 1, 2, 4):
        d, g, _ = one_channel(name)
        t = 0.
        seq = [0] + [g.units * (i + 1) for i in range(between)] + [0]  # the same bank: pages 0, U, 2U, ...
        for page in seq:
            d.submit(t, [page], [0], [64], [page], True)
            t = max(drain(d, t).values())
        hit = d.hits == 1
        out.append(check("A2: a page read again after %d other pages of its bank: a buffer hit" % between, hit,
                         between <= 1, hit == (between <= 1)))
    return out


def rounds_of(d, groups, w, gap):
    t, slow, total = 0., 0, 0.
    for pages in groups:
        d.submit(t, pages, [0] * len(pages), [64] * len(pages), pages, True)
        end = max(drain(d, t).values())
        slow += end - t > 1.5 * w["tR"]
        total += end - t
        t = end + gap
    return slow / len(groups), total / len(groups) / w["tR"]


def a3(name="C", W=16, groups=500):
    """A3 (SSD: which page a node lands in matters, not which bank). On HBF the same pages numbered so that the pages
    read together sit on distinct banks (a colouring) avoid the collisions of a random numbering."""
    d, g, w = one_channel(name)
    rng = random.Random(3)
    sets = [rng.sample(range(g.units * 1000), W) for _ in range(groups)]  # random numbering
    coloured = [[(i * 7 + j) % g.units + g.units * rng.randrange(1000) for j in range(W)] for i in range(groups)]
    slow_random, mean_random = rounds_of(d, sets, w, 10 * w["tR"])
    d2, _, _ = one_channel(name)
    slow_col, mean_col = rounds_of(d2, coloured, w, 10 * w["tR"])
    return [check("A3: rounds of %d co-read pages, random numbering: rounds that collide" % W, round(slow_random, 3),
                  round(birthday(W, g.units), 3), abs(slow_random - birthday(W, g.units)) < .05),
            check("A3: the same rounds coloured onto distinct banks: rounds that collide", round(slow_col, 3), 0., slow_col == 0.,
                  mean_round_tR={"random": round(mean_random, 3), "coloured": round(mean_col, 3)})]


def a4(name="C", k=3):
    """A4 (SSD: speculative reads are free when the device is not busy). On HBF a speculative read queued ahead of the
    critical read on its bank delays it by a tR each (strict order)."""
    lat = {}
    for first in ("speculative", "critical"):
        d, g, w = one_channel(name)
        spec = [g.units * (i + 1) for i in range(k)]  # bank 0
        order = spec + [0] if first == "speculative" else [0] + spec
        ids = d.submit(0., order, [0] * len(order), [64] * len(order), order, True)
        done = drain(d, 0.)
        lat[first] = done[ids[order.index(0)]]
    diff = (lat["speculative"] - lat["critical"]) / w["tR"]
    return check("A4: %d speculative reads on the bank ahead of the critical one delay it by k tR" % k, round(diff, 3), k,
                 abs(diff - k) < .1, latency_us={k_: round(v, 3) for k_, v in lat.items()})


def closed_loop(d, g, w, clients, W, extra, rounds_each, seed=5):
    """N clients, each reading W random critical pages plus `extra` speculative ones per round and waiting for its
    critical pages; returns (rounds per us after warm-up, mean round time in us)."""
    rng = random.Random(seed)
    space = g.channels * g.units * 400
    need, started, t = {}, {}, 0.
    times, finished = [], []
    tag = 0

    def start(c, t):
        nonlocal tag
        pages = rng.sample(range(space), W + extra)
        tags = list(range(tag, tag + W + extra))
        tag += W + extra
        d.submit(t, pages, [0] * len(pages), [64] * len(pages), tags, True)
        need[c] = set(tags[:W])
        started[c] = t
        for x in tags[:W]:
            owner[x] = c

    owner = {}
    for c in range(clients):
        start(c, 0.)
    count = {c: 0 for c in range(clients)}
    while any(v < rounds_each for v in count.values()):
        t = d.next_wake(t, count=1)
        for time, r in d.take_done():
            x = d.r_tag[r]
            c = owner.pop(x, None)
            if c is None:
                continue
            need[c].discard(x)
            if not need[c]:
                times.append(time - started[c])
                finished.append(time)
                count[c] += 1
                if count[c] < rounds_each:
                    start(c, time)
    finished.sort()
    lo, hi = len(finished) // 4, len(finished) - 1
    return (hi - lo) / (finished[hi] - finished[lo]), sum(times) / len(times)


def a5(W=8, extra=4):
    """A5 (SSD: extra reads only use spare bandwidth). On HBF, under load, 50% speculative reads take power (profile C)
    or bank time (D) from the critical reads: throughput drops by about 1/3; alone, they are nearly free."""
    out = []
    for name in ("C", "D"):
        res = {}
        for spec in (0, extra):
            d, g, w = one_channel(name)
            res[spec] = closed_loop(d, g, w, clients=4 * g.units, W=W, extra=spec, rounds_each=12)
        drop = 1. - res[extra][0] / res[0][0]
        out.append(check("A5: profile %s, high load: throughput lost to %d%% speculative reads" % (name, 100 * extra // W),
                         round(drop, 3), round(extra / (W + extra), 3), abs(drop - extra / (W + extra)) < .07))
    d, g, w = one_channel("C")
    alone = closed_loop(d, g, w, clients=1, W=W, extra=0, rounds_each=200)[1]
    d, g, w = one_channel("C")
    alone_spec = closed_loop(d, g, w, clients=1, W=W, extra=extra, rounds_each=200)[1]
    out.append(check("A5: profile C, one client: round latency with speculative reads / without", round(alone_spec / alone, 3),
                     "about 1 (at most the extra collisions)", alone_spec / alone < 1.25))
    return out


def hot_die_run(name, policy, arrivals, X=4000.):
    """Single 64 B reads at the given times, half of them on die 0 (the hot die); refresh paid at once (the moment a plane
    owes more than a page, even with reads queued) or in idle time (the stock policy, slack 8 pages). Returns latencies."""
    refresh, d_max = {"at once": ("scheme", .5), "idle": ("default", 8.)}[policy]
    d, g, w = one_channel(name, refresh=refresh, disturb_threshold=X, d_max=d_max, retry_p=0.)
    rng = random.Random(9)
    hot = [p for p in range(g.units * 50) if (p % g.units) // g.banks == 0]
    ids = {}
    for i, t in enumerate(arrivals):
        page = rng.choice(hot) if rng.random() < .5 else rng.randrange(g.units * 50)
        ids[d.submit(t, [page], [0], [64], [i], True)[0]] = t
    done = drain(d, arrivals[-1])
    lat = sorted(done[r] - t for r, t in ids.items())
    return lat, d.stats(1.)["refresh_ops"]


def refresh_timing(name="C"):
    """Refresh on a hot die paid at once against paid in idle time. Read disturb is accelerated
    (X = 4000) so that refresh takes a visible share of the hot die within the run.
    - Long idle gaps (bursts of reads 2 ms apart, the die idle in between for much longer than a program): paying in idle
      time keeps the programs out of the bursts, so the reads' tail drops by about a program.
    - Short gaps only (steady random arrivals, a few us apart): no idle gap fits a 200 us program, so paying in idle time
      cannot hide it (HBFlex fig. 8: short compute gaps cannot hide a full program); the two tails are the same.
    The first effect needs the long gaps; the second check records that."""
    t_prog = PROFILES[name]["t_prog"]
    out = []
    rng = random.Random(4)
    bursts = [2000. * b + .5 * i for b in range(60) for i in range(40)]
    steady, t = [], 0.
    for _ in range(4000):
        t += rng.expovariate(1. / 2.5)
        steady.append(t)
    for label, arrivals in (("long idle gaps", bursts), ("short gaps only", steady)):
        res = {}
        for policy in ("at once", "idle"):
            lat, ops = hot_die_run(name, policy, arrivals)
            res[policy] = {"p99": lat[int(.99 * len(lat))], "p90": lat[int(.9 * len(lat))], "refresh_ops": ops}
        diff = res["at once"]["p99"] - res["idle"]["p99"]
        detail = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in res.items()}
        if label == "long idle gaps":
            out.append(check("refresh on a hot die, %s: p99, paid at once minus paid in idle time" % label, round(diff, 1),
                             "about one program (%g us)" % t_prog, diff > .5 * t_prog, detail=detail, unit="us"))
        else:
            out.append(check("refresh on a hot die, %s: p99, paid at once minus paid in idle time" % label, round(diff, 1),
                             "about 0: no gap hides a program (HBFlex fig. 8)", abs(diff) < .1 * t_prog, detail=detail,
                             unit="us"))
    return out


# ---- §9.3: other HBF models ----
def hbflex_rounds(trials=20):
    """HBFlex §3.1: a 12 MiB read is 3072 pages, one per plane under ideal placement; placed at random, collisions force
    extra read waves (HBFlex reports 5.97x averaged over its workloads). The model: 3072 random pages over 3072 planes
    (profile B's structure: 96 channels x 32 planes) finish after the busiest plane's waves."""
    waves = []
    for i in range(trials):
        d, g, w = device("B", stacks=6, seed=i)
        rng = random.Random(100 + i)
        planes = g.channels * g.units
        pages = [rng.randrange(planes * 64) for _ in range(planes)]
        d.submit(0., pages, [0] * planes, [64] * planes, list(range(planes)), True)
        end = max(drain(d, 0.).values())
        waves.append(end / w["tR"])
    mean = sum(waves) / trials
    return check("HBFlex: 3072 random pages over 3072 planes, completion in tR (HBFlex: 5.97x on average)", round(mean, 2),
                 "about 6 (balls into bins: 6.09)", 5. <= mean <= 7.5)


def hbflex_program():
    """HBFlex fig. 8: a 75 us program cannot be preempted: the next read to its plane waits for it, other planes do not."""
    d, g, w = one_channel("B", refresh="scheme", disturb_threshold=4., pages_per_block=4)
    d.submit(0., [0], [0], [64], [0], True)
    drain(d, 0.)  # plane 0 owes a page
    t0 = 20.
    d.refresh(t0, 0, 1)  # the program holds plane 0 from t0 for tR + 75 us
    a = d.submit(t0 + 1., [g.units], [0], [64], [1], True)[0]  # plane 0
    b = d.submit(t0 + 1., [1], [0], [64], [2], True)[0]  # plane 1
    done = drain(d, t0 + 1.)
    held = done[a] - (t0 + 1.)
    free = done[b] - (t0 + 1.)
    op = w["tR"] + w["t_prog"]
    return check("HBFlex: a read behind a program on its plane waits for it (profile B, 75 us)", round(held, 2),
                 round(op - 1. + w["tR"], 2), abs(held - (op - 1. + w["tR"])) < 1. and free < 2 * w["tR"],
                 other_plane_us=round(free, 2), unit="us")


def run_all():
    out = profiles() + [ocp_13_1(), haven()] + a1() + a2() + a3() + [a4()] + a5() + refresh_timing() + [hbflex_rounds(), hbflex_program()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = run_all()
    for r in rows:
        print(("HOLDS  " if r["holds"] else "FAILS  ") + r["name"] + ": " + str(r["measured"]) + " (expected " + str(r["expected"]) + ")")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1, default=str)


if __name__ == "__main__":
    main()
