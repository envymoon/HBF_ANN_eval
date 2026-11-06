"""Tasks and weather (one draw per seed and task).

A task fixes the ability class, the dataset, the load rule, the platform (m: how many of the 8 positions are HBM) and any
weather overrides. The device is one of four self-consistent product profiles (the hardware is fixed, not drawn
parameter by parameter); a seed picks the profile (rotating through the four across the tasks of a seed, unless the task
pins one) and draws the firmware details a host does not know: the buffer policy, the base-die timer, the in-flight cap,
the SRAM size, the read-disturb threshold, the HBM capacity factor, and the tR jitter. Two runs with one (task, seed) are
identical.
"""
import math
import random

from .data import stream_seed

CLASSES = ("high_load", "low_load", "tail", "bottleneck_ends", "tR_ends", "data", "skew", "scale")
LEVELS = (.80, .90, .95, .99)

# ---- the product profiles, per channel; a stack has 16 channels ----
# tR in us; link in GB/s per channel (OCP link tiers 1, 2, 3: 384 GB/s, 1.536 and 3.072 TB/s per stack); t_prog in us;
# scope: what a program holds (the die, or only its plane); e_sense in uJ per 4 KiB sense.
PROFILES = {
    # slow medium (HBF-Sim, Characterizing HBF): structure, power and link all about 6-7 full-page senses/us per channel
    "A": {"tR": 20., "dies": 4, "banks_per_die": 32, "coredie": 2, "channel_GBps": 24., "t_prog": 200., "scope": "die",
          "e_sense": .27},
    # the industry configuration of HBFlex: one die of 32 planes per channel, programs hold only their plane
    "B": {"tR": 4., "dies": 1, "banks_per_die": 32, "coredie": 1, "channel_GBps": 96., "t_prog": 75., "scope": "plane",
          "e_sense": .22},
    # the OCP §4.6 example structure at the top link tier, with the energy of HAVEN / Characterizing HBF: power binds first
    "C": {"tR": 4., "dies": 4, "banks_per_die": 16, "coredie": 2, "channel_GBps": 192., "t_prog": 200., "scope": "die",
          "e_sense": .27},
    # the OCP top tier taken literally: tR and energy low enough to feed 3.072 TB/s within 30 W (about 1.2 pJ/bit)
    "D": {"tR": 1.5, "dies": 4, "banks_per_die": 16, "coredie": 2, "channel_GBps": 192., "t_prog": 75., "scope": "die",
          "e_sense": .039},
}
ROTATION = ("A", "B", "C", "D")
STACK_POWER_W = 30.  # per stack (HAVEN; about an HBM stack's package power)
E_PROG = 3.28  # uJ per 4 KiB page programmed (100 pJ/bit, Characterizing HBF)
BURST_TR = 10.  # power bucket depth: P x 10 tR
D_MAX = 4.  # pages of refresh a plane may owe before reads retry (and, under the scheme's own policy, 2x before forced)
RETRY_P = .2  # retry probability of a sense on a plane owing more than D_MAX (direction only: no public HBF data)
M_DEFAULT = 4  # HBM positions of the platform

# ---- firmware details drawn per seed ----
JITTER = .1
POLICIES = {"main": ("lru", "fifo"), "transfer": ("lru", "fifo"), "unseen": ("random",)}
TMR = (0., 2.)  # us
MOCS = (256, 16384)  # log-uniform
SRAM = (0, 2 << 20)  # bytes per channel, in 64 KiB steps
DISTURB = (1e5, 1e6)  # reads per block before a refresh, log-uniform 
HBM_FRACTION = (.1, .5)
HBM_STACK_BW = 1e6  # bytes/us (1 TB/s per stack, Characterizing HBF Table 1)
HBM_LATENCY, SRAM_LATENCY = .1, .05  # us
PAGES_PER_BLOCK = 256

# Fixed devices for the SSD comparison checks, not drawn: every seed gets the same device, with none of the
# extra mechanisms (no power budget, no core-die channels, no refresh).
# ssd: an NVMe SSD (PCIe 4 x4 class): 8 channels x 16 parallel units, tR 80 us (sense plus controller), 7 GB/s in total,
#      no base-die timer, a deep queue; "HBM" plays the host DRAM (the PQ codes, the node cache).
# memory: everything as fast as GPU memory: tR 0.1 us, 128 units per channel, 1.5 TB/s at S scale (3 TB/s per stack).
PRESETS = {
    "ssd": {"tR": 80., "units_per_channel": 16, "stack_bw": 16 * 875., "policy": "lru", "tmr": 0., "mocs": 1024,
            "sram_per_channel": 0, "hbm_fraction": .5},
    "memory": {"tR": .1, "units_per_channel": 128, "stack_bw": 3.072e6, "policy": "lru", "tmr": 0., "mocs": 1 << 20,
               "sram_per_channel": 0, "hbm_fraction": .5},
}
NO_V2 = {"banks_per_die": 16, "coredie": 0, "stack_power_w": 0., "e_sense": 0., "e_prog": 0., "t_prog": 0., "scope": "die",
         "disturb_threshold": 0., "d_max": math.inf, "retry_p": 0.}


def log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def profile_of(seed, spec):
    """The product profile of a (seed, task): pinned by the task, else rotating through the four across a seed's tasks."""
    over = spec.get("weather", {})
    if "profiles" in over:
        names = over["profiles"]
        return names[random.Random(stream_seed("profile", seed, spec["name"])).randrange(len(names))]
    return ROTATION[(seed + sum(map(ord, spec["name"]))) % len(ROTATION)]


def draw_weather(seed, spec, split, cpp):
    """The device for one (seed, task): a product profile plus the firmware details drawn for this seed."""
    rng = random.Random(stream_seed("weather", seed))  # the firmware draws: one per seed, shared by every task of the seed
    firmware = {"policy": rng.choice(POLICIES[split]), "tmr": rng.uniform(*TMR), "mocs": int(round(log_uniform(rng, *MOCS))),
                "sram_per_channel": 65536 * rng.randrange(SRAM[1] // 65536 + 1), "disturb_threshold": log_uniform(rng, *DISTURB),
                "hbm_fraction": rng.uniform(*HBM_FRACTION)}
    over = spec.get("weather", {})
    if "preset" in over:
        w = dict(PRESETS[over["preset"]], **NO_V2, profile=over["preset"], jitter=0.)
        return finish(w, cpp, int(spec.get("m", M_DEFAULT)))
    return profile_weather(profile_of(seed, spec), cpp, int(spec.get("m", M_DEFAULT)), **firmware)


def profile_weather(name, cpp, m, jitter=JITTER, **firmware):
    """The weather of product profile `name` with the given firmware details (what draw_weather draws; the HBF checks
    pass fixed ones)."""
    p = PROFILES[name]
    w = dict(firmware, profile=name, tR=p["tR"], units_per_channel=p["dies"] * p["banks_per_die"],
             banks_per_die=p["banks_per_die"], coredie=p["coredie"], stack_bw=p["channel_GBps"] * 1e3 * 16,
             stack_power_w=STACK_POWER_W, e_sense=p["e_sense"], e_prog=E_PROG, t_prog=p["t_prog"], scope=p["scope"],
             d_max=D_MAX, retry_p=RETRY_P, jitter=jitter)
    return finish(w, cpp, m)


def finish(w, cpp, m):
    w["tR_nominal"] = w["tR"]
    w.update(hbm_stack_bw=HBM_STACK_BW, hbm_latency=HBM_LATENCY, sram_latency=SRAM_LATENCY, pages_per_block=PAGES_PER_BLOCK,
             channels_per_position=cpp, m=m)
    return w


def clients(spec, weather):
    """N, fixed by the task: per parallel unit of the HBF positions, or an absolute count."""
    kind, value = spec["load"]
    if kind == "count":
        return int(value)
    units = (8 - weather["m"]) * weather["channels_per_position"] * weather["units_per_channel"]
    return max(1, int(round(value * units)))


def total_queries(spec, n_clients, scale):
    return int(min(scale["total_max"], max(scale["total_min"], scale["total_per_client"] * n_clients)))


def query_order(seed, spec, pool, total, queries):
    """Indices into the query pool: a seeded permutation, or (skew) Zipf over regions without repeats."""
    import torch
    g = torch.Generator().manual_seed(stream_seed("queries", seed, spec["name"]))
    if spec.get("queries") != "zipf":
        return torch.randperm(pool, generator=g)[:total]
    regions = 32
    centres = queries[torch.randperm(pool, generator=g)[:regions]]
    home = torch.cdist(queries, centres).argmin(dim=1)
    rank = torch.randperm(regions, generator=g)
    weights = 1. / (rank.double() + 1.) ** 1.1
    members = [torch.nonzero(home == r).flatten().tolist() for r in range(regions)]
    rng = random.Random(stream_seed("zipf", seed, spec["name"]))
    for m in members:
        rng.shuffle(m)
    out, w = [], weights.tolist()
    while len(out) < total and any(members):
        r = rng.choices(range(regions), weights=[w[i] if members[i] else 0. for i in range(regions)])[0]
        out.append(members[r].pop())
    return torch.tensor(out, dtype=torch.int64)


def task(name, cls, dataset, n, load, score="qps", queries="uniform", weather=None, m=M_DEFAULT):
    spec = {"name": name, "class": cls, "dataset": dataset, "n": n, "load": list(load), "score": score, "queries": queries,
            "weather": weather or {}, "m": m}
    return {"name": name, "group": cls, "spec": spec}


def suite(split, scale):
    """The task list of a split: "main", "transfer" (another dataset) or "unseen" (another dataset and a buffer policy the main
    split never draws). scale["mode"] = "speedup" picks the compact suite (4 main tasks); without it, the full suite."""
    n = scale["n"]
    if scale.get("mode") == "speedup":
        return suite_compact(split, n)
    if split == "main":
        hi, mid = ("per_unit", 1.5), ("per_unit", .5)
        return [task("high_load", "high_load", "sift", n, hi),
                task("low_load_1", "low_load", "sift", n, ("count", 1)),
                task("low_load_4", "low_load", "sift", n, ("count", 4)),
                task("tail", "tail", "sift", n, ("per_unit", .25), score="p99"),
                task("power_bound", "bottleneck_ends", "sift", n, hi, weather={"profiles": ["C"]}),
                task("power_free", "bottleneck_ends", "sift", n, hi, weather={"profiles": ["D"]}),
                task("tR_slow", "tR_ends", "sift", n, hi, weather={"profiles": ["A"]}),
                task("tR_fast", "tR_ends", "sift", n, hi, weather={"profiles": ["D"]}),
                task("data_sift", "data", "sift", n, mid),
                task("data_deep", "data", "deep", n, mid),
                task("data_glove", "data", "glove", n, mid),
                task("skew", "skew", "sift", n, hi, queries="zipf"),
                task("scale_full", "scale", "sift", n, mid),
                task("scale_half", "scale", "sift", n // 2, mid)]
    if split == "transfer":
        return [task("val_gist_high", "high_load", "gist", n, ("per_unit", 1.5)),
                task("val_gist_low", "low_load", "gist", n, ("count", 2))]
    if split == "unseen":
        return [task("hold_t2i_high", "high_load", "t2i", n, ("per_unit", 1.5)),
                task("hold_t2i_low", "low_load", "t2i", n, ("count", 2)),
                task("hold_t2i_tail", "tail", "t2i", n, ("per_unit", .25), score="p99")]
    return []


def suite_compact(split, n):
    """The compact suite: one task per key case, every task its own class. The other splits as in the full suite."""
    hi, mid = ("per_unit", .75), ("per_unit", .5)  # half the full suite's high load (still many reads in flight per unit), for cost
    if split == "main":
        return [task("high_load", "high_load", "sift", n, hi),
                task("low_load_4", "low_load", "sift", n, ("count", 4)),
                task("power_bound", "bottleneck_ends", "sift", n, hi, weather={"profiles": ["C"]}),
                task("data_deep", "data", "deep", n, mid)]
    if split == "transfer":
        return [task("val_gist_high", "high_load", "gist", n, hi),
                task("val_gist_low", "low_load", "gist", n, ("count", 2))]
    if split == "unseen":
        return [task("hold_t2i_high", "high_load", "t2i", n, hi),
                task("hold_t2i_low", "low_load", "t2i", n, ("count", 2))]
    return []
