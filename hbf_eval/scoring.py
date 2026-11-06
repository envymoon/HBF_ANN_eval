"""Scoring one task: the area under the recall-throughput curve, each recall level normalised on a log scale
between batched brute force (0) and the hardware roof (1)."""
import math

from .arch import POSITIONS, Geometry
from .compute_table import TABLE
from .device import PAGE
from .tasks import LEVELS


def frontier(points):
    """Upper envelope of (recall, value) points, value higher-is-better: ascending recall, descending value."""
    front, best = [], -math.inf
    for r, v in sorted(points, key=lambda p: (-p[0], -p[1])):
        if v > best and v > 0:
            front.append((r, v))
            best = v
    front.reverse()
    return front


def value_at(front, level):
    """The value reachable at `level`, log-interpolated between frontier points; None when the level is not reached."""
    if not front or front[-1][0] < level:
        return None
    for i, (r, v) in enumerate(front):
        if r >= level:
            if i == 0 or r == front[i - 1][0]:
                return v
            r0, v0 = front[i - 1]
            w = (level - r0) / (r - r0)
            return math.exp(math.log(v0) + w * (math.log(v) - math.log(v0)))
    return None


def normalise(value, lo, hi):
    if value is None:
        return 0.
    if hi <= lo * (1 + 1e-9):
        return 1. if value >= hi else 0.
    return max(0., min(1., math.log(value / lo) / math.log(hi / lo)))


def area(points, lo, hi):
    front = frontier(points)
    levels = [normalise(value_at(front, level), lo, hi) for level in LEVELS]
    return sum(levels) / len(levels), levels


SPEEDUP_LEVEL, SPEEDUP_FLOOR = .90, -8.  # speedup score: default recall level and floor


def speedup(points, lo, level=SPEEDUP_LEVEL):
    """Speedup score for one task: log2 of the value (qps, or 1 / p99) reachable at recall 0.9 over batched brute force's, so
    0 = as fast as the scan, +1 = twice as fast, -1 = half as fast. Floored at -8 (256x slower); a curve that never reaches
    0.9 scores the floor."""
    v = value_at(frontier(points), level)
    return SPEEDUP_FLOOR if v is None else max(SPEEDUP_FLOOR, math.log2(v / lo))


def sense_cost(weather, g):
    """Per sense: (unit time, energy) including the refresh work it causes (pages_per_block / X pages of
    tR + t_prog and E_sense + E_prog each). Spread evenly, a die-scope operation retires one page on every plane at once, so
    either scope costs each plane the same time per sense."""
    x = weather["disturb_threshold"]
    owed = weather["pages_per_block"] / x if x > 0 else 0.
    return weather["tR"] + owed * (weather["tR"] + weather["t_prog"]), g.e_sense + owed * (g.e_sense + g.e_prog)


def sense_rate(weather, g):
    """Senses per us the HBF positions can sustain: the least of the structure, the power budget and the core-die
    channels (every sense moves 4 KiB into the base die)."""
    t, e = sense_cost(weather, g)
    rates = [g.channels * g.units / t]
    if g.power > 0 and e > 0:
        rates.append((POSITIONS - g.m) * g.power / e)
    if g.coredie:
        rates.append(g.channels * g.coredie * g.coredie_bw / PAGE)
    return min(rates)


def brute(weather, data_bytes, n, dim, clients, table=TABLE):
    """Batched brute force: the N queries in flight scan all data once, the share that fits in HBM from HBM
    and the rest from HBF (whole pages, every limit of the device model), then one distance kernel and one top-k.
    Returns (batch time in us, m)."""
    g = Geometry(weather, data_bytes)
    dh = min(data_bytes, g.hbm_budget)
    df = data_bytes - dh
    scan_h = dh / g.hbm_bw + g.hbm_latency if dh > 0 else 0.
    pages = math.ceil(df / PAGE)
    scan_f = weather["tR"] + max(df / (g.channels * g.channel_bw), pages / sense_rate(weather, g)) if df > 0 else 0.
    scan = max(scan_h, scan_f)
    F, B, L = table["flops_per_us"], table["bytes_per_us"], table["launch_us"]
    dist = L + max(2. * clients * n * dim / F, (n * dim * 4 + clients * dim * 4 + clients * n * 4) / B)
    topk = L + max(4. * clients * n * math.log2(max(n, 2)) / F, 2. * clients * n * 4 / B)
    return table["step_us"] + scan + dist + topk, g.m


def roof(weather, data_bytes, k, vector_bytes, clients, table=TABLE):
    """Hardware roof of the platform's 8 - m HBF positions: min(sense rate / P_min, bandwidth / bytes_min,
    N / latency_min), the sense rate limited by structure, power and core-die channels. Returns (qps, minimum latency in us,
    which roof binds)."""
    g = Geometry(weather, data_bytes)
    p_min = math.ceil(k * vector_bytes / PAGE)
    cdc = PAGE / g.coredie_bw if g.coredie else 0.
    latency = table["step_us"] + table["launch_us"] + weather["tR"] + cdc + vector_bytes / g.channel_bw
    roofs = {"sense": sense_rate(weather, g) / p_min, "bandwidth": g.channels * g.channel_bw / (k * vector_bytes),
             "latency": clients / latency}
    which = min(roofs, key=roofs.get)
    return roofs[which] * 1e6, latency, which


def bounds(weather, data_bytes, n, dim, k, vector_bytes, clients, table=TABLE):
    batch, m = brute(weather, data_bytes, n, dim, clients, table)
    qps_hi, latency, which = roof(weather, data_bytes, k, vector_bytes, clients, table)
    return {"qps_lo": clients / batch * 1e6, "qps_hi": qps_hi, "lat_lo": batch, "lat_hi": latency, "roof": which, "brute_m": m}
