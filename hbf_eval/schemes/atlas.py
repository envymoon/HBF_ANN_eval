"""Cluster-ordered layout + cluster-centroid entry points. Built on the baseline graph search (baseline.py)."""

KNOBS = {
    'alpha': (1.0, 1.0, 1.3, 'linear'),
    'build_list': (64, 32, 96, 'int'),
    'cluster_size': (256, 128, 1024, 'int'),
    'degree': (32, 24, 40, 'int'),
    'entry_k': (128, 8, 1024, 'int'),
    'entry_seeds': (3, 1, 8, 'int'),
    'kmeans_iters': (8, 4, 12, 'int'),
    'max_hops': (1000, 16, 4000, 'int'),
    'pool_hi': (256, 20, 800, 'int'),
    'pool_lo': (16, 10, 100, 'int'),
    'probe': (6, 2, 12, 'int'),
    'search_list': (48, 32, 64, 'int'),
    'search_width': (8, 4, 12, 'int'),
    'vamana_share': (0.1, 0.0, 0.25, 'linear'),
}


# A proximity graph of out-degree `degree` (vamana with the `alpha` knob, default 1: plain relative-neighbourhood
# pruning).
def build(vectors, ctx):
    return {"graph": vamana(vectors, ctx)}


# layout plus budget-sized entry table in HBM
def layout(index, vectors, ctx):
    s = ctx.spec()
    graph = index["graph"]
    size, per = record_geometry(s, graph.shape[1])
    n = vectors.shape[0]
    cent, assign = kmeans(vectors, max(1, n // ctx.knobs["cluster_size"]), ctx.knobs["kmeans_iters"], ctx)
    order = torch.argsort(assign, stable=True)
    pos = torch.zeros(n, dtype=torch.int64)
    pos[order] = torch.arange(n)
    rec = torch.cat([pack_rows(vectors[order], s), graph[order].to(torch.int32).contiguous().view(torch.uint8)], dim=1)
    rec = torch.cat([rec, torch.zeros(rec.shape[0], size - rec.shape[1], dtype=torch.uint8)], dim=1)
    budget = s["hbm_bytes"] - 4 * n - 256
    hbm = entry_table(vectors, budget, ctx)
    hbm["pos"] = pos.to(torch.int32)
    return {"pages": to_pages(rec), "hbm": hbm, "sram": {}, "interleave": 1}


# Visited records in (exact distances), expanded records in (read their new neighbours), new queries start, then every
# query with nothing in flight expands its best open candidate or is answered. Tag = (qid * n + node) * 2 + kind.
def serve(state, events, ctx):
    s = ctx.spec()
    if "id" not in state:
        state = empty_state(s, ctx)
    tags = events["tags"]
    if tags.shape[0] > 0:
        n = s["n"]
        kind, node, qid = tags % 2, (tags // 2) % n, tags // (2 * n)
        seen = kind == 0
        if bool(seen.any()):
            state = take_visits(state, qid[seen], node[seen], events["data"][seen], s)
        if bool((~seen).any()):
            state = take_expansions(state, qid[~seen], events["data"][~seen], s, ctx)
    if events["qids"].shape[0] > 0:
        state = admit(state, events["qids"], events["queries"], s, ctx)
    return next_hop(state, s, ctx)


# seed queries from nearest centroid entry nodes
def admit(state, qids, queries, s, ctx):
    a, L = qids.shape[0], state["pid"].shape[1]
    cent = ctx.hbm("ecent")
    eid = ctx.hbm("eid").long()
    sc = (cent * cent).sum(1)[None, :] - 2. * queries @ cent.T
    m = min(int(ctx.knobs["entry_seeds"]), cent.shape[0])
    near = torch.topk(sc, m, dim=1, largest=False).indices
    read_records(qids.repeat_interleave(m), eid[near].reshape(-1), 0, s, ctx)
    z = torch.zeros(a, dtype=torch.int64)
    new = {"id": qids, "q": queries, "pid": torch.full((a, L), -1, dtype=torch.int64), "pd": torch.full((a, L), float("inf")),
           "px": torch.zeros(a, L, dtype=torch.bool), "out": z + m, "hops": z}
    return {key: torch.cat([state[key], new[key]]) for key in state}


# Vamana's construction search for a batch of points (exact distances, in memory): greedy beam search from the entry
# with a list of L and width W; returns every node it expanded (-1 padded): the path from the entry, which supplies long
# edges.
def build_search(vectors, graph, entry, pts, L, W):
    B, R, T = pts.shape[0], graph.shape[1], 2 * L // W + 2
    q = vectors[pts]
    cid = torch.full((B, L), -1, dtype=torch.int64)
    cd = torch.full((B, L), float("inf"))
    cx = torch.zeros(B, L, dtype=torch.bool)
    cid[:, 0] = entry
    cd[:, 0] = ((vectors[entry] - q) ** 2).sum(1)
    seen = torch.full((B, T * W), -1, dtype=torch.int64)
    act = torch.arange(B)
    for t in range(T):
        open_d = torch.where(cx[act] | (cid[act] < 0), torch.full_like(cd[act], float("inf")), cd[act])
        pick = torch.topk(open_d, W, dim=1, largest=False).indices
        live = torch.gather(open_d, 1, pick) < float("inf")
        go = live.any(1)
        if not bool(go.any()):
            break
        act, pick, live = act[go], pick[go], live[go]
        A = act.shape[0]
        a_cid, a_cd, a_cx = cid[act], cd[act], cx[act]
        a_cx[torch.arange(A)[:, None], pick] = a_cx[torch.arange(A)[:, None], pick] | live
        nodes = torch.where(live, torch.gather(a_cid, 1, pick), torch.full_like(pick, -1))
        seen[act, t * W:(t + 1) * W] = nodes
        nb = graph[nodes.clamp(min=0)]
        nb = torch.where(live[:, :, None] & (nb >= 0), nb, torch.full_like(nb, -1)).reshape(A, W * R)
        new = first_only(torch.cat([a_cid, nb], dim=1))[:, L:]
        ok = new >= 0
        owner = act[:, None].expand_as(new)[ok]
        d = torch.full(new.shape, float("inf"))
        d[ok] = ((vectors[new[ok]] - q[owner]) ** 2).sum(1)
        all_d = torch.cat([a_cd, d], dim=1)
        top = torch.topk(all_d, L, dim=1, largest=False).indices
        cd[act] = torch.gather(all_d, 1, top)
        cid[act] = torch.gather(torch.cat([a_cid, new], dim=1), 1, top)
        cx[act] = torch.gather(torch.cat([a_cx, torch.zeros(A, W * R, dtype=torch.bool)], dim=1), 1, top)
    return seen


# Per-query state: id, query, candidate list (L from ctx.effort, capped by the state budget: id, exact distance,
# expanded), reads in flight, hops.
def empty_state(s, ctx):
    lo, hi = ctx.knobs["pool_lo"], ctx.knobs["pool_hi"]
    L = max(s["k"], int(round(lo + ctx.effort * (hi - lo))))
    room = (s["state_bytes"] // max(1, s["clients"]) - 64 - 4 * s["dim"]) // 16
    L = max(s["k"], min(L, room))
    z = torch.zeros(0, dtype=torch.int64)
    return {"id": z, "q": torch.zeros(0, s["dim"]), "pid": torch.zeros(0, L, dtype=torch.int64), "pd": torch.zeros(0, L),
            "px": torch.zeros(0, L, dtype=torch.bool), "out": z, "hops": z}


# budget-sized coarse centroid table with nearest real node per centroid
def entry_table(vectors, budget, ctx):
    n, dim = vectors.shape[0], vectors.shape[1]
    cap = int(budget * 0.9) // (4 * dim + 4)
    K = max(1, min(ctx.knobs["entry_k"], n // 4, cap))
    cent, assign = kmeans(vectors, K, 4, ctx)
    K = cent.shape[0]
    d = ((vectors - cent[assign]) ** 2).sum(1)
    mn = torch.full((K,), float("inf")).scatter_reduce(0, assign, d, "amin", include_self=True)
    best = d == mn[assign]
    eid = torch.zeros(K, dtype=torch.int64)
    eid[assign[best]] = torch.arange(n)[best]
    return {"ecent": cent, "eid": eid.to(torch.int32)}


# Within a row, keep the first copy of each id.
def first_only(new):
    order = torch.sort(new, dim=1, stable=True)
    repeated = torch.zeros_like(new, dtype=torch.bool)
    repeated[:, 1:] = (order.values[:, 1:] == order.values[:, :-1]) & (order.values[:, 1:] >= 0)
    drop = torch.zeros_like(new, dtype=torch.bool).scatter(1, order.indices, repeated)
    return torch.where(drop, torch.full_like(new, -1), new)


# Approximate neighbour candidates: k-means into clusters of about `cluster_size`, each point's L nearest among the
# points of its cluster's `probe` nearest clusters (-1 where fewer).
def ivf_candidates(vectors, L, ctx):
    n = vectors.shape[0]
    K = max(1, n // ctx.knobs["cluster_size"])
    cent, assign = kmeans(vectors, K, ctx.knobs["kmeans_iters"], ctx)
    K = cent.shape[0]
    near = torch.topk(torch.cdist(cent, cent), min(K, ctx.knobs["probe"]), dim=1, largest=False).indices.tolist()
    order = torch.argsort(assign, stable=True)
    count = torch.bincount(assign, minlength=K)
    start = (torch.cumsum(count, 0) - count).tolist()
    count = count.tolist()
    out = torch.full((n, L), -1, dtype=torch.int64)
    for c in range(K):
        if count[c] == 0:
            continue
        mine = order[start[c]:start[c] + count[c]]
        pool = torch.cat([order[start[j]:start[j] + count[j]] for j in near[c]])
        d = torch.cdist(vectors[mine], vectors[pool])
        d[pool[None, :] == mine[:, None]] = float("inf")
        t = min(L, pool.shape[0] - 1)
        if t > 0:
            out[mine, :t] = pool[torch.topk(d, t, dim=1, largest=False).indices]
    return out


# Lloyd's k-means from K random points; returns the centroids and the assignment.
def kmeans(x, K, iters, ctx):
    n = x.shape[0]
    K = max(1, min(K, n))
    c = x[torch.argsort(ctx.uniform(n))[:K]].clone()
    assign = nearest(x, c)
    for _ in range(iters):
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        count = torch.bincount(assign, minlength=K).to(x.dtype)
        c = torch.where(count[:, None] > 0, sums / count.clamp(min=1.)[:, None], c)
        assign = nearest(x, c)
    return c, assign


# The point nearest the centroid: the search entry.
def medoid(vectors):
    return torch.argmin(((vectors - vectors.mean(dim=0, keepdim=True)) ** 2).sum(1)).reshape(1)


# Index of the nearest row of c for every row of x (chunked).
def nearest(x, c):
    cc = (c * c).sum(1)
    out = []
    for i in range(0, x.shape[0], 8192):
        out.append(torch.argmin(cc[None, :] - 2. * x[i:i + 8192] @ c.T, dim=1))
    return torch.cat(out)


# Queries with nothing in flight: read the record of the best unexpanded candidate (to expand it); a query with none
# left (or after max_hops) is answered with its k best.
def next_hop(state, s, ctx):
    if state["id"].shape[0] == 0:
        return state
    idle = state["out"] == 0
    open_d = torch.where(state["px"] | (state["pid"] < 0), torch.full_like(state["pd"], float("inf")), state["pd"])
    best = torch.argmin(open_d, dim=1)
    rows = torch.arange(state["id"].shape[0])
    live = idle & (open_d[rows, best] < float("inf")) & (state["hops"] < ctx.knobs["max_hops"])
    stop = idle & ~live
    px = state["px"].clone()
    px[rows[live], best[live]] = True
    if bool(live.any()):
        read_records(state["id"][live], state["pid"][rows[live], best[live]], 1, s, ctx)
    state = dict(state, px=px, out=state["out"] + live.long(), hops=state["hops"] + live.long())
    if bool(stop.any()):
        ctx.finish(state["id"][stop], state["pid"][stop][:, :s["k"]].clamp(min=0))
        state = {key: v[~stop] for key, v in state.items()}
    return state


# Vectors to their stored bytes: uint8 data stays one byte per value, float32 four.
def pack_rows(vectors, s):
    if s["dtype"] == "uint8":
        return vectors.round().clamp(0, 255).to(torch.uint8)
    return vectors.contiguous().view(torch.uint8)


# RobustPrune over all points, in batches.
def prune_all(vectors, cand, alpha, R):
    n = cand.shape[0]
    out = []
    for i in range(0, n, 1024):
        out.append(robust_prune(vectors, torch.arange(i, min(n, i + 1024)), cand[i:i + 1024], alpha, R))
    return torch.cat(out)


# Look up the record slot in HBM before reading.
def read_records(qids, nodes, kind, s, ctx):
    size, per = record_geometry(s, ctx.knobs["degree"])
    pos = ctx.hbm("pos", nodes).long()
    ctx.submit(pos // per, (pos % per) * size, size, (qids * s["n"] + nodes) * 2 + kind, True)


# A node record (vector, then R int32 neighbour ids) rounded up to 64 B, and records per page.
def record_geometry(s, R):
    size = ((s["vector_bytes"] + 4 * R + 63) // 64) * 64
    return size, 4096 // size


# For every node, up to `cap` nodes that point to it, those that rank it higher first.
def reverse_edges(graph, cap):
    n, R = graph.shape
    rank = torch.arange(R).repeat(n)
    src = torch.arange(n).repeat_interleave(R)
    dst = graph.reshape(-1)
    ok = dst >= 0
    src, dst, rank = src[ok], dst[ok], rank[ok]
    order = torch.argsort(rank, stable=True)
    src, dst = src[order], dst[order]
    order = torch.argsort(dst, stable=True)
    src, dst = src[order], dst[order]
    slot = torch.arange(dst.shape[0]) - torch.searchsorted(dst, dst)
    keep = slot < cap
    rev = torch.full((n, cap), -1, dtype=torch.int64)
    rev[dst[keep], slot[keep]] = src[keep]
    return rev


# DiskANN's RobustPrune for a batch of points: take the nearest live candidate, drop every candidate c with alpha *
# d(chosen, c) <= d(p, c), repeat up to R edges (-1 padded).
def robust_prune(vectors, pts, cand, alpha, R):
    B = pts.shape[0]
    cand = first_only(torch.where(cand == pts[:, None], torch.full_like(cand, -1), cand))
    v = vectors[cand.clamp(min=0)]
    d = ((v - vectors[pts][:, None, :]) ** 2).sum(2)
    d = torch.where(cand >= 0, d, torch.full_like(d, float("inf")))
    d, order = torch.sort(d, dim=1)
    cand = torch.gather(cand, 1, order)
    v = v[torch.arange(B)[:, None], order]
    vv = (v * v).sum(2)
    pair = vv[:, :, None] + vv[:, None, :] - 2. * v @ v.transpose(1, 2)
    alive = d < float("inf")
    rows = torch.arange(B)
    out = torch.full((B, R), -1, dtype=torch.int64)
    for r in range(R):
        has = alive.any(1)
        if not bool(has.any()):
            break
        sel = torch.argmax(alive.to(torch.int8), dim=1)
        out[:, r] = torch.where(has, cand[rows, sel], torch.full_like(sel, -1))
        alive = alive & (alpha * alpha * pair[rows, sel] > d)
    return out


# Group completions by query row: the order, the sorted rows and each one's slot within its row.
def slots(row):
    order = torch.argsort(row, stable=True)
    sorted_row = row[order]
    slot = torch.arange(row.shape[0]) - torch.searchsorted(sorted_row, sorted_row)
    return order, sorted_row, slot


# Expanded records in: every neighbour not already in the candidate list (or twice) is read to be scored.
def take_expansions(state, qids, data, s, ctx):
    R, vb = ctx.knobs["degree"], s["vector_bytes"]
    row = torch.searchsorted(state["id"], qids)
    kids = data[:, vb:vb + 4 * R].contiguous().view(torch.int32).long()
    order, row_s, slot = slots(row)
    Q, L, C = state["id"].shape[0], state["pid"].shape[1], int(slot.max()) + 1
    cand = torch.full((Q, C * R), -1, dtype=torch.int64)
    cand[row_s[:, None], slot[:, None] * R + torch.arange(R)[None, :]] = kids[order]
    cand = first_only(torch.cat([state["pid"], cand], dim=1))[:, L:]
    ok = cand >= 0
    owner = torch.arange(Q)[:, None].expand_as(cand)[ok]
    if bool(ok.any()):
        read_records(state["id"][owner], cand[ok], 0, s, ctx)
    return dict(state, out=state["out"] - torch.bincount(row, minlength=Q) + ok.sum(dim=1))


# Scored records in: exact distances merged into each query's candidate list; with the list full only those strictly
# better than its worst enter, so a node once pushed out never comes back (the search always ends).
def take_visits(state, qids, nodes, data, s):
    row = torch.searchsorted(state["id"], qids)
    vb = s["vector_bytes"]
    d = ((unpack_rows(data[:, :vb], s) - state["q"][row]) ** 2).sum(1)
    order, row_s, slot = slots(row)
    Q, L, C = state["id"].shape[0], state["pid"].shape[1], int(slot.max()) + 1
    nd = torch.full((Q, C), float("inf"))
    ni = torch.full((Q, C), -1, dtype=torch.int64)
    nd[row_s, slot] = d[order]
    ni[row_s, slot] = nodes[order]
    worst = torch.where(state["pid"] >= 0, state["pd"], torch.full_like(state["pd"], float("inf"))).max(dim=1).values
    better = nd < worst[:, None]
    nd = torch.where(better, nd, torch.full_like(nd, float("inf")))
    ni = torch.where(better, ni, torch.full_like(ni, -1))
    all_d = torch.cat([state["pd"], nd], dim=1)
    top = torch.topk(all_d, L, dim=1, largest=False).indices
    return dict(state, pd=torch.gather(all_d, 1, top), pid=torch.gather(torch.cat([state["pid"], ni], dim=1), 1, top),
                px=torch.gather(torch.cat([state["px"], torch.zeros(Q, C, dtype=torch.bool)], dim=1), 1, top),
                out=state["out"] - torch.bincount(row, minlength=Q))


# Records of one size, packed in order, as many per 4 KiB page as fit.
def to_pages(rec):
    size = rec.shape[1]
    per = 4096 // size
    n = rec.shape[0]
    count = (n + per - 1) // per
    rec = torch.cat([rec, torch.zeros(count * per - n, size, dtype=torch.uint8)])
    pages = rec.reshape(count, per * size)
    return torch.cat([pages, torch.zeros(count, 4096 - per * size, dtype=torch.uint8)], dim=1)


# Stored bytes (last dimension) back to float32 vectors.
def unpack_rows(raw, s):
    if s["dtype"] == "uint8":
        return raw.float()
    return raw.contiguous().view(torch.float32)


# A Vamana graph: start from RobustPrune (alpha 1) of IVF candidates, empty slots filled at random; then one Vamana pass
# in random order and batches: search from the medoid, RobustPrune (alpha > 1) of the expanded nodes and the current
# edges; finally add reverse edges and prune again.
def vamana(vectors, ctx):
    R, alpha, L = ctx.knobs["degree"], ctx.knobs["alpha"], ctx.knobs["build_list"]
    n = vectors.shape[0]
    graph = prune_all(vectors, ivf_candidates(vectors, L, ctx), 1., R)
    extra = (ctx.uniform(n, R) * n).long()
    graph = torch.where(graph >= 0, graph, extra)
    entry = medoid(vectors)
    order = torch.argsort(ctx.uniform(n))[:int(round(ctx.knobs["vamana_share"] * n))]
    for i in range(0, order.shape[0], 2048):
        pts = order[i:i + 2048]
        seen = build_search(vectors, graph, entry, pts, ctx.knobs["search_list"], ctx.knobs["search_width"])
        graph[pts] = robust_prune(vectors, pts, torch.cat([seen, graph[pts]], dim=1), alpha, R)
    return prune_all(vectors, torch.cat([graph, reverse_edges(graph, R)], dim=1), alpha, R)
