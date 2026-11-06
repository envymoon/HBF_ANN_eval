"""DiskANN (Subramanya et al., NeurIPS 2019): Vamana graph, PQ codes in HBM for routing, full records read from HBF."""

KNOBS = {
    'alpha': (1.2, 1.0, 2.0, 'linear'),
    'beam': (4, 1, 16, 'int'),
    'build_list': (64, 16, 200, 'int'),
    'cache_nodes': (2000, 0, 100000, 'int'),
    'cluster_size': (256, 32, 4096, 'int'),
    'degree': (32, 8, 64, 'int'),
    'kmeans_iters': (8, 1, 30, 'int'),
    'pool_hi': (1024, 20, 2048, 'int'),
    'pool_lo': (16, 10, 100, 'int'),
    'pq_bytes': (32, 1, 64, 'int'),
    'pq_centroids': (256, 16, 256, 'int'),
    'pq_sample': (20000, 1000, 200000, 'int'),
    'probe': (6, 1, 32, 'int'),
    'search_list': (48, 8, 200, 'int'),
    'search_width': (8, 1, 16, 'int'),
    'vamana_share': (0.1, 0.0, 1.0, 'linear'),
}


# Vamana graph, medoid entry, and PQ codebooks and codes at pq_bytes, /2, ... 1 bytes per vector (layout picks one).
def build(vectors, ctx):
    index = {"graph": vamana(vectors, ctx), "entry": medoid(vectors)}
    M = ctx.knobs["pq_bytes"]
    while M >= 1:
        book, codes = pq_train(vectors, M, ctx)
        index["book%d" % M] = book
        index["codes%d" % M] = codes
        if M == 1:
            return index
        M = M // 2
    return index


# Renumber (entry and its BFS neighbourhood first), node records of the uncached nodes packed into pages in id order;
# HBM holds the PQ codebook, the PQ codes and the cached records.
def layout(index, vectors, ctx):
    s = ctx.spec()
    graph = index["graph"]
    n, R = graph.shape
    size = record_size(s, R)
    M = pick_pq(index, s, ctx)
    book, codes = index["book%d" % M].half(), index["codes%d" % M]
    room = s["hbm_bytes"] - codes.numel() - book.numel() * 2
    C = max(0, min(ctx.knobs["cache_nodes"], room // size, n - 1))
    perm = bfs_order(graph, index["entry"][0], C)
    inv = torch.zeros(n, dtype=torch.int64)
    inv[perm] = torch.arange(n)
    nbr = graph[perm]
    nbr = torch.where(nbr >= 0, inv[nbr.clamp(min=0)], nbr)
    rec = torch.cat([pack_rows(vectors[perm], s), nbr.to(torch.int32).contiguous().view(torch.uint8),
                     perm.to(torch.int32).reshape(n, 1).contiguous().view(torch.uint8)], dim=1)
    rec = torch.cat([rec, torch.zeros(n, size - rec.shape[1], dtype=torch.uint8)], dim=1)
    return {"pages": to_pages(rec[C:]), "hbm": {"book": book, "codes": codes[perm].contiguous(), "cache": rec[:C].clone()},
            "sram": {}, "interleave": 1}


# Admit new queries, take in completed records, run the beam rounds of the queries with nothing in flight.
def serve(state, events, ctx):
    s = ctx.spec()
    n = s["n"]
    if "id" not in state:
        state = empty_state(s, ctx)
    if events["qids"].shape[0] > 0:
        state = admit(state, events["qids"], events["queries"], s, ctx)
    if events["tags"].shape[0] > 0:
        qids = events["tags"] // n
        state = absorb(state, qids, events["data"], s, ctx)
        state = dict(state, out=state["out"] - torch.bincount(torch.searchsorted(state["id"], qids), minlength=state["id"].shape[0]))
    return expand(state, s, ctx)


# Records in: exact distances of the expanded nodes into the results (implicit re-rank), their neighbours into the pool.
def absorb(state, qids, data, s, ctx):
    R, vb, k = ctx.knobs["degree"], s["vector_bytes"], s["k"]
    row = torch.searchsorted(state["id"], qids)
    vec = unpack_rows(data[:, :vb], s)
    nbr = data[:, vb:vb + 4 * R].contiguous().view(torch.int32).long()
    orig = data[:, vb + 4 * R:vb + 4 * R + 4].contiguous().view(torch.int32).long().reshape(-1)
    exact = ((vec - state["q"][row]) ** 2).sum(1)
    order, row_s, slot = slots(row)
    Q, C = state["id"].shape[0], int(slot.max()) + 1
    ed = torch.full((Q, C), float("inf"))
    ei = torch.zeros(Q, C, dtype=torch.int64)
    ed[row_s, slot] = exact[order]
    ei[row_s, slot] = orig[order]
    all_d = torch.cat([state["rd"], ed], dim=1)
    top = torch.topk(all_d, k, dim=1, largest=False).indices
    state = dict(state, rd=torch.gather(all_d, 1, top), ri=torch.gather(torch.cat([state["ri"], ei], dim=1), 1, top))
    cand = torch.full((Q, C * R), -1, dtype=torch.int64)
    cand[row_s[:, None], slot[:, None] * R + torch.arange(R)[None, :]] = nbr[order]
    return merge_pool(state, cand, ctx)


# Asymmetric PQ distances from the 8-bit table: offset + scale x the sum of the codes' entries, per (query row) in rows.
def adc(tab, tb, ts, rows, codes):
    Q, M, C = tab.shape
    idx = (rows[:, None] * M + torch.arange(M)[None, :]) * C + codes.long()
    return tb[rows] + ts[rows] * tab.reshape(-1)[idx].float().sum(1)


# New queries: their PQ tables (codebook read from HBM), and the entry (node 0) as the only candidate.
def admit(state, qids, queries, s, ctx):
    a, L, k = qids.shape[0], state["pid"].shape[1], s["k"]
    tab, tb, ts = pq_lut(queries, ctx.hbm("book").float())
    pid = torch.full((a, L), -1, dtype=torch.int64)
    pid[:, 0] = 0
    pd = torch.full((a, L), float("inf"))
    pd[:, 0] = 0.
    new = {"id": qids, "q": queries, "tab": tab, "tb": tb, "ts": ts, "pid": pid, "pd": pd, "px": torch.zeros(a, L, dtype=torch.bool),
           "rd": torch.full((a, k), float("inf")), "ri": torch.zeros(a, k, dtype=torch.int64), "out": torch.zeros(a, dtype=torch.int64)}
    return {key: torch.cat([state[key], new[key]]) for key in state}


# New order: the entry, then BFS from it up to `count` nodes, then everything else by id.
def bfs_order(graph, entry, count):
    n = graph.shape[0]
    seen = torch.zeros(n, dtype=torch.bool)
    seen[entry] = True
    first, frontier, total = [entry.reshape(1)], entry.reshape(1), 1
    while total < count and frontier.shape[0] > 0:
        nb = graph[frontier].reshape(-1)
        nb = torch.unique(nb[nb >= 0])
        nb = nb[~seen[nb]]
        seen[nb] = True
        first.append(nb)
        total += nb.shape[0]
        frontier = nb
    first = torch.cat(first)[:max(count, 1)]
    rest = torch.ones(n, dtype=torch.bool)
    rest[first] = False
    return torch.cat([first, torch.nonzero(rest).flatten()])


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


# The per-query state, no queries yet: id, query, PQ table, candidate pool (id, PQ distance, expanded), best exact
# results, reads in flight.
def empty_state(s, ctx):
    L, k = pool_size(s, ctx), s["k"]
    M, C = s["hbm"]["book"][0], s["hbm"]["book"][1]
    z = torch.zeros(0, dtype=torch.int64)
    return {"id": z, "q": torch.zeros(0, s["dim"]), "tab": torch.zeros(0, M, C, dtype=torch.uint8), "tb": torch.zeros(0), "ts": torch.zeros(0),
            "pid": torch.zeros(0, L, dtype=torch.int64), "pd": torch.zeros(0, L), "px": torch.zeros(0, L, dtype=torch.bool),
            "rd": torch.zeros(0, k), "ri": torch.zeros(0, k, dtype=torch.int64), "out": z}


# Beam rounds for the queries with nothing in flight: answer a query whose pool is fully expanded, else expand its W
# best open candidates: cached records come from HBM at once (and the round repeats), the others are read from HBF as
# one batch.
def expand(state, s, ctx):
    W, n, L = ctx.knobs["beam"], s["n"], state["pid"].shape[1]
    cached = s["hbm"]["cache"][0]
    size = record_size(s, ctx.knobs["degree"])
    per = 4096 // size
    for _ in range(L + 1):
        idle = state["out"] == 0
        if not bool(idle.any()):
            return state
        rows = torch.nonzero(idle).flatten()
        open_d = torch.where(state["px"][rows] | (state["pid"][rows] < 0), torch.full_like(state["pd"][rows], float("inf")), state["pd"][rows])
        pick = torch.topk(open_d, min(W, L), dim=1, largest=False).indices
        live = torch.gather(open_d, 1, pick) < float("inf")
        done = ~live.any(dim=1)
        if bool(done.any()):
            state = retire(state, rows[done], ctx)
            continue
        px = state["px"].clone()
        px[rows[:, None], pick] = px[rows[:, None], pick] | live
        nodes = torch.gather(state["pid"][rows], 1, pick)[live]
        owner = rows[:, None].expand_as(pick)[live]
        state = dict(state, px=px)
        hot = nodes < cached
        if bool((~hot).any()):
            j, o = nodes[~hot] - cached, owner[~hot]
            ctx.submit(j // per, (j % per) * size, size, state["id"][o] * n + nodes[~hot], True)
            state = dict(state, out=state["out"].index_add(0, o, torch.ones_like(o)))
        if not bool(hot.any()):
            return state
        state = absorb(state, state["id"][owner[hot]], ctx.hbm("cache", nodes[hot]), s, ctx)
    return state


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


# New candidates not already in the pool: PQ codes from HBM, PQ distances; with the pool full only those strictly better
# than its worst enter, so an expanded node once pushed out never comes back; keep the L best.
def merge_pool(state, cand, ctx):
    L = state["pid"].shape[1]
    cand = first_only(torch.cat([state["pid"], cand], dim=1))[:, L:]
    ok = cand >= 0
    if not bool(ok.any()):
        return state
    owner = torch.arange(cand.shape[0])[:, None].expand_as(cand)[ok]
    d = torch.full(cand.shape, float("inf"))
    d[ok] = adc(state["tab"], state["tb"], state["ts"], owner, ctx.hbm("codes", cand[ok]))
    worst = torch.where(state["pid"] >= 0, state["pd"], torch.full_like(state["pd"], float("inf"))).max(dim=1).values
    better = d < worst[:, None]
    d = torch.where(better, d, torch.full_like(d, float("inf")))
    cand = torch.where(better, cand, torch.full_like(cand, -1))
    all_d = torch.cat([state["pd"], d], dim=1)
    top = torch.topk(all_d, L, dim=1, largest=False).indices
    return dict(state, pd=torch.gather(all_d, 1, top), pid=torch.gather(torch.cat([state["pid"], cand], dim=1), 1, top),
                px=torch.gather(torch.cat([state["px"], torch.zeros(cand.shape, dtype=torch.bool)], dim=1), 1, top))


# Index of the nearest row of c for every row of x (chunked).
def nearest(x, c):
    cc = (c * c).sum(1)
    out = []
    for i in range(0, x.shape[0], 8192):
        out.append(torch.argmin(cc[None, :] - 2. * x[i:i + 8192] @ c.T, dim=1))
    return torch.cat(out)


# Vectors to their stored bytes: uint8 data stays one byte per value, float32 four.
def pack_rows(vectors, s):
    if s["dtype"] == "uint8":
        return vectors.round().clamp(0, 255).to(torch.uint8)
    return vectors.contiguous().view(torch.uint8)


# Largest PQ size whose codes and codebook fit the HBM budget; else the smallest.
def pick_pq(index, s, ctx):
    n = index["graph"].shape[0]
    sizes = pq_sizes(index)
    for M in sizes:
        if n * M + index["book%d" % M].numel() * 2 <= s["hbm_bytes"]:
            return M
    return sizes[-1]


# Candidate list size L from ctx.effort, capped so that a full load of queries fits the state cap.
def pool_size(s, ctx):
    lo, hi = ctx.knobs["pool_lo"], ctx.knobs["pool_hi"]
    L = max(s["k"], int(round(lo + ctx.effort * (hi - lo))))
    M, C = s["hbm"]["book"][0], s["hbm"]["book"][1]
    fixed = 24 + 4 * s["dim"] + M * C + 16 * s["k"]
    room = (s["state_bytes"] // max(1, s["clients"]) - fixed) // 17
    return max(s["k"], min(L, room))


# The PQ table quantised to 8 bits as in FAISS fast-scan: one scale per query, one offset per sub-space (kept summed).
# Half the bytes of float16, so a 32-byte PQ table fits the per-query state budget at high load.
def pq_lut(q, book):
    tab = pq_table(q, book)
    lo = tab.min(dim=2).values
    scale = (tab.max(dim=2).values - lo).max(dim=1).values.clamp(min=1e-12) / 255.
    q8 = ((tab - lo[:, :, None]) / scale[:, None, None]).round().clamp(0, 255).to(torch.uint8)
    return q8, lo.sum(1), scale


# The PQ sizes build trained, largest first.
def pq_sizes(index):
    return sorted([int(key[4:]) for key in index if key[:4] == "book"], reverse=True)


# Per-query PQ distance table [a, M, C]: squared distance of each query sub-vector to each centroid.
def pq_table(q, book):
    M, C, ds = book.shape
    qp = F.pad(q, (0, M * ds - q.shape[1])).reshape(q.shape[0], M, ds)
    return (qp * qp).sum(2)[:, :, None] + (book * book).sum(2)[None] - 2. * torch.einsum("amd,mcd->amc", qp, book)


# Product quantisation: M sub-spaces (zero padded), `pq_centroids` k-means centroids each; returns the codebook [M, C,
# ds] and the codes uint8[n, M].
def pq_train(vectors, M, ctx):
    n, D = vectors.shape
    ds = (D + M - 1) // M
    x = F.pad(vectors, (0, ds * M - D))
    sample = x[torch.argsort(ctx.uniform(n))[:min(n, ctx.knobs["pq_sample"])]]
    books = [kmeans(sample[:, j * ds:(j + 1) * ds].contiguous(), ctx.knobs["pq_centroids"], ctx.knobs["kmeans_iters"], ctx)[0]
             for j in range(M)]
    C = min(b.shape[0] for b in books)
    book = torch.stack([b[:C] for b in books])
    codes = torch.stack([nearest(x[:, j * ds:(j + 1) * ds].contiguous(), book[j]) for j in range(M)], dim=1)
    return book, codes.to(torch.uint8)


# RobustPrune over all points, in batches.
def prune_all(vectors, cand, alpha, R):
    n = cand.shape[0]
    out = []
    for i in range(0, n, 1024):
        out.append(robust_prune(vectors, torch.arange(i, min(n, i + 1024)), cand[i:i + 1024], alpha, R))
    return torch.cat(out)


# A node record: vector, R int32 neighbour ids, int32 original id, rounded up to 64 B.
def record_size(s, R):
    return ((s["vector_bytes"] + 4 * R + 4 + 63) // 64) * 64


# Answer queries with their k best exactly ranked nodes and drop them from the state.
def retire(state, rows, ctx):
    ctx.finish(state["id"][rows], state["ri"][rows])
    keep = torch.ones(state["id"].shape[0], dtype=torch.bool)
    keep[rows] = False
    return {key: v[keep] for key, v in state.items()}


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
