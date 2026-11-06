"""CAGRA-like (Ootomo et al., ICDE 2024): fixed-degree graph, random entry points, batched best-first search."""

KNOBS = {
    'cluster_size': (256, 32, 4096, 'int'),
    'degree': (32, 8, 64, 'int'),
    'inter_degree': (64, 16, 128, 'int'),
    'kmeans_iters': (8, 1, 30, 'int'),
    'max_iters': (200, 8, 1000, 'int'),
    'pool_hi': (128, 20, 400, 'int'),
    'pool_lo': (16, 10, 100, 'int'),
    'probe': (6, 1, 32, 'int'),
    'random_entries': (32, 1, 256, 'int'),
    'search_width': (1, 1, 8, 'int'),
}


# kNN graph of `inter_degree` (IVF candidates), detour-count pruning to `degree`, merged with reverse edges.
def build(vectors, ctx):
    R = ctx.knobs["degree"]
    knn = ivf_candidates(vectors, max(ctx.knobs["inter_degree"], R), ctx)
    return {"graph": fixed_degree(knn, detour_prune(knn, R), R)}


# The graph (adjacency records) then the dataset (vector records), each packed in id order; no HBM, all 8 positions HBF.
def layout(index, vectors, ctx):
    s = ctx.spec()
    graph = index["graph"]
    n, R = graph.shape
    adj, vec, _ = sizes(s, R)
    a = graph.to(torch.int32).contiguous().view(torch.uint8)
    a = torch.cat([a, torch.zeros(n, adj - a.shape[1], dtype=torch.uint8)], dim=1)
    v = pack_rows(vectors, s)
    v = torch.cat([v, torch.zeros(n, vec - v.shape[1], dtype=torch.uint8)], dim=1)
    return {"pages": torch.cat([to_pages(a), to_pages(v)]), "hbm": {}, "sram": {}, "interleave": 1}


# One synchronous iteration for every query: children's vectors in, adjacency lists in (read their new children), the
# others pick parents (or finish); new queries start; then wait until nothing is in flight. Tag = (qid * n + node) * 2 +
# kind.
def serve(state, events, ctx):
    s = ctx.spec()
    n = s["n"]
    if "id" not in state:
        state = empty_state(s, ctx)
    tags = events["tags"]
    busy = torch.zeros(state["id"].shape[0], dtype=torch.bool)
    if tags.shape[0] > 0:
        kind, node, qid = tags % 2, (tags // 2) % n, tags // (2 * n)
        vec = kind == 0
        if bool(vec.any()):
            state = take_vectors(state, qid[vec], node[vec], events["data"][vec], s)
        if bool((~vec).any()):
            state, busy = take_children(state, qid[~vec], events["data"][~vec], s, ctx)
    state, busy = step(state, busy, s, ctx)
    if events["qids"].shape[0] > 0:
        state = admit(state, events["qids"], events["queries"], s, ctx)
    ctx.wait(1 << 40)
    return state


# New queries: read the vectors of `random_entries` random nodes.
def admit(state, qids, queries, s, ctx):
    a, L, n = qids.shape[0], state["pid"].shape[1], s["n"]
    E = ctx.knobs["random_entries"]
    ids = first_only((ctx.uniform(a, E) * n).long().clamp(max=n - 1))
    ok = ids >= 0
    owner = qids[:, None].expand_as(ids)[ok]
    read_vectors(owner, ids[ok], s, ctx)
    new = {"id": qids, "q": queries, "pid": torch.full((a, L), -1, dtype=torch.int64), "pd": torch.full((a, L), float("inf")),
           "px": torch.zeros(a, L, dtype=torch.bool), "it": torch.zeros(a, dtype=torch.int64)}
    return {key: torch.cat([state[key], new[key]]) for key in state}


# CAGRA's rank-based reordering: an edge A->B (rank j) is detourable via C (rank i < j) if B is in C's list at rank < j;
# keep the R edges with the fewest detours, ties by rank.
def detour_prune(knn, R):
    n, K = knn.shape
    out = []
    for i in range(0, n, 512):
        a = knn[i:i + 512]
        nk = knn[a.clamp(min=0)]
        sv, si = torch.sort(nk, dim=2)
        target = a[:, None, :].expand(-1, K, -1).contiguous()
        pos = torch.searchsorted(sv, target).clamp(max=K - 1)
        found = torch.gather(sv, 2, pos) == target
        rank = torch.gather(si, 2, pos)
        ii, jj = torch.arange(K)[None, :, None], torch.arange(K)[None, None, :]
        detour = found & (ii < jj) & (rank < jj) & (a[:, :, None] >= 0) & (a[:, None, :] >= 0)
        key = detour.sum(1) * K + torch.arange(K)[None, :]
        key = torch.where(a >= 0, key, torch.full_like(key, K * K * 2))
        out.append(torch.gather(a, 1, torch.argsort(key, dim=1)[:, :R]))
    return torch.cat(out)


# Per-query state: id, query, top list (L from ctx.effort, capped by the state budget: id, exact distance, expanded),
# iterations.
def empty_state(s, ctx):
    lo, hi = ctx.knobs["pool_lo"], ctx.knobs["pool_hi"]
    L = max(s["k"], int(round(lo + ctx.effort * (hi - lo))))
    room = (s["state_bytes"] // max(1, s["clients"]) - 32 - 4 * s["dim"]) // 17
    L = max(s["k"], min(L, room))
    z = torch.zeros(0, dtype=torch.int64)
    return {"id": z, "q": torch.zeros(0, s["dim"]), "pid": torch.zeros(0, L, dtype=torch.int64), "pd": torch.zeros(0, L),
            "px": torch.zeros(0, L, dtype=torch.bool), "it": z}


# Within a row, keep the first copy of each id.
def first_only(new):
    order = torch.sort(new, dim=1, stable=True)
    repeated = torch.zeros_like(new, dtype=torch.bool)
    repeated[:, 1:] = (order.values[:, 1:] == order.values[:, :-1]) & (order.values[:, 1:] >= 0)
    drop = torch.zeros_like(new, dtype=torch.bool).scatter(1, order.indices, repeated)
    return torch.where(drop, torch.full_like(new, -1), new)


# Fixed out-degree R: half from the pruned graph, half from the reverse graph, interleaved, then the rest, then the kNN
# list.
def fixed_degree(knn, pruned, R):
    n, half = pruned.shape[0], R // 2
    rev = reverse_edges(pruned, R)
    cand = torch.cat([pruned[:, :half], rev[:, :half], pruned[:, half:], rev[:, half:], knn], dim=1)
    cand = torch.where(cand == torch.arange(n)[:, None], torch.full_like(cand, -1), cand)
    cand = first_only(cand)
    order = torch.argsort((cand < 0).to(torch.int8), dim=1, stable=True)
    return torch.gather(cand, 1, order)[:, :R]


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


# Read adjacency records (kind 1).
def read_adjacency(qids, nodes, s, ctx):
    adj, vec, pa = sizes(s, ctx.knobs["degree"])
    per = 4096 // adj
    ctx.submit(nodes // per, (nodes % per) * adj, adj, (qids * s["n"] + nodes) * 2 + 1, True)


# Read vector records (kind 0).
def read_vectors(qids, nodes, s, ctx):
    adj, vec, pa = sizes(s, ctx.knobs["degree"])
    per = 4096 // vec
    ctx.submit(pa + nodes // per, (nodes % per) * vec, vec, (qids * s["n"] + nodes) * 2, True)


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


# Adjacency record and vector record sizes (64 B multiples) and the number of adjacency pages.
def sizes(s, R):
    adj = ((4 * R + 63) // 64) * 64
    vec = ((s["vector_bytes"] + 63) // 64) * 64
    return adj, vec, -(-s["n"] // (4096 // adj))


# Group completions by query row: the order, the sorted rows and each one's slot within its row.
def slots(row):
    order = torch.argsort(row, stable=True)
    sorted_row = row[order]
    slot = torch.arange(row.shape[0]) - torch.searchsorted(sorted_row, sorted_row)
    return order, sorted_row, slot


# Queries not busy: pick the `search_width` best unexpanded nodes as parents and read their adjacency lists; a query
# with none left (or after max_iters) is answered with its top k.
def step(state, busy, s, ctx):
    W = ctx.knobs["search_width"]
    L = state["pid"].shape[1]
    if state["id"].shape[0] == 0:
        return state, busy
    open_d = torch.where(state["px"] | (state["pid"] < 0), torch.full_like(state["pd"], float("inf")), state["pd"])
    pick = torch.topk(open_d, min(W, L), dim=1, largest=False).indices
    live = torch.gather(open_d, 1, pick) < float("inf")
    stop = ~busy & (~live.any(dim=1) | (state["it"] >= ctx.knobs["max_iters"]))
    live = live & ~busy[:, None] & ~stop[:, None]
    rows = torch.arange(state["id"].shape[0])
    px = state["px"].clone()
    px[rows[:, None], pick] = px[rows[:, None], pick] | live
    if bool(live.any()):
        owner = rows[:, None].expand_as(pick)[live]
        read_adjacency(state["id"][owner], torch.gather(state["pid"], 1, pick)[live], s, ctx)
    state = dict(state, px=px, it=state["it"] + live.any(dim=1).long())
    if bool(stop.any()):
        ctx.finish(state["id"][stop], state["pid"][stop][:, :s["k"]].clamp(min=0))
        state = {key: v[~stop] for key, v in state.items()}
        busy = busy[~stop]
    return state, busy


# Parents' adjacency lists in: children not already in the top list (or twice) are read; those queries are busy this
# round.
def take_children(state, qids, data, s, ctx):
    R = ctx.knobs["degree"]
    row = torch.searchsorted(state["id"], qids)
    kids = data[:, :4 * R].contiguous().view(torch.int32).long()
    order, row_s, slot = slots(row)
    Q, L, C = state["id"].shape[0], state["pid"].shape[1], int(slot.max()) + 1
    cand = torch.full((Q, C * R), -1, dtype=torch.int64)
    cand[row_s[:, None], slot[:, None] * R + torch.arange(R)[None, :]] = kids[order]
    cand = first_only(torch.cat([state["pid"], cand], dim=1))[:, L:]
    ok = cand >= 0
    owner = torch.arange(Q)[:, None].expand_as(cand)[ok]
    if bool(ok.any()):
        read_vectors(state["id"][owner], cand[ok], s, ctx)
    busy = torch.zeros(Q, dtype=torch.bool)
    busy[owner] = True
    return state, busy


# Children's vectors in: exact distances, merged into each query's top list; with the list full only those strictly
# better than its worst enter, so an expanded node once pushed out never comes back.
def take_vectors(state, qids, nodes, data, s):
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
                px=torch.gather(torch.cat([state["px"], torch.zeros(Q, C, dtype=torch.bool)], dim=1), 1, top))


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
