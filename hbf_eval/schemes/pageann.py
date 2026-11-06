"""PageANN (arXiv 2509.25487): page-aligned graph; the neighbour lists and PQ codes of a page's vectors live together."""

KNOBS = {
    'alpha': (1.2, 1.0, 2.0, 'linear'),
    'beam': (4, 1, 16, 'int'),
    'build_list': (64, 16, 200, 'int'),
    'cluster_size': (256, 32, 4096, 'int'),
    'degree': (32, 8, 64, 'int'),
    'entries': (4, 1, 64, 'int'),
    'kmeans_iters': (8, 1, 30, 'int'),
    'page_degree': (24, 4, 64, 'int'),
    'pool_hi': (256, 8, 512, 'int'),
    'pool_lo': (8, 2, 64, 'int'),
    'pq_bytes': (32, 1, 32, 'int'),
    'pq_centroids': (256, 16, 256, 'int'),
    'pq_sample': (20000, 1000, 200000, 'int'),
    'probe': (6, 1, 32, 'int'),
    'route_ratio': (32, 1, 1024, 'int'),
    'search_list': (48, 8, 200, 'int'),
    'search_width': (8, 1, 16, 'int'),
    'vamana_share': (0.1, 0.0, 1.0, 'linear'),
}


# Pages of similar vectors, the page graph, the PQ codebook and the codes of the page centroids, and the routing pages
# (the pages nearest to k-means centres of the page centroids, one per `route_ratio` pages).
def build(vectors, ctx):
    s = ctx.spec()
    M = ctx.knobs["pq_bytes"]
    per, Rp = page_geometry(s, ctx.knobs["page_degree"], M)
    page_of, P = balanced_pages(vectors, per, ctx)
    order = torch.argsort(page_of, stable=True)
    sp = page_of[order]
    members = torch.full((P, per), -1, dtype=torch.int64)
    members[sp, torch.arange(order.shape[0]) - torch.searchsorted(sp, sp)] = order
    count = torch.bincount(page_of, minlength=P).float()
    cent = torch.zeros(P, vectors.shape[1]).index_add_(0, page_of, vectors) / count[:, None]
    book, _ = pq_train(vectors, M, ctx)
    S = max(1, P // ctx.knobs["route_ratio"])
    rc, _ = kmeans(cent, S, ctx.knobs["kmeans_iters"], ctx)
    route = torch.unique(nearest(rc, cent))
    return {"members": members, "pnbr": page_graph(vamana(vectors, ctx), page_of, P, Rp), "book": book,
            "codes": pq_encode(cent, book), "cent": cent, "route": route}


# One page per page node: member vectors, member ids, neighbour page ids, neighbour codes. HBM: codebook and routing
# table.
def layout(index, vectors, ctx):
    s = ctx.spec()
    members, pnbr, codes = index["members"], index["pnbr"], index["codes"]
    P, per = members.shape
    Rp, vb = pnbr.shape[1], s["vector_bytes"]
    ok = members >= 0
    vec = pack_rows(vectors[members.clamp(min=0)].reshape(P * per, -1), s).reshape(P, per, vb)
    vec = torch.where(ok[:, :, None], vec, torch.zeros_like(vec)).reshape(P, per * vb)
    nb_codes = torch.where((pnbr >= 0)[:, :, None], codes[pnbr.clamp(min=0)], torch.zeros(P, Rp, codes.shape[1], dtype=torch.uint8))
    page = torch.cat([vec, members.to(torch.int32).contiguous().view(torch.uint8), pnbr.to(torch.int32).contiguous().view(torch.uint8),
                      nb_codes.reshape(P, -1)], dim=1)
    page = torch.cat([page, torch.zeros(P, 4096 - page.shape[1], dtype=torch.uint8)], dim=1)
    hbm = {"book": index["book"].half(), "route_vec": index["cent"][index["route"]].half(), "route_page": index["route"].clone()}
    return {"pages": page, "hbm": hbm, "sram": {}, "interleave": 1}


# Admit new queries, take in completed pages, run the page rounds of the queries with nothing in flight. Tag = qid * P +
# page.
def serve(state, events, ctx):
    s = ctx.spec()
    P = s["pages"]
    if "id" not in state:
        state = empty_state(s, ctx)
    if events["qids"].shape[0] > 0:
        state = admit(state, events["qids"], events["queries"], s, ctx)
    if events["tags"].shape[0] > 0:
        qids = events["tags"] // P
        state = absorb(state, qids, events["data"], s, ctx)
        state = dict(state, out=state["out"] - torch.bincount(torch.searchsorted(state["id"], qids), minlength=state["id"].shape[0]))
    return expand(state, s, ctx)


# Pages in: every member vector scored exactly into the results; the neighbour pages scored from their PQ codes in the
# page.
def absorb(state, qids, data, s, ctx):
    k, vb = s["k"], s["vector_bytes"]
    M = state["tab"].shape[1]
    per, Rp = page_geometry(s, ctx.knobs["page_degree"], M)
    c = data.shape[0]
    row = torch.searchsorted(state["id"], qids)
    vec = unpack_rows(data[:, :per * vb].reshape(c, per, vb), s)
    o = per * vb
    ids = data[:, o:o + 4 * per].contiguous().view(torch.int32).long()
    nbr = data[:, o + 4 * per:o + 4 * per + 4 * Rp].contiguous().view(torch.int32).long()
    codes = data[:, o + 4 * per + 4 * Rp:o + 4 * per + 4 * Rp + Rp * M].reshape(c, Rp, M)
    d = ((vec - state["q"][row][:, None, :]) ** 2).sum(2)
    d = torch.where(ids >= 0, d, torch.full_like(d, float("inf")))
    order, row_s, slot = slots(row)
    Q, C = state["id"].shape[0], int(slot.max()) + 1
    cols = slot[:, None] * per + torch.arange(per)[None, :]
    ed = torch.full((Q, C * per), float("inf"))
    ei = torch.zeros(Q, C * per, dtype=torch.int64)
    ed[row_s[:, None], cols] = d[order]
    ei[row_s[:, None], cols] = ids[order].clamp(min=0)
    all_d = torch.cat([state["rd"], ed], dim=1)
    top = torch.topk(all_d, k, dim=1, largest=False).indices
    state = dict(state, rd=torch.gather(all_d, 1, top), ri=torch.gather(torch.cat([state["ri"], ei], dim=1), 1, top))
    nd = adc(state["tab"], state["tb"], state["ts"], row.repeat_interleave(Rp), codes.reshape(c * Rp, M)).reshape(c, Rp)
    nd = torch.where(nbr >= 0, nd, torch.full_like(nd, float("inf")))
    cols = slot[:, None] * Rp + torch.arange(Rp)[None, :]
    cand = torch.full((Q, C * Rp), -1, dtype=torch.int64)
    cd = torch.full((Q, C * Rp), float("inf"))
    cand[row_s[:, None], cols] = nbr[order]
    cd[row_s[:, None], cols] = nd[order]
    return merge_pages(state, cand, cd)


# Asymmetric PQ distances from the 8-bit table: offset + scale x the sum of the codes' entries, per (query row) in rows.
def adc(tab, tb, ts, rows, codes):
    Q, M, C = tab.shape
    idx = (rows[:, None] * M + torch.arange(M)[None, :]) * C + codes.long()
    return tb[rows] + ts[rows] * tab.reshape(-1)[idx].float().sum(1)


# New queries: PQ tables; entry pages from the routing table (the `entries` nearest routing centroids).
def admit(state, qids, queries, s, ctx):
    a, L, k = qids.shape[0], state["pid"].shape[1], s["k"]
    tab, tb, ts = pq_lut(queries, ctx.hbm("book").float())
    rv = ctx.hbm("route_vec").float()
    d = (queries * queries).sum(1)[:, None] + (rv * rv).sum(1)[None, :] - 2. * queries @ rv.T
    t = torch.topk(d, min(L, rv.shape[0], ctx.knobs["entries"]), dim=1, largest=False)
    pid = torch.full((a, L), -1, dtype=torch.int64)
    pd = torch.full((a, L), float("inf"))
    pid[:, :t.indices.shape[1]] = ctx.hbm("route_page", t.indices.reshape(-1)).reshape(a, -1)
    pd[:, :t.values.shape[1]] = t.values
    new = {"id": qids, "q": queries, "tab": tab, "tb": tb, "ts": ts, "pid": pid, "pd": pd, "px": torch.zeros(a, L, dtype=torch.bool),
           "rd": torch.full((a, k), float("inf")), "ri": torch.zeros(a, k, dtype=torch.int64), "out": torch.zeros(a, dtype=torch.int64)}
    return {key: torch.cat([state[key], new[key]]) for key in state}


# Balanced k-means into pages of at most `per` vectors: in rounds, every unplaced vector applies to its nearest centre
# that still has room, and each centre takes its closest applicants up to its room; empty pages are dropped.
def balanced_pages(vectors, per, ctx):
    n = vectors.shape[0]
    cent, _ = kmeans(vectors, -(-n // per), ctx.knobs["kmeans_iters"], ctx)
    K = cent.shape[0]
    cc = (cent * cent).sum(1)
    assign = torch.full((n,), -1, dtype=torch.int64)
    load = torch.zeros(K, dtype=torch.int64)
    for _ in range(100):
        free = torch.nonzero(assign < 0).flatten()
        if free.shape[0] == 0:
            break
        full = (load >= per)[None, :]
        ds, cs = [], []
        for i in range(0, free.shape[0], 8192):
            d = cc[None, :] - 2. * vectors[free[i:i + 8192]] @ cent.T
            t = torch.where(full, torch.full_like(d, float("inf")), d).min(dim=1)
            ds.append(t.values)
            cs.append(t.indices)
        d, cl = torch.cat(ds), torch.cat(cs)
        o = torch.argsort(d, stable=True)
        free, cl = free[o], cl[o]
        o = torch.argsort(cl, stable=True)
        free, cl = free[o], cl[o]
        ok = torch.arange(cl.shape[0]) - torch.searchsorted(cl, cl) < per - load[cl]
        assign[free[ok]] = cl[ok]
        load = load + torch.bincount(cl[ok], minlength=K)
    used = torch.bincount(assign, minlength=K) > 0
    renum = torch.cumsum(used.long(), 0) - 1
    return renum[assign], int(used.sum())


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


# Per-query state: id, query, PQ table, page pool (page, estimated distance, expanded; L pages from ctx.effort, capped
# by the state budget), best exact results, reads in flight.
def empty_state(s, ctx):
    lo, hi, k = ctx.knobs["pool_lo"], ctx.knobs["pool_hi"], s["k"]
    L = max(1, int(round(lo + ctx.effort * (hi - lo))))
    M, C = s["hbm"]["book"][0], s["hbm"]["book"][1]
    room = (s["state_bytes"] // max(1, s["clients"]) - 24 - 4 * s["dim"] - M * C - 16 * k) // 17
    L = max(1, min(L, room))
    z = torch.zeros(0, dtype=torch.int64)
    return {"id": z, "q": torch.zeros(0, s["dim"]), "tab": torch.zeros(0, M, C, dtype=torch.uint8), "tb": torch.zeros(0), "ts": torch.zeros(0),
            "pid": torch.zeros(0, L, dtype=torch.int64), "pd": torch.zeros(0, L), "px": torch.zeros(0, L, dtype=torch.bool),
            "rd": torch.zeros(0, k), "ri": torch.zeros(0, k, dtype=torch.int64), "out": z}


# Page rounds for the queries with nothing in flight: answer a query whose pool is fully expanded, else read its W best
# open pages as one batch.
def expand(state, s, ctx):
    W, L, P = ctx.knobs["beam"], state["pid"].shape[1], s["pages"]
    for _ in range(2):
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
        pages = torch.gather(state["pid"][rows], 1, pick)[live]
        owner = rows[:, None].expand_as(pick)[live]
        ctx.submit(pages, torch.zeros_like(pages), 4096, state["id"][owner] * P + pages, True)
        return dict(state, px=px, out=state["out"].index_add(0, owner, torch.ones_like(owner)))
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


# Neighbour pages not already in the pool, merged by estimated distance; with the pool full only those strictly better
# than its worst enter, so an expanded page once pushed out never comes back; keep the L best.
def merge_pages(state, cand, cd):
    L = state["pid"].shape[1]
    cand = first_only(torch.cat([state["pid"], cand], dim=1))[:, L:]
    worst = torch.where(state["pid"] >= 0, state["pd"], torch.full_like(state["pd"], float("inf"))).max(dim=1).values
    better = (cand >= 0) & (cd < worst[:, None])
    cd = torch.where(better, cd, torch.full_like(cd, float("inf")))
    cand = torch.where(better, cand, torch.full_like(cand, -1))
    all_d = torch.cat([state["pd"], cd], dim=1)
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


# Vectors per page and neighbour pages per page: per vectors (bytes + int32 id), then Rp neighbours (int32 page + M code
# bytes).
def page_geometry(s, Rp, M):
    vb = s["vector_bytes"]
    per = (4096 - Rp * (4 + M)) // (vb + 4)
    while per < 1 and Rp > 1:
        Rp = Rp // 2
        per = (4096 - Rp * (4 + M)) // (vb + 4)
    return max(per, 1), Rp


# Page edges: the pages the member vectors' graph edges point to, most shared first, up to Rp.
def page_graph(graph, page_of, P, Rp):
    n, R = graph.shape
    src = page_of.repeat_interleave(R)
    dst = graph.reshape(-1)
    ok = dst >= 0
    src, dst = src[ok], page_of[dst[ok]]
    ok = src != dst
    key, cnt = torch.unique(src[ok] * P + dst[ok], return_counts=True)
    a, b = key // P, key % P
    o = torch.argsort(-cnt, stable=True)
    a, b = a[o], b[o]
    o = torch.argsort(a, stable=True)
    a, b = a[o], b[o]
    slot = torch.arange(a.shape[0]) - torch.searchsorted(a, a)
    keep = slot < Rp
    out = torch.full((P, Rp), -1, dtype=torch.int64)
    out[a[keep], slot[keep]] = b[keep]
    return out


# PQ codes of rows of x.
def pq_encode(x, book):
    M, C, ds = book.shape
    x = F.pad(x, (0, M * ds - x.shape[1]))
    return torch.stack([nearest(x[:, j * ds:(j + 1) * ds].contiguous(), book[j]) for j in range(M)], dim=1).to(torch.uint8)


# The PQ table quantised to 8 bits as in FAISS fast-scan: one scale per query, one offset per sub-space (kept summed).
# Half the bytes of float16, so a 32-byte PQ table fits the per-query state budget at high load.
def pq_lut(q, book):
    tab = pq_table(q, book)
    lo = tab.min(dim=2).values
    scale = (tab.max(dim=2).values - lo).max(dim=1).values.clamp(min=1e-12) / 255.
    q8 = ((tab - lo[:, :, None]) / scale[:, None, None]).round().clamp(0, 255).to(torch.uint8)
    return q8, lo.sum(1), scale


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


# Answer queries with their k best exactly scored vectors and drop them from the state.
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
