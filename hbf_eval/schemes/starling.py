"""Starling (Wang et al., SIGMOD 2024): block-shuffled layout and an in-memory navigation graph over a Vamana graph."""

KNOBS = {
    'alpha': (1.2, 1.0, 2.0, 'linear'),
    'beam': (4, 1, 16, 'int'),
    'build_list': (64, 16, 200, 'int'),
    'cluster_size': (256, 32, 4096, 'int'),
    'degree': (32, 8, 64, 'int'),
    'entries': (4, 1, 32, 'int'),
    'kmeans_iters': (8, 1, 30, 'int'),
    'nav_share': (0.01, 0.001, 0.1, 'log'),
    'pool_hi': (1024, 20, 2048, 'int'),
    'pool_lo': (16, 10, 100, 'int'),
    'pq_bytes': (32, 1, 64, 'int'),
    'pq_centroids': (256, 16, 256, 'int'),
    'pq_sample': (20000, 1000, 200000, 'int'),
    'probe': (6, 1, 32, 'int'),
    'search_list': (48, 8, 200, 'int'),
    'search_width': (8, 1, 16, 'int'),
    'shuffle_rounds': (16, 0, 32, 'int'),
    'vamana_share': (0.1, 0.0, 1.0, 'linear'),
}


# Vamana graph, the block-shuffled order (original id of every slot, -1 for an empty slot), PQ at pq_bytes, /2, ... 1.
def build(vectors, ctx):
    s = ctx.spec()
    graph = vamana(vectors, ctx)
    index = {"graph": graph, "order": block_shuffle(graph, 4096 // record_size(s, graph.shape[1]), ctx)}
    M = ctx.knobs["pq_bytes"]
    while M >= 1:
        book, codes = pq_train(vectors, M, ctx)
        index["book%d" % M] = book
        index["codes%d" % M] = codes
        if M == 1:
            return index
        M = M // 2
    return index


# Records in block-shuffled order (one block per page; empty slots zero, original id -1); HBM holds the PQ codebook and
# codes (by slot) and the navigation sample (vectors and slots).
def layout(index, vectors, ctx):
    s = ctx.spec()
    graph, order = index["graph"], index["order"]
    n, R = graph.shape
    nslot, size = order.shape[0], record_size(s, R)
    ok = order >= 0
    new = torch.zeros(n, dtype=torch.int64)
    new[order[ok]] = torch.arange(nslot)[ok]
    nbr = graph[order.clamp(min=0)]
    nbr = torch.where((nbr >= 0) & ok[:, None], new[nbr.clamp(min=0)], torch.full_like(nbr, -1))
    vec = pack_rows(vectors[order.clamp(min=0)], s)
    vec = torch.where(ok[:, None], vec, torch.zeros_like(vec))
    rec = torch.cat([vec, nbr.to(torch.int32).contiguous().view(torch.uint8),
                     order.to(torch.int32).reshape(nslot, 1).contiguous().view(torch.uint8)], dim=1)
    rec = torch.cat([rec, torch.zeros(nslot, size - rec.shape[1], dtype=torch.uint8)], dim=1)
    pick = torch.nonzero(ok).flatten()
    nav = pick[torch.argsort(ctx.uniform(pick.shape[0]))[:max(1, int(ctx.knobs["nav_share"] * n))]].sort().values
    nav_vec = vectors[order[nav]].half()
    M = pick_pq(index, s, ctx, nslot, nav_vec.numel() * 2 + nav.numel() * 8)
    codes = torch.where(ok[:, None], index["codes%d" % M][order.clamp(min=0)], torch.zeros(nslot, M, dtype=torch.uint8))
    hbm = {"book": index["book%d" % M].half(), "codes": codes, "nav_vec": nav_vec, "nav_id": nav}
    return {"pages": to_pages(rec), "hbm": hbm, "sram": {}, "interleave": 1}


# Admit new queries, take in completed pages, run the beam rounds of the queries with nothing in flight. Tag = qid *
# slots + node (slots = pages x records per page).
def serve(state, events, ctx):
    s = ctx.spec()
    per = 4096 // record_size(s, ctx.knobs["degree"])
    nslot = s["pages"] * per
    if "id" not in state:
        state = empty_state(s, ctx)
    if events["qids"].shape[0] > 0:
        state = admit(state, events["qids"], events["queries"], s, ctx)
    if events["tags"].shape[0] > 0:
        qids = events["tags"] // nslot
        state = absorb(state, qids, events["tags"] % nslot, events["data"], s, ctx)
        state = dict(state, out=state["out"] - torch.bincount(torch.searchsorted(state["id"], qids), minlength=state["id"].shape[0]))
    return expand(state, s, ctx)


# Pages in (block search): every record scored exactly into the results (a page read again adds no duplicates); the
# target and every record that would enter the top L are expanded now: marked expanded, their neighbours into the pool.
def absorb(state, qids, nodes, data, s, ctx):
    R, vb, k = ctx.knobs["degree"], s["vector_bytes"], s["k"]
    size = record_size(s, R)
    per, c = 4096 // size, data.shape[0]
    rec = data[:, :per * size].reshape(c, per, size)
    row = torch.searchsorted(state["id"], qids)
    vec = unpack_rows(rec[:, :, :vb], s)
    nbr = rec[:, :, vb:vb + 4 * R].contiguous().view(torch.int32).long()
    orig = rec[:, :, vb + 4 * R:vb + 4 * R + 4].contiguous().view(torch.int32).long().reshape(c, per)
    exact = ((vec - state["q"][row][:, None, :]) ** 2).sum(2)
    exact = torch.where(orig >= 0, exact, torch.full_like(exact, float("inf")))
    worst = torch.where(state["pid"] >= 0, state["pd"], torch.full_like(state["pd"], float("inf"))).max(dim=1).values
    slot_ids = (nodes // per * per)[:, None] + torch.arange(per)[None, :]
    use = ((slot_ids == nodes[:, None]) | (exact < worst[row][:, None])) & (orig >= 0)
    order, row_s, slot = slots(row)
    Q, C = state["id"].shape[0], int(slot.max()) + 1
    cols = slot[:, None] * per + torch.arange(per)[None, :]
    ed = torch.full((Q, C * per), float("inf"))
    ei = torch.zeros(Q, C * per, dtype=torch.int64)
    ex = torch.full((Q, C * per), -1, dtype=torch.int64)
    ed[row_s[:, None], cols] = exact[order]
    ei[row_s[:, None], cols] = orig[order].clamp(min=0)
    ex[row_s[:, None], cols] = torch.where(use, slot_ids, torch.full_like(slot_ids, -1))[order]
    seen = torch.cat([torch.where(state["rd"] < float("inf"), state["ri"], torch.full_like(state["ri"], -1)),
                      torch.where(ed < float("inf"), ei, torch.full_like(ei, -1))], dim=1)
    all_d = torch.where(first_only(seen) >= 0, torch.cat([state["rd"], ed], dim=1), torch.full_like(seen, float("inf"), dtype=torch.float32))
    top = torch.topk(all_d, k, dim=1, largest=False).indices
    px = state["px"] | (state["pid"][:, :, None] == ex[:, None, :]).any(dim=2)
    state = dict(state, px=px, rd=torch.gather(all_d, 1, top), ri=torch.gather(torch.cat([state["ri"], ei], dim=1), 1, top))
    cand = torch.where(use[:, :, None], nbr, torch.full_like(nbr, -1)).reshape(c, per * R)
    dense = torch.full((Q, C * per * R), -1, dtype=torch.int64)
    dense[row_s[:, None], slot[:, None] * per * R + torch.arange(per * R)[None, :]] = cand[order]
    return merge_pool(state, dense, ctx)


# Asymmetric PQ distances from the 8-bit table: offset + scale x the sum of the codes' entries, per (query row) in rows.
def adc(tab, tb, ts, rows, codes):
    Q, M, C = tab.shape
    idx = (rows[:, None] * M + torch.arange(M)[None, :]) * C + codes.long()
    return tb[rows] + ts[rows] * tab.reshape(-1)[idx].float().sum(1)


# New queries: PQ tables; entries: the `entries` nearest nodes of the navigation sample (one distance kernel).
def admit(state, qids, queries, s, ctx):
    a, L, k = qids.shape[0], state["pid"].shape[1], s["k"]
    tab, tb, ts = pq_lut(queries, ctx.hbm("book").float())
    nv = ctx.hbm("nav_vec").float()
    d = (queries * queries).sum(1)[:, None] + (nv * nv).sum(1)[None, :] - 2. * queries @ nv.T
    t = torch.topk(d, min(L, nv.shape[0], ctx.knobs["entries"]), dim=1, largest=False)
    pid = torch.full((a, L), -1, dtype=torch.int64)
    pd = torch.full((a, L), float("inf"))
    pid[:, :t.indices.shape[1]] = ctx.hbm("nav_id", t.indices.reshape(-1)).reshape(a, -1)
    pd[:, :t.values.shape[1]] = t.values
    new = {"id": qids, "q": queries, "tab": tab, "tb": tb, "ts": ts, "pid": pid, "pd": pd, "px": torch.zeros(a, L, dtype=torch.bool),
           "rd": torch.full((a, k), float("inf")), "ri": torch.zeros(a, k, dtype=torch.int64), "out": torch.zeros(a, dtype=torch.int64)}
    return {key: torch.cat([state[key], new[key]]) for key in state}


# Starling's partitioner: ceil(n / per) blocks, all full but the last. A greedy pass in id order (a node, then its
# unplaced neighbours, into the current block until it is full); then shuffle_rounds LDG rounds: each node in a random
# order moves to the unfilled block with most of its in- and out-neighbours, weighted by the block's free share (any
# unfilled block if none). Neighbours not yet re-placed in a round count at their last block, as in the original.
def block_shuffle(graph, per, ctx):
    n = graph.shape[0]
    nb = [[v for v in row if v >= 0] for row in graph.tolist()]
    near = [list(row) for row in nb]
    for u in range(n):
        for v in nb[u]:
            near[v].append(u)
    blocks = -(-n // per)
    pid, size, p = [-1] * n, [0] * blocks, 0
    for u in range(n):
        if pid[u] >= 0:
            continue
        if size[p] == per:
            p += 1
        pid[u], size[p] = p, size[p] + 1
        for v in nb[u]:
            if pid[v] >= 0:
                continue
            if size[p] == per:
                p += 1
                break
            pid[v], size[p] = p, size[p] + 1
    for _ in range(ctx.knobs["shuffle_rounds"]):
        size, free = [0] * blocks, 0
        for u in torch.argsort(ctx.uniform(n)).tolist():
            count = {}
            for v in near[u]:
                count[pid[v]] = count.get(pid[v], 0) + 1
            best, top = -1, 0.
            for q, c in count.items():
                w = c * (1 - size[q] / per)
                if size[q] < per and w > top:
                    best, top = q, w
            if best < 0:
                while size[free] >= per:
                    free += 1
                best = free
            pid[u], size[best] = best, size[best] + 1
    order, fill = [-1] * (blocks * per), [0] * blocks
    for u in range(n):
        order[pid[u] * per + fill[pid[u]]] = u
        fill[pid[u]] += 1
    return torch.tensor(order, dtype=torch.int64)


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


# Beam rounds for the queries with nothing in flight: answer a query whose pool is fully expanded, else read the pages
# of its W best open candidates as one batch.
def expand(state, s, ctx):
    W, L = ctx.knobs["beam"], state["pid"].shape[1]
    per = 4096 // record_size(s, ctx.knobs["degree"])
    nslot = s["pages"] * per
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
        nodes = torch.gather(state["pid"][rows], 1, pick)[live]
        owner = rows[:, None].expand_as(pick)[live]
        ctx.submit(nodes // per, torch.zeros_like(nodes), 4096, state["id"][owner] * nslot + nodes, True)
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


# Largest PQ size that fits the HBM budget with the navigation sample; else the smallest size.
def pick_pq(index, s, ctx, nslot, extra):
    sizes = pq_sizes(index)
    for M in sizes:
        if nslot * M + index["book%d" % M].numel() * 2 + extra <= s["hbm_bytes"]:
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
