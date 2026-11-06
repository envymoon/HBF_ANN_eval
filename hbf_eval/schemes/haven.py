"""HAVEN-style IVF-PQ with re-ranking: PQ scan of the probed clusters on chip, exact re-rank reads from HBF."""

KNOBS = {
    'block': (8, 1, 64, 'int'),
    'chunk': (32, 1, 1024, 'int'),
    'kmeans_iters': (8, 1, 30, 'int'),
    'nlist': (1024, 8, 16384, 'int'),
    'nprobe_hi': (48, 1, 1024, 'int'),
    'nprobe_lo': (4, 1, 256, 'int'),
    'pq_bytes': (16, 1, 64, 'int'),
    'pq_centroids': (256, 16, 256, 'int'),
    'pq_sample': (20000, 1000, 200000, 'int'),
    'rerank_hi': (512, 10, 8192, 'int'),
    'rerank_lo': (32, 10, 4096, 'int'),
    'train_sample': (100000, 1000, 1000000, 'int'),
}


# Coarse k-means (trained on a sample) into nlist lists; residual PQ codebooks and codes at pq_bytes, /2, ... 1 bytes
# per vector.
def build(vectors, ctx):
    n = vectors.shape[0]
    K = max(1, min(ctx.knobs["nlist"], n // 8))
    sample = vectors[torch.argsort(ctx.uniform(n))[:min(n, ctx.knobs["train_sample"])]]
    cent, _ = kmeans(sample, K, ctx.knobs["kmeans_iters"], ctx)
    assign = nearest(vectors, cent)
    resid = vectors - cent[assign]
    out = {"cent": cent, "assign": assign}
    M = ctx.knobs["pq_bytes"]
    for _ in range(8):
        book, codes = pq_train(resid, M, ctx)
        out["book%d" % M], out["codes%d" % M] = book, codes
        if M == 1:
            break
        M = max(1, M // 2)
    return out


# HBF: the full vectors in id order. HBM: coarse centroids, codebook, the lists (largest code size that fits the
# budget).
def layout(index, vectors, ctx):
    s = ctx.spec()
    n, K, size = vectors.shape[0], index["cent"].shape[0], record_size(s)
    rec = pack_rows(vectors, s)
    rec = torch.cat([rec, torch.zeros(n, size - rec.shape[1], dtype=torch.uint8)], dim=1)
    cent = index["cent"].half()
    for M in pq_sizes(index):
        lists, lstart = list_blocks(index["assign"], index["codes%d" % M], K, ctx.knobs["block"])
        need = cent.numel() * 2 + index["book%d" % M].numel() * 2 + lists.numel() + lstart.numel() * 8
        if need <= s["hbm_bytes"]:
            break
    hbm = {"cent": cent, "book": index["book%d" % M].half(), "lists": lists, "lstart": lstart}
    return {"pages": to_pages(rec), "hbm": hbm, "sram": {}, "interleave": 1}


# Re-rank reads in (exact distances into the top k), new queries probed and their re-rank reads issued, finished queries
# answered.
def serve(state, events, ctx):
    s = ctx.spec()
    if "id" not in state:
        z = torch.zeros(0, dtype=torch.int64)
        state = {"id": z, "q": torch.zeros(0, s["dim"]), "rd": torch.zeros(0, s["k"]),
                 "ri": torch.zeros(0, s["k"], dtype=torch.int64), "out": z}
    if events["tags"].shape[0] > 0:
        state = rerank_in(state, events, s)
    if events["qids"].shape[0] > 0:
        state = probe(state, events["qids"], events["queries"], s, ctx)
    return answer(state, ctx)


# Answer the queries whose re-rank reads are all in.
def answer(state, ctx):
    done = state["out"] == 0
    if not bool(done.any()):
        return state
    ctx.finish(state["id"][done], state["ri"][done])
    keep = ~done
    return {key: v[keep] for key, v in state.items()}


# nprobe and N_rerank from ctx.effort, each between its lo and hi knob.
def effort_knobs(ctx, K):
    e = ctx.effort
    P = int(round(ctx.knobs["nprobe_lo"] + e * (ctx.knobs["nprobe_hi"] - ctx.knobs["nprobe_lo"])))
    R = int(round(ctx.knobs["rerank_lo"] + e * (ctx.knobs["rerank_hi"] - ctx.knobs["rerank_lo"])))
    return max(1, min(P, K)), max(1, R)


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


# The inverted lists as HBM rows: each list padded to whole blocks of B entries (M code bytes + int32 id, id -1 for
# padding); lstart[c] is list c's first block (lstart[K] the total).
def list_blocks(assign, codes, K, B):
    n, M = codes.shape
    order = torch.argsort(assign, stable=True)
    count = torch.bincount(assign, minlength=K)
    blocks = (count + B - 1) // B
    bstart = torch.cumsum(blocks, 0) - blocks
    lstart = torch.cat([bstart, bstart[-1:] + blocks[-1:]])
    a = assign[order]
    first = torch.cumsum(count, 0) - count
    pos = torch.arange(n) - first[a]
    slot = (bstart[a] + pos // B) * B + pos % B
    total = int(lstart[-1]) * B
    ids = torch.full((total,), -1, dtype=torch.int32)
    ids[slot] = order.to(torch.int32)
    ent = torch.zeros(total, M + 4, dtype=torch.uint8)
    ent[slot, :M] = codes[order]
    ent[:, M:] = ids.contiguous().view(torch.uint8).reshape(total, 4)
    return ent.reshape(total // B, B * (M + 4)), lstart


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


# Steps 1-3 for new queries: nearest nprobe lists, the PQ scan per chunk of queries, one Batch Read of all re-rank
# vectors.
def probe(state, qids, queries, s, ctx):
    cent, lstart, book = ctx.hbm("cent").float(), ctx.hbm("lstart"), ctx.hbm("book").float()
    P, R = effort_knobs(ctx, cent.shape[0])
    a, C = qids.shape[0], ctx.knobs["chunk"]
    lists = torch.topk(torch.cdist(queries, cent), P, dim=1, largest=False).indices
    parts = [shortlist(queries[i:i + C], lists[i:i + C], cent, book, lstart, R, s, ctx) for i in range(0, a, C)]
    ids = torch.cat(parts)
    ok = ids >= 0
    owner = torch.arange(a)[:, None].expand_as(ids)[ok]
    j = ids[ok]
    size = record_size(s)
    per = 4096 // size
    if j.shape[0] > 0:
        ctx.submit(j // per, (j % per) * size, size, qids[owner] * s["n"] + j, True)
    k = s["k"]
    new = {"id": qids, "q": queries, "rd": torch.full((a, k), float("inf")), "ri": torch.zeros(a, k, dtype=torch.int64),
           "out": ok.sum(dim=1)}
    return {key: torch.cat([state[key], new[key]]) for key in state}


# A full-precision vector on HBF, rounded up to the 64 B read granularity.
def record_size(s):
    return ((s["vector_bytes"] + 63) // 64) * 64


# Step 3: full vectors in, exact distances merged into each query's k best.
def rerank_in(state, events, s):
    n, vb, k = s["n"], s["vector_bytes"], s["k"]
    tags = events["tags"]
    qids, ids = tags // n, tags % n
    row = torch.searchsorted(state["id"], qids)
    d = ((unpack_rows(events["data"][:, :vb], s) - state["q"][row]) ** 2).sum(1)
    order, row_s, slot = slots(row)
    Q, W = state["id"].shape[0], int(slot.max()) + 1
    ed = torch.full((Q, W), float("inf"))
    ei = torch.zeros(Q, W, dtype=torch.int64)
    ed[row_s, slot] = d[order]
    ei[row_s, slot] = ids[order]
    all_d = torch.cat([state["rd"], ed], dim=1)
    top = torch.topk(all_d, k, dim=1, largest=False).indices
    return dict(state, rd=torch.gather(all_d, 1, top), ri=torch.gather(torch.cat([state["ri"], ei], dim=1), 1, top),
                out=state["out"] - torch.bincount(row, minlength=Q))


# Step 2 for a chunk of queries: read the probed lists' blocks from HBM, residual PQ distances (a table per query and
# list), the R best ids per query (-1 padded).
def shortlist(q, lists, cent, book, lstart, R, s, ctx):
    a, P = lists.shape
    M, C, ds = book.shape
    B = ctx.knobs["block"]
    b0 = lstart[lists].reshape(-1)
    nb = (lstart[lists + 1] - lstart[lists]).reshape(-1)
    pair = torch.repeat_interleave(torch.arange(a * P), nb)
    rows = b0[pair] + torch.arange(pair.shape[0]) - torch.repeat_interleave(torch.cumsum(nb, 0) - nb, nb)
    ent = ctx.hbm("lists", rows).reshape(-1, M + 4)
    pair = pair.repeat_interleave(B)
    ids = ent[:, M:].contiguous().view(torch.int32).reshape(-1).long()
    tab = pq_table((q[:, None, :] - cent[lists]).reshape(a * P, -1), book)
    idx = (pair[:, None] * M + torch.arange(M)[None, :]) * C + ent[:, :M].long()
    d = tab.reshape(-1)[idx].sum(1)
    d = torch.where(ids >= 0, d, torch.full_like(d, float("inf")))
    order, row_s, slot = slots(pair // P)
    W = int(slot.max()) + 1 if slot.shape[0] > 0 else 1
    dd = torch.full((a, W), float("inf"))
    ii = torch.full((a, W), -1, dtype=torch.int64)
    dd[row_s, slot] = d[order]
    ii[row_s, slot] = ids[order]
    top = torch.topk(dd, min(R, W), dim=1, largest=False)
    pick = torch.where(top.values < float("inf"), torch.gather(ii, 1, top.indices), torch.full_like(top.indices, -1))
    return torch.cat([pick, torch.full((a, R - pick.shape[1]), -1, dtype=torch.int64)], dim=1)


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
