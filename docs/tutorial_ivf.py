"""Tutorial scheme: IVF-Flat on HBF. k-means clusters; each cluster's full vectors packed into its own run of HBF pages;
centroids and the page range of every cluster in HBM. A query reads every page of its nprobe nearest clusters and keeps
the k nearest records it sees. Written to be read alongside the manual (docs/index.html), not to be fast."""

KNOBS = {
    "nlist": (256, 8, 4096, "int"),        # number of clusters
    "kmeans_iters": (8, 1, 30, "int"),     # Lloyd iterations in build
    "nprobe_lo": (2, 1, 256, "int"),       # clusters probed at effort 0
    "nprobe_hi": (32, 1, 1024, "int"),     # clusters probed at effort 1 (--wide opens it to 1024)
}


# ---------------------------------------------------------------- build: offline, not timed
def build(vectors, ctx):
    n = vectors.shape[0]
    K = min(ctx.knobs["nlist"], n)
    cent = vectors[torch.argsort(ctx.uniform(n))[:K]].clone()      # K random vectors as the first centroids
    for it in range(ctx.knobs["kmeans_iters"]):
        assign = nearest(vectors, cent)
        sums = torch.zeros_like(cent).index_add_(0, assign, vectors)
        count = torch.bincount(assign, minlength=K).to(vectors.dtype)
        cent = torch.where(count[:, None] > 0, sums / count.clamp(min=1.)[:, None], cent)
    return {"cent": cent, "assign": nearest(vectors, cent)}


# Index of the nearest row of c for every row of x, in chunks of 8192 rows to bound memory.
def nearest(x, c):
    cc = (c * c).sum(1)
    out = []
    for i in range(0, x.shape[0], 8192):
        out.append(torch.argmin(cc[None, :] - 2. * x[i:i + 8192] @ c.T, dim=1))
    return torch.cat(out)


# ---------------------------------------------------------------- layout: offline, not timed
def layout(index, vectors, ctx):
    s = ctx.spec()
    n, K = vectors.shape[0], index["cent"].shape[0]
    size = record_size(s)                     # bytes of one record: 4-byte id + the vector, rounded up to 64
    per = 4096 // size                        # records per page
    assign = index["assign"]
    order = torch.argsort(assign, stable=True)          # vector ids grouped by cluster
    count = torch.bincount(assign, minlength=K)         # vectors per cluster
    npages = (count + per - 1) // per                   # pages per cluster (each cluster starts on a fresh page)
    first_page = torch.cumsum(npages, 0) - npages       # first page of each cluster
    first_rec = torch.cumsum(count, 0) - count          # first position of each cluster in `order`
    a = assign[order]
    pos = torch.arange(n) - first_rec[a]                # position of each vector inside its cluster
    slot = (first_page[a] + pos // per) * per + pos % per   # global record slot = page * per + slot in page
    P = int(npages.sum())
    body = torch.zeros(P * per, size, dtype=torch.uint8)
    body[:, :4] = 255                                   # id -1 (four 0xFF bytes) marks an empty slot
    body[slot, :4] = order.to(torch.int32).view(torch.uint8).reshape(n, 4)
    body[slot, 4:4 + s["vector_bytes"]] = pack_rows(vectors[order], s)
    pages = torch.zeros(P, 4096, dtype=torch.uint8)
    pages[:, :per * size] = body.reshape(P, per * size)
    lpage = torch.cat([first_page, first_page[-1:] + npages[-1:]])   # cluster c = pages lpage[c] .. lpage[c+1]-1
    return {"pages": pages, "hbm": {"cent": index["cent"], "lpage": lpage}, "sram": {}, "interleave": 1}


def record_size(s):
    return ((4 + s["vector_bytes"] + 63) // 64) * 64


# Vectors to their stored bytes: uint8 data one byte per value, float32 four.
def pack_rows(v, s):
    if s["dtype"] == "uint8":
        return v.round().clamp(0, 255).to(torch.uint8)
    return v.contiguous().view(torch.uint8)


def unpack_rows(raw, s):
    if s["dtype"] == "uint8":
        return raw.float()
    return raw.contiguous().view(torch.float32)


# ---------------------------------------------------------------- serve: online, timed
def serve(state, events, ctx):
    s = ctx.spec()
    if "qid" not in state:
        state = empty_state(s)
    if events["tags"].shape[0] > 0:
        state = absorb(state, events, s)
    if events["qids"].shape[0] > 0:
        state = start(state, events["qids"], events["queries"], s, ctx)
    return answer(state, ctx)


# One row per open query, rows kept in qid order (qids arrive in increasing order and rows are only appended or removed).
def empty_state(s):
    k = s["k"]
    return {"qid": torch.zeros(0, dtype=torch.int64),          # query id
            "q": torch.zeros(0, s["dim"]),                     # query vector
            "bd": torch.zeros(0, k),                           # best distances so far
            "bi": torch.zeros(0, k, dtype=torch.int64),        # their ids
            "left": torch.zeros(0, dtype=torch.int64)}         # page reads still in flight


def nprobe(ctx, K):
    lo, hi = ctx.knobs["nprobe_lo"], ctx.knobs["nprobe_hi"]
    return max(1, min(K, int(round(lo + ctx.effort * (hi - lo)))))


# New queries: nearest clusters from the HBM centroids, then one read per page of those clusters.
def start(state, qids, queries, s, ctx):
    cent = ctx.hbm("cent")                                    # [K, D]: waits for the HBM transfer
    lpage = ctx.hbm("lpage")                                  # [K + 1]
    a, P = qids.shape[0], nprobe(ctx, cent.shape[0])
    lists = torch.topk(torch.cdist(queries, cent), P, dim=1, largest=False).indices   # [a, P]
    first, cnt = lpage[lists].reshape(-1), (lpage[lists + 1] - lpage[lists]).reshape(-1)
    pair = torch.repeat_interleave(torch.arange(a * P), cnt)  # one entry per page to read
    base = torch.cumsum(cnt, 0) - cnt
    pages = first[pair] + torch.arange(pair.shape[0]) - base[pair]
    owner = pair // P                                         # row of the new query that reads it
    size = record_size(s)
    used = ((4096 // size) * size + 63) // 64 * 64            # bytes of a page that hold records
    ctx.submit(pages, torch.zeros_like(pages), used, qids[owner] * s["pages"] + pages, batch=True)
    k = s["k"]
    new = {"qid": qids, "q": queries, "bd": torch.full((a, k), float("inf")),
           "bi": torch.zeros(a, k, dtype=torch.int64), "left": torch.bincount(owner, minlength=a)}
    return {key: torch.cat([state[key], new[key]]) for key in state}


# Completed reads: decode the tag, compute exact distances of every record on the page, merge into each query's k best.
def absorb(state, events, s):
    size, vb, k = record_size(s), s["vector_bytes"], s["k"]
    per = 4096 // size
    tags = events["tags"]
    row = torch.searchsorted(state["qid"], tags // s["pages"])        # which state row each completion belongs to
    rec = events["data"][:, :per * size].reshape(-1, per, size)       # [c, per, size]
    ids = rec[:, :, :4].contiguous().view(torch.int32).squeeze(2).long()   # [c, per]
    vec = unpack_rows(rec[:, :, 4:4 + vb], s)                          # [c, per, D]
    d = ((vec - state["q"][row][:, None, :]) ** 2).sum(2)
    d = torch.where(ids >= 0, d, torch.full_like(d, float("inf")))
    crow = row[:, None].expand(-1, per).reshape(-1)
    order = torch.argsort(crow, stable=True)
    rs = crow[order]
    col = torch.arange(rs.shape[0]) - torch.searchsorted(rs, rs)      # position of each candidate within its row
    Q, W = state["qid"].shape[0], int(col.max()) + 1
    cd = torch.full((Q, W), float("inf"))
    ci = torch.zeros(Q, W, dtype=torch.int64)
    cd[rs, col] = d.reshape(-1)[order]
    ci[rs, col] = ids.reshape(-1)[order].clamp(min=0)
    alld, alli = torch.cat([state["bd"], cd], 1), torch.cat([state["bi"], ci], 1)
    top = torch.topk(alld, k, dim=1, largest=False).indices
    return dict(state, bd=torch.gather(alld, 1, top), bi=torch.gather(alli, 1, top),
                left=state["left"] - torch.bincount(row, minlength=Q))


# Answer every query whose reads are all in, and drop its row.
def answer(state, ctx):
    done = state["left"] == 0
    if not bool(done.any()):
        return state
    ctx.finish(state["qid"][done], state["bi"][done])
    keep = ~done
    return {key: v[keep] for key, v in state.items()}
