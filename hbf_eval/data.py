"""Datasets, subsets and exact neighbours . The exact neighbours never leave the evaluator.

A prepared dataset is a directory <root>/<name>/ with base.npy (uint8 or float32 [n, D]), query.npy (float32 [q, D]) and
meta.json {"metric": "l2" | "angular", "dtype": "uint8" | "float32"}. Without one, a synthetic stand-in of the same shape is
generated from the name, and every record says so ("synthetic": true). Stand-in: a mixture of Gaussians in a low-dimensional
latent space (clusters of different widths), mapped to D dimensions by a random linear map, plus a little isotropic noise, so
the local intrinsic dimension is near the real set's (about 15 to 30) and not D: clean clusters in D dimensions make every
neighbour equidistant, which no real set does and which breaks graph pruning.

Base subset: fixed per (dataset, n), so build is cached across seeds . Query pool: fixed per
dataset; each seed draws its own order from it (tasks.py). Angular data is unit-normalised, so L2 order is angular order.
"""
import hashlib
import json
import os
import tempfile

# name -> (dim, metric, dtype, latent dim, clusters, spread, noise): stand-ins shaped like the real sets
STANDINS = {"sift": (128, "l2", "uint8", 16, 64, 2., 1.), "deep": (96, "l2", "float32", 12, 48, 2., .5),
            "glove": (100, "angular", "float32", 24, 80, 1.5, 1.5), "gist": (960, "l2", "float32", 24, 32, 2., 1.),
            "t2i": (200, "angular", "float32", 20, 40, 2., 1.), "tiny": (32, "l2", "float32", 8, 8, 2., .5),
            # weakly clustered: same shape and latent dimension as sift / deep, but 1024
            # overlapping components with centres closer than their widths, so there are no separable clusters
            "sift_flat": (128, "l2", "uint8", 16, 1024, .5, 1.), "deep_flat": (96, "l2", "float32", 12, 1024, .5, .5)}
STANDIN_VERSION = "latent-v2"
BASE_POOL = 1000  # queries drawn jointly with a stand-in's base ; more come from a separate stream
_MEMO = {}


def stream_seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little") >> 1


def standin(name, n, pool):
    """The base and the first BASE_POOL queries come from one joint draw (latent-v2, unchanged so that the base, and every
    index built on it, stays the same); queries beyond BASE_POOL come from their own stream, from the same mixture."""
    import torch
    dim, metric, dtype, latent, clusters, spread, noise = STANDINS[name]
    g = torch.Generator().manual_seed(stream_seed("standin", STANDIN_VERSION, name))
    centres = torch.randn(clusters, latent, generator=g) * spread
    weights = torch.rand(clusters, generator=g) + .2
    widths = torch.rand(clusters, latent, generator=g) * 1.5 + .25
    first = BASE_POOL
    which = torch.multinomial(weights, n + first, replacement=True, generator=g)
    z = centres[which] + torch.randn(n + first, latent, generator=g) * widths[which]
    shift = None
    if name == "t2i":  # queries from a shifted distribution, as in Text2Image
        shift = .6 * torch.randn(1, latent, generator=g)
        z[n:] = z[n:] * .8 + shift
    lift = torch.randn(latent, dim, generator=g) / latent ** .5
    points = z @ lift + noise * torch.randn(n + first, dim, generator=g)
    lo, hi = points[:n].min(), points[:n].max()
    queries = points[n:n + min(pool, first)]
    if pool > first:
        h = torch.Generator().manual_seed(stream_seed("standin-queries", STANDIN_VERSION, name))
        extra = pool - first
        w2 = torch.multinomial(weights, extra, replacement=True, generator=h)
        z2 = centres[w2] + torch.randn(extra, latent, generator=h) * widths[w2]
        if shift is not None:
            z2 = z2 * .8 + shift
        queries = torch.cat([queries, z2 @ lift + noise * torch.randn(extra, dim, generator=h)])
    base = points[:n]
    if dtype == "uint8":
        base = ((base - lo) / (hi - lo) * 255.).round().clamp(0., 255.)
        queries = ((queries - lo) / (hi - lo) * 255.).round().clamp(0., 255.)
    return base.contiguous(), queries.contiguous(), metric, dtype


def prepared(root, name, n, pool):
    import numpy as np
    import torch
    folder = os.path.join(root, name)
    meta = json.load(open(os.path.join(folder, "meta.json")))
    base = np.load(os.path.join(folder, "base.npy"), mmap_mode="r")
    query = np.load(os.path.join(folder, "query.npy"), mmap_mode="r")
    if n > base.shape[0] or pool > query.shape[0]:
        raise ValueError("%s has %d base and %d query vectors; asked for %d and %d" % (name, base.shape[0], query.shape[0], n, pool))
    g = torch.Generator().manual_seed(stream_seed("subset", name, n))
    pick = torch.randperm(base.shape[0], generator=g)[:n].sort().values.numpy()
    return (torch.from_numpy(np.ascontiguousarray(base[pick]).astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(query[:pool]).astype(np.float32)), meta["metric"], meta["dtype"])


def neighbours(base, queries, k, chunk=512):
    import torch
    out = []
    norms = (base * base).sum(1)
    for i in range(0, queries.shape[0], chunk):
        q = queries[i:i + chunk]
        d = norms[None, :] - 2. * q @ base.T  # + |q|^2, which does not change the order
        out.append(torch.topk(d, k, dim=1, largest=False).indices)
    return torch.cat(out) if out else torch.zeros(0, k, dtype=torch.int64)


def load(name, n, pool, k, root=None, cache=None):
    """{"base", "queries", "truth", "metric", "dtype", "vector_bytes", "synthetic", "id"}; memoised per process."""
    import torch
    key = (name, n, pool, k)
    if key in _MEMO:
        return _MEMO[key]
    real = bool(root) and os.path.isdir(os.path.join(root, name))
    base, queries, metric, dtype = prepared(root, name, n, pool) if real else standin(name, n, pool)
    if metric == "angular":
        base = base / base.norm(dim=1, keepdim=True).clamp(min=1e-12)
        queries = queries / queries.norm(dim=1, keepdim=True).clamp(min=1e-12)
    # "id" names the base (the build cache key): for a stand-in it keeps the first form, whose base was drawn with
    # BASE_POOL queries, so indexes built then stay valid whatever the pool; "queries_id" names the query pool too.
    ident = "real-%s-n%d" % (name, n) if real else "standin-%s-%s-n%d-q%d" % (STANDIN_VERSION, name, n, BASE_POOL)
    qident = "%s-%s-n%d-q%d" % ("real" if real else "standin-" + STANDIN_VERSION, name, n, pool)
    truth, path = None, os.path.join(cache, "truth", qident + "-k%d.pt" % k) if cache else None
    if path and os.path.exists(path):
        try:
            saved = torch.load(path, weights_only=True)
            if saved["check"] == checksum(base, queries):
                truth = saved["truth"]
        except Exception:
            truth = None
    if truth is None:
        truth = neighbours(base, queries, k)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    torch.save({"truth": truth, "check": checksum(base, queries)}, f)
                os.replace(tmp, path)  # private name, then rename: parallel workers cannot collide
            except OSError:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    item = {"base": base, "queries": queries, "truth": truth, "metric": metric, "dtype": dtype, "dim": base.shape[1],
            "vector_bytes": base.shape[1] * (1 if dtype == "uint8" else 4), "synthetic": not real, "id": ident,
            "queries_id": qident}
    if len(_MEMO) >= 3:
        _MEMO.pop(next(iter(_MEMO)))
    _MEMO[key] = item
    return item


def checksum(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()
