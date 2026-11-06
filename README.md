# hbf_eval — an evaluator for ANN search on High Bandwidth Flash

`hbf_eval` scores an approximate-nearest-neighbour (ANN) search scheme on a simulated GPU platform that has **High Bandwidth
Flash (HBF)** next to HBM. A scheme is a small Python module; the evaluator runs it against an event-level model of the HBF
device (channels, dies, banks, page buffers, power budget, refresh) and reports how fast it is, relative to the obvious
alternative (a batched brute-force scan), at fixed recall.

It exists because the HBF simulator published with the HBF literature needs a specific NVIDIA GPU and runs LLM workloads, not
ANN search. Nothing here needs a GPU: the platform is simulated and the scheme's GPU work is *billed* from tensor shapes.

## Score

For each task: `speedup = log2(throughput of the scheme at recall@10 = R / throughput of batched brute force on the same device)`.

- 0 = as fast as the scan, +1 = twice as fast, −1 = half as fast. A scheme that never reaches recall R scores the floor, −8.
- Throughput is closed-loop: each client sends its next query when the last is answered. With few clients it is
  `clients / mean latency`, so low-load tasks compare latency.
- The score of a scheme is the mean over the four tasks, read at R = 0.90 (also 0.95 and 0.99).

| Task | What it tests | Data | Load | Device profile |
|---|---|---|---|---|
| `high_load` | throughput when parallel units are saturated | SIFT-like, 128-d uint8, L2 | 0.75 clients per parallel unit | rotates with the seed |
| `low_load_4` | latency: every dependent step waits for one read | SIFT-like | 4 clients | rotates with the seed |
| `power_bound` | the stack power budget binds before bandwidth | SIFT-like | 0.75 clients per unit | profile C |
| `data_deep` | another data distribution | DEEP-like, 96-d float32, L2 | 0.5 clients per unit | rotates with the seed |

Recall is measured against exact neighbours. Datasets are synthetic stand-ins of the same shape as the real sets (a mixture of
Gaussians in a low-dimensional latent space); put real data under `--data-root` as `<name>/base.npy, query.npy, meta.json`
(see `hbf_eval/data.py`). A suite with more tasks (tail latency, skew, scale, other datasets) is in `hbf_eval/tasks.py`.

## The platform model

- **Device** (`device.py`, same model in C in `device_c.c`, bit for bit): reads arrive at a channel's base die, are held by the
  base-die timer (or flushed by a Batch Read), dispatch to a bank (one plane), hit one of the bank's 2 page buffers or sense
  a page (tR with jitter, read-disturb charge), draw sense energy from the stack's power budget, cross the core-die channel
  into the base die, then the link. Refresh debt accrues per sense; programs hold a die or a plane; reads past the allowed debt
  retry. Based on the OCP HBF specification v0.7.0 and the HBF literature (HBFSim, *Characterizing HBF*, HBFlex, HAVEN).
- **Platform** (`arch.py`): 8 positions around the GPU, `m` of them HBM (default 4) with their own links, the rest HBF stacks.
- **Hardware profiles A–D** (`tasks.py`): tR 1.5–20 µs, 32–128 parallel units per channel, 24–192 GB/s per channel. The profile
  rotates with the seed; the firmware details a host cannot see (buffer replacement policy, base-die timer, in-flight cap,
  SRAM size, read-disturb threshold, HBM capacity factor, tR jitter) are drawn from the seed ("weather").
- **GPU compute** (`compute_table.py`): every torch operation a scheme runs in `serve` is billed from its shapes
  (a fixed cost per wake-up, on-chip bandwidth, a sync cost for every value read back to the host).
- **Brute-force reference and hardware roof** (`scoring.py`): the scan and the best any scheme could do on the platform.

## Run

```bash
pip install -r requirements.txt             # torch, numpy; a C compiler speeds up the device model (falls back to the Python reference)
python3 -m unittest discover tests          # device known-answer tests, scheme loader
python3 -m hbf_eval.hbfcheck                # the 25 HBF phenomena checks, with their numbers
python3 -m hbf_eval.bench list              # names of the bundled schemes
```

### Scoring schemes: `python3 -m hbf_eval.bench run`

`--schemes` takes a comma-separated list; each item is either a bundled name (`haven`) or **a path to your own file**
(`my/scheme.py`). Without `--schemes` it runs `baseline, diskann, starling, pipeann, pageann, haven`.

```bash
# one bundled scheme
python3 -m hbf_eval.bench run --schemes haven --scale 100k --data clustered
# your own file next to two references, 3 processes in parallel
python3 -m hbf_eval.bench run --schemes my/scheme.py,baseline,haven --workers 3
# your own file with two knobs changed and an 8 GiB memory cap
python3 -m hbf_eval.bench run --schemes my/scheme.py --set nprobe_hi=96 --set nlist=2048 --mem-gb 8
python3 -m hbf_eval.bench report                      # table of everything in results/results.jsonl
python3 -m hbf_eval.bench check my/scheme.py          # load a scheme file, list its knobs; runs nothing (seconds)
python3 -m hbf_eval.bench compare results/results.jsonl reference/results_100k.jsonl   # exit 1 on any difference
```

| Option | Meaning | Default |
|---|---|---|
| `--schemes a,b,path.py` | what to run | baseline + 5 published |
| `--scale 100k \| 1m` | vectors per dataset (1m is the 1M-vector setting, much slower and larger) | `100k` |
| `--data clustered \| flat \| ..._mK` | dataset variant, see below | `clustered` |
| `--wide` | open every `*_hi` knob to its declared maximum | off |
| `--set KNOB=VALUE` | override a declared knob (repeatable); clamped to its range; an unknown name is an error | knob defaults |
| `--workers N` | how many scheme processes run at once (one process = one scheme x one data x one scale) | 4 |
| `--mem-gb G` | address-space limit of *each* process (a cap, not a reservation) | 12 (100k), 14 (1m) |
| `--out DIR` | results (`results.jsonl`, `driver.log`, one `stderr_*.log` per job) | `results/` |
| `--cache DIR` | built indexes and layouts, keyed by the code and knobs they depend on, so a rerun that changes only `serve` does not rebuild | `cache/` |
| `--data-root DIR` | real datasets instead of the synthetic stand-ins | synthetic |

**Resources.** Each process uses one CPU thread for torch (fixed) and runs its four tasks one after the other, so wall time is
set by the slowest scheme and total memory is at most `workers x mem-gb`. If your machine has less memory, lower `--workers`,
not `--mem-gb` (a process that hits the cap fails with a memory error and is recorded with a nonzero `status`). On a
4-core, 15 GB machine the six default schemes at 100k took about 17 minutes wall time with 4 workers (HAVEN about 1 minute,
the graph schemes 7-16 minutes each). Finished (scheme, data, scale) jobs are skipped on rerun, so an interrupted run resumes.
With `--set` or `--wide` the result is labelled `scheme@wide` / `scheme@knob=value`, so variants sit side by side in `report`.

### Why `clustered`, `flat` and `--wide` exist

They answer one question: *is this scheme's score a property of the scheme, or of the benchmark setting it was run in?*
You do not need them for a first run; the defaults are the standard setting.

- **`clustered` (default)** - synthetic SIFT-like and DEEP-like data with well-separated clusters, roughly like real descriptor sets.
  This is the standard setting and what all scores in this README use. Schemes that exploit cluster structure (IVF lists, cluster-ordered
  layouts, centroid entry points) have an easy time on it.
- **`flat`** - the same shapes, but weakly clustered: no clear cluster boundaries, a harder case for anything that routes by cluster.
  Run it next to `clustered`; if a scheme's score collapses on `flat`, its result depended on the clusters, not on the memory system.
- **`clustered_m1`, `flat_m2`, ...** - the suffix `_mK` runs on a platform with K HBM positions out of 8 instead of 4, to see how much
  a scheme depends on having a lot of HBM.
- **`--wide`** - a scheme maps the host's effort (0 to 1) onto its search effort between a `*_lo` and a `*_hi` knob (for example `nprobe_lo` to
  `nprobe_hi`). The default `*_hi` is a moderate value, so a scheme may never reach recall 0.99 and then scores the floor (-8) at that level.
  `--wide` opens every `*_hi` knob to its declared maximum. Use it when the 0.95 or 0.99 column is -8, to tell "this scheme cannot reach that
  recall" from "its default range stops short". It changes nothing for a scheme without `*_hi` knobs.
- **Real data** - see below. With `--data-root` the tasks use the real SIFT and DEEP files instead of the synthetic stand-ins.

### What the score does not measure

The score covers **query serving on the simulated device**: `serve`, the HBF reads it issues, and the GPU work it runs. The offline stages
are free and unmeasured:

- **Index construction and preprocessing** (`build`: graph construction, k-means, PQ training, sorting, reordering) is not timed, not
  charged to the device, and its cost does not enter the score. The only limit is a wall-clock budget per scheme (1 h at 100k), and the
  size of `build` is not part of the result. A scheme that needs hours of preprocessing scores the same as one that needs seconds.
- **Writing the index to HBF** (`layout`) is free too: the pages appear on the device before the first query. Program time, write energy
  and wear for loading the index are not modelled; only refresh and programs *during* serving (refresh debt, `ctx.refresh`) are.
- **Updates and deletions**, **index size on flash** beyond the capacity limit, **endurance**, and **tail recall** (recall is the mean
  over queries; the worst-5% value is recorded in the diagnostics but not scored).

If your idea trades preprocessing for query speed (a better layout that takes a long time to compute), the score shows only the query-side gain.

### Real datasets

By default the tasks run on synthetic stand-ins with the shape of SIFT (128-d uint8, L2) and DEEP (96-d float32, angular). To use the real
files, prepare them once and pass `--data-root`:

```bash
pip install h5py pyarrow                    # only for the preparation script
python3 tools/prepare_dataset.py sift-128-euclidean.hdf5 data/sift --dtype uint8 --metric l2
python3 tools/prepare_dataset.py --parquet-base train-00001-of-00009.parquet --parquet-query test-00001-of-00001.parquet \
        data/deep --dtype float32 --metric angular --max-base 200000
python3 -m hbf_eval.bench run --schemes my/scheme.py,haven --data-root data --out results_real --cache cache_real
```

`sift-128-euclidean.hdf5` (about 0.5 GB, 1M x 128) is the ann-benchmarks file; the DEEP shard is the ann-benchmarks `deep-image-96-angular`
data (one train shard of about 0.45 GB plus the test file). The tasks take a fixed random 100K-vector subset of the base and 3072 of the
queries, and compute exact neighbours themselves. Use a separate `--out` and `--cache` for real data so it is not mixed with the synthetic
results.

## Reference schemes (`hbf_eval/schemes/`)

| File | Scheme |
|---|---|
| `baseline.py` | textbook proximity graph: RobustPrune graph, one record per node on HBF, entry at node 0, one expansion at a time, nothing in HBM |
| `diskann.py` | DiskANN: Vamana graph, PQ codes in HBM for routing, full records from HBF, beam search |
| `starling.py` | Starling: block shuffling of the layout plus an in-memory navigation graph |
| `pipeann.py` | PipeANN: pipelined, speculative beam |
| `pageann.py` | PageANN: page-aligned graph, neighbour lists and PQ codes of a page's vectors stored together |
| `haven.py` | HAVEN-style IVF-PQ: PQ scan of the probed clusters on chip, exact re-rank reads from HBF |
| `cagra.py` | CAGRA-like fixed-degree graph with batched search |

These are ports into the scheme language, not the authors' code; see each file's header and the papers for the originals.

### Our graph schemes (`compass.py` ... `cache.py`)

Seven variants of the baseline graph search, each adding mechanisms that shorten the dependent chain of reads, cut the number of
page reads, or use the page buffers. Scores are the mean of the four tasks at recall@10 = 0.90, 100K vectors (a 5-point effort grid, so they
differ slightly from `bench` output, which uses 9 points).

| Scheme | Score | Mechanisms |
|---|---|---|
| `compass.py` | −0.36 | entry table in HBM + locality-reordered layout + no host syncs |
| `atlas.py` | −0.42 | cluster-ordered layout + cluster-centroid entry |
| `router.py` | −0.48 | beam of 2 + query-routed entry |
| `reorder.py` | −0.65 | entry table + locality-reordered layout |
| `entry.py` | −0.80 | entry table only |
| `batch.py` | −0.90 | cluster-ordered layout + wake-up batching |
| `cache.py` | −1.00 | HBM cache of the adjacency lists of high in-degree nodes |
| `baseline.py` | −1.58 | none |

Their knob values are the tuned ones; `reference/results_100k.jsonl` has the `bench` output of Compass, Atlas and Router.

## Figures

`docs/figs/` has the charts of the first results (scoring, device-model checks, rankings per task, recall levels, weakly clustered data).

## Writing a scheme

A scheme is one `.py` file. To test an idea you write that file; there is nothing else to register. Start by copying a bundled
scheme that is closest to your idea (`baseline.py` for a graph, `haven.py` for IVF/PQ, `pageann.py` for a page-aligned graph) and change it.

### From an idea to the three functions

Think of the idea in three stages, which are the three entry functions:

| Stage | Function | What you decide | Time is charged? |
|---|---|---|---|
| Build the index | `build(vectors, ctx) -> index` | the data structure: graph, clusters, PQ codebooks, entry points. `vectors` is a float32 `[n, D]` tensor | no |
| Place it | `layout(index, vectors, ctx) -> plan` | which bytes go to HBF pages, HBM, or per-channel SRAM, and how pages interleave over channels | no |
| Answer queries | `serve(state, events, ctx) -> state` | what to read now, what to compute with what came back, when to answer | **yes** |

Ask of each part of your idea: *is it computed once, or per query?* Once: `build` (or `layout` if it is a placement decision).
Per query: `serve`. A "table in HBM" is built in `build`, placed in `layout` under `plan["hbm"]`, read in `serve` with `ctx.hbm`.
A "better page layout" is a `layout` decision (the order of records in `plan["pages"]`, and `plan["interleave"]`).
A "prefetch / speculate / batch" idea is a `serve` decision (which `ctx.submit` calls, with `batch=True` or not, and `ctx.wait`).

`serve` has no access to `vectors` or `index`: it sees only what `layout` placed, and every read goes through `ctx`
(HBF reads cost device time; HBM and SRAM reads cost their bandwidth and latency). That is the constraint that makes the
comparison fair, and also what you design around: whatever `serve` needs must be placed by `layout` first.

### How `serve` works

`serve` is called at every wake with `events`: new queries (`qids`, `queries`) and completed HBF reads (`tags`, `nbytes`,
`data`, the bytes you asked for). It is a state machine, because a read you issue now completes later:

1. For completed reads: use `data` (the bytes) and `tags` (whatever integer you attached to the read, e.g. `qid * n + node`).
2. For new queries: choose where to start and issue reads with `ctx.submit(pages, offsets, nbytes, tags)`.
3. Decide what to do next for each query: issue more reads, or `ctx.finish(qids, ids)` with `ids` an int64 `[len(qids), k]`.
4. Return `state`, a dict of tensors you keep between calls (capped at 64 KiB per client).

Process all queries of a wake together with batched tensor operations: one op over all queries costs about the same as one op
for one query. Turning a tensor into a Python value (`.item()`, `.tolist()`, `bool()`, `int()`) is billed as a read-back to the host,
so avoid it in loops. `ctx.effort` (0 to 1) is how the host sweeps search effort to draw the recall-throughput curve: map it
to your beam width, probes, re-rank count and so on; the score reads that curve at recall 0.90, 0.95 and 0.99.

### Skeleton

```python
"""My idea in one sentence."""

KNOBS = {                      # name: (default, low, high, "int" | "log" | "linear")
    "nlist": (1024, 8, 16384, "int"),
    "probe_lo": (4, 1, 256, "int"),
    "probe_hi": (48, 1, 1024, "int"),     # *_hi is what --wide opens
}

def build(vectors, ctx):                   # offline; ctx.knobs["nlist"], ctx.uniform(n), ctx.spec()
    return {"centroids": ...}

def layout(index, vectors, ctx):           # offline; ctx.spec() has the platform and device figures
    return {"pages": ...,                  # uint8 [P, 4096]: the HBF image
            "hbm": {"centroids": ...},     # tensors you will read with ctx.hbm
            "sram": {}, "interleave": 1}

def serve(state, events, ctx):             # online; the only timed stage
    ...                                    # read events, ctx.submit(...), ctx.finish(...)
    return state
```

This is an outline, not a runnable scheme: `haven.py` is a complete worked example of exactly this shape (IVF lists in HBM,
re-rank reads from HBF in one Batch Read).

### Rules the loader enforces

- Plain top-level `def`s, one `KNOBS` literal and a docstring; nothing else at module level. `torch`, `F` (torch.nn.functional)
  and `math` are available without importing.
- A knob is read as `ctx.knobs["name"]` with a literal name and must be declared in `KNOBS`. `build` and `layout` results are
  cached on the functions they reach and the knobs those read, so changing a knob only `serve` reads never rebuilds the index.
- No `import`, file or network IO, global random state (use `ctx.uniform`), or names starting with `_`; at most 60 functions,
  80 lines each, 160 000 characters. This is a static policy that keeps scheme code deterministic and away from the evaluator's
  state; it is not a security boundary, so run code you do not trust in a container.
- The full text of the interface, with every `ctx` method and the `spec` fields, is `TEXT` in `hbf_eval/contract.py`.

### Check, run, read

```bash
python3 -m hbf_eval.bench check my/scheme.py                      # loader and policy errors, knob table, which knobs build/layout read
python3 -m hbf_eval.bench run --schemes my/scheme.py,baseline     # score next to the baseline
python3 -m hbf_eval.bench report
```

`results/results.jsonl` has, per task, the speedups at 0.90 / 0.95 / 0.99, the recall-throughput curve, the bounds and a
`diag` block at the point nearest recall 0.95: `reads_per_query`, `senses_per_query` (page senses, which is what the device
pays for), `hit_rate` (page-buffer hits), `unit_util` (how busy the parallel units are), `compute_share` and `wake_share`
(time spent in GPU work and in wake-ups), `power_wait_share` (waiting for the power budget). Read them to see whether your idea
moved the thing you meant to move. A `failure` string means the scheme was rejected or crashed in that task, with the reason.

## Layout

```
hbf_eval/  device.py device_c.c fastdevice.py   device model (Python reference, C)
           arch.py tasks.py scoring.py data.py  platform, tasks and weather, scores, datasets
           ctx.py sim.py compute_table.py       the scheme's interface, the closed-loop run, GPU billing
           scheme.py policy.py contract.py      scheme files, code policy, interface text
           runner.py bench.py hbfcheck.py       one evaluation, the command line, the HBF checks
           schemes/                             reference schemes
tests/     reference/                           tests; the scores the reference schemes got when this evaluator was split out
```

## Current limitations

The framework is still under development. These are known limits of the current version, not design choices; they are planned work.

- **No dynamic updates.** The index is built once and read-only: no insertion, deletion or rebuild during serving, and writes are not simulated.
- **One platform: GPU + HBM + HBF.** No host CPU computation, no host memory or SSD tier, no multiple GPUs or network. Designs that depend on these cannot be expressed.
- **GPU computation is billed from tensor shapes.** Custom CUDA kernels, operator fusion and other low-level optimisations can be neither expressed nor measured.
- **The score covers throughput and latency only.** Build time, preprocessing, index size, update performance and tail recall do not affect it.
