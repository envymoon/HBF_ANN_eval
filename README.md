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
pip install -r requirements.txt            # torch, numpy; a C compiler speeds up the device model (it falls back to the Python reference)
python3 -m unittest discover tests          # device known-answer tests, HBF phenomena checks, scheme loader
python3 -m hbf_eval.hbfcheck                # the 25 HBF checks with their numbers
python3 -m hbf_eval.bench run --scale 100k --data clustered --workers 4    # baseline + DiskANN, Starling, PipeANN, PageANN, HAVEN-style
python3 -m hbf_eval.bench report
python3 -m hbf_eval.bench compare results/results.jsonl reference/results_100k.jsonl
```

`--data flat` uses weakly clustered stand-ins (no separable clusters); `--wide` opens every effort range to its limit;
`--data clustered_m1` runs the same data on a platform with one HBM position. `--scale 1m` is the 1M-vector setting.

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
page reads, or use the page buffers. Scores are the mean of the four tasks at recall@10 = 0.90, 100K vectors, from the report
(`docs/report.md`, section 6; a 5-point effort grid, so they differ slightly from `bench` output, which uses 9 points).

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

## Report

`docs/report.md` is the write-up: why a new evaluator, how scores are computed, how the device model was checked (HBF literature
phenomena, an independent MQSim-based simulator, parameter perturbations, 100K vs 1M rankings), and the results with figures
(`docs/figs/`).

## Writing a scheme

A scheme file has a docstring, a `KNOBS` table and top-level functions: `build(vectors, ctx)`, `layout(index, vectors, ctx)`,
`serve(state, events, ctx)` plus any helpers. `hbf_eval/contract.py` documents the `ctx` interface (issue HBF reads, read HBM and
SRAM tensors, answer queries, wait, refresh). A static policy (`policy.py`) keeps scheme code deterministic and away from the
evaluator's state: no imports, no file IO, no global RNG, no names starting with `_`. It is defence in depth, not a security
boundary; run code you do not trust in a container.

```bash
python3 -m hbf_eval.bench run --schemes path/to/my_scheme.py,diskann --scale 100k
```

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
