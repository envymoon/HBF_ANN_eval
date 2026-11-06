"""The interface a scheme implements (TEXT), and the entry functions the evaluator calls."""
from .scheme import ENTRIES, LIMITS

CONTRACT = "hbf.ann.v2"

TEXT = """## Scheme interface
A scheme is a Python module of top-level functions plus a KNOBS table (see scheme.py). The functions call each other by name. Entry functions (exact arity):
  build(vectors, ctx) -> index        offline, once per dataset. vectors float32[n, D], free to read. index: {name: tensor}, any content.
                                      ctx.spec(): n, dim, metric, k, vector_bytes, dtype. Cached on the functions build reaches and the knobs they read.
  layout(index, vectors, ctx) -> plan offline. ctx.spec() adds the platform and the device's datasheet figures: m (HBM positions out of 8,
                                      fixed by the platform; the other 8 - m are HBF), positions, channels_per_position, units_per_channel,
                                      dies_per_channel, banks_per_die, tR_us, channel_GBps, coredie_channels, power_W_per_position,
                                      sense_uJ, program_uJ, program_us, program_holds ("die" or "plane"), pages_per_block,
                                      refresh_slack_pages, hbm_GBps_per_position, hbm_latency_us, hbm_bytes_per_position, hbm_bytes,
                                      hbf_pages_per_position, sram_bytes_per_channel, sram_latency_us, mocs.
      plan = {"pages": uint8[P, 4096],   the HBF image: page p holds whatever bytes you put there (P <= (8 - m) * hbf_pages_per_position)
              "hbm": {name: tensor},     HBM, at most hbm_bytes in total
              "sram": {name: tensor},    per-channel scratchpad SRAM: row r on channel r mod channels, each channel at most sram_bytes_per_channel
              "interleave": int,         global page -> channel interleave in pages (>= 1): channel = (page // interleave) % channels
              "refresh": str}            optional. "default": the host's stock refresh policy pays read-disturb refresh when a die is idle,
                                         and at once when a plane owes more than refresh_slack_pages. "scheme": your serve pays it with
                                         ctx.refresh; the device still forces payment beyond 2 x refresh_slack_pages.
  serve(state, events, ctx) -> state  online and timed; called at every wake. state: your dict, kept between calls ({} at first).
      events = {"qids": int64[a], "queries": float32[a, D],            new queries (closed loop: clients wait for their answer)
                "tags": int64[c], "nbytes": int64[c], "data": uint8[c, W]}  completed HBF reads since the last call, zero padded
The vectors and the index are NOT available in serve: only what layout placed, through the charged ctx.
ctx in serve:
  ctx.submit(pages, offsets, nbytes, tags, batch=False)  issue HBF reads now (int64 tensors; nbytes may be one int); returns at once,
        completions arrive in later events. A read is 64 B to 4 KiB, 64 B aligned, inside one 4 KiB page, and still senses the whole page.
        batch=True is the Batch Read hint: dispatched at once instead of waiting for the base-die timer.
  ctx.hbm(name, rows=None) / ctx.sram(name, rows=None)  rows of a tensor layout placed there (int64 row ids, any shape); waits for the data.
  ctx.finish(qids, ids)   answer queries, ids int64[len(qids), k]; each client sends its next query immediately.
  ctx.wait(count=1, timeout_us=inf)  after this call returns, wake when count events wait, or after timeout_us, or when nothing is in flight.
  ctx.mapping(pages) -> int64[r, 3] channel, die, bank;  ctx.pending() -> int64 tags in flight;  ctx.now() -> simulated microseconds;
  ctx.debt() -> float64[dies]: pages of refresh each die owes (die id = channel * dies_per_channel + die);
  ctx.refresh(dies, counts=1)  start refresh operations (int64 tensors; counts may be one int): one operation waits for the die's reads in
        progress, then holds the die (or, where programs hold only a plane, the plane that owes most) for tR + program_us, and pays a page;
  ctx.spec() adds channels, pages, interleave, refresh, clients (N), state_bytes (cap), hbm/sram {name: shape};
  ctx.effort: a float in [0, 1] swept by the host to trace your recall-throughput curve (map it to your search effort; higher = more recall);
  ctx.knobs["name"], ctx.uniform(*shape) as everywhere.
Time in serve: each wake costs a fixed step and runs as one fused GPU kernel; every torch operation in it is billed from its shapes at
on-chip speed (max of flops / compute and bytes / on-chip bandwidth, no launch of its own), since its operands are the queries' state
and data you already read through ctx (those reads are billed there); views and shapes are free; turning a tensor into a Python value
(.item(), .tolist(), bool(), int()) is a sync and its bytes cross the slow host link.
Batch work across queries: one op over all queries costs about the same as one op for one query.
The state returned by serve is capped (spec state_bytes, at most 64 KiB per client); it cannot hold the data."""
