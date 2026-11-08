"""ctx.lanes: per-lane control flow billed as thread blocks on the GPU."""
import unittest

import torch

from hbf_eval.compute_table import TABLE
from hbf_eval.ctx import SchemeError, ServeCtx
from hbf_eval.device import Device
from hbf_eval.sim import Session

torch.set_num_threads(1)


def session():
    d = Device(channels=2, units=16, tR=10., channel_bw=1024., hbm_bw=1024., hbm_latency=.1, sram_latency=.05)
    plan = {"pages": torch.zeros(4, 4096, dtype=torch.uint8), "hbm": {"t": torch.zeros(64, 8)}, "sram": {}, "interleave": 1}
    s = Session(device=d, plan=plan, queries=torch.zeros(8, 4), k=1, n=10, clients=4, spec={}, knobs={}, effort=0.,
                generator=torch.Generator().manual_seed(0), state_cap=1 << 16)
    return s, ServeCtx(s)


def timed(body, lanes=None):
    s, ctx = session()
    with s.meter:
        if lanes is None:
            body(ctx, 0)
        else:
            for i in ctx.lanes(lanes):
                body(ctx, i)
    return s.cursor, s


class Lanes(unittest.TestCase):
    def test_branches_inside_lanes_are_cheap(self):
        branchy = lambda ctx, i: [bool(torch.ones(4).sum() > j) for j in range(10)]
        outside, _ = timed(lambda ctx, i: [branchy(ctx, j) for j in range(8)])
        inside, s = timed(branchy, lanes=8)
        self.assertGreater(outside, 80 * TABLE["sync_us"] - 1e-9)
        self.assertLess(inside, outside / 10)
        self.assertGreaterEqual(inside, TABLE["lane_phase_us"] + 10 * TABLE["lane_sync_us"] - 1e-9)
        self.assertEqual(s.lane_syncs, 80)

    def test_lanes_never_beat_the_gpu_and_one_lane_never_beats_an_sm(self):
        big = lambda ctx, i: torch.ones(256, 256) @ torch.ones(256, 256)
        one, _ = timed(big)
        many, _ = timed(big, lanes=64)
        self.assertGreaterEqual(many, 64 * one - 1e-9)
        single, _ = timed(big, lanes=1)
        self.assertAlmostEqual(single, TABLE["lane_phase_us"] + TABLE["lane_sms"] * one, places=6)

    def test_hbm_traffic_of_all_lanes_shares_the_link(self):
        read = lambda ctx, i: ctx.hbm("t")
        t, _ = timed(read, lanes=16)
        self.assertGreaterEqual(t, 16 * 64 * 32 / 1024.)

    def test_deferred_finish_happens_at_the_lane_time(self):
        s, ctx = session()
        for q in range(4):
            s.arrive_next()
            s.open.add(q)
        with s.meter:
            for i in ctx.lanes(2):
                for j in range(5 * (i + 1)):
                    bool(torch.ones(1).sum() > 0)
                ctx.finish(torch.tensor([i]), torch.zeros(1, 1, dtype=torch.int64))
        self.assertLess(s.finish_t[0], s.finish_t[1])
        self.assertLessEqual(s.finish_t[1], s.cursor)
        self.assertEqual(s.open, {2, 3})

    def test_misuse_fails(self):
        s, ctx = session()
        with self.assertRaises(SchemeError):
            for i in ctx.lanes(2):
                ctx.lanes(1)
        s, ctx = session()
        with self.assertRaises(SchemeError):
            for i in ctx.lanes(2):
                ctx.debt()
        with self.assertRaises(SchemeError):
            session()[1].lanes(-1)


if __name__ == "__main__":
    unittest.main()
