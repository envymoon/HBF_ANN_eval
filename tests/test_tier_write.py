"""Serve-time writes to HBM and SRAM: ctx.hbm_write / ctx.sram_write overwrite rows layout placed, are billed like reads of
the same rows, check dtype and shape, and leave the plan itself untouched."""
import unittest

import torch

from hbf_eval.ctx import SchemeError, ServeCtx
from hbf_eval.device import Device
from hbf_eval.sim import Session


def session(plan):
    d = Device(channels=2, units=16, tR=10., channel_bw=1024., hbm_bw=1024., hbm_latency=.1, sram_latency=.05)
    return Session(device=d, plan=plan, queries=torch.zeros(4, 8), k=1, n=10, clients=1, spec={}, knobs={}, effort=0.,
                   generator=torch.Generator().manual_seed(0), state_cap=1 << 16)


def plan():
    return {"pages": torch.zeros(1, 4096, dtype=torch.uint8), "hbm": {"cache": torch.zeros(8, 4)},
            "sram": {"tab": torch.zeros(4, 16, dtype=torch.int32)}, "interleave": 1}


class TierWrite(unittest.TestCase):
    def test_write_then_read_and_the_plan_is_untouched(self):
        p = plan()
        s = session(p)
        ctx = ServeCtx(s)
        ctx.hbm_write("cache", torch.tensor([1, 3]), torch.ones(2, 4))
        self.assertEqual(float(ctx.hbm("cache").sum()), 8.)
        self.assertEqual(float(p["hbm"]["cache"].sum()), 0.)
        self.assertEqual(float(session(p).tiers["hbm"]["cache"].sum()), 0.)
        ctx.sram_write("tab", None, torch.full((4, 16), 2, dtype=torch.int32))
        self.assertEqual(int(ctx.sram("tab", torch.tensor([2])).sum()), 32)
        self.assertEqual(s.written["hbm"], 2 * 32)
        self.assertEqual(s.written["sram"], 4 * 64)

    def test_a_write_is_billed_like_a_read(self):
        a, b = session(plan()), session(plan())
        ServeCtx(a).hbm("cache", torch.tensor([0, 1, 2]))
        ServeCtx(b).hbm_write("cache", torch.tensor([0, 1, 2]), torch.ones(3, 4))
        self.assertAlmostEqual(a.cursor, b.cursor)
        self.assertGreater(b.cursor, 0.)

    def test_bad_writes_fail(self):
        ctx = ServeCtx(session(plan()))
        for args, msg in ((("cache", torch.tensor([0]), torch.ones(1, 4, dtype=torch.float64)), "dtype"),
                          (("cache", torch.tensor([0, 1]), torch.ones(3, 4)), "len(rows)"),
                          (("cache", torch.tensor([1, 1]), torch.ones(2, 4)), "distinct"),
                          (("cache", torch.tensor([8]), torch.ones(1, 4)), "out of range"),
                          (("nope", None, torch.ones(8, 4)), "no tensor")):
            with self.assertRaises(SchemeError) as e:
                ctx.hbm_write(*args)
            self.assertIn(msg, str(e.exception))


if __name__ == "__main__":
    unittest.main()
