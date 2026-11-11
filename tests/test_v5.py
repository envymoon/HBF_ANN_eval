"""Evaluator v5: tail-recall floor, bisection at each recall level, graded floor, cost report and read rounds."""
import time
import unittest

import torch

from hbf_eval import runner, scheme, schemes
from hbf_eval.scoring import SPEEDUP_FLOOR, speedup
from hbf_eval.tasks import task

SCALE = {"cpp": 1, "pool": 64, "k": 10, "efforts": 3, "total_min": 32, "total_max": 64, "total_per_client": 2,
         "mode": "speedup", "n": 2000, "levels": [.9, .95]}


def run(scale, name="baseline"):
    torch.set_num_threads(1)
    sch = scheme.load(str(schemes.path(name)))
    t = task("low_load_4", "low_load", "tiny", scale["n"], ("count", 4))
    t["spec"]["split"] = "main"
    return runner.evaluate(sch, sch.knob_values({}), t, 1, {"scale": scale, "data_root": None, "cache": None},
                           time.monotonic() + 600)


class Scoring(unittest.TestCase):
    def test_graded_floor(self):
        pts = [(.5, 100.), (.8, 50.)]
        self.assertEqual(speedup(pts, 10.), SPEEDUP_FLOOR)
        g = speedup(pts, 10., graded=True)
        self.assertAlmostEqual(g, SPEEDUP_FLOOR * (1 + .1 / .9))
        self.assertLess(speedup([(.2, 100.)], 10., graded=True), g)  # further from 0.9: lower
        self.assertEqual(speedup([(.95, 40.)], 10., graded=True), 2.)  # reached: the usual log2 speedup

    def test_tail_ok(self):
        self.assertTrue(runner.tail_ok({"recall_worst5": .1}, None))
        self.assertFalse(runner.tail_ok({"recall_worst5": .1}, .3))
        self.assertTrue(runner.tail_ok({"recall_worst5": .3}, .3))


class Evaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = run(SCALE)
        cls.v5 = run(dict(SCALE, tail_floor=.3, refine=2, graded_floor=True))

    def test_records(self):
        for rec in (self.plain, self.v5):
            self.assertIsNone(rec["failure"])
            for key in ("build_seconds", "layout_seconds", "index_bytes", "hbf_bytes", "hbm_bytes", "hbm_share_of_budget"):
                self.assertIn(key, rec["cost"])
            for key in ("p50_over_hbf_floor", "p50_over_hbm_floor", "read_rounds_per_query"):
                self.assertIn(key, rec["diag"])
            self.assertGreaterEqual(rec["diag"]["read_rounds_per_query"], 1.)
        self.assertEqual(self.v5["tail_floor"], .3)
        self.assertNotIn("tail_floor", self.plain)

    def test_bisection_adds_points_between_efforts(self):
        efforts = sorted(c[0] for c in self.v5["curve"])
        self.assertGreaterEqual(len(efforts), len(self.plain["curve"]))
        grid = {0., .5, 1.}
        self.assertTrue(set(e for e in efforts if e not in grid) or len(efforts) == len(self.plain["curve"]))
        self.assertEqual(efforts, sorted(set(efforts)))  # no effort is run twice

    def test_deterministic_part(self):
        again = run(dict(SCALE, tail_floor=.3, refine=2, graded_floor=True))
        self.assertEqual(runner.deterministic_part(again), runner.deterministic_part(self.v5))


if __name__ == "__main__":
    unittest.main()


class InFlightCap(unittest.TestCase):
    """An effort whose reads overflow the in-flight cap ends the sweep; the points below it stand."""
    def test_cap_keeps_lower_points(self):
        real = runner.Session

        def capped_session(**kw):
            s = real(**kw)
            if kw["effort"] > .4:
                def boom(serve):
                    raise runner.SchemeError("submit: " + runner.CAP_ERROR)
                s.run = boom
            return s
        runner.Session = capped_session
        try:
            rec = run(dict(SCALE, levels=[.9, .95, .9999], refine=0))
        finally:
            runner.Session = real
        self.assertIsNone(rec["failure"])
        self.assertEqual(rec["capped_at"], .5)
        self.assertTrue(all(c[0] <= .4 for c in rec["curve"]))
        self.assertIn("speedups", rec)
