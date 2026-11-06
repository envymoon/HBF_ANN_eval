"""The HBF checks of the device model (hbf_eval/hbfcheck.py) must all hold: the profiles are
self-consistent, the SSD assumptions fail on HBF the way they should, and the model reproduces HBFlex, HAVEN and OCP §13.1."""
import unittest

from hbf_eval import fastdevice, hbfcheck


@unittest.skipIf(fastdevice.load() is None, "no C compiler (the checks run on the C model for speed)")
class HBFChecks(unittest.TestCase):
    def test_every_check_holds(self):
        rows = hbfcheck.run_all()
        self.assertGreaterEqual(len(rows), 24)
        failed = [(r["name"], r["measured"], r["expected"]) for r in rows if not r["holds"]]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
