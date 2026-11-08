"""bench run: a scheme process over its resident-memory budget is stopped and recorded as "memory over budget"."""
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest

from hbf_eval import bench


class ResidentBudget(unittest.TestCase):
    def test_resident_counts_a_live_process(self):
        p = subprocess.Popen([sys.executable, "-c", "x = bytearray(200 << 20)\nfor i in range(0, len(x), 4096): x[i] = 1\n"
                                                    "import time; time.sleep(30)"])
        try:
            deadline = time.time() + 20
            while bench.resident(p.pid) < 150 << 20 and time.time() < deadline:
                time.sleep(.2)
            self.assertGreater(bench.resident(p.pid), 150 << 20)
        finally:
            p.kill()
            p.wait()

    def test_over_budget_is_recorded(self):
        hog = "x = bytearray(400 << 20)\nfor i in range(0, len(x), 4096): x[i] = 1\nimport time; time.sleep(60)\n"
        real = subprocess.Popen
        with tempfile.TemporaryDirectory() as out:
            a = types.SimpleNamespace(out=out, schemes="baseline", set=None, scale="100k", data="clustered", wide=False,
                                      workers=1, cache=out, data_root=None, mem_gb=None, rss_gb=.2)
            subprocess.Popen = lambda cmd, **kw: real([sys.executable, "-c", hog], **kw)
            try:
                started = time.time()
                bench.run(a)
            finally:
                subprocess.Popen = real
            rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl"))]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["failure"].startswith("memory over budget"), rows[0])
        self.assertLess(time.time() - started, 30)


if __name__ == "__main__":
    unittest.main()
