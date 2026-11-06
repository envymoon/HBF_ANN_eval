"""The scheme loader: every reference scheme loads under the code policy, entries have the right arity, and a scheme that breaks
the policy or the interface is rejected."""
import unittest

from hbf_eval import schemes
from hbf_eval.policy import Rejected
from hbf_eval.scheme import Scheme, load

GOOD = '''"""tiny"""
KNOBS = {"w": (4, 1, 16, "int")}

def build(vectors, ctx):
    return {"n": torch.tensor([vectors.shape[0] + ctx.knobs["w"]])}

def layout(index, vectors, ctx):
    return {}

def serve(state, events, ctx):
    return state
'''


class SchemeTests(unittest.TestCase):
    def test_reference_schemes_load(self):
        for name in schemes.NAMES:
            s = load(schemes.path(name))
            self.assertEqual(s.live, set(s.functions), name + ": unreachable functions")
            self.assertTrue(set(s.knobs) >= set().union(*(s.refs[f] for f in s.live)))
            s.namespace()

    def test_good_scheme_and_keys(self):
        s = Scheme(GOOD)
        self.assertEqual(s.knob_values(), {"w": 4})
        self.assertEqual(s.knob_values({"w": 99}), {"w": 16})
        keys = s.stage_keys(s.knob_values())
        self.assertEqual(keys["build"]["knobs"], {"w": 4})
        self.assertEqual(keys["layout"]["knobs"], {})
        self.assertEqual(sorted(keys["build"]["units"]), ["build"])

    def test_identity_ignores_comments_and_docstrings(self):
        a = Scheme(GOOD).stage_keys({"w": 4})
        b = Scheme(GOOD.replace("def build(vectors, ctx):", "# a comment\ndef build(vectors, ctx):\n    'doc'")).stage_keys({"w": 4})
        self.assertEqual(a["build"]["units"], b["build"]["units"])

    def test_rejections(self):
        bad = {
            "import": GOOD.replace("return state", "import os\n    return state"),
            "underscore": GOOD.replace("return state", "return ctx._state"),
            "file io": GOOD.replace("return state", "return torch.load('x')"),
            "global rng": GOOD.replace("return state", "return torch.rand(3)"),
            "undeclared knob": GOOD.replace('ctx.knobs["w"]', 'ctx.knobs["z"]'),
            "dynamic knob": GOOD.replace('ctx.knobs["w"]', "ctx.knobs[name]"),
            "arity": GOOD.replace("def serve(state, events, ctx):", "def serve(state, events):"),
            "no entry": GOOD.replace("def layout", "def other"),
            "top-level statement": GOOD + "\nX = 1\n",
            "no knobs table": GOOD.replace("KNOBS = {\"w\": (4, 1, 16, \"int\")}", ""),
        }
        for what, source in bad.items():
            with self.assertRaises(Rejected, msg=what):
                Scheme(source)


if __name__ == "__main__":
    unittest.main()
