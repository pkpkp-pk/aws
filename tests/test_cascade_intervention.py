"""Cascade ranker + intervention handler with fakes. No network."""
import unittest

from decider_ranker.intervention import ToolCallRanker
from eval.rankers import cascade_ranker


class FakeRanker:
    """DeciderRanker stand-in: scores by table lookup on code substring."""
    def __init__(self, table):
        self.table = table  # substring -> (score, conf)
        self.total_latency_ms = 40.0

    def rank(self, source, failure, codes):
        from decider_ranker.ranker import RankedCandidate
        out = []
        for i, c in enumerate(codes):
            s, cf = next(v for m, v in self.table.items() if m in c)
            out.append(RankedCandidate(i, c, s, cf, {}))
        out.sort(key=lambda r: r.score, reverse=True)
        return out


class FakeJudge:
    def __init__(self, table):
        self.table = table
        self.calls = 0

    def score(self, source, failure, code):
        self.calls += 1
        return next(v for m, v in self.table.items() if m in code), 0.5


POOL = {
    "source": "def f(x):\n    return x - 1\n",
    "failure_output": "assert f(1) == 2",
    "candidates": [
        {"code": "def f(x):  # A\n    return x + 1\n", "passes": True},
        {"code": "def f(x):  # B\n    return x\n", "passes": False},
        {"code": "def f(x):  # C\n    return x * 2\n", "passes": False},
    ],
}


class TestCascade(unittest.TestCase):
    def test_low_confidence_escalates_to_judge(self):
        # Decider unsure about A (conf 0.3 < 0.6) -> judge overrides B's lead
        ranker = FakeRanker({"# A": (0.30, 0.30), "# B": (0.50, 0.95),
                             "# C": (0.10, 0.95)})
        judge = FakeJudge({"# A": 0.99, "# B": 0.10, "# C": 0.05})
        r = cascade_ranker(POOL, ranker, judge, conf_threshold=0.6, margin=0.0)
        self.assertEqual(r.order[0], 0)          # judge's high score wins
        self.assertEqual(judge.calls, 1)         # only A escalated
        self.assertAlmostEqual(r.latency_s, 0.54, places=2)  # decider + judge

    def test_confident_decider_skips_judge(self):
        ranker = FakeRanker({"# A": (0.90, 0.95), "# B": (0.20, 0.95),
                             "# C": (0.10, 0.95)})
        judge = FakeJudge({"# A": 0.01, "# B": 0.01, "# C": 0.01})
        r = cascade_ranker(POOL, ranker, judge, conf_threshold=0.6, margin=0.05)
        self.assertEqual(r.order[0], 0)
        self.assertEqual(judge.calls, 0)         # no escalation needed

    def test_close_top_scores_escalate_on_margin(self):
        ranker = FakeRanker({"# A": (0.50, 0.95), "# B": (0.48, 0.95),
                             "# C": (0.10, 0.95)})
        judge = FakeJudge({"# A": 0.40, "# B": 0.90, "# C": 0.05})
        r = cascade_ranker(POOL, ranker, judge, conf_threshold=0.5, margin=0.1)
        self.assertEqual(judge.calls, 2)         # A and B within margin
        self.assertEqual(r.order[0], 1)          # judge prefers B


class FakeDecider:
    def __init__(self, choice, conf):
        self._a = {"action": {"choice": choice, "confidence": conf}}

    def ask(self, state, questions):
        return self._a


def describe(call):
    return f"state for {call}", {"run": "execute the action",
                                 "stop": "do not execute"}


class TestIntervention(unittest.TestCase):
    def test_proceed_when_confident(self):
        h = ToolCallRanker(FakeDecider("run", 0.9), describe)
        self.assertEqual(h.before_tool_call("x").kind, "proceed")

    def test_guide_when_low_confidence(self):
        h = ToolCallRanker(FakeDecider("run", 0.4), describe)
        v = h.before_tool_call("x")
        self.assertEqual(v.kind, "guide")
        self.assertIn("confidence", v.feedback)

    def test_guide_when_alternative_chosen(self):
        h = ToolCallRanker(FakeDecider("stop", 0.9), describe)
        v = h.before_tool_call("x")
        self.assertEqual(v.kind, "guide")
        self.assertIn("stop", v.feedback)


if __name__ == "__main__":
    unittest.main()
