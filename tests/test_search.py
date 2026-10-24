"""Search loop tests with stub proposer/runner/ranker. Offline."""
import unittest

from agent.search import search_fix

BUGGY = "def f(x):\n    return x - 1\n"
FIX = "def f(x):\n    return x + 1\n"
TESTS = "from solution import f\nassert f(1) == 2\n"


def stub_runner(passes_on=(FIX,)):
    def run(code, tests):
        from agent.runner import RunResult
        return RunResult(passed=code in passes_on, output="", duration_s=0.01)
    return run


class StubRanker:
    def __init__(self, preferred):
        self.preferred = preferred
        self.total_latency_ms = 5.0

    def rank(self, source, failure, codes):
        from decider_ranker.ranker import RankedCandidate
        order = sorted(codes, key=lambda c: c != self.preferred)
        return [RankedCandidate(i, c, 1.0 if c == self.preferred else 0.1,
                                0.9, {}) for i, c in enumerate(order)]


class TestSearch(unittest.TestCase):
    def test_stops_at_first_pass_in_rank_order(self):
        wrong = "def f(x):\n    return x\n"
        calls = []
        proposer = lambda s, f, k: [wrong, FIX][:k]
        out = search_fix(BUGGY, TESTS, k=2, ranker=StubRanker(FIX),
                         proposer=proposer, runner=stub_runner())
        self.assertTrue(out.fixed)
        self.assertEqual(out.fix.rank, 1)          # ranker put fix first
        self.assertEqual(out.test_runs, 1)         # only one test run needed

    def test_unranked_runs_until_pass(self):
        wrong = "def f(x):\n    return x\n"
        proposer = lambda s, f, k: [wrong, FIX][:k]
        out = search_fix(BUGGY, TESTS, k=2, ranker=None,
                         proposer=proposer, runner=stub_runner())
        self.assertTrue(out.fixed)
        self.assertEqual(out.fix.rank, 2)
        self.assertEqual(out.test_runs, 2)

    def test_feedback_round_when_all_fail(self):
        seen_failures = []

        def proposer(source, failure, k):
            seen_failures.append(failure)
            return [FIX] if len(seen_failures) > 1 else ["def f(x):\n    return x\n"]

        out = search_fix(BUGGY, TESTS, k=1, max_rounds=2, ranker=None,
                         proposer=proposer, runner=stub_runner())
        self.assertTrue(out.fixed)
        self.assertEqual(out.fix.round, 2)
        self.assertIn("Previous attempts", seen_failures[1])

    def test_gives_up_after_max_rounds(self):
        proposer = lambda s, f, k: ["def f(x):\n    return x\n"] * k
        out = search_fix(BUGGY, TESTS, k=2, max_rounds=2, ranker=None,
                         proposer=proposer, runner=stub_runner())
        self.assertFalse(out.fixed)
        self.assertIsNone(out.fix)

    def test_passing_source_rejected(self):
        with self.assertRaises(ValueError):
            search_fix(FIX, TESTS, ranker=None,
                       proposer=lambda s, f, k: [], runner=stub_runner())


if __name__ == "__main__":
    unittest.main()
