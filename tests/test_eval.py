"""Metrics + rankers on synthetic pools."""
import unittest

from eval.metrics import (auroc, bootstrap_ci, brier, runs_to_first_pass,
                          top_k_hit)
from eval.rankers import (cascade_ranker, decider_ranker, heuristic_ranker,
                          llm_judge_ranker, random_ranker)


def make_pool(codes, passes, source="def f(x):\n    return x + 1\n"):
    return {
        "task": "t", "bug_class": "x", "split": "dev",
        "source": source, "failure_output": "assert f(1) == 2",
        "candidates": [{"code": c, "passes": p} for c, p in zip(codes, passes)],
    }


class TestMetrics(unittest.TestCase):
    def test_runs_to_first_pass(self):
        self.assertEqual(runs_to_first_pass([2, 0, 1], [True, False, False]), 2)
        self.assertEqual(runs_to_first_pass([1, 0, 2], [True, False, False]), 2)
        self.assertIsNone(runs_to_first_pass([0, 1], [False, False]))

    def test_top_k(self):
        self.assertTrue(top_k_hit([0, 1, 2], [False, False, True], 3))
        self.assertFalse(top_k_hit([0, 1, 2], [False, False, True], 2))

    def test_auroc(self):
        self.assertEqual(auroc([0.9, 0.1], [True, False]), 1.0)
        self.assertEqual(auroc([0.1, 0.9], [True, False]), 0.0)
        self.assertEqual(auroc([0.5, 0.5], [True, False]), 0.5)
        self.assertIsNone(auroc([0.5], [True]))

    def test_brier(self):
        self.assertEqual(brier([1.0, 0.0], [True, False]), 0.0)
        self.assertEqual(brier([0.0, 1.0], [True, False]), 1.0)

    def test_bootstrap_ci(self):
        lo, hi = bootstrap_ci([0.0, 1.0] * 10, n=500)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


class TestRankers(unittest.TestCase):
    def test_random_deterministic_per_seed(self):
        pool = make_pool(["a", "b", "c"], [True, False, False])
        self.assertEqual(random_ranker(pool, seed=1).order,
                         random_ranker(pool, seed=1).order)

    def test_heuristic_prefers_smallest_change(self):
        src = "def f(x):\n    return x + 1\n"
        near = "def f(x):\n    return x + 2\n"
        far = "def f(x):\n    # long comment rewriting everything\n    y = x * 2\n    return y + 2 - x\n"
        pool = make_pool([far, near], [True, True], source=src)
        r = heuristic_ranker(pool)
        self.assertEqual(r.order[0], 1)  # near first
        self.assertEqual(len(r.scores), 2)

    def test_decider_adapter(self):
        from tests.test_ranker import FakeDecider, fake_answers
        pool = make_pool(["# good\n", "# bad\n"], [True, False])
        table = {
            "good": fake_answers(0.9, 0.1, 2.8, 0.9),
            "bad": fake_answers(0.3, 0.7, 0.5, 0.3),
        }
        from decider_ranker.ranker import DeciderRanker
        r = decider_ranker(pool, DeciderRanker(decider=FakeDecider(table)))
        self.assertEqual(r.order[0], 0)
        self.assertAlmostEqual(r.scores[0], 0.9 * (2.8 / 3) * 0.9)

    def test_llm_judge_adapter(self):
        class FakeJudge:
            def score(self, src, fail, code):
                return (0.9 if "good" in code else 0.1), 0.5

        pool = make_pool(["# bad\n", "# good\n"], [False, True])
        r = llm_judge_ranker(pool, FakeJudge())
        self.assertEqual(r.order[0], 1)
        self.assertAlmostEqual(r.latency_s, 1.0)

    def test_cascade_escalates_low_confidence(self):
        from tests.test_ranker import FakeDecider, fake_answers
        from decider_ranker.ranker import DeciderRanker

        class FakeJudge:
            def __init__(self):
                self.seen = []

            def score(self, src, fail, code):
                self.seen.append(code)
                return (0.95 if "good" in code else 0.05), 0.5

        pool = make_pool(["# good\n", "# bad\n"], [True, False])
        table = {
            # decider is wrong: bad scores higher but with low confidence
            "good": fake_answers(0.4, 0.4, 1.0, 0.3),
            "bad": fake_answers(0.6, 0.2, 1.5, 0.3),
        }
        judge = FakeJudge()
        r = cascade_ranker(pool, DeciderRanker(decider=FakeDecider(table)),
                           judge, conf_threshold=0.6, margin=0.05)
        self.assertEqual(len(judge.seen), 2)  # both low confidence
        self.assertEqual(r.order[0], 0)       # judge flips the order


if __name__ == "__main__":
    unittest.main()
