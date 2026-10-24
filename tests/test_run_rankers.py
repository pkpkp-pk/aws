"""End-to-end: evaluate() over synthetic pools, cheap rankers only."""
import unittest

from eval.rankers import heuristic_ranker, random_ranker
from eval.run_rankers import evaluate


def pool(name, codes, passes, source="def f(x):\n    return x + 1\n"):
    return {"task": name, "bug_class": "x", "split": "heldout",
            "source": source, "failure_output": "assert f(1) == 2",
            "candidates": [{"code": c, "passes": p}
                           for c, p in zip(codes, passes)]}


class TestEvaluate(unittest.TestCase):
    def test_aggregate_shape(self):
        pools = [
            pool("t1", ["def f(x):\n    return x + 2\n",
                        "def g(y):\n    return y\n"], [True, False]),
            pool("t2", ["x = 1\n", "def f(x):\n    return x - 1\n"], [False, True]),
            pool("t3", ["pass\n", "pass\n"], [False, False]),  # unsolvable
        ]
        results = evaluate(pools, {"random": random_ranker,
                                   "heuristic": heuristic_ranker},
                           random_shuffles=20)
        for name in ("random", "heuristic"):
            agg = results[name]
            self.assertEqual(agg["n_tasks"], 3)
            self.assertEqual(agg["n_solvable"], 2)
            self.assertIsNotNone(agg["mean_runs_to_first_pass"])
            self.assertGreaterEqual(agg["top1_rate"], 0.0)
            self.assertLessEqual(agg["top1_rate"], 1.0)
        # heuristic emits scores -> AUROC defined; random does not
        self.assertIsNotNone(results["heuristic"]["auroc"])
        self.assertIsNone(results["random"]["auroc"])


if __name__ == "__main__":
    unittest.main()
