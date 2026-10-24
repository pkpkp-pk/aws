"""Dataset invariants: tests fail on buggy source, pass on reference."""
import unittest

from agent.runner import run_candidate
from eval.tasks import load_tasks


class TestTasks(unittest.TestCase):
    def test_buggy_fails_and_reference_passes(self):
        failures = []
        for task in load_tasks():
            buggy = run_candidate(task.source, task.tests)
            if buggy.passed:
                failures.append(f"{task.name}: buggy source PASSES tests")
            ref = run_candidate(task.reference, task.tests)
            if not ref.passed:
                failures.append(f"{task.name}: reference FAILS: {ref.output[-200:]}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_splits_nonempty(self):
        self.assertGreaterEqual(len(load_tasks("dev")), 8)
        self.assertGreaterEqual(len(load_tasks("heldout")), 15)

    def test_names_unique(self):
        names = [t.name for t in load_tasks()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
