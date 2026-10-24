"""Pool builder with a fake proposer: no AWS needed."""
import json
import unittest

from eval.build_pool import build_pool, save_pool, load_pool
from eval.tasks import load_tasks


def fake_proposer(source, failure, k):
    """One correct fix (swaps - to +) and one wrong patch, k ignored."""
    return [
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a * b\n",
    ]


FAKE_TASK_SRC = "def add(a, b):\n    return a - b\n"


class TestPool(unittest.TestCase):
    def test_build_labels_candidates(self):
        # ho_power is unused here; use a synthetic Task-shaped stand-in
        from eval.tasks import Task
        task = Task("fake_add", "wrong_operator", "dev",
                    FAKE_TASK_SRC,
                    "from solution import add\nassert add(2, 3) == 5\n",
                    "def add(a, b):\n    return a + b\n")
        pool = build_pool(task, proposer=fake_proposer)
        self.assertEqual(len(pool["candidates"]), 2)
        self.assertTrue(pool["candidates"][0]["passes"])
        self.assertFalse(pool["candidates"][1]["passes"])
        self.assertIn("assert", pool["failure_output"].lower())

    def test_save_load_roundtrip(self):
        from eval.tasks import Task
        import tempfile
        from pathlib import Path
        task = Task("fake_add", "wrong_operator", "dev",
                    FAKE_TASK_SRC,
                    "from solution import add\nassert add(2, 3) == 5\n",
                    "def add(a, b):\n    return a + b\n")
        pool = build_pool(task, proposer=fake_proposer)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_pool(pool, Path(tmp))
            loaded = load_pool("fake_add", Path(tmp))
        self.assertEqual(loaded["task"], "fake_add")
        self.assertEqual(len(loaded["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
