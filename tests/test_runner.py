"""Runner tests: pass, fail, timeout, and resource-limit behavior."""
import unittest

from agent.runner import run_candidate

GOOD = "def add(a, b):\n    return a + b\n"
BAD = "def add(a, b):\n    return a - b\n"
TESTS = (
    "from solution import add\n"
    "assert add(2, 3) == 5\n"
    "assert add(-1, 1) == 0\n"
)


class TestRunner(unittest.TestCase):
    def test_passing_candidate(self):
        r = run_candidate(GOOD, TESTS)
        self.assertTrue(r.passed, r.output)

    def test_failing_candidate(self):
        r = run_candidate(BAD, TESTS)
        self.assertFalse(r.passed)
        self.assertIn("assert", r.output.lower())

    def test_syntax_error(self):
        r = run_candidate("def add(:\n", TESTS)
        self.assertFalse(r.passed)
        self.assertIn("SyntaxError", r.output)

    def test_timeout(self):
        loop = "def add(a, b):\n    while True:\n        pass\n"
        tests = "from solution import add\nadd(1, 2)\n"
        r = run_candidate(loop, tests, timeout_s=2)
        self.assertFalse(r.passed)
        self.assertTrue(r.timed_out)

    def test_memory_limit(self):
        hog = "def add(a, b):\n    x = [0] * (10**9)\n    return a + b\n"
        tests = "from solution import add\nassert add(1, 2) == 3\n"
        r = run_candidate(hog, tests)
        self.assertFalse(r.passed)

    def test_output_trimmed(self):
        noisy = "def add(a, b):\n    print('x' * 100000)\n    return a - b\n"
        r = run_candidate(noisy, TESTS)
        self.assertLessEqual(len(r.output), 2100)


if __name__ == "__main__":
    unittest.main()
