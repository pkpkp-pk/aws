"""Ranker + client tests with a fake server (no GPU, no Colab)."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from decider_ranker.client import Decider
from decider_ranker.ranker import DeciderRanker, build_state, combine


def fake_answers(p_addr, p_side, score, conf):
    return {
        "addresses_failure": {"type": "noul", "noul": p_addr},
        "side_effects": {"type": "noul", "noul": p_side},
        "minimal": {"type": "noul", "noul": 0.5},
        "likely_passes": {"type": "score", "score": score, "confidence": conf},
    }


class FakeDecider:
    """Stands in for Decider.ask; returns canned answers per candidate."""
    def __init__(self, table):
        self.table = table  # candidate-code-substring -> answers
        self.calls = 0
        self.last_latency_ms = 10.0

    def ask(self, state, questions):
        self.calls += 1
        for marker, answers in self.table.items():
            if marker in state:
                return answers
        raise AssertionError("no canned answers for state")


class TestCombine(unittest.TestCase):
    def test_combine_formula(self):
        score, conf = combine(fake_answers(0.9, 0.1, 3.0, 0.8))
        self.assertAlmostEqual(score, 0.9 * 1.0 * 0.9)
        self.assertEqual(conf, 0.8)

    def test_score_normalized(self):
        score, _ = combine(fake_answers(1.0, 0.0, 1.5, 0.5))
        self.assertAlmostEqual(score, 0.5)  # 1.5 / 3 levels


class TestRanker(unittest.TestCase):
    def test_rank_orders_by_combined_score(self):
        table = {
            "good_patch": fake_answers(0.9, 0.05, 2.7, 0.9),
            "bad_patch": fake_answers(0.2, 0.8, 0.5, 0.4),
        }
        ranker = DeciderRanker(decider=FakeDecider(table))
        ranked = ranker.rank("def f(): ...", "assert ...",
                             ["# bad_patch\n", "# good_patch\n"])
        self.assertEqual(ranked[0].code, "# good_patch\n")
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertEqual(ranker.decider.calls, 2)

    def test_state_template_keeps_parts(self):
        s = build_state("SRC", "FAIL", "CAND")
        self.assertIn("SRC", s)
        self.assertIn("FAIL", s)
        self.assertIn("CAND", s)


class TestClientOverHTTP(unittest.TestCase):
    """Client against a local stub server: wire shape and answer parsing."""

    def test_ask_roundtrip(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.update(body)
                resp = {
                    "answers": {"q": {"type": "noul", "noul": 0.7}},
                    "usage": {"input_tokens": 42, "output_tokens": 1},
                    "latency_ms": 12.5,
                }
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                data = json.dumps({"checkpoint": "fake"}).encode()
                self.send_response(200)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            d = Decider(url=f"http://127.0.0.1:{server.server_port}")
            answers = d.ask("some state", {"q": Decider.noul("is it?")})
            self.assertEqual(answers["q"]["noul"], 0.7)
            self.assertEqual(seen["state"], "some state")
            self.assertEqual(seen["questions"]["q"]["type"], "noul")
            self.assertEqual(d.input_tokens, 42)
            self.assertEqual(d.last_latency_ms, 12.5)
            self.assertEqual(d.health()["checkpoint"], "fake")
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
