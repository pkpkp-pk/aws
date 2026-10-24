"""Rank candidate patches with a strands-decider server.

Per-candidate questions (never "which of these K is best" — that needs
lookahead the model does not have). One request per candidate, four
questions per request. Combined score:

    score = P(addresses_failure) * P(likely_passes) * (1 - P(side_effects))

The combination formula is a hyperparameter: tune on the dev split,
report on held-out.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from decider_ranker.client import Decider
from decider_ranker.questions import patch_questions

N_SCORE_LEVELS = 4  # len(questions.SCORE_LEVELS)

STATE_TEMPLATE = """\
Buggy function:
{source}
Failing test output:
{failure}
Candidate patch:
{candidate}"""


@dataclass(frozen=True)
class RankedCandidate:
    index: int           # position in the input list
    code: str
    score: float         # combined ranking score, 0..1
    confidence: float    # confidence on the likely_passes score
    answers: dict = field(compare=False)


def build_state(source: str, failure: str, candidate: str) -> str:
    return STATE_TEMPLATE.format(source=source.strip(), failure=failure.strip(),
                                 candidate=candidate.strip())


def combine(answers: dict) -> tuple[float, float]:
    """answers -> (combined score, confidence). Isolated for tuning."""
    p_addr = answers["addresses_failure"]["noul"]
    p_side = answers["side_effects"]["noul"]
    s = answers["likely_passes"]
    p_pass = s["score"] / (N_SCORE_LEVELS - 1)
    conf = s.get("confidence", max(p_pass, 1 - p_pass))
    return p_addr * p_pass * (1 - p_side), conf


class DeciderRanker:
    """Ranks candidates; the test suite stays the ground truth."""

    def __init__(self, decider: Decider | None = None) -> None:
        self.decider = decider or Decider()
        self.total_latency_ms = 0.0

    def score_one(self, state: str) -> tuple[dict, float, float]:
        answers = self.decider.ask(state, patch_questions())
        score, conf = combine(answers)
        return answers, score, conf

    def rank(self, source: str, failure: str,
             candidates: list[str]) -> list[RankedCandidate]:
        """Return candidates sorted by combined score, best first."""
        self.total_latency_ms = 0.0
        ranked = []
        for i, code in enumerate(candidates):
            t0 = time.monotonic()
            answers, score, conf = self.score_one(build_state(source, failure, code))
            ranked.append(RankedCandidate(
                index=i, code=code, score=score, confidence=conf, answers=answers,
            ))
            self.total_latency_ms += (time.monotonic() - t0) * 1000
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked
