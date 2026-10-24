"""Decision-guided patch search loop.

One round: propose K candidates -> rank with Decider -> run tests in ranked
order -> stop at first pass. If nothing passes, feed the best candidates'
failure output back into the proposer and try again (bounded depth, all
candidates kept — that is the backtrack).

The verifier is the ground truth; nothing counts as fixed unless tests pass.
All dependencies are injectable so tests run offline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.proposer import propose
from agent.runner import run_candidate


@dataclass(frozen=True)
class Fix:
    code: str
    round: int            # 1-based proposal round that produced it
    rank: int             # 1-based position in that round's ranking
    score: float          # ranker score it carried


@dataclass
class SearchOutcome:
    fixed: bool
    fix: Fix | None
    test_runs: int
    decider_latency_ms: float
    wall_s: float
    rounds: int
    log: list[dict] = field(default_factory=list)


def search_fix(source: str, tests: str, *, k: int = 8, max_rounds: int = 3,
               ranker=None, proposer=propose, runner=run_candidate,
               on_event=None) -> SearchOutcome:
    """Find a passing replacement for `source`.

    ranker: DeciderRanker-like with .rank(source, failure, codes) and
            .total_latency_ms. None -> proposal order (unranked baseline).
    on_event: optional callback(dict) for live demo output.
    """
    def emit(ev: dict) -> None:
        if on_event:
            on_event(ev)

    t_start = time.monotonic()
    failing = runner(source, tests)
    if failing.passed:
        raise ValueError("source already passes the tests — nothing to fix")

    test_runs = 0
    decider_ms = 0.0
    failure = failing.output
    feedback: list[str] = []  # prior attempts + why they failed
    log: list[dict] = []

    for round_no in range(1, max_rounds + 1):
        candidates = proposer(source, failure, k) if round_no == 1 else \
            proposer(source, failure + _feedback_block(feedback), k)
        if not candidates:
            emit({"t": "no_candidates", "round": round_no})
            continue

        if ranker is not None:
            ranked = ranker.rank(source, failure, [c for c in candidates])
            ordered = [(r.score, r.code) for r in ranked]
            decider_ms += ranker.total_latency_ms
        else:
            ordered = [(0.0, c) for c in candidates]
        emit({"t": "ranked", "round": round_no,
              "scores": [round(s, 3) for s, _ in ordered]})

        for rank, (score, code) in enumerate(ordered, start=1):
            result = runner(code, tests)
            test_runs += 1
            emit({"t": "ran", "round": round_no, "rank": rank,
                  "score": round(score, 3), "passed": result.passed})
            if result.passed:
                return SearchOutcome(
                    fixed=True, fix=Fix(code, round_no, rank, score),
                    test_runs=test_runs, decider_latency_ms=decider_ms,
                    wall_s=time.monotonic() - t_start, rounds=round_no, log=log)
            feedback.append(f"Attempt (score {score:.2f}) failed:\n"
                            f"```python\n{code}```\noutput:\n{result.output[:800]}")
        log.append({"round": round_no, "candidates": len(candidates)})

    return SearchOutcome(fixed=False, fix=None, test_runs=test_runs,
                         decider_latency_ms=decider_ms,
                         wall_s=time.monotonic() - t_start,
                         rounds=max_rounds, log=log)


def _feedback_block(feedback: list[str]) -> str:
    """Trimmed history for the proposer: most recent attempts matter most."""
    keep = feedback[-4:]
    return "\n\nPrevious attempts that still failed:\n" + "\n\n".join(keep)
