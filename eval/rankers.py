"""Ranker implementations over cached pools.

Protocol: rank(pool) -> Ranking(order, scores, latency_s, note)
- order: candidate indices, best first
- scores: per-candidate 0..1 score aligned with the INPUT index order
  (None for rankers that produce only an ordering)
- latency_s: wall time spent deciding (Decider HTTP / Bedrock calls),
  not test runs

Rankers never see labels — pools carry them for scoring only.
"""
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ranking:
    order: list[int]
    scores: list[float] | None
    latency_s: float = 0.0
    note: str = ""


def random_ranker(pool: dict, seed: int = 0) -> Ranking:
    idx = list(range(len(pool["candidates"])))
    random.Random(seed).shuffle(idx)
    return Ranking(order=idx, scores=None, note=f"seed={seed}")


def heuristic_ranker(pool: dict) -> Ranking:
    """Smallest-change-first: prefer candidates closest to the original
    source by difflib ratio, tie-break on shorter code. Compiles-check
    implicit: all pool candidates were AST-validated at build time."""
    src = pool["source"]
    cands = pool["candidates"]

    def key(i: int) -> tuple[float, int]:
        code = cands[i]["code"]
        return (-difflib.SequenceMatcher(None, src, code).ratio(), len(code))

    order = sorted(range(len(cands)), key=key)
    scores = [0.0] * len(cands)
    for rank, i in enumerate(order):
        scores[i] = 1.0 - rank / max(1, len(cands) - 1)
    return Ranking(order=order, scores=scores)


def decider_ranker(pool: dict, ranker) -> Ranking:
    """Adapter: DeciderRanker over a cached pool. `ranker` injected so
    tests can use a fake; production passes DeciderRanker()."""
    codes = [c["code"] for c in pool["candidates"]]
    ranked = ranker.rank(pool["source"], pool["failure_output"], codes)
    scores = [0.0] * len(codes)
    for r in ranked:
        scores[r.index] = r.score
    return Ranking(
        order=[r.index for r in ranked],
        scores=scores,
        latency_s=ranker.total_latency_ms / 1000,
    )


def llm_judge_ranker(pool: dict, judge, note: str = "") -> Ranking:
    """Adapter: an LLM judge with .score(source, failure, code) -> (0..1,
    latency_s). Per-candidate scoring (no position bias)."""
    src, failure = pool["source"], pool["failure_output"]
    scores, latency = [], 0.0
    for c in pool["candidates"]:
        s, dt = judge.score(src, failure, c["code"])
        scores.append(s)
        latency += dt
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return Ranking(order=order, scores=scores, latency_s=latency, note=note)


def cascade_ranker(pool: dict, ranker, judge, conf_threshold: float = 0.6,
                   margin: float = 0.1) -> Ranking:
    """Decider first; candidates whose likely_passes confidence is below
    threshold, or within `margin` of the top score, get re-scored by the
    large LLM judge. Final order: judge score overrides where called."""
    codes = [c["code"] for c in pool["candidates"]]
    ranked = ranker.rank(pool["source"], pool["failure_output"], codes)
    decider_latency = ranker.total_latency_ms / 1000
    scores = [0.0] * len(codes)
    confs = [0.0] * len(codes)
    for r in ranked:
        scores[r.index] = r.score
        confs[r.index] = r.confidence
    top = ranked[0].score if ranked else 0.0
    # Candidates close to the top are ambiguous; if any non-top candidate is
    # within margin, the top is re-judged too so compared scores share a scale.
    near_top = any(0 < (top - scores[i]) < margin for i in range(len(codes)))
    judge_latency = 0.0
    for r in ranked:
        i = r.index
        if (confs[i] < conf_threshold
                or 0 < (top - scores[i]) < margin
                or (near_top and scores[i] == top)):
            s, dt = judge.score(pool["source"], pool["failure_output"], codes[i])
            scores[i] = s
            judge_latency += dt
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return Ranking(order=order, scores=scores,
                   latency_s=decider_latency + judge_latency,
                   note=f"escalated with conf<{conf_threshold} or margin<{margin}")
