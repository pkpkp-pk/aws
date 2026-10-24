"""Evaluate all rankers on cached pools; write results/metrics.json.

Everything is computed from results/pools/*.json — the same candidates
and labels for every ranker. Random is averaged over many shuffles.

Usage:
    python -m eval.run_rankers                      # cheap rankers only
    python -m eval.run_rankers --with decider       # + Decider (needs server)
    python -m eval.run_rankers --with decider,judge_small,judge_large,cascade
    python -m eval.run_rankers --split heldout
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval.build_pool import POOLS_DIR, load_pool
from eval.metrics import (auroc, bootstrap_ci, brier, runs_to_first_pass,
                          top_k_hit)
from eval.rankers import (cascade_ranker, decider_ranker, heuristic_ranker,
                          llm_judge_ranker, random_ranker)

RANDOM_SHUFFLES = 200
RESULTS_DIR = POOLS_DIR.parent
OUT_PATH = RESULTS_DIR / "metrics.json"


def labels_of(pool: dict) -> list[bool]:
    return [c["passes"] for c in pool["candidates"]]


def eval_ranking(ranking, labels) -> dict:
    rfp = runs_to_first_pass(ranking.order, labels)
    return {
        "runs_to_first_pass": rfp,
        "solvable": rfp is not None,
        "top1": top_k_hit(ranking.order, labels, 1),
        "top3": top_k_hit(ranking.order, labels, 3),
        "latency_s": ranking.latency_s,
        "scores": ranking.scores,
    }


def aggregate(rows: list[dict], all_scores: list[float],
              all_labels: list[bool]) -> dict:
    solvable = [r for r in rows if r["solvable"]]
    rfps = [r["runs_to_first_pass"] for r in solvable]
    lo, hi = bootstrap_ci([float(x) for x in rfps]) if rfps else (float("nan"),) * 2
    return {
        "n_tasks": len(rows),
        "n_solvable": len(solvable),
        "mean_runs_to_first_pass": statistics.mean(rfps) if rfps else None,
        "median_runs_to_first_pass": statistics.median(rfps) if rfps else None,
        "rfp_ci95": [lo, hi],
        "top1_rate": statistics.mean(r["top1"] for r in solvable) if solvable else None,
        "top3_rate": statistics.mean(r["top3"] for r in solvable) if solvable else None,
        "auroc": auroc(all_scores, all_labels) if all_scores else None,
        "brier": brier(all_scores, all_labels) if all_scores else None,
        "mean_latency_s": statistics.mean(r["latency_s"] for r in rows),
    }


def evaluate(pools: list[dict], rankers: dict[str, callable],
             random_shuffles: int = RANDOM_SHUFFLES) -> dict:
    """rankers: name -> fn(pool) -> Ranking. Returns per-ranker aggregates."""
    per_ranker: dict[str, dict] = {name: {"rows": [], "scores": [], "labels": []}
                                   for name in rankers}
    for pool in pools:
        labels = labels_of(pool)
        for name, fn in rankers.items():
            if name == "random":
                # average over shuffles; scores undefined for random
                rs = [eval_ranking(random_ranker(pool, seed=s), labels)
                      for s in range(random_shuffles)]
                n_sol = sum(r["solvable"] for r in rs)
                row = {
                    "runs_to_first_pass": (
                        statistics.mean(r["runs_to_first_pass"] for r in rs if r["solvable"])
                        if n_sol else None),
                    "solvable": any(r["solvable"] for r in rs),
                    "top1": statistics.mean(r["top1"] for r in rs),
                    "top3": statistics.mean(r["top3"] for r in rs),
                    "latency_s": 0.0,
                    "scores": None,
                }
            else:
                row = eval_ranking(fn(pool), labels)
            per_ranker[name]["rows"].append(row)
            if row["scores"] is not None:
                per_ranker[name]["scores"].extend(row["scores"])
                per_ranker[name]["labels"].extend(labels)

    return {name: aggregate(d["rows"], d["scores"], d["labels"])
            for name, d in per_ranker.items()}


def make_live_rankers(names: list[str]) -> dict[str, callable]:
    """Construct rankers that need a server or Bedrock, by name."""
    out: dict[str, callable] = {}
    if "decider" in names or "cascade" in names:
        from decider_ranker.ranker import DeciderRanker
    if "judge_small" in names or "cascade" in names:
        from agent.judge import BedrockJudge, SMALL_MODEL
    if "judge_large" in names or "cascade" in names:
        from agent.judge import LARGE_MODEL

    if "decider" in names:
        out["decider"] = lambda pool: decider_ranker(pool, DeciderRanker())
    if "judge_small" in names:
        judge = BedrockJudge(SMALL_MODEL)
        out["judge_small"] = lambda pool: llm_judge_ranker(pool, judge, "haiku")
    if "judge_large" in names:
        judge = BedrockJudge(LARGE_MODEL)
        out["judge_large"] = lambda pool: llm_judge_ranker(pool, judge, "sonnet")
    if "cascade" in names:
        judge = BedrockJudge(LARGE_MODEL)
        out["cascade"] = lambda pool: cascade_ranker(pool, DeciderRanker(), judge)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout"], default=None)
    ap.add_argument("--with", dest="live", default="",
                    help="comma list of live rankers: decider,judge_small,judge_large,cascade")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    pools = []
    for path in sorted(POOLS_DIR.glob("*.json")):
        pool = json.loads(path.read_text())
        if args.split and pool["split"] != args.split:
            continue
        pools.append(pool)
    if not pools:
        raise SystemExit(f"no pools in {POOLS_DIR} — run eval.build_pool first")

    rankers: dict[str, callable] = {"random": random_ranker,
                                    "heuristic": heuristic_ranker}
    rankers.update(make_live_rankers([n for n in args.live.split(",") if n]))

    results = evaluate(pools, rankers)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
