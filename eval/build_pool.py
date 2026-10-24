"""Build cached candidate pools: propose K patches per task, label each
against the real tests, cache to results/pools/<task>.json.

All rankers evaluate offline against these identical pools — Bedrock
cost is paid once per task, never per ranker.

Usage:
    python -m eval.build_pool                 # all tasks, k=8
    python -m eval.build_pool --split dev     # dev split only
    python -m eval.build_pool --task dev_clamp --force
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent.proposer import propose
from agent.runner import run_candidate
from eval.tasks import load_tasks

POOLS_DIR = Path(__file__).resolve().parent.parent / "results" / "pools"


def build_pool(task, k: int = 8, proposer=propose) -> dict:
    """Generate and label one pool. Pure given a proposer function."""
    failing = run_candidate(task.source, task.tests)
    assert not failing.passed, f"{task.name}: buggy source passes (bad task)"
    t0 = time.monotonic()
    candidates = proposer(task.source, failing.output, k)
    proposal_s = time.monotonic() - t0
    pool = {
        "task": task.name,
        "bug_class": task.bug_class,
        "split": task.split,
        "source": task.source,
        "failure_output": failing.output,
        "proposal_s": proposal_s,
        "candidates": [],
    }
    for code in candidates:
        r = run_candidate(code, task.tests)
        pool["candidates"].append({
            "code": code,
            "passes": r.passed,
            "run_s": r.duration_s,
        })
    return pool


def save_pool(pool: dict, out_dir: Path = POOLS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pool['task']}.json"
    path.write_text(json.dumps(pool, indent=2))
    return path


def load_pool(name: str, pools_dir: Path = POOLS_DIR) -> dict:
    return json.loads((pools_dir / f"{name}.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout"], default=None)
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rebuild cached pools")
    args = ap.parse_args()

    tasks = load_tasks(args.split)
    if args.task:
        tasks = [t for t in tasks if t.name in args.task]
    for task in tasks:
        out = POOLS_DIR / f"{task.name}.json"
        if out.exists() and not args.force:
            print(f"skip {task.name} (cached)")
            continue
        print(f"building {task.name} (k={args.k}) ...", flush=True)
        pool = build_pool(task, k=args.k)
        n_pass = sum(c["passes"] for c in pool["candidates"])
        save_pool(pool)
        print(f"  {len(pool['candidates'])} candidates, {n_pass} pass, "
              f"propose took {pool['proposal_s']:.1f}s")


if __name__ == "__main__":
    main()
