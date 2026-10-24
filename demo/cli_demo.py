"""Live demo for the video: propose -> Decider ranks -> tests run in order.

    python demo/cli_demo.py --task dev_clamp
    python demo/cli_demo.py --task dev_clamp --no-rank   # unranked baseline

Shows each candidate's Decider score and confidence as tests run, and the
final search cost (test runs, Decider latency) side by side with what an
unranked search paid.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.search import search_fix
from eval.tasks import load_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-rank", action="store_true")
    args = ap.parse_args()

    task = next(t for t in load_tasks() if t.name == args.task)
    print(f"task: {task.name} ({task.bug_class})\n")
    print("buggy function:")
    print(task.source)

    ranker = None
    if not args.no_rank:
        from decider_ranker.ranker import DeciderRanker
        ranker = DeciderRanker()
        print(f"decider server: {ranker.decider.url}\n")

    def show(ev: dict) -> None:
        if ev["t"] == "ranked":
            print(f"round {ev['round']}: scores {ev['scores']}")
        elif ev["t"] == "ran":
            mark = "PASS" if ev["passed"] else "fail"
            print(f"  rank {ev['rank']} (score {ev['score']:.3f}) -> {mark}")

    out = search_fix(task.source, task.tests, k=args.k, ranker=ranker,
                     on_event=show)
    print()
    if out.fixed:
        print(f"FIXED in round {out.fix.round}, rank {out.fix.rank} "
              f"(score {out.fix.score:.3f})")
        print(out.fix.code)
    else:
        print("no fix found")
    print(f"test runs: {out.test_runs} | decider time: "
          f"{out.decider_latency_ms:.0f} ms | wall: {out.wall_s:.1f}s")


if __name__ == "__main__":
    main()
