# Decision-Guided Patch Search

An agent that fixes buggy Python functions by **searching** over candidate fixes.
Amazon Bedrock *proposes* K candidate patches. **Strands Decider 2B** *ranks* them
in ~100 ms per candidate by answering typed questions about each one. The agent
runs the real tests in ranked order and stops at the first pass.

Decider is a **fast prior**. The test suite is the ground truth. When Decider's
confidence is low, the search escalates those candidates to a larger Bedrock LLM
judge — a cascade that aims to sit on the accuracy/cost Pareto frontier.

**Research question:** do Decider's scores predict which patches pass the tests
well enough to beat random ordering, a cheap heuristic, and an LLM judge — once
you account for cost and latency?

## Why Decider here

- **Runtime-defined options.** Candidate patches did not exist when any model was
  trained. Decider's pointer head scores options described in the request.
- **Cheap, confidence-bearing decisions.** Ranking K=8 patches costs ~1 s of local
  compute, not 8 LLM completions. Every answer carries a calibrated confidence,
  which drives the escalation cascade.
- **Shared-prefix inference.** One request per candidate asks four typed
  questions; the state (buggy function + failing test + patch) is encoded once.

## The reusable decision pattern

`decider_ranker/` — per-candidate typed questions over a shared state:

| Question | Type | Asks |
|---|---|---|
| `addresses_failure` | yes/no | does the patch touch the failing code path? |
| `side_effects` | yes/no | does it change unrelated behavior? |
| `minimal` | yes/no | is it a focused change? |
| `likely_passes` | score 0–3 rubric | how likely is the failing test to pass? |

Combined score (hyperparameter, tuned on dev split only):

```
score = P(addresses_failure) * P(likely_passes) * (1 - P(side_effects))
```

Never ask "which of these K is best?" — that needs lookahead a 2B single-pass
model does not have. Per-candidate questions also kill option-position bias.

## Architecture

```
failing test + buggy function
          │
          ▼
  ┌─────────────────┐   K candidate patches
  │ Bedrock proposer│ ──────────────┐
  └─────────────────┘               ▼
                          ┌───────────────────┐
                          │ Decider ranker    │  per-candidate typed questions
                          │ (local, ~100 ms)  │  → score + confidence
                          └─────────┬─────────┘
                                    │ ranked list
        low confidence  ┌───────────┴────────────┐  high confidence
                        ▼                        ▼
              ┌──────────────────┐     ┌────────────────┐
              │ Bedrock LLM judge│────▶│ Test runner    │  ranked order,
              │ (escalation)     │     │ (sandboxed)    │  stop at first pass
              └──────────────────┘     └───────┬────────┘
                                               │ all failed?
                                               ▼
                              feed failure output back to proposer
                              (depth 2–3, keep alternatives, backtrack)
```

## Repo layout

```
decider_ranker/      # the reusable pattern: client, questions, ranker, intervention
agent/               # proposer (Bedrock), judge (Bedrock), runner (sandbox), search loop
eval/                # tasks, build_pool (cached labeled pools), run_rankers, metrics, plots
demo/                # cli_demo.py — live ranking view
tests/               # offline tests (fake Decider server, stub Bedrock)
results/             # cached pools, metrics.json, figures
```

## Setup

```bash
pip install -e '.[dev]'
pytest                    # offline: fake Decider server, no AWS calls
```

The Decider server runs separately (GPU recommended):

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
export STRANDS_DECIDER_URL=http://127.0.0.1:8099   # or a tunnel URL
```

Bedrock: configure AWS credentials and request model access. Model IDs are
env-configurable (`PROPOSER_MODEL`, `JUDGE_SMALL_MODEL`, `JUDGE_LARGE_MODEL`).

## The experiment

```bash
python -m eval.build_pool  --split dev        # propose + label K=8 candidates (Bedrock, once)
python -m eval.run_rankers --with decider,judge_small,judge_large,cascade
python -m eval.plots                          # Pareto, calibration, bars -> results/
```

All rankers evaluate on the **same cached candidate pools** — Bedrock cost is
paid once per task, never per ranker. Rankers never see labels; the test suite
is the only ground truth. Dev split tunes the formula and thresholds; held-out
split reports results with bootstrap CIs.

Rankers compared: random · smallest-diff heuristic · small LLM judge ·
large LLM judge · Decider · cascade (Decider + escalation).

Metrics: test runs to first pass (primary), top-1/top-3 hit rate, AUROC,
Brier/calibration, latency, cost.

## Hypotheses (pre-registered in the plan doc)

H1 Decider ≫ random · H2 Decider > heuristic, modestly · H3 large judge more
accurate but slow/costly · H4 cascade on the Pareto frontier · H5 Decider's
confidence is informative about itself, not a patch-quality proxy — examined
honestly either way.

## Demo

```bash
python demo/cli_demo.py --task dev_clamp   # live: propose → rank → run
```

## License

MIT (see LICENSE).
