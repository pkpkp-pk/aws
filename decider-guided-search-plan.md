# Decision-Guided Patch Search — Hackathon Plan (Whats and Hows)

Hackathon: **AWS Agent Experience Hackathon: Strands Decider**
Deadline: **Oct 23, 2026, 11:59 PM PDT** (= **Oct 24, ~12:29 PM IST**). Aim to submit by Oct 23 evening IST at the latest.
Prize: 3 winners, each $5,000 AWS credits + swag. Top 100 advance from Gate 1 (AI + human scoring) to Gate 2 (human panel).

---

## 1. The one-paragraph pitch

An agent that fixes buggy Python functions by **searching** over candidate patches. Amazon Bedrock *proposes* K candidate patches. **Strands Decider 2B** *ranks* them in ~100 ms by answering typed questions about each patch. The agent then runs the real tests in ranked order and stops at the first pass. The research question: **do Decider's scores predict which patches pass the tests well enough to beat random ordering, a cheap heuristic, and an LLM judge once you account for cost and latency?**

Decider is a **fast prior**. The test suite is the ground truth. A low-confidence Decider answer escalates to a bigger LLM judge or simply runs the test.

---

## 2. What we know about Decider 2B (from AWS's announcement)

- 2B-parameter decision model (Qwen3.5-2B torso, LM head replaced with a small pointer head, rank-16 LoRA). It **cannot generate text**.
- Takes a **state + typed question**; returns a **choice**, a **yes/no**, or a **0–1 score**, each with a **calibrated confidence**.
- Median latency ~115 ms (RTX 3090), ~153 ms on an M3 MacBook for small tasks. Latency grows roughly **linearly with token count**, so keep prompts short.
- Efficient to ask **several questions about the same prompt**.
- Weak at complex reasoning (single parallel pass, no lookahead). Great at easy, rote decisions.
- Open source: code `github.com/strands-labs/strands-decider`, weights on Hugging Face (`StrandsAgents`). CLI model id in the blog: `StrandsAgents/strands-decider-2B-hobson-v19`.
- Install: `pip install strands-decider`.
- Python API shape shown in the blog: `Decider.noul(question, {option: description, ...})`. Wiring: an `InterventionHandler` with `before_tool_call`, passed as `Agent(interventions=[...])`, returning `Proceed`, `Deny`, `Confirm`, or `Guide`.

> Verify the exact API (especially for score-type questions and batching several questions) in the repo's `examples/` before writing code. Do not guess signatures.

### Implication for our design

Do **not** ask "which of these 8 patches is best?" (that needs lookahead). Ask **per-candidate** questions ("does this patch address the failing assertion?") and rank by the resulting scores. This also avoids option-position bias and lets K grow beyond what fits in one prompt.

---

## 3. Rules checklist (what a valid submission needs)

- [ ] Agent that uses Decider 2B for at least one meaningful decision, **paired with Amazon Bedrock**
- [ ] Written **article on AWS Builder Center** (agent, what it does, the decision handed to Decider)
- [ ] **Public repo** with source code + at least one **reusable decision pattern** (typed question + option set, wiring such as an `InterventionHandler`, short usage note)
- [ ] **Demo video under 3 minutes**, publicly available
- [ ] One declared **decision pattern**
- [ ] Original, not previously published
- [ ] Tags added before the deadline: `#strands-decider-2b` and one pattern tag

**Pattern tag choice:** none of the five official patterns is "search". Closest: **`#tool-selection`** (choosing among candidate actions). Ask in the hackathon's Discussion tab whether a candidate-ranking use is accepted, and describe it explicitly in the article as action selection.

**Judging weights:** Creativity 30%, Effective use of Decider 30% (includes reusable pattern quality; a measured latency/cost gain is a bonus), Communication 25%, Originality 15%.

---

## 4. Architecture

```
failing test + buggy function
          │
          ▼
  ┌─────────────────┐   K candidate patches
  │ Bedrock proposer│ ───────────────────────┐
  └─────────────────┘                        ▼
                                  ┌─────────────────────┐
                                  │ Decider ranker      │  per-candidate typed questions
                                  │ (local, ~100 ms)    │  → score + confidence
                                  └─────────┬───────────┘
                                            │ ranked list
                  low confidence ┌──────────┴───────────┐ high confidence
                                 ▼                      ▼
                       ┌──────────────────┐     ┌───────────────┐
                       │ Bedrock LLM judge│────▶│ Test runner   │  run in rank order,
                       │ (escalation)     │     │ (sandboxed)   │  stop at first pass
                       └──────────────────┘     └──────┬────────┘
                                                       │ all failed?
                                                       ▼
                                      feed failure output back to proposer
                                      (depth 2–3, keep alternatives, backtrack)
```

Roles:
- **Bedrock**: generative parts (propose patches, LLM-judge baseline, optional explanation of the final fix).
- **Decider**: fast, cheap scoring of candidates, called many times.
- **Test runner**: the verifier. Nothing is "fixed" unless tests pass.

---

## 5. The Decider questions (the reusable decision pattern)

State given to Decider = failing test output (trimmed) + the original function + the candidate patch as a **short diff**. Keep it short; latency scales with tokens.

Draft question set (write these with `Decider.noul(...)`, following the blog's style of describing each option):

| Key | Type | Question | Options / range |
|---|---|---|---|
| `addresses_failure` | yes/no | Does this patch change the code path that the failing assertion exercises? | true / false |
| `likely_passes` | score 0–1 | How likely is it that this patch makes the failing test pass? | 0–1 |
| `side_effects` | yes/no | Does the patch change behaviour unrelated to the failing test? | true / false |
| `minimal` | yes/no | Is this a minimal, focused change? | true / false |

Combine into one ranking score, for example:

```
score = P(addresses_failure) * P(likely_passes) * (1 - P(side_effects))
```

Treat the combination formula as a hyperparameter. Tune it on a small dev split, **not** on the evaluation split.

**Confidence use (the cascade):** if the top-ranked candidates are separated by less than a margin, or the confidence on `likely_passes` is below a threshold, escalate to the Bedrock LLM judge for those candidates only.

### Reusable asset to ship

`decider_ranker/` as a small package:
- `questions.py`: the typed question set above
- `ranker.py`: `DeciderRanker.rank(state, candidates) -> list[(candidate, score, confidence)]`
- `intervention.py`: an `InterventionHandler` whose `before_tool_call` ranks several proposed tool calls/candidates and returns `Guide` with feedback (or `Proceed`) based on the scores. Start from the repo's `examples/strands/` grounded-arguments example.
- `README.md`: how to define your own question set, how to pick thresholds, known limits.

---

## 6. The experiment

### Dataset
- 30–50 small buggy Python functions, each with a test that fails before the fix and passes after.
- Source options: HumanEvalFix-style tasks, or hand-seed bugs into simple functions (off-by-one, wrong operator, wrong boundary, missing edge case, wrong variable).
- Split: **~10 dev** (tune formula and thresholds) / **rest held out** (report results).

### Candidate pool
For each task, have Bedrock generate **K = 8** candidate patches (temperature > 0). **Run all K against the tests once** to record the ground-truth label for each (pass/fail). Cache everything. This lets you evaluate every ranker offline on identical candidates without repeating expensive calls.

### Rankers compared (same candidate pools)
1. **Random** (average over many shuffles)
2. **Heuristic** (e.g., smallest diff first, or passes lint/type-check first)
3. **LLM judge, small Bedrock model**
4. **LLM judge, large Bedrock model**
5. **Decider** (per-candidate questions)
6. **Cascade** (Decider, escalate to the large judge when uncertain)

### Metrics
- **Test runs to first pass** (mean/median; primary, since it is the search cost)
- **Top-1 / top-3 hit rate**
- **AUROC** of the ranker score vs pass/fail across all candidates
- **Calibration**: reliability curve of Decider's score vs the empirical pass rate; Brier score
- **Latency per ranking** and **estimated cost per ranking** (Decider is local, so count compute time; Bedrock cost from token usage)
- **Ordering stability**: rerun Decider with shuffled candidate order and report score variance

### Headline figures for the article
1. Pareto chart: **accuracy (test runs saved) vs cost/latency** for all rankers
2. Calibration curve for Decider
3. Bar chart: mean test runs to first pass

### Hypotheses (write down before running; report whichever way it goes)
- H1: Decider beats random clearly.
- H2: Decider beats the cheap heuristic modestly.
- H3: The large LLM judge is more accurate but much slower and costlier.
- H4: The cascade sits on the Pareto frontier.
- H5: Decider's confidence is informative about *its own* correctness but not a reliable proxy for patch quality (this is the key finding to examine honestly).

### Pitfalls to control
- **Leakage:** the pre-test Decider must not see test results or the reference fix.
- **Compute-matched baselines:** compare at equal numbers of verifier calls and report the cost axis.
- **Position bias:** shuffle candidate order; use per-candidate questions.
- **Prompt length:** use diffs, trim traces; check how latency scales on your hardware.
- **Overfitting the combination formula:** tune on dev, report on held-out.
- **Small sample:** report confidence intervals (bootstrap) rather than point estimates alone.

---

## 7. Project structure

```
decider-patch-search/
├── README.md
├── pyproject.toml
├── decider_ranker/            # the reusable pattern
│   ├── questions.py
│   ├── ranker.py
│   └── intervention.py
├── agent/
│   ├── proposer.py            # Bedrock patch proposals
│   ├── judge.py               # Bedrock LLM-judge baseline(s)
│   ├── runner.py              # sandboxed test execution
│   └── search.py              # rank → run → feed back (beam / backtracking)
├── eval/
│   ├── tasks/                 # buggy functions + tests
│   ├── build_pool.py          # generate + label candidates, cache to disk
│   ├── run_rankers.py         # evaluate all rankers on cached pools
│   └── plots.py               # Pareto, calibration, bars
├── demo/
│   └── cli_demo.py            # live ranking view for the video
└── results/                   # cached pools, metrics JSON, figures
```

---

## 8. Build steps (how)

### Setup (day 1, morning)
1. Create/verify your AWS account; **request Bedrock model access** in the console for the models you will use; configure credentials locally.
2. `pip install strands-decider` and run the blog's CLI example to confirm the model loads. Note whether you have a usable GPU; if not, test CPU latency or use a cloud instance (the hackathon credits only come after winning, so budget small).
3. Clone `strands-labs/strands-decider` and read `examples/strands/` for the intervention pattern and the real API.
4. Create the public GitHub repo with an open-source license and the structure above.

### Core build
5. **Tasks:** write/collect 30–50 buggy functions with tests.
6. **Runner:** execute candidate code + tests in a subprocess with a timeout and a temp directory. Never run untrusted code unsandboxed; at minimum use subprocess isolation with resource limits.
7. **Proposer:** Bedrock call returning K patches as diffs or full-function replacements; parse robustly; deduplicate.
8. **Pool builder:** generate and label candidate pools; cache to JSON.
9. **Baselines:** random, heuristic, small and large LLM judge.
10. **Decider ranker:** implement questions, per-candidate scoring, shuffling check.
11. **Evaluation script:** compute all metrics and figures from cached pools.
12. **Agent loop:** wire proposer → ranker → (escalate) → runner, as a Strands agent. Add the `InterventionHandler` variant.
13. **Stretch:** multi-step search. If all K fail, include the failure message in the next proposal round, keep the best-ranked failed branches, and backtrack. Report depth-1 results first; depth 2–3 only if time allows.

### Demo
14. `cli_demo.py` prints the 8 candidates with Decider's scores and latency in real time, then runs tests in order and shows "fixed after N runs instead of ~M for random".

---

## 9. Timeline (Oct 20 → Oct 23)

| When | Goal |
|---|---|
| **Tue Oct 20** | Setup, Decider running, tasks + runner + proposer, baselines started |
| **Wed Oct 21** | Candidate pools labeled, all rankers evaluated, first figures, agent loop wired |
| **Thu Oct 22** | Intervention handler + README, demo video, article draft, polish repo |
| **Fri Oct 23 (PDT) / morning Oct 24 IST** | Buffer, publish article, tags, verify links, **submit with hours to spare** |

Cut order if behind: multi-step search, then the cascade, then the second LLM judge. Never cut: the baseline comparison, the reusable pattern, the video, the article.

---

## 10. Article outline (Builder Center)

1. **Title + one-line hook**: "Letting a 2B decision model decide which patch to test first"
2. **The problem:** searching candidate fixes is expensive because every check costs a test run or an LLM call.
3. **The decision handed to Decider:** state, typed questions, option set, how scores combine.
4. **Architecture:** diagram from section 4; Bedrock's role vs Decider's role.
5. **Decision pattern declared:** tool-selection (action selection); how to reuse it.
6. **Experiment:** dataset, baselines, metrics, hypotheses stated up front.
7. **Results:** Pareto chart, calibration curve, test-runs bar chart. Report honestly where Decider wins and loses.
8. **What I learned:** confidence vs value, position bias, prompt length vs latency.
9. **How to reuse it:** copy-paste snippet for the `InterventionHandler`.
10. **Limits and future work:** fine-tuning Decider on patch outcomes, other domains (layout search, planning), using scores inside beam search.
11. Link to the repo and the demo video; add tags `#strands-decider-2b` and `#tool-selection`.

---

## 11. Demo video script (target 2:30)

- **0:00–0:20** The problem and the idea in one sentence.
- **0:20–0:50** Architecture diagram: Bedrock proposes, Decider ranks, tests verify.
- **0:50–1:40** Live demo: buggy function, 8 patches scored in about a second, tests run in ranked order, fix found.
- **1:40–2:15** Results: the Pareto chart and calibration curve; one honest takeaway.
- **2:15–2:30** The reusable pattern and where to find the repo.

Keep the screen readable (large font), narrate clearly, and upload as a publicly viewable video.

---

## 12. Submission-day checklist

- [ ] Repo is public, has a license, and the README works from a clean clone
- [ ] Reusable pattern is in its own folder with a short usage note
- [ ] Results and figures are reproducible from cached data (document the command)
- [ ] Article is published on Builder Center
- [ ] Both tags added: `#strands-decider-2b` + `#tool-selection`
- [ ] Video is public, under 3 minutes, linked in the submission
- [ ] Submission form is complete and saved before the deadline

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Decider's scores barely beat random | Still a valid, publishable finding; show the calibration data and the cascade; emphasise cost/latency |
| Search pattern not accepted under the five tags | Ask in Discussion early; also ship the intervention-handler form of the pattern |
| API differs from the blog's snippets | Read repo examples first; keep a thin wrapper so only one file changes |
| Not enough time for multi-step search | Depth-1 reranking is a complete submission on its own |
| Slow local inference | Shorter prompts, batch questions per candidate, use a GPU instance if needed |
| Running generated code is unsafe | Subprocess isolation, timeouts, temp dirs, no network, resource limits |

---

## 14. Sources

- Introducing Strands Decider 2B: https://strandsagents.com/blog/introducing-strands-decider/
- Hackathon page and rules: https://builder.aws.com/build/hackathons/4f50957c-1d08-37ab-b11c-97516b9adeb4/agent-experience-hackathon
- Strands Decider repo: https://github.com/strands-labs/strands-decider
