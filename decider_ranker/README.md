# decider_ranker — reusable decision pattern

Rank runtime-generated candidates (patches, tool calls, plans) with a
Strands Decider 2B server. Per-candidate typed questions over one shared
state; combined score orders the candidates; confidence drives escalation.

## Pieces

- `client.py` — stdlib HTTP client. One state + many typed questions per
  round trip (`POST /v1/systemone`); the server encodes the state once.
  Server URL via `STRANDS_DECIDER_URL` (default `http://127.0.0.1:8099`).
- `questions.py` — the typed question set. Wording and criteria carry most
  of the signal: treat them as hyperparameters, tune on dev data only.
- `ranker.py` — `DeciderRanker.rank(source, failure, candidates)` →
  candidates sorted by combined score, with per-candidate confidence.
- `intervention.py` — `ToolCallRanker`, a `before_tool_call` handler for
  Strands `Agent(interventions=[...])`: Decider scores a proposed action
  and proceeds / guides before anything executes.

## Define your own question set

```python
from decider_ranker.client import Decider

d = Decider()  # server must be running
answers = d.ask(state_text, {
    "safe": Decider.noul("Is this action safe to run unsupervised?",
                         {"true": "reversible, no external effects",
                          "false": "touches shared state or is irreversible"}),
    "quality": Decider.score("Rate expected quality",
                             ["poor", "ok", "good", "excellent"]),
})
p_safe = answers["safe"]["noul"]
level  = answers["quality"]["score"]       # 0..3 index into your levels
conf   = answers["quality"]["confidence"]  # calibrated
```

Rules of thumb:

- Ask **per-candidate** questions, never "which of these K is best" —
  a single-pass pointer head has no lookahead, and listwise questions add
  position bias.
- Keep the state short. Latency grows roughly linearly with token count.
- Batch all questions for one candidate into one `ask` — the shared prefix
  is encoded once, extra questions are nearly free.
- Use `confidence` to escalate: below your threshold, re-ask a bigger model
  or run the real verifier.

## Known limits

- Weak at complex reasoning; great at easy, rote decisions. Narrow the
  questions until the task is rote.
- Combination formulas overfit easily — tune on a dev split, report held-out.
- Confidence is informative about the model's own certainty; it is not a
  guarantee about the world. The verifier stays the ground truth.
