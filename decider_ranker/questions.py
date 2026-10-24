"""The typed question set for ranking candidate patches.

The reusable decision pattern: given a state (buggy function + failing
test output + one candidate patch), ask four narrow questions in ONE
request — the server encodes the state once, so each extra question is
nearly free.

- addresses_failure (noul): does the patch touch the failing code path?
- side_effects     (noul): does it change unrelated behavior?
- minimal          (noul): is it a focused, minimal change?
- likely_passes    (score): 4-level rubric, normalized to 0..1.

Question wording and criteria carry most of the signal (per the
strands-decider docs) — treat them as hyperparameters: tune on the dev
split only, never on held-out tasks.
"""
from __future__ import annotations

from decider_ranker.client import Decider

QUESTION_KEYS = ("addresses_failure", "side_effects", "minimal", "likely_passes")

SCORE_LEVELS = ["very unlikely", "unlikely", "likely", "very likely"]


def patch_questions() -> dict[str, dict]:
    """Fresh question dicts (the server may not mutate them, but callers
    should treat returned answers as keyed by these names)."""
    return {
        "addresses_failure": Decider.noul(
            "Does this patch change the code path exercised by the failing assertion?",
            {
                "true": "the patch modifies the expression or branch the assertion checks",
                "false": "the patch leaves the failing code path unchanged",
            },
        ),
        "side_effects": Decider.noul(
            "Does the patch change behavior unrelated to the failing test?",
            {
                "true": "behavior outside the failing assertion changes",
                "false": "only the failing behavior is affected",
            },
        ),
        "minimal": Decider.noul(
            "Is this a minimal, focused change to the function?",
            {
                "true": "the patch changes only what the failure requires",
                "false": "the patch rewrites large parts or adds unrelated logic",
            },
        ),
        "likely_passes": Decider.score(
            "How likely is this patch to make the failing test pass?",
            SCORE_LEVELS,
        ),
    }
