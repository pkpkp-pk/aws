"""Strands Agent intervention wiring for the ranker.

Pattern: before an agent commits to a consequential tool call, Decider
scores the proposed call (and any alternatives) against typed questions.
High score -> proceed. Low score -> guide the agent with feedback.

The strands `InterventionHandler` contract is duck-typed here so this
module imports without strands installed: any handler exposing
`before_tool_call(call)` and returning Proceed/Deny/Guide-shaped
decisions composes the same way. When wiring into a real Agent, pass
`interventions=[handler]` and adapt the verdict objects to the SDK's
own Proceed/Deny/Guide types in one place (see `_to_sdk_verdict`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from decider_ranker.client import Decider


@dataclass(frozen=True)
class Verdict:
    kind: str          # "proceed" | "deny" | "guide"
    feedback: str = ""


def _to_sdk_verdict(v: Verdict, sdk: Any) -> Any:
    """Adapt to the strands SDK's verdict types. One seam, edit here only."""
    return {
        "proceed": getattr(sdk, "Proceed", lambda: v)(),
        "deny": getattr(sdk, "Deny", lambda feedback="": v)(feedback=v.feedback),
        "guide": getattr(sdk, "Guide", lambda feedback="": v)(feedback=v.feedback),
    }[v.kind]


class ToolCallRanker:
    """Decides whether a proposed tool call should run.

    describe(call) -> (state, options): state text for Decider plus a
    mapping of option label -> human description of what choosing that
    option means. The choice question's answer picks the action.
    """

    def __init__(self, decider: Decider,
                 describe: Callable[[Any], tuple[str, Mapping[str, str]]],
                 proceed_label: str = "run",
                 min_confidence: float = 0.7) -> None:
        self.decider = decider
        self.describe = describe
        self.proceed_label = proceed_label
        self.min_confidence = min_confidence

    def before_tool_call(self, call: Any) -> Verdict:
        state, options = self.describe(call)
        answers = self.decider.ask(state, {
            "action": Decider.choice(
                "The agent wants to perform the action described in the state. "
                "What should happen next?", dict(options)),
        })
        ans = answers["action"]
        choice = ans.get("choice") or ans.get("answer")
        conf = float(ans.get("confidence", 0.0))
        if choice == self.proceed_label and conf >= self.min_confidence:
            return Verdict("proceed")
        if choice == self.proceed_label:
            return Verdict("guide",
                           f"Proceed looks right but confidence is low ({conf:.2f}); "
                           "double-check the arguments before retrying.")
        return Verdict("guide",
                       f"Consider '{choice}' instead (confidence {conf:.2f}).")
