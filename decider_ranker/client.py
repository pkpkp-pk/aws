"""Synchronous HTTP client for a running strands-decider server.

Follows the shape of `examples/strands/_client.py` from the
strands-decider repo: stdlib urllib only, one state + many typed
questions per round trip (the state is encoded once on the server, so
batching questions is nearly free).

Two modes:
- local:  POST /v1/systemone, no key. Start: strands-decider serve <checkpoint> --port 8099
- hosted: POST /v1/decide, Authorization: Bearer <key> (AWS UG Madurai platform).
  Set DECIDER_URL + DECIDER_API_KEY.

Endpoint: env STRANDS_DECIDER_URL or DECIDER_URL, default http://127.0.0.1:8099.
Key: env DECIDER_API_KEY (presence selects hosted mode).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

ENV_URL = "STRANDS_DECIDER_URL"
DEFAULT_URL = f"http://127.0.0.1:{os.environ.get('STRANDS_DECIDER_PORT', '8099')}"
# The server answers as whatever checkpoint it loaded; this field only
# completes the wire shape.
MODEL_ID = "hobson-latest"
SERVE_HINT = "strands-decider serve <checkpoint> --port 8099"
HOSTED_URL = "https://da25tnxy06gjk.cloudfront.net"


class ServerUnavailable(RuntimeError):
    """The server is not answering. Raised with the command that starts one."""


class Decider:
    """One state, many typed questions, one round trip."""

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0) -> None:
        self.api_key = api_key if api_key is not None else \
            os.environ.get("DECIDER_API_KEY") or None
        default = HOSTED_URL if self.api_key else DEFAULT_URL
        self.url = (url or os.environ.get(ENV_URL)
                    or os.environ.get("DECIDER_URL") or default).rstrip("/")
        self.hosted = bool(self.api_key)
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.last_latency_ms: float | None = None

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no question. Answer is P(true); with two outcomes that is the confidence."""
        q: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            q["criteria"] = dict(criteria)
        return q

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N. criteria maps option name -> what the option means."""
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    def _unavailable(self, exc: Exception) -> ServerUnavailable:
        return ServerUnavailable(
            f"no strands-decider at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
        )

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate state against questions; answers come back under the same keys."""
        body = json.dumps(
            {"state": state, "model": MODEL_ID, "questions": dict(questions)}
        ).encode()
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        path = "/v1/decide" if self.hosted else "/v1/systemone"
        request = urllib.request.Request(
            f"{self.url}{path}", data=body, headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            hint = {401: "check DECIDER_API_KEY", 402: "monthly quota used up",
                    429: "rate limited; retry"}.get(exc.code, "")
            raise ServerUnavailable(
                f"decider returned {exc.code} {hint}: "
                f"{exc.read().decode(errors='replace')[:200]}") from exc
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc
        self.calls += 1
        self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
        self.last_latency_ms = payload.get("latency_ms")
        return payload["answers"]

    def health(self) -> dict[str, Any]:
        path = "/v1/models" if self.hosted else "/health"
        headers = ({"authorization": f"Bearer {self.api_key}"}
                   if self.api_key else {})
        request = urllib.request.Request(f"{self.url}{path}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc
