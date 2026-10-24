"""Bedrock LLM judge baseline: per-candidate pass-likelihood score.

One `converse` call per candidate (per-candidate, not listwise — avoids
position bias and matches how Decider is asked). Returns (score 0..1,
latency_s). Model id configurable; small/large judges differ only by id.
"""
from __future__ import annotations

import json
import os
import re
import time

import boto3

SMALL_MODEL = os.environ.get("JUDGE_SMALL_MODEL",
                             "us.anthropic.claude-haiku-4-5-20251001-v1:0")
LARGE_MODEL = os.environ.get("JUDGE_LARGE_MODEL",
                             "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")

_PROMPT = """\
A Python function has a bug. Here is the function:

```python
{source}
```

Its failing test output:

```
{failure}
```

A candidate fix:

```python
{candidate}
```

How likely is this candidate to make all the tests pass? Reply with ONLY
a JSON object: {{"p": <number from 0.0 to 1.0>}}"""


class BedrockJudge:
    def __init__(self, model: str, client=None) -> None:
        self.model = model
        self.client = client or boto3.client("bedrock-runtime", region_name=REGION)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def score(self, source: str, failure: str, candidate: str) -> tuple[float, float]:
        t0 = time.monotonic()
        resp = self.client.converse(
            modelId=self.model,
            messages=[{
                "role": "user",
                "content": [{"text": _PROMPT.format(
                    source=source, failure=failure, candidate=candidate)}],
            }],
            inferenceConfig={"temperature": 0.0, "maxTokens": 64},
        )
        dt = time.monotonic() - t0
        usage = resp.get("usage", {})
        self.total_input_tokens += usage.get("inputTokens", 0)
        self.total_output_tokens += usage.get("outputTokens", 0)
        text = resp["output"]["message"]["content"][0]["text"]
        m = re.search(r'"p"\s*:\s*([0-9]*\.?[0-9]+)', text)
        p = float(m.group(1)) if m else 0.0
        return max(0.0, min(1.0, p)), dt
