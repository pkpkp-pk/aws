"""Bedrock patch proposer.

One `converse` call returns K candidate full-function replacements as
JSON. Candidates are AST-validated (must parse and define the same
function name) and deduplicated by normalized AST dump.

Model id is configurable: set PROPOSER_MODEL or pass `model`.
"""
from __future__ import annotations

import ast
import json
import os
import re

import boto3

DEFAULT_MODEL = os.environ.get(
    "PROPOSER_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
REGION = os.environ.get("AWS_REGION", "us-east-1")

_PROMPT = """\
A Python function has a bug. Here is the function:

```python
{source}
```

Running its tests gives:

```
{failure}
```

Write {k} DIFFERENT corrected versions of the function. Each must be a
complete replacement (same function name and signature). Vary the
approach: minimal one-line fixes, rewrites, different algorithms.

Reply with ONLY a JSON object, no prose:
{{"patches": [{{"code": "<full function source>"}}, ...]}}
"""


def _fn_name(source: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    raise ValueError("no function definition in source")


def _normalize(source: str) -> str:
    """AST dump without locations: two textual variants of one fix dedupe."""
    return ast.dump(ast.parse(source))


def _extract_patches(text: str, want_fn: str) -> list[str]:
    """Pull candidate sources out of the model reply. Robust to prose
    around the JSON and to markdown fences."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in data.get("patches", []):
        code = item.get("code", "") if isinstance(item, dict) else ""
        try:
            if _fn_name(code) == want_fn:
                out.append(code if code.endswith("\n") else code + "\n")
        except (SyntaxError, ValueError):
            continue
    return out


def propose(source: str, failure: str, k: int = 8,
            model: str | None = None, client=None) -> list[str]:
    """Return up to k deduplicated, AST-valid candidate replacements."""
    client = client or boto3.client("bedrock-runtime", region_name=REGION)
    want_fn = _fn_name(source)
    resp = client.converse(
        modelId=model or DEFAULT_MODEL,
        messages=[{
            "role": "user",
            "content": [{"text": _PROMPT.format(source=source, failure=failure, k=k)}],
        }],
        inferenceConfig={"temperature": 0.8, "maxTokens": 4096},
    )
    text = resp["output"]["message"]["content"][0]["text"]
    patches = _extract_patches(text, want_fn)
    seen, unique = set(), []
    for p in patches:
        key = _normalize(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
