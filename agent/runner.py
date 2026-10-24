"""Sandboxed execution of candidate patches against tests.

A candidate is a full-function replacement (source string). Tests are a
plain Python script of asserts importing `solution` from the candidate
module. Execution happens in a subprocess with resource limits, a
timeout, a fresh temp directory, and a scrubbed environment.

Nothing runs in-process. The verifier is the ground truth: a patch is
"fixed" only when the test script exits 0.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

TIMEOUT_S = 5
MAX_OUTPUT_CHARS = 2000
# RLIMIT_AS (address space) caps memory; RLIMIT_CPU caps CPU seconds.
MAX_MEMORY_BYTES = 512 * 1024 * 1024
MAX_CPU_S = 4


@dataclass(frozen=True)
class RunResult:
    passed: bool
    output: str  # trimmed stdout+stderr
    duration_s: float
    timed_out: bool = False


def _limits() -> None:
    """preexec_fn: applied in the child before exec. Unix only."""
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_S, MAX_CPU_S + 1))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))


def run_candidate(candidate_src: str, test_src: str,
                  timeout_s: float = TIMEOUT_S) -> RunResult:
    """Run test_src against candidate_src in a sandboxed subprocess.

    Returns RunResult; passed=True iff the test script exits 0.
    Never raises for candidate/test failures — those are data.
    """
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "solution.py"), "w") as f:
            f.write(candidate_src)
        test_path = os.path.join(tmp, "run_tests.py")
        with open(test_path, "w") as f:
            f.write(test_src)

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmp,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = subprocess.run(
                # No -I: it implies -P, which removes the script's dir from
                # sys.path and breaks `from solution import ...`. Isolation
                # comes from the scrubbed env + tmpdir cwd + rlimits.
                [sys.executable, test_path],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "")
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            return RunResult(
                passed=False,
                output=(out + "\n<TIMEOUT>")[-MAX_OUTPUT_CHARS:],
                duration_s=time.monotonic() - start,
                timed_out=True,
            )

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return RunResult(
        passed=proc.returncode == 0,
        output=output[-MAX_OUTPUT_CHARS:],
        duration_s=time.monotonic() - start,
    )
