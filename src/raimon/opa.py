"""Evaluate the Rego policies through a local OPA server.

The policy files under `policies/` are the single definition of verdicts and deployment decisions. Python
code computes statistics and intervals; OPA decides.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import requests

POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OpaServer:
    """A local `opa run --server` process loaded with the repository's policies."""

    def __init__(self, opa_bin: str | None = None, policy_dir: Path = POLICY_DIR):
        binary = opa_bin or shutil.which("opa")
        if binary is None:
            raise FileNotFoundError("opa binary not found on PATH")
        self.port = _free_port()
        policies = [str(p) for p in sorted(policy_dir.glob("*.rego")) if not p.name.endswith("_test.rego")]
        self.proc = subprocess.Popen(
            [binary, "run", "--server", "--addr", f"127.0.0.1:{self.port}", "--log-level", "error", *policies],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self.session = requests.Session()
        self.url = f"http://127.0.0.1:{self.port}/v1/data/raimon/verdict"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                if self.session.get(f"http://127.0.0.1:{self.port}/health", timeout=1).ok:
                    return
            except requests.ConnectionError:
                time.sleep(0.05)
        self.close()
        raise RuntimeError("OPA server did not start")

    def evaluate(self, domains: list[str], checks: list[dict]) -> dict:
        """Return {"verdict": {id: v}, "decision": {domain: d}} for one round."""
        r = self.session.post(self.url, json={"input": {"domains": domains, "checks": checks}}, timeout=30)
        r.raise_for_status()
        result = r.json().get("result", {})
        return {"verdict": result.get("verdict", {}), "decision": result.get("decision", {})}

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=5)

    def __enter__(self) -> "OpaServer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
