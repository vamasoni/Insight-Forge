"""Analysis tool: runs statistics on query results in a separate, resource-limited Python process.

Two modes:
  - helper: call a vetted function from stats_lib (the default path the planner uses)
  - code:   run model-written pandas/scipy code after an AST allowlist check

Isolation here = subprocess + timeout + (POSIX) CPU/memory rlimits + empty env + temp cwd + AST checks.
That stops accidents and casual abuse. For hostile multi-tenant use, run the whole service in a container
with --network none and a read-only filesystem (see docker-compose.yml).
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from insightforge.evidence import CalcEvidence, EvidenceStore
from insightforge.tracing import get_tracer

ALLOWED_IMPORTS = {"pandas", "numpy", "scipy", "scipy.stats", "statsmodels", "statsmodels.api",
                   "statsmodels.formula.api", "plotly", "plotly.express", "plotly.graph_objects", "math",
                   "statistics", "insightforge.tools.stats_lib"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "__builtins__"}
_IO_ATTR_PREFIXES = ("read_", "to_csv", "to_pickle", "to_parquet", "to_excel", "to_sql", "to_hdf", "to_feather",
                     "to_stata", "to_html", "to_latex", "to_clipboard", "to_markdown", "to_xml", "to_orc",
                     "write", "save", "load", "system", "popen")


class UnsafeCode(Exception):
    pass


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m not in ALLOWED_IMPORTS:
                    raise UnsafeCode(f"import of {m!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCode(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeCode("dunder attribute access is not allowed")
            if node.attr.startswith(_IO_ATTR_PREFIXES):
                raise UnsafeCode(f"file/system I/O ({node.attr}) is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise UnsafeCode("global/nonlocal is not allowed")


@dataclass
class AnalysisRun:
    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)  # plotly JSON
    error: str = ""
    stdout: str = ""
    evidence: CalcEvidence | None = None


class AnalysisTool:
    def __init__(self, timeout_s: float = 30.0, memory_mb: int = 2048):
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    def run_helper(self, method: str, args: dict, frames: dict[str, pd.DataFrame], *,
                   store: EvidenceStore | None = None, description: str = "") -> AnalysisRun:
        from insightforge.tools.stats_lib import REGISTRY

        if method not in REGISTRY:
            return AnalysisRun(False, error=f"unknown analysis {method!r}; choose from {sorted(REGISTRY)}")
        job = {"mode": "helper", "method": method, "args": args, "input": next(iter(frames))}
        return self._execute(job, frames, store, description or f"{method} on {', '.join(frames)}", method, "")

    def run_code(self, code: str, frames: dict[str, pd.DataFrame], *, store: EvidenceStore | None = None,
                 description: str = "custom analysis") -> AnalysisRun:
        """`code` sees each frame as a variable named after its evidence id (Q1, Q2, ...) and must assign
        a dict to `result`. It may assign a plotly figure (or list of them) to `fig`."""
        try:
            check_code(code)
        except UnsafeCode as e:
            return AnalysisRun(False, error=f"rejected: {e}")
        return self._execute({"mode": "code", "code": code}, frames, store, description, "custom_code", code)

    def _execute(self, job: dict, frames: dict[str, pd.DataFrame], store, description, method, code) -> AnalysisRun:
        with get_tracer().span("analysis_tool", as_type="tool", input={"job": job, "frames": list(frames)}) as span:
            with tempfile.TemporaryDirectory(prefix="if_sandbox_") as tmp:
                tmpd = Path(tmp)
                with (tmpd / "frames.pkl").open("wb") as f:
                    pickle.dump(frames, f)
                (tmpd / "job.json").write_text(json.dumps(job, default=str))
                pkg_root = str(Path(__file__).resolve().parents[2])
                env = {"PYTHONPATH": os.pathsep.join([pkg_root, *[p for p in sys.path if p]]), "PATH": os.environ.get("PATH", ""),
                       "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
                if os.name == "nt":  # Windows needs these to start Python at all
                    env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "insightforge.tools.sandbox_runner", str(tmpd), str(self.memory_mb),
                         str(int(self.timeout_s))],
                        cwd=tmpd, env=env, capture_output=True, text=True, timeout=self.timeout_s + 5)
                except subprocess.TimeoutExpired:
                    run = AnalysisRun(False, error=f"analysis exceeded {self.timeout_s}s")
                    span["output"] = run.error
                    return run
                res_path = tmpd / "result.json"
                if not res_path.exists():
                    killed = proc.returncode is not None and proc.returncode < 0
                    msg = ("analysis killed: CPU or memory limit exceeded" if killed
                           else (proc.stderr or "sandbox produced no result")[-800:])
                    run = AnalysisRun(False, error=msg)
                    span["output"] = run.error
                    return run
                res = json.loads(res_path.read_text())
            run = AnalysisRun(res.get("ok", False), res.get("outputs") or {}, res.get("figures") or [],
                              res.get("error", ""), res.get("stdout", ""))
            if run.ok and store is not None:
                run.evidence = store.add_calc(description=description, method=run.outputs.get("method", method),
                                              inputs=list(frames), outputs=run.outputs, code=code)
            span["output"] = {"ok": run.ok, "error": run.error, "outputs": run.outputs}
            return run
