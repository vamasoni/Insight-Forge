"""Tracing adapter.

Langfuse is used when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set and the SDK is installed.
Otherwise spans go to runs/traces.jsonl so latency/token/cost data is never lost.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_TRACER = None


class _LocalTracer:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()  # span stacks are per thread (evals run questions in parallel)
        self._io = threading.Lock()

    @property
    def _stack(self) -> list[str]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @property
    def _trace_id(self) -> str | None:
        return getattr(self._local, "trace_id", None)

    @_trace_id.setter
    def _trace_id(self, v: str | None) -> None:
        self._local.trace_id = v

    def _write(self, rec: dict) -> None:
        with self._io, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    @contextlib.contextmanager
    def span(self, name: str, as_type: str = "span", input: Any = None, metadata: dict | None = None):
        sid = uuid.uuid4().hex[:12]
        root = not self._stack
        if root:
            self._trace_id = uuid.uuid4().hex[:16]
        parent = self._stack[-1] if self._stack else None
        self._stack.append(sid)
        t0 = time.perf_counter()
        holder: dict = {}
        err = None
        try:
            yield holder  # callers may put {"output": ...} in here
        except Exception as e:  # record and re-raise
            err = repr(e)
            raise
        finally:
            self._stack.pop()
            self._write({
                "trace_id": self._trace_id, "span_id": sid, "parent": parent, "name": name, "type": as_type,
                "latency_s": round(time.perf_counter() - t0, 4), "input": _clip(input),
                "output": _clip(holder.get("output")), "metadata": metadata, "error": err,
            })

    def generation(self, role, system, messages, response) -> None:
        self._write({
            "trace_id": self._trace_id, "parent": self._stack[-1] if self._stack else None,
            "name": f"llm:{role}", "type": "generation", "model": response.model,
            "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
            "cost_usd": round(response.cost_usd, 6), "latency_s": round(response.latency_s, 4),
        })

    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    def flush(self) -> None:
        pass


class _LangfuseTracer:
    def __init__(self):
        from langfuse import get_client

        self.client = get_client()

    @contextlib.contextmanager
    def span(self, name: str, as_type: str = "span", input: Any = None, metadata: dict | None = None):
        with self.client.start_as_current_observation(name=name, as_type=as_type, input=_clip(input),
                                                      metadata=metadata) as obs:
            holder: dict = {}
            yield holder
            if "output" in holder:
                obs.update(output=_clip(holder["output"]))

    def generation(self, role, system, messages, response) -> None:
        g = self.client.start_observation(
            name=f"llm:{role}", as_type="generation", model=response.model,
            input=[{"role": "system", "content": system}, *messages], output=response.text,
            usage_details={"input": response.input_tokens, "output": response.output_tokens},
            cost_details={"total": response.cost_usd},
            metadata={"latency_s": response.latency_s},
        )
        g.end()

    @property
    def trace_id(self) -> str | None:
        return self.client.get_current_trace_id()

    def flush(self) -> None:
        self.client.flush()


def _clip(x: Any, n: int = 4000) -> Any:
    if x is None:
        return None
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + "...<truncated>"


def get_tracer():
    global _TRACER
    if _TRACER is None:
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                _TRACER = _LangfuseTracer()
            except Exception:  # SDK missing or misconfigured -> never break the agent over tracing
                _TRACER = _LocalTracer(Path(os.getenv("RUNS_DIR", "runs")) / "traces.jsonl")
        else:
            _TRACER = _LocalTracer(Path(os.getenv("RUNS_DIR", "runs")) / "traces.jsonl")
    return _TRACER


def set_tracer(tracer) -> None:
    global _TRACER
    _TRACER = tracer
