"""Provider-agnostic LLM interface plus per-request usage accounting."""
from __future__ import annotations

import contextvars
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# USD per 1M tokens (input, output). Edit to match your provider's current pricing.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "mistral-small": (0.20, 0.60),
    "mistral-large": (2.00, 6.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
}


def price_for(model: str) -> tuple[float, float]:
    for key, p in PRICES.items():
        if model.startswith(key):
            return p
    return (0.0, 0.0)  # self-hosted / unknown -> track tokens only


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        pin, pout = price_for(self.model)
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    by_role: dict[str, dict] = field(default_factory=dict)

    def add(self, resp: LLMResponse, role: str) -> None:
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cost_usd += resp.cost_usd
        self.latency_s += resp.latency_s
        r = self.by_role.setdefault(role, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        r["calls"] += 1
        r["input_tokens"] += resp.input_tokens
        r["output_tokens"] += resp.output_tokens
        r["cost_usd"] += resp.cost_usd

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_latency_s": round(self.latency_s, 3),
            "by_role": self.by_role,
        }


_current_usage: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("usage", default=None)


class track_usage:
    """`with track_usage() as u: ...` collects every LLM call made inside the block."""

    def __enter__(self) -> Usage:
        self.usage = Usage()
        self._token = _current_usage.set(self.usage)
        return self.usage

    def __exit__(self, *exc) -> None:
        _current_usage.reset(self._token)


class LLMProvider(ABC):
    model: str = "unknown"
    temperature: float = 0.0

    @abstractmethod
    def _complete(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> LLMResponse:
        ...

    def complete(
        self,
        system: str,
        user: str | list[dict],
        *,
        role: str = "default",
        max_tokens: int = 1500,
        temperature: float | None = None,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": user}] if isinstance(user, str) else user
        t0 = time.perf_counter()
        resp = self._complete(system, messages, max_tokens, self.temperature if temperature is None else temperature)
        resp.latency_s = time.perf_counter() - t0
        usage = _current_usage.get()
        if usage is not None:
            usage.add(resp, role)
        from insightforge.tracing import get_tracer  # local import avoids a cycle

        get_tracer().generation(role=role, system=system, messages=messages, response=resp)
        return resp

    def complete_json(self, system: str, user: str, *, role: str = "default", max_tokens: int = 1500) -> dict | list:
        """Ask for JSON and parse it, with one retry that feeds the parse error back."""
        resp = self.complete(system, user, role=role, max_tokens=max_tokens)
        try:
            return extract_json(resp.text)
        except ValueError as e:
            retry = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": f"That was not valid JSON ({e}). Reply with only the JSON."},
            ]
            return extract_json(self.complete(system, retry, role=role, max_tokens=max_tokens).text)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict | list:
    m = _FENCE.search(text)
    candidate = (m.group(1) if m else text).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            s, e = candidate.find(open_c), candidate.rfind(close_c)
            if s != -1 and e > s:
                try:
                    return json.loads(candidate[s : e + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no parseable JSON found")
