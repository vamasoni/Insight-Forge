"""Concrete providers. Raw HTTP via httpx keeps dependencies small and behaviour explicit."""
from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from .base import LLMProvider, LLMResponse


class _HTTPProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None, base_url: str, timeout_s: float, temperature: float):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.client = httpx.Client(timeout=timeout_s)

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        for attempt in range(4):
            r = self.client.post(url, headers=headers, json=payload)

            if r.status_code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(2 ** attempt)
                continue

            if r.is_error:
                raise RuntimeError(
                    f"LLM API error {r.status_code}\n"
                    f"URL: {url}\n"
                    f"Response: {r.text}\n"
                    f"Headers: {dict(r.headers)}"
                )

            return r.json()

        raise RuntimeError("LLM request failed after retries")


class OpenAICompatibleProvider(_HTTPProvider):
    """OpenAI, Mistral (https://api.mistral.ai/v1), vLLM, Ollama, Groq ... any /chat/completions API."""

    def __init__(self, model, api_key=None, base_url=None, timeout_s=60.0, temperature=0.0):
        super().__init__(model, api_key, base_url or "https://api.openai.com/v1", timeout_s, temperature)

    def _complete(self, system, messages, max_tokens, temperature):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self._post(f"{self.base_url}/chat/completions", headers, payload)
        usage = data.get("usage") or {}
        return LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class AnthropicProvider(_HTTPProvider):
    def __init__(self, model, api_key=None, base_url=None, timeout_s=60.0, temperature=0.0):
        super().__init__(model, api_key, base_url or "https://api.anthropic.com/v1", timeout_s, temperature)

    def _complete(self, system, messages, max_tokens, temperature):
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self._post(f"{self.base_url}/messages", headers, payload)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


class FakeProvider(LLMProvider):
    """Deterministic provider for tests and CI.

    Pass `responder(system, messages, role) -> str`, or a `queue` of canned replies.
    """

    def __init__(self, responder: Callable[[str, list[dict], str], str] | None = None, queue: list[str] | None = None):
        self.model = "fake"
        self.responder = responder
        self.queue = list(queue or [])
        self.calls: list[dict] = []
        self._role = "default"

    def complete(self, system, user, *, role="default", max_tokens=1500, temperature=None):
        self._role = role
        return super().complete(system, user, role=role, max_tokens=max_tokens, temperature=temperature)

    def _complete(self, system, messages, max_tokens, temperature):
        self.calls.append({"role": self._role, "system": system, "messages": list(messages)})
        if self.responder:
            text = self.responder(system, messages, self._role)
        elif self.queue:
            text = self.queue.pop(0)
        else:
            raise RuntimeError("FakeProvider has no responses left")
        n_in = sum(len(str(m["content"])) for m in messages) // 4
        return LLMResponse(text=text, model=self.model, input_tokens=n_in, output_tokens=len(text) // 4)
