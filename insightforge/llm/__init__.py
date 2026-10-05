from .base import LLMProvider, LLMResponse, Usage, extract_json, track_usage
from .providers import AnthropicProvider, FakeProvider, OpenAICompatibleProvider


def get_llm(settings=None) -> LLMProvider:
    from insightforge.config import get_settings

    s = settings or get_settings()
    kw = dict(model=s.llm_model, api_key=s.llm_api_key, base_url=s.llm_base_url,
              timeout_s=s.llm_timeout_s, temperature=s.llm_temperature)
    if s.llm_provider == "anthropic":
        return AnthropicProvider(**kw)
    if s.llm_provider in ("openai", "mistral", "vllm", "ollama"):
        return OpenAICompatibleProvider(**kw)
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider!r} (FakeProvider is built directly in tests)")


__all__ = ["LLMProvider", "LLMResponse", "Usage", "extract_json", "track_usage", "get_llm",
           "AnthropicProvider", "OpenAICompatibleProvider", "FakeProvider"]
