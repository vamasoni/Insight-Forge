"""Runtime settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    return val if val not in (None, "") else default


@dataclass
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))  # openai | anthropic | fake
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_base_url: str | None = field(default_factory=lambda: _env("LLM_BASE_URL"))  # Mistral / vLLM / Ollama
    llm_api_key: str | None = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.0")))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "60")))

    # Database
    db_path: str = field(default_factory=lambda: _env("DB_PATH", "data/olist.duckdb"))
    db_dialect: str = field(default_factory=lambda: _env("DB_DIALECT", "duckdb"))  # duckdb | sqlite

    # SQL tool
    max_rows: int = field(default_factory=lambda: int(_env("MAX_ROWS", "5000")))
    analysis_max_rows: int = field(default_factory=lambda: int(_env("ANALYSIS_MAX_ROWS", "500000")))
    query_timeout_s: float = field(default_factory=lambda: float(_env("QUERY_TIMEOUT_S", "30")))
    max_repair_attempts: int = field(default_factory=lambda: int(_env("MAX_REPAIR_ATTEMPTS", "2")))
    use_schema_retrieval: bool = field(default_factory=lambda: _env("USE_SCHEMA_RETRIEVAL", "1") == "1")
    retrieval_top_tables: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_TABLES", "4")))

    # Agent
    max_sub_questions: int = field(default_factory=lambda: int(_env("MAX_SUB_QUESTIONS", "4")))
    max_critic_revisions: int = field(default_factory=lambda: int(_env("MAX_CRITIC_REVISIONS", "1")))
    analysis_timeout_s: float = field(default_factory=lambda: float(_env("ANALYSIS_TIMEOUT_S", "30")))

    # Storage
    runs_dir: Path = field(default_factory=lambda: Path(_env("RUNS_DIR", "runs")))

    @property
    def provider_label(self) -> str:
        return f"{self.llm_provider}:{self.llm_model}"


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise AttributeError(f"Unknown setting: {k}")
        setattr(s, k, v)
    return s
