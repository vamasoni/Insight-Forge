"""Shared construction of db / llm / agent for the CLI, API, MCP server and UI."""
from __future__ import annotations

from functools import lru_cache

from insightforge.agent import InsightForgeAgent
from insightforge.config import Settings, get_settings
from insightforge.db import Database, open_database
from insightforge.llm import LLMProvider, get_llm


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def database() -> Database:
    s = settings()
    return open_database(s.db_path, s.db_dialect)


@lru_cache(maxsize=1)
def llm() -> LLMProvider:
    return get_llm(settings())


@lru_cache(maxsize=1)
def agent() -> InsightForgeAgent:
    return InsightForgeAgent(database(), llm(), settings())
