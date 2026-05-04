"""Embedding clients for retrieval."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag_experiment.model_clients.providers import EMBEDDING_PROVIDERS


def get_embedding_model(
    provider: str = "dashscope",
    model: str | None = None,
    **overrides: Any,
) -> Embeddings:
    """Create an embedding model from a named provider."""

    if provider not in EMBEDDING_PROVIDERS:
        available = ", ".join(sorted(EMBEDDING_PROVIDERS))
        raise ValueError(f"Unknown embedding provider {provider!r}. Available: {available}")

    cfg = EMBEDDING_PROVIDERS[provider]
    api_key_env = overrides.pop("api_key_env", cfg["api_key_env"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing environment variable: {api_key_env}")

    return OpenAIEmbeddings(
        model=model or cfg["default_model"],
        api_key=api_key,
        base_url=overrides.pop("base_url", cfg["base_url"]),
        chunk_size=overrides.pop("chunk_size", cfg.get("chunk_size", 25)),
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        **overrides,
    )
