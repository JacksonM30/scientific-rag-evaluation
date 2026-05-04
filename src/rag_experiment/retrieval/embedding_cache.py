"""Local embedding vector-store cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.vectorstores import InMemoryVectorStore

from rag_experiment.data.hotpotqa import Passage


CACHE_SCHEMA_VERSION = "embedding-cache-v0.1"


@dataclass(frozen=True)
class EmbeddingCacheSpec:
    """Cache identity for one embedded passage pool."""

    namespace: str
    provider: str
    model: str
    dimensions: int | None
    base_url: str


@dataclass(frozen=True)
class EmbeddingCacheInfo:
    """Runtime metadata for artifact records."""

    enabled: bool
    hit: bool
    path: Path | None
    provider: str
    model: str
    dimensions: int | None

    def as_run_metadata(self) -> dict[str, Any]:
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_dimensions": self.dimensions,
            "embedding_cache_enabled": self.enabled,
            "embedding_cache_hit": self.hit,
            "embedding_cache_path": str(self.path) if self.path is not None else None,
        }


def cache_path_for(
    *,
    cache_dir: Path,
    passages: list[Passage],
    spec: EmbeddingCacheSpec,
) -> Path:
    """Return a deterministic cache path for the passage pool and model."""
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "namespace": spec.namespace,
                "provider": spec.provider,
                "model": spec.model,
                "dimensions": spec.dimensions,
                "base_url": spec.base_url,
                "passages": [
                    {
                        "id": passage.id,
                        "text_sha256": _sha256_text(passage.text),
                    }
                    for passage in passages
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    safe_model = spec.model.replace("/", "_").replace(":", "_")
    dim_label = spec.dimensions if spec.dimensions is not None else "default"
    filename = f"{spec.namespace}_{safe_model}_d{dim_label}_{digest[:16]}.json"
    return cache_dir / filename


def load_vectorstore(path: Path, embedding_model: Any) -> InMemoryVectorStore | None:
    """Load a cached LangChain in-memory vector store if it exists."""
    if not path.exists():
        return None
    return InMemoryVectorStore.load(str(path), embedding_model)


def save_vectorstore(path: Path, vectorstore: InMemoryVectorStore) -> None:
    """Save a LangChain in-memory vector store for later reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.dump(str(path))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
