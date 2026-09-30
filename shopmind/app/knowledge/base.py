from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any, Protocol

from shopmind.app.domain.search_result import RetrievedDocument


class KnowledgeMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


_MODEL_ID = re.compile(r"\b[a-z][a-z0-9-]*(?:[\s-]+\d+[a-z0-9-]*|[a-z0-9-]*\d[a-z0-9-]*)\b", re.IGNORECASE)


def resolve_knowledge_mode(mode: KnowledgeMode, query: str) -> KnowledgeMode:
    if mode != KnowledgeMode.ADAPTIVE:
        return mode
    model = _MODEL_ID.search(query)
    if model is None:
        return KnowledgeMode.DENSE
    # ponytail: model-token heuristic; replace with a measured router if retrieval evaluation shows misses.
    has_description = re.search(r"\w", query[:model.start()] + query[model.end():]) is not None
    return KnowledgeMode.HYBRID if has_description else KnowledgeMode.BM25


@dataclass(frozen=True)
class KnowledgeHit:
    document_id: str
    title: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]: ...


def rrf_knowledge_hits(rankings, *, k: int = 60, top_k: int = 40):
    if k <= 0 or top_k <= 0:
        raise ValueError("k and top_k must be positive")
    scores: dict[str, float] = {}
    first: dict[str, KnowledgeHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            first.setdefault(hit.document_id, hit)
            scores[hit.document_id] = scores.get(hit.document_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))[:top_k]
    return [(first[document_id], scores[document_id]) for document_id in ordered]
