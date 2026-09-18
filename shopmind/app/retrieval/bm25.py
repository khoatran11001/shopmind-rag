from __future__ import annotations

from typing import Protocol

from shopmind.app.domain.search_query import SearchRequest
from shopmind.app.domain.search_result import RawSearchHit, RetrievalScores, SearchResult


class _LexicalRepository(Protocol):
    def lexical_search(self, query: str | None, size: int, filters: dict[str, str | list[str]]) -> list[RawSearchHit]: ...


class BM25Retriever:
    def __init__(self, repository: _LexicalRepository) -> None:
        self.repository = repository

    def search(self, request: SearchRequest) -> list[SearchResult]:
        hits = self.repository.lexical_search(request.query, size=request.candidate_k, filters=request.filters)
        return [SearchResult(product_id=hit.product_id, title=hit.title, rank=rank, score=hit.score, source="bm25", metadata=dict(hit.metadata), retrieval_scores=RetrievalScores(bm25_rank=rank)) for rank, hit in enumerate(hits[: request.top_k], start=1)]
