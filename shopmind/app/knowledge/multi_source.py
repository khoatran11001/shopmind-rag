from __future__ import annotations

from shopmind.app.domain.search_result import RetrievedDocument
from shopmind.app.rag.models import SOURCES


class MultiSourceRetriever:
    def __init__(self, retrievers: dict[str, object], candidate_counts: dict[str, int]) -> None:
        self.retrievers = dict(retrievers)
        self.candidate_counts = dict(candidate_counts)

    def retrieve(self, query: str, sources: tuple[str, ...] | set[str], product_id: str | None = None) -> dict[str, list[RetrievedDocument]]:
        requested = set(sources)
        if not requested or requested - set(self.retrievers):
            raise ValueError("invalid knowledge sources")
        return {source: self.retrievers[source].retrieve(query, self.candidate_counts[source], product_id=product_id) for source in SOURCES if source in requested}
