from dataclasses import dataclass, field
from enum import StrEnum


class SearchMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    CROSS_MODAL = "cross_modal"


FilterValue = str | list[str]


@dataclass(frozen=True)
class SearchRequest:
    query: str | None = None
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = 10
    candidate_k: int = 100
    filters: dict[str, FilterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query = self.query.strip() if isinstance(self.query, str) else None
        object.__setattr__(self, "query", query or None)
        if not self.query and not self.filters:
            raise ValueError("query or at least one filter is required")
        if not self.query and self.mode != SearchMode.BM25:
            raise ValueError("filter-only search requires bm25 mode")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
