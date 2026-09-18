from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from shopmind.app.api.errors import InvalidFilter, UnsupportedSearchMode
from shopmind.app.domain.search_query import SearchMode, SearchRequest
from shopmind.app.domain.search_result import SearchResult

_ALLOWED_FILTERS = {"brand", "category", "product_id"}


class TextSearchRequest(BaseModel):
    query: str | None = None
    mode: str = SearchMode.HYBRID.value
    top_k: int = Field(default=10, gt=0)
    candidate_k: int | None = Field(default=None, gt=0)
    filters: dict[str, str | list[str]] = Field(default_factory=dict)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        cleaned = value.strip()
        return cleaned or None

    @field_validator("filters", mode="before")
    @classmethod
    def normalize_filter_values(cls, value: Any) -> dict[str, str | list[str]]:
        return normalize_filters(value)

    @model_validator(mode="after")
    def candidate_window_must_cover_top_k(self) -> "TextSearchRequest":
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        if self.query is None and not self.filters:
            raise ValueError("query or at least one filter is required")
        return self

    def to_domain(self) -> SearchRequest:
        try:
            mode = SearchMode(self.mode)
        except ValueError as exc:
            raise UnsupportedSearchMode(f"unsupported search mode: {self.mode}") from exc
        if self.query is None and mode is not SearchMode.BM25:
            raise UnsupportedSearchMode("filter-only search requires bm25 mode")
        unknown = sorted(set(self.filters) - _ALLOWED_FILTERS)
        if unknown:
            raise InvalidFilter(f"unsupported filter field: {unknown[0]}")
        candidate_k = self.candidate_k if self.candidate_k is not None else max(100, self.top_k)
        return SearchRequest(query=self.query, mode=mode, top_k=self.top_k, candidate_k=candidate_k, filters=dict(self.filters))


class ProductSearchResult(BaseModel):
    product_id: str
    title: str
    image_url: str | None = None
    brand: str | None = None
    category: str | None = None
    score: float
    rank: int

    @classmethod
    def from_domain(cls, result: SearchResult) -> "ProductSearchResult":
        return cls(product_id=result.product_id, title=result.title, image_url=_optional_str(result.metadata.get("image_url")), brand=_optional_str(result.metadata.get("brand")), category=_optional_str(result.metadata.get("category")), score=result.score, rank=result.rank)


class TextSearchResponse(BaseModel):
    query: str | None
    mode: str
    filter_only: bool = False
    results: list[ProductSearchResult]


class ImageSearchResponse(BaseModel):
    mode: str = "image_dense"
    results: list[ProductSearchResult]


class FilterOption(BaseModel):
    value: str
    count: int


class FilterOptionsResponse(BaseModel):
    field: str
    options: list[FilterOption]


def normalize_filters(value: Any) -> dict[str, str | list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("filters must be an object")
    normalized: dict[str, str | list[str]] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise ValueError("filter names must be strings")
        if raw is None:
            continue
        if isinstance(raw, str):
            cleaned = raw.strip()
            if cleaned:
                normalized[key] = cleaned
            continue
        if isinstance(raw, list):
            values: list[str] = []
            for item in raw:
                if not isinstance(item, str):
                    raise ValueError(f"filter {key} list must contain strings")
                cleaned = item.strip()
                if cleaned and cleaned not in values:
                    values.append(cleaned)
            if values:
                normalized[key] = values
            continue
        raise ValueError(f"filter {key} must be a string or list of strings")
    return normalized


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
