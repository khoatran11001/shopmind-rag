from __future__ import annotations

from typing import Any

from shopmind.app.domain.search_result import RawSearchHit

_FILTER_FIELDS = {"brand": "brand.keyword", "category": "category.keyword", "product_id": "product_id"}
_VECTOR_FIELDS = {"text_vector", "image_vector"}
_FACET_FIELDS = set(_FILTER_FIELDS)


def _values(value: str | list[str], key: str) -> list[str]:
    raw_values = [value] if isinstance(value, str) else value
    if not raw_values or not all(isinstance(item, str) and item.strip() for item in raw_values):
        raise ValueError(f"filter {key} must contain non-empty strings")
    return list(dict.fromkeys(item.strip() for item in raw_values))


def _escape_wildcard(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def _canonical_partial_value(key: str, value: str) -> str:
    cleaned = value.strip()
    if key == "category":
        cleaned = cleaned.replace("-", "_").replace(" ", "_")
    return _escape_wildcard(cleaned)


def build_filter_clauses(filters: dict[str, str | list[str]]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for key in sorted(filters):
        if key not in _FILTER_FIELDS:
            raise ValueError(f"unsupported filter field: {key}")
        field = _FILTER_FIELDS[key]
        value = filters[key]
        values = _values(value, key)
        if len(values) == 1:
            clauses.append({"term": {field: {"value": values[0], "case_insensitive": True}}})
        else:
            clauses.append({"terms": {field: values}})
    return clauses


def build_partial_filter_clauses(filters: dict[str, str | list[str]]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for key in sorted(filters):
        if key not in _FILTER_FIELDS:
            raise ValueError(f"unsupported filter field: {key}")
        field = _FILTER_FIELDS[key]
        values = _values(filters[key], key)
        queries = []
        for value in values:
            cleaned = _canonical_partial_value(key, value)
            if key == "product_id":
                queries.append({"prefix": {field: {"value": cleaned, "case_insensitive": True}}})
            else:
                queries.append({"wildcard": {field: {"value": f"*{cleaned}*", "case_insensitive": True}}})
        clauses.append(queries[0] if len(queries) == 1 else {"bool": {"should": queries, "minimum_should_match": 1}})
    return clauses


def _source_without_vectors(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key not in {"text_vector", "image_vector"}}


def _to_raw_hit(hit: dict[str, Any]) -> RawSearchHit:
    source = dict(hit.get("_source") or {})
    return RawSearchHit(product_id=str(source.get("product_id") or hit.get("_id") or ""), title=str(source.get("title") or ""), score=float(hit.get("_score") or 0.0), metadata=_source_without_vectors(source))


class ElasticsearchProductRepository:
    def __init__(self, client: Any, index_alias: str = "products") -> None:
        self.client = client
        self.index_alias = index_alias

    def ping(self) -> bool:
        return bool(self.client.ping())

    def alias_exists(self, alias: str) -> bool:
        return bool(self.client.indices.exists_alias(name=alias))

    def _lexical_once(self, query: str | None, size: int, clauses: list[dict[str, Any]]) -> list[RawSearchHit]:
        must = [{"match_all": {}}] if query is None else [{"multi_match": {"query": query, "fields": ["title^3", "brand^2", "category^2", "description", "attributes_text", "search_text"]}}]
        search_query: dict[str, Any] = {"bool": {"must": must, "filter": clauses}}
        kwargs: dict[str, Any] = {"index": self.index_alias, "query": search_query, "size": size, "source_excludes": ["text_vector", "image_vector"]}
        if query is None:
            kwargs["sort"] = [{"product_id": "asc"}]
        response = self.client.search(**kwargs)
        return [_to_raw_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def lexical_search(self, query: str | None, size: int, filters: dict[str, str | list[str]]) -> list[RawSearchHit]:
        if size <= 0:
            raise ValueError("size must be positive")
        hits = self._lexical_once(query, size, build_filter_clauses(filters))
        return hits or (self._lexical_once(query, size, build_partial_filter_clauses(filters)) if filters else [])

    def vector_search(self, field: str, vector: list[float], size: int, num_candidates: int, filters: dict[str, str | list[str]]) -> list[RawSearchHit]:
        if field not in _VECTOR_FIELDS:
            raise ValueError(f"unsupported vector field: {field}")
        if size <= 0 or num_candidates < size:
            raise ValueError("num_candidates must be >= positive size")
        clauses = build_filter_clauses(filters)

        def search_with(filter_clauses: list[dict[str, Any]]) -> list[RawSearchHit]:
            knn: dict[str, Any] = {"field": field, "query_vector": vector, "k": size, "num_candidates": num_candidates}
            if filter_clauses:
                knn["filter"] = filter_clauses[0] if len(filter_clauses) == 1 else {"bool": {"filter": filter_clauses}}
            response = self.client.search(index=self.index_alias, knn=knn, size=size, source_excludes=["text_vector", "image_vector"])
            return [_to_raw_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

        hits = search_with(clauses)
        return hits or (search_with(build_partial_filter_clauses(filters)) if filters else [])

    def filter_options(self, field: str, prefix: str = "", limit: int = 20) -> list[dict[str, Any]]:
        if field not in _FACET_FIELDS:
            raise ValueError(f"unsupported filter field: {field}")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        keyword_field = _FILTER_FIELDS[field]
        kwargs: dict[str, Any] = {
            "index": self.index_alias,
            "size": 0,
            "aggs": {"options": {"terms": {"field": keyword_field, "size": limit, "order": [{"_count": "desc"}, {"_key": "asc"}]}}},
        }
        if prefix:
            escaped = _canonical_partial_value(field, prefix)
            if field == "product_id":
                kwargs["query"] = {"prefix": {keyword_field: {"value": escaped, "case_insensitive": True}}}
            else:
                kwargs["query"] = {"wildcard": {keyword_field: {"value": f"*{escaped}*", "case_insensitive": True}}}
        response = self.client.search(**kwargs)
        buckets = response.get("aggregations", {}).get("options", {}).get("buckets", [])
        return [{"value": str(bucket["key"]), "count": int(bucket.get("doc_count", 0))} for bucket in buckets]

    def bulk_index(self, index_name: str, documents: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        if not documents:
            return 0, []
        try:
            from elasticsearch.helpers import bulk
        except ImportError as exc:
            raise RuntimeError("elasticsearch Python package is required for bulk indexing") from exc
        actions = [{"_op_type": "index", "_index": index_name, "_id": str(document["product_id"]), "_source": document} for document in documents]
        success, errors = bulk(self.client, actions, raise_on_error=False, raise_on_exception=False, refresh=False)
        return int(success), list(errors)
