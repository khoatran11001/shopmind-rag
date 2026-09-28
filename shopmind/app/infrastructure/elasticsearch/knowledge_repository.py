from __future__ import annotations

from typing import Any
from shopmind.app.knowledge.base import KnowledgeHit


class ElasticsearchKnowledgeRepository:
    def __init__(self, client: Any, index_alias: str, *, source: str) -> None:
        if source not in {"review", "policy"}:
            raise ValueError("source must be review or policy")
        self.client = client
        self.index_alias = index_alias
        self.source = source

    def ping(self) -> bool:
        return bool(self.client.ping())

    def alias_exists(self, alias: str | None = None) -> bool:
        return bool(self.client.indices.exists_alias(name=alias or self.index_alias))

    def lexical_search(self, query: str, size: int, product_id: str | None = None) -> list[KnowledgeHit]:
        if size <= 0:
            raise ValueError("size must be positive")
        fields = ["review_title^2", "content", "search_text"] if self.source == "review" else ["title^2", "content", "search_text"]
        search_query: dict[str, Any] = {"multi_match": {"query": query, "fields": fields}}
        if self.source == "review" and product_id is not None:
            search_query = {"bool": {"must": [search_query], "filter": [{"term": {"product_id": product_id}}]}}
        response = self.client.search(index=self.index_alias, query=search_query, size=size, source_excludes=["text_vector"])
        return [self._hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def vector_search(self, vector: list[float], size: int, num_candidates: int, product_id: str | None = None) -> list[KnowledgeHit]:
        if size <= 0 or num_candidates < size:
            raise ValueError("num_candidates must be >= positive size")
        knn: dict[str, Any] = {"field": "text_vector", "query_vector": vector, "k": size, "num_candidates": num_candidates}
        if self.source == "review" and product_id is not None:
            knn["filter"] = {"term": {"product_id": product_id}}
        response = self.client.search(index=self.index_alias, knn=knn, size=size, source_excludes=["text_vector"])
        return [self._hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def bulk_index(self, index_name: str, documents: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        if not documents:
            return 0, []
        from elasticsearch.helpers import bulk
        id_field = "review_id" if self.source == "review" else "section_id"
        actions = [{"_op_type": "index", "_index": index_name, "_id": str(document[id_field]), "_source": document} for document in documents]
        success, errors = bulk(self.client, actions, raise_on_error=False, raise_on_exception=False, refresh=False)
        return int(success), list(errors)

    def _hit(self, hit: dict[str, Any]) -> KnowledgeHit:
        source = {key: value for key, value in dict(hit.get("_source") or {}).items() if key != "text_vector"}
        id_field = "review_id" if self.source == "review" else "section_id"
        title_field = "review_title" if self.source == "review" else "title"
        return KnowledgeHit(document_id=str(source.get(id_field) or hit.get("_id") or ""), title=str(source.get(title_field) or ""), score=float(hit.get("_score") or 0.0), metadata=source)


__all__ = ["ElasticsearchKnowledgeRepository", "KnowledgeHit"]
