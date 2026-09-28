from __future__ import annotations

from typing import Any


def _vector(dimension: int) -> dict[str, Any]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return {"type": "dense_vector", "dims": dimension, "index": True, "similarity": "cosine"}


def build_reviews_mapping(dimension: int) -> dict[str, Any]:
    return {
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "review_id": {"type": "keyword"},
                "product_id": {"type": "keyword"},
                "asin": {"type": "keyword"},
                "parent_asin": {"type": "keyword"},
                "matched_by": {"type": "keyword"},
                "review_title": {"type": "text"},
                "content": {"type": "text"},
                "search_text": {"type": "text"},
                "rating": {"type": "float"},
                "verified_purchase": {"type": "boolean"},
                "helpful_vote": {"type": "integer"},
                "timestamp": {"type": "date", "format": "epoch_millis||strict_date_optional_time"},
                "text_vector": _vector(dimension),
                "embedding_model": {"type": "keyword"},
                "embedding_version": {"type": "keyword"},
                "metadata": {"type": "object", "dynamic": True},
            },
        }
    }


def build_policies_mapping(dimension: int) -> dict[str, Any]:
    return {
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "section_id": {"type": "keyword"},
                "policy_id": {"type": "keyword"},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "search_text": {"type": "text"},
                "policy_type": {"type": "keyword"},
                "version": {"type": "keyword"},
                "effective_date": {"type": "date"},
                "source_url": {"type": "keyword", "index": False},
                "text_vector": _vector(dimension),
                "embedding_model": {"type": "keyword"},
                "embedding_version": {"type": "keyword"},
                "metadata": {"type": "object", "dynamic": True},
            },
        }
    }


def create_knowledge_index(client: Any, index_name: str, source: str, dimension: int, *, recreate: bool = False) -> None:
    if source not in {"review", "policy"}:
        raise ValueError("source must be review or policy")
    mapping = build_reviews_mapping(dimension) if source == "review" else build_policies_mapping(dimension)
    exists = bool(client.indices.exists(index=index_name))
    if exists and not recreate:
        raise FileExistsError(index_name)
    if exists:
        indices = client.indices.get(index=index_name)
        if any(name != index_name or details.get("aliases") for name, details in indices.items()):
            raise ValueError(f"cannot recreate {index_name} while it has an alias; use a new versioned index")
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, **mapping)


def validate_knowledge_index(client: Any, index_name: str, source: str, expected_count: int, dimension: int, smoke_query: str, smoke_vector: list[float]) -> None:
    if source not in {"review", "policy"}:
        raise ValueError("source must be review or policy")
    if int(client.count(index=index_name)["count"]) != expected_count:
        raise ValueError("knowledge index document count mismatch")
    mapping = client.indices.get_mapping(index=index_name)
    entry = mapping[index_name] if index_name in mapping else next(iter(mapping.values()))
    if int(entry["mappings"]["properties"]["text_vector"]["dims"]) != dimension:
        raise ValueError("knowledge text_vector dimension mismatch")
    if expected_count == 0:
        return
    if len(smoke_vector) != dimension:
        raise ValueError("smoke vector dimension does not match index dimension")
    from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository
    repo = ElasticsearchKnowledgeRepository(client, index_name, source=source)
    if not repo.lexical_search(smoke_query, 1):
        raise ValueError("knowledge BM25 smoke query returned no results")
    if not repo.vector_search(smoke_vector, 1, max(1, min(expected_count, 10))):
        raise ValueError("knowledge kNN smoke query returned no results")
