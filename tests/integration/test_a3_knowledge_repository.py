from __future__ import annotations

import os
import uuid

import pytest

from shopmind.app.infrastructure.elasticsearch.knowledge_index import create_knowledge_index, validate_knowledge_index
from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository

pytestmark = pytest.mark.integration


def test_review_bm25_and_knn_keep_selected_product_scope():
    from elasticsearch import Elasticsearch

    client = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    if not client.ping():
        pytest.skip("Elasticsearch is not reachable")
    index_name = f"test_a3_reviews_{uuid.uuid4().hex}"
    documents = [
        {"review_id": "R1", "product_id": "P1", "review_title": "Comfortable", "content": "Comfortable fit", "search_text": "Comfortable fit", "text_vector": [1.0, 0.0, 0.0, 0.0], "embedding_model": "fake", "embedding_version": "test", "metadata": {}},
        {"review_id": "R2", "product_id": "P2", "review_title": "Comfortable", "content": "Comfortable fit", "search_text": "Comfortable fit", "text_vector": [0.9, 0.1, 0.0, 0.0], "embedding_model": "fake", "embedding_version": "test", "metadata": {}},
    ]
    try:
        create_knowledge_index(client, index_name, "review", 4)
        repo = ElasticsearchKnowledgeRepository(client, index_name, source="review")
        success, errors = repo.bulk_index(index_name, documents)
        assert (success, errors) == (2, [])
        client.indices.refresh(index=index_name)
        assert [hit.document_id for hit in repo.lexical_search("comfortable", 2, product_id="P1")] == ["R1"]
        assert [hit.document_id for hit in repo.vector_search([1, 0, 0, 0], 2, 2, product_id="P2")] == ["R2"]
        validate_knowledge_index(client, index_name, "review", 2, 4, "comfortable", [1, 0, 0, 0])
        client.indices.put_alias(index=index_name, name=f"{index_name}_alias")
        with pytest.raises(ValueError, match="alias"):
            create_knowledge_index(client, index_name, "review", 4, recreate=True)
        assert client.count(index=index_name)["count"] == 2
    finally:
        client.indices.delete(index=index_name, ignore_unavailable=True)
