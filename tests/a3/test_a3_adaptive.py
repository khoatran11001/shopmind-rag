from __future__ import annotations

import numpy as np
import pytest

from shopmind.app.core.config import RAGRetrievalConfig
from shopmind.app.domain.search_query import SearchMode
from shopmind.app.knowledge.policy_retriever import PolicyDocumentRetriever
from shopmind.app.knowledge.product_retriever import ProductDocumentRetriever
from shopmind.app.knowledge.review_retriever import ReviewDocumentRetriever


class Search:
    def __init__(self):
        self.requests = []

    def search_text(self, request):
        self.requests.append(request)
        return []

    def to_retrieved_documents(self, results):
        return []


class Repository:
    def __init__(self):
        self.calls = []

    def lexical_search(self, query, size, *, product_id=None):
        self.calls.append("bm25")
        return []

    def vector_search(self, vector, size, candidate_k, *, product_id=None):
        self.calls.append("dense")
        return []


class Embedder:
    def __init__(self):
        self.calls = 0

    def embed_texts(self, queries):
        self.calls += 1
        return np.array([[1.0, 0.0]], dtype=np.float32)


@pytest.mark.parametrize(
    ("query", "expected_mode", "expected_paths"),
    [
        ("RTX 5060", SearchMode.BM25, ["bm25"]),
        ("laptop nhẹ cho developer", SearchMode.DENSE, ["dense"]),
        ("laptop RTX 5060 nhẹ", SearchMode.HYBRID, ["bm25", "dense"]),
    ],
)
def test_adaptive_mode_routes_all_a3_sources(query, expected_mode, expected_paths):
    settings = RAGRetrievalConfig(product_mode="adaptive", review_mode="adaptive", policy_mode="adaptive")
    search = Search()
    review_repo = Repository()
    policy_repo = Repository()
    embedder = Embedder()

    ProductDocumentRetriever(search, mode=settings.product_mode).retrieve(query, 3)
    ReviewDocumentRetriever(review_repo, embedder, mode=settings.review_mode).retrieve(query, 3)
    PolicyDocumentRetriever(policy_repo, embedder, mode=settings.policy_mode).retrieve(query, 3)

    assert search.requests[0].mode == expected_mode
    assert review_repo.calls == expected_paths
    assert policy_repo.calls == expected_paths
    assert embedder.calls == 2 * ("dense" in expected_paths)


def test_adaptive_product_id_keeps_exact_lookup():
    search = Search()
    ProductDocumentRetriever(search, mode="adaptive").retrieve("laptop nhẹ cho developer", 3, product_id="P1")
    assert search.requests[0].query is None
    assert search.requests[0].mode == SearchMode.BM25
    assert search.requests[0].filters == {"product_id": "P1"}


def test_explicit_bm25_mode_ignores_adaptive_query_routing():
    search = Search()
    repo = Repository()
    ProductDocumentRetriever(search, mode="bm25").retrieve("laptop nhẹ cho developer", 3)
    ReviewDocumentRetriever(repo, None, mode="bm25").retrieve("laptop nhẹ cho developer", 3)
    assert search.requests[0].mode == SearchMode.BM25
    assert repo.calls == ["bm25"]
