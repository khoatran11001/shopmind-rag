from __future__ import annotations

from fastapi.testclient import TestClient

from shopmind.app.domain.search_result import RetrievalScores, SearchResult
from shopmind.app.main import create_app
from tests.api.conftest import FakeEmbedder, FakeRepository, FakeSearchService


def test_debug_search_explains_hybrid_scores_without_changing_ranking():
    service = FakeSearchService(results=[
        SearchResult(product_id="P2", title="Second", rank=1, score=0.2, source="hybrid", retrieval_scores=RetrievalScores(bm25_rank=2, dense_rank=1, rrf_score=1 / 19 + 1 / 18, reranker_score=0.2)),
        SearchResult(product_id="P1", title="First", rank=2, score=1 / 18, source="hybrid", retrieval_scores=RetrievalScores(bm25_rank=1, rrf_score=1 / 18)),
    ])
    app = create_app(search_service=service, repository=FakeRepository(), embedder=FakeEmbedder(), fusion_rrf_k=17, auto_wire=False)

    with TestClient(app) as client:
        response = client.post("/api/v1/debug/search", json={"query": "running shoes", "top_k": 2})
        public = client.post("/api/v1/search/text", json={"query": "running shoes", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["rrf_k"] == 17
    assert [item["product_id"] for item in body["results"]] == ["P2", "P1"]
    assert body["results"][0]["final_score"] == 0.2
    assert body["results"][0]["source"] == "hybrid"
    assert body["results"][0]["bm25_rank"] == 2
    assert body["results"][0]["dense_rank"] == 1
    assert body["results"][0]["bm25_rrf_contribution"] == 1 / 19
    assert body["results"][0]["dense_rrf_contribution"] == 1 / 18
    assert body["results"][0]["rrf_score"] == 1 / 19 + 1 / 18
    assert body["results"][0]["reranker_score"] == 0.2
    assert body["results"][0]["bm25_score"] is None
    assert body["results"][0]["dense_score"] is None
    assert body["results"][1]["dense_rrf_contribution"] is None
    assert [item["product_id"] for item in public.json()["results"]] == ["P2", "P1"]
    assert "rrf_score" not in public.json()["results"][0]


def test_debug_search_direct_mode_has_native_score_without_rrf(client, fake_service):
    fake_service.results = [SearchResult(product_id="P1", title="Runner", rank=1, score=3.4, source="bm25", retrieval_scores=RetrievalScores(bm25_rank=1))]

    response = client.post("/api/v1/debug/search", json={"query": "runner", "mode": "bm25"})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["final_score"] == 3.4
    assert result["bm25_score"] == 3.4
    assert result["dense_score"] is None
    assert result["bm25_rrf_contribution"] is None
    assert result["rrf_score"] is None
    assert fake_service.text_requests[0].mode.value == "bm25"
