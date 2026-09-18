import pytest
from shopmind.app.domain.search_query import SearchMode, SearchRequest
from shopmind.app.domain.product import Product
from shopmind.app.domain.search_result import RawSearchHit, RetrievalScores, RetrievedDocument, SearchResult


def test_search_request_rejects_blank_query():
    with pytest.raises(ValueError): SearchRequest(query="   ", mode=SearchMode.HYBRID)


def test_search_request_has_stable_defaults():
    request = SearchRequest(query="black running shoes")
    assert request.mode is SearchMode.HYBRID
    assert request.top_k == 10
    assert request.candidate_k == 100
    assert request.filters == {}


def test_search_request_allows_filter_only_without_query():
    request = SearchRequest(mode=SearchMode.BM25, filters={"brand": "Acme"})

    assert request.query is None


def test_search_request_rejects_invalid_rank_window():
    with pytest.raises(ValueError): SearchRequest(query="shoes", top_k=11, candidate_k=10)


def test_product_main_image_is_first_image():
    product = Product(product_id="p1", title="Shoe", description="A shoe", brand="Brand", category="Shoes", attributes={"color": "black"}, image_paths=("main.jpg", "alt.jpg"))
    assert product.main_image_path == "main.jpg"


def test_result_contracts_are_constructible():
    hit = RawSearchHit(product_id="p1", title="Shoe", score=1.2)
    scores = RetrievalScores(bm25_rank=1)
    result = SearchResult(product_id=hit.product_id, title=hit.title, rank=1, score=hit.score, source="bm25", retrieval_scores=scores)
    doc = RetrievedDocument(id="p1", content="Shoe", score=1.2, source="product")
    assert result.retrieval_scores.bm25_rank == 1
    assert doc.id == result.product_id
