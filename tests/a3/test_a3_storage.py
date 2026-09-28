import json
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from shopmind.app.domain.search_result import RetrievedDocument, SearchResult


def test_review_matches_only_exact_catalog_asin_or_parent_asin():
    from shopmind.pipeline.review_normalizer import match_review_product, normalize_review

    product_id = json.loads(Path("data/a1_abo_1500/products.jsonl").open().readline())["product_id"]
    catalog = {product_id}
    assert match_review_product({"asin": product_id + "X"}, catalog) == (None, None)
    assert match_review_product({"asin": "other", "parent_asin": product_id}, catalog) == (product_id, "parent_asin")
    review = normalize_review({"asin": product_id, "title": "Good", "text": "Fits", "user_id": "private"}, catalog)
    assert review.product_id == product_id
    assert review.matched_by == "asin"
    assert "private" not in json.dumps(review.to_dict())


def test_policy_corpus_uses_stable_namespaced_ids_and_experimental_provenance(tmp_path):
    from shopmind.pipeline.policy_loader import load_policy_corpus

    path = tmp_path / "policies.jsonl"
    path.write_text(json.dumps({
        "policy_id": "returns", "section_id": "returns.window", "title": "Return Window",
        "content": "Experimental returns within 30 days.", "policy_type": "returns",
        "version": "2026-09", "effective_date": "2026-09-01", "source_url": None,
        "metadata": {"provenance": "shopmind_curated_experimental"},
    }) + "\n")
    documents = load_policy_corpus(path)
    assert [document.document_id for document in documents] == ["policy:returns.window"]
    assert documents[0].metadata["version"] == "2026-09"
    assert documents[0].metadata["provenance"] == "shopmind_curated_experimental"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_policy_corpus(path)


def test_review_index_requires_embedding_order_and_keeps_product_link():
    from scripts.index_reviews import build_review_index_documents

    rows = [{"document_id": "review:r1", "source": "review", "title": "Good", "content": "Fits",
             "search_text": "Good\nFits", "metadata": {"review_id": "r1", "product_id": "P1", "matched_by": "asin"}}]
    vector = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="order"):
        build_review_index_documents(rows, ["review:wrong"], vector, embedding_model="m", embedding_version="v")
    documents = build_review_index_documents(rows, ["review:r1"], vector, embedding_model="m", embedding_version="v")
    assert documents[0]["review_id"] == "r1"
    assert documents[0]["product_id"] == "P1"
    assert documents[0]["text_vector"] == [1.0, 0.0]


def test_policy_index_keeps_section_id_and_version():
    from scripts.index_policies import build_policy_index_documents

    rows = [{"document_id": "policy:returns.window", "source": "policy", "title": "Return Window",
             "content": "30 days", "search_text": "Return Window\n30 days",
             "metadata": {"policy_id": "returns", "policy_type": "returns", "version": "2026-09",
                          "effective_date": "2026-09-01", "provenance": "shopmind_curated_experimental"}}]
    documents = build_policy_index_documents(rows, ["policy:returns.window"],
                                             np.array([[1.0, 0.0]], dtype=np.float32),
                                             embedding_model="m", embedding_version="v")
    assert documents[0]["section_id"] == "returns.window"
    assert documents[0]["version"] == "2026-09"
    assert documents[0]["metadata"]["provenance"] == "shopmind_curated_experimental"


def test_review_repository_filters_both_search_modes_by_exact_product_id():
    from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository

    class Client:
        def __init__(self):
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return {"hits": {"hits": [{"_id": "r1", "_score": 2.0, "_source": {
                "review_id": "r1", "review_title": "Good", "content": "Fits", "product_id": "P1",
                "text_vector": [1.0, 0.0],
            }}]}}

    client = Client()
    repo = ElasticsearchKnowledgeRepository(client, "reviews", source="review")
    assert repo.lexical_search("fits", 3, product_id="P1")[0].metadata["product_id"] == "P1"
    assert repo.vector_search([1.0, 0.0], 3, 3, product_id="P1")[0].document_id == "r1"
    assert client.calls[0]["query"]["bool"]["filter"] == [{"term": {"product_id": "P1"}}]
    assert client.calls[1]["knn"]["filter"] == {"term": {"product_id": "P1"}}


def test_selected_product_is_returned_even_when_query_terms_differ():
    from shopmind.app.knowledge.product_retriever import ProductDocumentRetriever

    class Search:
        def search_text(self, request):
            if request.query is None and request.filters == {"product_id": "P1"}:
                return [SearchResult("P1", "Red Shoe", 1, 0.0, "bm25", {"search_text": "Red Shoe\nSize 8"})]
            return []

        def to_retrieved_documents(self, results):
            return [RetrievedDocument(result.product_id, result.metadata["search_text"], result.score,
                                      "product", {"title": result.title}) for result in results]

    documents = ProductDocumentRetriever(Search()).retrieve("Is the sizing narrow?", 3, product_id="P1")
    assert [(document.id, document.content) for document in documents] == [("product:P1", "Red Shoe\nSize 8")]


def test_unknown_product_and_other_product_reviews_do_not_leak():
    from shopmind.app.knowledge.product_retriever import ProductDocumentRetriever
    from shopmind.app.knowledge.review_retriever import ReviewDocumentRetriever

    class Search:
        def search_text(self, request):
            return [SearchResult("P12", "Other", 1, 1.0, "bm25")]

        def to_retrieved_documents(self, results):
            return [RetrievedDocument(result.product_id, result.title, result.score, "product") for result in results]

    class Repo:
        def lexical_search(self, query, size, product_id=None):
            from shopmind.app.knowledge.base import KnowledgeHit
            return [KnowledgeHit("r1", "Good", 1.0, {"product_id": "P1", "content": "Fits"}),
                    KnowledgeHit("r2", "Other", 2.0, {"product_id": "P2", "content": "Leaks"})]

    assert ProductDocumentRetriever(Search()).retrieve("fit", 3, product_id="P1") == []
    reviews = ReviewDocumentRetriever(Repo(), None, mode="bm25").retrieve("fit", 3, product_id="P1")
    assert [review.id for review in reviews] == ["review:r1"]


def test_prepare_reviews_cli_keeps_only_exact_product_matches_and_omits_user_id(tmp_path):
    products = tmp_path / "products.jsonl"
    reviews = tmp_path / "reviews.jsonl.gz"
    output = tmp_path / "prepared.jsonl"
    products.write_text('{"product_id":"P1"}\n')
    with gzip.open(reviews, "wt", encoding="utf-8") as stream:
        stream.write('{"asin":"P12","title":"Other","text":"Wrong item"}\n')
        stream.write('{"asin":"P1","title":"Fits","text":"Comfortable","user_id":"private-123"}\n')

    subprocess.run([sys.executable, "-m", "scripts.prepare_reviews", "--products", str(products), "--reviews", str(reviews), "--output", str(output)], check=True)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["document_id"].startswith("review:")
    assert rows[0]["metadata"]["product_id"] == "P1"
    assert "private-123" not in output.read_text()


@pytest.mark.parametrize("output_name", ["products.jsonl", "reviews.jsonl"])
def test_prepare_reviews_rejects_output_overwriting_an_input(tmp_path, output_name):
    products = tmp_path / "products.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    products.write_text('{"product_id":"P1"}\n')
    reviews.write_text('{"asin":"P1","text":"Fits"}\n')
    before = {path: path.read_bytes() for path in (products, reviews)}

    result = subprocess.run(
        [sys.executable, "-m", "scripts.prepare_reviews", "--products", str(products), "--reviews", str(reviews), "--output", str(tmp_path / output_name)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--output" in result.stderr
    assert {path: path.read_bytes() for path in before} == before


def test_recreate_knowledge_index_rejects_an_active_alias():
    from shopmind.app.infrastructure.elasticsearch.knowledge_index import create_knowledge_index

    class Indices:
        def __init__(self):
            self.deleted = False

        def exists(self, *, index):
            return True

        def get(self, *, index):
            return {index: {"aliases": {"reviews": {}}}}

        def delete(self, *, index):
            self.deleted = True

        def create(self, *, index, **mapping):
            raise AssertionError("active index must not be recreated")

    client = type("Client", (), {"indices": Indices()})()
    with pytest.raises(ValueError, match="alias"):
        create_knowledge_index(client, "reviews_v1", "review", 4, recreate=True)
    assert not client.indices.deleted
