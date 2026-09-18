from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shopmind.app.domain.search_result import SearchResult
from shopmind.app.main import create_app


class FakeSearchService:
    def __init__(self, results=None):
        self.results = results if results is not None else [
            SearchResult(product_id="P1", title="Black Runner", rank=1, score=0.031, source="hybrid", metadata={"image_url": "https://example.test/p1.jpg", "brand": "Acme", "category": "Shoes", "search_text": "Black Runner"})
        ]
        self.text_requests = []
        self.image_requests = []

    def search_text(self, request):
        self.text_requests.append(request)
        return list(self.results)

    def search_image(self, image, top_k=10, candidate_k=100, filters=None):
        self.image_requests.append((image, top_k, candidate_k, filters))
        return list(self.results)


class FakeRepository:
    def __init__(self, ready=True, alias=True):
        self.ready = ready
        self.alias = alias

    def filter_options(self, field, prefix="", limit=20):
        del prefix, limit
        if field not in {"brand", "category", "product_id"}:
            raise ValueError(f"unsupported filter field: {field}")
        return [{"value": "Acme", "count": 2}] if field == "brand" else []

    def ping(self):
        return self.ready

    def alias_exists(self, alias):
        return self.alias


class FakeEmbedder:
    model_name = "fake"
    model_revision = "test"

    def __init__(self, ready=True):
        self.is_ready = ready


@pytest.fixture
def fake_service():
    return FakeSearchService()


@pytest.fixture
def client(fake_service):
    app = create_app(search_service=fake_service, repository=FakeRepository(), embedder=FakeEmbedder(), index_alias="products", max_image_bytes=1024 * 1024, auto_wire=False)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_with_unready_repo(fake_service):
    app = create_app(search_service=fake_service, repository=FakeRepository(ready=False), embedder=FakeEmbedder(), index_alias="products", auto_wire=False)
    with TestClient(app) as test_client:
        yield test_client
