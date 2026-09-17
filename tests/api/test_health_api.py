from fastapi.testclient import TestClient

from shopmind.app.main import create_app
from tests.api.conftest import FakeEmbedder, FakeRepository, FakeSearchService


def test_health_is_liveness_only(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_fails_when_elasticsearch_is_down(client_with_unready_repo):
    response = client_with_unready_repo.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "search_infrastructure_unavailable"


def test_ready_succeeds_when_repository_alias_and_embedding_are_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def _media_client(image_dir):
    app = create_app(search_service=FakeSearchService(), repository=FakeRepository(), embedder=FakeEmbedder(), product_image_dir=image_dir, auto_wire=False)
    return TestClient(app)


def test_product_media_serves_existing_image(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "shoe.jpg").write_bytes(b"fixture-image")

    response = _media_client(image_dir).get("/media/products/shoe.jpg")

    assert response.status_code == 200
    assert response.content == b"fixture-image"


def test_product_media_returns_404_for_missing_image(tmp_path):
    response = _media_client(tmp_path).get("/media/products/missing.jpg")
    assert response.status_code == 404


def test_product_media_rejects_path_traversal(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (tmp_path / "outside.txt").write_text("private")

    response = _media_client(image_dir).get("/media/products/%2e%2e/outside.txt")

    assert response.status_code == 404
