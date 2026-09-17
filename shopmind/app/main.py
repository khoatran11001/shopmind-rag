from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from shopmind.app.api.errors import EmbeddingUnavailable, ImageTooLarge, InvalidFilter, InvalidImage, SearchInfrastructureError, UnsupportedSearchMode
from shopmind.app.api.routes.health import router as health_router
from shopmind.app.api.routes.search import router as search_router

logger = logging.getLogger("shopmind")


def _error_body(request: Request, code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}, "request_id": getattr(request.state, "request_id", "unknown")}


def _wire_runtime(app: FastAPI) -> None:
    from shopmind.app.core.config import load_app_config
    from shopmind.app.embedding.siglip2 import SigLIP2EmbeddingProvider
    from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
    from shopmind.app.infrastructure.elasticsearch.repository import ElasticsearchProductRepository
    from shopmind.app.retrieval.bm25 import BM25Retriever
    from shopmind.app.retrieval.dense import CrossModalRetriever, ImageDenseRetriever, TextDenseRetriever
    from shopmind.app.retrieval.hybrid import HybridRetriever
    from shopmind.app.retrieval.reranker import NoOpReranker
    from shopmind.app.search.service import SearchService

    config_path = Path(os.getenv("SHOPMIND_CONFIG", "configs/app.yaml"))
    config = load_app_config(config_path)
    client = create_elasticsearch_client(config.elasticsearch)
    repository = ElasticsearchProductRepository(client, config.elasticsearch.index_alias)
    embedder = SigLIP2EmbeddingProvider(model_name=config.embedding.model_name, model_revision=None, hf_token=config.embedding.hf_token)
    bm25 = BM25Retriever(repository)
    dense = TextDenseRetriever(repository, embedder)
    cross_modal = CrossModalRetriever(repository, embedder)
    hybrid = HybridRetriever(bm25, dense, NoOpReranker(), rrf_k=config.fusion.rrf_k)
    image = ImageDenseRetriever(repository, embedder)
    app.state.search_service = SearchService(bm25=bm25, dense=dense, hybrid=hybrid, cross_modal=cross_modal, image=image)
    app.state.repository = repository
    app.state.embedder = embedder
    app.state.index_alias = config.elasticsearch.index_alias
    app.state.embedding_model = embedder.model_name
    app.state.embedding_version = embedder.model_revision
    app.state.max_image_bytes = config.api.max_image_bytes
    app.state.runtime_error = None


def create_app(*, search_service: Any | None = None, repository: Any | None = None, embedder: Any | None = None, index_alias: str = "products", max_image_bytes: int = 5 * 1024 * 1024, product_image_dir: str | Path | None = None, auto_wire: bool = True) -> FastAPI:
    if max_image_bytes <= 0:
        raise ValueError("max_image_bytes must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if auto_wire and app.state.search_service is None:
            try:
                _wire_runtime(app)
            except Exception as exc:
                logger.exception("runtime initialization failed")
                app.state.runtime_error = exc
        yield

    app = FastAPI(title="ShopMind Multimodal Retrieval", version="0.1.0", lifespan=lifespan)
    app.state.search_service = search_service
    app.state.repository = repository
    app.state.embedder = embedder
    app.state.index_alias = index_alias
    app.state.max_image_bytes = max_image_bytes
    app.state.embedding_model = getattr(embedder, "model_name", None)
    app.state.embedding_version = getattr(embedder, "model_revision", None)
    app.state.runtime_error = None
    app.mount("/media/products", StaticFiles(directory=product_image_dir or os.getenv("PRODUCT_IMAGE_DIR", "data/raw/images"), check_dir=False), name="product-images")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        del exc
        return JSONResponse(status_code=422, content=_error_body(request, "validation_error", "request validation failed"))

    @app.exception_handler(UnsupportedSearchMode)
    async def unsupported_mode_handler(request: Request, exc: UnsupportedSearchMode):
        return JSONResponse(status_code=400, content=_error_body(request, "unsupported_search_mode", str(exc)))

    @app.exception_handler(InvalidFilter)
    async def invalid_filter_handler(request: Request, exc: InvalidFilter):
        return JSONResponse(status_code=400, content=_error_body(request, "invalid_filter", str(exc)))

    @app.exception_handler(InvalidImage)
    async def invalid_image_handler(request: Request, exc: InvalidImage):
        return JSONResponse(status_code=400, content=_error_body(request, "invalid_image", str(exc)))

    @app.exception_handler(ImageTooLarge)
    async def image_too_large_handler(request: Request, exc: ImageTooLarge):
        return JSONResponse(status_code=413, content=_error_body(request, "image_too_large", str(exc)))

    @app.exception_handler(EmbeddingUnavailable)
    async def embedding_unavailable_handler(request: Request, exc: EmbeddingUnavailable):
        return JSONResponse(status_code=503, content=_error_body(request, "embedding_unavailable", str(exc)))

    @app.exception_handler(SearchInfrastructureError)
    async def infrastructure_error_handler(request: Request, exc: SearchInfrastructureError):
        return JSONResponse(status_code=503, content=_error_body(request, "search_infrastructure_unavailable", str(exc)))

    app.include_router(health_router)
    app.include_router(search_router)
    return app


app = create_app(auto_wire=True)
