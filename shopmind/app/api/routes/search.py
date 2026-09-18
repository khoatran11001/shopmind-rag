from __future__ import annotations

import logging
from io import BytesIO
from time import perf_counter

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from shopmind.app.api.errors import ImageTooLarge, InvalidFilter, InvalidImage, SearchInfrastructureError
from shopmind.app.api.schemas.search import FilterOptionsResponse, ImageSearchResponse, ProductSearchResult, TextSearchRequest, TextSearchResponse, normalize_filters
from shopmind.app.core.logging import log_search_event
from shopmind.app.search.service import SearchService

router = APIRouter(prefix="/api/v1/search", tags=["search"])
logger = logging.getLogger("shopmind.search")


def get_search_service(request: Request) -> SearchService:
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise SearchInfrastructureError("search service is not ready")
    return service


def get_repository(request: Request):
    repository = getattr(request.app.state, "repository", None)
    if repository is None or not hasattr(repository, "filter_options"):
        raise SearchInfrastructureError("search repository is not ready")
    return repository


def _collapse_singleton_filters(filters: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
    return {key: value[0] if isinstance(value, list) and len(value) == 1 else value for key, value in filters.items()}


@router.post("/text", response_model=TextSearchResponse)
def search_text(payload: TextSearchRequest, request: Request, service: SearchService = Depends(get_search_service)) -> TextSearchResponse:
    started = perf_counter()
    domain_request = payload.to_domain()
    results = service.search_text(domain_request)
    log_search_event(logger, request_id=request.state.request_id, query=domain_request.query, mode=domain_request.mode.value, latency_ms=round((perf_counter() - started) * 1000, 3), candidate_k=domain_request.candidate_k, result_count=len(results), index_alias=getattr(request.app.state, "index_alias", None), embedding_model=getattr(request.app.state, "embedding_model", None), embedding_version=getattr(request.app.state, "embedding_version", None))
    return TextSearchResponse(query=domain_request.query, mode=domain_request.mode.value, filter_only=domain_request.query is None, results=[ProductSearchResult.from_domain(result) for result in results])


@router.get("/filter-options", response_model=FilterOptionsResponse)
def filter_options(field: str = Query(...), prefix: str = Query(default="", max_length=64), limit: int = Query(default=20, ge=1, le=50), repository=Depends(get_repository)) -> FilterOptionsResponse:
    try:
        options = repository.filter_options(field=field, prefix=prefix.strip(), limit=limit)
    except ValueError as exc:
        raise InvalidFilter(str(exc)) from exc
    return FilterOptionsResponse(field=field, options=options)


@router.post("/image", response_model=ImageSearchResponse)
async def search_image(request: Request, image: UploadFile = File(...), top_k: int = Query(default=10, gt=0), candidate_k: int = Query(default=100, gt=0), brand: list[str] | None = Query(default=None), category: list[str] | None = Query(default=None), product_id: str | None = Query(default=None), service: SearchService = Depends(get_search_service)) -> ImageSearchResponse:
    if candidate_k < top_k:
        raise InvalidImage("candidate_k must be >= top_k")
    if not image.content_type or not image.content_type.lower().startswith("image/"):
        raise InvalidImage("uploaded file must have an image content type")
    max_bytes = int(getattr(request.app.state, "max_image_bytes", 5 * 1024 * 1024))
    raw = await image.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ImageTooLarge(f"image exceeds configured maximum of {max_bytes} bytes")
    try:
        with Image.open(BytesIO(raw)) as decoded:
            rgb = decoded.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage("uploaded bytes are not a valid image") from exc
    filters = _collapse_singleton_filters(normalize_filters({"brand": brand, "category": category, "product_id": product_id}))
    started = perf_counter()
    results = service.search_image(rgb, top_k=top_k, candidate_k=candidate_k, filters=filters)
    log_search_event(logger, request_id=request.state.request_id, mode="image_dense", latency_ms=round((perf_counter() - started) * 1000, 3), candidate_k=candidate_k, result_count=len(results), index_alias=getattr(request.app.state, "index_alias", None), embedding_model=getattr(request.app.state, "embedding_model", None), embedding_version=getattr(request.app.state, "embedding_version", None))
    return ImageSearchResponse(results=[ProductSearchResult.from_domain(result) for result in results])
