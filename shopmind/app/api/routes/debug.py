from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from shopmind.app.api.routes.search import get_search_service
from shopmind.app.api.schemas.search import DebugSearchResponse, DebugSearchResult, TextSearchRequest
from shopmind.app.search.service import SearchService

router = APIRouter(prefix="/api/v1/debug", tags=["debug"])


@router.post("/search", response_model=DebugSearchResponse)
def debug_search(payload: TextSearchRequest, request: Request, service: SearchService = Depends(get_search_service)) -> DebugSearchResponse:
    domain_request = payload.to_domain()
    results = service.search_text(domain_request)
    rrf_k = request.app.state.fusion_rrf_k
    return DebugSearchResponse(
        query=domain_request.query, mode=domain_request.mode.value, filter_only=domain_request.query is None,
        rrf_k=rrf_k, results=[DebugSearchResult.from_domain(result, rrf_k=rrf_k) for result in results],
    )
