from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request

from shopmind.app.api.schemas.rag import RAGAskRequest, RAGAskResponse, RAGSourceResponse
from shopmind.app.rag.models import RAGRequest

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


class RAGUnavailable(RuntimeError):
    pass


@router.post("/ask", response_model=RAGAskResponse)
def ask(payload: RAGAskRequest, request: Request) -> RAGAskResponse:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise RAGUnavailable("RAG service is not ready")
    answer = service.answer(RAGRequest(payload.question, product_id=payload.product_id, sources=tuple(payload.sources), debug=payload.debug))
    return RAGAskResponse(
        answer=answer.answer,
        sources=[RAGSourceResponse.model_validate(asdict(source)) for source in answer.sources],
        insufficient_evidence=answer.insufficient_evidence,
        metadata=answer.metadata if payload.debug else {},
    )
