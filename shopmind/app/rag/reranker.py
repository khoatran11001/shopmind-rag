from __future__ import annotations

from dataclasses import replace

from shopmind.app.domain.search_result import RetrievedDocument


class NoOpDocumentReranker:
    def rerank(self, query: str, documents: list[RetrievedDocument], top_k: int) -> list[RetrievedDocument]:
        del query
        return documents[:top_k]


class CrossEncoderDocumentReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model=None) -> None:
        self.model_name = model_name
        self._model = model

    def rerank(self, query: str, documents: list[RetrievedDocument], top_k: int) -> list[RetrievedDocument]:
        if not documents:
            return []
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("install the rag-rerank extra to enable the cross-encoder") from exc
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, document.content) for document in documents])
        scored = sorted(zip(documents, scores, strict=True), key=lambda item: (-float(item[1]), item[0].id))
        return [replace(document, score=float(score), metadata={**document.metadata, "pre_rerank_score": document.score, "score_type": "reranker"}) for document, score in scored[:top_k]]
