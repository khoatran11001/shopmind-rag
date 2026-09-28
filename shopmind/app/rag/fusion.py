from __future__ import annotations

from dataclasses import replace

from shopmind.app.domain.search_result import RetrievedDocument


def rrf_fuse_documents(rankings: list[list[RetrievedDocument]], *, k: int = 60, top_k: int = 40) -> list[RetrievedDocument]:
    if k <= 0 or top_k <= 0:
        raise ValueError("k and top_k must be positive")
    scores: dict[str, float] = {}
    first: dict[str, RetrievedDocument] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking, 1):
            first.setdefault(document.id, document)
            scores[document.id] = scores.get(document.id, 0.0) + 1 / (k + rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    return [replace(first[doc_id], score=scores[doc_id], metadata={**first[doc_id].metadata, "source_score": first[doc_id].score, "score_type": "rrf"}) for doc_id in ordered]


def concatenate_documents(rankings: list[list[RetrievedDocument]], *, k: int = 60, top_k: int = 40) -> list[RetrievedDocument]:
    del k
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    seen: set[str] = set()
    output: list[RetrievedDocument] = []
    for ranking in rankings:
        for document in ranking:
            if document.id not in seen:
                seen.add(document.id)
                output.append(replace(document, metadata={**document.metadata, "score_type": "source"}))
            if len(output) >= top_k:
                return output
    return output
