from __future__ import annotations

import re

from shopmind.app.domain.search_result import RetrievedDocument
from shopmind.app.rag.models import Citation


class CitationValidationError(ValueError):
    pass


class CitationValidator:
    def validate(self, citation_ids: tuple[str, ...], allowed_ids: set[str], *, answer: str, insufficient_evidence: bool) -> tuple[str, ...]:
        if any(citation_id not in allowed_ids for citation_id in citation_ids):
            raise CitationValidationError("citation is outside the supplied evidence")
        unique = tuple(dict.fromkeys(citation_ids))
        inline_ids = set(re.findall(r"\[((?:product|review|policy):[^\]\s]+)\]", answer))
        if inline_ids - set(unique):
            raise CitationValidationError("inline citation is missing from the validated source list")
        if not insufficient_evidence and answer.strip() and not unique:
            raise CitationValidationError("answer requires a citation")
        return unique


class CitationResolver:
    def resolve(self, citation_ids: tuple[str, ...], documents: tuple[RetrievedDocument, ...]) -> tuple[Citation, ...]:
        by_id = {document.id: document for document in documents}
        output: list[Citation] = []
        for citation_id in citation_ids:
            document = by_id[citation_id]
            details = document.metadata
            output.append(Citation(
                document_id=document.id,
                source=document.source,
                title=str(details.get("title") or document.id),
                score=document.score,
                product_id=details.get("product_id"),
                policy_version=details.get("version"),
                source_url=details.get("source_url"),
                metadata={key: details[key] for key in ("rating", "verified_purchase", "helpful_vote", "matched_by", "effective_date", "policy_type", "provenance", "source_score", "score_type", "pre_rerank_score") if key in details},
            ))
        return tuple(output)
