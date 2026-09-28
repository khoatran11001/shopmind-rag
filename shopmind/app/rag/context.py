from __future__ import annotations

import json

from shopmind.app.domain.search_result import RetrievedDocument
from shopmind.app.rag.models import RAGContext


class ContextBuilder:
    def __init__(self, max_documents: int, source_limits: dict[str, int], max_characters: int) -> None:
        if max_documents <= 0 or max_characters <= 0 or any(limit < 0 for limit in source_limits.values()):
            raise ValueError("context limits must be valid")
        self.max_documents = max_documents
        self.source_limits = dict(source_limits)
        self.max_characters = max_characters

    def build(self, documents: list[RetrievedDocument]) -> RAGContext:
        selected: list[RetrievedDocument] = []
        chunks: list[str] = []
        counts: dict[str, int] = {}
        used = 0
        for document in documents:
            if counts.get(document.source, 0) >= self.source_limits.get(document.source, self.max_documents):
                continue
            evidence = {"id": document.id, "source": document.source, "title": document.metadata.get("title", document.id), "content": document.content}
            if document.source == "policy":
                evidence.update({key: document.metadata[key] for key in ("version", "effective_date", "provenance") if key in document.metadata})
            chunk = json.dumps(evidence, ensure_ascii=False)
            size = len(chunk) + (1 if chunks else 0)
            if used + size > self.max_characters:
                continue
            chunks.append(chunk)
            selected.append(document)
            counts[document.source] = counts.get(document.source, 0) + 1
            used += size
            if len(selected) >= self.max_documents:
                break
        return RAGContext(tuple(selected), "\n".join(chunks), tuple(document.id for document in selected), {"source_counts": counts, "context_characters": used})
