from __future__ import annotations

from shopmind.app.domain.search_result import RetrievedDocument
from shopmind.app.knowledge.base import KnowledgeMode, rrf_knowledge_hits


class ReviewDocumentRetriever:
    def __init__(self, repository, embedder, *, mode: str = "hybrid", candidate_k: int = 40, rrf_k: int = 60) -> None:
        self.repository = repository
        self.embedder = embedder
        self.mode = KnowledgeMode(mode)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int, product_id: str | None = None) -> list[RetrievedDocument]:
        lexical = self.repository.lexical_search(query, self.candidate_k, product_id=product_id) if self.mode in (KnowledgeMode.BM25, KnowledgeMode.HYBRID) else []
        dense = []
        if self.mode in (KnowledgeMode.DENSE, KnowledgeMode.HYBRID):
            vector = self.embedder.embed_texts([query])[0].astype("float32").tolist()
            dense = self.repository.vector_search(vector, self.candidate_k, self.candidate_k, product_id=product_id)
        if product_id is not None:
            lexical = [hit for hit in lexical if hit.metadata.get("product_id") == product_id]
            dense = [hit for hit in dense if hit.metadata.get("product_id") == product_id]
        if self.mode == KnowledgeMode.HYBRID:
            ranked = rrf_knowledge_hits([lexical, dense], k=self.rrf_k, top_k=self.candidate_k)
        else:
            ranked = [(hit, hit.score) for hit in (lexical if self.mode == KnowledgeMode.BM25 else dense)]
        return [RetrievedDocument(f"review:{hit.document_id}", str(hit.metadata.get("content") or hit.title), float(score), "review", {**hit.metadata, "title": hit.title}) for hit, score in ranked[:top_k]]
