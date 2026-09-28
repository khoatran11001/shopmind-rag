from __future__ import annotations

from dataclasses import replace

from shopmind.app.domain.search_query import SearchMode, SearchRequest


class ProductDocumentRetriever:
    def __init__(self, search_service, *, mode: str = "hybrid", candidate_k: int = 20) -> None:
        self.search_service = search_service
        self.mode = SearchMode(mode)
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int, product_id: str | None = None):
        if product_id is None:
            request = SearchRequest(query=query, mode=self.mode, top_k=top_k, candidate_k=max(top_k, self.candidate_k))
        else:
            request = SearchRequest(mode=SearchMode.BM25, top_k=1, candidate_k=max(1, self.candidate_k), filters={"product_id": product_id})
        results = self.search_service.search_text(request)
        if product_id is not None:
            results = [result for result in results if result.product_id == product_id]
        documents = self.search_service.to_retrieved_documents(results)
        return [replace(document, id=f"product:{document.id.removeprefix('product:')}", metadata={**document.metadata, "product_id": document.id.removeprefix("product:")}) for document in documents]
