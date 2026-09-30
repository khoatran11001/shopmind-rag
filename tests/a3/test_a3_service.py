from __future__ import annotations

from dataclasses import dataclass

import pytest

from shopmind.app.domain.search_result import RetrievedDocument
from shopmind.app.rag.citations import CitationResolver, CitationValidator
from shopmind.app.rag.context import ContextBuilder
from shopmind.app.rag.fusion import rrf_fuse_documents
from shopmind.app.rag.models import RAGRequest
from shopmind.app.rag.reranker import NoOpDocumentReranker
from shopmind.app.rag.service import RAGGenerationError, RAGService
from shopmind.app.llm.base import LLMProviderError


@dataclass
class Retriever:
    source: str
    documents: list[RetrievedDocument]

    def __post_init__(self):
        self.calls = []

    def retrieve(self, query, top_k, product_id=None):
        self.calls.append((query, top_k, product_id))
        return self.documents[:top_k]


class LLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return next(self.responses)


def _service(retrievers, llm):
    from shopmind.app.knowledge.multi_source import MultiSourceRetriever

    return RAGService(
        MultiSourceRetriever(retrievers, {name: 3 for name in retrievers}),
        rrf_fuse_documents,
        NoOpDocumentReranker(),
        ContextBuilder(3, {"product": 1, "review": 1, "policy": 1}, 1000),
        llm,
        CitationValidator(),
        CitationResolver(),
        max_generation_attempts=2,
    )


def test_product_id_reaches_each_source_and_citation_score_is_server_resolved():
    from shopmind.app.llm.base import LLMResponse

    product = RetrievedDocument("product:P1", "Blue shoes", 99.0, "product", {"title": "Blue shoes", "product_id": "P1"})
    review = RetrievedDocument("review:R1", "Fits well", 0.4, "review", {"title": "Fits well", "product_id": "P1"})
    retrievers = {"product": Retriever("product", [product]), "review": Retriever("review", [review])}
    llm = LLM([LLMResponse("Customers say it fits well.", ("review:R1",), False)])

    answer = _service(retrievers, llm).answer(RAGRequest("Does it fit?", product_id="P1", sources=("product", "review")))

    assert retrievers["product"].calls == [("Does it fit?", 3, "P1")]
    assert retrievers["review"].calls == [("Does it fit?", 3, "P1")]
    assert answer.answer == "Customers say it fits well."
    assert [(source.document_id, source.source, source.product_id) for source in answer.sources] == [("review:R1", "review", "P1")]
    assert answer.sources[0].score > 0
    assert answer.sources[0].score != review.score
    assert answer.sources[0].metadata["score_type"] == "rrf"
    assert answer.sources[0].metadata["source_score"] == 0.4
    assert llm.calls[0].allowed_citation_ids == ("product:P1", "review:R1")


def test_invalid_citation_is_repaired_once_and_never_exposed():
    from shopmind.app.llm.base import LLMResponse

    product = RetrievedDocument("product:P1", "Blue shoes", 1.0, "product", {"title": "Blue shoes", "product_id": "P1"})
    llm = LLM([
        LLMResponse("Available in blue.", ("product:made-up",), False),
        LLMResponse("Available in blue.", ("product:P1",), False),
    ])

    answer = _service({"product": Retriever("product", [product])}, llm).answer(RAGRequest("What color?", product_id="P1", sources=("product",)))

    assert [source.document_id for source in answer.sources] == ["product:P1"]
    assert answer.metadata["generation_attempts"] == 2
    assert "product:made-up" not in answer.answer


def test_repeated_invalid_citations_fail_after_bounded_repair():
    from shopmind.app.llm.base import LLMResponse

    product = RetrievedDocument("product:P1", "Blue shoes", 1.0, "product", {"title": "Blue shoes"})
    llm = LLM([LLMResponse("Blue.", ("product:fake",), False)] * 2)

    with pytest.raises(RAGGenerationError):
        _service({"product": Retriever("product", [product])}, llm).answer(RAGRequest("Color?", sources=("product",)))
    assert len(llm.calls) == 2


def test_empty_evidence_abstains_without_calling_llm():
    llm = LLM([])

    answer = _service({"product": Retriever("product", [])}, llm).answer(RAGRequest("Unknown?", sources=("product",)))

    assert answer.insufficient_evidence is True
    assert answer.sources == ()
    assert llm.calls == []


def test_missing_llm_fails_even_when_retrieval_has_no_evidence():
    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
        _service({"product": Retriever("product", [])}, None).answer(RAGRequest("Unknown?", sources=("product",)))


def test_retrieval_trace_runs_without_an_llm_for_ablation():
    from shopmind.app.knowledge.base import KnowledgeMode

    product = RetrievedDocument("product:P1", "Blue shoes", 1.0, "product", {"title": "Blue shoes"})
    retriever = Retriever("product", [product])
    retriever.mode = KnowledgeMode.ADAPTIVE
    service = _service({"product": retriever}, None)

    trace = service.retrieve(RAGRequest("Blue shoes?", sources=("product",)))

    assert trace["retrieved_ids"] == ["product:P1"]
    assert trace["fused_ids"] == ["product:P1"]
    assert trace["reranked_ids"] == ["product:P1"]
    assert trace["retrieval_modes"] == {"product": "dense"}


def test_context_budget_and_source_caps_limit_llm_citations():
    documents = [
        RetrievedDocument("review:R1", "first review", 1, "review", {"title": "First"}),
        RetrievedDocument("review:R2", "second review", 1, "review", {"title": "Second"}),
        RetrievedDocument("policy:P1", "return window", 1, "policy", {"title": "Returns"}),
    ]

    context = ContextBuilder(3, {"review": 1, "policy": 1}, 200).build(documents)

    assert context.citation_ids == ("review:R1", "policy:P1")
    assert "review:R2" not in context.rendered_context


def test_policy_provenance_and_version_reach_context_and_citation():
    policy = RetrievedDocument("policy:returns.window", "Experimental returns within 30 days.", 0.02, "policy", {"title": "Return Window", "version": "2026-09", "provenance": "shopmind_curated_experimental"})

    context = ContextBuilder(1, {"policy": 1}, 1000).build([policy])
    source = CitationResolver().resolve((policy.id,), context.documents)[0]

    assert "shopmind_curated_experimental" in context.rendered_context
    assert source.policy_version == "2026-09"
    assert source.metadata["provenance"] == "shopmind_curated_experimental"


def test_request_rejects_blank_question_or_product_id():
    with pytest.raises(ValueError):
        RAGRequest("   ")
    with pytest.raises(ValueError):
        RAGRequest("Hi", product_id=" ")


def test_answer_text_cannot_smuggle_an_unknown_inline_citation():
    from shopmind.app.rag.citations import CitationValidationError

    with pytest.raises(CitationValidationError):
        CitationValidator().validate(("product:P1",), {"product:P1"}, answer="See [review:fake].", insufficient_evidence=False)


def test_openai_provider_requests_structured_output_and_validates_response():
    import json
    from types import SimpleNamespace

    from shopmind.app.llm.base import InvalidLLMOutput, LLMRequest
    from shopmind.app.llm.openai_provider import OpenAIProvider

    class Responses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(output_text=json.dumps({"answer": "Blue.", "citation_ids": ["product:P1"], "insufficient_evidence": False}), model="test-model", id="r1")

    client = SimpleNamespace(responses=Responses())
    provider = OpenAIProvider(client=client, model="test-model")
    request = LLMRequest("Color?", '{"id":"product:P1"}', ("product:P1",), "Use evidence only")

    response = provider.generate(request)

    assert response.citation_ids == ("product:P1",)
    assert client.responses.kwargs["store"] is False
    assert client.responses.kwargs["text"]["format"]["type"] == "json_schema"
    assert "product:P1" in str(client.responses.kwargs["input"])

    client.responses.create = lambda **kwargs: SimpleNamespace(output_text='{"answer":"Blue.","citation_ids":[7],"insufficient_evidence":false}')
    with pytest.raises(InvalidLLMOutput):
        provider.generate(request)
