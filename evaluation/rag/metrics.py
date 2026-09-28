from __future__ import annotations

from typing import Any

from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from evaluation.rag.datasets import RAGEvaluationCase


def score_case(case: RAGEvaluationCase, answer: Any, *, grader: Any = None) -> dict[str, float | None]:
    metadata = answer.metadata or {}
    relevant = set(case.relevant_document_ids)
    cited = [source.document_id for source in answer.sources]
    context = metadata.get("context_ids")
    retrieved = metadata.get("retrieved_ids")
    fused = metadata.get("fused_ids")
    ranked = metadata.get("reranked_ids", metadata.get("fused_ids"))
    relevance = {doc_id: 1 for doc_id in relevant}
    facts = [fact.casefold() for fact in case.acceptable_facts if fact]
    response = answer.answer.casefold()
    return {
        "candidate_recall": len(set(retrieved) & relevant) / len(relevant) if retrieved is not None and relevant else None,
        "retrieval_recall_at_10": recall_at_k(fused, relevance, 10) if fused is not None and relevant else None,
        "reranked_ndcg_at_10": ndcg_at_k(ranked, relevance, 10) if ranked is not None and relevant else None,
        "reranked_mrr_at_10": mrr_at_k(ranked, relevance, 10) if ranked is not None and relevant else None,
        "citation_validity": sum(doc_id in set(context) for doc_id in cited) / len(cited) if context is not None and cited else (1.0 if context is not None else None),
        "citation_precision": sum(doc_id in relevant for doc_id in cited) / len(cited) if cited else (1.0 if not relevant else 0.0),
        "citation_recall": len(set(cited) & relevant) / len(relevant) if relevant else 1.0,
        "citation_coverage": float(bool(cited)) if not case.expect_insufficient_evidence else None,
        "abstention_accuracy": float(bool(answer.insufficient_evidence) == case.expect_insufficient_evidence),
        "acceptable_fact_coverage": sum(fact in response for fact in facts) / len(facts) if facts else None,
        "answer_correctness": grader(case, answer) if grader is not None else None,
    }


def classify_failure(case: RAGEvaluationCase, metrics: dict[str, float | None]) -> str | None:
    if case.relevant_document_ids and metrics.get("candidate_recall") == 0:
        return "candidate_miss"
    if case.relevant_document_ids and metrics.get("retrieval_recall_at_10") == 0:
        return "fusion_miss"
    if metrics.get("citation_validity") is not None and metrics["citation_validity"] < 1:
        return "invalid_citation"
    if case.relevant_document_ids and metrics.get("citation_recall") is not None and metrics["citation_recall"] < 1:
        return "citation_miss"
    if metrics.get("abstention_accuracy") == 0:
        return "wrong_abstention"
    return None
