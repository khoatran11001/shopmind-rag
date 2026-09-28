from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


QUESTION_CLASSES = {"product_factual", "review_based", "policy", "multi_source", "unanswerable"}
SOURCES = {"product", "review", "policy"}


@dataclass(frozen=True)
class RAGEvaluationCase:
    question_id: str
    question: str
    question_class: str
    product_id: str | None
    expected_sources: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    acceptable_facts: tuple[str, ...]
    expect_insufficient_evidence: bool
    label_status: str = "seed_unreviewed"


def load_rag_cases(path: str | Path) -> list[RAGEvaluationCase]:
    rows: list[RAGEvaluationCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        question_id = str(raw.get("question_id") or "").strip()
        question = str(raw.get("question") or "").strip()
        question_class = raw.get("question_class")
        if not question_id or not question:
            raise ValueError(f"invalid RAG evaluation case at line {line_number}")
        if question_id in seen:
            raise ValueError(f"duplicate question_id: {question_id}")
        if question_class not in QUESTION_CLASSES:
            raise ValueError(f"unsupported question_class: {question_class}")
        expected_sources = tuple(raw.get("expected_sources") or ())
        if any(source not in SOURCES for source in expected_sources):
            raise ValueError(f"invalid expected_sources at line {line_number}")
        relevant_ids = tuple(raw.get("relevant_document_ids") or ())
        if any(not isinstance(doc_id, str) or not doc_id.startswith(tuple(f"{source}:" for source in SOURCES)) for doc_id in relevant_ids):
            raise ValueError("relevant_document_ids must be namespaced")
        insufficient = raw.get("expect_insufficient_evidence", False)
        if not isinstance(insufficient, bool):
            raise ValueError(f"expect_insufficient_evidence must be boolean at line {line_number}")
        seen.add(question_id)
        rows.append(RAGEvaluationCase(
            question_id=question_id,
            question=question,
            question_class=question_class,
            product_id=str(raw["product_id"]).strip() if raw.get("product_id") else None,
            expected_sources=expected_sources,
            relevant_document_ids=relevant_ids,
            acceptable_facts=tuple(str(fact) for fact in raw.get("acceptable_facts") or ()),
            expect_insufficient_evidence=insufficient,
            label_status=str(raw.get("label_status") or "seed_unreviewed"),
        ))
    if not rows:
        raise ValueError("RAG evaluation set is empty")
    return rows
