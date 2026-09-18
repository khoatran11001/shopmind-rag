from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FAILURE_CATEGORIES = {
    "lexical_mismatch",
    "semantic_confusion",
    "category_confusion",
    "visual_wrong_product_type",
    "missing_metadata",
    "poor_image_quality",
    "ambiguous_query",
    "uncategorized",
}


def categorize_failure(query: str, result_metadata: dict[str, Any]) -> str:
    """Suggest a failure category only when explicit evidence is present.

    The query text is accepted for future deterministic rules, but this A1
    implementation deliberately avoids guessing subjective causes from text
    alone. Explicit metadata/evidence flags take precedence.
    """
    del query

    missing_fields = result_metadata.get("missing_fields")
    if isinstance(missing_fields, list) and any(str(value).strip() for value in missing_fields):
        return "missing_metadata"
    if result_metadata.get("poor_image_quality") is True:
        return "poor_image_quality"
    if result_metadata.get("visual_product_type_mismatch") is True:
        return "visual_wrong_product_type"

    expected_category = result_metadata.get("expected_category")
    retrieved_category = result_metadata.get("retrieved_category")
    if (
        expected_category is not None
        and retrieved_category is not None
        and str(expected_category).strip()
        and str(retrieved_category).strip()
        and str(expected_category) != str(retrieved_category)
    ):
        return "category_confusion"

    lexical_overlap = result_metadata.get("lexical_overlap")
    expected_terms = result_metadata.get("expected_terms")
    if lexical_overlap == 0 and isinstance(expected_terms, list) and expected_terms:
        return "lexical_mismatch"
    if result_metadata.get("semantic_confusion") is True:
        return "semantic_confusion"
    if result_metadata.get("ambiguous_query") is True:
        return "ambiguous_query"
    return "uncategorized"


def export_error_cases(path: str | Path, cases: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            metadata = dict(case.get("result_metadata") or {})
            suggested = categorize_failure(str(case.get("query") or ""), metadata)
            if suggested not in _FAILURE_CATEGORIES:
                suggested = "uncategorized"
            row = {
                "query_id": str(case.get("query_id") or ""),
                "query_type": str(case.get("query_type") or "text"),
                "query": str(case.get("query") or ""),
                "relevant_ids": list(case.get("relevant_ids") or []),
                "top_retrieved_ids": list(case.get("top_retrieved_ids") or []),
                "metrics": dict(case.get("metrics") or {}),
                "suggested_category": suggested,
                "reviewer_note": "",
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
