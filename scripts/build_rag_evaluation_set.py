from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.rag.datasets import load_rag_cases


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rag_evaluation_set(products_path: str | Path, reviews_path: str | Path, policies_path: str | Path, output_path: str | Path) -> Path:
    products = {str(row["product_id"]): row for row in _rows(products_path)}
    review = next((row for row in _rows(reviews_path) if str(row.get("product_id") or row.get("metadata", {}).get("product_id")) in products), None)
    if review is None:
        raise ValueError("no review matches a product in the catalog")
    product_id = str(review.get("product_id") or review["metadata"]["product_id"])
    product = products[product_id]
    policies = _rows(policies_path)
    if not policies:
        raise ValueError("policy corpus is empty")
    policy = next((row for row in policies if row.get("section_id") == "returns.window"), policies[0])
    review_id = str(review.get("document_id") or f"review:{review.get('review_id') or review.get('metadata', {}).get('review_id')}")
    policy_id = str(policy.get("document_id") or f"policy:{policy['section_id']}")
    if review_id.endswith(":None"):
        raise ValueError("matched review lacks an ID")
    title = str(product.get("title") or product_id)
    policy_content = str(policy["content"])
    cases = [
        {"question_id": "product-1", "question": f"What is the title of product {product_id}?", "question_class": "product_factual", "product_id": None, "expected_sources": ["product"], "relevant_document_ids": [f"product:{product_id}"], "acceptable_facts": [title], "expect_insufficient_evidence": False},
        {"question_id": "review-1", "question": "What does a customer review say about this product?", "question_class": "review_based", "product_id": product_id, "expected_sources": ["review"], "relevant_document_ids": [review_id], "acceptable_facts": [], "expect_insufficient_evidence": False},
        {"question_id": "policy-1", "question": f"What does the experimental {policy.get('policy_type') or policy.get('policy_id')} policy say?", "question_class": "policy", "product_id": None, "expected_sources": ["policy"], "relevant_document_ids": [policy_id], "acceptable_facts": [policy_content], "expect_insufficient_evidence": False},
        {"question_id": "multi-1", "question": f"What is this product called and what does the experimental policy say?", "question_class": "multi_source", "product_id": product_id, "expected_sources": ["product", "policy"], "relevant_document_ids": [f"product:{product_id}", policy_id], "acceptable_facts": [title, policy_content], "expect_insufficient_evidence": False},
        {"question_id": "unknown-1", "question": "Does the evidence guarantee a ten year battery life?", "question_class": "unanswerable", "product_id": product_id, "expected_sources": [], "relevant_document_ids": [], "acceptable_facts": [], "expect_insufficient_evidence": True},
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps({**case, "label_status": "seed_unreviewed"}, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    load_rag_cases(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=Path, default=Path("data/a1_abo_1500/products.jsonl"))
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--policies", type=Path, default=Path("data/policies/shopmind_policies.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/rag_cases.jsonl"))
    args = parser.parse_args()
    print(build_rag_evaluation_set(args.products, args.reviews, args.policies, args.output))


if __name__ == "__main__":
    main()
