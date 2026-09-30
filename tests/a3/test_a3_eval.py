import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _case(**overrides):
    from evaluation.rag.datasets import RAGEvaluationCase

    values = {
        "question_id": "q1",
        "question": "What does this product review say?",
        "question_class": "review_based",
        "product_id": "p1",
        "expected_sources": ("review",),
        "relevant_document_ids": ("review:r1",),
        "acceptable_facts": ("comfortable",),
        "expect_insufficient_evidence": False,
    }
    return RAGEvaluationCase(**(values | overrides))


def _answer(ids=("review:r1",), *, context_ids=None, retrieved_ids=None, insufficient=False):
    context_ids = list(ids if context_ids is None else context_ids)
    retrieved_ids = list(ids if retrieved_ids is None else retrieved_ids)
    return SimpleNamespace(
        answer="Comfortable to wear.",
        sources=tuple(SimpleNamespace(document_id=doc_id, source=doc_id.split(":")[0], title="Review", score=1.0) for doc_id in ids),
        insufficient_evidence=insufficient,
        metadata={"retrieved_ids": retrieved_ids, "fused_ids": retrieved_ids, "reranked_ids": retrieved_ids, "context_ids": context_ids},
    )


def test_loader_rejects_duplicate_and_unnamespaced_labels(tmp_path: Path):
    from evaluation.rag.datasets import load_rag_cases

    path = tmp_path / "cases.jsonl"
    raw = {
        "question_id": "q1", "question": "Review?", "question_class": "review_based", "product_id": "p1",
        "expected_sources": ["review"], "relevant_document_ids": ["review:r1"],
        "acceptable_facts": ["comfortable"], "expect_insufficient_evidence": False,
    }
    path.write_text(json.dumps(raw) + "\n" + json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="duplicate question_id"):
        load_rag_cases(path)
    path.write_text(json.dumps(raw | {"relevant_document_ids": ["r1"]}) + "\n")
    with pytest.raises(ValueError, match="namespaced"):
        load_rag_cases(path)


def test_metrics_do_not_invent_answer_correctness_or_citation_context():
    from evaluation.rag.metrics import classify_failure, score_case

    case = _case()
    scored = score_case(case, _answer(("review:r1", "review:wrong"), context_ids=["review:r1"], retrieved_ids=["review:r1"]))
    assert scored["retrieval_recall_at_10"] == 1.0
    assert scored["citation_validity"] == 0.5
    assert scored["citation_precision"] == 0.5
    assert scored["citation_recall"] == 1.0
    assert scored["answer_correctness"] is None
    assert scored["acceptable_fact_coverage"] == 1.0

    answer_without_trace = _answer()
    answer_without_trace.metadata.clear()
    assert score_case(case, answer_without_trace)["citation_validity"] is None
    assert score_case(case, answer_without_trace)["retrieval_recall_at_10"] is None

    removed_by_fusion = _answer(retrieved_ids=["review:r1"])
    removed_by_fusion.metadata["fused_ids"] = ["review:wrong"]
    assert score_case(case, removed_by_fusion)["candidate_recall"] == 1.0
    assert score_case(case, removed_by_fusion)["retrieval_recall_at_10"] == 0.0
    assert classify_failure(case, score_case(case, removed_by_fusion)) == "fusion_miss"
    assert classify_failure(case, scored) == "invalid_citation"


def test_candidate_recall_includes_policy_after_other_source_candidates():
    from evaluation.rag.metrics import classify_failure, score_case

    case = _case(relevant_document_ids=("policy:returns",))
    answer = _answer((), retrieved_ids=[*(f"product:p{i}" for i in range(20)), *(f"review:r{i}" for i in range(40)), "policy:returns"])
    answer.metadata["fused_ids"] = ["policy:returns"]

    metrics = score_case(case, answer)
    assert metrics["candidate_recall"] == 1.0
    assert classify_failure(case, metrics) != "candidate_miss"


def test_runner_applies_retrieval_fusion_and_reranker_overrides(tmp_path: Path):
    from evaluation.rag.runner import run_rag_experiment

    configs = []
    for mode, fusion, rerank in (("bm25", "rrf", False), ("dense", "concat", True)):
        path = tmp_path / f"{mode}.yaml"
        path.write_text(
            f"experiment:\n  name: {mode}\nsources: [review]\nrag:\n"
            f"  retrieval:\n    review_mode: {mode}\n    review_candidates: 7\n"
            f"  fusion:\n    method: {fusion}\n    top_k: 9\n"
            f"  reranker:\n    enabled: {str(rerank).lower()}\n    top_k: 3\n"
        )
        configs.append(path)

    seen = []

    def factory(config):
        seen.append((config.retrieval.review_mode, config.retrieval.review_candidates, config.fusion.method, config.fusion.top_k, config.reranker.enabled, config.reranker.top_k))

        class Service:
            def answer(self, request):
                assert request.product_id == "p1"
                assert request.sources == ("review",)
                return _answer() if config.retrieval.review_mode == "bm25" else _answer(())

        return Service()

    first = run_rag_experiment(config_path=configs[0], service_factory=factory, cases=[_case()], runs_root=tmp_path / "runs")
    second = run_rag_experiment(config_path=configs[1], service_factory=factory, cases=[_case()], runs_root=tmp_path / "runs")
    assert seen == [("bm25", 7, "rrf", 9, False, 3), ("dense", 7, "concat", 9, True, 3)]
    assert first.metrics["retrieval_recall_at_10"] == 1.0
    assert second.metrics["citation_recall"] == 0.0
    for summary in (first, second):
        metrics = json.loads((summary.run_dir / "metrics.json").read_text())
        assert metrics["metrics"]["answer_correctness"] is None
        assert "error_counts" in metrics
        assert metrics["latency_ms"]["p50"] >= 0
        assert (summary.run_dir / "results.jsonl").exists()
        assert (summary.run_dir / "config.yaml").exists()
        assert (summary.run_dir / "summary.md").exists()


def test_runner_rejects_unknown_knobs_before_creating_service(tmp_path: Path):
    from evaluation.rag.runner import run_rag_experiment

    path = tmp_path / "bad.yaml"
    path.write_text("experiment:\n  name: bad\nrag:\n  fusion:\n    ignored_knob: true\n")
    called = False

    def factory(config):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="ignored_knob"):
        run_rag_experiment(config_path=path, service_factory=factory, cases=[_case()], runs_root=tmp_path / "runs")
    assert not called


def test_retrieval_only_ablation_never_calls_llm_and_leaves_answer_metrics_unmeasured(tmp_path: Path):
    from evaluation.rag.runner import run_rag_experiment

    config = tmp_path / "retrieval_only.yaml"
    config.write_text("experiment:\n  name: retrieval_only\n  generate_answers: false\nsources: [review]\n")

    class Service:
        def retrieve(self, request):
            assert request.sources == ("review",)
            return {
                "retrieved_ids": ["review:r1"],
                "fused_ids": ["review:r1"],
                "reranked_ids": ["review:r1"],
                "retrieval_modes": {"review": "bm25"},
            }

        def answer(self, request):
            raise AssertionError("retrieval-only experiments must not generate answers")

    summary = run_rag_experiment(config_path=config, service_factory=lambda rag: Service(), cases=[_case()], runs_root=tmp_path / "runs")
    metrics = json.loads((summary.run_dir / "metrics.json").read_text())["metrics"]
    result = json.loads((summary.run_dir / "results.jsonl").read_text())

    assert metrics["candidate_recall"] == 1.0
    assert metrics["reranked_ndcg_at_10"] == 1.0
    assert metrics["citation_precision"] is None
    assert metrics["answer_correctness"] is None
    assert result["answer"] is None
    assert result["citation_ids"] is None
    assert result["retrieval_modes"] == {"review": "bm25"}


def test_retrieval_only_flag_must_be_boolean(tmp_path: Path):
    from evaluation.rag.runner import load_experiment_config

    config = tmp_path / "bad.yaml"
    config.write_text("experiment:\n  name: bad\n  generate_answers: false-ish\n")
    with pytest.raises(ValueError, match="generate_answers"):
        load_experiment_config(config)


def test_builder_uses_a_review_matched_to_selected_product(tmp_path: Path):
    from scripts.build_rag_evaluation_set import build_rag_evaluation_set
    from evaluation.rag.datasets import load_rag_cases

    products = tmp_path / "products.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    policies = tmp_path / "policies.jsonl"
    output = tmp_path / "cases.jsonl"
    products.write_text('{"product_id":"p1","title":"Trail Shoe"}\n')
    reviews.write_text('{"review_id":"other","product_id":"p2","text":"wrong product"}\n{"review_id":"r1","product_id":"p1","text":"comfortable","rating":5}\n')
    policies.write_text('{"policy_id":"returns","section_id":"returns.window","title":"Return Window","content":"30 calendar days","policy_type":"returns","version":"2026-09","effective_date":"2026-09-01"}\n')
    build_rag_evaluation_set(products, reviews, policies, output)
    cases = load_rag_cases(output)
    assert {case.question_class for case in cases} == {"product_factual", "review_based", "policy", "multi_source", "unanswerable"}
    review_case = next(case for case in cases if case.question_class == "review_based")
    product_case = next(case for case in cases if case.question_class == "product_factual")
    assert product_case.product_id is None
    assert review_case.product_id == "p1"
    assert review_case.relevant_document_ids == ("review:r1",)
    assert all(case.label_status == "seed_unreviewed" for case in cases)
