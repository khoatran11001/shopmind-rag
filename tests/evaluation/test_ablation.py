from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.ablation import HybridNoRRFControlService, run_ablation
from evaluation.datasets import EvaluationQuery
from shopmind.app.domain.search_query import SearchMode, SearchRequest
from shopmind.app.domain.search_result import SearchResult


class FakeService:
    embedding_model = "fake"
    embedding_version = "v1"

    def search_text(self, request):
        if request.mode is SearchMode.BM25:
            ids = ["B", "A", "C"]
        elif request.mode is SearchMode.DENSE:
            ids = ["D", "B", "A"]
        elif request.mode is SearchMode.CROSS_MODAL:
            ids = ["D", "C", "B"]
        else:
            ids = ["B", "A", "D", "C"]
        return [SearchResult(pid, pid, rank, 1.0 / rank, request.mode.value, {}) for rank, pid in enumerate(ids[: request.top_k], start=1)]


def test_hybrid_no_rrf_control_concatenates_bm25_then_unseen_dense():
    service = HybridNoRRFControlService(FakeService())
    results = service.search_text(SearchRequest("q", mode=SearchMode.HYBRID, top_k=4, candidate_k=10))
    assert [result.product_id for result in results] == ["B", "A", "C", "D"]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert all(result.source == "hybrid_no_rrf" for result in results)


def test_run_ablation_writes_comparison_with_bm25_deltas(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    templates = {"bm25": ("bm25", "rrf"), "dense": ("dense", "rrf"), "cross_modal": ("cross_modal", "rrf"), "hybrid_no_rrf": ("hybrid", "none"), "hybrid_rrf": ("hybrid", "rrf")}
    paths = []
    for name, (mode, fusion_method) in templates.items():
        path = config_dir / f"{name}.yaml"
        path.write_text(f"""experiment:\n  name: {name}\n  dataset_version: fixture_v1\n  index_version: products_v1\nretrieval:\n  mode: {mode}\n  top_k: 10\n  candidate_k: 100\nfusion:\n  method: {fusion_method}\n  rrf_k: 60\nembedding:\n  model: fake\n  version: v1\nreranker:\n  enabled: false\n""")
        paths.append(path)
    queries = [EvaluationQuery("q1", "query")]
    qrels = {"q1": {"B": 2, "A": 1}}
    summary = run_ablation(paths, FakeService(), queries, qrels, tmp_path / "runs")
    assert (summary.comparison_dir / "comparison.csv").exists()
    assert (summary.comparison_dir / "comparison.md").exists()
    with (summary.comparison_dir / "comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment"] for row in rows] == list(templates)
    assert "delta_Recall@10_vs_bm25" in rows[0]
    assert float(rows[0]["delta_Recall@10_vs_bm25"]) == 0.0
    no_rrf = next(row for row in rows if row["experiment"] == "hybrid_no_rrf")
    assert int(no_rrf["successful_queries"]) == 1
    text = (summary.comparison_dir / "comparison.md").read_text()
    assert "BM25 baseline" in text
    assert "causality" in text.lower()
    error_cases = [json.loads(line) for line in (summary.comparison_dir / "error-cases.jsonl").read_text().splitlines()]
    assert error_cases
    assert {"query_id", "query_type", "suggested_category", "reviewer_note"} <= error_cases[0].keys()
