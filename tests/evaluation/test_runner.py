from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from evaluation.datasets import EvaluationQuery
from evaluation.runner import run_experiment
from shopmind.app.domain.search_result import SearchResult


class FakeSearchService:
    embedding_model = "fake-embedder"
    embedding_version = "v-test"

    def search_text(self, request):
        if request.query == "broken":
            raise RuntimeError("query failed")
        ids = ["P1", "P3", "P2"] if "black" in request.query else ["P3", "P2", "P1"]
        return [SearchResult(pid, f"Title {pid}", rank, 1.0 / rank, request.mode.value, {}) for rank, pid in enumerate(ids[: request.top_k], start=1)]

    def search_image(self, image, *, top_k, candidate_k, filters=None):
        assert image.mode == "RGB"
        return [SearchResult("P2", "Image P2", 1, 0.9, "image_dense", {})]


def _config(path: Path):
    path.write_text("""experiment:\n  name: hybrid_rrf\n  dataset_version: fixture_v1\n  index_version: products_v1\nretrieval:\n  mode: hybrid\n  top_k: 10\n  candidate_k: 100\nfusion:\n  method: rrf\n  rrf_k: 60\nembedding:\n  model: fake-embedder\n  version: v-test\nreranker:\n  enabled: false\n""")


def test_run_experiment_writes_exact_reproducible_artifacts(tmp_path):
    config = tmp_path / "experiment.yaml"
    _config(config)
    queries = [EvaluationQuery("q1", "black running shoes"), EvaluationQuery("q2", "oak chair")]
    qrels = {"q1": {"P1": 2, "P2": 1}, "q2": {"P3": 1}}
    summary = run_experiment(config, FakeSearchService(), queries, qrels, tmp_path / "runs")
    names = {path.name for path in summary.run_dir.iterdir()}
    assert names == {"config.yaml", "metrics.json", "results.jsonl", "errors.jsonl", "summary.md"}
    metrics = json.loads((summary.run_dir / "metrics.json").read_text())
    assert metrics["dataset_version"] == "fixture_v1"
    assert metrics["index_version"] == "products_v1"
    assert metrics["embedding_model"] == "fake-embedder"
    assert metrics["embedding_version"] == "v-test"
    assert metrics["retrieval"]["mode"] == "hybrid"
    assert set(metrics["metrics"]) == {"Recall@10", "nDCG@10", "MRR@10"}
    assert metrics["successful_queries"] == 2
    assert metrics["failed_queries"] == 0
    snapshot = yaml.safe_load((summary.run_dir / "config.yaml").read_text())
    assert snapshot["fusion"]["rrf_k"] == 60
    report = (summary.run_dir / "summary.md").read_text()
    assert "Recall@10" in report
    assert "Latency p95" in report
    assert "Metrics by query type" in report


def test_run_continues_after_one_query_failure(tmp_path):
    config = tmp_path / "experiment.yaml"
    _config(config)
    queries = [EvaluationQuery("q1", "broken"), EvaluationQuery("q2", "oak chair")]
    qrels = {"q1": {"P1": 1}, "q2": {"P3": 1}}
    summary = run_experiment(config, FakeSearchService(), queries, qrels, tmp_path / "runs")
    assert summary.successful_queries == 1
    assert summary.failed_queries == 1
    errors = [json.loads(line) for line in (summary.run_dir / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["query_id"] == "q1"
    assert errors[0]["error_type"] == "RuntimeError"


def test_run_experiment_dispatches_image_query_and_groups_metrics(tmp_path):
    config = tmp_path / "experiment.yaml"
    _config(config)
    image = tmp_path / "query.jpg"
    Image.new("RGB", (4, 4), "blue").save(image)
    queries = [
        EvaluationQuery("q1", "black running shoes", "text", None, "semantic", "en"),
        EvaluationQuery("i1", None, "image", image, "image", None),
    ]
    qrels = {"q1": {"P1": 2}, "i1": {"P2": 2}}

    summary = run_experiment(config, FakeSearchService(), queries, qrels, tmp_path / "runs")

    metrics = json.loads((summary.run_dir / "metrics.json").read_text())
    assert metrics["query_types"] == {"image": 1, "text": 1}
    assert metrics["metrics_by_query_type"]["image"]["MRR@10"] == 1.0
    assert metrics["metrics_by_language"]["en"]["Recall@10"] == 1.0
    assert metrics["metrics_by_group"]["semantic"]["nDCG@10"] == 1.0
    assert metrics["latency_ms"]["p95"] >= metrics["latency_ms"]["p50"] >= 0
    rows = [json.loads(line) for line in (summary.run_dir / "results.jsonl").read_text().splitlines()]
    assert rows[1]["query_type"] == "image"
    assert rows[1]["results"][0]["source"] == "image_dense"
    assert "retrieval_scores" in rows[1]["results"][0]


def test_query_type_filter_runs_only_requested_queries(tmp_path):
    config = tmp_path / "experiment.yaml"
    _config(config)
    queries = [
        EvaluationQuery("q1", "black running shoes", "text", None, "semantic", "en"),
        EvaluationQuery("i1", None, "image", Path("missing.jpg"), "image", None),
    ]
    qrels = {"q1": {"P1": 2}, "i1": {"P2": 2}}

    summary = run_experiment(config, FakeSearchService(), queries, qrels, tmp_path / "runs", query_type="text")

    assert summary.successful_queries == 1
