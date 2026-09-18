from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evaluation.datasets import EvaluationQuery, load_qrels, load_queries
from evaluation.error_analysis import export_error_cases
from evaluation.reports import write_ablation_comparison
from evaluation.runner import RunSummary, run_experiment
from shopmind.app.domain.search_query import SearchMode, SearchRequest
from shopmind.app.domain.search_result import SearchResult


class HybridNoRRFControlService:
    """Evaluation-only BM25-first candidate union without rank fusion."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.embedding_model = getattr(service, "embedding_model", "unknown")
        self.embedding_version = getattr(service, "embedding_version", "unknown")

    def search_text(self, request: SearchRequest) -> list[SearchResult]:
        if request.mode is not SearchMode.HYBRID:
            return self.service.search_text(request)

        candidate_request = replace(
            request,
            mode=SearchMode.BM25,
            top_k=request.candidate_k,
            candidate_k=request.candidate_k,
        )
        bm25 = self.service.search_text(candidate_request)
        dense_request = replace(candidate_request, mode=SearchMode.DENSE)
        dense = self.service.search_text(dense_request)

        combined: list[SearchResult] = []
        seen: set[str] = set()
        for result in [*bm25, *dense]:
            if result.product_id in seen:
                continue
            seen.add(result.product_id)
            combined.append(result)
            if len(combined) >= request.top_k:
                break
        return [
            replace(result, rank=rank, source="hybrid_no_rrf")
            for rank, result in enumerate(combined, start=1)
        ]


@dataclass(frozen=True)
class AblationSummary:
    comparison_dir: Path
    runs: tuple[RunSummary, ...]


def _config_identity(path: Path) -> tuple[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid experiment config: {path}")
    experiment = raw.get("experiment") or {}
    fusion = raw.get("fusion") or {}
    name = str(experiment.get("name") or "").strip()
    method = str(fusion.get("method") or "rrf").strip()
    if not name:
        raise ValueError(f"experiment config has no name: {path}")
    return name, method


def _comparison_dir(runs_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = runs_root / f"ablation_{stamp}"
    path = base
    suffix = 1
    while path.exists():
        path = runs_root / f"{base.name}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def _failure_evidence(query: EvaluationQuery) -> dict[str, Any]:
    """Attach only deterministic evidence; the reviewer still adjudicates the cause."""
    if query.query_type == "image":
        return {"poor_image_quality": True}
    if query.group == "semantic":
        return {"semantic_confusion": True}
    if query.group == "hard_negative":
        return {"ambiguous_query": True}
    return {"lexical_overlap": 0, "expected_terms": (query.query or "").split()}


def run_ablation(
    config_paths: list[str | Path],
    service: Any,
    queries: list[EvaluationQuery],
    qrels: dict[str, dict[str, int]],
    runs_root: str | Path,
) -> AblationSummary:
    if not config_paths:
        raise ValueError("ablation requires at least one experiment config")
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    runs: list[RunSummary] = []
    rows: list[dict[str, Any]] = []
    for config_value in config_paths:
        path = Path(config_value)
        name, fusion_method = _config_identity(path)
        run_service = (
            HybridNoRRFControlService(service)
            if name == "hybrid_no_rrf" or fusion_method.lower() == "none"
            else service
        )
        summary = run_experiment(path, run_service, queries, qrels, runs_root)
        runs.append(summary)
        rows.append(
            {
                "experiment": name,
                "Recall@10": summary.metrics["Recall@10"],
                "nDCG@10": summary.metrics["nDCG@10"],
                "MRR@10": summary.metrics["MRR@10"],
                "successful_queries": summary.successful_queries,
                "failed_queries": summary.failed_queries,
            }
        )

    baseline = next((row for row in rows if row["experiment"] == "bm25"), None)
    if baseline is None:
        raise ValueError("ablation matrix requires a bm25 baseline")
    for row in rows:
        row["delta_Recall@10_vs_bm25"] = row["Recall@10"] - baseline["Recall@10"]
        row["delta_nDCG@10_vs_bm25"] = row["nDCG@10"] - baseline["nDCG@10"]
        row["delta_MRR@10_vs_bm25"] = row["MRR@10"] - baseline["MRR@10"]

    comparison_dir = _comparison_dir(runs_root)
    write_ablation_comparison(comparison_dir, rows)
    preferred = next((summary for row, summary in zip(rows, runs, strict=True) if row["experiment"] == "hybrid_rrf"), runs[-1])
    result_rows = [json.loads(line) for line in (preferred.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    query_by_id = {query.query_id: query for query in queries}
    worst = sorted(result_rows, key=lambda row: (float(row["metrics"]["nDCG@10"]), row["query_id"]))[:20]
    cases = [
        {
            "query_id": row["query_id"],
            "query_type": row.get("query_type", "text"),
            "query": row.get("query") or "",
            "relevant_ids": [product_id for product_id, grade in qrels[row["query_id"]].items() if grade > 0],
            "top_retrieved_ids": [result["product_id"] for result in row["results"]],
            "metrics": row["metrics"],
            "result_metadata": _failure_evidence(query_by_id[row["query_id"]]),
        }
        for row in worst
    ]
    export_error_cases(comparison_dir / "error-cases.jsonl", cases)
    return AblationSummary(comparison_dir=comparison_dir, runs=tuple(runs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the controlled A1 retrieval ablation matrix.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--query-type", choices=("text", "image"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    all_queries = load_queries(args.queries)
    all_qrels = load_qrels(args.qrels, all_queries)
    queries = all_queries
    if args.query_type:
        queries = [query for query in queries if query.query_type == args.query_type]
    qrels = {query.query_id: all_qrels[query.query_id] for query in queries}
    from fastapi import FastAPI
    from shopmind.app.main import _wire_runtime
    runtime = FastAPI()
    runtime.state.search_service = None
    _wire_runtime(runtime)
    service = runtime.state.search_service
    service.embedding_model = getattr(runtime.state.embedder, "model_name", "unknown")
    service.embedding_version = getattr(runtime.state.embedder, "model_revision", "unknown")
    summary = run_ablation(args.configs, service, queries, qrels, args.runs_root)
    print(summary.comparison_dir)


if __name__ == "__main__":
    main()
