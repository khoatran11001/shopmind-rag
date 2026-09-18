from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import yaml
from PIL import Image

from evaluation.datasets import EvaluationQuery, load_qrels, load_queries
from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from evaluation.reports import write_run_summary
from shopmind.app.domain.search_query import SearchMode, SearchRequest


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_dir: Path
    metrics: dict[str, float]
    successful_queries: int
    failed_queries: int


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _resolve_config(config_path: Path, service: Any) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "experiment config root")
    experiment = _require_mapping(root.get("experiment"), "experiment")
    retrieval = _require_mapping(root.get("retrieval"), "retrieval")

    name = str(experiment.get("name") or "").strip()
    dataset_version = str(experiment.get("dataset_version") or "").strip()
    index_version = str(experiment.get("index_version") or "").strip()
    if not name or not dataset_version or not index_version:
        raise ValueError("experiment name, dataset_version, and index_version are required")

    mode_text = str(retrieval.get("mode") or "").strip()
    if mode_text != "image_dense":
        try:
            SearchMode(mode_text)
        except ValueError as exc:
            raise ValueError(f"unsupported retrieval mode: {mode_text}") from exc
    top_k = int(retrieval.get("top_k", 10))
    candidate_k = int(retrieval.get("candidate_k", 100))
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("retrieval requires positive top_k and candidate_k >= top_k")

    fusion = _require_mapping(root.get("fusion") or {}, "fusion")
    fusion.setdefault("method", "rrf")
    fusion.setdefault("rrf_k", 60)
    if int(fusion["rrf_k"]) <= 0:
        raise ValueError("fusion.rrf_k must be positive")

    embedding = _require_mapping(root.get("embedding") or {}, "embedding")
    embedding.setdefault("model", getattr(service, "embedding_model", "unknown"))
    embedding.setdefault("version", getattr(service, "embedding_version", "unknown"))

    reranker = _require_mapping(root.get("reranker") or {}, "reranker")
    reranker.setdefault("enabled", False)

    return {
        "experiment": {"name": name, "dataset_version": dataset_version, "index_version": index_version},
        "retrieval": {"mode": mode_text, "top_k": top_k, "candidate_k": candidate_k},
        "fusion": {"method": str(fusion["method"]), "rrf_k": int(fusion["rrf_k"])},
        "embedding": {"model": str(embedding.get("model") or "unknown"), "version": str(embedding.get("version") or "unknown")},
        "reranker": {"enabled": bool(reranker.get("enabled", False))},
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return cleaned or "experiment"


def _create_run_dir(runs_root: Path, name: str, now: datetime) -> tuple[str, Path]:
    runs_root.mkdir(parents=True, exist_ok=True)
    base = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{_sanitize_name(name)}"
    run_id = base
    run_dir = runs_root / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{base}_{suffix:02d}"
        run_dir = runs_root / run_id
        suffix += 1
    run_dir.mkdir()
    return run_id, run_dir


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {name: fmean(row[name] for row in rows) if rows else 0.0 for name in ("Recall@10", "nDCG@10", "MRR@10")}


def _group_metrics(result_rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, float]]] = {}
    for row in result_rows:
        value = row.get(key)
        if value is not None:
            groups.setdefault(str(value), []).append(row["metrics"])
    return {name: _aggregate(rows) for name, rows in sorted(groups.items())}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_experiment(config_path: str | Path, service: Any, queries: list[EvaluationQuery], qrels: dict[str, dict[str, int]], runs_root: str | Path, query_type: str | None = None) -> RunSummary:
    if query_type not in {None, "text", "image"}:
        raise ValueError("query_type must be text, image, or None")
    if query_type is not None:
        queries = [query for query in queries if query.query_type == query_type]
    if not queries:
        raise ValueError("experiment requires at least one query")
    missing_qrels = [query.query_id for query in queries if query.query_id not in qrels]
    if missing_qrels:
        raise ValueError(f"qrels missing query_id: {missing_qrels[0]}")

    config = _resolve_config(Path(config_path), service)
    now = _utc_now()
    run_id, run_dir = _create_run_dir(Path(runs_root), config["experiment"]["name"], now)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    mode_text = config["retrieval"]["mode"]
    if mode_text == "image_dense" and any(query.query_type != "image" for query in queries):
        raise ValueError("image_dense experiments accept image queries only")
    mode = None if mode_text == "image_dense" else SearchMode(mode_text)
    for query in queries:
        started = perf_counter()
        try:
            if query.query_type == "text":
                request = SearchRequest(query=query.query or "", mode=mode, top_k=int(config["retrieval"]["top_k"]), candidate_k=int(config["retrieval"]["candidate_k"]))
                results = service.search_text(request)
                result_mode = mode_text
            else:
                with Image.open(query.image_path) as image:
                    results = service.search_image(image.convert("RGB"), top_k=int(config["retrieval"]["top_k"]), candidate_k=int(config["retrieval"]["candidate_k"]), filters=None)
                result_mode = "image_dense"
            latency_ms = (perf_counter() - started) * 1000
            ranked_ids = [result.product_id for result in results]
            relevance = qrels[query.query_id]
            query_metrics = {"Recall@10": recall_at_k(ranked_ids, relevance, 10), "nDCG@10": ndcg_at_k(ranked_ids, relevance, 10), "MRR@10": mrr_at_k(ranked_ids, relevance, 10)}
            result_rows.append({"query_id": query.query_id, "query_type": query.query_type, "query": query.query, "image_path": str(query.image_path) if query.image_path else None, "group": query.group, "language": query.language, "mode": result_mode, "latency_ms": latency_ms, "results": [{"product_id": result.product_id, "score": result.score, "rank": result.rank, "source": result.source, "retrieval_scores": asdict(result.retrieval_scores)} for result in results], "metrics": query_metrics})
        except Exception as exc:
            error_rows.append({"query_id": query.query_id, "query_type": query.query_type, "query": query.query, "error_type": type(exc).__name__, "message": str(exc)})

    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    _write_jsonl(run_dir / "results.jsonl", result_rows)
    _write_jsonl(run_dir / "errors.jsonl", error_rows)

    successful = len(result_rows)
    failed = len(error_rows)
    aggregate = _aggregate([row["metrics"] for row in result_rows])
    latencies = [float(row["latency_ms"]) for row in result_rows]
    query_types = {kind: sum(row["query_type"] == kind for row in result_rows) for kind in ("image", "text")}
    query_types = {key: value for key, value in query_types.items() if value}
    metadata = {"run_id": run_id, "created_at": now.isoformat().replace("+00:00", "Z"), "experiment_name": config["experiment"]["name"], "dataset_version": config["experiment"]["dataset_version"], "index_version": config["experiment"]["index_version"], "embedding_model": config["embedding"]["model"], "embedding_version": config["embedding"]["version"], "retrieval": config["retrieval"], "fusion": config["fusion"], "reranker": config["reranker"], "metrics": aggregate, "metrics_by_query_type": _group_metrics(result_rows, "query_type"), "metrics_by_language": _group_metrics(result_rows, "language"), "metrics_by_group": _group_metrics(result_rows, "group"), "query_types": query_types, "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)}, "successful_queries": successful, "failed_queries": failed}
    (run_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_summary(run_dir / "summary.md", metadata)

    if successful == 0:
        raise RuntimeError(f"all queries failed; artifacts written to {run_dir}")
    return RunSummary(run_id=run_id, run_dir=run_dir, metrics=aggregate, successful_queries=successful, failed_queries=failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible ShopMind retrieval experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=Path("data/evaluation/queries.jsonl"))
    parser.add_argument("--qrels", type=Path, default=Path("data/evaluation/qrels.jsonl"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--query-type", choices=("text", "image"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels, queries)
    from fastapi import FastAPI
    from shopmind.app.main import _wire_runtime
    runtime = FastAPI()
    runtime.state.search_service = None
    _wire_runtime(runtime)
    service = runtime.state.search_service
    service.embedding_model = getattr(runtime.state.embedder, "model_name", "unknown")
    service.embedding_version = getattr(runtime.state.embedder, "model_revision", "unknown")
    summary = run_experiment(args.config, service, queries, qrels, args.runs_root, query_type=args.query_type)
    print(summary.run_dir)


if __name__ == "__main__":
    main()
