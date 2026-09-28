from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Callable, Iterable

import yaml

from evaluation.rag.datasets import RAGEvaluationCase, SOURCES
from evaluation.rag.metrics import classify_failure, score_case


@dataclass(frozen=True)
class RAGRunSummary:
    run_dir: Path
    metrics: dict[str, float | None]
    successful_cases: int
    failed_cases: int


def load_experiment_config(path: str | Path, base_config: Any = None) -> tuple[dict[str, Any], Any]:
    from shopmind.app.core.config import RAGConfig

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - {"experiment", "sources", "rag"}:
        raise ValueError("experiment config must contain only experiment, sources, and rag mappings")
    experiment = raw.get("experiment")
    if not isinstance(experiment, dict) or not str(experiment.get("name") or "").strip():
        raise ValueError("experiment.name is required")
    sources = raw.get("sources", ["product", "review", "policy"])
    if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources) or set(sources) - SOURCES:
        raise ValueError("sources must be a nonempty list of unique product, review, or policy values")
    overrides = raw.get("rag") or {}
    if not isinstance(overrides, dict):
        raise ValueError("rag must be a mapping")
    merged = (base_config or RAGConfig()).model_dump()
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section] = merged[section] | value
        else:
            merged[section] = value
    rag_config = RAGConfig.model_validate(merged)
    return {"experiment": experiment, "sources": sources}, rag_config


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_rag_experiment(
    *,
    config_path: str | Path,
    service_factory: Callable[[Any], Any],
    cases: list[RAGEvaluationCase],
    runs_root: str | Path,
    base_config: Any = None,
    grader: Callable[[RAGEvaluationCase, Any], float] | None = None,
) -> RAGRunSummary:
    if not cases:
        raise ValueError("experiment requires at least one case")
    experiment, rag_config = load_experiment_config(config_path, base_config)
    service = service_factory(rag_config)
    name = str(experiment["experiment"]["name"])
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "rag"
    run_dir = Path(runs_root) / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{safe_name}"
    run_dir.mkdir(parents=True)
    resolved = {
        **experiment,
        "rag": rag_config.model_dump(mode="json", exclude={"llm": {"api_key"}}),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    from shopmind.app.rag.models import RAGRequest

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    latencies: list[float] = []
    for case in cases:
        started = perf_counter()
        try:
            answer = service.answer(RAGRequest(question=case.question, product_id=case.product_id, sources=tuple(experiment["sources"]), debug=True))
            latency_ms = (perf_counter() - started) * 1000
            latencies.append(latency_ms)
            metrics = score_case(case, answer, grader=grader)
            category = classify_failure(case, metrics)
            if category is not None:
                error_counts[category] += 1
            trace = answer.metadata or {}
            results.append({
                "question_id": case.question_id,
                "question_class": case.question_class,
                "product_id": case.product_id,
                "answer": answer.answer,
                "insufficient_evidence": answer.insufficient_evidence,
                "citation_ids": [source.document_id for source in answer.sources],
                "retrieved_ids": trace.get("retrieved_ids"),
                "fused_ids": trace.get("fused_ids"),
                "reranked_ids": trace.get("reranked_ids"),
                "context_ids": trace.get("context_ids"),
                "latency_ms": latency_ms,
                "metrics": metrics,
                "error_category": category,
            })
        except Exception as exc:
            errors.append({"question_id": case.question_id, "error_type": type(exc).__name__, "latency_ms": (perf_counter() - started) * 1000})
            error_counts["runtime_error"] += 1

    names = sorted({name for row in results for name in row["metrics"]})
    metrics = {
        name: fmean(values) if (values := [row["metrics"][name] for row in results if row["metrics"][name] is not None]) else None
        for name in names
    }
    metrics_path = run_dir / "metrics.json"
    payload = {
        "experiment_name": name,
        "case_count": len(cases),
        "successful_cases": len(results),
        "failed_cases": len(errors),
        "error_counts": dict(error_counts),
        "label_statuses": sorted({case.label_status for case in cases}),
        "config": resolved,
        "metrics": metrics,
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(run_dir / "results.jsonl", results)
    _write_jsonl(run_dir / "errors.jsonl", errors)
    summary_lines = [f"# RAG experiment: {name}", "", f"Cases: {len(results)} successful, {len(errors)} failed", f"Labels: {', '.join(payload['label_statuses'])}", ""]
    summary_lines += [f"- {key}: {value:.4f}" if value is not None else f"- {key}: unmeasured" for key, value in metrics.items()]
    summary_lines += ["", "Answer correctness requires an external grader; fact coverage is only a string match proxy.", ""]
    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    if not results:
        raise RuntimeError(f"all RAG cases failed; artifacts written to {run_dir}")
    return RAGRunSummary(run_dir, metrics, len(results), len(errors))
