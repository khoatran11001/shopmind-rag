from __future__ import annotations

from pathlib import Path
from typing import Any


def render_run_summary(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        f"# Retrieval Experiment: {payload['experiment_name']}",
        "",
        "## Metadata",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Timestamp (UTC): `{payload['created_at']}`",
        f"- Dataset version: `{payload['dataset_version']}`",
        f"- Index version: `{payload['index_version']}`",
        f"- Embedding model: `{payload['embedding_model']}`",
        f"- Embedding version: `{payload['embedding_version']}`",
        f"- Retrieval mode: `{payload['retrieval']['mode']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Recall@10 | {metrics['Recall@10']:.6f} |",
        f"| nDCG@10 | {metrics['nDCG@10']:.6f} |",
        f"| MRR@10 | {metrics['MRR@10']:.6f} |",
        "",
        "## Latency",
        "",
        f"- Latency p50: {payload['latency_ms']['p50']:.3f} ms",
        f"- Latency p95: {payload['latency_ms']['p95']:.3f} ms",
        "",
        "## Metrics by query type",
        "",
    ]
    for query_type, values in payload["metrics_by_query_type"].items():
        lines.append(f"- {query_type}: Recall@10={values['Recall@10']:.6f}, nDCG@10={values['nDCG@10']:.6f}, MRR@10={values['MRR@10']:.6f}")
    lines.extend([
        "",
        "## Query Outcomes",
        "",
        f"- Successful queries: {payload['successful_queries']}",
        f"- Failed queries: {payload['failed_queries']}",
        "",
        "## Artifacts",
        "",
        "- `config.yaml` — exact resolved experiment configuration",
        "- `metrics.json` — aggregate metrics and version metadata",
        "- `results.jsonl` — per-query rankings and metrics",
        "- `errors.jsonl` — query-level execution failures",
        "",
        "No statistical-significance claim is made by this summary.",
        "",
    ])
    return "\n".join(lines)


def write_run_summary(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.write_text(render_run_summary(payload), encoding="utf-8")
    return path


def write_ablation_comparison(comparison_dir: str | Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    import csv

    comparison_dir = Path(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = comparison_dir / "comparison.csv"
    md_path = comparison_dir / "comparison.md"

    fieldnames = [
        "experiment",
        "Recall@10",
        "nDCG@10",
        "MRR@10",
        "successful_queries",
        "failed_queries",
        "delta_Recall@10_vs_bm25",
        "delta_nDCG@10_vs_bm25",
        "delta_MRR@10_vs_bm25",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    lines = [
        "# Retrieval Ablation Comparison",
        "",
        "Deltas are measured against the BM25 baseline. They describe the isolated",
        "component comparison only and do not establish statistical significance or causality.",
        "",
        "| Experiment | Recall@10 | nDCG@10 | MRR@10 | Success | Failed | Δ Recall vs BM25 | Δ nDCG vs BM25 | Δ MRR vs BM25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {Recall@10:.6f} | {nDCG@10:.6f} | {MRR@10:.6f} | "
            "{successful_queries} | {failed_queries} | {delta_Recall@10_vs_bm25:.6f} | "
            "{delta_nDCG@10_vs_bm25:.6f} | {delta_MRR@10_vs_bm25:.6f} |".format(**row)
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
