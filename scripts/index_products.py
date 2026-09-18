from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from shopmind.app.core.config import load_app_config
from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
from shopmind.app.infrastructure.elasticsearch.index import switch_alias, validate_product_index
from shopmind.app.infrastructure.elasticsearch.repository import ElasticsearchProductRepository
from shopmind.pipeline.embedding_artifacts import load_embedding_artifacts


def public_image_url(main_image_path: str | None) -> str | None:
    return f"/media/products/{Path(main_image_path).name}" if main_image_path else None


def _read_processed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def build_index_documents(processed_rows: list[dict[str, Any]], product_ids: list[str], text_embeddings: np.ndarray, image_embeddings: np.ndarray, *, embedding_model: str, embedding_version: str) -> list[dict[str, Any]]:
    row_ids = [str(row["product_id"]) for row in processed_rows]
    if row_ids != product_ids:
        raise ValueError("processed product order must exactly match embedding product_ids.json")
    if text_embeddings.shape[0] != len(row_ids) or image_embeddings.shape[0] != len(row_ids):
        raise ValueError("embedding row count does not match processed products")
    documents: list[dict[str, Any]] = []
    for row, text_vector, image_vector in zip(processed_rows, text_embeddings, image_embeddings, strict=True):
        attributes = dict(row.get("attributes") or {})
        metadata = dict(row.get("metadata") or {})
        metadata["missing_image"] = bool(np.isclose(np.linalg.norm(image_vector), 0.0, atol=1e-6))
        document = {"product_id": str(row["product_id"]), "title": str(row.get("title") or ""), "description": str(row.get("description") or ""), "brand": row.get("brand"), "category": row.get("category"), "attributes": attributes, "attributes_text": "\n".join(f"{key}: {attributes[key]}" for key in sorted(attributes)), "search_text": str(row.get("search_text") or ""), "image_url": public_image_url(row.get("main_image_path")), "text_vector": text_vector.astype(np.float32, copy=False).tolist(), "embedding_model": embedding_model, "embedding_version": embedding_version, "metadata": metadata}
        if not metadata["missing_image"]:
            document["image_vector"] = image_vector.astype(np.float32, copy=False).tolist()
        documents.append(document)
    return documents


def _chunks(values: list[dict[str, Any]], size: int):
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def index_products(repository: ElasticsearchProductRepository, *, index_name: str, documents: list[dict[str, Any]], batch_size: int) -> tuple[int, list[dict[str, Any]]]:
    total = 0
    errors: list[dict[str, Any]] = []
    for batch in _chunks(documents, batch_size):
        success, batch_errors = repository.bulk_index(index_name, batch)
        total += success
        errors.extend(batch_errors)
    return total, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bulk-index processed products and reusable embeddings.")
    parser.add_argument("--products", type=Path, default=Path("data/processed/products.jsonl"))
    parser.add_argument("--artifacts-dir", "--embeddings", dest="artifacts_dir", type=Path, required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--switch-alias", nargs="?", const="__DEFAULT__", default=None, metavar="ALIAS")
    parser.add_argument("--alias", default="products", help="legacy alias name used with bare --switch-alias")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_app_config(args.config)
    manifest, product_ids, text, image = load_embedding_artifacts(args.artifacts_dir)
    rows = _read_processed(args.products)
    documents = build_index_documents(rows, product_ids, text, image, embedding_model=manifest.model, embedding_version=manifest.model_revision or manifest.dataset_version)
    client = create_elasticsearch_client(config.elasticsearch)
    repository = ElasticsearchProductRepository(client, args.index_name)
    success, errors = index_products(repository, index_name=args.index_name, documents=documents, batch_size=args.batch_size)
    if errors or success != len(documents):
        raise RuntimeError(f"bulk indexing failed: {success}/{len(documents)} succeeded; errors={len(errors)}")
    client.indices.refresh(index=args.index_name)
    smoke_query = documents[0]["title"] if documents else ""
    smoke_vector = documents[0]["text_vector"] if documents else [0.0] * manifest.dimension
    validate_product_index(client, args.index_name, expected_count=len(documents), dimension=manifest.dimension, smoke_query=smoke_query, smoke_vector=smoke_vector)
    if args.switch_alias is not None:
        alias_name = args.alias if args.switch_alias == "__DEFAULT__" else args.switch_alias
        switch_alias(client, alias=alias_name, target=args.index_name)
    print(f"indexed and validated {success} products into {args.index_name}")


if __name__ == "__main__":
    main()
