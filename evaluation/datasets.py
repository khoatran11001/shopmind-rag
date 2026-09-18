from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str | None = None
    query_type: Literal["text", "image"] = "text"
    image_path: Path | None = None
    group: str = "general"
    language: Literal["en", "vi"] | None = None


def _jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        yield line_number, value


def load_queries(path: str | Path) -> list[EvaluationQuery]:
    path = Path(path)
    queries: list[EvaluationQuery] = []
    seen: set[str] = set()
    for line_number, row in _jsonl(path):
        query_id = str(row.get("query_id") or "").strip()
        query_type = str(row.get("query_type") or "text").strip()
        query = str(row.get("query") or "").strip() or None
        group = str(row.get("group") or "general").strip()
        language = str(row.get("language") or "").strip() or None
        if not query_id:
            raise ValueError(f"{path}:{line_number} query_id must not be blank")
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        if query_type not in {"text", "image"}:
            raise ValueError(f"{path}:{line_number} query_type must be text or image")
        if query_type == "text" and not query:
            raise ValueError(f"{path}:{line_number} query text must not be blank")
        if language not in {None, "en", "vi"}:
            raise ValueError(f"{path}:{line_number} language must be en or vi")
        image_path: Path | None = None
        if query_type == "image":
            image_value = str(row.get("image_path") or "").strip()
            if not image_value:
                raise ValueError(f"{path}:{line_number} image_path must not be blank")
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"{path}:{line_number} image_path must reference a valid image") from exc
        seen.add(query_id)
        queries.append(EvaluationQuery(query_id, query, query_type, image_path, group, language))
    return queries


def load_qrels(
    path: str | Path,
    queries: list[EvaluationQuery],
) -> dict[str, dict[str, int]]:
    path = Path(path)
    known = {query.query_id for query in queries}
    qrels: dict[str, dict[str, int]] = {query.query_id: {} for query in queries}
    for line_number, row in _jsonl(path):
        query_id = str(row.get("query_id") or "").strip()
        product_id = str(row.get("product_id") or "").strip()
        if query_id not in known:
            raise ValueError(f"qrels references unknown query_id: {query_id}")
        if not product_id:
            raise ValueError(f"{path}:{line_number} product_id must not be blank")
        relevance = row.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, int):
            raise ValueError(f"{path}:{line_number} relevance must be an integer")
        if relevance < 0:
            raise ValueError(f"{path}:{line_number} relevance must not be negative")
        if relevance not in {0, 1, 2}:
            raise ValueError(f"{path}:{line_number} relevance must be 0, 1, or 2")
        if product_id in qrels[query_id]:
            raise ValueError(f"duplicate qrel for {query_id}/{product_id}")
        qrels[query_id][product_id] = relevance
    missing_positive = [query_id for query_id, rows in qrels.items() if not any(value > 0 for value in rows.values())]
    if missing_positive:
        raise ValueError(f"qrels must contain a positive relevance for every query: {missing_positive[0]}")
    return qrels
