from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class NormalizedReview:
    review_id: str
    product_id: str
    asin: str
    parent_asin: str
    matched_by: str
    title: str
    text: str
    rating: float | None
    verified_purchase: bool | None
    helpful_vote: int
    timestamp: int | None
    search_text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_review_id(raw: dict[str, Any]) -> str:
    stable = {key: raw.get(key) for key in ("asin", "parent_asin", "timestamp", "title", "text")}
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def match_review_product(raw: dict[str, Any], product_ids: set[str]) -> tuple[str | None, str | None]:
    asin = str(raw.get("asin") or "").strip()
    parent_asin = str(raw.get("parent_asin") or "").strip()
    if asin in product_ids:
        return asin, "asin"
    if parent_asin in product_ids:
        return parent_asin, "parent_asin"
    return None, None


def normalize_review(raw: dict[str, Any], product_ids: set[str]) -> NormalizedReview | None:
    product_id, matched_by = match_review_product(raw, product_ids)
    if product_id is None or matched_by is None:
        return None
    title = str(raw.get("title") or "").strip()
    text = str(raw.get("text") or "").strip()
    search_text = "\n".join(part for part in (title, text) if part)
    if not search_text:
        return None
    verified = raw.get("verified_purchase")
    return NormalizedReview(
        review_id=deterministic_review_id(raw),
        product_id=product_id,
        asin=str(raw.get("asin") or "").strip(),
        parent_asin=str(raw.get("parent_asin") or "").strip(),
        matched_by=matched_by,
        title=title,
        text=text,
        rating=None if raw.get("rating") is None else float(raw["rating"]),
        verified_purchase=verified if isinstance(verified, bool) else None,
        helpful_vote=int(raw.get("helpful_vote") or 0),
        timestamp=None if raw.get("timestamp") is None else int(raw["timestamp"]),
        search_text=search_text,
        metadata={"matched_by": matched_by},
    )


def build_overlap_report(product_ids: set[str], reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scanned = retained = 0
    asin_matches: set[str] = set()
    parent_matches: set[str] = set()
    matched_products: set[str] = set()
    for raw in reviews:
        scanned += 1
        product_id, matched_by = match_review_product(raw, product_ids)
        if product_id is None:
            continue
        retained += 1
        matched_products.add(product_id)
        (asin_matches if matched_by == "asin" else parent_matches).add(product_id)
    return {
        "abo_product_count": len(product_ids),
        "review_record_count_scanned": scanned,
        "products_matched_by_asin": len(asin_matches),
        "products_matched_by_parent_asin": len(parent_matches),
        "unique_abo_products_with_reviews": len(matched_products),
        "retained_review_count": retained,
        "unmatched_review_count": scanned - retained,
        "product_review_coverage_pct": round(100 * len(matched_products) / len(product_ids), 4) if product_ids else 0.0,
    }
