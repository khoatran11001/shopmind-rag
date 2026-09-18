from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

_VI_CATEGORY = {
    "ACCESSORY": "phụ kiện",
    "BOOT": "bốt",
    "CELLULAR_PHONE_CASE": "ốp điện thoại",
    "CHAIR": "ghế",
    "FINERING": "nhẫn",
    "GROCERY": "thực phẩm",
    "HEALTH_PERSONAL_CARE": "đồ chăm sóc cá nhân",
    "HOME": "đồ gia dụng",
    "HOME_BED_AND_BATH": "đồ phòng ngủ và phòng tắm",
    "HOME_FURNITURE_AND_DECOR": "nội thất trang trí",
    "OFFICE_PRODUCTS": "đồ văn phòng",
    "SANDAL": "dép sandal",
    "SHOES": "giày",
    "TABLE": "bàn",
}
_VI_COLOR = {"black": "đen", "blue": "xanh dương", "brown": "nâu", "green": "xanh lá", "red": "đỏ", "white": "trắng"}


def _read_products(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _diverse_products(products: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    shuffled = list(products)
    rng.shuffle(shuffled)
    for row in shuffled:
        groups[str(row.get("category") or "UNKNOWN")].append(row)
    selected: list[dict[str, Any]] = []
    categories = sorted(groups)
    while len(selected) < count:
        added = False
        for category in categories:
            if groups[category]:
                selected.append(groups[category].popleft())
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(f"dataset has only {len(selected)} usable products; need {count}")
    return selected


def _color(row: dict[str, Any]) -> str:
    attributes = row.get("attributes") or {}
    return str(attributes.get("color") or "").split("(", 1)[0].strip()


def _category(row: dict[str, Any]) -> str:
    return str(row.get("category") or "product").replace("_", " ").lower()


def _ascii(value: Any) -> str:
    text = str(value or "").strip()
    return text if text.isascii() else ""


def _distractor(target: dict[str, Any], products: list[dict[str, Any]]) -> str:
    same_category = [row for row in products if row["product_id"] != target["product_id"] and row.get("category") == target.get("category")]
    pool = same_category or [row for row in products if row["product_id"] != target["product_id"]]
    return str(pool[0]["product_id"])


def _transform_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        margin_x = max(1, width // 20)
        margin_y = max(1, height // 20)
        cropped = rgb.crop((margin_x, margin_y, max(margin_x + 1, width - margin_x), max(margin_y + 1, height - margin_y)))
        resized = cropped.resize((width, height))
        ImageEnhance.Contrast(resized).enhance(0.95).save(destination, format="JPEG", quality=75, optimize=True)


def build_benchmark(products_path: str | Path, output_dir: str | Path, *, seed: int = 42) -> Path:
    products = _read_products(Path(products_path))
    if len(products) < 100:
        raise ValueError("A1 benchmark requires at least 100 products")
    selected = _diverse_products(products, 100, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []

    def add_text(query_id: str, row: dict[str, Any], text: str, group: str, language: str) -> None:
        queries.append({"query_id": query_id, "query_type": "text", "query": text, "group": group, "language": language})
        qrels.extend([
            {"query_id": query_id, "product_id": row["product_id"], "relevance": 2},
            {"query_id": query_id, "product_id": _distractor(row, products), "relevance": 0},
        ])

    for index, row in enumerate(selected[:40], start=1):
        color = _ascii(_color(row))
        category = _category(row)
        if index <= 15:
            text, group = _ascii(row.get("title")) or " ".join(value for value in (color, category) if value), "exact"
        elif index <= 35:
            text, group = " ".join(value for value in (color, category, "for everyday use") if value), "semantic"
        else:
            text, group = " ".join(value for value in (_ascii(row.get("brand")), color, category) if value), "hard_negative"
        add_text(f"e{index:03}", row, text, group, "en")

    for index, row in enumerate(selected[40:70], start=1):
        brand = str(row.get("brand") or "").strip()
        category = _VI_CATEGORY.get(str(row.get("category") or ""), "sản phẩm")
        raw_color = _color(row)
        color = _VI_COLOR.get(raw_color.lower(), "")
        if index <= 10:
            text, group = " ".join(value for value in ("mua", brand, category, color) if value), "exact"
        elif index <= 25:
            text, group = " ".join(value for value in ("tìm", brand, category, color, "để sử dụng hằng ngày") if value), "semantic"
        else:
            text, group = " ".join(value for value in ("mẫu", brand, category, color) if value), "hard_negative"
        add_text(f"v{index:03}", row, text, group, "vi")

    for index, row in enumerate(selected[70:80], start=1):
        language = "en" if index <= 5 else "vi"
        category = _category(row) if language == "en" else _VI_CATEGORY.get(str(row.get("category") or ""), "sản phẩm")
        color = _ascii(_color(row)) if language == "en" else _VI_COLOR.get(_color(row).lower(), "")
        brand = _ascii(row.get("brand")) if language == "en" else str(row.get("brand") or "").strip()
        text = " ".join(value for value in (("product" if language == "en" else "sản phẩm"), brand, category, color) if value)
        add_text(f"h{index:03}", row, text, "hard_negative", language)

    for index, row in enumerate(selected[80:], start=1):
        query_id = f"i{index:03}"
        relative_path = Path("images") / f"{query_id}.jpg"
        _transform_image(Path(row["main_image_path"]), output / relative_path)
        queries.append({"query_id": query_id, "query_type": "image", "image_path": str(relative_path), "group": "image"})
        qrels.extend([
            {"query_id": query_id, "product_id": row["product_id"], "relevance": 2},
            {"query_id": query_id, "product_id": _distractor(row, products), "relevance": 0},
        ])

    for filename, rows in (("queries.jsonl", queries), ("qrels.jsonl", qrels)):
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output / "README.md").write_text(
        "# ShopMind A1 benchmark\n\n"
        "100 deterministic queries: 80 text (45 English, 35 Vietnamese) and 20 transformed image queries.\n\n"
        "Qrels hiện là seed labels được sinh từ target product và explicit distractor. "
        "review chéo bởi hai thành viên và adjudication vẫn chưa hoàn tất; không được mô tả bộ nhãn này là human-reviewed trước khi checklist đó được ký xác nhận.\n\n"
        "See `REVIEW.md` for the required two-reviewer workflow and current status.\n",
        encoding="utf-8",
    )
    (output / "REVIEW.md").write_text(
        "# A1 qrels review checklist\n\n"
        "The generator creates one grade-2 target and one explicit grade-0 distractor per query.\n\n"
        "1. Reviewer 1 checks every target/distractor against catalog metadata and the transformed image.\n"
        "2. Reviewer 2 independently repeats the check.\n"
        "3. Record disagreements by `query_id`, adjudicate them, and keep grades in `{0, 1, 2}`.\n"
        "4. Record final reviewer names/date and the adjudication count here.\n\n"
        "Current status: seed labels generated; second-member review and adjudication pending.\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic ShopMind A1 evaluation benchmark.")
    parser.add_argument("--products", type=Path, default=Path("data/a1_abo_1500/products.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/benchmarks/a1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(build_benchmark(args.products, args.output, seed=args.seed))


if __name__ == "__main__":
    main()
