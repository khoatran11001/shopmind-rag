#!/usr/bin/env python3
"""
Prepare a compact Amazon Berkeley Objects (ABO) subset for ShopMind A1.

Output:
data/a1_abo_1500/
├── products.jsonl
├── image_manifest.jsonl
├── images/
│   ├── <item_id>.jpg
│   └── ...
└── DATASET.md

Each product has:
- product text metadata
- one local main image (ABO small image, max 256 px)
- original ABO image URL/path for traceability

Usage:
    python scripts/prepare_a1_abo_subset.py \
        --output data/a1_abo_1500 \
        --limit 1500 \
        --seed 42 \
        --workers 16

No AWS credentials are required; ABO is public.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
LISTING_URL = BASE + "/listings/metadata/listings_{shard}.json.gz"
IMAGE_META_URL = BASE + "/images/metadata/images.csv.gz"
SMALL_IMAGE_URL = BASE + "/images/small/{path}"

USER_AGENT = "ShopMind-A1-Dataset-Builder/1.0"


def fetch_bytes(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last = exc
            if attempt + 1 == retries:
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last}")


def english_value(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    for row in values:
        if isinstance(row, dict) and row.get("language_tag") == "en_US":
            value = row.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    for row in values:
        if isinstance(row, dict):
            value = row.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def all_values(values: Any, max_items: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for row in values:
        if not isinstance(row, dict):
            continue
        value = row.get("value")
        if isinstance(value, str) and value.strip() and value.strip() not in out:
            out.append(value.strip())
        if len(out) >= max_items:
            break
    return out


def load_image_map() -> dict[str, dict[str, Any]]:
    print("Downloading image metadata...")
    raw = fetch_bytes(IMAGE_META_URL)
    text = gzip.decompress(raw).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    mapping: dict[str, dict[str, Any]] = {}
    for row in reader:
        image_id = row.get("image_id")
        path = row.get("path")
        if not image_id or not path:
            continue
        mapping[image_id] = {
            "path": path,
            "width": int(row["width"]) if row.get("width") else None,
            "height": int(row["height"]) if row.get("height") else None,
        }
    print(f"Image metadata rows: {len(mapping):,}")
    return mapping


def load_candidates(
    image_map: dict[str, dict[str, Any]],
    target_pool: int,
    max_shards: int = 16,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_items: set[str] = set()

    for shard in range(max_shards):
        print(f"Downloading listing shard {shard}...")
        raw = fetch_bytes(LISTING_URL.format(shard=f"{shard:x}"))
        decoded = gzip.decompress(raw).decode("utf-8")

        for line in decoded.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)

            item_id = row.get("item_id")
            main_image_id = row.get("main_image_id")
            title = english_value(row.get("item_name"))

            if (
                not item_id
                or item_id in seen_items
                or not main_image_id
                or main_image_id not in image_map
                or not title
            ):
                continue

            image_meta = image_map[main_image_id]
            product_type = None
            product_types = row.get("product_type")
            if isinstance(product_types, list) and product_types:
                value = product_types[0].get("value") if isinstance(product_types[0], dict) else None
                if isinstance(value, str):
                    product_type = value

            node_paths: list[str] = []
            if isinstance(row.get("node"), list):
                for node in row["node"]:
                    if not isinstance(node, dict):
                        continue
                    path = node.get("node_name") or node.get("path")
                    if isinstance(path, str) and path.strip():
                        node_paths.append(path.strip())

            bullets = all_values(row.get("bullet_point"), max_items=8)
            description = english_value(row.get("product_description"))
            if not description and bullets:
                description = " ".join(bullets)

            candidates.append(
                {
                    "product_id": item_id,
                    "title": title,
                    "description": description or "",
                    "brand": english_value(row.get("brand")) or "",
                    "category": product_type or (node_paths[0] if node_paths else ""),
                    "product_type": product_type or "",
                    "color": english_value(row.get("color")) or "",
                    "material": english_value(row.get("material")) or "",
                    "style": english_value(row.get("style")) or "",
                    "bullet_points": bullets,
                    "node_paths": node_paths,
                    "main_image_id": main_image_id,
                    "abo_image_path": image_meta["path"],
                    "image_width": image_meta["width"],
                    "image_height": image_meta["height"],
                    "image_url": SMALL_IMAGE_URL.format(path=image_meta["path"]),
                }
            )
            seen_items.add(item_id)

            if len(candidates) >= target_pool:
                print(f"Candidate pool reached {len(candidates):,} products.")
                return candidates

    return candidates


def download_one(product: dict[str, Any], image_dir: Path) -> tuple[str, str | None]:
    product_id = product["product_id"]
    suffix = Path(product["abo_image_path"]).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"

    dest = image_dir / f"{product_id}{suffix}"
    if dest.exists() and dest.stat().st_size > 0:
        return product_id, str(dest)

    try:
        data = fetch_bytes(product["image_url"], retries=4, timeout=45)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return product_id, str(dest)
    except Exception as exc:
        print(f"[WARN] image failed for {product_id}: {exc}", file=sys.stderr)
        return product_id, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/a1_abo_1500"))
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.35,
        help="Download extra candidates so failed images can be replaced.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not 1000 <= args.limit <= 10000:
        parser.error("--limit must be between 1000 and 10000 for the A1 compact dataset.")

    output = args.output
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_map = load_image_map()
    pool_size = max(args.limit + 100, int(args.limit * args.candidate_multiplier))
    candidates = load_candidates(image_map, pool_size)
    if len(candidates) < args.limit:
        raise RuntimeError(
            f"Only {len(candidates)} valid candidates found; need {args.limit}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    print(f"Downloading images with {args.workers} workers...")
    downloaded: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, row, image_dir): row
            for row in candidates
        }
        for future in as_completed(futures):
            product_id, path = future.result()
            if path:
                downloaded[product_id] = path

    selected: list[dict[str, Any]] = []
    for row in candidates:
        local_path = downloaded.get(row["product_id"])
        if not local_path:
            continue
        row = dict(row)
        row["image_path"] = str(Path(local_path).relative_to(output))
        selected.append(row)
        if len(selected) == args.limit:
            break

    if len(selected) < args.limit:
        raise RuntimeError(
            f"Only {len(selected)} images downloaded successfully; need {args.limit}. "
            "Re-run the script; existing images are reused."
        )

    keep_ids = {row["product_id"] for row in selected}
    for path in image_dir.iterdir():
        if path.is_file() and path.stem not in keep_ids:
            path.unlink()

    products_path = output / "products.jsonl"
    manifest_path = output / "image_manifest.jsonl"

    with products_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(
                json.dumps(
                    {
                        "product_id": row["product_id"],
                        "main_image_id": row["main_image_id"],
                        "image_path": row["image_path"],
                        "image_url": row["image_url"],
                        "width": row["image_width"],
                        "height": row["image_height"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    dataset_md = f"""# ShopMind A1 ABO subset

- Source: Amazon Berkeley Objects (ABO)
- Records: {len(selected)}
- Image policy: one `main_image_id` per product
- Image resolution: ABO `images/small/` (maximum dimension 256 px)
- Random seed: {args.seed}
- Product metadata: official ABO listing metadata
- License: CC BY 4.0 according to the official ABO download page
- Attribution: Amazon.com and the ABO dataset authors
- Official page: https://amazon-berkeley-objects.s3.amazonaws.com/index.html

This is a derived subset. No image pixels are modified by this script.
"""
    (output / "DATASET.md").write_text(dataset_md, encoding="utf-8")

    print()
    print("Dataset ready:")
    print(f"  records : {len(selected):,}")
    print(f"  products: {products_path}")
    print(f"  images  : {image_dir}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
