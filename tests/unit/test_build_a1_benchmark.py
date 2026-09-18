import hashlib
import json
from pathlib import Path

from PIL import Image

from scripts.build_a1_benchmark import build_benchmark


def test_build_benchmark_creates_100_queries_and_nonidentical_image_inputs(tmp_path: Path):
    products_path = tmp_path / "products.jsonl"
    image_dir = tmp_path / "source-images"
    image_dir.mkdir()
    categories = ["SHOES", "CHAIR", "GROCERY", "HOME", "SANDAL", "BOOT", "TABLE", "OFFICE_PRODUCTS", "ACCESSORY", "HEALTH_PERSONAL_CARE"]
    rows = []
    for index in range(120):
        image_path = image_dir / f"P{index:03}.jpg"
        Image.new("RGB", (16, 16), (index % 255, 40, 80)).save(image_path, quality=95)
        rows.append({
            "product_id": f"P{index:03}",
            "title": f"Example {categories[index % len(categories)]} {index}",
            "brand": f"Brand {index % 7}",
            "category": categories[index % len(categories)],
            "attributes": {"color": ["Black", "Blue", "Red"][index % 3]},
            "main_image_path": str(image_path),
        })
    products_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    output = build_benchmark(products_path, tmp_path / "benchmark", seed=42)

    queries = [json.loads(line) for line in (output / "queries.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (output / "qrels.jsonl").read_text().splitlines()]
    assert len(queries) == 100
    assert sum(row["query_type"] == "text" for row in queries) == 80
    assert sum(row["query_type"] == "image" for row in queries) == 20
    assert sum(row.get("language") == "en" for row in queries) == 45
    assert sum(row.get("language") == "vi" for row in queries) == 35
    assert all(any(qrel["query_id"] == row["query_id"] and qrel["relevance"] > 0 for qrel in qrels) for row in queries)

    image_query = next(row for row in queries if row["query_type"] == "image")
    target = next(qrel["product_id"] for qrel in qrels if qrel["query_id"] == image_query["query_id"] and qrel["relevance"] == 2)
    source = next(row["main_image_path"] for row in rows if row["product_id"] == target)
    generated = output / image_query["image_path"]
    assert generated.is_file()
    assert hashlib.sha256(generated.read_bytes()).digest() != hashlib.sha256(Path(source).read_bytes()).digest()

    readme = (output / "README.md").read_text()
    assert "review chéo" in readme
    assert "chưa hoàn tất" in readme
