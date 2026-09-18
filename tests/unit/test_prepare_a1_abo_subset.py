from pathlib import Path

from scripts.prepare_a1_abo_subset import build_canonical_record, select_balanced_subset


def test_select_balanced_subset_is_deterministic_and_caps_categories():
    rows = [
        {"product_id": f"phone-{index}", "category": "PHONE_CASE"}
        for index in range(6)
    ] + [
        {"product_id": f"shoe-{index}", "category": "SHOES"}
        for index in range(3)
    ]

    first = select_balanced_subset(rows, limit=4, max_per_category=2, seed=42)
    second = select_balanced_subset(rows, limit=4, max_per_category=2, seed=42)

    assert first == second
    assert len(first) == 4
    assert sum(row["category"] == "PHONE_CASE" for row in first) == 2
    assert sum(row["category"] == "SHOES" for row in first) == 2


def test_select_balanced_subset_fails_when_pool_cannot_meet_limit():
    rows = [{"product_id": f"p{index}", "category": "ONE"} for index in range(3)]

    try:
        select_balanced_subset(rows, limit=2, max_per_category=1, seed=42)
    except RuntimeError as exc:
        assert "1/2" in str(exc)
    else:
        raise AssertionError("expected an undersized balanced pool to fail")


def test_build_canonical_record_matches_a1_processed_schema(tmp_path: Path):
    product = {
        "product_id": "B000TEST",
        "title": "Trail Shoe",
        "description": "Lightweight shoe",
        "brand": "Example",
        "category": "SHOES",
        "product_type": "SHOES",
        "color": "Black",
        "material": "Mesh",
        "style": "Running",
        "bullet_points": ["Breathable", "Rubber sole"],
        "node_paths": ["Sports > Shoes"],
        "main_image_id": "img1",
        "abo_image_path": "ab/cd/img1.jpg",
        "image_width": 256,
        "image_height": 200,
        "image_url": "https://example.test/img1.jpg",
    }
    image_path = tmp_path / "images" / "B000TEST.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")

    record = build_canonical_record(product, image_path=image_path, output_dir=tmp_path)

    assert set(record) == {
        "product_id",
        "title",
        "description",
        "brand",
        "category",
        "attributes",
        "image_paths",
        "main_image_path",
        "search_text",
        "metadata",
    }
    assert record["main_image_path"] == str(image_path)
    assert record["image_paths"] == [str(image_path)]
    assert record["attributes"] == {
        "color": "Black",
        "material": "Mesh",
        "style": "Running",
        "product_type": "SHOES",
    }
    assert "Trail Shoe" in record["search_text"]
    assert "Brand: Example" in record["search_text"]
    assert record["metadata"]["main_image_id"] == "img1"
    assert record["metadata"]["image_url"] == "https://example.test/img1.jpg"
