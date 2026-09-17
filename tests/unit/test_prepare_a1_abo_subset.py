from pathlib import Path

from scripts.prepare_a1_abo_subset import build_canonical_record


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
