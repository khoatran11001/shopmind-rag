from pathlib import Path
import pytest
from PIL import Image
from evaluation.datasets import EvaluationQuery, load_qrels, load_queries


def test_load_queries_preserves_file_order():
    queries=load_queries(Path("tests/fixtures/evaluation/queries.jsonl")); assert queries==[EvaluationQuery("q1","black running shoes"),EvaluationQuery("q2","oak chair")]

def test_duplicate_query_ids_are_rejected(tmp_path):
    path=tmp_path/"queries.jsonl"; path.write_text('{"query_id":"q1","query":"one"}\n{"query_id":"q1","query":"two"}\n')
    with pytest.raises(ValueError,match="duplicate query_id"): load_queries(path)

def test_blank_query_text_is_rejected(tmp_path):
    path=tmp_path/"queries.jsonl"; path.write_text('{"query_id":"q1","query":"   "}\n')
    with pytest.raises(ValueError,match="blank"): load_queries(path)

def test_load_queries_accepts_text_and_image_metadata(tmp_path):
    image = tmp_path / "query.jpg"
    Image.new("RGB", (4, 4), "red").save(image)
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query_id":"t1","query_type":"text","query":"black shoes","group":"semantic","language":"en"}\n'
        '{"query_id":"i1","query_type":"image","image_path":"query.jpg","group":"image"}\n'
    )

    queries = load_queries(path)

    assert queries[0] == EvaluationQuery("t1", "black shoes", "text", None, "semantic", "en")
    assert queries[1].query_type == "image"
    assert queries[1].image_path == image.resolve()
    assert queries[1].language is None

def test_image_query_requires_a_decodable_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query_id":"i1","query_type":"image","image_path":"bad.jpg","group":"image"}\n')

    with pytest.raises(ValueError, match="valid image"):
        load_queries(path)

def test_qrels_reference_known_queries_and_reject_negative_relevance(tmp_path):
    queries=[EvaluationQuery("q1","one")]; unknown=tmp_path/"unknown.jsonl"; unknown.write_text('{"query_id":"q2","product_id":"P1","relevance":1}\n')
    with pytest.raises(ValueError,match="unknown query_id"): load_qrels(unknown,queries)
    negative=tmp_path/"negative.jsonl"; negative.write_text('{"query_id":"q1","product_id":"P1","relevance":-1}\n')
    with pytest.raises(ValueError,match="negative"): load_qrels(negative,queries)

def test_qrels_require_a_positive_grade_and_a1_scale(tmp_path):
    queries = [EvaluationQuery("q1", "one")]
    zero = tmp_path / "zero.jsonl"
    zero.write_text('{"query_id":"q1","product_id":"P1","relevance":0}\n')
    with pytest.raises(ValueError, match="positive"):
        load_qrels(zero, queries)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"query_id":"q1","product_id":"P1","relevance":3}\n')
    with pytest.raises(ValueError, match="0, 1, or 2"):
        load_qrels(invalid, queries)

def test_load_qrels_builds_nested_mapping():
    queries=load_queries(Path("tests/fixtures/evaluation/queries.jsonl")); qrels=load_qrels(Path("tests/fixtures/evaluation/qrels.jsonl"),queries)
    assert qrels["q1"]=={"P1":2,"P2":1}; assert qrels["q2"]=={"P3":1}
