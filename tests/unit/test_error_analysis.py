import json
import pytest
from evaluation.error_analysis import categorize_failure, export_error_cases


@pytest.mark.parametrize(("metadata","expected"),[({"missing_fields":["brand"]},"missing_metadata"),({"poor_image_quality":True},"poor_image_quality"),({"visual_product_type_mismatch":True},"visual_wrong_product_type"),({"expected_category":"Shoes","retrieved_category":"Furniture"},"category_confusion"),({"lexical_overlap":0,"expected_terms":["running","shoe"]},"lexical_mismatch"),({"semantic_confusion":True},"semantic_confusion"),({"ambiguous_query":True},"ambiguous_query"),({},"uncategorized")])
def test_categorize_failure_requires_explicit_evidence(metadata, expected): assert categorize_failure("query text",metadata)==expected


def test_export_error_cases_writes_review_scaffold(tmp_path):
    output=tmp_path/"errors.jsonl"; export_error_cases(output,[{"query_id":"q1","query_type":"text","query":"black running shoes","relevant_ids":["P1"],"top_retrieved_ids":["P9","P8"],"metrics":{"Recall@10":0.0,"nDCG@10":0.0,"MRR@10":0.0},"result_metadata":{"expected_category":"Shoes","retrieved_category":"Furniture"}}])
    row=json.loads(output.read_text().strip()); assert row["query_id"]=="q1"; assert row["query_type"]=="text"; assert row["suggested_category"]=="category_confusion"; assert row["reviewer_note"]==""; assert row["relevant_ids"]==["P1"]
