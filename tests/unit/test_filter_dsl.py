import pytest
from shopmind.app.infrastructure.elasticsearch.repository import build_filter_clauses


def test_filters_support_single_and_multi_values():
    clauses = build_filter_clauses({"brand":"Acme","category":["Shoes","Boots"]})
    assert {"term":{"brand.keyword":{"value":"Acme","case_insensitive":True}}} in clauses
    assert clauses[1]["bool"]["minimum_should_match"] == 1
    assert clauses[1]["bool"]["should"] == [{"term": {"category.keyword": {"value": value, "case_insensitive": True}}} for value in ["Shoes", "Boots"]]


def test_product_id_uses_keyword_field_directly():
    assert build_filter_clauses({"product_id":"P1"}) == [{"term":{"product_id":{"value":"P1","case_insensitive":True}}}]


def test_partial_filter_clauses_use_contains_for_catalog_fields_and_prefix_for_product_id():
    from shopmind.app.infrastructure.elasticsearch.repository import build_partial_filter_clauses

    assert build_partial_filter_clauses({"brand": "amaz", "category": "cell phone", "product_id": "b01"}) == [
        {"wildcard": {"brand.keyword": {"value": "*amaz*", "case_insensitive": True}}},
        {"wildcard": {"category.keyword": {"value": "*cell_phone*", "case_insensitive": True}}},
        {"prefix": {"product_id": {"value": "b01", "case_insensitive": True}}},
    ]


def test_arbitrary_filter_field_is_rejected():
    with pytest.raises(ValueError, match="unsupported filter field"): build_filter_clauses({"metadata.secret":"x"})
