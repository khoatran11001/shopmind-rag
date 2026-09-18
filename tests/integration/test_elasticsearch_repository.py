from __future__ import annotations

import os
import uuid
import pytest

pytestmark=pytest.mark.integration


def _client_or_skip():
    try: from elasticsearch import Elasticsearch
    except ImportError: pytest.skip("elasticsearch Python package is not installed")
    url=os.getenv("ELASTICSEARCH_URL","http://localhost:9200"); client=Elasticsearch(url)
    try:
        if not client.ping(): pytest.skip(f"Elasticsearch is not reachable at {url}")
    except Exception as exc: pytest.skip(f"Elasticsearch is not reachable: {exc}")
    return client


def test_repository_bm25_vector_filter_and_alias_lifecycle():
    from shopmind.app.infrastructure.elasticsearch.index import create_versioned_index,switch_alias,validate_product_index
    from shopmind.app.infrastructure.elasticsearch.repository import ElasticsearchProductRepository
    client=_client_or_skip(); index_name=f"test_products_{uuid.uuid4().hex}"; alias=f"test_products_alias_{uuid.uuid4().hex}"
    documents=[{"product_id":"P1","title":"Black Running Shoe","description":"lightweight runner","brand":"Acme","category":"Shoes","attributes":{"color":"black"},"attributes_text":"color: black","search_text":"Black Running Shoe","image_url":None,"text_vector":[1.0,0.0,0.0,0.0],"image_vector":[1.0,0.0,0.0,0.0],"embedding_model":"fake","embedding_version":"test","metadata":{}},{"product_id":"P2","title":"Oak Chair","description":"wood dining chair","brand":"WoodCo","category":"Furniture","attributes":{"material":"oak"},"attributes_text":"material: oak","search_text":"Oak Chair","image_url":None,"text_vector":[0.0,1.0,0.0,0.0],"image_vector":[0.0,1.0,0.0,0.0],"embedding_model":"fake","embedding_version":"test","metadata":{}},{"product_id":"P3","title":"Trail Boot","description":"outdoor boot","brand":"Acme","category":"Shoes","attributes":{},"attributes_text":"","search_text":"Trail Boot","image_url":None,"text_vector":[0.9,0.1,0.0,0.0],"image_vector":[0.9,0.1,0.0,0.0],"embedding_model":"fake","embedding_version":"test","metadata":{}}]
    try:
        create_versioned_index(client,index_name,4); repo=ElasticsearchProductRepository(client,index_name); success,errors=repo.bulk_index(index_name,documents); assert success==3 and errors==[]; client.indices.refresh(index=index_name)
        lexical=repo.lexical_search("Black Running Shoe",3,{}); assert lexical[0].product_id=="P1"
        filtered_only=repo.lexical_search(None,3,{"brand":"acme"}); assert {hit.product_id for hit in filtered_only}=={"P1","P3"}
        partial=repo.lexical_search("shoe",3,{"category":"hoe"}); assert partial[0].product_id=="P1"
        vector=repo.vector_search("text_vector",[1,0,0,0],2,3,{}); assert vector[0].product_id=="P1"
        filtered=repo.vector_search("text_vector",[1,0,0,0],3,3,{"category":"Furniture"}); assert {hit.product_id for hit in filtered}=={"P2"}
        validate_product_index(client,index_name,expected_count=3,dimension=4,smoke_query="Black Running Shoe",smoke_vector=[1,0,0,0]); switch_alias(client,alias=alias,target=index_name); aliases=client.indices.get_alias(name=alias); assert index_name in aliases
    finally:
        try: client.indices.delete(index=index_name,ignore_unavailable=True)
        except Exception: pass
