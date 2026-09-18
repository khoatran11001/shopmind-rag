from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from shopmind.pipeline.embedding_artifacts import EmbeddingManifest, load_embedding_artifacts, validate_artifacts, write_embedding_artifacts
from scripts.generate_embeddings import generate_embeddings
from scripts.index_products import build_index_documents, public_image_url


class FakeEmbeddingProvider:
    model_name = "fake"; model_revision = "test"; dimension = 4; is_ready = True
    def embed_texts(self, texts):
        rows=[]
        for index,_ in enumerate(texts,start=1):
            row=np.array([index,1,1,1],dtype=np.float32); rows.append(row/np.linalg.norm(row))
        return np.vstack(rows) if rows else np.empty((0,4),dtype=np.float32)
    def embed_images(self, images):
        rows=[]
        for index,_ in enumerate(images,start=1):
            row=np.array([1,index,2,1],dtype=np.float32); rows.append(row/np.linalg.norm(row))
        return np.vstack(rows) if rows else np.empty((0,4),dtype=np.float32)


def _manifest(count: int = 2, dimension: int = 4) -> EmbeddingManifest:
    return EmbeddingManifest(model="fake", model_revision="test", dimension=dimension, dataset_version="fixture-v1", product_count=count, normalized=True, created_at="2026-09-07T00:00:00Z")


def test_artifact_validation_rejects_dimension_mismatch(tmp_path):
    np.save(tmp_path/"text_embeddings.npy",np.zeros((2,4),dtype=np.float32)); np.save(tmp_path/"image_embeddings.npy",np.zeros((2,3),dtype=np.float32)); (tmp_path/"product_ids.json").write_text('["P1", "P2"]')
    with pytest.raises(ValueError,match="dimension"): validate_artifacts(tmp_path,_manifest())


def test_write_and_load_artifacts_preserve_order_and_zero_image_sentinel(tmp_path):
    text=np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32); image=np.array([[0,0,1,0],[0,0,0,0]],dtype=np.float32); output=tmp_path/"artifacts"
    write_embedding_artifacts(output,["P2","P1"],text,image,_manifest(),force=False)
    manifest,ids,loaded_text,loaded_image=load_embedding_artifacts(output)
    assert manifest.product_count==2 and ids==["P2","P1"]
    np.testing.assert_array_equal(loaded_text,text); np.testing.assert_array_equal(loaded_image,image)


def test_validation_rejects_non_finite_and_non_float32(tmp_path):
    np.save(tmp_path/"text_embeddings.npy",np.array([[1,0,0,0]],dtype=np.float64)); np.save(tmp_path/"image_embeddings.npy",np.array([[1,0,0,0]],dtype=np.float32)); (tmp_path/"product_ids.json").write_text('["P1"]')
    with pytest.raises(ValueError,match="float32"): validate_artifacts(tmp_path,_manifest(count=1))


def test_generate_embeddings_keeps_product_order_and_missing_image_zero(tmp_path):
    processed=tmp_path/"products.jsonl"; image_path=Path("tests/fixtures/images/p1.png").resolve(); records=[{"product_id":"P2","title":"No image","description":"second","brand":None,"category":"Other","attributes":{},"image_paths":[],"main_image_path":None,"search_text":"No image\nCategory: Other\nsecond","metadata":{}},{"product_id":"P1","title":"Image item","description":"first","brand":"Acme","category":"Shoes","attributes":{"color":"black"},"image_paths":[str(image_path)],"main_image_path":str(image_path),"search_text":"Image item\nBrand: Acme\nCategory: Shoes\nfirst\ncolor: black","metadata":{}}]
    processed.write_text("\n".join(json.dumps(row) for row in records)+"\n"); output=tmp_path/"embeddings"
    generate_embeddings(input_path=processed,output_dir=output,provider=FakeEmbeddingProvider(),dataset_version="fixture-v1",limit=None,batch_size=2,force=False)
    _,ids,text,image=load_embedding_artifacts(output)
    assert ids==["P2","P1"] and text.shape==(2,4); np.testing.assert_array_equal(image[0],np.zeros(4,dtype=np.float32)); assert np.isclose(np.linalg.norm(image[1]),1.0)


def test_generate_embeddings_refuses_existing_directory_without_force(tmp_path):
    processed=tmp_path/"products.jsonl"; processed.write_text(json.dumps({"product_id":"P1","title":"One","description":"","brand":None,"category":None,"attributes":{},"image_paths":[],"main_image_path":None,"search_text":"One","metadata":{}})+"\n"); output=tmp_path/"embeddings"; provider=FakeEmbeddingProvider()
    generate_embeddings(processed,output,provider,"fixture-v1",None,2,False)
    with pytest.raises(FileExistsError): generate_embeddings(processed,output,provider,"fixture-v1",None,2,False)
    generate_embeddings(processed,output,provider,"fixture-v2",None,2,True); manifest,*_=load_embedding_artifacts(output); assert manifest.dataset_version=="fixture-v2"


def test_index_documents_omit_missing_image_zero_vector():
    rows=[{"product_id":"P1","title":"No image"},{"product_id":"P2","title":"Has image"}]
    text=np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
    image=np.array([[0,0,0,0],[0,0,1,0]],dtype=np.float32)
    documents=build_index_documents(rows,["P1","P2"],text,image,embedding_model="fake",embedding_version="test")
    assert "image_vector" not in documents[0] and documents[0]["metadata"]["missing_image"] is True
    assert documents[1]["image_vector"]==[0.0,0.0,1.0,0.0] and documents[1]["metadata"]["missing_image"] is False


def test_index_documents_publish_media_url_instead_of_local_path():
    rows = [{"product_id": "P1", "title": "Shoe", "main_image_path": "data/a1_abo_1500/images/P1.jpg"}]
    vector = np.array([[1, 0, 0, 0]], dtype=np.float32)

    documents = build_index_documents(rows, ["P1"], vector, vector, embedding_model="fake", embedding_version="test")

    assert public_image_url(rows[0]["main_image_path"]) == "/media/products/P1.jpg"
    assert documents[0]["image_url"] == "/media/products/P1.jpg"
