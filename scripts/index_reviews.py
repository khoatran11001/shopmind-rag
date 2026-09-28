from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from shopmind.app.core.config import load_app_config
from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
from shopmind.app.infrastructure.elasticsearch.index import switch_alias
from shopmind.app.infrastructure.elasticsearch.knowledge_index import create_knowledge_index, validate_knowledge_index
from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository
from shopmind.pipeline.knowledge_embedding_artifacts import load_knowledge_embedding_artifacts

def _load_jsonl(path:Path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def build_review_index_documents(rows,document_ids,vectors,*,embedding_model:str,embedding_version:str):
    row_ids=[str(row["document_id"]) for row in rows]
    if row_ids!=list(document_ids): raise ValueError("processed review order must exactly match embedding document IDs")
    output=[]
    for row,vector in zip(rows,vectors,strict=True):
        meta=dict(row.get("metadata") or {});output.append({"review_id":str(meta["review_id"]),"product_id":str(meta["product_id"]),"asin":str(meta.get("asin") or ""),"parent_asin":str(meta.get("parent_asin") or ""),"matched_by":str(meta["matched_by"]),"review_title":str(row.get("title") or ""),"content":str(row.get("content") or ""),"search_text":str(row["search_text"]),"rating":meta.get("rating"),"verified_purchase":meta.get("verified_purchase"),"helpful_vote":int(meta.get("helpful_vote") or 0),"timestamp":meta.get("timestamp"),"text_vector":np.asarray(vector,dtype=np.float32).tolist(),"embedding_model":embedding_model,"embedding_version":embedding_version,"metadata":meta})
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True);parser.add_argument("--embeddings",type=Path,required=True);parser.add_argument("--index-name",required=True);parser.add_argument("--alias",default="reviews");parser.add_argument("--config",type=Path,default=Path("configs/app.yaml"));parser.add_argument("--batch-size",type=int,default=500);parser.add_argument("--recreate",action="store_true");parser.add_argument("--switch-alias",action="store_true");args=parser.parse_args()
    if args.batch_size<=0: raise ValueError("batch-size must be positive")
    rows=_load_jsonl(args.input);manifest,document_ids,vectors=load_knowledge_embedding_artifacts(args.embeddings);documents=build_review_index_documents(rows,document_ids,vectors,embedding_model=manifest.model,embedding_version=manifest.model_revision or "default")
    if not documents: raise ValueError("cannot build review index from an empty document set")
    config=load_app_config(args.config);client=create_elasticsearch_client(config.elasticsearch);create_knowledge_index(client,args.index_name,"review",manifest.dimension,recreate=args.recreate);repository=ElasticsearchKnowledgeRepository(client,args.index_name,source="review");success,errors=0,[]
    for start in range(0,len(documents),args.batch_size): batch_success,batch_errors=repository.bulk_index(args.index_name,documents[start:start+args.batch_size]);success+=batch_success;errors.extend(batch_errors)
    if errors or success!=len(documents): raise RuntimeError(f"bulk indexing failed: {success}/{len(documents)}")
    client.indices.refresh(index=args.index_name);validate_knowledge_index(client,args.index_name,"review",len(documents),manifest.dimension,documents[0]["review_title"] or documents[0]["content"],documents[0]["text_vector"])
    if args.switch_alias: switch_alias(client,alias=args.alias,target=args.index_name)
    print(f"indexed {success} reviews into {args.index_name}")
if __name__=="__main__": main()
