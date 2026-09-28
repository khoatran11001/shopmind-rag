from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.generate_knowledge_embeddings import load_knowledge_documents
from shopmind.app.core.config import load_app_config
from shopmind.app.infrastructure.elasticsearch.client import create_elasticsearch_client
from shopmind.app.infrastructure.elasticsearch.index import switch_alias
from shopmind.app.infrastructure.elasticsearch.knowledge_index import create_knowledge_index, validate_knowledge_index
from shopmind.app.infrastructure.elasticsearch.knowledge_repository import ElasticsearchKnowledgeRepository
from shopmind.pipeline.knowledge_embedding_artifacts import load_knowledge_embedding_artifacts


def build_policy_index_documents(rows, document_ids, vectors, *, embedding_model: str, embedding_version: str):
    row_ids = [str(row["document_id"]) for row in rows]
    if row_ids != list(document_ids): raise ValueError("processed policy order must exactly match embedding document IDs")
    output=[]
    for row,vector in zip(rows,vectors,strict=True):
        meta=dict(row.get("metadata") or {}); section_id=str(row["document_id"]).removeprefix("policy:")
        output.append({"section_id":section_id,"policy_id":str(meta["policy_id"]),"title":str(row["title"]),"content":str(row["content"]),"search_text":str(row["search_text"]),"policy_type":str(meta["policy_type"]),"version":str(meta["version"]),"effective_date":str(meta["effective_date"]),"source_url":meta.get("source_url"),"text_vector":np.asarray(vector,dtype=np.float32).tolist(),"embedding_model":embedding_model,"embedding_version":embedding_version,"metadata":meta})
    return output

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True);parser.add_argument("--embeddings",type=Path,required=True);parser.add_argument("--index-name",required=True);parser.add_argument("--alias",default="policies");parser.add_argument("--config",type=Path,default=Path("configs/app.yaml"));parser.add_argument("--batch-size",type=int,default=500);parser.add_argument("--recreate",action="store_true");parser.add_argument("--switch-alias",action="store_true");args=parser.parse_args()
    if args.batch_size<=0: raise ValueError("batch-size must be positive")
    rows=[document.to_dict() for document in load_knowledge_documents(args.input,"policy")];manifest,document_ids,vectors=load_knowledge_embedding_artifacts(args.embeddings);documents=build_policy_index_documents(rows,document_ids,vectors,embedding_model=manifest.model,embedding_version=manifest.model_revision or "default")
    if not documents: raise ValueError("cannot build policy index from an empty document set")
    config=load_app_config(args.config);client=create_elasticsearch_client(config.elasticsearch);create_knowledge_index(client,args.index_name,"policy",manifest.dimension,recreate=args.recreate);repository=ElasticsearchKnowledgeRepository(client,args.index_name,source="policy");success,errors=0,[]
    for start in range(0,len(documents),args.batch_size): batch_success,batch_errors=repository.bulk_index(args.index_name,documents[start:start+args.batch_size]);success+=batch_success;errors.extend(batch_errors)
    if errors or success!=len(documents): raise RuntimeError(f"bulk indexing failed: {success}/{len(documents)}")
    client.indices.refresh(index=args.index_name);validate_knowledge_index(client,args.index_name,"policy",len(documents),manifest.dimension,documents[0]["title"],documents[0]["text_vector"])
    if args.switch_alias: switch_alias(client,alias=args.alias,target=args.index_name)
    print(f"indexed {success} policies into {args.index_name}")
if __name__=="__main__": main()
