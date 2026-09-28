from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from shopmind.app.core.config import load_app_config
from shopmind.app.embedding.siglip2 import SigLIP2EmbeddingProvider
from shopmind.pipeline.knowledge_document import KnowledgeDocument
from shopmind.pipeline.policy_loader import load_policy_sections
from shopmind.pipeline.knowledge_embedding_artifacts import KnowledgeEmbeddingManifest,write_knowledge_embedding_artifacts
def load_knowledge_documents(path,source):
    path=Path(path)
    if source=='policy':return [KnowledgeDocument(s.citation_id,'policy',s.title,s.content,s.search_text,{'policy_id':s.policy_id,'policy_type':s.policy_type,'version':s.version,'effective_date':s.effective_date,'source_url':s.source_url,**s.metadata}) for s in load_policy_sections(path)]
    if source!='review':raise ValueError('source must be review or policy')
    return [KnowledgeDocument(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
def embed_knowledge_documents(documents,provider,dataset_version,batch_size,output_dir,force=False):
    ids=[d.document_id for d in documents];mats=[]
    for i in range(0,len(documents),batch_size):mats.append(provider.embed_texts([d.search_text for d in documents[i:i+batch_size]]))
    vectors=np.concatenate(mats,axis=0).astype(np.float32,copy=False) if mats else np.empty((0,provider.dimension),dtype=np.float32);manifest=KnowledgeEmbeddingManifest(provider.model_name,provider.model_revision,provider.dimension,dataset_version,len(ids),True,datetime.now(timezone.utc).isoformat().replace('+00:00','Z'));return write_knowledge_embedding_artifacts(output_dir,ids,vectors,manifest,force=force)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['review','policy'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--dataset-version',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/app.yaml'));p.add_argument('--force',action='store_true');a=p.parse_args();cfg=load_app_config(a.config);provider=SigLIP2EmbeddingProvider(model_name=cfg.embedding.model_name,hf_token=cfg.embedding.hf_token);docs=load_knowledge_documents(a.input,a.source);print(embed_knowledge_documents(docs,provider,a.dataset_version,cfg.embedding.batch_size,a.output_dir,a.force))
if __name__=='__main__':main()
