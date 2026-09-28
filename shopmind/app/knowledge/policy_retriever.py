from shopmind.app.domain.search_result import RetrievedDocument
from shopmind.app.knowledge.base import KnowledgeMode, rrf_knowledge_hits
class PolicyDocumentRetriever:
    def __init__(self,repository,embedder,*,mode="hybrid",candidate_k=40,rrf_k=60): self.repository=repository; self.embedder=embedder; self.mode=KnowledgeMode(mode); self.candidate_k=candidate_k; self.rrf_k=rrf_k
    def retrieve(self,query,top_k,product_id=None):
        del product_id
        lex=self.repository.lexical_search(query,self.candidate_k) if self.mode in {KnowledgeMode.BM25,KnowledgeMode.HYBRID} else []
        dense=[]
        if self.mode in {KnowledgeMode.DENSE,KnowledgeMode.HYBRID}:
            vec=self.embedder.embed_texts([query])[0].astype("float32").tolist(); dense=self.repository.vector_search(vec,self.candidate_k,self.candidate_k)
        ranked=[(h,h.score) for h in (lex if self.mode==KnowledgeMode.BM25 else dense)]
        if self.mode==KnowledgeMode.HYBRID: ranked=rrf_knowledge_hits([lex,dense],k=self.rrf_k,top_k=self.candidate_k)
        return [RetrievedDocument(f"policy:{h.document_id}",str(h.metadata.get("content") or h.title),float(score),"policy",{**h.metadata,**(h.metadata.get("metadata") or {}),"title":h.title,"source_local_score":h.score}) for h,score in ranked[:top_k]]
