from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
from shopmind.pipeline.review_normalizer import normalize_review
from shopmind.pipeline.knowledge_document import KnowledgeDocument
def _iter(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--products',type=Path,required=True); p.add_argument('--reviews',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists() and any(a.output.samefile(path) for path in (a.products, a.reviews) if path.exists()):
        p.error('--output must differ from --products and --reviews')
    ids={str(r['product_id']) for r in _iter(a.products)}; a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8') as out:
        for raw in _iter(a.reviews):
            review=normalize_review(raw,ids)
            if review:
                doc=KnowledgeDocument(f'review:{review.review_id}','review',review.title,review.text,review.search_text,{'review_id':review.review_id,'product_id':review.product_id,'asin':review.asin,'parent_asin':review.parent_asin,'matched_by':review.matched_by,'rating':review.rating,'verified_purchase':review.verified_purchase,'helpful_vote':review.helpful_vote,'timestamp':review.timestamp})
                out.write(json.dumps(doc.to_dict(),ensure_ascii=False,sort_keys=True)+'\n')
if __name__=='__main__': main()
