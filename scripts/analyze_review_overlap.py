from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
from shopmind.pipeline.review_normalizer import build_overlap_report
def _iter(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--products',type=Path,required=True); p.add_argument('--reviews',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    ids={str(r['product_id']) for r in _iter(a.products)}; report=build_overlap_report(ids,_iter(a.reviews)); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
