# ShopMind A1 retrieval results

Run date: 2026-09-17. Dataset: `abo_a1_1500_v1` (1,500 ABO products, 215 categories, maximum 150/category). Index: `products_a1_1500_v1`. Model: `google/siglip2-base-patch16-224`, 768-dimensional unit vectors.

The reproducible run directories are under the local, ignored `runs/a1/` directory. The latest controlled text comparison is `runs/a1/ablation_20260917T165156Z/`; the image baseline is `runs/a1/20260917T165233Z_image_dense/`.

## Controlled text ablation (80 text queries)

| Run | Recall@10 | nDCG@10 | MRR@10 | p50 ms | p95 ms | Failed |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.462500 | 0.356813 | 0.324861 | 25.86 | 44.82 | 0 |
| Dense text | 0.012500 | 0.004167 | 0.001786 | 51.28 | 96.68 | 0 |
| Cross-modal text → image | 0.012500 | 0.003613 | 0.001250 | 40.56 | 59.53 | 0 |
| BM25 + dense, no RRF | 0.462500 | 0.356813 | 0.324861 | 67.04 | 102.53 | 0 |
| BM25 + dense + RRF | 0.325000 | 0.184633 | 0.141404 | 65.44 | 93.54 | 0 |

The result is a diagnostic baseline, not a claim that fusion is beneficial for every query distribution. BM25 uses lexical matching; dense uses cosine kNN; RRF combines component ranks with `1 / (60 + rank)`. Per-query component ranks/scores are in each `results.jsonl`.

Hybrid-RRF breakdown: English Recall@10 `0.422222`, Vietnamese `0.200000`; exact `0.360000`, semantic `0.142857`, hard-negative `0.350000`.

## Image baseline (20 transformed image queries)

Image kNN achieved Recall@10 `1.000000`, nDCG@10 `1.000000`, MRR@10 `1.000000`, p50 `140.82 ms`, p95 `230.01 ms`, with zero failed queries. Queries are crop/resize/contrast/JPEG transformations and no query file is byte-identical to its catalog image.

## Error analysis and data note

The latest ablation export contains the 20 lowest-nDCG cases in `runs/a1/ablation_20260917T165156Z/error-cases.jsonl`, with suggested categories (lexical mismatch and semantic confusion) plus an empty reviewer field. The benchmark qrels in `evaluation/benchmarks/a1/` are deterministic seed labels; second-member review and adjudication still must be completed before describing them as human-reviewed.

ABO's official download page states CC BY 4.0 while the AWS Registry states CC BY-NC 4.0. This coursework applies the conservative CC BY-NC 4.0 interpretation and attributes Amazon.com and the ABO authors.
