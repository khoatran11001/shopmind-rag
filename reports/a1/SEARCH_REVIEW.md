# Search accuracy review — 2026-09-19

Validated against the local `products` alias (1,500 documents).

## Changes

- Multi-value exact filters now use case-insensitive OR clauses.
- Each filter value is resolved against the catalog before retrieval. An existing exact value stays exact even when the text query or a combination of filters returns no results. Only unknown values use partial matching. BM25 and kNN therefore resolve the same filters independently of their result counts.
- Product ID prefix values are literal; wildcard escaping applies only to wildcard queries.
- BM25 accepts exact product IDs in the main query, with a strong keyword boost, and boosts title phrase matches.
- Domain validation rejects filter-only dense/hybrid/cross-modal requests.
- UI ignores cancelled suggestion responses and disables empty submissions.

## Measurements

Compared the repository implementation at HEAD with the working implementation against the same live index. Existing benchmark queries and qrels were not changed.

| Check | Before | After |
|---|---:|---:|
| Exact ID top-1, first 100 product IDs sorted ascending, lowercase queries | 0/100 | 100/100 |
| BM25 Recall@10, 80 text queries | 0.462500 | 0.462500 |
| BM25 nDCG@10, 80 text queries | 0.356813 | 0.356813 |
| BM25 MRR@10, 80 text queries | 0.324861 | 0.324861 |

The ID check is a deterministic lookup regression, not a semantic relevance benchmark. No improvement in general semantic retrieval is claimed. Historical dense/image figures in RESULTS.md were not rerun in this review.

## Verification and remaining limitations

- Backend: 106 passed, 1 skipped (opt-in real model test). Live Elasticsearch integration includes lowercase multi-values, incompatible filter combinations, partial vector filters, literal wildcard input and product ID lookup.
- Frontend: 8 tests passed; production build passed.
- Existing Vietnamese benchmark contains repeated generic queries with different target labels; qrels still require independent human review. Tuning against these labels risks misleading gains.
- Resolving filters adds one Elasticsearch count request per distinct value. Measure latency before scaling beyond this small catalog.
- Dockerfile contains a pre-existing unrelated shell command after CMD, which prevents a normal Docker build. It was preserved, not executed. Runtime API/UI deployment and browser acceptance were not verified in this review.
- Text embeddings use dynamic padding and the manifest has no pinned model revision. Batch invariance and model/artifact compatibility should be tested with the real model before changing preprocessing or regenerating embeddings.
