# ShopMind Multimodal Product Retrieval and A3 RAG

Python-first retrieval foundation for product search using Amazon Berkeley Objects (ABO), Elasticsearch BM25 + dense-vector kNN, SigLIP2 multimodal embeddings, Reciprocal Rank Fusion (RRF), FastAPI, and reproducible information-retrieval evaluation.

A1 stops at retrieval. A3 adds a separate answer layer using `RetrievedDocument[]`; product search remains available when RAG or its LLM is unavailable.

## Architecture

```text
ABO compact subset -> canonical products + local main images -> SigLIP2 text/image embeddings
                                                        |
                                                        v
                                             Elasticsearch products_vN
                                                /                 \
                                             BM25                 kNN
                                                \                 /
                                                 --- RRF hybrid ---
                                                         |
                                                   SearchService
                                                     /       \
                                                FastAPI   A3 RAG <- reviews + policies
                                                             |
                                                             v
                                                     answer + sources
```

Elasticsearch is the primary search store. Embeddings are generated offline and saved as reusable artifacts before any indexing step.

## Requirements

- Python 3.11+
- Docker + Docker Compose for local Elasticsearch
- Sufficient disk space for ~1,500 ABO product images and model cache
- Hugging Face model access if the configured SigLIP2 model requires it

The default A1 development dataset is **1,500 ABO products**, each with **one local main image** from the public ABO small-image collection (maximum dimension 256 px). The dataset builder is deterministic for the same seed and can be scaled to 1,000–10,000 products with `--limit`.

## A1 web interface

The React interface lives in the sibling `test-ui` repository. Start the complete local stack and open `http://localhost:5173`:

```bash
docker compose up -d --build api frontend
```

The UI compares two text retrieval modes side by side, supports Brand/Category/Product ID autocomplete filters, and provides SigLIP2 image search. Brand and Category can be selected multiple times; Product ID is a single-value lookup.

## Apply the current A1 workflow

Run from the repository root. The dataset, embedding artifacts, and `runs/` directory are local generated outputs and are intentionally not committed.

### 1. Install dependencies and build the runtime

```bash
python -m pip install -e '.[dev]'
docker compose up -d elasticsearch
docker compose build api frontend
```

The host Python 3.13/MPS combination can crash while loading SigLIP2. The commands below therefore run embedding and evaluation inside the pinned Docker runtime with CPU inference.

### 2. Rebuild the A1 dataset and index (only when needed)

Skip this section when `data/a1_abo_1500/` and `data/embeddings/abo_a1_1500_v1/` already exist.

```bash
python -m scripts.prepare_a1_abo_subset \
  --output data/a1_abo_1500 \
  --limit 1500 \
  --seed 42 \
  --max-per-category 150 \
  --workers 16 \
  --candidate-multiplier 1.35 \
  --force

docker compose run --rm -T --no-deps \
  -v "$PWD:/app" \
  api python -u -m scripts.generate_embeddings \
  --input /app/data/a1_abo_1500/products.jsonl \
  --output-dir /app/data/embeddings/abo_a1_1500_v1 \
  --config /app/configs/app.yaml \
  --dataset-version abo_a1_1500_v1 \
  --device cpu \
  --force

python -m scripts.create_index \
  --manifest data/embeddings/abo_a1_1500_v1/manifest.json \
  --index-name products_a1_1500_v1 \
  --recreate

python -m scripts.index_products \
  --products data/a1_abo_1500/products.jsonl \
  --embeddings data/embeddings/abo_a1_1500_v1 \
  --index-name products_a1_1500_v1 \
  --switch-alias products
```

`--force` and `--recreate` replace only the generated subset/artifact directory and the versioned index named above. The alias is switched only after document-count, BM25, and kNN validation.

### 3. Start API/UI and verify readiness

```bash
docker compose up -d api frontend
docker compose ps
until curl -fsS http://localhost:8000/ready; do sleep 5; done
curl http://localhost:9200/products/_count
curl http://localhost:9200/_alias/products
```

### 4. Generate the 100-query benchmark and run evaluation

```bash
python -m scripts.build_a1_benchmark \
  --products data/a1_abo_1500/products.jsonl \
  --output evaluation/benchmarks/a1 \
  --seed 42

docker compose run --rm -T --no-deps \
  -v "$PWD:/app" \
  -e ELASTICSEARCH_URL=http://elasticsearch:9200 \
  api sh -lc '
    python -m evaluation.ablation \
      --configs configs/experiments/bm25.yaml configs/experiments/dense.yaml \
        configs/experiments/cross_modal.yaml configs/experiments/hybrid_no_rrf.yaml \
        configs/experiments/hybrid_rrf.yaml \
      --queries evaluation/benchmarks/a1/queries.jsonl \
      --qrels evaluation/benchmarks/a1/qrels.jsonl \
      --query-type text \
      --runs-root runs/a1 &&
    python -m evaluation.runner \
      --config configs/experiments/image_dense.yaml \
      --queries evaluation/benchmarks/a1/queries.jsonl \
      --qrels evaluation/benchmarks/a1/qrels.jsonl \
      --query-type image \
      --runs-root runs/a1
  '
```

The latest report is [reports/a1/RESULTS.md](reports/a1/RESULTS.md), and the live demo commands are [reports/a1/DEMO.md](reports/a1/DEMO.md).

### 5. Run tests

```bash
pytest -q
pytest -m integration -q

cd ../test-ui
npm ci
npm test
npm run build
```

The default evaluation runner expects:

```text
data/evaluation/queries.jsonl
data/evaluation/qrels.jsonl
```

You can override these with `--queries` and `--qrels`.

## Compact A1 dataset

`scripts.prepare_a1_abo_subset` downloads a deterministic subset directly from the public Amazon Berkeley Objects dataset and writes:

```text
data/a1_abo_1500/
├── products.jsonl
├── image_manifest.jsonl
├── DATASET.md
└── images/
    ├── <product_id>.jpg
    └── ...
```

`products.jsonl` is already in the canonical schema expected by `scripts.generate_embeddings` and `scripts.index_products`, so the compact flow does **not** require a separate `download_abo` or `preprocess` step.

Each product contains:

```text
product_id
title
description
brand
category
attributes
image_paths
main_image_path
search_text
metadata
```

`main_image_path` points to the downloaded local image, while `metadata.image_url` preserves the original ABO small-image URL for traceability. Indexed results expose the public path `/media/products/<filename>`; Docker mounts the compact image directory at that path. Only successfully downloaded images are included in the final subset, so a 1,500-record dataset has 1,500 usable main images.

For a larger local experiment, change only the limit, for example:

```bash
python -m scripts.prepare_a1_abo_subset \
  --output data/a1_abo_5000 \
  --limit 5000 \
  --seed 42 \
  --workers 16
```

## Embedding artifacts

A generated artifact directory contains:

```text
product_ids.json
text_embeddings.npy
image_embeddings.npy
manifest.json
```

The manifest records model, revision, dimension, dataset version, product count, normalization status, and creation timestamp. Image rows are unit-normalized and every A1 record must have a usable main image. Embedding generation and Elasticsearch indexing are intentionally separate, so re-indexing does not require re-running SigLIP2.

Only the **main product image** is embedded in A1.

## Elasticsearch index versioning

Never treat a physical index name as the public contract.

```text
products_v1  <- physical version
products_v2  <- next version
products     <- stable alias used by the application
```

Create and validate a versioned index first. Switch the `products` alias only after document count, vector dimensions, BM25 smoke search, and kNN smoke search succeed. Alias updates are atomic.

## Text retrieval modes

`POST /api/v1/search/text` supports these four public modes:

| Mode | Behavior |
|---|---|
| `bm25` | Elasticsearch lexical BM25 over boosted product text fields |
| `dense` | query text embedding -> `text_vector` kNN |
| `cross_modal` | query text embedding -> `image_vector` kNN |
| `hybrid` | BM25 + dense candidate retrieval -> RRF -> optional reranker boundary |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "black running shoes",
    "mode": "hybrid",
    "top_k": 10,
    "candidate_k": 100,
    "filters": {"category": "Shoes"}
  }'
```

`query` may be omitted when at least one filter is supplied. Filter-only searches use `bm25` because dense and cross-modal modes require a text embedding:

```bash
curl -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "bm25",
    "filters": {
      "brand": ["AmazonBasics", "Rivet"],
      "category": ["SHOES"]
    }
  }'
```

The public filter fields are `brand`, `category`, and `product_id`. Values within Brand or Category are combined with OR; different fields are combined with AND. Each value is matched case-insensitively against the catalog first. Only values absent from the catalog use substring matching for Brand/Category or prefix matching for Product ID. Existing exact filters are never broadened merely because a query or filter combination has no results. BM25 also supports an exact Product ID in the main query.

Autocomplete values come from the catalog:

```bash
curl 'http://localhost:8000/api/v1/search/filter-options?field=brand&prefix=amaz&limit=10'
```

`GET /api/v1/search/filter-options` accepts `field=brand|category|product_id`, an optional `prefix`, and `limit` from 1 to 50. The UI uses the canonical option value when a suggestion is selected.

Public results contain `product_id`, `title`, `image_url`, `brand`, `category`, `score`, and `rank`. Internal component scores/ranks remain an application/research concern and are not exposed by default.

Requests without a query and without filters return HTTP `422`. Filter-only requests with a non-BM25 mode return HTTP `400`.

## Image retrieval

`POST /api/v1/search/image` accepts a multipart image upload. The API validates content type and byte size, decodes the file with Pillow, converts it to RGB, creates an image embedding, then searches `image_vector`.

```bash
curl -X POST 'http://localhost:8000/api/v1/search/image?top_k=10&candidate_k=100&category=SHOES&category=BOOT&brand=AmazonBasics' \
  -F 'image=@query.png;type=image/png'
```

Image filters use repeated `brand` and `category` query parameters when multiple values are selected; `product_id` remains a single-value filter.

Image bytes and embedding vectors are never written to structured search logs.

## Health and readiness

```text
GET /health  -> process liveness only
GET /ready   -> Elasticsearch ping + products alias + embedding provider readiness
```

A dependency failure returns `503`; liveness can remain healthy while readiness fails.

## Evaluation

The reproducible A1 benchmark is generated with seed 42 under `evaluation/benchmarks/a1/` and contains 80 text queries (45 English, 35 Vietnamese) plus 20 transformed image queries. Its README records the deterministic construction; human second-review and adjudication of qrels remain a required reporting step.

Evaluation inputs use JSONL.

`queries.jsonl`:

```json
{"query_id":"q1","query":"black running shoes"}
```

`qrels.jsonl`:

```json
{"query_id":"q1","product_id":"P1","relevance":2}
{"query_id":"q1","product_id":"P2","relevance":1}
```

Core metrics are:

- Recall@10
- nDCG@10
- MRR@10

Every experiment writes a self-contained run directory:

```text
runs/<timestamp>_<experiment>/
  config.yaml
  metrics.json
  results.jsonl
  errors.jsonl
  summary.md
```

The saved configuration includes dataset version, index version, embedding metadata, retrieval mode, candidate window, fusion settings, and reranker state.

The latest local A1 results and the license/qrels review status are summarized in [`reports/a1/RESULTS.md`](reports/a1/RESULTS.md); live smoke commands are in [`reports/a1/DEMO.md`](reports/a1/DEMO.md).

## Ablation

Core comparison matrix:

1. BM25 only
2. dense text only
3. cross-modal SigLIP2 text -> image
4. BM25-first + unseen dense candidate union without RRF (`hybrid_no_rrf`, evaluation-only)
5. BM25 + dense + RRF (`hybrid_rrf`)

`hybrid_no_rrf` is deliberately **not** a public API `SearchMode`; it exists only to isolate the contribution of RRF in evaluation.

Representative failure records use evidence-based categories such as lexical mismatch, semantic confusion, category confusion, wrong visual product type, missing metadata, poor image quality, ambiguous query, or `uncategorized` for manual review.

## Testing

Fast tests do not require Elasticsearch or a real model:

```bash
pytest tests/unit tests/api tests/evaluation -q
```

With Elasticsearch running:

```bash
pytest tests/integration -m integration -q
```

Real SigLIP2 smoke testing is opt-in because it can download a large model:

```bash
RUN_SLOW_MODEL_TESTS=1 pytest -m slow -q
```

## A3 product questions

The A3 API searches three evidence sources: the existing A1 product index, matched Amazon review records, and the versioned **experimental** ShopMind policies in `data/policies/shopmind_policies.jsonl`. No Amazon review data is bundled. Policies are examples for this prototype, not real store terms.

Prepare reviews from a supplied Amazon Reviews JSONL or JSONL.gz file. Review matching uses exact `asin` and then exact `parent_asin`; check the overlap report before indexing. Use the 1,500-product A1 catalog, not `data/processed/products.jsonl` (the four-row sample).

```bash
python -m scripts.analyze_review_overlap \
  --products data/a1_abo_1500/products.jsonl \
  --reviews /path/to/reviews.jsonl.gz \
  --output reports/a3/overlap.json
python -m scripts.prepare_reviews \
  --products data/a1_abo_1500/products.jsonl \
  --reviews /path/to/reviews.jsonl.gz \
  --output data/processed/reviews.jsonl
```

Generate embeddings with the configured SigLIP2 model, then index the two knowledge sources. The embedding commands use the Docker CPU runtime to avoid the local Python 3.13/MPS crash.

```bash
docker compose run --rm -T --no-deps -v "$PWD:/app" api \
  python -m scripts.generate_knowledge_embeddings --source review \
  --input data/processed/reviews.jsonl --dataset-version reviews_a3_v1 \
  --output-dir data/embeddings/reviews_a3_v1
docker compose run --rm -T --no-deps -v "$PWD:/app" api \
  python -m scripts.generate_knowledge_embeddings --source policy \
  --input data/policies/shopmind_policies.jsonl --dataset-version policies_a3_v1 \
  --output-dir data/embeddings/policies_a3_v1
python -m scripts.index_reviews --input data/processed/reviews.jsonl \
  --embeddings data/embeddings/reviews_a3_v1 --index-name reviews_a3_v1 \
  --switch-alias
python -m scripts.index_policies --input data/policies/shopmind_policies.jsonl \
  --embeddings data/embeddings/policies_a3_v1 --index-name policies_a3_v1 \
  --switch-alias
```

Each index command checks document count, embedding dimension, BM25, and vector search before switching its alias. To generate answers, install the package dependencies and set `OPENAI_API_KEY` in the API environment. Cross-encoder reranking is optional: install `pip install -e '.[rag-rerank]'` and set `rag.reranker.enabled: true` in `configs/app.yaml` when needed.

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What do reviews say about this item?", "product_id":"B07PH3GSND", "sources":["product","review","policy"]}'
```

The response has `answer`, `sources` (`document_id`, source, title, final score, product ID or policy version where applicable), and `insufficient_evidence`. The source metadata identifies whether `score` is a source score, RRF score, or reranker score; scores from different modes are not directly comparable. Source IDs must come from the exact context sent to the LLM; the server resolves titles, URLs, and scores, and allows one retry for an invalid citation. `product_id` restricts product and review evidence to an exact catalog ID. If an index exists but returns no evidence, the API abstains; if a requested source index is missing, it returns `503`. Requests can set `sources: ["product"]` before reviews are indexed. Without an API key, generation returns `503` while A1 search and `/ready` remain usable. The documented [OpenAI Responses API structured output format](https://developers.openai.com/api/docs/guides/structured-outputs) is used for the answer contract.

Build an unreviewed seed evaluation set after review preparation, then run the six A3 configurations. They compare product-only and all-source evidence, BM25/dense/hybrid retrieval, cross-source RRF, and optional reranking. Outputs under `runs/a3/` include case results, failure categories (candidate, fusion, citation, abstention, runtime), citation/retrieval metrics, and p50/p95 latency. Answer correctness stays `null` until a grader or human review is supplied.

```bash
python -m scripts.build_rag_evaluation_set \
  --reviews data/processed/reviews.jsonl \
  --output data/evaluation/rag_cases.jsonl
python -m scripts.run_rag_experiment \
  --configs configs/experiments/rag/*.yaml \
  --cases data/evaluation/rag_cases.jsonl
```
