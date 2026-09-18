# ShopMind A1 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn tất A1 thành một hệ multimodal product retrieval chạy trên 1.500 sản phẩm ABO thật, có text/image search qua API + UI, evaluation tái lập được, controlled ablation, error analysis và bằng chứng sẵn để đưa vào báo cáo.

**Architecture:** Giữ nguyên FastAPI + Elasticsearch + SigLIP2 + React hiện có. Chỉ sửa đường đi dữ liệu và evaluation: tạo subset cân bằng hơn, thống nhất URL ảnh, mở rộng runner để chạy cả text và image query, rồi xuất một gói kết quả A1 có version rõ ràng.

**Tech Stack:** Python 3.11+, FastAPI, Elasticsearch 8.15, Transformers/SigLIP2, NumPy, Pillow, pytest, React/Vite.

**Spec:** [README.md](/Users/yuji/research/rag/test/README.md) và session yêu cầu môn học: https://chatgpt.com/share/6a9e237a-c8c8-83ec-af1f-5bc95a9dbc9c?ogimg=plain

## Global Constraints

- Chỉ làm A1; không thêm LLM, RAG, agent, reranker thật, PostgreSQL/pgvector hoặc vector database thứ hai.
- Public demo phải hỗ trợ text query và image upload, gọi API và trả top-k ảnh sản phẩm.
- Ablation phải giữ nguyên dataset, qrels, top-k và model; mỗi comparison chỉ thay đổi một thành phần.
- Dataset mặc định là 1.500 sản phẩm, một main image mỗi sản phẩm, seed `42`.
- Model mặc định là `google/siglip2-base-patch16-224`; embedding sinh offline và được version hóa.
- Vì hai trang chính thức của ABO ghi license khác nhau, báo cáo phải ghi nhận cả `CC BY 4.0` và `CC BY-NC 4.0`, đồng thời áp dụng điều kiện bảo thủ `CC BY-NC 4.0` cho bài tập.

---

## Audit hiện trạng

| Yêu cầu A1 | Hiện trạng | Kết luận |
|---|---|---|
| Text search: BM25, dense, hybrid/RRF | Code + API + UI đã có | Giữ nguyên |
| Image search | Code + API + UI đã có | Cần chạy với ảnh thật và thêm evaluation |
| Top-k ảnh qua frontend/API | Contract đã có | Đường dẫn ảnh compact hiện không khớp `/media/products` |
| Dataset nhỏ, tái lập được | Builder đã có | Local `products.jsonl` đang là schema cũ; subset lệch mạnh về phone case |
| Embedding/index thật | Pipeline đã có | Artifact và Elasticsearch hiện chỉ có 4 record fixture |
| Evaluation | Metrics/runner đã có | Chưa có `data/evaluation`; runner chỉ chạy text query |
| Controlled ablation | 5 config + comparison đã có | Chưa có run thật; version trong config không khớp dataset/index dự kiến |
| Error analysis | Helper phân loại đã có | Chưa nối vào run thật, chưa có case đã review |
| Test/reproducibility | 75 unit/evaluation tests và 2 integration tests pass riêng | Full command fail collect API tests vì `tests` chưa là package cục bộ |

## Không làm trong A1

- Không triển khai `/ask`, RAG, review/policy retrieval hoặc A2 model benchmark.
- Không thêm `/debug/search`; giải thích score bằng công thức trong report và component ranks trong artifact evaluation là đủ.
- Không tăng lên 5.000–10.000 sản phẩm trước khi pipeline 1.500 sản phẩm chạy end-to-end.
- Không thêm charting dependency; CSV/Markdown và bảng trong report đủ cho A1.

---

### Task 1: Khóa baseline kiểm thử và version A1

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/api/__init__.py`
- Modify: `configs/experiments/bm25.yaml`
- Modify: `configs/experiments/dense.yaml`
- Modify: `configs/experiments/cross_modal.yaml`
- Modify: `configs/experiments/hybrid_no_rrf.yaml`
- Modify: `configs/experiments/hybrid_rrf.yaml`
- Create: `configs/experiments/image_dense.yaml`

**Interfaces:**
- Consumes: pytest discovery và các fake trong `tests/api/conftest.py`.
- Produces: một lệnh test duy nhất chạy được; mọi config dùng `dataset_version: abo_a1_1500_v1` và `index_version: products_a1_1500_v1`; image evaluation có config riêng.

- [ ] **Step 1: Ghi nhận lỗi collect hiện tại**

Run:

```bash
pytest tests/unit tests/api tests/evaluation -q
```

Expected: FAIL khi import `tests.api.conftest`.

- [ ] **Step 2: Tạo hai package marker rỗng**

```python
# tests/__init__.py và tests/api/__init__.py không cần nội dung.
```

- [ ] **Step 3: Đồng bộ version trong cả năm experiment config**

```yaml
experiment:
  dataset_version: abo_a1_1500_v1
  index_version: products_a1_1500_v1
```

Tạo `configs/experiments/image_dense.yaml`:

```yaml
experiment:
  name: image_dense
  dataset_version: abo_a1_1500_v1
  index_version: products_a1_1500_v1
retrieval:
  mode: image_dense
  top_k: 10
  candidate_k: 100
fusion:
  method: none
  rrf_k: 60
embedding:
  model: google/siglip2-base-patch16-224
  version: default
reranker:
  enabled: false
```

- [ ] **Step 4: Chạy toàn bộ fast test**

Run:

```bash
pytest tests/unit tests/api tests/evaluation -q
```

Expected: PASS, API tests được collect.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/api/__init__.py configs/experiments
git commit -m "test: stabilize A1 checks and experiment versions"
```

---

### Task 2: Tạo subset ABO phù hợp để đánh giá và thống nhất ảnh hiển thị

**Files:**
- Modify: `scripts/prepare_a1_abo_subset.py`
- Modify: `scripts/index_products.py`
- Modify: `docker-compose.yml`
- Modify: `/Users/yuji/research/rag/test-ui/src/api.js`
- Modify: `tests/unit/test_prepare_a1_abo_subset.py`
- Create: `tests/unit/test_index_products.py`
- Modify: `/Users/yuji/research/rag/test-ui/src/api.test.js`

**Interfaces:**
- Consumes: ABO listing/image metadata.
- Produces: canonical `products.jsonl`, tối đa 150 sản phẩm/category, và `image_url` dạng `/media/products/<filename>`.

- [ ] **Step 1: Viết test fail cho giới hạn category và media URL**

```python
def test_select_balanced_subset_caps_each_category():
    rows = [{"product_id": f"p{i}", "category": "PHONE_CASE"} for i in range(4)]
    rows += [{"product_id": "shoe-1", "category": "SHOES"}]
    selected = select_balanced_subset(rows, limit=3, max_per_category=2, seed=42)
    assert [row["product_id"] for row in selected] == ["p3", "p1", "shoe-1"]


def test_index_document_uses_public_media_url():
    row = {"product_id": "P1", "title": "Shoe", "main_image_path": "data/a1_abo_1500/images/P1.jpg"}
    assert public_image_url(row["main_image_path"]) == "/media/products/P1.jpg"
```

```javascript
test('resolveImageUrl accepts backend media URLs', () => {
  assert.equal(resolveImageUrl('/media/products/P1.jpg'), '/media/products/P1.jpg')
})
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run:

```bash
pytest tests/unit/test_prepare_a1_abo_subset.py tests/unit/test_index_products.py -q
npm test --prefix /Users/yuji/research/rag/test-ui
```

Expected: FAIL vì chưa có balanced selector/public URL support.

- [ ] **Step 3: Thêm lựa chọn round-robin theo category**

```python
def select_balanced_subset(rows, *, limit: int, max_per_category: int, seed: int):
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    counts: Counter[str] = Counter()
    selected = []
    for row in shuffled:
        category = str(row.get("category") or "UNKNOWN")
        if counts[category] >= max_per_category:
            continue
        selected.append(row)
        counts[category] += 1
        if len(selected) == limit:
            return selected
    raise RuntimeError(f"balanced candidate pool produced {len(selected)}/{limit} products")
```

Thêm CLI `--max-per-category 150`. Đọc pool metadata `limit * 5`, chọn trước khoảng `limit * candidate_multiplier` candidate cân bằng để tránh tải hàng nghìn ảnh thừa, sau download mới áp dụng cap 150 lần cuối để ghi đúng 1.500 record; giữ seed `42`.

- [ ] **Step 4: Public hóa ảnh bằng một contract duy nhất**

```python
def public_image_url(main_image_path: str | None) -> str | None:
    return f"/media/products/{Path(main_image_path).name}" if main_image_path else None
```

Trong `docker-compose.yml`, mount `./data/a1_abo_1500/images` vào `/app/data/a1_abo_1500/images` và đặt `PRODUCT_IMAGE_DIR=/app/data/a1_abo_1500/images`. Trong UI, trả nguyên URL bắt đầu bằng `/media/products/`.

- [ ] **Step 5: Ghi license bảo thủ vào `DATASET.md` do builder tạo**

```markdown
- License note: the official ABO download page states CC BY 4.0, while the AWS Registry states CC BY-NC 4.0. This coursework uses the conservative CC BY-NC 4.0 interpretation and includes attribution.
```

- [ ] **Step 6: Chạy checks**

Run:

```bash
pytest tests/unit/test_prepare_a1_abo_subset.py tests/unit/test_index_products.py -q
npm test --prefix /Users/yuji/research/rag/test-ui
npm run build --prefix /Users/yuji/research/rag/test-ui
```

Expected: PASS.

- [ ] **Step 7: Commit backend và frontend riêng**

```bash
git add scripts/prepare_a1_abo_subset.py scripts/index_products.py docker-compose.yml tests/unit
git commit -m "fix: make the A1 dataset and image contract reproducible"
git -C /Users/yuji/research/rag/test-ui add src/api.js src/api.test.js
git -C /Users/yuji/research/rag/test-ui commit -m "fix: render compact A1 product images"
```

---

### Task 3: Mở rộng evaluation cho text và image query

**Files:**
- Modify: `evaluation/datasets.py`
- Modify: `evaluation/runner.py`
- Modify: `evaluation/reports.py`
- Modify: `configs/experiments/image_dense.yaml`
- Modify: `tests/unit/test_evaluation_datasets.py`
- Modify: `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: JSONL query có `query_type: text|image`; image query có `image_path`; experiment mode `image_dense` chỉ nhận image query.
- Produces: cùng một schema result/metric cho cả hai loại, kèm `latency_ms`, p50 và p95.

- [ ] **Step 1: Viết test fail cho image query**

```python
def test_load_queries_accepts_text_and_image(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query_id":"t1","query_type":"text","query":"black shoes"}\n'
        '{"query_id":"i1","query_type":"image","image_path":"images/i1.jpg"}\n'
    )
    queries = load_queries(path)
    assert [(q.query_id, q.query_type) for q in queries] == [("t1", "text"), ("i1", "image")]
```

```python
def test_run_experiment_dispatches_image_query_and_records_latency(tmp_path):
    summary = run_experiment(config, service, queries, qrels, tmp_path / "runs")
    metrics = json.loads((summary.run_dir / "metrics.json").read_text())
    assert metrics["query_types"] == {"image": 1, "text": 1}
    assert metrics["latency_ms"]["p95"] >= metrics["latency_ms"]["p50"]
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run:

```bash
pytest tests/unit/test_evaluation_datasets.py tests/evaluation/test_runner.py -q
```

Expected: FAIL vì `EvaluationQuery` hiện chỉ có text.

- [ ] **Step 3: Mở rộng query model tối thiểu**

```python
@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query_type: Literal["text", "image"]
    query: str | None = None
    image_path: Path | None = None
```

Validation: text query phải có `query`; image query phải có file tồn tại và Pillow decode được.

- [ ] **Step 4: Dispatch đúng service method và lưu component evidence**

```python
if query.query_type == "text":
    results = service.search_text(request)
else:
    with Image.open(query.image_path) as image:
        results = service.search_image(image.convert("RGB"), top_k=top_k, candidate_k=candidate_k)
```

Mỗi result row lưu `source` và `retrieval_scores` ngoài `product_id`, `rank`, `score`; mỗi query row lưu `latency_ms`.

Config parser chấp nhận `image_dense` như evaluation-only mode. Nếu config `image_dense` gặp text query, hoặc config text gặp image query khi dùng `--query-type`, fail trước khi tạo run directory.

- [ ] **Step 5: Aggregate riêng và chung**

`metrics.json` phải có metrics tổng, metrics theo `text`/`image`, query count theo type, `latency_ms.p50` và `latency_ms.p95`. CLI có `--query-type text|image` để ablation text không bị trộn với image baseline. Không thêm thư viện: dùng `statistics` + NumPy đã cài.

- [ ] **Step 6: Chạy checks**

Run:

```bash
pytest tests/unit/test_evaluation_datasets.py tests/evaluation/test_runner.py tests/evaluation/test_ablation.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evaluation tests/unit/test_evaluation_datasets.py tests/evaluation
git commit -m "feat: evaluate A1 text and image retrieval"
```

---

### Task 4: Tạo benchmark A1 nhỏ nhưng đủ bảo vệ

**Files:**
- Create: `evaluation/benchmarks/a1/queries.jsonl`
- Create: `evaluation/benchmarks/a1/qrels.jsonl`
- Create: `evaluation/benchmarks/a1/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: deterministic `abo_a1_1500_v1` product IDs/images.
- Produces: 60 query gồm 40 text và 20 image; qrels graded `0..2` được hai thành viên review chéo.

- [ ] **Step 1: Rebuild canonical dataset**

Run:

```bash
python -m scripts.prepare_a1_abo_subset --output data/a1_abo_1500 --limit 1500 --seed 42 --workers 16 --candidate-multiplier 1.35 --max-per-category 150 --force
```

Expected: 1.500 canonical records, 1.500 ảnh, không category nào vượt 150 record.

- [ ] **Step 2: Chọn query set cố định**

Tạo đúng:

- 10 exact/entity queries (brand, product type, color/material).
- 20 semantic queries (ý định sử dụng, synonym, mô tả thuộc tính).
- 10 hard-negative queries (mã/thuộc tính gần giống, category dễ nhầm).
- 20 image queries dùng ảnh đã resize/crop nhẹ, không dùng byte-identical catalog image.

Mỗi dòng dùng schema:

```json
{"query_id":"t001","query_type":"text","query":"black leather wallet with RFID protection","group":"semantic"}
{"query_id":"i001","query_type":"image","image_path":"evaluation/benchmarks/a1/images/i001.jpg","group":"image"}
```

- [ ] **Step 3: Gán qrels có review chéo**

```json
{"query_id":"t001","product_id":"B083RBSL56","relevance":2}
```

Quy ước: `2 = đúng trực tiếp`, `1 = chấp nhận được`, `0 = hard negative đã xét`. Người viết query gán lần đầu; một thành viên khác review; bất đồng được ghi trong `README.md` và chốt bằng đồng thuận.

- [ ] **Step 4: Validate benchmark**

Run:

```bash
python - <<'PY'
from evaluation.datasets import load_qrels, load_queries
q = load_queries('evaluation/benchmarks/a1/queries.jsonl')
r = load_qrels('evaluation/benchmarks/a1/qrels.jsonl', q)
assert len(q) == 60
assert sum(x.query_type == 'text' for x in q) == 40
assert sum(x.query_type == 'image' for x in q) == 20
assert all(any(v > 0 for v in r[x.query_id].values()) for x in q)
print('A1 benchmark valid')
PY
```

Expected: `A1 benchmark valid`.

- [ ] **Step 5: Commit**

```bash
git add evaluation/benchmarks/a1 README.md
git commit -m "data: add reviewed A1 retrieval benchmark"
```

---

### Task 5: Chạy pipeline thật và xuất ablation/error analysis

**Files:**
- Modify: `evaluation/ablation.py`
- Modify: `evaluation/error_analysis.py`
- Create: `reports/a1/README.md`
- Create: `reports/a1/comparison.csv`
- Create: `reports/a1/comparison.md`
- Create: `reports/a1/error-cases.jsonl`
- Create: `reports/a1/metrics.json`
- Modify: `tests/evaluation/test_ablation.py`
- Modify: `tests/unit/test_error_analysis.py`

**Interfaces:**
- Consumes: real index `products_a1_1500_v1`, benchmark A1 và năm experiment configs.
- Produces: bảng ablation, metrics/latency, ít nhất 15 failure case đã review và kết luận có bằng chứng.

- [ ] **Step 1: Viết test fail cho export failure cases**

```python
def test_ablation_exports_worst_queries_for_review(tmp_path):
    summary = run_ablation(paths, service, queries, qrels, tmp_path / "runs")
    rows = [json.loads(line) for line in (summary.comparison_dir / "error-cases.jsonl").read_text().splitlines()]
    assert rows
    assert {"query_id", "query_type", "suggested_category", "reviewer_note"} <= rows[0].keys()
```

- [ ] **Step 2: Chạy test để xác nhận fail, rồi nối exporter hiện có vào ablation**

Run:

```bash
pytest tests/evaluation/test_ablation.py tests/unit/test_error_analysis.py -q
```

Expected trước sửa: FAIL vì comparison chưa xuất error cases. Expected sau sửa: PASS.

Thêm CLI mỏng vào `evaluation/ablation.py`: nhận `--configs`, `--queries`, `--qrels`, `--runs-root`, `--query-type`; tái sử dụng `_wire_runtime`, `load_queries`, `load_qrels` và `run_ablation`, không tạo orchestration layer mới.

- [ ] **Step 3: Sinh embedding thật**

Run:

```bash
python -m scripts.generate_embeddings --input data/a1_abo_1500/products.jsonl --output-dir data/embeddings/abo_a1_1500_v1 --config configs/app.yaml --dataset-version abo_a1_1500_v1 --device auto --force
```

Expected: manifest có `product_count: 1500`; text/image matrices có 1.500 non-zero rows.

- [ ] **Step 4: Tạo index versioned và switch alias**

Run:

```bash
python -m scripts.create_index --manifest data/embeddings/abo_a1_1500_v1/manifest.json --index-name products_a1_1500_v1
python -m scripts.index_products --products data/a1_abo_1500/products.jsonl --embeddings data/embeddings/abo_a1_1500_v1 --index-name products_a1_1500_v1 --switch-alias products
curl -fsS 'http://localhost:9200/products/_count'
```

Expected: alias `products` trỏ tới `products_a1_1500_v1`, count bằng `1500`.

- [ ] **Step 5: Chạy controlled ablation**

Run:

```bash
python -m evaluation.ablation \
  --configs configs/experiments/bm25.yaml configs/experiments/dense.yaml configs/experiments/cross_modal.yaml configs/experiments/hybrid_no_rrf.yaml configs/experiments/hybrid_rrf.yaml \
  --queries evaluation/benchmarks/a1/queries.jsonl \
  --qrels evaluation/benchmarks/a1/qrels.jsonl \
  --query-type text \
  --runs-root runs/a1
```

Expected: cả năm run dùng đúng 40 text query và không có failed query; comparison chứa Recall@10, nDCG@10, MRR@10, delta so với BM25 và latency p50/p95.

- [ ] **Step 6: Chạy image retrieval baseline riêng**

Run:

```bash
python -m evaluation.runner \
  --config configs/experiments/image_dense.yaml \
  --queries evaluation/benchmarks/a1/queries.jsonl \
  --qrels evaluation/benchmarks/a1/qrels.jsonl \
  --query-type image \
  --runs-root runs/a1
```

Expected: 20/20 image query chạy thành công; artifact báo mode `image_dense` và metrics riêng cho image retrieval.

- [ ] **Step 7: Review error cases**

Chọn ít nhất 15 query có nDCG@10 thấp nhất, điền `reviewer_note`, và dùng đúng taxonomy hiện có: lexical mismatch, semantic confusion, category confusion, wrong visual product type, missing metadata, poor image quality, ambiguous query hoặc uncategorized.

- [ ] **Step 8: Copy artifact nộp bài vào `reports/a1`**

Chỉ copy summary files; không commit embeddings, raw images, Elasticsearch data hoặc toàn bộ `runs/`.

- [ ] **Step 9: Commit**

```bash
git add evaluation tests reports/a1
git commit -m "eval: publish A1 ablation and error analysis"
```

---

### Task 6: Kiểm tra demo và đóng gói bằng chứng A1

**Files:**
- Modify: `README.md`
- Modify: `/Users/yuji/research/rag/test-ui/README.md`
- Create: `reports/a1/demo-checklist.md`

**Interfaces:**
- Consumes: backend/index/report hoàn tất.
- Produces: quy trình demo 5 phút chạy lại được trên máy khác.

- [ ] **Step 1: Chạy toàn bộ automated checks**

```bash
pytest -q
npm test --prefix /Users/yuji/research/rag/test-ui
npm run build --prefix /Users/yuji/research/rag/test-ui
```

Expected: PASS; slow model test chỉ skip nếu không bật `RUN_SLOW_MODEL_TESTS=1`.

- [ ] **Step 2: Start stack và kiểm tra readiness**

```bash
docker compose up --build -d
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
curl -fsS http://localhost:9200/products/_count
```

Expected: health/ready OK và count `1500`.

- [ ] **Step 3: Manual demo checklist**

`reports/a1/demo-checklist.md` phải xác nhận:

1. BM25 thắng ở một exact query.
2. Dense hoặc hybrid thắng ở một semantic query.
3. Cross-modal trả ảnh hợp nghĩa cho một text query.
4. Image upload trả top-k ảnh và tất cả thumbnail load được.
5. UI so sánh cùng query giữa hai mode.
6. Có thể giải thích BM25 score, cosine/dot-product similarity và RRF `1 / (60 + rank)`.
7. Bảng ablation và ba error case điển hình mở được khi bảo vệ.

- [ ] **Step 4: Final smoke request**

```bash
curl -fsS -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"black leather wallet with RFID protection","mode":"hybrid","top_k":10}'
```

Expected: 10 kết quả từ ABO thật, URL ảnh bắt đầu bằng `/media/products/`.

- [ ] **Step 5: Commit docs**

```bash
git add README.md reports/a1/demo-checklist.md
git commit -m "docs: finalize the A1 reproducible demo"
git -C /Users/yuji/research/rag/test-ui add README.md
git -C /Users/yuji/research/rag/test-ui commit -m "docs: add A1 demo checks"
```

---

## Definition of Done

- `products` alias chứa đúng 1.500 ABO product thật; không còn fixture 4 record.
- Text, cross-modal và image search chạy qua UI/API, thumbnail hiển thị được.
- Benchmark 60 query có qrels review chéo; image query không byte-identical với indexed image.
- Năm experiment chạy trên cùng dataset/model/qrels và xuất bảng comparison.
- Report có metric, latency, score explanation, tối thiểu 15 error case và license note.
- `pytest -q`, frontend test và frontend build đều pass.
