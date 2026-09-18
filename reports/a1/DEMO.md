# A1 demo checklist

Run the stack with `docker compose up --build`, then use these live checks:

```bash
# exact / BM25
curl -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"AmazonBasics PowerBank (5,600 mAh) and Lightning Cable (6 Feet)","mode":"bm25","top_k":3}'

# semantic English, compare dense and hybrid
curl -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"bathwater additive for everyday use","mode":"dense","top_k":3}'

# Vietnamese / cross-lingual text path
curl -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"mua AmazonBasics sản phẩm trắng","mode":"cross_modal","top_k":3}'

# text → image and image upload
curl -X POST http://localhost:8000/api/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query":"camera tripod for everyday use","mode":"cross_modal","top_k":3}'
curl -X POST 'http://localhost:8000/api/v1/search/image?top_k=3' \
  -F 'image=@evaluation/benchmarks/a1/images/i001.jpg;type=image/jpeg'
```

All result thumbnails should use `/media/products/<filename>` and load through the frontend proxy at `http://localhost:5173`.
