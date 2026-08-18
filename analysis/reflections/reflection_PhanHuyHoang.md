# Individual Reflection - Phan Huy Hoang

## 1. Đóng góp kỹ thuật

Tôi hoàn thiện cả năm module và luồng demo: ba chiến lược chunking; BM25 + dense Qdrant + RRF; CrossEncoder reranking; RAGAS adapter/failure analysis; enrichment có local fallback và kiểm soát chi phí. Kết quả cuối là 39/39 test pass.

## 2. Mapping bài giảng vào code

| Concept | Module/hàm | Quan sát thực tế |
|---|---|---|
| Semantic chunking | `chunk_semantic()` | Model được cache; có lexical fallback khi model không sẵn sàng. |
| Parent-child chunking | `chunk_hierarchical()` | 26 tài liệu tạo 104 child chunks có parent ID. |
| Hybrid retrieval | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Qdrant giữ 104 vector; BM25 được rebuild nhanh khi demo. |
| Cross-encoder | `CrossEncoderReranker.rerank()` | Chất lượng version query tăng sau khi thêm lifecycle adjustment. |
| Evaluation | `evaluate_ragas()` | Production RAGAS thật đạt 0.8129/0.7712/0.9417/0.7917. |
| Contextual enrichment | `_enrich_single_call()` | Local mode hoàn thành 104 chunks gần như tức thời và không phát sinh API cost. |

## 3. Khó khăn và cách giải quyết

- Python 3.13 khiến NumPy 1.26.4 phải compile và lỗi `Compiler cl cannot compile programs`; tạo venv Python 3.11 giải quyết được.
- Qdrant container ban đầu không publish port; recreate service đã đưa `6333/6334` ra localhost.
- Windows `cp1252` làm crash khi in ký tự Unicode; các entry point được cấu hình UTF-8.
- Placeholder `sk-...` bị coi là key thật và tạo 401; config hiện loại placeholder và tách cờ external call.
- Chính sách cũ cạnh tranh với bản hiện hành; reranker hiện phạt source v1/v2023 và ưu tiên v2/v2024.

## 4. Nếu làm tiếp

1. Parse bảng Markdown thành row-aware chunks.
2. Thêm query decomposition cho multi-hop và calculator cho bài toán số học.
3. Đưa `status`, `version`, `effective_date`, `supersedes` vào metadata chuẩn.
4. OCR hai PDF scan và đánh dấu chất lượng OCR trước khi index.
5. Chạy baseline bằng cùng RAGAS evaluator để có delta so sánh hợp lệ.

## 5. Tự đánh giá

| Tiêu chí | Điểm (1-5) |
|---|---:|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Kiểm chứng runtime | 5 |
| Phân tích failure | 4 |
