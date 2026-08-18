# Báo cáo kết quả - Lab 18 Production RAG

**Sinh viên:** Phan Huy Hoang

**Ngày kiểm chứng:** 18/08/2026

## Phạm vi hoàn thành

| Module | Nội dung | Trạng thái |
|---|---|---|
| M1 | Semantic, hierarchical, structure-aware chunking | Hoàn thành |
| M2 | Vietnamese BM25, dense Qdrant, RRF, local fallback | Hoàn thành |
| M3 | CrossEncoder reranking, cache model, version preference | Hoàn thành |
| M4 | RAGAS, lexical fallback, failure analysis | Hoàn thành |
| M5 | Summary, HyQA, contextual prepend, metadata, combined mode | Hoàn thành |

## Bằng chứng runtime

- Validator chính thức: **39/39 tests passed (100%)**.
- Corpus: 26 tài liệu đọc được, tạo 104 hierarchical child chunks.
- Qdrant: `lab18_production=104`, `lab18_naive=57` vectors.
- Demo version query trả đúng chu kỳ mật khẩu hiện hành: 120 ngày từ v2.0.
- Production OpenAI + RAGAS: 20 answers, 80 metric evaluations, `evaluation_mode=ragas`.
- Hai PDF scan không có text layer được bỏ qua có cảnh báo; chưa OCR.

## Production RAGAS

| Metric | Score | Trạng thái |
|---|---:|---|
| Faithfulness | 0.8129 | Đạt |
| Answer relevancy | 0.7712 | Đạt |
| Context precision | 0.9417 | Đạt |
| Context recall | 0.7917 | Đạt |

Cả bốn metric đều vượt 0.75. Baseline report hiện dùng lexical fallback nên chưa thể tính delta RAGAS hợp lệ nếu chưa chạy baseline qua cùng evaluator.

## Key findings

1. Context precision là điểm mạnh nhất (0.9417); hybrid retrieval + reranker lọc nhiễu tốt ở phần lớn câu hỏi.
2. Faithfulness giảm mạnh ở câu hỏi cần tính toán hoặc ghép evidence từ nhiều chunks.
3. Câu hỏi phủ định và multi-hop vẫn cần query decomposition/metadata routing.
4. Version-aware reranking sửa được lỗi chính sách mật khẩu v1 cạnh tranh với v2.
5. External calls được đặt opt-in; enrichment 104 chunks vẫn chạy local để kiểm soát chi phí.

## Demo

```powershell
.\.venv311\Scripts\Activate.ps1
python .\demo.py --question "Bao lâu phải đổi mật khẩu một lần?" --show-context
```

Kết quả kiểm chứng: trả `mỗi 120 ngày`, context từ chính sách mật khẩu v2.0.
