# Failure Analysis - Lab 18: Production RAG

**Sinh viên:** Phan Huy Hoang

**Nguồn:** `reports/ragas_report.json`

**Chế độ đánh giá:** RAGAS qua OpenAI, 20 câu hỏi

## Điểm Production RAGAS

| Metric | Score | Đạt 0.75 |
|---|---:|:---:|
| Faithfulness | 0.8129 | Có |
| Answer relevancy | 0.7712 | Có |
| Context precision | 0.9417 | Có |
| Context recall | 0.7917 | Có |

Baseline hiện vẫn là `local_lexical_fallback`, vì vậy không đưa vào bảng so sánh chính thức cho đến khi baseline contexts được phê duyệt và chấm bằng cùng RAGAS pipeline.

## Bottom-5 failures

### 1. Nghỉ phép không lương 20 ngày

- **Kết quả:** `Không tìm thấy.`
- **Worst metric:** Faithfulness = 0.0000.
- **Root cause:** Các context chưa làm nổi bật đồng thời ngưỡng 16-30 ngày và ảnh hưởng bảo hiểm trên 14 ngày.
- **Suggested fix:** Query decomposition theo `phê duyệt` và `bảo hiểm`, sau đó merge context cùng source.

### 2. Lương thử việc Junior cao nhất

- **Kết quả:** Trả đúng 17.000.000 VNĐ nhưng RAGAS faithfulness = 0 vì context không chứa đầy đủ phép tính trong cùng evidence span.
- **Worst metric:** Faithfulness = 0.0000.
- **Root cause:** Bảng lương Junior và quy tắc 85% nằm ở các đoạn khác nhau.
- **Suggested fix:** Parent expansion hoặc row-aware table chunking trước khi tính toán.

### 3. Tạm ứng 15 triệu, thanh toán sau 20 ngày

- **Kết quả:** Tính khoảng 50.025 đồng, gần ground-truth 50.000 đồng.
- **Worst metric:** Faithfulness = 0.0909.
- **Root cause:** LLM tự thực hiện phép tính từ quy tắc 2%/tháng; phép tính chi tiết không xuất hiện nguyên văn trong context.
- **Suggested fix:** Dùng calculator tool và trả kèm công thức cùng evidence nguồn.

### 4. Nhân viên thử việc có hưởng PVI không

- **Kết quả:** `Không tìm thấy.`
- **Worst metric:** Answer relevancy = 0.0000.
- **Root cause:** Retriever chưa đưa câu phủ định về đối tượng thử việc vào top context.
- **Suggested fix:** Tăng trọng số từ khóa phủ định và metadata `employment_status=probation`.

### 5. Laptop 30 triệu và xác nhận CNTT

- **Kết quả:** Đúng phần xác nhận CNTT nhưng chọn sai người phê duyệt và thiếu ba báo giá.
- **Worst metric:** Faithfulness = 0.6667.
- **Root cause:** Query nhiều điều kiện bị nhiễu bởi chính sách tạm ứng có cùng token số tiền/phê duyệt.
- **Suggested fix:** Tách query thành ba nhánh `approval threshold`, `IT confirmation`, `quotation count` rồi chỉ merge context từ `mua_sam.md`.

## Case study versioning

**Câu hỏi:** Bao lâu phải đổi mật khẩu một lần?

1. Hybrid search có thể lấy cả chính sách cũ và hiện hành.
2. Version-aware reranking phạt source `_v1` và ưu tiên `_v2`.
3. Demo runtime trả `120 ngày`; cả ba context đều thuộc v2.0.
4. Production tiếp theo nên dùng metadata chuẩn `version`, `status`, `effective_date`, `supersedes` thay cho suy luận tên file.
