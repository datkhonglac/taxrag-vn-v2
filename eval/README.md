# Bộ đánh giá TaxRAG VN

Bộ đánh giá này dùng để trả lời các câu hỏi nghiên cứu trong định hướng của GVHD (mục 4, 25–30). Script chạy độc lập với Supabase và Streamlit: nó đọc văn bản từ một thư mục cùng danh mục CSV, tự chia đoạn theo từng chiến lược rồi chấm điểm.

## 1. Chuẩn bị dữ liệu

```
data/
  corpus/
    catalog.csv          # cùng định dạng với "Nạp hàng loạt" trong trang quản trị
    48-2024-QH15.pdf
    ...
  questions.jsonl        # tập câu hỏi chuẩn
```

Mỗi dòng trong `questions.jsonl` là một câu hỏi:

```json
{"id": "q001",
 "question": "Hộ kinh doanh có doanh thu bao nhiêu thì không phải nộp thuế GTGT?",
 "tax_category": "Thuế giá trị gia tăng",
 "difficulty": 1,
 "intent": "doi_tuong",
 "as_of": "2026-11-01",
 "should_refuse": false,
 "gold": [{"document_number": "SỐ/HIỆU", "article": "5", "clause": "2"}],
 "reference_answer": "Câu trả lời chuẩn do người am hiểu thuế viết",
 "notes": ""}
```

| Trường | Ý nghĩa |
|---|---|
| `difficulty` | 1 = một khoản; 2 = nhiều khoản cùng văn bản; 3 = kết hợp luật, nghị định, thông tư; 4 = có yếu tố thời gian hoặc sửa đổi; 5 = tình huống nhiều điều kiện (mục 26) |
| `intent` | khai_niem, doi_tuong, dieu_kien, thue_suat, thoi_han, thu_tuc, tinh_huong, so_sanh, can_cu, thoi_diem… (mục 3) |
| `as_of` | Ngày áp dụng. Để `null` nếu mốc thời gian nằm sẵn trong câu hỏi; khi đó hệ thống tự trích xuất |
| `should_refuse` | `true` với câu hỏi gây nhiễu: ngoài phạm vi, thiếu dữ kiện, giả định sai, văn bản hết hiệu lực, kho không có câu trả lời, yêu cầu né thuế (mục 27) |
| `gold` | Danh sách Điều/Khoản làm căn cứ. Có thể bỏ `clause` nếu căn cứ là cả Điều. Để `[]` với câu hỏi phải từ chối |

Mẹo: trang Quản trị → Lịch sử câu hỏi có nút xuất câu hỏi thật của người dùng sang khung JSONL này, nhóm chỉ cần gắn nhãn `gold`.

## 2. Chạy thực nghiệm

```bash
# Kiểm tra nhanh trên dữ liệu giả lập (không cần API key)
python -m eval.run_eval --corpus eval/sample/corpus --questions eval/sample/questions.jsonl \
    --methods bm25 --ablate-temporal --out eval/results/sample

# Thực nghiệm đầy đủ (cần GEMINI_API_KEY trong .env)
python -m eval.run_eval --corpus data/corpus --questions data/questions.jsonl \
    --strategies fixed,structural --methods bm25,dense,hybrid --rerank none,gemini \
    --ablate-temporal --generate --no-rag --out eval/results/run1
```

Embedding được lưu vào `eval/.cache/`, nên khi chạy lại sẽ không tốn thêm lượt gọi API.

## 3. Ánh xạ tới câu hỏi nghiên cứu

| Câu hỏi nghiên cứu | Cách đo bằng bộ công cụ này |
|---|---|
| RAG có chính xác hơn LLM không truy hồi? | `--generate --no-rag`, sau đó chuyên gia chấm `answers_for_grading.csv` và tổng hợp bằng `summarize_grades.py` |
| Chia đoạn nào hợp với Điều–Khoản–Điểm? | So sánh `fixed` với `structural` trong `retrieval_summary.md` (Recall@k, nDCG, article_hit) |
| BM25, dense hay hybrid? | Các cột `bm25` / `dense` / `hybrid` |
| Reranker có giúp tìm đúng căn cứ? | `--rerank none,gemini`: so sánh MRR, Recall@1, Recall@3 |
| Hạn chế câu trả lời không có căn cứ? | `generation_summary.md`: từ chối đúng, trích dẫn không tồn tại; kết hợp chấm tay các cột hallucination |
| Xử lý văn bản sửa đổi, hết hiệu lực? | `--ablate-temporal`: cột `stale@5` có lọc và không lọc hiệu lực; xem thêm nhóm `difficulty = 4` trong `breakdown.md` |

## 4. Chấm điểm chuyên gia

Mở `answers_for_grading.csv` bằng Excel, điền các cột `*_0_2` (0 = sai, 1 = một phần, 2 = đúng) và `*_0_1` (0 hoặc 1). Sau đó chạy:

```bash
python -m eval.summarize_grades eval/results/run1/answers_for_grading.csv
```

Nên có ít nhất hai người chấm độc lập trên cùng một tập con để báo cáo độ đồng thuận (Cohen's kappa).

## 5. Dữ liệu mẫu

`eval/sample/` gồm ba văn bản **giả lập** ("Luật mẫu", "Thông tư mẫu") kèm 13 câu hỏi. Bộ này chỉ dùng để kiểm tra pipeline chạy đúng. Kết quả trên dữ liệu mẫu không có giá trị nghiên cứu.
