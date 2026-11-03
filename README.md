# TaxRAG VN v3

Hệ thống hỏi đáp chính sách thuế Việt Nam dựa trên kỹ thuật RAG, phục vụ doanh nghiệp nhỏ và vừa. Phạm vi gồm thuế giá trị gia tăng, hóa đơn điện tử, thủ tục khai và nộp thuế.

Đề tài NCKH sinh viên ĐTSV 356, Khoa HTTTKT, Học viện Tài chính.

## Kiến trúc

```
Văn bản (PDF/HTML/TXT) ──► Trích xuất + OCR ──► Làm sạch đầu/chân trang
        ──► Phân đoạn Điều → Khoản → Điểm (giữ đường dẫn, ngữ cảnh, đoạn cha)
        ──► Metadata hiệu lực + quan hệ sửa đổi/thay thế ──► Supabase (Postgres + pgvector)

Câu hỏi ──► Kiểm tra chính sách (né/trốn thuế) ──► Trích xuất thời điểm
        ──► Lọc văn bản theo hiệu lực ──► BM25 + Embedding ──► RRF (hybrid)
        ──► Reranker (Gemini) ──► Kiểm tra mức căn cứ (từ chối sớm)
        ──► Gemini sinh JSON có cấu trúc ──► Kiểm tra trích dẫn
        ──► Câu trả lời + căn cứ pháp lý + hiệu lực + cảnh báo ──► Ghi log + đánh giá 👍👎
```

## Cấu trúc mã nguồn

| Thư mục / tệp | Nội dung | Mục trong định hướng GVHD |
|---|---|---|
| `taxrag/ingest.py` | Đọc PDF/HTML/TXT, OCR tự động trang scan | 9–10 |
| `taxrag/chunking.py` | Chia đoạn `fixed` / `structural` / `hierarchical` | 11–12 |
| `taxrag/text.py` | Chuẩn hóa Unicode, tách âm tiết + bigram cho BM25 | 10, 16 |
| `taxrag/retrieval.py` | BM25, dense, hybrid (RRF) | 15–16 |
| `taxrag/rerank.py` | Reranker Gemini / cross-encoder | 17 |
| `taxrag/temporal.py` | Trích thời điểm, lọc hiệu lực, quan hệ văn bản | 18 |
| `taxrag/generation.py` | Prompt, JSON có cấu trúc, từ chối, kiểm tra trích dẫn | 19–21 |
| `taxrag/pipeline.py` | Ghép toàn bộ pipeline | 31 |
| `taxrag/metrics.py` | Recall@k, Precision@k, MRR, nDCG@k | 28 |
| `taxrag/db.py` | Supabase REST: phân trang, nạp văn bản an toàn, pgvector, log | 8, 13, 23 |
| `views/` | Các trang: hỏi đáp, tài liệu riêng, hướng dẫn, quản trị | 22–24 |
| `eval/` | Script thực nghiệm và chấm điểm, kèm dữ liệu mẫu giả lập | 25–30 |
| `tests/` | Unit test (pytest) | 33 |
| `supabase/schema.sql` | Lược đồ CSDL, chạy được trên CSDL v2 | 8, 13 |

## Cài đặt

### 1. Cơ sở dữ liệu
Mở Supabase → **SQL Editor**, dán và chạy toàn bộ `supabase/schema.sql`. File được viết kiểu `if not exists` nên chạy được trên CSDL v2 đang có. Lệnh này thêm các cột metadata, bảng quan hệ văn bản, bảng lịch sử câu hỏi và hàm `match_chunks` của pgvector.

### 2. Secrets (`.streamlit/secrets.toml` hoặc Streamlit Cloud → Settings → Secrets)

```toml
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"   # tùy chọn
EMBEDDING_DIM = "768"                      # phải khớp vector(768) trong schema.sql
RERANK_MODEL = ""                          # tùy chọn: mô hình rẻ hơn cho rerank
ADMIN_PASSWORD = "..."
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_KEY = "sb_secret_..."
```

### 3. Chạy

```bash
pip install -r requirements.txt
streamlit run app.py
```

Trên Streamlit Cloud, `packages.txt` cài Tesseract và gói tiếng Việt để OCR được trang scan.

### 4. Nạp lại dữ liệu cũ
Các đoạn của v2 được chia theo trang, sau khi chạy `schema.sql` sẽ được đánh dấu `legacy`. Vào **Quản trị → Nạp văn bản**, nạp lại từng văn bản kèm ngày hiệu lực. Cách nhanh hơn là dùng **Nạp hàng loạt** với tệp danh mục CSV. Sau đó vào **Bảo trì** và bấm **Tạo embedding** nếu cần.

## Kiểm thử

```bash
pip install -r requirements-dev.txt
pytest -q
python -m eval.run_eval --corpus eval/sample/corpus --questions eval/sample/questions.jsonl \
    --methods bm25 --ablate-temporal --out eval/results/sample
```

Xem `eval/README.md` để biết cách xây tập câu hỏi chuẩn và chạy thực nghiệm đầy đủ.

## Thay đổi so với v2

**Lỗi đã sửa**
- Kho trên 1.000 đoạn không còn bị cắt. Supabase giới hạn 1.000 dòng mỗi lần gọi, nay `db.get_all` đọc theo từng trang.
- OCR hoạt động trên Streamlit Cloud. `taxrag_packages.txt` được đổi tên thành `packages.txt`.
- Nạp lại văn bản không còn nguy cơ mất dữ liệu. Bản mới được ghi xong và chuyển sang `ready` rồi mới xóa bản cũ.
- Kho và chỉ mục chỉ dựng một lần rồi dùng chung cho mọi phiên, thay vì tải lại toàn bộ và tính lại TF-IDF ở mỗi lần bấm.
- Câu trả lời cũ vẫn giữ nguồn trích dẫn khi người dùng hỏi tiếp.
- Nội dung tài liệu do người dùng tải lên được đặt trong thẻ dữ liệu, hạn chế prompt injection.

**Tính năng mới theo định hướng**
- Chia đoạn theo Điều–Khoản–Điểm, có thêm chế độ phân cấp. Chia cố định được giữ làm đối chứng.
- Metadata đầy đủ: loại văn bản, cơ quan ban hành, ngày ban hành, ngày hiệu lực, ngày hết hiệu lực, đối tượng áp dụng, URL nguồn chính thức. Có thêm quan hệ sửa đổi, thay thế, hướng dẫn giữa các văn bản.
- Trả lời theo thời điểm: nhận diện ngày, tháng, quý, năm và các cách nói "hiện nay", "năm ngoái", rồi lọc văn bản theo hiệu lực.
- Truy hồi BM25, embedding (pgvector) và hybrid (RRF), có thêm reranker.
- Câu trả lời có cấu trúc: kết luận, điều kiện, căn cứ, hiệu lực, lưu ý. Mỗi câu trả lời mang một trong 7 trạng thái, gồm cả các trạng thái từ chối.
- Phát hiện trích dẫn tới nguồn không tồn tại. Công văn và hỏi đáp được gắn nhãn "chỉ tham khảo".
- Giao diện mới: thanh bên chọn loại thuế và ngày áp dụng, thẻ căn cứ pháp lý, đánh giá 👍👎, chế độ nghiên cứu hiển thị điểm truy hồi.
- Trang quản trị tách riêng, chỉ hiện khi đã đăng nhập. Có xem trước phân đoạn, nạp hàng loạt bằng CSV, sửa metadata, khai báo quan hệ văn bản, kiểm tra đoạn, lịch sử câu hỏi và bảo trì embedding.
- Bộ đánh giá `eval/` và unit test.

## Giới hạn hiện tại
- Chưa hỗ trợ tệp `.doc/.docx`. Hãy chuyển sang PDF hoặc lấy bản HTML từ vbpl.vn.
- Ngưỡng từ chối (`min_rerank_score`, `min_dense_score` trong `taxrag/config.py`) là giá trị khởi điểm, cần hiệu chỉnh bằng bộ đánh giá.
- Hệ thống chưa dùng lịch sử hội thoại: mỗi câu hỏi được xử lý độc lập, câu hỏi nối tiếp cần nêu lại ngữ cảnh.
