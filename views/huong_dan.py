"""Trang hướng dẫn và phạm vi hệ thống."""

import streamlit as st

from taxrag.config import TAX_CATEGORIES
from taxrag.services import load_knowledge_base

st.markdown('<div class="tr-hero" style="padding-top:1rem"><h1>Hướng dẫn và phạm vi</h1></div>',
            unsafe_allow_html=True)

kb, _ = load_knowledge_base()
if kb is not None:
    st.caption(f"Kho dữ liệu hiện có {len(kb.documents)} văn bản.")

st.markdown(f"""
### Hệ thống trả lời được gì

**Loại thuế:** {', '.join(TAX_CATEGORIES)}.

**Người dùng chính:** doanh nghiệp nhỏ và vừa, hộ kinh doanh, nhân viên kế toán, sinh viên
ngành kế toán – tài chính.

**Loại câu hỏi:** khái niệm, đối tượng áp dụng, điều kiện, thuế suất, thời hạn khai và nộp,
hồ sơ thủ tục, văn bản làm căn cứ, quy định có hiệu lực tại một thời điểm, so sánh quy định cũ và mới.

### Hệ thống không làm

- Tư vấn tối ưu thuế cho một trường hợp cụ thể, hoặc bất kỳ cách nào để né, giảm nghĩa vụ thuế trái quy định.
- Trả lời các sắc thuế ngoài phạm vi trên (thuế TNDN, TNCN, xuất nhập khẩu…) khi kho chưa có văn bản.
- Thay thế ý kiến chính thức của cơ quan thuế.

### Cách đặt câu hỏi để có kết quả tốt

1. Nêu rõ **đối tượng**: doanh nghiệp, hộ kinh doanh, cá nhân; loại hàng hóa, dịch vụ.
2. Nêu **thời điểm** nếu không hỏi về quy định hiện hành: “năm 2024”, “từ ngày 1/7/2025”, “quý 3/2025”.
   Bạn cũng có thể chọn ngày áp dụng ở thanh bên.
3. Mỗi câu hỏi một vấn đề. Nếu hệ thống báo **Cần thêm thông tin**, hãy bổ sung đúng các mục được liệt kê.

### Cách đọc câu trả lời

| Thành phần | Ý nghĩa |
|---|---|
| Nhãn trạng thái | *Đủ căn cứ*, *Trả lời một phần*, *Cần thêm thông tin*, *Chưa đủ căn cứ*, *Nguồn mâu thuẫn*, *Ngoài phạm vi*, *Từ chối* |
| [1], [2] | Số thứ tự căn cứ – tương ứng với thẻ văn bản bên dưới câu trả lời |
| Thẻ văn bản | Số hiệu, Điều – Khoản – Điểm, tình trạng hiệu lực, ngày hiệu lực và đường dẫn văn bản gốc |
| *Chỉ tham khảo* | Công văn, hỏi đáp chính sách – không có giá trị như luật, nghị định, thông tư |

### Giới hạn

- Câu trả lời được sinh bởi mô hình ngôn ngữ dựa trên các đoạn văn bản truy hồi được, vẫn có thể sai.
  Luôn đối chiếu với văn bản gốc trước khi áp dụng.
- Văn bản scan được nhận dạng ký tự (OCR) có thể sai số hiệu, số tiền hoặc dấu tiếng Việt.
- Câu hỏi và câu trả lời được lưu ẩn danh để cải thiện hệ thống. Tài liệu riêng bạn tải lên không được lưu.
""")
