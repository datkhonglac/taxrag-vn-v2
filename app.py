import hmac
import json
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="TaxRAG VN",
    page_icon="📘",
    layout="wide",
)

load_dotenv()

DATA_DIR = Path("data_store")
SHARED_DATA_FILE = DATA_DIR / "shared_chunks.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_PRIVATE_FILES = 5
MAX_PRIVATE_TOTAL_MB = 30


def get_setting(name: str, default: str = "") -> str:
    """Đọc cấu hình từ Streamlit Secrets, sau đó mới đọc biến môi trường."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.6-flash")
ADMIN_PASSWORD = get_setting("ADMIN_PASSWORD")


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# TRẠNG THÁI PHIÊN
# ============================================================

SESSION_DEFAULTS = {
    "shared_messages": [],
    "private_messages": [],
    "private_chunks": [],
    "private_names": [],
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value.copy()


# ============================================================
# KHO DỮ LIỆU DÙNG CHUNG
# ============================================================

def load_shared_chunks() -> list[dict[str, Any]]:
    if not SHARED_DATA_FILE.exists():
        return []

    try:
        data = json.loads(SHARED_DATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_shared_chunks(chunks: list[dict[str, Any]]) -> None:
    temp_file = SHARED_DATA_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(SHARED_DATA_FILE)


# ============================================================
# XỬ LÝ PDF
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    return text.strip()


def split_text(
    text: str,
    max_chars: int = 1400,
    overlap_chars: int = 220,
) -> list[str]:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip()

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        while len(paragraph) > max_chars:
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars - overlap_chars:]

        current = paragraph

    if current:
        chunks.append(current)

    return chunks


def pdf_to_chunks(
    pdf_bytes: bytes,
    filename: str,
    metadata: dict[str, str] | None = None,
    scope: str = "private",
) -> tuple[list[dict[str, Any]], int, int]:
    metadata = metadata or {}
    chunks: list[dict[str, Any]] = []
    empty_pages = 0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        page_count = len(document)

        for page_number, page in enumerate(document, start=1):
            page_text = clean_text(page.get_text("text"))

            if len(page_text) < 50:
                empty_pages += 1
                continue

            for chunk_index, content in enumerate(split_text(page_text)):
                chunk_id = sha256(
                    (
                        f"{scope}|{filename}|{page_number}|"
                        f"{chunk_index}|{content[:100]}"
                    ).encode("utf-8")
                ).hexdigest()

                chunks.append(
                    {
                        "id": chunk_id,
                        "scope": scope,
                        "filename": filename,
                        "document_name": metadata.get(
                            "document_name",
                            filename,
                        ),
                        "document_number": metadata.get(
                            "document_number",
                            "",
                        ),
                        "topic": metadata.get(
                            "topic",
                            "Tài liệu thuế riêng",
                        ),
                        "status": metadata.get(
                            "status",
                            "Tài liệu riêng",
                        ),
                        "source_url": metadata.get(
                            "source_url",
                            "",
                        ),
                        "page": page_number,
                        "content": content,
                    }
                )

    return chunks, page_count, empty_pages


# ============================================================
# TRUY XUẤT TF-IDF
# ============================================================

def retrieve(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    documents = [
        (
            f"{item.get('document_name', '')} "
            f"{item.get('document_number', '')} "
            f"{item.get('topic', '')} "
            f"{item.get('content', '')}"
        )
        for item in chunks
    ]

    try:
        word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50000,
        )
        char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=70000,
        )

        word_matrix = word_vectorizer.fit_transform(documents)
        char_matrix = char_vectorizer.fit_transform(documents)
        document_matrix = hstack([word_matrix, char_matrix])

        query_matrix = hstack(
            [
                word_vectorizer.transform([question]),
                char_vectorizer.transform([question]),
            ]
        )

        scores = cosine_similarity(
            query_matrix,
            document_matrix,
        ).flatten()

    except ValueError:
        return []

    safe_top_k = min(max(1, top_k), len(chunks))
    best_indices = np.argsort(scores)[::-1][:safe_top_k]

    results: list[dict[str, Any]] = []

    for index in best_indices:
        item = dict(chunks[int(index)])
        item["similarity"] = float(scores[int(index)])
        results.append(item)

    return results


# ============================================================
# GEMINI
# ============================================================

def build_context(
    sources: list[dict[str, Any]],
    source_label: str,
) -> str:
    blocks: list[str] = []

    for index, item in enumerate(sources, start=1):
        blocks.append(
            f"""
[{source_label.upper()} {index}]
Tên tài liệu: {item.get("document_name", "")}
Số hiệu: {item.get("document_number", "")}
Chủ đề: {item.get("topic", "")}
Trạng thái: {item.get("status", "")}
Trang: {item.get("page", "")}
URL: {item.get("source_url", "")}

Nội dung:
{item.get("content", "")}
""".strip()
        )

    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    sources: list[dict[str, Any]],
    private_mode: bool,
) -> str:
    client = get_gemini_client()

    if client is None:
        return (
            "Hệ thống chưa được cấu hình GEMINI_API_KEY trong "
            "Streamlit Secrets."
        )

    if private_mode:
        source_label = "Tài liệu"
        rules = """
- Chỉ sử dụng nội dung trong tài liệu người dùng vừa tải lên.
- Không mặc định tài liệu là văn bản pháp luật chính thức hoặc còn hiệu lực.
- Không tự tạo số liệu, điều khoản, thời hạn hoặc nghĩa vụ thuế.
- Gắn các nhận định quan trọng với [Tài liệu 1], [Tài liệu 2]...
- Khi thiếu thông tin, nói rõ: "Chưa đủ thông tin trong tài liệu để kết luận."
"""
    else:
        source_label = "Nguồn"
        rules = """
- Chỉ sử dụng ngữ cảnh được truy xuất từ kho dữ liệu dùng chung.
- Không tự tạo số hiệu văn bản, điều khoản, ngày tháng, mức thuế hoặc thời hạn.
- Gắn các nhận định quan trọng với [Nguồn 1], [Nguồn 2]...
- Khi thiếu căn cứ, nói rõ: "Chưa đủ căn cứ trong kho dữ liệu để kết luận."
"""

    prompt = f"""
Bạn là trợ lý tra cứu và phân tích chính sách thuế Việt Nam dành cho
doanh nghiệp vừa và nhỏ.

QUY TẮC:
{rules}
- Không hướng dẫn trốn thuế, che giấu doanh thu hoặc vi phạm pháp luật.
- Trả lời bằng tiếng Việt, rõ ràng và thực tế.
- Kết quả không thay thế tư vấn chính thức của cơ quan thuế.

CÂU HỎI:
{question}

NGỮ CẢNH:
{build_context(sources, source_label)}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return getattr(response, "text", None) or (
            "Mô hình không trả về nội dung."
        )
    except Exception:
        return (
            "Không thể gọi Gemini API. Hãy kiểm tra GEMINI_API_KEY "
            "và GEMINI_MODEL trong Streamlit Secrets."
        )


def show_sources(
    sources: list[dict[str, Any]],
    title: str,
) -> None:
    if not sources:
        return

    with st.expander(title):
        for index, item in enumerate(sources, start=1):
            line = (
                f"**{index}. {item.get('document_name', '')}** — "
                f"trang {item.get('page', '')} — "
                f"độ tương đồng "
                f"{item.get('similarity', 0):.2f}"
            )

            if item.get("document_number"):
                line += f" — {item['document_number']}"

            if item.get("source_url"):
                line += f" — [Mở nguồn]({item['source_url']})"

            st.markdown(line)


# ============================================================
# GIAO DIỆN
# ============================================================

st.title("TaxRAG VN")
st.caption(
    "Trợ lý hỏi đáp và phân tích tài liệu thuế dành cho "
    "doanh nghiệp vừa và nhỏ."
)

tab_shared, tab_private, tab_admin, tab_guide = st.tabs(
    [
        "Hỏi đáp chính sách",
        "Phân tích tài liệu riêng",
        "Quản trị dữ liệu",
        "Hướng dẫn",
    ]
)


# ------------------------------------------------------------
# 1. HỎI ĐÁP CHÍNH SÁCH
# ------------------------------------------------------------

with tab_shared:
    left, right = st.columns([1, 3])

    with left:
        active_only = st.checkbox(
            "Chỉ dùng văn bản còn hiệu lực",
            value=True,
        )

        shared_top_k = st.slider(
            "Số đoạn truy xuất",
            min_value=3,
            max_value=10,
            value=5,
            key="shared_top_k",
        )

        shared_threshold = st.slider(
            "Ngưỡng liên quan",
            min_value=0.00,
            max_value=0.80,
            value=0.08,
            step=0.01,
            key="shared_threshold",
        )

        st.metric(
            "Số đoạn trong kho",
            len(load_shared_chunks()),
        )

    with right:
        for message in st.session_state.shared_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        shared_question = st.chat_input(
            "Nhập câu hỏi về chính sách thuế...",
            key="shared_question",
        )

        if shared_question:
            st.session_state.shared_messages.append(
                {
                    "role": "user",
                    "content": shared_question,
                }
            )

            with st.chat_message("user"):
                st.markdown(shared_question)

            shared_corpus = load_shared_chunks()

            if active_only:
                shared_corpus = [
                    item
                    for item in shared_corpus
                    if item.get("status") == "Còn hiệu lực"
                ]

            shared_sources = retrieve(
                question=shared_question,
                chunks=shared_corpus,
                top_k=shared_top_k,
            )

            with st.chat_message("assistant"):
                if not shared_sources:
                    shared_answer = (
                        "Kho dữ liệu đang trống hoặc không có tài liệu "
                        "phù hợp với bộ lọc."
                    )
                elif (
                    shared_sources[0]["similarity"]
                    < shared_threshold
                ):
                    shared_answer = (
                        "Chưa đủ căn cứ trong kho dữ liệu để kết luận."
                    )
                else:
                    with st.spinner("Đang tạo câu trả lời..."):
                        shared_answer = generate_answer(
                            question=shared_question,
                            sources=shared_sources,
                            private_mode=False,
                        )

                st.markdown(shared_answer)
                show_sources(
                    shared_sources,
                    "Nguồn được truy xuất",
                )

            st.session_state.shared_messages.append(
                {
                    "role": "assistant",
                    "content": shared_answer,
                }
            )


# ------------------------------------------------------------
# 2. PHÂN TÍCH TÀI LIỆU RIÊNG
# ------------------------------------------------------------

with tab_private:
    st.subheader("Phân tích tài liệu thuế của bạn")

    st.info(
        "Tài liệu chỉ được giữ trong phiên sử dụng hiện tại và không "
        "được thêm vào kho dùng chung. Các đoạn liên quan sẽ được gửi "
        "tới Gemini API để tạo câu trả lời."
    )

    private_files = st.file_uploader(
        "Tải tối đa 5 PDF có lớp chữ, tổng không quá 30 MB",
        type=["pdf"],
        accept_multiple_files=True,
        key="private_uploader",
    )

    read_col, clear_col = st.columns(2)

    read_private = read_col.button(
        "Đọc tài liệu",
        type="primary",
        use_container_width=True,
    )

    clear_private = clear_col.button(
        "Xóa tài liệu khỏi phiên",
        use_container_width=True,
    )

    if clear_private:
        st.session_state.private_chunks = []
        st.session_state.private_names = []
        st.session_state.private_messages = []
        st.rerun()

    if read_private:
        if not private_files:
            st.error("Bạn chưa chọn PDF.")
        elif len(private_files) > MAX_PRIVATE_FILES:
            st.error(
                f"Chỉ được tải tối đa {MAX_PRIVATE_FILES} file."
            )
        else:
            total_size = sum(
                len(uploaded_file.getvalue())
                for uploaded_file in private_files
            )

            if total_size > MAX_PRIVATE_TOTAL_MB * 1024 * 1024:
                st.error(
                    f"Tổng dung lượng vượt quá "
                    f"{MAX_PRIVATE_TOTAL_MB} MB."
                )
            else:
                all_chunks: list[dict[str, Any]] = []
                document_names: list[str] = []
                failed_files: list[str] = []

                with st.spinner("Đang đọc và chia nhỏ tài liệu..."):
                    for uploaded_file in private_files:
                        try:
                            chunks, _, _ = pdf_to_chunks(
                                pdf_bytes=uploaded_file.getvalue(),
                                filename=uploaded_file.name,
                                scope="private",
                            )

                            if chunks:
                                all_chunks.extend(chunks)
                                document_names.append(
                                    uploaded_file.name
                                )
                            else:
                                failed_files.append(
                                    uploaded_file.name
                                )

                        except Exception:
                            failed_files.append(
                                uploaded_file.name
                            )

                if all_chunks:
                    st.session_state.private_chunks = all_chunks
                    st.session_state.private_names = document_names
                    st.session_state.private_messages = []

                    st.success(
                        f"Đã đọc {len(document_names)} tài liệu và tạo "
                        f"{len(all_chunks)} đoạn nội dung."
                    )
                else:
                    st.error(
                        "Không trích xuất được chữ. PDF có thể là "
                        "file scan hoặc được bảo vệ."
                    )

                if failed_files:
                    st.warning(
                        "Không đọc được: "
                        + ", ".join(failed_files)
                    )

    if st.session_state.private_chunks:
        st.caption(
            "Tài liệu đang dùng: "
            + ", ".join(st.session_state.private_names)
        )

        private_left, private_right = st.columns([1, 3])

        with private_left:
            private_top_k = st.slider(
                "Số đoạn phân tích",
                min_value=3,
                max_value=10,
                value=5,
                key="private_top_k",
            )

            private_threshold = st.slider(
                "Ngưỡng liên quan",
                min_value=0.00,
                max_value=0.80,
                value=0.05,
                step=0.01,
                key="private_threshold",
            )

            st.metric(
                "Số đoạn tài liệu",
                len(st.session_state.private_chunks),
            )

        with private_right:
            for message in st.session_state.private_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            private_question = st.chat_input(
                "Hỏi về tài liệu vừa tải lên...",
                key="private_question",
            )

            if private_question:
                st.session_state.private_messages.append(
                    {
                        "role": "user",
                        "content": private_question,
                    }
                )

                with st.chat_message("user"):
                    st.markdown(private_question)

                private_sources = retrieve(
                    question=private_question,
                    chunks=st.session_state.private_chunks,
                    top_k=private_top_k,
                )

                with st.chat_message("assistant"):
                    if not private_sources:
                        private_answer = (
                            "Không tìm được nội dung phù hợp trong "
                            "tài liệu đã tải lên."
                        )
                    elif (
                        private_sources[0]["similarity"]
                        < private_threshold
                    ):
                        private_answer = (
                            "Chưa đủ thông tin trong tài liệu để "
                            "kết luận."
                        )
                    else:
                        with st.spinner(
                            "Đang phân tích tài liệu..."
                        ):
                            private_answer = generate_answer(
                                question=private_question,
                                sources=private_sources,
                                private_mode=True,
                            )

                    st.markdown(private_answer)
                    show_sources(
                        private_sources,
                        "Đoạn tài liệu được sử dụng",
                    )

                st.session_state.private_messages.append(
                    {
                        "role": "assistant",
                        "content": private_answer,
                    }
                )


# ------------------------------------------------------------
# 3. QUẢN TRỊ DỮ LIỆU
# ------------------------------------------------------------

with tab_admin:
    st.subheader("Quản trị kho dữ liệu dùng chung")

    admin_allowed = False

    if not ADMIN_PASSWORD:
        st.error(
            "ADMIN_PASSWORD chưa được cấu hình trong "
            "Streamlit Secrets."
        )
    else:
        entered_password = st.text_input(
            "Mật khẩu quản trị",
            type="password",
            key="admin_password",
        )

        if entered_password:
            admin_allowed = hmac.compare_digest(
                entered_password,
                ADMIN_PASSWORD,
            )

            if not admin_allowed:
                st.error("Mật khẩu không đúng.")

    if admin_allowed:
        st.success("Đã mở quyền quản trị.")

        admin_file = st.file_uploader(
            "Chọn PDF có lớp chữ",
            type=["pdf"],
            key="admin_uploader",
        )

        document_name = st.text_input("Tên văn bản")
        document_number = st.text_input("Số hiệu văn bản")

        topic = st.selectbox(
            "Chủ đề",
            [
                "Thuế giá trị gia tăng",
                "Hóa đơn điện tử",
                "Thủ tục khai và nộp thuế",
            ],
        )

        status = st.selectbox(
            "Trạng thái",
            [
                "Còn hiệu lực",
                "Hết hiệu lực",
                "Chưa xác minh",
            ],
        )

        source_url = st.text_input(
            "URL nguồn chính thức",
            placeholder="https://...",
        )

        if st.button(
            "Xử lý và lưu vào kho dùng chung",
            type="primary",
        ):
            if admin_file is None:
                st.error("Bạn chưa chọn PDF.")
            elif not document_name.strip():
                st.error("Bạn chưa nhập tên văn bản.")
            elif not document_number.strip():
                st.error("Bạn chưa nhập số hiệu văn bản.")
            else:
                metadata = {
                    "document_name": document_name.strip(),
                    "document_number": document_number.strip(),
                    "topic": topic,
                    "status": status,
                    "source_url": source_url.strip(),
                }

                try:
                    with st.spinner("Đang xử lý PDF..."):
                        new_chunks, page_count, empty_pages = (
                            pdf_to_chunks(
                                pdf_bytes=admin_file.getvalue(),
                                filename=admin_file.name,
                                metadata=metadata,
                                scope="shared",
                            )
                        )

                    if not new_chunks:
                        st.error(
                            "Không trích xuất được chữ. PDF có thể "
                            "là file scan."
                        )
                    else:
                        existing_chunks = [
                            item
                            for item in load_shared_chunks()
                            if item.get("document_number")
                            != document_number.strip()
                        ]

                        save_shared_chunks(
                            existing_chunks + new_chunks
                        )

                        st.success(
                            f"Đã xử lý {page_count} trang và lưu "
                            f"{len(new_chunks)} đoạn dữ liệu."
                        )

                        if empty_pages:
                            st.warning(
                                f"Có {empty_pages} trang không "
                                "trích xuất được chữ."
                            )

                except Exception:
                    st.error(
                        "Không thể đọc PDF. File có thể bị lỗi, "
                        "được bảo vệ hoặc là PDF scan."
                    )

        st.divider()

        confirm_delete = st.checkbox(
            "Tôi xác nhận muốn xóa toàn bộ kho dùng chung"
        )

        if st.button(
            "Xóa toàn bộ dữ liệu dùng chung",
            disabled=not confirm_delete,
        ):
            save_shared_chunks([])
            st.session_state.shared_messages = []
            st.success("Đã xóa toàn bộ kho dữ liệu dùng chung.")


# ------------------------------------------------------------
# 4. HƯỚNG DẪN
# ------------------------------------------------------------

with tab_guide:
    st.markdown(
        """
### Ba khu vực của hệ thống

**Hỏi đáp chính sách**

Người dùng không cần mật khẩu. Câu trả lời dựa trên kho tài liệu
dùng chung do quản trị viên kiểm soát.

**Phân tích tài liệu riêng**

Người dùng tự tải PDF lên rồi đặt câu hỏi. Tài liệu chỉ được giữ
trong phiên hiện tại và không được thêm vào kho dùng chung.

**Quản trị dữ liệu**

Chỉ quản trị viên có mật khẩu mới được thêm hoặc xóa tài liệu trong
kho dùng chung.

### Phạm vi MVP

- Thuế giá trị gia tăng.
- Hóa đơn điện tử.
- Thủ tục khai và nộp thuế.
- Đối tượng chính: doanh nghiệp vừa và nhỏ.

### Giới hạn

- Chỉ đọc PDF có lớp chữ; chưa hỗ trợ OCR cho PDF scan.
- Kho dùng chung có thể mất khi Streamlit Cloud tạo lại môi trường.
- Kết quả không thay thế tư vấn chính thức của cơ quan thuế.
"""
    )
