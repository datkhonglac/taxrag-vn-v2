import hmac
import io
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
import requests
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="TaxRAG VN",
    page_icon="📘",
    layout="wide",
)

load_dotenv()

MAX_PRIVATE_FILES = 5
MAX_PRIVATE_TOTAL_MB = 30
OCR_TEXT_THRESHOLD = 120
OCR_RENDER_SCALE = 2.0
MAX_OCR_PAGES_PER_FILE = 80


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
SUPABASE_URL = get_setting("SUPABASE_URL")
SUPABASE_KEY = get_setting("SUPABASE_KEY")


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
# KHO DỮ LIỆU DÙNG CHUNG - SUPABASE REST API
# ============================================================

def supabase_rest_base() -> str:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Chưa cấu hình SUPABASE_URL hoặc SUPABASE_KEY."
        )
    return SUPABASE_URL.rstrip("/") + "/rest/v1"


def supabase_headers(
    prefer: str | None = None,
) -> dict[str, str]:
    """
    Key mới dạng sb_secret_ được gửi qua header apikey.
    Không đặt sb_secret_ vào Authorization: Bearer vì đây
    không phải JWT legacy service_role.
    """
    headers = {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def rest_request(
    method: str,
    table: str,
    *,
    params: dict[str, str] | None = None,
    json_data: Any = None,
    prefer: str | None = None,
) -> Any:
    url = f"{supabase_rest_base()}/{table}"
    response = requests.request(
        method=method,
        url=url,
        headers=supabase_headers(prefer),
        params=params,
        json=json_data,
        timeout=45,
    )

    if not response.ok:
        detail = response.text[:1200]
        raise RuntimeError(
            f"Supabase HTTP {response.status_code}: {detail}"
        )

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError:
        return response.text


def test_supabase_connection() -> tuple[bool, str]:
    try:
        rest_request(
            "GET",
            "documents",
            params={
                "select": "id",
                "limit": "1",
            },
        )
        return True, "Kết nối Supabase thành công."
    except Exception as exc:
        return False, str(exc)


def list_documents() -> list[dict[str, Any]]:
    data = rest_request(
        "GET",
        "documents",
        params={
            "select": "*",
            "order": "uploaded_at.desc",
        },
    )
    return data or []


def load_shared_chunks() -> list[dict[str, Any]]:
    documents = rest_request(
        "GET",
        "documents",
        params={"select": "*"},
    ) or []

    chunks = rest_request(
        "GET",
        "chunks",
        params={
            "select": "*",
            "order": "chunk_index.asc",
        },
    ) or []

    document_map = {
        item["id"]: item
        for item in documents
    }

    merged: list[dict[str, Any]] = []

    for chunk in chunks:
        document = document_map.get(chunk.get("document_id"))
        if not document:
            continue

        merged.append(
            {
                "id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "page": chunk.get("page_number"),
                "content": chunk.get("content", ""),
                "document_name": document.get(
                    "document_name",
                    "",
                ),
                "document_number": document.get(
                    "document_number",
                    "",
                ),
                "topic": document.get("topic", ""),
                "status": document.get("status", ""),
                "source_url": document.get(
                    "source_url",
                    "",
                ),
                "filename": document.get("filename", ""),
                "uploaded_by": document.get(
                    "uploaded_by",
                    "",
                ),
                "uploaded_at": document.get(
                    "uploaded_at",
                    "",
                ),
                "scope": "shared",
            }
        )

    return merged


def delete_document_from_supabase(document_id: str) -> None:
    rest_request(
        "DELETE",
        "documents",
        params={"id": f"eq.{document_id}"},
        prefer="return=minimal",
    )


def upsert_document_to_supabase(
    metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
    page_count: int,
    uploaded_by: str,
) -> str:
    document_number = metadata["document_number"]

    existing = rest_request(
        "GET",
        "documents",
        params={
            "select": "id",
            "document_number": f"eq.{document_number}",
        },
    ) or []

    for item in existing:
        delete_document_from_supabase(item["id"])

    document_payload = {
        "document_name": metadata["document_name"],
        "document_number": document_number,
        "topic": metadata.get("topic", ""),
        "status": metadata.get(
            "status",
            "Chưa xác minh",
        ),
        "source_url": metadata.get("source_url", ""),
        "filename": metadata.get("filename", ""),
        "uploaded_by": uploaded_by or "Admin",
        "page_count": page_count,
        "chunk_count": len(chunks),
    }

    inserted = rest_request(
        "POST",
        "documents",
        json_data=document_payload,
        prefer="return=representation",
    )

    if not inserted:
        raise RuntimeError("Không tạo được bản ghi documents.")

    document_id = inserted[0]["id"]

    payloads = [
        {
            "document_id": document_id,
            "chunk_index": index,
            "page_number": item.get("page"),
            "content": item.get("content", ""),
        }
        for index, item in enumerate(chunks)
    ]

    batch_size = 200

    for batch_start in range(0, len(payloads), batch_size):
        rest_request(
            "POST",
            "chunks",
            json_data=payloads[
                batch_start:batch_start + batch_size
            ],
            prefer="return=minimal",
        )

    return document_id


def delete_all_documents_from_supabase() -> None:
    documents = list_documents()
    for document in documents:
        delete_document_from_supabase(document["id"])


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



def text_quality_ok(text: str) -> bool:
    """
    Kiểm tra nhanh chất lượng text trước khi quyết định OCR.

    Điều kiện:
    - Có ít nhất OCR_TEXT_THRESHOLD ký tự.
    - Ít nhất 70% ký tự là chữ, số hoặc khoảng trắng.
    """
    if len(text) < OCR_TEXT_THRESHOLD:
        return False

    useful = sum(
        ch.isalnum() or ch.isspace()
        for ch in text
    )

    ratio = useful / max(len(text), 1)

    return ratio >= 0.7


def preprocess_ocr_image(image: Image.Image) -> Image.Image:
    """
    Tiền xử lý nhẹ để OCR tiếng Việt ổn định hơn:
    grayscale -> tăng tương phản -> sharpen.
    """
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(1.6)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def ocr_pdf_page(page: fitz.Page) -> str:
    """
    Render một trang PDF thành ảnh rồi OCR bằng Tesseract.
    Chỉ được gọi khi PyMuPDF không lấy đủ lớp chữ.
    """
    matrix = fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE)
    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = Image.open(
        io.BytesIO(pixmap.tobytes("png"))
    )
    image = preprocess_ocr_image(image)

    text = pytesseract.image_to_string(
        image,
        lang="vie+eng",
        config="--oem 3 --psm 6",
    )

    return clean_text(text)


def pdf_to_chunks(
    pdf_bytes: bytes,
    filename: str,
    metadata: dict[str, str] | None = None,
    scope: str = "private",
) -> tuple[list[dict[str, Any]], int, int, int]:
    """
    Trích xuất text theo cơ chế hybrid:
    1. Ưu tiên lớp chữ có sẵn bằng PyMuPDF.
    2. Nếu trang gần như không có chữ, tự động fallback sang OCR.
    3. Nếu OCR vẫn không đọc được thì đánh dấu trang rỗng.
    """
    metadata = metadata or {}
    chunks: list[dict[str, Any]] = []
    empty_pages = 0
    ocr_pages = 0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        page_count = len(document)

        for page_number, page in enumerate(document, start=1):
            page_text = clean_text(page.get_text("text"))
            extraction_method = "text"

            if not text_quality_ok(page_text):
                if ocr_pages >= MAX_OCR_PAGES_PER_FILE:
                    empty_pages += 1
                    continue

                try:
                    page_text = ocr_pdf_page(page)
                    extraction_method = "ocr"
                    ocr_pages += 1
                except Exception:
                    page_text = ""

            if not text_quality_ok(page_text):
                empty_pages += 1
                continue

            for chunk_index, content in enumerate(split_text(page_text)):
                chunk_id = sha256(
                    (
                        f"{scope}|{filename}|{page_number}|"
                        f"{chunk_index}|{content[:100]}"
                    ).encode("utf-8")
                ).hexdigest()

                # Gắn cờ OCR trực tiếp vào chunk cục bộ.
                # Database hiện tại chưa cần thêm cột mới.
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
                        "extraction_method": extraction_method,
                    }
                )

    return chunks, page_count, empty_pages, ocr_pages


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

        try:
            shared_chunk_count = len(load_shared_chunks())
        except Exception:
            shared_chunk_count = 0
        st.metric("Số đoạn trong kho", shared_chunk_count)

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
                total_ocr_pages = 0

                with st.spinner("Đang đọc và chia nhỏ tài liệu..."):
                    for uploaded_file in private_files:
                        try:
                            chunks, _, _, ocr_pages = pdf_to_chunks(
                                pdf_bytes=uploaded_file.getvalue(),
                                filename=uploaded_file.name,
                                scope="private",
                            )

                            if chunks:
                                all_chunks.extend(chunks)
                                total_ocr_pages += ocr_pages
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
                    if total_ocr_pages:
                        st.info(
                            f"Hệ thống đã tự động OCR "
                            f"{total_ocr_pages} trang scan."
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
        if not SUPABASE_URL or not SUPABASE_KEY:
            st.error(
                "Chưa cấu hình SUPABASE_URL hoặc SUPABASE_KEY "
                "trong Streamlit Secrets."
            )
        else:
            st.success("Đã mở quyền quản trị.")

            connection_ok, connection_message = (
                test_supabase_connection()
            )
            if connection_ok:
                st.success("Supabase: kết nối thành công.")
            else:
                st.error(
                    "Supabase chưa kết nối được: "
                    + connection_message
                )

            admin_display_name = st.text_input(
                "Tên người quản trị",
                placeholder="Ví dụ: Nguyễn Văn A",
                help=(
                    "Tên này được lưu cùng tài liệu để các admin "
                    "biết ai đã tải lên."
                ),
            )

            st.divider()
            st.markdown("### Tài liệu hiện có trong kho")

            try:
                documents = list_documents()
            except Exception as exc:
                documents = []
                st.error(
                    "Không đọc được Supabase. Hãy kiểm tra URL, "
                    "Secret key và hai bảng documents/chunks."
                )

            if documents:
                document_rows = [
                    {
                        "Tên văn bản": item.get(
                            "document_name",
                            "",
                        ),
                        "Số hiệu": item.get(
                            "document_number",
                            "",
                        ),
                        "Chủ đề": item.get("topic", ""),
                        "Trạng thái": item.get(
                            "status",
                            "",
                        ),
                        "Số trang": item.get(
                            "page_count",
                            0,
                        ),
                        "Số đoạn": item.get(
                            "chunk_count",
                            0,
                        ),
                        "Người tải": item.get(
                            "uploaded_by",
                            "",
                        ),
                        "Ngày tải": item.get(
                            "uploaded_at",
                            "",
                        ),
                    }
                    for item in documents
                ]

                st.dataframe(
                    document_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                document_options = {
                    (
                        f"{item.get('document_number', '')} — "
                        f"{item.get('document_name', '')}"
                    ): item["id"]
                    for item in documents
                }

                selected_document_label = st.selectbox(
                    "Chọn tài liệu để xóa",
                    options=list(document_options.keys()),
                )

                confirm_single_delete = st.checkbox(
                    "Tôi xác nhận xóa tài liệu đã chọn",
                    key="confirm_single_delete",
                )

                if st.button(
                    "Xóa tài liệu đã chọn",
                    disabled=not confirm_single_delete,
                ):
                    try:
                        delete_document_from_supabase(
                            document_options[
                                selected_document_label
                            ]
                        )
                        st.success(
                            "Đã xóa tài liệu và toàn bộ chunks "
                            "liên quan."
                        )
                        st.rerun()
                    except Exception:
                        st.error(
                            "Không xóa được tài liệu trong Supabase."
                        )
            else:
                st.info("Kho dữ liệu hiện chưa có tài liệu.")

            st.divider()
            st.markdown("### Tải tài liệu mới")

            admin_files = st.file_uploader(
                "Chọn một hoặc nhiều PDF có lớp chữ",
                type=["pdf"],
                accept_multiple_files=True,
                key="admin_uploader",
                help=(
                    "Có thể tải nhiều văn bản cùng lúc. "
                    "Mỗi file cần nhập metadata riêng."
                ),
            )

            admin_entries: list[dict[str, Any]] = []

            if admin_files:
                st.caption(
                    f"Đã chọn {len(admin_files)} file. "
                    "Hãy kiểm tra metadata của từng văn bản."
                )

                if len(admin_files) > 20:
                    st.error(
                        "Mỗi lần chỉ xử lý tối đa 20 file."
                    )

                for file_index, admin_file in enumerate(
                    admin_files,
                    start=1,
                ):
                    widget_id = sha256(
                        (
                            f"{file_index}|{admin_file.name}|"
                            f"{admin_file.size}"
                        ).encode("utf-8")
                    ).hexdigest()[:12]

                    default_name = Path(admin_file.name).stem

                    with st.expander(
                        f"{file_index}. {admin_file.name}",
                        expanded=(len(admin_files) <= 3),
                    ):
                        document_name = st.text_input(
                            "Tên văn bản",
                            value=default_name,
                            key=f"admin_name_{widget_id}",
                        )

                        document_number = st.text_input(
                            "Số hiệu văn bản",
                            placeholder="Ví dụ: 48/2024/QH15",
                            key=f"admin_number_{widget_id}",
                        )

                        topic = st.selectbox(
                            "Chủ đề",
                            [
                                "Thuế giá trị gia tăng",
                                "Hóa đơn điện tử",
                                "Thủ tục khai và nộp thuế",
                            ],
                            key=f"admin_topic_{widget_id}",
                        )

                        status = st.selectbox(
                            "Trạng thái",
                            [
                                "Còn hiệu lực",
                                "Hết hiệu lực",
                                "Chưa xác minh",
                            ],
                            key=f"admin_status_{widget_id}",
                        )

                        source_url = st.text_input(
                            "URL nguồn chính thức",
                            placeholder="https://...",
                            key=f"admin_url_{widget_id}",
                        )

                    admin_entries.append(
                        {
                            "file": admin_file,
                            "document_name": (
                                document_name.strip()
                            ),
                            "document_number": (
                                document_number.strip()
                            ),
                            "topic": topic,
                            "status": status,
                            "source_url": source_url.strip(),
                        }
                    )

            if st.button(
                "Xử lý và lưu tất cả vào Supabase",
                type="primary",
                disabled=(
                    not admin_files
                    or len(admin_files) > 20
                ),
            ):
                missing_metadata = [
                    entry["file"].name
                    for entry in admin_entries
                    if not entry["document_name"]
                    or not entry["document_number"]
                ]

                if missing_metadata:
                    st.error(
                        "Các file sau còn thiếu tên văn bản "
                        "hoặc số hiệu: "
                        + ", ".join(missing_metadata)
                    )
                else:
                    successful_files = 0
                    failed_files: list[str] = []
                    total_pages = 0
                    total_chunks = 0
                    total_ocr_pages_admin = 0

                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    for entry_index, entry in enumerate(
                        admin_entries,
                        start=1,
                    ):
                        uploaded_file = entry["file"]

                        progress_text.write(
                            f"Đang xử lý {entry_index}/"
                            f"{len(admin_entries)}: "
                            f"{uploaded_file.name}"
                        )

                        metadata = {
                            "document_name": (
                                entry["document_name"]
                            ),
                            "document_number": (
                                entry["document_number"]
                            ),
                            "topic": entry["topic"],
                            "status": entry["status"],
                            "source_url": (
                                entry["source_url"]
                            ),
                            "filename": uploaded_file.name,
                        }

                        try:
                            new_chunks, page_count, _, ocr_pages = (
                                pdf_to_chunks(
                                    pdf_bytes=(
                                        uploaded_file.getvalue()
                                    ),
                                    filename=(
                                        uploaded_file.name
                                    ),
                                    metadata=metadata,
                                    scope="shared",
                                )
                            )

                            if not new_chunks:
                                failed_files.append(
                                    f"{uploaded_file.name} "
                                    "(không trích xuất được chữ)"
                                )
                            else:
                                upsert_document_to_supabase(
                                    metadata=metadata,
                                    chunks=new_chunks,
                                    page_count=page_count,
                                    uploaded_by=(
                                        admin_display_name.strip()
                                        or "Admin"
                                    ),
                                )

                                successful_files += 1
                                total_pages += page_count
                                total_chunks += len(new_chunks)
                                total_ocr_pages_admin += ocr_pages

                        except Exception as exc:
                            failed_files.append(
                                f"{uploaded_file.name}: {exc}"
                            )

                        progress_bar.progress(
                            entry_index / len(admin_entries)
                        )

                    if successful_files:
                        st.success(
                            f"Đã lưu {successful_files}/"
                            f"{len(admin_entries)} văn bản vào "
                            f"Supabase, {total_pages} trang và "
                            f"{total_chunks} đoạn dữ liệu."
                        )
                        if total_ocr_pages_admin:
                            st.info(
                                f"Đã tự động OCR "
                                f"{total_ocr_pages_admin} trang scan."
                            )

                    if failed_files:
                        st.error(
                            "Không xử lý được: "
                            + "; ".join(failed_files)
                        )

                    progress_text.empty()

            st.divider()
            st.markdown("### Xóa toàn bộ kho dữ liệu")

            confirm_delete_all = st.checkbox(
                "Tôi xác nhận muốn xóa toàn bộ kho dùng chung",
                key="confirm_delete_all",
            )

            if st.button(
                "Xóa toàn bộ dữ liệu dùng chung",
                disabled=not confirm_delete_all,
            ):
                try:
                    delete_all_documents_from_supabase()
                    st.session_state.shared_messages = []
                    st.success(
                        "Đã xóa toàn bộ tài liệu và chunks."
                    )
                    st.rerun()
                except Exception:
                    st.error(
                        "Không thể xóa toàn bộ dữ liệu trong Supabase."
                    )


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

- Tự động OCR các trang PDF scan khi không có đủ lớp chữ.
- OCR có thể nhận sai số hiệu, số tiền hoặc dấu tiếng Việt; cần đối chiếu trang gốc.
- Kho dùng chung được lưu bền vững trong Supabase.
- Kết quả không thay thế tư vấn chính thức của cơ quan thuế.
"""
    )
