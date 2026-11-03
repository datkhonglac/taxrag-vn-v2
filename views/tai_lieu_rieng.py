"""Trang phân tích tài liệu riêng của người dùng (chỉ lưu trong phiên)."""

import uuid

import numpy as np
import streamlit as st

from taxrag.chunking import chunk_document, index_text
from taxrag.ingest import SUPPORTED_TYPES, extract_file
from taxrag.pipeline import KnowledgeBase, PipelineConfig, answer_question
from taxrag.services import get_reranker, get_settings
from taxrag.ui import render_answer, render_user

MAX_FILES = 5
MAX_TOTAL_MB = 30
MAX_EMBED_CHUNKS = 1500

settings = get_settings()
st.session_state.setdefault("pv_messages", [])
st.session_state.setdefault("pv_kb", None)
st.session_state.setdefault("pv_names", [])

st.markdown(
    """<div class="tr-hero" style="padding-top:1rem"><h1>Phân tích tài liệu riêng</h1>
    <p>Tải lên văn bản, hợp đồng, thông báo thuế… rồi đặt câu hỏi. Tài liệu chỉ giữ trong phiên
    làm việc này, không thêm vào kho dùng chung. Các đoạn liên quan sẽ được gửi tới Gemini API
    để tạo câu trả lời.</p></div>""",
    unsafe_allow_html=True,
)


def build_private_kb(files) -> tuple[KnowledgeBase | None, list[str], list[str]]:
    documents, chunks, names, problems = [], [], [], []
    for f in files:
        try:
            ext = extract_file(f.name, f.getvalue())
        except Exception as exc:
            problems.append(f"{f.name}: {exc}")
            continue
        problems += [f"{f.name}: {w}" for w in ext.warnings]
        if not ext.pages:
            problems.append(f"{f.name}: không trích xuất được chữ.")
            continue
        doc_id = f"private-{uuid.uuid4().hex[:8]}"
        documents.append({"id": doc_id, "document_number": f.name, "document_title": f.name,
                          "document_type": "Tài liệu riêng", "legal_status": "Chưa xác minh"})
        local = chunk_document(ext.pages, "structural")
        for c in local:
            d = c.to_dict()
            d["id"] = f"{doc_id}-{c.local_id}"
            d["parent_id"] = f"{doc_id}-{c.parent_local_id}" if c.parent_local_id is not None else None
            d["document_id"] = doc_id
            chunks.append(d)
        names.append(f.name)
    if not chunks:
        return None, names, problems

    vectors = None
    if settings.gemini_api_key and len(chunks) <= MAX_EMBED_CHUNKS:
        try:
            from taxrag.llm import embed_texts

            by_doc = {d["id"]: d for d in documents}
            vectors = embed_texts(settings, [index_text(c, by_doc[c["document_id"]]) for c in chunks])
        except Exception as exc:
            problems.append(f"Không tạo được embedding ({type(exc).__name__}) – chỉ dùng BM25.")
            vectors = None
    elif len(chunks) > MAX_EMBED_CHUNKS:
        problems.append("Tài liệu quá dài để tạo embedding – chỉ dùng tìm kiếm từ khóa (BM25).")
    return KnowledgeBase(documents, chunks, vectors=np.asarray(vectors) if vectors is not None else None), \
        names, problems


with st.container(border=True):
    files = st.file_uploader(
        f"Tối đa {MAX_FILES} tệp ({', '.join(SUPPORTED_TYPES).upper()}), tổng không quá {MAX_TOTAL_MB} MB",
        type=SUPPORTED_TYPES, accept_multiple_files=True, key="pv_uploader")
    c1, c2 = st.columns(2)
    read = c1.button("Đọc tài liệu", type="primary", use_container_width=True, disabled=not files)
    if c2.button("Xóa tài liệu khỏi phiên", use_container_width=True,
                 disabled=st.session_state.pv_kb is None):
        st.session_state.update(pv_kb=None, pv_names=[], pv_messages=[])
        st.rerun()

if read and files:
    total = sum(len(f.getvalue()) for f in files)
    if len(files) > MAX_FILES:
        st.error(f"Chỉ được tải tối đa {MAX_FILES} tệp.")
    elif total > MAX_TOTAL_MB * 1024 * 1024:
        st.error(f"Tổng dung lượng vượt quá {MAX_TOTAL_MB} MB.")
    else:
        with st.spinner("Đang đọc, phân đoạn và lập chỉ mục tài liệu…"):
            kb, names, problems = build_private_kb(files)
        for p in problems:
            st.warning(p)
        if kb is not None:
            st.session_state.update(pv_kb=kb, pv_names=names, pv_messages=[])
            n_leaf = sum(1 for c in kb.chunks if c.get("is_leaf", True))
            st.success(f"Đã đọc {len(names)} tài liệu, tạo {n_leaf} đoạn theo cấu trúc Điều–Khoản–Điểm.")

kb = st.session_state.pv_kb
if kb is not None:
    st.caption("Đang dùng: " + ", ".join(st.session_state.pv_names))
    for m in st.session_state.pv_messages:
        if m["role"] == "user":
            render_user(m["content"])
        else:
            render_answer(m["result"])
            st.write("")

    q = st.chat_input("Hỏi về tài liệu vừa tải lên…")
    if q:
        st.session_state.pv_messages.append({"role": "user", "content": q})
        render_user(q)
        cfg = PipelineConfig(method="hybrid" if kb.has_dense else "bm25",
                             use_reranker=bool(settings.gemini_api_key), include_reference=True,
                             private_mode=True)
        with st.spinner("Đang phân tích tài liệu…"):
            result = answer_question(kb, q, settings, cfg, reranker=get_reranker()).to_dict()
        st.session_state.pv_messages.append({"role": "assistant", "result": result})
        st.rerun()
else:
    st.info("Chưa có tài liệu nào trong phiên. Hãy tải tệp lên rồi bấm “Đọc tài liệu”.")
