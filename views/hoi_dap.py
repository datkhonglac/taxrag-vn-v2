"""Trang hỏi đáp chính sách thuế (kho dữ liệu dùng chung)."""

import uuid
from datetime import date

import streamlit as st

from taxrag.config import TAX_CATEGORIES
from taxrag.pipeline import PipelineConfig, answer_question
from taxrag.services import (
    get_reranker,
    get_settings,
    load_knowledge_base,
    log_query,
    set_feedback,
)
from taxrag.ui import render_answer, render_user

SUGGESTIONS = [
    "Hộ kinh doanh có thuộc đối tượng nộp thuế GTGT không?",
    "Điều kiện áp dụng thuế suất 0% với hàng hóa xuất khẩu là gì?",
    "Doanh nghiệp nào được khai thuế GTGT theo quý?",
    "Thời hạn nộp hồ sơ khai thuế GTGT theo quý là khi nào?",
    "Thời điểm lập hóa đơn điện tử khi cung cấp dịch vụ là khi nào?",
    "Thuế suất GTGT năm 2024 khác quy định hiện hành thế nào?",
]
METHOD_LABELS = {"hybrid": "Lai (BM25 + embedding)", "bm25": "BM25 (từ khóa)", "dense": "Embedding (ngữ nghĩa)"}
CONTEXT_LABELS = {"leaf": "Đoạn khoản/điểm", "parent": "Cả Điều chứa đoạn (phân cấp)"}

settings = get_settings()
st.session_state.setdefault("qa_messages", [])
st.session_state.setdefault("session_id", str(uuid.uuid4()))
kb, kb_error = load_knowledge_base()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("##### Phạm vi tra cứu")
    cats = st.pills("Loại thuế", TAX_CATEGORIES, selection_mode="multi",
                    default=TAX_CATEGORIES, key="qa_cats")
    time_mode = st.segmented_control("Thời điểm áp dụng", ["Theo câu hỏi", "Chọn ngày"],
                                     default="Theo câu hỏi", key="qa_time_mode")
    as_of = None
    if time_mode == "Chọn ngày":
        as_of = st.date_input("Ngày áp dụng", value=date.today(), format="DD/MM/YYYY", key="qa_as_of")
    else:
        st.caption("Tự nhận diện mốc thời gian trong câu hỏi (vd “năm 2024”, “từ 1/7/2025”); "
                   "nếu không có, áp dụng quy định hiện hành.")
    include_ref = st.toggle("Dùng cả công văn, hỏi đáp", value=True,
                            help="Công văn và câu trả lời hỏi đáp chỉ có giá trị tham khảo, "
                                 "không tương đương luật, nghị định, thông tư.")
    research = st.toggle("Chế độ nghiên cứu", value=False, key="qa_research",
                         help="Hiện điểm số truy hồi và cho phép đổi cấu hình để thực nghiệm.")

    method, use_rerank = settings.retrieval_method, settings.use_reranker
    context_mode, final_k = "leaf", settings.final_k
    if research:
        method = st.selectbox("Phương pháp truy hồi", list(METHOD_LABELS),
                              format_func=METHOD_LABELS.get, key="qa_method")
        use_rerank = st.toggle("Dùng reranker", value=True, key="qa_rerank")
        context_mode = st.radio("Ngữ cảnh đưa vào mô hình", list(CONTEXT_LABELS),
                                format_func=CONTEXT_LABELS.get, key="qa_ctx")
        final_k = st.slider("Số đoạn đưa vào mô hình", 3, 8, settings.final_k, key="qa_k")

    if kb is not None:
        n_leaf = sum(1 for c in kb.chunks if c.get("is_leaf", True))
        st.caption(f"Kho dữ liệu: {len(kb.documents)} văn bản · {n_leaf:,} đoạn".replace(",", "."))
    if st.session_state.qa_messages:
        if st.button("Cuộc hội thoại mới", use_container_width=True):
            st.session_state.qa_messages = []
            st.rerun()


def _ask(q: str) -> None:
    st.session_state.qa_pending = q


def _feedback(key: str) -> None:
    value = st.session_state.get(key)
    msg = next((m for m in st.session_state.qa_messages if m.get("fb_key") == key), None)
    if msg is not None and value is not None:
        set_feedback(msg.get("log_id"), int(value))


# ------------------------------------------------------------------ main
if kb_error:
    st.warning(kb_error + " Quản trị viên cần cấu hình Secrets và chạy supabase/schema.sql.")

messages = st.session_state.qa_messages
if not messages:
    st.markdown(
        """<div class="tr-hero"><h1>Hỏi đáp chính sách thuế</h1>
        <p>Tra cứu quy định về thuế giá trị gia tăng, hóa đơn điện tử và thủ tục khai, nộp thuế
        dành cho doanh nghiệp nhỏ và vừa. Mỗi câu trả lời đều kèm căn cứ pháp lý và ngày hiệu lực.</p></div>
        <div class="tr-scope">
          <div><b>Đúng văn bản</b><span>Trích dẫn số hiệu, Điều, Khoản, Điểm và đường dẫn văn bản gốc.</span></div>
          <div><b>Đúng thời điểm</b><span>Chỉ dùng văn bản có hiệu lực tại thời điểm bạn hỏi.</span></div>
          <div><b>Biết từ chối</b><span>Nói rõ khi chưa đủ căn cứ hoặc câu hỏi thiếu thông tin.</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("**Câu hỏi gợi ý**")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        cols[i % 2].button(s, key=f"sugg_{i}", on_click=_ask, args=(s,),
                           use_container_width=True, disabled=kb is None)

for m in messages:
    if m["role"] == "user":
        render_user(m["content"])
    else:
        render_answer(m["result"], research=research, feedback_key=m["fb_key"], on_feedback=_feedback)
        st.write("")

question = st.chat_input("Hỏi về thuế GTGT, hóa đơn điện tử, khai và nộp thuế…", disabled=kb is None)
question = question or st.session_state.pop("qa_pending", None)

if question and kb is not None:
    messages.append({"role": "user", "content": question})
    render_user(question)
    cfg = PipelineConfig(
        method=method, use_reranker=use_rerank, candidate_k=settings.candidate_k, final_k=final_k,
        context_mode=context_mode, tax_categories=list(cats or []) or None, as_of=as_of,
        include_reference=include_ref,
    )
    with st.spinner("Đang tra cứu văn bản và soạn câu trả lời…"):
        result = answer_question(kb, question, settings, cfg,
                                 reranker=get_reranker() if use_rerank else None).to_dict()
    log_id = log_query({
        "session_id": st.session_state.session_id,
        "question": question,
        "as_of_label": result["query_time"].get("label"),
        "status": result["answer"].get("status"),
        "answer": result["answer"],
        "sources": [{"n": s["n"], "document_number": s["document"].get("document_number"),
                     "path": s["chunk"].get("path"), "chunk_id": s["chunk"].get("id"),
                     "candidate": s["candidate"]} for s in result["sources"]],
        "config": result["config"],
        "latency_ms": int(result["timings"].get("total_ms", 0)),
    })
    messages.append({"role": "assistant", "result": result, "log_id": log_id,
                     "fb_key": f"fb_{uuid.uuid4().hex[:10]}"})
    st.rerun()
