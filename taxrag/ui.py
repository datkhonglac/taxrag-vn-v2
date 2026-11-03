"""Thành phần giao diện dùng chung: CSS, huy hiệu, thẻ căn cứ pháp lý, câu trả lời."""

from __future__ import annotations

import html
from typing import Any, Callable

import streamlit as st

from .generation import STATUSES

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
  font-family: 'Be Vietnam Pro', system-ui, sans-serif !important;
}
#MainMenu, footer, .stDeployButton {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 920px;}
[data-testid="stSidebar"] {border-right: 1px solid #E4E8EF;}
[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
h1, h2, h3 {letter-spacing: -0.01em; font-weight: 600 !important;}
.stChatMessage {background: transparent; padding: .25rem 0;}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {display:none;}
.tr-brand {display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem;}
.tr-logo {width:34px; height:34px; border-radius:9px; background:#1D4E89; color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:600; font-size:15px;}
.tr-brand b {font-size:17px; color:#1B2430; display:block; line-height:1.1}
.tr-brand span {font-size:12px; color:#6B7686}
.tr-hero {padding: 2.2rem 0 1rem;}
.tr-hero h1 {font-size: 30px; margin: 0 0 .4rem; color:#1B2430;}
.tr-hero p {color:#556070; font-size:16px; margin:0; max-width: 640px; line-height:1.6}
.tr-scope {display:grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap:.75rem; margin:1.4rem 0 1.6rem;}
.tr-scope div {border:1px solid #E4E8EF; border-radius:12px; padding:.9rem 1rem; background:#FAFBFD}
.tr-scope b {display:block; font-size:14px; color:#1B2430; margin-bottom:.2rem}
.tr-scope span {font-size:13px; color:#6B7686; line-height:1.5}
.tr-user {background:#F1F4F9; border-radius:14px; padding:.7rem 1rem; margin-left:auto;
  max-width:85%; width:fit-content; color:#1B2430; font-size:15px;}
.tr-badges {display:flex; gap:.4rem; flex-wrap:wrap; margin:.1rem 0 .6rem}
.tr-badge {font-size:12px; padding:.15rem .55rem; border-radius:6px; font-weight:500; white-space:nowrap}
.b-ok {background:#E6F4EA; color:#1E6B34} .b-warn {background:#FFF4DB; color:#8A5A00}
.b-bad {background:#FDECEC; color:#A12828} .b-info {background:#E8F0FB; color:#1D4E89}
.b-muted {background:#F1F3F6; color:#5A6472}
.tr-callout {border:1px solid #F3DFA9; background:#FFFBEF; color:#6E4E00; border-radius:10px;
  padding:.55rem .8rem; font-size:13px; margin:.5rem 0; line-height:1.5}
.tr-cards {display:grid; grid-template-columns: repeat(auto-fit, minmax(230px,1fr)); gap:.6rem; margin:.8rem 0 .4rem}
.tr-card {border:1px solid #E4E8EF; border-radius:10px; padding:.65rem .8rem; background:#fff; font-size:13px}
.tr-card .t {display:flex; justify-content:space-between; gap:.5rem; font-weight:600; color:#1B2430; font-size:13.5px}
.tr-card .t a {color:#1D4E89; text-decoration:none; font-weight:500; font-size:12px; white-space:nowrap}
.tr-card .p {color:#556070; margin:.15rem 0 .35rem}
.tr-card .m {display:flex; gap:.35rem; flex-wrap:wrap; align-items:center; color:#8A93A0; font-size:11.5px}
.tr-disclaimer {color:#8A93A0; font-size:12px; margin-top:.3rem}
.tr-src {border-left:3px solid #1D4E89; padding:.2rem 0 .2rem .8rem; margin:.4rem 0 1rem;
  white-space:pre-wrap; font-size:13.5px; color:#2A3442; line-height:1.6}
</style>
"""

STATUS_STYLE = {
    "answered": "b-ok", "partial": "b-warn", "need_more_info": "b-warn",
    "insufficient_evidence": "b-muted", "conflicting_sources": "b-bad",
    "out_of_scope": "b-muted", "declined": "b-bad", "error": "b-bad",
}
VALIDITY_LABEL = {
    "in_force": ("Còn hiệu lực", "b-ok"),
    "unknown": ("Chưa rõ hiệu lực", "b-warn"),
    "expired": ("Hết hiệu lực", "b-bad"),
    "not_yet": ("Chưa có hiệu lực", "b-warn"),
}

e = html.escape


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand(subtitle: str = "Hỏi đáp chính sách thuế") -> None:
    st.markdown(
        f'<div class="tr-brand"><div class="tr-logo">TR</div><div><b>TaxRAG VN</b>'
        f'<span>{e(subtitle)}</span></div></div>', unsafe_allow_html=True)


def badge(text: str, style: str = "b-info") -> str:
    return f'<span class="tr-badge {style}">{e(text)}</span>'


def _fmt_date(v: Any) -> str:
    if not v:
        return ""
    s = str(v)[:10]
    try:
        y, m, d = s.split("-")
        return f"{d}/{m}/{y}"
    except ValueError:
        return s


def citation_cards(sources: list[dict[str, Any]], cited: list[int] | None = None) -> None:
    if not sources:
        return
    cited_set = set(cited or [])
    cards = []
    for s in sources:
        if cited_set and s["n"] not in cited_set:
            continue
        doc, ch = s["document"], s["chunk"]
        label, style = VALIDITY_LABEL.get(s.get("validity_status", "unknown"), ("Chưa rõ", "b-warn"))
        title = f'{doc.get("document_type", "")} {doc.get("document_number", "")}'.strip()
        url = doc.get("source_url")
        link = f'<a href="{e(url)}" target="_blank">Văn bản gốc ↗</a>' if url else ""
        prov = " › ".join(x for x in [
            f'Điều {ch["article"]}' if ch.get("article") else "",
            f'Khoản {ch["clause"]}' if ch.get("clause") else "",
            f'Điểm {ch["point"]}' if ch.get("point") else "",
        ] if x) or ch.get("heading") or "Đoạn văn bản"
        page = f' · tr. {ch["page_start"]}' if ch.get("page_start") else ""
        meta = [badge(label, style)]
        if doc.get("reference_only"):
            meta.append(badge("Chỉ tham khảo", "b-muted"))
        eff = _fmt_date(doc.get("effective_date"))
        if eff:
            meta.append(f"từ {eff}")
        cards.append(
            f'<div class="tr-card"><div class="t"><span>[{s["n"]}] {e(title)}</span>{link}</div>'
            f'<div class="p">{e(prov)}{e(page)}</div><div class="m">{"".join(meta)}</div></div>')
    if cards:
        st.markdown(f'<div class="tr-cards">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_user(text: str) -> None:
    st.markdown(f'<div class="tr-user">{e(text)}</div>', unsafe_allow_html=True)


def render_answer(
    result: dict[str, Any],
    *,
    research: bool = False,
    feedback_key: str | None = None,
    on_feedback: Callable[[str], None] | None = None,
) -> None:
    ans = result.get("answer", {})
    status = ans.get("status", "answered")
    qt = result.get("query_time", {})
    badges = [badge(STATUSES.get(status, status), STATUS_STYLE.get(status, "b-info"))]
    if qt.get("label"):
        badges.append(badge(f"Áp dụng: {qt['label']}", "b-info"))
    if research and result.get("timings", {}).get("total_ms"):
        badges.append(badge(f'{result["timings"]["total_ms"] / 1000:.1f} giây', "b-muted"))
    st.markdown(f'<div class="tr-badges">{"".join(badges)}</div>', unsafe_allow_html=True)

    st.markdown(ans.get("markdown") or ans.get("conclusion") or "")

    for w in (ans.get("warnings") or []) + (result.get("warnings") or []):
        st.markdown(f'<div class="tr-callout">{e(w)}</div>', unsafe_allow_html=True)

    sources = result.get("sources") or []
    if status not in ("out_of_scope", "declined", "insufficient_evidence"):
        citation_cards(sources, ans.get("cited_sources"))

    if sources:
        with st.expander(f"Xem {len(sources)} đoạn văn bản được dùng làm ngữ cảnh"):
            for s in sources:
                doc, ch = s["document"], s["chunk"]
                st.markdown(f'**[{s["n"]}] {doc.get("document_number", "")} — {ch.get("path", "")}**')
                notes = s.get("validity_notes") or []
                if notes:
                    st.caption(" · ".join(notes))
                if research and ch.get("context"):
                    st.caption(f"Ngữ cảnh: {ch['context']}")
                st.markdown(f'<div class="tr-src">{e(ch.get("content", ""))}</div>',
                            unsafe_allow_html=True)

    if research:
        render_research_panel(result)

    if feedback_key:
        cols = st.columns([1, 6])
        with cols[0]:
            st.feedback("thumbs", key=feedback_key,
                        on_change=(lambda: on_feedback(feedback_key)) if on_feedback else None)
        with cols[1]:
            st.markdown('<div class="tr-disclaimer">Kết quả chỉ để tham khảo, không thay thế '
                        'ý kiến chính thức của cơ quan thuế.</div>', unsafe_allow_html=True)


def render_research_panel(result: dict[str, Any]) -> None:
    with st.expander("Chi tiết truy hồi (chế độ nghiên cứu)"):
        cfg = result.get("config", {})
        st.caption(
            f"Phương pháp: {cfg.get('method')} · Reranker: {'bật' if cfg.get('use_reranker') else 'tắt'}"
            f" · Ngữ cảnh: {cfg.get('context_mode')} · Văn bản hợp lệ: {result.get('allowed_documents')}"
        )
        cands = result.get("candidates") or []
        if cands:
            rows = [{
                "#": i + 1,
                "Văn bản": c.get("document_number"),
                "Vị trí": c.get("path"),
                "Hạng BM25": c.get("bm25_rank"),
                "Hạng dense": c.get("dense_rank"),
                "Cosine": None if c.get("dense_score") is None else round(c["dense_score"], 3),
                "Điểm hợp nhất": round(c.get("score", 0), 4),
                "Rerank (0-10)": c.get("rerank_score"),
                "Lý do rerank": c.get("rerank_reason"),
            } for i, c in enumerate(cands)]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        excluded = result.get("excluded_documents") or []
        if excluded:
            st.caption("Văn bản bị loại do hiệu lực: " + "; ".join(
                f'{x["document_number"]} ({", ".join(x.get("notes") or [x["status"]])})' for x in excluded))
        t = result.get("timings") or {}
        if t:
            st.caption(" · ".join(f"{k.replace('_ms', '')}: {v:.0f} ms" for k, v in t.items()))
