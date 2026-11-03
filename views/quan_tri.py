"""Trang quản trị dữ liệu (chỉ hiện khi đã đăng nhập quản trị) – mục 23 định hướng GVHD."""

import csv
import html
import io
import json
import statistics
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from taxrag.admin_ops import (
    catalog_template,
    embed_chunks,
    parse_catalog,
    prepare_document,
    structure_stats,
    validate_metadata,
)
from taxrag.config import (
    DOCUMENT_TYPES,
    ISSUING_AUTHORITIES,
    LEGAL_STATUSES,
    RELATION_TYPES,
    TAX_CATEGORIES,
)
from taxrag.generation import REFUSAL_STATUSES, STATUSES
from taxrag.ingest import SUPPORTED_TYPES
from taxrag.services import get_settings, get_store, invalidate_knowledge_base, load_knowledge_base

if not st.session_state.get("is_admin"):
    st.error("Bạn cần đăng nhập quản trị.")
    st.stop()

settings = get_settings()
store = get_store()
st.title("Quản trị dữ liệu")
if store is None:
    st.error("Chưa cấu hình SUPABASE_URL / SUPABASE_KEY trong Secrets.")
    st.stop()
ok, msg = store.ping()
if not ok:
    st.error(msg)
    st.stop()

admin_name = st.session_state.setdefault("admin_name", "")


def _date(v):
    if not v:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _iso(v):
    if v is None:
        return None
    try:
        import pandas as pd

        if pd.isna(v):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return v or None


def _save(filename: str, data: bytes, meta: dict, uploaded_by: str, cached=None, progress=None):
    ext, chunks = cached or prepare_document(filename, data)
    if not chunks:
        raise ValueError("Không trích xuất được chữ từ tệp.")
    vectors = None
    if settings.gemini_api_key:
        vectors = embed_chunks(settings, chunks, meta, progress=progress)
    clean = {k: _iso(v) if k.endswith("_date") else v for k, v in meta.items()}
    clean["filename"] = filename
    doc_id = store.save_document(clean, chunks, page_count=ext.page_count, uploaded_by=uploaded_by,
                                 embeddings=vectors, content_hash=ext.content_hash)
    return doc_id, ext, chunks


tabs = st.tabs(["Tổng quan", "Nạp văn bản", "Nạp hàng loạt", "Danh mục văn bản",
                "Quan hệ văn bản", "Kiểm tra đoạn", "Lịch sử câu hỏi", "Bảo trì"])

documents = store.list_documents(include_processing=True)
numbers = sorted({d["document_number"] for d in documents})

# ------------------------------------------------------------------ 1. Tổng quan
with tabs[0]:
    kb, _ = load_knowledge_base()
    ready = [d for d in documents if d.get("ingest_status", "ready") == "ready"]
    c = st.columns(4)
    c[0].metric("Văn bản", len(ready))
    c[1].metric("Đoạn (lá)", sum(1 for x in kb.chunks if x.get("is_leaf", True)) if kb else 0)
    c[2].metric("Còn hiệu lực", sum(1 for d in ready if d.get("legal_status") == "Còn hiệu lực"))
    c[3].metric("Chưa xác minh", sum(1 for d in ready if d.get("legal_status") == "Chưa xác minh"))

    issues = []
    no_eff = [d["document_number"] for d in ready if not d.get("effective_date")]
    if no_eff:
        issues.append(f"{len(no_eff)} văn bản chưa có ngày hiệu lực: {', '.join(no_eff[:8])}")
    no_url = [d["document_number"] for d in ready if not d.get("source_url")]
    if no_url:
        issues.append(f"{len(no_url)} văn bản chưa có URL nguồn.")
    stuck = [d["document_number"] for d in documents if d.get("ingest_status") == "processing"]
    if stuck:
        issues.append(f"{len(stuck)} văn bản đang ở trạng thái nạp dở (có thể do lỗi): {', '.join(stuck)}")
    if kb:
        legacy = sum(1 for x in kb.chunks if x.get("level") == "legacy")
        if legacy:
            issues.append(f"{legacy} đoạn còn chia theo trang kiểu v2 – hãy nạp lại văn bản để chia "
                          "theo Điều–Khoản–Điểm.")
    st.markdown("##### Cần xử lý")
    if issues:
        for i in issues:
            st.warning(i)
    else:
        st.success("Không phát hiện vấn đề về dữ liệu.")

    st.markdown("##### Phân bố theo loại thuế")
    rows = []
    for cat in TAX_CATEGORIES:
        sub = [d for d in ready if cat in (d.get("tax_categories") or [])]
        rows.append({"Loại thuế": cat, "Số văn bản": len(sub),
                     **{s: sum(1 for d in sub if d.get("legal_status") == s) for s in LEGAL_STATUSES}})
    st.dataframe(rows, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 2. Nạp văn bản
with tabs[1]:
    st.caption("Mỗi lần nạp một văn bản, có bước xem trước phân đoạn để kiểm tra chất lượng trước khi lưu. "
               "Nạp lại văn bản cùng số hiệu sẽ thay thế bản cũ (an toàn: bản cũ chỉ bị xóa khi bản mới đã lưu xong).")
    up = st.file_uploader("Tệp văn bản", type=SUPPORTED_TYPES, key="ad_file")
    with st.form("ad_meta"):
        c1, c2 = st.columns([2, 1])
        number = c1.text_input("Số hiệu văn bản *", placeholder="48/2024/QH15")
        doc_type = c2.selectbox("Loại văn bản *", DOCUMENT_TYPES)
        title = st.text_input("Tên văn bản *", value=Path(up.name).stem if up else "")
        c1, c2 = st.columns(2)
        authority = c1.selectbox("Cơ quan ban hành", ISSUING_AUTHORITIES)
        status = c2.selectbox("Tình trạng hiệu lực", LEGAL_STATUSES)
        c1, c2, c3 = st.columns(3)
        issued = c1.date_input("Ngày ban hành", value=None, format="DD/MM/YYYY")
        effective = c2.date_input("Ngày có hiệu lực", value=None, format="DD/MM/YYYY")
        expiry = c3.date_input("Ngày hết hiệu lực", value=None, format="DD/MM/YYYY")
        cats = st.multiselect("Loại thuế liên quan *", TAX_CATEGORIES)
        subjects = st.text_input("Đối tượng áp dụng", placeholder="Doanh nghiệp nhỏ và vừa, hộ kinh doanh…")
        url = st.text_input("URL nguồn chính thức", placeholder="https://vbpl.vn/…")
        notes = st.text_area("Ghi chú", height=70)
        uploader_name = st.text_input("Người nạp", value=admin_name)
        b1, b2 = st.columns(2)
        do_preview = b1.form_submit_button("Xem trước phân đoạn", use_container_width=True)
        do_save = b2.form_submit_button("Lưu vào kho", type="primary", use_container_width=True)

    meta = {
        "document_number": number.strip(), "document_title": title.strip(), "document_type": doc_type,
        "issuing_authority": authority, "issued_date": issued, "effective_date": effective,
        "expiry_date": expiry, "legal_status": status, "tax_categories": cats,
        "applicable_subjects": subjects.strip() or None, "source_url": url.strip() or None,
        "notes": notes.strip() or None, "retrieval_date": date.today(),
    }
    if (do_preview or do_save) and not up:
        st.error("Hãy chọn tệp văn bản.")
    elif do_preview or do_save:
        st.session_state.admin_name = uploader_name
        errors, warns = validate_metadata(meta)
        for w in warns:
            st.warning(w)
        data = up.getvalue()
        cache_key = f"prep_{up.name}_{len(data)}"
        if cache_key not in st.session_state:
            with st.spinner("Đang trích xuất và phân đoạn…"):
                st.session_state[cache_key] = prepare_document(up.name, data)
        ext, chunks = st.session_state[cache_key]
        dup = [d["document_number"] for d in documents
               if d.get("content_hash") == ext.content_hash and d["document_number"] != meta["document_number"]]
        if dup:
            st.warning(f"Nội dung tệp trùng với văn bản đã có: {', '.join(dup)}")
        if meta["document_number"] in numbers:
            st.info(f"Đã có văn bản số {meta['document_number']} – lưu sẽ thay thế bản hiện có.")
        for w in ext.warnings:
            st.warning(w)

        stats = structure_stats(chunks)
        m = st.columns(5)
        m[0].metric("Trang", ext.page_count)
        m[1].metric("Số Điều", stats["articles"])
        m[2].metric("Đoạn lá", stats["leaves"])
        m[3].metric("Đoạn khoản / điểm", f'{stats["clauses"]} / {stats["points"]}')
        m[4].metric("Đoạn dài nhất (từ)", stats["max_words"])
        if stats["articles"] == 0:
            st.warning("Không nhận diện được “Điều …” – văn bản sẽ được chia theo độ dài cố định. "
                       "Kiểm tra lại nếu đây là luật, nghị định hoặc thông tư.")
        st.dataframe(
            [{"#": i, "Cấp": c["level"], "Vị trí": c["path"], "Số từ": len(c["content"].split()),
              "Trang": c.get("page_start"), "Lá": c.get("is_leaf", True)} for i, c in enumerate(chunks)],
            hide_index=True, use_container_width=True, height=260,
        )
        if do_save:
            if errors:
                for e in errors:
                    st.error(e)
            else:
                bar = st.progress(0.0, text="Đang tạo embedding…")
                try:
                    doc_id, _, _ = _save(up.name, data, meta, uploader_name or "Admin",
                                         cached=(ext, chunks),
                                         progress=lambda p: bar.progress(p, text="Đang tạo embedding…"))
                    bar.empty()
                    invalidate_knowledge_base()
                    st.session_state.pop(cache_key, None)
                    st.success(f"Đã lưu {meta['document_number']} với {stats['leaves']} đoạn.")
                except Exception as exc:
                    bar.empty()
                    st.error(f"Không lưu được: {exc}")

# ------------------------------------------------------------------ 3. Nạp hàng loạt
with tabs[2]:
    st.caption("Nạp nhiều văn bản theo một tệp danh mục CSV (mỗi dòng một văn bản). "
               "Danh mục này cũng chính là bảng “Lập danh mục văn bản” của đề tài.")
    st.download_button("Tải mẫu danh mục CSV", catalog_template().encode("utf-8-sig"),
                       "danh_muc_van_ban.csv", "text/csv")
    cat_file = st.file_uploader("Danh mục (CSV, UTF-8)", type=["csv"], key="ad_catalog")
    files = st.file_uploader("Các tệp văn bản", type=SUPPORTED_TYPES, accept_multiple_files=True,
                             key="ad_batch_files")
    if cat_file and files:
        rows, problems = parse_catalog(cat_file.getvalue())
        for p in problems:
            st.warning(p)
        by_name = {f.name: f for f in files}
        plan = [{"Dòng": r["line"], "Tệp": r["file"], "Số hiệu": r["meta"]["document_number"],
                 "Có tệp": r["file"] in by_name,
                 "Lỗi": "; ".join(validate_metadata(r["meta"])[0])} for r in rows]
        st.dataframe(plan, hide_index=True, use_container_width=True)
        runnable = [r for r in rows if r["file"] in by_name and not validate_metadata(r["meta"])[0]]
        if st.button(f"Nạp {len(runnable)} văn bản hợp lệ", type="primary", disabled=not runnable):
            bar = st.progress(0.0)
            results = []
            for i, r in enumerate(runnable, start=1):
                bar.progress((i - 1) / len(runnable), text=f"{i}/{len(runnable)}: {r['file']}")
                f = by_name[r["file"]]
                try:
                    _, ext, chunks = _save(f.name, f.getvalue(), r["meta"], admin_name or "Admin")
                    results.append({"Tệp": f.name, "Kết quả": f"OK – {len(chunks)} đoạn"})
                except Exception as exc:
                    results.append({"Tệp": f.name, "Kết quả": f"Lỗi: {exc}"})
            bar.empty()
            invalidate_knowledge_base()
            st.dataframe(results, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 4. Danh mục
with tabs[3]:
    if not documents:
        st.info("Kho chưa có văn bản.")
    else:
        editable = [{
            "id": d["id"], "Số hiệu": d["document_number"], "Tên văn bản": d.get("document_title") or d.get("document_name"),
            "Loại": d.get("document_type"), "Cơ quan": d.get("issuing_authority"),
            "Ban hành": _date(d.get("issued_date")), "Hiệu lực từ": _date(d.get("effective_date")),
            "Hết hiệu lực": _date(d.get("expiry_date")), "Tình trạng": d.get("legal_status"),
            "Loại thuế": ", ".join(d.get("tax_categories") or []), "URL": d.get("source_url"),
            "Số đoạn": d.get("chunk_count"), "Nạp": d.get("ingest_status"),
        } for d in documents]
        edited = st.data_editor(
            editable, hide_index=True, use_container_width=True, key="doc_editor",
            disabled=["id", "Số hiệu", "Loại thuế", "Số đoạn", "Nạp"],
            column_config={
                "id": None,
                "Loại": st.column_config.SelectboxColumn(options=DOCUMENT_TYPES),
                "Cơ quan": st.column_config.SelectboxColumn(options=ISSUING_AUTHORITIES),
                "Tình trạng": st.column_config.SelectboxColumn(options=LEGAL_STATUSES),
                "Ban hành": st.column_config.DateColumn(format="DD/MM/YYYY"),
                "Hiệu lực từ": st.column_config.DateColumn(format="DD/MM/YYYY"),
                "Hết hiệu lực": st.column_config.DateColumn(format="DD/MM/YYYY"),
                "URL": st.column_config.LinkColumn(),
            },
        )
        field_map = {"Tên văn bản": "document_title", "Loại": "document_type", "Cơ quan": "issuing_authority",
                     "Ban hành": "issued_date", "Hiệu lực từ": "effective_date",
                     "Hết hiệu lực": "expiry_date", "Tình trạng": "legal_status", "URL": "source_url"}
        if st.button("Lưu thay đổi metadata", type="primary"):
            orig = {r["id"]: r for r in editable}
            rows = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
            changed = 0
            for r in rows:
                o = orig.get(r["id"])
                diff = {field_map[k]: _iso(r[k]) for k in field_map if o and r[k] != o[k]}
                if diff:
                    if "document_title" in diff:
                        diff["document_name"] = diff["document_title"]
                    store.update_document(r["id"], diff)
                    changed += 1
            invalidate_knowledge_base()
            st.success(f"Đã cập nhật {changed} văn bản.") if changed else st.info("Không có thay đổi.")

        st.divider()
        st.markdown("##### Xóa văn bản")
        to_delete = st.multiselect("Chọn văn bản", [f'{d["document_number"]} — {d.get("document_title") or ""}'
                                                    for d in documents], key="del_docs")
        confirm = st.checkbox("Tôi xác nhận xóa văn bản đã chọn và toàn bộ đoạn của chúng")
        if st.button("Xóa", disabled=not (to_delete and confirm)):
            labels = {f'{d["document_number"]} — {d.get("document_title") or ""}': d["id"] for d in documents}
            for lab in to_delete:
                store.delete_document(labels[lab])
            invalidate_knowledge_base()
            st.success(f"Đã xóa {len(to_delete)} văn bản.")
            st.rerun()

# ------------------------------------------------------------------ 5. Quan hệ văn bản
with tabs[4]:
    st.caption("Khai báo văn bản sửa đổi, thay thế, hướng dẫn để hệ thống cảnh báo đúng khi trả lời theo thời điểm.")
    with st.form("rel_form"):
        c1, c2, c3 = st.columns([2, 1, 2])
        src = c1.selectbox("Văn bản (mới)", numbers, index=None, placeholder="Chọn số hiệu")
        rtype = c2.selectbox("Quan hệ", list(RELATION_TYPES), format_func=RELATION_TYPES.get)
        tgt = c3.selectbox("Văn bản (bị tác động)", numbers, index=None, placeholder="Chọn số hiệu")
        note = st.text_input("Ghi chú", placeholder="Vd: sửa đổi khoản 2 Điều 5")
        auto_expire = st.checkbox("Với quan hệ “thay thế”: đặt ngày hết hiệu lực của văn bản bị thay thế "
                                  "bằng ngày có hiệu lực của văn bản mới", value=True)
        if st.form_submit_button("Thêm quan hệ", type="primary"):
            if not src or not tgt or src == tgt:
                st.error("Chọn hai văn bản khác nhau.")
            else:
                store.add_relation(src, tgt, rtype, note)
                if rtype == "replaces" and auto_expire:
                    new = next(d for d in documents if d["document_number"] == src)
                    old = next(d for d in documents if d["document_number"] == tgt)
                    if new.get("effective_date"):
                        store.update_document(old["id"], {"expiry_date": new["effective_date"],
                                                          "legal_status": "Hết hiệu lực"})
                invalidate_knowledge_base()
                st.success("Đã thêm quan hệ.")
    relations = store.list_relations()
    if relations:
        st.dataframe([{"Văn bản": r["source_number"], "Quan hệ": RELATION_TYPES.get(r["relation_type"]),
                       "Văn bản bị tác động": r["target_number"], "Ghi chú": r.get("note")}
                      for r in relations], hide_index=True, use_container_width=True)
        rid = st.selectbox("Xóa quan hệ", relations, index=None, format_func=lambda r:
                           f'{r["source_number"]} {RELATION_TYPES.get(r["relation_type"])} {r["target_number"]}')
        if rid and st.button("Xóa quan hệ đã chọn"):
            store.delete_relation(rid["id"])
            invalidate_knowledge_base()
            st.rerun()

# ------------------------------------------------------------------ 6. Kiểm tra đoạn
with tabs[5]:
    doc = st.selectbox("Văn bản", documents, index=None, placeholder="Chọn văn bản để kiểm tra",
                       format_func=lambda d: f'{d["document_number"]} — {d.get("document_title") or ""}')
    if doc:
        rows = store.load_chunks([doc["id"]])
        if not rows:
            st.info("Văn bản chưa có đoạn nào.")
        else:
            long_ = [r for r in rows if len((r.get("content") or "").split()) > 600 and r.get("is_leaf", True)]
            if long_:
                st.warning(f"{len(long_)} đoạn dài hơn 600 từ – nên kiểm tra nhận diện Khoản/Điểm.")
            st.dataframe([{"#": r.get("chunk_index"), "Cấp": r.get("level"), "Lá": r.get("is_leaf"),
                           "Vị trí": r.get("path"), "Số từ": len((r.get("content") or "").split()),
                           "Trang": r.get("page_number")} for r in rows],
                         hide_index=True, use_container_width=True, height=300)
            pick = st.number_input("Xem nội dung đoạn số", 0, len(rows) - 1, 0)
            r = rows[int(pick)]
            st.markdown(f"**{r.get('path') or ''}**")
            if r.get("context"):
                st.caption(f"Ngữ cảnh: {r['context']}")
            st.markdown(f'<div class="tr-src">{html.escape(r.get("content") or "")}</div>',
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ 7. Lịch sử câu hỏi
with tabs[6]:
    try:
        logs = store.list_query_logs(limit=1000)
    except Exception as exc:
        logs = []
        st.error(f"Chưa đọc được bảng query_logs – hãy chạy supabase/schema.sql. ({exc})")
    if logs:
        rated = [lg for lg in logs if lg.get("feedback") is not None]
        refused = [lg for lg in logs if lg.get("status") in REFUSAL_STATUSES]
        lat = [lg["latency_ms"] for lg in logs if lg.get("latency_ms")]
        c = st.columns(4)
        c[0].metric("Câu hỏi", len(logs))
        c[1].metric("Tỷ lệ từ chối / chưa đủ căn cứ", f"{len(refused) / len(logs):.0%}")
        c[2].metric("Đánh giá hữu ích", f"{sum(1 for x in rated if x['feedback'] == 1) / len(rated):.0%}"
                    if rated else "—")
        c[3].metric("Thời gian trung vị", f"{statistics.median(lat) / 1000:.1f} giây" if lat else "—")
        only_bad = st.toggle("Chỉ hiện câu bị đánh giá không hữu ích")
        view = [lg for lg in logs if not only_bad or lg.get("feedback") == 0]
        st.dataframe([{
            "Thời gian": lg["created_at"][:16].replace("T", " "), "Câu hỏi": lg["question"],
            "Trạng thái": STATUSES.get(lg.get("status"), lg.get("status")),
            "Thời điểm áp dụng": lg.get("as_of_label"),
            "Đánh giá": {1: "Hữu ích", 0: "Không hữu ích"}.get(lg.get("feedback"), ""),
            "ms": lg.get("latency_ms"),
        } for lg in view], hide_index=True, use_container_width=True)

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["created_at", "question", "status", "as_of", "feedback", "latency_ms", "answer"])
        for lg in logs:
            w.writerow([lg["created_at"], lg["question"], lg.get("status"), lg.get("as_of_label"),
                        lg.get("feedback"), lg.get("latency_ms"),
                        (lg.get("answer") or {}).get("markdown", "")])
        jsonl = "\n".join(json.dumps({
            "id": f"log-{lg['id'][:8]}", "question": lg["question"], "tax_category": "",
            "difficulty": None, "intent": "", "as_of": None, "should_refuse": None,
            "gold": [], "reference_answer": "", "source": "query_logs",
        }, ensure_ascii=False) for lg in logs)
        c1, c2 = st.columns(2)
        c1.download_button("Tải lịch sử (CSV)", buf.getvalue().encode("utf-8-sig"), "lich_su_cau_hoi.csv",
                           "text/csv", use_container_width=True)
        c2.download_button("Xuất khung tập câu hỏi đánh giá (JSONL)", jsonl.encode("utf-8"),
                           "cau_hoi_tu_log.jsonl", use_container_width=True,
                           help="Dùng câu hỏi thật của người dùng làm đầu vào cho eval/; "
                                "cần gắn nhãn căn cứ (gold) thủ công.")
    elif not logs:
        st.info("Chưa có câu hỏi nào được ghi nhận.")

# ------------------------------------------------------------------ 8. Bảo trì
with tabs[7]:
    st.markdown("##### Tạo embedding cho đoạn còn thiếu")
    if not settings.gemini_api_key:
        st.info("Chưa cấu hình GEMINI_API_KEY.")
    elif st.button("Kiểm tra và tạo embedding"):
        missing = [r for r in store.chunks_missing_embedding() if r.get("is_leaf") is not False]
        if not missing:
            st.success("Mọi đoạn đều đã có embedding.")
        else:
            from taxrag.chunking import index_text
            from taxrag.llm import embed_texts

            docs = {d["id"]: d for d in documents}
            bar = st.progress(0.0)
            for start in range(0, len(missing), 50):
                part = missing[start:start + 50]
                vecs = embed_texts(settings, [index_text(r, docs.get(r["document_id"], {})) for r in part])
                store.set_embeddings([r["id"] for r in part], vecs)
                bar.progress(min(1.0, (start + 50) / len(missing)), text=f"{start + len(part)}/{len(missing)}")
            invalidate_knowledge_base()
            st.success(f"Đã tạo embedding cho {len(missing)} đoạn.")

    st.markdown("##### Bộ nhớ đệm")
    if st.button("Làm mới kho dữ liệu trong bộ nhớ"):
        invalidate_knowledge_base()
        st.success("Đã làm mới. Lần hỏi tiếp theo sẽ nạp lại kho từ Supabase.")

    st.markdown("##### Xóa toàn bộ kho dùng chung")
    typed = st.text_input("Gõ XÓA TOÀN BỘ để xác nhận")
    if st.button("Xóa toàn bộ văn bản", disabled=typed.strip() != "XÓA TOÀN BỘ"):
        store.delete_all_documents()
        invalidate_knowledge_base()
        st.success("Đã xóa toàn bộ văn bản và đoạn.")
