"""Các dịch vụ dùng chung cho giao diện Streamlit (có bộ nhớ đệm).

Sửa lỗi v2: v2 tải lại toàn bộ kho từ Supabase và dựng lại TF-IDF ở *mỗi* lần
người dùng thao tác. Ở đây kho + chỉ mục được dựng một lần và dùng chung cho mọi
phiên; admin thay đổi dữ liệu thì gọi ``invalidate_knowledge_base()``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from .config import Settings, load_settings
from .db import SupabaseError, SupabaseStore
from .pipeline import KnowledgeBase
from .rerank import GeminiReranker


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return load_settings()


@st.cache_resource(show_spinner=False)
def get_store() -> SupabaseStore | None:
    try:
        return SupabaseStore(get_settings())
    except SupabaseError:
        return None


@st.cache_resource(show_spinner=False)
def get_reranker():
    s = get_settings()
    return GeminiReranker(s) if s.gemini_api_key else None


def db_chunk_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["page_start"] = row.pop("page_number", None)
    row.setdefault("is_leaf", True)
    if row.get("is_leaf") is None:
        row["is_leaf"] = True
    row["covers"] = row.get("covers") or []
    return row


@st.cache_resource(show_spinner="Đang nạp kho văn bản…", ttl=900)
def load_knowledge_base() -> tuple[KnowledgeBase | None, str | None]:
    store = get_store()
    if store is None:
        return None, "Chưa cấu hình Supabase (SUPABASE_URL, SUPABASE_KEY)."
    try:
        documents = store.list_documents()
        chunks = [db_chunk_to_dict(r) for r in store.load_chunks()]
        try:
            relations = store.list_relations()
        except SupabaseError:
            relations = []  # chưa chạy schema v3
        has_vectors = bool(store.request("GET", "chunks", params={
            "select": "id", "embedding": "not.is.null", "limit": "1"}))
    except SupabaseError as exc:
        return None, f"Không đọc được Supabase: {exc}"
    doc_ids = {d["id"] for d in documents}
    chunks = [c for c in chunks if c.get("document_id") in doc_ids]
    dense = store.match_chunks if has_vectors and get_settings().gemini_api_key else None
    return KnowledgeBase(documents, chunks, relations, dense_search=dense), None


def invalidate_knowledge_base() -> None:
    load_knowledge_base.clear()


def log_query(record: dict[str, Any]) -> str | None:
    store = get_store()
    if store is None:
        return None
    try:
        return store.log_query(record)
    except Exception:
        return None  # ghi log lỗi không được làm hỏng trải nghiệm hỏi đáp


def set_feedback(log_id: str | None, value: int, note: str = "") -> None:
    store = get_store()
    if store is None or not log_id:
        return
    try:
        store.set_feedback(log_id, value, note)
    except Exception:
        pass
