"""Cấu hình chung của hệ thống.

Thứ tự đọc: Streamlit Secrets (nếu đang chạy trong Streamlit) -> biến môi trường
-> giá trị mặc định. Module này không bắt buộc phải có Streamlit để các script
đánh giá (eval/) và unit test chạy được độc lập.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # python-dotenv là tùy chọn
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def get_setting(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


# ------------------------------------------------------------------
# Danh mục dùng chung (phạm vi đề tài)
# ------------------------------------------------------------------

TAX_CATEGORIES = [
    "Thuế giá trị gia tăng",
    "Hóa đơn điện tử",
    "Thủ tục khai và nộp thuế",
]

DOCUMENT_TYPES = [
    "Luật",
    "Nghị quyết",
    "Nghị định",
    "Thông tư",
    "Quyết định",
    "Công văn",
    "Hỏi đáp chính sách",
]

# Văn bản quy phạm pháp luật = căn cứ pháp lý chính.
# Công văn, hỏi đáp = chỉ có giá trị tham khảo (định hướng của GVHD, mục 7).
NORMATIVE_TYPES = {"Luật", "Nghị quyết", "Nghị định", "Thông tư", "Quyết định"}

LEGAL_STATUSES = [
    "Còn hiệu lực",
    "Hết hiệu lực một phần",
    "Chưa có hiệu lực",
    "Hết hiệu lực",
    "Chưa xác minh",
]

RELATION_TYPES = {
    "amends": "sửa đổi, bổ sung",
    "replaces": "thay thế",
    "guides": "hướng dẫn thi hành",
}

ISSUING_AUTHORITIES = [
    "Quốc hội",
    "Ủy ban Thường vụ Quốc hội",
    "Chính phủ",
    "Thủ tướng Chính phủ",
    "Bộ Tài chính",
    "Tổng cục Thuế / Cục Thuế",
    "Khác",
]


@dataclass
class Settings:
    gemini_api_key: str = field(default_factory=lambda: get_setting("GEMINI_API_KEY"))
    gemini_model: str = field(
        default_factory=lambda: get_setting("GEMINI_MODEL", "gemini-3.6-flash")
    )
    # Mô hình nhỏ/rẻ dùng cho reranking; mặc định dùng chung mô hình chính.
    rerank_model: str = field(default_factory=lambda: get_setting("RERANK_MODEL", ""))
    embedding_model: str = field(
        default_factory=lambda: get_setting("EMBEDDING_MODEL", "gemini-embedding-001")
    )
    # Phải khớp với kiểu vector(N) trong supabase/schema.sql
    embedding_dim: int = field(
        default_factory=lambda: int(get_setting("EMBEDDING_DIM", "768"))
    )
    admin_password: str = field(default_factory=lambda: get_setting("ADMIN_PASSWORD"))
    supabase_url: str = field(default_factory=lambda: get_setting("SUPABASE_URL"))
    supabase_key: str = field(default_factory=lambda: get_setting("SUPABASE_KEY"))

    # Truy hồi
    retrieval_method: str = "hybrid"  # bm25 | dense | hybrid
    use_reranker: bool = True
    candidate_k: int = 20  # số ứng viên trước rerank
    final_k: int = 5  # số đoạn đưa vào LLM
    # Ngưỡng từ chối sớm (trước khi gọi LLM). Cần hiệu chỉnh bằng eval/.
    min_rerank_score: float = 3.0  # thang 0-10
    min_dense_score: float = 0.35

    @property
    def effective_rerank_model(self) -> str:
        return self.rerank_model or self.gemini_model


def load_settings() -> Settings:
    return Settings()
