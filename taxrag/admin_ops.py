"""Nghiệp vụ quản trị dữ liệu (tách khỏi giao diện để kiểm thử được)."""

from __future__ import annotations

import csv
import io
from datetime import date
from typing import Any
from urllib.parse import urlparse

from .chunking import chunk_document, index_text
from .config import DOCUMENT_TYPES, LEGAL_STATUSES, TAX_CATEGORIES, Settings
from .ingest import ExtractedDocument, extract_file

# Nguồn chính thống (định hướng GVHD, mục 7). Các trang tổng hợp không chính thức
# có thể thiếu thông tin hiệu lực, sửa đổi.
OFFICIAL_DOMAINS = (
    "vbpl.vn", "chinhphu.vn", "congbao.chinhphu.vn", "vanban.chinhphu.vn", "quochoi.vn",
    "mof.gov.vn", "gdt.gov.vn", "thuedientu.gdt.gov.vn", "gov.vn",
)

CATALOG_COLUMNS = [
    "file", "document_number", "document_title", "document_type", "issuing_authority",
    "issued_date", "effective_date", "expiry_date", "legal_status", "tax_category",
    "applicable_subjects", "source_url",
]


def is_official_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def prepare_document(filename: str, data: bytes, strategy: str = "hierarchical"
                     ) -> tuple[ExtractedDocument, list[dict[str, Any]]]:
    ext = extract_file(filename, data)
    chunks = [c.to_dict() for c in chunk_document(ext.pages, strategy)] if ext.pages else []
    return ext, chunks


def structure_stats(chunks: list[dict[str, Any]]) -> dict[str, int]:
    leaves = [c for c in chunks if c.get("is_leaf", True)]
    return {
        "articles": len({c.get("article") for c in chunks if c.get("article")}),
        "leaves": len(leaves),
        "parents": len(chunks) - len(leaves),
        "clauses": sum(1 for c in leaves if c.get("level") == "clause"),
        "points": sum(1 for c in leaves if c.get("level") == "point"),
        "windows": sum(1 for c in leaves if c.get("level") in ("window", "appendix")),
        "max_words": max((len(c.get("content", "").split()) for c in leaves), default=0),
    }


def embed_chunks(settings: Settings, chunks: list[dict[str, Any]], doc: dict[str, Any],
                 progress=None) -> list[Any]:
    """Embedding cho đoạn lá; đoạn cha (chỉ dùng làm ngữ cảnh) để None."""
    from .llm import embed_texts

    idx = [i for i, c in enumerate(chunks) if c.get("is_leaf", True)]
    out: list[Any] = [None] * len(chunks)
    batch = 50
    for start in range(0, len(idx), batch):
        part = idx[start:start + batch]
        vecs = embed_texts(settings, [index_text(chunks[i], doc) for i in part])
        for i, v in zip(part, vecs):
            out[i] = v
        if progress:
            progress(min(1.0, (start + batch) / max(len(idx), 1)))
    return out


def validate_metadata(meta: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Trả về (lỗi bắt buộc sửa, cảnh báo)."""
    errors, warnings = [], []
    if not meta.get("document_number"):
        errors.append("Thiếu số hiệu văn bản.")
    if not meta.get("document_title"):
        errors.append("Thiếu tên văn bản.")
    if not meta.get("tax_categories"):
        errors.append("Chọn ít nhất một loại thuế liên quan.")
    if meta.get("document_type") not in DOCUMENT_TYPES:
        errors.append("Loại văn bản không hợp lệ.")
    eff, exp = meta.get("effective_date"), meta.get("expiry_date")
    if eff and exp and _d(exp) <= _d(eff):
        errors.append("Ngày hết hiệu lực phải sau ngày có hiệu lực.")
    if not eff:
        warnings.append("Chưa nhập ngày có hiệu lực – hệ thống sẽ gắn cảnh báo "
                        "“chưa xác định hiệu lực” khi dùng văn bản này.")
    if meta.get("legal_status") == "Hết hiệu lực" and not exp:
        warnings.append("Văn bản hết hiệu lực nên có ngày hết hiệu lực để trả lời câu hỏi theo thời điểm.")
    url = meta.get("source_url") or ""
    if not url:
        warnings.append("Chưa có URL nguồn – người dùng sẽ không mở được văn bản gốc.")
    elif not is_official_url(url):
        warnings.append("URL không thuộc cổng thông tin chính thức (vbpl.vn, chinhphu.vn, mof.gov.vn, "
                        "gdt.gov.vn…). Nên đối chiếu với nguồn chính thống.")
    return errors, warnings


def _d(v: Any) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])


def parse_catalog(raw: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows, problems = [], []
    missing = [c for c in ("file", "document_number", "document_title") if c not in (reader.fieldnames or [])]
    if missing:
        return [], [f"Thiếu cột bắt buộc: {', '.join(missing)}"]
    for i, r in enumerate(reader, start=2):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        cats = [c.strip() for c in (r.get("tax_category") or r.get("tax_categories") or "").split(";") if c.strip()]
        meta = {
            "document_number": r.get("document_number"),
            "document_title": r.get("document_title"),
            "document_type": r.get("document_type") or "Thông tư",
            "issuing_authority": r.get("issuing_authority") or None,
            "issued_date": r.get("issued_date") or None,
            "effective_date": r.get("effective_date") or None,
            "expiry_date": r.get("expiry_date") or None,
            "legal_status": r.get("legal_status") or "Chưa xác minh",
            "tax_categories": cats,
            "applicable_subjects": r.get("applicable_subjects") or None,
            "source_url": r.get("source_url") or None,
        }
        if meta["legal_status"] not in LEGAL_STATUSES:
            problems.append(f"Dòng {i}: tình trạng hiệu lực “{meta['legal_status']}” không hợp lệ.")
        unknown = [c for c in cats if c not in TAX_CATEGORIES]
        if unknown:
            problems.append(f"Dòng {i}: loại thuế ngoài danh mục: {', '.join(unknown)}")
        rows.append({"file": r.get("file"), "meta": meta, "line": i})
    return rows, problems


def catalog_template() -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CATALOG_COLUMNS)
    w.writerow(["ten-tep.pdf", "SỐ/HIỆU/VĂN-BẢN", "Tên đầy đủ của văn bản", "Thông tư", "Bộ Tài chính",
                "2025-01-01", "2025-02-15", "", "Còn hiệu lực",
                "Thuế giá trị gia tăng;Thủ tục khai và nộp thuế", "Doanh nghiệp nhỏ và vừa",
                "https://vbpl.vn/..."])
    return buf.getvalue()
