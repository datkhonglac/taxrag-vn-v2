"""Xử lý câu hỏi theo thời điểm (định hướng GVHD, mục 18).

Quy trình:
1. Trích xuất thời điểm/khoảng thời gian từ câu hỏi.
2. Lọc văn bản theo ngày có hiệu lực / hết hiệu lực.
3. Kiểm tra quan hệ sửa đổi, thay thế.
4. Chỉ dùng căn cứ phù hợp với thời điểm được hỏi.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .config import NORMATIVE_TYPES
from .text import normalize_doc_number


@dataclass
class QueryTime:
    start: date
    end: date
    label: str
    explicit: bool  # True nếu người dùng nêu rõ thời điểm
    matched: str = ""

    @property
    def is_range(self) -> bool:
        return self.start != self.end

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "label": self.label, "explicit": self.explicit, "matched": self.matched}


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


_DMY = r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})"
_DMY_WORDS = r"(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})"
_MY = r"tháng\s+(\d{1,2})\s*(?:[/\-.]|\s+năm\s+)\s*(\d{4})"
_QUARTER = r"quý\s+(I{1,3}|IV|[1-4])\s*(?:[/\-.]|\s+năm\s+)\s*(\d{4})"
_YEAR = r"(?:năm|kỳ tính thuế|kỳ thuế|niên độ)\s+(\d{4})"
_PREFIX = r"(từ|kể từ|sau|trước|đến|tới|tại|vào|trong)?\s*(?:ngày\s+)?"

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def extract_query_time(question: str, today: date | None = None) -> QueryTime:
    today = today or date.today()
    q = question.lower()

    for pattern in (_PREFIX + _DMY, _PREFIX + _DMY_WORDS):
        for m in re.finditer(pattern, q):
            prep = m.group(1) or ""
            d = _safe_date(int(m.group(4)), int(m.group(3)), int(m.group(2)))
            if d is None:
                continue
            if prep == "trước":
                d = d - timedelta(days=1)
            return QueryTime(d, d, f"ngày {d:%d/%m/%Y}", True, m.group(0).strip())

    m = re.search(_QUARTER, q, re.IGNORECASE)
    if m:
        qn = _ROMAN.get(m.group(1).upper(), None) or int(m.group(1))
        y = int(m.group(2))
        start = date(y, 3 * (qn - 1) + 1, 1)
        end_m = 3 * qn
        end = date(y, end_m, calendar.monthrange(y, end_m)[1])
        return QueryTime(start, end, f"quý {qn}/{y}", True, m.group(0))

    m = re.search(_MY, q)
    if m and 1 <= int(m.group(1)) <= 12:
        mo, y = int(m.group(1)), int(m.group(2))
        return QueryTime(date(y, mo, 1), date(y, mo, calendar.monthrange(y, mo)[1]),
                         f"tháng {mo}/{y}", True, m.group(0))

    m = re.search(_YEAR, q)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= today.year + 5:
            return QueryTime(date(y, 1, 1), date(y, 12, 31), f"năm {y}", True, m.group(0))

    relative = {
        "năm nay": today.year,
        "năm ngoái": today.year - 1,
        "năm trước": today.year - 1,
        "năm sau": today.year + 1,
        "năm tới": today.year + 1,
    }
    for phrase, y in relative.items():
        if phrase in q:
            return QueryTime(date(y, 1, 1), date(y, 12, 31), f"năm {y}", True, phrase)

    for phrase in ("hiện nay", "hiện hành", "hiện tại", "bây giờ", "mới nhất", "đang áp dụng"):
        if phrase in q:
            return QueryTime(today, today, f"hiện hành ({today:%d/%m/%Y})", True, phrase)

    return QueryTime(today, today, f"hiện hành ({today:%d/%m/%Y})", False)


def make_query_time(d: date, explicit: bool = True) -> QueryTime:
    return QueryTime(d, d, f"ngày {d:%d/%m/%Y}", explicit)


def _to_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@dataclass
class ValidityDecision:
    document_id: str
    usable: bool
    status: str  # in_force | not_yet | expired | unknown
    reference_only: bool
    notes: list[str] = field(default_factory=list)


def assess_document(doc: dict[str, Any], qt: QueryTime) -> ValidityDecision:
    """Xác định văn bản có dùng được làm căn cứ tại thời điểm hỏi hay không."""
    eff = _to_date(doc.get("effective_date"))
    exp = _to_date(doc.get("expiry_date"))
    status = (doc.get("legal_status") or "").strip()
    reference_only = doc.get("document_type") not in NORMATIVE_TYPES
    notes: list[str] = []

    if status == "Hết hiệu lực" and exp is None:
        # Đã hết hiệu lực nhưng không biết ngày -> chỉ dùng khi không xác định được
        return ValidityDecision(doc["id"], False, "expired", reference_only,
                                ["Văn bản đã hết hiệu lực (chưa nhập ngày hết hiệu lực)."])
    if eff is None:
        return ValidityDecision(doc["id"], True, "unknown", reference_only,
                                ["Chưa xác định được ngày có hiệu lực của văn bản."])
    if eff > qt.end:
        return ValidityDecision(doc["id"], False, "not_yet", reference_only,
                                [f"Văn bản chỉ có hiệu lực từ {eff:%d/%m/%Y}."])
    if exp is not None and exp <= qt.start:
        return ValidityDecision(doc["id"], False, "expired", reference_only,
                                [f"Văn bản hết hiệu lực từ {exp:%d/%m/%Y}."])
    if qt.is_range and (eff > qt.start or (exp is not None and exp <= qt.end)):
        notes.append("Văn bản chỉ có hiệu lực trong một phần khoảng thời gian được hỏi.")
    if status == "Hết hiệu lực một phần":
        notes.append("Văn bản đã hết hiệu lực một phần – cần kiểm tra văn bản sửa đổi.")
    return ValidityDecision(doc["id"], True, "in_force", reference_only, notes)


def filter_documents(
    documents: dict[str, dict[str, Any]],
    qt: QueryTime,
    *,
    tax_categories: list[str] | None = None,
    include_unknown: bool = True,
    include_reference: bool = True,
) -> tuple[set[str], dict[str, ValidityDecision]]:
    """Trả về (tập document_id được phép dùng, quyết định cho từng văn bản)."""
    allowed: set[str] = set()
    decisions: dict[str, ValidityDecision] = {}
    for doc_id, doc in documents.items():
        if tax_categories:
            cats = doc.get("tax_categories") or [doc.get("tax_category") or doc.get("topic")]
            if not set(cats) & set(tax_categories):
                continue
        dec = assess_document(doc, qt)
        decisions[doc_id] = dec
        if not dec.usable:
            continue
        if dec.status == "unknown" and not include_unknown:
            continue
        if dec.reference_only and not include_reference:
            continue
        allowed.add(doc_id)
    return allowed, decisions


def relation_notes(
    doc: dict[str, Any],
    relations: list[dict[str, Any]],
    documents_by_number: dict[str, dict[str, Any]],
    qt: QueryTime,
) -> list[str]:
    """Ghi chú về văn bản sửa đổi/thay thế liên quan, để hiển thị và đưa vào prompt."""
    from .config import RELATION_TYPES

    number = normalize_doc_number(doc.get("document_number"))
    notes = []
    for rel in relations:
        src = normalize_doc_number(rel.get("source_number"))
        tgt = normalize_doc_number(rel.get("target_number"))
        verb = RELATION_TYPES.get(rel.get("relation_type"), rel.get("relation_type"))
        if tgt == number:
            other = documents_by_number.get(src)
            eff = _to_date(other.get("effective_date")) if other else None
            when = f" (hiệu lực từ {eff:%d/%m/%Y})" if eff else ""
            pending = " – chưa có hiệu lực tại thời điểm hỏi" if eff and eff > qt.end else ""
            notes.append(f"Bị {verb} bởi {rel.get('source_number')}{when}{pending}.")
        elif src == number:
            notes.append(f"{verb.capitalize()} văn bản {rel.get('target_number')}.")
    return notes
