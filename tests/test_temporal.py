from datetime import date

from taxrag.temporal import assess_document, extract_query_time, filter_documents, make_query_time

TODAY = date(2026, 11, 3)


def qt(text):
    return extract_query_time(text, TODAY)


def test_explicit_dates():
    t = qt("Từ ngày 1/7/2025 thuế suất là bao nhiêu?")
    assert t.start == t.end == date(2025, 7, 1) and t.explicit
    assert qt("áp dụng ngày 15 tháng 3 năm 2024").start == date(2024, 3, 15)
    assert qt("trước ngày 01/07/2025 thì sao").end == date(2025, 6, 30)


def test_year_month_quarter():
    t = qt("Quy định áp dụng trong năm 2024 là gì?")
    assert (t.start, t.end) == (date(2024, 1, 1), date(2024, 12, 31)) and t.is_range
    t = qt("tháng 2/2024")
    assert (t.start, t.end) == (date(2024, 2, 1), date(2024, 2, 29))
    t = qt("tờ khai quý 3/2025")
    assert (t.start, t.end) == (date(2025, 7, 1), date(2025, 9, 30))
    assert qt("năm ngoái").start == date(2025, 1, 1)


def test_document_numbers_are_not_dates():
    t = qt("Luật 48/2024/QH15 quy định gì về thuế suất?")
    assert not t.explicit and t.start == TODAY


def test_default_is_today():
    t = qt("Hộ kinh doanh có phải nộp thuế GTGT không?")
    assert t.start == TODAY and not t.explicit
    assert qt("Thuế suất hiện nay?").explicit


DOCS = {
    "old": {"id": "old", "document_type": "Luật", "effective_date": "2021-01-01",
            "expiry_date": "2025-07-01", "legal_status": "Hết hiệu lực", "tax_categories": ["GTGT"]},
    "new": {"id": "new", "document_type": "Luật", "effective_date": "2025-07-01",
            "legal_status": "Còn hiệu lực", "tax_categories": ["GTGT"]},
    "cv": {"id": "cv", "document_type": "Công văn", "effective_date": "2025-09-01",
           "legal_status": "Còn hiệu lực", "tax_categories": ["HĐĐT"]},
    "nodate": {"id": "nodate", "document_type": "Thông tư", "legal_status": "Chưa xác minh",
               "tax_categories": ["GTGT"]},
}


def test_filter_by_point_in_time():
    allowed, dec = filter_documents(DOCS, make_query_time(date(2024, 5, 1)))
    assert allowed == {"old", "nodate"}
    assert dec["new"].status == "not_yet"
    allowed, _ = filter_documents(DOCS, make_query_time(date(2025, 7, 1)))
    assert "old" not in allowed and "new" in allowed  # ngày hết hiệu lực không còn áp dụng


def test_range_spanning_change_keeps_both_versions():
    allowed, dec = filter_documents(DOCS, qt("trong năm 2025"))
    assert {"old", "new"} <= allowed
    assert dec["new"].notes  # cảnh báo chỉ hiệu lực một phần khoảng


def test_category_and_reference_filters():
    allowed, _ = filter_documents(DOCS, make_query_time(TODAY), tax_categories=["HĐĐT"])
    assert allowed == {"cv"}
    allowed, _ = filter_documents(DOCS, make_query_time(TODAY), include_reference=False, include_unknown=False)
    assert allowed == {"new"}
    assert assess_document(DOCS["cv"], make_query_time(TODAY)).reference_only
