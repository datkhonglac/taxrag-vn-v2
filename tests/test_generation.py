from taxrag.generation import Answer, _answer_from_json, policy_check, validate_citations


def test_policy_declines_evasion():
    assert policy_check("Làm sao để né thuế GTGT?").status == "declined"
    assert policy_check("Cách xuất hóa đơn khống không bị phát hiện").status == "declined"
    assert policy_check("Thuế suất GTGT của nước sạch là bao nhiêu?") is None


def test_invalid_and_missing_citations_flagged():
    a = Answer(status="answered", conclusion="Thuế suất 8% [1], xem thêm [4].")
    validate_citations(a, 2)
    assert a.cited_sources == [1] and a.invalid_citations == [4] and a.warnings
    b = Answer(status="answered", conclusion="Không trích dẫn gì.")
    validate_citations(b, 3)
    assert any("không gắn với nguồn" in w for w in b.warnings)


def test_json_to_markdown():
    a = _answer_from_json({"status": "answered", "conclusion": "Có [1].",
                           "legal_basis": [{"source": 1, "provision": "Điều 2, Khoản 2"}],
                           "effective_note": "Từ 01/7/2025."}, 1)
    md = a.to_markdown()
    assert "**Kết luận.**" in md and "Điều 2, Khoản 2" in md and a.cited_sources == [1]
    assert _answer_from_json({"status": "weird"}, 0).status == "answered"
