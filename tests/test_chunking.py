from conftest import SAMPLE

from taxrag.chunking import chunk_document, parse_structure, remove_headers_footers

LAW = (SAMPLE / "lm-02-2024.txt").read_text(encoding="utf-8")


def test_parse_articles_clauses_points():
    articles, appendices, preamble = parse_structure(LAW)
    assert [a.article for a in articles] == ["1", "2", "5", "6", "7"]
    art5 = articles[2]
    assert art5.chapter.startswith("Chương II")
    assert [c.label for c in art5.children] == ["1", "2", "3"]
    assert [p.label for p in art5.children[0].children] == ["a", "b", "c"]
    assert preamble is not None and appendices == []


def test_recipients_block_not_part_of_last_article():
    articles, _, _ = parse_structure(LAW)
    last = LAW[articles[-1].start:articles[-1].end]
    assert "Nơi nhận" not in last


def test_structural_clause_level_with_paths():
    chunks = chunk_document([(1, LAW)], "structural", short_article_words=0, max_leaf_words=15)
    paths = [c.path for c in chunks]
    assert any(p.endswith("Điều 6 › Khoản 3") for p in paths)
    point = next(c for c in chunks if c.level == "point" and c.point == "b")
    assert point.covers == ["5", "5.1"]
    assert "0%" in point.context  # đoạn dẫn của khoản giữ ngữ cảnh cho điểm


def test_hierarchical_has_parent_links():
    chunks = chunk_document([(1, LAW)], "hierarchical", short_article_words=0, max_leaf_words=15)
    parents = {c.local_id: c for c in chunks if not c.is_leaf}
    leaves = [c for c in chunks if c.is_leaf and c.parent_local_id is not None]
    assert parents and leaves
    for leaf in leaves:
        assert parents[leaf.parent_local_id].article == leaf.article


def test_fixed_windows_record_covered_units():
    chunks = chunk_document([(1, LAW)], "fixed", window_words=40, overlap_words=10)
    assert len(chunks) > 3
    assert all(c.level == "window" for c in chunks)
    assert any("6.2" in c.covers for c in chunks)


def test_quoted_amendment_lines_are_not_articles():
    text = ("Điều 1. Sửa đổi\n1. Sửa đổi khoản 2 Điều 5 như sau:\n"
            "“Điều 5. Thuế suất\n2. Mức thuế suất 5%...”\nĐiều 2. Hiệu lực\nCó hiệu lực từ ngày ký.")
    articles, _, _ = parse_structure(text)
    assert [a.article for a in articles] == ["1", "2"]


def test_non_sequential_numbers_are_not_clauses():
    text = "Điều 3. Giải thích\n1. Một\n2024. Năm này không phải khoản\n2. Hai"
    art = parse_structure(text)[0][0]
    assert [c.label for c in art.children] == ["1", "2"]


def test_page_numbers_and_repeated_headers_removed():
    pages = [(i, f"CÔNG BÁO MẪU\nNội dung trang {i} có chữ.\n{i}") for i in range(1, 6)]
    cleaned = remove_headers_footers(pages)
    assert all("CÔNG BÁO" not in t and not t.strip().endswith(str(no)) for no, t in cleaned)


def test_page_mapping():
    half = len(LAW) // 2
    cut = LAW.rfind("\n", 0, half)
    chunks = chunk_document([(7, LAW[:cut]), (8, LAW[cut + 1:])], "structural",
                            short_article_words=0, max_leaf_words=15)
    assert chunks[0].page_start == 7
    assert chunks[-1].page_end == 8
