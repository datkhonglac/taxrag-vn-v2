"""Phân đoạn văn bản pháp luật.

Ba chiến lược (định hướng GVHD, mục 12):

* ``fixed``        – cửa sổ trượt theo số từ, có chồng lấn (dùng làm đối chứng).
* ``structural``   – theo Điều → Khoản → Điểm (phương pháp chính).
* ``hierarchical`` – giống ``structural`` nhưng mỗi đoạn lá giữ liên kết tới
                     đoạn cha cấp Điều; khi sinh câu trả lời, hệ thống đưa cả
                     Điều vào ngữ cảnh.

Mọi đoạn đều ghi lại ``covers`` – danh sách đơn vị pháp lý mà đoạn bao phủ
(vd ``["8", "8.1"]``). Nhờ đó bộ đánh giá có thể chấm cả các đoạn chia cố định
theo cùng một nhãn chuẩn "Điều 8, Khoản 1".
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .text import normalize_text

STRATEGIES = ("fixed", "structural", "hierarchical")

# Ngưỡng tính theo số từ (âm tiết). ~1 âm tiết ≈ 1–1,3 token.
SHORT_ARTICLE_WORDS = 250   # Điều ngắn hơn -> giữ nguyên một đoạn
MAX_LEAF_WORDS = 350        # Khoản dài hơn -> chia tiếp theo Điểm
FIXED_WINDOW_WORDS = 400
FIXED_OVERLAP_WORDS = 80
PARENT_MAX_WORDS = 1200     # giới hạn độ dài Điều khi đưa vào ngữ cảnh

VI_POINT_LETTERS = "a b c d đ e g h i k l m n o p q r s t u v x y".split()

_ARTICLE_RE = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*[.:]\s*(.*)$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\b\.?\s*(.*)$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^Mục\s+(\d+|[IVXLCDM]+)\b\.?\s*(.*)$", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^Phụ\s+lục\b\s*(.*)$", re.IGNORECASE)
_RECIPIENTS_RE = re.compile(r"^Nơi\s+nhận\s*:", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"^(\d{1,3})([a-zđ]?)\.\s+\S")
_POINT_RE = re.compile(r"^([a-zđ])\)\s+\S")
_PAGE_NO_RE = re.compile(r"^\s*(trang\s*)?\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.IGNORECASE)


@dataclass
class Chunk:
    local_id: int
    strategy: str
    level: str  # preamble | article | clause | point | window | appendix
    content: str
    heading: str = ""
    chapter: str = ""
    section: str = ""
    article: str = ""
    article_title: str = ""
    clause: str = ""
    point: str = ""
    context: str = ""  # đoạn dẫn của Điều/Khoản cha, giúp đoạn lá không mất ngữ cảnh
    path: str = ""
    covers: list[str] = field(default_factory=list)
    parent_local_id: int | None = None
    is_leaf: bool = True
    page_start: int | None = None
    page_end: int | None = None
    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class _Unit:
    kind: str
    start: int
    end: int
    label: str = ""
    title: str = ""
    chapter: str = ""
    section: str = ""
    article: str = ""
    article_title: str = ""
    clause: str = ""
    children: list["_Unit"] = field(default_factory=list)
    body_start: int = 0  # vị trí bắt đầu phần thân (sau dòng tiêu đề)


# ------------------------------------------------------------------
# Làm sạch theo trang
# ------------------------------------------------------------------

def _line_signature(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_headers_footers(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Bỏ số trang và dòng đầu/chân trang lặp lại trên nhiều trang."""
    cleaned = [(no, normalize_text(text)) for no, text in pages]
    if len(cleaned) >= 3:
        counter: Counter[str] = Counter()
        for _, text in cleaned:
            lines = [ln for ln in text.split("\n") if ln.strip()]
            edge = lines[:2] + lines[-2:]
            counter.update({_line_signature(ln) for ln in edge})
        threshold = max(3, int(0.5 * len(cleaned)))
        repeated = {sig for sig, c in counter.items() if c >= threshold and sig}
    else:
        repeated = set()

    result = []
    for no, text in cleaned:
        lines = text.split("\n")
        keep = []
        for idx, ln in enumerate(lines):
            if _PAGE_NO_RE.match(ln):
                continue
            near_edge = idx < 3 or idx >= len(lines) - 3
            if near_edge and _line_signature(ln) in repeated:
                continue
            keep.append(ln)
        result.append((no, "\n".join(keep).strip()))
    return result


def join_pages(pages: list[tuple[int, str]]) -> tuple[str, list[int], list[int]]:
    """Ghép các trang thành một chuỗi; trả về (text, offsets, page_numbers)."""
    parts, offsets, numbers = [], [], []
    pos = 0
    for no, text in pages:
        if not text:
            continue
        offsets.append(pos)
        numbers.append(no)
        parts.append(text)
        pos += len(text) + 1  # +1 cho ký tự "\n" nối trang
    return "\n".join(parts), offsets, numbers


# ------------------------------------------------------------------
# Phân tích cấu trúc
# ------------------------------------------------------------------

def _iter_lines(text: str):
    pos = 0
    for line in text.split("\n"):
        yield pos, line
        pos += len(line) + 1


def _point_index(letter: str) -> int:
    try:
        return VI_POINT_LETTERS.index(letter)
    except ValueError:
        return -1


def parse_structure(text: str) -> tuple[list[_Unit], list[_Unit], _Unit | None]:
    """Trả về (articles, appendices, preamble)."""
    markers: list[tuple[int, str, re.Match, int]] = []
    lines = list(_iter_lines(text))
    for i, (pos, raw) in enumerate(lines):
        line = raw.strip()
        if not line or line[0] in "\"“”'‘":
            continue  # dòng bắt đầu bằng ngoặc kép = trích dẫn văn bản được sửa đổi
        for kind, rx in (
            ("article", _ARTICLE_RE),
            ("chapter", _CHAPTER_RE),
            ("section", _SECTION_RE),
            ("appendix", _APPENDIX_RE),
            ("recipients", _RECIPIENTS_RE),
        ):
            m = rx.match(line)
            if m:
                markers.append((pos, kind, m, i))
                break

    # Một khi đã vào Phụ lục thì không nhận "Điều" nữa
    filtered = []
    in_appendix = False
    for item in markers:
        if item[1] == "appendix":
            in_appendix = True
        elif in_appendix and item[1] in ("article", "chapter", "section"):
            continue
        filtered.append(item)
    markers = filtered

    articles: list[_Unit] = []
    appendices: list[_Unit] = []
    chapter = section = ""
    first_marker = markers[0][0] if markers else len(text)

    for idx, (pos, kind, m, line_idx) in enumerate(markers):
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
        if kind == "chapter":
            title = m.group(2).strip()
            if not title and line_idx + 1 < len(lines):
                nxt = lines[line_idx + 1][1].strip()
                if nxt and nxt.upper() == nxt and not _ARTICLE_RE.match(nxt):
                    title = nxt
            if title.isupper():
                title = title[:1] + title[1:].lower()
            chapter = f"Chương {m.group(1).upper()}" + (f". {title}" if title else "")
            section = ""
        elif kind == "section":
            title = m.group(2).strip()
            section = f"Mục {m.group(1)}" + (f". {title}" if title else "")
        elif kind == "article":
            line_end = text.find("\n", pos)
            body_start = len(text) if line_end == -1 else line_end + 1
            art = _Unit(
                kind="article",
                start=pos,
                end=end,
                label=m.group(1),
                title=m.group(2).strip(),
                chapter=chapter,
                section=section,
                article=m.group(1),
                article_title=m.group(2).strip(),
                body_start=min(body_start, end),
            )
            _parse_article_children(text, art)
            articles.append(art)
        elif kind == "appendix":
            appendices.append(
                _Unit(kind="appendix", start=pos, end=end, title=m.group(1).strip())
            )
        # "recipients": chỉ dùng làm điểm cắt cho Điều cuối cùng

    preamble = None
    if first_marker > 0 and text[:first_marker].strip():
        preamble = _Unit(kind="preamble", start=0, end=first_marker)
    return articles, appendices, preamble


def _parse_article_children(text: str, art: _Unit) -> None:
    body = text[art.body_start:art.end]
    clause_marks: list[tuple[int, str]] = []
    point_marks: list[tuple[int, str]] = []
    expected_clause = 1
    for rel, line in _iter_lines(body):
        s = line.lstrip()
        lead = len(line) - len(s)
        cm = _CLAUSE_RE.match(s)
        if cm and int(cm.group(1)) in (expected_clause, expected_clause - 1) and (
            int(cm.group(1)) == expected_clause or cm.group(2)
        ):
            clause_marks.append((art.body_start + rel + lead, cm.group(1) + cm.group(2)))
            if not cm.group(2):
                expected_clause += 1
            continue
        pm = _POINT_RE.match(s)
        if pm:
            point_marks.append((art.body_start + rel + lead, pm.group(1)))

    def build_points(start: int, end: int, parent_clause: str) -> list[_Unit]:
        marks = [(p, l) for p, l in point_marks if start <= p < end]
        seq, expected = [], 0
        for p, letter in marks:
            idx = _point_index(letter)
            if idx == expected or (idx == 0 and not seq):
                seq.append((p, letter))
                expected = idx + 1
        units = []
        for i, (p, letter) in enumerate(seq):
            e = seq[i + 1][0] if i + 1 < len(seq) else end
            units.append(
                _Unit(kind="point", start=p, end=e, label=letter,
                      article=art.article, clause=parent_clause)
            )
        return units

    if clause_marks:
        for i, (p, label) in enumerate(clause_marks):
            e = clause_marks[i + 1][0] if i + 1 < len(clause_marks) else art.end
            clause = _Unit(kind="clause", start=p, end=e, label=label,
                           article=art.article, clause=label)
            clause.children = build_points(p, e, label)
            art.children.append(clause)
    else:
        art.children = build_points(art.body_start, art.end, "")


# ------------------------------------------------------------------
# Sinh đoạn
# ------------------------------------------------------------------

class _PageMapper:
    def __init__(self, offsets: list[int], numbers: list[int]):
        self.offsets = offsets
        self.numbers = numbers

    def page(self, pos: int) -> int | None:
        if not self.offsets:
            return None
        i = bisect.bisect_right(self.offsets, pos) - 1
        return self.numbers[max(i, 0)]


def _words(text: str) -> int:
    return len(text.split())


def _window_spans(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    tokens = [(m.start() + start, m.end() + start) for m in re.finditer(r"\S+", text[start:end])]
    if not tokens:
        return []
    spans, step = [], max(1, size - overlap)
    for i in range(0, len(tokens), step):
        window = tokens[i:i + size]
        spans.append((window[0][0], window[-1][1]))
        if i + size >= len(tokens):
            break
    return spans


def _path(*parts: str) -> str:
    return " › ".join(p for p in parts if p)


def _covers_for_article(art: _Unit) -> list[str]:
    return [art.article] + [f"{art.article}.{c.label}" for c in art.children if c.kind == "clause"]


def chunk_document(
    pages: list[tuple[int, str]],
    strategy: str = "structural",
    *,
    clean: bool = True,
    short_article_words: int = SHORT_ARTICLE_WORDS,
    max_leaf_words: int = MAX_LEAF_WORDS,
    window_words: int = FIXED_WINDOW_WORDS,
    overlap_words: int = FIXED_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chia một văn bản (danh sách (số trang, nội dung)) thành các đoạn."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy phải là một trong {STRATEGIES}")
    if clean:
        pages = remove_headers_footers(pages)
    else:
        pages = [(no, normalize_text(t)) for no, t in pages]
    text, offsets, numbers = join_pages(pages)
    mapper = _PageMapper(offsets, numbers)
    articles, appendices, preamble = parse_structure(text)

    chunks: list[Chunk] = []

    def add(**kw) -> Chunk:
        start, end = kw["char_start"], kw["char_end"]
        c = Chunk(
            local_id=len(chunks),
            strategy=strategy,
            page_start=mapper.page(start),
            page_end=mapper.page(max(start, end - 1)),
            **kw,
        )
        c.content = c.content.strip()
        chunks.append(c)
        return c

    if strategy == "fixed" or not articles:
        # Văn bản không nhận diện được Điều (công văn, hỏi đáp...) cũng đi nhánh này.
        clause_units = [
            (u.start, u.end, f"{a.article}.{u.label}")
            for a in articles for u in a.children if u.kind == "clause"
        ]
        art_units = [(a.start, a.end, a.article) for a in articles]
        for s, e in _window_spans(text, 0, len(text), window_words, overlap_words):
            covers = [lab for us, ue, lab in art_units if us < e and ue > s]
            covers += [lab for us, ue, lab in clause_units if us < e and ue > s]
            first_art = next((a for a in articles if a.start < e and a.end > s), None)
            add(
                level="window", content=text[s:e], char_start=s, char_end=e,
                covers=covers,
                article=first_art.article if first_art else "",
                article_title=first_art.article_title if first_art else "",
                chapter=first_art.chapter if first_art else "",
                heading=f"Điều {first_art.article}. {first_art.article_title}" if first_art else "",
                path=_path(first_art.chapter if first_art else "",
                           f"Điều {first_art.article}" if first_art else "Đoạn văn bản"),
            )
        return chunks

    if preamble is not None:
        s, e = preamble.start, preamble.end
        words = text[s:e].split()
        e = s + len(" ".join(words[:300])) if len(words) > 300 else e
        add(level="preamble", content=text[s:e], char_start=s, char_end=e,
            heading="Phần mở đầu / căn cứ ban hành", path="Phần mở đầu")

    for art in articles:
        art_heading = f"Điều {art.article}. {art.article_title}".strip()
        art_text = text[art.start:art.end]
        art_path = _path(art.chapter, art.section, f"Điều {art.article}")
        base = dict(chapter=art.chapter, section=art.section, article=art.article,
                    article_title=art.article_title)

        short = _words(art_text) <= short_article_words or not art.children
        if short and _words(art_text) <= max_leaf_words * 2:
            add(level="article", content=art_text, heading=art_heading, path=art_path,
                covers=_covers_for_article(art), char_start=art.start, char_end=art.end, **base)
            continue

        parent_id = None
        if strategy == "hierarchical":
            words = art_text.split()
            pe = art.start + len(" ".join(words[:PARENT_MAX_WORDS])) if len(words) > PARENT_MAX_WORDS else art.end
            parent = add(level="article", content=text[art.start:pe], heading=art_heading,
                         path=art_path, covers=_covers_for_article(art), is_leaf=False,
                         char_start=art.start, char_end=art.end, **base)
            parent_id = parent.local_id

        if not art.children:
            # Điều dài không có khoản/điểm -> cửa sổ trong phạm vi Điều
            for i, (s, e) in enumerate(_window_spans(text, art.start, art.end, max_leaf_words, 50)):
                add(level="article", content=text[s:e], heading=f"{art_heading} (phần {i + 1})",
                    path=art_path, covers=_covers_for_article(art), parent_local_id=parent_id,
                    char_start=s, char_end=e, **base)
            continue

        # Đoạn dẫn của Điều (trước khoản/điểm đầu tiên) làm ngữ cảnh
        lead_in = text[art.body_start:art.children[0].start].strip()
        for child in art.children:
            if child.kind == "clause":
                cl_text = text[child.start:child.end]
                cl_path = _path(art_path, f"Khoản {child.label}")
                cl_cover = [art.article, f"{art.article}.{child.label}"]
                if _words(cl_text) > max_leaf_words and child.children:
                    cl_lead = text[child.start:child.children[0].start].strip()
                    for pt in child.children:
                        add(level="point", content=text[pt.start:pt.end],
                            heading=f"{art_heading} — Khoản {child.label}, Điểm {pt.label}",
                            path=_path(cl_path, f"Điểm {pt.label}"), clause=child.label,
                            point=pt.label, covers=cl_cover, parent_local_id=parent_id,
                            char_start=pt.start, char_end=pt.end,
                            context=_join_context(lead_in, cl_lead), **base)
                else:
                    add(level="clause", content=cl_text,
                        heading=f"{art_heading} — Khoản {child.label}",
                        path=cl_path, clause=child.label, covers=cl_cover,
                        parent_local_id=parent_id, char_start=child.start, char_end=child.end,
                        context=_shorten(lead_in), **base)
            else:  # điểm trực tiếp dưới Điều
                add(level="point", content=text[child.start:child.end],
                    heading=f"{art_heading} — Điểm {child.label}",
                    path=_path(art_path, f"Điểm {child.label}"), point=child.label,
                    covers=[art.article], parent_local_id=parent_id,
                    char_start=child.start, char_end=child.end,
                    context=_shorten(lead_in), **base)

    for ap in appendices:
        title = f"Phụ lục {ap.title}".strip()
        for i, (s, e) in enumerate(_window_spans(text, ap.start, ap.end, window_words, overlap_words)):
            add(level="appendix", content=text[s:e], heading=f"{title} (phần {i + 1})",
                path=title, covers=["PL"], char_start=s, char_end=e)

    return chunks


def _shorten(text: str, max_words: int = 60) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def _join_context(*parts: str) -> str:
    """Ghép các đoạn dẫn (đã rút gọn) của Điều và Khoản cha."""
    return " | ".join(_shorten(p) for p in parts if p.strip())


def index_text(chunk: dict[str, Any], doc: dict[str, Any] | None = None) -> str:
    """Văn bản dùng để đánh chỉ mục: metadata + tiêu đề + nội dung."""
    doc = doc or {}
    parts = [
        doc.get("document_type", ""),
        doc.get("document_number", ""),
        doc.get("document_title") or doc.get("document_name", ""),
        chunk.get("chapter", ""),
        chunk.get("heading", ""),
        chunk.get("context", ""),
        chunk.get("content", ""),
    ]
    return "\n".join(p for p in parts if p)
