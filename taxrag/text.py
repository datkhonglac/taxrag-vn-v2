"""Chuẩn hóa và tách từ tiếng Việt cho truy hồi từ khóa (BM25)."""

from __future__ import annotations

import re
import unicodedata

# Stopword chỉ áp dụng cho unigram. Giữ lại "không", "được", "phải", "chịu"...
# vì các từ này đổi nghĩa pháp lý ("không chịu thuế" ≠ "chịu thuế").
VI_STOPWORDS = {
    "là", "của", "và", "các", "những", "cho", "theo", "thì", "với", "này",
    "đó", "khi", "tại", "trong", "về", "như", "để", "có", "một", "ra", "vào",
    "nào", "gì", "ạ", "nhé", "vậy", "hay", "hoặc", "bị", "do", "từ", "đến",
    "trên", "dưới", "cũng", "đã", "sẽ", "đang", "thế", "sao", "bao", "nhiêu",
    "tôi", "mình", "em", "anh", "chị", "bạn", "cần", "muốn", "hỏi", "xin",
}

# Token: chữ/số, cho phép nối bằng / - . để giữ nguyên số hiệu văn bản
# (vd: 48/2024/qh15, 70/2025/nđ-cp) và số thập phân.
_TOKEN_RE = re.compile(r"\w+(?:[/\-.]\w+)*", re.UNICODE)


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC, khoảng trắng, dấu gạch nối cuối dòng."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", " ").replace(" ", " ").replace("​", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_vi(text: str, bigrams: bool = True) -> list[str]:
    """Tách âm tiết + ghép bigram âm tiết.

    Tiếng Việt viết cách theo âm tiết, nhiều từ có 2 âm tiết ("hóa đơn",
    "doanh nghiệp"). Ghép bigram là cách xấp xỉ tách từ mà không cần thư viện
    ngoài; có thể thay bằng pyvi/underthesea trong thực nghiệm.
    """
    text = unicodedata.normalize("NFC", text.lower())
    syllables = _TOKEN_RE.findall(text)
    tokens = [s for s in syllables if s not in VI_STOPWORDS]
    if bigrams:
        tokens.extend(
            f"{a}_{b}" for a, b in zip(syllables, syllables[1:])
        )
    return tokens


def normalize_doc_number(value: str | None) -> str:
    """'48/2024/QH15 ' -> '48/2024/qh15' để so khớp số hiệu văn bản."""
    if not value:
        return ""
    value = unicodedata.normalize("NFC", value).strip().lower()
    value = re.sub(r"\s+", "", value)
    return value.replace("–", "-").replace("—", "-")


def word_count(text: str) -> int:
    return len(text.split())
