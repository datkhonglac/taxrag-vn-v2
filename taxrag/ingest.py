"""Đọc văn bản từ PDF / HTML / TXT, tự động OCR trang scan (mục 9–10)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from hashlib import sha256
from html.parser import HTMLParser

from .text import normalize_text

OCR_TEXT_THRESHOLD = 120
OCR_RENDER_SCALE = 2.0
MAX_OCR_PAGES_PER_FILE = 80
SUPPORTED_TYPES = ["pdf", "html", "htm", "txt"]


@dataclass
class ExtractedDocument:
    pages: list[tuple[int, str]]
    page_count: int
    ocr_pages: int = 0
    empty_pages: int = 0
    content_hash: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.pages)


def text_quality_ok(text: str) -> bool:
    if len(text) < OCR_TEXT_THRESHOLD:
        return False
    useful = sum(ch.isalnum() or ch.isspace() for ch in text)
    return useful / max(len(text), 1) >= 0.7


def _ocr_page(page) -> str:
    import fitz
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    image = ImageEnhance.Contrast(image).enhance(1.6).filter(ImageFilter.SHARPEN)
    return normalize_text(pytesseract.image_to_string(image, lang="vie+eng", config="--oem 3 --psm 6"))


def ocr_available() -> bool:
    try:
        import pytesseract

        return "vie" in pytesseract.get_languages(config="")
    except Exception:
        return False


def extract_pdf(data: bytes) -> ExtractedDocument:
    import fitz

    pages: list[tuple[int, str]] = []
    ocr_pages = empty = 0
    warnings: list[str] = []
    can_ocr = None
    with fitz.open(stream=data, filetype="pdf") as doc:
        count = len(doc)
        for no, page in enumerate(doc, start=1):
            text = normalize_text(page.get_text("text", sort=True))
            if not text_quality_ok(text):
                if can_ocr is None:
                    can_ocr = ocr_available()
                    if not can_ocr:
                        warnings.append("Máy chủ chưa cài Tesseract (gói tiếng Việt) – "
                                        "không OCR được trang scan. Kiểm tra packages.txt.")
                if can_ocr and ocr_pages < MAX_OCR_PAGES_PER_FILE:
                    try:
                        text = _ocr_page(page)
                        ocr_pages += 1
                    except Exception as exc:
                        warnings.append(f"OCR lỗi ở trang {no}: {type(exc).__name__}")
            if not text_quality_ok(text):
                empty += 1
                continue
            pages.append((no, text))
    if ocr_pages:
        warnings.append(f"Đã OCR {ocr_pages} trang – cần đối chiếu số hiệu, số tiền với bản gốc.")
    return ExtractedDocument(pages, count, ocr_pages, empty, sha256(data).hexdigest(), warnings)


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form", "aside"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "cp1258", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_html(data: bytes) -> ExtractedDocument:
    parser = _HTMLText()
    parser.feed(_decode(data))
    text = normalize_text("".join(parser.parts))
    text = re.sub(r"\n[ |]+\n", "\n", text)
    return ExtractedDocument([(1, text)] if text else [], 1, content_hash=sha256(data).hexdigest())


def extract_txt(data: bytes) -> ExtractedDocument:
    text = normalize_text(_decode(data))
    return ExtractedDocument([(1, text)] if text else [], 1, content_hash=sha256(data).hexdigest())


def extract_file(filename: str, data: bytes) -> ExtractedDocument:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_pdf(data)
    if ext in ("html", "htm"):
        return extract_html(data)
    if ext == "txt":
        return extract_txt(data)
    raise ValueError(f"Chưa hỗ trợ định dạng .{ext}")
