"""Pipeline hỏi đáp hoàn chỉnh (mục 31 – kiến trúc hệ thống).

Câu hỏi → kiểm tra chính sách → xác định thời điểm → lọc văn bản theo hiệu lực
→ truy hồi (BM25 / dense / hybrid) → rerank → kiểm tra mức căn cứ
→ sinh câu trả lời (JSON) → kiểm tra trích dẫn → kết quả + căn cứ + cảnh báo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np

from .config import NORMATIVE_TYPES, Settings
from .generation import (
    Answer,
    format_source,
    generate_answer,
    insufficient,
    policy_check,
)
from .retrieval import Candidate, Corpus
from .temporal import (
    QueryTime,
    extract_query_time,
    filter_documents,
    make_query_time,
    relation_notes,
)
from .text import normalize_doc_number

DenseSearchFn = Callable[[np.ndarray, int, set[str]], list[tuple[str, float]]]


@dataclass
class PipelineConfig:
    method: str = "hybrid"           # bm25 | dense | hybrid
    use_reranker: bool = True
    candidate_k: int = 20
    final_k: int = 5
    context_mode: str = "leaf"       # leaf | parent (chia đoạn phân cấp)
    tax_categories: list[str] | None = None
    as_of: date | None = None        # None = tự trích xuất từ câu hỏi
    include_reference: bool = True   # công văn, hỏi đáp
    include_unknown: bool = True     # văn bản chưa rõ ngày hiệu lực
    generate: bool = True            # False = chỉ truy hồi (dùng cho eval)
    private_mode: bool = False       # tài liệu riêng: bỏ cảnh báo hiệu lực
    temporal_filter: bool = True     # False = thí nghiệm đối chứng không lọc hiệu lực

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["as_of"] = self.as_of.isoformat() if self.as_of else None
        return d


class KnowledgeBase:
    def __init__(self, documents: list[dict[str, Any]], chunks: list[dict[str, Any]],
                 relations: list[dict[str, Any]] | None = None,
                 vectors: np.ndarray | None = None,
                 dense_search: DenseSearchFn | None = None):
        self.documents = {d["id"]: self._prepare_doc(d) for d in documents}
        for c in chunks:
            doc = self.documents.get(c.get("document_id"), {})
            c.setdefault("document_number", doc.get("document_number", ""))
        self.chunks = chunks
        self.relations = relations or []
        self.by_number = {normalize_doc_number(d.get("document_number")): d
                          for d in self.documents.values()}
        self.corpus = Corpus(chunks, self.documents, vectors=vectors)
        self.dense_search = dense_search

    @staticmethod
    def _prepare_doc(d: dict[str, Any]) -> dict[str, Any]:
        d = dict(d)
        d["document_title"] = d.get("document_title") or d.get("document_name") or ""
        d["reference_only"] = d.get("document_type") not in NORMATIVE_TYPES
        cats = d.get("tax_categories")
        if not cats:
            one = d.get("tax_category") or d.get("topic")
            d["tax_categories"] = [one] if one else []
        return d

    @property
    def has_dense(self) -> bool:
        return self.dense_search is not None or self.corpus.dense.available


@dataclass
class SourceView:
    n: int
    chunk: dict[str, Any]
    document: dict[str, Any]
    candidate: dict[str, Any]
    validity_status: str
    validity_notes: list[str]
    relation_notes: list[str]
    used_parent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PipelineResult:
    question: str
    query_time: dict[str, Any]
    answer: dict[str, Any]
    sources: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    allowed_documents: int = 0
    excluded_documents: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def retrieve(
    kb: KnowledgeBase,
    question: str,
    qt: QueryTime,
    cfg: PipelineConfig,
    settings: Settings,
    *,
    reranker=None,
    query_vector: np.ndarray | None = None,
    timings: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> tuple[list[Candidate], set[str], dict]:
    """Bước truy hồi dùng chung cho ứng dụng và bộ đánh giá."""
    timings = timings if timings is not None else {}
    warnings = warnings if warnings is not None else []
    allowed, decisions = filter_documents(
        kb.documents, qt, tax_categories=cfg.tax_categories,
        include_unknown=cfg.include_unknown, include_reference=cfg.include_reference,
    )
    if not cfg.temporal_filter:  # đối chứng: dùng mọi văn bản trong phạm vi loại thuế
        allowed = set(decisions)
    if not allowed:
        return [], allowed, decisions

    method = cfg.method
    if method in ("dense", "hybrid") and not kb.has_dense and query_vector is None:
        method = "bm25"  # kho chưa có embedding (hoặc chưa có API key)
    dense_hits = None
    if method in ("dense", "hybrid"):
        t0 = time.perf_counter()
        try:
            if query_vector is None:
                from .llm import embed_query

                query_vector = embed_query(settings, question)
            if kb.dense_search is not None:
                dense_hits = kb.dense_search(query_vector, cfg.candidate_k * 2, allowed)
        except Exception as exc:  # mất embedding -> lùi về BM25 thay vì lỗi
            warnings.append(f"Không tính được embedding ({type(exc).__name__}); dùng BM25.")
            method, query_vector = "bm25", None
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cands = kb.corpus.search(
        question, method=method, top_n=cfg.candidate_k, allowed_documents=allowed,
        query_vector=query_vector, dense_hits=dense_hits,
    )
    timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

    if cfg.use_reranker and reranker is not None and cands:
        t0 = time.perf_counter()
        cands = reranker.rerank(question, cands, kb.corpus.chunks)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    return cands, allowed, decisions


def answer_question(
    kb: KnowledgeBase,
    question: str,
    settings: Settings,
    cfg: PipelineConfig | None = None,
    *,
    reranker=None,
) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []

    qt = make_query_time(cfg.as_of) if cfg.as_of else extract_query_time(question)
    result = PipelineResult(question=question, query_time=qt.to_dict(), answer={},
                            config=cfg.to_dict(), timings=timings, warnings=warnings)

    declined = policy_check(question)
    if declined:
        result.answer = declined.to_dict()
        return result

    cands, allowed, decisions = retrieve(kb, question, qt, cfg, settings, reranker=reranker,
                                         timings=timings, warnings=warnings)
    result.allowed_documents = len(allowed)
    result.excluded_documents = [
        {"document_number": kb.documents[d].get("document_number"), "status": dec.status,
         "notes": dec.notes}
        for d, dec in decisions.items() if not dec.usable
    ]
    result.candidates = [
        {**c.to_dict(), "path": kb.corpus.chunks[c.index].get("path"),
         "document_number": kb.corpus.chunks[c.index].get("document_number")}
        for c in cands
    ]

    if not allowed:
        result.answer = insufficient(
            f"Không có văn bản nào trong kho thuộc phạm vi đã chọn và có hiệu lực tại {qt.label}."
        ).to_dict()
        return result
    if not cands:
        result.answer = insufficient("Không tìm thấy đoạn văn bản liên quan.").to_dict()
        return result

    top = cands[0]
    if top.rerank_score is not None and top.rerank_score < settings.min_rerank_score:
        result.answer = insufficient(
            "Các đoạn tìm được không trực tiếp trả lời câu hỏi."
        ).to_dict()
        result.sources = []
        return result
    if top.rerank_score is None and top.dense_score is not None and \
            cfg.method == "dense" and top.dense_score < settings.min_dense_score:
        result.answer = insufficient("Mức tương đồng ngữ nghĩa quá thấp.").to_dict()
        return result

    selected = cands[: cfg.final_k]
    if top.rerank_score is not None:
        floor = settings.min_rerank_score / 2
        selected = [c for c in selected if (c.rerank_score or 0) >= floor] or selected[:1]

    # Xây ngữ cảnh; chế độ phân cấp thay đoạn lá bằng cả Điều cha (khử trùng lặp)
    sources: list[SourceView] = []
    blocks: list[str] = []
    seen_parent: set[Any] = set()
    for c in selected:
        chunk = kb.corpus.chunks[c.index]
        doc = kb.documents.get(chunk.get("document_id"), {})
        content, used_parent = None, False
        if cfg.context_mode == "parent":
            parent = kb.corpus.parent_of(chunk)
            if parent is not None:
                if parent.get("id") in seen_parent:
                    continue
                seen_parent.add(parent.get("id"))
                content, used_parent = parent.get("content"), True
        dec = decisions.get(chunk.get("document_id"))
        rel = relation_notes(doc, kb.relations, kb.by_number, qt)
        n = len(sources) + 1
        sources.append(SourceView(
            n=n, chunk=chunk, document=doc, candidate=c.to_dict(),
            validity_status=dec.status if dec else "unknown",
            validity_notes=(dec.notes if dec else []) + rel,
            relation_notes=rel, used_parent=used_parent,
        ))
        blocks.append(format_source(n, chunk, doc, (dec.notes if dec else []) + rel, content))
    result.sources = [s.to_dict() for s in sources]

    if not cfg.generate:
        result.answer = Answer(status="answered").to_dict()
        return result

    t0 = time.perf_counter()
    try:
        ans = generate_answer(settings, question, qt, "\n\n".join(blocks), len(sources))
    except Exception as exc:
        ans = Answer(
            status="error",
            conclusion="Không gọi được mô hình ngôn ngữ. Các căn cứ tìm được vẫn hiển thị bên dưới.",
            notes=f"Chi tiết kỹ thuật: {type(exc).__name__}. Kiểm tra GEMINI_API_KEY / GEMINI_MODEL.",
        )
    timings["generate_ms"] = (time.perf_counter() - t0) * 1000

    for n in ([] if cfg.private_mode else ans.cited_sources):
        s = sources[n - 1]
        if s.document.get("reference_only"):
            ans.warnings.append(f"Nguồn [{n}] là {s.document.get('document_type', 'văn bản')} – "
                                "chỉ có giá trị tham khảo.")
        if s.validity_status == "unknown":
            ans.warnings.append(f"Nguồn [{n}] chưa xác định được ngày hiệu lực.")
    if cfg.private_mode:
        pass
    elif not qt.explicit:
        warnings.append(f"Câu hỏi không nêu thời điểm – hệ thống áp dụng quy định {qt.label}.")
    elif qt.is_range:
        warnings.append(f"Câu hỏi về khoảng thời gian {qt.label}: nếu quy định thay đổi giữa kỳ, "
                        "cần xác định ngày phát sinh nghiệp vụ.")

    result.answer = ans.to_dict()
    timings["total_ms"] = (time.perf_counter() - t_start) * 1000
    return result
