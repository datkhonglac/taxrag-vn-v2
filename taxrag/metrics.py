"""Chỉ số đánh giá truy hồi (mục 28) – độc lập với chiến lược chia đoạn.

Nhãn chuẩn của một câu hỏi là danh sách đơn vị pháp lý:
    {"document_number": "48/2024/QH15", "article": "5", "clause": "2"}
Một đoạn được coi là liên quan tới nhãn nếu cùng số hiệu văn bản và ``covers``
của đoạn chứa Điều (và Khoản, nếu nhãn có Khoản).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .text import normalize_doc_number


def unit_key(g: dict[str, Any]) -> str:
    art = str(g.get("article") or "").strip()
    cl = str(g.get("clause") or "").strip()
    key = normalize_doc_number(g.get("document_number"))
    return f"{key}#{art}.{cl}" if cl else f"{key}#{art}"


def chunk_matches(chunk: dict[str, Any], g: dict[str, Any], level: str = "clause") -> bool:
    if normalize_doc_number(chunk.get("document_number")) != normalize_doc_number(g.get("document_number")):
        return False
    if level == "document":
        return True
    art = str(g.get("article") or "").strip()
    if not art:
        return True
    covers = set(chunk.get("covers") or [])
    if level == "article" or not g.get("clause"):
        return art in covers
    return f"{art}.{str(g.get('clause')).strip()}" in covers


def retrieval_metrics(
    ranked: Sequence[dict[str, Any]],
    gold: Sequence[dict[str, Any]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Tính Recall@k, Precision@k, MRR, nDCG@k, tỷ lệ đúng văn bản / đúng điều-khoản."""
    out: dict[str, float] = {}
    if not gold:
        return out
    rel = [any(chunk_matches(c, g) for g in gold) for c in ranked]
    for k in ks:
        top = ranked[:k]
        covered = sum(1 for g in gold if any(chunk_matches(c, g) for c in top))
        out[f"recall@{k}"] = covered / len(gold)
        out[f"precision@{k}"] = sum(rel[:k]) / k
        out[f"hit@{k}"] = 1.0 if any(rel[:k]) else 0.0
        out[f"doc_hit@{k}"] = 1.0 if any(chunk_matches(c, g, "document") for c in top for g in gold) else 0.0
        out[f"article_hit@{k}"] = 1.0 if any(chunk_matches(c, g, "article") for c in top for g in gold) else 0.0
        dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 1 / (first + 1) if first is not None else 0.0
    return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    return {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys} if rows else {}
