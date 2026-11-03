"""Mô-đun truy hồi: BM25 (sparse), embedding (dense) và kết hợp lai (hybrid).

Thiết kế để dùng chung cho ứng dụng Streamlit và script đánh giá (eval/):
mỗi phương pháp trả về danh sách (chỉ số đoạn, điểm) để so sánh công bằng.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .chunking import index_text
from .text import tokenize_vi

RRF_K = 60  # hằng số Reciprocal Rank Fusion (Cormack và cộng sự, 2009)


# ------------------------------------------------------------------
# BM25
# ------------------------------------------------------------------

class BM25Index:
    """Okapi BM25 cài đặt trên ma trận thưa (đủ nhanh cho vài chục nghìn đoạn)."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75,
                 tokenizer: Callable[[str], list[str]] = tokenize_vi):
        self.k1, self.b, self.tokenizer = k1, b, tokenizer
        docs = [Counter(tokenizer(t)) for t in texts]
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, counts in enumerate(docs):
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
        n_docs, n_terms = len(docs), len(self.vocab)
        tf = csr_matrix((vals, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32)
        df = np.bincount(cols, minlength=n_terms).astype(np.float32) if cols else np.zeros(0)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(lengths.mean()) if n_docs else 0.0
        # Tiền tính trọng số BM25 cho từng (đoạn, từ)
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * lengths[tf.row] / max(avgdl, 1e-9))
        weights = tf.data * (k1 + 1) / (tf.data + norm)
        self.matrix = csr_matrix((weights, (tf.row, tf.col)), shape=(n_docs, n_terms)).tocsc()
        self.n_docs = n_docs

    def scores(self, query: str) -> np.ndarray:
        if self.n_docs == 0:
            return np.zeros(0, dtype=np.float32)
        ids = [self.vocab[t] for t in set(self.tokenizer(query)) if t in self.vocab]
        if not ids:
            return np.zeros(self.n_docs, dtype=np.float32)
        sub = self.matrix[:, ids]
        return np.asarray(sub @ self.idf[ids]).ravel()


# ------------------------------------------------------------------
# Dense
# ------------------------------------------------------------------

class DenseIndex:
    """Chỉ mục vector trong bộ nhớ (cosine). Dùng cho eval và tài liệu riêng.

    Với kho dùng chung, ứng dụng gọi hàm ``match_chunks`` của pgvector trên
    Supabase (xem ``taxrag/db.py``) thay vì nạp toàn bộ vector về.
    """

    def __init__(self, vectors: np.ndarray | None):
        if vectors is None or len(vectors) == 0:
            self.vectors = None
        else:
            v = np.asarray(vectors, dtype=np.float32)
            self.vectors = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)

    @property
    def available(self) -> bool:
        return self.vectors is not None

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        return self.vectors @ q


# ------------------------------------------------------------------
# Kết hợp
# ------------------------------------------------------------------

def top_indices(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[int]:
    if scores.size == 0:
        return []
    s = scores.astype(np.float64).copy()
    if allowed is not None:
        s[~allowed] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-s[idx])]]


def rrf_fuse(rankings: Iterable[list[int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass
class Candidate:
    index: int
    score: float
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None
    rerank_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Corpus:
    """Tập đoạn văn bản + metadata văn bản + các chỉ mục.

    ``chunks``: danh sách dict (từ DB hoặc từ chunk_document)
    ``documents``: dict document_id -> metadata văn bản
    """

    chunks: list[dict[str, Any]]
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    vectors: np.ndarray | None = None
    leaf_only: bool = True

    def __post_init__(self):
        self.searchable = np.array(
            [bool(c.get("is_leaf", True)) or not self.leaf_only for c in self.chunks], dtype=bool
        )
        texts = [index_text(c, self.documents.get(c.get("document_id"), {})) for c in self.chunks]
        self.bm25 = BM25Index(texts)
        self.dense = DenseIndex(self.vectors)
        self.by_id = {c.get("id"): i for i, c in enumerate(self.chunks) if c.get("id") is not None}

    def __len__(self) -> int:
        return len(self.chunks)

    def mask_for_documents(self, document_ids: set[str] | None) -> np.ndarray:
        mask = self.searchable.copy()
        if document_ids is not None:
            mask &= np.array([c.get("document_id") in document_ids for c in self.chunks], dtype=bool)
        return mask

    def search(
        self,
        query: str,
        *,
        method: str = "hybrid",
        top_n: int = 20,
        allowed_documents: set[str] | None = None,
        query_vector: np.ndarray | None = None,
        dense_hits: list[tuple[str, float]] | None = None,
    ) -> list[Candidate]:
        """Trả về top_n ứng viên.

        - ``query_vector``: dùng chỉ mục dense trong bộ nhớ.
        - ``dense_hits``: kết quả dense đã tính sẵn ở nơi khác (pgvector),
          dạng [(chunk_id, similarity)].
        """
        mask = self.mask_for_documents(allowed_documents)
        bm25_rank: list[int] = []
        dense_rank: list[int] = []
        bm25_scores = dense_scores = None

        if method in ("bm25", "hybrid"):
            bm25_scores = self.bm25.scores(query)
            bm25_rank = [i for i in top_indices(bm25_scores, top_n, mask) if bm25_scores[i] > 0]

        if method in ("dense", "hybrid"):
            if dense_hits is not None:
                dense_scores = np.full(len(self.chunks), np.nan, dtype=np.float32)
                for cid, sim in dense_hits:
                    i = self.by_id.get(cid)
                    if i is not None and mask[i]:
                        dense_scores[i] = sim
                        dense_rank.append(i)
                dense_rank = dense_rank[:top_n]
            elif query_vector is not None and self.dense.available:
                dense_scores = self.dense.scores(query_vector)
                dense_rank = top_indices(dense_scores, top_n, mask)

        if method == "bm25" or (method == "hybrid" and not dense_rank):
            order = [(i, float(bm25_scores[i])) for i in bm25_rank]
        elif method == "dense" or (method == "hybrid" and not bm25_rank):
            order = [(i, float(dense_scores[i])) for i in dense_rank]
        else:
            fused = rrf_fuse([bm25_rank, dense_rank])
            order = sorted(fused.items(), key=lambda kv: -kv[1])[:top_n]

        bm25_pos = {i: r for r, i in enumerate(bm25_rank, start=1)}
        dense_pos = {i: r for r, i in enumerate(dense_rank, start=1)}
        results = []
        for i, score in order[:top_n]:
            results.append(
                Candidate(
                    index=i,
                    score=score,
                    bm25_score=float(bm25_scores[i]) if bm25_scores is not None else None,
                    bm25_rank=bm25_pos.get(i),
                    dense_score=(
                        float(dense_scores[i])
                        if dense_scores is not None and not math.isnan(float(dense_scores[i]))
                        else None
                    ),
                    dense_rank=dense_pos.get(i),
                )
            )
        return results

    def parent_of(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        pid = chunk.get("parent_id")
        if pid is None:
            return None
        i = self.by_id.get(pid)
        return self.chunks[i] if i is not None else None
