import numpy as np

from taxrag.retrieval import BM25Index, Corpus, rrf_fuse, top_indices
from taxrag.text import tokenize_vi


def test_tokenizer_keeps_document_numbers_and_bigrams():
    toks = tokenize_vi("Theo Luật 48/2024/QH15 về hóa đơn điện tử")
    assert "48/2024/qh15" in toks
    assert "hóa_đơn" in toks and "điện_tử" in toks
    assert "về" not in toks


def test_bm25_ranks_relevant_first():
    texts = ["thuế suất 0% áp dụng hàng hóa xuất khẩu", "thời hạn nộp tờ khai thuế theo quý",
             "hóa đơn điện tử có mã của cơ quan thuế"]
    idx = BM25Index(texts)
    s = idx.scores("hạn nộp tờ khai quý")
    assert int(np.argmax(s)) == 1
    assert idx.scores("từ không có trong kho").sum() == 0


def test_rrf_and_mask():
    fused = rrf_fuse([[1, 2, 3], [3, 1]])
    assert max(fused, key=fused.get) in (1, 3)
    s = np.array([0.9, 0.5, 0.7])
    assert top_indices(s, 2, np.array([False, True, True])) == [2, 1]


def _corpus(vectors=None):
    chunks = [
        {"id": "a", "document_id": "d1", "content": "thuế suất 0% hàng xuất khẩu", "is_leaf": True},
        {"id": "p", "document_id": "d1", "content": "Điều 5 toàn văn", "is_leaf": False},
        {"id": "b", "document_id": "d2", "content": "hạn nộp tờ khai thuế quý", "is_leaf": True},
    ]
    return Corpus(chunks, {"d1": {}, "d2": {}}, vectors=vectors)


def test_parent_chunks_not_searchable_and_doc_filter():
    c = _corpus()
    res = c.search("Điều 5 toàn văn thuế", method="bm25", top_n=5)
    assert all(c.chunks[r.index]["id"] != "p" for r in res)
    res = c.search("thuế", method="bm25", top_n=5, allowed_documents={"d2"})
    assert [c.chunks[r.index]["id"] for r in res] == ["b"]


def test_hybrid_combines_ranks():
    vec = np.array([[1, 0], [0, 0], [0, 1]], dtype=np.float32)
    c = _corpus(vec)
    res = c.search("hạn nộp tờ khai", method="hybrid", top_n=2, query_vector=np.array([1.0, 0.0]))
    ids = [c.chunks[r.index]["id"] for r in res]
    assert set(ids) == {"a", "b"}
    assert res[0].bm25_rank is not None or res[0].dense_rank is not None
    # dense_hits từ pgvector
    res = c.search("x", method="dense", top_n=2, dense_hits=[("b", 0.8), ("a", 0.4)])
    assert [c.chunks[r.index]["id"] for r in res] == ["b", "a"]
