from taxrag.metrics import chunk_matches, retrieval_metrics

GOLD = [{"document_number": "48/2024/QH15", "article": "5", "clause": "2"}]


def c(num, covers):
    return {"document_number": num, "covers": covers}


def test_matching_levels():
    assert chunk_matches(c("48/2024/qh15 ", ["5", "5.2"]), GOLD[0])
    assert not chunk_matches(c("48/2024/QH15", ["5", "5.1"]), GOLD[0])
    assert chunk_matches(c("48/2024/QH15", ["5", "5.1"]), GOLD[0], "article")
    assert chunk_matches(c("48/2024/QH15", []), GOLD[0], "document")


def test_metrics_values():
    ranked = [c("X", ["1"]), c("48/2024/QH15", ["5", "5.1"]), c("48/2024/QH15", ["5", "5.2"])]
    m = retrieval_metrics(ranked, GOLD, ks=(1, 3))
    assert m["recall@1"] == 0 and m["recall@3"] == 1
    assert abs(m["mrr"] - 1 / 3) < 1e-9
    assert m["article_hit@3"] == 1 and m["doc_hit@1"] == 0
    assert 0 < m["ndcg@3"] < 1
