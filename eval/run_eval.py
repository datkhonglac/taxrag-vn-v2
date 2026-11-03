"""Chạy thực nghiệm đánh giá TaxRAG VN (định hướng GVHD, mục 25–30).

Ví dụ (chạy từ thư mục gốc repo):

    # Chỉ BM25, không cần API key – kiểm tra nhanh pipeline trên dữ liệu mẫu
    python -m eval.run_eval --corpus eval/sample/corpus --questions eval/sample/questions.jsonl \
        --methods bm25 --out eval/results/sample

    # Đầy đủ: 2 chiến lược chia đoạn × 3 phương pháp truy hồi × có/không reranker
    #         + đối chứng không lọc hiệu lực + sinh câu trả lời + baseline không RAG
    python -m eval.run_eval --corpus data/corpus --questions data/questions.jsonl \
        --strategies fixed,structural --methods bm25,dense,hybrid --rerank none,gemini \
        --ablate-temporal --generate --no-rag --out eval/results/run1

Đầu ra trong thư mục --out:
    retrieval_summary.csv / .md   bảng so sánh các cấu hình (Recall@k, MRR, nDCG…)
    retrieval_details.jsonl       kết quả từng câu hỏi × cấu hình
    breakdown.md                  kết quả theo mức độ khó và loại câu hỏi
    answers.jsonl                 câu trả lời sinh ra (nếu --generate)
    answers_for_grading.csv       bảng chấm điểm cho chuyên gia (mục 29–30)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from taxrag.admin_ops import parse_catalog  # noqa: E402
from taxrag.chunking import chunk_document, index_text  # noqa: E402
from taxrag.config import load_settings  # noqa: E402
from taxrag.generation import REFUSAL_STATUSES  # noqa: E402
from taxrag.ingest import extract_file  # noqa: E402
from taxrag.metrics import chunk_matches, mean_metrics, retrieval_metrics  # noqa: E402
from taxrag.pipeline import KnowledgeBase, PipelineConfig, answer_question, retrieve  # noqa: E402
from taxrag.temporal import assess_document, extract_query_time, make_query_time  # noqa: E402

GRADING_COLUMNS = [
    "dung_noi_dung_0_2", "day_du_0_2", "lien_quan_0_2", "co_can_cu_0_2", "trich_dan_dung_0_2",
    "dung_thoi_diem_0_2", "xu_ly_ngoai_le_0_2", "tu_choi_dung_0_1", "de_hieu_0_2",
    "co_thong_tin_ngoai_nguon_0_1", "trich_sai_van_ban_0_1", "dung_van_ban_sai_dieu_khoan_0_1",
    "dung_van_ban_het_hieu_luc_0_1", "ghi_chu",
]


# ------------------------------------------------------------------ dữ liệu
def load_corpus(corpus_dir: Path, catalog: Path) -> tuple[list[dict], dict[str, list]]:
    rows, problems = parse_catalog(catalog.read_bytes())
    for p in problems:
        print("  [danh mục]", p)
    documents, pages = [], {}
    for r in rows:
        f = corpus_dir / r["file"]
        if not f.exists():
            print(f"  [bỏ qua] không thấy tệp {f}")
            continue
        ext = extract_file(f.name, f.read_bytes())
        doc = {**r["meta"], "id": r["meta"]["document_number"]}
        documents.append(doc)
        pages[doc["id"]] = ext.pages
    return documents, pages


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_chunks(documents: list[dict], pages: dict[str, list], strategy: str) -> list[dict]:
    chunks = []
    for doc in documents:
        for c in chunk_document(pages[doc["id"]], strategy):
            d = c.to_dict()
            d["id"] = f'{doc["id"]}::{strategy}::{c.local_id}'
            d["parent_id"] = (f'{doc["id"]}::{strategy}::{c.parent_local_id}'
                              if c.parent_local_id is not None else None)
            d["document_id"] = doc["id"]
            d["document_number"] = doc["document_number"]
            chunks.append(d)
    return chunks


class EmbeddingCache:
    """Lưu embedding ra đĩa để chạy lại thực nghiệm không tốn thêm lượt gọi API."""

    def __init__(self, settings, cache_dir: Path):
        self.settings = settings
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"emb_{settings.embedding_model.replace('/', '_')}_{settings.embedding_dim}.pkl"
        self.data: dict[str, np.ndarray] = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}

    def get(self, texts: list[str], task: str) -> np.ndarray:
        from taxrag.llm import embed_texts

        keys = [hashlib.sha1(f"{task}|{t}".encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.data]
        if missing:
            vecs = embed_texts(self.settings, [t for _, t in missing], task_type=task)
            for (k, _), v in zip(missing, vecs):
                self.data[k] = v
            self.path.write_bytes(pickle.dumps(self.data))
        return np.stack([self.data[k] for k in keys])


# ------------------------------------------------------------------ chạy
def query_time_for(q: dict):
    return make_query_time(date.fromisoformat(q["as_of"])) if q.get("as_of") else \
        extract_query_time(q["question"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--catalog", type=Path, help="mặc định: <corpus>/catalog.csv")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--strategies", default="fixed,structural")
    ap.add_argument("--methods", default="bm25,dense,hybrid")
    ap.add_argument("--rerank", default="none", help="none,gemini,cross-encoder")
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--ablate-temporal", action="store_true",
                    help="thêm cấu hình không lọc hiệu lực để đo tác dụng của lọc theo thời điểm")
    ap.add_argument("--generate", action="store_true", help="sinh câu trả lời để chấm điểm")
    ap.add_argument("--generate-configs", default="structural|hybrid|gemini,hierarchical|hybrid|gemini",
                    help="cấu hình dùng khi sinh câu trả lời: chiến lược|phương pháp|reranker")
    ap.add_argument("--no-rag", action="store_true", help="thêm baseline LLM không truy hồi")
    args = ap.parse_args()

    settings = load_settings()
    ks = [int(k) for k in args.ks.split(",")]
    strategies = args.strategies.split(",")
    methods = args.methods.split(",")
    rerankers = args.rerank.split(",")
    args.out.mkdir(parents=True, exist_ok=True)

    print("Đọc kho văn bản…")
    documents, pages = load_corpus(args.corpus, args.catalog or args.corpus / "catalog.csv")
    questions = load_questions(args.questions)
    print(f"  {len(documents)} văn bản, {len(questions)} câu hỏi")

    needs_dense = any(m in ("dense", "hybrid") for m in methods) or args.generate
    emb = EmbeddingCache(settings, ROOT / "eval" / ".cache") if needs_dense and settings.gemini_api_key else None
    if needs_dense and emb is None:
        print("  [cảnh báo] chưa có GEMINI_API_KEY – bỏ các cấu hình dense/hybrid")
        methods = [m for m in methods if m == "bm25"]

    reranker_objs: dict[str, Any] = {"none": None}
    for r in rerankers:
        if r == "gemini" and settings.gemini_api_key:
            from taxrag.rerank import GeminiReranker

            reranker_objs[r] = GeminiReranker(settings)
        elif r.startswith("cross-encoder"):
            from taxrag.rerank import CrossEncoderReranker

            name = r.split(":", 1)[1] if ":" in r else "BAAI/bge-reranker-v2-m3"
            reranker_objs[r] = CrossEncoderReranker(name)
    rerankers = [r for r in rerankers if r in reranker_objs]

    kbs: dict[str, KnowledgeBase] = {}
    all_strats = set(strategies)
    if args.generate:
        all_strats |= {c.split("|")[0] for c in args.generate_configs.split(",")}
    for strat in sorted(all_strats):
        chunks = build_chunks(documents, pages, strat)
        vectors = None
        if emb is not None:
            docs = {d["id"]: d for d in documents}
            vectors = emb.get([index_text(c, docs[c["document_id"]]) for c in chunks], "RETRIEVAL_DOCUMENT")
        kbs[strat] = KnowledgeBase(documents, chunks, vectors=vectors)
        n_leaf = sum(1 for c in chunks if c.get("is_leaf", True))
        print(f"  chiến lược {strat}: {n_leaf} đoạn lá")

    qvecs = {}
    if emb is not None:
        vs = emb.get([q["question"] for q in questions], "RETRIEVAL_QUERY")
        qvecs = {q["id"]: v for q, v in zip(questions, vs)}

    # ---------------------------------------------------------- truy hồi
    configs = [(s, m, r, True) for s in strategies for m in methods for r in rerankers]
    if args.ablate_temporal:
        configs += [(s, m, "none", False) for s in strategies for m in methods]

    details, summary = [], []
    for strat, method, rr, tfilter in configs:
        name = f"{strat}|{method}|{rr}" + ("" if tfilter else "|không-lọc-hiệu-lực")
        kb = kbs[strat]
        rows, latencies, stale_hits = [], [], []
        for q in questions:
            if not q.get("gold"):
                continue
            qt = query_time_for(q)
            cfg = PipelineConfig(method=method, use_reranker=rr != "none",
                                 candidate_k=args.candidate_k, temporal_filter=tfilter, generate=False)
            t0 = time.perf_counter()
            cands, _, decisions = retrieve(kb, q["question"], qt, cfg, settings,
                                           reranker=reranker_objs[rr], query_vector=qvecs.get(q["id"]))
            latencies.append((time.perf_counter() - t0) * 1000)
            ranked = [kb.corpus.chunks[c.index] for c in cands]
            m = retrieval_metrics(ranked, q["gold"], ks)
            # Tỷ lệ top-5 chứa văn bản không có hiệu lực tại thời điểm hỏi (mục 30)
            invalid = any(not assess_document(kb.documents[c["document_id"]], qt).usable for c in ranked[:5])
            m["stale@5"] = 1.0 if invalid else 0.0
            stale_hits.append(invalid)
            rows.append(m)
            details.append({"config": name, "id": q["id"], "difficulty": q.get("difficulty"),
                            "intent": q.get("intent"), **m,
                            "top5": [f'{c["document_number"]} {c.get("path", "")}' for c in ranked[:5]]})
        agg = mean_metrics(rows)
        agg["latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
        summary.append({"config": name, "n": len(rows), **agg})
        print(f"  {name:45s} R@5={agg.get('recall@5', 0):.3f} MRR={agg.get('mrr', 0):.3f} "
              f"nDCG@5={agg.get('ndcg@5', 0):.3f} stale@5={agg.get('stale@5', 0):.2f}")

    write_summary(args.out, summary, ks)
    with (args.out / "retrieval_details.jsonl").open("w", encoding="utf-8") as fh:
        for d in details:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    write_breakdown(args.out, details)

    # ---------------------------------------------------------- sinh câu trả lời
    if args.generate or args.no_rag:
        run_generation(args, settings, questions, kbs, reranker_objs)
    print(f"Xong. Kết quả ở {args.out}")


def write_summary(out: Path, summary: list[dict], ks: list[int]) -> None:
    cols = ["config", "n"] + [f"recall@{k}" for k in ks] + [f"precision@{k}" for k in ks] + \
        ["mrr"] + [f"ndcg@{k}" for k in ks] + [f"doc_hit@{k}" for k in ks] + \
        [f"article_hit@{k}" for k in ks] + ["stale@5", "latency_ms"]
    with (out / "retrieval_summary.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    short = ["config", "n", "recall@1", "recall@3", "recall@5", "mrr", "ndcg@5",
             "article_hit@5", "stale@5", "latency_ms"]
    lines = ["| " + " | ".join(short) + " |", "|" + "---|" * len(short)]
    for s in sorted(summary, key=lambda x: -x.get("recall@5", 0)):
        lines.append("| " + " | ".join(
            f"{s.get(c, 0):.3f}" if isinstance(s.get(c), float) and c != "latency_ms"
            else f"{s.get(c, 0):.0f}" if c == "latency_ms" else str(s.get(c, "")) for c in short) + " |")
    (out / "retrieval_summary.md").write_text(
        "# Kết quả truy hồi\n\n" + "\n".join(lines) +
        "\n\nstale@5 = tỷ lệ câu hỏi mà top-5 chứa văn bản không có hiệu lực tại thời điểm hỏi.\n",
        encoding="utf-8")


def write_breakdown(out: Path, details: list[dict]) -> None:
    lines = ["# Kết quả theo nhóm câu hỏi (Recall@5 / MRR)\n"]
    for key, title in (("difficulty", "Mức độ khó"), ("intent", "Loại câu hỏi")):
        groups: dict[tuple, list] = defaultdict(list)
        for d in details:
            groups[(d["config"], d.get(key))].append(d)
        configs = sorted({d["config"] for d in details})
        values = sorted({d.get(key) for d in details}, key=str)
        lines += [f"\n## {title}\n", "| cấu hình | " + " | ".join(str(v) for v in values) + " |",
                  "|---|" + "---|" * len(values)]
        for c in configs:
            cells = []
            for v in values:
                g = groups.get((c, v), [])
                cells.append(f'{np.mean([x["recall@5"] for x in g]):.2f} / {np.mean([x["mrr"] for x in g]):.2f}'
                             if g else "–")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    (out / "breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_generation(args, settings, questions, kbs, reranker_objs) -> None:
    from taxrag.generation import generate_without_rag

    if not settings.gemini_api_key:
        print("  [bỏ qua sinh câu trả lời] chưa có GEMINI_API_KEY")
        return
    gen_configs = [c.split("|") for c in args.generate_configs.split(",")] if args.generate else []
    records = []
    for q in questions:
        qt = query_time_for(q)
        for strat, method, rr in gen_configs:
            kb = kbs[strat]
            cfg = PipelineConfig(method=method, use_reranker=rr != "none", candidate_k=args.candidate_k,
                                 context_mode="parent" if strat == "hierarchical" else "leaf",
                                 as_of=qt.end if q.get("as_of") else None)
            res = answer_question(kb, q["question"], settings, cfg, reranker=reranker_objs.get(rr))
            records.append(_grade_auto(q, f"{strat}|{method}|{rr}", res.answer, res.sources))
        if args.no_rag:
            ans = generate_without_rag(settings, q["question"], qt)
            records.append(_grade_auto(q, "LLM-không-RAG", ans.to_dict(), []))
        print(f"  sinh câu trả lời: {q['id']}")

    with (args.out / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.out / "answers_for_grading.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        base = ["id", "config", "question", "should_refuse", "status", "answer", "cited",
                "gold", "reference_answer"]
        w = csv.DictWriter(fh, fieldnames=base + GRADING_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({**r, "gold": json.dumps(r["gold"], ensure_ascii=False),
                        "cited": json.dumps(r["cited"], ensure_ascii=False)})

    # Chỉ số tự động (không thay thế chấm điểm chuyên gia)
    by_cfg = defaultdict(list)
    for r in records:
        by_cfg[r["config"]].append(r)
    lines = ["# Chỉ số tự động cho câu trả lời\n",
             "| cấu hình | từ chối đúng | từ chối sai | trích dẫn đúng văn bản | đúng điều-khoản | trích dẫn không tồn tại |",
             "|---|---|---|---|---|---|"]
    for cfg_name, rs in by_cfg.items():
        neg = [r for r in rs if r["should_refuse"]]
        pos = [r for r in rs if not r["should_refuse"]]
        refuse_ok = np.mean([r["refused"] for r in neg]) if neg else float("nan")
        refuse_bad = np.mean([r["refused"] for r in pos]) if pos else float("nan")
        cited_pos = [r for r in pos if r["cited"]]
        doc_ok = np.mean([r["cite_doc_ok"] for r in cited_pos]) if cited_pos else float("nan")
        prov_ok = np.mean([r["cite_provision_ok"] for r in cited_pos]) if cited_pos else float("nan")
        invalid = np.mean([bool(r["invalid_citations"]) for r in rs])
        lines.append(f"| {cfg_name} | {refuse_ok:.2f} | {refuse_bad:.2f} | {doc_ok:.2f} | {prov_ok:.2f} | {invalid:.2f} |")
    lines.append("\nVới LLM-không-RAG, cột trích dẫn để trống vì không có nguồn truy hồi; "
                 "cần chuyên gia chấm cột co_thong_tin_ngoai_nguon trong answers_for_grading.csv.")
    (args.out / "generation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _grade_auto(q, config, answer: dict, sources: list[dict]) -> dict:
    cited_chunks = [sources[n - 1]["chunk"] for n in answer.get("cited_sources", []) if n <= len(sources)]
    gold = q.get("gold") or []
    return {
        "id": q["id"], "config": config, "question": q["question"],
        "should_refuse": bool(q.get("should_refuse")), "status": answer.get("status"),
        "refused": answer.get("status") in REFUSAL_STATUSES,
        "answer": answer.get("markdown", ""),
        "cited": [f'{c.get("document_number")} {c.get("path", "")}' for c in cited_chunks],
        "cite_doc_ok": any(chunk_matches(c, g, "document") for c in cited_chunks for g in gold),
        "cite_provision_ok": any(chunk_matches(c, g) for c in cited_chunks for g in gold),
        "invalid_citations": answer.get("invalid_citations", []),
        "gold": gold, "reference_answer": q.get("reference_answer", ""),
    }


if __name__ == "__main__":
    main()
