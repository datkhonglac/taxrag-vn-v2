"""Tổng hợp bảng chấm điểm của chuyên gia (answers_for_grading.csv đã điền).

    python -m eval.summarize_grades eval/results/run1/answers_for_grading.csv

In ra điểm trung bình từng tiêu chí và các tỷ lệ hallucination (mục 29–30) theo cấu hình.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

HALLUCINATION = {
    "co_thong_tin_ngoai_nguon_0_1": "Tỷ lệ chứa thông tin không có trong nguồn",
    "trich_sai_van_ban_0_1": "Tỷ lệ trích dẫn sai văn bản",
    "dung_van_ban_sai_dieu_khoan_0_1": "Tỷ lệ đúng văn bản nhưng sai điều, khoản",
    "dung_van_ban_het_hieu_luc_0_1": "Tỷ lệ dùng văn bản hết hiệu lực",
}


def main(path: str) -> None:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig")))
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    criteria = [c for c in rows[0] if c.endswith(("_0_2", "_0_1"))] if rows else []
    out = ["| tiêu chí | " + " | ".join(by_cfg) + " |", "|---|" + "---|" * len(by_cfg)]
    for c in criteria:
        cells = []
        for rs in by_cfg.values():
            vals = [float(r[c]) for r in rs if r.get(c, "").strip() != ""]
            cells.append(f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "–")
        out.append(f"| {HALLUCINATION.get(c, c)} | " + " | ".join(cells) + " |")
    # Trả lời khi đáng ra phải từ chối
    cells = []
    for rs in by_cfg.values():
        neg = [r for r in rs if r["should_refuse"] == "True"]
        bad = [r for r in neg if r["status"] in ("answered", "partial")]
        cells.append(f"{len(bad) / len(neg):.2f} (n={len(neg)})" if neg else "–")
    out.append("| Tỷ lệ trả lời khi đáng ra phải từ chối | " + " | ".join(cells) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
