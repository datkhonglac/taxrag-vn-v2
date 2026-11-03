"""Truy cập Supabase qua REST API (PostgREST).

Sửa các lỗi của v2:
* Phân trang khi đọc: PostgREST của Supabase giới hạn mặc định 1.000 dòng / lần gọi,
  v2 đọc một lần nên kho > 1.000 đoạn bị cắt mà không báo lỗi.
* Nạp văn bản an toàn: tạo bản mới ở trạng thái ``processing`` → ghi đoạn → chuyển
  ``ready`` → mới xóa bản cũ. Lỗi giữa chừng thì xóa bản mới, bản cũ còn nguyên.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import requests

from .config import Settings
from .text import normalize_doc_number

PAGE_SIZE = 1000
CHUNK_COLUMNS = (
    "id,document_id,chunk_index,page_number,page_end,strategy,level,is_leaf,parent_id,"
    "heading,context,path,chapter,section,article,article_title,clause,point,covers,content"
)
DOC_DATE_FIELDS = ("issued_date", "effective_date", "expiry_date", "retrieval_date")


class SupabaseError(RuntimeError):
    pass


class SupabaseStore:
    def __init__(self, settings: Settings, timeout: int = 60):
        if not settings.supabase_url or not settings.supabase_key:
            raise SupabaseError("Chưa cấu hình SUPABASE_URL hoặc SUPABASE_KEY.")
        self.base = settings.supabase_url.rstrip("/") + "/rest/v1"
        self.key = settings.supabase_key
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------
    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        # Khóa mới dạng sb_secret_/sb_publishable_ gửi qua header apikey;
        # khóa JWT kiểu cũ cần thêm Authorization.
        h = {"apikey": self.key, "Content-Type": "application/json", "Accept": "application/json"}
        if self.key.count(".") == 2:
            h["Authorization"] = f"Bearer {self.key}"
        if prefer:
            h["Prefer"] = prefer
        return h

    def request(self, method: str, path: str, *, params: dict | None = None,
                json_data: Any = None, prefer: str | None = None) -> Any:
        resp = self.session.request(
            method, f"{self.base}/{path}", headers=self._headers(prefer),
            params=params, data=json.dumps(json_data, default=_json_default) if json_data is not None else None,
            timeout=self.timeout,
        )
        if not resp.ok:
            raise SupabaseError(f"Supabase HTTP {resp.status_code}: {resp.text[:600]}")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def get_all(self, table: str, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Đọc toàn bộ bảng theo từng trang 1.000 dòng."""
        params = dict(params or {})
        params.setdefault("order", "id.asc")
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.request("GET", table, params={**params, "limit": str(PAGE_SIZE),
                                                       "offset": str(offset)}) or []
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE

    def ping(self) -> tuple[bool, str]:
        try:
            self.request("GET", "documents", params={"select": "id", "limit": "1"})
            return True, "Kết nối Supabase thành công."
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Văn bản
    # ------------------------------------------------------------------
    def list_documents(self, include_processing: bool = False) -> list[dict[str, Any]]:
        params = {"select": "*", "order": "effective_date.desc.nullslast,uploaded_at.desc,id.asc"}
        if not include_processing:
            params["ingest_status"] = "eq.ready"
        return self.get_all("documents", params)

    def load_chunks(self, document_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        params = {"select": CHUNK_COLUMNS, "order": "document_id.asc,chunk_index.asc,id.asc"}
        if document_ids is not None:
            ids = list(document_ids)
            if not ids:
                return []
            params["document_id"] = f"in.({','.join(ids)})"
        return self.get_all("chunks", params)

    def update_document(self, document_id: str, fields: dict[str, Any]) -> None:
        fields = {**fields, "updated_at": datetime.now(timezone.utc).isoformat()}
        self.request("PATCH", "documents", params={"id": f"eq.{document_id}"},
                     json_data=fields, prefer="return=minimal")

    def delete_document(self, document_id: str) -> None:
        self.request("DELETE", "documents", params={"id": f"eq.{document_id}"},
                     prefer="return=minimal")

    def delete_all_documents(self) -> None:
        self.request("DELETE", "documents", params={"id": "not.is.null"}, prefer="return=minimal")

    def find_by_number(self, document_number: str) -> list[dict[str, Any]]:
        rows = self.request("GET", "documents", params={
            "select": "id,document_number,ingest_status",
            "document_number": f"ilike.{document_number.strip()}",
        }) or []
        key = normalize_doc_number(document_number)
        return [r for r in rows if normalize_doc_number(r.get("document_number")) == key]

    def save_document(
        self,
        metadata: dict[str, Any],
        chunks: list[dict[str, Any]],
        *,
        page_count: int,
        uploaded_by: str,
        embeddings: list[np.ndarray | None] | np.ndarray | None = None,
        content_hash: str = "",
    ) -> str:
        """Lưu văn bản + các đoạn. Nếu đã có văn bản cùng số hiệu thì thay thế an toàn."""
        old = self.find_by_number(metadata["document_number"])
        new_id = str(uuid.uuid4())
        doc_row = {
            **{k: v for k, v in metadata.items() if v not in ("", None)},
            "id": new_id,
            "document_name": metadata.get("document_title", ""),
            "page_count": page_count,
            "chunk_count": len(chunks),
            "uploaded_by": uploaded_by or "Admin",
            "content_hash": content_hash,
            "ingest_status": "processing",
        }
        self.request("POST", "documents", json_data=doc_row, prefer="return=minimal")
        try:
            id_map = {c["local_id"]: str(uuid.uuid4()) for c in chunks}
            payload = []
            for i, c in enumerate(chunks):
                row = {
                    "id": id_map[c["local_id"]],
                    "document_id": new_id,
                    "chunk_index": i,
                    "page_number": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "strategy": c.get("strategy"),
                    "level": c.get("level"),
                    "is_leaf": c.get("is_leaf", True),
                    "parent_id": id_map.get(c.get("parent_local_id")),
                    "heading": c.get("heading"),
                    "context": c.get("context"),
                    "path": c.get("path"),
                    "chapter": c.get("chapter"),
                    "section": c.get("section"),
                    "article": c.get("article"),
                    "article_title": c.get("article_title"),
                    "clause": c.get("clause"),
                    "point": c.get("point"),
                    "covers": c.get("covers") or [],
                    "content": c.get("content", ""),
                }
                if embeddings is not None and embeddings[i] is not None:
                    row["embedding"] = _vec(embeddings[i])
                payload.append(row)
            for start in range(0, len(payload), 100):
                self.request("POST", "chunks", json_data=payload[start:start + 100],
                             prefer="return=minimal")
            self.update_document(new_id, {"ingest_status": "ready"})
        except Exception:
            try:
                self.delete_document(new_id)
            except Exception:
                pass
            raise
        for row in old:  # chỉ xóa bản cũ khi bản mới đã sẵn sàng
            if row["id"] != new_id:
                self.delete_document(row["id"])
        return new_id

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def chunks_missing_embedding(self) -> list[dict[str, Any]]:
        return self.get_all("chunks", {
            "select": "id,document_id,heading,context,content,is_leaf",
            "embedding": "is.null",
        })

    def set_embeddings(self, ids: list[str], vectors: np.ndarray) -> None:
        for cid, vec in zip(ids, vectors):
            self.request("PATCH", "chunks", params={"id": f"eq.{cid}"},
                         json_data={"embedding": _vec(vec)}, prefer="return=minimal")

    def match_chunks(self, query_vec: np.ndarray, top_n: int,
                     doc_ids: set[str] | None) -> list[tuple[str, float]]:
        rows = self.request("POST", "rpc/match_chunks", json_data={
            "query_embedding": _vec(query_vec),
            "match_count": int(top_n),
            "doc_ids": sorted(doc_ids) if doc_ids is not None else None,
        }) or []
        return [(r["id"], float(r["similarity"])) for r in rows]

    # ------------------------------------------------------------------
    # Quan hệ văn bản
    # ------------------------------------------------------------------
    def list_relations(self) -> list[dict[str, Any]]:
        return self.get_all("document_relations", {"select": "*"})

    def add_relation(self, source_number: str, target_number: str, relation_type: str,
                     note: str = "") -> None:
        self.request("POST", "document_relations", json_data={
            "source_number": source_number.strip(), "target_number": target_number.strip(),
            "relation_type": relation_type, "note": note,
        }, prefer="resolution=ignore-duplicates,return=minimal")

    def delete_relation(self, relation_id: str) -> None:
        self.request("DELETE", "document_relations", params={"id": f"eq.{relation_id}"},
                     prefer="return=minimal")

    # ------------------------------------------------------------------
    # Lịch sử câu hỏi
    # ------------------------------------------------------------------
    def log_query(self, record: dict[str, Any]) -> str | None:
        record = {**record, "id": record.get("id") or str(uuid.uuid4())}
        self.request("POST", "query_logs", json_data=record, prefer="return=minimal")
        return record["id"]

    def set_feedback(self, log_id: str, value: int, note: str = "") -> None:
        self.request("PATCH", "query_logs", params={"id": f"eq.{log_id}"},
                     json_data={"feedback": value, "feedback_note": note or None},
                     prefer="return=minimal")

    def list_query_logs(self, limit: int = 500) -> list[dict[str, Any]]:
        return self.request("GET", "query_logs", params={
            "select": "id,created_at,question,as_of_label,status,latency_ms,feedback,feedback_note,sources,answer,config",
            "order": "created_at.desc", "limit": str(limit),
        }) or []


def _vec(v: Any) -> str:
    """pgvector nhận chuỗi dạng '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.6f}" for x in np.asarray(v).ravel()) + "]"


def _json_default(o: Any):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"Không tuần tự hóa được {type(o)}")
