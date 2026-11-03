import json
from types import SimpleNamespace

import pytest

from taxrag.config import Settings
from taxrag.db import PAGE_SIZE, SupabaseError, SupabaseStore


class FakeResp:
    def __init__(self, data, ok=True, status=200):
        self._d, self.ok, self.status_code = data, ok, status
        self.content = json.dumps(data).encode() if data is not None else b""
        self.text = self.content.decode()

    def json(self):
        return self._d


def store_with(handler):
    s = SupabaseStore(Settings(supabase_url="https://x.supabase.co", supabase_key="sb_secret_x"))
    s.session = SimpleNamespace(request=handler)
    return s


def test_get_all_paginates_past_1000_rows():
    total = 2500
    calls = []

    def handler(method, url, headers, params, data, timeout):
        off, lim = int(params["offset"]), int(params["limit"])
        calls.append(off)
        return FakeResp([{"id": i} for i in range(off, min(off + lim, total))])

    rows = store_with(handler).get_all("chunks")
    assert len(rows) == total and calls == [0, PAGE_SIZE, 2 * PAGE_SIZE]


def test_save_document_keeps_old_version_when_insert_fails():
    log = []

    def handler(method, url, headers, params, data, timeout):
        table = url.rsplit("/", 1)[-1]
        log.append((method, table, params))
        if method == "GET":
            return FakeResp([{"id": "old-id", "document_number": "01/2025/TT-BTC"}])
        if method == "POST" and table == "chunks":
            return FakeResp({"message": "boom"}, ok=False, status=500)
        return FakeResp(None)

    s = store_with(handler)
    with pytest.raises(SupabaseError):
        s.save_document({"document_number": "01/2025/TT-BTC", "document_title": "TT"},
                        [{"local_id": 0, "content": "x"}], page_count=1, uploaded_by="t")
    deletes = [p["id"] for m, t, p in log if m == "DELETE"]
    assert deletes and all(d != "eq.old-id" for d in deletes)  # chỉ xóa bản mới, bản cũ còn


def test_save_document_replaces_old_after_success():
    log = []

    def handler(method, url, headers, params, data, timeout):
        log.append((method, url.rsplit("/", 1)[-1], params, data))
        if method == "GET":
            return FakeResp([{"id": "old-id", "document_number": "01/2025/tt-btc"}])
        return FakeResp(None)

    s = store_with(handler)
    s.save_document({"document_number": "01/2025/TT-BTC", "document_title": "TT"},
                    [{"local_id": 0, "content": "a"}, {"local_id": 1, "content": "b", "parent_local_id": 0}],
                    page_count=1, uploaded_by="t")
    order = [(m, t) for m, t, _, _ in log]
    assert order.index(("PATCH", "documents")) < order.index(("DELETE", "documents"))
    chunk_rows = json.loads(next(d for m, t, _, d in log if t == "chunks"))
    assert chunk_rows[1]["parent_id"] == chunk_rows[0]["id"]
