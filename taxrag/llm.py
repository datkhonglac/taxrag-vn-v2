"""Lớp mỏng bao quanh Gemini API: sinh văn bản, sinh JSON, embedding.

Import ``google.genai`` muộn để unit test và bộ đánh giá BM25 chạy được
khi chưa cài thư viện hoặc chưa có API key.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .config import Settings


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _client(api_key: str):
    if not api_key:
        raise LLMUnavailable("Chưa cấu hình GEMINI_API_KEY.")
    from google import genai

    return genai.Client(api_key=api_key)


def _retry(fn, attempts: int = 3, base_delay: float = 1.5):
    for i in range(attempts):
        try:
            return fn()
        except LLMUnavailable:
            raise
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))


def generate_json(
    settings: Settings,
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    from google.genai import types

    client = _client(settings.gemini_api_key)
    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature,
        response_mime_type="application/json",
    )
    response = _retry(
        lambda: client.models.generate_content(
            model=model or settings.gemini_model, contents=prompt, config=cfg
        )
    )
    return parse_json(getattr(response, "text", "") or "")


def generate_text(settings: Settings, prompt: str, *, system: str = "",
                  temperature: float = 0.2) -> str:
    from google.genai import types

    client = _client(settings.gemini_api_key)
    cfg = types.GenerateContentConfig(system_instruction=system or None, temperature=temperature)
    response = _retry(
        lambda: client.models.generate_content(
            model=settings.gemini_model, contents=prompt, config=cfg
        )
    )
    return getattr(response, "text", "") or ""


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    return data if isinstance(data, dict) else {"items": data}


def embed_texts(
    settings: Settings,
    texts: Sequence[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 50,
) -> np.ndarray:
    """Embedding bằng Gemini. task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY."""
    from google.genai import types

    client = _client(settings.gemini_api_key)
    cfg = types.EmbedContentConfig(
        task_type=task_type, output_dimensionality=settings.embedding_dim
    )
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[start:start + batch_size]]
        result = _retry(
            lambda b=batch: client.models.embed_content(
                model=settings.embedding_model, contents=b, config=cfg
            )
        )
        out.extend(e.values for e in result.embeddings)
    arr = np.asarray(out, dtype=np.float32)
    # Khi giảm số chiều (output_dimensionality), cần chuẩn hóa lại độ dài vector
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


def embed_query(settings: Settings, text: str) -> np.ndarray:
    return embed_texts(settings, [text], task_type="RETRIEVAL_QUERY")[0]
