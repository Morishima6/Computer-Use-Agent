from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

_embedding_client: Optional[OpenAI] = None


def _get_embedding_api_key() -> Optional[str]:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if api_key:
        return api_key

    api_key = os.getenv("QWEN_API_KEY")
    if api_key:
        return api_key

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key

    return None


def _get_embedding_client() -> OpenAI:
    global _embedding_client
    if _embedding_client is None:
        api_key = _get_embedding_api_key()
        if not api_key:
            raise RuntimeError(
                "Embedding API key is not configured. "
                "Set SILICONFLOW_API_KEY "
                "(or QWEN_API_KEY / DASHSCOPE_API_KEY for backward compatibility)."
            )
        _embedding_client = OpenAI(api_key=api_key, base_url=SILICONFLOW_BASE_URL)
    return _embedding_client


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[List[float]]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return None

    try:
        response = _get_embedding_client().embeddings.create(
            model=model,
            input=normalized_text,
        )
    except Exception as exc:
        print(f"Embedding request failed: {exc}")
        return None

    if not getattr(response, "data", None):
        return None

    embedding = response.data[0].embedding
    return embedding if isinstance(embedding, list) and embedding else None


__all__ = ["DEFAULT_EMBEDDING_MODEL", "get_embedding"]
