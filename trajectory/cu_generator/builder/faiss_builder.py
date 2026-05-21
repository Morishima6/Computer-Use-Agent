"""Build and persist FAISS resources for Canonical Unit retrieval."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .schemas import CanonicalUnitRecord

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

try:
    from ...retrieval.common_llm_call import get_embedding
except ImportError:
    get_embedding = None


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def build_faiss_artifacts(
    canonical_units: Sequence[CanonicalUnitRecord],
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[object | None, Dict[str, str], List[str]]:
    """Build an in-memory FAISS index and its id mapping."""

    warnings: List[str] = []

    if not canonical_units:
        return None, {}, warnings

    if faiss is None or np is None:
        warnings.append("faiss or numpy is unavailable; skipping FAISS index build.")
        return None, {}, warnings

    if get_embedding is None:
        warnings.append("Embedding function is unavailable; skipping FAISS index build.")
        return None, {}, warnings

    vectors: List[List[float]] = []
    faiss_to_cu: Dict[str, str] = {}
    expected_dim: Optional[int] = None

    total_units = len(canonical_units)
    for index, canonical_unit in enumerate(canonical_units, start=1):
        text = str(canonical_unit.intent or "").strip()
        if not text:
            warnings.append(f"CU {canonical_unit.cu_id} has empty intent; skipped in FAISS index.")
            if progress_callback is not None:
                progress_callback(index, total_units, f"{canonical_unit.cu_id} skipped: empty intent")
            continue

        embedding = get_embedding(text, model=embedding_model)
        if not isinstance(embedding, list) or not embedding:
            warnings.append(
                f"Embedding generation failed for CU {canonical_unit.cu_id}; skipped in FAISS index."
            )
            if progress_callback is not None:
                progress_callback(index, total_units, f"{canonical_unit.cu_id} skipped: embedding failed")
            continue

        if expected_dim is None:
            expected_dim = len(embedding)
        elif len(embedding) != expected_dim:
            warnings.append(
                f"Embedding dimension mismatch for CU {canonical_unit.cu_id}; expected {expected_dim}, got {len(embedding)}."
            )
            if progress_callback is not None:
                progress_callback(index, total_units, f"{canonical_unit.cu_id} skipped: dimension mismatch")
            continue

        faiss_index_position = str(len(vectors))
        vectors.append([float(value) for value in embedding])
        faiss_to_cu[faiss_index_position] = canonical_unit.cu_id
        if progress_callback is not None:
            progress_callback(index, total_units, canonical_unit.cu_id)

    if not vectors or expected_dim is None:
        warnings.append("No valid CU embeddings were collected; FAISS index was not created.")
        return None, {}, warnings

    matrix = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(expected_dim)
    index.add(matrix)
    return index, faiss_to_cu, warnings


__all__ = ["DEFAULT_EMBEDDING_MODEL", "build_faiss_artifacts"]
