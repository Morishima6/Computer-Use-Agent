"""Cluster parameterized units into Canonical Unit candidate groups."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .schemas import ClusterAssignment, ParameterizedUnitRecord

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    AgglomerativeClustering = None

try:
    from ...retrieval.common_llm_call import get_embedding
except ImportError:
    get_embedding = None


# WANDER Phase 3 uses abstract intent embedding similarity for clustering.
DEFAULT_SIMILARITY_THRESHOLD = 0.82
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_CACHE_FILENAME = "abstract_intent_embeddings.json"


def cluster_parameterized_units(
    units: Sequence[ParameterizedUnitRecord],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cache_root: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[ClusterAssignment]:
    """Cluster parameterized units within hard app groups."""

    if not units:
        return []

    cluster_results: List[ClusterAssignment] = []
    cluster_counter = 1

    grouped_units = _group_units_by_app(units)
    total_groups = len(grouped_units)

    for index, (group_key, group_units) in enumerate(grouped_units.items(), start=1):
        labels, embedding_stats = _cluster_group(
            group_units,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            cache_root=cache_root,
        )
        for cluster_units in _labels_to_clusters(group_units, labels):
            cluster_id = f"cluster_{cluster_counter:05d}"
            cluster_results.append(
                ClusterAssignment(
                    cluster_id=cluster_id,
                    group_key=group_key,
                    app_name=cluster_units[0].app_name if cluster_units else "UNKNOWN",
                    unit_type=cluster_units[0].unit_type if cluster_units else "UNKNOWN",
                    unit_instances=cluster_units,
                    centroid_instance_id=_select_centroid_instance_id(cluster_units),
                )
            )
            cluster_counter += 1
        if progress_callback is not None:
            progress_callback(
                index,
                total_groups,
                (
                    f"{group_key} ({len(group_units)} units, "
                    f"cache_hits={embedding_stats['cache_hits']}, fetched={embedding_stats['fetched']})"
                ),
            )

    return cluster_results


def build_cluster_text(unit: ParameterizedUnitRecord) -> str:
    """Build the semantic text used for clustering."""

    return str(unit.abstract_intent or "").strip()


def _group_units_by_app(
    units: Sequence[ParameterizedUnitRecord],
) -> Dict[str, List[ParameterizedUnitRecord]]:
    groups: Dict[str, List[ParameterizedUnitRecord]] = defaultdict(list)
    for unit in units:
        group_key = _normalize_app_group_key(unit.app_name)
        groups[group_key].append(unit)

    return {
        key: sorted(
            value,
            key=lambda item: (
                item.source_user,
                item.trace_id or "",
                item.segment_order,
                item.step_indices[0] if item.step_indices else 0,
                item.instance_id,
            ),
        )
        for key, value in sorted(groups.items(), key=lambda pair: pair[0])
    }


def _normalize_app_group_key(app_name: Optional[str]) -> str:
    normalized = str(app_name or "").strip()
    if not normalized:
        return "unknown"
    return normalized.casefold()


def _cluster_group(
    units: Sequence[ParameterizedUnitRecord],
    *,
    similarity_threshold: float,
    embedding_model: str,
    cache_root: Optional[Path],
) -> tuple[List[int], Dict[str, int]]:
    if len(units) <= 1:
        return [0] * len(units), {"cache_hits": 0, "fetched": 0}

    if AgglomerativeClustering is None or np is None:
        raise RuntimeError(
            "Phase 3 clustering requires sklearn and numpy. "
            "Install these dependencies to run the documented agglomerative clustering pipeline."
        )

    embeddings, embedding_stats = _resolve_unit_embeddings(
        units,
        embedding_model=embedding_model,
        cache_root=cache_root,
    )
    similarity_matrix = _build_similarity_matrix(embeddings)
    distance_matrix = [
        [max(0.0, 1.0 - similarity_matrix[row][col]) for col in range(len(units))]
        for row in range(len(units))
    ]

    return _cluster_with_agglomerative(distance_matrix, similarity_threshold), embedding_stats


def _build_similarity_matrix(
    embeddings: Sequence[Sequence[float]],
) -> List[List[float]]:
    matrix: List[List[float]] = []

    for row in range(len(embeddings)):
        row_values: List[float] = []
        for col in range(len(embeddings)):
            if row == col:
                row_values.append(1.0)
                continue

            similarity = _unit_similarity(
                row_embedding=embeddings[row],
                col_embedding=embeddings[col],
            )
            row_values.append(similarity)
        matrix.append(row_values)

    return matrix


def _resolve_unit_embeddings(
    units: Sequence[ParameterizedUnitRecord],
    *,
    embedding_model: str,
    cache_root: Optional[Path],
) -> tuple[List[List[float]], Dict[str, int]]:
    text_cache: Dict[str, List[float]] = {}
    user_cache_payloads = _load_embedding_cache_payloads(cache_root, units)
    modified_users: set[str] = set()
    embeddings: List[List[float]] = []
    stats = {"cache_hits": 0, "fetched": 0}

    for unit in units:
        text = str(build_cluster_text(unit) or "").strip()
        if not text:
            raise RuntimeError(
                "Phase 3 clustering requires non-empty abstract_intent for every unit."
            )

        entry = user_cache_payloads.get(unit.source_user, {}).get(unit.instance_id)
        embedding = _embedding_from_cache_entry(
            entry,
            expected_text=text,
            embedding_model=embedding_model,
        )
        if embedding is not None:
            embeddings.append(embedding)
            text_cache.setdefault(text, embedding)
            stats["cache_hits"] += 1
            continue

        embedding = text_cache.get(text)
        if embedding is None:
            embedding = _get_text_embedding(text, embedding_model=embedding_model)
            text_cache[text] = embedding
            stats["fetched"] += 1

        user_cache_payloads.setdefault(unit.source_user, {})[unit.instance_id] = {
            "text": text,
            "model": embedding_model,
            "embedding": embedding,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        modified_users.add(unit.source_user)
        embeddings.append(list(embedding))

    if modified_users:
        _save_embedding_cache_payloads(
            cache_root,
            user_cache_payloads,
            modified_users=modified_users,
        )

    return embeddings, stats


def _load_embedding_cache_payloads(
    cache_root: Optional[Path],
    units: Sequence[ParameterizedUnitRecord],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if cache_root is None:
        return {}

    payloads: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source_user in sorted({unit.source_user for unit in units if unit.source_user}):
        cache_path = _embedding_cache_path(cache_root, source_user)
        if not cache_path.exists():
            payloads[source_user] = {}
            continue
        try:
            raw_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payloads[source_user] = {}
            continue
        entries = raw_payload.get("entries", {})
        if isinstance(entries, dict):
            payloads[source_user] = {
                str(key): value
                for key, value in entries.items()
                if isinstance(value, dict)
            }
        else:
            payloads[source_user] = {}
    return payloads


def _save_embedding_cache_payloads(
    cache_root: Optional[Path],
    payloads: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    modified_users: set[str],
) -> None:
    if cache_root is None:
        return

    for source_user in sorted(modified_users):
        cache_path = _embedding_cache_path(cache_root, source_user)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "source_user": source_user,
            "entries": payloads.get(source_user, {}),
        }
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, cache_path)


def _embedding_cache_path(cache_root: Path, source_user: str) -> Path:
    return Path(cache_root) / str(source_user) / EMBEDDING_CACHE_FILENAME


def _embedding_from_cache_entry(
    entry: Optional[Dict[str, Any]],
    *,
    expected_text: str,
    embedding_model: str,
) -> Optional[List[float]]:
    if not isinstance(entry, dict):
        return None
    if str(entry.get("text") or "").strip() != expected_text:
        return None
    if str(entry.get("model") or "").strip() != embedding_model:
        return None
    embedding = entry.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return None
    try:
        return [float(value) for value in embedding]
    except (TypeError, ValueError):
        return None


def _get_text_embedding(text: str, *, embedding_model: str) -> List[float]:
    if get_embedding is None:
        raise RuntimeError(
            "Phase 3 clustering requires a working embedding function for abstract_intent."
        )

    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise RuntimeError(
            "Phase 3 clustering requires non-empty abstract_intent for every unit."
        )

    embedding = _get_text_embedding_cached(normalized_text, embedding_model)
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(
            f"Embedding generation failed for abstract_intent: {normalized_text!r}"
        )
    return [float(value) for value in embedding]


@lru_cache(maxsize=2048)
def _get_text_embedding_cached(text: str, embedding_model: str):
    return get_embedding(text, model=embedding_model)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        numerator += float(left_value) * float(right_value)
        left_norm += float(left_value) ** 2
        right_norm += float(right_value) ** 2

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    return numerator / math.sqrt(left_norm * right_norm)


def _unit_similarity(
    *,
    row_embedding: Sequence[float],
    col_embedding: Sequence[float],
) -> float:
    return max(0.0, min(1.0, _cosine_similarity(row_embedding, col_embedding)))


def _cluster_with_agglomerative(
    distance_matrix: List[List[float]],
    similarity_threshold: float,
) -> List[int]:
    matrix = np.asarray(distance_matrix, dtype="float32")
    kwargs = {
        "n_clusters": None,
        "distance_threshold": max(0.0, 1.0 - similarity_threshold),
        "linkage": "average",
    }

    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)

    return [int(label) for label in model.fit_predict(matrix)]


def _labels_to_clusters(
    units: Sequence[ParameterizedUnitRecord],
    labels: Sequence[int],
) -> List[List[ParameterizedUnitRecord]]:
    grouped: Dict[int, List[ParameterizedUnitRecord]] = defaultdict(list)
    for unit, label in zip(units, labels):
        grouped[int(label)].append(unit)

    return [grouped[label] for label in sorted(grouped)]


def _select_centroid_instance_id(units: Sequence[ParameterizedUnitRecord]) -> Optional[str]:
    if not units:
        return None
    ranked = sorted(
        units,
        key=lambda unit: (
            -len(unit.abstract_intent or ""),
            -len(unit.unit_before_state or ""),
            -len(unit.unit_after_state or ""),
            unit.instance_id,
        ),
    )
    return ranked[0].instance_id


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_EMBEDDING_MODEL",
    "build_cluster_text",
    "cluster_parameterized_units",
]
