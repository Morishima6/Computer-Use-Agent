"""Persist Phase 3 build artifacts to the cu_base directory."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .schemas import CanonicalUnitRecord

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


def persist_phase3_artifacts(
    output_dir: Path,
    *,
    canonical_units: Sequence[CanonicalUnitRecord],
    transitions: Dict[str, Dict[str, int]],
    instance_to_cu: Dict[str, str],
    faiss_to_cu: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    faiss_index: object | None = None,
) -> Dict[str, Path]:
    """Write the Phase 3 outputs to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_metadata = _build_metadata(
        canonical_units=canonical_units,
        instance_to_cu=instance_to_cu,
        config=config or {},
        metadata=metadata or {},
    )

    cu_base_path = target_dir / "cu_base.json"
    transitions_path = target_dir / "transitions.json"
    mappings_path = target_dir / "mappings.json"
    config_path = target_dir / "config.json"
    faiss_path = target_dir / "faiss_intent.index"

    _write_json(
        cu_base_path,
        {
            "metadata": resolved_metadata,
            "canonical_units": [canonical_unit.to_dict() for canonical_unit in canonical_units],
        },
    )
    _write_json(transitions_path, transitions)
    _write_json(
        mappings_path,
        {
            "faiss_to_cu": dict(sorted(faiss_to_cu.items(), key=lambda item: int(item[0]))),
            "instance_to_cu": dict(sorted(instance_to_cu.items(), key=lambda item: item[0])),
        },
    )
    _write_json(config_path, config or {})

    written_paths = {
        "cu_base": cu_base_path,
        "transitions": transitions_path,
        "mappings": mappings_path,
        "config": config_path,
    }

    if faiss_index is not None and faiss is not None:
        tmp_faiss_path = _tmp_path_for(faiss_path)
        faiss.write_index(faiss_index, str(tmp_faiss_path))
        os.replace(tmp_faiss_path, faiss_path)
        written_paths["faiss_index"] = faiss_path

    return written_paths


def _build_metadata(
    *,
    canonical_units: Sequence[CanonicalUnitRecord],
    instance_to_cu: Dict[str, str],
    config: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(metadata)
    payload.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")
    payload.setdefault("canonical_unit_count", len(canonical_units))
    payload.setdefault("covered_instance_count", len(instance_to_cu))
    payload.setdefault("config", dict(config))
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = _tmp_path_for(path)
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _tmp_path_for(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


__all__ = ["persist_phase3_artifacts"]
