"""Main entrypoint and orchestration for the Phase 3 builder."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from .clusterer import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    cluster_parameterized_units,
)
from .cu_builder import DEFAULT_ACTION_MATCH_THRESHOLD, build_canonical_units
from .faiss_builder import build_faiss_artifacts
from .persist import persist_phase3_artifacts
from .schemas import CanonicalUnitRecord, ClusterAssignment, ParameterizedUnitRecord
from .transition_builder import build_transition_table
from .units_loader import load_parameterized_units


DEFAULT_SEGMENTS_DIR = Path(__file__).resolve().parents[1] / "segments"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "cu_base"
CHECKPOINT_DIRNAME = ".phase3_checkpoints"
STAGE_SEQUENCE = ("load", "cluster", "build-cu", "transitions", "faiss", "persist")
CHECKPOINT_STAGE_TO_FILE = {
    "load": "units",
    "cluster": "clusters",
    "build-cu": "canonical_units",
    "transitions": "transitions",
}


class ProgressReporter:
    """Lightweight CLI progress reporter with throttled loop updates."""

    def __init__(self) -> None:
        self._progress_state: Dict[str, tuple[int, float, float]] = {}

    def info(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def start_stage(self, label: str, detail: str = "") -> float:
        suffix = f" ({detail})" if detail else ""
        self.info(f"{label} started{suffix}")
        return time.monotonic()

    def finish_stage(self, label: str, started_at: float, detail: str = "") -> None:
        elapsed = time.monotonic() - started_at
        suffix = f" - {detail}" if detail else ""
        self.info(f"{label} finished in {elapsed:.1f}s{suffix}")

    def make_progress_callback(
        self,
        stage_key: str,
        *,
        noun: str,
        every: int = 25,
        min_interval_seconds: float = 5.0,
    ) -> Callable[[int, int, str], None]:
        def _callback(completed: int, total: int, message: str) -> None:
            now = time.monotonic()
            last_completed, last_print_at, started_at = self._progress_state.get(
                stage_key,
                (0, 0.0, now),
            )
            if stage_key not in self._progress_state:
                started_at = now

            should_print = False
            if total <= 0 or completed <= 0:
                should_print = True
            elif completed == 1 or completed == total:
                should_print = True
            elif completed - last_completed >= max(1, every):
                should_print = True
            elif now - last_print_at >= min_interval_seconds:
                should_print = True

            self._progress_state[stage_key] = (completed, last_print_at, started_at)
            if not should_print:
                return

            elapsed = max(0.001, now - started_at)
            rate = completed / elapsed if completed else 0.0
            remaining = max(total - completed, 0)
            eta_seconds = (remaining / rate) if rate > 0 else None
            eta_text = f", eta={eta_seconds:.1f}s" if eta_seconds is not None else ""
            message_suffix = f" - {message}" if message else ""
            self.info(
                f"{stage_key}: {completed}/{total} {noun} ({rate:.2f}/s{eta_text}){message_suffix}"
            )
            self._progress_state[stage_key] = (completed, now, started_at)

        return _callback


def run_phase3_build(
    *,
    segments_root: Path = DEFAULT_SEGMENTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    users: Optional[Sequence[str]] = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    action_match_threshold: float = DEFAULT_ACTION_MATCH_THRESHOLD,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    resume: bool = False,
    start_from: str = "load",
) -> Dict[str, Any]:
    """Run the full Phase 3 build pipeline and persist the outputs."""

    normalized_segments_root = Path(segments_root)
    normalized_output_dir = Path(output_dir)
    normalized_users = list(users) if users else None
    reporter = ProgressReporter()
    resolved_start_stage = _resolve_start_stage(
        output_dir=normalized_output_dir,
        resume=resume,
        start_from=start_from,
    )

    config = {
        "segments_root": str(normalized_segments_root),
        "output_dir": str(normalized_output_dir),
        "users": normalized_users or [],
        "similarity_threshold": similarity_threshold,
        "action_match_threshold": action_match_threshold,
        "embedding_model": embedding_model,
    }
    reporter.info(
        "Phase 3 build configuration: "
        f"start_from={resolved_start_stage}, resume={resume}, output_dir={normalized_output_dir}"
    )

    units: list[ParameterizedUnitRecord]
    clusters: list[ClusterAssignment]
    canonical_units: list[CanonicalUnitRecord]
    instance_to_cu: Dict[str, str]
    transitions: Dict[str, Dict[str, int]]
    faiss_index: object | None
    faiss_to_cu: Dict[str, str]
    faiss_warnings: list[str]

    if _stage_index(resolved_start_stage) <= _stage_index("load"):
        stage_started = reporter.start_stage("Stage 1/6: load units")
        units = load_parameterized_units(
            normalized_segments_root,
            users=normalized_users,
            progress_callback=reporter.make_progress_callback(
                "load",
                noun="segments",
                every=10,
                min_interval_seconds=2.0,
            ),
        )
        _save_checkpoint(
            normalized_output_dir,
            "load",
            {
                "config": config,
                "units": [unit.to_dict() for unit in units],
            },
        )
        reporter.finish_stage(
            "Stage 1/6: load units",
            stage_started,
            detail=f"loaded {len(units)} units",
        )
    else:
        units_payload = _load_checkpoint(normalized_output_dir, "load")
        units = _deserialize_units(units_payload.get("units", []))
        reporter.info(f"Resumed units from checkpoint: {len(units)} loaded")

    if _stage_index(resolved_start_stage) <= _stage_index("cluster"):
        stage_started = reporter.start_stage("Stage 2/6: cluster units")
        clusters = cluster_parameterized_units(
            units,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            cache_root=normalized_segments_root,
            progress_callback=reporter.make_progress_callback(
                "cluster",
                noun="groups",
                every=1,
                min_interval_seconds=1.0,
            ),
        )
        _save_checkpoint(
            normalized_output_dir,
            "cluster",
            {
                "config": config,
                "clusters": [cluster.to_dict() for cluster in clusters],
            },
        )
        reporter.finish_stage(
            "Stage 2/6: cluster units",
            stage_started,
            detail=f"built {len(clusters)} clusters",
        )
    else:
        clusters_payload = _load_checkpoint(normalized_output_dir, "cluster")
        clusters = _deserialize_clusters(clusters_payload.get("clusters", []))
        reporter.info(f"Resumed clusters from checkpoint: {len(clusters)} loaded")

    if _stage_index(resolved_start_stage) <= _stage_index("build-cu"):
        stage_started = reporter.start_stage("Stage 3/6: build canonical units")
        canonical_units, instance_to_cu = build_canonical_units(
            clusters,
            action_match_threshold=action_match_threshold,
            embedding_model=embedding_model,
            progress_callback=reporter.make_progress_callback(
                "build-cu",
                noun="clusters",
                every=10,
                min_interval_seconds=2.0,
            ),
        )
        _save_checkpoint(
            normalized_output_dir,
            "build-cu",
            {
                "config": config,
                "canonical_units": [unit.to_dict() for unit in canonical_units],
                "instance_to_cu": instance_to_cu,
            },
        )
        reporter.finish_stage(
            "Stage 3/6: build canonical units",
            stage_started,
            detail=f"built {len(canonical_units)} canonical units",
        )
    else:
        canonical_payload = _load_checkpoint(normalized_output_dir, "build-cu")
        canonical_units = _deserialize_canonical_units(canonical_payload.get("canonical_units", []))
        instance_to_cu = {
            str(key): str(value)
            for key, value in dict(canonical_payload.get("instance_to_cu") or {}).items()
        }
        reporter.info(
            "Resumed canonical units from checkpoint: "
            f"{len(canonical_units)} units, {len(instance_to_cu)} mappings"
        )

    if _stage_index(resolved_start_stage) <= _stage_index("transitions"):
        stage_started = reporter.start_stage("Stage 4/6: build transitions")
        transitions = build_transition_table(units, instance_to_cu)
        _save_checkpoint(
            normalized_output_dir,
            "transitions",
            {
                "config": config,
                "transitions": transitions,
            },
        )
        reporter.finish_stage(
            "Stage 4/6: build transitions",
            stage_started,
            detail=f"built {len(transitions)} transition rows",
        )
    else:
        transitions_payload = _load_checkpoint(normalized_output_dir, "transitions")
        transitions = {
            str(key): {
                str(inner_key): int(inner_value)
                for inner_key, inner_value in dict(inner_payload or {}).items()
            }
            for key, inner_payload in dict(transitions_payload.get("transitions") or {}).items()
        }
        reporter.info(f"Resumed transitions from checkpoint: {len(transitions)} rows")

    stage_started = reporter.start_stage("Stage 5/6: build FAISS artifacts")
    faiss_index, faiss_to_cu, faiss_warnings = build_faiss_artifacts(
        canonical_units,
        embedding_model=embedding_model,
        progress_callback=reporter.make_progress_callback(
            "faiss",
            noun="canonical_units",
            every=25,
            min_interval_seconds=3.0,
        ),
    )
    reporter.finish_stage(
        "Stage 5/6: build FAISS artifacts",
        stage_started,
        detail=f"indexed {len(faiss_to_cu)} canonical units",
    )

    metadata = {
        "warnings": faiss_warnings,
        "source_segment_count": len(
            {
                (unit.source_user, unit.trace_id or "", unit.segment_id)
                for unit in units
            }
        ),
    }
    stage_started = reporter.start_stage("Stage 6/6: persist artifacts")
    written_paths = persist_phase3_artifacts(
        normalized_output_dir,
        canonical_units=canonical_units,
        transitions=transitions,
        instance_to_cu=instance_to_cu,
        faiss_to_cu=faiss_to_cu,
        config=config,
        metadata=metadata,
        faiss_index=faiss_index,
    )
    reporter.finish_stage(
        "Stage 6/6: persist artifacts",
        stage_started,
        detail=", ".join(f"{name}={path.name}" for name, path in written_paths.items()),
    )

    return {
        "units": units,
        "clusters": clusters,
        "canonical_units": canonical_units,
        "instance_to_cu": instance_to_cu,
        "transitions": transitions,
        "faiss_to_cu": faiss_to_cu,
        "warnings": faiss_warnings,
        "written_paths": written_paths,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Phase 3 builder."""

    parser = argparse.ArgumentParser(description="Build Canonical Units from Phase 2 segments.")
    parser.add_argument(
        "--segments-root",
        type=Path,
        default=DEFAULT_SEGMENTS_DIR,
        help="Root directory containing Phase 2 segment JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Phase 3 artifacts will be written.",
    )
    parser.add_argument(
        "--users",
        nargs="*",
        default=None,
        help="Optional list of users to include, such as user_1 user_2.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Cluster similarity threshold.",
    )
    parser.add_argument(
        "--action-match-threshold",
        type=float,
        default=DEFAULT_ACTION_MATCH_THRESHOLD,
        help="Semantic similarity threshold used when merging action nodes into the Unit Tree.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name used for clustering and FAISS build.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest available checkpoint under the output directory.",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["auto", *STAGE_SEQUENCE],
        default="auto",
        help="Stage to start from. Use 'auto' with --resume to continue from the latest checkpoint.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entrypoint for running the full Phase 3 builder."""

    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_phase3_build(
        segments_root=args.segments_root,
        output_dir=args.output_dir,
        users=args.users,
        similarity_threshold=float(args.similarity_threshold),
        action_match_threshold=float(args.action_match_threshold),
        embedding_model=str(args.embedding_model),
        resume=bool(args.resume),
        start_from=str(args.start_from),
    )

    print(
        "Phase 3 build complete:",
        f"units={len(result['units'])},",
        f"clusters={len(result['clusters'])},",
        f"canonical_units={len(result['canonical_units'])},",
        f"output_dir={args.output_dir}",
    )
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    return result


def _checkpoint_dir(output_dir: Path) -> Path:
    return Path(output_dir) / CHECKPOINT_DIRNAME


def _checkpoint_path(output_dir: Path, stage_name: str) -> Path:
    file_stem = CHECKPOINT_STAGE_TO_FILE[stage_name]
    return _checkpoint_dir(output_dir) / f"{file_stem}.json"


def _save_checkpoint(output_dir: Path, stage_name: str, payload: Dict[str, Any]) -> Path:
    checkpoint_dir = _checkpoint_dir(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(output_dir, stage_name)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def _load_checkpoint(output_dir: Path, stage_name: str) -> Dict[str, Any]:
    checkpoint_path = _checkpoint_path(output_dir, stage_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for stage '{stage_name}' was not found: {checkpoint_path}"
        )
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def _deserialize_units(payload: Iterable[Dict[str, Any]]) -> list[ParameterizedUnitRecord]:
    return [
        ParameterizedUnitRecord.from_dict(unit_payload)
        for unit_payload in payload
        if isinstance(unit_payload, dict)
    ]


def _deserialize_clusters(payload: Iterable[Dict[str, Any]]) -> list[ClusterAssignment]:
    return [
        ClusterAssignment.from_dict(cluster_payload)
        for cluster_payload in payload
        if isinstance(cluster_payload, dict)
    ]


def _deserialize_canonical_units(payload: Iterable[Dict[str, Any]]) -> list[CanonicalUnitRecord]:
    return [
        CanonicalUnitRecord.from_dict(unit_payload)
        for unit_payload in payload
        if isinstance(unit_payload, dict)
    ]


def _stage_index(stage_name: str) -> int:
    return STAGE_SEQUENCE.index(stage_name)


def _resolve_start_stage(*, output_dir: Path, resume: bool, start_from: str) -> str:
    if start_from != "auto":
        _ensure_stage_requirements(output_dir, start_from)
        return start_from
    if not resume:
        return "load"

    available_stages = [
        stage_name
        for stage_name in CHECKPOINT_STAGE_TO_FILE
        if _checkpoint_path(output_dir, stage_name).exists()
    ]
    if not available_stages:
        return "load"

    latest_stage = max(available_stages, key=_stage_index)
    next_index = _stage_index(latest_stage) + 1
    if next_index >= len(STAGE_SEQUENCE):
        return "persist"
    return STAGE_SEQUENCE[next_index]


def _ensure_stage_requirements(output_dir: Path, start_from: str) -> None:
    if start_from == "load":
        return

    required_stages: list[str] = []
    if start_from == "cluster":
        required_stages = ["load"]
    elif start_from == "build-cu":
        required_stages = ["cluster"]
    elif start_from == "transitions":
        required_stages = ["load", "build-cu"]
    elif start_from in {"faiss", "persist"}:
        required_stages = ["load", "build-cu", "transitions"]

    missing = [
        stage_name
        for stage_name in required_stages
        if not _checkpoint_path(output_dir, stage_name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot resume from "
            f"'{start_from}' because the following checkpoints are missing: {', '.join(missing)}"
        )


__all__ = [
    "DEFAULT_SEGMENTS_DIR",
    "DEFAULT_OUTPUT_DIR",
    "build_arg_parser",
    "main",
    "run_phase3_build",
]


if __name__ == "__main__":
    main()
