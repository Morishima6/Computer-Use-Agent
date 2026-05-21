"""Load Phase 2 parameterized units from segment artifacts."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .schemas import ParameterizedAction, ParameterizedUnitRecord


SEGMENT_FILE_PATTERN = "seg_*.json"
SEGMENT_ORDER_RE = re.compile(r"seg_(\d+)\.json$", re.IGNORECASE)
UNIT_SUFFIX_RE = re.compile(r"_(\d+)$")


def discover_segment_files(
    segments_root: Path,
    users: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Return segment files under the given root, optionally filtered by user."""

    root = Path(segments_root)
    allowed_users = set(users or [])
    segment_files: List[Path] = []

    for path in root.rglob(SEGMENT_FILE_PATTERN):
        if not path.is_file():
            continue
        source_user = _infer_source_user(root, path)
        if allowed_users and source_user not in allowed_users:
            continue
        segment_files.append(path)

    return sorted(
        segment_files,
        key=lambda path: (
            _infer_source_user(root, path),
            _infer_trace_id(root, path),
            _parse_segment_order(path),
            str(path),
        ),
    )


def load_parameterized_units(
    segments_root: Path,
    users: Optional[Sequence[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[ParameterizedUnitRecord]:
    """Load and normalize all phase2 `parameterized units` records."""

    root = Path(segments_root)
    records: List[ParameterizedUnitRecord] = []
    segment_files = discover_segment_files(root, users=users)
    total_segments = len(segment_files)

    for index, segment_path in enumerate(segment_files, start=1):
        records.extend(load_segment_parameterized_units(segment_path, segments_root=root))
        if progress_callback is not None:
            progress_callback(index, total_segments, str(segment_path))

    return sorted(
        records,
        key=lambda record: (
            record.source_user,
            record.trace_id or "",
            record.segment_order,
            record.step_indices[0] if record.step_indices else 0,
            record.instance_id,
        ),
    )


def load_segment_parameterized_units(
    segment_path: Path,
    *,
    segments_root: Optional[Path] = None,
) -> List[ParameterizedUnitRecord]:
    """Load normalized phase2 units from one segment file."""

    segment_file = Path(segment_path)
    root = Path(segments_root) if segments_root is not None else segment_file.parents[1]
    payload = json.loads(segment_file.read_text(encoding="utf-8"))

    source_user = _infer_source_user(root, segment_file)
    trace_id = _infer_trace_id(root, segment_file)
    segment_id = str(payload.get("segment_id") or segment_file.stem)
    segment_order = _parse_segment_order(segment_file, fallback_segment_id=segment_id)
    app_name = str(payload.get("app") or "UNKNOWN")
    env = dict(payload.get("env") or {})
    app_context = _build_app_context(app_name=app_name, env=env)
    step_map = _build_step_map(payload.get("steps", []))

    raw_units = payload.get("parameterized units", [])
    if not isinstance(raw_units, list):
        return []

    records: List[ParameterizedUnitRecord] = []
    for unit_payload in raw_units:
        if not isinstance(unit_payload, dict):
            continue
        if str(unit_payload.get("phase2_status", "")).lower() != "done":
            continue

        step_indices = _normalize_step_indices(unit_payload.get("step_indices", []))
        timestamp_start, timestamp_end = _resolve_unit_time_range(step_indices, step_map)

        actions = [
            _normalize_parameterized_action(action_payload)
            for action_payload in unit_payload.get("parameterized_action_sequence", [])
            if isinstance(action_payload, dict)
        ]

        unit_id = str(unit_payload.get("unit_id") or "")
        param_unit_id = str(unit_payload.get("param_unit_id") or unit_id)
        instance_id = _build_instance_id(
            source_user=source_user,
            trace_id=trace_id,
            segment_id=segment_id,
            raw_unit_id=unit_id,
        )
        record = ParameterizedUnitRecord(
            instance_id=instance_id,
            param_unit_id=param_unit_id,
            unit_id=unit_id,
            segment_id=segment_id,
            source_user=source_user,
            app_name=app_name,
            unit_type=str(unit_payload.get("unit_type") or "UNKNOWN"),
            abstract_intent=str(unit_payload.get("abstract_intent") or ""),
            unit_intent=str(unit_payload.get("unit_intent") or ""),
            unit_before_state=str(unit_payload.get("unit_before_state") or ""),
            unit_after_state=str(unit_payload.get("unit_after_state") or ""),
            unit_precondition=_normalize_string_list(unit_payload.get("unit_precondition", [])),
            unit_effect=_normalize_string_list(unit_payload.get("unit_effect", [])),
            parameters=dict(unit_payload.get("parameters") or {}),
            parameterized_action_sequence=actions,
            step_indices=step_indices,
            segment_order=segment_order,
            trace_id=trace_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            app_context=app_context,
            env=env,
            raw_segment_path=str(segment_file),
            raw_payload=unit_payload,
        )
        records.append(record)

    return records


def _build_step_map(steps: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    step_map: Dict[int, Dict[str, Any]] = {}
    for index, step_payload in enumerate(steps, start=1):
        if not isinstance(step_payload, dict):
            continue
        step_id = str(step_payload.get("step_id") or "")
        numeric_index = _parse_step_id(step_id) or index
        step_map[numeric_index] = step_payload
    return step_map


def _parse_step_id(step_id: str) -> Optional[int]:
    match = re.search(r"(\d+)$", str(step_id))
    return int(match.group(1)) if match else None


def _normalize_step_indices(raw_step_indices: Any) -> List[int]:
    if not isinstance(raw_step_indices, list):
        return []
    normalized = []
    for value in raw_step_indices:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(normalized)


def _resolve_unit_time_range(
    step_indices: Sequence[int],
    step_map: Dict[int, Dict[str, Any]],
) -> tuple[Optional[str], Optional[str]]:
    if not step_indices:
        return None, None

    first_step = step_map.get(min(step_indices), {})
    last_step = step_map.get(max(step_indices), {})
    first_state = dict(first_step.get("now_state") or {})
    last_state = dict(last_step.get("now_state") or {})

    timestamp_start = _first_non_empty(
        first_state.get("screenshot_time_before"),
        first_state.get("screeenshot_time_before"),
        first_state.get("screenshot_time_after"),
        first_state.get("screeenshot_time_after"),
    )
    timestamp_end = _first_non_empty(
        last_state.get("screeenshot_time_after"),
        last_state.get("screenshot_time_after"),
        last_state.get("screenshot_time_before"),
        last_state.get("screeenshot_time_before"),
    )
    return timestamp_start, timestamp_end


def _normalize_parameterized_action(action_payload: Dict[str, Any]) -> ParameterizedAction:
    raw_action = action_payload.get("action")
    return ParameterizedAction(
        action_type=_infer_action_type(raw_action),
        action_template=str(action_payload.get("action_template") or ""),
        raw_action=raw_action,
    )


def _infer_action_type(raw_action: Any) -> str:
    if isinstance(raw_action, dict):
        action_type = raw_action.get("type")
        return str(action_type or "UNKNOWN").upper()

    if isinstance(raw_action, str):
        stripped = raw_action.strip()
        if not stripped:
            return "UNKNOWN"
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                return str(parsed.get("type") or "UNKNOWN").upper()
        return stripped.upper()

    return "UNKNOWN"


def _normalize_string_list(raw_values: Any) -> List[str]:
    if not isinstance(raw_values, list):
        return []
    return [str(value) for value in raw_values if str(value).strip()]


def _build_app_context(app_name: str, env: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "app_name": app_name,
        "os": env.get("os", ""),
        "screen": env.get("screen", ""),
        "url": env.get("url", ""),
        "locale": env.get("locale", ""),
    }


def _infer_source_user(segments_root: Path, segment_path: Path) -> str:
    relative_parts = segment_path.relative_to(segments_root).parts
    return relative_parts[0] if relative_parts else "unknown_user"


def _infer_trace_id(segments_root: Path, segment_path: Path) -> str:
    relative_parts = segment_path.relative_to(segments_root).parts
    if len(relative_parts) <= 2:
        return relative_parts[0] if relative_parts else "default_trace"
    return "/".join(relative_parts[1:-1])


def _parse_segment_order(segment_path: Path, fallback_segment_id: str = "") -> int:
    match = SEGMENT_ORDER_RE.search(segment_path.name)
    if match:
        return int(match.group(1))

    fallback_match = re.search(r"(\d+)$", fallback_segment_id)
    if fallback_match:
        return int(fallback_match.group(1))

    return 0


def _build_instance_id(
    *,
    source_user: str,
    trace_id: str,
    segment_id: str,
    raw_unit_id: str,
) -> str:
    user_part = _normalize_id_part(source_user, fallback="unknown_user")
    trace_part = _normalize_id_part(trace_id, fallback=user_part)
    segment_part = _normalize_segment_id(segment_id)
    unit_suffix = _extract_unit_suffix(raw_unit_id)
    return f"u_{user_part}_{trace_part}_{segment_part}_{unit_suffix}"


def _normalize_id_part(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip())
    normalized = normalized.strip("_").lower()
    return normalized or fallback


def _normalize_segment_id(segment_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(segment_id or "").strip())
    normalized = normalized.strip("_").lower()
    if normalized:
        return normalized
    return "seg_000"


def _extract_unit_suffix(raw_unit_id: str) -> str:
    match = UNIT_SUFFIX_RE.search(str(raw_unit_id or ""))
    if match:
        return match.group(1)
    return "00"


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


__all__ = [
    "discover_segment_files",
    "load_parameterized_units",
    "load_segment_parameterized_units",
]
