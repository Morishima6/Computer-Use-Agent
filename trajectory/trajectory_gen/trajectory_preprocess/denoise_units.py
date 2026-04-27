from __future__ import annotations

import argparse
import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils.vlm_similarity_gate import judge_multimodal_json_with_meta


ENV_PATH = Path(".env")
DEFAULT_UNIT_OUTPUT_DIRNAME = "segments_units_denoised"
DEFAULT_AUDIT_FILENAME = "segments_units_denoise_audit.json"
DEFAULT_VLM_BACKEND = "qwen"
DEFAULT_VLM_VERIFICATION_REPEATS = 3
DEFAULT_HTTP_TRUST_ENV = False

UNIT_SYSTEM_PROMPT = """You are an GUI action judgement. You are dealing with one already-segmented GUI interaction unit.

Your task is to decide which specific steps inside this unit are accidental, redundant or performed aimlessly based the given screenshots and choose them.

You will be given:
    - The full JSON content of one extracted unit payload.
    - The screenshots referenced by that unit.

For eash step in the unit, you should inspect its:
    - action type
    - action before screenshot: the state of screen before this action
    - action after screenshot: the state of screen after 1s of this action 
    - the next action before screenshot (if exists): the state of screen before the next action (maybe after 2-10s of the current action)

Note: 
    - If there is a red marker in the before screenshot, it is only used to indicate the mouse position; please IGNORE the red marker.
    - Do not judge only by similarity. You know, sometimes selecting a toolbar button may produce similar before and after frames, but this does not mean that the action is aimless. At the same time, we must not overlook genuine redundancy, such as clicking in a blank space or an invalid input.
    - Not only look at the before and after screenshots of this action, pay attention to the next action before screenshot. Because sometimes an action doesn't manifest within 1 second, but may change over 2-10 seconds.

If every step in the unit should be chosen, you may set chosen_all=true.

Choosing step Principle:
    - Choose when this action is accidental, redundant, failed or performed aimlessly. 
    - Don't choose it when this action is meaningful, sucessful or purposeful.

Output Formate (JSON):
{
  "chosen_all": false,
  "chosen_step_ids": ["s3", "s4"],
  "reason": "short explanation",
  "step_reasons": {
    "s3": "short reason",
    "s4": "short reason"
  }
}

IMPORTANT: Your output must be in ENGLISH.
"""

SCREENSHOT_PATH_KEYS = (
    "screenshot_path_before",
    "screenshot_path_after",
    "screenshot_path_before_part",
)


def parse_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ENV_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise unit annotations under segments_units by sending one unit and its screenshots to a VLM."
    )
    parser.add_argument(
        "input_path",
        help="A trajectory result directory, a segments_units directory, or a single segment JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for denoised segment files. Defaults to <segments_units_dir>/../segments_units_denoised.",
    )
    parser.add_argument(
        "--output-audit",
        default=None,
        help="Audit JSON path. Defaults to <segments_units_dir>/../segments_units_denoise_audit.json.",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=["none", "auto", "qwen", "kimi", "ark", "gpt"],
        default=DEFAULT_VLM_BACKEND,
        help="Model backend used for per-unit denoising.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    key: List[Any] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log_with_timestamp(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def iter_segment_files(segments_units_dir: Path) -> Iterable[Path]:
    for path in sorted(segments_units_dir.glob("*.json"), key=natural_sort_key):
        if path.is_file():
            yield path


def resolve_segments_units_dir(input_path: str) -> Tuple[Path, List[Path]]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Input file is not a JSON segment file: {path}")
        return path.parent.resolve(), [path.resolve()]

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir() and path.name == "segments_units":
        segments_units_dir = path.resolve()
    elif path.is_dir() and (path / "segments_units").is_dir():
        segments_units_dir = (path / "segments_units").resolve()
    elif path.is_dir() and (path / "result" / "segments_units").is_dir():
        segments_units_dir = (path / "result" / "segments_units").resolve()
    else:
        raise FileNotFoundError(
            f"Could not find a segments_units directory under: {path}. "
            "Expected input to be a segment JSON, a segments_units dir, or a result dir containing segments_units."
        )

    segment_files = list(iter_segment_files(segments_units_dir))
    if not segment_files:
        raise FileNotFoundError(f"No segment JSON files found under: {segments_units_dir}")
    return segments_units_dir, segment_files


def resolve_output_paths(
    segments_units_dir: Path,
    output_dir_arg: Optional[str],
    output_audit_arg: Optional[str],
) -> Tuple[Path, Path]:
    base_dir = segments_units_dir.parent
    output_dir = Path(output_dir_arg).resolve() if output_dir_arg else (base_dir / DEFAULT_UNIT_OUTPUT_DIRNAME).resolve()
    output_audit = Path(output_audit_arg).resolve() if output_audit_arg else (base_dir / DEFAULT_AUDIT_FILENAME).resolve()
    return output_dir, output_audit


def resolve_screenshot_path(segment_file: Path, rel_path: str) -> Path:
    rel = Path(rel_path)
    candidate_parents: List[Path] = []
    current = segment_file.parent
    for _ in range(6):
        if current in candidate_parents:
            break
        candidate_parents.append(current)
        if current.parent == current:
            break
        current = current.parent

    for parent in candidate_parents:
        candidate = (parent / rel).resolve()
        if candidate.exists():
            return candidate
    if rel.exists():
        return rel.resolve()
    raise FileNotFoundError(f"Cannot resolve screenshot path: {rel_path}")


def normalize_step_indices(raw_step_indices: Any) -> List[int]:
    normalized: List[int] = []
    if not isinstance(raw_step_indices, list):
        return normalized
    for item in raw_step_indices:
        if isinstance(item, int):
            normalized.append(item)
            continue
        stripped = str(item).strip().lower()
        if stripped.startswith("s") and stripped[1:].isdigit():
            normalized.append(int(stripped[1:]))
        elif stripped.isdigit():
            normalized.append(int(stripped))
    return normalized


def extract_unit_steps(segment_payload: Dict[str, Any], unit_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_steps = segment_payload.get("steps") or []
    if not isinstance(all_steps, list):
        return []

    unit_steps: List[Dict[str, Any]] = []
    for step_index in normalize_step_indices(unit_payload.get("step_indices")):
        if 1 <= step_index <= len(all_steps):
            raw_step = all_steps[step_index - 1]
            if isinstance(raw_step, dict):
                unit_steps.append(copy.deepcopy(raw_step))
    return unit_steps


def build_unit_prompt_payload(segment_payload: Dict[str, Any], unit_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "segment_id": segment_payload.get("segment_id"),
        "app": segment_payload.get("app"),
        "env": copy.deepcopy(segment_payload.get("env") or {}),
        "unit": copy.deepcopy(unit_payload),
        "unit_steps": extract_unit_steps(segment_payload, unit_payload),
    }


def get_prompt_steps(unit_prompt_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_steps = unit_prompt_payload.get("unit_steps") or unit_prompt_payload.get("steps") or []
    return raw_steps if isinstance(raw_steps, list) else []


def build_unit_source_name(segment_file: Path, unit_payload: Dict[str, Any]) -> str:
    unit_id = str(unit_payload.get("unit_id", "")).strip() or "unknown_unit"
    return f"{segment_file.name}::{unit_id}"


def collect_unit_screenshots(unit_prompt_payload: Dict[str, Any], segment_file: Path) -> List[Dict[str, Any]]:
    screenshot_items: List[Dict[str, Any]] = []
    seen_paths: Set[Path] = set()
    for step in get_prompt_steps(unit_prompt_payload):
        step_id = str(step.get("step_id", "")).strip() or "unknown_step"
        now_state = step.get("now_state", {})
        for field_name in SCREENSHOT_PATH_KEYS:
            rel_path = now_state.get(field_name)
            if not rel_path:
                continue
            resolved = resolve_screenshot_path(segment_file, str(rel_path))
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            screenshot_items.append(
                {
                    "step_id": step_id,
                    "field": field_name,
                    "path": resolved,
                    "rel_path": str(rel_path),
                }
            )
    return screenshot_items


def build_unit_text_prompt(unit_payload: Dict[str, Any], source_name: str) -> str:
    return (
        f"Source unit: {source_name}\n\n"
        "Decide which step_ids inside this unit are judged accidental, redundant or performed aimlessly.\n"
        "Use both the unit JSON and the provided screenshots.\n"
        "Return JSON only.\n\n"
        f"Unit JSON:\n{json.dumps(unit_payload, ensure_ascii=False, indent=2)}"
    )


def build_unit_screenshot_inputs(screenshots: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "path": item["path"],
            "label": f"Screenshot for step {item['step_id']} field {item['field']}: {item['rel_path']}",
        }
        for item in screenshots
    ]


def judge_unit_payload(
    unit_payload: Dict[str, Any],
    segment_file: Path,
    source_name: str,
    backend: str,
    *,
    trust_env: bool,
) -> Tuple[Dict[str, Any], str, str, int]:
    screenshots = collect_unit_screenshots(unit_payload, segment_file)
    try:
        parsed, provider, model = judge_multimodal_json_with_meta(
            system_prompt_text=UNIT_SYSTEM_PROMPT,
            prompt_text=build_unit_text_prompt(unit_payload, source_name),
            screenshots=build_unit_screenshot_inputs(screenshots),
            backend=backend,
            http_trust_env=trust_env,
            max_tokens=1024,
            compact_images=True,
            openai_use_system_message=True,
            provider_request_kwargs={
                "ark": {"temperature": 0},
                "qwen": {"temperature": 0},
                "gpt": {"temperature": 0},
            },
        )
        return parsed, provider, model, len(screenshots)
    except Exception as exc:
        log_with_timestamp(f"unit={source_name} backend={backend} failed: {exc}")
        raise


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_step_reasons(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in raw.items():
        step_id = str(key).strip()
        if not step_id:
            continue
        normalized[step_id] = str(value).strip()
    return normalized


def normalize_choice_payload(raw_payload: Dict[str, Any], step_ids: Sequence[str]) -> Tuple[List[str], str, Dict[str, str], bool]:
    valid_step_ids = [str(step_id).strip() for step_id in step_ids if str(step_id).strip()]
    valid_step_id_set = set(valid_step_ids)

    raw_chosen_ids = raw_payload.get("chosen_step_ids")
    if raw_chosen_ids is None:
        raw_chosen_ids = raw_payload.get("drop_step_ids")
    if raw_chosen_ids is None:
        raw_chosen_ids = raw_payload.get("step_ids_to_delete")
    if raw_chosen_ids is None:
        raw_chosen_ids = raw_payload.get("delete_step_ids")

    normalized_ids: List[str] = []
    if isinstance(raw_chosen_ids, list):
        for item in raw_chosen_ids:
            step_id = str(item).strip()
            if step_id in valid_step_id_set and step_id not in normalized_ids:
                normalized_ids.append(step_id)

    chosen_all = (
        coerce_bool(raw_payload.get("chosen_all", False))
        or coerce_bool(raw_payload.get("choose_all", False))
        or coerce_bool(raw_payload.get("drop_all", False))
        or coerce_bool(raw_payload.get("delete_all", False))
        or coerce_bool(raw_payload.get("drop_unit", False))
    )
    if chosen_all:
        normalized_ids = list(valid_step_ids)

    reason = str(raw_payload.get("reason", raw_payload.get("Reason", ""))).strip()
    step_reasons = normalize_step_reasons(raw_payload.get("step_reasons"))
    if not reason:
        reason = "No reason provided by the model."
    return normalized_ids, reason, step_reasons, chosen_all


def get_unit_resume_key(unit_payload: Dict[str, Any], unit_index: int) -> str:
    unit_id = str(unit_payload.get("unit_id", "")).strip()
    return unit_id or f"__index_{unit_index}"


def get_record_resume_key(unit_record: Dict[str, Any]) -> str:
    unit_id = str(unit_record.get("unit_id", "")).strip()
    if unit_id:
        return unit_id
    unit_index = unit_record.get("unit_index")
    return f"__index_{unit_index}"


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_existing_audit_state(audit_path: Path) -> Dict[str, Any]:
    if not audit_path.is_file():
        return {"segments": []}
    raw = load_json(audit_path)
    if not isinstance(raw, dict):
        return {"segments": []}

    normalized_segments: List[Dict[str, Any]] = []
    for raw_segment in raw.get("segments", []):
        if not isinstance(raw_segment, dict):
            continue
        segment_record = copy.deepcopy(raw_segment)
        units = [unit for unit in segment_record.get("units", []) if isinstance(unit, dict)]
        units.sort(key=lambda item: coerce_int(item.get("unit_index"), -1))
        segment_record["units"] = units
        segment_record["original_unit_count"] = max(
            coerce_int(segment_record.get("original_unit_count"), len(units)),
            len(units),
        )
        segment_record["processed_unit_count"] = len(units)
        status = str(segment_record.get("status", "")).strip().lower()
        if status not in {"in_progress", "completed"}:
            status = "completed" if len(units) >= segment_record["original_unit_count"] else "in_progress"
        segment_record["status"] = status
        normalized_segments.append(segment_record)

    return {"segments": normalized_segments}


def find_segment_record(
    audit_state: Dict[str, Any],
    segment_file: Path,
    segment_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    source_path = str(segment_file.resolve()).strip().lower()
    fallback_match: Optional[Dict[str, Any]] = None
    for segment_record in audit_state.get("segments", []):
        record_source_path = str(segment_record.get("source_path", "")).strip().lower()
        if record_source_path == source_path:
            return segment_record
        if (
            str(segment_record.get("file_name", "")).strip() == segment_file.name
            and (segment_id is None or str(segment_record.get("segment_id", "")).strip() == segment_id)
        ):
            fallback_match = segment_record
    return fallback_match


def ensure_segment_record(
    audit_state: Dict[str, Any],
    *,
    index: int,
    segment_file: Path,
    segment_payload: Dict[str, Any],
    unit_count: int,
    step_count: int,
) -> Dict[str, Any]:
    existing = find_segment_record(
        audit_state,
        segment_file,
        segment_id=str(segment_payload.get("segment_id", "")).strip() or None,
    )
    if existing is not None:
        existing["index"] = index
        existing["file_name"] = segment_file.name
        existing["source_path"] = str(segment_file.resolve())
        existing["segment_id"] = segment_payload.get("segment_id")
        existing["original_step_count"] = step_count
        existing["original_unit_count"] = max(coerce_int(existing.get("original_unit_count"), unit_count), unit_count)
        existing["processed_unit_count"] = len([unit for unit in existing.get("units", []) if isinstance(unit, dict)])
        existing.setdefault("units", [])
        existing.setdefault("status", "in_progress")
        return existing

    segment_record = {
        "index": index,
        "file_name": segment_file.name,
        "source_path": str(segment_file.resolve()),
        "segment_id": segment_payload.get("segment_id"),
        "original_step_count": step_count,
        "kept_step_count": 0,
        "removed_step_count": 0,
        "original_unit_count": unit_count,
        "processed_unit_count": 0,
        "kept_unit_count": 0,
        "removed_unit_count": 0,
        "status": "in_progress",
        "units": [],
    }
    audit_state.setdefault("segments", []).append(segment_record)
    return segment_record


def build_segment_outputs(
    segment_payload: Dict[str, Any],
    original_units: Sequence[Dict[str, Any]],
    segment_unit_records: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chosen_step_ids_in_segment: Set[str] = set()
    output_units: List[Dict[str, Any]] = []
    original_steps = segment_payload.get("steps") or []
    record_map = {
        get_record_resume_key(record): record
        for record in segment_unit_records
        if isinstance(record, dict)
    }

    for unit_index, unit_payload in enumerate(original_units):
        if not isinstance(unit_payload, dict):
            continue
        resume_key = get_unit_resume_key(unit_payload, unit_index)
        record = record_map.get(resume_key)
        if record is None:
            continue

        chosen_step_ids = list(record.get("chosen_step_ids", []))
        chosen_step_id_set = set(chosen_step_ids)
        chosen_step_ids_in_segment.update(chosen_step_id_set)
        verification = record.get("verification", {})
        if verification.get("chosen_all"):
            continue

        unit_output = copy.deepcopy(unit_payload)
        unit_output["unit_denoise_meta"] = {
            "chosen_step_ids": chosen_step_ids,
            "chosen_all": verification.get("chosen_all", False),
            "reason": verification.get("Reason", ""),
            "backend": verification.get("backend"),
            "model": verification.get("model"),
            "screenshot_count": verification.get("screenshot_count"),
            "required_repeat_count": verification.get("required_repeat_count"),
        }
        unit_output["_kept_step_ids"] = list(record.get("kept_step_ids", []))
        output_units.append(unit_output)

    output_steps = [
        copy.deepcopy(step)
        for step in original_steps
        if str(step.get("step_id", "")).strip() not in chosen_step_ids_in_segment
    ]
    step_id_to_new_index = {
        str(step.get("step_id", "")).strip(): new_index
        for new_index, step in enumerate(output_steps, start=1)
        if str(step.get("step_id", "")).strip()
    }

    finalized_units: List[Dict[str, Any]] = []
    for unit_output in output_units:
        kept_step_ids = unit_output.pop("_kept_step_ids", [])
        unit_output["step_indices"] = [
            step_id_to_new_index[step_id]
            for step_id in kept_step_ids
            if step_id in step_id_to_new_index
        ]
        finalized_units.append(unit_output)

    output_segment = copy.deepcopy(segment_payload)
    output_segment["steps"] = output_steps
    output_segment["units"] = finalized_units
    output_segment["segment_denoise_meta"] = {
        "chosen_step_ids": [
            str(step.get("step_id", "")).strip()
            for step in original_steps
            if str(step.get("step_id", "")).strip() in chosen_step_ids_in_segment
        ],
        "original_step_count": len(original_steps) if isinstance(original_steps, list) else 0,
        "kept_step_count": len(output_steps),
        "removed_step_count": len(chosen_step_ids_in_segment),
        "original_unit_count": len([unit for unit in original_units if isinstance(unit, dict)]),
        "kept_unit_count": len(finalized_units),
        "removed_unit_count": len(
            [record for record in segment_unit_records if record.get("verification", {}).get("chosen_all")]
        ),
        "vlm_backend": output_segment.get("segment_denoise_meta", {}).get("vlm_backend"),
        "unit_verification_repeats": DEFAULT_VLM_VERIFICATION_REPEATS,
    }

    segment_stats = {
        "kept_step_count": len(output_steps),
        "removed_step_count": len(chosen_step_ids_in_segment),
        "kept_unit_count": len(finalized_units),
        "removed_unit_count": len(
            [record for record in segment_unit_records if record.get("verification", {}).get("chosen_all")]
        ),
    }
    return output_segment, segment_stats


def save_audit_state(
    audit_path: Path,
    audit_state: Dict[str, Any],
    *,
    output_dir: Path,
    total_segment_count: int,
    total_original_unit_count: int,
    backend: str,
) -> Dict[str, Any]:
    kept_units: List[Dict[str, Any]] = []
    dropped_units: List[Dict[str, Any]] = []
    partially_denoised_unit_count = 0
    untouched_unit_count = 0
    total_removed_steps = 0
    completed_segment_count = 0
    in_progress_segment_count = 0

    segments = [segment for segment in audit_state.get("segments", []) if isinstance(segment, dict)]
    segments.sort(key=lambda item: coerce_int(item.get("index"), 10**9))

    for segment_record in segments:
        status = str(segment_record.get("status", "")).strip().lower()
        if status == "completed":
            completed_segment_count += 1
        else:
            in_progress_segment_count += 1

        output_path = str(segment_record.get("output_path", "")).strip()
        for unit_record in segment_record.get("units", []):
            if not isinstance(unit_record, dict):
                continue
            chosen_step_ids = list(unit_record.get("chosen_step_ids", []))
            total_removed_steps += len(chosen_step_ids)
            if chosen_step_ids:
                partially_denoised_unit_count += 1
            else:
                untouched_unit_count += 1

            flat_record = copy.deepcopy(unit_record)
            if output_path:
                flat_record["output_path"] = output_path
            if flat_record.get("verification", {}).get("chosen_all"):
                dropped_units.append(flat_record)
            else:
                kept_units.append(flat_record)

    audit_payload = {
        "summary": {
            "original_segment_count": total_segment_count,
            "kept_segment_count": completed_segment_count,
            "in_progress_segment_count": in_progress_segment_count,
            "original_unit_count": total_original_unit_count,
            "kept_unit_count": len(kept_units),
            "fully_dropped_unit_count": len(dropped_units),
            "partially_denoised_unit_count": partially_denoised_unit_count,
            "untouched_unit_count": untouched_unit_count,
            "total_removed_steps": total_removed_steps,
            "vlm_backend": backend,
            "unit_verification_repeats": DEFAULT_VLM_VERIFICATION_REPEATS,
            "output_dir": str(output_dir.resolve()),
        },
        "segments": segments,
        "kept_units": kept_units,
        "fully_dropped_units": dropped_units,
    }
    dump_json(audit_path, audit_payload)
    return audit_payload


def verify_unit_steps(
    unit_payload: Dict[str, Any],
    segment_file: Path,
    source_name: str,
    backend: str,
    *,
    trust_env: bool,
) -> Dict[str, Any]:
    all_step_ids = [str(step.get("step_id", "")).strip() for step in get_prompt_steps(unit_payload)]
    all_step_ids = [step_id for step_id in all_step_ids if step_id]

    if backend == "none":
        return {
            "chosen_step_ids": [],
            "chosen_all": False,
            "Reason": "VLM verification disabled.",
            "attempts": [],
            "backend": "none",
            "model": None,
            "screenshot_count": len(collect_unit_screenshots(unit_payload, segment_file)),
            "required_repeat_count": DEFAULT_VLM_VERIFICATION_REPEATS,
        }

    attempts: List[Dict[str, Any]] = []
    consensus_ids: Optional[Set[str]] = None
    provider_label: Optional[str] = None
    model_label: Optional[str] = None
    screenshot_count = 0

    for attempt_idx in range(DEFAULT_VLM_VERIFICATION_REPEATS):
        log_with_timestamp(
            f"verifying unit={source_name} attempt={attempt_idx + 1}/{DEFAULT_VLM_VERIFICATION_REPEATS} backend={backend}"
        )
        parsed, provider, model, screenshot_count = judge_unit_payload(
            unit_payload,
            segment_file,
            source_name,
            backend,
            trust_env=trust_env,
        )
        chosen_ids, reason, step_reasons, chosen_all = normalize_choice_payload(parsed, all_step_ids)
        attempt_chosen_set = set(chosen_ids)
        consensus_ids = attempt_chosen_set if consensus_ids is None else consensus_ids & attempt_chosen_set
        attempts.append(
            {
                "attempt": attempt_idx + 1,
                "backend": provider,
                "model": model,
                "chosen_all": chosen_all,
                "chosen_step_ids": chosen_ids,
                "reason": reason,
                "step_reasons": step_reasons,
            }
        )
        provider_label = provider
        model_label = model
        log_with_timestamp(
            f"verified unit={source_name} attempt={attempt_idx + 1}/{DEFAULT_VLM_VERIFICATION_REPEATS} provider={provider} model={model} chosen_step_ids={chosen_ids}"
        )

    final_chosen_ids = [step_id for step_id in all_step_ids if consensus_ids and step_id in consensus_ids]
    final_chosen_all = bool(final_chosen_ids) and len(final_chosen_ids) == len(all_step_ids)
    final_reason = (
        f"Chose {len(final_chosen_ids)} step(s) that all {DEFAULT_VLM_VERIFICATION_REPEATS} VLM calls agreed were accidental, redundant or aimless."
        if final_chosen_ids
        else "No step reached unanimous VLM choosing consensus."
    )
    log_with_timestamp(
        f"finished unit={source_name} final_chosen_step_ids={final_chosen_ids} chosen_all={final_chosen_all}"
    )
    return {
        "chosen_step_ids": final_chosen_ids,
        "chosen_all": final_chosen_all,
        "Reason": final_reason,
        "attempts": attempts,
        "backend": provider_label,
        "model": model_label,
        "screenshot_count": screenshot_count,
        "required_repeat_count": DEFAULT_VLM_VERIFICATION_REPEATS,
    }


def denoise_units(
    segment_files: Sequence[Path],
    output_dir: Path,
    audit_path: Path,
    *,
    backend: str,
    trust_env: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_state = load_existing_audit_state(audit_path)
    total_original_unit_count = 0
    for segment_file in segment_files:
        segment_payload = load_json(segment_file)
        raw_units = segment_payload.get("units") or []
        total_original_unit_count += len([unit for unit in raw_units if isinstance(unit, dict)])

    log_with_timestamp(
        f"start denoise segments={len(segment_files)} backend={backend} output_dir={output_dir}"
    )

    for index, segment_file in enumerate(segment_files):
        segment_payload = load_json(segment_file)
        original_steps = segment_payload.get("steps") or []
        original_units = segment_payload.get("units") or []
        segment_id = str(segment_payload.get("segment_id", "")).strip() or segment_file.stem
        unit_count = len([unit for unit in original_units if isinstance(unit, dict)]) if isinstance(original_units, list) else 0
        segment_output_path = output_dir / segment_file.name
        segment_record = ensure_segment_record(
            audit_state,
            index=index,
            segment_file=segment_file,
            segment_payload=segment_payload,
            unit_count=unit_count,
            step_count=len(original_steps) if isinstance(original_steps, list) else 0,
        )
        segment_record["status"] = "completed" if (
            len([unit for unit in segment_record.get("units", []) if isinstance(unit, dict)]) >= unit_count
            and unit_count > 0
        ) else str(segment_record.get("status", "in_progress"))

        if segment_record.get("status") == "completed" and segment_output_path.is_file():
            log_with_timestamp(
                f"skip completed segment={segment_id} file={segment_file.name} output={segment_output_path}"
            )
            continue

        log_with_timestamp(
            f"processing segment={segment_id} file={segment_file.name} index={index + 1}/{len(segment_files)} units={unit_count}"
        )
        segment_unit_records = [unit for unit in segment_record.get("units", []) if isinstance(unit, dict)]
        processed_unit_keys = {get_record_resume_key(record) for record in segment_unit_records}
        if processed_unit_keys:
            log_with_timestamp(
                f"resume state segment={segment_id} processed_units={len(segment_unit_records)}/{unit_count}"
            )

        for unit_index, unit_payload in enumerate(original_units):
            if not isinstance(unit_payload, dict):
                continue

            resume_key = get_unit_resume_key(unit_payload, unit_index)
            unit_id = str(unit_payload.get("unit_id", "")).strip() or f"{segment_id}_unit_{unit_index + 1}"
            if resume_key in processed_unit_keys:
                log_with_timestamp(
                    f"resume skip segment={segment_id} unit={unit_id} index={unit_index + 1}/{unit_count}"
                )
                continue

            unit_prompt_payload = build_unit_prompt_payload(segment_payload, unit_payload)
            source_name = build_unit_source_name(segment_file, unit_payload)
            log_with_timestamp(
                f"processing segment={segment_id} unit={unit_id} index={unit_index + 1}/{unit_count}"
            )
            verification = verify_unit_steps(
                unit_prompt_payload,
                segment_file,
                source_name,
                backend,
                trust_env=trust_env,
            )

            original_unit_steps = get_prompt_steps(unit_prompt_payload)
            original_step_ids = [
                str(step.get("step_id", "")).strip()
                for step in original_unit_steps
                if str(step.get("step_id", "")).strip()
            ]
            chosen_step_ids = list(verification["chosen_step_ids"])
            chosen_step_id_set = set(chosen_step_ids)
            kept_step_ids = [step_id for step_id in original_step_ids if step_id not in chosen_step_id_set]

            record = {
                "segment_index": index,
                "unit_index": unit_index,
                "resume_unit_key": resume_key,
                "file_name": segment_file.name,
                "source_path": str(segment_file.resolve()),
                "segment_id": segment_payload.get("segment_id"),
                "unit_id": unit_payload.get("unit_id"),
                "unit_type": unit_payload.get("unit_type"),
                "unit_intent": unit_payload.get("unit_intent"),
                "original_step_count": len(original_step_ids),
                "kept_step_count": len(kept_step_ids),
                "chosen_step_count": len(chosen_step_ids),
                "chosen_step_ids": chosen_step_ids,
                "kept_step_ids": kept_step_ids,
                "verification": verification,
            }
            segment_unit_records.append(record)
            segment_unit_records.sort(key=lambda item: coerce_int(item.get("unit_index"), 10**9))
            processed_unit_keys.add(resume_key)
            segment_record["units"] = segment_unit_records
            segment_record["processed_unit_count"] = len(segment_unit_records)
            segment_record["status"] = "in_progress"
            save_audit_state(
                audit_path,
                audit_state,
                output_dir=output_dir,
                total_segment_count=len(segment_files),
                total_original_unit_count=total_original_unit_count,
                backend=backend,
            )
            log_with_timestamp(
                f"checkpoint saved segment={segment_id} processed_units={len(segment_unit_records)}/{unit_count}"
            )

            if verification["chosen_all"]:
                log_with_timestamp(
                    f"drop unit={unit_id} segment={segment_id} chosen_all=true chosen_step_ids={chosen_step_ids}"
                )
                continue

            if chosen_step_ids:
                log_with_timestamp(
                    f"partial denoise unit={unit_id} segment={segment_id} chosen_step_ids={chosen_step_ids}"
                )
            else:
                log_with_timestamp(
                    f"keep unit={unit_id} segment={segment_id} chosen_step_ids=[]"
                )

        if len(segment_unit_records) < unit_count:
            segment_record["units"] = segment_unit_records
            segment_record["processed_unit_count"] = len(segment_unit_records)
            segment_record["status"] = "in_progress"
            continue

        output_segment, segment_stats = build_segment_outputs(segment_payload, original_units, segment_unit_records)
        output_segment["segment_denoise_meta"]["vlm_backend"] = backend
        dump_json(segment_output_path, output_segment)
        log_with_timestamp(
            f"written segment={segment_id} output={segment_output_path} removed_steps={segment_stats['removed_step_count']} kept_units={segment_stats['kept_unit_count']}"
        )
        segment_record.update(
            {
                "kept_step_count": segment_stats["kept_step_count"],
                "removed_step_count": segment_stats["removed_step_count"],
                "kept_unit_count": segment_stats["kept_unit_count"],
                "removed_unit_count": segment_stats["removed_unit_count"],
                "processed_unit_count": len(segment_unit_records),
                "output_path": str(segment_output_path.resolve()),
                "status": "completed",
                "units": segment_unit_records,
            }
        )
        save_audit_state(
            audit_path,
            audit_state,
            output_dir=output_dir,
            total_segment_count=len(segment_files),
            total_original_unit_count=total_original_unit_count,
            backend=backend,
        )

    audit_payload = save_audit_state(
        audit_path,
        audit_state,
        output_dir=output_dir,
        total_segment_count=len(segment_files),
        total_original_unit_count=total_original_unit_count,
        backend=backend,
    )
    log_with_timestamp(f"written audit={audit_path}")
    return audit_payload


def main() -> None:
    args = parse_args()
    segments_units_dir, segment_files = resolve_segments_units_dir(args.input_path)
    output_dir, audit_path = resolve_output_paths(segments_units_dir, args.output_dir, args.output_audit)
    log_with_timestamp(
        f"resolved input segments_units_dir={segments_units_dir} output_dir={output_dir} output_audit={audit_path}"
    )
    audit_payload = denoise_units(
        segment_files,
        output_dir,
        audit_path,
        backend=args.vlm_backend,
        trust_env=parse_env_bool("VLM_HTTP_TRUST_ENV", DEFAULT_HTTP_TRUST_ENV),
    )
    print(
        json.dumps(
            {
                "segments_units_dir": str(segments_units_dir),
                "output_dir": str(output_dir.resolve()),
                "output_audit": str(audit_path.resolve()),
                **audit_payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
