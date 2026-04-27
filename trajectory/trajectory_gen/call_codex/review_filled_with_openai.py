from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI


DEFAULT_ROOT = Path("tmp_trace") / "natural" / "impress_final"
DEFAULT_JSON_FILE = "report_filled.json"
DEFAULT_MODEL = "gpt-5.5"
REVIEW_STATUS_FILENAME = "_openai_step_review_status.json"
REVIEW_FIELDS = (
    "step_goal",
    "app",
    "action_preconditions",
    "nl_position",
    "action_before_state",
    "action_after_effects",
    "nl_explanation",
)
RUN_LOG_PATH: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an OpenAI multimodal model to review and correct filled step annotations, "
            "then optionally sync corrected steps back into segments_units JSON files."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Batch root containing trajectory folders. Default: tmp_trace/natural/impress_final.",
    )
    parser.add_argument(
        "--trajectory",
        default=None,
        help="Process a single trajectory folder. If omitted, batch-process direct child folders under --root.",
    )
    parser.add_argument(
        "--json-file",
        default=DEFAULT_JSON_FILE,
        help="Filled trajectory JSON filename to review inside each trajectory folder.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model used for review. Default: gpt-5.5.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL if set.",
    )
    parser.add_argument(
        "--no-auto-v1",
        action="store_true",
        help="Do not append /v1 automatically when the configured base URL looks like a gateway root.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name containing the API key. Default: OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--skip-segments",
        action="store_true",
        help="Only review the trajectory JSON; do not update segments_units/*.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing done status and review steps again.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batch mode only: process at most this many trajectory folders.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Debug option: review at most this many steps per trajectory.",
    )
    parser.add_argument(
        "--start-step",
        default=None,
        help="Resume/debug option: skip steps before this step_id, e.g. s12.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model responses under _openai_step_review_raw/.",
    )
    parser.add_argument(
        "--save-model-log",
        action="store_true",
        help="Save per-step model input/output logs, including prompt payload, image paths, raw response, parsed response, and applied fields.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for runtime logs and optional model logs. Defaults to <root>/_openai_step_review_logs.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Print runtime logs to console only; do not write a run log file.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before modifying a JSON file.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry each model call this many times before failing the current trajectory.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Initial retry delay in seconds. Subsequent retries use exponential backoff.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Call the model and print progress, but do not write report or segment files.",
    )
    return parser.parse_args()


def format_timestamp(dt: Optional[datetime] = None) -> str:
    return (dt or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{format_timestamp()}] {message}"
    print(line, flush=True)
    if RUN_LOG_PATH is not None:
        RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def configure_run_log(log_dir: Path, enabled: bool) -> None:
    global RUN_LOG_PATH
    if not enabled:
        RUN_LOG_PATH = None
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    RUN_LOG_PATH.write_text("", encoding="utf-8")


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


def read_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_backup(path: Path, enabled: bool) -> Optional[Path]:
    if not enabled or not path.is_file():
        return None
    backup_path = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def normalize_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        value = "; ".join(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return text if text else None


def extract_nl_position(step: Dict[str, Any]) -> Optional[str]:
    action = step.get("action")
    target = action.get("target") if isinstance(action, dict) else None
    if not isinstance(target, dict):
        return None
    nl_position = target.get("nl_position")
    return normalize_optional_text(nl_position)


def compact_neighbor_step(step: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(step, dict):
        return None
    return {
        "step_id": step.get("step_id"),
        "step_goal": step.get("step_goal"),
        "action": step.get("action"),
        "action_preconditions": step.get("action_preconditions"),
        "action_before_state": step.get("action_before_state"),
        "action_after_effects": step.get("action_after_effects"),
        "nl_explanation": step.get("nl_explanation"),
    }


def build_review_payload(
    *,
    report: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    step_index: int,
) -> Dict[str, Any]:
    step = steps[step_index]
    return {
        "task_context": {
            "task_id": report.get("task_id"),
            "task_category": report.get("task_category"),
            "task_title": report.get("task_title"),
            "instruction": report.get("instruction"),
            "trajectory_app": report.get("app"),
            "env": report.get("env") or {},
        },
        "step_context": {
            "step_index_1_based": step_index + 1,
            "total_steps": len(steps),
            "previous_step": compact_neighbor_step(steps[step_index - 1] if step_index > 0 else None),
            "next_step": compact_neighbor_step(steps[step_index + 1] if step_index + 1 < len(steps) else None),
        },
        "current_step": {
            "step_id": step.get("step_id"),
            "now_state": step.get("now_state"),
            "action": step.get("action"),
            "existing_filled_fields": {
                "step_goal": step.get("step_goal"),
                "app": step.get("app") or report.get("app"),
                "action_preconditions": step.get("action_preconditions"),
                "nl_position": extract_nl_position(step),
                "action_before_state": step.get("action_before_state"),
                "action_after_effects": step.get("action_after_effects"),
                "nl_explanation": step.get("nl_explanation"),
            },
        },
    }


def resolve_artifact_path(trajectory_dir: Path, rel_path: Any) -> Optional[Path]:
    if not rel_path:
        return None
    path = Path(str(rel_path))
    candidates = [path] if path.is_absolute() else [trajectory_dir / path, trajectory_dir.parent / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    return None


def image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def collect_step_images(
    *,
    trajectory_dir: Path,
    steps: Sequence[Dict[str, Any]],
    step_index: int,
) -> List[Tuple[str, Path]]:
    step = steps[step_index]
    now_state = step.get("now_state") if isinstance(step, dict) else {}
    if not isinstance(now_state, dict):
        now_state = {}

    image_specs: List[Tuple[str, Any]] = [
        ("before", now_state.get("screenshot_path_before")),
        ("before_part", now_state.get("screenshot_path_before_part")),
        ("after", now_state.get("screenshot_path_after")),
    ]
    if step_index + 1 < len(steps):
        next_now_state = steps[step_index + 1].get("now_state")
        if isinstance(next_now_state, dict):
            image_specs.append(("next_before", next_now_state.get("screenshot_path_before")))

    images: List[Tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, rel_path in image_specs:
        image_path = resolve_artifact_path(trajectory_dir, rel_path)
        if image_path is None or image_path in seen:
            continue
        images.append((label, image_path))
        seen.add(image_path)
    return images


def build_system_prompt() -> str:
    return """You are a strict reviewer for GUI trajectory step annotations.
You receive one step at a time, its already-filled annotation fields, neighboring step context, and screenshots labeled before, before_part, after, and next_before when available.

Your job is to verify whether the existing filled fields are accurate and concrete. If a field is inaccurate, vague, inconsistent with screenshots, or affected by OCR/encoding artifacts, rewrite it. If it is already accurate, keep its meaning but you may lightly polish wording.

Field requirements:
- step_goal: immediate goal of this exact step, not the whole task.
- app: target application used in this step.
- action_preconditions: list of concrete conditions visible or required before the action.
- nl_position: actual targeted UI element or location. Use null for keyboard-only actions with no target.
- action_before_state: concrete UI state before the action.
- action_after_effects: list of observable effects after the action, using next_before as stabilized evidence when useful.
- nl_explanation: concise explanation of what the action does and why.

Do not modify or invent action metadata such as click coordinates, key names, typed text, or screenshot paths. Do not describe irrelevant background windows unless they are the actual target application. Return only valid JSON.

MUST output one JSON object with this exact schema:
{{
  "is_revision_needed": true,
  "revision_reason": "brief reason, or no change needed",
  "fields": {{
    "step_goal": "...",
    "app": "...",
    "action_preconditions": ["..."],
    "nl_position": "..." or null,
    "action_before_state": "...",
    "action_after_effects": ["..."],
    "nl_explanation": "..."
}}

IMPORTANT: Your output must be in ENGLISH.
"""


def build_user_content(
    *,
    review_payload: Dict[str, Any],
    images: Sequence[Tuple[str, Path]],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for label, image_path in images:
        content.append(
            {
                "type": "text",
                "text": f"[Screenshot: {label}] {image_path.name}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{image_mime_type(image_path)};base64,"
                        f"{encode_image(image_path)}"
                    )
                },
            }
        )

    prompt = f"""Review this single step and return one JSON object with this exact schema:
{{
  "is_revision_needed": true,
  "revision_reason": "brief reason, or no change needed",
  "fields": {{
    "step_goal": "...",
    "app": "...",
    "action_preconditions": ["..."],
    "nl_position": "..." or null,
    "action_before_state": "...",
    "action_after_effects": ["..."],
    "nl_explanation": "..."
  }}
}}

Important:
1. The fields object must always contain all seven fields, even when no revision is needed.
2. Keep the annotation language consistent with the existing data, usually English.
3. Ground the answer in the screenshots and the current step action.

Step review payload:
{json.dumps(review_payload, ensure_ascii=False, indent=2)}"""
    content.append({"type": "text", "text": prompt})
    return content


def extract_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    if not text:
        raise ValueError("Model returned an empty response.")
    lower_text = text[:200].lower()
    if lower_text.startswith("<!doctype html") or lower_text.startswith("<html"):
        raise ValueError(
            "Model endpoint returned HTML instead of JSON. "
            "The base URL is probably pointing at a web UI root; use the OpenAI-compatible API base, usually ending with /v1."
        )
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                text = candidate
                break
    start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if start_positions:
        start = min(start_positions)
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(text[start:])
        return payload
    raise ValueError(f"Could not find JSON payload in model response:\n{raw_text}")


def normalize_base_url(base_url: Optional[str], *, auto_v1: bool) -> Optional[str]:
    if not base_url:
        return None
    normalized = base_url.strip().rstrip("/")
    if not auto_v1:
        return normalized
    lower_url = normalized.lower()
    if lower_url.endswith("/v1") or "/v1/" in lower_url:
        return normalized
    if "127.0.0.1" in lower_url or "localhost" in lower_url:
        return f"{normalized}/v1"
    return normalized


def create_openai_client(api_key_env: str, base_url: Optional[str], *, auto_v1: bool) -> OpenAI:
    load_env_file(Path(".env"))
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")
    resolved_base_url = normalize_base_url(base_url or os.getenv("OPENAI_BASE_URL"), auto_v1=auto_v1)
    if resolved_base_url:
        log(f"OpenAI base URL: {resolved_base_url}")
        return OpenAI(api_key=api_key, base_url=resolved_base_url)
    return OpenAI(api_key=api_key)


def _extract_content_from_message(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def extract_chat_completion_text(response: Any) -> str:
    if isinstance(response, str):
        text = response.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        nested_text = extract_chat_completion_text(parsed)
        return nested_text or text

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                content = _extract_content_from_message(first_choice.get("message"))
                if content:
                    return content
                content = first_choice.get("text")
                if isinstance(content, str):
                    return content
        content = response.get("content")
        if isinstance(content, str):
            return content
        for key in ("output_text", "response", "result", "text"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
        output = response.get("output")
        if isinstance(output, list):
            parts: List[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content_items = item.get("content")
                if isinstance(content_items, list):
                    for content_item in content_items:
                        if isinstance(content_item, dict):
                            text = content_item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "\n".join(parts).strip()
        data = response.get("data")
        if isinstance(data, dict):
            nested_text = extract_chat_completion_text(data)
            if nested_text:
                return nested_text
        return json.dumps(response, ensure_ascii=False)

    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = _extract_content_from_message(message)
        if content:
            return content
        text = getattr(first_choice, "text", None)
        if isinstance(text, str):
            return text

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    raise TypeError(f"Unsupported model response type: {type(response).__name__}")


def call_review_model(
    *,
    client: OpenAI,
    model: str,
    user_content: List[Dict[str, Any]],
    retries: int,
    retry_delay: float,
) -> Tuple[Dict[str, Any], str]:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
            raw_text = extract_chat_completion_text(response)
            parsed = extract_json_payload(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("Model response is not a JSON object.")
            return parsed, raw_text
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, retries):
                break
            delay = retry_delay * (2 ** (attempt - 1))
            log(f"Model call failed; retrying in {delay:.1f}s ({attempt}/{retries}): {type(exc).__name__}: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Review model call failed after {retries} attempt(s): {last_error}") from last_error


def find_review_fields(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates: List[Any] = [parsed.get("fields"), parsed]
    for key in ("corrected_fields", "revised_fields", "annotation", "step", "result", "data"):
        value = parsed.get(key)
        if isinstance(value, dict):
            candidates.extend([value.get("fields"), value])

    for key in ("annotations", "revised_steps", "steps"):
        value = parsed.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                candidates.extend([first.get("fields"), first.get("changes"), first])

    for candidate in candidates:
        if isinstance(candidate, dict) and all(field in candidate for field in REVIEW_FIELDS):
            return candidate
    return None


def validate_review_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    fields = find_review_fields(parsed)
    if not isinstance(fields, dict):
        raise ValueError(
            'Model response missing review fields. Expected either top-level "fields" '
            "or all seven review fields in a recognizable nested object."
        )

    missing = [field for field in REVIEW_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"Model response missing required field(s): {missing}")

    normalized = {
        "step_goal": str(fields.get("step_goal") or "").strip(),
        "app": str(fields.get("app") or "").strip(),
        "action_preconditions": normalize_text_list(fields.get("action_preconditions")),
        "nl_position": normalize_optional_text(fields.get("nl_position")),
        "action_before_state": str(fields.get("action_before_state") or "").strip(),
        "action_after_effects": normalize_text_list(fields.get("action_after_effects")),
        "nl_explanation": str(fields.get("nl_explanation") or "").strip(),
    }

    empty_fields = [
        field
        for field in (
            "step_goal",
            "app",
            "action_before_state",
            "nl_explanation",
        )
        if not normalized[field]
    ]
    if not normalized["action_preconditions"]:
        empty_fields.append("action_preconditions")
    if not normalized["action_after_effects"]:
        empty_fields.append("action_after_effects")
    if empty_fields:
        raise ValueError(f"Model response contains empty required field(s): {empty_fields}")
    return normalized


def apply_review_fields(report: Dict[str, Any], step: Dict[str, Any], fields: Dict[str, Any]) -> None:
    step["step_goal"] = fields["step_goal"]
    if "app" in step:
        step["app"] = fields["app"]
    elif fields["app"]:
        report["app"] = fields["app"]

    step["action_preconditions"] = fields["action_preconditions"]
    step["action_before_state"] = fields["action_before_state"]
    step["action_after_effects"] = fields["action_after_effects"]
    step["nl_explanation"] = fields["nl_explanation"]

    action = step.get("action")
    target = action.get("target") if isinstance(action, dict) else None
    if isinstance(target, dict):
        if fields["nl_position"] is None:
            target.pop("nl_position", None)
        else:
            target["nl_position"] = [fields["nl_position"]]


def read_status(status_path: Path) -> Dict[str, Any]:
    if not status_path.is_file():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_status(status_path: Path, status: Dict[str, Any], dry_run: bool) -> None:
    status["updated_at"] = format_timestamp()
    if dry_run:
        return
    write_json_file(status_path, status)


def is_step_done(status: Dict[str, Any], json_file: str, step_id: Any) -> bool:
    if status.get("json_file") != json_file:
        return False
    reviewed_steps = status.get("reviewed_steps")
    if not isinstance(reviewed_steps, dict):
        return False
    item = reviewed_steps.get(str(step_id))
    return isinstance(item, dict) and item.get("status") == "done"


def should_skip_before_start_step(step_id: Any, start_step: Optional[str], started: bool) -> Tuple[bool, bool]:
    if not start_step or started:
        return False, True
    if str(step_id) == start_step:
        return False, True
    return True, False


def save_raw_response(raw_dir: Path, step_id: Any, raw_text: str, dry_run: bool) -> None:
    if dry_run:
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{step_id}.json"
    raw_path.write_text(raw_text, encoding="utf-8")


def save_model_interaction_log(
    *,
    model_log_dir: Optional[Path],
    step_id: Any,
    step_index: int,
    model: str,
    review_payload: Dict[str, Any],
    images: Sequence[Tuple[str, Path]],
    prompt_text: str,
    raw_text: str,
    parsed: Dict[str, Any],
    applied_fields: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Optional[Path]:
    if model_log_dir is None:
        return None

    model_log_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".error.json" if error else ".json"
    log_path = model_log_dir / f"{step_index:04d}_{step_id}{suffix}"
    payload = {
        "logged_at": format_timestamp(),
        "model": model,
        "step_id": step_id,
        "step_index": step_index,
        "images": [
            {
                "label": label,
                "path": str(image_path),
            }
            for label, image_path in images
        ],
        "review_payload": review_payload,
        "prompt_text": prompt_text,
        "raw_response": raw_text,
        "parsed_response": parsed,
        "applied_fields": applied_fields,
        "error": error,
    }
    write_json_file(log_path, payload)
    return log_path


def review_report(
    *,
    client: OpenAI,
    trajectory_dir: Path,
    json_file: str,
    model: str,
    force: bool,
    max_steps: Optional[int],
    start_step: Optional[str],
    save_raw: bool,
    backup: bool,
    retries: int,
    retry_delay: float,
    model_log_dir: Optional[Path],
    dry_run: bool,
) -> Dict[str, Any]:
    report_path = trajectory_dir / json_file
    if not report_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {report_path}")

    status_path = trajectory_dir / REVIEW_STATUS_FILENAME
    status = read_status(status_path)
    if status.get("status") == "done" and status.get("json_file") == json_file and not force:
        log(f"Skip trajectory {trajectory_dir.name}: already marked done")
        return read_json_file(report_path)

    report = read_json_file(report_path)
    steps = report.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{report_path} does not contain a list field: steps")

    backup_path = maybe_backup(report_path, backup and not dry_run)
    if backup_path:
        log(f"Created backup: {backup_path}")

    if force or status.get("json_file") != json_file:
        status = {}
    status.setdefault("reviewed_steps", {})
    status.update(
        {
            "status": "running",
            "json_file": json_file,
            "model": model,
            "trajectory_dir": str(trajectory_dir),
            "started_at": status.get("started_at") or format_timestamp(),
        }
    )
    write_status(status_path, status, dry_run)

    reviewed_count = 0
    start_step_reached = start_step is None
    total_steps = len(steps)
    raw_dir = trajectory_dir / "_openai_step_review_raw"

    for step_index, step in enumerate(steps):
        step_id = step.get("step_id") or f"step_{step_index + 1}"
        skip_before, start_step_reached = should_skip_before_start_step(
            step_id, start_step, start_step_reached
        )
        if skip_before:
            continue
        if max_steps is not None and reviewed_count >= max_steps:
            log(f"Reached --max-steps={max_steps}; stop reviewing later steps in this trajectory")
            break
        if not force and is_step_done(status, json_file, step_id):
            log(f"  [{step_index + 1}/{total_steps}] Skip step {step_id}: already reviewed")
            continue

        images = collect_step_images(
            trajectory_dir=trajectory_dir,
            steps=steps,
            step_index=step_index,
        )
        image_labels = ", ".join(label for label, _ in images) or "no screenshots"
        log(f"  [{step_index + 1}/{total_steps}] Review step {step_id}; images: {image_labels}")

        review_payload = build_review_payload(report=report, steps=steps, step_index=step_index)
        user_content = build_user_content(review_payload=review_payload, images=images)
        prompt_text = str(user_content[-1].get("text", "")) if user_content else ""
        parsed, raw_text = call_review_model(
            client=client,
            model=model,
            user_content=user_content,
            retries=retries,
            retry_delay=retry_delay,
        )
        try:
            fields = validate_review_fields(parsed)
        except Exception as exc:
            error_log_path = save_model_interaction_log(
                model_log_dir=model_log_dir,
                step_id=step_id,
                step_index=step_index + 1,
                model=model,
                review_payload=review_payload,
                images=images,
                prompt_text=prompt_text,
                raw_text=raw_text,
                parsed=parsed,
                applied_fields=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            if error_log_path is not None:
                log(f"    Saved failed model log: {error_log_path}")
            raise

        model_log_path = save_model_interaction_log(
            model_log_dir=model_log_dir,
            step_id=step_id,
            step_index=step_index + 1,
            model=model,
            review_payload=review_payload,
            images=images,
            prompt_text=prompt_text,
            raw_text=raw_text,
            parsed=parsed,
            applied_fields=fields,
        )
        if model_log_path is not None:
            log(f"    Saved model log: {model_log_path}")
        apply_review_fields(report, step, fields)

        if save_raw:
            save_raw_response(raw_dir, step_id, raw_text, dry_run)
        if not dry_run:
            write_json_file(report_path, report)

        reviewed_steps = status.setdefault("reviewed_steps", {})
        reviewed_steps[str(step_id)] = {
            "status": "done",
            "step_index": step_index + 1,
            "is_revision_needed": bool(parsed.get("is_revision_needed")),
            "revision_reason": str(parsed.get("revision_reason") or "").strip(),
            "model_log_path": str(model_log_path) if model_log_path is not None else None,
            "updated_at": format_timestamp(),
        }
        write_status(status_path, status, dry_run)
        reviewed_count += 1

    if start_step and not start_step_reached:
        raise ValueError(f"--start-step {start_step} was not found in {report_path}")

    all_done = len(status.get("reviewed_steps", {})) >= len(steps)
    if max_steps is None and all_done:
        status["status"] = "review_done"
        write_status(status_path, status, dry_run)
    return report


def natural_json_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10**9, path.name.lower())


def sync_segments_from_report(
    *,
    trajectory_dir: Path,
    report: Dict[str, Any],
    backup: bool,
    dry_run: bool,
) -> int:
    segments_dir = trajectory_dir / "segments_units"
    if not segments_dir.is_dir():
        log(f"segments_units not found; skip sync: {segments_dir}")
        return 0

    steps = report.get("steps") or []
    step_by_id = {
        str(step.get("step_id")): deepcopy(step)
        for step in steps
        if isinstance(step, dict) and step.get("step_id") is not None
    }
    if not step_by_id:
        raise ValueError("Corrected report has no usable step_id values.")

    updated_files = 0
    segment_paths = sorted(segments_dir.glob("*.json"), key=natural_json_sort_key)
    for segment_path in segment_paths:
        segment = read_json_file(segment_path)
        segment_steps = segment.get("steps")
        if not isinstance(segment_steps, list):
            log(f"  Skip segment without steps list: {segment_path.name}")
            continue

        changed = False
        replaced_steps: List[Dict[str, Any]] = []
        for segment_step in segment_steps:
            step_id = str(segment_step.get("step_id")) if isinstance(segment_step, dict) else ""
            corrected = step_by_id.get(step_id)
            if corrected is None:
                replaced_steps.append(segment_step)
                continue
            replaced_steps.append(deepcopy(corrected))
            changed = True

        if not changed:
            continue
        segment["steps"] = replaced_steps
        maybe_backup(segment_path, backup and not dry_run)
        if not dry_run:
            write_json_file(segment_path, segment)
        updated_files += 1
        log(f"  Synced segment: {segment_path.name}")

    return updated_files


def finalize_status(
    *,
    trajectory_dir: Path,
    json_file: str,
    status_value: str,
    segment_updated_files: Optional[int],
    dry_run: bool,
) -> None:
    status_path = trajectory_dir / REVIEW_STATUS_FILENAME
    status = read_status(status_path)
    if status.get("json_file") != json_file:
        status["json_file"] = json_file
    status["status"] = status_value
    if segment_updated_files is not None:
        status["segment_sync"] = {
            "status": "done",
            "updated_files": segment_updated_files,
            "updated_at": format_timestamp(),
        }
    write_status(status_path, status, dry_run)


def process_trajectory(
    *,
    client: OpenAI,
    trajectory_dir: Path,
    args: argparse.Namespace,
) -> bool:
    log(f"Start trajectory: {trajectory_dir}")
    model_log_dir = None
    if args.save_model_log:
        model_log_dir = args.resolved_log_dir / trajectory_dir.name / "model_outputs"
    try:
        report = review_report(
            client=client,
            trajectory_dir=trajectory_dir,
            json_file=args.json_file,
            model=args.model,
            force=args.force,
            max_steps=args.max_steps,
            start_step=args.start_step,
            save_raw=args.save_raw,
            backup=not args.no_backup,
            retries=args.retries,
            retry_delay=args.retry_delay,
            model_log_dir=model_log_dir,
            dry_run=args.dry_run,
        )

        segment_updated_files: Optional[int] = None
        if not args.skip_segments and args.max_steps is None:
            log(f"Start syncing segments_units: {trajectory_dir.name}")
            segment_updated_files = sync_segments_from_report(
                trajectory_dir=trajectory_dir,
                report=report,
                backup=not args.no_backup,
                dry_run=args.dry_run,
            )
            log(f"Finished syncing segments_units; updated files: {segment_updated_files}")
        elif args.max_steps is not None:
            log("Skip segments_units sync because --max-steps is active and the report may be partially reviewed")

        status_value = "done" if args.max_steps is None else "partial"
        finalize_status(
            trajectory_dir=trajectory_dir,
            json_file=args.json_file,
            status_value=status_value,
            segment_updated_files=segment_updated_files,
            dry_run=args.dry_run,
        )
        log(f"Finished trajectory: {trajectory_dir.name}")
        return True
    except Exception as exc:
        log(f"Trajectory failed: {trajectory_dir.name}: {type(exc).__name__}: {exc}")
        finalize_status(
            trajectory_dir=trajectory_dir,
            json_file=args.json_file,
            status_value="failed",
            segment_updated_files=None,
            dry_run=args.dry_run,
        )
        return False


def discover_trajectory_dirs(root: Path, json_file: str) -> List[Path]:
    if (root / json_file).is_file():
        return [root]
    if not root.is_dir():
        raise NotADirectoryError(f"Root directory does not exist: {root}")
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / json_file).is_file()],
        key=lambda path: path.name.lower(),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if args.trajectory:
        trajectory_dirs = [Path(args.trajectory).expanduser().resolve()]
    else:
        trajectory_dirs = discover_trajectory_dirs(root, args.json_file)
        if args.limit is not None:
            trajectory_dirs = trajectory_dirs[: args.limit]

    if not trajectory_dirs:
        raise FileNotFoundError(f"No trajectory folders with {args.json_file} found under {root}")

    if args.log_dir:
        resolved_log_dir = Path(args.log_dir).expanduser().resolve()
    elif args.trajectory:
        resolved_log_dir = trajectory_dirs[0] / "_openai_step_review_logs"
    else:
        resolved_log_dir = root / "_openai_step_review_logs"
    args.resolved_log_dir = resolved_log_dir
    configure_run_log(resolved_log_dir, enabled=not args.no_log_file)

    client = create_openai_client(args.api_key_env, args.base_url, auto_v1=not args.no_auto_v1)

    if RUN_LOG_PATH is not None:
        log(f"Runtime log file: {RUN_LOG_PATH}")
    if args.save_model_log:
        log(f"Per-step model logs root: {resolved_log_dir}")
    log(
        f"Trajectories to process: {len(trajectory_dirs)}; model: {args.model}; "
        f"json file: {args.json_file}; sync segments: {not args.skip_segments}"
    )

    success_count = 0
    failed_count = 0
    for index, trajectory_dir in enumerate(trajectory_dirs, start=1):
        log(f"[{index}/{len(trajectory_dirs)}] Current trajectory: {trajectory_dir.name}")
        if process_trajectory(client=client, trajectory_dir=trajectory_dir, args=args):
            success_count += 1
        else:
            failed_count += 1

    log(f"All done. Success: {success_count}; failed: {failed_count}")
    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
