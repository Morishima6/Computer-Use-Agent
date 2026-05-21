from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from call_codex_0413 import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_RETRIES,
    build_step_diagnostics,
    call_codex,
    extract_json_payload,
    infer_primary_app,
    load_report,
    normalize_text_list,
    resolve_artifact_path,
    resolve_report_path,
    sanitize_text_artifacts,
    save_report,
    save_text,
)


STATUS_FILENAME = "_call_codex_step_status.json"
TARGET_FIELDS = (
    "step_goal",
    "app",
    "action_preconditions",
    "nl_position",
    "action_before_state",
    "action_after_effects",
    "nl_explanation",
)

MOUSE_ACTION_TYPES = {
    "click",
    "double_click",
    "drag",
    "drag_to",
    "mouse_move",
    "move",
    "scroll",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill trajectory annotations one step at a time with Codex."
    )
    parser.add_argument(
        "conversation_path",
        nargs="?",
        default=None,
        help="Folder containing the target JSON file and screenshots, a direct path to report*.json, or a batch root when --batch is enabled.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all trajectory session directories under conversation_path.",
    )
    parser.add_argument(
        "--subdir",
        default="",
        help="Subdirectory under each session directory that contains the target JSON file. Default: empty, meaning the session root. For extracted trajectories, use: result.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for --batch. Default: 1.",
    )
    parser.add_argument(
        "--json-file",
        default="report.json",
        help="Name of the JSON file to read inside the conversation folder.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Optional output JSON path. Defaults to overwriting the selected --json-file in place. "
            "In --batch mode, use a relative name such as report_denoised_filled.json so each session writes beside its input JSON."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help="Codex model used for each step.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw per-step Codex responses.",
    )
    parser.add_argument(
        "--codex-retries",
        type=int,
        default=DEFAULT_CODEX_RETRIES,
        help="Retry Codex CLI transient stream failures this many times per step.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore step status and regenerate all selected steps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filled fields. By default, only empty fields are filled.",
    )
    parser.add_argument(
        "--start-step",
        default=None,
        help="Start from this step_id, e.g. s12.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Process at most this many non-skipped steps.",
    )
    parser.add_argument(
        "--no-status",
        action="store_true",
        help="Do not use the status file for resume.",
    )
    return parser.parse_args()


def prompt_for_folder() -> str:
    return input(
        "Please input the conversation folder path or target report JSON path: "
    ).strip()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_status(status_path: Path) -> Dict[str, Any]:
    if not status_path.is_file():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_status(status_path: Path, payload: Dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_output_path(report_path: Path, output_json: Optional[str]) -> Path:
    if not output_json:
        return report_path
    output_candidate = Path(output_json).expanduser()
    if output_candidate.is_absolute():
        return output_candidate.resolve()
    return (report_path.parent / output_candidate).resolve()


def validate_batch_output_json(output_json: Optional[str]) -> None:
    if not output_json:
        return
    output_candidate = Path(output_json).expanduser()
    if output_candidate.is_absolute():
        raise ValueError("--output-json must be a relative path in --batch mode")
    if any(part == ".." for part in output_candidate.parts):
        raise ValueError("--output-json cannot contain '..' in --batch mode")


def load_working_report(report_path: Path, output_path: Path, *, force: bool) -> Dict[str, Any]:
    if output_path != report_path and output_path.is_file() and not force:
        return load_report(output_path)
    return load_report(report_path)


def resolve_input_paths(conversation_path: str, json_file: str) -> Tuple[Path, Path]:
    input_path = Path(conversation_path).expanduser().resolve()
    if input_path.is_file():
        return input_path.parent, input_path
    if input_path.is_dir():
        return input_path, resolve_report_path(input_path, json_file)
    if input_path.suffix.lower() == ".json":
        raise FileNotFoundError(f"JSON file does not exist: {input_path}")
    raise NotADirectoryError(f"Conversation folder does not exist: {input_path}")


def find_batch_report_paths(input_path: str, json_file: str, subdir: str) -> List[Path]:
    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() != ".json":
            raise ValueError(f"Batch input file is not a JSON file: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Batch input directory does not exist: {root}")

    report_paths: List[Path] = []
    root_report_dir = root / subdir if subdir else root
    root_report_path = root_report_dir / json_file
    if root_report_path.is_file():
        report_paths.append(root_report_path.resolve())

    for session_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        report_dir = session_dir / subdir if subdir else session_dir
        report_path = report_dir / json_file
        if report_path.is_file():
            report_paths.append(report_path.resolve())

    return report_paths


def extract_nl_position(step: Dict[str, Any]) -> Optional[str]:
    action = step.get("action")
    target = action.get("target") if isinstance(action, dict) else None
    if not isinstance(target, dict):
        return None
    nl_position = target.get("nl_position")
    if isinstance(nl_position, list):
        text = "; ".join(str(item).strip() for item in nl_position if str(item).strip())
        return text or None
    if isinstance(nl_position, str):
        return nl_position.strip() or None
    return None


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


def get_action_type(step: Dict[str, Any]) -> str:
    action = step.get("action") if isinstance(step, dict) else {}
    if not isinstance(action, dict):
        return ""
    return str(action.get("type") or "").strip().lower()


def select_before_screenshot_path(step: Dict[str, Any]) -> Any:
    now_state = step.get("now_state") if isinstance(step, dict) else {}
    if not isinstance(now_state, dict):
        return None
    field_name = (
        "screenshot_path_before"
        if get_action_type(step) in MOUSE_ACTION_TYPES
        else "screenshot_path_before_raw"
    )
    return now_state.get(field_name)


def collect_step_screenshot_paths(
    *,
    report_path: Path,
    steps: Sequence[Dict[str, Any]],
    step_index: int,
) -> List[Dict[str, Any]]:
    step = steps[step_index]
    now_state = step.get("now_state") if isinstance(step, dict) else {}
    if not isinstance(now_state, dict):
        now_state = {}

    image_specs: List[Tuple[str, Any]] = [
        ("before", select_before_screenshot_path(step)),
        ("before_part", now_state.get("screenshot_path_before_part")),
        ("after", now_state.get("screenshot_path_after")),
    ]
    if step_index + 1 < len(steps):
        next_step = steps[step_index + 1]
        next_now_state = next_step.get("now_state")
        if isinstance(next_now_state, dict):
            image_specs.append(("next_before", select_before_screenshot_path(next_step)))

    screenshots: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for label, rel_path in image_specs:
        resolved = resolve_artifact_path(report_path, rel_path)
        if resolved is not None:
            if resolved in seen:
                continue
            seen.add(resolved)
        screenshots.append(
            {
                "label": label,
                "json_path": rel_path,
                "absolute_path": str(resolved) if resolved is not None else None,
                "exists": resolved is not None,
            }
        )
    return screenshots


def build_step_payload(
    *,
    report: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    step_index: int,
    diagnostics: Sequence[Dict[str, Any]],
    report_path: Path,
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
            "source_json_file": report_path.name,
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
        "screenshots": collect_step_screenshot_paths(
            report_path=report_path,
            steps=steps,
            step_index=step_index,
        ),
        "diagnostics": diagnostics[step_index] if step_index < len(diagnostics) else {},
    }


def build_system_prompt(primary_app: str) -> str:
    return f"""You are an action-behavior analyst and recorder.

Your task is to fill annotation fields for exactly one GUI trajectory step.
Use only the provided step JSON, neighboring step context, and the screenshots listed in the prompt.

The target application for this trajectory is: {primary_app}.
Stay tightly focused on this target application only. Ignore unrelated desktop, terminal, wallpaper, notifications, or background windows unless they are directly involved in the step.

For the current step, inspect these screenshot roles when available:
- before: pre-action UI state.
- before_part: cropped pre-action evidence, often around the red pointer marker.
- after: immediate post-action UI state.
- next_before: next step's before screenshot, useful as the stabilized post-action state when after is transitional.

Produce exactly one annotation object with these fields:
- step_goal: short phrase for the immediate goal of this step, not the whole task.
- app: application used in this step.
- action_preconditions: concrete conditions that must already be true before the action.
- nl_position: actual targeted UI element or location. Use null for keyboard-only actions with no real on-screen target.
- action_before_state: concrete UI state before the action.
- action_after_effects: observable changes caused by the action, using next_before when it is the clearer stable result.
- nl_explanation: concise natural-language explanation of the action and its purpose.

Special rules:
1. The red cross marker and the visible mouse cursor may be misaligned. Treat the red cross marker and its center point as the authoritative action target, and use the response changes in the after screenshot as supporting evidence.
2. Do not modify or invent action metadata such as coordinates, key names, typed text, screenshot paths, or URLs.
3. If a click confirms success, deselects an object, or appears to be a misclick/no-op, describe that explicitly.
4. Keep descriptions concrete and screen-grounded. Avoid vague text such as "the page changes".
5. The output must be in ENGLISH.

Output requirements:
- Return JSON only.
- Return one top-level object with exactly one field named "annotations".
- "annotations" must be an array containing exactly one object.
- Do not include markdown or explanations outside JSON."""


def build_user_prompt(
    *,
    source_folder: Path,
    step_payload: Dict[str, Any],
) -> str:
    return f"""Source folder:
{source_folder}

Read only the current step payload below and the referenced screenshot files.
Do not process the full trajectory JSON as one batch.
This call is responsible for exactly one step.

Screenshot files for this step:
{json.dumps(step_payload.get("screenshots", []), ensure_ascii=False, indent=2)}

Current step payload:
{json.dumps(step_payload, ensure_ascii=False, indent=2)}

Return only this JSON shape:
{{
  "annotations": [
    {{
      "step_goal": "...",
      "app": "...",
      "action_preconditions": ["..."],
      "nl_position": "..." or null,
      "action_before_state": "...",
      "action_after_effects": ["..."],
      "nl_explanation": "..."
    }}
  ]
}}"""


def parse_step_annotation(raw_text: str) -> Dict[str, Any]:
    payload = extract_json_payload(raw_text)
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        annotations = payload["annotations"]
    elif isinstance(payload, list):
        annotations = payload
    else:
        raise ValueError(f"Expected annotations array, got {type(payload).__name__}.")

    if len(annotations) != 1 or not isinstance(annotations[0], dict):
        raise ValueError("Expected exactly one annotation object.")
    annotation, _ = sanitize_text_artifacts(annotations[0])
    if not isinstance(annotation, dict):
        raise ValueError("Annotation became invalid after sanitization.")
    return annotation


def is_empty_value(value: Any) -> bool:
    return value in (None, "", [])


def should_set_field(current_value: Any, *, overwrite: bool) -> bool:
    return overwrite or is_empty_value(current_value)


def apply_annotation_to_step(
    report: Dict[str, Any],
    step: Dict[str, Any],
    annotation: Dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if should_set_field(step.get("step_goal"), overwrite=overwrite) and annotation.get("step_goal"):
        step["step_goal"] = annotation["step_goal"]

    if annotation.get("app"):
        if "app" in step and should_set_field(step.get("app"), overwrite=overwrite):
            step["app"] = annotation["app"]
        elif should_set_field(report.get("app"), overwrite=overwrite):
            report["app"] = annotation["app"]

    preconditions = normalize_text_list(annotation.get("action_preconditions"))
    if preconditions and should_set_field(step.get("action_preconditions"), overwrite=overwrite):
        step["action_preconditions"] = preconditions

    action = step.get("action")
    target = action.get("target") if isinstance(action, dict) else None
    if isinstance(target, dict) and should_set_field(target.get("nl_position"), overwrite=overwrite):
        nl_position = annotation.get("nl_position")
        if nl_position is None:
            target.pop("nl_position", None)
        elif str(nl_position).strip():
            target["nl_position"] = [str(nl_position).strip()]

    if should_set_field(step.get("action_before_state"), overwrite=overwrite) and annotation.get("action_before_state"):
        step["action_before_state"] = annotation["action_before_state"]

    after_effects = normalize_text_list(annotation.get("action_after_effects"))
    if after_effects and should_set_field(step.get("action_after_effects"), overwrite=overwrite):
        step["action_after_effects"] = after_effects

    if should_set_field(step.get("nl_explanation"), overwrite=overwrite) and annotation.get("nl_explanation"):
        step["nl_explanation"] = annotation["nl_explanation"]


def is_step_done(status: Dict[str, Any], source_json: str, step_id: Any) -> bool:
    if status.get("source_json_file") != source_json:
        return False
    done_steps = status.get("done_steps")
    return isinstance(done_steps, dict) and str(step_id) in done_steps


def process_conversation(conversation_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    folder_path, report_path = resolve_input_paths(conversation_path, args.json_file)
    output_path = resolve_output_path(report_path, args.output_json)
    report = load_working_report(report_path, output_path, force=args.force)
    steps = report.get("steps")
    if not isinstance(steps, list):
        raise RuntimeError(f"Expected {report_path.name}.steps to be a list.")

    diagnostics = build_step_diagnostics(report_path, report)
    primary_app = infer_primary_app(report)
    system_prompt = build_system_prompt(primary_app)
    status_path = output_path.with_name(STATUS_FILENAME)
    status = {} if args.force else read_status(status_path)
    status.setdefault("done_steps", {})
    status.update(
        {
            "status": "running",
            "source_json_file": report_path.name,
            "output_json_file": str(output_path),
            "model": args.model,
            "overwrite": args.overwrite,
            "started_at": status.get("started_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_status(status_path, status, enabled=not args.no_status)

    raw_dir = output_path.parent / "_call_codex_step_raw"
    processed_count = 0
    start_reached = args.start_step is None

    for step_index, step in enumerate(steps):
        step_id = step.get("step_id") or f"step_{step_index + 1}"
        if not start_reached:
            if str(step_id) == args.start_step:
                start_reached = True
            else:
                continue
        if args.max_steps is not None and processed_count >= args.max_steps:
            log(f"Reached --max-steps={args.max_steps}; stop.")
            break
        if not args.force and not args.no_status and is_step_done(status, report_path.name, step_id):
            log(f"[{step_index + 1}/{len(steps)}] skip {step_id}: already done")
            continue

        step_payload = build_step_payload(
            report=report,
            steps=steps,
            step_index=step_index,
            diagnostics=diagnostics,
            report_path=report_path,
        )
        image_labels = ", ".join(
            item["label"] for item in step_payload.get("screenshots", []) if item.get("exists")
        ) or "no screenshots"
        log(f"[{step_index + 1}/{len(steps)}] fill {step_id}; screenshots: {image_labels}")

        user_prompt = build_user_prompt(
            source_folder=folder_path,
            step_payload=step_payload,
        )
        raw_text = call_codex(args.model, system_prompt, user_prompt, retries=args.codex_retries)
        if args.save_raw:
            save_text(raw_dir / f"{step_index + 1:04d}_{step_id}.json", raw_text)

        annotation = parse_step_annotation(raw_text)
        annotation["step_id"] = step_id
        apply_annotation_to_step(report, step, annotation, overwrite=args.overwrite)
        save_report(output_path, report)

        status["done_steps"][str(step_id)] = {
            "step_index": step_index + 1,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_status(status_path, status, enabled=not args.no_status)
        processed_count += 1

    if args.start_step and not start_reached:
        raise ValueError(f"--start-step {args.start_step} was not found in {report_path}")

    report["fill_meta_step"] = {
        "source_json_file": report_path.name,
        "codex_model": args.model,
        "stepwise": True,
        "overwrite": args.overwrite,
        "processed_steps_this_run": processed_count,
    }
    save_report(output_path, report)

    status["status"] = "done" if args.max_steps is None else "partial"
    status["processed_steps_this_run"] = processed_count
    write_status(status_path, status, enabled=not args.no_status)
    log(f"Filled fields step-by-step in: {output_path}")
    return {
        "report_path": str(report_path),
        "output_path": str(output_path),
        "total_steps": len(steps),
        "processed_steps_this_run": processed_count,
        "status": status["status"],
    }


def process_batch_conversation(report_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    try:
        meta = process_conversation(str(report_path), args)
        return {
            "report_path": str(report_path),
            "meta": meta,
            "error": None,
        }
    except Exception as exc:
        return {
            "report_path": str(report_path),
            "meta": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.batch and args.workers != 1:
        raise ValueError("--workers can only be greater than 1 when --batch is enabled")
    if args.batch:
        validate_batch_output_json(args.output_json)

    conversation_path = args.conversation_path.strip() if args.conversation_path else prompt_for_folder()
    if not conversation_path:
        conversation_path = str(Path.cwd())

    if not args.batch:
        process_conversation(conversation_path, args)
        return

    report_paths = find_batch_report_paths(conversation_path, args.json_file, args.subdir)
    if not report_paths:
        raise FileNotFoundError(
            f"No {args.json_file} found under {Path(conversation_path).expanduser().resolve()} "
            f"with subdir={args.subdir!r}"
        )

    max_workers = min(args.workers, len(report_paths))
    log(
        f"Batch start reports={len(report_paths)} subdir={args.subdir!r} "
        f"json_file={args.json_file!r} output_json={args.output_json!r} workers={max_workers}"
    )
    failed: List[Dict[str, str]] = []
    total_steps = 0
    total_processed_steps = 0

    def collect_batch_result(index: int, report_path: Path, result: Dict[str, Any]) -> None:
        nonlocal total_steps, total_processed_steps
        error = result.get("error")
        if error:
            failed.append({"report_path": str(report_path), "error": str(error)})
            log(f"[{index}/{len(report_paths)}] failed {report_path}: {error}")
            return

        meta = result["meta"]
        total_steps += int(meta.get("total_steps", 0))
        total_processed_steps += int(meta.get("processed_steps_this_run", 0))
        log(
            f"[{index}/{len(report_paths)}] done {report_path}; "
            f"processed_steps_this_run={meta.get('processed_steps_this_run', 0)}"
        )

    if args.workers == 1:
        for index, report_path in enumerate(report_paths, start=1):
            log(f"[{index}/{len(report_paths)}] start {report_path}")
            collect_batch_result(index, report_path, process_batch_conversation(report_path, args))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {}
            for index, report_path in enumerate(report_paths, start=1):
                log(f"[{index}/{len(report_paths)}] submitted {report_path}")
                future = executor.submit(process_batch_conversation, report_path, args)
                future_to_item[future] = (index, report_path)

            for future in as_completed(future_to_item):
                index, report_path = future_to_item[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "report_path": str(report_path),
                        "meta": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                collect_batch_result(index, report_path, result)

    summary = {
        "report_count": len(report_paths),
        "failed_count": len(failed),
        "total_steps": total_steps,
        "total_processed_steps_this_run": total_processed_steps,
        "failed": failed,
    }
    log(
        f"Batch finished reports={summary['report_count']} failed={summary['failed_count']} "
        f"processed_steps_this_run={summary['total_processed_steps_this_run']}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
