from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from call_codex_0413 import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_RETRIES,
    DEFAULT_MINIMAX_MODEL,
    apply_minimax_revisions,
    build_minimax_review_payload,
    build_step_diagnostics,
    call_codex,
    extract_json_payload,
    infer_primary_app,
    load_report,
    maybe_run_minimax_review,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill trajectory annotations one step at a time with Codex."
    )
    parser.add_argument(
        "conversation_folder",
        nargs="?",
        default=None,
        help="Folder containing the target JSON file and screenshots.",
    )
    parser.add_argument(
        "--json-file",
        default="report.json",
        help="Name of the JSON file to read inside the conversation folder.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path. Defaults to overwriting the selected --json-file in place.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help="Codex model used for each step.",
    )
    parser.add_argument(
        "--minimax-model",
        default=DEFAULT_MINIMAX_MODEL,
        help="MiniMax model used for optional review of anomalous steps.",
    )
    parser.add_argument(
        "--minimax-review-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to run MiniMax review after all step calls.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw per-step Codex responses and optional MiniMax response.",
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
        "Please input the conversation folder path (where the JSON file and screenshots are located): "
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


def load_working_report(report_path: Path, output_path: Path, *, force: bool) -> Dict[str, Any]:
    if output_path != report_path and output_path.is_file() and not force:
        return load_report(output_path)
    return load_report(report_path)


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
        ("before", now_state.get("screenshot_path_before")),
        ("before_part", now_state.get("screenshot_path_before_part")),
        ("after", now_state.get("screenshot_path_after")),
    ]
    if step_index + 1 < len(steps):
        next_now_state = steps[step_index + 1].get("now_state")
        if isinstance(next_now_state, dict):
            image_specs.append(("next_before", next_now_state.get("screenshot_path_before")))

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
1. Do not modify or invent action metadata such as coordinates, key names, typed text, screenshot paths, or URLs.
2. If a click confirms success, deselects an object, or appears to be a misclick/no-op, describe that explicitly.
3. Keep descriptions concrete and screen-grounded. Avoid vague text such as "the page changes".
4. The output must be in ENGLISH.

Output requirements:
- Return JSON only.
- Return one top-level object with exactly one field named "annotations".
- "annotations" must be an array containing exactly one object.
- Do not include markdown or explanations outside JSON."""


def build_user_prompt(
    *,
    conversation_folder: Path,
    step_payload: Dict[str, Any],
) -> str:
    return f"""Conversation folder:
{conversation_folder}

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


def annotation_from_step(report: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step_id": step.get("step_id"),
        "step_goal": step.get("step_goal"),
        "app": step.get("app") or report.get("app"),
        "action_preconditions": step.get("action_preconditions"),
        "nl_position": extract_nl_position(step),
        "action_before_state": step.get("action_before_state"),
        "action_after_effects": step.get("action_after_effects"),
        "nl_explanation": step.get("nl_explanation"),
    }


def is_step_done(status: Dict[str, Any], source_json: str, step_id: Any) -> bool:
    if status.get("source_json_file") != source_json:
        return False
    done_steps = status.get("done_steps")
    return isinstance(done_steps, dict) and str(step_id) in done_steps


def main() -> None:
    args = parse_args()
    conversation_folder = args.conversation_folder.strip() if args.conversation_folder else prompt_for_folder()
    if not conversation_folder:
        conversation_folder = os.getcwd()

    folder_path = Path(conversation_folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Conversation folder does not exist: {folder_path}")

    report_path = resolve_report_path(folder_path, args.json_file)
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
            conversation_folder=folder_path,
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

    ai_data = [annotation_from_step(report, step) for step in steps]
    review_payload = build_minimax_review_payload(
        report=report,
        diagnostics=diagnostics,
        ai_data=ai_data,
    )
    minimax_review = maybe_run_minimax_review(
        review_mode=args.minimax_review_mode,
        review_payload=review_payload,
        model=args.minimax_model,
        raw_output_path=(
            output_path.with_suffix(output_path.suffix + ".minimax_step.raw.txt") if args.save_raw else None
        ),
    )
    minimax_review, review_sanitized_count = sanitize_text_artifacts(minimax_review)
    if review_sanitized_count:
        print(
            f"[warn] Sanitized {review_sanitized_count} suspicious text value(s) in MiniMax review output.",
            file=sys.stderr,
        )
    apply_minimax_revisions(ai_data, minimax_review)
    for step, annotation in zip(steps, ai_data):
        apply_annotation_to_step(report, step, annotation, overwrite=True)

    report["fill_meta_step"] = {
        "source_json_file": report_path.name,
        "codex_model": args.model,
        "stepwise": True,
        "overwrite": args.overwrite,
        "minimax_review_mode": args.minimax_review_mode,
        "minimax_model": args.minimax_model if minimax_review is not None else None,
        "processed_steps_this_run": processed_count,
    }
    save_report(output_path, report)

    status["status"] = "done" if args.max_steps is None else "partial"
    status["processed_steps_this_run"] = processed_count
    write_status(status_path, status, enabled=not args.no_status)
    log(f"Filled fields step-by-step in: {output_path}")


if __name__ == "__main__":
    main()
