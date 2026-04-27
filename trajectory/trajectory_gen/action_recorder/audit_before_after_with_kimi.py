from __future__ import annotations

import argparse
import ast
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

try:
    from .audit_model_clients import (
        SUPPORTED_PROVIDERS,
        call_audit_model,
        normalize_provider_name,
        resolve_provider_api_key,
        resolve_provider_base_url,
        resolve_provider_model,
    )
except ImportError:
    from audit_model_clients import (
        SUPPORTED_PROVIDERS,
        call_audit_model,
        normalize_provider_name,
        resolve_provider_api_key,
        resolve_provider_base_url,
        resolve_provider_model,
    )


DEFAULT_PROVIDER = "kimi_coding"
DEFAULT_BASE_URL = "https://api.kimi.com/coding/"
DEFAULT_MODEL = "kimi-2.5"
DEFAULT_OUTPUT_NAME = "before_after_audit_problems.json"
DEFAULT_PER_TRAJECTORY_DIRNAME = "before_after_audit_by_trajectory"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Kimi Coding Plan to audit whether the before/after screenshots recorded in "
            "trajectory result/report.json are correct for each step."
        )
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default="tmp_trace/documents-export-2026-4-7",
        help="Root directory that contains trajectory folders. Default: tmp_trace/documents-export-2026-4-7",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output JSON path. Defaults to <input_root>/before_after_audit_problems.json."
        ),
    )
    parser.add_argument(
        "--per-trajectory-dir",
        default="",
        help=(
            "Directory for per-trajectory problematic-step JSON files. "
            "Defaults to <output_stem>_by_trajectory beside the aggregate output."
        ),
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=(
            "Model calling backend. "
            "kimi_coding uses Kimi Coding Plan by default; "
            "ark_coding uses Ark Coding Plan. "
            "Short aliases kimi/ark/openai are also accepted."
        ),
    )
    parser.add_argument("--api-key", default="", help="Provider API key. Overrides environment variables.")
    parser.add_argument(
        "--base-url",
        default="",
        help=f"Provider base URL. Default for kimi_coding: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default="",
        help=f"Model name override. Default for kimi_coding: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="Max output tokens for the model response. Default: 1024",
    )
    parser.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=10000,
        help="Thinking budget tokens for kimi_coding. Default: 10000",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable provider-side reasoning/thinking when the backend supports it.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent model requests. Default: 1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional hard limit on how many steps to audit. 0 means no limit.",
    )
    parser.add_argument(
        "--save-all-steps",
        action="store_true",
        help="Save all step judgments instead of only problematic ones.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime(TIMESTAMP_FORMAT)}] {message}", flush=True)


def normalize_input_path(raw_path: str) -> Path:
    cleaned = raw_path.strip()
    if cleaned.startswith("@"):
        cleaned = cleaned[1:]
    return Path(cleaned).expanduser().resolve()


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def parse_json_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find a JSON object in model output: {raw_text}")

    candidate = text[start : end + 1]

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', candidate)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    python_like = re.sub(r"\btrue\b", "True", repaired, flags=re.IGNORECASE)
    python_like = re.sub(r"\bfalse\b", "False", python_like, flags=re.IGNORECASE)
    python_like = re.sub(r"\bnull\b", "None", python_like, flags=re.IGNORECASE)

    try:
        parsed = ast.literal_eval(python_like)
        if isinstance(parsed, dict):
            return parsed
    except (SyntaxError, ValueError):
        pass

    raise ValueError(f"Failed to parse model output as JSON: {raw_text}")


def parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), TIMESTAMP_FORMAT)
    except ValueError:
        return None


def format_seconds_as_hms(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    whole_seconds = int(round(total_seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def find_report_paths(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("report.json") if path.parent.name == "result")


def resolve_report_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name != "report.json":
            raise FileNotFoundError(f"Expected report.json file, got: {input_path}")
        return [input_path]

    direct_report = input_path / "result" / "report.json"
    if direct_report.exists():
        return [direct_report]

    direct_report_alt = input_path / "report.json"
    if direct_report_alt.exists() and input_path.name == "result":
        return [direct_report_alt]

    return find_report_paths(input_path)


def resolve_image_path(trajectory_dir: Path, relative_path: Any) -> Path | None:
    if not relative_path or not isinstance(relative_path, str):
        return None

    rel = Path(relative_path)
    candidates = [
        trajectory_dir / rel,
        trajectory_dir / rel.name,
        trajectory_dir / "result" / rel.name,
        trajectory_dir / "result" / "screenshots" / rel.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_initial_record_time(steps: list[dict[str, Any]]) -> datetime | None:
    timestamps: list[datetime] = []
    for step in steps:
        now_state = step.get("now_state") or {}
        for key in ("screenshot_time_before", "screenshot_time_after"):
            parsed = parse_timestamp(now_state.get(key))
            if parsed is not None:
                timestamps.append(parsed)
    if not timestamps:
        return None
    return min(timestamps)


def extract_relative_timestamp(
    now_state: dict[str, Any],
    initial_time: datetime | None,
) -> tuple[float | None, str | None, str | None]:
    for key in ("screenshot_time_before", "screenshot_time_after"):
        parsed = parse_timestamp(now_state.get(key))
        if parsed is None:
            continue
        if initial_time is None:
            return None, None, key
        delta_seconds = (parsed - initial_time).total_seconds()
        return delta_seconds, format_seconds_as_hms(delta_seconds), key
    return None, None, None


def make_step_sort_key(step_id: Any) -> tuple[int, str]:
    text = str(step_id or "")
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1)), text
    return 10**9, text


def sanitize_path_fragment(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "__", value.strip())
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("._") or "trajectory"


def get_trajectory_label(input_root: Path, trajectory_dir: Path) -> str:
    candidates = [input_root]
    if input_root.is_file():
        candidates.append(input_root.parent)
        candidates.append(input_root.parent.parent)
    else:
        candidates.append(input_root.parent)

    for base in candidates:
        try:
            relative = trajectory_dir.relative_to(base).as_posix()
            if relative == ".":
                return trajectory_dir.name
            return relative
        except ValueError:
            continue
    return trajectory_dir.as_posix()


def build_per_trajectory_output_path(
    per_trajectory_dir: Path,
    input_root: Path,
    trajectory_dir: Path,
) -> Path:
    label = get_trajectory_label(input_root, trajectory_dir)
    filename = f"{sanitize_path_fragment(label)}.json"
    return per_trajectory_dir / filename


def get_default_output_path(input_root: Path) -> Path:
    if input_root.is_file():
        if input_root.name == "report.json" and input_root.parent.name == "result":
            return input_root.parent.parent / DEFAULT_OUTPUT_NAME
        return input_root.parent / DEFAULT_OUTPUT_NAME

    if input_root.name == "result" and (input_root / "report.json").exists():
        return input_root.parent / DEFAULT_OUTPUT_NAME

    return input_root / DEFAULT_OUTPUT_NAME


def build_step_prompt(
    *,
    report_path: Path,
    report_data: dict[str, Any],
    step: dict[str, Any],
    before_exists: bool,
    after_exists: bool,
) -> str:
    action = step.get("action") or {}
    now_state = step.get("now_state") or {}
    payload = {
        "trajectory_instruction": report_data.get("instruction", ""),
        "task_title": report_data.get("task_title", ""),
        "step_id": step.get("step_id", ""),
        "step_goal": step.get("step_goal", ""),
        "action": action,
        "action_before_state": step.get("action_before_state", ""),
        "action_after_effects": step.get("action_after_effects", []),
        "nl_explanation": step.get("nl_explanation", ""),
        "app_title_before": now_state.get("app_title_before", ""),
        "app_title_after": now_state.get("app_title_after", ""),
        "screenshot_time_before": now_state.get("screenshot_time_before", ""),
        "screenshot_time_after": now_state.get("screenshot_time_after", ""),
        "screenshot_path_before": now_state.get("screenshot_path_before", ""),
        "screenshot_path_after": now_state.get("screenshot_path_after", ""),
        "before_image_exists": before_exists,
        "after_image_exists": after_exists,
        "report_path": str(report_path),
    }

    return (
        "You are auditing a GUI trajectory record.\n"
        "You will receive:\n"
        "1. A screenshot that is supposed to be captured immediately BEFORE the action.\n"
        "2. A screenshot that is supposed to be captured immediately AFTER the action.\n"
        "3. The recorded step metadata.\n\n"
        "Decide whether the recorded before screenshot is correct and whether the recorded after screenshot is correct.\n"
        "A screenshot is correct only if it plausibly matches the intended temporal position around this action.\n\n"
        "Important rules:\n"
        "- Judge BEFORE and AFTER independently.\n"
        "- Use the action type, target position, visible UI changes, typed text, focus changes, movement, scrolling, dragging, and app/window context.\n"
        "- If the action could legitimately cause little or no visible change, AFTER may still be correct.\n"
        "- If evidence is ambiguous but still plausible, prefer true.\n"
        "- Return false only when there is a clear mismatch, missing required evidence, or the screenshot looks swapped, stale, or unrelated.\n"
        "- Do not assume the task must succeed globally; only judge whether these screenshots match the local step timing.\n\n"
        "Return ONLY a raw JSON object with this exact schema:\n"
        "{\n"
        '  "before_correct": true,\n'
        '  "after_correct": true,\n'
        '  "confidence": 0.0,\n'
        '  "before_reason": "short reason",\n'
        '  "after_reason": "short reason",\n'
        '  "summary": "short overall summary"\n'
        "}\n\n"
        "Step metadata:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def judge_step_with_model(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    max_output_tokens: int,
    enable_thinking: bool,
    thinking_budget_tokens: int,
    report_path: Path,
    trajectory_dir: Path,
    report_data: dict[str, Any],
    step: dict[str, Any],
    initial_time: datetime | None,
) -> dict[str, Any]:
    step_id = str(step.get("step_id", "")).strip() or "<unknown>"
    action = step.get("action") or {}
    action_type = str(action.get("type", "")).strip().lower()
    now_state = step.get("now_state") or {}

    before_rel = now_state.get("screenshot_path_before")
    after_rel = now_state.get("screenshot_path_after")
    before_path = resolve_image_path(trajectory_dir, before_rel)
    after_path = resolve_image_path(trajectory_dir, after_rel)
    before_exists = bool(before_path and before_path.exists())
    after_exists = bool(after_path and after_path.exists())

    relative_seconds, relative_hms, timestamp_basis = extract_relative_timestamp(
        now_state, initial_time
    )

    result: dict[str, Any] = {
        "trajectory_dir": str(trajectory_dir),
        "report_path": str(report_path),
        "instruction": report_data.get("instruction", ""),
        "step_id": step_id,
        "action_type": action_type,
        "relative_timestamp_seconds": relative_seconds,
        "relative_timestamp_hms": relative_hms,
        "relative_timestamp_basis": timestamp_basis,
        "before_correct": True,
        "after_correct": True,
        "before_reason": "",
        "after_reason": "",
        "summary": "",
        "confidence": None,
        "before_path": str(before_path) if before_path else None,
        "after_path": str(after_path) if after_path else None,
        "audit_status": "ok",
    }

    local_errors: list[str] = []
    if not before_rel:
        result["before_correct"] = False
        result["before_reason"] = "report.json does not contain screenshot_path_before"
        local_errors.append("missing_before_path")
    elif not before_exists:
        result["before_correct"] = False
        result["before_reason"] = "before screenshot file is missing on disk"
        local_errors.append("missing_before_file")

    if not after_rel:
        result["after_correct"] = False
        result["after_reason"] = "report.json does not contain screenshot_path_after"
        local_errors.append("missing_after_path")
    elif not after_exists:
        result["after_correct"] = False
        result["after_reason"] = "after screenshot file is missing on disk"
        local_errors.append("missing_after_file")

    if local_errors:
        result["audit_status"] = "local_validation_failed"
        result["summary"] = ", ".join(local_errors)
        result["confidence"] = 1.0
        return result

    prompt = build_step_prompt(
        report_path=report_path,
        report_data=report_data,
        step=step,
        before_exists=before_exists,
        after_exists=after_exists,
    )
    system_prompt = "You are a careful GUI trajectory auditor. Output JSON only."

    try:
        model_response = call_audit_model(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            before_path=before_path,
            after_path=after_path,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        raw_content = normalize_message_content(model_response["raw_text"])
        parsed = parse_json_response(raw_content)

        result["before_correct"] = bool(parsed.get("before_correct", True))
        result["after_correct"] = bool(parsed.get("after_correct", True))
        result["before_reason"] = str(parsed.get("before_reason", "")).strip()
        result["after_reason"] = str(parsed.get("after_reason", "")).strip()
        result["summary"] = str(parsed.get("summary", "")).strip()
        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)):
            result["confidence"] = float(confidence)
        else:
            result["confidence"] = None
        result["model_raw"] = parsed
        if model_response.get("reasoning"):
            result["model_reasoning"] = model_response["reasoning"]
        if model_response.get("provider_meta"):
            result["provider_meta"] = model_response["provider_meta"]
    except Exception as exc:
        result["audit_status"] = "model_error"
        result["audit_error"] = str(exc)
        result["before_correct"] = None
        result["after_correct"] = None
        result["before_reason"] = ""
        result["after_reason"] = ""
        result["summary"] = "model_audit_failed"
        result["confidence"] = None
    return result


def is_model_error_result(item: dict[str, Any]) -> bool:
    return str(item.get("audit_status", "")).strip().lower() == "model_error"


def is_problematic_result(item: dict[str, Any]) -> bool:
    if is_model_error_result(item):
        return False
    return item.get("before_correct") is False or item.get("after_correct") is False


def load_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_output_payload(
    *,
    input_root: Path,
    output_path: Path,
    provider: str,
    model: str,
    base_url: str,
    report_paths: list[Path],
    results: list[dict[str, Any]],
    save_all_steps: bool,
) -> dict[str, Any]:
    problem_results = [item for item in results if is_problematic_result(item)]
    model_error_results = [item for item in results if is_model_error_result(item)]
    selected_results = results if save_all_steps else [*problem_results, *model_error_results]
    return {
        "input_root": str(input_root),
        "output_path": str(output_path),
        "generated_at": datetime.now().strftime(TIMESTAMP_FORMAT),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "trajectories_found": len(report_paths),
        "steps_audited": len(results),
        "problematic_steps_count": len(problem_results),
        "problematic_before_count": sum(item.get("before_correct") is False for item in problem_results),
        "problematic_after_count": sum(item.get("after_correct") is False for item in problem_results),
        "audit_error_steps_count": len(model_error_results),
        "results": selected_results,
    }


def build_trajectory_payload(
    *,
    input_root: Path,
    trajectory_dir: Path,
    report_path: Path,
    output_path: Path,
    provider: str,
    model: str,
    base_url: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    problem_results = [item for item in results if is_problematic_result(item)]
    model_error_results = [item for item in results if is_model_error_result(item)]
    return {
        "input_root": str(input_root),
        "trajectory_dir": str(trajectory_dir),
        "trajectory_label": get_trajectory_label(input_root, trajectory_dir),
        "report_path": str(report_path),
        "output_path": str(output_path),
        "generated_at": datetime.now().strftime(TIMESTAMP_FORMAT),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "steps_audited": len(results),
        "problematic_steps_count": len(problem_results),
        "problematic_before_count": sum(item.get("before_correct") is False for item in problem_results),
        "problematic_after_count": sum(item.get("after_correct") is False for item in problem_results),
        "audit_error_steps_count": len(model_error_results),
        "results": [*problem_results, *model_error_results],
    }


def write_json_file(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_trajectory(
    *,
    input_root: Path,
    report_path: Path,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    max_output_tokens: int,
    enable_thinking: bool,
    thinking_budget_tokens: int,
    max_workers: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    trajectory_dir = report_path.parent.parent
    report_data = load_report(report_path)
    steps = report_data.get("steps") or []
    if max_steps > 0:
        steps = steps[:max_steps]
    initial_time = extract_initial_record_time(steps)

    log(
        f"Analyzing trajectory: {get_trajectory_label(input_root, trajectory_dir)} "
        f"({len(steps)} steps)"
    )

    jobs = [
        {
            "report_path": report_path,
            "trajectory_dir": trajectory_dir,
            "report_data": report_data,
            "step": step,
            "initial_time": initial_time,
        }
        for step in steps
    ]

    results: list[dict[str, Any]] = []
    if max_workers == 1:
        for job in jobs:
            results.append(
                judge_step_with_model(
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget_tokens,
                    report_path=job["report_path"],
                    trajectory_dir=job["trajectory_dir"],
                    report_data=job["report_data"],
                    step=job["step"],
                    initial_time=job["initial_time"],
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    judge_step_with_model,
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget_tokens,
                    report_path=job["report_path"],
                    trajectory_dir=job["trajectory_dir"],
                    report_data=job["report_data"],
                    step=job["step"],
                    initial_time=job["initial_time"],
                ): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                results.append(future.result())

    results.sort(
        key=lambda item: (
            item.get("trajectory_dir") or "",
            make_step_sort_key(item.get("step_id")),
        )
    )
    return results


def main() -> int:
    load_dotenv()
    args = parse_args()
    args.provider = normalize_provider_name(args.provider)
    if args.provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {args.provider}. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    input_root = normalize_input_path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else get_default_output_path(input_root)
    )
    per_trajectory_dir = (
        Path(args.per_trajectory_dir).expanduser().resolve()
        if args.per_trajectory_dir
        else output_path.parent / f"{output_path.stem}_by_trajectory"
    )

    api_key = resolve_provider_api_key(args.provider, args.api_key)
    base_url = resolve_provider_base_url(args.provider, args.base_url)
    model = resolve_provider_model(args.provider, args.model)
    max_workers = max(1, int(args.max_workers))
    enable_thinking = not args.disable_thinking

    report_paths = resolve_report_paths(input_root)
    if not report_paths:
        raise FileNotFoundError(f"No result/report.json found under: {input_root}")

    log(
        f"Found {len(report_paths)} trajectory report(s) under: {input_root}; "
        f"provider={args.provider}; model={model}"
    )
    results: list[dict[str, Any]] = []
    per_trajectory_files: list[str] = []
    remaining_steps = int(args.max_steps)

    for report_index, report_path in enumerate(report_paths, start=1):
        trajectory_dir = report_path.parent.parent
        trajectory_label = get_trajectory_label(input_root, trajectory_dir)
        trajectory_max_steps = remaining_steps if remaining_steps > 0 else 0

        log(f"[{report_index}/{len(report_paths)}] Current trajectory: {trajectory_label}")
        trajectory_results = process_trajectory(
            input_root=input_root,
            report_path=report_path,
            provider=args.provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_output_tokens=args.max_output_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=args.thinking_budget_tokens,
            max_workers=max_workers,
            max_steps=trajectory_max_steps,
        )
        results.extend(trajectory_results)

        trajectory_output_path = build_per_trajectory_output_path(
            per_trajectory_dir=per_trajectory_dir,
            input_root=input_root,
            trajectory_dir=trajectory_dir,
        )
        trajectory_payload = build_trajectory_payload(
            input_root=input_root,
            trajectory_dir=trajectory_dir,
            report_path=report_path,
            output_path=trajectory_output_path,
            provider=args.provider,
            model=model,
            base_url=base_url,
            results=trajectory_results,
        )
        write_json_file(trajectory_output_path, trajectory_payload, overwrite=args.overwrite)
        per_trajectory_files.append(str(trajectory_output_path))

        log(
            f"Finished trajectory: {trajectory_label}; "
            f"problematic steps={trajectory_payload['problematic_steps_count']}; "
            f"saved to {trajectory_output_path}"
        )

        if remaining_steps > 0:
            remaining_steps -= len(trajectory_results)
            if remaining_steps <= 0:
                log("Reached --max-steps limit; stopping early.")
                break

    payload = build_output_payload(
        input_root=input_root,
        output_path=output_path,
        provider=args.provider,
        model=model,
        base_url=base_url,
        report_paths=report_paths,
        results=results,
        save_all_steps=args.save_all_steps,
    )
    payload["per_trajectory_dir"] = str(per_trajectory_dir)
    payload["per_trajectory_files"] = per_trajectory_files

    write_json_file(output_path, payload, overwrite=args.overwrite)

    log(f"Trajectories found: {payload['trajectories_found']}")
    log(f"Steps audited: {payload['steps_audited']}")
    log(f"Problematic steps: {payload['problematic_steps_count']}")
    log(f"Saved aggregate results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
