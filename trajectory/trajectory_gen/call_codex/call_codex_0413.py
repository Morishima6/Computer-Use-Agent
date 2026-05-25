from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"
DEFAULT_CODEX_RETRIES = 3
POINTER_ACTION_TYPES = {"click", "drag_to"}
SUSPICIOUS_MOJIBAKE_MARKERS = (
    "\u9225",
    "\u95b3",
    "\u00e2\u20ac",
    "\u00c3\u00a2",
    "\u20ac\u2122",
    "鑱",
    "閰",
    "鏄",
    "妗",
    "紱",
    "绱",
    "娲",
    "鐨",
    "鍦",
    "鎼",
    "灞",
    "姝",
    "€",
)
TEXT_NORMALIZATION_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)
CODEX_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_goal": {"type": "string"},
                    "app": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "action_preconditions": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "nl_position": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                    "action_before_state": {"type": "string"},
                    "action_after_effects": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "nl_explanation": {"type": "string"},
                },
                "required": [
                    "step_goal",
                    "app",
                    "action_preconditions",
                    "nl_position",
                    "action_before_state",
                    "action_after_effects",
                    "nl_explanation",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["annotations"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing trajectory fields in a chosen JSON file using Codex, with optional MiniMax review."
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
        help="Name of the JSON file to read and update inside the conversation folder.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path. Defaults to overwriting the selected --json-file in place.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CODEX_MODEL,
        help="Codex model used for the main fill step.",
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
        help="Whether to run MiniMax review for steps with missing screenshots. 'auto' skips review if dependencies or credentials are unavailable.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw Codex and MiniMax responses beside the processed JSON file.",
    )
    parser.add_argument(
        "--codex-retries",
        type=int,
        default=DEFAULT_CODEX_RETRIES,
        help="Retry Codex CLI transient stream failures this many times.",
    )
    return parser.parse_args()


def prompt_for_folder() -> str:
    return input(
        "Please input the conversation folder path (where the JSON file and screenshots are located): "
    ).strip()


def find_codex_path() -> str:
    # return "codex"
    return r"D:\Program Files\nodejs\node_global\codex.cmd"


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


def extract_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    if not text:
        raise ValueError("Model returned an empty response.")

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    decoder = json.JSONDecoder()
    for start_char in ("[", "{"):
        start = text.find(start_char)
        if start == -1:
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
            return payload
        except json.JSONDecodeError:
            repaired = escape_unescaped_inner_quotes(text[start:])
            if repaired != text[start:]:
                try:
                    payload, _ = decoder.raw_decode(repaired)
                    return payload
                except json.JSONDecodeError:
                    pass
            continue
    raise ValueError(f"Failed to extract JSON from model response:\n{raw_text}")


def extract_message_text(message: Any) -> str:
    texts: List[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text = getattr(block, "text", "")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _count_suspicious_markers(text: str) -> int:
    return sum(text.count(marker) for marker in SUSPICIOUS_MOJIBAKE_MARKERS)


def repair_mojibake_text(text: str) -> str:
    if not any(marker in text for marker in SUSPICIOUS_MOJIBAKE_MARKERS):
        return text.translate(TEXT_NORMALIZATION_TABLE)

    best = text
    best_score = _count_suspicious_markers(text)
    for encoding in ("cp936", "gb18030", "gbk"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        candidate_score = _count_suspicious_markers(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score

    return best.translate(TEXT_NORMALIZATION_TABLE)


def decode_model_output(data: bytes, *, source: str) -> str:
    had_decode_error = False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        had_decode_error = True

    repaired = repair_mojibake_text(text)

    if had_decode_error:
        print(
            f"[warn] {source} contained non-UTF-8 bytes; decoded with replacement fallback.",
            file=sys.stderr,
        )
    if repaired != text:
        print(
            f"[warn] Repaired suspicious text artifacts in {source}.",
            file=sys.stderr,
        )

    return repaired


def escape_unescaped_inner_quotes(text: str) -> str:
    result: List[str] = []
    in_string = False
    escaped = False

    for idx, char in enumerate(text):
        if not in_string:
            if char == '"':
                in_string = True
            result.append(char)
            continue

        if escaped:
            result.append(char)
            escaped = False
            continue

        if char == "\\":
            result.append(char)
            escaped = True
            continue

        if char == '"':
            next_idx = idx + 1
            while next_idx < len(text) and text[next_idx] in " \t\r\n":
                next_idx += 1
            if next_idx == len(text) or text[next_idx] in ":,]}":
                in_string = False
                result.append(char)
            else:
                result.append('\\"')
            continue

        result.append(char)

    return "".join(result)


def sanitize_text_artifacts(value: Any) -> Tuple[Any, int]:
    if isinstance(value, str):
        repaired = repair_mojibake_text(value)
        return repaired, int(repaired != value)
    if isinstance(value, list):
        changed = 0
        repaired_list: List[Any] = []
        for item in value:
            repaired_item, item_changed = sanitize_text_artifacts(item)
            repaired_list.append(repaired_item)
            changed += item_changed
        return repaired_list, changed
    if isinstance(value, dict):
        changed = 0
        repaired_dict: Dict[Any, Any] = {}
        for key, item in value.items():
            repaired_item, item_changed = sanitize_text_artifacts(item)
            repaired_dict[key] = repaired_item
            changed += item_changed
        return repaired_dict, changed
    return value, 0


def build_codex_command(model: str) -> List[str]:
    cmd = [
        find_codex_path(),
        "exec",
        "--skip-git-repo-check",
        "-c",
        "features.fast_mode=true",
        "-c",
        "service_tier=fast",
        "-c",
        "reasoning_effort=medium",
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append("--json")
    return cmd


def tail_text(text: str, *, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if not lines:
        return "(empty)"
    return "\n".join(lines[-max_lines:])


def format_codex_failure(stdout_text: str, stderr_text: str) -> str:
    return (
        "Codex failed before producing a final assistant message.\n\n"
        f"Codex stdout tail:\n{tail_text(stdout_text)}\n\n"
        f"Codex stderr tail:\n{tail_text(stderr_text)}"
    )


def is_transient_codex_failure(stdout_text: str, stderr_text: str) -> bool:
    combined = f"{stdout_text}\n{stderr_text}".lower()
    if "invalid_request_error" in combined or "invalid_json_schema" in combined:
        return False
    return (
        "stream disconnected" in combined
        or "no last agent message" in combined
        or "timeout" in combined
        or "temporarily unavailable" in combined
    )


def call_codex(model: str, system_prompt: str, user_prompt: str, *, retries: int = DEFAULT_CODEX_RETRIES) -> str:
    prompt = (
        f"{system_prompt.rstrip()}\n\n{user_prompt}\n\n"
        'For this structured-output call, return a JSON object with exactly one top-level field "annotations". '
        'The value of "annotations" must be the JSON array described above.'
    ).strip()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        prompt_file = handle.name
        handle.write(prompt)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        output_file = handle.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        schema_file = handle.name
        json.dump(CODEX_OUTPUT_SCHEMA, handle, ensure_ascii=False)

    try:
        cmd = build_codex_command(model)
        cmd.extend(["--output-schema", schema_file, "--output-last-message", output_file])
        attempts = max(1, retries)
        for attempt in range(1, attempts + 1):
            Path(output_file).write_text("", encoding="utf-8")
            with open(prompt_file, "r", encoding="utf-8") as prompt_handle:
                result = subprocess.run(
                    cmd,
                    stdin=prompt_handle,
                    capture_output=True,
                )

            stdout_text = decode_model_output(result.stdout, source="Codex stdout")
            stderr_text = decode_model_output(result.stderr, source="Codex stderr")
            final_content = Path(output_file).read_text(encoding="utf-8").strip()

            if result.returncode == 0 and final_content:
                final_content = repair_mojibake_text(final_content)
                final_payload = extract_json_payload(final_content)
                if isinstance(final_payload, dict) and isinstance(final_payload.get("annotations"), list):
                    return json.dumps(final_payload["annotations"], ensure_ascii=False, indent=2)
                if isinstance(final_payload, list):
                    return json.dumps(final_payload, ensure_ascii=False, indent=2)
                raise RuntimeError(f"Codex final message did not contain annotations: {final_content}")

            if attempt < attempts and is_transient_codex_failure(stdout_text, stderr_text):
                print(
                    f"[warn] Codex transient failure on attempt {attempt}/{attempts}; retrying...",
                    file=sys.stderr,
                )
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            raise RuntimeError(format_codex_failure(stdout_text, stderr_text))
    finally:
        os.unlink(prompt_file)
        os.unlink(schema_file)
        os.unlink(output_file)


def call_minimax_review(
    *,
    review_payload: Dict[str, Any],
    model: str,
) -> Tuple[Dict[str, Any], str]:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()
    load_env_file(Path(".env"))

    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for MiniMax review.") from exc

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set, so MiniMax review cannot run.")

    base_url = os.getenv("ANTHROPIC_BASE_URL")
    client = anthropic.Anthropic(base_url=base_url) if base_url else anthropic.Anthropic()

    system_prompt = (
        "You are a strict editor for GUI trajectory annotations. "
        "Focus only on the steps flagged as anomalous because screenshots are missing. "
        "Check whether the generated fields are noticeably shorter or weaker than the normal steps. "
        "Only expand fields when needed; otherwise keep them unchanged. Return JSON only."
    )
    user_prompt = (
        "Review the generated annotations for anomalous steps.\n\n"
        "Rules:\n"
        "1. Only review the flagged anomalous steps.\n"
        "2. The main target is to avoid under-detailed text caused by missing screenshots.\n"
        "3. If a field is already comparable in detail to the provided normal examples, leave it unchanged.\n"
        "4. If a field is too short, vague, or weaker than the normal examples, expand it while staying faithful to the available context.\n"
        "5. Never invent impossible UI changes.\n"
        "6. Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "summary": "short summary",\n'
        '  "revised_steps": [\n'
        "    {\n"
        '      "step_id": "s1",\n'
        '      "changes": {\n'
        '        "action_preconditions": ["..."],\n'
        '        "action_before_state": "...",\n'
        '        "action_after_effects": "...",\n'
        '        "nl_explanation": "..."\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Review payload:\n{json.dumps(review_payload, ensure_ascii=False, indent=2)}"
    )

    message = client.messages.create(
        model=model,
        max_tokens=12000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        ],
    )
    response_text = extract_message_text(message)
    parsed = extract_json_payload(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("MiniMax review did not return a JSON object.")
    return parsed, response_text


def resolve_report_path(conversation_folder: Path, json_file: str) -> Path:
    report_path = (conversation_folder / json_file).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {report_path}")
    return report_path


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_artifact_path(report_path: Path, rel_path: Optional[str]) -> Optional[Path]:
    if not rel_path:
        return None
    rel = Path(rel_path)
    candidates = [
        report_path.parent / rel,
        report_path.parent.parent / rel,
        Path.cwd() / rel,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def normalize_action_type(step: Dict[str, Any]) -> str:
    return str(step.get("action", {}).get("type", "")).lower()


def infer_primary_app(report: Dict[str, Any]) -> str:
    app = report.get("app")
    if isinstance(app, list):
        apps = [str(item).strip() for item in app if str(item).strip()]
        if apps:
            return ", ".join(apps)
    if isinstance(app, str) and app.strip():
        return app.strip()
    env = report.get("env")
    if isinstance(env, dict):
        env_app = str(env.get("app") or "").strip()
        if env_app:
            return env_app
    return "the target application"


def normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result
    return []


def build_step_diagnostics(report_path: Path, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for step in report.get("steps", []):
        now_state = step.get("now_state", {})
        before_rel = now_state.get("screenshot_path_before")
        before_part_rel = now_state.get("screenshot_path_before_part")
        after_rel = now_state.get("screenshot_path_after")
        action_type = normalize_action_type(step)

        before_exists = resolve_artifact_path(report_path, before_rel) is not None
        before_part_exists = resolve_artifact_path(report_path, before_part_rel) is not None
        after_exists = resolve_artifact_path(report_path, after_rel) is not None

        flags: List[str] = []
        if before_rel and not before_exists:
            flags.append("missing_before_screenshot")
        if before_part_rel and not before_part_exists:
            flags.append("missing_before_part_screenshot")
        if after_rel and not after_exists:
            flags.append("missing_after_screenshot")
        if not before_rel:
            flags.append("before_path_empty")
        if not before_part_rel:
            flags.append("before_part_path_empty")
        if not after_rel:
            flags.append("after_path_empty")
        if action_type in POINTER_ACTION_TYPES and not before_part_exists:
            flags.append("pointer_step_without_reliable_before_part")

        diagnostics.append(
            {
                "step_id": step.get("step_id"),
                "action_type": action_type,
                "before_path": before_rel,
                "before_part_path": before_part_rel,
                "after_path": after_rel,
                "before_exists": before_exists,
                "before_part_exists": before_part_exists,
                "after_exists": after_exists,
                "flags": flags,
            }
        )
    return diagnostics


def build_system_prompt(json_filename: str, primary_app: str) -> str:
    return (
        "You are an action-behavior analyst and recorder.\n"
        "\n"
        "Your task:\n"
        f"Analyze one trajectory JSON file named {json_filename} that records a single task containing multiple UI actions, together with its screenshots. "
        "For each action (step) you must infer structured descriptions strictly from the JSON metadata and screenshots, and return a machine-readable JSON array.\n"
        "For each step, analyze the current before screenshot as the pre-action state, including the current before(part) screenshot if it exists, and analyze both the current after screenshot and the next step's before screenshot as evidence of the post-action state whenever the next before screenshot exists.\n"
        "If the current after screenshot appears transitional, partially loaded, or inconsistent, prefer the next step's before screenshot as the more reliable reference for the stabilized result.\n"       # 这两行是新加的，引导读取before, after, next_before三个截图
        "\n"
        f"The target application for this trajectory is: {primary_app}.\n"
        "You must stay tightly focused on this target application only.\n"
        "If a screenshot contains other irrelevant content such as a terminal window, desktop wallpaper, dock, background apps, notifications, or unrelated windows, ignore them unless they directly affect the target application's current interaction.\n"
        "Do not describe irrelevant apps or background content in action_preconditions, action_before_state, action_after_effects, nl_position, or nl_explanation.\n"
        "\n"
        "For each step you MUST produce one JSON object with exactly these fields:\n"
        '- \"step_goal\": A short phrase describing the immediate goal of this step within the overall task.\n'
        '- \"app\": The software/application used during the task.\n'
        '- \"action_preconditions\": What must already be true before the action occurs.\n'
        '- \"nl_position\": A natural-language description of the actual targeted UI element or mouse location (based on the red marker in the before screenshot). '
        'If the step has no on-screen target (for example, a typing or press action where "action.target" is missing or an empty object in report.json), set this field to null instead of describing any location. If you cannot confidently identify what the element is or what text it contains, instead describe its visual appearance (shape, color, approximate size) and relative location (for example, "a blue rectangular button near the top-right corner").\n'
        # 'If the step has no on-screen target, set this field to null.\n'
        '- \"action_before_state\": The UI state or condition before the action.\n'
        '- \"action_after_effects\": The changes caused by the action, based on the current step\'s after screenshot and, when available, the next step\'s before screenshot. '
        'If the after screenshot appears transitional, not fully loaded, or otherwise unreliable, use the next step\'s before screenshot as the primary reference for the stabilized result. IMPORTANT NOTES:\n'
        '  * If clicking on empty/blank area but the state barely changes, this could be to CONFIRM whether the previous operation succeeded, or to DESELECT/cancel the current selection.\n'
        '  * Watch for misclicks or invalid actions - if the click seems to have no purpose or effect, it might be an accidental click, clicking wrong element, or clicking on wrong page. Please identify and note these cases appropriately.\n'
        '- \"nl_explanation\": A concise natural-language explanation of the action and its purpose. When explaining, consider whether the action might be a confirmation check, deselection, or a misclick, and whether the result is better confirmed by the next step\'s before screenshot.\n'
        "\n"
        "Special handling rules:\n"
        # "1. If a step is missing its before or after screenshot, do not fail. Infer the missing screenshot-dependent fields from the step metadata, "  # 现在没有这个问题了？
        # "task context, neighboring steps, window titles, URLs, and any available screenshots.\n"
        # "2. If a click or drag step does not have a reliable before-part screenshot with the red marker, treat the raw numeric position as possibly wrong. "
        # "Inspect the full before screenshot instead and infer the actual clicked or dragged UI element visually.\n"
        # "3. For such pointer steps, nl_position and nl_explanation must describe the real target seen in the screenshot rather than blindly copying the provided coordinates.\n"
        "1. If a click seems to confirm success, deselect something, or is a likely misclick, describe that explicitly in action_after_effects and nl_explanation.\n"
        "2. Keep outputs concrete and sufficiently detailed. Avoid weak text such as 'page changes' or 'something is selected' when a more grounded description is possible.\n\n"
        "Reference detail example:\n"
        'Too short: "The page changes."\n'
        'Acceptable: "Chrome navigates to the Natural Product Information page, where a search box and an A-Z list of natural products are visible near the top of the page."\n\n'
        "Output format requirements (very important):\n"
        "- The FINAL answer must be a single JSON array (e.g. [ { ... }, { ... }, ... ]) with one object per action.\n"
        "- Do not print any explanations, comments, or non-JSON text in the final answer.\n"
        "- Do not include trailing commas. The JSON must be strictly valid.\n"
        "- The array length must match the number of steps.\n\n"
        "IMPORTANT: Your output must be in ENGLISH.\n"
    )

def build_user_prompt(
    *,
    conversation_folder: Path,
    json_filename: str,
    report: Dict[str, Any],
    diagnostics: Sequence[Dict[str, Any]],
) -> str:
    anomalous = [entry for entry in diagnostics if entry["flags"]]
    primary_app = infer_primary_app(report)
    return (
        f"You are given a conversation folder located at:\n{conversation_folder}\n\n"
        f"Inside this folder there is a {json_filename} JSON file and a screenshots/ subfolder referenced by it.\n\n"
        # f"The target JSON file to analyze is:\n{json_filename}\n\n"
        "Your job now:\n"
        # "Read the target JSON file and inspect the screenshots referenced by each step.\n\n"
        # "When filling each step, use:\n"
        "1) Read the target JSON file in that folder.\n"
        f"2) For each step in {json_filename}.steps, carefully inspect:\n"
        "   - The step metadata in the JSON, including any screenshot paths (such as screenshot_path_before_part).\n"
        "   - The current step's before screenshot and before-part screenshot when available, for the pre-action state, targeted element, and pointer position and so on.\n"
        "   - The current step's after screenshot and the next step's before screenshot when available, for the post-action state and so on.\n"
        "   - If `after` looks transitional, partially loaded, delayed, or inconsistent, prefer `next_before` as the more reliable stable result.\n"
        "   - The red-highlighted mouse position.\n"
        "   - Any relevant application and URL information\n"
        # "   - Neighboring steps before and after the current step when screenshots are missing.\n\n"
        "3) Then produce ONE JSON array as the final answer. Each element in the array corresponds to one step, "
        "and must contain the fields described in the system prompt: step_goal, app, "
        "action_preconditions, nl_position, action_before_state, action_after_effects, nl_explanation.\n"
        "\n"
        "Note:\n"
        f"  - The current trajectory should be interpreted only with respect to this application: {primary_app}.\n"
        "   - When a screenshot includes unrelated UI outside the target application, treat that content as noise and ignore it. "
        "For example: if the target app is Chrome but part of the screenshot also shows a terminal, do not mention the terminal unless the step is truly interacting with it.\n\n"
        "Important anomaly summary for this file:\n"
        f"{json.dumps(anomalous, ensure_ascii=False, indent=2)}\n\n"
        "Remember: the final answer must be ONLY that JSON array, with no extra commentary or text."
    )


def build_minimax_review_payload(
    *,
    report: Dict[str, Any],
    diagnostics: Sequence[Dict[str, Any]],
    ai_data: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    anomalous_steps: List[Dict[str, Any]] = []
    normal_examples: List[Dict[str, Any]] = []

    for step, diag, generated in zip(report.get("steps", []), diagnostics, ai_data):
        if not isinstance(generated, dict):
            continue
        entry = {
            "step_id": step.get("step_id"),
            "action_type": normalize_action_type(step),
            "flags": diag["flags"],
            "generated_fields": {
                "action_preconditions": normalize_text_list(generated.get("action_preconditions")),
                "action_before_state": generated.get("action_before_state"),
                "action_after_effects": generated.get("action_after_effects"),
                "nl_explanation": generated.get("nl_explanation"),
            },
            "context": {
                "step_goal": generated.get("step_goal"),
                "url": generated.get("url"),
                "app_title_before": step.get("now_state", {}).get("app_title_before"),
                "app_title_after": step.get("now_state", {}).get("app_title_after"),
            },
        }
        if diag["flags"] and (
            "missing_before_screenshot" in diag["flags"]
            or "missing_after_screenshot" in diag["flags"]
            or "before_path_empty" in diag["flags"]
            or "after_path_empty" in diag["flags"]
        ):
            anomalous_steps.append(entry)
        elif len(normal_examples) < 4:
            normal_examples.append(entry)

    if not anomalous_steps:
        return None

    return {
        "task_title": report.get("task_title"),
        "instruction": report.get("instruction"),
        "normal_reference_examples": normal_examples,
        "anomalous_steps": anomalous_steps,
    }


def maybe_run_minimax_review(
    *,
    review_mode: str,
    review_payload: Optional[Dict[str, Any]],
    model: str,
    raw_output_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if review_mode == "off" or review_payload is None:
        return None
    try:
        parsed, raw_text = call_minimax_review(review_payload=review_payload, model=model)
    except Exception as exc:
        if review_mode == "on":
            raise
        print(f"[warn] MiniMax review skipped: {exc}", file=sys.stderr)
        return None

    if raw_output_path is not None:
        save_text(raw_output_path, raw_text)
    return parsed


def apply_minimax_revisions(
    ai_data: List[Dict[str, Any]],
    minimax_review: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(minimax_review, dict):
        return
    revisions = minimax_review.get("revised_steps")
    if not isinstance(revisions, list):
        return

    by_step_id = {
        item.get("step_id"): item.get("changes")
        for item in revisions
        if isinstance(item, dict) and isinstance(item.get("changes"), dict)
    }

    for action_info in ai_data:
        if not isinstance(action_info, dict):
            continue
        step_id = action_info.get("step_id")
        changes = by_step_id.get(step_id)
        if not isinstance(changes, dict):
            continue
        for field_name, value in changes.items():
            if field_name == "action_preconditions":
                normalized = normalize_text_list(value)
                if normalized:
                    action_info[field_name] = normalized
            elif isinstance(value, str) and value.strip():
                action_info[field_name] = value.strip()


def merge_ai_into_report(report: Dict[str, Any], ai_data: Sequence[Dict[str, Any]]) -> None:
    steps = report.get("steps", [])

    if ai_data:
        first = ai_data[0]
        if isinstance(first, dict):
            if not report.get("task_title") and first.get("task_title"):
                report["task_title"] = first["task_title"]
            if not report.get("app") and first.get("app"):
                report["app"] = first["app"]
            if first.get("url"):
                report.setdefault("env", {})
                if not report["env"].get("url"):
                    report["env"]["url"] = first["url"]

    for idx, step in enumerate(steps):
        if idx >= len(ai_data):
            break
        action_info = ai_data[idx]
        if not isinstance(action_info, dict):
            continue

        if step.get("step_goal") in (None, "") and action_info.get("step_goal"):
            step["step_goal"] = action_info["step_goal"]

        preconditions = normalize_text_list(action_info.get("action_preconditions"))
        if step.get("action_preconditions") in (None, [], "") and preconditions:
            step["action_preconditions"] = preconditions

        action = step.get("action")
        target = action.get("target") if isinstance(action, dict) else None
        if isinstance(target, dict) and "position" in target:
            nl_position = target.get("nl_position")
            if nl_position in (None, [], "") and action_info.get("nl_position"):
                target["nl_position"] = [action_info["nl_position"]]

        if step.get("action_before_state") in (None, "") and action_info.get("action_before_state"):
            step["action_before_state"] = action_info["action_before_state"]

        after_effects = normalize_text_list(action_info.get("action_after_effects"))
        if step.get("action_after_effects") in (None, [], "") and after_effects:
            step["action_after_effects"] = after_effects

        if step.get("nl_explanation") in (None, "") and action_info.get("nl_explanation"):
            step["nl_explanation"] = action_info["nl_explanation"]


def ensure_step_ids(ai_data: List[Dict[str, Any]], report: Dict[str, Any]) -> None:
    for generated, step in zip(ai_data, report.get("steps", [])):
        if isinstance(generated, dict) and not generated.get("step_id"):
            generated["step_id"] = step.get("step_id")


def main() -> None:
    args = parse_args()
    conversation_folder = args.conversation_folder.strip() if args.conversation_folder else prompt_for_folder()
    if not conversation_folder:
        conversation_folder = os.getcwd()

    folder_path = Path(conversation_folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Conversation folder does not exist: {folder_path}")

    report_path = resolve_report_path(folder_path, args.json_file)
    report = load_report(report_path)
    diagnostics = build_step_diagnostics(report_path, report)
    primary_app = infer_primary_app(report)

    system_prompt = build_system_prompt(report_path.name, primary_app)
    user_prompt = build_user_prompt(
        conversation_folder=folder_path,
        json_filename=report_path.name,
        report=report,
        diagnostics=diagnostics,
    )

    codex_raw = call_codex(args.model, system_prompt, user_prompt, retries=args.codex_retries)
    if args.save_raw:
        save_text(report_path.with_suffix(report_path.suffix + ".codex_0413.raw.txt"), codex_raw)

    ai_data = extract_json_payload(codex_raw)
    ai_data, sanitized_count = sanitize_text_artifacts(ai_data)
    if sanitized_count:
        print(
            f"[warn] Sanitized {sanitized_count} suspicious text value(s) in Codex output.",
            file=sys.stderr,
        )
    if not isinstance(ai_data, list):
        raise RuntimeError(f"Expected a JSON array of step annotations, got: {type(ai_data).__name__}")
    ensure_step_ids(ai_data, report)

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
            report_path.with_suffix(report_path.suffix + ".minimax_0413.raw.txt") if args.save_raw else None
        ),
    )
    minimax_review, review_sanitized_count = sanitize_text_artifacts(minimax_review)
    if review_sanitized_count:
        print(
            f"[warn] Sanitized {review_sanitized_count} suspicious text value(s) in MiniMax review output.",
            file=sys.stderr,
        )
    apply_minimax_revisions(ai_data, minimax_review)

    merge_ai_into_report(report, ai_data)
    report["fill_meta_0413"] = {
        "source_json_file": report_path.name,
        "codex_model": args.model,
        "minimax_review_mode": args.minimax_review_mode,
        "minimax_model": args.minimax_model if minimax_review is not None else None,
        "anomalous_steps": [entry for entry in diagnostics if entry["flags"]],
    }

    if args.output_json:
        output_candidate = Path(args.output_json).expanduser()
        output_path = (
            output_candidate.resolve()
            if output_candidate.is_absolute()
            else (report_path.parent / output_candidate).resolve()
        )
    else:
        output_path = report_path
    save_report(output_path, report)
    print(f"Filled missing fields in: {output_path}")

if __name__ == "__main__":
    main()
