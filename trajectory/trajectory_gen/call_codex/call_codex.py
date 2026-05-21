import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple


SUSPICIOUS_MOJIBAKE_MARKERS = (
    "\u9225",
    "\u95b3",
    "\u00e2\u20ac",
    "\u00c3\u00a2",
    "\u20ac\u2122",
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


def _find_codex_path() -> str:
    # return "codex"
    return r"D:\Program Files\nodejs\node_global\codex.cmd"


def _build_codex_command(model: Optional[str], json_mode: bool = True) -> List[str]:
    cmd = [
        _find_codex_path(),
        "exec",
        "--skip-git-repo-check",
        "-c",
        "features.fast_mode=true",
        "-c",
        "service_tier=fast",
        "-c",
        "reasoning_effort=medium",
    ]
    if json_mode:
        cmd.append("--json")
    if model:
        cmd.extend(["--model", model])
    return cmd


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


def decode_codex_output(data: bytes, *, source: str) -> str:
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


def call_codex(model: str, system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to Codex and return the final assistant message."""

    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        prompt_file = handle.name
        handle.write(prompt)

    try:
        cmd = _build_codex_command(model, json_mode=True)
        with open(prompt_file, "r", encoding="utf-8") as prompt_handle:
            result = subprocess.run(
                cmd,
                stdin=prompt_handle,
                capture_output=True,
            )
    finally:
        os.unlink(prompt_file)

    stdout_text = decode_codex_output(result.stdout, source="Codex stdout")
    stderr_text = decode_codex_output(result.stderr, source="Codex stderr")

    if result.returncode != 0:
        stderr_msg = stderr_text if stderr_text else stdout_text
        raise RuntimeError(f"Codex error: {stderr_msg}")

    last_content: Optional[str] = None
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue

        if event.get("type") in ("item.started", "item.completed"):
            item = event.get("item", {})
            item_type = item.get("type") or item.get("item_type")
            if item_type in ("assistant_message", "agent_message"):
                content = item.get("text") or item.get("content")
                if isinstance(content, list):
                    content = "".join(
                        c if isinstance(c, str) else c.get("text", "")
                        for c in content
                    )
                last_content = content

    if last_content is not None:
        return last_content

    stdout_tail = "\n".join(stdout_text.splitlines()[-30:]) or "(empty)"
    stderr_tail = "\n".join(stderr_text.splitlines()[-30:]) or "(empty)"
    raise RuntimeError(
        "Codex did not return an assistant message.\n\n"
        f"Codex stdout tail:\n{stdout_tail}\n\n"
        f"Codex stderr tail:\n{stderr_tail}"
    )


def call_codex_streaming(model: str, system_prompt: str, user_prompt: str) -> None:
    """
    Stream Codex output directly to the terminal, similar to running codex exec manually.
    """

    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()
    cmd = _build_codex_command(model, json_mode=False)
    cmd.append(prompt)

    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError("Codex exited with non-zero status")


if __name__ == "__main__":
    model = "gpt-5.5"

    if len(sys.argv) > 1:
        conversation_folder = sys.argv[1].strip()
    else:
        conversation_folder = input(
            "Please input the conversation folder path (where report.json and screenshots are located): "
        ).strip()

    if not conversation_folder:
        conversation_folder = os.getcwd()

    report_path = os.path.join(conversation_folder, "report.json")

    # system_prompt = (
    #     "You are an action-behavior analyst and recorder.\n"
    #     "\n"
    #     "Your task:\n"
    #     "Analyze a report.json file that records a single task containing multiple UI actions. "
    #     "For each action (step) you must infer a structured description strictly from the JSON metadata and screenshots, "
    #     "and output a machine-readable JSON array that can be used to fill the empty fields in report.json.\n\n"
    #     "For each action/step you MUST produce one JSON object with exactly these fields:\n"
    #     '- "task_title": A concise title summarizing the overall task (same value for all actions).\n'
    #     '- "step_goal": A short phrase describing the immediate goal of this specific action within the overall task.\n'
    #     '- "app": The software/application used during the task.\n'
    #     '- "url": Any URL relevant to the task or the specific action.\n'
    #     '- "action_preconditions": What must be true or present before the action occurs (based on the before screenshot).\n'
    #     '- "nl_position": A natural-language description of the mouse location or targeted UI element (based on the red marker in the before screenshot). If the step has no on-screen target (for example, a typing or press action where "action.target" is missing or an empty object in report.json), set this field to null instead of describing any location. If you cannot confidently identify what the element is or what text it contains, instead describe its visual appearance (shape, color, approximate size) and relative location (for example, "a blue rectangular button near the top-right corner").\n'
    #     '- "action_before_state": The UI state or condition before the action.\n'
    #     '- "action_after_effects": The changes caused by the action (based on the after screenshot). IMPORTANT NOTES:\n'
    #     '  * If clicking on empty/blank area but the state barely changes, this could be to CONFIRM whether the previous operation succeeded, or to DESELECT/cancel the current selection.\n'
    #     '  * Watch for misclicks or invalid actions - if the click seems to have no purpose or effect, it might be an accidental click, clicking wrong element, or clicking on wrong page. Please identify and note these cases appropriately.\n'
    #     '- "nl_explanation": A concise, natural-language explanation of the action and its purpose, written without referring to "the user" (describe the step itself, for example, "Click the Save button to store the changes."). When explaining, consider whether the action might be a confirmation check, deselection, or a misclick.\n\n'
    #     "Output format requirements (very important):\n"
    #     "- The FINAL answer must be a single JSON array (e.g. [ { ... }, { ... }, ... ]) with one object per action.\n"
    #     "- Do not print any explanations, comments, or non-JSON text in the final answer.\n"
    #     "- Do not include trailing commas. The JSON must be strictly valid.\n"
    # )

    # user_prompt = (
    #     "You are given a conversation folder located at:\n"
    #     f"{conversation_folder}\n\n"
    #     "Inside this folder there is a report.json file and a screenshots/ subfolder referenced by it.\n\n"
    #     "Your job now:\n"
    #     "1) Read report.json in that folder.\n"
    #     "2) For each step in report.json.steps, carefully inspect:\n"
    #     "   - The overall task instruction or user prompt in report.json (for example, the \"instruction\" field)\n"
    #     "   - Its metadata in the JSON, including any screenshot paths (such as screenshot_path_before_part)\n"
    #     "   - The before screenshot, the partial before screenshot near the signed position and the after screenshot\n"
    #     "   - The red-highlighted mouse position\n"
    #     "   - Any relevant application and URL information\n"
    #     "3) Then produce ONE JSON array as the final answer. Each element in the array corresponds to one step, "
    #     "and must contain the fields described in the system prompt: task_title, step_goal, app, url, "
    #     "action_preconditions, nl_position, action_before_state, action_after_effects, nl_explanation.\n\n"
    #     "Remember: the final answer must be ONLY that JSON array, with no extra commentary or text."
    # )

    system_prompt = (
        "You are an action-behavior analyst and recorder.\n"
        "\n"
        "Your task:\n"
        "Analyze a report.json file that records a single task containing multiple UI actions. "
        "For each action (step) you must infer a structured description strictly from the JSON metadata and screenshots, "
        "and output a machine-readable JSON array that can be used to fill the empty fields in report.json.\n\n"
        "For each action/step you MUST produce one JSON object with exactly these fields:\n"
        # '- "task_title": A concise title summarizing the overall task (same value for all actions).\n'
        '- "step_goal": A short phrase describing the immediate goal of this specific action within the overall task.\n'
        '- "app": The software/application used during the task.\n'
        # '- "url": Any URL relevant to the task or the specific action.\n'
        '- "action_preconditions": What must be true or present before the action occurs (based on the before screenshot).\n'
        '- "nl_position": A natural-language description of the mouse location or targeted UI element (based on the red marker in the before screenshot if exists). If the step has no on-screen target (for example, a typing or press action where "action.target" is missing or an empty object in report.json), set this field to null instead of describing any location. If you cannot confidently identify what the element is or what text it contains, instead describe its visual appearance (shape, color, approximate size) and relative location (for example, "a blue rectangular button near the top-right corner").\n'
        '- "action_before_state": The UI state or condition before the action.\n'
        '- "action_after_effects": The changes caused by the action (based on the after screenshot). IMPORTANT NOTES:\n'
        '  * If clicking on empty/blank area but the state barely changes, this could be to CONFIRM whether the previous operation succeeded, or to DESELECT/cancel the current selection.\n'
        '  * Watch for misclicks or invalid actions - if the click seems to have no purpose or effect, it might be an accidental click, clicking wrong element, or clicking on wrong page. Please identify and note these cases appropriately.\n'
        '- "nl_explanation": A concise, natural-language explanation of the action and its purpose, written without referring to "the user" (describe the step itself, for example, "Click the Save button to store the changes."). When explaining, consider whether the action might be a confirmation check, deselection, or a misclick.\n\n'
        "Output format requirements (very important):\n"
        "- The FINAL answer must be a single JSON array (e.g. [ { ... }, { ... }, ... ]) with one object per action.\n"
        "- Do not print any explanations, comments, or non-JSON text in the final answer.\n"
        "- Do not include trailing commas. The JSON must be strictly valid.\n"
    )
    # 修改了 2) 第三条
    user_prompt = (
        "You are given a conversation folder located at:\n"
        f"{conversation_folder}\n\n"
        "Inside this folder there is a report.json file and a screenshots/ subfolder referenced by it.\n\n"
        "Your job now:\n"
        "1) Read report.json in that folder.\n"
        "2) For each step in report.json.steps, carefully inspect:\n"
        "   - The overall task instruction or user prompt in report.json (for example, the \"instruction\" field)\n"
        "   - Its metadata in the JSON, including any screenshot paths (such as screenshot_path_before_part)\n"
        "   - The before screenshot, the partial before screenshot near the signed position, the after screenshot and next step's before screenshot (if it exists)\n"
        "   - The red-highlighted mouse position\n"
        "   - Any relevant application and URL information\n"
        "3) Then produce ONE JSON array as the final answer. Each element in the array corresponds to one step, "
        "and must contain the fields described in the system prompt: task_title, step_goal, app, url, "
        "action_preconditions, nl_position, action_before_state, action_after_effects, nl_explanation.\n\n"
        "Remember: the final answer must be ONLY that JSON array, with no extra commentary or text."
    )

    ai_response = call_codex(model, system_prompt, user_prompt)

    try:
        ai_data = json.loads(ai_response)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse model response as JSON:\n{ai_response}")

    ai_data, sanitized_count = sanitize_text_artifacts(ai_data)
    if sanitized_count:
        print(
            f"[warn] Sanitized {sanitized_count} suspicious text value(s) in Codex output.",
            file=sys.stderr,
        )

    if not isinstance(ai_data, list):
        raise RuntimeError(f"Expected a JSON array of actions, got: {type(ai_data).__name__}")

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"report.json not found at: {report_path}")

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    steps = report.get("steps", [])

    if ai_data:
        first = ai_data[0]
        if isinstance(first, dict):
            if not report.get("task_title") and first.get("task_title"):
                report["task_title"] = first["task_title"]
            if not report.get("app") and first.get("app"):
                report["app"] = first["app"]
            url_val = first.get("url")
            if url_val:
                report.setdefault("env", {})
                if not report["env"].get("url"):
                    report["env"]["url"] = url_val

    for idx, step in enumerate(steps):
        if idx >= len(ai_data):
            break
        action_info = ai_data[idx]
        if not isinstance(action_info, dict):
            continue

        if step.get("step_goal") in (None, "") and action_info.get("step_goal"):
            step["step_goal"] = action_info["step_goal"]

        if step.get("action_preconditions") in (None, [], "") and action_info.get("action_preconditions"):
            step["action_preconditions"] = [action_info["action_preconditions"]]

        action = step.get("action")
        target = action.get("target") if isinstance(action, dict) else None
        if isinstance(target, dict) and "position" in target:
            nl_pos = target.get("nl_position")
            if nl_pos in (None, [], "") and action_info.get("nl_position"):
                target["nl_position"] = [action_info["nl_position"]]

        if step.get("action_before_state") in (None, "") and action_info.get("action_before_state"):
            step["action_before_state"] = action_info["action_before_state"]

        if step.get("action_after_effects") in (None, [], "") and action_info.get("action_after_effects"):
            step["action_after_effects"] = [action_info["action_after_effects"]]

        if step.get("nl_explanation") in (None, "") and action_info.get("nl_explanation"):
            step["nl_explanation"] = action_info["nl_explanation"]

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"report.json filled where empty at: {report_path}")
