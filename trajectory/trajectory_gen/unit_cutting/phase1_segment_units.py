import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

LOG_FILE_PATH: Optional[Path] = None

MOUSE_ACTION_TYPES = {
    "click",
    "double_click",
    "drag",
    "drag_to",
    "mouse_move",
    "move",
    "scroll",
}


SYSTEM_PROMPT = """You are an expert at analyzing GUI interaction trajectories and segmenting them into short reusable units.

Your task is to jointly perform:
1. Segmentation: decide which consecutive steps should be grouped into one unit.
2. Concrete annotation: generate unit-level structured semantic descriptions.

These two subproblems must be solved together, not separately, because the grouping boundary depends on semantic understanding of the whole local interaction.

A unit is a short, coherent, reusable interaction fragment composed of one or more consecutive steps.

A good unit should satisfy the following principles:

[Segmentation Principles]
- Transitional states should be merged:
  If an intermediate state is only a transient operational state rather than a stable state that users care about, it should stay inside the same unit.

- Co-occurring operations should be merged:
  If several actions usually appear together and splitting them apart would make them lose independent meaning, they should be grouped into one unit.

- Independent operations should stay independent:
  If a single step already expresses a complete semantic action with clear meaning, it can remain as a single-step unit. (e.g. "Ctrl+S" to save the change.)

- Intent is a sub-goal, not an action paraphrase:
  unit_intent must describe the deeper user sub-goal (like "update shipping address"), not surface motor actions (e.g. "type in the text box").

- Prefer short units:
  Prefer 1-4 steps per unit.
  5-6 steps are allowed only when the steps are still tightly coupled and clearly form one micro-goal.
  Do not create overly long units.

- Boundaries should be placed when one of these changes:
  1. the micro-goal changes
  2. the main operated object changes
  3. the interaction mode changes
  4. a stable reusable result has already been achieved

[Annotation Principles]
For each unit, describe:
- step_indices: step numbers included in this unit
- unit_before_state: the concrete UI state before the unit
- unit_precondition: what must already be true before the unit can be executed
- unit_after_state: the concrete UI state after the unit
- unit_effect: the observable effect / verification signal
- unit_type: the semantic unit type
- unit_intent: the deep sub-goal (intent)

Important constraints:
- Units must cover all steps exactly once, in order, with no overlap and no gaps.
- unit_before_state and unit_after_state must be concrete, screen-grounded natural language descriptions.
- unit_precondition must be actionable and verifiable.
- unit_effect must describe observable results or verification signals, not vague statements.
- Return only valid JSON, Do NOT output markdown, Do NOT output explanations outside the JSON.

IMPORTANT: Your output must be in ENGLISH.
"""


USER_PROMPT_TEMPLATE = """Please analyze the following segment and output only the segmented and concretely annotated units.

Input assumptions:
- The input is one segment with consecutive steps.
- The steps are already filtered and ordered.
- You must jointly decide segmentation boundaries and generate unit-level semantic annotations.

Output example:
Return only a JSON object with this schema:

{
  "units": [
    {
      "unit_id": "u_seg_001_01",
      "step_indices": [1, 2, 3],
      "unit_intent": "Change the upper headline text color to yellow using a custom hex color.",
      "unit_type": "APPLY_CUSTOM_TEXT_COLOR",
      "unit_before_state": "LibreOffice Impress is showing slide 1 with the upper headline visible in its original light text color. The headline text box can be activated for editing, and the Properties sidebar contains the character color controls needed to modify the text color.",
      "unit_after_state": "The upper headline text on slide 1 is updated to yellow after confirming the custom color value, and the slide returns to the main editing view with the new color applied.",
      "unit_precondition": [
        "The address entry page has been reached.",
        "The input field is active."
      ],
      "unit_effect": [
        "The Pick a Color dialog opens and shows the current title color values, including hex 5f77bb."
      ]
    }
  ]
}

Additional rules:
- unit_type is a semantic abstraction such as:
  "REPLACE_CONTENT", "OPEN_CUSTOM_COLOR_DIALOG", "APPLY_CUSTOM_TEXT_COLOR",
  "SWITCH_SLIDE", "EXIT_EDITING_STATE", "SAVE_DOCUMENT"
- unit_intent must describe what the user is trying to achieve at the sub-goal level.
- If a step is just a transition or refocus step with little standalone meaning, merge it into its surrounding unit if it serves the same micro-goal.
- If a single step creates a new working context or completes an independent action, it may remain as a single-step unit.
- Prefer short, semantically clean, reusable units.

Reference segmentation style example:
- [select all -> delete -> type new content] should usually be grouped as one content-replacement unit.
- [switch to slide 2] can remain a single-step unit.
- [click object -> click blank area to deselect] can be grouped as one return-to-stable-canvas unit.
- [save document] usually remains a single-step unit.

Boundary example for a LibreOffice Impress editing segment:
- s1-s3: locate the slide 1 title color entry
- s4-s6: select and copy or replace the hex value inside the color dialog
- s7: close or confirm the color dialog if it is an independent completion step
- s8: switch to slide 2 if this creates a new working context
- s9-s12: clear the old title and enter the new title
- s13-s18: select the new title, open custom color, change hex value, and confirm application
- s19-s20: exit text/object selection and return to a stable canvas state
- s21: save the file

Now process this segment JSON and output only the "units" JSON object:

<SEGMENT_JSON>
{segment_json_here}
</SEGMENT_JSON>"""


LOW_SIGNAL_INTENTS = (
    "click the button",
    "click button",
    "press enter",
    "press the button",
    "type text",
    "input text",
    "enter text",
    "click the text box",
    "click the input field",
    "select the text box",
    "select the field",
    "type in the text box",
)


def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def log_with_timestamp(message: str) -> None:
    line = f"[{format_timestamp(datetime.now())}] {message}"
    print(line, flush=True)
    if LOG_FILE_PATH is not None:
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def configure_log_file(log_file: str) -> None:
    global LOG_FILE_PATH
    LOG_FILE_PATH = Path(log_file).resolve() if log_file else None


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


def natural_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return (int(match.group(1)), path.name.lower())
    return (10**9, path.name.lower())


def normalize_tasks_subdir(tasks_subdir: str) -> Path:
    normalized = Path(tasks_subdir.strip().strip("/\\"))
    if not normalized.parts:
        raise ValueError("--tasks-subdir must not be empty")
    if normalized.is_absolute():
        raise ValueError("--tasks-subdir must be a relative subdirectory path")
    return normalized


def resolve_session_paths(input_path: Path, tasks_subdir: Path) -> Tuple[Path, Path]:
    if input_path.is_dir() and input_path.name == tasks_subdir.name:
        session_dir = input_path
        for _ in tasks_subdir.parts:
            session_dir = session_dir.parent
        candidate_dir = session_dir / tasks_subdir
        if candidate_dir == input_path:
            return session_dir, candidate_dir
    if input_path.is_dir() and (input_path / tasks_subdir).is_dir():
        return input_path, input_path / tasks_subdir
    raise FileNotFoundError(
        f"Could not find a session directory with {tasks_subdir.as_posix()} under: {input_path}"
    )


def discover_session_dirs(input_path: Path, tasks_subdir: Path) -> List[Path]:
    try:
        session_dir, _ = resolve_session_paths(input_path, tasks_subdir)
        return [session_dir]
    except FileNotFoundError:
        pass

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Could not find a session directory or batch root under: {input_path}"
        )

    session_dirs = []
    for task_json_dir in input_path.rglob(tasks_subdir.name):
        if not task_json_dir.is_dir():
            continue
        if task_json_dir.relative_to(input_path).parts[-len(tasks_subdir.parts):] != tasks_subdir.parts:
            continue
        session_dir = task_json_dir
        for _ in tasks_subdir.parts:
            session_dir = session_dir.parent
        session_dirs.append(session_dir)
    session_dirs = sorted(set(session_dirs), key=lambda path: str(path).lower())
    if not session_dirs:
        raise FileNotFoundError(
            f"No session directories containing {tasks_subdir.as_posix()} were found under: {input_path}"
        )
    return session_dirs


def discover_batch_session_dirs(input_path: Path, tasks_subdir: Path, recursive: bool) -> List[Path]:
    if not input_path.is_dir():
        raise FileNotFoundError(f"Batch root is not a directory: {input_path}")

    if recursive:
        session_dirs = []
        for task_json_dir in input_path.rglob(tasks_subdir.name):
            if not task_json_dir.is_dir():
                continue
            if task_json_dir.relative_to(input_path).parts[-len(tasks_subdir.parts):] != tasks_subdir.parts:
                continue
            session_dir = task_json_dir
            for _ in tasks_subdir.parts:
                session_dir = session_dir.parent
            session_dirs.append(session_dir)
        session_dirs = sorted(set(session_dirs), key=lambda path: str(path).lower())
    else:
        session_dirs = sorted(
            [
                child
                for child in input_path.iterdir()
                if child.is_dir() and (child / tasks_subdir).is_dir()
            ],
            key=lambda path: str(path).lower(),
        )

    if not session_dirs:
        search_mode = "recursively" if recursive else "under direct children"
        raise FileNotFoundError(
            f"No session directories containing {tasks_subdir.as_posix()} were found {search_mode}: {input_path}"
        )
    return session_dirs


def read_json_file(json_path: Path) -> Dict[str, Any]:
    if not json_path.is_file():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_status_file(status_path: Path, payload: Dict[str, Any]) -> None:
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def status_file_path_for_session(session_dir: Path) -> Path:
    return session_dir / "_phase1_unit_cutting_status.json"


def resolve_output_dir_for_session(
    session_dir: Path,
    explicit_output_dir: str,
    batch_mode: bool,
) -> Path:
    if not explicit_output_dir:
        return session_dir / "segments_units"

    resolved = Path(explicit_output_dir).resolve()
    if batch_mode:
        return resolved / session_dir.name / "segments_units"
    return resolved


def choose_image_path(step: Dict[str, Any], session_dir: Path) -> Optional[Path]:
    now_state = step.get("now_state") or {}
    action = step.get("action") or {}
    action_type = str(action.get("type") or "").strip().lower() if isinstance(action, dict) else ""
    before_field = (
        "screenshot_path_before"
        if action_type in MOUSE_ACTION_TYPES
        else "screenshot_path_before_raw"
    )
    candidates = [
        now_state.get("screenshot_path_before_part"),
        now_state.get(before_field),
        now_state.get("screenshot_path_after"),
    ]
    for rel_path in candidates:
        if not rel_path:
            continue
        image_path = session_dir / str(rel_path)
        if image_path.is_file():
            return image_path
    return None


def select_image_step_indices(total_steps: int, max_images: int) -> List[int]:
    if total_steps <= 0 or max_images <= 0:
        return []
    if total_steps <= max_images:
        return list(range(1, total_steps + 1))

    picks = set()
    for image_idx in range(max_images):
        ratio = image_idx / max(max_images - 1, 1)
        step_index = int(round(ratio * (total_steps - 1))) + 1
        picks.add(step_index)
    return sorted(picks)


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def normalize_target_description(target: Any) -> str:
    if not isinstance(target, dict):
        return ""

    nl_position = target.get("nl_position")
    if isinstance(nl_position, list):
        return "; ".join(str(item) for item in nl_position if str(item).strip())
    if isinstance(nl_position, str):
        return nl_position.strip()

    describe = target.get("describe")
    if isinstance(describe, dict):
        return json.dumps(describe, ensure_ascii=False)
    if isinstance(describe, str):
        return describe.strip()
    return ""


def compact_step_for_prompt(step: Dict[str, Any], step_index: int) -> Dict[str, Any]:
    action = step.get("action") or {}
    now_state = step.get("now_state") or {}
    return {
        "step_index": step_index,
        "step_id": step.get("step_id"),
        "step_goal": step.get("step_goal"),
        "app_title_before": now_state.get("app_title_before"),
        "app_title_after": now_state.get("app_title_after"),
        "action_type": action.get("type"),
        "target_description": normalize_target_description(action.get("target")),
        "action_parameters": action.get("param") or {},
        "action_preconditions": step.get("action_preconditions") or [],
        "action_before_state": step.get("action_before_state"),
        "action_after_effects": step.get("action_after_effects") or [],
        "nl_explanation": step.get("nl_explanation"),
    }


def build_prompt_segment(
    segment_id: str,
    source_data: Dict[str, Any],
    step_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    steps = source_data.get("steps") or []
    if step_indices is None:
        indexed_steps = list(enumerate(steps, start=1))
    else:
        indexed_steps = list(zip(step_indices, steps))

    compact_steps = [
        compact_step_for_prompt(step, idx)
        for idx, step in indexed_steps
    ]
    return {
        "segment_id": segment_id,
        "app": source_data.get("app"),
        "env": source_data.get("env") or {},
        "steps": compact_steps,
    }


def build_user_content(
    segment_prompt_json: Dict[str, Any],
    source_data: Dict[str, Any],
    session_dir: Path,
    max_images: int,
    step_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    steps = source_data.get("steps") or []
    selected_indices = select_image_step_indices(len(steps), max_images)
    content: List[Dict[str, Any]] = []

    # # 没有截图就不需要了？
    # if selected_indices:
    #     content.append(
    #         {
    #             "type": "text",
    #             "text": (
    #                 "Representative screenshots are attached for selected steps. "
    #                 "Each attached image corresponds to the step label immediately before it. "
    #                 "When a cropped before screenshot exists, that cropped image is preferred."
    #             ),
    #         }
    #     )

    for step_index in selected_indices:
        image_path = choose_image_path(steps[step_index - 1], session_dir)
        if image_path is None:
            continue
        global_step_index = step_indices[step_index - 1] if step_indices else step_index
        content.append(
            {
                "type": "text",
                "text": f"[Screenshot for step {global_step_index}]",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(image_path)}",
                },
            }
        )

    prompt_text = USER_PROMPT_TEMPLATE.replace(
        "{segment_json_here}",
        json.dumps(segment_prompt_json, ensure_ascii=False, indent=2),
    )
    content.append({"type": "text", "text": prompt_text})
    return content


def build_codex_user_prompt(
    segment_prompt_json: Dict[str, Any],
    source_path: Path,
    session_dir: Path,
) -> str:
    prompt_text = USER_PROMPT_TEMPLATE.replace(
        "{segment_json_here}",
        json.dumps(segment_prompt_json, ensure_ascii=False, indent=2),
    )
    return (
        f"The source segment JSON file is located at:\n{source_path}\n\n"
        # f"The related screenshots directory is located at:\n{session_dir / 'screenshots'}\n\n"   # 去掉截图看看效果
        # "Use the screenshots referenced by the segment JSON when they help determine boundaries or concrete states. "
        "You may inspect the local files directly before answering.\n\n"
        f"{prompt_text}"
    )


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _find_codex_path() -> str:
    return "codex"


def call_codex(model: str, system_prompt: str, user_prompt: str) -> str:
    import tempfile

    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        encoding="utf-8",
        delete=False,
    ) as file:
        prompt_file = file.name
        file.write(prompt)

    try:
        codex_path = _find_codex_path()
        base = f'"{codex_path}" exec --skip-git-repo-check -c reasoning_effort=medium'
        if model:
            base += f" --model {shlex.quote(model)}"
        base += " --json"

        if os.name == "nt":
            cmd = f'cmd /c "{base} < {prompt_file}"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        else:
            cmd = shlex.split(base)
            with open(prompt_file, "r", encoding="utf-8") as stdin_file:
                result = subprocess.run(
                    cmd,
                    stdin=stdin_file,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
    finally:
        os.unlink(prompt_file)

    if result.returncode != 0:
        stderr_msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(f"Codex error: {stderr_msg}")

    last_content: Optional[str] = None
    for line in result.stdout.splitlines():
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
                        part if isinstance(part, str) else part.get("text", "")
                        for part in content
                    )
                last_content = content

    if last_content is not None:
        return last_content
    return "(no assistant message returned)"


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if "{" in candidate and "}" in candidate:
                text = candidate
                break

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_response_json(raw_text: str) -> Dict[str, Any]:
    return json.loads(extract_json_text(raw_text))


def normalize_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_step_indices(step_indices: Any) -> List[int]:
    normalized: List[int] = []
    if not isinstance(step_indices, list):
        return normalized
    for item in step_indices:
        if isinstance(item, int):
            normalized.append(item)
            continue
        if isinstance(item, str):
            stripped = item.strip().lower()
            if stripped.startswith("s") and stripped[1:].isdigit():
                normalized.append(int(stripped[1:]))
            elif stripped.isdigit():
                normalized.append(int(stripped))
    return normalized


def normalize_units(raw_units: Any, segment_id: str) -> List[Dict[str, Any]]:
    if not isinstance(raw_units, list):
        return []

    units: List[Dict[str, Any]] = []
    for unit_index, raw_unit in enumerate(raw_units, start=1):
        if not isinstance(raw_unit, dict):
            continue

        unit_type = raw_unit.get("unit_type")
        if not unit_type and "unit.type" in raw_unit:
            unit_type = raw_unit.get("unit.type")

        unit = {
            "unit_id": f"u_{segment_id}_{unit_index:02d}",
            "step_indices": normalize_step_indices(raw_unit.get("step_indices")),
            "unit_intent": str(raw_unit.get("unit_intent", "")).strip(),
            "unit_type": str(unit_type or "").strip(),
            "unit_before_state": str(raw_unit.get("unit_before_state", "")).strip(),
            "unit_after_state": str(raw_unit.get("unit_after_state", "")).strip(),
            "unit_precondition": normalize_text_list(raw_unit.get("unit_precondition")),
            "unit_effect": normalize_text_list(raw_unit.get("unit_effect")),
        }
        units.append(unit)
    return units


def is_low_signal_intent(intent: str) -> bool:
    normalized = re.sub(r"\s+", " ", intent.strip().lower())
    return any(phrase in normalized for phrase in LOW_SIGNAL_INTENTS)


def validate_units(
    units: Sequence[Dict[str, Any]],
    expected_step_indices: Sequence[int],
) -> List[str]:
    errors: List[str] = []
    if not units:
        return ["No units were produced."]

    expected_index_list = list(expected_step_indices)
    expected_index_set = set(expected_index_list)
    all_indices: List[int] = []
    previous_end = expected_index_list[0] - 1 if expected_index_list else 0

    for unit_index, unit in enumerate(units, start=1):
        required_text_fields = (
            "unit_id",
            "unit_intent",
            "unit_type",
            "unit_before_state",
            "unit_after_state",
        )
        for field_name in required_text_fields:
            if not str(unit.get(field_name, "")).strip():
                errors.append(f"Unit {unit_index} is missing {field_name}.")

        step_indices = unit.get("step_indices") or []
        if not isinstance(step_indices, list) or not step_indices:
            errors.append(f"Unit {unit_index} has empty step_indices.")
            continue

        if sorted(step_indices) != step_indices:
            errors.append(f"Unit {unit_index} step_indices are not sorted.")
        if len(set(step_indices)) != len(step_indices):
            errors.append(f"Unit {unit_index} step_indices contain duplicates.")
        if any(idx not in expected_index_set for idx in step_indices):
            errors.append(f"Unit {unit_index} step_indices are out of range.")

        expected_span = list(range(step_indices[0], step_indices[-1] + 1))
        if step_indices != expected_span:
            errors.append(f"Unit {unit_index} step_indices are not contiguous.")
        if len(step_indices) > 6:
            errors.append(
                f"Unit {unit_index} is too long ({len(step_indices)} steps)."
            )

        if step_indices[0] <= previous_end:
            errors.append(f"Unit {unit_index} overlaps with the previous unit.")
        previous_end = max(previous_end, step_indices[-1])
        all_indices.extend(step_indices)

        if not unit.get("unit_precondition"):
            errors.append(f"Unit {unit_index} has empty unit_precondition.")
        if not unit.get("unit_effect"):
            errors.append(f"Unit {unit_index} has empty unit_effect.")

        if len(str(unit.get("unit_before_state", ""))) < 20:
            errors.append(f"Unit {unit_index} unit_before_state is too short.")
        if len(str(unit.get("unit_after_state", ""))) < 20:
            errors.append(f"Unit {unit_index} unit_after_state is too short.")

        if is_low_signal_intent(str(unit.get("unit_intent", ""))):
            errors.append(f"Unit {unit_index} unit_intent looks like a low-level action paraphrase.")

    if all_indices != expected_index_list:
        errors.append("Units do not cover all steps exactly once in order.")

    return errors


def call_segmentation_model(
    backend: str,
    client: Optional[OpenAI],
    codex_call,
    model: str,
    system_prompt: str,
    user_content: Optional[List[Dict[str, Any]]] = None,
    user_prompt: str = "",
) -> str:
    if backend == "codex":
        if codex_call is None:
            raise RuntimeError("codex backend requested but call_codex is not available.")
        if not user_prompt.strip():
            raise RuntimeError("codex backend requires a non-empty user_prompt.")
        content = codex_call(model, system_prompt, user_prompt)
        if not content:
            raise RuntimeError("Codex returned empty content.")
        return content

    if client is None:
        raise RuntimeError("openai-compatible backend requested but client is not initialized.")
    if user_content is None:
        raise RuntimeError("openai-compatible backend requires user_content.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty content.")
    return content


def build_output_payload(
    segment_id: str,
    source_data: Dict[str, Any],
    units: List[Dict[str, Any]],
    phase1_error: str = "",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "segment_id": segment_id,
        "app": source_data.get("app"),
        "env": source_data.get("env") or {},
        "steps": source_data.get("steps") or [],
        "units": units,
    }
    if phase1_error:
        payload["phase1_error"] = phase1_error
    return payload


def is_completed_segment_output(output_path: Path, segment_id: str, source_path: Path) -> bool:
    if not output_path.is_file():
        return False

    payload = read_json_file(output_path)
    if not payload or payload.get("phase1_error"):
        return False
    if str(payload.get("segment_id", "")).strip() != segment_id:
        return False

    source_data = read_json_file(source_path)
    source_steps = source_data.get("steps") or []
    if payload.get("steps") != source_steps:
        return False

    units = payload.get("units")
    if not isinstance(units, list):
        return False

    expected_step_indices = list(range(1, len(source_steps) + 1))
    return not validate_units(units, expected_step_indices=expected_step_indices)


def window_cache_path(debug_dir: Path, debug_prefix: str) -> Path:
    return debug_dir / f"{debug_prefix}_units.json"


def has_window_cache_outputs(debug_dir: Path, segment_id: str) -> bool:
    return any(debug_dir.glob(f"{segment_id}_window_*_units.json"))


def read_valid_window_cache(
    cache_path: Path,
    segment_id: str,
    source_data: Dict[str, Any],
    expected_step_indices: Sequence[int],
) -> Optional[List[Dict[str, Any]]]:
    payload = read_json_file(cache_path)
    if not payload or payload.get("phase1_error"):
        return None
    if str(payload.get("segment_id", "")).strip() != segment_id:
        return None
    if payload.get("expected_step_indices") != list(expected_step_indices):
        return None
    if payload.get("steps") != (source_data.get("steps") or []):
        return None

    units = payload.get("units")
    if not isinstance(units, list):
        return None
    if validate_units(units, expected_step_indices=expected_step_indices):
        return None
    return units


def write_window_cache(
    cache_path: Path,
    segment_id: str,
    window_index: int,
    start: int,
    end: int,
    source_data: Dict[str, Any],
    expected_step_indices: Sequence[int],
    units: List[Dict[str, Any]],
) -> None:
    payload = {
        "cache_kind": "phase1_window_units",
        "segment_id": segment_id,
        "window_id": f"{segment_id}_window_{window_index:02d}",
        "step_start": start,
        "step_end": end,
        "expected_step_indices": list(expected_step_indices),
        "steps": source_data.get("steps") or [],
        "units": units,
        "cached_at": format_timestamp(datetime.now()),
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_phase1_status(units: List[Dict[str, Any]], status: str) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for unit in units:
        copied = dict(unit)
        copied["phase1_status"] = status
        updated.append(copied)
    return updated


def build_source_slice(source_data: Dict[str, Any], expected_step_indices: Sequence[int]) -> Dict[str, Any]:
    sliced = dict(source_data)
    all_steps = source_data.get("steps") or []
    sliced["steps"] = [all_steps[idx - 1] for idx in expected_step_indices]
    return sliced


def generate_validated_units(
    source_data: Dict[str, Any],
    source_path: Path,
    segment_id: str,
    session_dir: Path,
    debug_dir: Path,
    debug_prefix: str,
    backend: str,
    client: Optional[OpenAI],
    codex_call,
    model: str,
    max_images: int,
    max_retries: int,
    retry_delay: float,
    expected_step_indices: Sequence[int],
) -> Tuple[bool, List[Dict[str, Any]], str]:
    segment_prompt_json = build_prompt_segment(
        segment_id=segment_id,
        source_data=source_data,
        step_indices=expected_step_indices,
    )
    user_content: Optional[List[Dict[str, Any]]] = None
    base_user_prompt = ""
    if backend == "codex":
        base_user_prompt = build_codex_user_prompt(
            segment_prompt_json=segment_prompt_json,
            source_path=source_path,
            session_dir=session_dir,
        )
    else:
        user_content = build_user_content(
            segment_prompt_json=segment_prompt_json,
            source_data=source_data,
            session_dir=session_dir,
            max_images=max_images,
            step_indices=expected_step_indices,
        )

    last_error = ""
    for attempt in range(1, max_retries + 1):
        attempt_start = time.perf_counter()
        log_with_timestamp(f"{debug_prefix}: attempt {attempt}/{max_retries}")
        if attempt > 1 and last_error:
            correction_text = (
                "Previous output failed validation. Please regenerate the full JSON from scratch and fix these issues:\n"
                f"{last_error}\n"
                "Do not create any unit longer than 6 steps."
            )
            if backend == "codex":
                attempt_user_prompt = f"{base_user_prompt}\n\n{correction_text}"
                attempt_content = None
            else:
                attempt_content = user_content + [{"type": "text", "text": correction_text}]
                attempt_user_prompt = ""
        else:
            attempt_content = user_content
            attempt_user_prompt = base_user_prompt

        raw_output = ""
        try:
            raw_output = call_segmentation_model(
                backend=backend,
                client=client,
                codex_call=codex_call,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_content=attempt_content,
                user_prompt=attempt_user_prompt,
            )
            (debug_dir / f"{debug_prefix}_attempt_{attempt}_raw.txt").write_text(
                raw_output,
                encoding="utf-8",
            )

            parsed = parse_response_json(raw_output)
            units = normalize_units(parsed.get("units"), segment_id=segment_id)
            validation_errors = validate_units(
                units,
                expected_step_indices=expected_step_indices,
            )
            if validation_errors:
                last_error = "; ".join(validation_errors)
                log_with_timestamp(
                    f"{debug_prefix}: attempt {attempt} validation failed after "
                    f"{format_duration(time.perf_counter() - attempt_start)}: {last_error}"
                )
            else:
                log_with_timestamp(
                    f"{debug_prefix}: attempt {attempt} succeeded in "
                    f"{format_duration(time.perf_counter() - attempt_start)}"
                )
                return True, units, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log_with_timestamp(
                f"{debug_prefix}: attempt {attempt} request failed after "
                f"{format_duration(time.perf_counter() - attempt_start)}: {last_error}"
            )
            if raw_output:
                (debug_dir / f"{debug_prefix}_attempt_{attempt}_raw.txt").write_text(
                    raw_output,
                    encoding="utf-8",
                )

        if attempt < max_retries:
            time.sleep(retry_delay)

    return False, [], last_error


def resolve_overlap(
    existing_units: List[Dict[str, Any]],
    new_units: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not existing_units:
        return list(new_units)
    if not new_units:
        return []

    last_existing = existing_units[-1]
    last_existing_end = max(last_existing["step_indices"])
    resolved: List[Dict[str, Any]] = []

    for unit in new_units:
        unit_start = min(unit["step_indices"])
        if unit_start <= last_existing_end:
            if len(unit["step_indices"]) > len(last_existing["step_indices"]):
                existing_units.pop()
                resolved.append(unit)
        else:
            resolved.append(unit)

    return resolved


def generate_units_with_windows(
    source_data: Dict[str, Any],
    source_path: Path,
    segment_id: str,
    session_dir: Path,
    debug_dir: Path,
    backend: str,
    client: Optional[OpenAI],
    codex_call,
    model: str,
    max_images: int,
    max_retries: int,
    retry_delay: float,
    window_size: int,
    overlap: int,
    force: bool,
) -> Tuple[bool, List[Dict[str, Any]], str]:
    all_steps = source_data.get("steps") or []
    step_count = len(all_steps)
    if step_count == 0:
        return False, [], "Source JSON has no steps."

    if window_size <= overlap:
        return False, [], "window_size must be greater than overlap."

    merged_units: List[Dict[str, Any]] = []
    start = 1
    window_index = 1

    while start <= step_count:
        end = min(start + window_size - 1, step_count)
        expected_indices = list(range(start, end + 1))
        window_source = build_source_slice(source_data, expected_indices)
        debug_prefix = f"{segment_id}_window_{window_index:02d}_{start:03d}_{end:03d}"
        cache_path = window_cache_path(debug_dir, debug_prefix)
        window_start = time.perf_counter()
        log_with_timestamp(
            f"{segment_id}: window {window_index} start "
            f"(steps {start}-{end}, size={len(expected_indices)})"
        )

        cached_units = None
        if not force:
            cached_units = read_valid_window_cache(
                cache_path=cache_path,
                segment_id=segment_id,
                source_data=window_source,
                expected_step_indices=expected_indices,
            )

        if cached_units is not None:
            success = True
            window_units = cached_units
            error = ""
            log_with_timestamp(
                f"{segment_id}: window {window_index} cache hit ({cache_path.name})"
            )
        else:
            success, window_units, error = generate_validated_units(
                source_data=window_source,
                source_path=source_path,
                segment_id=segment_id,
                session_dir=session_dir,
                debug_dir=debug_dir,
                debug_prefix=debug_prefix,
                backend=backend,
                client=client,
                codex_call=codex_call,
                model=model,
                max_images=max_images,
                max_retries=max_retries,
                retry_delay=retry_delay,
                expected_step_indices=expected_indices,
            )
            if success:
                write_window_cache(
                    cache_path=cache_path,
                    segment_id=segment_id,
                    window_index=window_index,
                    start=start,
                    end=end,
                    source_data=window_source,
                    expected_step_indices=expected_indices,
                    units=window_units,
                )

        if not success:
            log_with_timestamp(
                f"{segment_id}: window {window_index} failed after "
                f"{format_duration(time.perf_counter() - window_start)}"
            )
            return False, [], f"Window {window_index} ({start}-{end}) failed: {error}"
        log_with_timestamp(
            f"{segment_id}: window {window_index} finished in "
            f"{format_duration(time.perf_counter() - window_start)}"
        )

        resolved_units = resolve_overlap(merged_units, window_units)
        merged_units.extend(resolved_units)

        if end == step_count:
            break
        start += window_size - overlap
        window_index += 1

    final_errors = validate_units(
        merged_units,
        expected_step_indices=list(range(1, step_count + 1)),
    )
    if final_errors:
        return False, [], "; ".join(final_errors)

    return True, merged_units, ""


def segment_one_file(
    source_path: Path,
    segment_id: str,
    session_dir: Path,
    output_dir: Path,
    debug_dir: Path,
    backend: str,
    client: Optional[OpenAI],
    codex_call,
    model: str,
    max_images: int,
    max_retries: int,
    retry_delay: float,
    window_fallback_threshold: int,
    window_size: int,
    window_overlap: int,
    force: bool,
) -> Tuple[bool, str]:
    segment_wall_start = datetime.now()
    segment_perf_start = time.perf_counter()
    source_data = json.loads(source_path.read_text(encoding="utf-8"))
    steps = source_data.get("steps") or []
    log_with_timestamp(
        f"{segment_id}: start processing {source_path.name} "
        f"with {len(steps) if isinstance(steps, list) else 0} steps"
    )
    if not isinstance(steps, list) or not steps:
        payload = build_output_payload(
            segment_id=segment_id,
            source_data=source_data,
            units=[],
            phase1_error="Source JSON has no steps.",
        )
        (output_dir / f"{segment_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_with_timestamp(
            f"{segment_id}: failed in {format_duration(time.perf_counter() - segment_perf_start)} "
            f"(no steps)"
        )
        return False, "Source JSON has no steps."

    has_window_cache = (
        not force
        and len(steps) >= window_fallback_threshold
        and has_window_cache_outputs(debug_dir, segment_id)
    )
    if has_window_cache:
        success = False
        units: List[Dict[str, Any]] = []
        last_error = "Existing window cache found; resume with sliding windows."
        log_with_timestamp(
            f"{segment_id}: window cache detected, resuming sliding windows "
            f"(window_size={window_size}, overlap={window_overlap})"
        )
    else:
        success, units, last_error = generate_validated_units(
            source_data=source_data,
            source_path=source_path,
            segment_id=segment_id,
            session_dir=session_dir,
            debug_dir=debug_dir,
            debug_prefix=segment_id,
            backend=backend,
            client=client,
            codex_call=codex_call,
            model=model,
            max_images=max_images,
            max_retries=max_retries,
            retry_delay=retry_delay,
            expected_step_indices=list(range(1, len(steps) + 1)),
        )

    if (not success) and len(steps) >= window_fallback_threshold:
        log_with_timestamp(
            f"{segment_id}: full-segment generation failed, falling back to sliding windows "
            f"(window_size={window_size}, overlap={window_overlap})"
        )
        success, units, last_error = generate_units_with_windows(
            source_data=source_data,
            source_path=source_path,
            segment_id=segment_id,
            session_dir=session_dir,
            debug_dir=debug_dir,
            backend=backend,
            client=client,
            codex_call=codex_call,
            model=model,
            max_images=max_images,
            max_retries=max_retries,
            retry_delay=retry_delay,
            window_size=window_size,
            overlap=window_overlap,
            force=force,
        )

    if success:
        final_units = ensure_phase1_status(units, status="done")
        payload = build_output_payload(
            segment_id=segment_id,
            source_data=source_data,
            units=final_units,
        )
        (output_dir / f"{segment_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_with_timestamp(
            f"{segment_id}: finished successfully in "
            f"{format_duration(time.perf_counter() - segment_perf_start)} "
            f"(started {format_timestamp(segment_wall_start)})"
        )
        return True, ""

    failed_payload = build_output_payload(
        segment_id=segment_id,
        source_data=source_data,
        units=[],
        phase1_error=last_error,
    )
    (output_dir / f"{segment_id}.json").write_text(
        json.dumps(failed_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_with_timestamp(
        f"{segment_id}: failed in {format_duration(time.perf_counter() - segment_perf_start)} "
        f"(started {format_timestamp(segment_wall_start)})"
    )
    return False, last_error


def iter_task_files(tasks_abs_dir: Path) -> Iterable[Path]:
    files = sorted(tasks_abs_dir.glob("*.json"), key=natural_sort_key)
    for file_path in files:
        if file_path.is_file():
            yield file_path


def count_completed_segment_outputs(
    tasks_abs_dir: Path,
    output_dir: Path,
    segment_limit: int,
) -> Tuple[int, int]:
    task_files = list(iter_task_files(tasks_abs_dir))
    if segment_limit > 0:
        task_files = task_files[:segment_limit]

    completed_count = 0
    for file_index, source_path in enumerate(task_files, start=1):
        segment_id = f"seg_{file_index:03d}"
        output_path = output_dir / f"{segment_id}.json"
        if is_completed_segment_output(output_path, segment_id, source_path):
            completed_count += 1
    return completed_count, len(task_files)


def session_outputs_completed(
    tasks_abs_dir: Path,
    output_dir: Path,
    segment_limit: int,
) -> bool:
    completed_count, total_count = count_completed_segment_outputs(
        tasks_abs_dir=tasks_abs_dir,
        output_dir=output_dir,
        segment_limit=segment_limit,
    )
    return total_count > 0 and completed_count == total_count


def resolve_api_key(explicit_api_key: str) -> str:
    if explicit_api_key:
        return explicit_api_key
    for env_name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    raise RuntimeError("No API key found. Set QWEN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY.")


def resolve_base_url(explicit_base_url: str) -> str:
    if explicit_base_url:
        return explicit_base_url
    return (
        os.getenv("QWEN_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def resolve_default_model(backend: str, explicit_model: str) -> str:
    if explicit_model:
        return explicit_model
    if backend == "codex":
        return "gpt-5.5"
    return "qwen-vl-plus"


def process_session(
    session_dir: Path,
    tasks_abs_dir: Path,
    output_dir: Path,
    status_path: Optional[Path],
    backend: str,
    client: Optional[OpenAI],
    codex_call,
    model: str,
    max_images: int,
    max_retries: int,
    retry_delay: float,
    window_fallback_threshold: int,
    window_size: int,
    window_overlap: int,
    segment_limit: int,
    force: bool,
) -> Tuple[bool, List[Tuple[str, str]], int]:
    debug_dir = output_dir / "_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    task_files = list(iter_task_files(tasks_abs_dir))
    if not task_files:
        raise FileNotFoundError(f"No JSON files found under {tasks_abs_dir}")
    if segment_limit > 0:
        task_files = task_files[:segment_limit]

    log_with_timestamp(
        f"Session start: session={session_dir.name}, backend={backend}, model={model}, "
        f"segments={len(task_files)}, output_dir={output_dir}"
    )

    failures: List[Tuple[str, str]] = []
    completed_segments = 0
    skipped_segments = 0
    failed_segments = 0
    total_segments = len(task_files)
    for file_index, source_path in enumerate(task_files, start=1):
        segment_id = f"seg_{file_index:03d}"
        output_path = output_dir / f"{segment_id}.json"
        if not force and is_completed_segment_output(output_path, segment_id, source_path):
            completed_segments += 1
            skipped_segments += 1
            log_with_timestamp(
                f"{session_dir.name} [{file_index}/{total_segments}]: skip {source_path.name} -> {segment_id} "
                f"(completed output exists)"
            )
            if status_path is not None:
                status_data = read_json_file(status_path)
                status_data.update(
                    {
                        "current_segment_index": file_index,
                        "current_segment_id": segment_id,
                        "current_source": str(source_path),
                        "segments_total": total_segments,
                        "segments_completed_so_far": completed_segments,
                        "segments_skipped_so_far": skipped_segments,
                        "segments_failed_so_far": failed_segments,
                        "last_progress_at": format_timestamp(datetime.now()),
                    }
                )
                write_status_file(status_path, status_data)
            continue

        log_with_timestamp(
            f"{session_dir.name} [{file_index}/{total_segments}]: processing {source_path.name} -> {segment_id}"
        )
        if status_path is not None:
            status_data = read_json_file(status_path)
            status_data.update(
                {
                    "current_segment_index": file_index,
                    "current_segment_id": segment_id,
                    "current_source": str(source_path),
                    "segments_total": total_segments,
                    "segments_completed_so_far": completed_segments,
                    "segments_skipped_so_far": skipped_segments,
                    "segments_failed_so_far": failed_segments,
                    "last_progress_at": format_timestamp(datetime.now()),
                }
            )
            write_status_file(status_path, status_data)

        ok, error = segment_one_file(
            source_path=source_path,
            segment_id=segment_id,
            session_dir=session_dir,
            output_dir=output_dir,
            debug_dir=debug_dir,
            backend=backend,
            client=client,
            codex_call=codex_call,
            model=model,
            max_images=max_images,
            max_retries=max_retries,
            retry_delay=retry_delay,
            window_fallback_threshold=window_fallback_threshold,
            window_size=window_size,
            window_overlap=window_overlap,
            force=force,
        )
        if not ok:
            failed_segments += 1
            failures.append((segment_id, error))
        else:
            completed_segments += 1

        if status_path is not None:
            status_data = read_json_file(status_path)
            status_data.update(
                {
                    "current_segment_index": file_index,
                    "current_segment_id": segment_id,
                    "current_source": str(source_path),
                    "segments_total": total_segments,
                    "segments_completed_so_far": completed_segments,
                    "segments_skipped_so_far": skipped_segments,
                    "segments_failed_so_far": failed_segments,
                    "last_progress_at": format_timestamp(datetime.now()),
                }
            )
            write_status_file(status_path, status_data)

    return not failures, failures, len(task_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-1 unit segmentation and concrete annotation for one session or a batch of sessions.",
    )
    parser.add_argument(
        "input_path",
        help="Session directory path, task-JSON directory path, or a batch root containing multiple session directories.",
    )
    parser.add_argument(
        "--tasks-subdir",
        default="tasks_abs",
        help="Relative subdirectory under each session that contains the segment JSON files. Default: tasks_abs. Example: splits/tasks",
    )
    parser.add_argument(
        "--backend",
        choices=("codex", "qwen"),
        default="codex",
        help="Model backend. 'codex' uses the built-in codex CLI call in this file; 'qwen' uses the OpenAI-compatible multimodal API path. Default: codex.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated outputs. Single-session mode writes directly here; batch mode writes to <output-dir>/<session_name>/segments_units. Defaults to <session>/segments_units.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Model name. Default: gpt-5.5 for codex, qwen-vl-plus for qwen.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. If omitted, environment variables are used.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional base URL. If omitted, QWEN_BASE_URL / OPENAI_BASE_URL / DashScope default is used.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum representative screenshots to attach per segment. Default: 20.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum generation retries per segment. Default: 3.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between retries. Default: 2.0.",
    )
    parser.add_argument(
        "--window-fallback-threshold",
        type=int,
        default=40,
        help="If a segment has at least this many steps and full-segment generation fails, use sliding-window fallback. Default: 40.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Sliding-window size for fallback mode. Default: 6.",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=2,
        help="Sliding-window overlap for fallback mode. Default: 2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of segment JSON files to process per trajectory. 0 means no limit.",
    )
    parser.add_argument(
        "--trajectory-limit",
        type=int,
        default=0,
        help="Optional maximum number of trajectory directories to process in batch mode. 0 means no limit.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat input_path as a batch root and process trajectory directories under it.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --batch is used, recursively search for trajectory directories. Without it, only direct children are scanned.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one trajectory fails. By default, batch mode continues with the next trajectory.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional file path for timestamped runtime logs. Console output is still printed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess trajectories even if they are already marked as done.",
    )
    args = parser.parse_args()
    run_wall_start = datetime.now()
    run_perf_start = time.perf_counter()

    repo_root = Path(__file__).resolve().parent
    load_env_file(repo_root / ".env")
    configure_log_file(args.log_file)

    input_path = Path(args.input_path).resolve()
    tasks_subdir = normalize_tasks_subdir(args.tasks_subdir)
    if args.batch:
        session_dirs = discover_batch_session_dirs(
            input_path=input_path,
            tasks_subdir=tasks_subdir,
            recursive=args.recursive,
        )
        batch_mode = True
    else:
        session_dirs = discover_session_dirs(input_path, tasks_subdir)
        batch_mode = len(session_dirs) > 1 or (
            input_path.is_dir() and input_path not in session_dirs
        )
    if args.trajectory_limit > 0:
        session_dirs = session_dirs[: args.trajectory_limit]

    model = resolve_default_model(args.backend, args.model)
    client: Optional[OpenAI] = None
    codex_call = None
    if args.backend == "codex":
        codex_call = call_codex
    else:
        client = create_client(
            api_key=resolve_api_key(args.api_key),
            base_url=resolve_base_url(args.base_url),
        )

    log_with_timestamp(
        f"Run start: backend={args.backend}, model={model}, trajectories={len(session_dirs)}, "
        f"batch_mode={batch_mode}, recursive={args.recursive}, tasks_subdir={tasks_subdir.as_posix()}"
    )
    if LOG_FILE_PATH is not None:
        log_with_timestamp(f"Runtime log file: {LOG_FILE_PATH}")

    trajectory_failures: List[Tuple[str, str]] = []
    processed_sessions = 0
    skipped_sessions = 0

    for session_index, session_dir in enumerate(session_dirs, start=1):
        _, tasks_abs_dir = resolve_session_paths(session_dir, tasks_subdir)
        output_dir = resolve_output_dir_for_session(
            session_dir=session_dir,
            explicit_output_dir=args.output_dir,
            batch_mode=batch_mode,
        )
        status_path = status_file_path_for_session(session_dir)
        status_data = read_json_file(status_path)
        previous_status = str(status_data.get("status", "")).strip().lower()

        outputs_are_complete = session_outputs_completed(
            tasks_abs_dir=tasks_abs_dir,
            output_dir=output_dir,
            segment_limit=args.limit,
        )

        if previous_status == "done" and outputs_are_complete and not args.force:
            skipped_sessions += 1
            log_with_timestamp(
                f"[{session_index}/{len(session_dirs)}] skip {session_dir.name}: already marked done at "
                f"{status_data.get('finished_at', 'unknown time')}"
            )
            continue
        if previous_status == "done" and not outputs_are_complete and not args.force:
            log_with_timestamp(
                f"[{session_index}/{len(session_dirs)}] resume {session_dir.name}: status is done but "
                f"some segment outputs are missing or invalid"
            )

        session_wall_start = datetime.now()
        session_perf_start = time.perf_counter()
        write_status_file(
            status_path,
            {
                "status": "in_progress",
                "session_dir": str(session_dir),
                "tasks_abs_dir": str(tasks_abs_dir),
                "tasks_subdir": tasks_subdir.as_posix(),
                "output_dir": str(output_dir),
                "backend": args.backend,
                "model": model,
                "started_at": format_timestamp(session_wall_start),
                "finished_at": "",
                "duration_seconds": 0.0,
                "segments_total": 0,
                "segments_succeeded": 0,
                "segments_failed": 0,
                "failed_segments": [],
            },
        )

        log_with_timestamp(
            f"[{session_index}/{len(session_dirs)}] processing trajectory {session_dir.name}"
        )

        ok = False
        failures: List[Tuple[str, str]] = []
        segment_count = 0
        error_message = ""
        try:
            ok, failures, segment_count = process_session(
                session_dir=session_dir,
                tasks_abs_dir=tasks_abs_dir,
                output_dir=output_dir,
                status_path=status_path,
                backend=args.backend,
                client=client,
                codex_call=codex_call,
                model=model,
                max_images=args.max_images,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                window_fallback_threshold=args.window_fallback_threshold,
                window_size=args.window_size,
                window_overlap=args.window_overlap,
                segment_limit=args.limit,
                force=args.force,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            failures = [("session", error_message)]
            ok = False

        duration_seconds = time.perf_counter() - session_perf_start
        segment_failed_count = sum(
            1 for segment_id, _ in failures if segment_id != "session"
        )
        segment_success_count = max(0, segment_count - segment_failed_count)
        write_status_file(
            status_path,
            {
                "status": "done" if ok else "failed",
                "session_dir": str(session_dir),
                "tasks_abs_dir": str(tasks_abs_dir),
                "tasks_subdir": tasks_subdir.as_posix(),
                "output_dir": str(output_dir),
                "backend": args.backend,
                "model": model,
                "started_at": format_timestamp(session_wall_start),
                "finished_at": format_timestamp(datetime.now()),
                "duration_seconds": round(duration_seconds, 3),
                "segments_total": segment_count,
                "segments_succeeded": segment_success_count,
                "segments_failed": segment_failed_count,
                "failed_segments": [
                    {"segment_id": segment_id, "error": error}
                    for segment_id, error in failures
                ],
            },
        )

        processed_sessions += 1
        if ok:
            log_with_timestamp(
                f"Trajectory {session_dir.name} finished successfully in {format_duration(duration_seconds)}"
            )
        else:
            trajectory_failures.append(
                (session_dir.name, error_message or "; ".join(f"{sid}: {err}" for sid, err in failures))
            )
            log_with_timestamp(
                f"Trajectory {session_dir.name} failed in {format_duration(duration_seconds)}"
            )
            if args.stop_on_error:
                log_with_timestamp("Stop on error is enabled; aborting remaining trajectories.")
                break

    log_with_timestamp(
        f"Run finished in {format_duration(time.perf_counter() - run_perf_start)} "
        f"(started {format_timestamp(run_wall_start)})"
    )
    print(
        f"\nProcessed trajectories: {processed_sessions}, skipped: {skipped_sessions}, "
        f"failed: {len(trajectory_failures)}"
    )
    if batch_mode:
        print(f"Batch root: {input_path}")
    else:
        print(f"Session root: {session_dirs[0]}")

    if args.output_dir:
        print(f"Output base: {Path(args.output_dir).resolve()}")
    else:
        print("Output base: each session directory / segments_units")

    if trajectory_failures:
        print("Failed trajectories:")
        for session_name, error in trajectory_failures:
            print(f"  - {session_name}: {error}")
        return 1

    print("All requested trajectories finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
