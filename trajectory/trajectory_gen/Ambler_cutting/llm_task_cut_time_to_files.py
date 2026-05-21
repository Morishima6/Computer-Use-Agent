import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_codex_path() -> str:
    return "codex"


def call_codex(model: str, system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to Codex and return the final assistant message."""

    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as f:
        prompt_file = f.name
        f.write(prompt)

    try:
        codex_path = _find_codex_path()
        base = f'"{codex_path}" exec --skip-git-repo-check -c reasoning_effort=medium'
        if model:
            base += f" --model {shlex.quote(model)}"
        base += " --json"

        if os.name == "nt":
            cmd = f"cmd /c \"{base} < {prompt_file}\""
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
                        c if isinstance(c, str) else c.get("text", "")
                        for c in content
                    )
                last_content = content

    if last_content is not None:
        return last_content
    return "(no assistant message returned)"


def build_prompt_for_segment(segment_key: str, steps: List[Dict[str, Any]]) -> str:
    """
    Build a prompt for one steps_timeX segment.
    Only uses step_id, step_goal, action_before_state, action_after_effects,
    and nl_explanation.
    """
    lines: List[str] = []
    lines.append(
        "You are an analyst that groups low-level UI steps into higher-level user tasks."
    )
    lines.append(
        f"The following steps all belong to one contiguous time-based segment: {segment_key}."
    )
    lines.append(
        "Your job is to decide how to cut these steps into one or more tasks, "
        "and for each task provide its instruction and task goal."
    )
    lines.append("")
    lines.append("Steps:")

    for step in steps:
        step_id = step.get("step_id", "")
        step_goal = step.get("step_goal", "")
        action_before_state = step.get("action_before_state", "")
        action_after_effects = step.get("action_after_effects", [])
        nl_explanation = step.get("nl_explanation", "")

        if isinstance(action_after_effects, list):
            after_text = " ".join(str(x) for x in action_after_effects)
        else:
            after_text = str(action_after_effects)

        lines.append(f"- step_id: {step_id}")
        lines.append(f"  step_goal: {step_goal}")
        lines.append(f"  action_before_state: {action_before_state}")
        lines.append(f"  action_after_effects: {after_text}")
        lines.append(f"  nl_explanation: {nl_explanation}")
        lines.append("")

    lines.append(
        "Now, cut these steps into one or more higher-level tasks. "
        "A task is a coherent sub-goal from the user's perspective."
    )
    lines.append(
        "For each task, you must specify:\n"
        "- start_step_id (the first step of this task)\n"
        "- end_step_id (the last step of this task)\n"
        "- step_ids (all step_ids included in this task in order)\n"
        "- instruction (a short imperative instruction describing what the agent should do)\n"
        "- task_goal (a short description of the goal / outcome of this task)."
    )
    lines.append(
        "Return ONLY valid JSON with the following schema:\n"
        "{\n"
        '  "segment_id": "<same as the segment key I gave you>",\n'
        '  "tasks": [\n'
        "    {\n"
        '      "task_index": 1,\n'
        '      "start_step_id": "sX",\n'
        '      "end_step_id": "sY",\n'
        '      "step_ids": ["sX", "...", "sY"],\n'
        '      "instruction": "...",\n'
        '      "task_goal": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    lines.append("IMPORTANT: Your output must be in ENGLISH.\n")

    return "\n".join(lines)


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()

    if not raw_text:
        raise ValueError("LLM returned empty string, cannot parse JSON")

    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        candidate = None
        for part in parts:
            if "{" in part and "}" in part:
                candidate = part.strip()
                break
        if candidate:
            raw_text = candidate

    try:
        return json.loads(raw_text)
    except Exception:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw_text[start : end + 1]
            return json.loads(snippet)
        raise


def extract_base_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("steps_time"):
            continue
        if key.startswith("tasks_steps_time"):
            continue
        if key == "llm_task_segments":
            continue
        meta[key] = value
    return meta


def build_task_reports(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = extract_base_meta(source)
    task_keys = [
        key
        for key in source.keys()
        if key.startswith("tasks_steps_time") and "window" not in key
    ]

    task_reports: List[Dict[str, Any]] = []

    for tasks_key in sorted(task_keys):
        seg_key = tasks_key.replace("tasks_", "", 1)
        seg_steps = source.get(seg_key, [])
        llm_result = source.get(tasks_key, {})

        if not isinstance(seg_steps, list) or not seg_steps:
            continue
        if not isinstance(llm_result, dict):
            continue

        tasks = llm_result.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            continue

        id2step: Dict[str, Dict[str, Any]] = {}
        for step in seg_steps:
            if isinstance(step, dict) and "step_id" in step:
                id2step[str(step["step_id"])] = step

        for task in tasks:
            if not isinstance(task, dict):
                continue

            step_ids = task.get("step_ids") or []
            if not isinstance(step_ids, list):
                continue

            task_steps: List[Dict[str, Any]] = []
            for step_id in step_ids:
                step_obj = id2step.get(str(step_id))
                if step_obj is not None:
                    task_steps.append(step_obj)

            if not task_steps:
                continue

            report: Dict[str, Any] = dict(meta)
            report["instruction"] = task.get("instruction", "")
            report["task_goal"] = task.get("task_goal", "")
            report["source_segment"] = seg_key
            report["task_index"] = task.get("task_index", None)
            report["step_ids"] = step_ids
            report["steps"] = task_steps

            task_reports.append(report)

    return task_reports


def run_codex_cut_and_split_for_time(
    input_path: str,
    output_dir: str,
    model: str = "gpt-5.5",
    prefix: str = "report_cutting_llm_task_",
) -> None:
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    source: Dict[str, Any] = dict(data)

    segment_keys = [
        key
        for key in data.keys()
        if key.startswith("steps_time") and "window" not in key
    ]

    if not segment_keys:
        print("No steps_timeX segments found; nothing to cut.")
        return

    for seg_key in sorted(segment_keys):
        steps = data.get(seg_key, [])
        if not isinstance(steps, list) or not steps:
            continue

        print(f"Processing segment (time): {seg_key} (steps: {len(steps)})")

        prompt = build_prompt_for_segment(seg_key, steps)

        try:
            raw_output = call_codex(model, "", prompt)
        except Exception as exc:
            print(f"Codex call failed for segment {seg_key}: {exc}")
            raw_output = ""

        try:
            parsed = parse_llm_json(raw_output)
        except Exception as exc:
            print(f"Failed to parse JSON for segment {seg_key}: {exc}")
            parsed = {
                "segment_id": seg_key,
                "error": str(exc),
                "raw_output": raw_output,
            }

        source[f"tasks_{seg_key}"] = parsed

    task_reports = build_task_reports(source)
    if not task_reports:
        print("No task reports generated (check Codex outputs).")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, report in enumerate(task_reports, start=1):
        out_path = out_dir / f"{prefix}{idx}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved task {idx} to: {out_path}")

    print(f"\nTotal tasks exported from time-based segments: {len(task_reports)}")


def default_output_dir_for_input(input_path: Path) -> Path:
    if input_path.parent.name == "splits":
        return input_path.parent / "tasks"
    return input_path.parent / "splits" / "tasks"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use Codex to split time-based segments into task files."
    )
    parser.add_argument("input_path", help="Path to the report_cutting_time.json file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to write task JSON files into. Defaults to splits/tasks beside the input file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-5.5",
        help="Model name for the Codex call. Default: gpt-5.5.",
    )
    parser.add_argument(
        "--prefix",
        default="report_cutting_llm_task_",
        help="Prefix for generated task files. Default: report_cutting_llm_task_.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir_for_input(input_path)

    run_codex_cut_and_split_for_time(
        input_path=str(input_path),
        output_dir=str(output_dir),
        model=args.model,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
