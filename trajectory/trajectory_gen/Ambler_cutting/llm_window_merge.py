import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_codex_path() -> str:
    return "codex"


def call_codex(model: str, system_prompt: str, user_prompt: str) -> str:
    """发送 prompt 给 Codex，并返回 assistant_message（效果与 codex exec 一致）"""

    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()

    # 把 prompt 写入临时文件
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
            # Windows: 使用 type 命令从文件读取 stdin
            cmd = f"cmd /c \"{base} < {prompt_file}\""
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
        else:
            cmd = shlex.split(base)
            result = subprocess.run(
                cmd,
                stdin=open(prompt_file),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
    finally:
        os.unlink(prompt_file)

    if result.returncode != 0:
        stderr_msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(f"Codex error: {stderr_msg}")

    # 从 stdout 解析 JSON event 流，取最后一个 assistant_message/agent_message
    last_content: Optional[str] = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except:
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


def find_time_window_segments(data: Dict[str, Any]) -> Dict[str, List[Tuple[int, str]]]:
    """
    从 demo_report_cutting_window.json 中提取按 time 分组的 window 段。
    返回结构：
    {
      "steps_time1": [(1, "steps_time1_window1"), (2, "steps_time1_window2")],
      "steps_time2": [(1, "steps_time2_window1"), ...],
      ...
    }
    """
    groups: Dict[str, List[Tuple[int, str]]] = {}

    for key in data.keys():
        if not key.startswith("steps_time"):
            continue
        if "window" not in key:
            continue

        # 形如 steps_time1_window2
        try:
            time_part, window_part = key.split("_window", 1)
            window_idx = int(window_part)
        except Exception:
            # 非预期格式，跳过
            continue

        groups.setdefault(time_part, []).append((window_idx, key))

    # 每个 time 内按 window 序号排序
    for time_key, items in groups.items():
        items.sort(key=lambda x: x[0])
        groups[time_key] = items

    return groups


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """
    尝试从 LLM 返回的文本中解析 JSON。

    - 支持返回被 ```json ... ``` 或 ``` ... ``` 包裹的情况；
    - 如果前后有自然语言，尝试截取第一个 '{' 到最后一个 '}' 之间的片段再解析。
    """
    raw_text = raw_text.strip()

    if not raw_text:
        raise ValueError("LLM returned empty string, cannot parse JSON")

    # 常见情况：LLM 用 ```json ... ``` 或 ``` ... ``` 包裹
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        # 选出包含 '{' 的那一段作为真正的 JSON 内容
        candidate = None
        for part in parts:
            if "{" in part and "}" in part:
                candidate = part.strip()
                break
        if candidate:
            raw_text = candidate

    # 第一轮：直接整体解析
    try:
        return json.loads(raw_text)
    except Exception:
        # 第二轮：有些模型会在 JSON 前后加自然语言，尝试截取第一个 { 到最后一个 } 之间
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw_text[start : end + 1]
            return json.loads(snippet)

        # 仍然失败就抛出异常，由上层去记录 raw_output
        raise


def build_merge_prompt(
    time_key: str,
    prev_key: str,
    next_key: str,
    prev_last_step: Dict[str, Any],
    next_first_step: Dict[str, Any],
) -> str:
    """
    构造给 Codex 的提示词：
    比较同一 time 段内，相邻两个 window 段（例如 steps_time1_window1 和 steps_time1_window2）
    的"衔接性"和"连贯性"，判断是否可以认为它们属于同一个小任务的连续步骤。
    """
    def format_step(step: Dict[str, Any], position: str) -> List[str]:
        now_state = step.get("now_state", {}) or {}
        step_id = step.get("step_id", "")
        step_goal = step.get("step_goal", "")
        action_before_state = step.get("action_before_state", "")
        action_after_effects = step.get("action_after_effects", [])
        nl_explanation = step.get("nl_explanation", "")

        if isinstance(action_after_effects, list):
            after_text = " ".join(str(x) for x in action_after_effects)
        else:
            after_text = str(action_after_effects)

        app_before = now_state.get("app_title_before", "")
        app_after = now_state.get("app_title_after", "")

        ss_before = now_state.get("screenshot_path_before", "")
        ss_after = now_state.get("screenshot_path_after", "")

        lines = [
            f"{position} step_id: {step_id}",
            f"{position} step_goal: {step_goal}",
            f"{position} app_title_before: {app_before}",
            f"{position} app_title_after: {app_after}",
            f"{position} screenshot_before_path: {ss_before}",
            f"{position} screenshot_after_path: {ss_after}",
            f"{position} action_before_state: {action_before_state}",
            f"{position} action_after_effects: {after_text}",
            f"{position} nl_explanation: {nl_explanation}",
        ]
        return lines

    lines: List[str] = []
    lines.append(
        "You are an expert at analyzing sequences of UI interaction steps "
        "and deciding whether two segments of steps belong to the same small, coherent task."
    )
    lines.append(
        f"We are looking at two consecutive window segments within the same time group: {time_key}."
    )
    lines.append(
        f"The previous segment is {prev_key} (we take its LAST step), "
        f"and the next segment is {next_key} (we take its FIRST step)."
    )
    lines.append(
        "Even though the active window title may change, we want to know whether "
        "these two steps are still part of the SAME small task (i.e., they are strongly connected and continuous)."
    )
    lines.append("")
    lines.extend(format_step(prev_last_step, position="PREV_LAST"))
    lines.append("")
    lines.extend(format_step(next_first_step, position="NEXT_FIRST"))
    lines.append("")
    lines.append(
        "Based on all of the above information (goals, before/after states, app titles, and screenshots paths), "
        "decide whether the NEXT_FIRST step is a natural continuation of the PREV_LAST step in the SAME small task."
    )
    lines.append(
        "If they belong to the same small task, we can safely MERGE the two window segments "
        "as one continuous task. Otherwise, they should remain as separate tasks."
    )
    lines.append("")
    lines.append(
        "Return ONLY valid JSON with the following schema:\n"
        "{\n"
        '  "time_group": "<e.g. steps_time1>",\n'
        '  "prev_segment": "<e.g. steps_time1_window1>",\n'
        '  "next_segment": "<e.g. steps_time1_window2>",\n'
        '  "can_merge": true or false,\n'
        '  "reason": "<short explanation in English>"\n'
        "}"
    )

    return "\n".join(lines)


def check_window_merges_with_codex(
    input_path: str,
    model: str = "gpt-5.5",
) -> Dict[str, Any]:
    """
    读取 demo_report_cutting_window.json，
    对于每个 time 组内相邻的 window 段（如 steps_time1_window1 和 steps_time1_window2），
    把前一个段的最后一个 step 和后一个段的第一个 step 交给 Codex 判断是否可以合并为同一个小任务。

    结果保存到 output_path（例如 report_window_merge_codex.json），结构示例：
    {
      "decisions": [
        {
          "time_group": "steps_time1",
          "prev_segment": "steps_time1_window1",
          "next_segment": "steps_time1_window2",
          "can_merge": true,
          "reason": "..."
        },
        ...
      ]
    }
    """
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    groups = find_time_window_segments(data)
    decisions: List[Dict[str, Any]] = []

    # (time_group, prev_seg, next_seg) -> can_merge(bool)
    merge_edges: Dict[Tuple[str, str, str], bool] = {}

    if not groups:
        print("No steps_timeX_windowY segments found; nothing to check.")

    for time_key, items in sorted(groups.items()):
        # items: List[(window_idx, segment_key)]
        if len(items) < 2:
            continue  # 只有一个 window 段，无需比较

        for (idx1, seg1), (idx2, seg2) in zip(items[:-1], items[1:]):
            steps1 = data.get(seg1, [])
            steps2 = data.get(seg2, [])
            if not isinstance(steps1, list) or not steps1:
                continue
            if not isinstance(steps2, list) or not steps2:
                continue

            prev_last_step = steps1[-1]
            next_first_step = steps2[0]

            print(
                f"Checking merge between {seg1} (last step_id={prev_last_step.get('step_id')}) "
                f"and {seg2} (first step_id={next_first_step.get('step_id')})"
            )

            prompt = build_merge_prompt(
                time_key=time_key,
                prev_key=seg1,
                next_key=seg2,
                prev_last_step=prev_last_step,
                next_first_step=next_first_step,
            )

            try:
                raw_output = call_codex(model, "", prompt)
            except Exception as e:
                print(f"Codex call failed for pair {seg1} -> {seg2}: {e}")
                raw_output = ""

            try:
                parsed = parse_llm_json(raw_output)
            except Exception as e:
                print(f"Failed to parse JSON for pair {seg1} -> {seg2}: {e}")
                parsed = {
                    "time_group": time_key,
                    "prev_segment": seg1,
                    "next_segment": seg2,
                    "error": str(e),
                    "raw_output": raw_output,
                }
                # 解析失败时，保守起见视为不能合并
                merge_edges[(time_key, seg1, seg2)] = False
            else:
                # 标准化字段并提取 can_merge
                t_group = parsed.get("time_group", time_key)
                prev_seg = parsed.get("prev_segment", seg1)
                next_seg = parsed.get("next_segment", seg2)
                can_merge = bool(parsed.get("can_merge", False))
                merge_edges[(t_group, prev_seg, next_seg)] = can_merge

            decisions.append(parsed)

    # 基于 merge_edges，把同一 time 组内的 window 段合并成 mix 段
    mixed_data: Dict[str, Any] = dict(data)

    for time_key, items in sorted(groups.items()):
        if not items:
            continue

        # 保持原始 window 段不变，仅额外生成 *_mix_windowX
        ordered_seg_keys = [seg for _, seg in items]

        # 从第一个 window 段开始累积
        current_steps: List[Dict[str, Any]] = []
        mixed_segments: List[List[Dict[str, Any]]] = []

        # 初始化为第一个段的步骤
        first_seg = ordered_seg_keys[0]
        first_steps = data.get(first_seg, [])
        if isinstance(first_steps, list):
            current_steps.extend(first_steps)

        # 依次查看后续段，决定是否并入当前 mix 段
        for prev_seg, next_seg in zip(ordered_seg_keys[:-1], ordered_seg_keys[1:]):
            edge_key = (time_key, prev_seg, next_seg)
            can_merge = merge_edges.get(edge_key, False)

            next_steps = data.get(next_seg, [])
            if not isinstance(next_steps, list):
                next_steps = []

            if can_merge:
                # 可以合并：把 next 段的步骤接在当前 mix 段后面
                current_steps.extend(next_steps)
            else:
                # 不能合并：先收束当前 mix 段，再开启新的 mix 段
                if current_steps:
                    mixed_segments.append(current_steps)
                current_steps = list(next_steps)

        # 最后一段收束
        if current_steps:
            mixed_segments.append(current_steps)

        # 写入新的 mix 段键：steps_timeN_mix_window1, 2, ...
        for idx, seg_steps in enumerate(mixed_segments, start=1):
            mix_key = f"{time_key}_mix_window{idx}"
            mixed_data[mix_key] = seg_steps

    # 也把原始的 Codex 决策记录下来，方便调试
    mixed_data["window_merge_decisions"] = decisions

    return mixed_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use Codex to decide whether adjacent window segments should be merged."
    )
    parser.add_argument("input_path", help="Path to the report_cutting_window.json file.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Path to the output JSON file. Defaults to <input_stem>_mix.json.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-5.5",
        help="Model name for the Codex call. Default: gpt-5.5.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path.with_name(
        f"{input_path.stem}_mix.json"
    )

    mixed_data = check_window_merges_with_codex(
        input_path=str(input_path),
        model=args.model,
    )

    # 保存为「带 *_mix_windowX 的」新报告
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mixed_data, f, indent=2, ensure_ascii=False)

    print(f"\nMixed window report saved to: {output_path}")


if __name__ == "__main__":
    main()
