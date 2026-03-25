import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import openai  # type: ignore


def build_prompt_for_segment(segment_key: str, steps: List[Dict[str, Any]]) -> str:
    """
    根据一个 steps_timeX_mix_windowY 段构造给 LLM 的提示词。
    只包含 step_id、step_goal、action_before_state、action_after_effects、nl_explanation 这几个字段。
    """
    lines: List[str] = []
    lines.append(
        "You are an analyst that groups low-level UI steps into higher-level user tasks."
    )
    lines.append(
        f"The following steps all belong to one contiguous segment: {segment_key}."
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
        "}"
    )

    return "\n".join(lines)


def call_llm(prompt: str, model: str = "gpt-5.1") -> str:
    """
    调用 OpenAI / 兼容 OpenAI 接口的 LLM，返回纯文本输出。

    - 使用环境变量 OPENAI_API_KEY 作为密钥；
    - 如需自定义网关（例如代理服务），可通过环境变量 OPENAI_BASE_URL 配置 base_url。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

    base_url = os.getenv("OPENAI_BASE_URL")

    if base_url:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)  # pragma: no cover
    else:
        # 默认走官方地址 https://api.openai.com/v1
        client = openai.OpenAI(api_key=api_key)  # pragma: no cover

    resp = client.chat.completions.create(  # pragma: no cover
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    if not resp.choices:
        raise RuntimeError("LLM returned no choices.")  # pragma: no cover

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned empty content.")  # pragma: no cover

    return content


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


def extract_base_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 LLM 切割后的总 JSON 中抽取通用的元信息，
    去掉所有 steps_xxx、tasks_steps_xxx、llm_task_segments 这些字段。
    """
    meta: Dict[str, Any] = {}
    for k, v in data.items():
        if k.startswith("steps_time"):
            continue
        if k.startswith("tasks_steps_time"):
            continue
        if k == "llm_task_segments":
            continue
        meta[k] = v
    return meta


def build_task_reports(
    source: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    根据 source 中的 tasks_steps_timeX_mix_windowY / tasks_steps_timeX_windowY 索引，
    为每一个 task 生成一个独立的 task 报告 JSON 对象。

    返回的列表中，每个元素对应一个任务，结构示例：
    {
      ...  // 原始元信息（task_id, env 等）
      "instruction": <LLM 生成的 instruction>,
      "task_goal": <LLM 生成的 task_goal>,
      "source_segment": "steps_time1_mix_window1",
      "task_index": 1,
      "step_ids": ["s1", "s2", "s3"],
      "steps": [ ... 对应的 step 对象 ... ]
    }
    """
    meta = extract_base_meta(source)

    # 找出所有 LLM 任务结果的键：tasks_steps_timeX_*
    task_keys = [
        k for k in source.keys() if k.startswith("tasks_steps_time") and "window" in k
    ]

    task_reports: List[Dict[str, Any]] = []

    for tasks_key in sorted(task_keys):
        seg_key = tasks_key.replace("tasks_", "", 1)  # 对应的 steps_timeX_*
        seg_steps = source.get(seg_key, [])
        llm_result = source.get(tasks_key, {})

        if not isinstance(seg_steps, list) or not seg_steps:
            continue
        if not isinstance(llm_result, dict):
            continue

        tasks = llm_result.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            # 可能解析失败，仅有 error/raw_output
            continue

        # 建立 step_id -> step 对象 的索引，方便按 ID 取 step
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

            # 按 step_ids 的顺序取出完整 step 对象
            task_steps: List[Dict[str, Any]] = []
            for sid in step_ids:
                sid_str = str(sid)
                step_obj = id2step.get(sid_str)
                if step_obj is not None:
                    task_steps.append(step_obj)

            if not task_steps:
                continue

            report: Dict[str, Any] = dict(meta)  # 复制一份元信息
            report["instruction"] = task.get("instruction", "")
            report["task_goal"] = task.get("task_goal", "")
            report["source_segment"] = seg_key
            report["task_index"] = task.get("task_index", None)
            report["step_ids"] = step_ids
            report["steps"] = task_steps

            task_reports.append(report)

    return task_reports


def run_llm_cut_and_split_for_mix(
    input_path: str,
    output_dir: str,
    model: str = "gpt-5.1",
    prefix: str = "report_cutting_llm_task_",
) -> None:
    """
    读取 demo_report_cutting_window_mix.json，针对 steps_timeX_mix_windowY 段：
      1）把每个段内的所有 step 拼接成 prompt，交给 LLM 进行 task 切割；
      2）根据 LLM 输出的任务边界，生成独立的 task 级别 JSON 文件：
         report_cutting_llm_task_1.json, report_cutting_llm_task_2.json, ...

    注意：这里不再使用原始的 steps_timeX_windowY，而是使用 steps_timeX_mix_windowY。
    """
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # 在原始数据上追加 LLM 结果（tasks_steps_timeX_mix_windowY）
    source: Dict[str, Any] = dict(data)

    # 找到所有 mix 段：steps_timeX_mix_windowY
    segment_keys = [
        k for k in data.keys() if k.startswith("steps_time") and "mix_window" in k
    ]

    if not segment_keys:
        print("No steps_timeX_mix_windowY segments found; nothing to cut.")
        return

    for seg_key in sorted(segment_keys):
        steps = data.get(seg_key, [])
        if not isinstance(steps, list) or not steps:
            continue

        print(f"Processing segment (mix): {seg_key} (steps: {len(steps)})")

        prompt = build_prompt_for_segment(seg_key, steps)
        raw_output = call_llm(prompt, model=model)

        try:
            parsed = parse_llm_json(raw_output)
        except Exception as e:
            print(f"Failed to parse JSON for segment {seg_key}: {e}")
            parsed = {
                "segment_id": seg_key,
                "error": str(e),
                "raw_output": raw_output,
            }

        # 写入 tasks_steps_timeX_mix_windowY
        tasks_key = f"tasks_{seg_key}"
        source[tasks_key] = parsed

    # 利用已有的 build_task_reports，将 tasks_* + steps_* 结构转为独立 task 报告
    task_reports = build_task_reports(source)
    if not task_reports:
        print("No task reports generated (check LLM outputs).")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, report in enumerate(task_reports, start=1):
        out_path = out_dir / f"{prefix}{idx}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved task {idx} to: {out_path}")

    print(f"\nTotal tasks exported from mix segments: {len(task_reports)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use an LLM to split mixed window segments into task files."
    )
    parser.add_argument("input_path", help="Path to the report_cutting_window_mix.json file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to write task JSON files into. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-5.1",
        help="Model name for the LLM call. Default: gpt-5.1.",
    )
    parser.add_argument(
        "--prefix",
        default="report_cutting_llm_task_",
        help="Prefix for generated task files. Default: report_cutting_llm_task_.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    run_llm_cut_and_split_for_mix(
        input_path=str(input_path),
        output_dir=str(output_dir),
        model=args.model,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
