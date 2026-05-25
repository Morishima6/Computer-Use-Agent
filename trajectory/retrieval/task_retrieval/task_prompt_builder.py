import json
import os
from typing import Any, Dict, Optional


def find_trace_by_instruction(
    instruction: str,
    trace_root: str,
) -> Optional[Dict[str, Any]]:
    """
    在 trajectory_base 目录下递归查找与给定 instruction 完全相同的 report.json，
    找到后返回整个 JSON（字典）。
    """
    if not instruction:
        return None

    if not os.path.isdir(trace_root):
        raise FileNotFoundError(f"trace_root 不存在或不是目录: {trace_root}")

    for root, _dirs, files in os.walk(trace_root):
        if "report.json" not in files:
            continue

        report_path = os.path.join(root, "report.json")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if data.get("instruction") == instruction:
            return data

    return None


def build_task_reference_prompt(trace: Dict[str, Any]) -> str:
    """只提取 task_title/instruction/steps[].nl_explanation 生成 prompt。"""
    if not trace:
        return ""

    task_title = trace.get("task_title", "")
    instruction = trace.get("instruction", "")
    steps = trace.get("steps") or []

    lines = []

    if task_title:
        lines.append(f"Task Title: {task_title}")

    if instruction:
        lines.append(f"Instruction: {instruction}")

    if steps:
        lines.append("Step Explanations:")
        for idx, step in enumerate(steps, start=1):
            nl_explanation = (step or {}).get("nl_explanation", "")
            if nl_explanation:
                lines.append(f"{idx}. {nl_explanation}")

    return "\n".join(lines).strip()


def build_prompt_from_trace(trace: Dict[str, Any]) -> str:
    return build_task_reference_prompt(trace)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python task_prompt_builder.py '<instruction 文本>'")
        sys.exit(1)

    query_instruction = sys.argv[1]

    # 根据当前文件位置推导 trajectory_base 目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trace_root_dir = os.path.join(base_dir, "trajectory_base")

    trace_data = find_trace_by_instruction(query_instruction, trace_root_dir)
    if not trace_data:
        print("未在 trajectory_base 中找到匹配的 instruction 对应的 report.json")
        sys.exit(0)

    prompt_text = build_prompt_from_trace(trace_data)
    print(prompt_text)
