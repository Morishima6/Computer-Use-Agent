import os
import json
import re
from pathlib import Path
from typing import List

try:
    from .common_llm_call import get_embedding
except Exception:
    from common_llm_call import get_embedding


# Tool1 - embed_report_file: 嵌入 report.json，指定需要嵌入哪些 key
""" A Demo to Use Tool1:
    workspace_root = "/Users/yepyoung/Desktop/MyProject/Ambler-yangyaopeng"
    embed_fields = ["action_before_state", "step_goal", "instruction"]
    trace_base_dir = Path(workspace_root) / "Ambler-Agent" / "trajectory" / "trajectory_base"
    target_folders = [
        "chrome_0d8b7de3-e8de-4d86-b9fd-dd2dce58a217_20251128_193658",
        "chrome_2ad9387a-65d8-4e33-ad5b-7580065a27ca_20251128_195352"
    ]
    for folder in target_folders:
        report_file_path = trace_base_dir / folder / "report.json"
        embed_report_file(str(report_file_path), fields=embed_fields)
"""




def save_json_compact_embeddings(data, file_path):
    """
    保存 JSON，但确保所有以 _embedding 结尾的字段数组在一行内。
    """
    json_text = None
    pattern = None
    f = None

    json_text = json.dumps(data, indent=2, ensure_ascii=False)
    
    pattern = r'("\w+_embedding":\s*\[)([^\]]+)(\])'
    
    def collapse_array(match):
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        collapsed_content = re.sub(r'\s+', ' ', content).strip()
        return f"{prefix}{collapsed_content}{suffix}"

    json_text = re.sub(pattern, collapse_array, json_text)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_text)

def embed_report_file(
    report_path: str, fields: List[str] = ["action_before_state"], overwrite: bool = False
):
    """
    处理 report.json：根据指定的 fields 生成 embedding。
    支持: instruction (任务级), step_goal, action_before_state, nl_explanation (步骤级)
    """
    # C89 风格
    report_path_obj = None
    data = None
    steps = None
    step = None
    field = None
    text = None
    embedding = None
    emb_key = None
    f = None

    report_path_obj = Path(report_path)
    if not report_path_obj.exists():
        print(f"文件不存在: {report_path}")
        return

    try:
        with open(report_path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. 处理任务级字段 (如 instruction)
        for field in fields:
            if field in data and isinstance(data[field], str):
                emb_key = f"{field}_embedding"
                if overwrite or emb_key not in data:
                    print(f"🧶 正在为任务级字段 '{field}' 生成 embedding...")
                    embedding = get_embedding(data[field])
                    if embedding:
                        data[emb_key] = embedding

        # 2. 处理步骤级字段 (如 step_goal, action_before_state, nl_explanation)
        steps = data.get('steps', [])
        for step in steps:
            for field in fields:
                # 如果字段存在于步骤中且尚未生成 embedding
                if field in step and isinstance(step[field], str):
                    emb_key = f"{field}_embedding"
                    if overwrite or emb_key not in step:
                        print(f"🧶 正在为步骤 {step.get('step_id')} 的 '{field}' 生成 embedding...")
                        embedding = get_embedding(step[field])
                        if embedding:
                            step[emb_key] = embedding
                        else:
                            print(f"警告: 无法为 {step.get('step_id')} 的 {field} 生成 embedding")
    
        save_json_compact_embeddings(data, str(report_path_obj))
        print(f"🌵 成功更新文件: {report_path}")
        
    except Exception as e:
        print(f"处理文件 {report_path} 时出错: {e}")


def embed_trajectory_base(
    trace_root_dir: str, fields: List[str], overwrite: bool = False
) -> int:
    """
    遍历 trajectory_base/**/report.json 并为指定 fields 批量写入 embedding。
    fields 可包含: instruction, step_goal, action_before_state, nl_explanation 等。
    """
    updated = 0
    base = Path(trace_root_dir)
    if not base.exists():
        return 0

    report_paths = list(base.glob("*/report.json"))
    for report_path in report_paths:
        before_mtime = report_path.stat().st_mtime
        embed_report_file(str(report_path), fields=fields, overwrite=overwrite)
        after_mtime = report_path.stat().st_mtime
        if after_mtime != before_mtime:
            updated += 1

    return updated


def embed_task_instruction_embeddings(trace_root_dir: str, overwrite: bool = False) -> int:
    """批量写入 task 级 instruction_embedding。"""
    return embed_trajectory_base(trace_root_dir, fields=["instruction"], overwrite=overwrite)
