#!/usr/bin/env python3
"""
数据处理流水线脚本
==================

功能：
1. 预处理：调用 remove_first_step.py 删除第一步
2. 单次处理：调用 Codex 填充 report.json，然后按顺序执行以下处理步骤：
   - generate_report_cutting_time.py  (按时间切分)
   - generate_report_cutting_window.py (按窗口切分)
   - llm_window_merge.py               (LLM 合并窗口)
   - llm_task_cut_mix_to_files.py     (LLM 任务切割并生成文件)

2. 批量处理：支持对整个数据文件夹批量执行上述流程

用法：
    # 单次处理单个会话文件夹
    python pipeline.py --single /path/to/session_folder

    # 批量处理整个数据目录
    python pipeline.py --batch /path/to/data_directory

    # 仅执行特定步骤（跳过 Codex 调用）
    python pipeline.py --single /path/to/session_folder --skip-codex

    # 跳过删除第一步的预处理步骤
    python pipeline.py --single /path/to/session_folder --skip-remove-first-step

    # 自定义参数
    python pipeline.py --single /path/to/session_folder --codex-model gpt-5.4 --llm-model gpt-5.1
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil


# ============ 配置区 ============
# 各脚本的路径（根据实际情况调整）
SCRIPT_DIR = Path(__file__).parent.resolve()

CALL_CODEX_DIR = SCRIPT_DIR / "call_codex"
AMBLER_CUTTING_DIR = SCRIPT_DIR / "Ambler_cutting"
TRAJECTORY_PREPROCESS_DIR = SCRIPT_DIR / "trajectory_preprocess"

CALL_CODEX_SCRIPT = CALL_CODEX_DIR / "call_codex.py"
CUTTING_TIME_SCRIPT = AMBLER_CUTTING_DIR / "generate_report_cutting_time.py"
CUTTING_WINDOW_SCRIPT = AMBLER_CUTTING_DIR / "generate_report_cutting_window.py"
WINDOW_MERGE_SCRIPT = AMBLER_CUTTING_DIR / "llm_window_merge.py"
TASK_CUT_SCRIPT = AMBLER_CUTTING_DIR / "llm_task_cut_mix_to_files.py"
REMOVE_FIRST_STEP_SCRIPT = TRAJECTORY_PREPROCESS_DIR / "remove_first_step.py"

# 默认模型
DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_LLM_MODEL = "gpt-5.4"

# 时间切分阈值（秒）
DEFAULT_TIME_THRESHOLD = 30

# 标记文件名
SIGN_FILE = "sign.txt"


# ============ 工具函数 ============
def get_current_time_str() -> str:
    """获取当前时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """格式化时长为可读字符串"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def write_sign_file(session_folder: Path) -> None:
    """创建 sign.txt 标记文件，记录处理完成时间"""
    sign_path = session_folder / SIGN_FILE
    try:
        with open(sign_path, "w", encoding="utf-8") as f:
            f.write(f"Pipeline processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except OSError as e:
        print(f"[警告] 无法写入 sign.txt: {e}")


def run_python_script(script_path: Path, *args, **kwargs) -> int:
    """运行 Python 脚本并返回退出码"""
    cmd = [sys.executable, str(script_path)] + list(args)
    for k, v in kwargs.items():
        # 将 --key value 转换为 --key=value 格式
        cmd.append(f"--{k}")
        if v is not True:  # 如果不是布尔 flag
            cmd.append(str(v))

    print(f"\n{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace")
    return result.returncode


def find_report_json_files(data_dir: Path) -> List[Path]:
    """在目录中查找所有 report.json 文件，排除已有 sign.txt 的文件夹"""
    report_files = []
    for item in data_dir.iterdir():
        if item.is_dir():
            report_path = item / "report.json"
            sign_path = item / SIGN_FILE
            # 跳过已有 sign.txt 的文件夹（已处理过）
            if report_path.exists() and not sign_path.exists():
                report_files.append(item)  # 返回会话文件夹而非文件本身
    return sorted(report_files)


def get_screenshot_root(session_folder: Path) -> Path:
    """获取截图根目录"""
    # 通常截图在 session_folder/screenshots 或 session_folder 下
    ss_dir = session_folder / "screenshots"
    if ss_dir.exists():
        return ss_dir
    return session_folder


def process_single_session(
    session_folder: Path,
    codex_model: str,
    llm_model: str,
    time_threshold: int,
    skip_codex: bool,
    skip_llm: bool,
    skip_remove_first_step: bool = False,
) -> bool:
    """
    处理单个会话文件夹，返回是否成功
    """
    session_folder = Path(session_folder)
    report_path = session_folder / "report.json"

    # 记录开始时间
    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print(f"# 开始处理会话: {session_folder.name}")
    print(f"# 开始时间: {get_current_time_str()}")
    print(f"# report.json 路径: {report_path}")
    print(f"{'#'*60}")

    # ========== 步骤 0: 预处理 - 删除第一步 ==========
    if not skip_remove_first_step:
        print("\n>>> 步骤 0: 预处理 - 删除第一步 (remove_first_step.py)...")
        ret = run_python_script(
            REMOVE_FIRST_STEP_SCRIPT,
            str(session_folder),
        )
        if ret != 0:
            print(f"[错误] 删除第一步失败，退出码: {ret}")
            return False
        print(f"✓ 删除第一步完成")
        print(f"  当前时间: {get_current_time_str()}")
    else:
        print("\n>>> 步骤 0: 跳过删除第一步（--skip-remove-first-step）")
        print(f"  当前时间: {get_current_time_str()}")

    # ========== 步骤 1: 调用 Codex 填充 report.json ==========
    if not skip_codex:
        print("\n>>> 步骤 1: 调用 Codex 填充 report.json...")
        # 直接调用 call_codex 函数
        sys.path.insert(0, str(CALL_CODEX_DIR))
        try:
            from call_codex import call_codex as codex_call

            # 准备 prompt（与原始脚本相同）
            system_prompt = (
                "You are an action-behavior analyst and recorder.\n"
                "\n"
                "Your task:\n"
                "Analyze a report.json file that records a single task containing multiple UI actions. "
                "For each action (step) you must infer a structured description strictly from the JSON metadata and screenshots, "
                "and output a machine-readable JSON array that can be used to fill the empty fields in report.json.\n\n"
                "For each action/step you MUST produce one JSON object with exactly these fields:\n"
                '- "task_title": A concise title summarizing the overall task (same value for all actions).\n'
                '- "step_goal": A short phrase describing the immediate goal of this specific action within the overall task.\n'
                '- "app": The software/application used during the task.\n'
                '- "url": Any URL relevant to the task or the specific action.\n'
                '- "action_preconditions": What must be true or present before the action occurs (based on the before screenshot).\n'
                '- "nl_position": A natural-language description of the mouse location or targeted UI element (based on the red marker in the before screenshot). If the step has no on-screen target (for example, a typing or press action where "action.target" is missing or an empty object in report.json), set this field to null instead of describing any location. If you cannot confidently identify what the element is or what text it contains, instead describe its visual appearance (shape, color, approximate size) and relative location (for example, "a blue rectangular button near the top-right corner").\n'
                '- "action_before_state": The UI state or condition before the action.\n'
                '- "action_after_effects": The changes caused by the action (based on the after screenshot).\n'
                '- "nl_explanation": A concise, natural-language explanation of the action and its purpose, written without referring to "the user" (describe the step itself, for example, "Click the Save button to store the changes.").\n\n'
                "Output format requirements (very important):\n"
                "- The FINAL answer must be a single JSON array (e.g. [ { ... }, { ... }, ... ]) with one object per action.\n"
                "- Do not print any explanations, comments, or non-JSON text in the final answer.\n"
                "- Do not include trailing commas. The JSON must be strictly valid.\n"
            )

            user_prompt = (
                "You are given a conversation folder located at:\n"
                f"{session_folder}\n\n"
                "Inside this folder there is a report.json file and a screenshots/ subfolder referenced by it.\n\n"
                "Your job now:\n"
                "1) Read report.json in that folder.\n"
                "2) For each step in report.json.steps, carefully inspect:\n"
                "   - The overall task instruction or user prompt in report.json (for example, the \"instruction\" field)\n"
                "   - Its metadata in the JSON, including any screenshot paths (such as screenshot_path_before_part)\n"
                "   - The before screenshot, the partial before screenshot near the signed position, and the after screenshot\n"
                "   - The red-highlighted mouse position\n"
                "   - Any relevant application and URL information\n"
                "3) Then produce ONE JSON array as the final answer. Each element in the array corresponds to one step, "
                "and must contain the fields described in the system prompt: task_title, step_goal, app, url, "
                "action_preconditions, nl_position, action_before_state, action_after_effects, nl_explanation.\n\n"
                "Remember: the final answer must be ONLY that JSON array, with no extra commentary or text."
            )

            print(f"调用 Codex 模型: {codex_model}")
            ai_response = codex_call(codex_model, system_prompt, user_prompt)

            # 解析并填充 report.json
            try:
                ai_data = json.loads(ai_response)
            except json.JSONDecodeError as e:
                print(f"[错误] 解析 Codex 返回的 JSON 失败: {e}")
                return False

            if not isinstance(ai_data, list):
                print(f"[错误] Codex 返回的不是 JSON 数组: {type(ai_data).__name__}")
                return False

            # 读入原始 report.json 并填充
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            steps = report.get("steps", [])

            # 填充顶层字段
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

            # 填充每个 step
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
                    if (nl_pos in (None, [], "")) and action_info.get("nl_position"):
                        target["nl_position"] = [action_info["nl_position"]]

                if step.get("action_before_state") in (None, "") and action_info.get("action_before_state"):
                    step["action_before_state"] = action_info["action_before_state"]

                if step.get("action_after_effects") in (None, [], "") and action_info.get("action_after_effects"):
                    step["action_after_effects"] = [action_info["action_after_effects"]]

                if step.get("nl_explanation") in (None, "") and action_info.get("nl_explanation"):
                    step["nl_explanation"] = action_info["nl_explanation"]

            # 写回 report_filled.json（保留原始 report.json）
            filled_path = report_path.parent / "report_filled.json"
            with open(filled_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"✓ Codex 填充完成: {filled_path}")
            print(f"  当前时间: {get_current_time_str()}")

        except Exception as e:
            print(f"[错误] Codex 调用失败: {e}")
            return False
        finally:
            sys.path.pop(0)
    else:
        print("\n>>> 步骤 1: 跳过 Codex 调用（--skip-codex）")
        print(f"  当前时间: {get_current_time_str()}")
        filled_path = report_path  # 使用原始文件继续后续步骤

    # ========== 创建 splits 文件夹 ==========
    splits_dir = session_folder / "splits"
    splits_dir.mkdir(exist_ok=True)

    # ========== 步骤 2: 按时间切分 ==========
    print("\n>>> 步骤 2: 按时间切分 (generate_report_cutting_time.py)...")
    time_output = splits_dir / f"{report_path.stem}_cutting_time.json"
    ret = run_python_script(
        CUTTING_TIME_SCRIPT,
        str(filled_path),
        "-o", str(time_output),
        "-t", str(time_threshold),
    )
    if ret != 0:
        print(f"[错误] 时间切分失败，退出码: {ret}")
        return False
    print(f"✓ 时间切分完成: {time_output}")
    print(f"  当前时间: {get_current_time_str()}")

    # ========== 步骤 3: 按窗口切分 ==========
    print("\n>>> 步骤 3: 按窗口切分 (generate_report_cutting_window.py)...")
    window_output = splits_dir / f"{report_path.stem}_cutting_window.json"

    ret = run_python_script(
        CUTTING_WINDOW_SCRIPT,
        str(time_output),
        "-o", str(window_output),
    )
    if ret != 0:
        print(f"[错误] 窗口切分失败，退出码: {ret}")
        return False
    print(f"✓ 窗口切分完成: {window_output}")
    print(f"  当前时间: {get_current_time_str()}")

    # ========== 步骤 4: Codex 合并窗口 ==========
    if not skip_llm:
        print("\n>>> 步骤 4: Codex 合并窗口 (llm_window_merge.py)...")
        merge_output = splits_dir / f"{window_output.stem}_mix.json"

        ret = run_python_script(
            WINDOW_MERGE_SCRIPT,
            str(window_output),
            "-o", str(merge_output),
            "-m", codex_model,  # 使用 codex_model
        )
        if ret != 0:
            print(f"[错误] Codex 窗口合并失败，退出码: {ret}")
            return False
        print(f"✓ Codex 窗口合并完成: {merge_output}")
        print(f"  当前时间: {get_current_time_str()}")
    else:
        print("\n>>> 步骤 4: 跳过 Codex 窗口合并（--skip-llm）")
        print(f"  当前时间: {get_current_time_str()}")
        merge_output = window_output

    # ========== 步骤 5: Codex 任务切割并生成文件 ==========
    if not skip_llm:
        print("\n>>> 步骤 5: Codex 任务切割 (llm_task_cut_mix_to_files.py)...")
        # 任务文件输出到 splits_dir/tasks/
        output_dir = splits_dir / "tasks"
        output_dir.mkdir(exist_ok=True)

        ret = run_python_script(
            TASK_CUT_SCRIPT,
            str(merge_output),
            "-o", str(output_dir),
            "-m", codex_model,  # 使用 codex_model
        )
        if ret != 0:
            print(f"[错误] Codex 任务切割失败，退出码: {ret}")
            return False
        print(f"✓ Codex 任务切割完成，输出目录: {output_dir}")
        print(f"  当前时间: {get_current_time_str()}")
    else:
        print("\n>>> 步骤 5: 跳过 Codex 任务切割（--skip-llm）")
        print(f"  当前时间: {get_current_time_str()}")

    # 计算总耗时
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print(f"\n{'#'*60}")
    print(f"# 会话处理完成: {session_folder.name}")
    print(f"# 完成时间: {get_current_time_str()}")
    print(f"# 总耗时: {format_duration(total_duration)}")
    print(f"{'#'*60}\n")

    # 成功后写入 sign.txt 标记
    write_sign_file(session_folder)

    return True


def process_batch(
    data_dir: Path,
    codex_model: str,
    llm_model: str,
    time_threshold: int,
    skip_codex: bool,
    skip_llm: bool,
    skip_remove_first_step: bool = False,
) -> None:
    """
    批量处理数据目录中的所有会话文件夹
    """
    data_dir = Path(data_dir)

    # 统计已处理的文件夹（有 sign.txt 的）
    all_folders_with_report = []
    for item in data_dir.iterdir():
        if item.is_dir() and (item / "report.json").exists():
            all_folders_with_report.append(item)

    skipped_count = sum(1 for f in all_folders_with_report if (f / SIGN_FILE).exists())

    # 查找所有包含 report.json 且没有 sign.txt 的文件夹
    session_folders = find_report_json_files(data_dir)

    if not session_folders:
        print(f"\n[信息] 在 {data_dir} 中未找到需要处理的会话文件夹")
        if skipped_count > 0:
            print(f"[信息] {skipped_count} 个文件夹已有 sign.txt，已跳过")
        return

    print(f"\n{'='*60}")
    print(f"批量处理模式")
    print(f"数据目录: {data_dir}")
    print(f"找到 {len(session_folders)} 个会话文件夹待处理")
    if skipped_count > 0:
        print(f"{skipped_count} 个文件夹已有 sign.txt，将跳过")
    print(f"开始时间: {get_current_time_str()}")
    print(f"{'='*60}\n")

    batch_start_time = datetime.now()
    success_count = 0
    fail_count = 0

    for i, session_folder in enumerate(session_folders, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(session_folders)}] 准备处理: {session_folder.name}")
        print(f"{'='*60}")

        try:
            success = process_single_session(
                session_folder,
                codex_model=codex_model,
                llm_model=llm_model,
                time_threshold=time_threshold,
                skip_codex=skip_codex,
                skip_llm=skip_llm,
                skip_remove_first_step=skip_remove_first_step,
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"[错误] 处理会话 {session_folder.name} 时发生异常: {e}")
            fail_count += 1

    # 计算批量处理总耗时
    batch_end_time = datetime.now()
    batch_total_duration = (batch_end_time - batch_start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"完成时间: {get_current_time_str()}")
    print(f"总耗时: {format_duration(batch_total_duration)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(session_folders)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="数据处理流水线：串接 Codex 调用和 Ambler 切分脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单次处理单个会话文件夹
  python pipeline.py --single "E:/data/session_001"

  # 批量处理整个数据目录
  python pipeline.py --batch "E:/osworld_data"

  # 跳过 Codex 调用（已有填充好的 report.json）
  python pipeline.py --single "E:/data/session_001" --skip-codex

  # 跳过 LLM 步骤（仅执行切分）
  python pipeline.py --single "E:/data/session_001" --skip-llm

  # 自定义模型
  python pipeline.py --single "E:/data/session_001" --codex-model gpt-5.4 --llm-model gpt-5.1
        """
    )

    # 互斥组：单次处理 vs 批量处理
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--single",
        metavar="FOLDER",
        help="单次处理：指定单个会话文件夹路径",
    )
    mode_group.add_argument(
        "--batch",
        metavar="DIRECTORY",
        help="批量处理：指定数据目录（将处理目录下所有包含 report.json 的子文件夹）",
    )

    # 可选参数
    parser.add_argument(
        "--codex-model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex 模型名称 (默认: {DEFAULT_CODEX_MODEL})",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM 模型名称 (默认: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "-t", "--time-threshold",
        type=int,
        default=DEFAULT_TIME_THRESHOLD,
        help=f"时间切分阈值，秒 (默认: {DEFAULT_TIME_THRESHOLD})",
    )
    parser.add_argument(
        "--skip-codex",
        action="store_true",
        help="跳过 Codex 调用步骤（当 report.json 已填充完成时使用）",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="跳过 LLM 相关步骤（窗口合并和任务切割）",
    )
    parser.add_argument(
        "--skip-remove-first-step",
        action="store_true",
        help="跳过删除第一步的预处理步骤",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要处理的文件夹列表，不实际执行",
    )

    args = parser.parse_args()

    # 验证脚本路径
    for script in [CALL_CODEX_SCRIPT, CUTTING_TIME_SCRIPT, CUTTING_WINDOW_SCRIPT,
                   WINDOW_MERGE_SCRIPT, TASK_CUT_SCRIPT, REMOVE_FIRST_STEP_SCRIPT]:
        if not script.exists():
            print(f"[错误] 找不到脚本: {script}")
            sys.exit(1)

    # 执行处理
    if args.single:
        folder = Path(args.single)
        if not folder.exists():
            print(f"[错误] 文件夹不存在: {folder}")
            sys.exit(1)

        if args.dry_run:
            print(f"[Dry Run] 将处理单个会话: {folder}")
        else:
            process_single_session(
                folder,
                codex_model=args.codex_model,
                llm_model=args.llm_model,
                time_threshold=args.time_threshold,
                skip_codex=args.skip_codex,
                skip_llm=args.skip_llm,
                skip_remove_first_step=args.skip_remove_first_step,
            )

    elif args.batch:
        directory = Path(args.batch)
        if not directory.exists():
            print(f"[错误] 目录不存在: {directory}")
            sys.exit(1)

        if args.dry_run:
            folders = find_report_json_files(directory)
            print(f"[Dry Run] 将处理以下 {len(folders)} 个会话文件夹:")
            for f in folders:
                print(f"  - {f}")
        else:
            process_batch(
                directory,
                codex_model=args.codex_model,
                llm_model=args.llm_model,
                time_threshold=args.time_threshold,
                skip_codex=args.skip_codex,
                skip_llm=args.skip_llm,
                skip_remove_first_step=args.skip_remove_first_step,
            )


if __name__ == "__main__":
    main()
