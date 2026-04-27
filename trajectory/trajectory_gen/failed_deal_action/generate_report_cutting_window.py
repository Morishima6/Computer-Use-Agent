import argparse
import json
from pathlib import Path


def cut_steps_by_window(report_path: str, output_path: str):
    """
    读取已经按时间切割的 JSON 报告文件（包含 steps_time1, steps_time2 等），
    对每个 steps_timeX 数组，根据 app_title 的变化进行二次切割。
    当 app_title_before 与 app_title_after 不同时，认为该 step 是「新片段的起点」，
    在该步骤之前进行切割（即边界 step 作为新片段的第一个 step）。
    最终生成 steps_time1_window1, steps_time1_window2, steps_time2_window1 等。
    """
    path = Path(report_path)
    if not path.is_file():
        print(f"Report file not found: {path}")
        return
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 找到所有 steps_timeX 的键
    time_step_keys = [key for key in data.keys() if key.startswith("steps_time")]
    
    if not time_step_keys:
        print("No steps_timeX keys found in the input file")
        return
    
    # 构建新的数据结构
    result = {k: v for k, v in data.items() if not k.startswith("steps_time")}
    
    # 对每个 steps_timeX 进行窗口切割
    for time_key in sorted(time_step_keys):
        steps = data[time_key]
        
        # 找到所有边界步骤的索引
        boundary_indices = []
        
        for i, step in enumerate(steps):
            step_id = step.get("step_id", "")
            now_state = step.get("now_state", {}) or {}
            before_title = now_state.get("app_title_before")
            after_title = now_state.get("app_title_after")
            
            # 如果缺少任一标题字段，则跳过该 step
            if not before_title or not after_title:
                continue
            
            # APP+窗口（标题）在一个 step 内发生变化，视为边界步骤
            if before_title != after_title:
                boundary_indices.append(i)
                print(f"Found window boundary in {time_key} at step {step_id}: '{before_title}' -> '{after_title}'")
        
        # 根据边界索引切割步骤：
        # 边界索引本身作为新片段的开始
        cut_steps = []
        if boundary_indices:
            boundary_indices = sorted(set(boundary_indices))
            cut_points = [0] + boundary_indices + [len(steps)]

            for start_idx, end_idx in zip(cut_points[:-1], cut_points[1:]):
                if start_idx >= end_idx:
                    continue
                cut_steps.append(steps[start_idx:end_idx])
        else:
            if steps:
                cut_steps.append(steps)
        
        # 添加切割后的步骤数组
        for i, step_group in enumerate(cut_steps, 1):
            result[f"{time_key}_window{i}"] = step_group
        
        print(f"Cut {time_key} into {len(cut_steps)} window-based groups")
    
    # 保存结果
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully completed window-based cutting")
    print(f"Saved to: {output_path}")


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_cutting_time"):
        stem = f"{stem[:-len('_cutting_time')]}_cutting_window"
    else:
        stem = f"{stem}_cutting_window"
    return input_path.with_name(f"{stem}.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut time-based step groups into window-based groups."
    )
    parser.add_argument("input_path", help="Path to the report_cutting_time.json file.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Path to the output JSON file. Defaults to replacing _cutting_time with _cutting_window.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else _default_output_path(input_path)
    cut_steps_by_window(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
