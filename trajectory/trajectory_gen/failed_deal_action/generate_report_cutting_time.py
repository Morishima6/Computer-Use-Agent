import argparse
import json
from datetime import datetime
from pathlib import Path


def cut_steps_by_time(report_path: str, output_path: str, threshold_seconds: int = 30):
    """
    读取给定的 JSON 报告文件，根据时间差切割步骤。
    当 screenshot_time_before 和 screenshot_time_after 的时间差大于 threshold_seconds 时，
    认为该 step 是「新片段的起点」，在该步骤**之前**进行切割，
    将原始的 "steps" 数组切分成多个 steps_time1, steps_time2, ...
    """
    path = Path(report_path)
    if not path.is_file():
        print(f"Report file not found: {path}")
        return
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    steps = data.get("steps", [])
    time_format = "%Y-%m-%d %H:%M:%S"
    
    # 找到所有边界步骤的索引（这些 step 视为新片段的起点）
    boundary_indices = []
    
    for i, step in enumerate(steps):
        step_id = step.get("step_id", "")
        now_state = step.get("now_state", {})
        before = now_state.get("screenshot_time_before")
        after = now_state.get("screenshot_time_after")
        
        # 如果缺少任一时间字段，则跳过该 step
        if not before or not after:
            continue
        
        try:
            t_before = datetime.strptime(before, time_format)
            t_after = datetime.strptime(after, time_format)
        except ValueError:
            # 时间格式异常时，跳过该 step
            continue
        
        diff_seconds = (t_after - t_before).total_seconds()
        
        if diff_seconds > threshold_seconds:
            # 将该步骤视为新片段的起点
            boundary_indices.append(i)
            print(f"Found time boundary at step {step_id} with time difference: {diff_seconds}s")
    
    # 根据边界索引切割步骤：
    # 边界索引本身作为新片段的开始，例如：
    # steps: [0,1,2,3,4,5,6,7,8]
    # boundary_indices: [4,8]
    # -> [0:4] [4:8] [8:9]  即 3 个片段
    cut_steps = []
    
    if boundary_indices:
        # 确保有序且去重
        boundary_indices = sorted(set(boundary_indices))
        
        # 构造切割点：起点 0 + 所有边界索引 + 末尾 len(steps)
        cut_points = [0] + boundary_indices + [len(steps)]
        
        for start_idx, end_idx in zip(cut_points[:-1], cut_points[1:]):
            # 忽略空片段
            if start_idx >= end_idx:
                continue
            cut_steps.append(steps[start_idx:end_idx])
    else:
        # 没有边界就整个数组作为一个片段
        if steps:
            cut_steps.append(steps)
    
    # 理论上如果 steps 为空，cut_steps 也会为空，这里不再强行添加空数组
    
    # 构建新的数据结构
    result = {k: v for k, v in data.items() if k != "steps"}
    
    # 添加切割后的步骤数组
    for i, step_group in enumerate(cut_steps, 1):
        result[f"steps_time{i}"] = step_group
    
    # 保存结果
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully cut steps into {len(cut_steps)} time-based groups")
    print(f"Saved to: {output_path}")


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_cutting_time.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut report steps into time-based groups."
    )
    parser.add_argument("input_path", help="Path to the source report.json file.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Path to the output JSON file. Defaults to <input_stem>_cutting_time.json.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=30,
        help="Time boundary threshold in seconds. Default: 30.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else _default_output_path(input_path)
    cut_steps_by_time(str(input_path), str(output_path), args.threshold)


if __name__ == "__main__":
    main()
