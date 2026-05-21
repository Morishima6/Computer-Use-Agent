import argparse
import json
from copy import deepcopy
from pathlib import Path


BEFORE_KEYS = (
    "screenshot_path_before",
    "screenshot_path_before_part",
    "screenshot_time_before",
    "app_title_before",
)

AFTER_KEYS = (
    "screenshot_path_after",
    "screeenshot_time_after",
    "app_title_after",
)

def _normalize_position(step: dict):
    target = (step.get("action") or {}).get("target") or {}
    position = target.get("position")
    if isinstance(position, (list, tuple)) and len(position) == 2:
        return list(position)
    return None


def _get_button(step: dict):
    param = (step.get("action") or {}).get("param") or {}
    return param.get("button")


def _parse_step_num(step_id) -> int | None:
    if not isinstance(step_id, str) or not step_id.startswith("s"):
        return None
    try:
        return int(step_id[1:])
    except ValueError:
        return None


def _is_double_click_then_click(first_step: dict, second_step: dict) -> bool:
    first_action = first_step.get("action") or {}
    second_action = second_step.get("action") or {}

    if first_action.get("type") != "double_click":
        return False
    if second_action.get("type") != "click":
        return False

    first_position = _normalize_position(first_step)
    second_position = _normalize_position(second_step)
    if first_position is None or first_position != second_position:
        return False

    first_button = _get_button(first_step)
    second_button = _get_button(second_step)
    if first_button is None or first_button != second_button:
        return False

    return True


def _merge_text(first_text, second_text):
    first_text = (first_text or "").strip()
    second_text = (second_text or "").strip()

    if not first_text:
        return second_text
    if not second_text:
        return first_text
    if first_text == second_text:
        return first_text
    return f"{first_text} {second_text}"


def _merge_list(first_list, second_list):
    merged = []
    for item in (first_list or []):
        if item not in merged:
            merged.append(item)
    for item in (second_list or []):
        if item not in merged:
            merged.append(item)
    return merged


def _copy_now_state_fields(base_step: dict, source_step: dict, field_keys) -> dict:
    merged_now_state = deepcopy(base_step.get("now_state", {}))
    source_now_state = source_step.get("now_state", {}) or {}
    for key in field_keys:
        merged_now_state[key] = source_now_state.get(key)
    return merged_now_state


def _build_triple_click_step(double_step: dict, click_step: dict) -> dict:
    merged_step = deepcopy(double_step)

    merged_step["step_goal"] = _merge_text(
        double_step.get("step_goal"),
        click_step.get("step_goal"),
    )

    merged_step["action_preconditions"] = _merge_list(
        double_step.get("action_preconditions"),
        click_step.get("action_preconditions"),
    )

    merged_step["action_before_state"] = double_step.get("action_before_state", "")
    merged_step["action_after_effects"] = deepcopy(click_step.get("action_after_effects", []))
    merged_step["nl_explanation"] = _merge_text(
        double_step.get("nl_explanation"),
        click_step.get("nl_explanation"),
    )

    # before 相关字段沿用 double_click，after 相关字段沿用 click
    merged_now_state = _copy_now_state_fields(double_step, double_step, BEFORE_KEYS)
    merged_now_state = _copy_now_state_fields({"now_state": merged_now_state}, click_step, AFTER_KEYS)
    merged_step["now_state"] = merged_now_state

    merged_action = deepcopy(double_step.get("action", {}))
    merged_action["type"] = "triple_click"

    merged_param = deepcopy(merged_action.get("param", {}))
    merged_param["num_click"] = 3
    merged_action["param"] = merged_param

    merged_target = deepcopy(merged_action.get("target", {}))
    click_target = click_step.get("action", {}).get("target", {}) or {}
    if not merged_target.get("nl_position") and click_target.get("nl_position"):
        merged_target["nl_position"] = deepcopy(click_target.get("nl_position"))
    merged_action["target"] = merged_target

    merged_step["action"] = merged_action
    return merged_step


def _merge_steps_and_collect_removed(steps: list[dict]):
    merged_steps = []
    removed_step_nums = []
    merge_count = 0
    idx = 0

    while idx < len(steps):
        if idx + 1 < len(steps) and _is_double_click_then_click(steps[idx], steps[idx + 1]):
            merged_steps.append(_build_triple_click_step(steps[idx], steps[idx + 1]))
            second_step_num = _parse_step_num(steps[idx + 1].get("step_id"))
            if second_step_num is not None:
                removed_step_nums.append(second_step_num)
            merge_count += 1
            idx += 2
            continue

        merged_steps.append(deepcopy(steps[idx]))
        idx += 1

    return merged_steps, removed_step_nums, merge_count


def _build_removed_step_set(removed_step_nums) -> set[int]:
    return set(sorted(num for num in removed_step_nums if isinstance(num, int)))


def _remap_step_num(step_num: int, removed_step_nums: set[int]) -> int:
    shift = sum(1 for removed_num in removed_step_nums if removed_num <= step_num)
    return step_num - shift


def _remap_step_id(step_id, removed_step_nums: set[int]):
    step_num = _parse_step_num(step_id)
    if step_num is None:
        return step_id
    return f"s{_remap_step_num(step_num, removed_step_nums)}"


def _renumber_steps_in_report(report: dict, removed_step_nums: set[int]) -> None:
    steps = report.get("steps")
    if not isinstance(steps, list):
        return

    for step in steps:
        step["step_id"] = _remap_step_id(step.get("step_id"), removed_step_nums)


def _refresh_report_step_metadata(report: dict) -> None:
    steps = report.get("steps") or []
    step_ids = [step.get("step_id") for step in steps if step.get("step_id")]

    if "step_ids" in report:
        report["step_ids"] = step_ids
    if "start_step_id" in report:
        report["start_step_id"] = step_ids[0] if step_ids else None
    if "end_step_id" in report:
        report["end_step_id"] = step_ids[-1] if step_ids else None


def _apply_merges_and_renumber(report: dict, removed_step_nums: set[int] | None = None):
    steps = report.get("steps")
    if not isinstance(steps, list) or not steps:
        return 0, set(), False

    merged_steps, local_removed_step_nums, merge_count = _merge_steps_and_collect_removed(steps)
    local_removed_step_nums = _build_removed_step_set(local_removed_step_nums)
    effective_removed_step_nums = removed_step_nums if removed_step_nums is not None else local_removed_step_nums

    changed = merge_count > 0
    if merged_steps != steps:
        report["steps"] = merged_steps
        changed = True

    if effective_removed_step_nums:
        before = json.dumps(report, ensure_ascii=False, sort_keys=True)
        _renumber_steps_in_report(report, effective_removed_step_nums)
        _refresh_report_step_metadata(report)
        after = json.dumps(report, ensure_ascii=False, sort_keys=True)
        if before != after:
            changed = True
    else:
        _refresh_report_step_metadata(report)

    return merge_count, effective_removed_step_nums, changed


def _load_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(json_path: Path, data: dict) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _iter_trajectory_dirs(target: Path):
    visited = set()

    def add(path: Path):
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        yield resolved

    if target.is_file():
        if target.name in {"report_filled.json", "report.json"}:
            yield from add(target.parent)
            return
        if target.suffix.lower() == ".json" and target.parent.name == "tasks_abs":
            yield from add(target.parent.parent)
            return
        yield from add(target.parent)
        return

    if target.name == "tasks_abs" and target.is_dir():
        yield from add(target.parent)
        return

    if (
        (target / "report_filled.json").is_file()
        or (target / "report.json").is_file()
        or (target / "tasks_abs").is_dir()
    ):
        yield from add(target)
        return

    for report_filled_path in sorted(target.rglob("report_filled.json")):
        yield from add(report_filled_path.parent)

    for report_path in sorted(target.rglob("report.json")):
        yield from add(report_path.parent)

    for tasks_abs_dir in sorted(p for p in target.rglob("tasks_abs") if p.is_dir()):
        yield from add(tasks_abs_dir.parent)


def process_trajectory_dir(trajectory_dir: Path):
    report_filled_path = trajectory_dir / "report_filled.json"
    tasks_abs_dir = trajectory_dir / "tasks_abs"

    task_files = sorted(tasks_abs_dir.glob("*.json")) if tasks_abs_dir.is_dir() else []
    touched_files = 0
    total_merges = 0

    removed_step_nums: set[int] = set()

    if report_filled_path.is_file():
        report = _load_json(report_filled_path)
        merge_count, removed_step_nums, changed = _apply_merges_and_renumber(report)
        if changed:
            _save_json(report_filled_path, report)
            touched_files += 1
        total_merges = len(removed_step_nums)
    elif task_files:
        for task_file in task_files:
            report = _load_json(task_file)
            _, local_removed_step_nums, _ = _apply_merges_and_renumber(deepcopy(report))
            removed_step_nums.update(local_removed_step_nums)
        total_merges = len(removed_step_nums)

    for task_file in task_files:
        report = _load_json(task_file)
        _, _, changed = _apply_merges_and_renumber(report, removed_step_nums)
        if changed:
            _save_json(task_file, report)
            touched_files += 1

    return touched_files, total_merges, removed_step_nums


def main():
    parser = argparse.ArgumentParser(
        description="Merge adjacent double_click + click steps into triple_click and renumber later step ids/screenshots."
    )
    parser.add_argument(
        "target",
        help="A task JSON file, a tasks_abs directory, a trajectory directory, or a root directory.",
    )
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")

    trajectory_dirs = list(_iter_trajectory_dirs(target))
    if not trajectory_dirs:
        print(f"No trajectory directories found under: {target}")
        return

    processed_dirs = 0
    touched_files = 0
    total_merges = 0

    for trajectory_dir in trajectory_dirs:
        dir_touched_files, dir_merges, removed_step_nums = process_trajectory_dir(trajectory_dir)
        processed_dirs += 1
        touched_files += dir_touched_files
        total_merges += dir_merges

        if dir_merges > 0 or dir_touched_files > 0:
            removed_str = ", ".join(f"s{num}" for num in sorted(removed_step_nums)) if removed_step_nums else "none"
            print(
                f"Updated trajectory: {trajectory_dir} | merges={dir_merges}, "
                f"files={dir_touched_files}, removed={removed_str}"
            )

    print(
        f"Done. Processed {processed_dirs} trajectory dir(s), updated {touched_files} JSON file(s), "
        f"merged {total_merges} triple-click pair(s)."
    )


if __name__ == "__main__":
    main()
