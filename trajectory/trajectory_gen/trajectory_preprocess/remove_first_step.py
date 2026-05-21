import json
import os
import sys


SCREENSHOT_KEYS = (
    "screenshot_path_before",
    "screenshot_path_before_part",
    "screenshot_path_after",
)


def _decrement_step_path(old_path: str) -> str:
    """Rename `sN_*` to `s(N-1)_*` using the same prefix replacement logic."""
    normalized = old_path.replace("\\", "/")
    old_step = normalized.split("_")[0].split("/")[-1]

    if not old_step.startswith("s"):
        return old_path

    try:
        file_old_step_num = int(old_step[1:])
    except ValueError:
        return old_path

    file_new_step_num = file_old_step_num - 1
    new_normalized = normalized.replace(old_step, f"s{file_new_step_num}", 1)

    if "\\" in old_path and "/" not in old_path:
        return new_normalized.replace("/", "\\")
    return new_normalized


def _rewrite_step_paths(step: dict) -> None:
    now_state = step.get("now_state", {})
    for key in SCREENSHOT_KEYS:
        value = now_state.get(key)
        if value:
            now_state[key] = _decrement_step_path(value)


def _rename_screenshot_files(screenshots_dir: str) -> None:
    """Rename screenshot files on disk with the same decrement logic."""
    if not os.path.isdir(screenshots_dir):
        return

    files_with_num: list[tuple[int, str]] = []
    for fname in os.listdir(screenshots_dir):
        if not fname.startswith("s") or "_" not in fname:
            continue

        old_step = fname.split("_")[0].split("/")[-1]
        try:
            file_old_step_num = int(old_step[1:])
        except ValueError:
            continue
        files_with_num.append((file_old_step_num, fname))

    files_with_num.sort(key=lambda x: x[0])

    for _, fname in files_with_num:
        old_step = fname.split("_")[0].split("/")[-1]
        file_old_step_num = int(old_step[1:])
        file_new_step_num = file_old_step_num - 1

        if file_new_step_num < 1:
            continue

        new_fname = fname.replace(old_step, f"s{file_new_step_num}", 1)
        old_path = os.path.join(screenshots_dir, fname)
        new_path = os.path.join(screenshots_dir, new_fname)

        if os.path.exists(old_path):
            if os.path.exists(new_path):
                os.remove(new_path)
                print(f"Deleted existing: {new_path}")
            os.rename(old_path, new_path)
            print(f"Renamed screenshot: {fname} -> {new_fname}")


def remove_first_step_and_renumber(report_path: str, output_path: str = None) -> None:
    """Remove the first step from report.json and renumber later steps/files."""
    if output_path is None:
        output_path = report_path

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not steps:
        print("No steps found in report.json")
        return

    report_dir = os.path.dirname(report_path)
    screenshots_dir = os.path.join(report_dir, "screenshots")

    first_step = steps[0]
    first_step_id = first_step.get("step_id", "s1")
    print(f"Removing step: {first_step_id}")

    remaining_steps = steps[1:]
    new_steps = []

    for i, step in enumerate(remaining_steps, start=1):
        step["step_id"] = f"s{i}"
        _rewrite_step_paths(step)
        new_steps.append(step)

    data["steps"] = new_steps
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    _rename_screenshot_files(screenshots_dir)

    print(f"\nDone! Removed {first_step_id}, renumbered {len(new_steps)} steps.")
    print(f"Output saved to: {output_path}")


def process_folder(folder_path: str) -> None:
    """Process a folder that contains report.json."""
    report_path = os.path.join(folder_path, "report.json")
    if not os.path.exists(report_path):
        print(f"report.json not found in: {folder_path}")
        return

    remove_first_step_and_renumber(report_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1].strip()

        if os.path.isfile(target) and target.endswith("report.json"):
            remove_first_step_and_renumber(target)
        elif os.path.isdir(target):
            report_path = os.path.join(target, "report.json")
            if os.path.exists(report_path):
                remove_first_step_and_renumber(report_path)
            else:
                for item in os.listdir(target):
                    item_path = os.path.join(target, item)
                    if os.path.isdir(item_path):
                        report_path = os.path.join(item_path, "report.json")
                        if os.path.exists(report_path):
                            print(f"\n=== Processing: {item} ===")
                            remove_first_step_and_renumber(report_path)
        else:
            print(f"Invalid path: {target}")
    else:
        print("Usage: python remove_first_step.py <report.json path or folder>")
        target = input("Enter report.json path or folder path: ").strip()
        if target and os.path.exists(target):
            if os.path.isfile(target):
                remove_first_step_and_renumber(target)
            else:
                process_folder(target)
