import json
import os
import sys
from typing import Iterable, Optional


def _parse_step_number(step_value) -> Optional[int]:
    """Accept step values like 1, "1", or "s1" and return the numeric part."""
    if isinstance(step_value, int):
        return step_value if step_value > 0 else None

    text = str(step_value).strip()
    if not text:
        return None

    if text.startswith("s"):
        text = text[1:]

    try:
        number = int(text)
    except ValueError:
        return None

    return number if number > 0 else None


def _parse_step_id_list(raw_value: str) -> list[int]:
    """Parse user input like '[1, 8, 9]', '1,8,9', or 's1,s8,s9'."""
    text = raw_value.strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    numbers: list[int] = []
    for chunk in text.split(","):
        step_num = _parse_step_number(chunk)
        if step_num is None:
            continue
        if step_num not in numbers:
            numbers.append(step_num)
    return numbers


def _is_json_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".json")


def _find_report_file_in_folder(folder_path: str) -> Optional[str]:
    """
    Prefer the conventional report filenames when the user passes a folder.
    """
    candidate_names = (
        "report.json",
        "report_denoised.json",
    )

    for name in candidate_names:
        candidate_path = os.path.join(folder_path, name)
        if os.path.isfile(candidate_path):
            return candidate_path

    return None


def remove_steps_and_renumber(
    report_path: str,
    step_ids_to_remove: Iterable[int],
    output_path: Optional[str] = None,
) -> None:
    """
    Remove selected steps from report.json and renumber remaining step_id values.

    Screenshot files on disk are not touched.
    Screenshot paths inside each remaining step are preserved as-is so they still
    point to the original files.
    """
    if output_path is None:
        output_path = report_path

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    if not isinstance(steps, list) or not steps:
        print("No steps found in report.json")
        return

    remove_set = {step_num for step_num in step_ids_to_remove if step_num > 0}
    if not remove_set:
        print("No valid step ids to remove.")
        return

    kept_steps = []
    removed_step_ids = []
    seen_step_numbers = set()

    for step in steps:
        step_num = _parse_step_number(step.get("step_id"))
        if step_num is None:
            kept_steps.append(step)
            continue

        seen_step_numbers.add(step_num)
        if step_num in remove_set:
            removed_step_ids.append(f"s{step_num}")
            continue
        kept_steps.append(step)

    missing_step_ids = [f"s{step_num}" for step_num in sorted(remove_set - seen_step_numbers)]

    for new_index, step in enumerate(kept_steps, start=1):
        step["step_id"] = f"s{new_index}"

    data["steps"] = kept_steps
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Removed {len(removed_step_ids)} step(s): {', '.join(removed_step_ids) or 'none'}")
    if missing_step_ids:
        print(f"Step ids not found: {', '.join(missing_step_ids)}")
    print(f"Remaining steps: {len(kept_steps)}")
    print("Screenshot files were not renamed or deleted.")
    print("Screenshot paths in JSON were kept unchanged for remaining steps.")
    print(f"Output saved to: {output_path}")


def process_folder(folder_path: str, step_ids_to_remove: Iterable[int]) -> None:
    """Process a folder that contains a supported report JSON file."""
    report_path = _find_report_file_in_folder(folder_path)
    if report_path is None:
        print(f"No supported report JSON found in: {folder_path}")
        return

    remove_steps_and_renumber(report_path, step_ids_to_remove)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        target = sys.argv[1].strip()
        step_ids = _parse_step_id_list(sys.argv[2])

        if _is_json_file(target):
            remove_steps_and_renumber(target, step_ids)
        elif os.path.isdir(target):
            report_path = _find_report_file_in_folder(target)
            if report_path is not None:
                remove_steps_and_renumber(report_path, step_ids)
            else:
                for item in os.listdir(target):
                    item_path = os.path.join(target, item)
                    if os.path.isdir(item_path):
                        report_path = _find_report_file_in_folder(item_path)
                        if report_path is not None:
                            print(f"\n=== Processing: {item} ===")
                            remove_steps_and_renumber(report_path, step_ids)
        else:
            print(f"Invalid path: {target}")
    else:
        print("Usage: python remove_steps_by_ids.py <json path or folder> <step_id list>")
        target = input("Enter JSON file path or folder path: ").strip()
        raw_step_ids = input("Enter step ids to remove, e.g. [1, 8, 9]: ").strip()
        step_ids = _parse_step_id_list(raw_step_ids)
        if target and os.path.exists(target):
            if os.path.isfile(target):
                remove_steps_and_renumber(target, step_ids)
            else:
                process_folder(target, step_ids)
