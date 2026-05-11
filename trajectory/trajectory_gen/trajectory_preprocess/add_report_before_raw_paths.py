from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


RAW_FIELD = "screenshot_path_before_raw"
PROGRESS_FILE_SUFFIX = ".add_before_raw_progress.json"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add screenshot_path_before_raw to every step.now_state in a trajectory JSON, "
            "and verify that the raw screenshot exists before writing the path."
        )
    )
    parser.add_argument(
        "trace_path",
        type=Path,
        help=(
            "Trajectory path. It can be a JSON file, a result directory containing the input JSON, "
            "a trajectory directory containing result/<input-json-name>, "
            "or a batch root when --batch is enabled."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Process every result/<input-json-name> found under trace_path. "
            "This intentionally ignores report.json directly under each session directory."
        ),
    )
    parser.add_argument(
        "--input-json-name",
        default="report.json",
        help='Input JSON filename when trace_path is a directory. Default: "report.json".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to overwriting the input JSON in place.",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=None,
        help="Screenshots directory. Defaults to <input_json_dir>/screenshots.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save after every N processed steps for resume support. Default: 20.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore the progress file and recheck all steps.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Default: 2.",
    )
    return parser.parse_args()


def resolve_input_json(trace_path: Path, input_json_name: str) -> Path:
    path = trace_path.resolve()
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Input file must be a JSON file: {path}")
        return path

    result_json = path / "result" / input_json_name
    if result_json.is_file():
        return result_json

    direct_json = path / input_json_name
    if direct_json.is_file():
        return direct_json

    raise FileNotFoundError(f"Input JSON not found: {path} / {input_json_name}")


def find_batch_input_jsons(trace_path: Path, input_json_name: str) -> list[Path]:
    path = trace_path.resolve()
    if path.is_file():
        return [resolve_input_json(path, input_json_name)]
    if not path.is_dir():
        raise FileNotFoundError(f"Trace path not found: {path}")

    input_jsons = [
        candidate.resolve()
        for candidate in path.rglob(input_json_name)
        if candidate.parent.name == "result" and candidate.is_file()
    ]
    return sorted(set(input_jsons), key=lambda item: str(item).lower())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def build_progress_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}{PROGRESS_FILE_SUFFIX}")


def load_progress(progress_path: Path) -> set[str]:
    if not progress_path.is_file():
        return set()

    data = load_json(progress_path)
    return set(data.get("completed_step_ids", []))


def save_progress(progress_path: Path, completed_step_ids: set[str]) -> None:
    data = {
        "updated_at": now_text(),
        "completed_step_ids": sorted(
            completed_step_ids,
            key=lambda step_id: (
                0,
                int(step_id[1:]),
            )
            if step_id.startswith("s") and step_id[1:].isdigit()
            else (1, step_id),
        ),
    }
    atomic_write_json(progress_path, data, indent=2)


def build_raw_relative_path(before_path: str | None) -> str:
    if not before_path:
        return ""

    before = Path(before_path)
    raw_name = f"{before.stem}(raw){before.suffix}"
    raw_path = before.with_name(raw_name)
    return raw_path.as_posix()


def set_raw_field(now_state: dict[str, Any], raw_value: str) -> None:
    rebuilt: dict[str, Any] = {}
    inserted = False

    for key, value in now_state.items():
        if key == RAW_FIELD:
            continue

        rebuilt[key] = value
        if key == "screenshot_path_before_part":
            rebuilt[RAW_FIELD] = raw_value
            inserted = True

    if not inserted:
        insert_after_before = RAW_FIELD not in rebuilt
        rebuilt = {}
        for key, value in now_state.items():
            if key == RAW_FIELD:
                continue
            rebuilt[key] = value
            if insert_after_before and key == "screenshot_path_before":
                rebuilt[RAW_FIELD] = raw_value
                insert_after_before = False
                inserted = True

    if not inserted:
        rebuilt[RAW_FIELD] = raw_value

    now_state.clear()
    now_state.update(rebuilt)


def process_report(
    report_data: dict[str, Any],
    screenshots_dir: Path,
    output_path: Path,
    progress_path: Path,
    completed_step_ids: set[str],
    save_every: int,
    indent: int,
) -> dict[str, int]:
    steps = report_data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Input JSON does not contain a list field: steps")

    stats = {
        "total": len(steps),
        "processed": 0,
        "resumed_skipped": 0,
        "raw_exists": 0,
        "raw_missing": 0,
        "invalid_step": 0,
    }

    for index, step in enumerate(steps, start=1):
        step_id = str(step.get("step_id") or f"index_{index}")
        now_state = step.get("now_state")

        if (
            step_id in completed_step_ids
            and isinstance(now_state, dict)
            and RAW_FIELD in now_state
        ):
            stats["resumed_skipped"] += 1
            log(f"Skip completed step {index}/{stats['total']}: {step_id}")
            continue

        if not isinstance(now_state, dict):
            stats["invalid_step"] += 1
            step["now_state"] = {RAW_FIELD: ""}
            completed_step_ids.add(step_id)
            log(f"Process step {index}/{stats['total']}: {step_id} now_state is not dict, set empty")
        else:
            before_path = now_state.get("screenshot_path_before")
            raw_relative_path = build_raw_relative_path(before_path)
            raw_abs_path = screenshots_dir / Path(raw_relative_path).name if raw_relative_path else None
            raw_value = raw_relative_path if raw_abs_path and raw_abs_path.is_file() else ""

            set_raw_field(now_state, raw_value)
            completed_step_ids.add(step_id)

            if raw_value:
                stats["raw_exists"] += 1
                log(f"Process step {index}/{stats['total']}: {step_id} found {raw_value}")
            else:
                stats["raw_missing"] += 1
                log(f"Process step {index}/{stats['total']}: {step_id} raw screenshot missing, set empty")

        stats["processed"] += 1

        if save_every > 0 and stats["processed"] % save_every == 0:
            atomic_write_json(output_path, report_data, indent=indent)
            save_progress(progress_path, completed_step_ids)
            log(f"Saved checkpoint: processed={stats['processed']}")

    atomic_write_json(output_path, report_data, indent=indent)
    save_progress(progress_path, completed_step_ids)
    return stats


def process_input_json(
    input_json: Path,
    output: Path | None,
    screenshots_dir: Path | None,
    no_resume: bool,
    save_every: int,
    indent: int,
) -> dict[str, int]:
    input_json = input_json.resolve()
    input_dir = input_json.parent
    resolved_screenshots_dir = (screenshots_dir or input_dir / "screenshots").resolve()
    output_path = (output or input_json).resolve()
    progress_path = build_progress_path(output_path)

    if not resolved_screenshots_dir.is_dir():
        raise FileNotFoundError(f"Screenshots directory not found: {resolved_screenshots_dir}")

    log(f"Read input JSON: {input_json}")
    log(f"Screenshots dir: {resolved_screenshots_dir}")
    log(f"Output path: {output_path}")
    log(f"Progress file: {progress_path}")

    completed_step_ids = set() if no_resume else load_progress(progress_path)
    load_path = output_path if completed_step_ids and output_path.is_file() else input_json
    report_data = load_json(load_path)
    if load_path != input_json:
        log(f"Resume from existing output: {load_path}")
    if completed_step_ids:
        log(f"Resume enabled, completed step count: {len(completed_step_ids)}")

    stats = process_report(
        report_data=report_data,
        screenshots_dir=resolved_screenshots_dir,
        output_path=output_path,
        progress_path=progress_path,
        completed_step_ids=completed_step_ids,
        save_every=save_every,
        indent=indent,
    )

    log("Done")
    log(
        "Stats: "
        f"total={stats['total']}, "
        f"processed={stats['processed']}, "
        f"resumed_skipped={stats['resumed_skipped']}, "
        f"raw_exists={stats['raw_exists']}, "
        f"raw_missing={stats['raw_missing']}, "
        f"invalid_step={stats['invalid_step']}"
    )
    return stats


def main() -> int:
    args = parse_args()
    if args.save_every < 0:
        raise ValueError("--save-every cannot be less than 0")
    if args.batch and args.output:
        raise ValueError("--output cannot be used with --batch because each report is overwritten in place")

    if not args.batch:
        input_json = resolve_input_json(args.trace_path, args.input_json_name)
        process_input_json(
            input_json=input_json,
            output=args.output,
            screenshots_dir=args.screenshots_dir,
            no_resume=args.no_resume,
            save_every=args.save_every,
            indent=args.indent,
        )
        return 0

    input_jsons = find_batch_input_jsons(args.trace_path, args.input_json_name)
    if not input_jsons:
        raise FileNotFoundError(
            f"No result/{args.input_json_name} found under: {args.trace_path.resolve()}"
        )

    log(f"Batch mode: found {len(input_jsons)} report(s)")
    total_stats = {
        "total": 0,
        "processed": 0,
        "resumed_skipped": 0,
        "raw_exists": 0,
        "raw_missing": 0,
        "invalid_step": 0,
    }
    failed: list[tuple[Path, str]] = []

    for index, input_json in enumerate(input_jsons, start=1):
        log(f"Batch item {index}/{len(input_jsons)}: {input_json}")
        try:
            stats = process_input_json(
                input_json=input_json,
                output=None,
                screenshots_dir=args.screenshots_dir,
                no_resume=args.no_resume,
                save_every=args.save_every,
                indent=args.indent,
            )
        except Exception as exc:
            failed.append((input_json, str(exc)))
            log(f"Failed: {input_json} | {exc}")
            continue

        for key in total_stats:
            total_stats[key] += stats[key]

    log("Batch done")
    log(
        "Batch stats: "
        f"reports={len(input_jsons)}, "
        f"failed={len(failed)}, "
        f"total={total_stats['total']}, "
        f"processed={total_stats['processed']}, "
        f"resumed_skipped={total_stats['resumed_skipped']}, "
        f"raw_exists={total_stats['raw_exists']}, "
        f"raw_missing={total_stats['raw_missing']}, "
        f"invalid_step={total_stats['invalid_step']}"
    )
    if failed:
        log("Failed reports:")
        for input_json, error in failed:
            log(f"- {input_json}: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
