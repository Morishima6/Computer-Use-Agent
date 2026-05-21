#!/usr/bin/env python3
"""
Trajectory processing pipeline.

Default flow for each session folder:
1. Optionally remove the first step from the selected input JSON.
2. Optionally call Codex to fill missing fields in the selected input JSON.
3. Run time-based cutting.
4. Run window-based cutting.
5. Optionally run LLM window merge.
6. Optionally run LLM task cutting.

Key extension in this version:
- `--json-file` lets the pipeline work on files such as `report_denoised.json`
  instead of always assuming `report.json`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


SCRIPT_DIR = Path(__file__).parent.resolve()

CALL_CODEX_DIR = SCRIPT_DIR / "call_codex"
AMBLER_CUTTING_DIR = SCRIPT_DIR / "Ambler_cutting"
TRAJECTORY_PREPROCESS_DIR = SCRIPT_DIR / "trajectory_preprocess"

CALL_CODEX_SCRIPT = CALL_CODEX_DIR / "call_codex_0413.py"
CUTTING_TIME_SCRIPT = AMBLER_CUTTING_DIR / "generate_report_cutting_time.py"
CUTTING_WINDOW_SCRIPT = AMBLER_CUTTING_DIR / "generate_report_cutting_window.py"
WINDOW_MERGE_SCRIPT = AMBLER_CUTTING_DIR / "llm_window_merge.py"
TASK_CUT_SCRIPT = AMBLER_CUTTING_DIR / "llm_task_cut_mix_to_files.py"

DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_LLM_MODEL = "gpt-5.5"
DEFAULT_TIME_THRESHOLD = 30
DEFAULT_JSON_FILE = "report.json"
SIGN_FILE = "sign.txt"


def get_current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def write_sign_file(session_folder: Path, json_file: str) -> None:
    sign_path = session_folder / SIGN_FILE
    try:
        sign_path.write_text(
            f"Pipeline processed at: {get_current_time_str()}\n"
            f"Input JSON: {json_file}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[warn] Failed to write sign.txt: {exc}")


def run_python_script(script_path: Path, *args: str) -> int:
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n{'=' * 72}")
    print("Running:", " ".join(cmd))
    print(f"{'=' * 72}\n")
    result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace")
    return result.returncode


def find_session_folders(data_dir: Path, json_file: str) -> List[Path]:
    session_folders: List[Path] = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and (item / json_file).is_file() and not (item / SIGN_FILE).exists():
            session_folders.append(item)
    return session_folders


def count_skipped_sessions(data_dir: Path, json_file: str) -> int:
    skipped = 0
    for item in data_dir.iterdir():
        if item.is_dir() and (item / json_file).is_file() and (item / SIGN_FILE).exists():
            skipped += 1
    return skipped


def remove_first_step_inplace(report_path: Path) -> None:
    sys.path.insert(0, str(TRAJECTORY_PREPROCESS_DIR))
    try:
        from remove_first_step import remove_first_step_and_renumber

        remove_first_step_and_renumber(str(report_path))
    finally:
        sys.path.pop(0)


def run_call_codex_fill(
    *,
    session_folder: Path,
    json_file: str,
    codex_model: str,
    filled_path: Path,
) -> int:
    return run_python_script(
        CALL_CODEX_SCRIPT,
        str(session_folder),
        "--json-file",
        json_file,
        "--model",
        codex_model,
        "--output-json",
        filled_path.name,
    )


def process_single_session(
    session_folder: Path,
    *,
    json_file: str,
    codex_model: str,
    llm_model: str,
    time_threshold: int,
    skip_codex: bool,
    skip_llm: bool,
    skip_remove_first_step: bool = False,
) -> bool:
    session_folder = session_folder.resolve()
    report_path = session_folder / json_file
    if not report_path.is_file():
        print(f"[error] Input JSON not found: {report_path}")
        return False

    start_time = datetime.now()
    print(f"\n{'#' * 72}")
    print(f"# Session: {session_folder.name}")
    print(f"# Start:   {get_current_time_str()}")
    print(f"# Input:   {report_path}")
    print(f"{'#' * 72}")

    if not skip_remove_first_step:
        print("\n>>> Step 0: remove first step")
        try:
            remove_first_step_inplace(report_path)
        except Exception as exc:
            print(f"[error] remove_first_step failed: {exc}")
            return False
        print(f"[ok] remove_first_step finished at {get_current_time_str()}")
    else:
        print("\n>>> Step 0: skip remove first step")

    if not skip_codex:
        print("\n>>> Step 1: fill missing fields with Codex")
        filled_path = report_path.parent / f"{report_path.stem}_filled.json"
        ret = run_call_codex_fill(
            session_folder=session_folder,
            json_file=json_file,
            codex_model=codex_model,
            filled_path=filled_path,
        )
        if ret != 0:
            print(f"[error] call_codex_0413.py failed with exit code {ret}")
            return False
        print(f"[ok] Codex fill output: {filled_path}")
    else:
        print("\n>>> Step 1: skip Codex fill")
        filled_path = report_path

    splits_dir = session_folder / "splits"
    splits_dir.mkdir(exist_ok=True)

    print("\n>>> Step 2: time-based cutting")
    time_output = splits_dir / f"{filled_path.stem}_cutting_time.json"
    ret = run_python_script(
        CUTTING_TIME_SCRIPT,
        str(filled_path),
        "-o",
        str(time_output),
        "-t",
        str(time_threshold),
    )
    if ret != 0:
        print(f"[error] generate_report_cutting_time.py failed with exit code {ret}")
        return False
    print(f"[ok] Time cutting output: {time_output}")

    print("\n>>> Step 3: window-based cutting")
    window_output = splits_dir / f"{filled_path.stem}_cutting_window.json"
    ret = run_python_script(
        CUTTING_WINDOW_SCRIPT,
        str(time_output),
        "-o",
        str(window_output),
    )
    if ret != 0:
        print(f"[error] generate_report_cutting_window.py failed with exit code {ret}")
        return False
    print(f"[ok] Window cutting output: {window_output}")

    if not skip_llm:
        print("\n>>> Step 4: LLM window merge")
        merge_output = splits_dir / f"{window_output.stem}_mix.json"
        ret = run_python_script(
            WINDOW_MERGE_SCRIPT,
            str(window_output),
            "-o",
            str(merge_output),
            "-m",
            llm_model,
        )
        if ret != 0:
            print(f"[error] llm_window_merge.py failed with exit code {ret}")
            return False
        print(f"[ok] Window merge output: {merge_output}")

        print("\n>>> Step 5: LLM task cutting")
        tasks_dir = splits_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        ret = run_python_script(
            TASK_CUT_SCRIPT,
            str(merge_output),
            "-o",
            str(tasks_dir),
            "-m",
            llm_model,
        )
        if ret != 0:
            print(f"[error] llm_task_cut_mix_to_files.py failed with exit code {ret}")
            return False
        print(f"[ok] Task cutting output dir: {tasks_dir}")
    else:
        print("\n>>> Step 4-5: skip LLM merge/task cutting")

    total_duration = (datetime.now() - start_time).total_seconds()
    print(f"\n{'#' * 72}")
    print(f"# Session finished: {session_folder.name}")
    print(f"# End:      {get_current_time_str()}")
    print(f"# Duration: {format_duration(total_duration)}")
    print(f"{'#' * 72}\n")

    write_sign_file(session_folder, json_file)
    return True


def process_batch(
    data_dir: Path,
    *,
    json_file: str,
    codex_model: str,
    llm_model: str,
    time_threshold: int,
    skip_codex: bool,
    skip_llm: bool,
    skip_remove_first_step: bool = False,
) -> None:
    data_dir = data_dir.resolve()
    session_folders = find_session_folders(data_dir, json_file)
    skipped_count = count_skipped_sessions(data_dir, json_file)

    if not session_folders:
        print(f"[info] No session folders containing {json_file} need processing under: {data_dir}")
        if skipped_count:
            print(f"[info] {skipped_count} folders already have sign.txt and were skipped.")
        return

    print(f"\n{'=' * 72}")
    print("Batch mode")
    print(f"Root dir:        {data_dir}")
    print(f"Target JSON:     {json_file}")
    print(f"To process:      {len(session_folders)}")
    print(f"Already skipped: {skipped_count}")
    print(f"Start:           {get_current_time_str()}")
    print(f"{'=' * 72}\n")

    batch_start = datetime.now()
    success_count = 0
    fail_count = 0

    for idx, session_folder in enumerate(session_folders, start=1):
        print(f"\n{'=' * 72}")
        print(f"[{idx}/{len(session_folders)}] {session_folder.name}")
        print(f"{'=' * 72}")
        try:
            ok = process_single_session(
                session_folder,
                json_file=json_file,
                codex_model=codex_model,
                llm_model=llm_model,
                time_threshold=time_threshold,
                skip_codex=skip_codex,
                skip_llm=skip_llm,
                skip_remove_first_step=skip_remove_first_step,
            )
        except Exception as exc:
            print(f"[error] Session crashed: {exc}")
            ok = False
        if ok:
            success_count += 1
        else:
            fail_count += 1

    total_duration = (datetime.now() - batch_start).total_seconds()
    print(f"\n{'=' * 72}")
    print("Batch finished")
    print(f"End:      {get_current_time_str()}")
    print(f"Duration: {format_duration(total_duration)}")
    print(f"Success:  {success_count}")
    print(f"Failed:   {fail_count}")
    print(f"Total:    {len(session_folders)}")
    print(f"{'=' * 72}\n")


def validate_scripts() -> None:
    required_scripts = [
        CALL_CODEX_SCRIPT,
        CUTTING_TIME_SCRIPT,
        CUTTING_WINDOW_SCRIPT,
        WINDOW_MERGE_SCRIPT,
        TASK_CUT_SCRIPT,
    ]
    for script in required_scripts:
        if not script.exists():
            print(f"[error] Missing script: {script}")
            sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trajectory pipeline: remove-first-step -> fill fields -> split -> merge -> task cut",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py --single E:/data/session_001\n"
            "  python pipeline.py --single E:/data/session_001 --json-file report_denoised.json\n"
            "  python pipeline.py --batch E:/data/root --json-file report_denoised.json\n"
            "  python pipeline.py --single E:/data/session_001 --skip-codex\n"
            "  python pipeline.py --single E:/data/session_001 --skip-remove-first-step\n"
        ),
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single", metavar="FOLDER", help="Process one session folder.")
    mode_group.add_argument("--batch", metavar="DIRECTORY", help="Process all matching session folders under a directory.")

    parser.add_argument(
        "--json-file",
        default=DEFAULT_JSON_FILE,
        help=f"Input JSON filename inside each session folder (default: {DEFAULT_JSON_FILE}).",
    )
    parser.add_argument(
        "--codex-model",
        default=DEFAULT_CODEX_MODEL,
        help=f"Codex model name (default: {DEFAULT_CODEX_MODEL}).",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model used by later merge/cut scripts (default: {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument(
        "-t",
        "--time-threshold",
        type=int,
        default=DEFAULT_TIME_THRESHOLD,
        help=f"Time split threshold in seconds (default: {DEFAULT_TIME_THRESHOLD}).",
    )
    parser.add_argument("--skip-codex", action="store_true", help="Skip the Codex fill step.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM window merge and task cutting.")
    parser.add_argument(
        "--skip-remove-first-step",
        action="store_true",
        help="Skip the preprocessing step that removes the first step.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print matching session folders.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_scripts()

    if args.single:
        folder = Path(args.single).resolve()
        if not folder.exists():
            print(f"[error] Folder does not exist: {folder}")
            sys.exit(1)
        if args.dry_run:
            print(f"[dry-run] Would process: {folder}")
            print(f"[dry-run] Target JSON: {args.json_file}")
            return
        ok = process_single_session(
            folder,
            json_file=args.json_file,
            codex_model=args.codex_model,
            llm_model=args.llm_model,
            time_threshold=args.time_threshold,
            skip_codex=args.skip_codex,
            skip_llm=args.skip_llm,
            skip_remove_first_step=args.skip_remove_first_step,
        )
        if not ok:
            sys.exit(1)
        return

    directory = Path(args.batch).resolve()
    if not directory.exists():
        print(f"[error] Directory does not exist: {directory}")
        sys.exit(1)

    if args.dry_run:
        folders = find_session_folders(directory, args.json_file)
        print(f"[dry-run] Found {len(folders)} folders containing {args.json_file}:")
        for folder in folders:
            print(f"  - {folder}")
        return

    process_batch(
        directory,
        json_file=args.json_file,
        codex_model=args.codex_model,
        llm_model=args.llm_model,
        time_threshold=args.time_threshold,
        skip_codex=args.skip_codex,
        skip_llm=args.skip_llm,
        skip_remove_first_step=args.skip_remove_first_step,
    )


if __name__ == "__main__":
    main()
