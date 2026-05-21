import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generate_report_cutting_time import cut_steps_by_time
from generate_report_cutting_window import cut_steps_by_window
from llm_task_cut_mix_to_files import run_codex_cut_and_split_for_mix
from llm_window_merge import check_window_merges_with_codex


DEFAULT_REPORT_CANDIDATES = (
    "report_denoised_filled_step.json",
    "report_denoised.json",
    "report.json",
)

STATUS_FILE_NAME = "pipeline_status.json"
STAGE_CUTTING_TIME = "cutting_time"
STAGE_CUTTING_WINDOW = "cutting_window"
STAGE_WINDOW_MERGE = "window_merge"
STAGE_TASK_FILES = "task_files"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now_text()}] {message}", flush=True)


def _resolve_relative_to_trace(path_arg: Optional[str], trace_dir: Path, default: Path) -> Path:
    if not path_arg:
        return default

    path = Path(path_arg)
    if path.is_absolute():
        return path
    return trace_dir / path


def _resolve_source_report(trajectory_path: str, json_file: Optional[str]) -> Tuple[Path, Path]:
    path = Path(trajectory_path)

    if path.is_file():
        trace_dir = path.parent
        source_report = path
    else:
        trace_dir = path
        if json_file:
            source_report = Path(json_file)
            if not source_report.is_absolute():
                source_report = trace_dir / source_report
        else:
            source_report = _find_default_report(trace_dir)

    if not source_report.is_file():
        raise FileNotFoundError(f"Input report not found: {source_report}")

    return trace_dir, source_report


def _find_default_report(trace_dir: Path) -> Path:
    for filename in DEFAULT_REPORT_CANDIDATES:
        candidate = trace_dir / filename
        if candidate.is_file():
            return candidate

    names = ", ".join(DEFAULT_REPORT_CANDIDATES)
    raise FileNotFoundError(
        f"No default report found in {trace_dir}. Tried: {names}. "
        "Use --json-file to choose an input JSON explicitly."
    )


def _output_base_stem(source_report: Path) -> str:
    stem = source_report.stem
    if stem.endswith("_step"):
        return stem[: -len("_step")]
    return stem


def _load_status(status_path: Path) -> Dict[str, Any]:
    if not status_path.is_file():
        return {"stages": {}}

    with status_path.open("r", encoding="utf-8") as f:
        status = json.load(f)

    if not isinstance(status, dict):
        return {"stages": {}}
    if not isinstance(status.get("stages"), dict):
        status["stages"] = {}
    return status


def _save_status(status_path: Path, status: Dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = _now_text()
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def _set_stage_status(
    status_path: Path,
    status: Dict[str, Any],
    stage: str,
    state: str,
    **extra: Any,
) -> None:
    stages = status.setdefault("stages", {})
    stage_info = dict(stages.get(stage, {}))
    stage_info.update(extra)
    stage_info["status"] = state
    stage_info[f"{state}_at"] = _now_text()
    stages[stage] = stage_info
    _save_status(status_path, status)


def _stage_completed(
    status: Dict[str, Any],
    stage: str,
    output_path: Optional[Path] = None,
) -> bool:
    stage_info = status.get("stages", {}).get(stage, {})
    if stage_info.get("status") != "completed":
        return False
    if output_path is not None and not output_path.is_file():
        return False
    return True


def _task_stage_completed(status: Dict[str, Any], tasks_dir: Path, task_prefix: str) -> bool:
    stage_info = status.get("stages", {}).get(STAGE_TASK_FILES, {})
    if stage_info.get("status") != "completed":
        return False

    recorded_count = int(stage_info.get("task_count", 0) or 0)
    if recorded_count <= 0:
        return False
    return _count_task_files(tasks_dir, task_prefix) >= recorded_count


def _verify_output(output_path: Path, stage: str) -> None:
    if not output_path.is_file():
        raise RuntimeError(f"Stage {stage} finished but output file was not created: {output_path}")


def _count_task_files(tasks_dir: Path, task_prefix: str) -> int:
    if not tasks_dir.is_dir():
        return 0
    return len(list(tasks_dir.glob(f"{task_prefix}*.json")))


def _has_source_report(trace_dir: Path, json_file: Optional[str]) -> bool:
    if json_file:
        return (trace_dir / json_file).is_file()
    return any((trace_dir / filename).is_file() for filename in DEFAULT_REPORT_CANDIDATES)


def discover_trajectory_dirs(
    batch_dir: str,
    json_file: Optional[str] = None,
    recursive: bool = False,
) -> List[Path]:
    root = Path(batch_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Batch directory not found: {root}")
    if json_file and Path(json_file).is_absolute():
        raise ValueError("In batch mode, --json-file should be a filename or relative path under each trajectory directory.")

    candidates = root.rglob("*") if recursive else root.iterdir()
    trace_dirs: List[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if candidate.name in {"screenshots", "splits", "__pycache__"}:
            continue
        if not (candidate / "screenshots").is_dir():
            continue
        if not _has_source_report(candidate, json_file):
            continue
        trace_dirs.append(candidate)

    return sorted(trace_dirs, key=lambda p: str(p).lower())


def run_pipeline(
    trajectory_path: str,
    json_file: Optional[str] = None,
    reports_dir_arg: Optional[str] = None,
    tasks_dir_arg: Optional[str] = None,
    threshold_seconds: int = 30,
    model: str = "gpt-5.5",
    task_prefix: str = "report_cutting_llm_task_",
    resume: bool = True,
    progress_label: Optional[str] = None,
) -> Dict[str, Path]:
    trace_dir, source_report = _resolve_source_report(trajectory_path, json_file)

    default_reports_dir = trace_dir / "splits"
    reports_dir = _resolve_relative_to_trace(reports_dir_arg, trace_dir, default_reports_dir)
    tasks_dir = _resolve_relative_to_trace(tasks_dir_arg, trace_dir, reports_dir / "tasks")

    base_stem = _output_base_stem(source_report)
    cutting_time_path = reports_dir / f"{base_stem}_cutting_time.json"
    cutting_window_path = reports_dir / f"{base_stem}_cutting_window.json"
    cutting_window_mix_path = reports_dir / f"{base_stem}_cutting_window_mix.json"

    reports_dir.mkdir(parents=True, exist_ok=True)
    status_path = reports_dir / STATUS_FILE_NAME
    status = _load_status(status_path)
    status.update(
        {
            "trace_dir": str(trace_dir),
            "source_report": str(source_report),
            "reports_dir": str(reports_dir),
            "tasks_dir": str(tasks_dir),
            "model": model,
            "threshold_seconds": threshold_seconds,
            "task_prefix": task_prefix,
        }
    )
    _save_status(status_path, status)

    prefix = f"{progress_label} " if progress_label else ""

    _log(f"{prefix}Trajectory cutting pipeline started")
    _log(f"{prefix}Trace directory: {trace_dir}")
    _log(f"{prefix}Input report: {source_report}")
    _log(f"{prefix}Intermediate reports directory: {reports_dir}")
    _log(f"{prefix}Task reports directory: {tasks_dir}")
    _log(f"{prefix}Resume: {'enabled' if resume else 'disabled'}")

    if resume and _stage_completed(status, STAGE_CUTTING_TIME, cutting_time_path):
        _log(f"{prefix}[1/4] Skip time cutting, existing output: {cutting_time_path}")
    else:
        _log(f"{prefix}[1/4] Cutting by time")
        _set_stage_status(status_path, status, STAGE_CUTTING_TIME, "running", output=str(cutting_time_path))
        try:
            cut_steps_by_time(
                report_path=str(source_report),
                output_path=str(cutting_time_path),
                threshold_seconds=threshold_seconds,
            )
            _verify_output(cutting_time_path, STAGE_CUTTING_TIME)
        except Exception as exc:
            _set_stage_status(status_path, status, STAGE_CUTTING_TIME, "failed", error=str(exc))
            raise
        _set_stage_status(status_path, status, STAGE_CUTTING_TIME, "completed", output=str(cutting_time_path))
        _log(f"{prefix}[1/4] Time report saved: {cutting_time_path}")

    if resume and _stage_completed(status, STAGE_CUTTING_WINDOW, cutting_window_path):
        _log(f"{prefix}[2/4] Skip window cutting, existing output: {cutting_window_path}")
    else:
        _log(f"{prefix}[2/4] Cutting by window")
        _set_stage_status(status_path, status, STAGE_CUTTING_WINDOW, "running", output=str(cutting_window_path))
        try:
            cut_steps_by_window(
                report_path=str(cutting_time_path),
                output_path=str(cutting_window_path),
            )
            _verify_output(cutting_window_path, STAGE_CUTTING_WINDOW)
        except Exception as exc:
            _set_stage_status(status_path, status, STAGE_CUTTING_WINDOW, "failed", error=str(exc))
            raise
        _set_stage_status(status_path, status, STAGE_CUTTING_WINDOW, "completed", output=str(cutting_window_path))
        _log(f"{prefix}[2/4] Window report saved: {cutting_window_path}")

    if resume and _stage_completed(status, STAGE_WINDOW_MERGE, cutting_window_mix_path):
        _log(f"{prefix}[3/4] Skip LLM window merge, existing output: {cutting_window_mix_path}")
    else:
        _log(f"{prefix}[3/4] Merging adjacent windows with LLM")
        _set_stage_status(status_path, status, STAGE_WINDOW_MERGE, "running", output=str(cutting_window_mix_path))
        try:
            mixed_data = check_window_merges_with_codex(
                input_path=str(cutting_window_path),
                model=model,
            )
            with cutting_window_mix_path.open("w", encoding="utf-8") as f:
                json.dump(mixed_data, f, indent=2, ensure_ascii=False)
            _verify_output(cutting_window_mix_path, STAGE_WINDOW_MERGE)
        except Exception as exc:
            _set_stage_status(status_path, status, STAGE_WINDOW_MERGE, "failed", error=str(exc))
            raise
        _set_stage_status(status_path, status, STAGE_WINDOW_MERGE, "completed", output=str(cutting_window_mix_path))
        _log(f"{prefix}[3/4] Mixed window report saved: {cutting_window_mix_path}")

    if resume and _task_stage_completed(status, tasks_dir, task_prefix):
        task_count = status.get("stages", {}).get(STAGE_TASK_FILES, {}).get("task_count", 0)
        _log(f"{prefix}[4/4] Skip LLM task cutting, completed task count: {task_count}")
    else:
        _log(f"{prefix}[4/4] Cutting mixed segments into task files with LLM")
        _set_stage_status(status_path, status, STAGE_TASK_FILES, "running", output_dir=str(tasks_dir))
        try:
            run_codex_cut_and_split_for_mix(
                input_path=str(cutting_window_mix_path),
                output_dir=str(tasks_dir),
                model=model,
                prefix=task_prefix,
            )
            task_count = _count_task_files(tasks_dir, task_prefix)
            if task_count <= 0:
                raise RuntimeError(f"No task JSON files generated under {tasks_dir}")
        except Exception as exc:
            _set_stage_status(status_path, status, STAGE_TASK_FILES, "failed", error=str(exc))
            raise
        _set_stage_status(
            status_path,
            status,
            STAGE_TASK_FILES,
            "completed",
            output_dir=str(tasks_dir),
            task_count=task_count,
        )
        _log(f"{prefix}[4/4] Task reports directory: {tasks_dir} (files: {task_count})")

    _log(f"{prefix}Pipeline completed")
    return {
        "trace_dir": trace_dir,
        "source_report": source_report,
        "cutting_time": cutting_time_path,
        "cutting_window": cutting_window_path,
        "cutting_window_mix": cutting_window_mix_path,
        "tasks_dir": tasks_dir,
        "status": status_path,
    }


def run_batch_pipeline(
    batch_dir: str,
    json_file: Optional[str] = None,
    reports_dir_arg: Optional[str] = None,
    tasks_dir_arg: Optional[str] = None,
    threshold_seconds: int = 30,
    model: str = "gpt-5.5",
    task_prefix: str = "report_cutting_llm_task_",
    resume: bool = True,
    recursive: bool = False,
    stop_on_error: bool = False,
) -> Tuple[int, List[Tuple[Path, str]]]:
    trace_dirs = discover_trajectory_dirs(
        batch_dir=batch_dir,
        json_file=json_file,
        recursive=recursive,
    )
    total = len(trace_dirs)
    if total == 0:
        raise FileNotFoundError(
            f"No trajectory directories found under {batch_dir}. "
            "A trajectory directory should contain the selected report JSON and screenshots/."
        )

    _log(f"Batch pipeline started: {batch_dir}")
    _log(f"Discovered trajectories: {total}")

    success_count = 0
    failures: List[Tuple[Path, str]] = []
    for index, trace_dir in enumerate(trace_dirs, start=1):
        label = f"[trajectory {index}/{total}]"
        _log(f"{label} Processing: {trace_dir}")
        try:
            run_pipeline(
                trajectory_path=str(trace_dir),
                json_file=json_file,
                reports_dir_arg=reports_dir_arg,
                tasks_dir_arg=tasks_dir_arg,
                threshold_seconds=threshold_seconds,
                model=model,
                task_prefix=task_prefix,
                resume=resume,
                progress_label=label,
            )
        except Exception as exc:
            failures.append((trace_dir, str(exc)))
            _log(f"{label} Failed: {exc}")
            if stop_on_error:
                raise
            continue

        success_count += 1
        _log(f"{label} Finished")

    _log(f"Batch pipeline completed: success={success_count}, failed={len(failures)}, total={total}")
    if failures:
        _log("Failed trajectories:")
        for trace_dir, error in failures:
            _log(f"- {trace_dir}: {error}")

    return success_count, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full trajectory cutting pipeline: time -> window -> LLM merge -> LLM task files."
    )
    parser.add_argument(
        "trajectory_path",
        help="Path to the trajectory result directory. A JSON file path is also accepted.",
    )
    parser.add_argument(
        "-j",
        "--json-file",
        default=None,
        help=(
            "Input JSON filename under the trajectory directory, or an absolute JSON path. "
            "Defaults to report_denoised_filled_step.json, then report_denoised.json, then report.json."
        ),
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory for intermediate reports. Relative paths are resolved under trajectory_path. Default: ./splits.",
    )
    parser.add_argument(
        "-o",
        "--tasks-dir",
        default=None,
        help="Directory for final task JSON files. Relative paths are resolved under trajectory_path. Default: ./splits/tasks.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=30,
        help="Time boundary threshold in seconds. Default: 30.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-5.5",
        help="Model name for Codex LLM calls. Default: gpt-5.5.",
    )
    parser.add_argument(
        "--prefix",
        default="report_cutting_llm_task_",
        help="Prefix for generated task files. Default: report_cutting_llm_task_.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat trajectory_path as a parent directory and process all child trajectory directories.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --batch is used, recursively search for trajectory directories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Disable resume and rerun every stage even if previous outputs/status exist.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="When --batch is used, stop immediately if one trajectory fails.",
    )
    args = parser.parse_args()

    resume = not args.force
    if args.batch:
        _, failures = run_batch_pipeline(
            batch_dir=args.trajectory_path,
            json_file=args.json_file,
            reports_dir_arg=args.reports_dir,
            tasks_dir_arg=args.tasks_dir,
            threshold_seconds=args.threshold,
            model=args.model,
            task_prefix=args.prefix,
            resume=resume,
            recursive=args.recursive,
            stop_on_error=args.stop_on_error,
        )
        if failures:
            raise SystemExit(1)
    else:
        run_pipeline(
            trajectory_path=args.trajectory_path,
            json_file=args.json_file,
            reports_dir_arg=args.reports_dir,
            tasks_dir_arg=args.tasks_dir,
            threshold_seconds=args.threshold,
            model=args.model,
            task_prefix=args.prefix,
            resume=resume,
        )


if __name__ == "__main__":
    main()
