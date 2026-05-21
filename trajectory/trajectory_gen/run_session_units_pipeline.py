#!/usr/bin/env python3
"""
Run the session trajectory pipeline in order:
1. Ambler rule/LLM task cutting -> splits/ and splits/tasks/
2. Phase-1 unit cutting -> segments_units/
3. Phase-2 unit parameterization -> segments_units_phase2/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
CUT_REPORT_SCRIPT = SCRIPT_DIR / "Ambler_cutting" / "pipeline_cut_report.py"
PHASE1_SCRIPT = SCRIPT_DIR / "unit_cutting" / "phase1_segment_units.py"
PHASE2_SCRIPT = SCRIPT_DIR / "unit_cutting" / "phase2_parameterize_units.py"

STATUS_FILE_NAME = "_trajectory_units_pipeline_status.json"
LOG_FILE_NAME = "_trajectory_units_pipeline.log"

STAGE_CUT_REPORT = "cut_report"
STAGE_SEGMENT_UNITS = "segment_units"
STAGE_PARAMETERIZE_UNITS = "parameterize_units"


@dataclass(frozen=True)
class PipelineTarget:
    session_dir: Path
    json_file: str
    report_path: Path


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class PipelineLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{now_text()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def resolve_session_and_json(input_path: str, json_file: str) -> Tuple[Path, str, Path]:
    path = Path(input_path).resolve()
    if path.is_file():
        session_dir = path.parent
        selected_json = path.name
        report_path = path
    else:
        session_dir = path
        selected_json = json_file
        report_path = Path(json_file)
        if not report_path.is_absolute():
            report_path = session_dir / report_path

    if not session_dir.is_dir():
        raise NotADirectoryError(f"Session directory not found: {session_dir}")
    if not selected_json:
        raise ValueError("Please provide --json-file when input_path is a session directory.")
    if not report_path.is_file():
        raise FileNotFoundError(f"Input report JSON not found: {report_path}")
    return session_dir, selected_json, report_path


def resolve_target_from_path(input_path: str, json_file: str, session_subdir: str) -> PipelineTarget:
    path = Path(input_path).resolve()
    if path.is_file():
        return PipelineTarget(session_dir=path.parent, json_file=path.name, report_path=path)

    if not json_file:
        raise ValueError("Please provide --json-file when input_path is a directory.")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path not found: {path}")

    direct_report = path / json_file
    if direct_report.is_file():
        return PipelineTarget(session_dir=path, json_file=json_file, report_path=direct_report)

    normalized_session_dir = path / session_subdir
    normalized_report = normalized_session_dir / json_file
    if normalized_report.is_file():
        return PipelineTarget(
            session_dir=normalized_session_dir,
            json_file=json_file,
            report_path=normalized_report,
        )

    raise FileNotFoundError(
        f"Could not resolve a trajectory session from {path}. Tried:\n"
        f"  - {direct_report}\n"
        f"  - {normalized_report}"
    )


def discover_targets_under_root(root_path: str, json_file: str, session_subdir: str, recursive: bool) -> List[PipelineTarget]:
    if not json_file:
        raise ValueError("--json-file is required in --batch mode.")

    root = Path(root_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Batch root not found: {root}")

    targets: List[PipelineTarget] = []

    # 兼容把真实 result 子目录直接作为 batch root 传入的情况。
    direct_report = root / json_file
    if direct_report.is_file():
        targets.append(PipelineTarget(session_dir=root, json_file=json_file, report_path=direct_report))

    if recursive:
        candidates = [
            candidate
            for candidate in root.rglob(session_subdir)
            if candidate.is_dir()
        ]
    else:
        candidates = [
            child / session_subdir
            for child in sorted(root.iterdir())
            if child.is_dir()
        ]

    for session_dir in sorted(candidates):
        report_path = session_dir / json_file
        if report_path.is_file():
            targets.append(PipelineTarget(session_dir=session_dir, json_file=json_file, report_path=report_path))

    return deduplicate_targets(targets)


def deduplicate_targets(targets: Sequence[PipelineTarget]) -> List[PipelineTarget]:
    unique: List[PipelineTarget] = []
    seen: set[Path] = set()
    for target in targets:
        key = target.report_path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    return unique


def resolve_requested_targets(args: argparse.Namespace) -> List[PipelineTarget]:
    targets: List[PipelineTarget] = []
    if args.batch:
        for input_path in args.input_paths:
            targets.extend(
                discover_targets_under_root(
                    root_path=input_path,
                    json_file=args.json_file,
                    session_subdir=args.session_subdir,
                    recursive=args.recursive,
                )
            )
    else:
        for input_path in args.input_paths:
            targets.append(
                resolve_target_from_path(
                    input_path=input_path,
                    json_file=args.json_file,
                    session_subdir=args.session_subdir,
                )
            )

    targets = deduplicate_targets(targets)
    if args.trajectory_limit > 0:
        targets = targets[: args.trajectory_limit]
    if not targets:
        raise FileNotFoundError("No trajectory sessions were resolved from the provided input paths.")
    return targets


def count_json_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob("*.json") if item.is_file())


def stage_output_complete(stage: str, session_dir: Path, limit: int = 0) -> Tuple[bool, str]:
    tasks_dir = session_dir / "splits" / "tasks"
    segments_units_dir = session_dir / "segments_units"
    phase2_dir = session_dir / "segments_units_phase2"

    if stage == STAGE_CUT_REPORT:
        task_count = count_json_files(tasks_dir)
        return task_count > 0, f"{task_count} task JSON files under {tasks_dir}"

    if stage == STAGE_SEGMENT_UNITS:
        task_count = count_json_files(tasks_dir)
        output_count = count_json_files(segments_units_dir)
        expected = min(task_count, limit) if limit > 0 else task_count
        return expected > 0 and output_count >= expected, (
            f"{output_count}/{expected} segment-unit JSON files under {segments_units_dir}"
        )

    if stage == STAGE_PARAMETERIZE_UNITS:
        segment_count = count_json_files(segments_units_dir)
        output_count = count_json_files(phase2_dir)
        expected = min(segment_count, limit) if limit > 0 else segment_count
        return expected > 0 and output_count >= expected, (
            f"{output_count}/{expected} phase2 JSON files under {phase2_dir}"
        )

    raise ValueError(f"Unknown stage: {stage}")


def update_pipeline_status(
    status_path: Path,
    *,
    stage: Optional[str] = None,
    stage_payload: Optional[Dict[str, Any]] = None,
    **top_level: Any,
) -> Dict[str, Any]:
    status = read_json_file(status_path)
    if not status:
        status = {"stages": {}}
    if not isinstance(status.get("stages"), dict):
        status["stages"] = {}
    status.update(top_level)
    status["updated_at"] = now_text()
    if stage is not None and stage_payload is not None:
        stages = status.setdefault("stages", {})
        previous = stages.get(stage, {})
        if not isinstance(previous, dict):
            previous = {}
        previous.update(stage_payload)
        stages[stage] = previous
    write_json_file(status_path, status)
    return status


def run_streaming_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    logger: PipelineLogger,
    stage_label: str,
) -> int:
    logger.log(f"{stage_label} command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        logger.log(f"{stage_label} {line}")
    return process.wait()


def build_stage_commands(args: argparse.Namespace, session_dir: Path, json_file: str) -> List[Tuple[str, List[str]]]:
    python_cmd = [sys.executable, "-u"]

    cut_cmd = [
        *python_cmd,
        str(CUT_REPORT_SCRIPT),
        str(session_dir),
        "--json-file",
        json_file,
        "--reports-dir",
        "splits",
        "--tasks-dir",
        "splits/tasks",
        "--threshold",
        str(args.threshold),
        "--model",
        args.model,
        "--prefix",
        args.task_prefix,
    ]
    if args.force:
        cut_cmd.append("--force")

    phase1_log = session_dir / "_phase1_unit_cutting.log"
    phase1_cmd = [
        *python_cmd,
        str(PHASE1_SCRIPT),
        str(session_dir),
        "--tasks-subdir",
        "splits/tasks",
        "--output-dir",
        str(session_dir / "segments_units"),
        "--backend",
        args.phase1_backend,
        "--max-images",
        str(args.max_images),
        "--max-retries",
        str(args.max_retries),
        "--retry-delay",
        str(args.retry_delay),
        "--window-fallback-threshold",
        str(args.window_fallback_threshold),
        "--window-size",
        str(args.window_size),
        "--window-overlap",
        str(args.window_overlap),
        "--limit",
        str(args.limit),
        "--log-file",
        str(phase1_log),
    ]
    if args.phase1_model:
        phase1_cmd.extend(["--model", args.phase1_model])
    elif args.phase1_backend == "codex":
        phase1_cmd.extend(["--model", args.model])
    else:
        phase1_cmd.extend(["--model", "qwen-vl-plus"])
    if args.api_key:
        phase1_cmd.extend(["--api-key", args.api_key])
    if args.base_url:
        phase1_cmd.extend(["--base-url", args.base_url])
    if args.force:
        phase1_cmd.append("--force")
    if args.stop_on_error:
        phase1_cmd.append("--stop-on-error")

    phase2_log = session_dir / "_phase2_unit_parameterization.log"
    phase2_cmd = [
        *python_cmd,
        str(PHASE2_SCRIPT),
        str(session_dir),
        "--units-subdir",
        "segments_units",
        "--output-dir",
        str(session_dir / "segments_units_phase2"),
        "--model",
        args.phase2_model or args.model,
        "--max-retries",
        str(args.max_retries),
        "--retry-delay",
        str(args.retry_delay),
        "--limit",
        str(args.limit),
        "--log-file",
        str(phase2_log),
    ]
    if args.force:
        phase2_cmd.append("--force")
    if args.stop_on_error:
        phase2_cmd.append("--stop-on-error")

    return [
        (STAGE_CUT_REPORT, cut_cmd),
        (STAGE_SEGMENT_UNITS, phase1_cmd),
        (STAGE_PARAMETERIZE_UNITS, phase2_cmd),
    ]


def run_pipeline(args: argparse.Namespace) -> int:
    session_dir, json_file, report_path = resolve_session_and_json(args.input_path, args.json_file)
    status_path = session_dir / STATUS_FILE_NAME
    logger = PipelineLogger(session_dir / LOG_FILE_NAME)
    started_at = datetime.now()
    progress_label = str(getattr(args, "progress_label", "") or "").strip()
    log_prefix = f"{progress_label} " if progress_label else ""

    update_pipeline_status(
        status_path,
        status="in_progress",
        session_dir=str(session_dir),
        input_report=str(report_path),
        json_file=json_file,
        started_at=now_text(),
        finished_at="",
        duration_seconds=0.0,
        model=args.model,
        force=args.force,
    )

    logger.log("=" * 80)
    logger.log(f"{log_prefix}Session units pipeline started")
    logger.log(f"{log_prefix}Session directory: {session_dir}")
    logger.log(f"{log_prefix}Input JSON: {report_path}")
    logger.log(f"{log_prefix}Resume: {'disabled (--force)' if args.force else 'enabled'}")
    logger.log(f"{log_prefix}Stage order: cut_report -> segment_units -> parameterize_units")

    commands = build_stage_commands(args, session_dir, json_file)
    total = len(commands)

    for index, (stage, cmd) in enumerate(commands, start=1):
        stage_label = f"{log_prefix}[{index}/{total} {stage}]"
        complete, detail = stage_output_complete(stage, session_dir, limit=args.limit)
        if complete and not args.force:
            logger.log(f"{stage_label} skip: existing output complete ({detail})")
            update_pipeline_status(
                status_path,
                stage=stage,
                stage_payload={
                    "status": "skipped",
                    "detail": detail,
                    "skipped_at": now_text(),
                },
            )
            continue

        logger.log(f"{stage_label} start")
        logger.log(f"{stage_label} current output state before run: {detail}")
        stage_start = time.perf_counter()
        update_pipeline_status(
            status_path,
            stage=stage,
            stage_payload={
                "status": "running",
                "started_at": now_text(),
                "command": cmd,
                "detail_before": detail,
            },
        )

        return_code = run_streaming_command(
            cmd,
            cwd=SCRIPT_DIR,
            logger=logger,
            stage_label=stage_label,
        )
        duration_seconds = time.perf_counter() - stage_start
        complete_after, detail_after = stage_output_complete(stage, session_dir, limit=args.limit)

        if return_code != 0 or not complete_after:
            status = "failed"
            logger.log(
                f"{stage_label} failed: return_code={return_code}, "
                f"output_complete={complete_after}, detail={detail_after}"
            )
            update_pipeline_status(
                status_path,
                status="failed",
                stage=stage,
                stage_payload={
                    "status": status,
                    "return_code": return_code,
                    "finished_at": now_text(),
                    "duration_seconds": round(duration_seconds, 3),
                    "detail_after": detail_after,
                },
                finished_at=now_text(),
                duration_seconds=round((datetime.now() - started_at).total_seconds(), 3),
            )
            return return_code or 1

        logger.log(f"{stage_label} completed in {format_duration(duration_seconds)} ({detail_after})")
        update_pipeline_status(
            status_path,
            stage=stage,
            stage_payload={
                "status": "done",
                "return_code": return_code,
                "finished_at": now_text(),
                "duration_seconds": round(duration_seconds, 3),
                "detail_after": detail_after,
            },
        )

    total_duration = (datetime.now() - started_at).total_seconds()
    update_pipeline_status(
        status_path,
        status="done",
        finished_at=now_text(),
        duration_seconds=round(total_duration, 3),
    )
    logger.log(f"{log_prefix}Session units pipeline completed in {format_duration(total_duration)}")
    logger.log(f"{log_prefix}Status file: {status_path}")
    logger.log(f"{log_prefix}Log file: {logger.log_path}")
    return 0


def make_session_args(args: argparse.Namespace, target: PipelineTarget, index: int, total: int) -> argparse.Namespace:
    session_args = copy.copy(args)
    session_args.input_path = str(target.session_dir)
    session_args.json_file = target.json_file
    session_args.progress_label = f"[session {index}/{total} {target.session_dir.parent.name}/{target.session_dir.name}]"
    return session_args


def default_batch_status_path(args: argparse.Namespace, targets: Sequence[PipelineTarget]) -> Path:
    if args.batch_status_file:
        return Path(args.batch_status_file).resolve()

    common_root = Path(os.path.commonpath([str(target.session_dir) for target in targets]))
    if common_root.is_file():
        common_root = common_root.parent
    return common_root / "_trajectory_units_pipeline_batch_status.json"


def write_batch_status(
    status_path: Path,
    *,
    status: str,
    targets: Sequence[PipelineTarget],
    results: Dict[str, Dict[str, Any]],
    started_at: datetime,
    workers: int,
) -> None:
    sessions = []
    for target in targets:
        key = str(target.session_dir)
        sessions.append(
            {
                "session_dir": key,
                "json_file": target.json_file,
                "report_path": str(target.report_path),
                **results.get(key, {"status": "pending"}),
            }
        )

    write_json_file(
        status_path,
        {
            "status": status,
            "started_at": format_timestamp(started_at),
            "updated_at": now_text(),
            "finished_at": now_text() if status in {"done", "failed"} else "",
            "duration_seconds": round((datetime.now() - started_at).total_seconds(), 3),
            "workers": workers,
            "total": len(targets),
            "succeeded": sum(1 for item in results.values() if item.get("status") == "done"),
            "failed": sum(1 for item in results.values() if item.get("status") == "failed"),
            "sessions": sessions,
        },
    )


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def run_one_target(args: argparse.Namespace, target: PipelineTarget, index: int, total: int) -> Tuple[PipelineTarget, int, str]:
    session_args = make_session_args(args, target, index, total)
    try:
        return_code = run_pipeline(session_args)
    except Exception as exc:
        return target, 1, f"{type(exc).__name__}: {exc}"
    return target, return_code, ""


def run_targets(args: argparse.Namespace, targets: Sequence[PipelineTarget]) -> int:
    total = len(targets)
    workers = max(1, min(args.workers, total))
    started_at = datetime.now()
    batch_status_path = default_batch_status_path(args, targets)
    results: Dict[str, Dict[str, Any]] = {}

    print(f"[{now_text()}] Batch pipeline started: sessions={total}, workers={workers}", flush=True)
    print(f"[{now_text()}] Batch status file: {batch_status_path}", flush=True)
    for index, target in enumerate(targets, start=1):
        print(f"[{now_text()}] [{index}/{total}] target: {target.session_dir} ({target.json_file})", flush=True)

    write_batch_status(
        batch_status_path,
        status="in_progress",
        targets=targets,
        results=results,
        started_at=started_at,
        workers=workers,
    )

    failures: List[Tuple[PipelineTarget, str]] = []

    if workers == 1:
        for index, target in enumerate(targets, start=1):
            key = str(target.session_dir)
            results[key] = {"status": "running", "started_at": now_text()}
            write_batch_status(
                batch_status_path,
                status="in_progress",
                targets=targets,
                results=results,
                started_at=started_at,
                workers=workers,
            )
            finished_target, return_code, error = run_one_target(args, target, index, total)
            finished_key = str(finished_target.session_dir)
            results[finished_key] = {
                "status": "done" if return_code == 0 else "failed",
                "return_code": return_code,
                "error": error,
                "finished_at": now_text(),
            }
            if return_code != 0:
                failures.append((finished_target, error or f"return_code={return_code}"))
            write_batch_status(
                batch_status_path,
                status="in_progress",
                targets=targets,
                results=results,
                started_at=started_at,
                workers=workers,
            )
            if return_code != 0 and args.stop_on_error:
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_target: Dict[concurrent.futures.Future[Tuple[PipelineTarget, int, str]], PipelineTarget] = {}
            for index, target in enumerate(targets, start=1):
                key = str(target.session_dir)
                results[key] = {"status": "running", "started_at": now_text()}
                future = executor.submit(run_one_target, args, target, index, total)
                future_to_target[future] = target

            write_batch_status(
                batch_status_path,
                status="in_progress",
                targets=targets,
                results=results,
                started_at=started_at,
                workers=workers,
            )

            should_stop = False
            for future in concurrent.futures.as_completed(future_to_target):
                target = future_to_target[future]
                if future.cancelled():
                    key = str(target.session_dir)
                    results[key] = {
                        "status": "failed",
                        "return_code": 1,
                        "error": "cancelled after --stop-on-error",
                        "finished_at": now_text(),
                    }
                    failures.append((target, "cancelled after --stop-on-error"))
                    continue
                finished_target, return_code, error = future.result()
                key = str(finished_target.session_dir)
                results[key] = {
                    "status": "done" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "error": error,
                    "finished_at": now_text(),
                }
                if return_code != 0:
                    failures.append((finished_target, error or f"return_code={return_code}"))
                    if args.stop_on_error and not should_stop:
                        should_stop = True
                        for pending in future_to_target:
                            if not pending.done():
                                pending.cancel()
                write_batch_status(
                    batch_status_path,
                    status="in_progress",
                    targets=targets,
                    results=results,
                    started_at=started_at,
                    workers=workers,
                )

    final_status = "failed" if failures else "done"
    write_batch_status(
        batch_status_path,
        status=final_status,
        targets=targets,
        results=results,
        started_at=started_at,
        workers=workers,
    )

    succeeded = sum(1 for item in results.values() if item.get("status") == "done")
    print(
        f"[{now_text()}] Batch pipeline finished: success={succeeded}, "
        f"failed={len(failures)}, total={total}, duration={format_duration((datetime.now() - started_at).total_seconds())}",
        flush=True,
    )
    if failures:
        print(f"[{now_text()}] Failed sessions:", flush=True)
        for target, error in failures:
            print(f"[{now_text()}] - {target.session_dir}: {error}", flush=True)
        return 1
    return 0


def print_resolved_targets(targets: Sequence[PipelineTarget]) -> None:
    print(f"[{now_text()}] Resolved trajectory sessions: {len(targets)}", flush=True)
    for index, target in enumerate(targets, start=1):
        print(
            f"[{now_text()}] [{index}/{len(targets)}] session_dir={target.session_dir} "
            f"json_file={target.json_file}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cut_report -> phase1 unit cutting -> phase2 parameterization for one or more "
            "trajectory sessions, with timestamped streaming logs and resumable status."
        )
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help=(
            "Trajectory session/result directory, concrete input report JSON path, or batch root "
            "when --batch is enabled."
        ),
    )
    parser.add_argument(
        "-j",
        "--json-file",
        default="",
        help="Input JSON filename under each real result/session directory. Required for directory inputs.",
    )
    parser.add_argument(
        "--session-subdir",
        default="result",
        help=(
            "Subdirectory to process under each trajectory session folder. Default: result. "
            "Example: --session-subdir result-trimmed."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Treat each input path as a data root and process child session folders under it. "
            "Each child is normalized to <child>/<session-subdir>."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="In --batch mode, recursively search for directories named --session-subdir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of trajectory sessions to process in parallel. Default: 1.",
    )
    parser.add_argument(
        "--trajectory-limit",
        type=int,
        default=0,
        help="Optional maximum number of resolved trajectory sessions to process. 0 means no limit.",
    )
    parser.add_argument(
        "--batch-status-file",
        default="",
        help="Optional path for the aggregate batch status JSON file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve and print target sessions; do not run any pipeline stage.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Default model for the cutting pipeline, Phase-1 Codex backend, and Phase-2. Default: gpt-5.5.",
    )
    parser.add_argument(
        "--phase1-model",
        default="",
        help="Optional model override for phase1_segment_units.py. Leave empty to use --model for codex.",
    )
    parser.add_argument(
        "--phase2-model",
        default="",
        help="Optional model override for phase2_parameterize_units.py. Leave empty to use --model.",
    )
    parser.add_argument(
        "--phase1-backend",
        choices=("codex", "qwen"),
        default="codex",
        help="Backend for phase1_segment_units.py. Default: codex.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Time boundary threshold in seconds for cut_report. Default: 30.",
    )
    parser.add_argument(
        "--task-prefix",
        default="report_cutting_llm_task_",
        help="Prefix for task JSON files generated under splits/tasks.",
    )
    parser.add_argument("--api-key", default="", help="Optional API key for Phase-1 qwen backend.")
    parser.add_argument("--base-url", default="", help="Optional base URL for Phase-1 qwen backend.")
    parser.add_argument("--max-images", type=int, default=20, help="Phase-1 max screenshots per segment.")
    parser.add_argument("--max-retries", type=int, default=3, help="Generation retries per segment/unit.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Seconds between retries.")
    parser.add_argument("--window-fallback-threshold", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--window-overlap", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit task/segment JSON files per trajectory for Phase-1 and Phase-2. 0 means no limit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Disable resume checks and pass --force to all three stages.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Pass --stop-on-error to Phase-1 and Phase-2.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = resolve_requested_targets(args)
    if args.dry_run:
        print_resolved_targets(targets)
        return 0
    return run_targets(args, targets)


if __name__ == "__main__":
    raise SystemExit(main())
