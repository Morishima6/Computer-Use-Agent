#!/usr/bin/env python3
"""Offline frame extractor for main_v22-yyp.py recordings.

Reads <session_dir>/report.json + <session_dir>/recording.mp4 and extracts
before/after screenshots into <session_dir>/result/screenshots/, then writes
a new report.json to <session_dir>/result/report.json (original report is
NOT modified).

Timing rule (per user spec):
    before frame = action_ts - 100 ms
    after  frame = action_ts + 500 ms
where action_ts is the mono_ns captured by the recorder, rebased onto the
video timeline via video_start_mono_ns.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


BEFORE_LEAD_SECONDS = 0.10   # 抽帧点在 action 瞬间往前 100ms
AFTER_LEAD_SECONDS = 0.50    # 抽帧点在 action 瞬间往后 500ms
PRECISE_SEEK_WINDOW = 2.0    # ffmpeg 粗定位 / 细定位分界

# 这些动作类型不画红色十字、也不生成 (part).png
NO_MARKER_ACTIONS = {"typing", "press"}

# 与旧实现保持一致的 (part).png 裁剪尺寸 / 红色十字样式
CROP_WIDTH = 600
CROP_HEIGHT = 400
HIGHLIGHT_RADIUS = 30
HIGHLIGHT_WIDTH = 4


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def ffmpeg_bin() -> str:
    b = shutil.which("ffmpeg")
    if b is None:
        raise RuntimeError("未找到 ffmpeg，请先安装")
    return b


def ffprobe_duration(video_path: Path) -> float | None:
    probe = shutil.which("ffprobe")
    if probe is None:
        return None
    try:
        r = run([
            probe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ])
    except subprocess.CalledProcessError:
        return None
    raw = r.stdout.strip()
    try:
        return max(0.0, float(raw)) if raw else None
    except ValueError:
        return None


def extract_frame(video_path: Path, offset_s: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if offset_s <= PRECISE_SEEK_WINDOW:
        args = [
            ffmpeg_bin(), "-y",
            "-i", str(video_path),
            "-ss", f"{offset_s:.3f}",
            "-frames:v", "1",
            str(out_path),
        ]
    else:
        coarse = max(0.0, offset_s - PRECISE_SEEK_WINDOW)
        fine = offset_s - coarse
        args = [
            ffmpeg_bin(), "-y",
            "-ss", f"{coarse:.3f}",
            "-i", str(video_path),
            "-ss", f"{fine:.3f}",
            "-frames:v", "1",
            str(out_path),
        ]
    run(args)


def draw_marker(src: Path, dst: Path, anchor: tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        canvas = img.copy()
        draw = ImageDraw.Draw(canvas)
        x, y = anchor
        r = HIGHLIGHT_RADIUS
        w = HIGHLIGHT_WIDTH
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=w)
        draw.line((x - r, y, x + r, y), fill="red", width=w)
        draw.line((x, y - r, x, y + r), fill="red", width=w)
        canvas.save(dst)


def crop_part(src: Path, dst: Path, anchor: tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        width, height = img.size
        half_w = CROP_WIDTH // 2
        half_h = CROP_HEIGHT // 2
        x, y = anchor
        left = max(0, x - half_w)
        top = max(0, y - half_h)
        right = min(width, left + CROP_WIDTH)
        bottom = min(height, top + CROP_HEIGHT)
        left = max(0, right - CROP_WIDTH)
        top = max(0, bottom - CROP_HEIGHT)
        img.crop((left, top, right, bottom)).save(dst)


def file_time_str(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def clamp_offset(off: float, duration: float | None) -> float:
    if off < 0:
        off = 0.0
    if duration is not None and duration > 0:
        off = min(off, max(0.0, duration - 0.001))
    return off


def process_session(session_dir: Path) -> None:
    report_path = session_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"找不到 report.json: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    video_artifact = report.get("video_artifact") or {}
    video_rel = video_artifact.get("path")
    if not video_rel:
        raise ValueError("report.json 缺少 video_artifact.path")
    video_path = session_dir / video_rel
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    video_start_ns = video_artifact.get("video_start_mono_ns")
    if video_start_ns is None:
        raise ValueError("report.json 缺少 video_start_mono_ns")

    result_dir = session_dir / "result"
    screenshots_dir = result_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(video_path)

    for step in report.get("steps", []):
        plan = step.get("capture_plan") or {}
        before_ns = plan.get("before_ts_mono_ns")
        after_ns = plan.get("after_ts_mono_ns")
        anchor = plan.get("anchor_position")
        action = step.get("action") or {}
        action_type = str(action.get("type", ""))
        prefix = "drag" if action_type == "drag_to" else action_type
        if not prefix:
            continue
        step_id = step.get("step_id") or "step"

        if before_ns is None:
            print(f"[{step_id}] 跳过: 缺少 before_ts_mono_ns")
            continue
        if after_ns is None:
            after_ns = before_ns
            print(f"[{step_id}] 警告: after_ts_mono_ns 为空，已从 before_ts 估算")

        before_offset = (before_ns - video_start_ns) / 1_000_000_000 - BEFORE_LEAD_SECONDS
        if action_type == "typing":
            # typing 的 after_ts 是下一个动作触发时才记录的，
            # after 帧应往前取：拿到"打完字、下一个动作还没生效"的画面
            after_offset = (after_ns - video_start_ns) / 1_000_000_000 - BEFORE_LEAD_SECONDS
        else:
            after_offset = (after_ns - video_start_ns) / 1_000_000_000 + AFTER_LEAD_SECONDS
        before_offset = clamp_offset(before_offset, duration)
        after_offset = clamp_offset(after_offset, duration)

        before_out = screenshots_dir / f"{step_id}_before_{prefix}.png"
        before_raw_out = screenshots_dir / f"{step_id}_before_{prefix}(raw).png"
        before_part_out = screenshots_dir / f"{step_id}_before_{prefix}(part).png"
        after_out = screenshots_dir / f"{step_id}_after_{prefix}.png"

        try:
            with tempfile.TemporaryDirectory(prefix="extract_v22_") as tmp:
                raw_before = Path(tmp) / f"{step_id}_before_raw.png"
                extract_frame(video_path, before_offset, raw_before)

                draw_anchor = (action_type not in NO_MARKER_ACTIONS) and (
                    isinstance(anchor, list) and len(anchor) == 2
                )
                if draw_anchor:
                    anchor_tuple = (int(anchor[0]), int(anchor[1]))
                    draw_marker(raw_before, before_out, anchor_tuple)
                    shutil.copyfile(raw_before, before_raw_out)
                    crop_part(raw_before, before_part_out, anchor_tuple)
                else:
                    shutil.copyfile(raw_before, before_out)

                extract_frame(video_path, after_offset, after_out)
        except Exception as exc:
            print(f"[{step_id}] 抽帧失败，跳过: {exc}", file=sys.stderr)
            continue

        now_state = step.setdefault("now_state", {})
        now_state["screenshot_path_before"] = os.path.relpath(before_out, result_dir)
        now_state["screenshot_path_before_part"] = (
            os.path.relpath(before_part_out, result_dir) if before_part_out.exists() else None
        )
        now_state["screenshot_path_after"] = os.path.relpath(after_out, result_dir)
        now_state["screenshot_time_before"] = file_time_str(before_out)
        now_state["screenshot_time_after"] = file_time_str(after_out)

        print(f"[{step_id}] {action_type}: before@{before_offset:.3f}s after@{after_offset:.3f}s")

    result_report_path = result_dir / "report.json"
    result_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"抽帧完成: {screenshots_dir}")
    print(f"结果报告: {result_report_path}")
    print(f"原始报告未修改: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="根据 report.json 从录屏中抽取 before/after 截图")
    parser.add_argument("session_dir", help="会话目录（包含 report.json 与 recording.mp4）")
    args = parser.parse_args()
    session_dir = Path(args.session_dir).resolve()
    if not session_dir.is_dir():
        print(f"不是目录: {session_dir}", file=sys.stderr)
        return 2
    try:
        process_session(session_dir)
    except Exception as exc:
        print(f"抽帧失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
