from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.compare_screenshot_similarity import compare_screenshot_similarity
from utils.vlm_similarity_gate import judge_action


DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
ENV_PATH = Path(".env")
DEFAULT_FULL_SIMILARITY_THRESHOLD = 0.98
DEFAULT_SAME_REGION_RADIUS = 28.0
DEFAULT_SCROLL_CANCEL_GAP_SECONDS = 1.5
DEFAULT_SMALL_DRAG_THRESHOLD = 15.0
DEFAULT_TYPING_LOOKAHEAD_STEPS = 2
DEFAULT_KIMI_BASE_URL = "https://api.kimi.com/coding/"
DEFAULT_KIMI_MODEL = "kimi-2.5"
DEFAULT_KIMI_MAX_TOKENS = 512
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
DEFAULT_ARK_MODEL = "Kimi-K2.5"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.6-plus"
DEFAULT_VLM_VERIFICATION_REPEATS = 3

MODIFIER_KEYS = {"shift", "ctrl", "alt", "meta", "command", "cmd", "super"}
CLICK_LIKE_ACTION_TYPES = {"click", "double_click", "triple_click"}
POTENTIALLY_INEFFECTIVE_PRESS_KEYS = {
    "backspace",
    "delete",
    "del",
    "enter",
    "return",
    "escape",
    "esc",
    "home",
    "end",
    "pageup",
    "pagedown",
    "page_up",
    "page_down",
}
EDITING_OR_SELECTION_PRESS_KEYS = {
    "a",
    "backspace",
    "delete",
    "del",
    "enter",
    "return",
    "escape",
    "esc",
    "tab",
}
GENERIC_APP_TITLES = {"gnome shell", "desktop", "unknown", ""}
TERMINAL_KEYWORDS = {
    "terminal",
    "gnome-terminal",
    "gnome-terminal-server",
    "powershell",
    "command prompt",
    "cmd.exe",
    "cmd",
    "bash",
    "zsh",
    "konsole",
    "xterm",
}
KNOWN_APP_ALIASES = {
    "chrome": {"chrome", "google chrome", "chromium", "Chrome"},
    "thunderbird": {"thunderbird"},
    "vlc": {"vlc", "vlc media player"},
    "gimp": {"gimp"},
    "impress": {"libreoffice impress", "impress", "soffice"},
    "writer": {"libreoffice writer", "writer", "soffice"},
    "calc": {"libreoffice calc", "calc", "soffice"},
    "code": {"visual studio code", "vscode", "code", "vs_code"},
    "multi_apps": {"multi_apps", "multiple applications"},
    "os": {"os", "operating system", "system settings", "control panel"}
}
VLM_PREFILTER_SIMILARITY_THRESHOLD = 0.98


def parse_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ENV_PATH)


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {
                "timestamp": current_timestamp(),
                "event": event,
                **payload,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise GUI trajectory steps with screenshot similarity rules."
    )
    parser.add_argument(
        "input_path",
        help=(
            "Path to a report.json file, a session directory, or a batch root when --batch is enabled."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all session directories under input_path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for --batch. Default: 1.",
    )
    parser.add_argument(
        "--report-subdir",
        default="",
        help=(
            "Subdirectory under each session directory that contains report.json. "
            "Default is empty, meaning <session>/report.json. For SCUBA extracted results, use: result."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output path for the denoised report. Defaults to report_denoised.json beside report.json.",
    )
    parser.add_argument(
        "--output-audit",
        type=str,
        default=None,
        help="Output path for the denoise audit. Defaults to report_denoise_audit.json beside report.json.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save resume audit after every N newly processed steps. Default: 10.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing denoise audit and process reports from scratch.",
    )
    parser.add_argument(
        "--vlm-verify-backend",
        choices=["none", "auto", "kimi", "ark", "qwen"],
        default="auto",
        help="Optional multimodal screenshot-similarity verification for risky deletions. 'auto' prefers Kimi Anthropic-compatible access, then falls back to Ark when available.",
    )
    parser.set_defaults(
        full_similarity_threshold=DEFAULT_FULL_SIMILARITY_THRESHOLD,
        same_region_radius=DEFAULT_SAME_REGION_RADIUS,
        scroll_cancel_gap_seconds=DEFAULT_SCROLL_CANCEL_GAP_SECONDS,
        small_drag_threshold=DEFAULT_SMALL_DRAG_THRESHOLD,
        typing_lookahead_steps=DEFAULT_TYPING_LOOKAHEAD_STEPS,
        vlm_http_trust_env=parse_env_bool("VLM_HTTP_TRUST_ENV", False),
        kimi_api_key=os.environ.get("KIMI_API_KEY", ""),
        kimi_base_url=os.environ.get("KIMI_BASE_URL", DEFAULT_KIMI_BASE_URL),
        kimi_model=os.environ.get("KIMI_MODEL", DEFAULT_KIMI_MODEL),
        kimi_max_tokens=DEFAULT_KIMI_MAX_TOKENS,
        ark_api_key=os.environ.get("ARK_API_KEY", ""),
        ark_base_url=os.environ.get("ARK_BASE_URL", DEFAULT_ARK_BASE_URL),
        ark_model=os.environ.get("ARK_MODEL", DEFAULT_ARK_MODEL),
        qwen_api_key=os.environ.get("QWEN_API_KEY", ""),
        qwen_base_url=os.environ.get("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL),
        qwen_model=os.environ.get("QWEN_MODEL", DEFAULT_QWEN_MODEL),
    )
    return parser.parse_args()


def resolve_report_path(input_path: str, report_subdir: str = "") -> Path:
    path = Path(input_path)
    if path.is_dir():
        report_dir = path / report_subdir if report_subdir else path
        report_path = report_dir / "report.json"
    else:
        report_path = path
    if not report_path.exists():
        raise FileNotFoundError(f"Cannot find report.json at: {report_path}")
    return report_path.resolve()


def find_batch_report_paths(input_path: str, report_subdir: str) -> List[Path]:
    root = Path(input_path).resolve()
    if root.is_file():
        return [resolve_report_path(str(root), report_subdir)]
    if not root.is_dir():
        raise FileNotFoundError(f"Cannot find input directory: {root}")

    report_paths: List[Path] = []
    root_report_dir = root / report_subdir if report_subdir else root
    root_report_path = root_report_dir / "report.json"
    if root_report_path.is_file():
        report_paths.append(root_report_path.resolve())

    for session_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        report_dir = session_dir / report_subdir if report_subdir else session_dir
        report_path = report_dir / "report.json"
        if report_path.is_file():
            report_paths.append(report_path.resolve())

    return report_paths


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def atomic_dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    dump_json(tmp_path, payload)
    os.replace(tmp_path, path)


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, DATETIME_FMT)
    except ValueError:
        return None


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 2 or len(b) != 2:
        return float("inf")
    return math.dist((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))


def normalize_press_list(press_list: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(key).lower() for key in press_list)


def is_modifier_only_press(step: Dict[str, Any]) -> bool:
    action = step.get("action", {})
    if action.get("type") != "press":
        return False
    press_list = normalize_press_list(action.get("param", {}).get("press_list", []))
    return len(press_list) == 1 and press_list[0] in MODIFIER_KEYS


def is_undo_press(step: Dict[str, Any]) -> bool:
    action = step.get("action", {})
    if action.get("type") != "press":
        return False
    press_list = normalize_press_list(action.get("param", {}).get("press_list", []))
    return press_list in {("ctrl", "z"), ("command", "z"), ("meta", "z")}


def get_primary_press_key(step: Dict[str, Any]) -> Optional[str]:
    action = step.get("action", {})
    if action.get("type") != "press":
        return None
    press_list = normalize_press_list(action.get("param", {}).get("press_list", []))
    non_modifiers = [key for key in press_list if key not in MODIFIER_KEYS]
    return non_modifiers[-1] if non_modifiers else None


def is_potentially_ineffective_press(step: Dict[str, Any]) -> bool:
    action = step.get("action", {})
    if action.get("type") != "press":
        return False
    press_list = normalize_press_list(action.get("param", {}).get("press_list", []))
    if not press_list or is_modifier_only_press(step) or is_undo_press(step):
        return False
    non_modifiers = [key for key in press_list if key not in MODIFIER_KEYS]
    if len(non_modifiers) != 1:
        return False
    return non_modifiers[0] in POTENTIALLY_INEFFECTIVE_PRESS_KEYS


def is_editing_or_selection_step(step: Dict[str, Any]) -> bool:
    action_type = get_action_type(step)
    if action_type in {"typing", "drag_to", "double_click", "triple_click"}:
        return True
    if action_type != "press":
        return False

    press_list = normalize_press_list(step.get("action", {}).get("param", {}).get("press_list", []))
    if not press_list:
        return False
    non_modifiers = [key for key in press_list if key not in MODIFIER_KEYS]
    if len(non_modifiers) != 1:
        return False
    return non_modifiers[0] in EDITING_OR_SELECTION_PRESS_KEYS


def get_click_count(step: Dict[str, Any]) -> int:
    action = step.get("action", {})
    return int(action.get("param", {}).get("num_click", 1) or 1)


def get_action_type(step: Dict[str, Any]) -> str:
    return str(step.get("action", {}).get("type", "")).lower()


def is_click_like_mouse_action(step: Dict[str, Any]) -> bool:
    return get_action_type(step) in CLICK_LIKE_ACTION_TYPES


def get_model_before_rel(step: Dict[str, Any]) -> Optional[str]:
    now_state = step.get("now_state", {})
    if not isinstance(now_state, dict):
        return None
    raw_rel = now_state.get("screenshot_path_before_raw")
    return str(raw_rel) if raw_rel else None


def get_target_position(step: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    action = step.get("action", {})
    target = action.get("target", {}).get("position")
    if isinstance(target, (list, tuple)) and len(target) == 2:
        return float(target[0]), float(target[1])
    if isinstance(target, dict):
        start = target.get("start")
        end = target.get("end")
        if isinstance(start, (list, tuple)) and len(start) == 2 and isinstance(end, (list, tuple)) and len(end) == 2:
            return (
                (float(start[0]) + float(end[0])) / 2.0,
                (float(start[1]) + float(end[1])) / 2.0,
            )
    return None


def get_drag_positions(step: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    action = step.get("action", {})
    target = action.get("target", {}).get("position", {})
    start = target.get("start")
    end = target.get("end")
    if isinstance(start, (list, tuple)) and len(start) == 2:
        start_pos = (float(start[0]), float(start[1]))
    else:
        start_pos = None
    if isinstance(end, (list, tuple)) and len(end) == 2:
        end_pos = (float(end[0]), float(end[1]))
    else:
        end_pos = None
    return start_pos, end_pos


def get_step_start_time(step: Dict[str, Any]) -> Optional[datetime]:
    return parse_time(step.get("now_state", {}).get("screenshot_time_before"))


def get_step_end_time(step: Dict[str, Any]) -> Optional[datetime]:
    return parse_time(step.get("now_state", {}).get("screenshot_time_after"))


def normalize_text_token(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def get_step_titles(step: Dict[str, Any]) -> Tuple[str, str]:
    now_state = step.get("now_state", {})
    before = normalize_text_token(now_state.get("app_title_before") or "")
    after = normalize_text_token(now_state.get("app_title_after") or "")
    return before, after


def infer_target_app_keywords(report_path: Path, report: Dict[str, Any], steps: Sequence[Dict[str, Any]]) -> List[str]:
    candidates: List[str] = []

    for raw in (report.get("app"), report.get("task_category"), report.get("task_title"), report.get("instruction")):
        if isinstance(raw, str) and raw.strip():
            candidates.append(normalize_text_token(raw))

    for part in report_path.parts:
        token = normalize_text_token(part)
        if token:
            candidates.append(token)

    for step in steps:
        before, after = get_step_titles(step)
        if before and before not in GENERIC_APP_TITLES and not any(
            normalize_text_token(keyword) in before for keyword in TERMINAL_KEYWORDS
        ):
            candidates.append(before)
        if after and after not in GENERIC_APP_TITLES and not any(
            normalize_text_token(keyword) in after for keyword in TERMINAL_KEYWORDS
        ):
            candidates.append(after)

    keywords: List[str] = []
    for candidate in candidates:
        for alias_group in KNOWN_APP_ALIASES.values():
            for alias in alias_group:
                if alias in candidate and alias not in keywords:
                    keywords.append(alias)

    if not keywords:
        for candidate in candidates:
            if "chrome" in candidate and "chrome" not in keywords:
                keywords.append("chrome")
            if "libreoffice" in candidate and "libreoffice" not in keywords:
                keywords.append("libreoffice")
            if "impress" in candidate and "impress" not in keywords:
                keywords.append("impress")
            if "terminal" in candidate and "terminal" not in keywords:
                keywords.append("terminal")
    return keywords


def classify_app_relation(title: str, target_keywords: Sequence[str]) -> str:
    normalized = normalize_text_token(title)
    if normalized in GENERIC_APP_TITLES:
        return "unknown"
    if any(normalize_text_token(keyword) in normalized for keyword in TERMINAL_KEYWORDS):
        return "terminal"
    if any(normalize_text_token(keyword) in normalized for keyword in target_keywords):
        return "target"
    return "other"


def drop_unrelated_head_tail_steps(
    steps: List[Dict[str, Any]],
    keep: List[bool],
    audit: List[Optional[Dict[str, Any]]],
    features: List[StepFeatures],
    comparator: "ImageComparator",
    verifier: Optional[Dict[str, Any]],
    target_keywords: Sequence[str],
) -> None:
    if not target_keywords:
        return

    # Trim prefix steps until the trajectory clearly starts operating inside the target app.
    for idx, step in enumerate(steps):
        before_title, after_title = get_step_titles(step)
        before_relation = classify_app_relation(before_title, target_keywords)
        after_relation = classify_app_relation(after_title, target_keywords)
        if before_relation == "target":
            break
        if before_relation in {"terminal", "other"}:
            maybe_drop_step(
                keep=keep,
                audit=audit,
                idx=idx,
                rule_id="rule16_head_tail_unrelated_app_trim",
                reason="Leading step belongs to Terminal or another non-target app before the trajectory enters the target app.",
                features=features,
                steps=steps,
                comparator=comparator,
                verifier=verifier,
                extra={
                    "app_title_before": before_title or None,
                    "app_title_after": after_title or None,
                    "target_app_keywords": list(target_keywords),
                },
            )
            if after_relation == "target":
                break
            continue
        break

    # Trim suffix steps after the trajectory has already left the target app.
    for idx in range(len(steps) - 1, -1, -1):
        if not keep[idx]:
            continue
        step = steps[idx]
        before_title, after_title = get_step_titles(step)
        before_relation = classify_app_relation(before_title, target_keywords)
        after_relation = classify_app_relation(after_title, target_keywords)
        if after_relation == "target":
            break
        if after_relation in {"terminal", "other"}:
            maybe_drop_step(
                keep=keep,
                audit=audit,
                idx=idx,
                rule_id="rule16_head_tail_unrelated_app_trim",
                reason="Trailing step belongs to Terminal or another non-target app after the trajectory has already left the target app.",
                features=features,
                steps=steps,
                comparator=comparator,
                verifier=verifier,
                extra={
                    "app_title_before": before_title or None,
                    "app_title_after": after_title or None,
                    "target_app_keywords": list(target_keywords),
                },
            )
            if before_relation == "target":
                break
            continue
        break


@dataclass
class SimilarityMetrics:
    similarity: float
    method: str

    @property
    def as_dict(self) -> Dict[str, Any]:
        return {
            "similarity": round(self.similarity, 6),
            "method": self.method,
        }


@dataclass
class StepFeatures:
    global_similarity: Optional[SimilarityMetrics] = None
    unchanged_global: bool = False
    protected_focus_click: bool = False
    protected_blur_deselect_click: bool = False
    notes: List[str] = field(default_factory=list)
    vlm_verifications: List[Dict[str, Any]] = field(default_factory=list)


def serialize_feature(feature: StepFeatures) -> Dict[str, Any]:
    return {
        "global_similarity": feature.global_similarity.as_dict if feature.global_similarity else None,
        "unchanged_global": feature.unchanged_global,
        "protected_focus_click": feature.protected_focus_click,
        "protected_blur_deselect_click": feature.protected_blur_deselect_click,
        "notes": list(feature.notes),
        "vlm_verifications": list(feature.vlm_verifications),
    }


def deserialize_feature(payload: Dict[str, Any]) -> StepFeatures:
    similarity_payload = payload.get("global_similarity")
    global_similarity = None
    if isinstance(similarity_payload, dict):
        global_similarity = SimilarityMetrics(
            similarity=float(similarity_payload.get("similarity", 0.0)),
            method=str(similarity_payload.get("method", "")),
        )
    return StepFeatures(
        global_similarity=global_similarity,
        unchanged_global=bool(payload.get("unchanged_global", False)),
        protected_focus_click=bool(payload.get("protected_focus_click", False)),
        protected_blur_deselect_click=bool(payload.get("protected_blur_deselect_click", False)),
        notes=list(payload.get("notes", [])),
        vlm_verifications=list(payload.get("vlm_verifications", [])),
    )


def default_feature_progress(total_steps: int) -> Dict[str, Any]:
    return {
        "completed_step_indices": [],
        "features": [None for _ in range(total_steps)],
    }


def load_resume_audit(audit_path: Path, total_steps: int) -> Optional[Dict[str, Any]]:
    if not audit_path.is_file():
        return None

    audit = load_json(audit_path)
    summary = audit.get("summary", {})
    if int(summary.get("original_step_count", -1)) != total_steps:
        return None
    return audit


def is_completed_audit(audit: Dict[str, Any]) -> bool:
    summary = audit.get("summary", {})
    if summary.get("status") == "completed":
        return True
    return "kept_step_count" in summary and "dropped_step_count" in summary


def build_resume_audit(
    report_path: Path,
    total_steps: int,
    feature_progress: Dict[str, Any],
    phase: str,
    processing_started_at: str,
) -> Dict[str, Any]:
    completed_count = len(feature_progress.get("completed_step_indices", []))
    return {
        "summary": {
            "status": "in_progress",
            "phase": phase,
            "report_path": str(report_path.resolve()),
            "original_step_count": total_steps,
            "completed_feature_step_count": completed_count,
            "processing_started_at": processing_started_at,
            "updated_at": current_timestamp(),
        },
        "feature_progress": feature_progress,
        "dropped_steps": [],
        "kept_steps": [],
    }


class ImageComparator:
    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path

    def resolve_screenshot(self, rel_path: str) -> Path:
        candidate_parents = [
            self.report_path.parent,
            self.report_path.parent.parent,
            self.report_path.parent.parent.parent if self.report_path.parent.parent.parent != self.report_path.parent.parent else self.report_path.parent.parent,
        ]
        rel = Path(rel_path)
        for parent in candidate_parents:
            candidate = (parent / rel).resolve()
            if candidate.exists():
                return candidate
        if rel.exists():
            return rel.resolve()
        raise FileNotFoundError(f"Cannot resolve screenshot path: {rel_path}")

    def try_resolve_screenshot(self, rel_path: Optional[str]) -> Optional[Path]:
        if not rel_path:
            return None
        try:
            return self.resolve_screenshot(rel_path)
        except FileNotFoundError:
            return None

    def compare_full(self, before_path: Path, after_path: Path) -> SimilarityMetrics:
        return SimilarityMetrics(
            similarity=compare_screenshot_similarity(before_path, after_path),
            method="screenshot_similarity",
        )


def collect_features(
    steps: Sequence[Dict[str, Any]],
    comparator: ImageComparator,
    args: argparse.Namespace,
    audit_path: Path,
    report_path: Path,
    processing_started_at: str,
    resume_audit: Optional[Dict[str, Any]] = None,
) -> List[StepFeatures]:
    feature_progress = default_feature_progress(len(steps))
    if resume_audit:
        saved_progress = resume_audit.get("feature_progress", {})
        saved_features = saved_progress.get("features", [])
        if isinstance(saved_features, list) and len(saved_features) == len(steps):
            feature_progress["features"] = saved_features
            feature_progress["completed_step_indices"] = list(
                saved_progress.get("completed_step_indices", [])
            )

    completed_indices = {
        int(index)
        for index in feature_progress.get("completed_step_indices", [])
        if isinstance(index, int) or str(index).isdigit()
    }
    features: List[StepFeatures] = []
    processed_since_save = 0

    for index, step in enumerate(steps):
        saved_feature = feature_progress["features"][index]
        if index in completed_indices and isinstance(saved_feature, dict):
            features.append(deserialize_feature(saved_feature))
            continue

        now_state = step.get("now_state", {})
        before_rel = now_state.get("screenshot_path_before")
        after_rel = now_state.get("screenshot_path_after")
        feature = StepFeatures()
        if before_rel and after_rel:
            before_path = comparator.try_resolve_screenshot(before_rel)
            after_path = comparator.try_resolve_screenshot(after_rel)
            if before_path is None or after_path is None:
                missing_paths = []
                if before_path is None:
                    missing_paths.append(before_rel)
                if after_path is None:
                    missing_paths.append(after_rel)
                feature.notes.append(
                    "Skipped global similarity because screenshot file is missing: "
                    + ", ".join(str(path) for path in missing_paths)
                )
            else:
                feature.global_similarity = comparator.compare_full(before_path, after_path)
                feature.unchanged_global = feature.global_similarity.similarity >= args.full_similarity_threshold
        features.append(feature)
        completed_indices.add(index)
        feature_progress["completed_step_indices"] = sorted(completed_indices)
        feature_progress["features"][index] = serialize_feature(feature)
        processed_since_save += 1

        if args.save_every > 0 and processed_since_save % args.save_every == 0:
            atomic_dump_json(
                audit_path,
                build_resume_audit(
                    report_path=report_path,
                    total_steps=len(steps),
                    feature_progress=feature_progress,
                    phase="collect_features",
                    processing_started_at=processing_started_at,
                ),
            )
            log_progress(
                "denoise_feature_checkpoint_saved",
                report_path=str(report_path),
                completed_step_count=len(completed_indices),
                total_step_count=len(steps),
            )

    atomic_dump_json(
        audit_path,
        build_resume_audit(
            report_path=report_path,
            total_steps=len(steps),
            feature_progress=feature_progress,
            phase="collect_features_done",
            processing_started_at=processing_started_at,
        ),
    )
    mark_focus_click_protection(steps, features, args)
    mark_blur_deselect_click_protection(steps, features, args)
    return features


def is_click_outside_previous_target(
    step: Dict[str, Any],
    previous_step: Dict[str, Any],
    radius: float,
) -> bool:
    pos = get_target_position(step)
    previous_pos = get_target_position(previous_step)
    if pos is None:
        return False
    if previous_pos is None:
        return True
    return euclidean(pos, previous_pos) > radius


def mark_blur_deselect_click_protection(
    steps: Sequence[Dict[str, Any]],
    features: List[StepFeatures],
    args: argparse.Namespace,
) -> None:
    for i in range(1, len(steps)):
        if not is_click_like_mouse_action(steps[i]):
            continue
        if not features[i].unchanged_global:
            continue

        previous_step = steps[i - 1]
        if not is_editing_or_selection_step(previous_step):
            continue
        if not is_click_outside_previous_target(steps[i], previous_step, args.same_region_radius * 1.5):
            continue

        # 保护编辑或选中状态后的外部点击：这类点击常用于失焦、提交 blur 或取消选中，整屏变化通常很小。
        features[i].protected_blur_deselect_click = True
        features[i].notes.append(
            f"Protected as an outside click after {get_action_type(previous_step)} that may blur, commit, or deselect."
        )


def mark_focus_click_protection(
    steps: Sequence[Dict[str, Any]],
    features: List[StepFeatures],
    args: argparse.Namespace,
) -> None:
    n = len(steps)
    i = 0
    while i < n:
        if not is_click_like_mouse_action(steps[i]):
            i += 1
            continue
        cluster = [i]
        anchor = get_target_position(steps[i])
        j = i + 1
        while j < n and is_click_like_mouse_action(steps[j]):
            pos = get_target_position(steps[j])
            if anchor is None or pos is None or euclidean(anchor, pos) > args.same_region_radius:
                break
            cluster.append(j)
            j += 1
        next_typing_idx = None
        next_pos = None
        for k in range(j, min(n, j + args.typing_lookahead_steps + 1)):
            if get_action_type(steps[k]) == "typing":
                next_typing_idx = k
                next_pos = get_target_position(steps[k])
                break
            if get_action_type(steps[k]) not in {"click", "press"}:
                break
        if (
            next_typing_idx is not None
            and anchor is not None
            and next_pos is not None
            and euclidean(anchor, next_pos) <= args.same_region_radius * 1.5
            and all(features[idx].unchanged_global for idx in cluster)
        ):
            features[cluster[0]].protected_focus_click = True
            features[cluster[0]].notes.append(
                "Protected as the first focus or selection-establishing mouse action before nearby typing."
            )
        i = j


def drop_step(
    keep: List[bool],
    audit: List[Optional[Dict[str, Any]]],
    idx: int,
    rule_id: str,
    reason: str,
    features: StepFeatures,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not keep[idx]:
        return
    payload = {
        "rule_id": rule_id,
        "reason": reason,
        "global_similarity": features.global_similarity.as_dict if features.global_similarity else None,
        "protected_focus_click": features.protected_focus_click,
        "protected_blur_deselect_click": features.protected_blur_deselect_click,
    }
    if extra:
        payload.update(extra)
    keep[idx] = False
    audit[idx] = payload


def mutate_step(
    steps: List[Dict[str, Any]],
    mutations: Dict[int, Dict[str, Any]],
    idx: int,
    patch: Dict[str, Any],
    reason: str,
) -> None:
    action = steps[idx].setdefault("action", {})
    action_param = action.setdefault("param", {})
    action_param.update(patch)
    mutations[idx] = {"patch": patch, "reason": reason}


def find_previous_kept_index(keep: Sequence[bool], idx: int) -> Optional[int]:
    for j in range(idx - 1, -1, -1):
        if keep[j]:
            return j
    return None


def time_gap_seconds(prev_step: Dict[str, Any], next_step: Dict[str, Any]) -> Optional[float]:
    prev_end = get_step_end_time(prev_step) or get_step_start_time(prev_step)
    next_start = get_step_start_time(next_step) or get_step_end_time(next_step)
    if prev_end is None or next_start is None:
        return None
    return (next_start - prev_end).total_seconds()


def scroll_direction(step: Dict[str, Any]) -> Optional[str]:
    action = step.get("action", {})
    if action.get("type") != "scroll":
        return None
    direction = str(action.get("param", {}).get("type", "")).lower()
    if direction in {"up", "down", "left", "right"}:
        return direction
    delta = action.get("param", {}).get("delta")
    if isinstance(delta, (int, float)):
        return "down" if delta < 0 else "up"
    return None


def similar_regions(
    step_a: Dict[str, Any],
    step_b: Dict[str, Any],
    radius: float,
) -> bool:
    pos_a = get_target_position(step_a)
    pos_b = get_target_position(step_b)
    return pos_a is not None and pos_b is not None and euclidean(pos_a, pos_b) <= radius


RISKY_DROP_RULE_IDS = {
    "rule1_invalid_click_drop",
    "rule1b_invalid_press_drop",
    "rule4_multiclick_drop",
    "rule5_reverse_scroll_cancel",
    "rule7_focus_confirmation_drop",
    "rule9_isolated_modifier_drop",
    "rule10_invalid_drag_drop",
    "rule11_repeated_click_compress",
}


def maybe_drop_step(
    *,
    keep: List[bool],
    audit: List[Optional[Dict[str, Any]]],
    idx: int,
    rule_id: str,
    reason: str,
    features: List[StepFeatures],
    steps: List[Dict[str, Any]],
    comparator: ImageComparator,
    verifier: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not keep[idx]:
        return

    feature = features[idx]
    verification_payload: Optional[Dict[str, Any]] = None
    if verifier is not None and rule_id in RISKY_DROP_RULE_IDS:
        step = steps[idx]
        now_state = step.get("now_state", {})
        before_raw_rel = now_state.get("screenshot_path_before_raw")
        after_rel = now_state.get("screenshot_path_after")
        before_path = (
            comparator.try_resolve_screenshot(before_raw_rel)
            or comparator.try_resolve_screenshot(now_state.get("screenshot_path_before"))
        )
        after_path = comparator.try_resolve_screenshot(after_rel)
        if before_path is None or after_path is None:
            feature.notes.append(
                f"Similarity gate skipped for {rule_id}: missing before/after screenshots."
            )
            return
        try:
            numeric_similarity = compare_screenshot_similarity(before_path, after_path)
            log_progress(
                "before_after_similarity",
                step_index=idx,
                step_id=step.get("step_id"),
                rule_id=rule_id,
                similarity=round(numeric_similarity, 6),
            )
        except Exception as exc:
            feature.notes.append(
                f"Similarity gate skipped for {rule_id}: {type(exc).__name__}: {exc}"
            )
            return
        verification_payload = {
            "before_after_similarity": round(numeric_similarity, 6),
        }
        if numeric_similarity <= VLM_PREFILTER_SIMILARITY_THRESHOLD:
            keep_reason = (
                f"Current before/after similarity is {numeric_similarity:.4f}, "
                f"which does not exceed the threshold {VLM_PREFILTER_SIMILARITY_THRESHOLD:.4f}."
            )
            feature.notes.append(
                f"Similarity gate kept step for {rule_id}: {keep_reason}"
            )
            return

        next_idx = idx + 1
        if next_idx >= len(steps):
            keep_reason = "No next step exists for after-to-next-before verification."
            feature.notes.append(
                f"Similarity gate kept step for {rule_id}: {keep_reason}"
            )
            return

        next_before_rel = steps[next_idx].get("now_state", {}).get("screenshot_path_before")
        next_before_path = comparator.resolve_screenshot(next_before_rel) if next_before_rel else None
        
        if next_before_path is None:
            keep_reason = "Missing next step before screenshot."
            feature.notes.append(
                f"Similarity gate kept step for {rule_id}: {keep_reason}"
            )
            return

        try:
            next_similarity = compare_screenshot_similarity(after_path, next_before_path)
            log_progress(
                "after_next_before_similarity",
                step_index=idx,
                step_id=step.get("step_id"),
                next_step_id=steps[next_idx].get("step_id"),
                rule_id=rule_id,
                similarity=round(next_similarity, 6),
            )
        except Exception as exc:
            feature.notes.append(
                f"Similarity gate skipped next-step check for {rule_id}: {type(exc).__name__}: {exc}"
            )
            return

        verification_payload["after_next_before_similarity"] = round(next_similarity, 6)
        if next_similarity <= VLM_PREFILTER_SIMILARITY_THRESHOLD:
            keep_reason = (
                f"Current after/next before similarity is {next_similarity:.4f}, "
                f"which does not exceed the threshold {VLM_PREFILTER_SIMILARITY_THRESHOLD:.4f}."
            )
            feature.notes.append(
                f"Similarity gate kept step for {rule_id}: {keep_reason}"
            )
            return

        vlm_attempts: List[Dict[str, Any]] = []
        yes_count = 0
        model_before_rel = get_model_before_rel(step)
        model_next_before_rel = get_model_before_rel(steps[next_idx])
        model_before_path = comparator.resolve_screenshot(model_before_rel) if model_before_rel else None
        model_next_before_path = (
            comparator.resolve_screenshot(model_next_before_rel)
            if model_next_before_rel
            else None
        )
        if model_before_path is None or model_next_before_path is None:
            feature.notes.append(
                f"VLM similarity verification skipped for {rule_id}: missing raw before screenshot."
            )
            return
        for attempt_idx in range(DEFAULT_VLM_VERIFICATION_REPEATS):
            try:
                vlm_result = judge_action(
                    get_action_type(step),
                    model_before_path,
                    after_path,
                    model_next_before_path,
                    **verifier["judge_kwargs"],
                )
                log_progress(
                    "vlm_judgment",
                    step_index=idx,
                    step_id=step.get("step_id"),
                    rule_id=rule_id,
                    attempt=attempt_idx + 1,
                    result=vlm_result,
                )
            except Exception as exc:
                if verifier.get("strict", False):
                    raise
                feature.notes.append(
                    f"VLM similarity verification skipped for {rule_id}: {type(exc).__name__}: {exc}"
                )
                return

            judgment = str(vlm_result.get("judgment", vlm_result.get("Judgment", "no"))).strip().lower()
            reason_text = str(vlm_result.get("reason", vlm_result.get("Reason", ""))).strip()
            if judgment == "yes":
                yes_count += 1
            vlm_attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "Judgment": judgment,
                    "Reason": reason_text or "No reason provided by the VLM.",
                }
            )

        approved_drop = yes_count == DEFAULT_VLM_VERIFICATION_REPEATS
        first_reject_reason = next(
            (item["Reason"] for item in vlm_attempts if item["Judgment"] != "yes"),
            "",
        )
        verification_payload = {
            **verification_payload,
            "Judgment": "yes" if approved_drop else "no",
            "Reason": (
                f"All {DEFAULT_VLM_VERIFICATION_REPEATS} VLM verifications returned yes."
                if approved_drop
                else first_reject_reason
                or f"Only {yes_count}/{DEFAULT_VLM_VERIFICATION_REPEATS} VLM verifications returned yes."
            ),
            "yes_count": yes_count,
            "required_yes_count": DEFAULT_VLM_VERIFICATION_REPEATS,
            "attempts": vlm_attempts,
        }
        feature.vlm_verifications.append(
            {
                "rule_id": rule_id,
                **verification_payload,
            }
        )
        if not approved_drop:
            feature.notes.append(
                f"Similarity gate kept step for {rule_id}: {verification_payload['Reason']}"
            )
            return

    payload = dict(extra or {})
    if verification_payload is not None:
        payload["vlm_verification"] = verification_payload
    drop_step(
        keep,
        audit,
        idx,
        rule_id,
        reason,
        feature,
        payload or None,
    )


def apply_rules(
    steps: List[Dict[str, Any]],
    features: List[StepFeatures],
    args: argparse.Namespace,
    comparator: ImageComparator,
    verifier: Optional[Dict[str, Any]],
    target_keywords: Sequence[str],
) -> Tuple[List[bool], List[Optional[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    n = len(steps)
    keep = [True] * n
    audit: List[Optional[Dict[str, Any]]] = [None] * n
    mutations: Dict[int, Dict[str, Any]] = {}

    # Rule 16: trim obvious head/tail steps that belong to Terminal or another non-target app.
    drop_unrelated_head_tail_steps(steps, keep, audit, features, comparator, verifier, target_keywords)

    # Rule 2: Ctrl+Z undoes the previous kept step.
    for i, step in enumerate(steps):
        if not is_undo_press(step):
            continue
        prev_idx = find_previous_kept_index(keep, i)
        if prev_idx is None:
            continue
        drop_step(
            keep,
            audit,
            prev_idx,
            "rule2_fast_undo_cancel",
            f"{steps[i].get('step_id')} is Ctrl+Z and cancels the previous step.",
            features[prev_idx],
            {"paired_with_step_id": step.get("step_id")},
        )
        drop_step(
            keep,
            audit,
            i,
            "rule2_fast_undo_cancel",
            f"Ctrl+Z cancels previous kept step {steps[prev_idx].get('step_id')}.",
            features[i],
            {"paired_with_step_id": steps[prev_idx].get("step_id")},
        )

    # Rule 5: opposite-direction scroll pair cancellation.
    for i in range(n - 1):
        if not keep[i]:
            continue
        step_a = steps[i]
        if get_action_type(step_a) != "scroll":
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            step_b = steps[j]
            if get_action_type(step_b) != "scroll":
                break
            gap = time_gap_seconds(step_a, step_b)
            if gap is not None and gap > args.scroll_cancel_gap_seconds:
                break
            dir_a = scroll_direction(step_a)
            dir_b = scroll_direction(step_b)
            if dir_a is None or dir_b is None or dir_a == dir_b:
                continue
            sim_a_before_to_b_after = None
            before_rel = step_a.get("now_state", {}).get("screenshot_path_before")
            after_rel = step_b.get("now_state", {}).get("screenshot_path_after")
            if before_rel and after_rel:
                sim_a_before_to_b_after = comparator.compare_full(
                    comparator.resolve_screenshot(before_rel),
                    comparator.resolve_screenshot(after_rel),
                )
            if sim_a_before_to_b_after is not None and sim_a_before_to_b_after.similarity >= args.full_similarity_threshold:
                maybe_drop_step(
                    keep=keep,
                    audit=audit,
                    idx=i,
                    rule_id="rule5_reverse_scroll_cancel",
                    reason=f"Opposite-direction scroll pair returns close to the pre-scroll view with {steps[j].get('step_id')}.",
                    features=features,
                    steps=steps,
                    comparator=comparator,
                    verifier=verifier,
                    extra={
                        "paired_with_step_id": steps[j].get("step_id"),
                        "pair_roundtrip_similarity": sim_a_before_to_b_after.as_dict,
                    },
                )
                maybe_drop_step(
                    keep=keep,
                    audit=audit,
                    idx=j,
                    rule_id="rule5_reverse_scroll_cancel",
                    reason=f"Opposite-direction scroll pair returns close to the pre-scroll view with {steps[i].get('step_id')}.",
                    features=features,
                    steps=steps,
                    comparator=comparator,
                    verifier=verifier,
                    extra={
                        "paired_with_step_id": steps[i].get("step_id"),
                        "pair_roundtrip_similarity": sim_a_before_to_b_after.as_dict,
                    },
                )
            break

    # Rule 9: isolated modifier key with no visible effect.
    for i, step in enumerate(steps):
        if not keep[i]:
            continue
        if is_modifier_only_press(step) and features[i].unchanged_global:
            press_list = step.get("action", {}).get("param", {}).get("press_list", [])
            maybe_drop_step(
                keep=keep,
                audit=audit,
                idx=i,
                rule_id="rule9_isolated_modifier_drop",
                reason=f"Modifier-only press {press_list} has no visible effect.",
                features=features,
                steps=steps,
                comparator=comparator,
                verifier=verifier,
            )

    # 对 Backspace/Delete/Tab 等编辑或导航按键，只有画面持续近似不变并经 VLM 确认后才删除。
    for i, step in enumerate(steps):
        if not keep[i] or not is_potentially_ineffective_press(step):
            continue
        if not features[i].unchanged_global:
            continue
        press_list = list(normalize_press_list(step.get("action", {}).get("param", {}).get("press_list", [])))
        maybe_drop_step(
            keep=keep,
            audit=audit,
            idx=i,
            rule_id="rule1b_invalid_press_drop",
            reason=(
                f"Press {press_list} dropped because compare_screenshot_similarity exceeded "
                f"{VLM_PREFILTER_SIMILARITY_THRESHOLD:.4f} and VLM confirmed it was redundant."
            ),
            features=features,
            steps=steps,
            comparator=comparator,
            verifier=verifier,
            extra={
                "press_list": press_list,
                "primary_key": get_primary_press_key(step),
                "press_count": step.get("action", {}).get("param", {}).get("press_count"),
            },
        )

    # Rule 10: tiny or visually ineffective drag.
    for i, step in enumerate(steps):
        if not keep[i] or get_action_type(step) != "drag_to":
            continue
        distance = step.get("action", {}).get("param", {}).get("distance")
        start_pos, end_pos = get_drag_positions(step)
        if distance is None and start_pos and end_pos:
            distance = euclidean(start_pos, end_pos)
        tiny_drag = isinstance(distance, (int, float)) and float(distance) < args.small_drag_threshold
        visually_ineffective = features[i].unchanged_global
        if tiny_drag or visually_ineffective:
            maybe_drop_step(
                keep=keep,
                audit=audit,
                idx=i,
                rule_id="rule10_invalid_drag_drop",
                reason="Drag is tiny or does not create visible local/global change.",
                features=features,
                steps=steps,
                comparator=comparator,
                verifier=verifier,
                extra={"drag_distance": distance},
            )

    # Rule 11 + focus-confirmation protection:
    # compress repeated same-region clicks, keeping the first focus click before typing.
    i = 0
    while i < n:
        if not keep[i] or get_action_type(steps[i]) != "click":
            i += 1
            continue
        cluster = [i]
        j = i + 1
        while j < n and keep[j] and get_action_type(steps[j]) == "click" and similar_regions(
            steps[i], steps[j], args.same_region_radius
        ):
            cluster.append(j)
            j += 1
        if len(cluster) >= 2:
            changed_flags = [not features[idx].unchanged_global for idx in cluster]
            if any(changed_flags):
                first_changed_local = next(idx for idx, changed in zip(cluster, changed_flags) if changed)
                for idx in cluster:
                    if idx < first_changed_local and features[idx].unchanged_global:
                        maybe_drop_step(
                            keep=keep,
                            audit=audit,
                            idx=idx,
                            rule_id="rule11_repeated_click_compress",
                            reason=f"Earlier same-region click compressed into later effective click {steps[first_changed_local].get('step_id')}.",
                            features=features,
                            steps=steps,
                            comparator=comparator,
                            verifier=verifier,
                            extra={"kept_step_id": steps[first_changed_local].get("step_id")},
                        )
            elif any(features[idx].protected_focus_click for idx in cluster):
                protected_idx = next(idx for idx in cluster if features[idx].protected_focus_click)
                for idx in cluster:
                    if idx != protected_idx:
                        maybe_drop_step(
                            keep=keep,
                            audit=audit,
                            idx=idx,
                            rule_id="rule7_focus_confirmation_drop",
                            reason=f"Same-region confirmation click compressed after focus is already established by {steps[protected_idx].get('step_id')}.",
                            features=features,
                            steps=steps,
                            comparator=comparator,
                            verifier=verifier,
                            extra={"kept_step_id": steps[protected_idx].get("step_id")},
                        )
        i = j

    # Rule 4: degrade or drop visually ineffective multi-click.
    for i, step in enumerate(steps):
        if not keep[i] or get_action_type(step) != "click":
            continue
        num_click = get_click_count(step)
        if num_click <= 1 or not features[i].unchanged_global:
            continue
        nearby_typing = False
        pos = get_target_position(step)
        for j in range(i + 1, min(n, i + args.typing_lookahead_steps + 2)):
            if get_action_type(steps[j]) == "typing":
                next_pos = get_target_position(steps[j])
                if pos is not None and next_pos is not None and euclidean(pos, next_pos) <= args.same_region_radius * 1.5:
                    nearby_typing = True
                break
        if nearby_typing:
            mutate_step(
                steps,
                mutations,
                i,
                {"num_click": 1},
                "rule4_multiclick_degrade",
            )
        else:
            maybe_drop_step(
                keep=keep,
                audit=audit,
                idx=i,
                rule_id="rule4_multiclick_drop",
                reason="Multi-click has no visible effect and no nearby typing benefit.",
                features=features,
                steps=steps,
                comparator=comparator,
                verifier=verifier,
            )

    # Rule 1: invalid click-like mouse action with almost no visible change.
    # The similarity gate first checks current before/after, then current after/next before.
    # Only when both are above 0.98 does it ask the VLM whether the action is redundant.
    for i, step in enumerate(steps):
        if not keep[i] or not is_click_like_mouse_action(step):
            continue
        if features[i].protected_focus_click:
            continue
        if features[i].protected_blur_deselect_click:
            continue
        maybe_drop_step(
            keep=keep,
            audit=audit,
            idx=i,
            rule_id="rule1_invalid_click_drop",
            reason=(
                "Click-like mouse action dropped because compare_screenshot_similarity exceeded "
                f"{VLM_PREFILTER_SIMILARITY_THRESHOLD:.4f} and VLM confirmed high similarity."
            ),
            features=features,
            steps=steps,
            comparator=comparator,
            verifier=verifier,
        )

    return keep, audit, mutations


def build_outputs(
    report: Dict[str, Any],
    steps: List[Dict[str, Any]],
    keep: Sequence[bool],
    audit: Sequence[Optional[Dict[str, Any]]],
    mutations: Dict[int, Dict[str, Any]],
    features: Sequence[StepFeatures],
    verifier: Optional[Dict[str, Any]],
    inferred_target_keywords: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kept_indices = [i for i in range(len(steps)) if keep[i]]
    kept_steps = [copy.deepcopy(steps[i]) for i in kept_indices]
    step_id_mapping: Dict[str, str] = {}
    for new_index, (original_index, step_payload) in enumerate(zip(kept_indices, kept_steps), start=1):
        original_step_id = str(steps[original_index].get("step_id", "")).strip()
        new_step_id = f"s{new_index}"
        if original_step_id:
            step_id_mapping[original_step_id] = new_step_id
        step_payload["step_id"] = new_step_id

    counts = Counter()
    dropped_records = []
    for idx, record in enumerate(audit):
        if record is None:
            continue
        counts[record["rule_id"]] += 1
        dropped_records.append(
            {
                "index": idx,
                "step_id": steps[idx].get("step_id"),
                "action_type": get_action_type(steps[idx]),
                "vlm_verifications": list(features[idx].vlm_verifications),
                **record,
            }
        )

    denoised_report = copy.deepcopy(report)
    denoised_report["steps"] = kept_steps
    global_similarity_method = next(
        (feature.global_similarity.method for feature in features if feature.global_similarity is not None),
        None,
    )
    denoised_report["denoise_meta"] = {
        "status": "completed",
        "original_step_count": len(steps),
        "kept_step_count": len(kept_steps),
        "dropped_step_count": len(steps) - len(kept_steps),
        "dropped_by_rule": dict(counts),
        "global_similarity_method": global_similarity_method,
        "vlm_verifier_model": None if verifier is None else verifier["model"],
        "vlm_prefilter_similarity_threshold": VLM_PREFILTER_SIMILARITY_THRESHOLD,
        "inferred_target_app_keywords": list(inferred_target_keywords),
        "mutated_steps": {
            step_id_mapping.get(str(steps[idx].get("step_id", "")).strip(), steps[idx].get("step_id")): {
                **payload,
                "original_step_id": steps[idx].get("step_id"),
            }
            for idx, payload in mutations.items()
        },
        "step_id_mapping": step_id_mapping,
    }

    audit_report = {
        "summary": denoised_report["denoise_meta"],
        "dropped_steps": dropped_records,
        "kept_steps": [
            {
                "index": idx,
                "step_id": step_id_mapping.get(str(steps[idx].get("step_id", "")).strip(), steps[idx].get("step_id")),
                "original_step_id": steps[idx].get("step_id"),
                "action_type": get_action_type(steps[idx]),
                "protected_focus_click": features[idx].protected_focus_click,
                "protected_blur_deselect_click": features[idx].protected_blur_deselect_click,
                "global_similarity": features[idx].global_similarity.as_dict
                if features[idx].global_similarity
                else None,
                "vlm_verifications": list(features[idx].vlm_verifications),
                "notes": list(features[idx].notes),
            }
            for idx in range(len(steps))
            if keep[idx]
        ],
    }
    return denoised_report, audit_report


def build_vlm_verifier(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.vlm_verify_backend == "none":
        return None

    if args.vlm_verify_backend == "kimi":
        verifier_model = args.kimi_model
    elif args.vlm_verify_backend == "ark":
        verifier_model = args.ark_model
    elif args.vlm_verify_backend == "qwen":
        verifier_model = args.qwen_model
    else:
        verifier_model = f"auto(kimi:{args.kimi_model}, ark:{args.ark_model}, qwen:{args.qwen_model})"

    return {
        "strict": args.vlm_verify_backend in {"kimi", "ark", "qwen"},
        "model": verifier_model,
        "judge_kwargs": {
            "backend": args.vlm_verify_backend,
            "http_trust_env": args.vlm_http_trust_env,
            "kimi_api_key": args.kimi_api_key,
            "kimi_base_url": args.kimi_base_url,
            "kimi_model": args.kimi_model,
            "kimi_max_tokens": args.kimi_max_tokens,
            "ark_api_key": args.ark_api_key,
            "ark_base_url": args.ark_base_url,
            "ark_model": args.ark_model,
            "qwen_api_key": args.qwen_api_key,
            "qwen_base_url": args.qwen_base_url,
            "qwen_model": args.qwen_model
        },
    }


def process_report(report_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    processing_started_at = current_timestamp()
    processing_started_perf = time.perf_counter()
    log_progress("denoise_started", report_path=str(report_path))
    output_json = Path(args.output_json) if args.output_json else report_path.with_name("report_denoised.json")
    output_audit = Path(args.output_audit) if args.output_audit else report_path.with_name("report_denoise_audit.json")
    report = load_json(report_path)
    steps = copy.deepcopy(report.get("steps", []))
    resume_audit = None if args.no_resume else load_resume_audit(output_audit, len(steps))

    if resume_audit and is_completed_audit(resume_audit) and output_json.is_file():
        summary = dict(resume_audit.get("summary", {}))
        summary["resume_skipped"] = True
        log_progress(
            "denoise_resume_skipped_completed",
            report_path=str(report_path),
            output_json=str(output_json.resolve()),
            output_audit=str(output_audit.resolve()),
        )
        return summary

    comparator = ImageComparator(report_path)
    verifier = build_vlm_verifier(args)
    features = collect_features(
        steps,
        comparator,
        args,
        audit_path=output_audit,
        report_path=report_path,
        processing_started_at=processing_started_at,
        resume_audit=resume_audit,
    )
    inferred_target_keywords = infer_target_app_keywords(report_path, report, steps)
    keep, audit, mutations = apply_rules(steps, features, args, comparator, verifier, inferred_target_keywords)
    denoised_report, audit_report = build_outputs(
        report,
        steps,
        keep,
        audit,
        mutations,
        features,
        verifier,
        inferred_target_keywords,
    )

    processing_finished_at = current_timestamp()
    processing_elapsed_seconds = round(time.perf_counter() - processing_started_perf, 3)
    processing_meta = {
        "processing_started_at": processing_started_at,
        "processing_finished_at": processing_finished_at,
        "processing_elapsed_seconds": processing_elapsed_seconds,
    }
    denoised_report["denoise_meta"].update(processing_meta)
    audit_report["summary"].update(processing_meta)

    atomic_dump_json(output_json, denoised_report)
    atomic_dump_json(output_audit, audit_report)
    log_progress(
        "denoise_finished",
        report_path=str(report_path),
        output_json=str(output_json.resolve()),
        output_audit=str(output_audit.resolve()),
        elapsed_seconds=processing_elapsed_seconds,
    )

    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "output_json": str(output_json.resolve()),
                "output_audit": str(output_audit.resolve()),
                **denoised_report["denoise_meta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return denoised_report["denoise_meta"]


def process_batch_report(report_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    try:
        meta = process_report(report_path, args)
        return {
            "report_path": str(report_path),
            "meta": meta,
            "error": None,
        }
    except Exception as exc:
        return {
            "report_path": str(report_path),
            "meta": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    args = parse_args()
    if args.save_every < 0:
        raise ValueError("--save-every cannot be less than 0")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.batch and args.workers != 1:
        raise ValueError("--workers can only be greater than 1 when --batch is enabled")
    if args.batch and (args.output_json or args.output_audit):
        raise ValueError("--output-json and --output-audit cannot be used with --batch")

    if not args.batch:
        report_path = resolve_report_path(args.input_path, args.report_subdir)
        process_report(report_path, args)
        return 0

    report_paths = find_batch_report_paths(args.input_path, args.report_subdir)
    if not report_paths:
        raise FileNotFoundError(
            f"No report.json found under {Path(args.input_path).resolve()} "
            f"with report_subdir={args.report_subdir!r}"
        )

    log_progress(
        "batch_denoise_started",
        input_path=str(Path(args.input_path).resolve()),
        report_subdir=args.report_subdir,
        report_count=len(report_paths),
        workers=min(args.workers, len(report_paths)),
    )
    failed: List[Dict[str, str]] = []
    total_original_steps = 0
    total_kept_steps = 0
    total_dropped_steps = 0
    skipped_completed_count = 0

    def collect_batch_result(index: int, report_path: Path, result: Dict[str, Any]) -> None:
        nonlocal total_original_steps, total_kept_steps, total_dropped_steps, skipped_completed_count
        error = result.get("error")
        if error:
            failed.append({"report_path": str(report_path), "error": str(error)})
            log_progress(
                "batch_denoise_item_failed",
                index=index,
                total=len(report_paths),
                report_path=str(report_path),
                error=str(error),
            )
            return

        meta = result["meta"]
        if meta.get("resume_skipped"):
            skipped_completed_count += 1
        total_original_steps += int(meta.get("original_step_count", 0))
        total_kept_steps += int(meta.get("kept_step_count", 0))
        total_dropped_steps += int(meta.get("dropped_step_count", 0))

    if args.workers == 1:
        for index, report_path in enumerate(report_paths, start=1):
            log_progress(
                "batch_denoise_item_started",
                index=index,
                total=len(report_paths),
                report_path=str(report_path),
            )
            collect_batch_result(index, report_path, process_batch_report(report_path, args))
    else:
        max_workers = min(args.workers, len(report_paths))
        executor = ProcessPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            future_to_item = {}
            for index, report_path in enumerate(report_paths, start=1):
                log_progress(
                    "batch_denoise_item_submitted",
                    index=index,
                    total=len(report_paths),
                    report_path=str(report_path),
                )
                future = executor.submit(process_batch_report, report_path, args)
                future_to_item[future] = (index, report_path)

            for future in as_completed(future_to_item):
                index, report_path = future_to_item[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "report_path": str(report_path),
                        "meta": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                collect_batch_result(index, report_path, result)
        except KeyboardInterrupt:
            interrupted = True
            for future in future_to_item:
                future.cancel()
            pending_count = sum(1 for future in future_to_item if not future.done())
            executor.shutdown(wait=False, cancel_futures=True)
            log_progress(
                "batch_denoise_interrupted",
                report_count=len(report_paths),
                failed_count=len(failed),
                pending_count=pending_count,
                skipped_completed_count=skipped_completed_count,
                total_original_steps=total_original_steps,
                total_kept_steps=total_kept_steps,
                total_dropped_steps=total_dropped_steps,
            )
            return 130
        finally:
            if not interrupted:
                executor.shutdown()

    log_progress(
        "batch_denoise_finished",
        report_count=len(report_paths),
        failed_count=len(failed),
        skipped_completed_count=skipped_completed_count,
        total_original_steps=total_original_steps,
        total_kept_steps=total_kept_steps,
        total_dropped_steps=total_dropped_steps,
    )
    print(
        json.dumps(
            {
                "report_count": len(report_paths),
                "failed_count": len(failed),
                "skipped_completed_count": skipped_completed_count,
                "total_original_steps": total_original_steps,
                "total_kept_steps": total_kept_steps,
                "total_dropped_steps": total_dropped_steps,
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
