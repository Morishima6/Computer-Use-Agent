from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


DEFAULT_ORIGINAL_REPORT_NAME = "report.json"
DEFAULT_DENOISED_REPORT_NAME = "report_denoised.json"
DEFAULT_REPAIR_AUDIT_NAME = "report_denoised_repair_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore wrongly dropped original step_id values back into report_denoised.json "
            "using the original report.json as the source of truth."
        )
    )
    parser.add_argument(
        "input_path",
        help="A result directory, report.json, or report_denoised.json.",
    )
    parser.add_argument(
        "--restore-steps",
        required=True,
        help="Original step ids to restore, e.g. s24,s112 or [s24,s112].",
    )
    parser.add_argument(
        "--original-report",
        default=None,
        help="Path to the original report.json. Defaults to report.json beside/in the input directory.",
    )
    parser.add_argument(
        "--denoised-report",
        default=None,
        help="Path to report_denoised.json. Defaults to report_denoised.json beside/in the input directory.",
    )
    parser.add_argument(
        "--denoise-audit",
        default=None,
        help="Optional report_denoise_audit.json path, used to update dropped_by_rule when restored steps are known.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write the repaired report. Defaults to overwriting report_denoised.json.",
    )
    parser.add_argument(
        "--repair-audit",
        default=None,
        help=f"Repair audit path. Defaults to {DEFAULT_REPAIR_AUDIT_NAME} beside report_denoised.json.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak_<timestamp> copy when overwriting report_denoised.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes and write nothing.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_backup(path: Path, *, enabled: bool) -> Optional[Path]:
    if not enabled or not path.is_file():
        return None
    backup_path = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def normalize_step_id(raw_value: Any) -> Optional[str]:
    text = str(raw_value).strip().strip("'\"")
    if not text:
        return None
    if text.lower().startswith("s"):
        text = text[1:]
    if not text.isdigit():
        return None
    number = int(text)
    if number <= 0:
        return None
    return f"s{number}"


def parse_step_id_list(raw_value: str) -> List[str]:
    text = raw_value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    step_ids: List[str] = []
    for chunk in re.split(r"[,，\s]+", text):
        step_id = normalize_step_id(chunk)
        if step_id and step_id not in step_ids:
            step_ids.append(step_id)
    return step_ids


def step_number(step_id: str) -> int:
    normalized = normalize_step_id(step_id)
    if normalized is None:
        return 10**12
    return int(normalized[1:])


def step_id_of(step: Dict[str, Any]) -> str:
    return str(step.get("step_id", "")).strip()


def resolve_report_paths(
    input_path: str,
    original_report_arg: Optional[str],
    denoised_report_arg: Optional[str],
    output_json_arg: Optional[str],
    repair_audit_arg: Optional[str],
) -> Tuple[Path, Path, Path, Path]:
    path = Path(input_path).expanduser().resolve()

    if path.is_dir():
        result_dir = path
        default_original = result_dir / DEFAULT_ORIGINAL_REPORT_NAME
        default_denoised = result_dir / DEFAULT_DENOISED_REPORT_NAME
    elif path.is_file():
        if path.name == DEFAULT_ORIGINAL_REPORT_NAME:
            default_original = path
            default_denoised = path.with_name(DEFAULT_DENOISED_REPORT_NAME)
        else:
            default_denoised = path
            default_original = path.with_name(DEFAULT_ORIGINAL_REPORT_NAME)
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    original_report = (
        Path(original_report_arg).expanduser().resolve()
        if original_report_arg
        else default_original.resolve()
    )
    denoised_report = (
        Path(denoised_report_arg).expanduser().resolve()
        if denoised_report_arg
        else default_denoised.resolve()
    )
    output_json = (
        Path(output_json_arg).expanduser().resolve()
        if output_json_arg
        else denoised_report
    )
    repair_audit = (
        Path(repair_audit_arg).expanduser().resolve()
        if repair_audit_arg
        else denoised_report.with_name(DEFAULT_REPAIR_AUDIT_NAME)
    )

    if not original_report.is_file():
        raise FileNotFoundError(f"Original report not found: {original_report}")
    if not denoised_report.is_file():
        raise FileNotFoundError(f"Denoised report not found: {denoised_report}")
    return original_report, denoised_report, output_json, repair_audit


def screenshot_before_key(step: Dict[str, Any]) -> Optional[str]:
    now_state = step.get("now_state", {})
    if not isinstance(now_state, dict):
        return None
    raw = now_state.get("screenshot_path_before")
    return str(raw) if raw else None


def infer_step_id_mapping(
    original_steps: Sequence[Dict[str, Any]],
    denoised_steps: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    original_by_before: Dict[str, List[str]] = {}
    for step in original_steps:
        key = screenshot_before_key(step)
        if key:
            original_by_before.setdefault(key, []).append(step_id_of(step))

    mapping: Dict[str, str] = {}
    for step in denoised_steps:
        key = screenshot_before_key(step)
        if not key:
            continue
        original_ids = original_by_before.get(key, [])
        if len(original_ids) == 1:
            mapping[original_ids[0]] = step_id_of(step)
    return mapping


def build_existing_original_to_step(
    original_steps: Sequence[Dict[str, Any]],
    denoised_report: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    denoised_steps = denoised_report.get("steps", [])
    denoised_by_current_id = {step_id_of(step): step for step in denoised_steps if step_id_of(step)}

    meta_mapping = denoised_report.get("denoise_meta", {}).get("step_id_mapping", {})
    if not isinstance(meta_mapping, dict) or not meta_mapping:
        meta_mapping = infer_step_id_mapping(original_steps, denoised_steps)

    existing: Dict[str, Dict[str, Any]] = {}
    for original_id, denoised_id in meta_mapping.items():
        normalized_original_id = normalize_step_id(original_id)
        if normalized_original_id is None:
            continue
        step = denoised_by_current_id.get(str(denoised_id).strip())
        if step is not None:
            existing[normalized_original_id] = step
    return existing


def update_denoise_meta(
    repaired_report: Dict[str, Any],
    original_step_count: int,
    restored_step_ids: Sequence[str],
    step_id_mapping: Dict[str, str],
    restored_rule_by_step_id: Dict[str, str],
) -> None:
    meta = repaired_report.setdefault("denoise_meta", {})
    meta["original_step_count"] = original_step_count
    meta["kept_step_count"] = len(repaired_report.get("steps", []))
    meta["dropped_step_count"] = original_step_count - len(repaired_report.get("steps", []))
    meta["step_id_mapping"] = step_id_mapping

    dropped_by_rule = meta.get("dropped_by_rule")
    if isinstance(dropped_by_rule, dict):
        counts = Counter({str(rule_id): int(count) for rule_id, count in dropped_by_rule.items()})
        for step_id in restored_step_ids:
            rule_id = restored_rule_by_step_id.get(step_id)
            if rule_id:
                counts[rule_id] -= 1
        meta["dropped_by_rule"] = {
            rule_id: count
            for rule_id, count in counts.items()
            if count > 0
        }

    meta.setdefault("repair_history", [])
    if isinstance(meta["repair_history"], list):
        meta["repair_history"].append(
            {
                "repaired_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "restored_original_step_ids": list(restored_step_ids),
                "restored_rule_by_step_id": dict(restored_rule_by_step_id),
            }
        )


def repair_report(
    original_report: Dict[str, Any],
    denoised_report: Dict[str, Any],
    restore_step_ids: Sequence[str],
    restored_rule_by_step_id: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    restored_rule_by_step_id = restored_rule_by_step_id or {}
    original_steps = original_report.get("steps", [])
    denoised_steps = denoised_report.get("steps", [])
    if not isinstance(original_steps, list) or not isinstance(denoised_steps, list):
        raise ValueError("Both original and denoised reports must contain a list field named 'steps'.")

    original_by_id = {step_id_of(step): step for step in original_steps if step_id_of(step)}
    requested_ids = list(restore_step_ids)
    missing_requested = [step_id for step_id in requested_ids if step_id not in original_by_id]

    existing_by_original_id = build_existing_original_to_step(original_steps, denoised_report)
    existing_ids: Set[str] = set(existing_by_original_id)
    restore_set = {step_id for step_id in requested_ids if step_id in original_by_id}
    already_present = sorted(restore_set & existing_ids, key=step_number)
    restored_ids = sorted(restore_set - existing_ids, key=step_number)
    final_original_ids = existing_ids | restore_set

    repaired_steps: List[Dict[str, Any]] = []
    step_id_mapping: Dict[str, str] = {}
    restored_records: List[Dict[str, Any]] = []

    for original_step in original_steps:
        original_id = step_id_of(original_step)
        if original_id not in final_original_ids:
            continue

        if original_id in existing_by_original_id:
            step_payload = copy.deepcopy(existing_by_original_id[original_id])
            source = "denoised"
        else:
            step_payload = copy.deepcopy(original_step)
            source = "original"

        new_step_id = f"s{len(repaired_steps) + 1}"
        old_step_id = step_payload.get("step_id")
        step_payload["step_id"] = new_step_id
        repaired_steps.append(step_payload)
        step_id_mapping[original_id] = new_step_id

        if original_id in restored_ids:
            restored_records.append(
                {
                    "original_step_id": original_id,
                    "new_step_id": new_step_id,
                    "source": source,
                    "previous_step_id_in_source": old_step_id,
                }
            )

    repaired_report = copy.deepcopy(denoised_report)
    repaired_report["steps"] = repaired_steps
    update_denoise_meta(
        repaired_report,
        len(original_steps),
        restored_ids,
        step_id_mapping,
        restored_rule_by_step_id,
    )

    audit = {
        "repair_summary": {
            "original_step_count": len(original_steps),
            "input_denoised_step_count": len(denoised_steps),
            "output_step_count": len(repaired_steps),
            "requested_restore_step_ids": requested_ids,
            "restored_step_ids": restored_ids,
            "already_present_step_ids": already_present,
            "missing_requested_step_ids": missing_requested,
            "restored_rule_by_step_id": {
                step_id: restored_rule_by_step_id[step_id]
                for step_id in restored_ids
                if step_id in restored_rule_by_step_id
            },
        },
        "restored_steps": restored_records,
        "dropped_original_step_ids_after_repair": [
            step_id_of(step)
            for step in original_steps
            if step_id_of(step) and step_id_of(step) not in final_original_ids
        ],
        "output_step_id_mapping": step_id_mapping,
    }
    return repaired_report, audit


def load_dropped_rule_by_step_id(audit_path: Path) -> Dict[str, str]:
    if not audit_path.is_file():
        return {}
    audit = load_json(audit_path)
    dropped_steps = audit.get("dropped_steps", [])
    if not isinstance(dropped_steps, list):
        return {}

    rule_by_step_id: Dict[str, str] = {}
    for record in dropped_steps:
        if not isinstance(record, dict):
            continue
        step_id = normalize_step_id(record.get("step_id"))
        rule_id = str(record.get("rule_id", "")).strip()
        if step_id and rule_id:
            rule_by_step_id[step_id] = rule_id
    return rule_by_step_id


def main() -> None:
    args = parse_args()
    restore_step_ids = parse_step_id_list(args.restore_steps)
    if not restore_step_ids:
        raise ValueError("No valid --restore-steps values were provided.")

    original_path, denoised_path, output_path, repair_audit_path = resolve_report_paths(
        args.input_path,
        args.original_report,
        args.denoised_report,
        args.output_json,
        args.repair_audit,
    )
    denoise_audit_path = (
        Path(args.denoise_audit).expanduser().resolve()
        if args.denoise_audit
        else denoised_path.with_name("report_denoise_audit.json")
    )

    log(f"Original report: {original_path}")
    log(f"Denoised report: {denoised_path}")
    if denoise_audit_path.is_file():
        log(f"Denoise audit: {denoise_audit_path}")
    else:
        log(f"Denoise audit not found; dropped_by_rule will not be adjusted by rule id: {denoise_audit_path}")
    log(f"Restore original step ids: {', '.join(restore_step_ids)}")

    original_report = load_json(original_path)
    denoised_report = load_json(denoised_path)
    dropped_rule_by_step_id = load_dropped_rule_by_step_id(denoise_audit_path)
    repaired_report, audit = repair_report(
        original_report,
        denoised_report,
        restore_step_ids,
        dropped_rule_by_step_id,
    )

    summary = audit["repair_summary"]
    log(
        "Planned repair: "
        f"restore={summary['restored_step_ids']}, "
        f"already_present={summary['already_present_step_ids']}, "
        f"missing={summary['missing_requested_step_ids']}, "
        f"output_steps={summary['output_step_count']}"
    )

    if args.dry_run:
        log("Dry run enabled; no files were written.")
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    backup_path = maybe_backup(output_path, enabled=(not args.no_backup and output_path == denoised_path))
    if backup_path:
        log(f"Backup created: {backup_path}")

    dump_json(output_path, repaired_report)
    dump_json(repair_audit_path, audit)
    log(f"Repaired report saved to: {output_path}")
    log(f"Repair audit saved to: {repair_audit_path}")


if __name__ == "__main__":
    main()
