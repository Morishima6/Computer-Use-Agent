from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_SOURCE_DIRNAME = "segments_units"
DEFAULT_DENOISED_DIRNAME = "segments_units_denoised"
DEFAULT_REPAIR_AUDIT_FILENAME = "segments_units_denoise_repair_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair denoised segments_units JSON files by restoring or removing specified step_id values "
            "against the original segments_units directory."
        )
    )
    parser.add_argument(
        "input_path",
        help=(
            "A result/session directory, a segments_units directory, a segments_units_denoised directory, "
            "or one denoised segment JSON file."
        ),
    )
    parser.add_argument(
        "--restore-steps",
        default="",
        help="Step ids to restore from original segments_units, e.g. s12,s13 or [s12,s13].",
    )
    parser.add_argument(
        "--remove-steps",
        default="",
        help="Step ids to remove from denoised output, e.g. s8,s9 or [s8,s9].",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Original segments_units directory. Defaults to sibling/result segments_units.",
    )
    parser.add_argument(
        "--denoised-dir",
        default=None,
        help="Denoised segments_units_denoised directory. Defaults to sibling/result segments_units_denoised.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write repaired segment files. Defaults to overwriting the denoised directory.",
    )
    parser.add_argument(
        "--repair-audit",
        default=None,
        help="Repair audit JSON path. Defaults to <result>/segments_units_denoise_repair_audit.json.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak_<timestamp> copies when overwriting denoised segment files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes and write nothing.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    key: List[Any] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return tuple(key)


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
    text = str(raw_value).strip()
    if not text:
        return None
    if text.lower().startswith("s"):
        number_text = text[1:]
    else:
        number_text = text
    if not number_text.isdigit():
        return None
    number = int(number_text)
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
        cleaned = chunk.strip().strip("'\"")
        if not cleaned:
            continue
        step_id = normalize_step_id(cleaned)
        if step_id and step_id not in step_ids:
            step_ids.append(step_id)
    return step_ids


def iter_segment_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.json"), key=natural_sort_key):
        if path.is_file():
            yield path


def resolve_dirs(
    input_path: str,
    source_dir_arg: Optional[str],
    denoised_dir_arg: Optional[str],
    output_dir_arg: Optional[str],
    repair_audit_arg: Optional[str],
) -> Tuple[Path, Path, Path, Path, Optional[str]]:
    path = Path(input_path).expanduser().resolve()
    target_segment_name: Optional[str] = None

    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Input file is not a JSON segment file: {path}")
        target_segment_name = path.name
        denoised_dir = path.parent
        if denoised_dir.name == DEFAULT_DENOISED_DIRNAME:
            result_dir = denoised_dir.parent
        else:
            result_dir = denoised_dir.parent
    elif path.is_dir():
        if path.name == DEFAULT_SOURCE_DIRNAME:
            result_dir = path.parent
            denoised_dir = result_dir / DEFAULT_DENOISED_DIRNAME
        elif path.name == DEFAULT_DENOISED_DIRNAME:
            result_dir = path.parent
            denoised_dir = path
        elif (path / DEFAULT_SOURCE_DIRNAME).is_dir():
            result_dir = path
            denoised_dir = path / DEFAULT_DENOISED_DIRNAME
        elif (path / "result" / DEFAULT_SOURCE_DIRNAME).is_dir():
            result_dir = path / "result"
            denoised_dir = result_dir / DEFAULT_DENOISED_DIRNAME
        else:
            raise FileNotFoundError(
                f"Could not resolve result/segments_units directories from input path: {path}"
            )
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    source_dir = Path(source_dir_arg).expanduser().resolve() if source_dir_arg else result_dir / DEFAULT_SOURCE_DIRNAME
    denoised_dir = Path(denoised_dir_arg).expanduser().resolve() if denoised_dir_arg else denoised_dir.resolve()
    output_dir = Path(output_dir_arg).expanduser().resolve() if output_dir_arg else denoised_dir
    repair_audit = (
        Path(repair_audit_arg).expanduser().resolve()
        if repair_audit_arg
        else result_dir / DEFAULT_REPAIR_AUDIT_FILENAME
    )

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Original segments_units directory not found: {source_dir}")
    if not denoised_dir.is_dir():
        raise FileNotFoundError(f"Denoised segments_units_denoised directory not found: {denoised_dir}")

    return source_dir.resolve(), denoised_dir.resolve(), output_dir.resolve(), repair_audit.resolve(), target_segment_name


def step_id_of(step: Dict[str, Any]) -> str:
    return str(step.get("step_id", "")).strip()


def coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower().startswith("s"):
        text = text[1:]
    if not text.isdigit():
        return None
    return int(text)


def unit_step_ids(unit: Dict[str, Any], original_steps: Sequence[Dict[str, Any]]) -> List[str]:
    step_ids: List[str] = []
    raw_indices = unit.get("step_indices") or []
    if not isinstance(raw_indices, list):
        return step_ids

    for raw_index in raw_indices:
        step_index = coerce_int(raw_index)
        if step_index is None or step_index < 1 or step_index > len(original_steps):
            continue
        step_id = step_id_of(original_steps[step_index - 1])
        if step_id:
            step_ids.append(step_id)
    return step_ids


def update_unit_denoise_meta(unit: Dict[str, Any], original_unit_step_ids: Sequence[str], kept_step_ids: Sequence[str]) -> None:
    meta = unit.get("unit_denoise_meta")
    if not isinstance(meta, dict):
        return
    kept_set = set(kept_step_ids)
    chosen_step_ids = [step_id for step_id in original_unit_step_ids if step_id not in kept_set]
    meta["chosen_step_ids"] = chosen_step_ids
    meta["chosen_all"] = False
    if chosen_step_ids:
        meta["reason"] = str(meta.get("reason") or "Manual repair kept this unit with selected steps removed.")
    else:
        meta["reason"] = str(meta.get("reason") or "Manual repair restored all steps in this unit.")


def build_repaired_segment(
    *,
    original_payload: Dict[str, Any],
    denoised_payload: Dict[str, Any],
    restore_step_ids: Set[str],
    remove_step_ids: Set[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    original_steps = [step for step in original_payload.get("steps", []) if isinstance(step, dict)]
    denoised_steps = [step for step in denoised_payload.get("steps", []) if isinstance(step, dict)]
    original_units = [unit for unit in original_payload.get("units", []) if isinstance(unit, dict)]
    denoised_units = [unit for unit in denoised_payload.get("units", []) if isinstance(unit, dict)]

    original_step_by_id = {step_id_of(step): step for step in original_steps if step_id_of(step)}
    denoised_step_by_id = {step_id_of(step): step for step in denoised_steps if step_id_of(step)}
    original_step_ids = [step_id_of(step) for step in original_steps if step_id_of(step)]
    current_kept_ids = {step_id_of(step) for step in denoised_steps if step_id_of(step)}

    restorable_ids = {step_id for step_id in restore_step_ids if step_id in original_step_by_id}
    removable_ids = {step_id for step_id in remove_step_ids if step_id in original_step_by_id or step_id in denoised_step_by_id}

    desired_kept_ids = set(current_kept_ids)
    desired_kept_ids.update(restorable_ids)
    desired_kept_ids.difference_update(removable_ids)

    final_steps: List[Dict[str, Any]] = []
    restored_ids: List[str] = []
    removed_ids: List[str] = []
    for step_id in original_step_ids:
        if step_id not in desired_kept_ids:
            continue
        if step_id in denoised_step_by_id:
            final_steps.append(copy.deepcopy(denoised_step_by_id[step_id]))
        else:
            final_steps.append(copy.deepcopy(original_step_by_id[step_id]))
            restored_ids.append(step_id)

    for step_id in original_step_ids:
        if step_id in current_kept_ids and step_id not in desired_kept_ids:
            removed_ids.append(step_id)

    step_id_to_new_index = {
        step_id_of(step): index
        for index, step in enumerate(final_steps, start=1)
        if step_id_of(step)
    }
    denoised_unit_by_id = {
        str(unit.get("unit_id", "")).strip(): unit
        for unit in denoised_units
        if str(unit.get("unit_id", "")).strip()
    }

    final_units: List[Dict[str, Any]] = []
    restored_unit_ids: List[str] = []
    removed_unit_ids: List[str] = []
    changed_unit_ids: List[str] = []
    for original_unit in original_units:
        unit_id = str(original_unit.get("unit_id", "")).strip()
        original_unit_step_ids = unit_step_ids(original_unit, original_steps)
        kept_unit_step_ids = [step_id for step_id in original_unit_step_ids if step_id in step_id_to_new_index]
        if not kept_unit_step_ids:
            if unit_id and unit_id in denoised_unit_by_id:
                removed_unit_ids.append(unit_id)
            continue

        denoised_unit = denoised_unit_by_id.get(unit_id)
        if denoised_unit is not None:
            unit_out = copy.deepcopy(denoised_unit)
        else:
            unit_out = copy.deepcopy(original_unit)
            if unit_id:
                restored_unit_ids.append(unit_id)

        unit_out["step_indices"] = [step_id_to_new_index[step_id] for step_id in kept_unit_step_ids]
        update_unit_denoise_meta(unit_out, original_unit_step_ids, kept_unit_step_ids)

        unit_restore_ids = [step_id for step_id in restored_ids if step_id in original_unit_step_ids]
        unit_remove_ids = [step_id for step_id in removed_ids if step_id in original_unit_step_ids]
        if unit_restore_ids or unit_remove_ids:
            unit_out["unit_denoise_repair_meta"] = {
                "restored_step_ids": unit_restore_ids,
                "removed_step_ids": unit_remove_ids,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if unit_id:
                changed_unit_ids.append(unit_id)

        final_units.append(unit_out)

    repaired_payload = copy.deepcopy(denoised_payload)
    repaired_payload["steps"] = final_steps
    repaired_payload["units"] = final_units

    current_meta = repaired_payload.get("segment_denoise_meta")
    if not isinstance(current_meta, dict):
        current_meta = {}
    removed_step_ids_now = [step_id for step_id in original_step_ids if step_id not in step_id_to_new_index]
    repaired_payload["segment_denoise_meta"] = {
        **current_meta,
        "chosen_step_ids": removed_step_ids_now,
        "original_step_count": len(original_steps),
        "kept_step_count": len(final_steps),
        "removed_step_count": len(removed_step_ids_now),
        "original_unit_count": len(original_units),
        "kept_unit_count": len(final_units),
        "removed_unit_count": len(original_units) - len(final_units),
    }
    repaired_payload["segment_denoise_repair_meta"] = {
        "restored_step_ids": restored_ids,
        "removed_step_ids": removed_ids,
        "restored_unit_ids": restored_unit_ids,
        "removed_unit_ids": removed_unit_ids,
        "changed_unit_ids": changed_unit_ids,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    stats = {
        "before_step_count": len(denoised_steps),
        "after_step_count": len(final_steps),
        "before_unit_count": len(denoised_units),
        "after_unit_count": len(final_units),
        "restored_step_ids": restored_ids,
        "removed_step_ids": removed_ids,
        "restore_step_ids_not_found": sorted(restore_step_ids - set(original_step_by_id)),
        "remove_step_ids_not_found": sorted(
            step_id for step_id in remove_step_ids if step_id not in original_step_by_id and step_id not in denoised_step_by_id
        ),
        "restored_unit_ids": restored_unit_ids,
        "removed_unit_ids": removed_unit_ids,
        "changed_unit_ids": changed_unit_ids,
    }
    return repaired_payload, stats


def should_process_segment(
    original_payload: Dict[str, Any],
    denoised_payload: Dict[str, Any],
    restore_step_ids: Set[str],
    remove_step_ids: Set[str],
) -> bool:
    original_ids = {step_id_of(step) for step in original_payload.get("steps", []) if isinstance(step, dict)}
    denoised_ids = {step_id_of(step) for step in denoised_payload.get("steps", []) if isinstance(step, dict)}
    return bool((restore_step_ids & original_ids) or (remove_step_ids & (original_ids | denoised_ids)))


def write_complete_output_dir(
    *,
    source_dir: Path,
    denoised_dir: Path,
    output_dir: Path,
    repaired_by_name: Dict[str, Dict[str, Any]],
    dry_run: bool,
) -> None:
    if output_dir == denoised_dir:
        return
    for source_file in iter_segment_files(source_dir):
        output_path = output_dir / source_file.name
        if source_file.name in repaired_by_name:
            payload = repaired_by_name[source_file.name]
        else:
            denoised_file = denoised_dir / source_file.name
            if not denoised_file.is_file():
                continue
            payload = load_json(denoised_file)
        if not dry_run:
            dump_json(output_path, payload)


def repair_segments(
    *,
    source_dir: Path,
    denoised_dir: Path,
    output_dir: Path,
    repair_audit_path: Path,
    target_segment_name: Optional[str],
    restore_step_ids: Set[str],
    remove_step_ids: Set[str],
    backup: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    repaired_by_name: Dict[str, Dict[str, Any]] = {}
    segment_records: List[Dict[str, Any]] = []

    source_files = list(iter_segment_files(source_dir))
    if target_segment_name:
        source_files = [path for path in source_files if path.name == target_segment_name]
        if not source_files:
            raise FileNotFoundError(f"Target segment not found in source dir: {target_segment_name}")

    for source_file in source_files:
        denoised_file = denoised_dir / source_file.name
        if not denoised_file.is_file():
            continue

        original_payload = load_json(source_file)
        denoised_payload = load_json(denoised_file)
        if not should_process_segment(original_payload, denoised_payload, restore_step_ids, remove_step_ids):
            continue

        repaired_payload, stats = build_repaired_segment(
            original_payload=original_payload,
            denoised_payload=denoised_payload,
            restore_step_ids=restore_step_ids,
            remove_step_ids=remove_step_ids,
        )
        changed = (
            stats["restored_step_ids"]
            or stats["removed_step_ids"]
            or stats["restored_unit_ids"]
            or stats["removed_unit_ids"]
        )
        if not changed:
            segment_records.append(
                {
                    "segment_file": source_file.name,
                    "status": "unchanged",
                    **stats,
                }
            )
            continue

        output_path = output_dir / source_file.name
        backup_path = None
        if not dry_run and output_dir == denoised_dir:
            backup_path = maybe_backup(output_path, enabled=backup)
        if not dry_run:
            dump_json(output_path, repaired_payload)
        repaired_by_name[source_file.name] = repaired_payload

        segment_records.append(
            {
                "segment_file": source_file.name,
                "status": "dry_run" if dry_run else "repaired",
                "output_path": str(output_path),
                "backup_path": str(backup_path) if backup_path else None,
                **stats,
            }
        )

        log(
            f"{source_file.name}: restore={stats['restored_step_ids']} remove={stats['removed_step_ids']} "
            f"steps {stats['before_step_count']}->{stats['after_step_count']} "
            f"units {stats['before_unit_count']}->{stats['after_unit_count']}"
        )

    write_complete_output_dir(
        source_dir=source_dir,
        denoised_dir=denoised_dir,
        output_dir=output_dir,
        repaired_by_name=repaired_by_name,
        dry_run=dry_run,
    )

    audit_payload = {
        "summary": {
            "source_dir": str(source_dir),
            "denoised_dir": str(denoised_dir),
            "output_dir": str(output_dir),
            "target_segment_name": target_segment_name,
            "requested_restore_step_ids": sorted(restore_step_ids, key=lambda item: int(item[1:])),
            "requested_remove_step_ids": sorted(remove_step_ids, key=lambda item: int(item[1:])),
            "processed_segment_count": len(segment_records),
            "repaired_segment_count": len([record for record in segment_records if record.get("status") in {"repaired", "dry_run"}]),
            "dry_run": dry_run,
            "updated_at": timestamp,
        },
        "segments": segment_records,
    }
    if not dry_run:
        dump_json(repair_audit_path, audit_payload)
    return audit_payload


def main() -> None:
    args = parse_args()
    restore_step_ids = set(parse_step_id_list(args.restore_steps))
    remove_step_ids = set(parse_step_id_list(args.remove_steps))
    if not restore_step_ids and not remove_step_ids:
        raise ValueError("No step ids provided. Use --restore-steps and/or --remove-steps.")
    conflict_ids = restore_step_ids & remove_step_ids
    if conflict_ids:
        raise ValueError(f"Step ids cannot be both restored and removed: {sorted(conflict_ids)}")

    source_dir, denoised_dir, output_dir, repair_audit_path, target_segment_name = resolve_dirs(
        args.input_path,
        args.source_dir,
        args.denoised_dir,
        args.output_dir,
        args.repair_audit,
    )
    log(
        f"resolved source_dir={source_dir} denoised_dir={denoised_dir} "
        f"output_dir={output_dir} repair_audit={repair_audit_path}"
    )

    audit_payload = repair_segments(
        source_dir=source_dir,
        denoised_dir=denoised_dir,
        output_dir=output_dir,
        repair_audit_path=repair_audit_path,
        target_segment_name=target_segment_name,
        restore_step_ids=restore_step_ids,
        remove_step_ids=remove_step_ids,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )
    summary = audit_payload["summary"]
    log(
        f"done processed_segments={summary['processed_segment_count']} "
        f"repaired_segments={summary['repaired_segment_count']} dry_run={args.dry_run}"
    )
    if not args.dry_run:
        log(f"repair audit written: {repair_audit_path}")


if __name__ == "__main__":
    main()
