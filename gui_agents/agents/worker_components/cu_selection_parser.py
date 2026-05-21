# CU 选择结果解析模块，从 planner 输出里抽取结构化 CU 选择结果

import ast
import re
from typing import Any, Dict, Optional


def _normalize_cu_id(raw_text: str) -> Optional[str]:
    text = (raw_text or "").strip()
    if not text:
        return None

    match = re.search(r"\bcu_\d+\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(0).lower()


def _normalize_path_id(raw_text: str) -> Optional[str]:
    text = (raw_text or "").strip()
    if not text:
        return None

    match = re.search(r"\bpath_\d+\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(0).lower()


def parse_reference_params(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        return {}
    try:
        value = ast.literal_eval(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {}


def extract_cu_selection_section(plan: str) -> str:
    section_match = re.search(
        r"\(CU Selection\)(.*?)(?:\n\([A-Za-z ].*?\)|\Z)",
        plan,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        return ""
    return section_match.group(1).strip()


def parse_cu_selection_from_plan(plan: str) -> Dict[str, Any]:
    selected_cu = None
    selected_path = None
    reference_params: Dict[str, Any] = {}

    section = extract_cu_selection_section(plan)
    source_text = section or plan

    cu_match = re.search(r"Selected CU:\s*(.+)", source_text, flags=re.IGNORECASE)
    if cu_match:
        value = cu_match.group(1).strip()
        if value and value.upper() != "NONE":
            selected_cu = _normalize_cu_id(value)

    path_match = re.search(r"Selected Path:\s*(.+)", source_text, flags=re.IGNORECASE)
    if path_match:
        value = path_match.group(1).strip()
        if value and value.upper() != "NONE":
            selected_path = _normalize_path_id(value)

    params_block_match = re.search(
        r"Reference Params:\s*(\{.*?\})",
        source_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if params_block_match:
        reference_params = parse_reference_params(params_block_match.group(1))
    else:
        params_line_match = re.search(
            r"Reference Params:\s*(\{.*?\})", source_text, flags=re.IGNORECASE
        )
        if params_line_match:
            reference_params = parse_reference_params(params_line_match.group(1))

    return {
        "selected_cu_id": selected_cu,
        "selected_path_id": selected_path,
        "reference_params": reference_params,
    }
 

def validate_cu_selection(
    parsed_selection: Dict[str, Any], retrieval_result: Any
) -> Dict[str, Any]:
    selected_cu_id = parsed_selection.get("selected_cu_id")
    selected_path_id = parsed_selection.get("selected_path_id")
    reference_params = dict(parsed_selection.get("reference_params", {}))

    if not selected_cu_id:
        return {
            "selected_cu_id": None,
            "selected_path_id": None,
            "reference_params": {},
            "selection_source": None,
            "valid": False,
        }

    candidate_map = {
        candidate.cu_id: candidate for candidate in retrieval_result.merged_candidates
    }
    candidate = candidate_map.get(selected_cu_id)
    if candidate is None:
        return {
            "selected_cu_id": None,
            "selected_path_id": None,
            "reference_params": {},
            "selection_source": None,
            "valid": False,
        }

    path_stats = dict(candidate.canonical_unit.get("path_stats", {}))
    if selected_path_id not in path_stats:
        if len(path_stats) == 1:
            selected_path_id = next(iter(path_stats))
        else:
            selected_path_id = None

    return {
        "selected_cu_id": selected_cu_id,
        "selected_path_id": selected_path_id,
        "reference_params": reference_params,
        "selection_source": candidate.source,
        "valid": True,
    }
