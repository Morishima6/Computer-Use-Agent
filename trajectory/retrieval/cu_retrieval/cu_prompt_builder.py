from __future__ import annotations

from typing import Any, Dict, List

from .schemas import MergedCandidate, RetrievalResult


def _format_parameter_defs(parameter_defs: List[Dict[str, Any]]) -> str:
    if not parameter_defs:
        return "none"

    lines: List[str] = []
    for param in parameter_defs:
        name = str(param.get("param_name", "unknown"))
        param_type = str(param.get("param_type", "unknown"))
        observed_values = list(param.get("observed_values", []))[:3]
        lines.append(
            f"{name} ({param_type}), examples={observed_values if observed_values else '[]'}"
        )
    return "; ".join(lines)


def _format_paths(path_stats: Dict[str, Dict[str, Any]]) -> str:
    if not path_stats:
        return "none"

    lines: List[str] = []
    ranked_paths = sorted(
        path_stats.items(),
        key=lambda item: (
            -int(item[1].get("path_count", 0) or 0)
        ),
    )
    for path_id, payload in ranked_paths:
        steps = payload.get("steps", []) or []
        step_desc = " -> ".join(str(step.get("description", "")) for step in steps)
        path_count = payload.get("path_count", 0)
        lines.append(
            f"--{path_id}: path_count={path_count}, steps={step_desc}"
        )
    return "\n".join(lines)


def _render_candidate(candidate: MergedCandidate, include_transition: bool) -> str:
    cu = candidate.canonical_unit
    parts = [
        f'CU {candidate.cu_id}: "{cu.get("intent", "")}"',
        f'source={candidate.source}',
    ]
    if include_transition and candidate.transition_count is not None:
        parts.append(f"transition_count={candidate.transition_count}")
    parts.extend(
        [
            f'execution_count={cu.get("execution_count", 0)}',
            "paths:",
            _format_paths(dict(cu.get("path_stats", {}))),
            f'parameters={_format_parameter_defs(list(cu.get("parameter_defs", [])))}',
        ]
    )
    return "\n".join(parts)


def build_cu_retrieval_prompt(retrieval_result: RetrievalResult) -> str:
    # 这个 prompt 只是“规划提示”，不是强制指令。
    # planner 可以采用其中一个 CU，也可以一个都不用，
    # 但当前轮仍然必须只输出一个 grounded action。
    sections: List[str] = []

    if retrieval_result.warnings:
        sections.append("")
        sections.append("CU RETRIEVAL WARNINGS:")
        for warning in retrieval_result.warnings:
            sections.append(f"- {warning}")

    if retrieval_result.state_candidates:
        sections.append("")
        sections.append("CU CANDIDATES FROM CURRENT STATE:")
        for idx, candidate in enumerate(retrieval_result.state_candidates, start=1):
            merged = MergedCandidate(
                cu_id=candidate.cu_id,
                canonical_unit=candidate.canonical_unit,
                source="state_similarity",
                similarity_score=candidate.similarity_score,
            )
            sections.append(f"{idx}.")
            sections.append(_render_candidate(merged, include_transition=False))
            sections.append("")

    if retrieval_result.transition_candidates:
        sections.append("")
        sections.append("CU CANDIDATES FROM TRANSITION HISTORY:")
        for idx, candidate in enumerate(retrieval_result.transition_candidates, start=1):
            merged = MergedCandidate(
                cu_id=candidate.cu_id,
                canonical_unit=candidate.canonical_unit,
                source="transition",
                transition_count=candidate.transition_count,
            )
            sections.append(f"{idx}.")
            sections.append(_render_candidate(merged, include_transition=True))
            sections.append("")

    if retrieval_result.merged_candidates:
        sections.append("")
        sections.append("CU PLANNING GUIDANCE:")
        sections.append(
            "Use these CU candidates as optional planning hints only. "
            "Choose at most one atomic action for this turn. "
            "If no candidate fits the current screen, ignore them and act autonomously."
        )
        sections.append(
            "If you decide to follow a CU candidate, include a `(CU Selection)` section in your response with exact fields: "
            "`Selected CU: <cu_id or NONE>`, `Selected Path: <path_id or NONE>`, and "
            "`Reference Params: <JSON object, or {} if none>`."
        )
        sections.append(
            "Always prioritize task fit and current UI fit first. "
            "Do not choose a CU or path only because it has a higher history count if it conflicts with the requested route, parameters, or current screen."
        )
        sections.append(
            "If multiple CU candidates fit the current task and screen equally well, prefer the CU with higher `execution_count`."
        )
        sections.append(
            "Within a selected CU, if multiple paths fit equally well, prefer the path with higher `path_count`. "
            "Use historical frequency only as a tie-breaker after confirming compatibility."
        )
    return "\n".join(sections).strip()
