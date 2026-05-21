# Active CU prompt 构造模块，把当前 CU/path 状态渲染给 planner

from typing import Any, Dict, Optional


def get_active_cu(worker: Any) -> Optional[Dict[str, Any]]:
    if worker.cu_store is None or not worker.active_cu_id:
        return None
    return worker.cu_store.get_cu(worker.active_cu_id)


def get_active_path_payload(worker: Any) -> Optional[Dict[str, Any]]:
    cu = get_active_cu(worker)
    if cu is None or not worker.active_path_id:
        return None
    path_stats = cu.get("path_stats", {})
    payload = path_stats.get(worker.active_path_id)
    return payload if isinstance(payload, dict) else None


def build_active_cu_prompt(worker: Any) -> str:
    cu = get_active_cu(worker)
    path_payload = get_active_path_payload(worker)
    if cu is None or path_payload is None:
        return ""

    steps = list(path_payload.get("steps", []) or [])

    lines = [
        "ACTIVE CU EXECUTION CONTEXT:",
        f"Current Mode: {worker.cu_runtime_mode}",
        f"CU ID: {worker.active_cu_id}",
        f'CU Intent: {cu.get("intent", "")}',
        f"Path ID: {worker.active_path_id}",
    ]

    if steps:
        lines.append("Reference Path Steps:")
        for idx, step in enumerate(steps, start=1):
            step_type = step.get("type", "UNKNOWN")
            description = step.get("description", "")
            lines.append(f"{idx}. {step_type}: {description}")
    else:
        lines.append("Reference Path Steps: none")

    lines.append(
        f"Reference Params: {worker.active_reference_params if worker.active_reference_params else {}}"
    )

    abstract_after = str(cu.get("abstract_state_after", "")).strip()
    if abstract_after:
        lines.extend(["Expected After State:", abstract_after])

    lines.extend([
            "\nContinue executing this CU path unless the current screenshot clearly shows that the path is no longer applicable.",
            "Do not perform a new CU search mentally in this mode; use the active path as the primary planning reference while still grounding actions from the screenshot.",
            "Infer progress from the current screenshot and UI state instead of assuming the path has advanced automatically.",
            "Even when following this path, output only the single next atomic action that best fits the current screen.",
        ])

    if str(worker.cu_control.get("status", "")).strip().lower() == "cu_blocked":
        lines.extend([
                "RECOVERY GUIDANCE:",
                "Reflection indicates the previous grounded action likely affected the wrong control or target, but the current CU/path is still valid.",
                "Before continuing the main CU steps, first choose the single safest recovery action that restores the UI to the intended state for this CU.",
                "Prefer a local rollback or recovery action that clearly fits the screenshot, such as undoing an incorrect edit, dismissing a mistaken popup, escaping an unintended mode, or reselecting the correct target.",
                "Do not abandon this CU or start a fresh retrieval unless the screenshot now shows that this CU/path is no longer applicable.",
            ])

    return "\n".join(lines).strip()
