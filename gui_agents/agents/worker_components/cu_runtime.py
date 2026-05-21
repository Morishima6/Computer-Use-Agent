# CU retrieval运行时状态机模块

import re
from typing import Any, Dict


CU_RETRIEVAL_CONTEXT_BEGIN = "<BEGIN CU RETRIEVAL CONTEXT>"
CU_RETRIEVAL_CONTEXT_END = "<END CU RETRIEVAL CONTEXT>"


def reset_cu_runtime_to_idle(worker: Any) -> None:
    clear_active_cu_state(worker)
    worker.cu_runtime_mode = "idle"
    worker.intent_predict = ""
    worker.cu_retry_count = 0
    worker.consecutive_no_progress_count = 0
    worker.cu_control = {
        "selection_source": None,
        "valid": False,
        "status": "",
        "reason": "",
        "progress_assessment": "",
    }


def update_cu_selection_state(worker: Any, validated_selection: Dict[str, Any]) -> None:
    worker.bad_reflection_count = 0
    selected_cu_id = validated_selection.get("selected_cu_id")
    if not selected_cu_id:
        reset_cu_runtime_to_idle(worker)
        return

    worker.intent_predict = ""
    worker.active_cu_id = selected_cu_id
    worker.active_path_id = validated_selection.get("selected_path_id")
    worker.active_reference_params = dict(
        validated_selection.get("reference_params", {})
    )
    worker.cu_runtime_mode = "active_cu"
    worker.cu_control = {
        "selection_source": validated_selection.get("selection_source"),
        "valid": bool(validated_selection.get("valid", False)),
        "status": "selected",
        "reason": "",
        "progress_assessment": "",
    }


def should_run_cu_retrieval(worker: Any) -> bool:
    return (
        worker.enable_cu_retrieval
        and worker.cu_retriever is not None
        and worker.cu_runtime_mode in {"need_retrieve", "completed_cu"}
        and worker.cu_control.get("status") != "task_completed"
    )


def build_cu_retrieval_query(worker: Any, instruction: str) -> str:
    parts = [
        f"Task: {instruction}" if instruction.strip() else "",
        f"Predicted Intent: {worker.intent_predict}" if worker.intent_predict.strip() else ""
    ]
    return "\n".join(part for part in parts if part.strip())


def wrap_cu_retrieval_context(retrieval_prompt: str) -> str:
    text = (retrieval_prompt or "").strip()
    if not text:
        return ""
    return (
        f"{CU_RETRIEVAL_CONTEXT_BEGIN}\n"
        f"{text}\n"
        f"{CU_RETRIEVAL_CONTEXT_END}"
    )


def strip_cu_retrieval_context(message_text: str) -> str:
    text = message_text or ""
    pattern = (
        rf"\n*{re.escape(CU_RETRIEVAL_CONTEXT_BEGIN)}\n.*?\n"
        rf"{re.escape(CU_RETRIEVAL_CONTEXT_END)}\n*"
    )
    stripped = re.sub(pattern, "\n\n", text, flags=re.DOTALL)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def clear_active_cu_state(worker: Any) -> None:
    worker.active_cu_id = None
    worker.active_path_id = None
    worker.active_reference_params = {}


def apply_reflection_control(worker: Any, control: Dict[str, Any]) -> None:
    status = str(control.get("status", "")).strip().lower()
    reason = str(control.get("reason", "")).strip()
    progress_assessment = str(control.get("progress_assessment", "")).strip()
    intent_predict = str(control.get("intent_predict", "")).strip()

    worker.intent_predict = intent_predict
    print("-" * 100)
    print("**intent_predict**:\n", intent_predict)
    current_selection_source = worker.cu_control.get("selection_source")
    current_valid = bool(worker.cu_control.get("valid", False))
    worker.cu_control = {
        "selection_source": current_selection_source,
        "valid": current_valid,
        "status": status or worker.cu_control.get("status", "idle"),
        "reason": reason,
        "progress_assessment": progress_assessment,
    }

    if not status:
        return

    if status == "continue_current_cu":
        if worker.active_cu_id:
            worker.cu_runtime_mode = "active_cu"
        worker.cu_retry_count = 0
        worker.consecutive_no_progress_count = 0
        return

    if status == "cu_blocked":
        if worker.active_cu_id:
            worker.cu_runtime_mode = "active_cu"
        worker.consecutive_no_progress_count += 1
        return

    if status == "cu_completed":
        if worker.active_cu_id:
            worker.last_completed_cu_id = worker.active_cu_id
        clear_active_cu_state(worker)
        worker.cu_runtime_mode = "completed_cu"
        worker.cu_retry_count = 0
        worker.consecutive_no_progress_count = 0
        return

    if status in {"cu_failed"}:
        clear_active_cu_state(worker)
        worker.cu_runtime_mode = "need_retrieve"
        worker.cu_retry_count += 1
        worker.consecutive_no_progress_count += 1
        return

    if status == "task_completed":
        if worker.active_cu_id:
            worker.last_completed_cu_id = worker.active_cu_id
        clear_active_cu_state(worker)
        worker.cu_runtime_mode = "completed_cu"
        worker.cu_retry_count = 0
        worker.consecutive_no_progress_count = 0
