# Reflection 控制模块

import re
import textwrap
from typing import Any, Dict, Tuple

from gui_agents.agents.worker_components.active_cu_prompt import (
    get_active_cu,
    get_active_path_payload,
)
from gui_agents.agents.worker_components.cu_selection_parser import (
    extract_cu_selection_section,
    parse_cu_selection_from_plan,
)
from gui_agents.utils.common_utils import call_llm_safe, split_thinking_response


def parse_reflection_control(reflection_text: str) -> Dict[str, Any]:
    text = (reflection_text or "").strip()
    result = {
        "status": "",
        "reason": "",
        "progress_assessment": "",
        "intent_predict": "",
    }
    if not text:
        return result

    patterns = {
        "status": r"Status:\s*(.+)",
        "reason": r"Reason:\s*(.+)",
        "progress_assessment": r"Progress Assessment:\s*(.+)",
        "intent_predict": r"Intent Predict:\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    allowed_statuses = {
        "continue_current_cu",
        "cu_blocked",
        "cu_completed",
        "cu_failed",
        "task_completed",
    }
    status = result["status"].lower()
    if status not in allowed_statuses:
        result["status"] = ""
    else:
        result["status"] = status

    if result["intent_predict"].upper() == "NONE":
        result["intent_predict"] = ""

    return result


def sanitize_reflection_for_planner(reflection_text: str) -> str:
    text = (reflection_text or "").strip()
    if not text:
        return ""

    sanitized_lines = []
    for line in text.splitlines():
        if re.match(r"^\s*Intent Predict\s*:", line, flags=re.IGNORECASE):
            continue
        sanitized_lines.append(line)

    sanitized = "\n".join(sanitized_lines).strip()
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized


def is_valid_reflection_control(control: Dict[str, Any]) -> bool:
    status = str(control.get("status", "")).strip().lower()
    reason = str(control.get("reason", "")).strip()
    progress_assessment = str(control.get("progress_assessment", "")).strip()
    return bool(status and reason and progress_assessment)


def build_reflection_fallback_control(worker: Any) -> Dict[str, Any]:
    if worker.active_cu_id and worker.bad_reflection_count < 2:
        return {
            "status": "continue_current_cu",
            "reason": "Reflection output was malformed; conservatively keep the current CU for this turn.",
            "progress_assessment": "Reflection parsing failed.",
            "intent_predict": "",
        }
    return {
        "status": "cu_failed",
        "reason": "Reflection output was malformed repeatedly; fall back to a fresh CU retrieval.",
        "progress_assessment": "Reflection parsing failed.",
        "intent_predict": "",
    }


def latest_worker_explicitly_selected_no_cu(worker: Any) -> bool:
    if not getattr(worker, "worker_history", None):
        return False

    latest_plan = worker.worker_history[-1]
    if not extract_cu_selection_section(latest_plan):
        return False

    parsed_selection = parse_cu_selection_from_plan(latest_plan)
    return parsed_selection.get("selected_cu_id") is None


def normalize_reflection_control(worker: Any, control: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(control or {})
    status = str(normalized.get("status", "")).strip().lower()

    if status == "task_completed" or worker.active_cu_id:
        return normalized

    if latest_worker_explicitly_selected_no_cu(worker):
        normalized["status"] = "cu_failed"
        normalized["reason"] = (
            "The latest planner output explicitly selected no CU, so the next turn should run a fresh retrieval."
        )
        normalized["progress_assessment"] = (
            "The previous turn proceeded without binding a CU."
        )
        # normalized["intent_predict"] = ""

    return normalized


def build_active_path_reflection_context(worker: Any) -> str:
    cu = get_active_cu(worker)
    path_payload = get_active_path_payload(worker)
    if cu is None or path_payload is None:
        return ""

    lines = [
        f'Active CU Intent: {cu.get("intent", "")}',
    ]

    steps = list(path_payload.get("steps", []) or [])
    if steps:
        lines.append("Reference Path Steps:")
        for idx, step in enumerate(steps, start=1):
            step_type = step.get("type", "UNKNOWN")
            description = step.get("description", "")
            lines.append(f"{idx}. {step_type}: {description}")
    lines.append(
        f"Reference Params: {worker.active_reference_params if worker.active_reference_params else {}}"
    )

    abstract_after = str(cu.get("abstract_state_after", "")).strip()
    if abstract_after:
        lines.extend(
            [
                "Expected After State:",
                abstract_after,
            ]
        )

    return "\n".join(lines)


def generate_reflection(
    worker: Any, instruction: str, model_obs: Dict, logger: Any
) -> Tuple[str, str, Dict[str, Any]]:
    reflection = None
    reflection_thoughts = None
    reflection_control: Dict[str, Any] = {
        "status": "",
        "reason": "",
        "progress_assessment": "",
        "intent_predict": "",
    }
    if worker.enable_reflection:
        if worker.turn_count == 0:
            text_content = textwrap.dedent(
                f"""
                Task Description: {instruction}
                Current Trajectory below:
                """
            )
            updated_sys_prompt = worker.reflection_agent.system_prompt + "\n" + text_content
            worker.reflection_agent.add_system_prompt(updated_sys_prompt)
            worker.reflection_agent.add_message(
                text_content="The initial screen is provided. No action has been taken yet.",
                image_content=model_obs["screenshot"],
                role="user",
            )
        else:
            active_context_lines = [
                f"Active CU ID: {worker.active_cu_id or 'NONE'}",
                f"Active Path ID: {worker.active_path_id or 'NONE'}",
            ]
            active_path_context = build_active_path_reflection_context(worker)
            if active_path_context:
                active_context_lines.append(active_path_context)
            print("**reflection prompt**:","\n".join(active_context_lines)+ "\n\nLatest Worker Output:\n"+ worker.worker_history[-1])
            worker.reflection_agent.add_message(
                text_content="\n".join(active_context_lines)
                + "\n\nLatest Worker Output:\n"
                + worker.worker_history[-1],
                image_content=model_obs["screenshot"],
                role="user",
            )
            full_reflection = call_llm_safe(
                worker.reflection_agent,
                temperature=worker.temperature,
                use_thinking=worker.use_thinking,
            )
            reflection, reflection_thoughts = split_thinking_response(full_reflection)
            reflection_control = parse_reflection_control(reflection)
            reflection_control = normalize_reflection_control(worker, reflection_control)
            if is_valid_reflection_control(reflection_control):
                worker.bad_reflection_count = 0
            else:
                worker.bad_reflection_count += 1
                logger.warning(
                    f"Reflection control parse failed on turn {worker.turn_count} (bad count={worker.bad_reflection_count}). Raw reflection:{reflection}"
                )
                reflection_control = build_reflection_fallback_control(worker)
            worker.reflections.append(reflection)
            logger.info("REFLECTION THOUGHTS: %s", reflection_thoughts)
            logger.info("REFLECTION: %s", reflection)
            logger.info("REFLECTION CONTROL: %s", reflection_control)
    return reflection, reflection_thoughts, reflection_control
