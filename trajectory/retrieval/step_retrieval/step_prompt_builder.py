import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..common_llm_call import llm_judge_step_append_to_plan
except ImportError:
    _parent_dir = str(Path(__file__).parent.parent)
    if _parent_dir not in sys.path:
        sys.path.append(_parent_dir)
    from common_llm_call import llm_judge_step_append_to_plan

def find_trace_by_instruction(
    instruction: str,
    trace_root: str,
) -> Optional[Dict[str, Any]]:
    from trajectory.retrieval.task_retrieval.task_prompt_builder import (
        find_trace_by_instruction as _f,
    )

    return _f(instruction, trace_root)


def build_prompt_from_trace(trace: Dict[str, Any]) -> str:
    from trajectory.retrieval.task_retrieval.task_prompt_builder import (
        build_task_reference_prompt,
    )

    return build_task_reference_prompt(trace)


# def _load_report_context(report_path: str) -> Dict[str, Any]:
#     if not report_path:
#         return {}
#     try:
#         return json.loads(Path(report_path).read_text(encoding="utf-8"))
#     except Exception:
#         return {}


def _stringify_preconditions(preconditions: Any) -> str:
    if not preconditions:
        return ""
    if isinstance(preconditions, list):
        items = [str(item).strip() for item in preconditions if str(item).strip()]
        return "\n".join(f"- {item}" for item in items)
    text = str(preconditions).strip()
    return f"- {text}" if text else ""


def _stringify_action(action: Dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return ""

    action_type = str(action.get("type") or "").strip()
    target = action.get("target") or {}
    param = action.get("param") or {}
    target_desc = ""

    if isinstance(target, dict):
        nl_position = target.get("nl_position")
        if isinstance(nl_position, list) and nl_position:
            target_desc = str(nl_position[0]).strip()
        elif isinstance(nl_position, str):
            target_desc = nl_position.strip()
        elif isinstance(target.get("text"), str):
            target_desc = str(target.get("text")).strip()

    parts: List[str] = []
    if action_type:
        parts.append(f"type={action_type}")
    if target_desc:
        parts.append(f"target={target_desc}")
    if isinstance(param, dict) and param:
        compact_param = ", ".join(
            f"{k}={v}" for k, v in param.items() if v not in (None, "", [], {})
        )
        if compact_param:
            parts.append(f"param={compact_param}")
    return "; ".join(parts)


def _stringify_after_effects(effects: Any) -> str:
    if not effects:
        return ""
    if isinstance(effects, list):
        lines: List[str] = []
        for effect in effects:
            if isinstance(effect, str):
                text = effect.strip()
                if text:
                    lines.append(f"- {text}")
            elif isinstance(effect, dict):
                desc = str(effect.get("desc") or "").strip()
                success_signal = str(effect.get("success_signal") or "").strip()
                if desc and success_signal:
                    lines.append(f"- {desc} (success signal: {success_signal})")
                elif desc:
                    lines.append(f"- {desc}")
                elif success_signal:
                    lines.append(f"- success signal: {success_signal}")
        return "\n".join(lines)
    text = str(effects).strip()
    return f"- {text}" if text else ""

def build_prompt_from_step(step_data: Dict[str, Any]) -> str:
    response_prompt = {}
    if not step_data:
        return ""

    similarity_above_k = step_data.get("similarity_above_k", False)
    response_prompt["similarity_above_k"] = similarity_above_k

    if not similarity_above_k:
        full_step = step_data.get("full_step_data") or {}
        # report_context = _load_report_context(step_data.get("report_path", ""))
        # task_title = str(report_context.get("task_title") or "").strip()
        # instruction = str(report_context.get("instruction") or "").strip()
        # app = str(report_context.get("app") or "").strip()
        step_goal = str(full_step.get("step_goal") or "").strip()
        action_before_state = str(full_step.get("action_before_state") or "").strip()
        action_preconditions = _stringify_preconditions(
            full_step.get("action_preconditions")
        )
        action_summary = str(full_step.get("nl_explanation") or "").strip()
        action_structure = _stringify_action(full_step.get("action") or {})
        after_effects = _stringify_after_effects(full_step.get("action_after_effects"))

        lines = []
        lines.append("STEP RETRIEVAL REFERENCE")
        lines.append("This is a similar historical step for planning reference only.")
        # if task_title:
        #     lines.append(f"Historical Task Title: {task_title}")
        # if instruction:
        #     lines.append(f"Historical Task Instruction: {instruction}")
        # if app:
        #     lines.append(f"Application: {app}")
        # if step_data.get("task_id"):
        #     lines.append(f"Historical Task ID: {step_data.get('task_id')}")
        # if step_data.get("step_id"):
        #     lines.append(f"Historical Step ID: {step_data.get('step_id')}")
        # lines.append("Precondition Filter Result: passed")
        lines.append("")
        if step_goal:
            lines.append("Applicable Goal:")
            lines.append(f"- {step_goal}")
        if action_preconditions:
            lines.append("Applicable When:")
            lines.append(action_preconditions)
        if action_before_state:
            lines.append("Historical State:")
            lines.append(f"- {action_before_state}")
        if action_summary:
            lines.append("Historical Action Summary:")
            lines.append(f"- {action_summary}")
        if action_structure:
            lines.append("Historical Action Structure:")
            lines.append(f"- {action_structure}")
        if after_effects:
            lines.append("Expected Result After Action:")
            lines.append(after_effects)
        lines.append("Planner Guidance:")
        lines.append("- Reuse the action idea only if the current screen shows the same key preconditions.")
        lines.append("- Prefer the same target type, focused region, or control/panel pattern if present.")
        lines.append("- Do not copy blindly if the current focus, visible text, or panel state is different.")
        content = "\n".join(lines).strip()
        response_prompt["content"] = content
        is_append = False
        try:
            is_append = llm_judge_step_append_to_plan(content)
        except Exception as e:
            print(f"Judge append-to-plan failed: {e}")
            is_append = False
            
        print(f"****LLM judge step append to plan: {is_append}")
        print("-" * 50)
        response_prompt["isAppend2Plan"] = bool(is_append)
        return response_prompt

    full_step = step_data.get("full_step_data")
    action = full_step.get("action") or {}
    action_type = action.get("type", "")
    target = action.get("target") or {}
    param = action.get("param") or {}
    nl_position = ""


    if isinstance(target, dict):
        nl_position_list = target.get("nl_position", [])
        if isinstance(nl_position_list, list) and nl_position_list:
            nl_position = nl_position_list[0]
        elif isinstance(nl_position_list, str):
            nl_position = nl_position_list

    element_desc = nl_position.strip() if isinstance(nl_position, str) else ""

    nl_explanation = full_step.get("nl_explanation", "")

    grounded_code = ""

    # IMPORTANT: the grounding model localizes UI elements from natural-language descriptions;
    # do not generate actions based on explicit x/y coordinates.
    if action_type in ["click", "double_click"] and element_desc:
        button = param.get("button", "left")
        num_clicks = param.get("num_click", 2 if action_type == "double_click" else 1)
        grounded_code = f"agent.click({element_desc!r}, {int(num_clicks)}, {button!r})"

    elif action_type == "typing":
        text = param.get("text", "")
        if element_desc:
            grounded_code = f"agent.type({element_desc!r}, text={text!r})"
        else:
            grounded_code = f"agent.type(text={text!r})"

    elif action_type == "press":
        press_list = param.get("press_list", [])
        if isinstance(press_list, str):
            press_list = [press_list]
        if not isinstance(press_list, list):
            press_list = [str(press_list)]
        press_list = [str(k) for k in press_list]
        grounded_code = f"agent.hotkey({press_list!r})"

    elif action_type == "scroll" and element_desc:
        direction = (param.get("type") or "down").lower()
        shift = direction in ["left", "right"]
        clicks = 10
        if direction in ["down", "left"]:
            clicks = -10
        grounded_code = f"agent.scroll({element_desc!r}, {clicks}, shift={shift})"

    if not grounded_code:
        return ""

    plan_lines = [
        "(Next Action)",
        nl_explanation,
        "",
        "(Grounded Action)",
        "```python",
        grounded_code,
        "```",
    ]
    response_prompt["content"] = "\n".join(plan_lines).strip()
    return response_prompt


if __name__ == "__main__":
    import sys
    from step_matcher import find_step_by_similarity

    runtime_nl_explain = "Only the upper part of the sitemap is visible, so sections like Tools & resources and For health professionals, including the Natural products link, are below the fold"
    k = 0.9

    step_data = find_step_by_similarity(
        retrieval_query=runtime_nl_explain,
        screen_evidence=runtime_nl_explain,
        k=k,
    )
    if not step_data:
        print("🚧 未找到匹配的步骤。")
        sys.exit(0)

    prompt_text = build_prompt_from_step(step_data)
    print(prompt_text)
