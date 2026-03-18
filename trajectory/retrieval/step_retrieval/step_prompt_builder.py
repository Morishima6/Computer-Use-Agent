import sys
from pathlib import Path
from typing import Any, Dict, Optional

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

def build_prompt_from_step(step_data: Dict[str, Any]) -> str:
    response_prompt = {}
    if not step_data:
        return ""

    similarity_above_k = step_data.get("similarity_above_k", False)
    response_prompt["similarity_above_k"] = similarity_above_k

    if not similarity_above_k:
        lines = []
        lines.append("Below is a description of a similar action retrieved externally and its execution status.")
        lines.append(f"- Step_Goal: {step_data.get('full_step_data').get('step_goal')}")
        lines.append(f"- Action_Before_State: {step_data.get('full_step_data').get('action_before_state')}")
        lines.append(f"- Action: {step_data.get('full_step_data').get('action')}")
        lines.append(f"- Execute_Description: {step_data.get('full_step_data').get('nl_explanation')}")
        lines.append(f"- similarity: {step_data.get('similarity', 'N/A')}")
        content = "\n".join(lines).strip()
        response_prompt["content"] = content
        is_append = False
        try:
            is_append = llm_judge_step_append_to_plan(content)
        except Exception as e:
            print(f"Judge append-to-plan failed: {e}")
            is_append = False
            
        print(f"****LLM judge step append to plan: {is_append}")
        print("-" * 20)
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

    step_data = find_step_by_similarity(runtime_nl_explain, k=k)
    if not step_data:
        print("🚧 未找到匹配的步骤。")
        sys.exit(0)

    prompt_text = build_prompt_from_step(step_data)
    print(prompt_text)
