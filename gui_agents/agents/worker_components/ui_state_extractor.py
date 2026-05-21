# UI 状态文本生成模块

import os
import textwrap
from typing import Any, Dict

from gui_agents.core.mllm import LMMAgent
from gui_agents.utils.common_utils import call_llm_safe


STATE_TEXT_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a UI state describer for canonical unit retrieval.
    Given a screenshot of a desktop app, describe only the current actionable UI state.
    Focus on:
    - active app or window
    - current view, mode, or dialog
    - currently selected or focused object
    - controls or regions most relevant to the next action
    Output plain text only.
    """
).strip()

STATE_TEXT_USER_PROMPT = textwrap.dedent(
    """
    Describe the current screenshot for CU retrieval.
    Keep it concise and action-oriented.
    Do not output coordinates or code.
    """
).strip()


def get_state_text_agent(worker: Any) -> LMMAgent:
    if worker._state_text_agent is None:
        state_text_engine_params = {
            "engine_type": "openai",
            "model": "qwen3-vl-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
            "temperature": 0.0,
        }
        worker._state_text_agent = LMMAgent(
            engine_params=state_text_engine_params,
            system_prompt=STATE_TEXT_SYSTEM_PROMPT,
        )
    return worker._state_text_agent


def generate_ui_state_text(worker: Any, model_obs: Dict, logger: Any) -> str:
    if "screenshot" not in model_obs:
        return ""
    try:
        agent = get_state_text_agent(worker)
        agent.reset()
        agent.add_message(
            text_content=STATE_TEXT_USER_PROMPT,
            image_content=model_obs["screenshot"],
            role="user",
            put_text_last=True,
        )
        return call_llm_safe(agent, temperature=0.0).strip()
    except Exception as exc:
        logger.error("UI state text generation failed: %s", exc)
        return ""
