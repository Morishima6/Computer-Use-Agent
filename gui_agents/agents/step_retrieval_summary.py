import textwrap
from typing import Dict, Optional

from gui_agents.core.mllm import LMMAgent
from gui_agents.utils.common_utils import call_llm_safe


STEP_RETRIEVAL_SUMMARIZER_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a retrieval-oriented UI state summarizer.

    Your job is to produce a short screen-state summary for semantic step retrieval.
    The summary should help match the current UI state to a historical "action_before_state".

    Focus only on action-relevant information:
    1. active application or page/view
    2. selected / focused object or field
    3. visible controls that are directly relevant to the likely next action
    4. critical UI state changes (enabled/disabled, checked/unchecked, expanded/collapsed, dialog opened, object selected, sidebar visible, etc.)
    5. short key text only if it helps identify the actionable region

    Ignore:
    - decorative/background content
    - repeated layout details
    - unrelated text blocks
    - global descriptions of the whole page unless necessary

    Output rules:
    - plain text only
    - 2 to 4 short sentences
    - no coordinates
    - no code
    - emphasize the current actionable state, not the full screenshot
    """
).strip()

STEP_RETRIEVAL_SUMMARIZER_USER_PROMPT = textwrap.dedent(
    """
    Summarize this screenshot for step retrieval.

    Requirements:
    - Describe only the UI state most relevant to the next action.
    - Prefer the currently selected object, focused control, open panel, or actionable region.
    - Mention at most 3 directly relevant controls or state cues.
    - Keep it concise and retrieval-friendly.
    """
).strip()


def create_step_retrieval_summarizer(engine_params: Dict) -> LMMAgent:
    return LMMAgent(
        engine_params=engine_params,
        system_prompt=STEP_RETRIEVAL_SUMMARIZER_SYSTEM_PROMPT,
    )


def generate_step_retrieval_summary(
    image_content,
    engine_params: Dict,
    summarizer: Optional[LMMAgent] = None,
) -> str:
    agent = summarizer or create_step_retrieval_summarizer(engine_params)
    normalized_image = str(image_content) if hasattr(image_content, "__fspath__") else image_content
    agent.reset()
    agent.add_message(
        text_content=STEP_RETRIEVAL_SUMMARIZER_USER_PROMPT,
        image_content=normalized_image,
        role="user",
        put_text_last=True,
    )
    return call_llm_safe(agent, temperature=0.0).strip()
