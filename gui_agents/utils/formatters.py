"""This file contains various formatting checks used to reprompt an agent for correctly formatted responses."""

import re

from gui_agents.utils.common_utils import (
    extract_agent_functions,
    parse_code_from_string,
    create_pyautogui_code,
    split_thinking_response,
)

single_action_check = (
    lambda response: len(extract_agent_functions(parse_code_from_string(response))) == 1
)
single_action_error_msg = (
    "Incorrect code: There must be a single agent action in the code response."
)
SINGLE_ACTION_FORMATTER = lambda response: (
    single_action_check(response),
    single_action_error_msg,
)


_REQUIRED_SECTION_HEADERS = [
    "(Previous action verification)",
    "(Screenshot Analysis)",
    "(CU Selection)",
    "(Next Action)",
    "(Grounded Action)",
]


def _count_section_occurrences(response, section_header):
    return len(
        re.findall(re.escape(section_header), response or "", flags=re.IGNORECASE)
    )


def _has_single_section_set(response):
    return all(
        _count_section_occurrences(response, section_header) == 1
        for section_header in _REQUIRED_SECTION_HEADERS
    )


single_section_set_error_msg = (
    "Incorrect response: The response must contain exactly one complete set of the required planner sections."
)
SINGLE_SECTION_SET_FORMATTER = lambda response: (
    _has_single_section_set(response),
    single_section_set_error_msg,
)


def _count_code_blocks(response):
    return len(re.findall(r"```(?:\w+\s+)?(.*?)```", response or "", re.DOTALL))


single_code_block_error_msg = (
    "Incorrect response: The response must contain exactly one Python code block."
)
SINGLE_CODE_BLOCK_FORMATTER = lambda response: (
    _count_code_blocks(response) == 1,
    single_code_block_error_msg,
)


def _attempt_code_creation(agent, code, obs):
    """Attempts to create a pyautogui code snippet from the response code"""
    try:
        return create_pyautogui_code(agent, code, obs)
    except Exception as e:
        return None


code_valid_check = (
    lambda agent, obs, response: _attempt_code_creation(
        agent, parse_code_from_string(response), obs
    ) is not None
)
code_valid_error_msg = "Incorrect code: The agent action must be a valid function and use valid parameters from the docstring list."
CODE_VALID_FORMATTER = lambda agent, obs, response: (
    code_valid_check(agent, obs, response),
    code_valid_error_msg,
)

thoughts_answer_tag_check = lambda response: split_thinking_response(response)[1] != ""
thoughts_answer_tag_error_msg = "Incorrect response: The response must contain both <thoughts>...</thoughts> and <answer>...</answer> tags."
THOUGHTS_ANSWER_TAG_FORMATTER = lambda response: (
    thoughts_answer_tag_check(response),
    thoughts_answer_tag_error_msg,
)

integer_answer_check = (
    lambda response: split_thinking_response(response)[0].strip().isdigit()
)
integer_answer_error_msg = (
    "Incorrect response: The <answer>...</answer> tag must contain a single integer."
)
INTEGER_ANSWER_FORMATTER = lambda response: (
    integer_answer_check(response),
    integer_answer_error_msg,
)
