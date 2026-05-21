import json
import os
import shlex
import subprocess
import sys
from typing import Optional, List


def _find_codex_path() -> str:
    return "codex"


def _build_codex_command(model: Optional[str]) -> List[str]:
    codex_path = _find_codex_path()

    base = f"{codex_path} exec --skip-git-repo-check -c reasoning_effort=medium --json"

    if model:
        base += f" --model {shlex.quote(model)}"
    if os.name == "nt":
        return ["cmd", "/c", base]
    return shlex.split(base)

# 静默等待，没有输出，直到Codex完成任务
def call_codex(model: str, system_prompt: str, user_prompt: str) -> str:
    """发送 prompt 给 Codex，并返回 assistant_message（效果与 codex exec 一致）"""

    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()
    cmd = _build_codex_command(model)

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(f"Codex error: {result.stderr}")

    # 从 stdout 解析 JSON event 流，取最后一个 assistant_message/agent_message
    last_content: Optional[str] = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except:
            continue

        if event.get("type") in ("item.started", "item.completed"):
            item = event.get("item", {})
            item_type = item.get("type") or item.get("item_type")
            if item_type in ("assistant_message", "agent_message"):
                content = item.get("text") or item.get("content")

                if isinstance(content, list):
                    content = "".join(
                        c if isinstance(c, str) else c.get("text", "")
                        for c in content
                    )
                last_content = content

    if last_content is not None:
        return last_content
    return "(no assistant message returned)"


# 由Codex充分发挥Agent功能，自己决定文件读写
def call_codex_streaming(model: str, system_prompt: str, user_prompt: str) -> None:
    """
    流式调用 Codex：让 codex CLI 直接把输出打印到终端，
    行为等同于你在命令行手动运行 `codex exec`。
    不在 Python 里解析结果，由 LLM 自己决定如何处理（包括是否修改文件）。
    """

    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()
    # 把 prompt 作为 CLI 的 [PROMPT] 参数传给 codex exec
    cmd = _build_codex_command(model, json_mode=False)
    cmd.append(prompt)

    # 不捕获 stdout/stderr，让它们直接继承当前终端，便于边生成边看
    process = subprocess.Popen(cmd)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError("Codex exited with non-zero status")



# 调用call codex -> 解析返回的数据 -> 填充report.json
if __name__ == "__main__":
    model = "gpt-5.1"

    # 会话文件夹路径（包含 report.json 和截图子文件夹）：
    # 1) 优先使用命令行参数：python call_codex.py /path/to/folder
    # 2) 若无参数，则在运行时通过 input() 询问
    if len(sys.argv) > 1:
        conversation_folder = sys.argv[1].strip()
    else:
        conversation_folder = input(
            "Please input the conversation folder path (where report.json and screenshots are located): "
        ).strip()

    # 如果最终仍为空，则使用当前工作目录
    if not conversation_folder:
        conversation_folder = os.getcwd()

    report_path = os.path.join(conversation_folder, "report.json")

    # system_prompt：设定模型的角色、职责和整体输出规范（通常较稳定）
    system_prompt = (
        "You are an action-behavior analyst and recorder.\n"
        "\n"
        "Your task:\n"
        "Analyze a report.json file that records a single task containing multiple UI actions. "
        "For each action (step) you must infer a structured description strictly from the JSON metadata and screenshots, "
        "and output a machine-readable JSON array that can be used to fill the empty fields in report.json.\n\n"
        "For each action/step you MUST produce one JSON object with exactly these fields:\n"
        '- \"task_title\": A concise title summarizing the overall task (same value for all actions).\n'
        '- \"step_goal\": A short phrase describing the immediate goal of this specific action within the overall task.\n'
        '- \"app\": The software/application used during the task.\n'
        '- \"url\": Any URL relevant to the task or the specific action.\n'
        '- \"action_preconditions\": What must be true or present before the action occurs (based on the before screenshot).\n'
        '- \"nl_position\": A natural-language description of the mouse location or targeted UI element (based on the red marker in the before screenshot). If the step has no on-screen target (for example, a typing or press action where \"action.target\" is missing or an empty object in report.json), set this field to null instead of describing any location. If you cannot confidently identify what the element is or what text it contains, instead describe its visual appearance (shape, color, approximate size) and relative location (for example, \"a blue rectangular button near the top-right corner\").\n'
        '- \"action_before_state\": The UI state or condition before the action.\n'
        '- \"action_after_effects\": The changes caused by the action (based on the after screenshot).\n'
        '- \"nl_explanation\": A concise, natural-language explanation of the action and its purpose, written without referring to \"the user\" (describe the step itself, for example, \"Click the Save button to store the changes.\").\n\n'
        "Output format requirements (very important):\n"
        "- The FINAL answer must be a single JSON array (e.g. [ { ... }, { ... }, ... ]) with one object per action.\n"
        "- Do not print any explanations, comments, or non-JSON text in the final answer.\n"
        "- Do not include trailing commas. The JSON must be strictly valid.\n"
    )

    # user_prompt：这一次具体要做的任务，并带上会话文件夹路径
    user_prompt = (
        "You are given a conversation folder located at:\n"
        f"{conversation_folder}\n\n"
        "Inside this folder there is a report.json file and a screenshots/ subfolder referenced by it.\n\n"
        "Your job now:\n"
        "1) Read report.json in that folder.\n"
        "2) For each step in report.json.steps, carefully inspect:\n"
        "   - The overall task instruction or user prompt in report.json (for example, the \"instruction\" field)\n"
        "   - Its metadata in the JSON, including any screenshot paths (such as screenshot_path_before_part)\n"
        "   - The before screenshot, the partial before screenshot near the signed position, and the after screenshot\n"
        "   - The red-highlighted mouse position\n"
        "   - Any relevant application and URL information\n"
        "3) Then produce ONE JSON array as the final answer. Each element in the array corresponds to one step, "
        "and must contain the fields described in the system prompt: task_title, step_goal, app, url, "
        "action_preconditions, nl_position, action_before_state, action_after_effects, nl_explanation.\n\n"
        "Remember: the final answer must be ONLY that JSON array, with no extra commentary or text."
    )

    # 调用 Codex，获取每个 step 的结构化 JSON 描述
    ai_response = call_codex(model, system_prompt, user_prompt)

    try:
        ai_data = json.loads(ai_response)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse model response as JSON:\n{ai_response}")

    if not isinstance(ai_data, list):
        raise RuntimeError(f"Expected a JSON array of actions, got: {type(ai_data).__name__}")

    # 读入原始 report.json
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"report.json not found at: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    steps = report.get("steps", [])

    # 顶层 task_title / app / env.url：仅在原本为空时填充
    if ai_data:
        first = ai_data[0]
        if isinstance(first, dict):
            if not report.get("task_title") and first.get("task_title"):
                report["task_title"] = first["task_title"]
            if not report.get("app") and first.get("app"):
                report["app"] = first["app"]
            url_val = first.get("url")
            if url_val:
                report.setdefault("env", {})
                if not report["env"].get("url"):
                    report["env"]["url"] = url_val

    # 按顺序将每个 action 信息填入对应 step 的空字段
    for idx, step in enumerate(steps):
        if idx >= len(ai_data):
            break
        action_info = ai_data[idx]
        if not isinstance(action_info, dict):
            continue

        # step_goal: 字符串
        if step.get("step_goal") in (None, "") and action_info.get("step_goal"):
            step["step_goal"] = action_info["step_goal"]

        # action_preconditions: 列表
        if step.get("action_preconditions") in (None, [], "") and action_info.get("action_preconditions"):
            step["action_preconditions"] = [action_info["action_preconditions"]]

        # nl_position: 嵌套在 action.target.nl_position（列表）
        action = step.get("action")
        target = action.get("target") if isinstance(action, dict) else None
        # 只有当原始步骤中存在屏幕坐标（例如点击操作）时，才填充 nl_position
        if isinstance(target, dict) and "position" in target:
            nl_pos = target.get("nl_position")
            if (nl_pos in (None, [], "")) and action_info.get("nl_position"):
                target["nl_position"] = [action_info["nl_position"]]

        # action_before_state: 字符串
        if step.get("action_before_state") in (None, "") and action_info.get("action_before_state"):
            step["action_before_state"] = action_info["action_before_state"]

        # action_after_effects: 列表
        if step.get("action_after_effects") in (None, [], "") and action_info.get("action_after_effects"):
            step["action_after_effects"] = [action_info["action_after_effects"]]

        # nl_explanation: 字符串
        if step.get("nl_explanation") in (None, "") and action_info.get("nl_explanation"):
            step["nl_explanation"] = action_info["nl_explanation"]

    # 写回 report.json（只是在空字段上“填空”，不新增其它字段）
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"report.json filled where empty at: {report_path}")
