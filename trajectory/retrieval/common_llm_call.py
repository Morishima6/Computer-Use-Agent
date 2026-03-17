import os
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()

qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

qwen_client: Optional[OpenAI] = None

def _get_qwen_api_key() -> Optional[str]:
    # Preferred key name in this repo
    api_key = os.getenv("QWEN_API_KEY")
    if api_key:
        return api_key
    # Commonly used name in DashScope docs/setups
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key
    return None

def get_qwen_client():
    global qwen_client
    if qwen_client is None:
        api_key = _get_qwen_api_key()
        if not api_key:
            raise RuntimeError(
                "Qwen API Key 未设置：请在环境变量或 .env 中配置 `QWEN_API_KEY`（推荐）或 `DASHSCOPE_API_KEY`。"
            )
        qwen_client = OpenAI(api_key=api_key, base_url=qwen_base_url)
    return qwen_client



def get_embedding(text: str, model: str = "text-embedding-v4") -> Optional[list[float]]:
    try:
        response = get_qwen_client().embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None



def llm_judge_step_precondition(
    action_preconditions: List[str], runtime_nl_explanation: str
) -> bool:
    if not action_preconditions or str(action_preconditions).strip() == "":
        return False

    prompt = f"""
# Role
You are a senior QA Automation Engineer specialized in UI automation validation. Your task is to determine whether the required preconditions for a given automation step are satisfied by the current screen state.

# Input
## Required Preconditions
{action_preconditions}

## Current Screen State (Context)
{runtime_nl_explanation}

# Decision Rules
Return "True" only if the Current Screen State logically entails (supports) every Required Precondition.

1. Logical entailment:
   - If a precondition requires a specific fact (e.g., "Username field is filled"), the Current Screen State must explicitly confirm it or provide a clear equivalent signal.

2. Contradictions:
   - If the Current Screen State explicitly contradicts any precondition, return False.
   - Example: Precondition: "Cart is empty" vs. Current: "Cart has 2 items" => False.

3. Missing information (conservative default):
   - If the Current Screen State does not mention an element/state needed to verify a precondition, treat it as NOT satisfied and return False.
   - Do not assume facts that are not supported by the provided context.

4. Semantic / implicit confirmation:
   - You may infer satisfaction only when the Current Screen State provides strong, unambiguous UI evidence.
   - Example: Precondition: "User is logged in" can be satisfied if the screen shows a "Logout" button, account avatar with user name, or a "My Profile" area that clearly indicates an authenticated session.
   - Weak or ambiguous signals are not sufficient; when in doubt, return False.

# Output Requirements
Return ONLY a raw JSON object (no markdown, no extra text). Use this schema exactly:
{{
  "analysis": "A brief explanation referencing the specific evidence (or contradiction/missing info) from the Current Screen State.",
  "is_satisfied": true or false
}}
"""

    try:
        completion = get_qwen_client().chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a strict logic evaluator. Output must be raw JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1
        )

        content = completion.choices[0].message.content
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)
        print(result)
        return bool(result.get("is_satisfied", False))

    except Exception as e:
        print(f"Precondition LLM check failed: {e}")
        return False


def llm_judge_step_append_to_plan(retrieval_summary: str) -> bool:

    if not retrieval_summary or str(retrieval_summary).strip() == "":
        return False

    prompt = f"""
You are a computer operation evaluation assistant tasked with assessing whether operations retrieved from the retrieval system should be provided as reference information to the planning agent.

Your task is to determine whether the retrieved operations:
1. Are relevant to the current task
2. Are reasonable and executable
3. Will not mislead or contaminate the planning agent's decision-making

Below is the retrieved operation description and execution status:
{retrieval_summary}

If the retrieved action is reasonable, relevant, and suitable for reference, respond with "YES".
If the retrieved action is unreasonable, irrelevant, or may mislead the worker agent, respond with "NO".

Please respond only with "YES" or "NO" without adding any additional content.
"""

    try:
        completion = get_qwen_client().chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a strict operation evaluator. Answer only YES or NO."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0
        )

        content = completion.choices[0].message.content
        content_clean = str(content).strip().upper()
        if content_clean.startswith("YES"):
            return True
        return False
    except Exception as e:
        print(f"Append-to-plan LLM check failed: {e}")
        return False
