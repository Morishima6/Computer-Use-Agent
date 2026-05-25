# generating testing data using qwen llm
from typing import List, Optional, Any, Tuple,Dict
from dataclasses import dataclass, field
import json
from pathlib import Path
import argparse
import hashlib
import sys
import numpy as np

try:
    from ..common_llm_call import get_qwen_client, get_embedding
except ImportError:
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from common_llm_call import get_qwen_client, get_embedding

try:
    from .train_k import cosine_similarity_01, iter_report_steps, read_step_train_cases_for_k
except ImportError:
    from train_k import cosine_similarity_01, iter_report_steps, read_step_train_cases_for_k

# --- 1. 最内层结构 ---

@dataclass
class Target:
    """定义动作的目标，例如点击的按钮文本"""
    text: str = field(metadata={'description': '目标元素的文本内容'})

@dataclass
class Action:
    """定义一个步骤中执行的动作"""
    type: str = field(metadata={'description': '动作类型，如 click, type, scroll'})
    target: Target = field(metadata={'description': '动作目标对象的详细信息'})


@dataclass
class ActionEffect:
    """定义动作执行后产生的影响，用于验证成功信号"""
    desc: str = field(metadata={'description': '影响的文本描述'})
    success_signal: str = field(metadata={'description': '成功信号的特征，如页面文本或URL变化'})


# --- 2. 步骤和环境结构 ---

@dataclass
class Environment:
    """定义任务执行的环境参数"""
    os: str = field(metadata={'description': '操作系统，如 Windows11'})
    screen: str = field(metadata={'description': '屏幕分辨率，如 1920x1080'})
    url: str = field(metadata={'description': '任务开始时的初始 URL'})
    locale: str = field(metadata={'description': '本地化语言，如 en_US'})


@dataclass
class Step:
    """定义任务中的单个步骤"""
    step_id: str = field(metadata={'description': '步骤的唯一ID'})
    step_goal: str = field(metadata={'description': '步骤的目标描述，如点击Proceed to checkout'})
    
    # 用于相似度计算的关键字段
    action_before_state: str = field(metadata={'description': '执行动作前屏幕的详细描述 (screenshot_explanation)'})
    
    action_preconditions: str = field(metadata={'description': '执行动作的前提条件'})
    action: Action = field(metadata={'description': '执行的具体动作和目标'})
    action_after_effects: List[ActionEffect] = field(metadata={'description': '动作成功执行后的影响列表'})
    nl_explanation: str = field(metadata={'description': '对该步骤的自然语言总结'})


# --- 3. 任务总结构 ---

@dataclass
class Task:
    """定义一个完整的自动化任务"""
    task_id: str = field(metadata={'description': '任务的全局唯一ID'})
    task_category: str = field(metadata={'description': '任务类别，如 Daily'})
    task_title: str = field(metadata={'description': '任务标题，如 pay_amazon_with_firefox'})
    instruction: str = field(metadata={'description': '用户对任务的自然语言指令'})
    app: str = field(metadata={'description': '使用的应用程序，如 Firefox'})
    env: Environment = field(metadata={'description': '任务执行的环境配置'})
    steps: List[Step] = field(metadata={'description': '任务包含的步骤列表'})
    
    
# --- 4. 辅助函数：将字典转换为数据类实例 ---    
def _dict_to_target(d: Dict[str, Any]) -> Target:
    return Target(text=d.get("text", ""))

def _dict_to_action(d: Dict[str, Any]) -> Action:
    target = d.get("target", {}) or {}
    return Action(type=d.get("type", ""), target=_dict_to_target(target))

def _dict_to_action_effect(d: Dict[str, Any]) -> ActionEffect:
    return ActionEffect(desc=d.get("desc", ""), success_signal=d.get("success_signal", ""))

def _dict_to_environment(d: Dict[str, Any]) -> Environment:
    return Environment(
        os=d.get("os", ""),
        screen=d.get("screen", ""),
        url=d.get("url", ""),
        locale=d.get("locale", "")
    )

def _dict_to_step(d: Dict[str, Any]) -> Step:
    action = _dict_to_action(d.get("action", {}) or {})
    effects = [_dict_to_action_effect(e) for e in (d.get("action_after_effects") or [])]
    return Step(
        step_id=d.get("step_id", ""),
        step_goal=d.get("step_goal", ""),
        action_before_state=d.get("action_before_state", ""),
        action_preconditions=d.get("action_preconditions", ""),
        action=action,
        action_after_effects=effects,
        nl_explanation=d.get("nl_explanation", "")
    )

def _dict_to_task(d: Dict[str, Any]) -> Task:
    env = _dict_to_environment(d.get("env", {}) or {})
    steps = [_dict_to_step(s) for s in (d.get("steps") or [])]
    return Task(
        task_id=str(d.get("task_id", "")),
        task_category=d.get("task_category", ""),
        task_title=d.get("task_title", ""),
        instruction=d.get("instruction", ""),
        app=d.get("app", ""),
        env=env,
        steps=steps
    )


generate_task_data_prompt = """
# ROLE: Topic-Guided Multi-Step UI Automation Task Generator
You are an expert in UI Automation and structured data generation. Your task is to generate **one** high-quality UI automation task based **strictly on the provided topic**.

# INPUT TOPIC
The task must be centered around the following topic:
"{TOPIC}"
TaskId is "{TASK_ID}".
StepId must be the format of "TASKID_X", where X is the step number starting from 1.

# GENERATION PROCESS
1.  **INTERPRET THE TOPIC:** Understand the topic and derive a realistic, non-trivial user goal within a modern web application context (e.g., if topic is 'online banking', a valid goal could be 'transfer money to a new recipient').
2.  **DESIGN A SCENARIO:** Create a coherent, end-to-end user scenario that requires **3 to 8 sequential interactions** with the UI.
3.  **GENERATE A SINGLE TASK:** Output exactly one Task object reflecting this scenario.

# GENERATION CONSTRAINTS
1.  **Output Format:** You MUST output a single, valid JSON array containing **exactly one** Task object. Do not include any text, comments, or markdown—only JSON.
2.  **Step Count:** The task MUST contain **between 3 and 8 steps** (inclusive). Steps must be logically connected and represent realistic user actions.
3.  **Content Quality:**
    a.  Each step's `action_before_state` must be highly detailed: describe the visible UI elements, current field values, enabled/disabled states, active tabs, or any relevant context **before** the action is performed.
    b.  The `app`, `task_title`, and `instruction` fields must clearly align with the given topic and reflect the user’s goal.
    c.  Use professional, clear English throughout.
4.  **Environment Assumptions:** 
    - Platform: Web application (responsive, desktop view)
    - OS: Windows 11
    - Browser: Google Chrome
    - Screen: 1920x1080

# JSON REQUIREMENTS
- The output must be a JSON Object not a array
- The Task must conform to the provided JSON schema.
- Do not invent multiple tasks or deviate from the given topic.

# JSON SCHEMA {JSON_SCHEMA}
---
Generate a JSON array with exactly one Task object (3–8 steps) based on the topic: "{TOPIC}".
"""


SCHEMA_CONTENT = {
    "task_id": {"type": "string", "description": "任务的全局唯一ID"},
    "task_category": {"type": "string", "description": "任务类别，如 Daily"},
    "task_title": {"type": "string", "description": "任务标题，如 pay_amazon_with_firefox"},
    "instruction": {"type": "string", "description": "用户对任务的自然语言指令"},
    "app": {"type": "string", "description": "使用的应用程序，如 Firefox"},
    "env": {"$ref": "#/$defs/Environment", "description": "任务执行的环境配置"},
    "steps": {"type": "array", "description": "任务包含的步骤列表", "items": {"$ref": "#/$defs/Step"}}
}
DEFINITIONS_CONTENT = {
    "Target": {"type": "object", "description": "定义动作的目标", "properties": {"text": {"type": "string", "description": "目标元素的文本内容"}}, "required": ["text"]},
    "Action": {"type": "object", "description": "定义一个步骤中执行的动作", "properties": {"type": {"type": "string", "description": "动作类型"}, "target": {"$ref": "#/$defs/Target", "description": "动作目标"}}, "required": ["type", "target"]},
    "ActionEffect": {"type": "object", "description": "定义动作执行后产生的影响", "properties": {"desc": {"type": "string", "description": "影响的文本描述"}, "success_signal": {"type": "string", "description": "成功信号的特征"}}, "required": ["desc", "success_signal"]},
    "Environment": {"type": "object", "description": "定义任务执行的环境参数", "properties": {"os": {"type": "string", "description": "操作系统"}, "screen": {"type": "string", "description": "屏幕分辨率"}, "url": {"type": "string", "description": "初始 URL"}, "locale": {"type": "string", "description": "本地化语言"}}, "required": ["os", "screen", "url", "locale"]},
    "Step": {"type": "object", "description": "定义任务中的单个步骤", "properties": {"step_id": {"type": "string", "description": "步骤的唯一ID"}, "step_goal": {"type": "string", "description": "步骤的目标描述"}, "action_before_state": {"type": "string", "description": "执行动作前屏幕的详细描述"}, "action_preconditions": {"type": "string", "description": "执行动作的前提条件"}, "action": {"$ref": "#/$defs/Action", "description": "执行的具体动作和目标"}, "action_after_effects": {"type": "array", "description": "动作成功执行后的影响列表", "items": {"$ref": "#/$defs/ActionEffect"}}, "nl_explanation": {"type": "string", "description": "对该步骤的自然语言总结"}}, "required": ["step_id", "step_goal", "action_before_state", "action_preconditions", "action", "action_after_effects", "nl_explanation"]}
}

FINAL_SCHEMA = {
    "type": "object",
    "description": "A single UI automation task object.",
    "properties": SCHEMA_CONTENT,
    "required": list(SCHEMA_CONTENT.keys()),
    "$defs": DEFINITIONS_CONTENT
}

def read_topic_from_file(file_path: str) -> List[Tuple[str, str]]:
    p = Path(file_path)
    text = p.read_text(encoding="utf-8")
    clean = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    data = json.loads(clean)
    result: List[Tuple[Any, str]] = []
    for item in data:
        tid = item.get("task_id")
        instr = item.get("instruction", "")
        result.append((str(tid), instr))
    return result


def generate_and_save_one_task_data(topic: str, task_id: str):
    prompt = generate_task_data_prompt.replace("{JSON_SCHEMA}", json.dumps(FINAL_SCHEMA, indent=4))
    prompt = prompt.replace("{TOPIC}", topic).replace("{TASK_ID}", task_id)
    try: 
        completion = get_qwen_client().chat.completions.create(
            model="qwen-plus",
            messages=[
                        {"role": "system", "content": "You are a strict evaluator. Output must be raw JSON only."},
                        {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        try:
            filePath = f"test/{task_id}.json"
            parsed = json.loads(completion.choices[0].message.content)
            with open(filePath, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=4, ensure_ascii=False)
        except json.JSONDecodeError as e:
            print("Received invalid JSON:", e)
            with open(filePath, "w", encoding="utf-8") as f:
                f.write(completion.choices[0].message.content)
        return
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return 
 

def generate_task_data() -> str:
    topics = read_topic_from_file("test/topic.json")
    total = len(topics)
    if total == 0:
        print("No topics found.")
        return ""

    bar_len = 40
    for idx, (tid, topic) in enumerate(topics, start=1):
        print(f"Generating task for Topic: {topic} (TaskId: {tid})")
        generate_and_save_one_task_data(topic, tid)
        filled = int(bar_len * idx / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\rProgress: [{bar}] {idx}/{total}", end="", flush=True)
        print()

    print()  
    return ""

def read_task_info(taskId: int) -> Optional[Task]:
    """
    读取 test/{taskId}.json 并返回 Task 对象（若出错返回 None）。
    """
    p = Path("test") / f"{taskId}.json"
    if not p.exists():
        print(f"file not found: {p}")
        return None
    try:
        raw = p.read_text(encoding="utf-8")
        obj = json.loads(raw)
        task = _dict_to_task(obj)
        return task
    except Exception as e:
        print(f"read_task_info error: {e}")
        return None
    
    
def read_step_info(task_id: int, step_id: str) -> Optional[Step]:
    """
    返回指定 task 的第 stepIndex 步（1-based）。越界或出错返回 None。
    """
    task = read_task_info(task_id)
    if task is None:
        return None
    for step in task.steps:
        if step.step_id == step_id:
            return step
    return None

def read_all_task_id() -> List[int]:
    """
    读取 test/ 目录下所有 taskId 列表。
    """
    p = Path("test")
    task_ids = []
    for file in p.glob("*.json"):
        try:
            name = file.stem
            tid = int(name)
            task_ids.append(tid)
        except ValueError:
            continue
    return task_ids

def read_all_step_id(task_id: str)-> List[str]:
    """
    读取指定 task_id 下所有 step_id 列表。
    """
    task = read_task_info(int(task_id))
    if task is None:
        return []
    step_ids = [step.step_id for step in task.steps if step is not None]
    return step_ids


def is_test_split(key: str, test_ratio: float, seed: str) -> bool:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < test_ratio


def evaluate_k(
    k: float,
    report_root: Path,
    cases_root: Path,
    test_ratio: float = 1.0,
    seed: str = "42",
    limit_steps: Optional[int] = None,
    *,
    embedding_model: str = "text-embedding-v4",
) -> Dict[str, object]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []

    embedding_cache: Dict[str, List[float]] = {}

    processed = 0
    for step in iter_report_steps(report_root):
        processed += 1
        if limit_steps is not None and processed > limit_steps:
            break

        key = f"{step['task_id']}:{step['step_id']}"
        if test_ratio < 1.0 and not is_test_split(key, test_ratio=test_ratio, seed=seed):
            continue

        cases = read_step_train_cases_for_k(
            cases_root,
            step["task_id"],
            step["step_id"],
            embedding_model=str(embedding_model),
        )
        if not cases:
            continue

        anchor = step["action_before_state_embedding"]
        vec_a = np.asarray(anchor, dtype=np.float32)

        def score_text(text: str, embedding: Optional[List[float]]) -> float:
            if embedding is None:
                if text not in embedding_cache:
                    emb = get_embedding(text, model=str(embedding_model))
                    if not emb:
                        return 0.0
                    embedding_cache[text] = emb
                embedding = embedding_cache[text]
            vec_b = np.asarray(embedding, dtype=np.float32)
            return cosine_similarity_01(vec_a, vec_b)

        for item in cases.get("positives", []):
            sim = score_text(str(item["text"]), item.get("embedding"))
            y_true.append(1)
            y_scores.append(sim)
            y_pred.append(1 if sim >= k else 0)

        for item in cases.get("hard_negatives", []):
            sim = score_text(str(item["text"]), item.get("embedding"))
            y_true.append(0)
            y_scores.append(sim)
            y_pred.append(1 if sim >= k else 0)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) else 0.0

    return {
        "k": float(k),
        "n_pairs": int(len(y_true)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {"precision": precision, "recall": recall, "f0.5": f05},
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument(
        "--k-file",
        type=str,
        default="Ambler-Agent/trajectory/retrieval/step_retrieval/train_k-result/k.json",
    )
    parser.add_argument("--embedding-model", type=str, default="text-embedding-v4")
    parser.add_argument(
        "--report-root",
        type=str,
        default="Ambler-Agent/trajectory/trajectory_base",
        help="Directory containing */report.json",
    )
    parser.add_argument(
        "--cases-root",
        type=str,
        default="Ambler-Agent/trajectory/retrieval/step_retrieval/train_k-data",
    )
    parser.add_argument("--test-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default="Ambler-Agent/trajectory/retrieval/step_retrieval/train_k-result/k_eval.json",
    )
    args = parser.parse_args(argv)

    k = args.k
    embedding_model = str(args.embedding_model)
    if k is None:
        k_data = json.loads(Path(args.k_file).read_text(encoding="utf-8"))
        k = float(k_data.get("optimal_k"))
        if (args.embedding_model == "text-embedding-v4") and ("embedding_model" in k_data):
            embedding_model = str(k_data.get("embedding_model") or embedding_model)

    result = evaluate_k(
        k=float(k),
        report_root=Path(args.report_root),
        cases_root=Path(args.cases_root),
        test_ratio=float(args.test_ratio),
        seed=str(args.seed),
        limit_steps=args.limit_steps,
        embedding_model=embedding_model,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"k={result['k']:.4f} metrics={result['metrics']} n_pairs={result['n_pairs']}")
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
