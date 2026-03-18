"""
本模块用于通过语义相似度匹配历史轨迹中的步骤。
它包含了预加载步骤索引、计算余弦相似度以及根据输入描述搜索相似步骤的功能。
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from ..common_llm_call import get_embedding, llm_judge_step_precondition
    from ..common_retrieval_tools import save_json_compact_embeddings
except ImportError:
    _parent_dir = str(Path(__file__).parent.parent)
    if _parent_dir not in sys.path:
        sys.path.append(_parent_dir)
    from common_llm_call import get_embedding, llm_judge_step_precondition
    from common_retrieval_tools import save_json_compact_embeddings

_step_index: Optional[List[Dict[str, Any]]] = None

def _load_step_index():
    global _step_index
    if _step_index is not None:
        return _step_index

    _step_index = []
    trajectory_base_dir = Path(__file__).parent.parent.parent / "trajectory_base"
    
    report_paths = list(trajectory_base_dir.glob("*/report.json"))
    
    for report_path in report_paths:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            task_id = data.get("task_id")
            steps = data.get("steps", [])
            report_modified = False

            for step in steps:
                embedding = step.get("action_before_state_embedding")
                action_before_state = step.get("action_before_state")

                if (
                    (not isinstance(embedding, list) or len(embedding) == 0)
                    and isinstance(action_before_state, str)
                    and action_before_state.strip()
                ):
                    embedding = get_embedding(action_before_state)
                    if isinstance(embedding, list) and embedding:
                        step["action_before_state_embedding"] = embedding
                        report_modified = True

                if isinstance(embedding, list) and embedding:
                    _step_index.append({
                        "report_path": str(report_path),
                        "task_id": task_id,
                        "step_id": step.get("step_id"),
                        "action_before_state": action_before_state,
                        "action_preconditions": step.get("action_preconditions", []),
                        "embedding": embedding
                    })

            if report_modified:
                save_json_compact_embeddings(data, str(report_path))
        except Exception as e:
            print(f"加载 {report_path} 时出错: {e}")
            
    print(f"已加载 {len(_step_index)} 条步骤索引。")
    return _step_index

def cal_step_similarity(step_embedding: List[float], explanation_embedding: List[float]) -> float:
    if len(step_embedding) != len(explanation_embedding):
        print("embedding length mismatch.")
        return 0.0
    
    try:
        dot_product = sum(a*b for a, b in zip(step_embedding, explanation_embedding))
        norm_step = sum(a*a for a in step_embedding) ** 0.5
        norm_explanation = sum(b*b for b in explanation_embedding) ** 0.5

        if norm_step == 0.0 or norm_explanation == 0.0:
            return 0.0
            
        similarity = dot_product / (norm_step * norm_explanation)
        similarity = max(-1.0, min(1.0, similarity))
        similarity = (similarity + 1) / 2
        return similarity
    except Exception as e:
        print(f"计算相似度时出错：{e}")
    return 0.0


def find_step_by_similarity(
    retrieval_query: str,
    screen_evidence: Optional[str] = None,
    k: float = 0.8,
):
    print(f"🔍 正在搜索与 '{retrieval_query}' 相似的步骤...")

    query_embedding = get_embedding(retrieval_query)
    if not query_embedding:
        print("无法获取查询文本的 embedding。")
        return None

    screen_evidence = screen_evidence or retrieval_query

    index = _load_step_index()
    if not index:
        print("步骤索引为空。")
        return None

    results = []
    for item in index:
        similarity = cal_step_similarity(item["embedding"], query_embedding)
        
        results.append({
            "report_path": item["report_path"],
            "task_id": item["task_id"],
            "step_id": item["step_id"],
            "similarity": similarity,
            "action_before_state": item["action_before_state"],
            "action_preconditions": item["action_preconditions"]
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    if not results:
        print("未找到可用步骤。")
        return None

    top_k = results[:10]
    above_threshold = [r for r in top_k if r["similarity"] >= k]
    candidates = above_threshold if above_threshold else top_k[:5]

    print(f"找到 {len(candidates)} 条候选步骤（阈值 k={k}, top={len(candidates)}）:")
    for item in candidates:
        print(f"- [相似度: {item['similarity']:.4f}]")
        print(f"  路径: {item['report_path']}")
        print(f"  Task ID: {item['task_id']}, Step ID: {item['step_id']}")
        print(f"  状态描述: {item['action_before_state']}")
        print("-" * 20)

    print("\n🧶 正在对候选步骤进行 LLM 前置条件判断...")
    best_step = None
    best_similarity = -1.0
    precondition_satisfied = False
    
    for item in candidates:
        if llm_judge_step_precondition(
            item["action_preconditions"],
            screen_evidence,
            item.get("action_before_state", "") or "",
        ):
            if item["similarity"] > best_similarity:
                best_similarity = item["similarity"]
                best_step = item
                precondition_satisfied = True
            
    if not best_step:
        print("🌵 没有步骤满足前置条件，返回 None。")
        return None
    
    similarity_above_k = best_step["similarity"] >= k
    print(f"🌵 找到相似度最高且满足前置条件的步骤:")
    print(f"- [相似度: {best_step['similarity']:.4f}] {'(≥k)' if similarity_above_k else '(<k)'}")
    print(f"  Task ID: {best_step['task_id']}, Step ID: {best_step['step_id']}")
    print(f"  状态描述: {best_step['action_before_state']}")
    print("-" * 20)
        
    full_step_data = None
    try:
        with open(best_step["report_path"], 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        steps = report_data.get("steps", [])
        for step in steps:
            if step.get("step_id") == best_step["step_id"]:
                full_step_data = step
                break
    except Exception as e:
        print(f"读取完整步骤信息时出错: {e}")
    
    return {
        "report_path": best_step.get("report_path"),
        "task_id": best_step.get("task_id"),
        "step_id": best_step.get("step_id"),
        "similarity": best_step.get("similarity"),
        "precondition_satisfied": precondition_satisfied,
        "similarity_above_k": similarity_above_k,
        "full_step_data": full_step_data,
    }
    
if __name__ == "__main__":
    find_step_by_similarity("Table.", screen_evidence="Table.", k=0.64)
