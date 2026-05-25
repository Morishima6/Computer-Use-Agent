"""
Train the similarity threshold `k` for step retrieval.

This script uses the existing trajectory reports under:
`Ambler-Agent/trajectory/trajectory_base/*/report.json`.

Workflow:
1) (Optional) Generate per-step augmentation cases via LLM (positives/hard_negatives).
2) Build a labeled dataset from those cases.
3) Compute cosine similarities and pick an optimal threshold k.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve

try:
    from ..common_llm_call import get_qwen_client, get_embedding
except ImportError:
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from common_llm_call import get_qwen_client, get_embedding

generate_train_case_prompt = """
You are a QA Data Augmentation Specialist. I will provide you with a UI description (action_before_state) from a software test step.
Your task is to generate training data for a semantic similarity model.

Input Description:
"{DESC}"

Please generate a JSON object with the following fields:
1. "positives": A list of 2 strings. Rewrite the input description using different vocabulary, sentence structures, or summary styles, but KEEP the functional meaning and UI state exactly the same.
2. "hard_negatives": A list of 2 strings. Subtly modify the input description to change a CRITICAL state (e.g., change 'enabled' to 'disabled', 'checked' to 'unchecked', 'visible' to 'hidden', or change the specific text content of a target element). The result must look very similar to the original but represent a functionally different state.

Output strictly valid JSON only:
{{
    "positives": ["...", "..."],
    "hard_negatives": ["...", "..."]
}}
"""

_NUMERIC_ARRAY_PATTERN = re.compile(r'("([^"]+)":\s*\[)([0-9eE\+\-\.,\s]+)(\])')


def _json_dumps_compact_numeric_arrays(data: object, *, indent: int = 4) -> str:
    """
    Like json.dumps(indent=...), but keeps numeric arrays in a single line.
    Intended for embedding vectors stored in JSON.
    """
    text = json.dumps(data, indent=indent, ensure_ascii=False)

    def collapse_array(match: re.Match[str]) -> str:
        prefix = match.group(1)
        content = match.group(3)
        suffix = match.group(4)
        collapsed = re.sub(r"\s+", " ", content).strip()
        return f"{prefix}{collapsed}{suffix}"

    return _NUMERIC_ARRAY_PATTERN.sub(collapse_array, text)


def _has_multiline_numeric_arrays(text: str) -> bool:
    # Detect patterns like: "text-embedding-v4": [\n 0.1, ...]
    return bool(re.search(r'"\w[^"]*":\s*\[\s*\n\s*[-0-9]', text))


def llm_generate_train_case_for_k(desc: str) -> Optional[str]:
    prompt = generate_train_case_prompt.replace("{DESC}", desc)
    try:
        completion = get_qwen_client().chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Output must be raw JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None
    
    
def iter_report_paths(report_root: Path) -> List[Path]:
    return sorted(report_root.glob("*/report.json"))


def iter_report_steps(report_root: Path) -> Iterable[Dict]:
    for report_path in iter_report_paths(report_root):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        task_id = data.get("task_id") or report_path.parent.name
        steps = data.get("steps", []) or []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_id = step.get("step_id")
            action_before_state = step.get("action_before_state")
            step_embedding = step.get("action_before_state_embedding")
            if not step_id or not action_before_state or not step_embedding:
                continue
            yield {
                "task_id": str(task_id),
                "step_id": str(step_id),
                "action_before_state": str(action_before_state),
                "action_before_state_embedding": step_embedding,
                "report_path": str(report_path),
            }


def generate_train_cases_for_reports(
    report_root: Path,
    cases_root: Path,
    limit_steps: Optional[int] = None,
    overwrite: bool = False,
) -> int:
    created = 0
    processed = 0
    for step in iter_report_steps(report_root):
        processed += 1
        if limit_steps is not None and processed > limit_steps:
            break

        task_id = step["task_id"]
        step_id = step["step_id"]
        out_dir = cases_root / f"ask_{task_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{step_id}_train_case.json"
        if out_path.exists() and not overwrite:
            continue

        train_case_text = llm_generate_train_case_for_k(step["action_before_state"])
        if not train_case_text:
            continue
        try:
            parsed = json.loads(train_case_text)
        except json.JSONDecodeError:
            continue

        out_path.write_text(
            json.dumps(parsed, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        created += 1
    return created
    

def read_step_train_cases_for_k(
    cases_root: Path,
    task_id: str,
    step_id: str,
    *,
    embedding_model: str = "text-embedding-v4",
) -> Optional[Dict[str, List[Dict[str, object]]]]:
    task_dir = cases_root / f"ask_{task_id}"
    file_path = task_dir / f"{step_id}_train_case.json"

    if not file_path.exists():
        return None

    text = file_path.read_text(encoding="utf-8").strip()
    if not text or text.lower() == "null":
        return None
    format_dirty = _has_multiline_numeric_arrays(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    chosen_key = str(embedding_model)
    dirty = bool(format_dirty)

    def _ensure_embeddings_dict(item: Dict[str, Any]) -> Dict[str, List[float]]:
        nonlocal dirty
        emb = item.get("embeddings")
        if isinstance(emb, dict):
            return emb  # type: ignore[return-value]
        dirty = True
        return {}

    def _maybe_promote_legacy_embedding(item: Dict[str, Any], embeddings: Dict[str, List[float]]) -> None:
        nonlocal dirty
        legacy = item.get("embedding")
        if (
            chosen_key not in embeddings
            and isinstance(legacy, list)
            and all(isinstance(v, (int, float)) for v in legacy)
        ):
            embeddings[chosen_key] = legacy  # type: ignore[assignment]
            dirty = True

    def _normalize_item(item: object) -> Dict[str, object]:
        nonlocal dirty
        if isinstance(item, dict):
            text_value = item.get("text") or item.get("content") or ""
            out: Dict[str, Any] = dict(item)
            out["text"] = str(text_value)
            if out.get("text") != item.get("text"):
                dirty = True
        else:
            out = {"text": str(item)}
            dirty = True

        embeddings = _ensure_embeddings_dict(out)
        _maybe_promote_legacy_embedding(out, embeddings)

        if chosen_key not in embeddings:
            emb = get_embedding(str(out.get("text", "")), model=embedding_model)
            if emb:
                embeddings[chosen_key] = emb
                dirty = True

        out["embeddings"] = embeddings
        return {"text": str(out.get("text", "")), "embedding": embeddings.get(chosen_key), "embeddings": embeddings}

    def normalize_items(x: object) -> List[Dict[str, object]]:
        nonlocal dirty
        if x is None:
            return []
        if isinstance(x, list):
            items = x
        else:
            items = [x]

        normalized: List[Dict[str, object]] = []
        updated_items: List[Dict[str, Any]] = []
        for item in items:
            norm = _normalize_item(item)
            if norm.get("text"):
                normalized.append(norm)
                updated_items.append(
                    {
                        "text": norm["text"],
                        "embeddings": norm.get("embeddings") or {},
                    }
                )
        if updated_items != items:
            dirty = True
        return normalized

    result = {
        "positives": normalize_items(parsed.get("positives")),
        "hard_negatives": normalize_items(parsed.get("hard_negatives")),
    }
    if dirty:
        out_payload = {
            "positives": [{"text": i["text"], "embeddings": i.get("embeddings") or {}} for i in result["positives"]],
            "hard_negatives": [
                {"text": i["text"], "embeddings": i.get("embeddings") or {}}
                for i in result["hard_negatives"]
            ],
        }
        file_path.write_text(
            _json_dumps_compact_numeric_arrays(out_payload, indent=4),
            encoding="utf-8",
        )

    return result

    
def cosine_similarity_01(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    return (cosine + 1.0) / 2.0


def construct_dataset_for_k(
    report_root: Path,
    cases_root: Path,
    limit_steps: Optional[int] = None,
    *,
    embedding_model: str = "text-embedding-v4",
) -> List[Tuple[List[float], str, Optional[List[float]], int]]:
    """
    Returns:
        pairs: List[(anchor_embedding, comparison_text, comparison_embedding, label)]
            label 1 = Positive, 0 = Negative
    """
    pairs: List[Tuple[List[float], str, Optional[List[float]], int]] = []
    processed = 0
    for step in iter_report_steps(report_root):
        processed += 1
        if limit_steps is not None and processed > limit_steps:
            break

        task_id = step["task_id"]
        step_id = step["step_id"]
        train_case = read_step_train_cases_for_k(
            cases_root,
            task_id,
            step_id,
            embedding_model=embedding_model,
        )
        if not train_case:
            continue

        anchor_emb = step["action_before_state_embedding"]
        for pos in train_case.get("positives", []):
            pairs.append((anchor_emb, str(pos["text"]), pos.get("embedding"), 1))
        for neg in train_case.get("hard_negatives", []):
            pairs.append((anchor_emb, str(neg["text"]), neg.get("embedding"), 0))

    return pairs

    
    
def train_k(
    report_root: Path,
    cases_root: Path,
    target_precision: float = 0.99,
    limit_pairs: Optional[int] = None,
    limit_steps: Optional[int] = None,
    *,
    embedding_model: str = "text-embedding-v4",
) -> Dict:
    """
    计算所有样本的相似度，并寻找满足目标 Precision 的最佳阈值 k。
    Args:
        target_precision: 期望的最低查准率 (默认 0.99)
    Returns:
        Dict: 包含最佳 k 值和对应的评估指标
    """
    chosen_key = str(embedding_model)
    pairs = construct_dataset_for_k(
        report_root,
        cases_root,
        limit_steps=limit_steps,
        embedding_model=embedding_model,
    )
    if limit_pairs is not None:
        pairs = pairs[:limit_pairs]
    if not pairs:
        return {}
    
    y_true: List[int] = []
    y_scores: List[float] = []
    embedding_cache: Dict[str, List[float]] = {}

    for anchor_emb_list, compare_text, compare_emb_list, label in pairs:
        if compare_emb_list is None:
            if compare_text not in embedding_cache:
                emb = get_embedding(compare_text, model=embedding_model)
                if not emb:
                    continue
                embedding_cache[compare_text] = emb
            compare_emb_list = embedding_cache[compare_text]

        vec_a = np.asarray(anchor_emb_list, dtype=np.float32)
        vec_b = np.asarray(compare_emb_list, dtype=np.float32)
        sim = cosine_similarity_01(vec_a, vec_b)
        y_true.append(int(label))
        y_scores.append(float(sim))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 2. 计算 Precision-Recall 曲线
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    best_k = 0.85  # 默认初始值
    best_f05 = 0
    best_metrics = {}
    
    best_target_k = None
    best_target_metrics = None
    
    # 3. 遍历寻找最佳点
    # thresholds 的长度比 precisions 少 1
    for i in range(len(thresholds)):
        k = thresholds[i]
        p = precisions[i]
        r = recalls[i]
        
        # 核心逻辑：F0.5 Score (偏向 Precision)
        # F0.5 = (1.25 * P * R) / (0.25 * P + R)
        if p + r == 0:
            f05 = 0
        else:
            f05 = (1.25 * p * r) / (0.25 * p + r)
        
        # 记录 F0.5 最高点
        if f05 > best_f05:
            best_f05 = f05
            best_k = k
            best_metrics = {'precision': p, 'recall': r, 'f0.5': f05}
            
        # 优先满足硬性 Precision 指标
        if p >= target_precision:
            if best_target_metrics is None or r > best_target_metrics["recall"]:
                best_target_k = k
                best_target_metrics = {"precision": p, "recall": r, "f0.5": f05}

    chosen_k = float(best_k)
    chosen_metrics = best_metrics
    if best_target_k is not None and best_target_metrics is not None:
        chosen_k = float(best_target_k)
        chosen_metrics = best_target_metrics

    return {
        "optimal_k": chosen_k,
        "metrics": chosen_metrics,
        "n_pairs": int(len(y_true)),
        "embedding_model": str(embedding_model),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
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
        help="Directory to read/write per-step train cases",
    )
    parser.add_argument("--generate-cases", action="store_true")
    parser.add_argument("--overwrite-cases", action="store_true")
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--embedding-model", type=str, default="text-embedding-v4")
    parser.add_argument(
        "--only-materialize-embeddings",
        action="store_true",
        help="Only write embeddings into train_case.json then exit (no threshold training).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Ambler-Agent/trajectory/retrieval/step_retrieval/train_k-result/k.json",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-test-ratio", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=str, default="42")
    parser.add_argument(
        "--eval-out",
        type=str,
        default="Ambler-Agent/trajectory/retrieval/step_retrieval/train_k-result/k_eval.json",
    )

    args = parser.parse_args(argv)
    report_root = Path(args.report_root)
    cases_root = Path(args.cases_root)
    embedding_model = str(args.embedding_model)

    if args.generate_cases:
        created = generate_train_cases_for_reports(
            report_root=report_root,
            cases_root=cases_root,
            limit_steps=args.limit_steps,
            overwrite=args.overwrite_cases,
        )
        print(f"generated_cases={created}")

    if args.only_materialize_embeddings:
        _ = construct_dataset_for_k(
            report_root=report_root,
            cases_root=cases_root,
            limit_steps=args.limit_steps,
            embedding_model=embedding_model,
        )
        print("materialized_embeddings=done")
        return 0

    result = train_k(
        report_root=report_root,
        cases_root=cases_root,
        target_precision=float(args.target_precision),
        limit_pairs=args.limit_pairs,
        limit_steps=args.limit_steps,
        embedding_model=embedding_model,
    )
    if not result:
        print("no_training_pairs")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"optimal_k={result['optimal_k']:.4f} metrics={result['metrics']} n_pairs={result['n_pairs']}")
    print(f"saved={out_path}")

    if args.eval:
        try:
            from .test_k import evaluate_k
        except ImportError:
            from test_k import evaluate_k

        eval_result = evaluate_k(
            k=float(result["optimal_k"]),
            report_root=report_root,
            cases_root=cases_root,
            test_ratio=float(args.eval_test_ratio),
            seed=str(args.eval_seed),
            limit_steps=args.limit_steps,
        )
        eval_out_path = Path(args.eval_out)
        eval_out_path.parent.mkdir(parents=True, exist_ok=True)
        eval_out_path.write_text(
            json.dumps(eval_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"eval_metrics={eval_result['metrics']} eval_pairs={eval_result['n_pairs']}")
        print(f"eval_saved={eval_out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
