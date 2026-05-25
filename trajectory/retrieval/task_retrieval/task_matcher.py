import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trajectory.retrieval.common_llm_call import get_embedding


task_index: Optional[List[Dict[str, Any]]] = None


def _normalize_os_name(os_name: Optional[str]) -> str:
    if not os_name:
        return ""
    name = str(os_name).strip().lower()
    aliases = {
        "mac": "darwin",
        "macos": "darwin",
        "osx": "darwin",
        "darwin": "darwin",
        "linux": "linux",
        "windows": "windows",
        "win": "windows",
    }
    return aliases.get(name, name)


def cosine_similarity_01(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (norm_a * norm_b)
    sim = max(-1.0, min(1.0, sim))
    return (sim + 1.0) / 2.0


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_task_index_from_trajectory_base(
    trajectory_base_dir: Path,
    *,
    persist_embeddings: bool = True,
    overwrite_embeddings: bool = False,
) -> List[Dict[str, Any]]:
    global task_index
    if task_index is not None:
        return task_index

    task_index = []
    report_paths = list(trajectory_base_dir.glob("*/report.json"))
    for report_path in report_paths:
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        instruction = data.get("instruction") or ""
        if not instruction:
            continue

        embedding = data.get("instruction_embedding")
        if overwrite_embeddings or not isinstance(embedding, list) or len(embedding) == 0:
            try:
                embedding = get_embedding(instruction)
            except Exception:
                embedding = None

            if persist_embeddings and isinstance(embedding, list) and embedding:
                data["instruction_embedding"] = embedding
                try:
                    _write_json(report_path, data)
                except Exception:
                    pass

        if not isinstance(embedding, list) or not embedding:
            continue

        task_index.append(
            {
                "report_path": str(report_path),
                "task_id": data.get("task_id"),
                "task_title": data.get("task_title"),
                "instruction": instruction,
                "app": data.get("app"),
                "env": data.get("env"),
                "embedding": embedding,
            }
        )

    return task_index


def is_task_compatible(task: Dict[str, Any], runtime_os: Optional[str]) -> bool:
    if not runtime_os:
        return True

    runtime = _normalize_os_name(runtime_os)

    env = task.get("env") or {}
    env_os = _normalize_os_name(env.get("os") if isinstance(env, dict) else None)
    if env_os and runtime and env_os != runtime:
        return False

    app = str(task.get("app") or "")
    if runtime != "linux" and ("ubuntu" in app.lower() or "gnome" in app.lower()):
        return False

    return True


class TaskMatcher:
    def __init__(self, trajectory_base_dir: Optional[str] = None):
        self.trajectory_base_dir = (
            Path(trajectory_base_dir) if trajectory_base_dir else None
        )
        self.db: List[Dict[str, Any]] = []

    def load_data(
        self,
        trajectory_base_dir: str,
        *,
        persist_embeddings: bool = True,
        overwrite_embeddings: bool = False,
    ) -> bool:
        self.trajectory_base_dir = Path(trajectory_base_dir)
        if not self.trajectory_base_dir.exists():
            return False
        self.db = load_task_index_from_trajectory_base(
            self.trajectory_base_dir,
            persist_embeddings=persist_embeddings,
            overwrite_embeddings=overwrite_embeddings,
        )
        return True

    def find_task(
        self,
        query: str,
        *,
        threshold: float = 0.7,
        runtime_os: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        if not query:
            return None, 0.0

        if not self.db:
            if self.trajectory_base_dir is None:
                return None, 0.0
            self.db = load_task_index_from_trajectory_base(self.trajectory_base_dir)

        query_embedding = get_embedding(query)
        if not isinstance(query_embedding, list) or not query_embedding:
            return None, 0.0

        best_task = None
        best_score = -1.0

        for task in self.db:
            if not is_task_compatible(task, runtime_os):
                continue
            score = cosine_similarity_01(task["embedding"], query_embedding)
            if score > best_score:
                best_score = score
                best_task = task

        if best_task is None:
            return None, 0.0

        if best_score < threshold:
            return None, best_score

        return best_task, best_score


if __name__ == "__main__":
    matcher = TaskMatcher()
    repo_root = Path(__file__).resolve().parents[2]
    trace_root = repo_root / "trajectory_base"
    ok = matcher.load_data(str(trace_root))
    print("loaded:", ok, "count:", len(matcher.db))
