from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import StateCandidate, TransitionCandidate

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

try:
    from ..common_llm_call import get_embedding
except ImportError:
    get_embedding = None


class CUStore:
    def __init__(self, cu_base_root: Path, embedding_model: str = "Qwen/Qwen3-Embedding-8B"):
        self.cu_base_root = Path(cu_base_root)
        self.embedding_model = embedding_model

        self.metadata: Dict[str, Any] = {}
        self.canonical_units: List[Dict[str, Any]] = []
        self.cu_by_id: Dict[str, Dict[str, Any]] = {}
        self.transitions: Dict[str, Dict[str, int]] = {}
        self.instance_to_cu: Dict[str, str] = {}
        self.faiss_to_cu: Dict[str, str] = {}
        self.faiss_index = None
        self.warnings: List[str] = []

        self._load()

    def _warn_once(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _load(self) -> None:
        # CU 运行时依赖 4 份持久化产物：
        # 1) canonical units
        # 2) transition 图
        # 3) instance/id 映射
        # 4) 用于状态相似检索的 FAISS 索引（可选）
        cu_base_path = self.cu_base_root / "cu_base.json"
        transitions_path = self.cu_base_root / "transitions.json"
        mappings_path = self.cu_base_root / "mappings.json"
        faiss_path = self.cu_base_root / "faiss_intent.index"

        cu_base_payload = json.loads(cu_base_path.read_text(encoding="utf-8"))
        mappings_payload = json.loads(mappings_path.read_text(encoding="utf-8"))
        self.transitions = json.loads(transitions_path.read_text(encoding="utf-8"))

        self.metadata = cu_base_payload.get("metadata", {})
        self.canonical_units = list(cu_base_payload.get("canonical_units", []))
        self.cu_by_id = {
            str(unit["cu_id"]): unit
            for unit in self.canonical_units
            if isinstance(unit, dict) and unit.get("cu_id")
        }
        self.instance_to_cu = dict(mappings_payload.get("instance_to_cu", {}))
        self.faiss_to_cu = dict(mappings_payload.get("faiss_to_cu", {}))

        if faiss is None:
            self._warn_once("faiss is not installed; state similarity retrieval is unavailable.")
            return

        if not faiss_path.exists():
            self._warn_once(f"FAISS index not found: {faiss_path}")
            return

        self.faiss_index = faiss.read_index(str(faiss_path))

    def get_cu(self, cu_id: str) -> Optional[Dict[str, Any]]:
        return self.cu_by_id.get(str(cu_id))

    def get_transition_candidates(
        self, cu_id: Optional[str], top_k: int = 5
    ) -> List[TransitionCandidate]:
        # Transition 检索通道回答的问题是：
        # “上一个 CU 执行完之后，通常下一个会接什么 CU？”
        # 它和基于当前界面相似度的检索通道是独立的。
        if not cu_id:
            return []

        outgoing = self.transitions.get(str(cu_id), {})
        ranked_pairs = sorted(outgoing.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: List[TransitionCandidate] = []
        for next_cu_id, transition_count in ranked_pairs:
            canonical_unit = self.get_cu(next_cu_id)
            if canonical_unit is None:
                continue
            results.append(
                TransitionCandidate(
                    cu_id=next_cu_id,
                    canonical_unit=canonical_unit,
                    transition_count=int(transition_count),
                )
            )
        return results

    def search_by_text(self, query_text: str, top_k: int = 5) -> List[StateCandidate]:
        # State 检索通道会先把当前 query 文本做 embedding，
        # 再去搜索由 canonical intent/state 构建好的 FAISS 索引。
        if not query_text or not str(query_text).strip():
            return []
        if get_embedding is None:
            self._warn_once("Embedding function is unavailable; state similarity retrieval is unavailable.")
            return []

        embedding = get_embedding(query_text, model=self.embedding_model)
        if not isinstance(embedding, list) or not embedding:
            self._warn_once("Failed to generate query embedding for CU retrieval.")
            return []

        return self.search_by_embedding(embedding, top_k=top_k)

    def search_by_embedding(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[StateCandidate]:
        if self.faiss_index is None:
            return []
        if faiss is None or np is None:
            return []
        if not query_embedding:
            return []

        # 这里先做归一化，再进行近似余弦相似度搜索；
        # 持久化索引构建时采用的是同一套约定。
        query = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self.faiss_index.search(query, top_k)

        results: List[StateCandidate] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            cu_id = self.faiss_to_cu.get(str(int(index)))
            if cu_id is None:
                continue
            canonical_unit = self.get_cu(cu_id)
            if canonical_unit is None:
                continue
            results.append(
                StateCandidate(
                    cu_id=cu_id,
                    canonical_unit=canonical_unit,
                    similarity_score=float(score),
                )
            )
        return results
