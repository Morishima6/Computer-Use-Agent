from __future__ import annotations

from typing import Dict, Optional

from .cu_store import CUStore
from .schemas import MergedCandidate, RetrievalResult


class CURetriever:
    def __init__(
        self,
        store: CUStore,
        *,
        state_top_k: int = 8,
        transition_top_k: int = 5,
    ):
        self.store = store
        self.state_top_k = state_top_k
        self.transition_top_k = transition_top_k

    def retrieve(
        self,
        query_text: str,
        *,
        last_selected_cu_id: Optional[str] = None,
    ) -> RetrievalResult:
        # 检索分成两条通道：
        # - state_candidates：和当前 UI / 任务状态相似的 CU
        # - transition_candidates：上一个 CU 之后高概率会接上的 CU
        state_candidates = self.store.search_by_text(query_text, top_k=self.state_top_k)
        transition_candidates = self.store.get_transition_candidates(
            last_selected_cu_id, top_k=self.transition_top_k
        )

        merged: Dict[str, MergedCandidate] = {}

        for candidate in state_candidates:
            merged[candidate.cu_id] = MergedCandidate(
                cu_id=candidate.cu_id,
                canonical_unit=candidate.canonical_unit,
                source="state_similarity",
                similarity_score=candidate.similarity_score,
            )

        for candidate in transition_candidates:
            existing = merged.get(candidate.cu_id)
            if existing is None:
                merged[candidate.cu_id] = MergedCandidate(
                    cu_id=candidate.cu_id,
                    canonical_unit=candidate.canonical_unit,
                    source="transition",
                    transition_count=candidate.transition_count,
                )
                continue

            existing.source = "both"
            existing.transition_count = candidate.transition_count

        # 排序时优先保留“双通道都支持”的候选，
        # 再用 similarity 和 transition count 做细粒度排序，
        # 给 planner 一个合并后的紧凑候选列表。
        merged_candidates = sorted(
            merged.values(),
            key=lambda item: (
                0 if item.source == "both" else 1,
                -(item.similarity_score if item.similarity_score is not None else -1.0),
                -(item.transition_count if item.transition_count is not None else -1),
                -int(item.canonical_unit.get("execution_count", 0) or 0),
                -float(item.canonical_unit.get("success_rate", 0.0) or 0.0),
            ),
        )

        return RetrievalResult(
            query_text=query_text,
            state_candidates=state_candidates,
            transition_candidates=transition_candidates,
            merged_candidates=merged_candidates,
            warnings=list(self.store.warnings),
        )
