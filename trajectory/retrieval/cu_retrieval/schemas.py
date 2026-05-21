from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


CandidateSource = Literal["state_similarity", "transition", "both"]


@dataclass
class StateCandidate:
    cu_id: str
    canonical_unit: Dict[str, Any]
    similarity_score: float
    source: Literal["state_similarity"] = "state_similarity"


@dataclass
class TransitionCandidate:
    cu_id: str
    canonical_unit: Dict[str, Any]
    transition_count: int
    source: Literal["transition"] = "transition"


@dataclass
class MergedCandidate:
    cu_id: str
    canonical_unit: Dict[str, Any]
    source: CandidateSource
    similarity_score: Optional[float] = None
    transition_count: Optional[int] = None


@dataclass
class RetrievalResult:
    query_text: str
    state_candidates: List[StateCandidate] = field(default_factory=list)
    transition_candidates: List[TransitionCandidate] = field(default_factory=list)
    merged_candidates: List[MergedCandidate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
