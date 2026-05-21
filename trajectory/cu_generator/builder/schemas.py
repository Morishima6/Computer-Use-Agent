"""Shared data schemas for the Phase 3 builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass
class ParameterizedAction:
    """One action step inside a parameterized unit path."""

    action_type: str
    action_template: str
    raw_action: Any = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "ParameterizedAction":
        return cls(
            action_type=str(payload.get("action_type") or ""),
            action_template=str(payload.get("action_template") or ""),
            raw_action=payload.get("raw_action"),
        )


@dataclass
class ParameterizedUnitRecord:
    """Normalized Phase 2 unit record used as Phase 3 input."""

    instance_id: str
    param_unit_id: str
    unit_id: str
    segment_id: str
    source_user: str
    app_name: str
    unit_type: str
    abstract_intent: str
    unit_intent: str
    unit_before_state: str
    unit_after_state: str
    unit_precondition: List[str] = field(default_factory=list)
    unit_effect: List[str] = field(default_factory=list)
    parameters: JsonDict = field(default_factory=dict)
    parameterized_action_sequence: List[ParameterizedAction] = field(default_factory=list)
    step_indices: List[int] = field(default_factory=list)
    segment_order: int = 0
    trace_id: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    app_context: JsonDict = field(default_factory=dict)
    env: JsonDict = field(default_factory=dict)
    raw_segment_path: str = ""
    raw_payload: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "ParameterizedUnitRecord":
        actions = [
            ParameterizedAction.from_dict(action_payload)
            for action_payload in payload.get("parameterized_action_sequence", [])
            if isinstance(action_payload, dict)
        ]
        return cls(
            instance_id=str(payload.get("instance_id") or ""),
            param_unit_id=str(payload.get("param_unit_id") or ""),
            unit_id=str(payload.get("unit_id") or ""),
            segment_id=str(payload.get("segment_id") or ""),
            source_user=str(payload.get("source_user") or ""),
            app_name=str(payload.get("app_name") or ""),
            unit_type=str(payload.get("unit_type") or ""),
            abstract_intent=str(payload.get("abstract_intent") or ""),
            unit_intent=str(payload.get("unit_intent") or ""),
            unit_before_state=str(payload.get("unit_before_state") or ""),
            unit_after_state=str(payload.get("unit_after_state") or ""),
            unit_precondition=[str(value) for value in payload.get("unit_precondition", [])],
            unit_effect=[str(value) for value in payload.get("unit_effect", [])],
            parameters=dict(payload.get("parameters") or {}),
            parameterized_action_sequence=actions,
            step_indices=[int(value) for value in payload.get("step_indices", [])],
            segment_order=int(payload.get("segment_order") or 0),
            trace_id=payload.get("trace_id"),
            timestamp_start=payload.get("timestamp_start"),
            timestamp_end=payload.get("timestamp_end"),
            app_context=dict(payload.get("app_context") or {}),
            env=dict(payload.get("env") or {}),
            raw_segment_path=str(payload.get("raw_segment_path") or ""),
            raw_payload=dict(payload.get("raw_payload") or {}),
        )


@dataclass
class ClusterAssignment:
    """Grouping result for one semantic cluster."""

    cluster_id: str
    group_key: str
    app_name: str
    unit_type: str
    unit_instances: List[ParameterizedUnitRecord] = field(default_factory=list)
    centroid_instance_id: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "ClusterAssignment":
        instances = [
            ParameterizedUnitRecord.from_dict(unit_payload)
            for unit_payload in payload.get("unit_instances", [])
            if isinstance(unit_payload, dict)
        ]
        return cls(
            cluster_id=str(payload.get("cluster_id") or ""),
            group_key=str(payload.get("group_key") or ""),
            app_name=str(payload.get("app_name") or ""),
            unit_type=str(payload.get("unit_type") or ""),
            unit_instances=instances,
            centroid_instance_id=payload.get("centroid_instance_id"),
        )


@dataclass
class UnitTreeNode:
    """One node inside a Canonical Unit action tree."""

    node_id: str
    action_type: str
    description: str
    params: JsonDict = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    source_count: int = 0
    can_terminate: bool = False
    terminate_count: int = 0

    def to_dict(self) -> JsonDict:
        payload = {
            "type": self.action_type,
            "description": self.description,
            "params": self.params,
            "children": self.children,
            "source_count": self.source_count,
        }
        if self.can_terminate:
            payload["can_terminate"] = True
            payload["terminate_count"] = self.terminate_count
        return payload


@dataclass
class CanonicalUnitRecord:
    """Canonical Unit payload aligned with the runtime store contract."""

    cu_id: str
    intent: str
    intent_embedding_id: int
    unit_type: str
    abstract_state_before: str
    abstract_state_after: str
    unit_tree: JsonDict = field(default_factory=dict)
    parameter_defs: List[JsonDict] = field(default_factory=list)
    execution_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    path_stats: JsonDict = field(default_factory=dict)
    source_users: List[str] = field(default_factory=list)
    source_instance_ids: List[str] = field(default_factory=list)
    app_context: JsonDict = field(default_factory=dict)
    category: str = "Uncategorized"
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "CanonicalUnitRecord":
        return cls(
            cu_id=str(payload.get("cu_id") or ""),
            intent=str(payload.get("intent") or ""),
            intent_embedding_id=int(payload.get("intent_embedding_id") or 0),
            unit_type=str(payload.get("unit_type") or ""),
            abstract_state_before=str(payload.get("abstract_state_before") or ""),
            abstract_state_after=str(payload.get("abstract_state_after") or ""),
            unit_tree=dict(payload.get("unit_tree") or {}),
            parameter_defs=list(payload.get("parameter_defs") or []),
            execution_count=int(payload.get("execution_count") or 0),
            success_count=int(payload.get("success_count") or 0),
            success_rate=float(payload.get("success_rate") or 0.0),
            path_stats=dict(payload.get("path_stats") or {}),
            source_users=[str(value) for value in payload.get("source_users", [])],
            source_instance_ids=[str(value) for value in payload.get("source_instance_ids", [])],
            app_context=dict(payload.get("app_context") or {}),
            category=str(payload.get("category") or "Uncategorized"),
            first_seen=payload.get("first_seen"),
            last_seen=payload.get("last_seen"),
        )


@dataclass
class Phase3Artifacts:
    """All in-memory outputs produced by the Phase 3 builder."""

    canonical_units: List[CanonicalUnitRecord] = field(default_factory=list)
    transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    instance_to_cu: Dict[str, str] = field(default_factory=dict)
    faiss_to_cu: Dict[str, str] = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    config: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "canonical_units": [unit.to_dict() for unit in self.canonical_units],
            "transitions": self.transitions,
            "instance_to_cu": self.instance_to_cu,
            "faiss_to_cu": self.faiss_to_cu,
            "metadata": self.metadata,
            "config": self.config,
        }


__all__ = [
    "JsonDict",
    "ParameterizedAction",
    "ParameterizedUnitRecord",
    "ClusterAssignment",
    "UnitTreeNode",
    "CanonicalUnitRecord",
    "Phase3Artifacts",
]
