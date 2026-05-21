"""Build Canonical Unit records from clustered parameterized units."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .schemas import (
    CanonicalUnitRecord,
    ClusterAssignment,
    ParameterizedUnitRecord,
    UnitTreeNode,
)

try:
    from ...retrieval.common_llm_call import get_embedding
except ImportError:
    get_embedding = None


DEFAULT_ACTION_MATCH_THRESHOLD = 0.85
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def build_canonical_units(
    clusters: Sequence[ClusterAssignment],
    *,
    action_match_threshold: float = DEFAULT_ACTION_MATCH_THRESHOLD,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[CanonicalUnitRecord], Dict[str, str]]:
    """Convert clusters into canonical units and instance-to-CU mappings."""

    canonical_units: List[CanonicalUnitRecord] = []
    instance_to_cu: Dict[str, str] = {}

    total_clusters = len(clusters)
    for index, cluster in enumerate(clusters):
        cu_id = f"cu_{index:05d}"
        canonical_unit = _build_single_canonical_unit(
            cluster,
            cu_id=cu_id,
            intent_embedding_id=index,
            action_match_threshold=action_match_threshold,
            embedding_model=embedding_model,
        )
        canonical_units.append(canonical_unit)

        for instance in cluster.unit_instances:
            instance_to_cu[instance.instance_id] = cu_id
        if progress_callback is not None:
            progress_callback(
                index + 1,
                total_clusters,
                f"{cluster.app_name}/{cluster.unit_type} -> {cu_id}",
            )

    return canonical_units, instance_to_cu


def _build_single_canonical_unit(
    cluster: ClusterAssignment,
    *,
    cu_id: str,
    intent_embedding_id: int,
    action_match_threshold: float,
    embedding_model: str,
) -> CanonicalUnitRecord:
    units = list(cluster.unit_instances)
    representative = _select_representative_unit(units)
    intent = _select_representative_intent(units)
    abstract_state_before = _build_representative_state(units, field_name="unit_before_state")
    abstract_state_after = _build_representative_state(units, field_name="unit_after_state")
    unit_tree, path_stats = _build_unit_tree_and_paths(
        units,
        action_match_threshold=action_match_threshold,
        embedding_model=embedding_model,
    )

    execution_count = len(units)
    success_count = execution_count
    success_rate = float(success_count / execution_count) if execution_count else 0.0

    return CanonicalUnitRecord(
        cu_id=cu_id,
        intent=intent,
        intent_embedding_id=intent_embedding_id,
        unit_type=cluster.unit_type,
        abstract_state_before=abstract_state_before,
        abstract_state_after=abstract_state_after,
        unit_tree=unit_tree,
        parameter_defs=_merge_parameter_defs(units),
        execution_count=execution_count,
        success_count=success_count,
        success_rate=success_rate,
        path_stats=path_stats,
        source_users=sorted({unit.source_user for unit in units}),
        source_instance_ids=[unit.instance_id for unit in units],
        app_context=dict(representative.app_context),
        category=_infer_category(representative.app_name),
        first_seen=_select_first_seen(units),
        last_seen=_select_last_seen(units),
    )


def _select_representative_unit(
    units: Sequence[ParameterizedUnitRecord],
) -> ParameterizedUnitRecord:
    ranked = sorted(
        units,
        key=lambda unit: (
            -len(unit.abstract_intent or ""),
            -len(unit.unit_intent or ""),
            -len(unit.unit_before_state or ""),
            -len(unit.unit_after_state or ""),
            unit.instance_id,
        ),
    )
    return ranked[0]


def _select_representative_intent(units: Sequence[ParameterizedUnitRecord]) -> str:
    if not units:
        return ""

    scored: List[Tuple[float, str]] = []
    candidates = [unit.abstract_intent or unit.unit_intent for unit in units]
    for candidate in candidates:
        normalized_candidate = _normalize_free_text(candidate)
        similarity_sum = 0.0
        for other in candidates:
            similarity_sum += _token_cosine_similarity(
                normalized_candidate,
                _normalize_free_text(other),
            )
        scored.append((similarity_sum, candidate))

    scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    return scored[0][1]


def _build_representative_state(
    units: Sequence[ParameterizedUnitRecord],
    *,
    field_name: str,
) -> str:
    if not units:
        return ""

    candidates = [str(getattr(unit, field_name, "") or "") for unit in units]
    best_text = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        normalized_candidate = _abstract_state_text(candidate)
        similarity_sum = 0.0
        for other in candidates:
            similarity_sum += _token_cosine_similarity(
                normalized_candidate,
                _abstract_state_text(other),
            )
        score = similarity_sum + min(len(normalized_candidate), 400) / 1000.0
        if score > best_score:
            best_score = score
            best_text = normalized_candidate
    return best_text


def _build_unit_tree_and_paths(
    units: Sequence[ParameterizedUnitRecord],
    *,
    action_match_threshold: float,
    embedding_model: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if not units:
        return {"root": None, "nodes": {}}, {}

    nodes: Dict[str, UnitTreeNode] = {}
    root_id = "root"
    nodes[root_id] = UnitTreeNode(
        node_id=root_id,
        action_type="ROOT",
        description="Canonical Unit root",
        children=[],
        source_count=len(units),
    )

    next_node_index = 1
    node_source_instance_ids: Dict[str, List[str]] = {root_id: []}
    for unit in _order_units_for_tree_insertion(units):
        actions = list(unit.parameterized_action_sequence or [])
        if not actions:
            nodes[root_id].can_terminate = True
            nodes[root_id].terminate_count += 1
            continue

        insertion = _find_best_anchor_for_path(
            nodes=nodes,
            actions=actions,
            action_match_threshold=action_match_threshold,
            embedding_model=embedding_model,
        )
        if insertion is None:
            matched_node_ids: List[str] = []
            current_node_id = root_id
            next_action_index = 0
        else:
            matched_node_ids = list(insertion[0])
            current_node_id = matched_node_ids[-1]
            next_action_index = len(matched_node_ids)

        for action, node_id in zip(actions, matched_node_ids):
            canonical_description = _canonicalize_action_template(action.action_template)
            _merge_action_into_node(nodes[node_id], action, canonical_description)
            nodes[node_id].source_count += 1
            node_source_instance_ids.setdefault(node_id, []).append(unit.instance_id)

        while next_action_index < len(actions):
            action = actions[next_action_index]
            canonical_description = _canonicalize_action_template(action.action_template)
            child_id = _find_best_matching_child(
                nodes=nodes,
                parent_id=current_node_id,
                action_type=action.action_type,
                action_description=canonical_description,
                action_match_threshold=action_match_threshold,
                embedding_model=embedding_model,
            )
            if child_id is None:
                child_id = f"a{next_node_index}"
                next_node_index += 1
                node = UnitTreeNode(
                    node_id=child_id,
                    action_type=action.action_type,
                    description=canonical_description,
                    params=_extract_action_params(action.action_template),
                    children=[],
                    source_count=0,
                )
                nodes[child_id] = node
                node_source_instance_ids[child_id] = []
                nodes[current_node_id].children.append(child_id)
            else:
                _merge_action_into_node(nodes[child_id], action, canonical_description)

            nodes[child_id].source_count += 1
            node_source_instance_ids.setdefault(child_id, []).append(unit.instance_id)
            current_node_id = child_id
            next_action_index += 1

        if current_node_id in nodes:
            nodes[current_node_id].can_terminate = True
            nodes[current_node_id].terminate_count += 1

    path_stats = _build_leaf_path_stats(nodes, root_id, node_source_instance_ids)

    return {
        "root": root_id,
        "nodes": {node_id: node.to_dict() for node_id, node in nodes.items()},
    }, path_stats


def _find_best_anchor_for_path(
    *,
    nodes: Dict[str, UnitTreeNode],
    actions: Sequence[Any],
    action_match_threshold: float,
    embedding_model: str,
) -> Tuple[List[str], float] | None:
    """Find the best existing node sequence that overlaps the start of a new path."""

    if not actions:
        return None

    node_depths = _compute_node_depths(nodes)
    best_match: Tuple[List[str], float] | None = None
    best_rank: Tuple[int, float, int, int, str] | None = None

    first_action = actions[0]
    first_description = _canonicalize_action_template(first_action.action_template)
    normalized_first_type = str(first_action.action_type or "UNKNOWN").upper()

    for node_id, node in nodes.items():
        if node_id == "root":
            continue
        if str(node.action_type or "UNKNOWN").upper() != normalized_first_type:
            continue

        first_score = _action_semantic_similarity(
            node.description,
            first_description,
            embedding_model=embedding_model,
        )
        if first_score < action_match_threshold:
            continue

        matched_node_ids, average_score = _match_existing_path_from_anchor(
            nodes=nodes,
            anchor_id=node_id,
            actions=actions,
            first_score=first_score,
            action_match_threshold=action_match_threshold,
            embedding_model=embedding_model,
        )
        if not matched_node_ids:
            continue

        last_node = nodes[matched_node_ids[-1]]
        rank = (
            len(matched_node_ids),
            average_score,
            node_depths.get(node_id, 0),
            int(last_node.source_count or 0),
            node_id,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_match = (matched_node_ids, average_score)

    return best_match


def _order_units_for_tree_insertion(
    units: Sequence[ParameterizedUnitRecord],
) -> List[ParameterizedUnitRecord]:
    return sorted(
        units,
        key=lambda unit: (
            -len(unit.parameterized_action_sequence or []),
            unit.source_user,
            unit.trace_id or "",
            unit.segment_order,
            unit.step_indices[0] if unit.step_indices else 0,
            unit.instance_id,
        ),
    )


def _match_existing_path_from_anchor(
    *,
    nodes: Dict[str, UnitTreeNode],
    anchor_id: str,
    actions: Sequence[Any],
    first_score: float,
    action_match_threshold: float,
    embedding_model: str,
) -> Tuple[List[str], float]:
    matched_node_ids = [anchor_id]
    scores = [first_score]
    current_node_id = anchor_id

    for action in actions[1:]:
        canonical_description = _canonicalize_action_template(action.action_template)
        child_id, score = _find_best_matching_child_with_score(
            nodes=nodes,
            parent_id=current_node_id,
            action_type=action.action_type,
            action_description=canonical_description,
            action_match_threshold=action_match_threshold,
            embedding_model=embedding_model,
        )
        if child_id is None:
            break
        matched_node_ids.append(child_id)
        scores.append(score)
        current_node_id = child_id

    average_score = sum(scores) / len(scores) if scores else 0.0
    return matched_node_ids, average_score


def _compute_node_depths(nodes: Dict[str, UnitTreeNode]) -> Dict[str, int]:
    depths: Dict[str, int] = {"root": 0}
    stack: List[Tuple[str, int]] = [("root", 0)]
    while stack:
        node_id, depth = stack.pop()
        node = nodes.get(node_id)
        if node is None:
            continue
        for child_id in reversed(node.children):
            if child_id in depths:
                continue
            depths[child_id] = depth + 1
            stack.append((child_id, depth + 1))
    return depths


def _merge_action_into_node(
    node: UnitTreeNode,
    action: Any,
    canonical_description: str,
) -> None:
    if len(canonical_description) > len(str(node.description or "")):
        node.description = canonical_description
    node.params = _merge_node_params(node.params, _extract_action_params(action.action_template))


def _build_leaf_path_stats(
    nodes: Dict[str, UnitTreeNode],
    root_id: str,
    node_source_instance_ids: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    path_stats: Dict[str, Dict[str, Any]] = {}
    path_counter = 1

    for path_node_ids in _iter_root_to_leaf_paths(nodes, root_id):
        if not path_node_ids:
            continue
        path_steps = _build_path_steps_from_node_ids(nodes, path_node_ids)
        leaf_id = path_node_ids[-1]
        source_instance_ids = list(dict.fromkeys(node_source_instance_ids.get(leaf_id, [])))
        path_count = len(source_instance_ids) or int(nodes[leaf_id].source_count or 0)
        path_id = f"path_{path_counter:03d}"
        path_counter += 1
        path_stats[path_id] = {
            "path_label": _build_path_label(path_steps),
            "path_count": path_count,
            "path_success": path_count,
            "path_success_rate": 1.0 if path_count else 0.0,
            "steps": path_steps,
            "source_instance_ids": source_instance_ids,
        }

    return path_stats


def _iter_root_to_leaf_paths(
    nodes: Dict[str, UnitTreeNode],
    root_id: str,
) -> List[List[str]]:
    paths: List[List[str]] = []

    def _visit(node_id: str, prefix: List[str]) -> None:
        node = nodes.get(node_id)
        if node is None:
            return
        if node_id != root_id:
            prefix = [*prefix, node_id]
        if not node.children:
            paths.append(prefix)
            return
        for child_id in node.children:
            _visit(child_id, prefix)

    _visit(root_id, [])
    return paths


def _find_best_matching_child(
    *,
    nodes: Dict[str, UnitTreeNode],
    parent_id: str,
    action_type: str,
    action_description: str,
    action_match_threshold: float,
    embedding_model: str,
) -> str | None:
    child_id, _score = _find_best_matching_child_with_score(
        nodes=nodes,
        parent_id=parent_id,
        action_type=action_type,
        action_description=action_description,
        action_match_threshold=action_match_threshold,
        embedding_model=embedding_model,
    )
    return child_id


def _find_best_matching_child_with_score(
    *,
    nodes: Dict[str, UnitTreeNode],
    parent_id: str,
    action_type: str,
    action_description: str,
    action_match_threshold: float,
    embedding_model: str,
) -> Tuple[str | None, float]:
    normalized_action_type = str(action_type or "UNKNOWN").upper()
    best_child_id: str | None = None
    best_score = -1.0

    for child_id in nodes[parent_id].children:
        child = nodes[child_id]
        if str(child.action_type or "UNKNOWN").upper() != normalized_action_type:
            continue
        score = _action_semantic_similarity(
            child.description,
            action_description,
            embedding_model=embedding_model,
        )
        if score > best_score:
            best_score = score
            best_child_id = child_id

    if best_child_id is not None and best_score >= action_match_threshold:
        return best_child_id, best_score
    return None, best_score


def _extract_action_params(action_template: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for raw_name in _find_placeholders(action_template):
        canonical_name = _canonical_param_name(raw_name)
        params[canonical_name] = "{{" + canonical_name + "}}"
    return params


def _merge_node_params(existing_params: Dict[str, str], new_params: Dict[str, str]) -> Dict[str, str]:
    merged = dict(existing_params or {})
    for key, value in new_params.items():
        merged.setdefault(key, value)
    return merged


def _build_path_steps_from_node_ids(
    nodes: Dict[str, UnitTreeNode],
    path_node_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    path_steps: List[Dict[str, Any]] = []
    for node_id in path_node_ids:
        node = nodes[node_id]
        path_steps.append(
            {
                "type": node.action_type,
                "description": node.description,
                "params": dict(node.params or {}),
            }
        )
    return path_steps


def _canonicalize_action_template(action_template: str) -> str:
    text = str(action_template or "")
    for raw_name in _find_placeholders(text):
        canonical_name = _canonical_param_name(raw_name)
        text = text.replace("{{" + raw_name + "}}", "{{" + canonical_name + "}}")
    return " ".join(text.split())


def _find_placeholders(text: str) -> List[str]:
    results: List[str] = []
    start = 0
    while True:
        left = text.find("{{", start)
        if left < 0:
            break
        right = text.find("}}", left + 2)
        if right < 0:
            break
        placeholder = text[left + 2 : right].strip()
        if placeholder and placeholder not in results:
            results.append(placeholder)
        start = right + 2
    return results


def _merge_parameter_defs(
    units: Sequence[ParameterizedUnitRecord],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for unit in units:
        for param_name, param_value in sorted(unit.parameters.items(), key=lambda item: item[0]):
            canonical_name = _canonical_param_name(param_name)
            if canonical_name not in merged:
                merged[canonical_name] = {
                    "param_name": canonical_name,
                    "param_type": _infer_param_type(param_value),
                    "description": _build_param_description(canonical_name),
                    "observed_values": [],
                }

            observed_values = merged[canonical_name]["observed_values"]
            normalized_value = _normalize_observed_value(param_value)
            if normalized_value not in observed_values:
                observed_values.append(normalized_value)

    return [merged[name] for name in sorted(merged)]


def _infer_param_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _build_param_description(param_name: str) -> str:
    readable = param_name.replace("_", " ")
    if param_name.endswith("element"):
        return f"Target UI element for {readable}"
    if param_name.endswith("value"):
        return f"Concrete value bound to {readable}"
    if param_name.endswith("text"):
        return f"Text content bound to {readable}"
    return f"Canonical parameter for {readable}"


def _normalize_observed_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [ _normalize_observed_value(item) for item in value ]
    if isinstance(value, dict):
        return {str(key): _normalize_observed_value(item) for key, item in sorted(value.items())}
    return str(value)


def _canonical_param_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    aliases = {
        "target_text_element": "target_element",
        "target_title_box": "target_element",
        "target_slide_thumbnail": "target_element",
        "target_field": "target_element",
        "font_color_control": "color_control",
        "custom_color_option": "custom_color_option",
        "recent_color_swatch": "color_source",
        "color_field": "color_value",
        "new_title_text": "content_text",
        "confirm_button": "confirm_button",
        "intermediate_object": "target_object",
        "empty_workspace_area": "workspace_area",
    }
    return aliases.get(normalized, normalized)


def _select_first_seen(units: Iterable[ParameterizedUnitRecord]) -> str | None:
    timestamps = sorted(
        str(unit.timestamp_start)
        for unit in units
        if unit.timestamp_start
    )
    return timestamps[0] if timestamps else None


def _select_last_seen(units: Iterable[ParameterizedUnitRecord]) -> str | None:
    timestamps = sorted(
        str(unit.timestamp_end)
        for unit in units
        if unit.timestamp_end
    )
    return timestamps[-1] if timestamps else None


def _infer_category(app_name: str) -> str:
    normalized = str(app_name or "").lower()
    if any(keyword in normalized for keyword in ("word", "excel", "powerpoint", "impress", "writer", "calc")):
        return "Office"
    if any(keyword in normalized for keyword in ("chrome", "firefox", "browser", "edge", "safari")):
        return "Daily"
    return "Uncategorized"


def _normalize_action_signature_text(text: str) -> str:
    normalized = _canonicalize_action_template(text).lower()
    normalized = normalized.replace("right-click", "click")
    normalized = normalized.replace("right click", "click")
    normalized = normalized.replace("ctrl+a", "select_all")
    normalized = normalized.replace("ctrl+c", "copy")
    normalized = normalized.replace("ctrl+v", "paste")
    normalized = normalized.replace("backspace", "delete")
    normalized = normalized.replace("pick a color", "color_dialog")
    normalized = normalized.replace("custom color", "color_dialog")
    normalized = normalized.replace("font color", "text_color")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _action_semantic_similarity(
    left_text: str,
    right_text: str,
    *,
    embedding_model: str,
) -> float:
    normalized_left = _normalize_action_signature_text(left_text)
    normalized_right = _normalize_action_signature_text(right_text)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    left_embedding = _get_action_embedding(normalized_left, embedding_model=embedding_model)
    right_embedding = _get_action_embedding(normalized_right, embedding_model=embedding_model)
    return max(0.0, min(1.0, _cosine_similarity(left_embedding, right_embedding)))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        numerator += float(left_value) * float(right_value)
        left_norm += float(left_value) ** 2
        right_norm += float(right_value) ** 2

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    return numerator / math.sqrt(left_norm * right_norm)


def _get_action_embedding(text: str, *, embedding_model: str) -> List[float]:
    if get_embedding is None:
        raise RuntimeError(
            "Unit Tree action matching requires a working embedding function."
        )

    embedding = _get_action_embedding_cached(text, embedding_model)
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(
            f"Embedding generation failed for action template: {text!r}"
        )
    return [float(value) for value in embedding]


@lru_cache(maxsize=4096)
def _get_action_embedding_cached(text: str, embedding_model: str):
    return get_embedding(text, model=embedding_model)


def _build_path_label(path_steps: Sequence[Dict[str, Any]]) -> str:
    labels: List[str] = []
    for step in path_steps[:3]:
        action_type = str(step.get("type", "UNKNOWN")).lower()
        description = _normalize_action_signature_text(str(step.get("description", "")))
        compact = _compact_path_phrase(description)
        labels.append(f"{action_type}:{compact}")
    return " -> ".join(labels) if labels else "empty_path"


def _abstract_state_text(text: str) -> str:
    normalized = str(text or "")
    stable_labels = {
        '"Pick a Color"': "__COLOR_DIALOG__",
        '"Custom Color..."': "__CUSTOM_COLOR_OPTION__",
        '"LibreOffice Impress"': "__APP_NAME__",
    }
    for original, marker in stable_labels.items():
        normalized = normalized.replace(original, marker)
    normalized = re.sub(r"\bslide\s+\d+\b", "target slide", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d+_\d+\.pptx\b", "<document>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b#?[0-9a-fA-F]{6}\b", "<hex_color>", normalized)
    normalized = re.sub(r"\bRGB\s*\d+\s*/\s*\d+\s*/\s*\d+\b", "RGB <color_value>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\"[^\"]{2,}\"", "\"<text>\"", normalized)
    for original, marker in stable_labels.items():
        normalized = normalized.replace(marker, original)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _compact_path_phrase(description: str) -> str:
    if "select_all" in description:
        return "select_all"
    if "paste" in description:
        return "paste_value"
    if "copy" in description:
        return "copy_value"
    if "color_dialog" in description:
        return "open_color_dialog"
    if "text_color" in description:
        return "open_color_palette"
    if "confirm" in description or "ok button" in description:
        return "confirm"
    if "target_element" in description and "select" in description:
        return "select_target_element"
    tokens = description.split()
    return "_".join(tokens[:2]) if tokens else "step"


def _token_cosine_similarity(left_text: str, right_text: str) -> float:
    left_counts = _token_counts(left_text)
    right_counts = _token_counts(right_text)
    if not left_counts or not right_counts:
        return 0.0

    numerator = 0.0
    for token, left_value in left_counts.items():
        numerator += left_value * right_counts.get(token, 0.0)

    left_norm = sum(value * value for value in left_counts.values()) ** 0.5
    right_norm = sum(value * value for value in right_counts.values()) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _token_counts(text: str) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    normalized = _normalize_free_text(text)
    for token in re.findall(r"[a-z0-9_]+", normalized):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _normalize_free_text(text: str) -> str:
    normalized = _abstract_state_text(text).lower()
    normalized = normalized.replace("text box", "text_element")
    normalized = normalized.replace("title text", "text_element")
    normalized = normalized.replace("font color", "text_color")
    normalized = normalized.replace("custom color", "color_dialog")
    normalized = normalized.replace("pick a color", "color_dialog")
    normalized = normalized.replace("ctrl+a", "select_all")
    normalized = re.sub(r"\{\{[^}]+\}\}", " param ", normalized)
    normalized = re.sub(r"[^a-z0-9_]+", " ", normalized)
    return " ".join(normalized.split())


__all__ = [
    "DEFAULT_ACTION_MATCH_THRESHOLD",
    "DEFAULT_EMBEDDING_MODEL",
    "build_canonical_units",
]
