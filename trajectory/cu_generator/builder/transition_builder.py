"""Build Canonical Unit transition statistics from ordered unit instances."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from .schemas import ParameterizedUnitRecord


TraceKey = Tuple[str, str]


def build_transition_table(
    units: Sequence[ParameterizedUnitRecord],
    instance_to_cu: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    """Build the CU transition adjacency table from ordered unit instances."""

    transitions: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for trace_units in _group_units_by_trace(units).values():
        cu_sequence = [
            instance_to_cu[unit.instance_id]
            for unit in trace_units
            if unit.instance_id in instance_to_cu
        ]

        for previous_cu, next_cu in zip(cu_sequence, cu_sequence[1:]):
            if previous_cu == next_cu:
                continue
            transitions[previous_cu][next_cu] += 1

    return {
        source_cu: dict(sorted(targets.items(), key=lambda item: item[0]))
        for source_cu, targets in sorted(transitions.items(), key=lambda item: item[0])
    }


def _group_units_by_trace(
    units: Sequence[ParameterizedUnitRecord],
) -> Dict[TraceKey, List[ParameterizedUnitRecord]]:
    grouped: DefaultDict[TraceKey, List[ParameterizedUnitRecord]] = defaultdict(list)
    for unit in units:
        grouped[_trace_key(unit)].append(unit)

    return {
        trace_key: sorted(trace_units, key=_unit_sort_key)
        for trace_key, trace_units in sorted(grouped.items(), key=lambda item: item[0])
    }


def _trace_key(unit: ParameterizedUnitRecord) -> TraceKey:
    return (unit.source_user, unit.trace_id or unit.source_user)


def _unit_sort_key(unit: ParameterizedUnitRecord) -> Tuple[int, int, str, str]:
    return (
        int(unit.segment_order or 0),
        unit.step_indices[0] if unit.step_indices else 0,
        unit.timestamp_start or "",
        unit.instance_id,
    )


__all__ = ["build_transition_table"]
