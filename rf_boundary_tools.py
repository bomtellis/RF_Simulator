"""Geometry helpers for RF planner boundary suggestions.

This module deliberately has no Qt dependency so the outer-wall chain logic can
be tested independently from the desktop GUI.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union


def _polygon_parts(geometry) -> List[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [
        part
        for part in getattr(geometry, "geoms", [])
        if getattr(part, "geom_type", "") == "Polygon" and not part.is_empty
    ]


def _normalise_polygon(geometry) -> List[Polygon]:
    """Return valid polygon parts from an arbitrary Shapely geometry."""
    if geometry is None or geometry.is_empty:
        return []
    try:
        repaired = geometry if geometry.is_valid else geometry.buffer(0)
    except Exception:
        return []
    return [part for part in _polygon_parts(repaired) if float(part.area) > 1e-6]


def _estimate_polygon_thickness(polygon: Polygon) -> float:
    """Estimate the short side of a mostly rectangular wall polygon."""
    try:
        rectangle = polygon.minimum_rotated_rectangle
        coords = list(rectangle.exterior.coords)
    except Exception:
        return 0.0
    lengths: List[float] = []
    for first, second in zip(coords, coords[1:]):
        length = math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
        if length > 1e-6:
            lengths.append(length)
    return min(lengths) if lengths else 0.0


def estimate_outer_wall_gap_tolerance(wall_polygons: Iterable[Polygon]) -> float:
    """Return a practical default gap-closing distance in model metres.

    The tolerance is intentionally large enough to bridge ordinary door and
    curtain-wall openings, but is capped so detached buildings are not usually
    merged into one planner boundary.
    """
    thicknesses: List[float] = []
    for geometry in wall_polygons:
        for polygon in _normalise_polygon(geometry):
            value = _estimate_polygon_thickness(polygon)
            if 0.03 <= value <= 3.0:
                thicknesses.append(value)
    typical = median(thicknesses) if thicknesses else 0.20
    return max(0.75, min(2.00, float(typical) * 5.0))


def suggest_external_boundary_polygons(
    wall_polygons: Sequence[Polygon],
    gap_tolerance_m: Optional[float] = None,
    simplify_tolerance_m: Optional[float] = None,
    minimum_area_m2: float = 1.0,
) -> Tuple[List[Polygon], dict]:
    """Trace one or more external planner polygons from wall geometry.

    Wall polygons are joined with a configurable morphological closing buffer.
    Each connected wall network then contributes its outer ring, with holes
    removed so internal partitions cannot create forbidden islands. The buffer
    is subsequently removed to return the chain close to the outer wall faces.

    The returned metadata is suitable for a GUI preview/acceptance summary.
    """
    cleaned: List[Polygon] = []
    for geometry in wall_polygons:
        cleaned.extend(_normalise_polygon(geometry))
    if not cleaned:
        return [], {
            "wall_count": 0,
            "gap_tolerance_m": 0.0,
            "typical_wall_thickness_m": 0.0,
            "warnings": ["No valid wall polygons were available."],
        }

    thicknesses = [
        value
        for value in (_estimate_polygon_thickness(polygon) for polygon in cleaned)
        if 0.03 <= value <= 3.0
    ]
    typical_thickness = float(median(thicknesses)) if thicknesses else 0.20
    gap = (
        estimate_outer_wall_gap_tolerance(cleaned)
        if gap_tolerance_m is None
        else max(0.05, min(10.0, float(gap_tolerance_m)))
    )
    simplify = (
        max(0.02, min(0.25, typical_thickness * 0.35))
        if simplify_tolerance_m is None
        else max(0.0, min(2.0, float(simplify_tolerance_m)))
    )

    try:
        wall_union = unary_union(cleaned)
        joined = wall_union.buffer(gap, cap_style=2, join_style=2)
    except Exception as exc:
        return [], {
            "wall_count": len(cleaned),
            "gap_tolerance_m": gap,
            "typical_wall_thickness_m": typical_thickness,
            "warnings": [f"Could not join the wall geometry: {exc}"],
        }

    joined_parts = sorted(_polygon_parts(joined), key=lambda polygon: float(polygon.area), reverse=True)
    if not joined_parts:
        return [], {
            "wall_count": len(cleaned),
            "gap_tolerance_m": gap,
            "typical_wall_thickness_m": typical_thickness,
            "warnings": ["The wall network did not form a usable outer chain."],
        }

    largest_joined_area = float(joined_parts[0].area)
    minimum_component_area = max(float(minimum_area_m2), largest_joined_area * 0.005)
    suggestions: List[Polygon] = []
    warnings: List[str] = []
    collapsed_components = 0

    for component in joined_parts:
        if float(component.area) < minimum_component_area:
            continue
        try:
            filled_outer = Polygon(component.exterior)
            inset = filled_outer.buffer(-gap, join_style=2)
        except Exception:
            continue

        inset_parts = sorted(_polygon_parts(inset), key=lambda polygon: float(polygon.area), reverse=True)
        if not inset_parts:
            collapsed_components += 1
            continue

        candidate = inset_parts[0]
        # A very small inset compared with the buffered component normally means
        # the wall chain still has a large opening. Do not turn a thin wall strip
        # into a planning area; tell the user to increase the bridge tolerance.
        reference_area = max(float(filled_outer.area), 1e-9)
        envelope_area = max(float(candidate.envelope.area), 1e-9)
        if (
            float(candidate.area) / reference_area < 0.15
            or float(candidate.area) / envelope_area < 0.12
        ):
            collapsed_components += 1
            continue

        try:
            candidate = Polygon(candidate.exterior)
            if simplify > 0.0:
                candidate = candidate.simplify(simplify, preserve_topology=True)
            if not candidate.is_valid:
                candidate = candidate.buffer(0)
        except Exception:
            continue

        candidate_parts = sorted(_polygon_parts(candidate), key=lambda polygon: float(polygon.area), reverse=True)
        if not candidate_parts:
            continue
        candidate = Polygon(candidate_parts[0].exterior)
        if float(candidate.area) < float(minimum_area_m2):
            continue
        suggestions.append(candidate)

    if collapsed_components:
        warnings.append(
            f"{collapsed_components} wall chain(s) remained open. Increase the maximum wall-gap bridge and preview again."
        )
    if not suggestions and not warnings:
        warnings.append("The wall network did not produce a usable external boundary.")

    suggestions.sort(key=lambda polygon: float(polygon.area), reverse=True)
    return suggestions, {
        "wall_count": len(cleaned),
        "gap_tolerance_m": gap,
        "typical_wall_thickness_m": typical_thickness,
        "component_count": len(suggestions),
        "total_area_m2": sum(float(polygon.area) for polygon in suggestions),
        "vertex_count": sum(max(0, len(polygon.exterior.coords) - 1) for polygon in suggestions),
        "warnings": warnings,
    }
