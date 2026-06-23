"""Deterministic 2.5D RF multipath helpers for the RF Attenuation Simulator.

The module is deliberately independent from Qt and the simulator data classes so
it can be imported by multiprocessing workers.  It provides bounded image-source
reflections, corner diffraction, Fresnel material coefficients, coherent field
combination, delay-spread metrics and deterministic spatial fading.
"""
from __future__ import annotations

import cmath
import functools
import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from shapely.geometry import LineString, Point, box
    from shapely.strtree import STRtree
except Exception:  # pragma: no cover - the main application reports Shapely setup errors
    LineString = Point = box = None
    STRtree = None

C_M_PER_S = 299_792_458.0
EPSILON_0_F_PER_M = 8.854_187_8128e-12


@dataclass(frozen=True)
class ReflectionSurface:
    """One finite reflecting edge extracted from an IFC/RF polygon."""

    a: Tuple[float, float]
    b: Tuple[float, float]
    material: str
    object_id: str
    category: str = "wall"

    @property
    def length_m(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


@dataclass(frozen=True)
class RayPath:
    """A geometric propagation path before link-budget attenuation is applied."""

    points: Tuple[Tuple[float, float], ...]
    length_m: float
    coefficient: complex
    extra_loss_db: float
    kind: str
    interacted_object_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReflectionGeometryPath:
    """Frequency-independent image-source path geometry."""

    points: Tuple[Tuple[float, float], ...]
    length_m: float
    surfaces: Tuple[ReflectionSurface, ...]
    incidence_angles_rad: Tuple[float, ...]
    kind: str
    interacted_object_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DiffractionGeometryPath:
    """Frequency-independent corner-detour geometry."""

    points: Tuple[Tuple[float, float], ...]
    length_m: float
    excess_path_m: float
    object_id: str


@dataclass(frozen=True)
class PathPower:
    """A received path contribution used for coherent combination and delay spread."""

    power_dbm: float
    length_m: float
    phase_rad: float
    kind: str


def dbm_to_mw(value_dbm: float) -> float:
    if not math.isfinite(value_dbm):
        return 0.0
    return 10.0 ** (float(value_dbm) / 10.0)


def mw_to_dbm(value_mw: float, floor_dbm: float = -200.0) -> float:
    if not math.isfinite(value_mw) or value_mw <= 0.0:
        return float(floor_dbm)
    return max(float(floor_dbm), 10.0 * math.log10(value_mw))


def coherent_power_dbm(paths: Sequence[PathPower], floor_dbm: float = -200.0) -> float:
    """Coherently combine ray fields using each path's phase."""
    field = 0j
    for path in paths:
        power_mw = dbm_to_mw(path.power_dbm)
        if power_mw <= 0.0:
            continue
        field += math.sqrt(power_mw) * cmath.exp(1j * float(path.phase_rad))
    return mw_to_dbm(abs(field) ** 2, floor_dbm)


def incoherent_power_dbm(values_dbm: Sequence[float], floor_dbm: float = -200.0) -> float:
    return mw_to_dbm(sum(dbm_to_mw(value) for value in values_dbm), floor_dbm)


def rms_delay_spread_ns(paths: Sequence[PathPower]) -> float:
    """Power-weighted RMS excess delay in nanoseconds."""
    if len(paths) <= 1:
        return 0.0
    powers = [dbm_to_mw(path.power_dbm) for path in paths]
    total = sum(powers)
    if total <= 0.0:
        return 0.0
    delays = [float(path.length_m) / C_M_PER_S for path in paths]
    mean = sum(power * delay for power, delay in zip(powers, delays)) / total
    variance = sum(power * (delay - mean) ** 2 for power, delay in zip(powers, delays)) / total
    return max(0.0, math.sqrt(max(0.0, variance)) * 1e9)


@functools.lru_cache(maxsize=64)
def wavelength_m(frequency_mhz: float) -> float:
    return C_M_PER_S / max(1.0, float(frequency_mhz) * 1e6)


def propagation_phase_rad(length_m: float, frequency_mhz: float, extra_phase_rad: float = 0.0) -> float:
    lam = wavelength_m(frequency_mhz)
    return (2.0 * math.pi * float(length_m) / lam + float(extra_phase_rad)) % (2.0 * math.pi)


def _normalise_material_name(value: str) -> str:
    text = str(value or "default").strip().lower()
    return text or "default"


def _material_properties(material: str, profiles: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    name = _normalise_material_name(material)
    best = None
    best_len = -1
    for key, profile in (profiles or {}).items():
        key_text = str(key).strip().lower()
        if key_text == "default":
            continue
        if key_text and key_text in name and len(key_text) > best_len:
            best = profile
            best_len = len(key_text)
    if best is None:
        best = (profiles or {}).get(name) or (profiles or {}).get("default") or {}
    return {
        "relative_permittivity": max(1.0, float(best.get("relative_permittivity", 4.0))),
        "conductivity_s_per_m": max(0.0, float(best.get("conductivity_s_per_m", 0.02))),
        "roughness_m": max(0.0, float(best.get("roughness_m", 0.003))),
        "reflection_scale": max(0.0, float(best.get("reflection_scale", 1.0))),
    }


def fresnel_reflection_coefficient(
    material: str,
    incidence_angle_rad: float,
    frequency_mhz: float,
    profiles: Dict[str, Dict[str, float]],
) -> complex:
    """Return an averaged complex Fresnel coefficient for a lossy material.

    The incidence angle is measured from the surface normal.  Parallel and
    perpendicular polarisations are averaged because the simulator does not
    currently model antenna polarisation per radio.
    """
    properties = _material_properties(material, profiles)
    frequency_hz = max(1.0, float(frequency_mhz) * 1e6)
    omega = 2.0 * math.pi * frequency_hz
    complex_eps_r = complex(
        properties["relative_permittivity"],
        -properties["conductivity_s_per_m"] / (omega * EPSILON_0_F_PER_M),
    )
    theta = max(0.0, min(math.radians(89.5), abs(float(incidence_angle_rad))))
    sin_theta = math.sin(theta)
    cos_theta = max(1e-6, math.cos(theta))
    root = cmath.sqrt(complex_eps_r - sin_theta * sin_theta)
    perpendicular = (cos_theta - root) / (cos_theta + root)
    parallel = (complex_eps_r * cos_theta - root) / (complex_eps_r * cos_theta + root)
    # For unknown polarisation, average reflected *power*, not the complex
    # coefficients directly. Direct complex averaging can unrealistically cancel
    # TE and TM components because their reference phases differ. Preserve the
    # perpendicular component phase and use the RMS magnitude of both modes.
    magnitude = math.sqrt((abs(perpendicular) ** 2 + abs(parallel) ** 2) * 0.5)
    coefficient = magnitude * cmath.exp(1j * cmath.phase(perpendicular))

    # Rayleigh roughness reduction: rough surfaces lose coherent specular energy.
    roughness = properties["roughness_m"]
    lam = wavelength_m(frequency_mhz)
    roughness_factor = math.exp(-((4.0 * math.pi * roughness * cos_theta / lam) ** 2))
    coefficient *= roughness_factor * properties["reflection_scale"]
    return coefficient


def _point_reflected_across_line(point: Tuple[float, float], surface: ReflectionSurface) -> Optional[Tuple[float, float]]:
    ax, ay = surface.a
    bx, by = surface.b
    px, py = point
    vx = bx - ax
    vy = by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return None
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    qx = ax + t * vx
    qy = ay + t * vy
    return (2.0 * qx - px, 2.0 * qy - py)


def _line_intersection_with_surface(
    start: Tuple[float, float],
    end: Tuple[float, float],
    surface: ReflectionSurface,
    endpoint_margin: float = 1e-5,
) -> Optional[Tuple[float, float]]:
    """Intersect the infinite start/end line with a finite surface segment."""
    x1, y1 = start
    x2, y2 = end
    x3, y3 = surface.a
    x4, y4 = surface.b
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    denominator = dx1 * dy2 - dy1 * dx2
    if abs(denominator) <= 1e-12:
        return None
    rx = x3 - x1
    ry = y3 - y1
    t = (rx * dy2 - ry * dx2) / denominator
    u = (rx * dy1 - ry * dx1) / denominator
    if not (endpoint_margin < t < 1.0 - endpoint_margin):
        return None
    if not (endpoint_margin < u < 1.0 - endpoint_margin):
        return None
    return (x1 + t * dx1, y1 + t * dy1)


def _surface_incidence_angle(
    incoming_from: Tuple[float, float],
    reflection_point: Tuple[float, float],
    surface: ReflectionSurface,
) -> float:
    in_x = reflection_point[0] - incoming_from[0]
    in_y = reflection_point[1] - incoming_from[1]
    in_length = math.hypot(in_x, in_y)
    sx = surface.b[0] - surface.a[0]
    sy = surface.b[1] - surface.a[1]
    s_length = math.hypot(sx, sy)
    if in_length <= 1e-12 or s_length <= 1e-12:
        return 0.0
    # Either normal is acceptable because incidence is folded into [0, pi/2].
    nx = -sy / s_length
    ny = sx / s_length
    cosine = abs((in_x / in_length) * nx + (in_y / in_length) * ny)
    cosine = max(0.0, min(1.0, cosine))
    return math.acos(cosine)


def polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    )


def image_source_path(
    source: Tuple[float, float],
    receiver: Tuple[float, float],
    surfaces: Sequence[ReflectionSurface],
    frequency_mhz: float,
    material_profiles: Dict[str, Dict[str, float]],
    minimum_coefficient: float = 0.015,
) -> Optional[RayPath]:
    """Build a finite first/higher-order image-source reflection path."""
    if not surfaces:
        return None
    if len({surface.object_id for surface in surfaces}) < len(surfaces):
        # Repeatedly bouncing on the same finite wall face is usually a numerical
        # duplicate and creates unstable zero-length paths.
        return None

    images: List[Tuple[float, float]] = [source]
    current = source
    for surface in surfaces:
        current = _point_reflected_across_line(current, surface)
        if current is None:
            return None
        images.append(current)

    reflection_points_reversed: List[Tuple[float, float]] = []
    target = receiver
    current_image = images[-1]
    for index in range(len(surfaces) - 1, -1, -1):
        surface = surfaces[index]
        point = _line_intersection_with_surface(target, current_image, surface)
        if point is None:
            return None
        reflection_points_reversed.append(point)
        target = point
        current_image = images[index]

    reflection_points = list(reversed(reflection_points_reversed))
    points: List[Tuple[float, float]] = [source] + reflection_points + [receiver]
    if any(math.hypot(b[0] - a[0], b[1] - a[1]) <= 1e-4 for a, b in zip(points, points[1:])):
        return None

    # A specular reflection from an opaque face must arrive and leave on
    # the same side of that face. Reject image solutions that reach a rear
    # face by passing through the wall whose attenuation is intentionally
    # excluded at the reflection point.
    for index, surface in enumerate(surfaces):
        previous_point = points[index]
        next_point = points[index + 2]
        sx = surface.b[0] - surface.a[0]
        sy = surface.b[1] - surface.a[1]
        previous_side = sx * (previous_point[1] - surface.a[1]) - sy * (previous_point[0] - surface.a[0])
        next_side = sx * (next_point[1] - surface.a[1]) - sy * (next_point[0] - surface.a[0])
        if previous_side * next_side < -1e-8:
            return None

    coefficient = 1.0 + 0.0j
    previous = source
    for point, surface in zip(reflection_points, surfaces):
        angle = _surface_incidence_angle(previous, point, surface)
        coefficient *= fresnel_reflection_coefficient(
            surface.material, angle, frequency_mhz, material_profiles
        )
        previous = point
    if abs(coefficient) < max(0.0, float(minimum_coefficient)):
        return None

    return RayPath(
        points=tuple(points),
        length_m=polyline_length(points),
        coefficient=coefficient,
        extra_loss_db=0.0,
        kind=f"reflection_{len(surfaces)}",
        interacted_object_ids=tuple(surface.object_id for surface in surfaces),
    )


def image_source_geometry(
    source: Tuple[float, float],
    receiver: Tuple[float, float],
    surfaces: Sequence[ReflectionSurface],
) -> Optional[ReflectionGeometryPath]:
    """Build image-source geometry once so it can be reused by all bands."""
    if not surfaces or len({surface.object_id for surface in surfaces}) < len(surfaces):
        return None
    images: List[Tuple[float, float]] = [source]
    current = source
    for surface in surfaces:
        current = _point_reflected_across_line(current, surface)
        if current is None:
            return None
        images.append(current)
    reversed_points: List[Tuple[float, float]] = []
    target = receiver
    current_image = images[-1]
    for index in range(len(surfaces) - 1, -1, -1):
        point = _line_intersection_with_surface(target, current_image, surfaces[index])
        if point is None:
            return None
        reversed_points.append(point)
        target = point
        current_image = images[index]
    reflection_points = list(reversed(reversed_points))
    points: List[Tuple[float, float]] = [source] + reflection_points + [receiver]
    if any(math.hypot(b[0] - a[0], b[1] - a[1]) <= 1e-4 for a, b in zip(points, points[1:])):
        return None
    for index, surface in enumerate(surfaces):
        previous_point = points[index]
        next_point = points[index + 2]
        sx = surface.b[0] - surface.a[0]
        sy = surface.b[1] - surface.a[1]
        previous_side = sx * (previous_point[1] - surface.a[1]) - sy * (previous_point[0] - surface.a[0])
        next_side = sx * (next_point[1] - surface.a[1]) - sy * (next_point[0] - surface.a[0])
        if previous_side * next_side < -1e-8:
            return None
    angles: List[float] = []
    previous = source
    for point, surface in zip(reflection_points, surfaces):
        angles.append(_surface_incidence_angle(previous, point, surface))
        previous = point
    return ReflectionGeometryPath(
        points=tuple(points),
        length_m=polyline_length(points),
        surfaces=tuple(surfaces),
        incidence_angles_rad=tuple(angles),
        kind=f"reflection_{len(surfaces)}",
        interacted_object_ids=tuple(surface.object_id for surface in surfaces),
    )


def evaluate_reflection_geometry(
    geometry: ReflectionGeometryPath,
    frequency_mhz: float,
    material_profiles: Dict[str, Dict[str, float]],
    minimum_coefficient: float = 0.015,
) -> Optional[RayPath]:
    coefficient = 1.0 + 0.0j
    for surface, angle in zip(geometry.surfaces, geometry.incidence_angles_rad):
        coefficient *= fresnel_reflection_coefficient(surface.material, angle, frequency_mhz, material_profiles)
    if abs(coefficient) < max(0.0, float(minimum_coefficient)):
        return None
    return RayPath(
        points=geometry.points,
        length_m=geometry.length_m,
        coefficient=coefficient,
        extra_loss_db=0.0,
        kind=geometry.kind,
        interacted_object_ids=geometry.interacted_object_ids,
    )


def precompute_reflection_sequences(
    source: Tuple[float, float],
    receiver_hint: Tuple[float, float],
    index: Optional["ReflectionIndex"],
    maximum_order: int,
    maximum_surfaces: int,
    search_radius_m: float,
) -> Tuple[Tuple[ReflectionSurface, ...], ...]:
    """Precompute AP/tile reflection-surface orderings once per worker tile."""
    if index is None or maximum_order <= 0 or maximum_surfaces <= 0:
        return ()
    # Keep every surface in the already-pruned AP/tile index so the receiver-specific
    # top-N filter below remains numerically identical to the original point query.
    candidates = list(index.surfaces)
    if len(candidates) > 24 or (int(maximum_order) >= 3 and len(candidates) > 12):
        # Avoid combinatorial sequence tables for deliberately extreme detailed settings.
        return ()
    sequences: List[Tuple[ReflectionSurface, ...]] = []
    for order in range(1, max(1, int(maximum_order)) + 1):
        sequences.extend(tuple(sequence) for sequence in itertools.permutations(candidates, order))
    return tuple(sequences)


def generate_reflection_geometries(
    source: Tuple[float, float],
    receiver: Tuple[float, float],
    index: Optional["ReflectionIndex"],
    maximum_order: int,
    maximum_surfaces: int,
    maximum_paths: int,
    search_radius_m: float,
    surface_sequences: Optional[Sequence[Sequence[ReflectionSurface]]] = None,
) -> List[ReflectionGeometryPath]:
    if index is None or maximum_order <= 0 or maximum_paths <= 0:
        return []
    if surface_sequences is None:
        candidates = index.query_surfaces(source, receiver, search_radius_m, maximum_surfaces)
        if not candidates:
            return []
        sequences: Iterable[Sequence[ReflectionSurface]] = (
            sequence
            for order in range(1, max(1, int(maximum_order)) + 1)
            for sequence in itertools.permutations(candidates, order)
        )
    else:
        candidates = index.query_surfaces(source, receiver, search_radius_m, maximum_surfaces)
        candidate_ids = {id(surface) for surface in candidates}
        sequences = (
            sequence for sequence in surface_sequences
            if all(id(surface) in candidate_ids for surface in sequence)
        )
    direct_length = math.hypot(receiver[0] - source[0], receiver[1] - source[1])
    geometries: List[ReflectionGeometryPath] = []
    for sequence in sequences:
        geometry = image_source_geometry(source, receiver, sequence)
        if geometry is None:
            continue
        if geometry.length_m > direct_length + 2.0 * max(1.0, float(search_radius_m)):
            continue
        geometries.append(geometry)
    geometries.sort(key=lambda path: path.length_m)
    unique: List[ReflectionGeometryPath] = []
    signatures = set()
    for path in geometries:
        signature = tuple((round(x, 3), round(y, 3)) for x, y in path.points[1:-1])
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(path)
        if len(unique) >= int(maximum_paths):
            break
    return unique


def generate_diffraction_geometries(
    source: Tuple[float, float],
    receiver: Tuple[float, float],
    index: Optional["ReflectionIndex"],
    maximum_paths: int,
    search_radius_m: float,
    preferred_object_ids: Optional[Sequence[str]] = None,
) -> List[DiffractionGeometryPath]:
    if index is None or maximum_paths <= 0:
        return []
    direct_length = math.hypot(receiver[0] - source[0], receiver[1] - source[1])
    corners = index.query_corners(
        source, receiver, search_radius_m, max(maximum_paths * 4, maximum_paths), preferred_object_ids
    )
    paths: List[DiffractionGeometryPath] = []
    for x, y, object_id in corners:
        points = (source, (x, y), receiver)
        length = polyline_length(points)
        if length <= direct_length + 1e-6:
            continue
        paths.append(DiffractionGeometryPath(points, length, length - direct_length, object_id))
    paths.sort(key=lambda path: (path.excess_path_m, path.length_m))
    return paths[:max(0, int(maximum_paths))]


def evaluate_diffraction_geometry(
    geometry: DiffractionGeometryPath,
    frequency_mhz: float,
    minimum_loss_db: float,
) -> RayPath:
    return RayPath(
        points=geometry.points,
        length_m=geometry.length_m,
        coefficient=cmath.exp(-1j * math.pi / 4.0),
        extra_loss_db=knife_edge_diffraction_loss_db(geometry.excess_path_m, frequency_mhz, minimum_loss_db),
        kind="diffraction",
        interacted_object_ids=(geometry.object_id,),
    )


class ReflectionIndex:
    """Spatial index for reflection faces and diffraction corners."""

    def __init__(self, surfaces: Iterable[ReflectionSurface]):
        self.surfaces = [surface for surface in surfaces if surface.length_m >= 0.25]
        self._lines = [LineString([surface.a, surface.b]) for surface in self.surfaces] if LineString else []
        self._line_by_id = {id(line): index for index, line in enumerate(self._lines)}
        self._tree = None
        if STRtree is not None and self._lines:
            try:
                self._tree = STRtree(self._lines)
            except Exception:
                self._tree = None

        corner_map: Dict[Tuple[int, int, str], Tuple[float, float, str]] = {}
        for surface in self.surfaces:
            for point in (surface.a, surface.b):
                key = (round(point[0] * 1000), round(point[1] * 1000), surface.object_id)
                corner_map[key] = (point[0], point[1], surface.object_id)
        self.corners = list(corner_map.values())
        self._corner_points = [Point(x, y) for x, y, _ in self.corners] if Point else []
        self._corner_by_id = {id(point): index for index, point in enumerate(self._corner_points)}
        self._corner_tree = None
        if STRtree is not None and self._corner_points:
            try:
                self._corner_tree = STRtree(self._corner_points)
            except Exception:
                self._corner_tree = None

    def query_surfaces(
        self,
        source: Tuple[float, float],
        receiver: Tuple[float, float],
        radius_m: float,
        maximum: int,
    ) -> List[ReflectionSurface]:
        if not self.surfaces or maximum <= 0:
            return []
        direct = LineString([source, receiver])
        radius = max(0.1, float(radius_m))
        candidates: List[int] = []
        if self._tree is not None:
            try:
                hits = self._tree.query(direct.buffer(radius, cap_style=2, join_style=2))
                for hit in hits:
                    if isinstance(hit, int) or getattr(hit, "dtype", None) is not None:
                        candidates.append(int(hit))
                    else:
                        index = self._line_by_id.get(id(hit))
                        if index is not None:
                            candidates.append(index)
            except Exception:
                candidates = []
        if not candidates:
            candidates = list(range(len(self.surfaces)))

        ranked = []
        seen = set()
        for index in candidates:
            if index < 0 or index >= len(self.surfaces) or index in seen:
                continue
            seen.add(index)
            line = self._lines[index]
            try:
                distance = float(line.distance(direct))
            except Exception:
                distance = radius + 1.0
            if distance <= radius:
                surface = self.surfaces[index]
                edge_x = float(surface.b[0]) - float(surface.a[0])
                edge_y = float(surface.b[1]) - float(surface.a[1])
                side_source = edge_x * (float(source[1]) - float(surface.a[1])) - edge_y * (float(source[0]) - float(surface.a[0]))
                side_receiver = edge_x * (float(receiver[1]) - float(surface.a[1])) - edge_y * (float(receiver[0]) - float(surface.a[0]))
                # A specular reflection on an opaque finite face requires source and receiver
                # to lie on the same side of its infinite support line. Rejecting the other
                # half-space here avoids expensive image-source construction for impossible rays.
                if side_source * side_receiver < -1e-8:
                    continue
                midpoint = line.interpolate(0.5, normalized=True)
                source_leg = math.hypot(midpoint.x - source[0], midpoint.y - source[1])
                receiver_leg = math.hypot(midpoint.x - receiver[0], midpoint.y - receiver[1])
                path_hint = source_leg + receiver_leg
                direct_length = math.hypot(receiver[0] - source[0], receiver[1] - source[1])
                if path_hint > direct_length + 2.0 * radius:
                    continue
                ranked.append((path_hint + distance * 0.25, index))
        ranked.sort(key=lambda item: item[0])
        return [self.surfaces[index] for _, index in ranked[: max(1, int(maximum))]]

    def subset_for_source_tile(
        self,
        source: Tuple[float, float],
        tile_bounds: Tuple[float, float, float, float],
        radius_m: float,
        maximum: int,
    ) -> "ReflectionIndex":
        """Return a small reflection index relevant to one AP/tile envelope."""
        if not self.surfaces:
            return ReflectionIndex([])
        minx, miny, maxx, maxy = (float(v) for v in tile_bounds)
        radius = max(0.1, float(radius_m))
        envelope = box(
            min(minx, source[0]) - radius,
            min(miny, source[1]) - radius,
            max(maxx, source[0]) + radius,
            max(maxy, source[1]) + radius,
        )
        indices: List[int] = []
        if self._tree is not None:
            try:
                for hit in self._tree.query(envelope):
                    if isinstance(hit, int) or getattr(hit, "dtype", None) is not None:
                        indices.append(int(hit))
                    else:
                        index = self._line_by_id.get(id(hit))
                        if index is not None:
                            indices.append(index)
            except Exception:
                indices = []
        if not indices:
            indices = list(range(len(self.surfaces)))
        ranked = []
        centre = ((minx + maxx) * 0.5, (miny + maxy) * 0.5)
        for index in set(indices):
            if not (0 <= index < len(self.surfaces)):
                continue
            surface = self.surfaces[index]
            midpoint = ((surface.a[0] + surface.b[0]) * 0.5, (surface.a[1] + surface.b[1]) * 0.5)
            score = math.hypot(midpoint[0] - source[0], midpoint[1] - source[1]) + math.hypot(midpoint[0] - centre[0], midpoint[1] - centre[1])
            ranked.append((score, surface))
        ranked.sort(key=lambda item: item[0])
        limit = max(1, int(maximum))
        return ReflectionIndex([surface for _, surface in ranked[:limit]])

    def query_corners(
        self,
        source: Tuple[float, float],
        receiver: Tuple[float, float],
        radius_m: float,
        maximum: int,
        preferred_object_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[float, float, str]]:
        if not self.corners or maximum <= 0:
            return []
        direct = LineString([source, receiver])
        radius = max(0.1, float(radius_m))
        preferred = set(preferred_object_ids or [])
        candidate_indices: List[int] = []
        if self._corner_tree is not None:
            try:
                hits = self._corner_tree.query(direct.buffer(radius, cap_style=2, join_style=2))
                for hit in hits:
                    if isinstance(hit, int) or getattr(hit, "dtype", None) is not None:
                        candidate_indices.append(int(hit))
                    else:
                        index = self._corner_by_id.get(id(hit))
                        if index is not None:
                            candidate_indices.append(index)
            except Exception:
                candidate_indices = []
        if not candidate_indices:
            candidate_indices = list(range(len(self.corners)))

        ranked = []
        seen = set()
        for index in candidate_indices:
            if index < 0 or index >= len(self.corners) or index in seen:
                continue
            seen.add(index)
            x, y, object_id = self.corners[index]
            point = self._corner_points[index] if index < len(self._corner_points) else Point(x, y)
            distance = float(point.distance(direct))
            if distance > radius:
                continue
            detour = math.hypot(x - source[0], y - source[1]) + math.hypot(
                receiver[0] - x, receiver[1] - y
            ) - math.hypot(receiver[0] - source[0], receiver[1] - source[1])
            preference = 0 if object_id in preferred else 1
            ranked.append((preference, max(0.0, detour), distance, x, y, object_id))
        ranked.sort()
        return [(x, y, object_id) for _, _, _, x, y, object_id in ranked[: max(1, int(maximum))]]


def generate_reflection_paths(
    source: Tuple[float, float],
    receiver: Tuple[float, float],
    index: Optional[ReflectionIndex],
    frequency_mhz: float,
    material_profiles: Dict[str, Dict[str, float]],
    maximum_order: int,
    maximum_surfaces: int,
    maximum_paths: int,
    search_radius_m: float,
    minimum_coefficient: float,
) -> List[RayPath]:
    if index is None or maximum_order <= 0 or maximum_paths <= 0:
        return []
    candidates = index.query_surfaces(source, receiver, search_radius_m, maximum_surfaces)
    if not candidates:
        return []
    paths: List[RayPath] = []
    direct_length = math.hypot(receiver[0] - source[0], receiver[1] - source[1])
    for order in range(1, max(1, int(maximum_order)) + 1):
        for sequence in itertools.permutations(candidates, order):
            path = image_source_path(
                source,
                receiver,
                sequence,
                frequency_mhz,
                material_profiles,
                minimum_coefficient,
            )
            if path is None:
                continue
            # Reject extreme paths that are unlikely to contribute and consume
            # disproportionate wall-intersection work.
            if path.length_m > direct_length + 2.0 * max(1.0, float(search_radius_m)):
                continue
            paths.append(path)
    paths.sort(key=lambda path: (path.length_m, -abs(path.coefficient)))

    # Suppress near-identical image solutions from the two faces of very thin
    # wall polygons while keeping physically distinct bounces.
    unique: List[RayPath] = []
    signatures = set()
    for path in paths:
        signature = tuple((round(x, 3), round(y, 3)) for x, y in path.points[1:-1])
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(path)
        if len(unique) >= int(maximum_paths):
            break
    return unique


def knife_edge_diffraction_loss_db(excess_path_m: float, frequency_mhz: float, minimum_loss_db: float = 6.0) -> float:
    """ITU-style single knife-edge loss using geometric excess path length."""
    excess = max(0.0, float(excess_path_m))
    if excess <= 1e-9:
        return max(0.0, float(minimum_loss_db))
    v = math.sqrt(max(0.0, 2.0 * excess / wavelength_m(frequency_mhz)))
    loss = 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return max(float(minimum_loss_db), min(45.0, loss))


def generate_diffraction_paths(
    source: Tuple[float, float],
    receiver: Tuple[float, float],
    index: Optional[ReflectionIndex],
    frequency_mhz: float,
    maximum_paths: int,
    search_radius_m: float,
    minimum_loss_db: float,
    preferred_object_ids: Optional[Sequence[str]] = None,
) -> List[RayPath]:
    if index is None or maximum_paths <= 0:
        return []
    direct_length = math.hypot(receiver[0] - source[0], receiver[1] - source[1])
    corners = index.query_corners(
        source,
        receiver,
        search_radius_m,
        max(maximum_paths * 4, maximum_paths),
        preferred_object_ids,
    )
    paths = []
    for x, y, object_id in corners:
        points = (source, (x, y), receiver)
        length = polyline_length(points)
        if length <= direct_length + 1e-6:
            continue
        loss = knife_edge_diffraction_loss_db(length - direct_length, frequency_mhz, minimum_loss_db)
        paths.append(RayPath(
            points=points,
            length_m=length,
            coefficient=cmath.exp(-1j * math.pi / 4.0),
            extra_loss_db=loss,
            kind="diffraction",
            interacted_object_ids=(object_id,),
        ))
    paths.sort(key=lambda path: (path.extra_loss_db, path.length_m))
    return paths[: max(0, int(maximum_paths))]


def _mix_u32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def _lattice_noise(ix: int, iy: int, key: int) -> float:
    mixed = _mix_u32(ix * 0x1F123BB5 ^ iy * 0x5F356495 ^ key * 0x9E3779B9)
    # Approximately uniform [-1, 1]. Two mixed values are averaged to move the
    # distribution towards a smooth Gaussian-like residual without random state.
    mixed2 = _mix_u32(mixed ^ 0xA5A5A5A5)
    return ((mixed / 0xFFFFFFFF) * 2.0 - 1.0 + (mixed2 / 0xFFFFFFFF) * 2.0 - 1.0) * 0.5


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def deterministic_spatial_fading_db(
    x: float,
    y: float,
    correlation_distance_m: float,
    sigma_db: float,
    seed: int,
    link_key: int,
) -> float:
    """Stable bilinearly correlated residual fading field in dB."""
    scale = max(0.05, float(correlation_distance_m))
    gx = float(x) / scale
    gy = float(y) / scale
    ix = math.floor(gx)
    iy = math.floor(gy)
    tx = _smoothstep(gx - ix)
    ty = _smoothstep(gy - iy)
    key = int(seed) ^ int(link_key)
    n00 = _lattice_noise(ix, iy, key)
    n10 = _lattice_noise(ix + 1, iy, key)
    n01 = _lattice_noise(ix, iy + 1, key)
    n11 = _lattice_noise(ix + 1, iy + 1, key)
    nx0 = n00 + (n10 - n00) * tx
    nx1 = n01 + (n11 - n01) * tx
    value = nx0 + (nx1 - nx0) * ty
    return float(sigma_db) * value * 1.8


@functools.lru_cache(maxsize=4096)
def stable_link_key(ap_name: str, frequency_mhz: float, channel: str = "") -> int:
    """Process-independent integer key; avoids Python's randomised hash()."""
    text = f"{ap_name}|{float(frequency_mhz):.6f}|{channel}"
    value = 2166136261
    for byte in text.encode("utf-8", errors="replace"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value
