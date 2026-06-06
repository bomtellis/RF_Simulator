"""
RF Attenuation Simulator - IFC Wi-Fi RSSI planning tool.

Run:
    pip install PySide6 numpy ifcopenshell shapely contourpy
    python rf_simulator.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    import contourpy
except Exception:
    contourpy = None

from PySide6.QtCore import QObject, QPointF, QRunnable, QThreadPool, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QBrush, QFont, QPen, QPolygonF, QPainterPath
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
        QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

try:
    import ifcopenshell
    import ifcopenshell.geom
except Exception:  # pragma: no cover
    ifcopenshell = None

try:
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install shapely: pip install shapely") from exc


# ----------------------------- Data models -----------------------------

@dataclass
class Wall2D:
    guid: str
    name: str
    floor: str
    source_file: str
    type_name: str
    material: str
    polygon: Polygon
    attenuation_by_band_db: Dict[float, float] = field(default_factory=lambda: {2400.0: 5.0, 5000.0: 7.0, 6000.0: 8.0})

    @property
    def label(self) -> str:
        key = self.material or self.type_name or "Unknown"
        return f"{key} | {self.name or self.guid[:8]}"

    def attenuation_db_for_frequency(self, frequency_mhz: float) -> float:
        """Return wall attenuation for the selected Wi-Fi band.

        Values are stored against the common Wi-Fi centre bands used by this
        tool: 2.4 GHz, 5 GHz and 6 GHz. If the user enters another frequency,
        linear interpolation is used between the nearest bands.
        """
        if not self.attenuation_by_band_db:
            return 0.0
        bands = sorted(float(k) for k in self.attenuation_by_band_db.keys())
        if frequency_mhz <= bands[0]:
            return float(self.attenuation_by_band_db[bands[0]])
        if frequency_mhz >= bands[-1]:
            return float(self.attenuation_by_band_db[bands[-1]])
        for lo, hi in zip(bands, bands[1:]):
            if lo <= frequency_mhz <= hi:
                lo_v = float(self.attenuation_by_band_db[lo])
                hi_v = float(self.attenuation_by_band_db[hi])
                t = (frequency_mhz - lo) / (hi - lo)
                return lo_v + (hi_v - lo_v) * t
        return float(self.attenuation_by_band_db[bands[-1]])


@dataclass
class AntennaPattern:
    """Simple AP antenna pattern used by the RF engine.

    azimuth_points/elevation_points are lists of (angle_degrees, gain_dbi).
    Angles are relative to the AP boresight. 0 degrees is straight ahead in
    azimuth and horizontal in elevation. The built-in profiles are deliberately
    simple so the simulator remains usable without manufacturer data sheets.
    Use Load pattern CSV to replace them with measured/manufacturer data.
    """

    name: str
    peak_gain_dbi: float = 0.0
    azimuth_points: List[Tuple[float, float]] = field(default_factory=list)
    elevation_points: List[Tuple[float, float]] = field(default_factory=list)

    def gain_dbi(self, azimuth_rel_deg: float, elevation_rel_deg: float = 0.0) -> float:
        az_gain = self._interp_gain(self.azimuth_points, azimuth_rel_deg, self.peak_gain_dbi)
        el_gain = self._interp_gain(self.elevation_points, elevation_rel_deg, 0.0)
        return az_gain + el_gain

    @staticmethod
    def _wrap_deg(angle: float) -> float:
        return ((angle + 180.0) % 360.0) - 180.0

    @classmethod
    def _interp_gain(cls, points: List[Tuple[float, float]], angle: float, default: float) -> float:
        if not points:
            return default
        angle = cls._wrap_deg(angle)
        pts = sorted((cls._wrap_deg(float(a)), float(g)) for a, g in points)
        pts_ext = [(pts[-1][0] - 360.0, pts[-1][1])] + pts + [(pts[0][0] + 360.0, pts[0][1])]
        for (a0, g0), (a1, g1) in zip(pts_ext, pts_ext[1:]):
            if a0 <= angle <= a1:
                if abs(a1 - a0) < 1e-9:
                    return g0
                t = (angle - a0) / (a1 - a0)
                return g0 + (g1 - g0) * t
        return pts[0][1]


def built_in_antenna_patterns() -> Dict[str, AntennaPattern]:
    return {
        "Omni ceiling AP": AntennaPattern(
            name="Omni ceiling AP",
            peak_gain_dbi=3.0,
            azimuth_points=[(-180, 3), (-90, 3), (0, 3), (90, 3), (180, 3)],
            elevation_points=[(-90, -8), (-60, -3), (-30, 1), (0, 3), (30, 1), (60, -3), (90, -8)],
        ),
        "Wall patch 60 degree": AntennaPattern(
            name="Wall patch 60 degree",
            peak_gain_dbi=8.0,
            azimuth_points=[(-180, -18), (-120, -15), (-90, -10), (-60, -4), (-30, 5), (0, 8), (30, 5), (60, -4), (90, -10), (120, -15), (180, -18)],
            elevation_points=[(-90, -15), (-45, -5), (0, 8), (45, -5), (90, -15)],
        ),
        "Directional sector 90 degree": AntennaPattern(
            name="Directional sector 90 degree",
            peak_gain_dbi=6.0,
            azimuth_points=[(-180, -14), (-135, -12), (-90, -6), (-45, 3), (0, 6), (45, 3), (90, -6), (135, -12), (180, -14)],
            elevation_points=[(-90, -12), (-45, -4), (0, 6), (45, -4), (90, -12)],
        ),
    }


@dataclass
class AccessPoint:
    name: str
    x: float
    y: float
    floor: str
    tx_power_dbm: float = 20.0
    frequency_mhz: float = 2400.0
    reference_loss_db_at_1m: float = 40.0
    path_loss_exponent: float = 2.2
    antenna_pattern: str = "Omni ceiling AP"
    azimuth_deg: float = 0.0
    downtilt_deg: float = 0.0
    mount_height_m: float = 2.7
    rx_height_m: float = 1.2


@dataclass
class Space2D:
    guid: str
    name: str
    floor: str
    source_file: str
    polygon: Polygon


@dataclass
class FloorModel:
    name: str
    elevation: float
    walls: List[Wall2D] = field(default_factory=list)
    spaces: List[Space2D] = field(default_factory=list)
    slab_attenuation_by_band_db: Dict[float, float] = field(default_factory=lambda: {2400.0: 12.0, 5000.0: 18.0, 6000.0: 22.0})

    def slab_attenuation_db_for_frequency(self, frequency_mhz: float) -> float:
        """Return floor/slab penetration attenuation for the selected band."""
        if not self.slab_attenuation_by_band_db:
            return 0.0
        bands = sorted(float(k) for k in self.slab_attenuation_by_band_db.keys())
        if frequency_mhz <= bands[0]:
            return float(self.slab_attenuation_by_band_db[bands[0]])
        if frequency_mhz >= bands[-1]:
            return float(self.slab_attenuation_by_band_db[bands[-1]])
        for lo, hi in zip(bands, bands[1:]):
            if lo <= frequency_mhz <= hi:
                lo_v = float(self.slab_attenuation_by_band_db[lo])
                hi_v = float(self.slab_attenuation_by_band_db[hi])
                t = (frequency_mhz - lo) / (hi - lo)
                return lo_v + (hi_v - lo_v) * t
        return float(self.slab_attenuation_by_band_db[bands[-1]])


@dataclass
class SimulationResult:
    xs: np.ndarray
    ys: np.ndarray
    rssi: np.ndarray


@dataclass
class RSSIZone:
    name: str
    min_dbm: float
    max_dbm: float
    colour: str
    alpha: int = 135


@dataclass
class HeatmapSettings:
    minimum_client_rssi_dbm: float = -82.0
    # True contour levels. Positive values in the JSON are treated as RSSI magnitudes
    # and converted to negative dBm, e.g. 70 -> -70 dBm.
    isoline_bands_dbm: List[float] = field(default_factory=lambda: [
        -10.0, -20.0, -30.0, -40.0, -55.0, -60.0, -65.0, -70.0,
        -75.0, -80.0, -85.0, -90.0, -95.0, -100.0, -105.0, -110.0,
    ])
    # CSV pattern files can be listed in the settings file. Relative paths are
    # resolved relative to the settings JSON location.
    rf_pattern_files: List[str] = field(default_factory=list)
    contour_label_font_size: int = 4
    sample_label_font_size: int = 4
    sample_cross_size: float = 0.22
    sample_stride_x: int = 8
    sample_stride_y: int = 6
    zones: List[RSSIZone] = field(default_factory=lambda: [
        RSSIZone("Excellent", -55.0, 0.0, "#00AA50", 135),
        RSSIZone("Good", -67.0, -55.0, "#A0C800", 135),
        RSSIZone("Marginal", -75.0, -67.0, "#FFAA00", 135),
        RSSIZone("Poor", -82.0, -75.0, "#DC0000", 135),
        RSSIZone("Disconnect", -200.0, -82.0, "#555555", 95),
    ])

    @classmethod
    def default(cls) -> "HeatmapSettings":
        return cls()

    @classmethod
    def _normalise_band(cls, value: float) -> float:
        value = float(value)
        return -abs(value) if value > 0 else value

    @classmethod
    def from_json_file(cls, path: Path) -> "HeatmapSettings":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        minimum = float(data.get("minimum_client_rssi_dbm", data.get("disconnect_threshold_dbm", -82.0)))
        zones = []
        for z in data.get("zones", []):
            zones.append(RSSIZone(
                name=str(z.get("name", "RSSI zone")),
                min_dbm=cls._normalise_band(float(z.get("min_dbm", -200.0))),
                max_dbm=cls._normalise_band(float(z.get("max_dbm", 0.0))) if float(z.get("max_dbm", 0.0)) != 0.0 else 0.0,
                colour=str(z.get("colour", z.get("color", "#555555"))),
                alpha=int(z.get("alpha", 135)),
            ))
        settings = cls(minimum_client_rssi_dbm=minimum)
        raw_bands = data.get("isoline_bands_dbm", data.get("isoline_bands", data.get("contour_bands_dbm", [])))
        if raw_bands:
            settings.isoline_bands_dbm = sorted(
                {cls._normalise_band(float(v)) for v in raw_bands},
                reverse=True,
            )
        settings.rf_pattern_files = [str(v) for v in data.get("rf_pattern_files", data.get("antenna_pattern_files", []))]
        settings.contour_label_font_size = int(data.get("contour_label_font_size", 4))
        settings.sample_label_font_size = int(data.get("sample_label_font_size", 4))
        settings.sample_cross_size = float(data.get("sample_cross_size", 0.22))
        settings.sample_stride_x = max(1, int(data.get("sample_stride_x", 8)))
        settings.sample_stride_y = max(1, int(data.get("sample_stride_y", 6)))
        if zones:
            # Highest RSSI first, so -55 to 0 is before -67 to -55.
            settings.zones = sorted(zones, key=lambda z: z.min_dbm, reverse=True)
        settings.ensure_disconnect_zone()
        return settings

    def ensure_disconnect_zone(self):
        has_disconnect = any(z.max_dbm <= self.minimum_client_rssi_dbm for z in self.zones)
        if not has_disconnect:
            self.zones.append(RSSIZone("Disconnect", -200.0, self.minimum_client_rssi_dbm, "#555555", 95))
        self.zones = sorted(self.zones, key=lambda z: z.min_dbm, reverse=True)
        self.isoline_bands_dbm = sorted({self._normalise_band(v) for v in self.isoline_bands_dbm}, reverse=True)

    def zone_for_rssi(self, rssi: float) -> RSSIZone:
        for zone in self.zones:
            if zone.min_dbm <= rssi < zone.max_dbm:
                return zone
        if rssi >= max(z.max_dbm for z in self.zones):
            return self.zones[0]
        return self.zones[-1]

    def colour_for_rssi(self, rssi: float) -> QColor:
        zone = self.zone_for_rssi(rssi)
        colour = QColor(zone.colour)
        if not colour.isValid():
            colour = QColor("#555555")
        colour.setAlpha(zone.alpha)
        return colour


class IFCModelLoader:
    """Extracts storeys and wall plan polygons from an IFC file.

    Multiple IFCs can be loaded and merged by the MainWindow. Each file is
    treated as one discipline/model package in the same shared IFC coordinate
    system. If an architectural model, structural model, and fit-out model use
    the same project coordinates, their walls will line up automatically.
    """

    def __init__(self, path: Path, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        if ifcopenshell is None:
            raise RuntimeError("ifcopenshell is not installed. Run: pip install ifcopenshell")
        self.path = path
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.ifc = ifcopenshell.open(str(path))
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def load(self) -> Dict[str, FloorModel]:
        storeys = self._storeys()
        floors = {name: FloorModel(name=name, elevation=elev) for name, elev in storeys.items()}
        if not floors:
            floors["Default"] = FloorModel(name="Default", elevation=0.0)

        for wall in list(self.ifc.by_type("IfcWall")) + list(self.ifc.by_type("IfcWallStandardCase")):
            floor_name = self._container_storey_name(wall) or self._nearest_floor_name(floors, 0.0)
            poly = self._plan_polygon_from_geometry(wall)
            if poly is None or poly.area <= 0:
                continue
            mat = self._material_name(wall)
            type_name = self._type_name(wall)
            floors.setdefault(floor_name, FloorModel(name=floor_name, elevation=0.0)).walls.append(
                Wall2D(
                    guid=getattr(wall, "GlobalId", ""),
                    name=getattr(wall, "Name", "") or "Wall",
                    floor=floor_name,
                    source_file=self.path.name,
                    type_name=type_name,
                    material=mat,
                    polygon=poly,
                    attenuation_by_band_db=self._default_attenuation_profile(mat, type_name),
                )
            )

        # Extract IfcSpace footprints so room/space names can be shown on the plan.
        # Missing/invalid space geometry is ignored so models without spaces still load.
        for space in self.ifc.by_type("IfcSpace"):
            floor_name = self._container_storey_name(space) or self._nearest_floor_name(floors, 0.0)
            poly = self._plan_polygon_from_geometry(space)
            if poly is None or poly.area <= 0:
                continue
            floors.setdefault(floor_name, FloorModel(name=floor_name, elevation=0.0)).spaces.append(
                Space2D(
                    guid=getattr(space, "GlobalId", ""),
                    name=getattr(space, "LongName", None) or getattr(space, "Name", "") or "Space",
                    floor=floor_name,
                    source_file=self.path.name,
                    polygon=poly,
                )
            )
        return floors

    def _storeys(self) -> Dict[str, float]:
        out = {}
        for st in self.ifc.by_type("IfcBuildingStorey"):
            name = st.Name or st.GlobalId
            out[name] = float(getattr(st, "Elevation", 0.0) or 0.0) + self.dz
        return out

    def _container_storey_name(self, product) -> Optional[str]:
        for rel in getattr(product, "ContainedInStructure", []) or []:
            s = getattr(rel, "RelatingStructure", None)
            if s and s.is_a("IfcBuildingStorey"):
                return s.Name or s.GlobalId
        return None

    @staticmethod
    def _nearest_floor_name(floors: Dict[str, FloorModel], z: float) -> str:
        return min(floors.values(), key=lambda f: abs(f.elevation - z)).name

    def _plan_polygon_from_geometry(self, product) -> Optional[Polygon]:
        """Project wall mesh vertices onto XY and make a rectangle/bbox footprint."""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            verts = np.array(shape.geometry.verts, dtype=float).reshape((-1, 3))
        except Exception:
            return None
        if verts.size == 0:
            return None
        verts[:, 0] += self.dx
        verts[:, 1] += self.dy
        verts[:, 2] += self.dz
        xy = verts[:, :2]
        # Use oriented min rectangle if possible; it is more useful than axis-aligned bbox.
        hull = Polygon(xy).convex_hull
        if hull.is_empty:
            return None
        rect = hull.minimum_rotated_rectangle
        if rect.area <= 0:
            minx, miny = xy.min(axis=0)
            maxx, maxy = xy.max(axis=0)
            rect = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        return rect

    def _type_name(self, product) -> str:
        try:
            for rel in product.IsTypedBy or []:
                t = rel.RelatingType
                return t.Name or t.is_a()
        except Exception:
            pass
        return product.is_a()

    def _material_name(self, product) -> str:
        names = []
        try:
            for rel in product.HasAssociations or []:
                mat = getattr(rel, "RelatingMaterial", None)
                if mat is None:
                    continue
                if mat.is_a("IfcMaterial"):
                    names.append(mat.Name)
                elif mat.is_a("IfcMaterialLayerSetUsage"):
                    layers = mat.ForLayerSet.MaterialLayers or []
                    names.extend([ly.Material.Name for ly in layers if ly.Material])
                elif mat.is_a("IfcMaterialLayerSet"):
                    names.extend([ly.Material.Name for ly in mat.MaterialLayers if ly.Material])
        except Exception:
            pass
        return ", ".join(dict.fromkeys([n for n in names if n]))

    @staticmethod
    def _default_attenuation_profile(material: str, type_name: str) -> Dict[float, float]:
        """Default attenuation assumptions by Wi-Fi band.

        These are editable planning assumptions, not certified material test
        values. Higher frequencies usually suffer greater loss through the
        same construction.
        """
        text = f"{material} {type_name}".lower()
        if "concrete" in text or "block" in text:
            return {2400.0: 12.0, 5000.0: 16.0, 6000.0: 20.0}
        if "brick" in text or "masonry" in text:
            return {2400.0: 8.0, 5000.0: 11.0, 6000.0: 14.0}
        if "glass" in text:
            return {2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0}
        if "plaster" in text or "drywall" in text or "partition" in text:
            return {2400.0: 3.0, 5000.0: 4.0, 6000.0: 5.0}
        if "metal" in text or "steel" in text:
            return {2400.0: 20.0, 5000.0: 28.0, 6000.0: 35.0}
        return {2400.0: 5.0, 5000.0: 7.0, 6000.0: 8.0}




class CombinedIFCModel:
    """Utility for merging wall/floor data from several IFC files."""

    @staticmethod
    def merge(target: Dict[str, FloorModel], incoming: Dict[str, FloorModel], source_name: str) -> Dict[str, FloorModel]:
        for floor_name, inc_floor in incoming.items():
            key = CombinedIFCModel._match_floor_key(target, inc_floor) or floor_name
            if key not in target:
                target[key] = FloorModel(name=key, elevation=inc_floor.elevation)
            for wall in inc_floor.walls:
                wall.floor = key
                # Prefix the GUID with source file to avoid collisions and to make
                # table edits stable when several IFCs contain identical object ids.
                wall.guid = f"{source_name}:{wall.guid}"
                target[key].walls.append(wall)
            for space in inc_floor.spaces:
                space.floor = key
                space.guid = f"{source_name}:{space.guid}"
                target[key].spaces.append(space)
        return target

    @staticmethod
    def _match_floor_key(existing: Dict[str, FloorModel], incoming: FloorModel) -> Optional[str]:
        # Prefer exact storey name match. If names differ between discipline IFCs,
        # merge storeys whose elevations are effectively the same.
        if incoming.name in existing:
            return incoming.name
        for key, floor in existing.items():
            if abs(floor.elevation - incoming.elevation) <= 0.05:
                return key
        return None

# ----------------------------- RF engine -----------------------------

class RFEngine:

    @staticmethod
    def free_space_loss_db_at_1m(frequency_mhz: float) -> float:
        """Free-space path loss at 1 metre for the selected Wi-Fi band.

        FSPL(dB) = 32.44 + 20 log10(distance_km) + 20 log10(frequency_MHz).
        At 1 m, distance_km is 0.001.
        """
        frequency_mhz = max(float(frequency_mhz), 1.0)
        return 32.44 + 20.0 * math.log10(0.001) + 20.0 * math.log10(frequency_mhz)
    @staticmethod
    def rssi_at(
        x: float,
        y: float,
        receiver_floor: FloorModel,
        ap: AccessPoint,
        floors: Dict[str, FloorModel],
        patterns: Optional[Dict[str, AntennaPattern]] = None,
        include_inter_floor: bool = True,
    ) -> float:
        """Calculate RSSI at a receiver point on receiver_floor.

        This is a 3D link-budget approximation. For same-floor links it behaves
        like the original model. For inter-floor links it adds:
        - vertical separation between AP mounting height and receiver height;
        - floor/slab penetration loss for every storey boundary crossed;
        - wall intersections on both the AP floor and receiver floor.
        """
        ap_floor = floors.get(ap.floor)
        if ap_floor is None:
            return -120.0
        if ap.floor != receiver_floor.name and not include_inter_floor:
            return -120.0

        horizontal_d = max(math.hypot(x - ap.x, y - ap.y), 0.1)
        ap_z = float(ap_floor.elevation) + float(ap.mount_height_m)
        rx_z = float(receiver_floor.elevation) + float(ap.rx_height_m)
        dz = ap_z - rx_z
        d_3d = max(math.hypot(horizontal_d, dz), 1.0)
        reference_loss = RFEngine.free_space_loss_db_at_1m(ap.frequency_mhz)
        path_loss = reference_loss + 10.0 * ap.path_loss_exponent * math.log10(d_3d)

        bearing = math.degrees(math.atan2(y - ap.y, x - ap.x))
        az_rel = AntennaPattern._wrap_deg(bearing - ap.azimuth_deg)
        elev_angle = math.degrees(math.atan2((rx_z - ap_z), horizontal_d))
        elev_rel = elev_angle + ap.downtilt_deg
        pattern_gain = 0.0
        if patterns:
            pattern = patterns.get(ap.antenna_pattern)
            if pattern:
                pattern_gain = pattern.gain_dbi(az_rel, elev_rel)

        line = LineString([(ap.x, ap.y), (x, y)])
        wall_loss = 0.0
        checked_wall_guids = set()
        for wall in receiver_floor.walls:
            if wall.guid not in checked_wall_guids and wall.polygon.intersects(line):
                wall_loss += wall.attenuation_db_for_frequency(ap.frequency_mhz)
                checked_wall_guids.add(wall.guid)
        if ap_floor.name != receiver_floor.name:
            for wall in ap_floor.walls:
                if wall.guid not in checked_wall_guids and wall.polygon.intersects(line):
                    wall_loss += wall.attenuation_db_for_frequency(ap.frequency_mhz)
                    checked_wall_guids.add(wall.guid)

        floor_loss = RFEngine.floor_penetration_loss_db(receiver_floor, ap_floor, floors, ap.frequency_mhz)
        return ap.tx_power_dbm + pattern_gain - path_loss - wall_loss - floor_loss

    @staticmethod
    def floor_penetration_loss_db(receiver_floor: FloorModel, ap_floor: FloorModel, floors: Dict[str, FloorModel], frequency_mhz: float) -> float:
        if receiver_floor.name == ap_floor.name:
            return 0.0
        ordered = sorted(floors.values(), key=lambda f: (f.elevation, f.name))
        try:
            rx_i = next(i for i, f in enumerate(ordered) if f.name == receiver_floor.name)
            ap_i = next(i for i, f in enumerate(ordered) if f.name == ap_floor.name)
        except StopIteration:
            # Fallback when floor references are incomplete: one slab loss per
            # approximate 3.5 m of vertical separation, minimum one slab.
            crossed = max(1, int(round(abs(receiver_floor.elevation - ap_floor.elevation) / 3.5)))
            return crossed * receiver_floor.slab_attenuation_db_for_frequency(frequency_mhz)
        lo, hi = sorted((rx_i, ap_i))
        crossed_boundaries = ordered[lo + 1:hi + 1]
        if not crossed_boundaries:
            crossed_boundaries = [receiver_floor]
        return sum(f.slab_attenuation_db_for_frequency(frequency_mhz) for f in crossed_boundaries)

    @staticmethod
    def simulate(
        floor: FloorModel,
        floors: Dict[str, FloorModel],
        aps: List[AccessPoint],
        resolution_m: float = 2.0,
        patterns: Optional[Dict[str, AntennaPattern]] = None,
        include_inter_floor: bool = True,
    ) -> Optional[SimulationResult]:
        if not floor.walls and not floor.spaces:
            return None
        if not aps:
            return None
        bounds = RFEngine._floor_bounds(floor, aps)
        minx, miny, maxx, maxy = bounds
        xs = np.arange(minx, maxx + resolution_m, resolution_m)
        ys = np.arange(miny, maxy + resolution_m, resolution_m)
        grid = np.full((len(ys), len(xs)), -120.0)
        candidate_aps = [a for a in aps if include_inter_floor or a.floor == floor.name]
        if not candidate_aps:
            return None
        for iy, yy in enumerate(ys):
            for ix, xx in enumerate(xs):
                grid[iy, ix] = max(
                    RFEngine.rssi_at(xx, yy, floor, ap, floors, patterns, include_inter_floor)
                    for ap in candidate_aps
                )
        return SimulationResult(xs=xs, ys=ys, rssi=grid)

    @staticmethod
    def _floor_bounds(floor: FloorModel, aps: List[AccessPoint]) -> Tuple[float, float, float, float]:
        bounds = [w.polygon.bounds for w in floor.walls] + [s.polygon.bounds for s in floor.spaces]
        if not bounds:
            bounds = [(0.0, 0.0, 1.0, 1.0)]
        for ap in aps:
            if ap.floor == floor.name:
                bounds.append((ap.x, ap.y, ap.x, ap.y))
        minx = min(b[0] for b in bounds) - 5
        miny = min(b[1] for b in bounds) - 5
        maxx = max(b[2] for b in bounds) + 5
        maxy = max(b[3] for b in bounds) + 5
        return minx, miny, maxx, maxy



# ----------------------------- Threaded IFC loading -----------------------------

class IFCLoadSignals(QObject):
    finished = Signal(object, object, str)  # path, floors, source name
    error = Signal(object, str)


class IFCLoadWorker(QRunnable):
    """Loads one IFC file away from the Qt GUI thread."""

    def __init__(self, path: Path, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        super().__init__()
        self.path = path
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.signals = IFCLoadSignals()

    @Slot()
    def run(self):
        try:
            floors = IFCModelLoader(self.path, self.dx, self.dy, self.dz).load()
            self.signals.finished.emit(self.path, floors, self.path.name)
        except Exception as exc:
            self.signals.error.emit(self.path, str(exc))

# ----------------------------- GUI -----------------------------

class PlanView(QGraphicsView):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)
        self.scale(1, -1)  # IFC Y-up style plan view

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if self.main.floor is None:
            return
        pos = self.mapToScene(event.position().toPoint())
        self.main.add_ap(pos.x(), pos.y())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Attenuation Simulator")
        self.resize(1400, 900)
        self.floors: Dict[str, FloorModel] = {}
        self.loaded_ifc_paths: List[Path] = []
        self.floor: Optional[FloorModel] = None
        self.aps: List[AccessPoint] = []
        self.antenna_patterns: Dict[str, AntennaPattern] = built_in_antenna_patterns()
        self.last_result: Optional[SimulationResult] = None
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(max(1, min(4, (os.cpu_count() or 2))))
        self._load_pending = 0
        self._load_errors: List[str] = []
        self._loading_replace = False
        self._loading_active = False
        self.heatmap_settings = HeatmapSettings.default()
        self.heatmap_settings_path: Optional[Path] = None

        self.view = PlanView(self)
        self.rssi_legend = QLabel()
        self.rssi_legend.setWordWrap(True)
        self.rssi_legend.setMinimumHeight(72)
        self.rssi_legend.setTextFormat(Qt.RichText)
        self.rssi_legend.setStyleSheet(
            "QLabel { background: #f6f6f6; border-top: 1px solid #b8b8b8; padding: 8px; }"
        )
        self.floor_combo = QComboBox()
        self.wall_table = QTableWidget(0, 7)
        self.wall_table.setHorizontalHeaderLabels([
            "Wall/material/type",
            "2.4 GHz dB",
            "5 GHz dB",
            "6 GHz dB",
            "Name",
            "Source IFC",
            "GUID",
        ])
        self.wall_table.itemChanged.connect(self._wall_table_changed)

        self.ap_table = QTableWidget(0, 9)
        self.ap_table.setHorizontalHeaderLabels([
            "AP", "Floor", "X", "Y", "Pattern", "Azimuth", "Downtilt", "TX dBm", "Freq MHz"
        ])
        self.ap_table.itemChanged.connect(self._ap_table_changed)

        self.resolution = QDoubleSpinBox()
        self.resolution.setRange(0.5, 10.0)
        self.resolution.setValue(2.0)
        self.resolution.setSuffix(" m")
        self.tx_power = QDoubleSpinBox()
        self.tx_power.setRange(-10.0, 40.0)
        self.tx_power.setValue(20.0)
        self.tx_power.setSuffix(" dBm")
        self.freq = QDoubleSpinBox()
        self.freq.setRange(2400.0, 7125.0)
        self.freq.setValue(2400.0)
        self.freq.setSingleStep(100.0)
        self.freq.setSuffix(" MHz")
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(list(self.antenna_patterns.keys()))
        self.azimuth = QDoubleSpinBox()
        self.azimuth.setRange(-180.0, 180.0)
        self.azimuth.setValue(0.0)
        self.azimuth.setSuffix("°")
        self.downtilt = QDoubleSpinBox()
        self.downtilt.setRange(-45.0, 90.0)
        self.downtilt.setValue(0.0)
        self.downtilt.setSuffix("°")
        self.mount_height = QDoubleSpinBox()
        self.mount_height.setRange(0.5, 10.0)
        self.mount_height.setValue(2.7)
        self.mount_height.setSuffix(" m")
        self.rx_height = QDoubleSpinBox()
        self.rx_height.setRange(0.1, 3.0)
        self.rx_height.setValue(1.2)
        self.rx_height.setSuffix(" m")
        self.ple = QDoubleSpinBox()
        self.ple.setRange(1.6, 5.0)
        self.ple.setValue(2.2)
        self.min_client_rssi = QDoubleSpinBox()
        self.min_client_rssi.setRange(-100.0, -40.0)
        self.min_client_rssi.setValue(self.heatmap_settings.minimum_client_rssi_dbm)
        self.min_client_rssi.setSuffix(" dBm")
        self.min_client_rssi.valueChanged.connect(self._minimum_rssi_changed)
        self.include_inter_floor = QCheckBox("Include APs from other floors")
        self.include_inter_floor.setChecked(True)
        self.slab_att_24 = QDoubleSpinBox()
        self.slab_att_24.setRange(0.0, 80.0)
        self.slab_att_24.setValue(12.0)
        self.slab_att_24.setSuffix(" dB")
        self.slab_att_5 = QDoubleSpinBox()
        self.slab_att_5.setRange(0.0, 80.0)
        self.slab_att_5.setValue(18.0)
        self.slab_att_5.setSuffix(" dB")
        self.slab_att_6 = QDoubleSpinBox()
        self.slab_att_6.setRange(0.0, 80.0)
        self.slab_att_6.setValue(22.0)
        self.slab_att_6.setSuffix(" dB")

        controls = QWidget()
        form = QFormLayout(controls)
        form.addRow("Floor", self.floor_combo)
        form.addRow("Grid resolution", self.resolution)
        form.addRow("AP TX power", self.tx_power)
        form.addRow("Frequency / Wi-Fi band", self.freq)
        form.addRow("AP antenna pattern", self.pattern_combo)
        form.addRow("AP azimuth", self.azimuth)
        form.addRow("AP downtilt", self.downtilt)
        form.addRow("AP mount height", self.mount_height)
        form.addRow("Receiver height", self.rx_height)
        form.addRow("Path loss exponent", self.ple)
        form.addRow("Client disconnect RSSI", self.min_client_rssi)
        form.addRow("Inter-floor RF", self.include_inter_floor)
        form.addRow("Slab loss 2.4 GHz", self.slab_att_24)
        form.addRow("Slab loss 5 GHz", self.slab_att_5)
        form.addRow("Slab loss 6 GHz", self.slab_att_6)
        form.addRow(QLabel("Double-click the model to place an AP using the selected pattern/orientation."))

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(controls)
        side_layout.addWidget(QLabel("Access points and antenna patterns"))
        side_layout.addWidget(self.ap_table)
        side_layout.addWidget(QLabel("Wall attenuation values"))
        side_layout.addWidget(self.wall_table)

        model_panel = QWidget()
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(0)
        model_layout.addWidget(self.view, 1)
        model_layout.addWidget(self.rssi_legend, 0)

        split = QSplitter()
        split.addWidget(model_panel)
        split.addWidget(side)
        split.setSizes([1000, 400])
        self.setCentralWidget(split)
        self._update_rssi_legend()

        tb = QToolBar("Main")
        self.addToolBar(tb)
        self.open_action = QAction("Open IFC(s)", self)
        self.open_action.triggered.connect(self.open_ifc)
        self.add_action = QAction("Add IFC(s)", self)
        self.add_action.triggered.connect(self.add_ifc)
        self.sim_action = QAction("Simulate RSSI", self)
        self.sim_action.triggered.connect(self.simulate)
        self.export_action = QAction("Export CSV", self)
        self.export_action.triggered.connect(self.export_csv)
        self.clear_ap_action = QAction("Clear APs", self)
        self.clear_ap_action.triggered.connect(self.clear_aps)
        self.load_pattern_action = QAction("Load pattern CSV", self)
        self.load_pattern_action.triggered.connect(self.load_pattern_csv)
        self.load_heatmap_settings_action = QAction("Load heatmap settings", self)
        self.load_heatmap_settings_action.triggered.connect(self.load_heatmap_settings)
        tb.addActions([self.open_action, self.add_action, self.sim_action, self.export_action, self.clear_ap_action, self.load_pattern_action, self.load_heatmap_settings_action])
        self.floor_combo.currentTextChanged.connect(self.select_floor)
        self.include_inter_floor.stateChanged.connect(lambda *_: self.draw_floor())

    def open_ifc(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open IFC file(s)", "", "IFC files (*.ifc);;All files (*.*)")
        if not paths:
            return
        self.floors = {}
        self.loaded_ifc_paths = []
        self.aps.clear()
        self._load_ifc_paths([Path(p) for p in paths], replace=True)

    def add_ifc(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add IFC file(s) to current model", "", "IFC files (*.ifc);;All files (*.*)")
        if not paths:
            return
        self._load_ifc_paths([Path(p) for p in paths], replace=False)

    def _load_ifc_paths(self, paths: List[Path], replace: bool = False):
        """Start parallel IFC parsing jobs and merge their outputs on completion."""
        if self._loading_active:
            QMessageBox.information(self, "IFC load in progress", "Wait for the current IFC load to finish before adding more files.")
            return
        if replace:
            self.floors = {}
            self.loaded_ifc_paths = []
            self.aps.clear()
            self.last_result = None

        unique_paths = []
        seen = {p.resolve() for p in self.loaded_ifc_paths if p.exists()}
        for path in paths:
            rp = path.resolve() if path.exists() else path
            if rp not in seen:
                unique_paths.append(path)
                seen.add(rp)
        if not unique_paths:
            return

        self._load_pending = len(unique_paths)
        self._load_errors = []
        self._loading_replace = replace
        self._loading_active = True
        self._set_loading_ui(True)
        self.statusBar().showMessage(
            f"Loading {self._load_pending} IFC file(s) using {self.thread_pool.maxThreadCount()} worker thread(s)..."
        )

        for path in unique_paths:
            worker = IFCLoadWorker(path)
            worker.signals.finished.connect(self._ifc_worker_finished)
            worker.signals.error.connect(self._ifc_worker_error)
            self.thread_pool.start(worker)

    @Slot(object, object, str)
    def _ifc_worker_finished(self, path_obj, incoming, source_name: str):
        path = Path(path_obj)
        CombinedIFCModel.merge(self.floors, incoming, source_name)
        self.loaded_ifc_paths.append(path)
        self._load_pending -= 1
        total_loaded = len(self.loaded_ifc_paths)
        self.statusBar().showMessage(f"Loaded {path.name}. Waiting for {self._load_pending} IFC file(s)... Total loaded: {total_loaded}")
        self._finish_ifc_batch_if_ready()

    @Slot(object, str)
    def _ifc_worker_error(self, path_obj, message: str):
        path = Path(path_obj)
        self._load_errors.append(f"{path.name}: {message}")
        self._load_pending -= 1
        self.statusBar().showMessage(f"Failed to load {path.name}. Waiting for {self._load_pending} IFC file(s)...")
        self._finish_ifc_batch_if_ready()

    def _finish_ifc_batch_if_ready(self):
        if self._load_pending > 0:
            return
        self._loading_active = False
        self._set_loading_ui(False)
        self._refresh_floor_combo()
        total_walls = sum(len(f.walls) for f in self.floors.values())
        total_spaces = sum(len(f.spaces) for f in self.floors.values())
        msg = f"Loaded {len(self.loaded_ifc_paths)} IFC file(s), {len(self.floors)} floor(s), {total_walls} wall(s), {total_spaces} space(s)"
        if self._load_errors:
            QMessageBox.warning(self, "Some IFC files failed", "\n".join(self._load_errors))
            msg += f". {len(self._load_errors)} file(s) failed."
        self.statusBar().showMessage(msg)

    def _refresh_floor_combo(self):
        self.floor_combo.blockSignals(True)
        current = self.floor_combo.currentText()
        self.floor_combo.clear()
        floor_names = sorted(self.floors.keys(), key=lambda n: (self.floors[n].elevation, n))
        self.floor_combo.addItems(floor_names)
        if current in self.floors:
            self.floor_combo.setCurrentText(current)
        self.floor_combo.blockSignals(False)
        self.select_floor(self.floor_combo.currentText())

    def _set_loading_ui(self, loading: bool):
        self.open_action.setEnabled(not loading)
        self.add_action.setEnabled(not loading)
        self.sim_action.setEnabled(not loading)
        self.export_action.setEnabled(not loading)
        self.clear_ap_action.setEnabled(not loading)
        self.load_pattern_action.setEnabled(not loading)
        self.floor_combo.setEnabled(not loading)

    def select_floor(self, name: str):
        self.floor = self.floors.get(name)
        self.last_result = None
        self._load_slab_attenuation_to_ui()
        self.draw_floor()
        self.populate_ap_table()
        self.populate_wall_table()

    def add_ap(self, x: float, y: float):
        if not self.floor:
            return
        ap = AccessPoint(
            name=f"AP-{len(self.aps)+1}",
            x=x,
            y=y,
            floor=self.floor.name,
            tx_power_dbm=float(self.tx_power.value()),
            frequency_mhz=float(self.freq.value()),
            path_loss_exponent=float(self.ple.value()),
            antenna_pattern=self.pattern_combo.currentText(),
            azimuth_deg=float(self.azimuth.value()),
            downtilt_deg=float(self.downtilt.value()),
            mount_height_m=float(self.mount_height.value()),
            rx_height_m=float(self.rx_height.value()),
        )
        self.aps.append(ap)
        self.draw_floor()
        self.populate_ap_table()

    def clear_aps(self):
        self.aps = [a for a in self.aps if not self.floor or a.floor != self.floor.name]
        self.draw_floor()
        self.populate_ap_table()

    def draw_floor(self):
        scene = self.view.scene()
        scene.clear()
        if not self.floor:
            return
        # Draw heatmap first so the building geometry remains visible above it.
        if self.last_result:
            self._draw_heatmap(self.last_result)

        # Draw spaces/floor areas with a stronger outline and subtle fill so the
        # extent of the floor remains readable even before a simulation is run.
        for space in self.floor.spaces:
            coords = list(space.polygon.exterior.coords)
            poly = QPolygonF([QPointF(x, y) for x, y in coords])
            item = QGraphicsPolygonItem(poly)
            item.setPen(QPen(QColor(70, 110, 160), 0.35))
            item.setBrush(QBrush(QColor(210, 230, 250, 55)))
            scene.addItem(item)

            if space.name:
                centroid = space.polygon.representative_point()
                label = QGraphicsSimpleTextItem(str(space.name))
                label.setBrush(QBrush(QColor(25, 45, 70)))
                label.setFont(QFont("Arial", 8))
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                label.setZValue(20)
                label.setPos(QPointF(centroid.x(), centroid.y()))
                scene.addItem(label)

        for wall in self.floor.walls:
            coords = list(wall.polygon.exterior.coords)
            poly = QPolygonF([QPointF(x, y) for x, y in coords])
            item = QGraphicsPolygonItem(poly)
            item.setPen(QPen(QColor(20, 20, 20), 0.45))
            item.setBrush(QBrush(QColor(105, 105, 105, 215)))
            item.setZValue(10)
            item.setFlag(QGraphicsItem.ItemIsSelectable, True)
            scene.addItem(item)
        visible_aps = [a for a in self.aps if a.floor == self.floor.name or self.include_inter_floor.isChecked()]
        for ap in visible_aps:
            same_floor = ap.floor == self.floor.name
            radius = 0.75 if same_floor else 0.45
            colour = QColor(0, 80, 255) if same_floor else QColor(120, 0, 180)
            dot = QGraphicsEllipseItem(ap.x - radius, ap.y - radius, radius * 2.0, radius * 2.0)
            dot.setBrush(QBrush(colour))
            dot.setPen(QPen(QColor(0, 0, 80), 0.2))
            dot.setZValue(30 if same_floor else 25)
            scene.addItem(dot)
            # Draw a short boresight arrow so directional antenna orientation can be checked.
            length = 5.0 if same_floor else 3.0
            ang = math.radians(ap.azimuth_deg)
            x2 = ap.x + length * math.cos(ang)
            y2 = ap.y + length * math.sin(ang)
            scene.addLine(ap.x, ap.y, x2, y2, QPen(colour, 0.25))
            if not same_floor:
                label = QGraphicsSimpleTextItem(ap.floor)
                label.setBrush(QBrush(colour))
                label.setFont(QFont("Arial", 7))
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                label.setZValue(31)
                label.setPos(QPointF(ap.x + 0.8, ap.y + 0.8))
                scene.addItem(label)
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-10, -10, 10, 10))
        self.view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def _draw_heatmap(self, result: SimulationResult):
        """Draw smooth filled RSSI contours, isolines and sampled RSSI points.

        contourpy is used so the coloured fill follows the interpolated contour
        boundary rather than being drawn as square grid-cell blocks. Text sizes
        and sample marker size are controlled by rf_heatmap_settings.json.
        """
        scene = self.view.scene()
        if len(result.xs) < 2 or len(result.ys) < 2:
            return
        if contourpy is None:
            QMessageBox.warning(self, "Missing dependency", "contourpy is required for smooth filled contours. Run: pip install contourpy")
            return

        grid = np.asarray(result.rssi, dtype=float)
        rows, cols = grid.shape
        if rows < 2 or cols < 2:
            return

        levels = sorted({float(v) for v in self.heatmap_settings.isoline_bands_dbm}, reverse=True)
        if not levels:
            return

        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            return
        data_min = float(np.nanmin(finite))
        data_max = float(np.nanmax(finite))
        fill_low = min(data_min, min(levels)) - 0.1
        fill_high = max(data_max, max(levels)) + 0.1

        try:
            cg = contourpy.contour_generator(
                x=np.asarray(result.xs, dtype=float),
                y=np.asarray(result.ys, dtype=float),
                z=grid,
                name="serial",
                line_type=contourpy.LineType.Separate,
                fill_type=contourpy.FillType.OuterOffset,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Contour generation failed", str(exc))
            return

        def path_from_outer_offset(polygons, offsets) -> QPainterPath:
            path = QPainterPath()
            for points, offs in zip(polygons, offsets):
                if points is None or offs is None:
                    continue
                pts = np.asarray(points, dtype=float)
                off = np.asarray(offs, dtype=int)
                for i in range(len(off) - 1):
                    ring = pts[off[i]:off[i + 1]]
                    if len(ring) < 3:
                        continue
                    path.moveTo(QPointF(float(ring[0, 0]), float(ring[0, 1])))
                    for x, y in ring[1:]:
                        path.lineTo(QPointF(float(x), float(y)))
                    path.closeSubpath()
            return path

        def add_filled_band(lower: float, upper: float, colour_ref: float):
            if upper <= data_min or lower >= data_max:
                return
            lower_c = max(lower, fill_low)
            upper_c = min(upper, fill_high)
            if upper_c <= lower_c:
                return
            try:
                polygons, offsets = cg.filled(lower_c, upper_c)
            except Exception:
                return
            path = path_from_outer_offset(polygons, offsets)
            if path.isEmpty():
                return
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(self.heatmap_settings.colour_for_rssi(colour_ref)))
            item.setPen(QPen(Qt.NoPen))
            item.setZValue(-20)
            scene.addItem(item)

        # Smooth filled bands between configured isoline thresholds.
        bounds = [fill_low] + sorted(levels) + [fill_high]
        for lower, upper in zip(bounds[:-1], bounds[1:]):
            add_filled_band(lower, upper, (lower + upper) / 2.0)

        contour_font_size = max(1, int(self.heatmap_settings.contour_label_font_size))
        contour_label_limit = 5
        contour_min_spacing = 18.0

        for level in levels:
            if level < data_min or level > data_max:
                continue
            line_colour = self.heatmap_settings.colour_for_rssi(level)
            line_colour.setAlpha(245)
            pen = QPen(line_colour, 0.36)
            pen.setCosmetic(True)
            pen.setStyle(Qt.SolidLine)

            try:
                lines = cg.lines(level)
            except Exception:
                continue

            label_positions: List[Tuple[float, float]] = []
            labels_on_level = 0
            for line in lines:
                pts = np.asarray(line, dtype=float)
                if len(pts) < 2:
                    continue
                path = QPainterPath(QPointF(float(pts[0, 0]), float(pts[0, 1])))
                for x, y in pts[1:]:
                    path.lineTo(QPointF(float(x), float(y)))
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(QBrush(Qt.NoBrush))
                item.setZValue(-4)
                scene.addItem(item)

                if labels_on_level >= contour_label_limit:
                    continue
                seg = np.diff(pts, axis=0)
                seg_len = np.hypot(seg[:, 0], seg[:, 1])
                total_len = float(np.sum(seg_len))
                if total_len < 4.0:
                    continue
                target = total_len * 0.5
                acc = 0.0
                chosen = None
                for i, length in enumerate(seg_len):
                    if acc + length >= target and length > 0:
                        t = (target - acc) / length
                        x = pts[i, 0] + (pts[i + 1, 0] - pts[i, 0]) * t
                        y = pts[i, 1] + (pts[i + 1, 1] - pts[i, 1]) * t
                        angle = math.degrees(math.atan2(seg[i, 1], seg[i, 0]))
                        chosen = (float(x), float(y), angle)
                        break
                    acc += float(length)
                if chosen is None:
                    continue
                x, y, angle = chosen
                if any(math.hypot(x - px, y - py) < contour_min_spacing for px, py in label_positions):
                    continue
                label = QGraphicsSimpleTextItem(f"{level:.0f} dBm")
                label.setBrush(QBrush(QColor(15, 15, 15)))
                label.setFont(QFont("Arial", contour_font_size, QFont.Bold))
                label.setZValue(-2)
                label.setPos(QPointF(x, y))
                label.setRotation(angle)
                scene.addItem(label)
                label_positions.append((x, y))
                labels_on_level += 1

        # Small blue + markers showing sample locations only.
        stride_x = max(1, int(self.heatmap_settings.sample_stride_x))
        stride_y = max(1, int(self.heatmap_settings.sample_stride_y))
        sample_font_size = max(1, int(self.heatmap_settings.sample_label_font_size))
        cross_size = max(0.05, float(self.heatmap_settings.sample_cross_size))
        sample_pen = QPen(QColor(0, 80, 255), 0.0)
        sample_pen.setCosmetic(True)
        for iy in range(0, rows, stride_y):
            for ix in range(0, cols, stride_x):
                rssi = float(grid[iy, ix])
                if not math.isfinite(rssi):
                    continue
                x = float(result.xs[ix])
                y = float(result.ys[iy])
                h = scene.addLine(x - cross_size, y, x + cross_size, y, sample_pen)
                vline = scene.addLine(x, y - cross_size, x, y + cross_size, sample_pen)
                h.setZValue(35)
                vline.setZValue(35)
                label = QGraphicsSimpleTextItem(f"{rssi:.0f} dBm")
                label.setBrush(QBrush(QColor(0, 80, 255)))
                label.setFont(QFont("Arial", sample_font_size))
                label.setZValue(36)
                label.setPos(QPointF(x + cross_size * 1.8, y + cross_size * 1.2))
                scene.addItem(label)

    def _update_rssi_legend(self):
        """Render the RSSI zone key in a fixed panel below the IFC view.

        The legend is deliberately outside the QGraphicsScene so it does not
        pan/zoom with the IFC model and does not obscure floor geometry.
        """
        if not hasattr(self, "rssi_legend"):
            return
        if not self.heatmap_settings.zones:
            self.rssi_legend.setText("<b>RSSI zones</b>: no heatmap settings loaded")
            return

        band_text = ", ".join(f"{v:.0f}" for v in self.heatmap_settings.isoline_bands_dbm)
        pattern_count = len(self.heatmap_settings.rf_pattern_files)
        pieces = [
            f"<b>RSSI isolines</b> &nbsp; "
            f"<span style='color:#555;'>Bands: {band_text} dBm. "
            f"Clients disconnect below {self.heatmap_settings.minimum_client_rssi_dbm:.0f} dBm. "
            f"Pattern files in settings: {pattern_count}</span>"
        ]
        for zone in self.heatmap_settings.zones:
            colour = QColor(zone.colour)
            if not colour.isValid():
                colour = QColor("#555555")
            fg = "#ffffff" if colour.lightness() < 130 else "#202020"
            pieces.append(
                "<span style='display:inline-block; margin-left:10px; "
                f"padding:3px 7px; border:1px solid #666; background:{colour.name()}; color:{fg};'>"
                f"{zone.name}: {zone.min_dbm:.0f} to {zone.max_dbm:.0f} dBm"
                "</span>"
            )
        self.rssi_legend.setText(" &nbsp; ".join(pieces))

    def _rssi_colour(self, rssi: float) -> QColor:
        return self.heatmap_settings.colour_for_rssi(rssi)

    def _minimum_rssi_changed(self, value: float):
        self.heatmap_settings.minimum_client_rssi_dbm = float(value)
        self.heatmap_settings.ensure_disconnect_zone()
        self._update_rssi_legend()
        if self.last_result is not None:
            self.draw_floor()

    def load_heatmap_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load heatmap settings", "rf_heatmap_settings.json", "JSON files (*.json);;All files (*.*)")
        if not path:
            return
        try:
            self.heatmap_settings = HeatmapSettings.from_json_file(Path(path))
            self.heatmap_settings_path = Path(path)
            self.min_client_rssi.blockSignals(True)
            self.min_client_rssi.setValue(self.heatmap_settings.minimum_client_rssi_dbm)
            self.min_client_rssi.blockSignals(False)
            loaded_patterns = []
            base_dir = Path(path).parent
            for pattern_file in self.heatmap_settings.rf_pattern_files:
                pattern_path = Path(pattern_file)
                if not pattern_path.is_absolute():
                    pattern_path = base_dir / pattern_path
                loaded_patterns.append(self.load_pattern_csv_path(pattern_path))
            self._update_rssi_legend()
            self.draw_floor()
            extra = f"\nLoaded RF pattern files: {', '.join(loaded_patterns)}" if loaded_patterns else ""
            QMessageBox.information(self, "Heatmap settings loaded", f"Loaded RSSI isoline settings from {Path(path).name}{extra}")
        except Exception as exc:
            QMessageBox.warning(self, "Heatmap settings failed", str(exc))

    def populate_ap_table(self):
        self.ap_table.blockSignals(True)
        self.ap_table.setRowCount(0)
        if not self.floor:
            self.ap_table.blockSignals(False)
            return
        for ap in [a for a in self.aps if a.floor == self.floor.name]:
            row = self.ap_table.rowCount()
            self.ap_table.insertRow(row)
            values = [
                ap.name, ap.floor, f"{ap.x:.2f}", f"{ap.y:.2f}", ap.antenna_pattern,
                f"{ap.azimuth_deg:.1f}", f"{ap.downtilt_deg:.1f}", f"{ap.tx_power_dbm:.1f}", f"{ap.frequency_mhz:.0f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, ap.name)
                self.ap_table.setItem(row, col, item)
        self.ap_table.resizeColumnsToContents()
        self.ap_table.blockSignals(False)

    def _ap_table_changed(self, item: QTableWidgetItem):
        if not self.floor:
            return
        ap_name = item.data(Qt.UserRole)
        ap = next((a for a in self.aps if a.name == ap_name), None)
        if not ap:
            return
        try:
            if item.column() == 4:
                if item.text() in self.antenna_patterns:
                    ap.antenna_pattern = item.text()
            elif item.column() == 5:
                ap.azimuth_deg = float(item.text())
            elif item.column() == 6:
                ap.downtilt_deg = float(item.text())
            elif item.column() == 7:
                ap.tx_power_dbm = float(item.text())
            elif item.column() == 8:
                ap.frequency_mhz = float(item.text())
        except ValueError:
            return
        self.last_result = None
        self.draw_floor()

    def populate_wall_table(self):
        self.wall_table.blockSignals(True)
        self.wall_table.setRowCount(0)
        if not self.floor:
            self.wall_table.blockSignals(False)
            return
        for wall in self.floor.walls:
            row = self.wall_table.rowCount()
            self.wall_table.insertRow(row)
            self.wall_table.setItem(row, 0, QTableWidgetItem(wall.label))
            for col, band in [(1, 2400.0), (2, 5000.0), (3, 6000.0)]:
                att = QTableWidgetItem(str(wall.attenuation_by_band_db.get(band, 0.0)))
                att.setData(Qt.UserRole, wall.guid)
                att.setData(Qt.UserRole + 1, band)
                self.wall_table.setItem(row, col, att)
            self.wall_table.setItem(row, 4, QTableWidgetItem(wall.name))
            self.wall_table.setItem(row, 5, QTableWidgetItem(wall.source_file))
            self.wall_table.setItem(row, 6, QTableWidgetItem(wall.guid))
        self.wall_table.resizeColumnsToContents()
        self.wall_table.blockSignals(False)

    def _wall_table_changed(self, item: QTableWidgetItem):
        if item.column() not in (1, 2, 3) or not self.floor:
            return
        guid = item.data(Qt.UserRole)
        band = item.data(Qt.UserRole + 1)
        try:
            val = float(item.text())
        except ValueError:
            return
        for wall in self.floor.walls:
            if wall.guid == guid:
                wall.attenuation_by_band_db[float(band)] = val
                break

    def _sync_slab_attenuation_from_ui(self):
        if not self.floor:
            return
        self.floor.slab_attenuation_by_band_db = {
            2400.0: float(self.slab_att_24.value()),
            5000.0: float(self.slab_att_5.value()),
            6000.0: float(self.slab_att_6.value()),
        }

    def _load_slab_attenuation_to_ui(self):
        if not self.floor:
            return
        self.slab_att_24.blockSignals(True)
        self.slab_att_5.blockSignals(True)
        self.slab_att_6.blockSignals(True)
        self.slab_att_24.setValue(float(self.floor.slab_attenuation_by_band_db.get(2400.0, 12.0)))
        self.slab_att_5.setValue(float(self.floor.slab_attenuation_by_band_db.get(5000.0, 18.0)))
        self.slab_att_6.setValue(float(self.floor.slab_attenuation_by_band_db.get(6000.0, 22.0)))
        self.slab_att_24.blockSignals(False)
        self.slab_att_5.blockSignals(False)
        self.slab_att_6.blockSignals(False)

    def simulate(self):
        if not self.floor:
            return
        self._sync_slab_attenuation_from_ui()
        if not self.aps:
            QMessageBox.information(self, "No APs", "Double-click the floor plan to place at least one AP.")
            return
        for ap in self.aps:
            ap.path_loss_exponent = float(self.ple.value())
            # Keep per-AP TX/frequency/pattern edits from the AP table.
            ap.rx_height_m = float(self.rx_height.value())
        self.last_result = RFEngine.simulate(
            self.floor,
            self.floors,
            self.aps,
            self.resolution.value(),
            self.antenna_patterns,
            include_inter_floor=self.include_inter_floor.isChecked(),
        )
        self.draw_floor()

    def load_pattern_csv_path(self, path: Path) -> str:
        """Load one antenna pattern CSV and return the pattern name."""
        azimuth_points: List[Tuple[float, float]] = []
        elevation_points: List[Tuple[float, float]] = []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                plane = (row.get("plane") or "").strip().lower()
                angle = float(row.get("angle_deg") or row.get("angle") or 0.0)
                gain = float(row.get("gain_dbi") or row.get("gain") or 0.0)
                if plane.startswith("az"):
                    azimuth_points.append((angle, gain))
                elif plane.startswith("el"):
                    elevation_points.append((angle, gain))
        if not azimuth_points and not elevation_points:
            raise ValueError(f"No azimuth/elevation points found in {path}")
        name = path.stem
        peak = max([g for _, g in azimuth_points + elevation_points] or [0.0])
        self.antenna_patterns[name] = AntennaPattern(
            name=name,
            peak_gain_dbi=peak,
            azimuth_points=azimuth_points,
            elevation_points=elevation_points,
        )
        if hasattr(self, "pattern_combo"):
            current = self.pattern_combo.currentText()
            self.pattern_combo.clear()
            self.pattern_combo.addItems(list(self.antenna_patterns.keys()))
            self.pattern_combo.setCurrentText(current if current in self.antenna_patterns else name)
        return name

    def load_pattern_csv(self):
        """Load a manufacturer-style antenna pattern CSV.

        Expected columns: plane,angle_deg,gain_dbi
        plane must be either azimuth or elevation. A single file creates one
        new pattern named after the CSV filename.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load antenna pattern CSV", "", "CSV files (*.csv);;All files (*.*)")
        if not path:
            return
        try:
            name = self.load_pattern_csv_path(Path(path))
            self.pattern_combo.setCurrentText(name)
            QMessageBox.information(self, "Pattern loaded", f"Loaded antenna pattern: {name}")
        except Exception as exc:
            QMessageBox.warning(self, "Pattern load failed", str(exc))

    def export_csv(self):
        if not self.last_result:
            QMessageBox.information(self, "No result", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export RSSI CSV", "rssi_heatmap.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["floor", "x", "y", "frequency_mhz", "rssi_dbm", "rssi_zone", "client_connected", "disconnect_threshold_dbm", "ap_count", "include_inter_floor", "slab_loss_24_db", "slab_loss_5_db", "slab_loss_6_db", "patterns_used"])
            for iy, y in enumerate(self.last_result.ys):
                for ix, x in enumerate(self.last_result.xs):
                    rssi = float(self.last_result.rssi[iy, ix])
                    zone = self.heatmap_settings.zone_for_rssi(rssi)
                    writer.writerow([self.floor.name if self.floor else "", x, y, float(self.freq.value()), rssi, zone.name, rssi >= self.heatmap_settings.minimum_client_rssi_dbm, self.heatmap_settings.minimum_client_rssi_dbm, len(self.aps), self.include_inter_floor.isChecked(), float(self.slab_att_24.value()), float(self.slab_att_5.value()), float(self.slab_att_6.value()), ";".join(sorted({a.antenna_pattern for a in self.aps}))])


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
