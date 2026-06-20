"""
RF Attenuation Simulator - IFC Wi-Fi RSSI planning tool.

Run:
    pip install PySide6 numpy ifcopenshell shapely contourpy
    python rf_simulator.py
"""
from __future__ import annotations

import csv
import concurrent.futures
import json
import math
import multiprocessing
import os
import uuid
from rf_dxf_prealign import DxfPreAlignDialog, SimilarityTransform2D, two_point_transform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    from scipy.ndimage import zoom as scipy_zoom
except Exception:
    scipy_zoom = None
try:
    import contourpy
except Exception:
    contourpy = None

from PySide6.QtCore import QPointF, Qt, Slot, QTimer
from PySide6.QtGui import QAction, QColor, QBrush, QFont, QPen, QPolygonF, QPainterPath, QPalette, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressDialog,
)

try:
    import ifcopenshell
    import ifcopenshell.geom
except Exception:  # pragma: no cover
    ifcopenshell = None

try:
    import ezdxf
except Exception:  # pragma: no cover
    ezdxf = None

try:
    from shapely.geometry import LineString, Point, Polygon, box
    from shapely.ops import unary_union
    from shapely.affinity import affine_transform
    try:
        from shapely import concave_hull as shapely_concave_hull
    except Exception:  # Shapely < 2.0
        shapely_concave_hull = None
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
    # Vertical extent in project/model metres. IFC containment often assigns an
    # element to only one storey, but shafts, curtain walls, risers and facade
    # elements may physically span several storeys. These values allow the UI
    # and RF engine to include the same construction on every floor it crosses.
    z_min: float = 0.0
    z_max: float = 0.0
    source_storey: str = ""
    # True when the object has been projected onto this floor because it forms
    # part of the building envelope and physically spans several storeys.
    # Normal internal walls/partitions/spaces remain visible only on their
    # assigned storey to avoid cluttering every floor view.
    projected_to_floor: bool = False
    envelope_wall: bool = False
    attenuation_by_band_db: Dict[float, float] = field(default_factory=lambda: {2400.0: 5.0, 5000.0: 7.0, 6000.0: 8.0})
    # RF-only metadata. IFC geometry and authoring data remain untouched; these
    # values are simulator overrides used for attenuation and saved RF plans.
    rf_type_override: str = ""
    rf_customised: bool = False
    is_user_created: bool = False
    user_wall_thickness_m: float = 0.15
    # RF-only geometry edits retained for compatibility with saved plans. The
    # original polygon remains available so a plan can restore imported IFC
    # geometry before applying any saved RF geometry override.
    rf_geometry_customised: bool = False
    rf_original_polygon: Optional[Polygon] = field(default=None, repr=False, compare=False)

    @property
    def label(self) -> str:
        key = self.rf_type_override or self.material or self.type_name or "Unknown"
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
class APRadio:
    """One RF radio fitted to an AP.

    A physical access point can expose several radios, for example 2.4 GHz,
    5 GHz and 6 GHz Wi-Fi, or non-Wi-Fi planning bands such as 433 MHz and
    868 MHz. Each radio has its own link budget and antenna pattern.
    """

    name: str = "Radio-1"
    frequency_mhz: float = 2400.0
    tx_power_dbm: float = 20.0
    antenna_pattern: str = "Omni ceiling AP"
    enabled: bool = True
    cutoff_radius_m: float = 0.0  # 0 means use settings/default; samples outside this radius are disconnected/skipped
    antenna_gain_dbi: float = 0.0  # Additional configured gain beyond the selected pattern data.
    channel: str = ""
    channel_width_mhz: float = 20.0
    spectrum_occupancy_percent: float = 0.0


@dataclass
class AccessPoint:
    name: str
    x: float
    y: float
    floor: str
    # Legacy/default radio fields are retained for backwards compatibility with
    # existing saved APs and older table-edit logic. New calculations use radios.
    tx_power_dbm: float = 20.0
    frequency_mhz: float = 2400.0
    reference_loss_db_at_1m: float = 40.0
    path_loss_exponent: float = 2.2
    antenna_pattern: str = "Omni ceiling AP"
    azimuth_deg: float = 0.0
    downtilt_deg: float = 0.0
    mount_height_m: float = 2.7
    rx_height_m: float = 1.2
    radios: List[APRadio] = field(default_factory=list)
    max_clients: int = 50
    planned: bool = False

    def active_radios(self) -> List[APRadio]:
        if not self.radios:
            return [APRadio(
                name="Radio-1",
                frequency_mhz=float(self.frequency_mhz),
                tx_power_dbm=float(self.tx_power_dbm),
                antenna_pattern=self.antenna_pattern,
                enabled=True,
                cutoff_radius_m=0.0,
                antenna_gain_dbi=0.0,
                channel="",
                channel_width_mhz=20.0,
                spectrum_occupancy_percent=0.0,
            )]
        return [r for r in self.radios if getattr(r, "enabled", True)]


@dataclass
class PlannerRadioRequirement:
    name: str = "5 GHz"
    enabled: bool = True
    frequency_mhz: float = 5000.0
    tx_power_dbm: float = 20.0
    antenna_pattern: str = "Omni ceiling AP"
    antenna_gain_dbi: float = 0.0
    channel_width_mhz: float = 20.0
    channels: List[str] = field(default_factory=lambda: ["36", "40", "44", "48"])
    spectrum_occupancy_percent: float = 20.0
    minimum_rssi_dbm: float = -67.0
    cutoff_radius_m: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "frequency_mhz": self.frequency_mhz,
            "tx_power_dbm": self.tx_power_dbm,
            "antenna_pattern": self.antenna_pattern,
            "antenna_gain_dbi": self.antenna_gain_dbi,
            "channel_width_mhz": self.channel_width_mhz,
            "channels": list(self.channels),
            "spectrum_occupancy_percent": self.spectrum_occupancy_percent,
            "minimum_rssi_dbm": self.minimum_rssi_dbm,
            "cutoff_radius_m": self.cutoff_radius_m,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PlannerRadioRequirement":
        channels = data.get("channels", []) if isinstance(data, dict) else []
        if isinstance(channels, str):
            channels = [v.strip() for v in channels.replace(";", ",").split(",") if v.strip()]
        return cls(
            name=str(data.get("name", "Radio")),
            enabled=bool(data.get("enabled", True)),
            frequency_mhz=float(data.get("frequency_mhz", 5000.0)),
            tx_power_dbm=float(data.get("tx_power_dbm", 20.0)),
            antenna_pattern=str(data.get("antenna_pattern", "Omni ceiling AP")),
            antenna_gain_dbi=float(data.get("antenna_gain_dbi", 0.0)),
            channel_width_mhz=float(data.get("channel_width_mhz", 20.0)),
            channels=[str(v) for v in channels] or ["1"],
            spectrum_occupancy_percent=max(0.0, min(100.0, float(data.get("spectrum_occupancy_percent", 20.0)))),
            minimum_rssi_dbm=float(data.get("minimum_rssi_dbm", -67.0)),
            cutoff_radius_m=max(0.0, float(data.get("cutoff_radius_m", 0.0))),
        )


@dataclass
class AutoPlannerSettings:
    target_coverage_percent: float = 95.0
    coverage_mode: str = "all"  # all selected frequencies, or any selected frequency
    sample_spacing_m: float = 3.0
    candidate_spacing_m: float = 6.0
    minimum_ap_spacing_m: float = 8.0
    maximum_aps: int = 64
    # ``auto`` uses IfcSpace footprints when present and infers a planning
    # footprint from walls when spaces have not been modelled. ``spaces``
    # requires IfcSpace geometry; ``walls`` always uses the inferred footprint.
    planning_area_mode: str = "auto"
    wall_footprint_margin_m: float = 0.0
    expected_clients: int = 250
    clients_per_ap: int = 50
    keep_existing_aps: bool = True
    remove_previous_planned_aps: bool = True
    radio_requirements: List[PlannerRadioRequirement] = field(default_factory=lambda: [
        PlannerRadioRequirement(
            name="2.4 GHz", frequency_mhz=2400.0, tx_power_dbm=18.0,
            channel_width_mhz=20.0, channels=["1", "6", "11"],
            spectrum_occupancy_percent=35.0, minimum_rssi_dbm=-67.0,
        ),
        PlannerRadioRequirement(
            name="5 GHz", frequency_mhz=5000.0, tx_power_dbm=20.0,
            channel_width_mhz=40.0, channels=["36", "44", "52", "60", "100", "108", "116", "124"],
            spectrum_occupancy_percent=20.0, minimum_rssi_dbm=-67.0,
        ),
        PlannerRadioRequirement(
            name="6 GHz", enabled=False, frequency_mhz=6000.0, tx_power_dbm=20.0,
            channel_width_mhz=80.0, channels=["5", "21", "37", "53", "69", "85"],
            spectrum_occupancy_percent=10.0, minimum_rssi_dbm=-67.0,
        ),
    ])

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_coverage_percent": self.target_coverage_percent,
            "coverage_mode": self.coverage_mode,
            "sample_spacing_m": self.sample_spacing_m,
            "candidate_spacing_m": self.candidate_spacing_m,
            "minimum_ap_spacing_m": self.minimum_ap_spacing_m,
            "maximum_aps": self.maximum_aps,
            "planning_area_mode": self.planning_area_mode,
            "wall_footprint_margin_m": self.wall_footprint_margin_m,
            "expected_clients": self.expected_clients,
            "clients_per_ap": self.clients_per_ap,
            "keep_existing_aps": self.keep_existing_aps,
            "remove_previous_planned_aps": self.remove_previous_planned_aps,
            "radio_requirements": [r.to_dict() for r in self.radio_requirements],
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "AutoPlannerSettings":
        base = cls()
        if not isinstance(data, dict):
            return base
        radios = data.get("radio_requirements", data.get("radios", []))
        if isinstance(radios, list) and radios:
            parsed = [PlannerRadioRequirement.from_dict(r) for r in radios if isinstance(r, dict)]
            if parsed:
                base.radio_requirements = parsed
        base.target_coverage_percent = max(1.0, min(100.0, float(data.get("target_coverage_percent", base.target_coverage_percent))))
        base.coverage_mode = "any" if str(data.get("coverage_mode", base.coverage_mode)).lower().startswith("any") else "all"
        base.sample_spacing_m = max(0.5, float(data.get("sample_spacing_m", base.sample_spacing_m)))
        base.candidate_spacing_m = max(1.0, float(data.get("candidate_spacing_m", base.candidate_spacing_m)))
        base.minimum_ap_spacing_m = max(0.0, float(data.get("minimum_ap_spacing_m", base.minimum_ap_spacing_m)))
        base.maximum_aps = max(1, min(10_000, int(data.get("maximum_aps", base.maximum_aps))))
        raw_area_mode = str(data.get("planning_area_mode", data.get("planner_area_mode", base.planning_area_mode))).strip().lower()
        if raw_area_mode.startswith("space"):
            base.planning_area_mode = "spaces"
        elif raw_area_mode.startswith("wall") or raw_area_mode.startswith("floor"):
            base.planning_area_mode = "walls"
        else:
            base.planning_area_mode = "auto"
        base.wall_footprint_margin_m = max(0.0, min(100.0, float(data.get("wall_footprint_margin_m", base.wall_footprint_margin_m))))
        base.expected_clients = max(0, int(data.get("expected_clients", base.expected_clients)))
        base.clients_per_ap = max(1, int(data.get("clients_per_ap", base.clients_per_ap)))
        base.keep_existing_aps = bool(data.get("keep_existing_aps", base.keep_existing_aps))
        base.remove_previous_planned_aps = bool(data.get("remove_previous_planned_aps", base.remove_previous_planned_aps))
        return base


@dataclass
class Space2D:
    guid: str
    name: str
    floor: str
    source_file: str
    polygon: Polygon
    z_min: float = 0.0
    z_max: float = 0.0
    source_storey: str = ""


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
    # Common RF planning frequencies. These values drive default attenuation
    # profiles, AP radio presets, and frequency selectors in the GUI.
    common_frequencies_mhz: List[float] = field(default_factory=lambda: [433.0, 868.0, 2400.0, 5000.0, 6000.0])
    # RF performance cut-off. When enabled, expensive wall/floor intersection
    # calculations are skipped for AP radios that cannot physically contribute
    # to a sample point. If a grid point is outside all active AP/radio cut-off
    # zones it remains at the disconnected RSSI value.
    enable_ap_cutoff_zones: bool = True
    disconnected_rssi_dbm: float = -120.0
    default_ap_cutoff_radius_m: float = 45.0
    ap_cutoff_radius_by_frequency_m: Dict[float, float] = field(default_factory=lambda: {
        433.0: 120.0,
        868.0: 90.0,
        2400.0: 45.0,
        5000.0: 35.0,
        6000.0: 30.0,
    })
    show_ap_cutoff_zones: bool = True
    ap_cutoff_zone_line_width: float = 0.12
    ap_cutoff_zone_alpha: int = 120
    default_floor_attenuation_by_frequency_db: Dict[float, float] = field(default_factory=lambda: {
        433.0: 8.0,
        868.0: 10.0,
        2400.0: 12.0,
        5000.0: 18.0,
        6000.0: 22.0,
    })
    default_wall_attenuation_by_material_db: Dict[str, Dict[float, float]] = field(default_factory=lambda: {
        "default": {433.0: 2.0, 868.0: 3.0, 2400.0: 5.0, 5000.0: 7.0, 6000.0: 8.0},
        "concrete": {433.0: 5.0, 868.0: 7.0, 2400.0: 12.0, 5000.0: 16.0, 6000.0: 20.0},
        "brick": {433.0: 4.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 11.0, 6000.0: 14.0},
        "masonry": {433.0: 4.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 11.0, 6000.0: 14.0},
        "glass": {433.0: 1.0, 868.0: 2.0, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0},
        "plasterboard": {433.0: 1.0, 868.0: 1.5, 2400.0: 3.0, 5000.0: 4.0, 6000.0: 5.0},
        "partition": {433.0: 1.0, 868.0: 1.5, 2400.0: 3.0, 5000.0: 4.0, 6000.0: 5.0},
        "metal": {433.0: 12.0, 868.0: 16.0, 2400.0: 20.0, 5000.0: 28.0, 6000.0: 35.0},
        "steel": {433.0: 12.0, 868.0: 16.0, 2400.0: 20.0, 5000.0: 28.0, 6000.0: 35.0}
    })
    default_ap_radios: List[Dict[str, object]] = field(default_factory=lambda: [
        {"name": "2.4 GHz", "frequency_mhz": 2400.0, "tx_power_dbm": 20.0, "antenna_pattern": "Omni ceiling AP", "enabled": True, "cutoff_radius_m": 45.0, "antenna_gain_dbi": 0.0, "channel": "1", "channel_width_mhz": 20.0, "spectrum_occupancy_percent": 35.0},
        {"name": "5 GHz", "frequency_mhz": 5000.0, "tx_power_dbm": 20.0, "antenna_pattern": "Omni ceiling AP", "enabled": True, "cutoff_radius_m": 35.0, "antenna_gain_dbi": 0.0, "channel": "36", "channel_width_mhz": 40.0, "spectrum_occupancy_percent": 20.0},
        {"name": "6 GHz", "frequency_mhz": 6000.0, "tx_power_dbm": 20.0, "antenna_pattern": "Omni ceiling AP", "enabled": False, "cutoff_radius_m": 30.0, "antenna_gain_dbi": 0.0, "channel": "5", "channel_width_mhz": 80.0, "spectrum_occupancy_percent": 10.0}
    ])
    auto_planner_settings: Dict[str, object] = field(default_factory=lambda: AutoPlannerSettings().to_dict())
    user_wall_default_type: str = "partition"
    user_wall_default_thickness_m: float = 0.15
    # Text uses model-scaled scene units so it zooms naturally with the IFC view.
    # The numeric font sizes below are small logical sizes which are multiplied
    # by text_model_scale before drawing. Increase text_model_scale if labels
    # are too small at your normal zoom level.
    contour_label_font_size: int = 6
    sample_label_font_size: int = 5
    space_label_font_size: int = 7
    ap_label_font_size: int = 7
    text_model_scale: float = 0.035
    sample_cross_size: float = 0.08
    contour_line_width: float = 1.25
    # Multiprocessing options. IFC loading now uses process workers only; Qt
    # thread-pool based IFC loading has been removed to avoid mixed concurrency
    # paths and last-file batch completion issues.
    enable_ifc_multiprocessing: bool = True
    max_ifc_loader_processes: int = 0
    # Safety watchdog for process-based IFC loading. Large files can legitimately
    # take time, so this is intentionally generous. Set to 0 to disable.
    ifc_load_timeout_seconds: int = 900
    # For very large single IFC files, do a lightweight index pass and split
    # geometry extraction into GlobalId chunks processed by multiple processes.
    enable_chunked_ifc_geometry_extraction: bool = True
    chunk_ifc_files_over_mb: float = 100.0
    ifc_geometry_chunk_size: int = 250
    # Large IFC safety guard. Chunking by GlobalId can improve smaller/medium
    # models, but for 500-1000 MB IFCs it is unsafe because every worker process
    # reopens the entire model. That multiplies RAM and can crash ifcopenshell/OCC.
    huge_ifc_single_process_threshold_mb: float = 512.0
    max_parallel_huge_ifc_processes: int = 1
    # Keep all IFC parsing out of the GUI process by default. If a worker dies,
    # the GUI reports the error instead of falling back to an in-process parse.
    allow_blocking_ifc_fallback: bool = False
    enable_rf_multiprocessing: bool = True
    max_rf_worker_processes: int = 0
    rf_multiprocessing_min_points: int = 5000
    rf_tile_rows: int = 16
    # Multi-storey model handling. Normal model elements are drawn only on their
    # assigned/nearest storey. External/envelope walls may be projected through
    # all intersected storeys so the RF model still sees a complete external
    # envelope on each floor slice.
    project_external_walls_across_floors: bool = True
    external_wall_keywords: List[str] = field(default_factory=lambda: [
        "external", "exterior", "outer", "facade", "façade", "curtain", "envelope", "perimeter"
    ])
    contour_line_cosmetic: bool = True
    contour_line_colour: str = "#111111"
    contour_line_colour_light: str = "#111111"
    contour_line_colour_dark: str = "#FFFFFF"
    contour_line_alpha: int = 255
    use_band_colour_for_contour_lines: bool = False
    sample_cross_line_width: float = 1.0
    wall_line_width: float = 0.18
    # DXF unit handling. By default the loader reads $INSUNITS from the DXF
    # and converts the overlay to metres to match IFC/project coordinates.
    # dxf_unit_scale is an additional manual multiplier for unusual exports.
    dxf_auto_unit_scale: bool = True
    dxf_unit_scale: float = 1.0
    dxf_unit_name: str = "auto"
    dxf_overlay_line_width: float = 0.08
    dxf_overlay_alpha: int = 190

    # All non-RSSI-band display colours are configurable in the settings JSON.
    # Both British (colour) and US (color) spellings are accepted when loading.
    colours_light: Dict[str, str] = field(default_factory=lambda: {
        "background": "#FAFAFA",
        "legend_background": "#F5F5F5",
        "legend_text": "#202020",
        "legend_border": "#BBBBBB",
        "space_pen": "#5F5F5F",
        "space_fill": "#E1E1E1",
        "space_text": "#282828",
        "wall_pen": "#191919",
        "ifc_wall_fill": "#D7D7D7",
        "ifc_linked_wall_fill": "#B9C3CD",
        "contour_text": "#141414",
        "sample_cross": "#0055FF",
        "sample_text": "#0055FF",
        "ap_same_floor": "#0050FF",
        "ap_other_floor": "#7800B4",
        "ap_outline": "#000050",
        "ap_cutoff_zone": "#0050FF",
        "dxf_overlay": "#0096FF"
    })
    colours_dark: Dict[str, str] = field(default_factory=lambda: {
        "background": "#2A2A2A",
        "legend_background": "#2F2F2F",
        "legend_text": "#EEEEEE",
        "legend_border": "#555555",
        "space_pen": "#969696",
        "space_fill": "#3A3A3A",
        "space_text": "#DCDCDC",
        "wall_pen": "#EBEBEB",
        "ifc_wall_fill": "#1E1E1E",
        "ifc_linked_wall_fill": "#414146",
        "contour_text": "#F0F0F0",
        "sample_cross": "#55A0FF",
        "sample_text": "#55A0FF",
        "ap_same_floor": "#4D8DFF",
        "ap_other_floor": "#C77DFF",
        "ap_outline": "#D8E4FF",
        "ap_cutoff_zone": "#4D8DFF",
        "dxf_overlay": "#62B7FF"
    })
    alpha_light: Dict[str, int] = field(default_factory=lambda: {
        "space_fill": 45,
        "ifc_wall_fill": 255,
        "ifc_linked_wall_fill": 255
    })
    alpha_dark: Dict[str, int] = field(default_factory=lambda: {
        "space_fill": 38,
        "ifc_wall_fill": 255,
        "ifc_linked_wall_fill": 255
    })

    contour_interpolation_factor: int = 4
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
        settings.common_frequencies_mhz = [float(v) for v in data.get("common_frequencies_mhz", data.get("common_frequencies", settings.common_frequencies_mhz))]
        settings.enable_ap_cutoff_zones = bool(data.get("enable_ap_cutoff_zones", data.get("enable_ap_cutoff_zone", True)))
        settings.disconnected_rssi_dbm = float(data.get("disconnected_rssi_dbm", -120.0))
        settings.default_ap_cutoff_radius_m = float(data.get("default_ap_cutoff_radius_m", 45.0))
        settings.show_ap_cutoff_zones = bool(data.get("show_ap_cutoff_zones", True))
        settings.ap_cutoff_zone_line_width = float(data.get("ap_cutoff_zone_line_width", 0.12))
        settings.ap_cutoff_zone_alpha = int(data.get("ap_cutoff_zone_alpha", 120))
        def _freq_dict(raw, fallback):
            if not isinstance(raw, dict):
                return fallback
            return {float(k): float(v) for k, v in raw.items()}
        settings.ap_cutoff_radius_by_frequency_m = _freq_dict(
            data.get("ap_cutoff_radius_by_frequency_m", data.get("ap_cutoff_radius_by_frequency", {})),
            settings.ap_cutoff_radius_by_frequency_m,
        )
        settings.default_floor_attenuation_by_frequency_db = _freq_dict(
            data.get("default_floor_attenuation_by_frequency_db", data.get("floor_attenuation_by_frequency_db", {})),
            settings.default_floor_attenuation_by_frequency_db,
        )
        raw_materials = data.get("default_wall_attenuation_by_material_db", data.get("wall_attenuation_by_material_db", {}))
        if isinstance(raw_materials, dict):
            parsed = {}
            for mat, profile in raw_materials.items():
                if isinstance(profile, dict):
                    parsed[str(mat).lower()] = {float(k): float(v) for k, v in profile.items()}
            if parsed:
                settings.default_wall_attenuation_by_material_db = parsed
        raw_radios = data.get("default_ap_radios", settings.default_ap_radios)
        if isinstance(raw_radios, list):
            settings.default_ap_radios = [dict(r) for r in raw_radios if isinstance(r, dict)]
        raw_planner = data.get("auto_planner_settings", data.get("predictive_ap_planner", {}))
        if isinstance(raw_planner, dict):
            settings.auto_planner_settings = AutoPlannerSettings.from_dict(raw_planner).to_dict()
        settings.user_wall_default_type = str(data.get("user_wall_default_type", "partition"))
        settings.user_wall_default_thickness_m = max(0.02, float(data.get("user_wall_default_thickness_m", 0.15)))
        # Font sizes are logical values converted to model-scaled scene text.
        # Text follows the view transform, so it enlarges when zooming in.
        settings.contour_label_font_size = int(data.get("contour_label_font_size", 6))
        settings.sample_label_font_size = int(data.get("sample_label_font_size", 5))
        settings.space_label_font_size = int(data.get("space_label_font_size", 7))
        settings.ap_label_font_size = int(data.get("ap_label_font_size", 7))
        settings.text_model_scale = float(data.get("text_model_scale", 0.035))
        settings.sample_cross_size = float(data.get("sample_cross_size", 0.08))
        settings.contour_line_width = float(data.get("contour_line_width", 1.25))
        settings.enable_ifc_multiprocessing = bool(data.get("enable_ifc_multiprocessing", data.get("ifc_loading_multiprocessing", True)))
        settings.max_ifc_loader_processes = max(0, int(data.get("max_ifc_loader_processes", data.get("ifc_loader_processes", 0))))
        settings.ifc_load_timeout_seconds = max(0, int(data.get("ifc_load_timeout_seconds", data.get("ifc_loader_timeout_seconds", 900))))
        settings.enable_chunked_ifc_geometry_extraction = bool(data.get("enable_chunked_ifc_geometry_extraction", data.get("chunked_ifc_geometry_extraction", True)))
        settings.chunk_ifc_files_over_mb = max(0.0, float(data.get("chunk_ifc_files_over_mb", data.get("ifc_chunk_files_over_mb", 100.0))))
        settings.ifc_geometry_chunk_size = max(25, int(data.get("ifc_geometry_chunk_size", data.get("ifc_chunk_size", 250))))
        settings.huge_ifc_single_process_threshold_mb = max(0.0, float(data.get("huge_ifc_single_process_threshold_mb", data.get("ifc_huge_file_single_process_threshold_mb", 512.0))))
        settings.max_parallel_huge_ifc_processes = max(1, int(data.get("max_parallel_huge_ifc_processes", data.get("ifc_max_parallel_huge_processes", 1))))
        settings.allow_blocking_ifc_fallback = bool(data.get("allow_blocking_ifc_fallback", False))
        settings.enable_rf_multiprocessing = bool(data.get("enable_rf_multiprocessing", data.get("rf_multiprocessing", True)))
        settings.max_rf_worker_processes = max(0, int(data.get("max_rf_worker_processes", data.get("rf_worker_processes", 0))))
        settings.rf_multiprocessing_min_points = max(1, int(data.get("rf_multiprocessing_min_points", 5000)))
        settings.rf_tile_rows = max(1, int(data.get("rf_tile_rows", 16)))
        settings.project_external_walls_across_floors = bool(data.get("project_external_walls_across_floors", True))
        raw_external_keywords = data.get("external_wall_keywords", data.get("envelope_wall_keywords", settings.external_wall_keywords))
        if isinstance(raw_external_keywords, list):
            settings.external_wall_keywords = [str(v).strip().lower() for v in raw_external_keywords if str(v).strip()]
        settings.contour_line_cosmetic = bool(data.get("contour_line_cosmetic", True))
        settings.contour_line_colour = str(data.get("contour_line_colour", data.get("contour_line_color", "#111111")))
        settings.contour_line_colour_light = str(data.get("contour_line_colour_light", data.get("contour_line_color_light", settings.contour_line_colour)))
        settings.contour_line_colour_dark = str(data.get("contour_line_colour_dark", data.get("contour_line_color_dark", "#FFFFFF")))
        settings.contour_line_alpha = int(data.get("contour_line_alpha", 255))
        settings.use_band_colour_for_contour_lines = bool(data.get("use_band_colour_for_contour_lines", False))
        settings.sample_cross_line_width = float(data.get("sample_cross_line_width", 1.0))
        settings.wall_line_width = float(data.get("wall_line_width", 0.18))
        settings.dxf_auto_unit_scale = bool(data.get("dxf_auto_unit_scale", data.get("dxf_auto_units", True)))
        settings.dxf_unit_scale = float(data.get("dxf_unit_scale", 1.0))
        settings.dxf_unit_name = str(data.get("dxf_unit_name", "auto"))
        settings.dxf_overlay_line_width = float(data.get("dxf_overlay_line_width", 0.08))
        settings.dxf_overlay_alpha = int(data.get("dxf_overlay_alpha", 190))

        # Display colours. Supports either:
        #   "colours": {"light": {...}, "dark": {...}, "alpha_light": {...}, "alpha_dark": {...}}
        # or US spelling "colors". Top-level legacy keys still work for contour lines.
        colour_block = data.get("colours", data.get("colors", {}))
        if isinstance(colour_block, dict):
            light = colour_block.get("light", colour_block.get("Light", {}))
            dark = colour_block.get("dark", colour_block.get("Dark", {}))
            alpha_light = colour_block.get("alpha_light", colour_block.get("light_alpha", {}))
            alpha_dark = colour_block.get("alpha_dark", colour_block.get("dark_alpha", {}))
            if isinstance(light, dict):
                settings.colours_light.update({str(k): str(v) for k, v in light.items()})
            if isinstance(dark, dict):
                settings.colours_dark.update({str(k): str(v) for k, v in dark.items()})
            if isinstance(alpha_light, dict):
                settings.alpha_light.update({str(k): int(v) for k, v in alpha_light.items()})
            if isinstance(alpha_dark, dict):
                settings.alpha_dark.update({str(k): int(v) for k, v in alpha_dark.items()})

        settings.contour_interpolation_factor = max(1, int(data.get("contour_interpolation_factor", 4)))
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

    def contour_line_qcolour(self, rssi: float, dark_theme: bool = False) -> QColor:
        """Return the independent contour boundary colour.

        By default this deliberately does not use the filled band colour, because
        matching the line to the fill can make isolines invisible. Set
        use_band_colour_for_contour_lines=true in the settings JSON if you want
        the old behaviour.
        """
        if self.use_band_colour_for_contour_lines:
            colour = self.colour_for_rssi(rssi)
            colour.setAlpha(max(0, min(255, int(self.contour_line_alpha))))
            return colour
        colour_name = self.contour_line_colour_dark if dark_theme else self.contour_line_colour_light
        if not colour_name:
            colour_name = self.contour_line_colour
        colour = QColor(colour_name)
        if not colour.isValid():
            colour = QColor("#111111")
        colour.setAlpha(max(0, min(255, int(self.contour_line_alpha))))
        return colour

    def display_qcolour(self, key: str, dark_theme: bool = False, fallback: str = "#000000") -> QColor:
        """Return a configured display colour by key for the current theme."""
        palette = self.colours_dark if dark_theme else self.colours_light
        alpha_map = self.alpha_dark if dark_theme else self.alpha_light
        colour = QColor(str(palette.get(key, fallback)))
        if not colour.isValid():
            colour = QColor(fallback)
        if key in alpha_map:
            colour.setAlpha(max(0, min(255, int(alpha_map[key]))))
        return colour


class IFCModelLoader:
    """Extracts storeys and wall plan polygons from an IFC file.

    Multiple IFCs can be loaded and merged by the MainWindow. Each file is
    treated as one discipline/model package in the same shared IFC coordinate
    system. If an architectural model, structural model, and fit-out model use
    the same project coordinates, their walls will line up automatically.
    """

    def __init__(self, path: Path, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 project_external_walls_across_floors: bool = True,
                 external_wall_keywords: Optional[List[str]] = None):
        if ifcopenshell is None:
            raise RuntimeError("ifcopenshell is not installed. Run: pip install ifcopenshell")
        self.path = path
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.project_external_walls_across_floors = bool(project_external_walls_across_floors)
        self.external_wall_keywords = [
            str(v).strip().lower() for v in (external_wall_keywords or [
                "external", "exterior", "outer", "facade", "façade", "curtain", "envelope", "perimeter"
            ]) if str(v).strip()
        ]
        self.ifc = ifcopenshell.open(str(path))
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def load(self) -> Dict[str, FloorModel]:
        return self.load_filtered()

    def load_filtered(
        self,
        wall_guids: Optional[Iterable[str]] = None,
        space_guids: Optional[Iterable[str]] = None,
        storeys_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, FloorModel]:
        """Load only selected wall/space GlobalIds.

        This is used by the large-file multiprocessing path. Each worker
        reopens the same IFC and extracts geometry only for its assigned
        GlobalIds, while the initial index/storey pass remains lightweight.
        """
        storeys = dict(storeys_override or self._storeys())
        floors = {name: FloorModel(name=name, elevation=elev) for name, elev in storeys.items()}
        if not floors:
            floors["Default"] = FloorModel(name="Default", elevation=0.0)

        wanted_walls = set(wall_guids or [])
        wanted_spaces = set(space_guids or [])
        filter_walls = bool(wall_guids is not None)
        filter_spaces = bool(space_guids is not None)

        seen_wall_guids = set()
        for wall in list(self.ifc.by_type("IfcWall")) + list(self.ifc.by_type("IfcWallStandardCase")):
            guid = getattr(wall, "GlobalId", "") or ""
            if guid in seen_wall_guids:
                continue
            seen_wall_guids.add(guid)
            if filter_walls and guid not in wanted_walls:
                continue
            source_floor = self._container_storey_name(wall)
            geom = self._plan_polygon_from_geometry(wall)
            if geom is None:
                continue
            poly, z_min, z_max = geom
            if poly is None or poly.area <= 0:
                continue
            mat = self._material_name(wall)
            type_name = self._type_name(wall)
            assigned_floor = source_floor or self._nearest_floor_name(floors, z_min)
            is_envelope = self._is_external_or_envelope_wall(wall, mat, type_name)
            if is_envelope and self.project_external_walls_across_floors:
                floor_names = self._floor_names_for_z_span(floors, z_min, z_max, assigned_floor)
            else:
                floor_names = [assigned_floor]
            for floor_name in floor_names:
                floors.setdefault(floor_name, FloorModel(name=floor_name, elevation=0.0)).walls.append(
                    Wall2D(
                        guid=guid,
                        name=getattr(wall, "Name", "") or "Wall",
                        floor=floor_name,
                        source_file=self.path.name,
                        type_name=type_name,
                        material=mat,
                        polygon=poly,
                        z_min=z_min,
                        z_max=z_max,
                        source_storey=source_floor or "",
                        projected_to_floor=(floor_name != assigned_floor),
                        envelope_wall=is_envelope,
                        attenuation_by_band_db=self._default_attenuation_profile(mat, type_name),
                    )
                )

        for space in self.ifc.by_type("IfcSpace"):
            guid = getattr(space, "GlobalId", "") or ""
            if filter_spaces and guid not in wanted_spaces:
                continue
            source_floor = self._container_storey_name(space)
            geom = self._plan_polygon_from_geometry(space)
            if geom is None:
                continue
            poly, z_min, z_max = geom
            if poly is None or poly.area <= 0:
                continue
            floor_names = [source_floor or self._nearest_floor_name(floors, z_min)]
            for floor_name in floor_names:
                floors.setdefault(floor_name, FloorModel(name=floor_name, elevation=0.0)).spaces.append(
                    Space2D(
                        guid=guid,
                        name=getattr(space, "LongName", None) or getattr(space, "Name", "") or "Space",
                        floor=floor_name,
                        source_file=self.path.name,
                        polygon=poly,
                        z_min=z_min,
                        z_max=z_max,
                        source_storey=source_floor or "",
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

    def _plan_polygon_from_geometry(self, product) -> Optional[Tuple[Polygon, float, float]]:
        """Project product mesh vertices onto XY and return footprint plus Z span."""
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
        z_min = float(np.min(verts[:, 2]))
        z_max = float(np.max(verts[:, 2]))
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
        return rect, z_min, z_max

    @staticmethod
    def _floor_names_for_z_span(
        floors: Dict[str, FloorModel],
        z_min: float,
        z_max: float,
        fallback_floor: Optional[str] = None,
        tolerance_m: float = 0.15,
    ) -> List[str]:
        """Return every storey slice intersected by an element's vertical extent.

        IFC containment can be misleading for multi-storey items. This method
        uses actual geometry Z extents and storey elevations so items such as
        lift shafts, risers, facade panels, curtain walls and stair-core walls
        are visible and attenuate RF on each floor they pass through.
        """
        if not floors:
            return [fallback_floor or "Default"]
        ordered = sorted(floors.values(), key=lambda f: (f.elevation, f.name))
        names: List[str] = []
        # Estimate a top slice height from neighbouring storeys.
        deltas = [ordered[i + 1].elevation - ordered[i].elevation for i in range(len(ordered) - 1) if ordered[i + 1].elevation > ordered[i].elevation]
        default_height = float(np.median(deltas)) if deltas else 3.5
        for i, floor in enumerate(ordered):
            lower = float(floor.elevation) - tolerance_m
            upper = (float(ordered[i + 1].elevation) if i + 1 < len(ordered) else float(floor.elevation) + default_height) + tolerance_m
            if z_max >= lower and z_min <= upper:
                names.append(floor.name)
        if not names and fallback_floor:
            names.append(fallback_floor)
        if not names:
            names.append(IFCModelLoader._nearest_floor_name(floors, z_min))
        return list(dict.fromkeys(names))

    def _ifc_is_external_property(self, product) -> Optional[bool]:
        """Return IFC IsExternal when present on the product/type property sets."""
        try:
            import ifcopenshell.util.element as element_util
            for obj in [product] + [getattr(rel, "RelatingType", None) for rel in (getattr(product, "IsTypedBy", []) or [])]:
                if obj is None:
                    continue
                psets = element_util.get_psets(obj) or {}
                for props in psets.values():
                    if isinstance(props, dict) and "IsExternal" in props:
                        return bool(props.get("IsExternal"))
        except Exception:
            pass
        return None

    def _is_external_or_envelope_wall(self, product, material: str, type_name: str) -> bool:
        """Classify walls that may be safely projected through intersected floors.

        Only external/envelope constructions are projected to complete the
        building RF envelope. Internal partitions and other tall objects remain
        on their assigned storey so floor plans are not overloaded.
        """
        ext = self._ifc_is_external_property(product)
        if ext is not None:
            return bool(ext)
        text = " ".join([
            str(getattr(product, "Name", "") or ""),
            str(getattr(product, "ObjectType", "") or ""),
            str(material or ""),
            str(type_name or ""),
        ]).lower()
        return any(keyword in text for keyword in self.external_wall_keywords)

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
            return {433.0: 5.0, 868.0: 7.0, 2400.0: 12.0, 5000.0: 16.0, 6000.0: 20.0}
        if "brick" in text or "masonry" in text:
            return {433.0: 4.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 11.0, 6000.0: 14.0}
        if "glass" in text:
            return {433.0: 1.0, 868.0: 2.0, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0}
        if "plaster" in text or "drywall" in text or "partition" in text:
            return {433.0: 1.0, 868.0: 1.5, 2400.0: 3.0, 5000.0: 4.0, 6000.0: 5.0}
        if "metal" in text or "steel" in text:
            return {433.0: 12.0, 868.0: 16.0, 2400.0: 20.0, 5000.0: 28.0, 6000.0: 35.0}
        return {433.0: 2.0, 868.0: 3.0, 2400.0: 5.0, 5000.0: 7.0, 6000.0: 8.0}


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
    def cutoff_radius_m_for_radio(radio: APRadio, settings: Optional[HeatmapSettings] = None) -> float:
        """Return the effective planning cut-off radius for one AP radio.

        A radio-specific value wins. If omitted/zero, the settings frequency
        table is used, falling back to default_ap_cutoff_radius_m. Values <= 0
        disable the cut-off for that radio.
        """
        explicit = float(getattr(radio, "cutoff_radius_m", 0.0) or 0.0)
        if explicit > 0.0:
            return explicit
        if settings is None:
            return 0.0
        table = getattr(settings, "ap_cutoff_radius_by_frequency_m", {}) or {}
        if table:
            bands = sorted(float(k) for k in table.keys())
            freq = float(radio.frequency_mhz)
            closest = min(bands, key=lambda b: abs(b - freq))
            return float(table.get(closest, settings.default_ap_cutoff_radius_m))
        return float(getattr(settings, "default_ap_cutoff_radius_m", 0.0) or 0.0)

    @staticmethod
    def point_is_inside_radio_cutoff(x: float, y: float, receiver_floor: FloorModel, ap: AccessPoint, floors: Dict[str, FloorModel], radio: APRadio, settings: Optional[HeatmapSettings]) -> bool:
        if not settings or not getattr(settings, "enable_ap_cutoff_zones", True):
            return True
        radius = RFEngine.cutoff_radius_m_for_radio(radio, settings)
        if radius <= 0.0:
            return True
        ap_floor = floors.get(ap.floor)
        if ap_floor is None:
            return False
        horizontal_d = math.hypot(x - ap.x, y - ap.y)
        ap_z = float(ap_floor.elevation) + float(ap.mount_height_m)
        rx_z = float(receiver_floor.elevation) + float(ap.rx_height_m)
        d_3d = math.hypot(horizontal_d, ap_z - rx_z)
        return d_3d <= radius

    @staticmethod
    def rssi_at(
        x: float,
        y: float,
        receiver_floor: FloorModel,
        ap: AccessPoint,
        floors: Dict[str, FloorModel],
        patterns: Optional[Dict[str, AntennaPattern]] = None,
        radio: Optional[APRadio] = None,
        include_inter_floor: bool = True,
        heatmap_settings: Optional[HeatmapSettings] = None,
    ) -> float:
        """Calculate RSSI at a receiver point on receiver_floor.

        This is a 3D link-budget approximation. For same-floor links it behaves
        like the original model. For inter-floor links it adds:
        - vertical separation between AP mounting height and receiver height;
        - floor/slab penetration loss for every storey boundary crossed;
        - wall intersections on both the AP floor and receiver floor.
        """
        radio = radio or ap.active_radios()[0]
        ap_floor = floors.get(ap.floor)
        if ap_floor is None:
            return -120.0
        disconnected = float(getattr(heatmap_settings, "disconnected_rssi_dbm", -120.0) if heatmap_settings else -120.0)
        if ap.floor != receiver_floor.name and not include_inter_floor:
            return disconnected
        if not RFEngine.point_is_inside_radio_cutoff(x, y, receiver_floor, ap, floors, radio, heatmap_settings):
            return disconnected

        horizontal_d = max(math.hypot(x - ap.x, y - ap.y), 0.1)
        ap_z = float(ap_floor.elevation) + float(ap.mount_height_m)
        rx_z = float(receiver_floor.elevation) + float(ap.rx_height_m)
        dz = ap_z - rx_z
        d_3d = max(math.hypot(horizontal_d, dz), 1.0)
        reference_loss = RFEngine.free_space_loss_db_at_1m(radio.frequency_mhz)
        path_loss = reference_loss + 10.0 * ap.path_loss_exponent * math.log10(d_3d)

        bearing = math.degrees(math.atan2(y - ap.y, x - ap.x))
        az_rel = AntennaPattern._wrap_deg(bearing - ap.azimuth_deg)
        elev_angle = math.degrees(math.atan2((rx_z - ap_z), horizontal_d))
        elev_rel = elev_angle + ap.downtilt_deg
        pattern_gain = 0.0
        if patterns:
            pattern = patterns.get(radio.antenna_pattern)
            if pattern:
                pattern_gain = pattern.gain_dbi(az_rel, elev_rel)

        line = LineString([(ap.x, ap.y), (x, y)])
        wall_loss = 0.0
        checked_wall_guids = set()
        # Include walls on every floor crossed by the 3D path. This matters for
        # multi-storey elements which have been copied into each intersected
        # storey slice during IFC loading, such as lift shafts, riser walls,
        # atrium glazing and external facade elements. A GUID is counted only
        # once so the same spanning object does not multiply its attenuation.
        for path_floor in RFEngine.floors_between_inclusive(receiver_floor, ap_floor, floors):
            for wall in path_floor.walls:
                wall_key = wall.guid or f"{wall.source_file}:{wall.name}:{wall.z_min:.3f}:{wall.z_max:.3f}"
                if wall_key not in checked_wall_guids and wall.polygon.intersects(line):
                    wall_loss += wall.attenuation_db_for_frequency(radio.frequency_mhz)
                    checked_wall_guids.add(wall_key)

        floor_loss = RFEngine.floor_penetration_loss_db(receiver_floor, ap_floor, floors, radio.frequency_mhz)
        return radio.tx_power_dbm + pattern_gain + float(getattr(radio, "antenna_gain_dbi", 0.0) or 0.0) - path_loss - wall_loss - floor_loss

    @staticmethod
    def floors_between_inclusive(receiver_floor: FloorModel, ap_floor: FloorModel, floors: Dict[str, FloorModel]) -> List[FloorModel]:
        if receiver_floor.name == ap_floor.name:
            return [receiver_floor]
        ordered = sorted(floors.values(), key=lambda f: (f.elevation, f.name))
        try:
            rx_i = next(i for i, f in enumerate(ordered) if f.name == receiver_floor.name)
            ap_i = next(i for i, f in enumerate(ordered) if f.name == ap_floor.name)
        except StopIteration:
            return [receiver_floor, ap_floor]
        lo, hi = sorted((rx_i, ap_i))
        return ordered[lo:hi + 1]

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
        heatmap_settings: Optional[HeatmapSettings] = None,
        progress_callback=None,
    ) -> Optional[SimulationResult]:
        if not floor.walls and not floor.spaces:
            return None
        if not aps:
            return None
        bounds = RFEngine._floor_bounds(floor, aps)
        minx, miny, maxx, maxy = bounds
        xs = np.arange(minx, maxx + resolution_m, resolution_m)
        ys = np.arange(miny, maxy + resolution_m, resolution_m)
        disconnected = float(getattr(heatmap_settings, "disconnected_rssi_dbm", -120.0) if heatmap_settings else -120.0)
        grid = np.full((len(ys), len(xs)), disconnected)
        candidate_aps = [a for a in aps if include_inter_floor or a.floor == floor.name]
        if not candidate_aps:
            return None
        total_points = int(len(xs) * len(ys))
        use_mp = bool(getattr(heatmap_settings, "enable_rf_multiprocessing", False)) if heatmap_settings else False
        min_points = int(getattr(heatmap_settings, "rf_multiprocessing_min_points", 5000) if heatmap_settings else 5000)
        if use_mp and total_points >= min_points and len(ys) > 1:
            requested = int(getattr(heatmap_settings, "max_rf_worker_processes", 0) or 0)
            process_count = min(_logical_process_count(requested), max(1, len(ys)))
            tile_rows = max(1, int(getattr(heatmap_settings, "rf_tile_rows", 16) or 16))
            jobs = []
            for start in range(0, len(ys), tile_rows):
                jobs.append((
                    floor, floors, aps, patterns, include_inter_floor, heatmap_settings,
                    xs, ys[start:start + tile_rows], start, disconnected,
                ))
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as executor:
                    completed_rows = 0
                    for start, tile in executor.map(_rf_grid_tile_worker, jobs):
                        grid[start:start + tile.shape[0], :] = tile
                        completed_rows += tile.shape[0]
                        if progress_callback:
                            progress_callback(min(completed_rows, len(ys)), len(ys))
                return SimulationResult(xs=xs, ys=ys, rssi=grid)
            except Exception:
                # Fall back to single-process calculation if a model object is not
                # pickleable on a particular platform/build. The UI still gets a
                # result rather than failing the survey.
                pass

        if progress_callback:
            progress_callback(0, len(ys))

        for iy, yy in enumerate(ys):
            for ix, xx in enumerate(xs):
                values = []
                for ap in candidate_aps:
                    for radio in ap.active_radios():
                        # Fast cut-off test avoids expensive wall/floor intersection
                        # calculations when this radio cannot contribute to the
                        # sample point. This is the main large-IFC speed-up.
                        if RFEngine.point_is_inside_radio_cutoff(xx, yy, floor, ap, floors, radio, heatmap_settings):
                            values.append(RFEngine.rssi_at(xx, yy, floor, ap, floors, patterns, radio, include_inter_floor, heatmap_settings))
                grid[iy, ix] = max(values) if values else disconnected
                if progress_callback:
                    progress_callback(iy + 1, len(ys))
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


# ----------------------------- DXF overlay and IFC alignment -----------------------------

@dataclass
class DxfPrimitive:
    kind: str
    layer: str
    points: List[Tuple[float, float]]


@dataclass
class DxfOverlay:
    path: str
    primitives: List[DxfPrimitive] = field(default_factory=list)
    visible: bool = True
    source_units_code: int = 0
    source_units_name: str = "unitless"
    metres_per_dxf_unit: float = 1.0
    manual_scale: float = 1.0
    effective_scale_to_metres: float = 1.0


DXF_INSUNITS_TO_METRES: Dict[int, Tuple[str, float]] = {
    0: ("unitless", 1.0),
    1: ("inches", 0.0254),
    2: ("feet", 0.3048),
    3: ("miles", 1609.344),
    4: ("millimetres", 0.001),
    5: ("centimetres", 0.01),
    6: ("metres", 1.0),
    7: ("kilometres", 1000.0),
    8: ("microinches", 0.0000000254),
    9: ("mils", 0.0000254),
    10: ("yards", 0.9144),
    11: ("angstroms", 1e-10),
    12: ("nanometres", 1e-9),
    13: ("microns", 1e-6),
    14: ("decimetres", 0.1),
    15: ("decametres", 10.0),
    16: ("hectometres", 100.0),
    17: ("gigametres", 1e9),
    18: ("astronomical units", 149597870700.0),
    19: ("light years", 9.4607304725808e15),
    20: ("parsecs", 3.085677581491367e16),
}


def dxf_units_to_metres(doc, auto_units: bool = True, manual_scale: float = 1.0) -> Tuple[int, str, float, float]:
    """Return DXF unit metadata and the final coordinate scale to metres.

    IFC geometry in this simulator is kept in metres. DXF files are often
    exported in millimetres. When auto_units is enabled, $INSUNITS is used as
    the base conversion and manual_scale remains available as an extra override.
    """
    try:
        code = int(doc.header.get("$INSUNITS", 0) or 0)
    except Exception:
        code = 0
    name, metres = DXF_INSUNITS_TO_METRES.get(code, (f"INSUNITS {code}", 1.0))
    if not auto_units:
        metres = 1.0
        name = "manual/unitless"
    manual = float(manual_scale or 1.0)
    return code, name, metres, metres * manual


@dataclass
class AlignmentTransform:
    """2D transform used to align IFC/model coordinates to the DXF reference.

    Order: scale about origin -> rotate about origin -> translate.
    """
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0

    def matrix(self) -> Tuple[float, float, float, float, float, float]:
        angle = math.radians(float(self.rotation_deg))
        scale = float(self.scale)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return (
            scale * cos_a,
            -scale * sin_a,
            scale * sin_a,
            scale * cos_a,
            float(self.dx),
            float(self.dy),
        )

    def map_xy(self, x: float, y: float) -> Tuple[float, float]:
        a, b, d, e, xoff, yoff = self.matrix()
        return a * x + b * y + xoff, d * x + e * y + yoff

    @staticmethod
    def _invert_matrix(m: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float, float, float, float]:
        a, b, d, e, xoff, yoff = m
        det = a * e - b * d
        if abs(det) < 1e-12:
            raise ValueError("Alignment scale/rotation produces a non-invertible transform.")
        ia = e / det
        ib = -b / det
        id_ = -d / det
        ie = a / det
        ix = -(ia * xoff + ib * yoff)
        iy = -(id_ * xoff + ie * yoff)
        return ia, ib, id_, ie, ix, iy

    @staticmethod
    def _compose(m2: Tuple[float, float, float, float, float, float],
                 m1: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float, float, float, float]:
        """Return matrix that applies m1 then m2."""
        a1, b1, d1, e1, x1, y1 = m1
        a2, b2, d2, e2, x2, y2 = m2
        return (
            a2 * a1 + b2 * d1,
            a2 * b1 + b2 * e1,
            d2 * a1 + e2 * d1,
            d2 * b1 + e2 * e1,
            a2 * x1 + b2 * y1 + x2,
            d2 * x1 + e2 * y1 + y2,
        )

    @classmethod
    def delta_matrix(cls, old: "AlignmentTransform", new: "AlignmentTransform") -> Tuple[float, float, float, float, float, float]:
        return cls._compose(new.matrix(), cls._invert_matrix(old.matrix()))

    @classmethod
    def from_matrix(cls, m: Tuple[float, float, float, float, float, float]) -> "AlignmentTransform":
        a, b, d, e, xoff, yoff = m
        scale_x = math.hypot(a, d)
        scale_y = math.hypot(b, e)
        scale = (scale_x + scale_y) / 2.0 if scale_x > 0 or scale_y > 0 else 1.0
        rotation = math.degrees(math.atan2(d, a)) if scale > 1e-12 else 0.0
        return cls(dx=float(xoff), dy=float(yoff), rotation_deg=float(rotation), scale=float(scale))

    @staticmethod
    def two_point_matrix(ifc_a: Tuple[float, float], ifc_b: Tuple[float, float],
                         dxf_a: Tuple[float, float], dxf_b: Tuple[float, float]) -> Tuple[float, float, float, float, float, float]:
        ix1, iy1 = ifc_a; ix2, iy2 = ifc_b
        dx1, dy1 = dxf_a; dx2, dy2 = dxf_b
        ivx, ivy = ix2 - ix1, iy2 - iy1
        dvx, dvy = dx2 - dx1, dy2 - dy1
        ilen = math.hypot(ivx, ivy)
        dlen = math.hypot(dvx, dvy)
        if ilen <= 1e-9 or dlen <= 1e-9:
            raise ValueError("Two-point alignment needs two distinct IFC points and two distinct DXF points.")
        scale = dlen / ilen
        angle = math.atan2(dvy, dvx) - math.atan2(ivy, ivx)
        cos_a = math.cos(angle); sin_a = math.sin(angle)
        a = scale * cos_a
        b = -scale * sin_a
        d = scale * sin_a
        e = scale * cos_a
        xoff = dx1 - (a * ix1 + b * iy1)
        yoff = dy1 - (d * ix1 + e * iy1)
        return a, b, d, e, xoff, yoff


def load_dxf_overlay(path: Path, unit_scale: float = 1.0, auto_units: bool = True) -> DxfOverlay:
    """Load basic 2D DXF entities as a reference overlay.

    Supported: LINE, LWPOLYLINE, POLYLINE, CIRCLE and ARC. Coordinates are
    converted to metres using the DXF $INSUNITS header when available, with
    ``unit_scale`` applied as an additional manual multiplier.
    """
    if ezdxf is None:
        raise RuntimeError("ezdxf is required. Install with: pip install ezdxf")
    doc = ezdxf.readfile(str(path))
    unit_code, unit_name, metres_per_unit, effective_scale = dxf_units_to_metres(
        doc, auto_units=auto_units, manual_scale=unit_scale
    )
    msp = doc.modelspace()
    prims: List[DxfPrimitive] = []

    def sc(pt) -> Tuple[float, float]:
        return float(pt[0]) * effective_scale, float(pt[1]) * effective_scale

    def iter_entities(layout):
        for e in layout:
            if e.dxftype() == "INSERT":
                try:
                    yield from e.virtual_entities()
                except Exception:
                    continue
            else:
                yield e

    for e in iter_entities(msp):
        et = e.dxftype()
        layer = str(getattr(e.dxf, "layer", "0") or "0")
        try:
            if et == "LINE":
                prims.append(DxfPrimitive("polyline", layer, [sc(e.dxf.start), sc(e.dxf.end)]))
            elif et == "LWPOLYLINE":
                pts = [(float(x) * effective_scale, float(y) * effective_scale) for x, y, *_ in e.get_points()]
                if len(pts) >= 2:
                    if e.closed:
                        pts.append(pts[0])
                    prims.append(DxfPrimitive("polyline", layer, pts))
            elif et == "POLYLINE":
                pts = [sc(v.dxf.location) for v in e.vertices]
                if len(pts) >= 2:
                    if e.is_closed:
                        pts.append(pts[0])
                    prims.append(DxfPrimitive("polyline", layer, pts))
            elif et == "CIRCLE":
                cx, cy = sc(e.dxf.center)
                r = float(e.dxf.radius) * effective_scale
                pts = [(cx + math.cos(a) * r, cy + math.sin(a) * r) for a in [2 * math.pi * i / 96 for i in range(97)]]
                prims.append(DxfPrimitive("polyline", layer, pts))
            elif et == "ARC":
                cx, cy = sc(e.dxf.center)
                r = float(e.dxf.radius) * effective_scale
                start = math.radians(float(e.dxf.start_angle))
                end = math.radians(float(e.dxf.end_angle))
                if end < start:
                    end += 2 * math.pi
                steps = max(8, int(abs(end - start) / (2 * math.pi) * 96))
                pts = [(cx + math.cos(start + (end - start) * i / steps) * r,
                        cy + math.sin(start + (end - start) * i / steps) * r) for i in range(steps + 1)]
                prims.append(DxfPrimitive("polyline", layer, pts))
            elif et in ("SPLINE", "ELLIPSE"):
                pts = [sc(p) for p in e.flattening(0.05)]
                if len(pts) >= 2:
                    prims.append(DxfPrimitive("polyline", layer, pts))
        except Exception:
            continue
    return DxfOverlay(
        path=str(path),
        primitives=prims,
        source_units_code=unit_code,
        source_units_name=unit_name,
        metres_per_dxf_unit=metres_per_unit,
        manual_scale=float(unit_scale or 1.0),
        effective_scale_to_metres=effective_scale,
    )


def load_dxf_overlay_with_similarity_transform(path: Path, transform: SimilarityTransform2D) -> DxfOverlay:
    """Load a DXF using raw DXF coordinates, then map it into IFC coordinates.

    The DxfPreAlignDialog calculates a transform from raw DXF coordinates to
    IFC/model metres. That transform already includes the DXF $INSUNITS unit
    conversion, so this function deliberately reads the DXF with auto unit
    conversion disabled and then applies the two-point transform to every point.
    """
    overlay = load_dxf_overlay(path, unit_scale=1.0, auto_units=False)
    transformed: List[DxfPrimitive] = []
    affine = getattr(transform, "affine_matrix", None)
    for prim in overlay.primitives:
        if affine:
            a, b, d, e, tx, ty = affine
            pts = [(a * float(x) + b * float(y) + tx, d * float(x) + e * float(y) + ty) for x, y in prim.points]
        else:
            pts = [transform.map_point(float(x), float(y)) for x, y in prim.points]
        transformed.append(DxfPrimitive(prim.kind, prim.layer, pts))
    overlay.primitives = transformed
    overlay.source_units_name = "pre-aligned DXF"
    overlay.effective_scale_to_metres = float(transform.scale)
    return overlay


class DxfAlignmentDialog(QDialog):
    def __init__(self, parent=None, transform: Optional[AlignmentTransform] = None, apply_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Align IFC to DXF")
        self.transform = transform or AlignmentTransform()
        self.apply_callback = apply_callback
        layout = QVBoxLayout(self)
        info = QLabel("Adjust the IFC model against the DXF overlay. Values are model metres/degrees. Click Apply to update geometry used by RF simulation.")
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QFormLayout()
        self.dx = QDoubleSpinBox(); self.dx.setRange(-1_000_000, 1_000_000); self.dx.setDecimals(4); self.dx.setSingleStep(0.1); self.dx.setValue(self.transform.dx)
        self.dy = QDoubleSpinBox(); self.dy.setRange(-1_000_000, 1_000_000); self.dy.setDecimals(4); self.dy.setSingleStep(0.1); self.dy.setValue(self.transform.dy)
        self.rot = QDoubleSpinBox(); self.rot.setRange(-360, 360); self.rot.setDecimals(4); self.rot.setSingleStep(0.25); self.rot.setValue(self.transform.rotation_deg); self.rot.setSuffix("°")
        self.scale_spin = QDoubleSpinBox(); self.scale_spin.setRange(0.000001, 1000000); self.scale_spin.setDecimals(6); self.scale_spin.setSingleStep(0.01); self.scale_spin.setValue(self.transform.scale)
        form.addRow("IFC offset X", self.dx)
        form.addRow("IFC offset Y", self.dy)
        form.addRow("IFC rotation", self.rot)
        form.addRow("IFC scale", self.scale_spin)
        layout.addLayout(form)
        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        reset_btn = QPushButton("Reset")
        apply_btn.clicked.connect(self.apply)
        reset_btn.clicked.connect(self.reset)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)

    def current_transform(self) -> AlignmentTransform:
        return AlignmentTransform(
            dx=float(self.dx.value()),
            dy=float(self.dy.value()),
            rotation_deg=float(self.rot.value()),
            scale=float(self.scale_spin.value()),
        )

    def apply(self):
        self.transform = self.current_transform()
        if self.apply_callback:
            self.apply_callback(self.transform)

    def reset(self):
        self.dx.setValue(0.0)
        self.dy.setValue(0.0)
        self.rot.setValue(0.0)
        self.scale_spin.setValue(1.0)
        self.apply()


# ----------------------------- IFC origin/site information -----------------------------

def _ifc_decimal_degrees(value) -> Optional[float]:
    """Convert IFC compound plane-angle values to decimal degrees."""
    if value is None:
        return None
    try:
        parts = list(value)
        if not parts:
            return None
        sign = -1.0 if float(parts[0]) < 0 else 1.0
        deg = abs(float(parts[0]))
        minute = abs(float(parts[1])) if len(parts) > 1 else 0.0
        second = abs(float(parts[2])) if len(parts) > 2 else 0.0
        millionth = abs(float(parts[3])) if len(parts) > 3 else 0.0
        return sign * (deg + minute / 60.0 + (second + millionth / 1_000_000.0) / 3600.0)
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def _ifc_local_placement_summary(product) -> Dict[str, float]:
    """Return absolute placement origin and plan rotation for an IFC product."""
    out = {"x": 0.0, "y": 0.0, "z": 0.0, "rotation_from_x_deg": 0.0}
    placement = getattr(product, "ObjectPlacement", None)
    if placement is None:
        return out
    try:
        import ifcopenshell.util.placement as placement_util
        matrix = np.asarray(placement_util.get_local_placement(placement), dtype=float)
        out["x"] = float(matrix[0, 3])
        out["y"] = float(matrix[1, 3])
        out["z"] = float(matrix[2, 3])
        out["rotation_from_x_deg"] = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
        return out
    except Exception:
        pass

    # Conservative fallback for simple IfcLocalPlacement chains.
    try:
        x = y = z = angle = 0.0
        chain = []
        current = placement
        while current is not None:
            chain.append(current)
            current = getattr(current, "PlacementRelTo", None)
        for local in reversed(chain):
            rel = getattr(local, "RelativePlacement", None)
            loc = getattr(getattr(rel, "Location", None), "Coordinates", None)
            if loc:
                lx = float(loc[0]) if len(loc) > 0 else 0.0
                ly = float(loc[1]) if len(loc) > 1 else 0.0
                lz = float(loc[2]) if len(loc) > 2 else 0.0
                ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                x += lx * ca - ly * sa
                y += lx * sa + ly * ca
                z += lz
            direction = getattr(getattr(rel, "RefDirection", None), "DirectionRatios", None)
            if direction and len(direction) >= 2:
                angle += math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        out.update(x=x, y=y, z=z, rotation_from_x_deg=angle)
    except Exception:
        pass
    return out


def _extract_ifc_origin_information(model, path: Path) -> Dict[str, object]:
    """Extract import origin, site orientation, true north and map CRS metadata."""
    try:
        import ifcopenshell.util.unit as ifc_unit_util
        length_unit_scale_to_m = float(ifc_unit_util.calculate_unit_scale(model) or 1.0)
    except Exception:
        length_unit_scale_to_m = 1.0

    info: Dict[str, object] = {
        "file": path.name,
        "path": str(path),
        "schema": str(getattr(model, "schema", "")),
        "length_unit_scale_to_m": length_unit_scale_to_m,
        "project": {},
        "sites": [],
        "buildings": [],
        "contexts": [],
        "map_conversions": [],
    }
    try:
        projects = model.by_type("IfcProject")
        if projects:
            project = projects[0]
            info["project"] = {
                "name": str(getattr(project, "Name", "") or ""),
                "global_id": str(getattr(project, "GlobalId", "") or ""),
                "long_name": str(getattr(project, "LongName", "") or ""),
            }
    except Exception:
        pass

    for type_name, key in (("IfcSite", "sites"), ("IfcBuilding", "buildings")):
        try:
            for entity in model.by_type(type_name):
                row: Dict[str, object] = {
                    "name": str(getattr(entity, "Name", "") or ""),
                    "global_id": str(getattr(entity, "GlobalId", "") or ""),
                    "placement": _ifc_local_placement_summary(entity),
                }
                if type_name == "IfcSite":
                    row["latitude_deg"] = _ifc_decimal_degrees(getattr(entity, "RefLatitude", None))
                    row["longitude_deg"] = _ifc_decimal_degrees(getattr(entity, "RefLongitude", None))
                    elevation = getattr(entity, "RefElevation", None)
                    row["reference_elevation"] = float(elevation) if elevation is not None else None
                    row["land_title_number"] = str(getattr(entity, "LandTitleNumber", "") or "")
                info[key].append(row)
        except Exception:
            pass

    try:
        contexts = list(model.by_type("IfcGeometricRepresentationContext"))
    except Exception:
        contexts = []
    for context in contexts:
        row: Dict[str, object] = {
            "context_type": str(getattr(context, "ContextType", "") or ""),
            "coordinate_space_dimension": getattr(context, "CoordinateSpaceDimension", None),
            "precision": getattr(context, "Precision", None),
        }
        world = getattr(context, "WorldCoordinateSystem", None)
        location = getattr(getattr(world, "Location", None), "Coordinates", None)
        if location:
            row["world_origin"] = [float(v) for v in location]
        ref = getattr(getattr(world, "RefDirection", None), "DirectionRatios", None)
        if ref and len(ref) >= 2:
            row["world_axis_rotation_from_x_deg"] = math.degrees(math.atan2(float(ref[1]), float(ref[0])))
        true_north = getattr(getattr(context, "TrueNorth", None), "DirectionRatios", None)
        if true_north and len(true_north) >= 2:
            dx, dy = float(true_north[0]), float(true_north[1])
            row["true_north_direction"] = [dx, dy]
            # IFC TrueNorth is expressed relative to the model +Y axis.
            row["true_north_angle_from_model_y_deg"] = math.degrees(math.atan2(dx, dy))
        info["contexts"].append(row)

    try:
        conversions = list(model.by_type("IfcMapConversion"))
    except Exception:
        conversions = []
    for conv in conversions:
        row: Dict[str, object] = {}
        for attr in ("Eastings", "Northings", "OrthogonalHeight", "XAxisAbscissa", "XAxisOrdinate", "Scale"):
            value = getattr(conv, attr, None)
            row[attr] = float(value) if value is not None else None
        target = getattr(conv, "TargetCRS", None)
        if target is not None:
            row["target_crs"] = {
                "name": str(getattr(target, "Name", "") or ""),
                "description": str(getattr(target, "Description", "") or ""),
                "geodetic_datum": str(getattr(target, "GeodeticDatum", "") or ""),
                "vertical_datum": str(getattr(target, "VerticalDatum", "") or ""),
                "map_projection": str(getattr(target, "MapProjection", "") or ""),
                "map_zone": str(getattr(target, "MapZone", "") or ""),
                "map_unit": str(getattr(getattr(target, "MapUnit", None), "Name", "") or ""),
            }
        if row.get("XAxisAbscissa") is not None and row.get("XAxisOrdinate") is not None:
            row["map_x_axis_rotation_from_east_deg"] = math.degrees(
                math.atan2(float(row["XAxisOrdinate"]), float(row["XAxisAbscissa"]))
            )
        info["map_conversions"].append(row)
    return info


# ----------------------------- Multiprocessing helpers -----------------------------

def _logical_process_count(requested_count: int = 0) -> int:
    """Resolve a safe process count for CPU-bound IFC/RF work."""
    cpu_count = max(1, os.cpu_count() or 1)
    if requested_count and int(requested_count) > 0:
        return max(1, min(int(requested_count), cpu_count))
    return cpu_count


def _load_ifc_file_in_process(args):
    """Top-level IFC worker so it is pickleable on Windows spawn."""
    path_str, dx, dy, dz, project_external_walls, external_keywords = args
    path = Path(path_str)
    loader = IFCModelLoader(
        path,
        float(dx),
        float(dy),
        float(dz),
        project_external_walls_across_floors=bool(project_external_walls),
        external_wall_keywords=list(external_keywords or []),
    )
    floors = loader.load()
    origin_info = _extract_ifc_origin_information(loader.ifc, path)
    return path_str, floors, path.name, origin_info


def _index_ifc_file_for_chunking(args):
    """Lightweight IFC index pass for large-file chunked geometry extraction."""
    path_str = args[0]
    if ifcopenshell is None:
        raise RuntimeError("ifcopenshell is not installed. Run: pip install ifcopenshell")
    path = Path(path_str)
    model = ifcopenshell.open(str(path))
    storeys: Dict[str, float] = {}
    for st in model.by_type("IfcBuildingStorey"):
        name = getattr(st, "Name", None) or getattr(st, "GlobalId", None) or "Storey"
        storeys[str(name)] = float(getattr(st, "Elevation", 0.0) or 0.0)
    seen = set()
    wall_guids: List[str] = []
    for wall in list(model.by_type("IfcWall")) + list(model.by_type("IfcWallStandardCase")):
        guid = getattr(wall, "GlobalId", "") or ""
        if guid and guid not in seen:
            seen.add(guid)
            wall_guids.append(guid)
    space_guids = [getattr(sp, "GlobalId", "") or "" for sp in model.by_type("IfcSpace")]
    space_guids = [g for g in space_guids if g]
    origin_info = _extract_ifc_origin_information(model, path)
    return path_str, storeys, wall_guids, space_guids, path.name, origin_info


def _load_ifc_geometry_chunk_in_process(args):
    """Extract geometry for one GlobalId chunk from a large IFC file."""
    (
        path_str, dx, dy, dz, project_external_walls, external_keywords,
        storeys, wall_guids, space_guids, chunk_index, chunk_count,
    ) = args
    path = Path(path_str)
    floors = IFCModelLoader(
        path,
        float(dx),
        float(dy),
        float(dz),
        project_external_walls_across_floors=bool(project_external_walls),
        external_wall_keywords=list(external_keywords or []),
    ).load_filtered(
        wall_guids=list(wall_guids or []),
        space_guids=list(space_guids or []),
        storeys_override={str(k): float(v) for k, v in dict(storeys or {}).items()},
    )
    return path_str, floors, path.name, int(chunk_index), int(chunk_count)


def _rf_grid_tile_worker(args):
    """Calculate one horizontal strip of the RF grid in a separate process."""
    (
        floor,
        floors,
        aps,
        patterns,
        include_inter_floor,
        heatmap_settings,
        xs,
        ys_slice,
        start_index,
        disconnected,
    ) = args
    xs = np.asarray(xs, dtype=float)
    ys_slice = np.asarray(ys_slice, dtype=float)
    tile = np.full((len(ys_slice), len(xs)), float(disconnected), dtype=float)
    candidate_aps = [a for a in aps if include_inter_floor or a.floor == floor.name]
    for iy, yy in enumerate(ys_slice):
        for ix, xx in enumerate(xs):
            values = []
            for ap in candidate_aps:
                for radio in ap.active_radios():
                    if RFEngine.point_is_inside_radio_cutoff(float(xx), float(yy), floor, ap, floors, radio, heatmap_settings):
                        values.append(RFEngine.rssi_at(float(xx), float(yy), floor, ap, floors, patterns, radio, include_inter_floor, heatmap_settings))
            tile[iy, ix] = max(values) if values else float(disconnected)
    return int(start_index), tile

# ----------------------------- Process-based IFC loading -----------------------------
# IFC loading intentionally uses multiprocessing only. A QTimer polls the futures
# from the GUI thread so no Qt worker thread is required. This keeps all GUI
# updates on the main thread while IFC parsing happens in child processes.

# ----------------------------- Drawing layers -----------------------------
# Higher z-values are drawn above lower z-values. Keep heatmap colour below IFC
# geometry, but keep contour boundaries, sample markers and text readable above it.
Z_HEATMAP_FILL = -30
Z_IFC_SPACE_FILL = -20
Z_IFC_SPACE_OUTLINE = -10
Z_IFC_WALL = 0
Z_DXF_OVERLAY = 10
Z_CONTOUR_LINE = 20
Z_SAMPLE_MARK = 30
Z_TEXT = 40
Z_AP = 50
Z_AP_LABEL = 55

class WallAttenuationDialog(QDialog):
    """Edit an IFC or user-created wall's RF type and attenuation profile."""

    def __init__(self, parent, wall: Wall2D, bands: List[float], profiles: Dict[str, Dict[float, float]]):
        super().__init__(parent)
        self.wall = wall
        self.bands = list(bands)
        self.profiles = profiles
        self.setWindowTitle("Wall RF type and attenuation")
        self.resize(620, 420)

        layout = QVBoxLayout(self)
        details = QLabel(
            f"<b>{wall.name or 'Wall'}</b><br>Source: {wall.source_file or 'User RF wall'}<br>"
            f"IFC type: {wall.type_name or '—'} &nbsp;&nbsp; Material: {wall.material or '—'}<br>"
            f"GUID: {wall.guid}"
        )
        details.setWordWrap(True)
        layout.addWidget(details)

        form = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.setEditable(True)
        self.type_combo.addItems(sorted({str(k) for k in profiles.keys()} | {"default", "partition", "glass", "brick", "concrete", "metal"}))
        self.type_combo.setCurrentText(wall.rf_type_override or wall.material or wall.type_name or "default")
        form.addRow("RF wall type", self.type_combo)
        layout.addLayout(form)

        self.table = QTableWidget(len(self.bands), 2)
        self.table.setHorizontalHeaderLabels(["Frequency", "Attenuation (dB)"])
        for row, band in enumerate(self.bands):
            label = f"{band / 1000:g} GHz" if band >= 1000 else f"{band:g} MHz"
            freq_item = QTableWidgetItem(label)
            freq_item.setFlags(freq_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, freq_item)
            self.table.setItem(row, 1, QTableWidgetItem(f"{wall.attenuation_db_for_frequency(band):.3f}"))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        preset = QPushButton("Apply attenuation preset for selected wall type")
        preset.clicked.connect(self._apply_preset)
        layout.addWidget(preset)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_preset(self):
        key = self.type_combo.currentText().strip().lower()
        profile = self.profiles.get(key)
        if profile is None:
            profile = next((candidate for name, candidate in self.profiles.items() if name != "default" and name.lower() in key), None)
        profile = profile or self.profiles.get("default", {})
        for row, band in enumerate(self.bands):
            value = profile.get(float(band))
            if value is None and profile:
                keys = sorted(float(v) for v in profile)
                nearest = min(keys, key=lambda v: abs(v - float(band)))
                value = profile[nearest]
            if value is not None:
                self.table.item(row, 1).setText(f"{float(value):.3f}")

    def values(self) -> Tuple[str, Dict[float, float]]:
        attenuation: Dict[float, float] = {}
        for row, band in enumerate(self.bands):
            try:
                attenuation[float(band)] = float(self.table.item(row, 1).text())
            except Exception:
                attenuation[float(band)] = self.wall.attenuation_db_for_frequency(float(band))
        return self.type_combo.currentText().strip(), attenuation


class AutoPlannerSettingsDialog(QDialog):
    """Configure predictive AP coverage, capacity, channel and radio requirements."""

    HEADERS = ["Enabled", "Name", "Frequency MHz", "Pattern", "TX dBm", "Gain dBi", "Width MHz", "Channels", "Occupancy %", "Min RSSI dBm", "Cut-off m"]

    def __init__(self, parent, settings: AutoPlannerSettings, pattern_names: List[str]):
        super().__init__(parent)
        self.pattern_names = list(pattern_names)
        self.setWindowTitle("Predictive AP planner settings")
        self.resize(1120, 650)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.target = QDoubleSpinBox(); self.target.setRange(1.0, 100.0); self.target.setSuffix(" %"); self.target.setValue(settings.target_coverage_percent)
        self.coverage_mode = QComboBox(); self.coverage_mode.addItem("Each selected frequency", "all"); self.coverage_mode.addItem("Any selected frequency", "any"); self.coverage_mode.setCurrentIndex(1 if settings.coverage_mode == "any" else 0)
        self.sample_spacing = QDoubleSpinBox(); self.sample_spacing.setRange(0.5, 25.0); self.sample_spacing.setSuffix(" m"); self.sample_spacing.setValue(settings.sample_spacing_m)
        self.candidate_spacing = QDoubleSpinBox(); self.candidate_spacing.setRange(1.0, 50.0); self.candidate_spacing.setSuffix(" m"); self.candidate_spacing.setValue(settings.candidate_spacing_m)
        self.minimum_spacing = QDoubleSpinBox(); self.minimum_spacing.setRange(0.0, 100.0); self.minimum_spacing.setSuffix(" m"); self.minimum_spacing.setValue(settings.minimum_ap_spacing_m)
        self.maximum_aps = QSpinBox(); self.maximum_aps.setRange(1, 10_000); self.maximum_aps.setValue(settings.maximum_aps)
        self.area_mode = QComboBox()
        self.area_mode.addItem("Automatic — use IFC spaces, otherwise infer from walls", "auto")
        self.area_mode.addItem("IFC spaces only", "spaces")
        self.area_mode.addItem("Infer floor footprint from IFC walls", "walls")
        area_index = self.area_mode.findData(settings.planning_area_mode)
        self.area_mode.setCurrentIndex(max(0, area_index))
        self.wall_margin = QDoubleSpinBox(); self.wall_margin.setRange(0.0, 100.0); self.wall_margin.setDecimals(2); self.wall_margin.setSuffix(" m"); self.wall_margin.setValue(settings.wall_footprint_margin_m)
        self.expected_clients = QSpinBox(); self.expected_clients.setRange(0, 1000000); self.expected_clients.setValue(settings.expected_clients)
        self.clients_per_ap = QSpinBox(); self.clients_per_ap.setRange(1, 10000); self.clients_per_ap.setValue(settings.clients_per_ap)
        self.keep_existing = QCheckBox("Count and retain manually placed APs on this floor"); self.keep_existing.setChecked(settings.keep_existing_aps)
        self.remove_planned = QCheckBox("Replace APs created by the previous planner run"); self.remove_planned.setChecked(settings.remove_previous_planned_aps)
        form.addRow("Target floor coverage", self.target)
        form.addRow("Frequency coverage rule", self.coverage_mode)
        form.addRow("Coverage sample spacing", self.sample_spacing)
        form.addRow("Candidate AP spacing", self.candidate_spacing)
        form.addRow("Minimum AP separation", self.minimum_spacing)
        form.addRow("Maximum planned APs", self.maximum_aps)
        form.addRow("Planning area source", self.area_mode)
        form.addRow("Inferred wall-footprint margin", self.wall_margin)
        form.addRow("Expected connected clients", self.expected_clients)
        form.addRow("Maximum clients per AP", self.clients_per_ap)
        form.addRow(self.keep_existing)
        form.addRow(self.remove_planned)
        layout.addLayout(form)

        note = QLabel("Spectrum occupancy reduces effective client capacity. Channels are allocated to minimise nearby co-channel reuse. Additional antenna gain is added to the selected pattern data. When an IFC contains no IfcSpace objects, Automatic mode derives a floor footprint from imported wall geometry so APs can still be positioned.")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        layout.addWidget(self.table, 1)
        row_buttons = QHBoxLayout()
        add_btn = QPushButton("Add frequency")
        remove_btn = QPushButton("Remove selected frequency")
        add_btn.clicked.connect(lambda: self._add_radio(PlannerRadioRequirement(name="New radio", frequency_mhz=5000.0)))
        remove_btn.clicked.connect(self._remove_selected)
        row_buttons.addWidget(add_btn); row_buttons.addWidget(remove_btn); row_buttons.addStretch(1)
        layout.addLayout(row_buttons)

        for radio in settings.radio_requirements:
            self._add_radio(radio)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.resizeColumnsToContents()

    def _add_radio(self, radio: PlannerRadioRequirement):
        row = self.table.rowCount(); self.table.insertRow(row)
        values = [
            "Yes" if radio.enabled else "No", radio.name, f"{radio.frequency_mhz:g}", radio.antenna_pattern,
            f"{radio.tx_power_dbm:g}", f"{radio.antenna_gain_dbi:g}", f"{radio.channel_width_mhz:g}",
            ", ".join(radio.channels), f"{radio.spectrum_occupancy_percent:g}", f"{radio.minimum_rssi_dbm:g}", f"{radio.cutoff_radius_m:g}",
        ]
        for col, value in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(str(value)))
        # Do not retain a QTableWidgetItem underneath the live combo box. Some
        # Qt styles paint both objects, which makes the pattern text appear
        # duplicated/overlaid across the control.
        self.table.takeItem(row, 3)
        pattern = QComboBox()
        pattern.addItems(self.pattern_names)
        pattern.setEditable(True)
        pattern.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        pattern.setMinimumContentsLength(18)
        pattern.setCurrentText(radio.antenna_pattern)
        self.table.setCellWidget(row, 3, pattern)
        self.table.setRowHeight(row, max(self.table.rowHeight(row), pattern.sizeHint().height() + 4))

    def _remove_selected(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def settings(self) -> AutoPlannerSettings:
        radios: List[PlannerRadioRequirement] = []
        for row in range(self.table.rowCount()):
            def value(col: int, default: str = "") -> str:
                item = self.table.item(row, col)
                return item.text().strip() if item is not None else default
            pattern_widget = self.table.cellWidget(row, 3)
            pattern = pattern_widget.currentText().strip() if isinstance(pattern_widget, QComboBox) else value(3, "Omni ceiling AP")
            channels = [v.strip() for v in value(7, "1").replace(";", ",").split(",") if v.strip()]
            try:
                radios.append(PlannerRadioRequirement(
                    enabled=value(0, "Yes").lower() in {"yes", "y", "true", "1", "on", "enabled"},
                    name=value(1, f"Radio-{row + 1}"), frequency_mhz=float(value(2, "5000")), antenna_pattern=pattern,
                    tx_power_dbm=float(value(4, "20")), antenna_gain_dbi=float(value(5, "0")), channel_width_mhz=float(value(6, "20")),
                    channels=channels or ["1"], spectrum_occupancy_percent=max(0.0, min(100.0, float(value(8, "0")))),
                    minimum_rssi_dbm=float(value(9, "-67")), cutoff_radius_m=max(0.0, float(value(10, "0"))),
                ))
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in radio row {row + 1}: {exc}")
        if not radios:
            raise ValueError("At least one radio requirement is required.")
        return AutoPlannerSettings(
            target_coverage_percent=float(self.target.value()), coverage_mode=str(self.coverage_mode.currentData()),
            sample_spacing_m=float(self.sample_spacing.value()), candidate_spacing_m=float(self.candidate_spacing.value()),
            minimum_ap_spacing_m=float(self.minimum_spacing.value()), maximum_aps=int(self.maximum_aps.value()),
            planning_area_mode=str(self.area_mode.currentData() or "auto"),
            wall_footprint_margin_m=float(self.wall_margin.value()),
            expected_clients=int(self.expected_clients.value()), clients_per_ap=int(self.clients_per_ap.value()),
            keep_existing_aps=self.keep_existing.isChecked(), remove_previous_planned_aps=self.remove_planned.isChecked(),
            radio_requirements=radios,
        )

    def accept(self):
        try:
            self._validated_settings = self.settings()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid planner settings", str(exc)); return
        super().accept()


class IFCOriginDialog(QDialog):
    def __init__(self, parent, origin_info: Dict[str, Dict[str, object]], alignment: AlignmentTransform):
        super().__init__(parent)
        self.main = parent
        self.setWindowTitle("Imported IFC origin and site orientation")
        self.resize(820, 650)
        layout = QVBoxLayout(self)
        explanation = QLabel("Coordinates shown below come from the imported IFC metadata. The simulator view rotation is display-only and does not change IFC coordinates or RF calculations.")
        explanation.setWordWrap(True); layout.addWidget(explanation)
        self.text = QTextEdit(); self.text.setReadOnly(True); self.text.setPlainText(self._format_info(origin_info, alignment)); layout.addWidget(self.text, 1)
        row = QHBoxLayout()
        rotate = QPushButton("Rotate view to first available True North")
        rotate.clicked.connect(self._rotate_true_north)
        reset = QPushButton("Reset view rotation")
        reset.clicked.connect(parent.reset_view_rotation)
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        row.addWidget(rotate); row.addWidget(reset); row.addStretch(1); row.addWidget(close)
        layout.addLayout(row)

    @staticmethod
    def _format_info(origin_info: Dict[str, Dict[str, object]], alignment: AlignmentTransform) -> str:
        lines = [
            "Current simulator IFC alignment:",
            f"  Offset X/Y: {alignment.dx:.6f}, {alignment.dy:.6f} m",
            f"  Rotation: {alignment.rotation_deg:.6f}°",
            f"  Scale: {alignment.scale:.9g}", "",
        ]
        if not origin_info:
            lines.append("No IFC origin metadata has been captured yet.")
            return "\n".join(lines)
        for key, info in origin_info.items():
            lines.append(f"FILE: {info.get('file', key)}")
            lines.append(f"  Path: {info.get('path', '')}")
            lines.append(f"  Schema: {info.get('schema', '')}")
            lines.append(f"  IFC length-unit scale: {float(info.get('length_unit_scale_to_m', 1.0) or 1.0):.12g} m/unit")
            project = info.get("project", {}) or {}
            lines.append(f"  Project: {project.get('name', '')}  [{project.get('global_id', '')}]")
            for label, collection in (("Site", info.get("sites", [])), ("Building", info.get("buildings", []))):
                for idx, entity in enumerate(collection or [], start=1):
                    place = entity.get("placement", {}) or {}
                    lines.append(f"  {label} {idx}: {entity.get('name', '')}  [{entity.get('global_id', '')}]")
                    lines.append(f"    Placement origin: X={place.get('x', 0):.6f}, Y={place.get('y', 0):.6f}, Z={place.get('z', 0):.6f} m")
                    lines.append(f"    Placement rotation from +X: {place.get('rotation_from_x_deg', 0):.6f}°")
                    if label == "Site":
                        lines.append(f"    Reference latitude/longitude: {entity.get('latitude_deg', None)}, {entity.get('longitude_deg', None)}")
                        lines.append(f"    Reference elevation: {entity.get('reference_elevation', None)}")
            for idx, context in enumerate(info.get("contexts", []) or [], start=1):
                lines.append(f"  Geometric context {idx}: {context.get('context_type', '')}")
                if "world_origin" in context: lines.append(f"    World coordinate system origin: {context.get('world_origin')}")
                if "world_axis_rotation_from_x_deg" in context: lines.append(f"    World axis rotation from +X: {context.get('world_axis_rotation_from_x_deg'):.6f}°")
                if "true_north_direction" in context:
                    lines.append(f"    True North direction: {context.get('true_north_direction')}")
                    lines.append(f"    True North angle from model +Y: {context.get('true_north_angle_from_model_y_deg'):.6f}°")
            for idx, conv in enumerate(info.get("map_conversions", []) or [], start=1):
                lines.append(f"  Map conversion {idx}:")
                lines.append(f"    Eastings/Northings/Height: {conv.get('Eastings')}, {conv.get('Northings')}, {conv.get('OrthogonalHeight')}")
                lines.append(f"    X axis abscissa/ordinate: {conv.get('XAxisAbscissa')}, {conv.get('XAxisOrdinate')}; scale={conv.get('Scale')}")
                if "map_x_axis_rotation_from_east_deg" in conv: lines.append(f"    Map X-axis rotation from East: {conv.get('map_x_axis_rotation_from_east_deg'):.6f}°")
                crs = conv.get("target_crs", {}) or {}
                if crs: lines.append(f"    Target CRS: {crs.get('name', '')}; projection={crs.get('map_projection', '')}; zone={crs.get('map_zone', '')}; datum={crs.get('geodetic_datum', '')}")
            lines.append("")
        return "\n".join(lines)

    def _rotate_true_north(self):
        if not self.main.rotate_view_to_true_north():
            QMessageBox.information(self, "True North unavailable", "The imported IFC files do not contain a usable TrueNorth direction.")


# ----------------------------- ACCESS POINT GRAPHIC ITEM -----------------------------

class WallGraphicsItem(QGraphicsPolygonItem):
    def __init__(self, main, wall: Wall2D, polygon: QPolygonF, pen: QPen, brush: QBrush):
        super().__init__(polygon)
        self.main = main
        self.wall = wall
        self.setPen(pen); self.setBrush(brush); self.setZValue(Z_IFC_WALL)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setToolTip(f"{wall.label}\nRight-click to inspect or edit RF attenuation")

    def contextMenuEvent(self, event):
        menu = QMenu()
        self.setSelected(True)
        edit_action = menu.addAction("Edit wall type and attenuation…")
        rotate_ifc_action = None
        if not self.wall.is_user_created:
            rotate_ifc_action = menu.addAction(
                "Rotate IFC about insertion point so this wall is 0° to X-axis"
            )
        reset_action = menu.addAction("Reset RF attenuation from IFC type/material")
        delete_action = None
        if self.wall.is_user_created:
            menu.addSeparator()
            delete_action = menu.addAction("Delete user-created RF wall")
        chosen = menu.exec(event.screenPos())
        if chosen == edit_action:
            self.main.edit_wall_rf_properties(self.wall)
        elif rotate_ifc_action is not None and chosen == rotate_ifc_action:
            self.main.rotate_ifc_to_align_wall_with_x_axis(self.wall)
        elif chosen == reset_action:
            self.main.reset_wall_rf_properties(self.wall)
        elif delete_action is not None and chosen == delete_action:
            self.main.delete_user_wall(self.wall)
        event.accept()


class AccessPointGraphicsItem(QGraphicsEllipseItem):
    def __init__(self, main, ap: AccessPoint, radius: float, colour: QColor):
        super().__init__(ap.x - radius, ap.y - radius, radius * 2.0, radius * 2.0)
        self.main = main
        self.ap = ap
        self.radius = radius

        self.setBrush(QBrush(colour))
        self.setPen(QPen(main._theme_colours()["ap_outline"], 0.2))
        self.setZValue(Z_AP)
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.OpenHandCursor)
        radio_summary = ", ".join(
            f"{r.frequency_mhz:g} MHz ch {r.channel or 'auto'} / {r.channel_width_mhz:g} MHz"
            for r in ap.active_radios()
        )
        self.setToolTip(f"{ap.name}{' (predicted)' if ap.planned else ''}\n{radio_summary}\nClients/AP: {ap.max_clients}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        scene_pos = self.sceneBoundingRect().center()
        self.ap.x = float(scene_pos.x())
        self.ap.y = float(scene_pos.y())
        self.main.last_result = None
        self.main.populate_ap_table()
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        self.main.aps = [a for a in self.main.aps if a is not self.ap]
        self.main.last_result = None
        self.main.draw_floor()
        self.main.populate_ap_table()
        event.accept()

# ----------------------------- GUI -----------------------------

class PlanView(QGraphicsView):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Keep cosmetic pens and device-independent text crisp on high-DPI screens.
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, False)
        self.setMouseTracking(True)
        self.scale(1, -1)  # IFC Y-up style plan view
        self._middle_panning = False
        self._last_pan_pos = None

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        if getattr(self.main, "wall_draw_mode", False) and getattr(self.main, "_wall_draw_start", None) is not None:
            pos = self.mapToScene(event.position().toPoint())
            snap, _ = self.main.nearest_ifc_connection_point(pos)
            self.main.show_user_wall_preview(snap)

        if self._middle_panning and self._last_pan_pos is not None:
            delta = event.position().toPoint() - self._last_pan_pos
            self._last_pan_pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if getattr(self.main, "alignment_pick_mode", None) in {"ifc_1", "ifc_2"}:
            pos = self.mapToScene(event.position().toPoint())
            snap = self.main.nearest_ifc_snap_point(pos)
            self.main.show_ifc_snap_marker(snap)

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if getattr(self.main, "wall_draw_mode", False):
            if event.button() == Qt.RightButton:
                self.main.cancel_user_wall_drawing()
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                self.main.capture_user_wall_point(pos)
                event.accept()
                return

        if event.button() == Qt.MiddleButton:
            self._middle_panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if getattr(self.main, "alignment_pick_mode", None) in {"ifc_1", "ifc_2"}:
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                snap = self.main.nearest_ifc_snap_point(pos)
                self.main.capture_alignment_point(snap.x(), snap.y())
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._middle_panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.main.floor is None:
            return
        if getattr(self.main, "alignment_pick_mode", None) or getattr(self.main, "wall_draw_mode", False):
            return
        pos = self.mapToScene(event.position().toPoint())
        self.main.add_ap(pos.x(), pos.y())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # IFC multiprocessing/chunk-loader state
        # Initialised defensively so process loading cannot fall back because
        # chunk tracking attributes do not exist.
        self._ifc_chunk_remaining = {}
        self._ifc_chunk_total = {}
        self._ifc_chunk_results = []
        self._ifc_chunk_errors = []
        self._ifc_chunk_futures = []
        self._ifc_active_batch_id = 0
        self._ifc_process_executor = None
        self.setWindowTitle("RF Attenuation Simulator")
        self.resize(1400, 900)
        self.floors: Dict[str, FloorModel] = {}
        self.loaded_ifc_paths: List[Path] = []
        self.floor: Optional[FloorModel] = None
        self.aps: List[AccessPoint] = []
        self.antenna_patterns: Dict[str, AntennaPattern] = built_in_antenna_patterns()
        self.last_result: Optional[SimulationResult] = None
        self._ifc_process_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._ifc_process_futures: Dict[concurrent.futures.Future, Path] = {}
        self._ifc_process_poll_timer = QTimer(self)
        self._ifc_process_poll_timer.setInterval(100)
        self._ifc_process_poll_timer.timeout.connect(self._poll_ifc_process_futures)
        self._load_pending = 0
        self._load_errors: List[str] = []
        self._load_total = 0
        self._load_completed = 0
        self._load_progress_dialog: Optional[QProgressDialog] = None
        self._load_started_keys = set()
        self._load_finished_keys = set()
        self._load_batch_id = 0
        self._loading_replace = False
        self._loading_active = False
        self.heatmap_settings = HeatmapSettings.default()
        self.auto_planner_settings = AutoPlannerSettings.from_dict(self.heatmap_settings.auto_planner_settings)
        self.heatmap_settings_path: Optional[Path] = None
        self.ifc_origin_info: Dict[str, Dict[str, object]] = {}
        self.view_rotation_deg: float = 0.0
        self.wall_draw_mode: bool = False
        self._wall_draw_start: Optional[QPointF] = None
        self._wall_preview_items: List[QGraphicsItem] = []
        self._pending_plan_data: Optional[Dict[str, object]] = None
        self.dxf_overlay: Optional[DxfOverlay] = None
        self.ifc_alignment = AlignmentTransform()
        self.alignment_pick_mode: Optional[str] = None
        self.alignment_pick_points: Dict[str, Tuple[float, float]] = {}
        self._alignment_pick_sequence: List[str] = []
        self._pending_dxf_alignment_path: Optional[Path] = None
        self._ifc_snap_marker_items: List[QGraphicsItem] = []
        self._ifc_pick_marker_items: List[QGraphicsItem] = []
        self.dark_theme = self._detect_dark_theme()

        self._view_has_been_fitted = False
        self._preserve_view_on_redraw = False

        self.view = PlanView(self)
        self.rssi_legend = QLabel()
        self.rssi_legend.setWordWrap(True)
        self.rssi_legend.setMinimumHeight(72)
        self.rssi_legend.setTextFormat(Qt.RichText)
        self._apply_theme_styles()
        self.floor_combo = QComboBox()
        self.wall_table = QTableWidget(0, 0)
        self._configure_wall_table_headers()
        self.wall_table.itemChanged.connect(self._wall_table_changed)

        self.ap_table = QTableWidget(0, 16)
        self.ap_table.setHorizontalHeaderLabels([
            "AP", "Radio", "Enabled", "Floor", "X", "Y", "Pattern", "Azimuth", "Downtilt",
            "TX dBm", "Gain dBi", "Freq MHz", "Channel", "Width MHz", "Occupancy %", "Clients/AP"
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
        self.freq.setRange(1.0, 100000.0)
        self.freq.setValue(2400.0)
        self.freq.setSingleStep(100.0)
        self.freq.setSuffix(" MHz")

        self.rssi_view_frequency = QComboBox()
        self.rssi_view_frequency.currentTextChanged.connect(self._rssi_view_frequency_changed)
        self._refresh_rssi_frequency_dropdown()

        self.rssi_results_by_frequency: Dict[float, SimulationResult] = {}

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
        form.addRow("RSSI view frequency", self.rssi_view_frequency)
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
        self.open_dxf_action = QAction("Open DXF overlay", self)
        self.open_dxf_action.triggered.connect(self.open_dxf_overlay)
        self.align_ifc_action = QAction("Align IFC to DXF", self)
        self.align_ifc_action.triggered.connect(self.show_dxf_alignment_dialog)
        self.two_point_align_action = QAction("2-point align IFC/DXF", self)
        self.two_point_align_action.triggered.connect(self.start_two_point_alignment)
        self.clear_dxf_action = QAction("Clear DXF", self)
        self.clear_dxf_action.triggered.connect(self.clear_dxf_overlay)
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
        self.planner_settings_action = QAction("Planner settings", self)
        self.planner_settings_action.triggered.connect(self.show_auto_planner_settings)
        self.predict_aps_action = QAction("Predict AP locations", self)
        self.predict_aps_action.triggered.connect(self.run_auto_planner)
        self.draw_wall_action = QAction("Draw RF wall", self)
        self.draw_wall_action.setCheckable(True)
        self.draw_wall_action.toggled.connect(self.toggle_wall_draw_mode)
        self.ifc_origin_action = QAction("IFC origin / orientation", self)
        self.ifc_origin_action.triggered.connect(self.show_ifc_origin_dialog)
        self.rotate_left_action = QAction("Rotate view left", self)
        self.rotate_left_action.triggered.connect(lambda: self.rotate_view(-15.0))
        self.rotate_right_action = QAction("Rotate view right", self)
        self.rotate_right_action.triggered.connect(lambda: self.rotate_view(15.0))
        self.reset_rotation_action = QAction("Reset view rotation", self)
        self.reset_rotation_action.triggered.connect(self.reset_view_rotation)
        self.save_plan_action = QAction("Save RF plan", self)
        self.save_plan_action.triggered.connect(self.save_rf_plan)
        self.load_plan_action = QAction("Load RF plan", self)
        self.load_plan_action.triggered.connect(self.load_rf_plan)
        tb.addActions([
            self.open_action, self.add_action, self.open_dxf_action, self.align_ifc_action, self.two_point_align_action,
            self.clear_dxf_action, self.ifc_origin_action, self.rotate_left_action, self.rotate_right_action,
            self.reset_rotation_action, self.draw_wall_action, self.planner_settings_action, self.predict_aps_action,
            self.sim_action, self.export_action, self.clear_ap_action, self.load_pattern_action,
            self.load_heatmap_settings_action, self.save_plan_action, self.load_plan_action,
        ])
        self.floor_combo.currentTextChanged.connect(self.select_floor)
        self.include_inter_floor.stateChanged.connect(lambda *_: self.draw_floor())

    # ----------------------------- View/origin tools -----------------------------

    def rotate_view(self, delta_degrees: float):
        delta = float(delta_degrees)
        if abs(delta) < 1e-9:
            return
        self.view.rotate(delta)
        self.view_rotation_deg = ((self.view_rotation_deg + delta + 180.0) % 360.0) - 180.0
        self._preserve_view_on_redraw = True
        self.statusBar().showMessage(f"View rotation: {self.view_rotation_deg:.1f}° (display only)")

    def reset_view_rotation(self):
        if abs(self.view_rotation_deg) > 1e-9:
            self.view.rotate(-self.view_rotation_deg)
        self.view_rotation_deg = 0.0
        self._preserve_view_on_redraw = True
        self.statusBar().showMessage("View rotation reset to model orientation")

    def rotate_view_to_true_north(self) -> bool:
        for info in self.ifc_origin_info.values():
            for context in info.get("contexts", []) or []:
                angle = context.get("true_north_angle_from_model_y_deg")
                if angle is None:
                    continue
                target = -float(angle)
                self.rotate_view(target - self.view_rotation_deg)
                self.statusBar().showMessage(
                    f"View rotated to IFC True North ({float(angle):.3f}° from model +Y)"
                )
                return True
        return False

    def show_ifc_origin_dialog(self):
        dlg = IFCOriginDialog(self, self.ifc_origin_info, self.ifc_alignment)
        dlg.exec()

    # ----------------------------- RF wall creation/editing -----------------------------

    def _clear_wall_preview(self):
        scene = self.view.scene()
        for item in list(getattr(self, "_wall_preview_items", [])):
            try:
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self._wall_preview_items = []

    def toggle_wall_draw_mode(self, enabled: bool):
        self.wall_draw_mode = bool(enabled)
        self._wall_draw_start = None
        self._clear_wall_preview()
        if enabled:
            if not self.floor:
                QMessageBox.information(self, "No floor selected", "Load an IFC and select a floor before drawing an RF wall.")
                self.draw_wall_action.blockSignals(True); self.draw_wall_action.setChecked(False); self.draw_wall_action.blockSignals(False)
                self.wall_draw_mode = False
                return
            self.view.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage("Draw RF wall: click an existing IFC wall/space edge for the first endpoint. Right-click to cancel.")
        else:
            self.view.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("RF wall drawing stopped")

    def cancel_user_wall_drawing(self):
        self._wall_draw_start = None
        self._clear_wall_preview()
        self.wall_draw_mode = False
        if hasattr(self, "draw_wall_action"):
            self.draw_wall_action.blockSignals(True); self.draw_wall_action.setChecked(False); self.draw_wall_action.blockSignals(False)
        self.view.setCursor(Qt.ArrowCursor)
        self.statusBar().showMessage("RF wall drawing cancelled")

    def nearest_ifc_connection_point(self, scene_pos: QPointF) -> Tuple[QPointF, bool]:
        """Snap to a vertex or nearest boundary point on existing IFC/user geometry."""
        if not self.floor:
            return scene_pos, False
        try:
            p0 = self.view.mapToScene(0, 0)
            p1 = self.view.mapToScene(28, 0)
            snap_radius = max(0.02, abs(p1.x() - p0.x()))
        except Exception:
            snap_radius = 0.75
        sx, sy = float(scene_pos.x()), float(scene_pos.y())
        source = Point(sx, sy)
        best = QPointF(sx, sy)
        best_distance = snap_radius
        for obj in list(self.floor.walls) + list(self.floor.spaces):
            try:
                boundary = obj.polygon.exterior
                projected = boundary.interpolate(boundary.project(source))
                distance = float(projected.distance(source))
                if distance <= best_distance:
                    best = QPointF(float(projected.x), float(projected.y)); best_distance = distance
                for x, y in boundary.coords:
                    distance = math.hypot(float(x) - sx, float(y) - sy)
                    if distance <= best_distance:
                        best = QPointF(float(x), float(y)); best_distance = distance
            except Exception:
                continue
        return best, best_distance <= snap_radius

    def show_user_wall_preview(self, end_point: QPointF):
        self._clear_wall_preview()
        if self._wall_draw_start is None or self.view.scene() is None:
            return
        pen = QPen(QColor("#FF7800"), 0)
        pen.setCosmetic(True); pen.setStyle(Qt.DashLine)
        line = self.view.scene().addLine(
            self._wall_draw_start.x(), self._wall_draw_start.y(), end_point.x(), end_point.y(), pen
        )
        line.setZValue(Z_AP_LABEL + 20)
        marker = self.view.scene().addEllipse(
            self._wall_draw_start.x() - 0.15, self._wall_draw_start.y() - 0.15, 0.3, 0.3,
            QPen(QColor("#FF7800"), 0), QBrush(QColor("#FF7800")),
        )
        marker.setZValue(Z_AP_LABEL + 21)
        self._wall_preview_items = [line, marker]

    def capture_user_wall_point(self, scene_pos: QPointF):
        if not self.floor:
            return
        snap, snapped = self.nearest_ifc_connection_point(scene_pos)
        if not snapped:
            self.statusBar().showMessage("RF wall endpoints must connect to an existing IFC wall or space boundary. Click closer to an edge.")
            return
        if self._wall_draw_start is None:
            self._wall_draw_start = snap
            self.show_user_wall_preview(snap)
            self.statusBar().showMessage("First wall endpoint captured. Click another existing IFC element/edge for the second endpoint.")
            return
        start = self._wall_draw_start
        if math.hypot(start.x() - snap.x(), start.y() - snap.y()) < 0.05:
            self.statusBar().showMessage("The second endpoint must be different from the first endpoint.")
            return
        thickness = max(0.02, float(self.heatmap_settings.user_wall_default_thickness_m))
        polygon = LineString([(start.x(), start.y()), (snap.x(), snap.y())]).buffer(
            thickness / 2.0, cap_style=2, join_style=2
        )
        wall_type = self.heatmap_settings.user_wall_default_type or "partition"
        profile = dict(
            self.heatmap_settings.default_wall_attenuation_by_material_db.get(
                wall_type.lower(), self.heatmap_settings.default_wall_attenuation_by_material_db.get("default", {})
            )
        )
        wall = Wall2D(
            guid=f"user-rf-wall-{uuid.uuid4().hex}", name="User RF blocking wall", floor=self.floor.name,
            source_file="User RF wall", type_name=wall_type, material=wall_type, polygon=polygon,
            z_min=float(self.floor.elevation), z_max=float(self.floor.elevation) + 3.0,
            source_storey=self.floor.name, attenuation_by_band_db=profile,
            rf_type_override=wall_type, rf_customised=True, is_user_created=True,
            user_wall_thickness_m=thickness,
        )
        self.floor.walls.append(wall)
        self._wall_draw_start = None
        self._clear_wall_preview()
        self.last_result = None
        self.edit_wall_rf_properties(wall)
        self.draw_floor(); self.populate_wall_table()
        self.statusBar().showMessage("Created RF blocking wall. Continue clicking to draw another, or right-click to finish.")

    def _wall_instances(self, wall: Wall2D) -> List[Wall2D]:
        if wall.is_user_created:
            return [wall]
        return [
            candidate for floor in self.floors.values() for candidate in floor.walls
            if candidate.guid == wall.guid and candidate.source_file == wall.source_file
        ] or [wall]

    def edit_wall_rf_properties(self, wall: Wall2D):
        dlg = WallAttenuationDialog(
            self, wall, self._frequency_bands(), self.heatmap_settings.default_wall_attenuation_by_material_db
        )
        if dlg.exec() != QDialog.Accepted:
            return
        wall_type, attenuation = dlg.values()
        for instance in self._wall_instances(wall):
            instance.rf_type_override = wall_type
            instance.attenuation_by_band_db.update({float(k): float(v) for k, v in attenuation.items()})
            instance.rf_customised = True
        self.last_result = None
        self.populate_wall_table(); self.draw_floor()
        self.statusBar().showMessage(f"Updated RF attenuation for {wall.name or wall.guid}")

    @staticmethod
    def _wall_major_axis_angle_deg(polygon: Polygon) -> float:
        """Return the dominant, unoriented plan angle of a wall polygon."""
        if polygon is None or polygon.is_empty:
            raise ValueError("The selected wall has no usable geometry.")
        rectangle = polygon.minimum_rotated_rectangle
        coords = list(rectangle.exterior.coords)
        if len(coords) < 3:
            raise ValueError("The selected wall has no usable axis.")
        edges = []
        for start, end in zip(coords, coords[1:]):
            dx = float(end[0]) - float(start[0])
            dy = float(end[1]) - float(start[1])
            length = math.hypot(dx, dy)
            if length > 1e-9:
                edges.append((length, math.degrees(math.atan2(dy, dx))))
        if not edges:
            raise ValueError("The selected wall has no usable axis.")
        return max(edges, key=lambda value: value[0])[1]

    def _origin_information_for_wall(self, wall: Wall2D) -> Optional[Dict[str, object]]:
        """Return imported origin metadata for the IFC that supplied ``wall``."""
        source_name = Path(str(wall.source_file or "")).name.casefold()
        if not source_name:
            return None
        for key, info in self.ifc_origin_info.items():
            if not isinstance(info, dict):
                continue
            candidates = {
                Path(str(key)).name.casefold(),
                Path(str(info.get("path", ""))).name.casefold(),
                Path(str(info.get("file", ""))).name.casefold(),
            }
            if source_name in candidates:
                return info
        return None

    def _ifc_insertion_point_for_wall(self, wall: Wall2D) -> Tuple[float, float, str]:
        """Return the selected IFC insertion point in current scene coordinates.

        Site placement is preferred, followed by building placement and the
        geometric representation context world origin. Placement coordinates
        are converted to metres before the current IFC alignment is applied.
        """
        info = self._origin_information_for_wall(wall)
        raw_x = raw_y = 0.0
        source = "IFC model origin"
        if info is not None:
            try:
                unit_scale = float(info.get("length_unit_scale_to_m", 1.0) or 1.0)
            except Exception:
                unit_scale = 1.0

            placement = None
            for collection_name, label in (("sites", "IfcSite placement"), ("buildings", "IfcBuilding placement")):
                collection = info.get(collection_name, []) or []
                if collection and isinstance(collection[0], dict):
                    candidate = collection[0].get("placement", {})
                    if isinstance(candidate, dict):
                        placement = candidate
                        source = label
                        break
            if placement is not None:
                raw_x = float(placement.get("x", 0.0) or 0.0) * unit_scale
                raw_y = float(placement.get("y", 0.0) or 0.0) * unit_scale
            else:
                for context in info.get("contexts", []) or []:
                    if not isinstance(context, dict):
                        continue
                    origin = context.get("world_origin", [])
                    if isinstance(origin, (list, tuple)) and len(origin) >= 2:
                        raw_x = float(origin[0]) * unit_scale
                        raw_y = float(origin[1]) * unit_scale
                        source = "IFC world-coordinate origin"
                        break

        scene_x, scene_y = self.ifc_alignment.map_xy(raw_x, raw_y)
        return float(scene_x), float(scene_y), source

    @staticmethod
    def _rotation_about_point_matrix(angle_deg: float, pivot_x: float, pivot_y: float) -> Tuple[float, float, float, float, float, float]:
        """Return a Shapely affine matrix that rotates about ``pivot``."""
        angle = math.radians(float(angle_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return (
            cos_a,
            -sin_a,
            sin_a,
            cos_a,
            float(pivot_x) - cos_a * float(pivot_x) + sin_a * float(pivot_y),
            float(pivot_y) - sin_a * float(pivot_x) - cos_a * float(pivot_y),
        )

    def rotate_ifc_to_align_wall_with_x_axis(self, wall: Wall2D):
        """Rotate the complete IFC model about its insertion point.

        The selected imported wall becomes 0 degrees to model +X. All imported
        walls, spaces, user RF walls and access points are transformed together
        so RF geometry remains registered. The DXF overlay and display-only view
        rotation are intentionally unchanged.
        """
        if wall.is_user_created:
            QMessageBox.information(
                self,
                "Imported IFC wall required",
                "Select a wall imported from an IFC file to define the IFC X-axis.",
            )
            return
        try:
            current_angle = self._wall_major_axis_angle_deg(wall.polygon)
            # Wall axes are unoriented, so select the smallest equivalent turn.
            delta = ((-current_angle + 90.0) % 180.0) - 90.0
            pivot_x, pivot_y, pivot_source = self._ifc_insertion_point_for_wall(wall)
            if abs(delta) <= 1e-9:
                self.statusBar().showMessage(
                    f"{wall.name or wall.guid} is already aligned to model X; "
                    f"IFC insertion point ({pivot_source}) is ({pivot_x:.3f}, {pivot_y:.3f}) m."
                )
                return
            delta_matrix = self._rotation_about_point_matrix(delta, pivot_x, pivot_y)
            self.apply_ifc_delta_alignment(
                delta_matrix,
                status_prefix=(
                    f"Rotated IFC {delta:.3f}° about {pivot_source} "
                    f"({pivot_x:.3f}, {pivot_y:.3f}) m"
                ),
            )
            self.last_result = None
            self.rssi_results_by_frequency = {}
            aligned_angle = self._wall_major_axis_angle_deg(wall.polygon)
            aligned_error = ((aligned_angle + 90.0) % 180.0) - 90.0
            self.statusBar().showMessage(
                f"Rotated IFC {delta:.3f}° about {pivot_source} "
                f"({pivot_x:.3f}, {pivot_y:.3f}) m; selected wall is "
                f"{aligned_error:.6f}° to model X."
            )
        except Exception as exc:
            QMessageBox.warning(self, "IFC rotation failed", str(exc))

    def reset_wall_rf_properties(self, wall: Wall2D):
        for instance in self._wall_instances(wall):
            instance.rf_type_override = ""
            instance.rf_customised = False
            instance.attenuation_by_band_db = self._profile_for_wall_from_settings(instance)
        self.last_result = None
        self.populate_wall_table(); self.draw_floor()

    def delete_user_wall(self, wall: Wall2D):
        if not wall.is_user_created:
            return
        floor = self.floors.get(wall.floor)
        if floor is not None:
            floor.walls = [candidate for candidate in floor.walls if candidate is not wall]
        self.last_result = None
        self.populate_wall_table(); self.draw_floor()

    # ----------------------------- Predictive AP planning -----------------------------

    def show_auto_planner_settings(self):
        dlg = AutoPlannerSettingsDialog(self, self.auto_planner_settings, list(self.antenna_patterns.keys()))
        if dlg.exec() == QDialog.Accepted:
            self.auto_planner_settings = dlg._validated_settings
            self.heatmap_settings.auto_planner_settings = self.auto_planner_settings.to_dict()
            self._apply_frequency_settings_to_model(replace_existing=False)
            self._refresh_rssi_frequency_dropdown()
            self.populate_wall_table()
            self.statusBar().showMessage("Predictive AP planner settings updated")

    def _planner_floor_area(self):
        """Return the area sampled by the predictive planner.

        IFC files frequently omit IfcSpace objects. In ``auto`` mode the
        planner therefore uses spaces when available and otherwise derives a
        practical footprint from the wall geometry. The inferred footprint is
        only a planning mask: wall polygons still provide the RF attenuation
        and candidate-point blocking behaviour.
        """
        if not self.floor:
            return None
        settings = self.auto_planner_settings
        mode = str(getattr(settings, "planning_area_mode", "auto") or "auto").lower()
        spaces = [space.polygon for space in self.floor.spaces if space.polygon is not None and not space.polygon.is_empty]

        if mode in {"auto", "spaces"} and spaces:
            try:
                area = unary_union(spaces)
                if not area.is_valid:
                    area = area.buffer(0)
                if not area.is_empty:
                    self._planner_area_source_label = f"{len(spaces)} IFC space footprint(s)"
                    return area
            except Exception:
                if mode == "spaces":
                    return None
        if mode == "spaces":
            self._planner_area_source_label = "IFC spaces only (none available)"
            return None

        wall_polygons = [wall.polygon for wall in self.floor.walls if wall.polygon is not None and not wall.polygon.is_empty]
        if not wall_polygons:
            self._planner_area_source_label = "no IFC spaces or wall geometry"
            return None
        try:
            wall_union = unary_union(wall_polygons)
            if not wall_union.is_valid:
                wall_union = wall_union.buffer(0)
            if wall_union.is_empty:
                return None

            area = None
            if shapely_concave_hull is not None:
                try:
                    area = shapely_concave_hull(wall_union, ratio=0.30, allow_holes=False)
                except Exception:
                    area = None
            # A closed wall ring may be returned as the wall material itself
            # rather than the enclosed floor. Detect that case and fill the
            # footprint using the convex hull so samples fall inside the plan.
            area_size = float(getattr(area, "area", 0.0)) if area is not None else 0.0
            wall_material_size = float(getattr(wall_union, "area", 0.0))
            if area is None or area.is_empty or area_size <= max(1e-9, wall_material_size * 1.05):
                area = wall_union.convex_hull
            if area is None or area.is_empty or float(getattr(area, "area", 0.0)) <= 1e-9:
                minx, miny, maxx, maxy = wall_union.bounds
                pad = max(0.5, float(getattr(settings, "candidate_spacing_m", 6.0)) / 2.0)
                area = box(minx - pad, miny - pad, maxx + pad, maxy + pad)

            margin = max(0.0, float(getattr(settings, "wall_footprint_margin_m", 0.0)))
            if margin > 0.0:
                area = area.buffer(margin, join_style=2)
            if not area.is_valid:
                area = area.buffer(0)
            if area.is_empty:
                return None
            self._planner_area_source_label = f"footprint inferred from {len(wall_polygons)} IFC/RF wall polygon(s)"
            return area
        except Exception:
            return None

    @staticmethod
    def _planner_tree_hits(tree, walls: List[Wall2D], geometry) -> List[Wall2D]:
        if tree is None:
            return walls
        try:
            hits = tree.query(geometry)
        except Exception:
            return walls
        result: List[Wall2D] = []
        for hit in hits:
            if isinstance(hit, (int, np.integer)):
                idx = int(hit)
                if 0 <= idx < len(walls): result.append(walls[idx])
            else:
                # Shapely 1.x may return geometry objects.
                for idx, candidate in enumerate(walls):
                    if candidate.polygon is hit:
                        result.append(candidate); break
        return result

    def _planner_point_is_blocked(self, point: Point, tree, walls: List[Wall2D]) -> bool:
        for wall in self._planner_tree_hits(tree, walls, point):
            try:
                if wall.polygon.covers(point):
                    return True
            except Exception:
                continue
        return False

    def _planner_grid_points(self, area, spacing: float, tree, walls: List[Wall2D], limit: int) -> List[Tuple[float, float]]:
        minx, miny, maxx, maxy = area.bounds
        spacing = max(0.25, float(spacing))
        # Avoid constructing an enormous intermediate grid merely to discard
        # most of it. Treat the configured spacing as the preferred minimum
        # and increase it only when the geometric area would exceed the point
        # budget by a substantial amount.
        try:
            estimated_points = max(0.0, float(area.area)) / max(1e-9, spacing * spacing)
            if limit > 0 and estimated_points > float(limit) * 1.25:
                spacing *= math.sqrt(estimated_points / float(limit))
        except Exception:
            pass
        points: List[Tuple[float, float]] = []
        y = miny + spacing / 2.0
        while y <= maxy:
            x = minx + spacing / 2.0
            while x <= maxx:
                point = Point(float(x), float(y))
                try:
                    inside = area.covers(point)
                except Exception:
                    inside = area.contains(point)
                if inside and not self._planner_point_is_blocked(point, tree, walls):
                    points.append((float(x), float(y)))
                x += spacing
            y += spacing
        if len(points) > limit:
            indices = np.linspace(0, len(points) - 1, limit, dtype=int)
            points = [points[int(i)] for i in indices]
        return points

    def _planner_rssi(self, x: float, y: float, ap: AccessPoint, radio: APRadio, wall_tree, walls: List[Wall2D]) -> float:
        if not self.floor:
            return -200.0
        dx = float(x) - float(ap.x); dy = float(y) - float(ap.y)
        dz = float(ap.mount_height_m) - float(ap.rx_height_m)
        distance = max(1.0, math.sqrt(dx * dx + dy * dy + dz * dz))
        radius = RFEngine.cutoff_radius_m_for_radio(radio, self.heatmap_settings)
        if radius > 0.0 and distance > radius:
            return float(self.heatmap_settings.disconnected_rssi_dbm)
        path_loss = RFEngine.free_space_loss_db_at_1m(radio.frequency_mhz) + 10.0 * float(ap.path_loss_exponent) * math.log10(distance)
        bearing = math.degrees(math.atan2(dy, dx))
        az_rel = bearing - float(ap.azimuth_deg)
        horizontal_distance = max(1e-9, math.hypot(dx, dy))
        elevation = math.degrees(math.atan2(float(ap.rx_height_m) - float(ap.mount_height_m), horizontal_distance))
        el_rel = elevation + float(ap.downtilt_deg)
        pattern = self.antenna_patterns.get(radio.antenna_pattern)
        pattern_gain = pattern.gain_dbi(az_rel, el_rel) if pattern else 0.0
        line = LineString([(float(ap.x), float(ap.y)), (float(x), float(y))])
        wall_loss = 0.0
        for wall in self._planner_tree_hits(wall_tree, walls, line):
            try:
                if wall.polygon.intersects(line):
                    wall_loss += wall.attenuation_db_for_frequency(radio.frequency_mhz)
            except Exception:
                continue
        return float(radio.tx_power_dbm) + float(radio.antenna_gain_dbi) + pattern_gain - path_loss - wall_loss

    @staticmethod
    def _combine_frequency_masks(masks: List[np.ndarray], mode: str, count: int) -> np.ndarray:
        if not masks:
            return np.zeros(count, dtype=bool)
        if mode == "any":
            return np.logical_or.reduce(masks)
        return np.logical_and.reduce(masks)

    def _planner_ap_masks(self, ap: AccessPoint, requirements: List[PlannerRadioRequirement], samples: List[Tuple[float, float]], wall_tree, walls: List[Wall2D]) -> List[np.ndarray]:
        masks: List[np.ndarray] = []
        for requirement in requirements:
            matching = [
                radio for radio in ap.active_radios()
                if abs(float(radio.frequency_mhz) - float(requirement.frequency_mhz)) < 1.0
            ]
            values = np.zeros(len(samples), dtype=bool)
            for radio in matching:
                for idx, (x, y) in enumerate(samples):
                    if values[idx]:
                        continue
                    values[idx] = self._planner_rssi(x, y, ap, radio, wall_tree, walls) >= requirement.minimum_rssi_dbm
            masks.append(values)
        return masks

    def _next_ap_name(self) -> str:
        used = {ap.name for ap in self.aps}
        idx = 1
        while f"AP-{idx}" in used:
            idx += 1
        return f"AP-{idx}"

    @staticmethod
    def _channel_center_mhz(frequency_mhz: float, channel: str) -> Optional[float]:
        try:
            number = float(str(channel).strip())
        except Exception:
            return None
        frequency_mhz = float(frequency_mhz)
        if 2300.0 <= frequency_mhz < 3000.0:
            if abs(number - 14.0) < 0.1:
                return 2484.0
            return 2407.0 + 5.0 * number
        if 4900.0 <= frequency_mhz < 5925.0:
            return 5000.0 + 5.0 * number
        if 5925.0 <= frequency_mhz < 7125.0:
            return 5950.0 + 5.0 * number
        return None

    @classmethod
    def _channel_overlap_fraction(cls, frequency_mhz: float, channel_a: str, width_a: float, channel_b: str, width_b: float) -> float:
        center_a = cls._channel_center_mhz(frequency_mhz, channel_a)
        center_b = cls._channel_center_mhz(frequency_mhz, channel_b)
        if center_a is None or center_b is None:
            return 1.0 if str(channel_a) == str(channel_b) and str(channel_a) else 0.0
        half_total = max(1.0, (float(width_a) + float(width_b)) / 2.0)
        separation = abs(center_a - center_b)
        return max(0.0, 1.0 - separation / half_total)

    def _assign_planner_channels(self, new_aps: List[AccessPoint], requirements: List[PlannerRadioRequirement]):
        new_ids = {id(ap) for ap in new_aps}
        assigned = [ap for ap in self.aps if id(ap) not in new_ids]
        for ap in new_aps:
            for req, radio in zip(requirements, ap.radios):
                channels = list(req.channels) or [""]
                best_channel = channels[0]
                best_cost = float("inf")
                radius = req.cutoff_radius_m or float(self.heatmap_settings.ap_cutoff_radius_by_frequency_m.get(req.frequency_mhz, 35.0))
                for channel in channels:
                    cost = 0.0
                    for other in assigned:
                        distance = math.hypot(ap.x - other.x, ap.y - other.y)
                        if distance > max(radius * 1.5, 10.0):
                            continue
                        for other_radio in other.active_radios():
                            if abs(other_radio.frequency_mhz - req.frequency_mhz) >= 1.0 or not str(other_radio.channel):
                                continue
                            overlap = self._channel_overlap_fraction(
                                req.frequency_mhz, str(channel), req.channel_width_mhz,
                                str(other_radio.channel), float(other_radio.channel_width_mhz),
                            )
                            if overlap > 0.0:
                                cost += overlap * (1.0 + req.spectrum_occupancy_percent / 100.0) / max(distance, 1.0)
                    if cost < best_cost:
                        best_cost = cost; best_channel = channel
                radio.channel = str(best_channel)
            assigned.append(ap)

    def run_auto_planner(self):
        if not self.floor:
            QMessageBox.information(self, "No floor selected", "Load an IFC and select a floor before predicting AP locations.")
            return
        settings = self.auto_planner_settings
        requirements = [r for r in settings.radio_requirements if r.enabled]
        if not requirements:
            QMessageBox.information(self, "No planner radios", "Enable at least one frequency in Planner settings.")
            return

        if settings.remove_previous_planned_aps:
            self.aps = [ap for ap in self.aps if not (ap.floor == self.floor.name and ap.planned)]
        existing = [ap for ap in self.aps if ap.floor == self.floor.name and settings.keep_existing_aps]
        area = self._planner_floor_area()
        if area is None or area.is_empty:
            if settings.planning_area_mode == "spaces":
                message = (
                    "This floor contains no usable IfcSpace geometry. Change Planning area source to "
                    "Automatic or Infer floor footprint from IFC walls to plan APs without spaces."
                )
            else:
                message = "No usable IFC space or wall geometry was found for the selected floor."
            QMessageBox.warning(self, "No plannable area", message)
            return
        walls = list(self.floor.walls)
        try:
            from shapely.strtree import STRtree
            wall_tree = STRtree([wall.polygon for wall in walls]) if walls else None
        except Exception:
            wall_tree = None

        samples = self._planner_grid_points(area, settings.sample_spacing_m, wall_tree, walls, 2500)
        # The previous fixed 450-location cap prevented the planner from ever
        # reaching the newly supported 10,000-AP limit. Scale the candidate
        # budget with the configured maximum while retaining a practical lower
        # bound for ordinary floor plans.
        candidate_location_limit = min(10_000, max(450, int(settings.maximum_aps) * 2))
        candidates = self._planner_grid_points(
            area, settings.candidate_spacing_m, wall_tree, walls, candidate_location_limit
        )
        # Include room representative points because they often place APs more naturally than a global grid.
        for space in self.floor.spaces:
            try:
                point = space.polygon.representative_point()
                candidate = (float(point.x), float(point.y))
                if candidate not in candidates and not self._planner_point_is_blocked(point, wall_tree, walls):
                    candidates.append(candidate)
            except Exception:
                pass
        if not samples or not candidates:
            QMessageBox.warning(self, "No planner points", "The selected floor did not produce usable coverage samples and AP candidates.")
            return

        directional = any(not req.antenna_pattern.lower().startswith("omni") for req in requirements)
        azimuths = list(range(0, 360, 45)) if directional else [0]
        candidate_specs = [(x, y, float(az)) for x, y in candidates for az in azimuths]
        candidate_spec_limit = min(20_000, max(700, int(settings.maximum_aps) * 2))
        if len(candidate_specs) > candidate_spec_limit:
            indices = np.linspace(0, len(candidate_specs) - 1, candidate_spec_limit, dtype=int)
            candidate_specs = [candidate_specs[int(i)] for i in indices]

        progress = QProgressDialog("Evaluating predictive AP locations…", "Cancel", 0, len(candidate_specs) + settings.maximum_aps, self)
        progress.setWindowTitle("Predictive AP planner"); progress.setWindowModality(Qt.WindowModal); progress.setMinimumDuration(0); progress.show()

        covered_by_radio = [np.zeros(len(samples), dtype=bool) for _ in requirements]
        for ap in existing:
            masks = self._planner_ap_masks(ap, requirements, samples, wall_tree, walls)
            covered_by_radio = [current | mask for current, mask in zip(covered_by_radio, masks)]

        candidate_records = []
        try:
            for idx, (x, y, azimuth) in enumerate(candidate_specs):
                if progress.wasCanceled():
                    raise RuntimeError("Predictive AP planning cancelled")
                radios = [APRadio(
                    name=req.name, frequency_mhz=req.frequency_mhz, tx_power_dbm=req.tx_power_dbm,
                    antenna_pattern=req.antenna_pattern, enabled=True, cutoff_radius_m=req.cutoff_radius_m,
                    antenna_gain_dbi=req.antenna_gain_dbi, channel="", channel_width_mhz=req.channel_width_mhz,
                    spectrum_occupancy_percent=req.spectrum_occupancy_percent,
                ) for req in requirements]
                temp_ap = AccessPoint(
                    name="candidate", x=x, y=y, floor=self.floor.name, radios=radios,
                    path_loss_exponent=float(self.ple.value()), azimuth_deg=azimuth,
                    mount_height_m=float(self.mount_height.value()), rx_height_m=float(self.rx_height.value()),
                    max_clients=settings.clients_per_ap, planned=True,
                )
                radio_masks = self._planner_ap_masks(temp_ap, requirements, samples, wall_tree, walls)
                candidate_records.append((temp_ap, radio_masks))
                progress.setValue(idx + 1)
                if idx % 5 == 0: QApplication.processEvents()

            overall = self._combine_frequency_masks(covered_by_radio, settings.coverage_mode, len(samples))
            target_fraction = settings.target_coverage_percent / 100.0
            average_occupancy = sum(req.spectrum_occupancy_percent for req in requirements) / len(requirements)
            effective_clients = max(1.0, settings.clients_per_ap * max(0.05, 1.0 - average_occupancy / 100.0))
            capacity_ap_count = int(math.ceil(settings.expected_clients / effective_clients)) if settings.expected_clients > 0 else 0
            selected: List[Tuple[AccessPoint, List[np.ndarray]]] = []
            positions = [(ap.x, ap.y) for ap in existing]
            remaining = list(range(len(candidate_records)))

            while remaining and len(selected) < settings.maximum_aps:
                coverage_fraction = float(np.count_nonzero(overall)) / max(1, len(overall))
                total_capacity_aps = len(existing) + len(selected)
                if coverage_fraction >= target_fraction and total_capacity_aps >= capacity_ap_count:
                    break
                best_index = None; best_score = -1.0
                coverage_needed = coverage_fraction < target_fraction
                for record_index in remaining:
                    candidate_ap, radio_masks = candidate_records[record_index]
                    if positions:
                        min_distance = min(math.hypot(candidate_ap.x - px, candidate_ap.y - py) for px, py in positions)
                        if min_distance + 1e-9 < settings.minimum_ap_spacing_m:
                            continue
                    else:
                        min_distance = settings.minimum_ap_spacing_m
                    proposed_by_radio = [current | mask for current, mask in zip(covered_by_radio, radio_masks)]
                    proposed_overall = self._combine_frequency_masks(proposed_by_radio, settings.coverage_mode, len(samples))
                    overall_gain = int(np.count_nonzero(proposed_overall & ~overall))
                    band_gain = sum(int(np.count_nonzero(mask & ~current)) for mask, current in zip(radio_masks, covered_by_radio))
                    if coverage_needed:
                        score = overall_gain * 1000.0 + band_gain * 10.0 + min_distance
                    else:
                        # Coverage is met but client capacity requires more APs: spread them across covered demand.
                        score = min_distance * 10.0 + sum(int(np.count_nonzero(mask)) for mask in radio_masks) / max(1, len(samples))
                    if score > best_score:
                        best_score = score; best_index = record_index
                if best_index is None:
                    break
                candidate_ap, radio_masks = candidate_records[best_index]
                selected.append((candidate_ap, radio_masks)); positions.append((candidate_ap.x, candidate_ap.y))
                covered_by_radio = [current | mask for current, mask in zip(covered_by_radio, radio_masks)]
                overall = self._combine_frequency_masks(covered_by_radio, settings.coverage_mode, len(samples))
                remaining.remove(best_index)
                progress.setValue(len(candidate_specs) + len(selected)); QApplication.processEvents()

            new_aps: List[AccessPoint] = []
            used_names = {ap.name for ap in self.aps}
            next_name_index = 1
            for candidate_ap, _ in selected:
                while f"AP-{next_name_index}" in used_names:
                    next_name_index += 1
                candidate_ap.name = f"AP-{next_name_index}"
                used_names.add(candidate_ap.name)
                next_name_index += 1
                candidate_ap.tx_power_dbm = candidate_ap.radios[0].tx_power_dbm
                candidate_ap.frequency_mhz = candidate_ap.radios[0].frequency_mhz
                candidate_ap.antenna_pattern = candidate_ap.radios[0].antenna_pattern
                self.aps.append(candidate_ap); new_aps.append(candidate_ap)
            self._assign_planner_channels(new_aps, requirements)

        except RuntimeError as exc:
            progress.close()
            self.statusBar().showMessage(str(exc))
            return
        except Exception as exc:
            progress.close()
            QMessageBox.warning(self, "Predictive AP planner failed", str(exc))
            return
        finally:
            progress.close()

        self.last_result = None; self.rssi_results_by_frequency = {}
        self._refresh_rssi_frequency_dropdown()
        self.populate_ap_table(); self.draw_floor()
        overall_pct = 100.0 * float(np.count_nonzero(overall)) / max(1, len(overall))
        band_lines = []
        for req, mask in zip(requirements, covered_by_radio):
            pct = 100.0 * float(np.count_nonzero(mask)) / max(1, len(mask))
            band_lines.append(f"{req.name} ({req.frequency_mhz:g} MHz): {pct:.1f}% at ≥ {req.minimum_rssi_dbm:g} dBm")
        available_capacity = int((len(existing) + len(new_aps)) * effective_clients)
        warnings = []
        if overall_pct + 1e-6 < settings.target_coverage_percent:
            warnings.append("The coverage target was not fully achievable with the configured AP limit, spacing, radio patterns and wall losses.")
        if available_capacity < settings.expected_clients:
            warnings.append(f"The estimated effective capacity ({available_capacity}) is below the expected {settings.expected_clients} clients.")
        warning = ("\n\n" + "\n".join(warnings)) if warnings else ""
        QMessageBox.information(
            self, "Predictive AP plan complete",
            f"Added {len(new_aps)} predicted AP(s) on {self.floor.name}.\n"
            f"Planning area: {getattr(self, '_planner_area_source_label', 'selected floor geometry')}.\n"
            f"Overall coverage ({'every band' if settings.coverage_mode == 'all' else 'any band'}): {overall_pct:.1f}%\n"
            f"Effective client capacity after {average_occupancy:.1f}% average spectrum occupancy: approximately {available_capacity} clients.\n\n"
            + "\n".join(band_lines) + warning,
        )

    # ----------------------------- RF plan persistence -----------------------------

    @staticmethod
    def _radio_to_dict(radio: APRadio) -> Dict[str, object]:
        return {
            "name": radio.name, "frequency_mhz": radio.frequency_mhz, "tx_power_dbm": radio.tx_power_dbm,
            "antenna_pattern": radio.antenna_pattern, "enabled": radio.enabled, "cutoff_radius_m": radio.cutoff_radius_m,
            "antenna_gain_dbi": radio.antenna_gain_dbi, "channel": radio.channel,
            "channel_width_mhz": radio.channel_width_mhz, "spectrum_occupancy_percent": radio.spectrum_occupancy_percent,
        }

    @staticmethod
    def _radio_from_dict(data: Dict[str, object]) -> APRadio:
        return APRadio(
            name=str(data.get("name", "Radio")), frequency_mhz=float(data.get("frequency_mhz", 2400.0)),
            tx_power_dbm=float(data.get("tx_power_dbm", 20.0)), antenna_pattern=str(data.get("antenna_pattern", "Omni ceiling AP")),
            enabled=bool(data.get("enabled", True)), cutoff_radius_m=float(data.get("cutoff_radius_m", 0.0)),
            antenna_gain_dbi=float(data.get("antenna_gain_dbi", 0.0)), channel=str(data.get("channel", "")),
            channel_width_mhz=float(data.get("channel_width_mhz", 20.0)),
            spectrum_occupancy_percent=float(data.get("spectrum_occupancy_percent", 0.0)),
        )

    def _rf_plan_data(self) -> Dict[str, object]:
        aps = []
        for ap in self.aps:
            aps.append({
                "name": ap.name, "x": ap.x, "y": ap.y, "floor": ap.floor,
                "tx_power_dbm": ap.tx_power_dbm, "frequency_mhz": ap.frequency_mhz,
                "reference_loss_db_at_1m": ap.reference_loss_db_at_1m, "path_loss_exponent": ap.path_loss_exponent,
                "antenna_pattern": ap.antenna_pattern, "azimuth_deg": ap.azimuth_deg, "downtilt_deg": ap.downtilt_deg,
                "mount_height_m": ap.mount_height_m, "rx_height_m": ap.rx_height_m,
                "max_clients": ap.max_clients, "planned": ap.planned,
                "radios": [self._radio_to_dict(radio) for radio in ap.radios],
            })
        user_walls = []
        overrides = []
        for floor in self.floors.values():
            for wall in floor.walls:
                if wall.is_user_created:
                    user_walls.append({
                        "guid": wall.guid, "name": wall.name, "floor": wall.floor, "type_name": wall.type_name,
                        "material": wall.material, "polygon": [[float(x), float(y)] for x, y in wall.polygon.exterior.coords],
                        "z_min": wall.z_min, "z_max": wall.z_max, "rf_type_override": wall.rf_type_override,
                        "attenuation_by_band_db": {str(k): float(v) for k, v in wall.attenuation_by_band_db.items()},
                        "thickness_m": wall.user_wall_thickness_m,
                    })
                elif wall.rf_customised or wall.rf_geometry_customised:
                    override = {
                        "guid": wall.guid, "source_file": wall.source_file, "floor": wall.floor,
                        "rf_type_override": wall.rf_type_override, "rf_customised": wall.rf_customised,
                        "attenuation_by_band_db": {str(k): float(v) for k, v in wall.attenuation_by_band_db.items()},
                        "rf_geometry_customised": wall.rf_geometry_customised,
                    }
                    if wall.rf_geometry_customised:
                        override["polygon"] = [[float(x), float(y)] for x, y in wall.polygon.exterior.coords]
                    overrides.append(override)
        return {
            "format": "rf-attenuation-plan", "version": 1,
            "ifc_paths": [str(path) for path in self.loaded_ifc_paths],
            "selected_floor": self.floor.name if self.floor else "", "view_rotation_deg": self.view_rotation_deg,
            "ifc_alignment": {"dx": self.ifc_alignment.dx, "dy": self.ifc_alignment.dy, "rotation_deg": self.ifc_alignment.rotation_deg, "scale": self.ifc_alignment.scale},
            "auto_planner_settings": self.auto_planner_settings.to_dict(),
            "access_points": aps, "user_walls": user_walls, "wall_overrides": overrides,
        }

    def save_rf_plan(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save RF plan", "rf_plan.rfplan.json", "RF plan (*.rfplan.json);;JSON files (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self._rf_plan_data(), indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"Saved RF plan: {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "RF plan save failed", str(exc))

    def load_rf_plan(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load RF plan", "", "RF plan (*.rfplan.json *.json);;All files (*.*)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("format") != "rf-attenuation-plan":
                raise ValueError("This is not an RF Attenuation Simulator plan file.")
        except Exception as exc:
            QMessageBox.warning(self, "RF plan load failed", str(exc)); return
        if not self.floors:
            paths = [Path(v) for v in data.get("ifc_paths", []) if str(v).strip()]
            existing_paths = [candidate for candidate in paths if candidate.exists()]
            if existing_paths:
                self._pending_plan_data = data
                self._load_ifc_paths(existing_paths, replace=True)
                return
        self._apply_rf_plan_data(data)

    def _apply_rf_plan_data(self, data: Dict[str, object]):
        alignment_data = data.get("ifc_alignment", {}) or {}
        try:
            target_alignment = AlignmentTransform(
                dx=float(alignment_data.get("dx", 0.0)), dy=float(alignment_data.get("dy", 0.0)),
                rotation_deg=float(alignment_data.get("rotation_deg", 0.0)), scale=float(alignment_data.get("scale", 1.0)),
            )
            self.apply_ifc_alignment(target_alignment)
        except Exception:
            pass
        # Remove prior user geometry before restoring the saved plan.
        for floor in self.floors.values():
            floor.walls = [wall for wall in floor.walls if not wall.is_user_created]
            for wall in floor.walls:
                if wall.rf_original_polygon is not None:
                    wall.polygon = wall.rf_original_polygon
                wall.rf_original_polygon = None
                wall.rf_geometry_customised = False
                wall.rf_customised = False; wall.rf_type_override = ""
                wall.attenuation_by_band_db = self._profile_for_wall_from_settings(wall)
        override_lookup_exact = {
            (str(item.get("source_file", "")), str(item.get("guid", "")), str(item.get("floor", ""))): item
            for item in data.get("wall_overrides", []) if isinstance(item, dict) and str(item.get("floor", ""))
        }
        override_lookup_legacy = {
            (str(item.get("source_file", "")), str(item.get("guid", ""))): item
            for item in data.get("wall_overrides", []) if isinstance(item, dict)
        }
        for floor in self.floors.values():
            for wall in floor.walls:
                item = override_lookup_exact.get((wall.source_file, wall.guid, wall.floor))
                if item is None:
                    item = override_lookup_legacy.get((wall.source_file, wall.guid))
                if item:
                    wall.rf_type_override = str(item.get("rf_type_override", ""))
                    wall.rf_customised = bool(item.get("rf_customised", bool(wall.rf_type_override)))
                    wall.attenuation_by_band_db.update({float(k): float(v) for k, v in dict(item.get("attenuation_by_band_db", {})).items()})
                    coords = item.get("polygon", [])
                    if bool(item.get("rf_geometry_customised", False)) and isinstance(coords, list) and len(coords) >= 3:
                        polygon = Polygon([(float(value[0]), float(value[1])) for value in coords])
                        if not polygon.is_valid:
                            polygon = polygon.buffer(0)
                        if not polygon.is_empty:
                            wall.rf_original_polygon = wall.polygon
                            wall.polygon = polygon
                            wall.rf_geometry_customised = True
        for item in data.get("user_walls", []):
            if not isinstance(item, dict): continue
            floor = self.floors.get(str(item.get("floor", "")))
            coords = item.get("polygon", [])
            if floor is None or not isinstance(coords, list) or len(coords) < 3: continue
            polygon = Polygon([(float(v[0]), float(v[1])) for v in coords])
            floor.walls.append(Wall2D(
                guid=str(item.get("guid", f"user-rf-wall-{uuid.uuid4().hex}")), name=str(item.get("name", "User RF blocking wall")),
                floor=floor.name, source_file="User RF wall", type_name=str(item.get("type_name", "partition")),
                material=str(item.get("material", "partition")), polygon=polygon,
                z_min=float(item.get("z_min", floor.elevation)), z_max=float(item.get("z_max", floor.elevation + 3.0)),
                source_storey=floor.name, attenuation_by_band_db={float(k): float(v) for k, v in dict(item.get("attenuation_by_band_db", {})).items()},
                rf_type_override=str(item.get("rf_type_override", item.get("type_name", "partition"))),
                rf_customised=True, is_user_created=True, user_wall_thickness_m=float(item.get("thickness_m", 0.15)),
            ))
        self.aps = []
        for item in data.get("access_points", []):
            if not isinstance(item, dict): continue
            radios = [self._radio_from_dict(v) for v in item.get("radios", []) if isinstance(v, dict)]
            self.aps.append(AccessPoint(
                name=str(item.get("name", self._next_ap_name())), x=float(item.get("x", 0.0)), y=float(item.get("y", 0.0)), floor=str(item.get("floor", "")),
                tx_power_dbm=float(item.get("tx_power_dbm", 20.0)), frequency_mhz=float(item.get("frequency_mhz", 2400.0)),
                reference_loss_db_at_1m=float(item.get("reference_loss_db_at_1m", 40.0)), path_loss_exponent=float(item.get("path_loss_exponent", 2.2)),
                antenna_pattern=str(item.get("antenna_pattern", "Omni ceiling AP")), azimuth_deg=float(item.get("azimuth_deg", 0.0)),
                downtilt_deg=float(item.get("downtilt_deg", 0.0)), mount_height_m=float(item.get("mount_height_m", 2.7)),
                rx_height_m=float(item.get("rx_height_m", 1.2)), radios=radios,
                max_clients=int(item.get("max_clients", 50)), planned=bool(item.get("planned", False)),
            ))
        self.auto_planner_settings = AutoPlannerSettings.from_dict(data.get("auto_planner_settings", {}))
        selected_floor = str(data.get("selected_floor", ""))
        if selected_floor in self.floors:
            self.floor_combo.setCurrentText(selected_floor)
        self.reset_view_rotation()
        saved_rotation = float(data.get("view_rotation_deg", 0.0))
        if abs(saved_rotation) > 1e-9:
            self.rotate_view(saved_rotation)
        self.last_result = None; self.rssi_results_by_frequency = {}
        self.populate_ap_table(); self.populate_wall_table(); self.draw_floor()
        self.statusBar().showMessage("Loaded RF plan")

    def _refresh_rssi_frequency_dropdown(self):
        if not hasattr(self, "rssi_view_frequency"):
            return

        current = self.rssi_view_frequency.currentText()
        self.rssi_view_frequency.blockSignals(True)
        self.rssi_view_frequency.clear()

        for band in self._frequency_bands():
            if band >= 1000:
                label = f"{band / 1000:g} GHz"
            else:
                label = f"{band:g} MHz"
            self.rssi_view_frequency.addItem(label, float(band))

        if current:
            idx = self.rssi_view_frequency.findText(current)
            if idx >= 0:
                self.rssi_view_frequency.setCurrentIndex(idx)

        self.rssi_view_frequency.blockSignals(False)

    def _selected_rssi_view_frequency(self) -> Optional[float]:
        if not hasattr(self, "rssi_view_frequency"):
            return None
        data = self.rssi_view_frequency.currentData()
        try:
            return float(data)
        except Exception:
            return None

    def _rssi_view_frequency_changed(self, *_):
        freq = self._selected_rssi_view_frequency()
        if freq is not None and freq in getattr(self, "rssi_results_by_frequency", {}):
            self.last_result = self.rssi_results_by_frequency[freq]
        else:
            self.last_result = None
        self.draw_floor()

    def _detect_dark_theme(self) -> bool:
        """Return True when the active Qt/OS palette appears to be dark."""
        app = QApplication.instance()
        palette = app.palette() if app is not None else self.palette()
        window_colour = palette.color(QPalette.Window)
        luminance = (0.299 * window_colour.red() + 0.587 * window_colour.green() + 0.114 * window_colour.blue())
        return luminance < 128

    def _theme_colours(self) -> Dict[str, QColor]:
        """Colours used by the scene, loaded from rf_heatmap_settings.json."""
        dark = bool(getattr(self, "dark_theme", False))
        hs = getattr(self, "heatmap_settings", HeatmapSettings.default())
        return {
            "background": hs.display_qcolour("background", dark, "#2A2A2A" if dark else "#FAFAFA"),
            "legend_background": hs.display_qcolour("legend_background", dark, "#2F2F2F" if dark else "#F5F5F5"),
            "legend_text": hs.display_qcolour("legend_text", dark, "#EEEEEE" if dark else "#202020"),
            "legend_border": hs.display_qcolour("legend_border", dark, "#555555" if dark else "#BBBBBB"),
            "space_pen": hs.display_qcolour("space_pen", dark, "#969696" if dark else "#5F5F5F"),
            "space_fill": hs.display_qcolour("space_fill", dark, "#3A3A3A" if dark else "#E1E1E1"),
            "space_text": hs.display_qcolour("space_text", dark, "#DCDCDC" if dark else "#282828"),
            "wall_pen": hs.display_qcolour("wall_pen", dark, "#EBEBEB" if dark else "#191919"),
            "wall_fill": hs.display_qcolour("ifc_wall_fill", dark, "#1E1E1E" if dark else "#D7D7D7"),
            "wall_alt_fill": hs.display_qcolour("ifc_linked_wall_fill", dark, "#414146" if dark else "#B9C3CD"),
            "contour_text": hs.display_qcolour("contour_text", dark, "#F0F0F0" if dark else "#141414"),
            "sample_cross": hs.display_qcolour("sample_cross", dark, "#55A0FF" if dark else "#0055FF"),
            "sample_text": hs.display_qcolour("sample_text", dark, "#55A0FF" if dark else "#0055FF"),
            "ap_same_floor": hs.display_qcolour("ap_same_floor", dark, "#4D8DFF" if dark else "#0050FF"),
            "ap_other_floor": hs.display_qcolour("ap_other_floor", dark, "#C77DFF" if dark else "#7800B4"),
            "ap_outline": hs.display_qcolour("ap_outline", dark, "#D8E4FF" if dark else "#000050"),
            "dxf_overlay": hs.display_qcolour("dxf_overlay", dark, "#62B7FF" if dark else "#0096FF"),
        }

    def _apply_theme_styles(self):
        """Apply non-scene styling that depends on the OS theme."""
        colours = self._theme_colours()
        if hasattr(self, "view") and self.view.scene() is not None:
            self.view.scene().setBackgroundBrush(QBrush(colours["background"]))
        if hasattr(self, "rssi_legend"):
            self.rssi_legend.setStyleSheet(
                "QLabel {{ background: {bg}; color: {fg}; border-top: 1px solid {border}; padding: 4px; }}".format(
                    bg=colours["legend_background"].name(),
                    fg=colours["legend_text"].name(),
                    border=colours["legend_border"].name(),
                )
            )

    @staticmethod
    def _point_xy(point) -> Tuple[float, float]:
        """Return x/y from either Qt-style points or geometry objects.

        Qt types such as ``QPointF`` expose ``x()`` and ``y()`` methods, while
        Shapely points expose ``x`` and ``y`` as float properties.  The simulator
        uses both, so this helper prevents ``TypeError: 'float' object is not
        callable`` when drawing labels from IFC-derived geometry.
        """
        x_attr = getattr(point, "x", 0.0)
        y_attr = getattr(point, "y", 0.0)
        x_val = x_attr() if callable(x_attr) else x_attr
        y_val = y_attr() if callable(y_attr) else y_attr
        return float(x_val), float(y_val)

    def _normalise_text_angle(self, angle_deg: float) -> float:
        """Keep text rotation readable on screen.

        The scene is displayed with a flipped Y axis to make IFC plans appear the
        right way up. Text ignores the view transform, so any optional line-label
        rotation must be clamped in screen space instead of model space.
        """
        angle = float(angle_deg or 0.0)
        while angle <= -180.0:
            angle += 360.0
        while angle > 180.0:
            angle -= 360.0
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        return angle

    def _font_point_size(self, configured_size: int) -> float:
        """Return a small logical font size before model scaling.

        Text is intentionally allowed to follow the QGraphicsView transform so
        it grows when the user zooms in and shrinks when they zoom out.  The
        separate text_model_scale setting converts the Qt font glyph size into
        sensible IFC/model units so labels are not building-sized.
        """
        return max(1.0, float(configured_size))

    def _add_upright_text(self, scene: QGraphicsScene, text: str, x: float, y: float, colour: QColor,
                          font_size: int, z_value: float, bold: bool = False, rotation_deg: float = 0.0):
        """Add model-scaled text that zooms with the plan but remains upright.

        The view uses ``scale(1, -1)`` so normal text would be mirrored/upside
        down.  Each text item is locally flipped in Y to cancel the view flip.
        Unlike the previous patch, the item does *not* use
        ItemIgnoresTransformations; therefore camera zoom behaves naturally and
        makes labels larger when zooming in.
        """
        item = QGraphicsSimpleTextItem(text)
        font = QFont(QApplication.font())
        font.setPointSizeF(self._font_point_size(font_size))
        font.setBold(bool(bold))
        item.setFont(font)
        item.setBrush(QBrush(colour))
        item.setZValue(z_value)
        item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        item.setAcceptedMouseButtons(Qt.NoButton)
        scene.addItem(item)

        rect = item.boundingRect()
        scale = max(0.001, float(getattr(self.heatmap_settings, "text_model_scale", 0.035)))
        angle = self._normalise_text_angle(rotation_deg)

        # Build a local transform about the text centre:
        #   1. move to desired scene position,
        #   2. scale the glyph dimensions down to model units,
        #   3. rotate if this is a contour label,
        #   4. flip in local Y so the global view flip does not invert the text,
        #   5. centre the text on the requested point.
        t = QTransform()
        t.translate(float(x), float(y))
        t.scale(scale, scale)
        t.rotate(angle)
        t.scale(1.0, -1.0)
        t.translate(-rect.width() / 2.0, -rect.height() / 2.0)
        item.setTransform(t)
        return item

    def _resolve_ifc_loader_process_count(self, requested_count: int = 0) -> int:
        return _logical_process_count(requested_count)

    def _ifc_loading_uses_multiprocessing(self) -> bool:
        # Multiprocessing is the only asynchronous IFC loading path now. If this
        # setting is false, loading falls back to a blocking single-process parse.
        return bool(getattr(self.heatmap_settings, "enable_ifc_multiprocessing", True))

    def _show_ifc_load_progress(self, total_files: int):
        """Show a visible progress bar while IFC loading is active."""
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.close()
        self._load_progress_dialog = QProgressDialog(
            "Preparing IFC loading...",
            "",
            0,
            max(1, int(total_files)),
            self,
        )
        self._load_progress_dialog.setWindowTitle("Loading IFC model")
        self._load_progress_dialog.setCancelButton(None)  # IFC parsing workers cannot be safely cancelled mid-read.
        self._load_progress_dialog.setWindowModality(Qt.WindowModal)
        self._load_progress_dialog.setMinimumDuration(0)
        self._load_progress_dialog.setAutoClose(False)
        self._load_progress_dialog.setAutoReset(False)
        self._load_progress_dialog.setValue(0)
        self._load_progress_dialog.show()
        QApplication.processEvents()

    def _update_ifc_load_progress(self, current_file: str = ""):
        if self._load_progress_dialog is None:
            return
        completed = max(0, int(self._load_completed))
        total = max(1, int(self._load_total))
        remaining = max(0, total - completed)
        suffix = f"\nCurrent file: {current_file}" if current_file else ""
        self._load_progress_dialog.setLabelText(
            f"Loading IFC files: {completed} of {total} complete\n"
            f"Remaining: {remaining}\n"
            f"Workers: {getattr(self, '_load_worker_label', 'process loader')}{suffix}"
        )
        self._load_progress_dialog.setMaximum(total)
        self._load_progress_dialog.setValue(min(completed, total))
        QApplication.processEvents()

    def _close_ifc_load_progress(self):
        if self._load_progress_dialog is not None:
            self._load_progress_dialog.setValue(max(1, self._load_total))
            self._load_progress_dialog.close()
            self._load_progress_dialog = None

    def open_dxf_overlay(self):
        """Open a DXF reference drawing directly without alignment.

        For project alignment use the toolbar action "2-point align IFC/DXF".
        That workflow asks for two snapped IFC points first, then opens the DXF
        in a separate alignment window before inserting the corrected overlay.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open DXF overlay without alignment", "", "DXF files (*.dxf);;All files (*.*)")
        if not path:
            return
        if ezdxf is None:
            QMessageBox.critical(self, "Missing dependency", "ezdxf is required. Install with: pip install ezdxf")
            return
        try:
            unit_scale = float(getattr(self.heatmap_settings, "dxf_unit_scale", 1.0))
            auto_units = bool(getattr(self.heatmap_settings, "dxf_auto_unit_scale", True))
            self.dxf_overlay = load_dxf_overlay(Path(path), unit_scale=unit_scale, auto_units=auto_units)
        except Exception as exc:
            QMessageBox.critical(self, "DXF load failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Loaded unaligned DXF overlay: {Path(path).name} ({len(self.dxf_overlay.primitives)} primitives), "
            f"units={self.dxf_overlay.source_units_name}, scale={self.dxf_overlay.effective_scale_to_metres:g} m/unit"
        )
        self.draw_floor()

    def clear_dxf_overlay(self):
        self.dxf_overlay = None
        self.statusBar().showMessage("Cleared DXF overlay")
        self.draw_floor()

    def show_dxf_alignment_dialog(self):
        if not self.floors:
            QMessageBox.information(self, "No IFC loaded", "Load an IFC model before aligning to a DXF.")
            return
        if self.dxf_overlay is None:
            QMessageBox.information(self, "No DXF overlay", "Open a DXF overlay first, then align the IFC to it.")
            return
        dlg = DxfAlignmentDialog(self, transform=self.ifc_alignment, apply_callback=self.apply_ifc_alignment)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def start_two_point_alignment(self):
        """Start the corrected two-stage DXF alignment workflow.

        Stage 1 happens in the IFC scene: select two IFC reference points with
        snapping to wall/space corners. Stage 2 opens the DXF in a separate
        pre-alignment dialog where the user selects two matching DXF points with
        DXF endpoint/corner snapping. Only the corrected DXF is then inserted.
        """
        if not self.floors or self.floor is None:
            QMessageBox.information(self, "No IFC loaded", "Load an IFC model and select a floor before aligning a DXF.")
            return
        if ezdxf is None:
            QMessageBox.critical(self, "Missing dependency", "ezdxf is required. Install with: pip install ezdxf")
            return
        self.alignment_pick_points = {}
        self._alignment_pick_sequence = ["ifc_1", "ifc_2"]
        self.alignment_pick_mode = "ifc_1"
        self._pending_dxf_alignment_path = None
        self.statusBar().showMessage("DXF alignment: click IFC reference point 1. Snapping is active on IFC corners/endpoints.")
        QMessageBox.information(
            self,
            "DXF pre-alignment",
            "Step 1: Select two reference points on the IFC first.\n\n"
            "Snapping is available to IFC wall and space corners.\n\n"
            "After the second IFC point is selected, choose the DXF file. The DXF "
            "will open in a separate alignment window where you select the two "
            "matching DXF points before the corrected DXF is inserted."
        )
        self.draw_floor()

    def _ifc_snap_points(self) -> List[QPointF]:
        """Return snap candidates for the current IFC floor."""
        pts: List[QPointF] = []
        if not self.floor:
            return pts
        for obj in list(self.floor.walls) + list(self.floor.spaces):
            try:
                coords = list(obj.polygon.exterior.coords)
            except Exception:
                continue
            for x, y in coords:
                pts.append(QPointF(float(x), float(y)))
        for ap in self.aps:
            if ap.floor == self.floor.name:
                pts.append(QPointF(float(ap.x), float(ap.y)))
        return pts

    def nearest_ifc_snap_point(self, scene_pos: QPointF) -> QPointF:
        """Return nearest IFC snap point within a screen-pixel snap radius."""
        points = self._ifc_snap_points()
        if not points:
            return scene_pos
        try:
            p0 = self.view.mapToScene(0, 0)
            p1 = self.view.mapToScene(18, 0)
            snap_radius = max(0.01, abs(p1.x() - p0.x()))
        except Exception:
            snap_radius = 0.5
        best = scene_pos
        best_d = snap_radius
        for p in points:
            d = math.hypot(p.x() - scene_pos.x(), p.y() - scene_pos.y())
            if d <= best_d:
                best = p
                best_d = d
        return best

    def show_ifc_snap_marker(self, point: QPointF):
        """Show a temporary snap marker in the IFC view."""
        scene = self.view.scene()
        if scene is None:
            return

        for item in list(getattr(self, "_ifc_snap_marker_items", [])):
            try:
                if item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                pass

        self._ifc_snap_marker_items = []

        size = 0.25
        pen = QPen(QColor("#00AEEF"), 0)
        pen.setCosmetic(True)

        h = scene.addLine(point.x() - size, point.y(), point.x() + size, point.y(), pen)
        v = scene.addLine(point.x(), point.y() - size, point.x(), point.y() + size, pen)

        h.setZValue(Z_AP_LABEL + 100)
        v.setZValue(Z_AP_LABEL + 100)

        self._ifc_snap_marker_items = [h, v]

    def capture_alignment_point(self, x: float, y: float):
        key = getattr(self, "alignment_pick_mode", None)
        if key not in {"ifc_1", "ifc_2"}:
            return
        self.alignment_pick_points[key] = (float(x), float(y))
        if key == "ifc_1":
            self.alignment_pick_mode = "ifc_2"
            self.statusBar().showMessage("Captured IFC point 1. Click IFC reference point 2.")
            self._draw_alignment_pick_marks()
            return

        self.alignment_pick_mode = None
        self.statusBar().showMessage("Captured IFC point 2. Choose DXF file for separate pre-alignment window.")
        self._draw_alignment_pick_marks()
        self._open_dxf_prealign_dialog()

    def _open_dxf_prealign_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DXF for pre-alignment", "", "DXF files (*.dxf);;All files (*.*)")
        if not path:
            self.statusBar().showMessage("DXF alignment cancelled before DXF selection.")
            return
        pts = self.alignment_pick_points
        try:
            ifc_a = QPointF(float(pts["ifc_1"][0]), float(pts["ifc_1"][1]))
            ifc_b = QPointF(float(pts["ifc_2"][0]), float(pts["ifc_2"][1]))
        except Exception:
            QMessageBox.warning(self, "IFC points missing", "Two IFC reference points are required before opening the DXF.")
            return

        self._pending_dxf_alignment_path = Path(path)
        dlg = DxfPreAlignDialog(str(path), ifc_a, ifc_b, self)
        dlg.alignmentReady.connect(self._apply_dxf_prealignment_result)
        dlg.exec()

    def _apply_dxf_prealignment_result(self, transform: SimilarityTransform2D):
        """Insert the corrected DXF overlay after the separate dialog alignment."""
        path = getattr(self, "_pending_dxf_alignment_path", None)
        if path is None:
            QMessageBox.warning(self, "DXF path missing", "No pending DXF path is available for alignment.")
            return
        try:
            self.dxf_overlay = load_dxf_overlay_with_similarity_transform(Path(path), transform)
        except Exception as exc:
            QMessageBox.critical(self, "Corrected DXF insertion failed", str(exc))
            return
        self._pending_dxf_alignment_path = None
        self.alignment_pick_points = {}
        self._alignment_pick_sequence = []
        self.draw_floor()
        self.statusBar().showMessage(
            f"Inserted pre-aligned DXF: {Path(path).name}. "
            f"Scale={transform.scale:g}, rotation={math.degrees(transform.rotation_rad):.4f}°, "
            f"offset=({transform.tx:.3f}, {transform.ty:.3f})"
        )

    def _draw_alignment_pick_marks(self):
        # Redraw the floor, then add temporary IFC pick crosses on top.
        self._preserve_view_on_redraw = True
        try:
            self.draw_floor()
        finally:
            self._preserve_view_on_redraw = False

        scene = self.view.scene()
        if scene is None:
            return

        pen = QPen(QColor("#FF00FF"), 0)
        pen.setCosmetic(True)
        size = 0.45
        for label, (x, y) in getattr(self, "alignment_pick_points", {}).items():
            l1 = scene.addLine(x - size, y, x + size, y, pen)
            l2 = scene.addLine(x, y - size, x, y + size, pen)
            l1.setZValue(Z_AP_LABEL + 90)
            l2.setZValue(Z_AP_LABEL + 90)
            self._add_upright_text(
                scene,
                label.replace("_", " ").upper(),
                x + size,
                y + size,
                QColor("#FF00FF"),
                4,
                Z_AP_LABEL + 95,
                bold=True,
            )

    def apply_two_point_alignment(self):
        """Deprecated old four-click IFC/DXF scene alignment.

        The active workflow is now:
            select IFC point 1 -> select IFC point 2 -> open DXF dialog ->
            select DXF point 1 -> select DXF point 2 -> insert corrected DXF.
        """
        QMessageBox.information(
            self,
            "Use DXF pre-alignment dialog",
            "Use the toolbar action '2-point align IFC/DXF'. The new workflow "
            "selects two IFC points first and then opens the DXF in a separate "
            "snapping alignment window."
        )

    def apply_ifc_delta_alignment(self, delta: Tuple[float, float, float, float, float, float], status_prefix: str = "Applied IFC alignment"):
        for floor in self.floors.values():
            for wall in floor.walls:
                wall.polygon = affine_transform(wall.polygon, delta)
                if wall.rf_original_polygon is not None:
                    wall.rf_original_polygon = affine_transform(wall.rf_original_polygon, delta)
            for space in floor.spaces:
                space.polygon = affine_transform(space.polygon, delta)
        for ap in self.aps:
            a, b, d, e, xoff, yoff = delta
            old_x, old_y = float(ap.x), float(ap.y)
            ap.x = a * old_x + b * old_y + xoff
            ap.y = d * old_x + e * old_y + yoff
        composed = AlignmentTransform._compose(delta, self.ifc_alignment.matrix())
        self.ifc_alignment = AlignmentTransform.from_matrix(composed)
        self.last_result = None
        self.draw_floor()
        self.populate_ap_table()
        self.populate_wall_table()
        self.statusBar().showMessage(
            f"{status_prefix}: X {self.ifc_alignment.dx:.3f}, Y {self.ifc_alignment.dy:.3f}, "
            f"rot {self.ifc_alignment.rotation_deg:.3f}°, scale {self.ifc_alignment.scale:.6f}"
        )

    def apply_ifc_alignment(self, new_transform: AlignmentTransform):
        """Apply an absolute IFC alignment by transforming current model geometry by the required delta."""
        try:
            delta = AlignmentTransform.delta_matrix(self.ifc_alignment, new_transform)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid alignment", str(exc))
            return

        self.apply_ifc_delta_alignment(delta, status_prefix="Applied IFC alignment")
        self.ifc_alignment = new_transform
        self.statusBar().showMessage(
            f"Applied IFC alignment: X {new_transform.dx:.3f}, Y {new_transform.dy:.3f}, "
            f"rot {new_transform.rotation_deg:.3f}°, scale {new_transform.scale:.6f}"
        )

    def _draw_dxf_overlay(self, scene: QGraphicsScene, colours: Dict[str, QColor]):
        overlay = getattr(self, "dxf_overlay", None)
        if overlay is None or not overlay.visible:
            return
        colour = QColor(colours.get("dxf_overlay", QColor("#0096FF")))
        colour.setAlpha(max(0, min(255, int(getattr(self.heatmap_settings, "dxf_overlay_alpha", 190)))))
        pen = QPen(colour, float(getattr(self.heatmap_settings, "dxf_overlay_line_width", 0.08)))
        pen.setCosmetic(True)
        for prim in overlay.primitives:
            if len(prim.points) < 2:
                continue
            path = QPainterPath(QPointF(prim.points[0][0], prim.points[0][1]))
            for x, y in prim.points[1:]:
                path.lineTo(QPointF(x, y))
            item = scene.addPath(path, pen)
            item.setZValue(Z_DXF_OVERLAY)

    def _ifc_path_key(self, path_obj) -> str:
        path = Path(path_obj)
        try:
            return str(path.resolve()) if path.exists() else str(path)
        except Exception:
            return str(path)

    def _mark_ifc_load_done(self, path_obj) -> bool:
        """Mark one IFC path as finished/failed. Returns False for duplicate signals."""
        key = self._ifc_path_key(path_obj)
        if key in self._load_finished_keys:
            return False
        self._load_finished_keys.add(key)
        if self._load_pending > 0:
            self._load_pending -= 1
        self._load_completed = min(self._load_total, self._load_completed + 1)
        return True

    def _schedule_ifc_load_timeout_watchdog(self):
        timeout_seconds = int(getattr(self.heatmap_settings, "ifc_load_timeout_seconds", 900) or 0)
        if timeout_seconds <= 0:
            return
        batch_id = int(self._load_batch_id)
        QTimer.singleShot(timeout_seconds * 1000, lambda bid=batch_id: self._ifc_load_timeout_check(bid))

    def _ifc_load_timeout_check(self, batch_id: int):
        if not self._loading_active or batch_id != self._load_batch_id:
            return
        missing = [k for k in self._load_started_keys if k not in self._load_finished_keys]
        if not missing:
            self._finish_ifc_batch_if_ready()
            return
        for key in missing:
            self._load_errors.append(f"{Path(key).name}: timed out during IFC loading")
            self._mark_ifc_load_done(key)
        self._update_ifc_load_progress("timeout")
        self._finish_ifc_batch_if_ready()

    def _shutdown_ifc_process_executor(self):
        if self._ifc_process_poll_timer.isActive():
            self._ifc_process_poll_timer.stop()
        executor = self._ifc_process_executor
        self._ifc_process_executor = None
        self._ifc_process_futures = {}
        if not hasattr(self, "_ifc_chunk_remaining") or not isinstance(self._ifc_chunk_remaining, dict):
            self._ifc_chunk_remaining = {}
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            except Exception:
                pass

    def _poll_ifc_process_futures(self):
        """Poll IFC worker-process futures safely from the GUI thread.

        Handles native crashes/BrokenProcessPool from huge IFC geometry reads
        without raising KeyError or hanging the progress dialog.
        """
        if not self._loading_active:
            self._shutdown_ifc_process_executor()
            return
        if not self._ifc_process_futures:
            self._finish_ifc_batch_if_ready()
            return

        completed = [future for future in list(self._ifc_process_futures.keys()) if future.done()]
        pool_broken = False

        for future in completed:
            meta = self._ifc_process_futures.pop(future, None)
            if meta is None:
                continue

            path = meta[0] if isinstance(meta, tuple) else meta
            kind = meta[1] if isinstance(meta, tuple) and len(meta) > 1 else "file"

            try:
                result = future.result()
                if kind == "chunk":
                    path_str, incoming, source_name, chunk_index, chunk_count = result
                    self._ifc_chunk_finished(Path(path_str), incoming, source_name, chunk_index, chunk_count)
                else:
                    path_str, incoming, source_name, origin_info = result
                    self._ifc_worker_finished(Path(path_str), incoming, source_name, origin_info)

            except concurrent.futures.process.BrokenProcessPool as exc:
                pool_broken = True
                self._load_errors.append(
                    f"{Path(path).name}: IFC worker process crashed while reading geometry. "
                    f"For very large IFCs, reduce concurrent IFC workers or increase RAM. {exc}"
                )
                if kind == "chunk":
                    self._ifc_chunk_error(path, "worker process crashed while reading geometry")
                else:
                    self._ifc_worker_error(path, "worker process crashed while reading geometry")
                break

            except Exception as exc:
                if kind == "chunk":
                    self._ifc_chunk_error(path, str(exc))
                else:
                    self._ifc_worker_error(path, str(exc))

        if pool_broken:
            remaining_meta = list(self._ifc_process_futures.values())
            self._ifc_process_futures.clear()
            seen_paths = set()
            for meta in remaining_meta:
                p = meta[0] if isinstance(meta, tuple) else meta
                key = self._ifc_path_key(p)
                if key in seen_paths or key in self._load_finished_keys:
                    continue
                seen_paths.add(key)
                self._load_errors.append(f"{Path(p).name}: cancelled because the IFC process pool crashed.")
                self._mark_ifc_load_done(p)
            self._shutdown_ifc_process_executor()
            self._update_ifc_load_progress("process pool crashed")
            self._finish_ifc_batch_if_ready()
            return

        if not self._ifc_process_futures:
            self._finish_ifc_batch_if_ready()

    def open_ifc(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open IFC file(s)", "", "IFC files (*.ifc);;All files (*.*)")
        if not paths:
            return
        self.floors = {}
        self.loaded_ifc_paths = []
        self.ifc_origin_info = {}
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
            if getattr(self, "wall_draw_mode", False) or getattr(self, "_wall_draw_start", None) is not None:
                self.cancel_user_wall_drawing()
            if abs(float(getattr(self, "view_rotation_deg", 0.0))) > 1e-9:
                self.reset_view_rotation()
            self.floors = {}
            self.loaded_ifc_paths = []
            self.ifc_origin_info = {}
            self.aps.clear()
            self.last_result = None
            self.ifc_alignment = AlignmentTransform()
        self.alignment_pick_mode: Optional[str] = None
        self.alignment_pick_points: Dict[str, Tuple[float, float]] = {}
        self._alignment_pick_sequence: List[str] = []

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
        self._load_total = len(unique_paths)
        self._load_completed = 0
        self._load_errors = []
        self._load_started_keys = {self._ifc_path_key(p) for p in unique_paths}
        self._load_finished_keys = set()
        self._load_batch_id += 1
        self._loading_replace = replace
        self._loading_active = True
        use_processes = self._ifc_loading_uses_multiprocessing()
        requested_processes = int(getattr(self.heatmap_settings, "max_ifc_loader_processes", 0) or 0)
        available_process_count = self._resolve_ifc_loader_process_count(requested_processes)

        # Do not cap the process pool by the number of IFC files when at least
        # one large file will be split into geometry chunks. The earlier version
        # used ``min(cpu_count, len(unique_paths))`` which meant a single 500 MB
        # IFC could only ever use one process, even though it was later split
        # into many chunk jobs.
        chunk_threshold_mb = float(getattr(self.heatmap_settings, "chunk_ifc_files_over_mb", 100.0) or 0.0)
        enable_chunking = bool(getattr(self.heatmap_settings, "enable_chunked_ifc_geometry_extraction", True))
        huge_threshold_mb = float(getattr(self.heatmap_settings, "huge_ifc_single_process_threshold_mb", 512.0) or 0.0)
        max_parallel_huge = max(1, int(getattr(self.heatmap_settings, "max_parallel_huge_ifc_processes", 1) or 1))
        has_large_chunked_ifc = False
        if use_processes and enable_chunking and chunk_threshold_mb > 0.0:
            for path in unique_paths:
                try:
                    if path.exists() and (path.stat().st_size / (1024.0 * 1024.0)) >= chunk_threshold_mb:
                        has_large_chunked_ifc = True
                        break
                except Exception:
                    pass

        if use_processes:
            # Always create a pool capable of using all configured/logical cores.
            # The previous logic capped the pool to the number of IFC files when
            # a file was not detected as "large" before chunking. That meant
            # loading one or two large/complex models often showed only one or two
            # busy cores in Task Manager. The job submission path below decides
            # how many chunks to submit; the pool should not be artificially capped.
            process_count = max(1, available_process_count)
        else:
            process_count = 1

        if use_processes and huge_threshold_mb > 0.0:
            huge_flags = []
            for p in unique_paths:
                try:
                    huge_flags.append(p.exists() and (p.stat().st_size / (1024.0 * 1024.0)) >= huge_threshold_mb)
                except Exception:
                    huge_flags.append(False)
            if huge_flags and all(huge_flags):
                process_count = min(process_count, max_parallel_huge)
        self._load_worker_label = f"{process_count} process(es)" if use_processes else "single process"
        self._set_loading_ui(True)
        self._show_ifc_load_progress(self._load_total)
        self._update_ifc_load_progress()
        self._schedule_ifc_load_timeout_watchdog()
        self.statusBar().showMessage(
            f"Loading {self._load_pending} IFC file(s) using {self._load_worker_label} on {os.cpu_count() or 1} logical CPU core(s)..."
        )

        base_jobs = [(
            str(path), 0.0, 0.0, 0.0,
            self.heatmap_settings.project_external_walls_across_floors,
            self.heatmap_settings.external_wall_keywords,
        ) for path in unique_paths]

        if use_processes:
            try:
                self._shutdown_ifc_process_executor()
                self._ifc_chunk_remaining = {}
                self._ifc_chunk_total = {}
                self._ifc_chunk_sources = {}
                self._ifc_chunk_had_error = {}
                self._ifc_process_executor = concurrent.futures.ProcessPoolExecutor(max_workers=process_count)
                self._ifc_process_futures = {}
                chunk_size = max(25, int(getattr(self.heatmap_settings, "ifc_geometry_chunk_size", 250) or 250))
                for job in base_jobs:
                    path = Path(job[0])
                    size_mb = (path.stat().st_size / (1024.0 * 1024.0)) if path.exists() else 0.0
                    # Chunk geometry whenever enabled and the file meets the configured
                    # threshold. A threshold of 0 forces chunking for all IFCs.
                    # This makes a single IFC capable of occupying multiple worker
                    # processes instead of only one whole-file worker.
                    is_huge_ifc = bool(huge_threshold_mb > 0.0 and size_mb >= huge_threshold_mb)
                    should_chunk_this_file = (
                        enable_chunking
                        and process_count > 1
                        and path.exists()
                        and size_mb >= max(0.0, chunk_threshold_mb)
                        and not is_huge_ifc
                    )
                    if is_huge_ifc:
                        # Memory-safe huge-model path. Do not split into many
                        # workers because every chunk worker reopens the full IFC.
                        self.statusBar().showMessage(
                            f"{path.name} is {size_mb:.0f} MB; using one isolated IFC process to avoid duplicate model loads..."
                        )
                        fut = self._ifc_process_executor.submit(_load_ifc_file_in_process, job)
                        self._ifc_process_futures[fut] = (path, "file")
                    elif should_chunk_this_file:
                        try:
                            self.statusBar().showMessage(
                                f"Indexing {path.name} before chunked multiprocessing..."
                            )
                            QApplication.processEvents()
                            path_str, storeys, wall_guids, space_guids, source_name, origin_info = _index_ifc_file_for_chunking(job)
                            self.ifc_origin_info[self._ifc_path_key(path)] = origin_info
                            records = [("wall", g) for g in wall_guids] + [("space", g) for g in space_guids]
                            if not records:
                                fut = self._ifc_process_executor.submit(_load_ifc_file_in_process, job)
                                self._ifc_process_futures[fut] = (path, "file")
                                continue
                            chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
                            key = self._ifc_path_key(path)
                            self._ifc_chunk_remaining[key] = len(chunks)
                            self._ifc_chunk_total[key] = len(chunks)
                            self._ifc_chunk_sources[key] = source_name
                            self._ifc_chunk_had_error[key] = False
                            for idx, chunk in enumerate(chunks, start=1):
                                chunk_wall_guids = [g for kind, g in chunk if kind == "wall"]
                                chunk_space_guids = [g for kind, g in chunk if kind == "space"]
                                chunk_job = (
                                    path_str, job[1], job[2], job[3], job[4], job[5],
                                    storeys, chunk_wall_guids, chunk_space_guids, idx, len(chunks),
                                )
                                fut = self._ifc_process_executor.submit(_load_ifc_geometry_chunk_in_process, chunk_job)
                                self._ifc_process_futures[fut] = (path, "chunk")
                            active_jobs = min(process_count, len(chunks))
                            self.statusBar().showMessage(
                                f"Chunked {path.name}: {len(records)} elements across {len(chunks)} geometry jobs "
                                f"using up to {active_jobs}/{process_count} process(es)..."
                            )
                        except Exception as exc:
                            self._load_errors.append(f"{path.name}: chunk index failed ({exc}); falling back to whole-file process loading")
                            fut = self._ifc_process_executor.submit(_load_ifc_file_in_process, job)
                            self._ifc_process_futures[fut] = (path, "file")
                    else:
                        fut = self._ifc_process_executor.submit(_load_ifc_file_in_process, job)
                        self._ifc_process_futures[fut] = (path, "file")
                self._ifc_process_poll_timer.start()
            except Exception as exc:
                self._load_errors.append(f"Process loader failed to start: {exc}. No in-GUI-process IFC fallback was used.")
                for job in base_jobs:
                    self._ifc_worker_error(Path(job[0]), "process loader failed to start")
                return

        if not use_processes:
            # Blocking fallback only when explicitly enabled/required. It is
            # disabled by default because huge IFC native crashes can terminate
            # the GUI process if parsed in-process.
            if not bool(getattr(self.heatmap_settings, "allow_blocking_ifc_fallback", False)):
                for job in base_jobs:
                    self._ifc_worker_error(Path(job[0]), "IFC multiprocessing is disabled and blocking fallback is not allowed")
                return
            for job in base_jobs:
                path = Path(job[0])
                try:
                    path_str, incoming, source_name, origin_info = _load_ifc_file_in_process(job)
                    self._ifc_worker_finished(Path(path_str), incoming, source_name, origin_info)
                except Exception as exc:
                    self._ifc_worker_error(path, str(exc))

    def _ifc_chunk_finished(self, path_obj, incoming, source_name: str, chunk_index: int, chunk_count: int):
        path = Path(path_obj)
        key = self._ifc_path_key(path)
        if key in self._load_finished_keys:
            return
        CombinedIFCModel.merge(self.floors, incoming, source_name)
        remaining = max(0, int(self._ifc_chunk_remaining.get(key, 1)) - 1)
        self._ifc_chunk_remaining[key] = remaining
        done = int(self._ifc_chunk_total.get(key, chunk_count)) - remaining
        self.statusBar().showMessage(
            f"Loaded geometry chunk {done}/{self._ifc_chunk_total.get(key, chunk_count)} for {path.name}."
        )
        if remaining <= 0:
            if self._mark_ifc_load_done(path):
                self._apply_frequency_settings_to_model(replace_existing=False)
                self.loaded_ifc_paths.append(path)
                self._update_ifc_load_progress(path.name)
                self.statusBar().showMessage(
                    f"Loaded {path.name} from {self._ifc_chunk_total.get(key, chunk_count)} geometry chunks. "
                    f"Waiting for {self._load_pending} IFC file(s)..."
                )
            self._finish_ifc_batch_if_ready()

    def _ifc_chunk_error(self, path_obj, message: str):
        path = Path(path_obj)
        key = self._ifc_path_key(path)
        if key in self._load_finished_keys:
            return
        self._ifc_chunk_had_error[key] = True
        self._load_errors.append(f"{path.name}: geometry chunk failed: {message}")
        remaining = max(0, int(self._ifc_chunk_remaining.get(key, 1)) - 1)
        self._ifc_chunk_remaining[key] = remaining
        if remaining <= 0:
            if self._mark_ifc_load_done(path):
                # Keep any chunks that did load, but report the file as partial.
                if any(f.walls or f.spaces for f in self.floors.values()):
                    self.loaded_ifc_paths.append(path)
                self._update_ifc_load_progress(path.name)
            self._finish_ifc_batch_if_ready()

    @Slot(object, object, str, object)
    def _ifc_worker_finished(self, path_obj, incoming, source_name: str, origin_info=None):

        # Defensive IFC chunk-state guard
        if not hasattr(self, "_ifc_chunk_remaining") or not isinstance(self._ifc_chunk_remaining, dict):
            self._ifc_chunk_remaining = {}
        if not hasattr(self, "_ifc_chunk_total") or not isinstance(self._ifc_chunk_total, dict):
            self._ifc_chunk_total = {}
        if not hasattr(self, "_ifc_chunk_results"):
            self._ifc_chunk_results = []
        if not hasattr(self, "_ifc_chunk_errors"):
            self._ifc_chunk_errors = []
        if not hasattr(self, "_ifc_chunk_futures"):
            self._ifc_chunk_futures = []
        path = Path(path_obj)
        if not self._mark_ifc_load_done(path):
            return
        CombinedIFCModel.merge(self.floors, incoming, source_name)
        if isinstance(origin_info, dict):
            self.ifc_origin_info[self._ifc_path_key(path)] = origin_info
        self._apply_frequency_settings_to_model(replace_existing=False)
        self.loaded_ifc_paths.append(path)
        total_loaded = len(self.loaded_ifc_paths)
        self._update_ifc_load_progress(path.name)
        self.statusBar().showMessage(f"Loaded {path.name}. Waiting for {self._load_pending} IFC file(s)... Total loaded: {total_loaded}")
        self._finish_ifc_batch_if_ready()

    @Slot(object, str)
    def _ifc_worker_error(self, path_obj, message: str):
        path = Path(path_obj)
        if not self._mark_ifc_load_done(path):
            return
        self._load_errors.append(f"{path.name}: {message}")
        self._update_ifc_load_progress(path.name)
        self.statusBar().showMessage(f"Failed to load {path.name}. Waiting for {self._load_pending} IFC file(s)...")
        self._finish_ifc_batch_if_ready()

    def _finish_ifc_batch_if_ready(self):

        # Defensive IFC chunk-state guard
        if not hasattr(self, "_ifc_chunk_remaining") or not isinstance(self._ifc_chunk_remaining, dict):
            self._ifc_chunk_remaining = {}
        if not hasattr(self, "_ifc_chunk_total") or not isinstance(self._ifc_chunk_total, dict):
            self._ifc_chunk_total = {}
        if not hasattr(self, "_ifc_chunk_results"):
            self._ifc_chunk_results = []
        if not hasattr(self, "_ifc_chunk_errors"):
            self._ifc_chunk_errors = []
        if not hasattr(self, "_ifc_chunk_futures"):
            self._ifc_chunk_futures = []
        if self._load_pending > 0:
            return
        self._loading_active = False
        self._set_loading_ui(False)
        self._close_ifc_load_progress()
        self._shutdown_ifc_process_executor()
        self._refresh_floor_combo()
        total_walls = sum(len(f.walls) for f in self.floors.values())
        total_spaces = sum(len(f.spaces) for f in self.floors.values())
        msg = f"Loaded {len(self.loaded_ifc_paths)} IFC file(s), {len(self.floors)} floor(s), {total_walls} wall(s), {total_spaces} space(s)"
        if self._load_errors:
            QMessageBox.warning(self, "Some IFC files failed", "\n".join(self._load_errors))
            msg += f". {len(self._load_errors)} file(s) failed."
        self.statusBar().showMessage(msg)
        if self._pending_plan_data is not None:
            pending = self._pending_plan_data
            self._pending_plan_data = None
            self._apply_rf_plan_data(pending)

    def _ensure_ifc_loader_state(self):
        """Ensure multiprocessing IFC state exists before any callback uses it."""
        if not hasattr(self, '_ifc_chunk_remaining') or not isinstance(self._ifc_chunk_remaining, dict):
            self._ifc_chunk_remaining = {}
        if not hasattr(self, '_ifc_chunk_total') or not isinstance(self._ifc_chunk_total, dict):
            self._ifc_chunk_total = {}
        if not hasattr(self, '_ifc_chunk_results'):
            self._ifc_chunk_results = []
        if not hasattr(self, '_ifc_batch_pending'):
            self._ifc_batch_pending = set()
        if not hasattr(self, '_ifc_batch_completed'):
            self._ifc_batch_completed = set()
        if not hasattr(self, '_ifc_batch_failed'):
            self._ifc_batch_failed = []
        if not hasattr(self, '_ifc_process_futures'):
            self._ifc_process_futures = []
        if not hasattr(self, '_ifc_process_executor'):
            self._ifc_process_executor = None

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
        self.open_dxf_action.setEnabled(not loading)
        self.align_ifc_action.setEnabled(not loading)
        self.clear_dxf_action.setEnabled(not loading)
        self.sim_action.setEnabled(not loading)
        self.export_action.setEnabled(not loading)
        self.clear_ap_action.setEnabled(not loading)
        self.load_pattern_action.setEnabled(not loading)
        for action_name in (
            "planner_settings_action", "predict_aps_action", "draw_wall_action", "ifc_origin_action",
            "rotate_left_action", "rotate_right_action", "reset_rotation_action", "save_plan_action", "load_plan_action",
        ):
            action = getattr(self, action_name, None)
            if action is not None:
                action.setEnabled(not loading)
        self.floor_combo.setEnabled(not loading)

    def select_floor(self, name: str):
        if getattr(self, "wall_draw_mode", False) or getattr(self, "_wall_draw_start", None) is not None:
            self.cancel_user_wall_drawing()
        # Avoid calling ``self.floors.get(...)`` directly. A previous settings/UI
        # change could accidentally shadow a ``get`` attribute with a numeric
        # value, which causes: TypeError: 'float' object is not callable.
        # Treat ``self.floors`` strictly as a mapping and use the dict type
        # implementation where possible.
        if not isinstance(self.floors, dict):
            QMessageBox.critical(
                self,
                "Internal model error",
                f"Expected self.floors to be a dict, got {type(self.floors).__name__}."
            )
            self.floor = None
            return
        self.floor = dict.get(self.floors, str(name), None)
        self.last_result = None
        self._load_slab_attenuation_to_ui()
        self.draw_floor()
        self.populate_ap_table()
        self.populate_wall_table()

    def add_ap(self, x: float, y: float):
        if not self.floor:
            return
        default_radios = []
        for idx, radio_def in enumerate(self.heatmap_settings.default_ap_radios or []):
            # Settings should provide each radio as a dict. If an older/edited
            # settings file has a bare number or other value, skip it instead of
            # trying to call ``radio_def.get(...)`` on a float.
            if not isinstance(radio_def, dict):
                continue
            get_value = dict.get
            pattern = str(get_value(radio_def, "antenna_pattern", self.pattern_combo.currentText()))
            if pattern not in self.antenna_patterns:
                pattern = self.pattern_combo.currentText()
            default_radios.append(APRadio(
                name=str(get_value(radio_def, "name", f"Radio-{idx + 1}")),
                frequency_mhz=float(get_value(radio_def, "frequency_mhz", self.freq.value())),
                tx_power_dbm=float(get_value(radio_def, "tx_power_dbm", self.tx_power.value())),
                antenna_pattern=pattern,
                enabled=bool(get_value(radio_def, "enabled", True)),
                cutoff_radius_m=float(get_value(radio_def, "cutoff_radius_m", 0.0)),
                antenna_gain_dbi=float(get_value(radio_def, "antenna_gain_dbi", 0.0)),
                channel=str(get_value(radio_def, "channel", "")),
                channel_width_mhz=float(get_value(radio_def, "channel_width_mhz", 20.0)),
                spectrum_occupancy_percent=float(get_value(radio_def, "spectrum_occupancy_percent", 0.0)),
            ))
        if not default_radios:
            default_radios = [APRadio(
                name="Radio-1",
                frequency_mhz=float(self.freq.value()),
                tx_power_dbm=float(self.tx_power.value()),
                antenna_pattern=self.pattern_combo.currentText(),
                enabled=True,
            )]
        first_radio = default_radios[0]
        ap = AccessPoint(
            name=self._next_ap_name(),
            x=x,
            y=y,
            floor=self.floor.name,
            tx_power_dbm=float(first_radio.tx_power_dbm),
            frequency_mhz=float(first_radio.frequency_mhz),
            path_loss_exponent=float(self.ple.value()),
            antenna_pattern=first_radio.antenna_pattern,
            azimuth_deg=float(self.azimuth.value()),
            downtilt_deg=float(self.downtilt.value()),
            mount_height_m=float(self.mount_height.value()),
            rx_height_m=float(self.rx_height.value()),
            radios=default_radios,
            max_clients=int(self.auto_planner_settings.clients_per_ap),
            planned=False,
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
        self._ifc_snap_marker_items = []
        self._wall_preview_items = []
        if not self.floor:
            return
        # Draw heatmap first so the building geometry remains visible above it.
        if self.last_result:
            self._draw_heatmap(self.last_result)

        colours = self._theme_colours()
        self._draw_dxf_overlay(scene, colours)
        # Draw spaces/floor areas with a stronger outline and subtle fill so the
        # extent of the floor remains readable even before a simulation is run.
        for space in self.floor.spaces:
            coords = list(space.polygon.exterior.coords)
            poly = QPolygonF([QPointF(x, y) for x, y in coords])
            item = QGraphicsPolygonItem(poly)
            pen = QPen(colours["space_pen"], 0.12)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QBrush(colours["space_fill"]))
            item.setZValue(Z_IFC_SPACE_FILL)
            scene.addItem(item)

            if space.name:
                centroid = space.polygon.representative_point()
                cx, cy = self._point_xy(centroid)
                self._add_upright_text(
                    scene, str(space.name), cx, cy,
                    colours["space_text"], self.heatmap_settings.space_label_font_size, Z_TEXT, bold=True
                )

        for wall in self.floor.walls:
            coords = list(wall.polygon.exterior.coords)
            poly = QPolygonF([QPointF(x, y) for x, y in coords])
            wall_pen_colour = QColor("#FF7800") if wall.is_user_created else colours["wall_pen"]
            wall_pen = QPen(wall_pen_colour, self.heatmap_settings.wall_line_width)
            wall_pen.setCosmetic(True)
            fill_key = "wall_alt_fill" if getattr(wall, "projected_to_floor", False) else "wall_fill"
            fill_colour = QColor("#FFB45A") if wall.is_user_created else QColor(colours[fill_key])
            if wall.is_user_created:
                fill_colour.setAlpha(150)
            item = WallGraphicsItem(self, wall, poly, wall_pen, QBrush(fill_colour))
            scene.addItem(item)
        visible_aps = [a for a in self.aps if a.floor == self.floor.name or self.include_inter_floor.isChecked()]
        for ap in visible_aps:
            same_floor = ap.floor == self.floor.name
            radius = 0.75 if same_floor else 0.45
            colour = colours["ap_same_floor"] if same_floor else colours["ap_other_floor"]
            if self.heatmap_settings.show_ap_cutoff_zones and same_floor:
                active_radii = []
                for radio in ap.active_radios():
                    r = RFEngine.cutoff_radius_m_for_radio(radio, self.heatmap_settings)
                    if r > 0:
                        active_radii.append(r)
                for r in sorted(set(round(v, 3) for v in active_radii)):
                    cut_colour = QColor(colours.get("ap_cutoff_zone", colour))
                    cut_colour.setAlpha(self.heatmap_settings.ap_cutoff_zone_alpha)
                    cut_pen = QPen(cut_colour, self.heatmap_settings.ap_cutoff_zone_line_width)
                    cut_pen.setCosmetic(True)
                    cut_item = QGraphicsEllipseItem(ap.x - r, ap.y - r, r * 2.0, r * 2.0)
                    cut_item.setPen(cut_pen)
                    cut_item.setBrush(QBrush(Qt.NoBrush))
                    cut_item.setZValue(Z_CONTOUR_LINE - 2)
                    scene.addItem(cut_item)

            if same_floor:
                dot = AccessPointGraphicsItem(self, ap, radius, colour)
                scene.addItem(dot)
            else:
                dot = QGraphicsEllipseItem(ap.x - radius, ap.y - radius, radius * 2.0, radius * 2.0)
                dot.setBrush(QBrush(colour))
                dot.setPen(QPen(colours["ap_outline"], 0.2))
                dot.setZValue(Z_AP - 5)
                scene.addItem(dot)

            # Draw a short boresight arrow so directional antenna orientation can be checked.
            length = 5.0 if same_floor else 3.0
            ang = math.radians(ap.azimuth_deg)
            x2 = ap.x + length * math.cos(ang)
            y2 = ap.y + length * math.sin(ang)
            arrow = scene.addLine(ap.x, ap.y, x2, y2, QPen(colour, 0.25))
            arrow.setZValue(Z_AP)
            if not same_floor:
                self._add_upright_text(scene, ap.floor, ap.x + 0.8, ap.y + 0.8, colour, self.heatmap_settings.ap_label_font_size, Z_AP_LABEL)

        old_transform = self.view.transform()
        old_h = self.view.horizontalScrollBar().value()
        old_v = self.view.verticalScrollBar().value()

        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-10, -10, 10, 10))

        if getattr(self, "_preserve_view_on_redraw", False) or getattr(self, "_view_has_been_fitted", False):
            self.view.setTransform(old_transform)
            self.view.horizontalScrollBar().setValue(old_h)
            self.view.verticalScrollBar().setValue(old_v)
        else:
            self.view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self._view_has_been_fitted = True

    def _draw_heatmap(self, result: SimulationResult):
        """Draw smooth filled RSSI contours, isolines and sampled RSSI points.

        The filled colour now follows interpolated contour polygons rather than
        square grid cells. scipy is used, when available, to upsample the RSSI
        grid before contourpy generates the filled bands and contour lines.
        Text is drawn as screen-sized upright text so it does not become huge or
        upside down when the IFC view is zoomed/flipped.
        """
        scene = self.view.scene()
        colours = self._theme_colours()
        if len(result.xs) < 2 or len(result.ys) < 2:
            return
        if contourpy is None:
            QMessageBox.warning(self, "Missing dependency", "contourpy is required for smooth filled contours. Run: pip install contourpy")
            return

        grid = np.asarray(result.rssi, dtype=float)
        rows, cols = grid.shape
        if rows < 2 or cols < 2:
            return

        factor = max(1, int(getattr(self.heatmap_settings, "contour_interpolation_factor", 4)))
        xs = np.asarray(result.xs, dtype=float)
        ys = np.asarray(result.ys, dtype=float)
        z = grid
        if factor > 1:
            fine_xs = np.linspace(float(xs[0]), float(xs[-1]), (len(xs) - 1) * factor + 1)
            fine_ys = np.linspace(float(ys[0]), float(ys[-1]), (len(ys) - 1) * factor + 1)
            if scipy_zoom is not None:
                z = scipy_zoom(grid, (factor, factor), order=3)
                # scipy_zoom can produce one or two extra samples depending on shape.
                z = z[:len(fine_ys), :len(fine_xs)]
                if z.shape != (len(fine_ys), len(fine_xs)):
                    z = np.resize(z, (len(fine_ys), len(fine_xs)))
            else:
                # Fallback: two-pass linear interpolation using numpy only.
                temp = np.vstack([np.interp(fine_xs, xs, row) for row in grid])
                z = np.vstack([np.interp(fine_ys, ys, temp[:, ix]) for ix in range(temp.shape[1])]).T
            xs, ys = fine_xs, fine_ys

        levels = sorted({float(v) for v in self.heatmap_settings.isoline_bands_dbm}, reverse=True)
        if not levels:
            return

        finite = z[np.isfinite(z)]
        if finite.size == 0:
            return
        data_min = float(np.nanmin(finite))
        data_max = float(np.nanmax(finite))
        fill_low = min(data_min, min(levels)) - 0.1
        fill_high = max(data_max, max(levels)) + 0.1

        try:
            cg = contourpy.contour_generator(
                x=xs,
                y=ys,
                z=z,
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
            item.setZValue(Z_HEATMAP_FILL)
            scene.addItem(item)

        bounds = [fill_low] + sorted(levels) + [fill_high]
        for lower, upper in zip(bounds[:-1], bounds[1:]):
            add_filled_band(lower, upper, (lower + upper) / 2.0)

        contour_font_size = max(1, int(self.heatmap_settings.contour_label_font_size))
        contour_label_limit = 3
        contour_min_spacing = 35.0
        line_width = max(0.01, float(getattr(self.heatmap_settings, "contour_line_width", 1.25)))
        contour_line_cosmetic = bool(getattr(self.heatmap_settings, "contour_line_cosmetic", True))
        # Cosmetic pens use screen pixels. Values below 1 px can disappear on
        # some Qt backends, which made the contour lines look as though they
        # were hidden even when their z-order was correct.
        if contour_line_cosmetic:
            line_width = max(1.0, line_width)

        for level in levels:
            if level < data_min or level > data_max:
                continue
            line_colour = self.heatmap_settings.contour_line_qcolour(level, bool(getattr(self, "dark_theme", False)))
            pen = QPen(line_colour, line_width)
            pen.setCosmetic(contour_line_cosmetic)
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
                item.setZValue(Z_CONTOUR_LINE)
                scene.addItem(item)

                if labels_on_level >= contour_label_limit:
                    continue
                seg = np.diff(pts, axis=0)
                seg_len = np.hypot(seg[:, 0], seg[:, 1])
                total_len = float(np.sum(seg_len))
                if total_len < 6.0:
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
                self._add_upright_text(
                    scene, f"{level:.0f} dBm", x, y, colours["contour_text"],
                    contour_font_size, Z_TEXT, bold=True, rotation_deg=angle
                )
                label_positions.append((x, y))
                labels_on_level += 1

        # Small blue + markers showing the original sample locations only.
        stride_x = max(1, int(self.heatmap_settings.sample_stride_x))
        stride_y = max(1, int(self.heatmap_settings.sample_stride_y))
        sample_font_size = max(1, int(self.heatmap_settings.sample_label_font_size))
        cross_size = max(0.01, float(self.heatmap_settings.sample_cross_size))
        sample_colour = colours["sample_text"]
        sample_cross_colour = colours["sample_cross"]
        sample_pen = QPen(sample_cross_colour, max(1.0, float(getattr(self.heatmap_settings, "sample_cross_line_width", 1.0))))
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
                h.setZValue(Z_SAMPLE_MARK)
                vline.setZValue(Z_SAMPLE_MARK)
                self._add_upright_text(
                    scene, f"{rssi:.0f} dBm", x + cross_size * 1.8, y + cross_size * 1.2,
                    sample_colour, sample_font_size, Z_TEXT
                )

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

        colours = self._theme_colours()
        band_text = ", ".join(f"{v:.0f}" for v in self.heatmap_settings.isoline_bands_dbm)
        pattern_count = len(self.heatmap_settings.rf_pattern_files)
        legend_text = colours["legend_text"].name()
        legend_border = colours["legend_border"].name()
        pieces = [
            f"<b>RSSI isolines</b> &nbsp; "
            f"<span style='color:{legend_text};'>Bands: {band_text} dBm. "
            f"Clients disconnect below {self.heatmap_settings.minimum_client_rssi_dbm:.0f} dBm. <br/> "
            f"Pattern files in settings: {pattern_count}</span>"
        ]
        for zone in self.heatmap_settings.zones:
            colour = QColor(zone.colour)
            if not colour.isValid():
                colour = QColor("#555555")
            fg = "#ffffff" if colour.lightness() < 130 else "#202020"
            pieces.append(
                "<span style='display:inline-block; margin-left:10px; "
                f"padding:3px 7px; border:1px solid {legend_border}; background:{colour.name()}; color:{fg};'>"
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
            self.auto_planner_settings = AutoPlannerSettings.from_dict(self.heatmap_settings.auto_planner_settings)
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
            self._apply_frequency_settings_to_model(replace_existing=False)
            self._refresh_rssi_frequency_dropdown()
            self.populate_ap_table()
            self.populate_wall_table()
            self._update_rssi_legend()
            self.draw_floor()
            extra = f"\nLoaded RF pattern files: {', '.join(loaded_patterns)}" if loaded_patterns else ""
            QMessageBox.information(self, "Heatmap settings loaded", f"Loaded RSSI isoline settings from {Path(path).name}{extra}")
        except Exception as exc:
            QMessageBox.warning(self, "Heatmap settings failed", str(exc))

    def _frequency_bands(self) -> List[float]:
        values = {float(v) for v in self.heatmap_settings.common_frequencies_mhz}
        planner = getattr(self, "auto_planner_settings", None)
        if planner is not None:
            values.update(float(r.frequency_mhz) for r in planner.radio_requirements)
        values.update(float(r.frequency_mhz) for ap in getattr(self, "aps", []) for r in ap.active_radios())
        return sorted(values)

    def _frequency_label(self, mhz: float) -> str:
        mhz = float(mhz)
        if mhz >= 1000.0:
            return f"{mhz / 1000.0:g} GHz dB"
        return f"{mhz:g} MHz dB"

    def _configure_wall_table_headers(self):
        bands = self._frequency_bands() if hasattr(self, "heatmap_settings") else [2400.0, 5000.0, 6000.0]
        headers = ["Wall/material/type"] + [self._frequency_label(b) for b in bands] + ["Name", "Source IFC", "GUID"]
        self.wall_table.setColumnCount(len(headers))
        self.wall_table.setHorizontalHeaderLabels(headers)

    def _profile_for_wall_from_settings(self, wall: Wall2D) -> Dict[float, float]:
        # Prefer the user-selected RF type when a wall has been overridden so
        # newly added planning frequencies inherit the intended RF material.
        override = wall.rf_type_override if wall.rf_customised else ""
        text = f"{override} {wall.material} {wall.type_name} {wall.name}".lower()
        profiles = self.heatmap_settings.default_wall_attenuation_by_material_db
        for key, profile in profiles.items():
            if key != "default" and key.lower() in text:
                return dict(profile)
        return dict(profiles.get("default", {}))

    def _apply_frequency_settings_to_model(self, replace_existing: bool = False):
        """Ensure loaded floors/walls contain attenuation values for all configured frequencies."""
        bands = self._frequency_bands()
        floor_profile = self.heatmap_settings.default_floor_attenuation_by_frequency_db
        for floor in self.floors.values():
            for band in bands:
                if replace_existing or band not in floor.slab_attenuation_by_band_db:
                    floor.slab_attenuation_by_band_db[band] = float(floor_profile.get(band, floor.slab_attenuation_db_for_frequency(band)))
            for wall in floor.walls:
                profile = self._profile_for_wall_from_settings(wall)
                for band in bands:
                    if replace_existing or band not in wall.attenuation_by_band_db:
                        wall.attenuation_by_band_db[band] = float(profile.get(band, wall.attenuation_db_for_frequency(band)))
        self._configure_wall_table_headers()

    def populate_ap_table(self):
        """Show one row per AP radio so a single AP can model multiple bands."""
        self.ap_table.blockSignals(True)
        self.ap_table.setRowCount(0)
        if not self.floor:
            self.ap_table.blockSignals(False)
            return
        for ap in [a for a in self.aps if a.floor == self.floor.name]:
            if not ap.radios:
                ap.radios = [APRadio(
                    name="Radio-1",
                    frequency_mhz=float(ap.frequency_mhz),
                    tx_power_dbm=float(ap.tx_power_dbm),
                    antenna_pattern=ap.antenna_pattern,
                    enabled=True,
                )]
            for radio_index, radio in enumerate(ap.radios):
                row = self.ap_table.rowCount()
                self.ap_table.insertRow(row)
                values = [
                    ap.name,
                    radio.name,
                    "Yes" if radio.enabled else "No",
                    ap.floor,
                    f"{ap.x:.2f}",
                    f"{ap.y:.2f}",
                    radio.antenna_pattern,
                    f"{ap.azimuth_deg:.1f}",
                    f"{ap.downtilt_deg:.1f}",
                    f"{radio.tx_power_dbm:.1f}",
                    f"{radio.antenna_gain_dbi:.1f}",
                    f"{radio.frequency_mhz:.0f}",
                    str(radio.channel),
                    f"{radio.channel_width_mhz:g}",
                    f"{radio.spectrum_occupancy_percent:g}",
                    str(ap.max_clients),
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.UserRole, ap.name)
                    item.setData(Qt.UserRole + 1, radio_index)
                    self.ap_table.setItem(row, col, item)
        self.ap_table.resizeColumnsToContents()
        self.ap_table.blockSignals(False)

    def _ap_table_changed(self, item: QTableWidgetItem):
        if not self.floor:
            return
        ap_name = item.data(Qt.UserRole)
        radio_index = item.data(Qt.UserRole + 1)
        ap = next((a for a in self.aps if a.name == ap_name), None)
        if not ap:
            return
        if not ap.radios:
            ap.radios = [APRadio(
                name="Radio-1",
                frequency_mhz=float(ap.frequency_mhz),
                tx_power_dbm=float(ap.tx_power_dbm),
                antenna_pattern=ap.antenna_pattern,
                enabled=True,
            )]
        try:
            radio_index = int(radio_index or 0)
            radio = ap.radios[radio_index]
            if item.column() == 1:
                radio.name = item.text().strip() or radio.name
            elif item.column() == 2:
                radio.enabled = item.text().strip().lower() in {"yes", "y", "true", "1", "on", "enabled"}
            elif item.column() == 6:
                if item.text() in self.antenna_patterns:
                    radio.antenna_pattern = item.text()
            elif item.column() == 7:
                ap.azimuth_deg = float(item.text())
            elif item.column() == 8:
                ap.downtilt_deg = float(item.text())
            elif item.column() == 9:
                radio.tx_power_dbm = float(item.text())
            elif item.column() == 10:
                radio.antenna_gain_dbi = float(item.text())
            elif item.column() == 11:
                radio.frequency_mhz = float(item.text())
                self._refresh_rssi_frequency_dropdown()
            elif item.column() == 12:
                radio.channel = item.text().strip()
            elif item.column() == 13:
                radio.channel_width_mhz = max(1.0, float(item.text()))
            elif item.column() == 14:
                radio.spectrum_occupancy_percent = max(0.0, min(100.0, float(item.text())))
            elif item.column() == 15:
                ap.max_clients = max(1, int(float(item.text())))
            # Keep legacy AP fields in sync with the first radio for older code/export.
            if ap.radios:
                ap.tx_power_dbm = float(ap.radios[0].tx_power_dbm)
                ap.frequency_mhz = float(ap.radios[0].frequency_mhz)
                ap.antenna_pattern = ap.radios[0].antenna_pattern
        except (ValueError, IndexError):
            return
        self.last_result = None
        self.draw_floor()

    def populate_wall_table(self):
        self.wall_table.blockSignals(True)
        self._configure_wall_table_headers()
        self.wall_table.setRowCount(0)
        if not self.floor:
            self.wall_table.blockSignals(False)
            return
        bands = self._frequency_bands()
        meta_start = 1 + len(bands)
        for wall in self.floor.walls:
            row = self.wall_table.rowCount()
            self.wall_table.insertRow(row)
            label_item = QTableWidgetItem(("[User] " if wall.is_user_created else "") + wall.label)
            label_item.setData(Qt.UserRole, wall.guid)
            label_item.setData(Qt.UserRole + 2, wall.source_file)
            self.wall_table.setItem(row, 0, label_item)
            for idx, band in enumerate(bands, start=1):
                att = QTableWidgetItem(str(round(float(wall.attenuation_by_band_db.get(band, wall.attenuation_db_for_frequency(band))), 3)))
                att.setData(Qt.UserRole, wall.guid)
                att.setData(Qt.UserRole + 1, band)
                self.wall_table.setItem(row, idx, att)
            self.wall_table.setItem(row, meta_start, QTableWidgetItem(wall.name))
            self.wall_table.setItem(row, meta_start + 1, QTableWidgetItem(wall.source_file))
            self.wall_table.setItem(row, meta_start + 2, QTableWidgetItem(wall.guid))
        self.wall_table.resizeColumnsToContents()
        self.wall_table.blockSignals(False)

    def _wall_table_changed(self, item: QTableWidgetItem):
        if not self.floor or item.column() < 1 or item.column() > len(self._frequency_bands()):
            return
        guid = item.data(Qt.UserRole)
        band = item.data(Qt.UserRole + 1)
        try:
            val = float(item.text())
        except ValueError:
            return
        wall = next((candidate for candidate in self.floor.walls if candidate.guid == guid), None)
        if wall is None:
            return
        for instance in self._wall_instances(wall):
            instance.attenuation_by_band_db[float(band)] = val
            instance.rf_customised = True
        self.last_result = None

    def _sync_slab_attenuation_from_ui(self):
        if not self.floor:
            return
        for band in self._frequency_bands():
            self.floor.slab_attenuation_by_band_db.setdefault(
                band,
                float(self.heatmap_settings.default_floor_attenuation_by_frequency_db.get(band, 0.0)),
            )
        self.floor.slab_attenuation_by_band_db[2400.0] = float(self.slab_att_24.value())
        self.floor.slab_attenuation_by_band_db[5000.0] = float(self.slab_att_5.value())
        self.floor.slab_attenuation_by_band_db[6000.0] = float(self.slab_att_6.value())

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

        progress = QProgressDialog("Calculating RSSI heatmap...", "Cancel", 0, 100, self)
        progress.setWindowTitle("RSSI calculation")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        def update_progress(done, total):
            if total <= 0:
                return
            progress.setValue(int((done / total) * 100))
            QApplication.processEvents()
            if progress.wasCanceled():
                raise RuntimeError("RSSI calculation cancelled")

        self.rssi_results_by_frequency = {}

        active_freqs = sorted(
            {float(r.frequency_mhz) for ap in self.aps for r in ap.active_radios()}
        )

        if not active_freqs:
            QMessageBox.information(
                self,
                "No active radios",
                "No enabled AP radios are available for RSSI calculation.",
            )
            return

        original_radios = [(ap, list(ap.radios)) for ap in self.aps]

        try:
            for i, freq in enumerate(active_freqs, start=1):
                for ap in self.aps:
                    ap.radios = [
                        r
                        for r in original_radios[[a for a, _ in original_radios].index(ap)][1]
                        if getattr(r, "enabled", True)
                        and abs(float(r.frequency_mhz) - freq) < 1e-6
                    ]

                if not any(ap.radios for ap in self.aps):
                    continue

                self.statusBar().showMessage(
                    f"Calculating RSSI for {freq:g} MHz ({i}/{len(active_freqs)})..."
                )

                result = RFEngine.simulate(
                    self.floor,
                    self.floors,
                    self.aps,
                    self.resolution.value(),
                    self.antenna_patterns,
                    include_inter_floor=self.include_inter_floor.isChecked(),
                    heatmap_settings=self.heatmap_settings,
                    progress_callback=update_progress,
                )

                if result is not None:
                    self.rssi_results_by_frequency[freq] = result

        finally:
            for ap, radios in original_radios:
                ap.radios = radios

        selected_freq = self._selected_rssi_view_frequency()

        if selected_freq in self.rssi_results_by_frequency:
            self.last_result = self.rssi_results_by_frequency[selected_freq]
        elif self.rssi_results_by_frequency:
            first_freq = sorted(self.rssi_results_by_frequency.keys())[0]
            self.last_result = self.rssi_results_by_frequency[first_freq]

            idx = self.rssi_view_frequency.findData(first_freq)
            if idx >= 0:
                self.rssi_view_frequency.blockSignals(True)
                self.rssi_view_frequency.setCurrentIndex(idx)
                self.rssi_view_frequency.blockSignals(False)
        else:
            self.last_result = None

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
            writer.writerow(["floor", "x", "y", "simulation_mode", "active_radio_frequencies_mhz", "rssi_dbm", "rssi_zone", "client_connected", "disconnect_threshold_dbm", "ap_count", "include_inter_floor", "slab_loss_24_db", "slab_loss_5_db", "slab_loss_6_db", "patterns_used"])
            for iy, y in enumerate(self.last_result.ys):
                for ix, x in enumerate(self.last_result.xs):
                    rssi = float(self.last_result.rssi[iy, ix])
                    zone = self.heatmap_settings.zone_for_rssi(rssi)
                    active_radios = [r for a in self.aps for r in a.active_radios()]
                    writer.writerow([self.floor.name if self.floor else "", x, y, "best_active_radio", ";".join(str(int(r.frequency_mhz)) for r in active_radios), rssi, zone.name, rssi >= self.heatmap_settings.minimum_client_rssi_dbm, self.heatmap_settings.minimum_client_rssi_dbm, len(self.aps), self.include_inter_floor.isChecked(), float(self.slab_att_24.value()), float(self.slab_att_5.value()), float(self.slab_att_6.value()), ";".join(sorted({r.antenna_pattern for r in active_radios}))])


def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
