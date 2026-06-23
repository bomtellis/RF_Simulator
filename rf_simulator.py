"""
RF Attenuation Simulator - IFC Wi-Fi RSSI planning tool.

Run:
    pip install PySide6 numpy ifcopenshell shapely contourpy
    pip install numba  # optional compiled RF kernels
    python rf_simulator.py
"""
from __future__ import annotations

import atexit
import copy
import csv
import cmath
import concurrent.futures
import functools
import json
import math
import multiprocessing
import os
import pickle
import tempfile
import threading
import uuid
import time
from rf_dxf_prealign import DxfPreAlignDialog, SimilarityTransform2D, two_point_transform
from rf_boundary_tools import (
    estimate_outer_wall_gap_tolerance,
    estimate_space_gap_tolerance,
    infer_space_polygons,
    missing_external_wall_polygons,
    suggest_external_boundary_polygons,
)
from rf_propagation import (
    PathPower,
    ReflectionIndex,
    ReflectionSurface,
    deterministic_spatial_fading_db,
    generate_diffraction_geometries,
    generate_reflection_geometries,
    precompute_reflection_sequences,
    evaluate_diffraction_geometry,
    evaluate_reflection_geometry,
    incoherent_power_dbm,
    propagation_phase_rad,
    stable_link_key,
)
from rf_performance import (
    BoundedLRU,
    SharedArraySpec,
    adaptive_refinement_mask,
    attach_shared_array,
    best_case_rssi_grid,
    coherent_path_metrics,
    create_shared_array,
    nearest_resample_regular_grid,
    resample_regular_grid,
    stable_digest,
)
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    from scipy.ndimage import zoom as scipy_zoom
except Exception:
    scipy_zoom = None
try:
    import contourpy
except Exception:
    contourpy = None

from PySide6.QtCore import QPointF, Qt, Slot, QTimer, QSize, QRectF, QMarginsF
from PySide6.QtGui import (
    QAction, QColor, QBrush, QFont, QPen, QPolygonF, QPainterPath, QPalette,
    QTransform, QIcon, QKeySequence, QPainter, QPdfWriter, QPageSize, QPageLayout,
    QImage, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
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
    from shapely.ops import nearest_points, unary_union
    from shapely.affinity import affine_transform, rotate as shapely_rotate
    from shapely.prepared import prep as prepare_geometry
    try:
        from shapely import intersects_xy as shapely_intersects_xy
    except Exception:  # Shapely < 2.0
        shapely_intersects_xy = None
    try:
        from shapely import concave_hull as shapely_concave_hull
    except Exception:  # Shapely < 2.0
        shapely_concave_hull = None
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install shapely: pip install shapely") from exc


# ----------------------------- Reusable RF process pool -----------------------------

_RF_PROCESS_EXECUTOR: Optional[concurrent.futures.ProcessPoolExecutor] = None
_RF_PROCESS_EXECUTOR_WORKERS = 0
_RF_PROCESS_EXECUTOR_LOCK = threading.Lock()
_RF_WORKER_CONTEXT_CACHE: Dict[str, tuple] = {}
_RF_WORKER_CONTEXT_ORDER: List[str] = []
_RF_CONTEXT_FILE_CACHE: Dict[str, str] = {}
_RF_CONTEXT_FILE_ORDER: List[str] = []
_RF_CONTEXT_FILE_LOCK = threading.Lock()
_RF_AP_FIELD_CACHE = BoundedLRU(maximum_items=192, maximum_bytes=768 * 1024 * 1024)


def _shutdown_rf_process_executor(wait: bool = False) -> None:
    """Stop the reusable RF worker pool. Safe to call repeatedly."""
    global _RF_PROCESS_EXECUTOR, _RF_PROCESS_EXECUTOR_WORKERS
    with _RF_PROCESS_EXECUTOR_LOCK:
        executor = _RF_PROCESS_EXECUTOR
        _RF_PROCESS_EXECUTOR = None
        _RF_PROCESS_EXECUTOR_WORKERS = 0
    if executor is not None:
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=wait)
        except Exception:
            pass
    with _RF_CONTEXT_FILE_LOCK:
        paths = list(_RF_CONTEXT_FILE_CACHE.values())
        _RF_CONTEXT_FILE_CACHE.clear()
        _RF_CONTEXT_FILE_ORDER.clear()
    for path in paths:
        try:
            os.unlink(path)
        except Exception:
            pass


def _get_rf_process_executor(max_workers: int, reuse: bool):
    """Return an RF process executor and whether the caller owns it."""
    global _RF_PROCESS_EXECUTOR, _RF_PROCESS_EXECUTOR_WORKERS
    max_workers = max(1, int(max_workers))
    if not reuse:
        return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers), True
    with _RF_PROCESS_EXECUTOR_LOCK:
        if _RF_PROCESS_EXECUTOR is not None and _RF_PROCESS_EXECUTOR_WORKERS != max_workers:
            stale = _RF_PROCESS_EXECUTOR
            _RF_PROCESS_EXECUTOR = None
            _RF_PROCESS_EXECUTOR_WORKERS = 0
            try:
                stale.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                stale.shutdown(wait=False)
            except Exception:
                pass
        if _RF_PROCESS_EXECUTOR is None:
            _RF_PROCESS_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            _RF_PROCESS_EXECUTOR_WORKERS = max_workers
        return _RF_PROCESS_EXECUTOR, False


atexit.register(_shutdown_rf_process_executor)


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


AP_TYPE_PRESETS: Dict[str, Dict[str, str]] = {
    "Ceiling AP": {
        "symbol": "circle_cross", "short": "C", "pattern": "Omni ceiling AP",
        "description": "Ceiling-mounted omnidirectional wireless access point.",
    },
    "Wall AP": {
        "symbol": "square", "short": "W", "pattern": "Wall patch 60 degree",
        "description": "Wall-mounted access point with a forward-facing coverage pattern.",
    },
    "Directional AP": {
        "symbol": "triangle", "short": "D", "pattern": "Directional sector 90 degree",
        "description": "Directional access point or sector antenna.",
    },
    "Outdoor AP": {
        "symbol": "hexagon", "short": "O", "pattern": "Omni ceiling AP",
        "description": "Weather-resistant outdoor access point.",
    },
    "Industrial AP": {
        "symbol": "octagon", "short": "I", "pattern": "Omni ceiling AP",
        "description": "Rugged or industrial wireless access point.",
    },
    "IoT Gateway": {
        "symbol": "diamond", "short": "G", "pattern": "Omni ceiling AP",
        "description": "Low-power or sub-GHz IoT/BLE/Zigbee gateway.",
    },
    "Cellular Small Cell": {
        "symbol": "double_circle", "short": "S", "pattern": "Omni ceiling AP",
        "description": "Indoor private-LTE/5G small cell.",
    },
}


RADIO_PROFILE_PRESETS: Dict[str, List[Dict[str, object]]] = {
    "Wi-Fi 5 dual-band": [
        {"name": "2.4 GHz", "frequency_mhz": 2400.0, "tx_power_dbm": 20.0, "channel": "1", "channel_width_mhz": 20.0, "cutoff_radius_m": 45.0, "spectrum_occupancy_percent": 35.0},
        {"name": "5 GHz", "frequency_mhz": 5000.0, "tx_power_dbm": 20.0, "channel": "36", "channel_width_mhz": 40.0, "cutoff_radius_m": 35.0, "spectrum_occupancy_percent": 20.0},
    ],
    "Wi-Fi 6 dual-band": [
        {"name": "2.4 GHz", "frequency_mhz": 2400.0, "tx_power_dbm": 20.0, "channel": "1", "channel_width_mhz": 20.0, "cutoff_radius_m": 45.0, "spectrum_occupancy_percent": 30.0},
        {"name": "5 GHz", "frequency_mhz": 5000.0, "tx_power_dbm": 20.0, "channel": "36", "channel_width_mhz": 80.0, "cutoff_radius_m": 35.0, "spectrum_occupancy_percent": 18.0},
    ],
    "Wi-Fi 6E tri-band": [
        {"name": "2.4 GHz", "frequency_mhz": 2400.0, "tx_power_dbm": 20.0, "channel": "1", "channel_width_mhz": 20.0, "cutoff_radius_m": 45.0, "spectrum_occupancy_percent": 30.0},
        {"name": "5 GHz", "frequency_mhz": 5000.0, "tx_power_dbm": 20.0, "channel": "36", "channel_width_mhz": 80.0, "cutoff_radius_m": 35.0, "spectrum_occupancy_percent": 18.0},
        {"name": "6 GHz", "frequency_mhz": 6000.0, "tx_power_dbm": 20.0, "channel": "5", "channel_width_mhz": 80.0, "cutoff_radius_m": 30.0, "spectrum_occupancy_percent": 10.0},
    ],
    "Wi-Fi 7 tri-band": [
        {"name": "2.4 GHz", "frequency_mhz": 2400.0, "tx_power_dbm": 20.0, "channel": "1", "channel_width_mhz": 20.0, "cutoff_radius_m": 45.0, "spectrum_occupancy_percent": 25.0},
        {"name": "5 GHz", "frequency_mhz": 5000.0, "tx_power_dbm": 20.0, "channel": "36", "channel_width_mhz": 160.0, "cutoff_radius_m": 35.0, "spectrum_occupancy_percent": 15.0},
        {"name": "6 GHz", "frequency_mhz": 6000.0, "tx_power_dbm": 20.0, "channel": "5", "channel_width_mhz": 320.0, "cutoff_radius_m": 30.0, "spectrum_occupancy_percent": 8.0},
    ],
    "2.4 GHz IoT / BLE": [
        {"name": "2.4 GHz IoT", "frequency_mhz": 2400.0, "tx_power_dbm": 10.0, "channel": "", "channel_width_mhz": 2.0, "cutoff_radius_m": 55.0, "spectrum_occupancy_percent": 10.0},
    ],
    "EU sub-GHz IoT": [
        {"name": "433 MHz", "frequency_mhz": 433.0, "tx_power_dbm": 10.0, "channel": "", "channel_width_mhz": 0.2, "cutoff_radius_m": 120.0, "spectrum_occupancy_percent": 5.0},
        {"name": "868 MHz", "frequency_mhz": 868.0, "tx_power_dbm": 14.0, "channel": "", "channel_width_mhz": 0.2, "cutoff_radius_m": 90.0, "spectrum_occupancy_percent": 8.0},
    ],
    "Private 5G indoor": [
        {"name": "3.5 GHz", "frequency_mhz": 3500.0, "tx_power_dbm": 24.0, "channel": "n78", "channel_width_mhz": 100.0, "cutoff_radius_m": 40.0, "spectrum_occupancy_percent": 20.0},
    ],
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
    ap_type: str = "Ceiling AP"
    radio_profile: str = "Project default radios"
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
    handover_enabled: bool = True
    target_handover_percent: float = 20.0
    handover_margin_db: float = 3.0
    prefer_non_overlapping_handover_channels: bool = True
    sample_spacing_m: float = 3.0
    candidate_spacing_m: float = 6.0
    minimum_ap_spacing_m: float = 8.0
    maximum_aps: int = 64
    # ``auto`` uses eligible IFC/manual/inferred space footprints when present,
    # otherwise shared user rectangular/polygon boundaries, and finally an
    # inferred wall footprint. ``spaces`` requires eligible space geometry;
    # ``boundaries`` uses only the shared planner boundaries; ``walls`` always
    # uses the inferred footprint. User
    # boundaries are a hard clipping limit in every mode whenever one exists.
    planning_area_mode: str = "auto"
    wall_footprint_margin_m: float = 0.0
    # Inferred simulator spaces can be used as predictive-planner coverage and
    # candidate-placement locations. This defaults to True to preserve the
    # behaviour of projects created before the option was exposed in the UI.
    use_inferred_spaces: bool = True
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
            "handover_enabled": self.handover_enabled,
            "target_handover_percent": self.target_handover_percent,
            "handover_margin_db": self.handover_margin_db,
            "prefer_non_overlapping_handover_channels": self.prefer_non_overlapping_handover_channels,
            "sample_spacing_m": self.sample_spacing_m,
            "candidate_spacing_m": self.candidate_spacing_m,
            "minimum_ap_spacing_m": self.minimum_ap_spacing_m,
            "maximum_aps": self.maximum_aps,
            "planning_area_mode": self.planning_area_mode,
            "wall_footprint_margin_m": self.wall_footprint_margin_m,
            "use_inferred_spaces": self.use_inferred_spaces,
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
        base.handover_enabled = bool(data.get("handover_enabled", base.handover_enabled))
        base.target_handover_percent = max(0.0, min(100.0, float(data.get("target_handover_percent", base.target_handover_percent))))
        base.handover_margin_db = max(0.0, min(30.0, float(data.get("handover_margin_db", base.handover_margin_db))))
        base.prefer_non_overlapping_handover_channels = bool(data.get(
            "prefer_non_overlapping_handover_channels", base.prefer_non_overlapping_handover_channels
        ))
        base.sample_spacing_m = max(0.5, float(data.get("sample_spacing_m", base.sample_spacing_m)))
        base.candidate_spacing_m = max(1.0, float(data.get("candidate_spacing_m", base.candidate_spacing_m)))
        base.minimum_ap_spacing_m = max(0.0, float(data.get("minimum_ap_spacing_m", base.minimum_ap_spacing_m)))
        base.maximum_aps = max(1, min(10_000, int(data.get("maximum_aps", base.maximum_aps))))
        raw_area_mode = str(data.get("planning_area_mode", data.get("planner_area_mode", base.planning_area_mode))).strip().lower()
        if raw_area_mode.startswith("space"):
            base.planning_area_mode = "spaces"
        elif raw_area_mode.startswith("bound") or raw_area_mode.startswith("box") or raw_area_mode.startswith("poly"):
            base.planning_area_mode = "boundaries"
        elif raw_area_mode.startswith("wall") or raw_area_mode.startswith("floor"):
            base.planning_area_mode = "walls"
        else:
            base.planning_area_mode = "auto"
        base.wall_footprint_margin_m = max(0.0, min(100.0, float(data.get("wall_footprint_margin_m", base.wall_footprint_margin_m))))
        base.use_inferred_spaces = bool(data.get(
            "use_inferred_spaces",
            data.get("include_inferred_spaces", base.use_inferred_spaces),
        ))
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
    # Spaces inferred in the simulator are RF/planning geometry only. They do
    # not modify the source IFC and are persisted in RF plan files.
    is_inferred: bool = False
    is_user_created: bool = False
    ap_planning_selected: bool = False
    assumption_note: str = ""


@dataclass
class IFCElement2D:
    guid: str
    name: str
    floor: str
    source_file: str
    type_name: str
    material: str
    polygon: Polygon
    z_min: float = 0.0
    z_max: float = 0.0
    source_storey: str = ""
    projected_to_floor: bool = False
    # Every imported IfcElement can carry an RF attenuation profile. Doors and
    # windows additionally replace their host wall at the opening, while other
    # element categories add their own loss when the 3D radio path crosses the
    # element volume. Zero-loss categories remain visual context only until the
    # user assigns attenuation through the bulk type manager.
    ifc_class: str = "IfcElement"
    rf_category: str = "other"
    host_wall_guid: str = ""
    attenuation_by_band_db: Dict[float, float] = field(default_factory=dict)
    rf_type_override: str = ""
    rf_customised: bool = False

    @property
    def is_rf_opening(self) -> bool:
        return self.rf_category in {"door", "window"} and self.is_rf_barrier

    @property
    def is_rf_barrier(self) -> bool:
        return any(abs(float(value)) > 1e-9 for value in self.attenuation_by_band_db.values())

    @property
    def label(self) -> str:
        key = self.rf_type_override or self.type_name or self.material or self.ifc_class or "IFC element"
        return f"{key} | {self.name or self.guid[:8]}"

    def attenuation_db_for_frequency(self, frequency_mhz: float) -> float:
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
class FloorModel:
    name: str
    elevation: float
    walls: List[Wall2D] = field(default_factory=list)
    spaces: List[Space2D] = field(default_factory=list)
    elements: List[IFCElement2D] = field(default_factory=list)
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
class PlannerBoundary2D:
    """User-defined AP planning extent shared by every imported IFC floor."""

    guid: str
    name: str
    polygon: Polygon
    shape_type: str = "polygon"  # rectangle or polygon


@dataclass
class SimulationResult:
    xs: np.ndarray
    ys: np.ndarray
    rssi: np.ndarray
    delay_spread_ns: Optional[np.ndarray] = None
    path_count: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    boundary_geometry: object = field(default=None, repr=False, compare=False)
    ignored_point_count: int = 0
    execution_mode: str = "single-process"
    worker_processes: int = 1
    elapsed_seconds: float = 0.0
    performance_note: str = ""
    progressive_fraction: float = 1.0
    approximate_points: int = 0
    exact_points: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    render_mode_override: str = ""
    render_cache: Dict[object, object] = field(default_factory=dict, repr=False, compare=False)


@dataclass
class PropagationSample:
    rssi_dbm: float
    delay_spread_ns: float = 0.0
    path_count: int = 1
    direct_rssi_dbm: float = -120.0


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
    default_door_attenuation_by_material_db: Dict[str, Dict[float, float]] = field(default_factory=lambda: {
        "default": {433.0: 2.0, 868.0: 3.0, 2400.0: 4.0, 5000.0: 6.0, 6000.0: 8.0},
        "timber": {433.0: 1.5, 868.0: 2.0, 2400.0: 3.0, 5000.0: 4.0, 6000.0: 5.0},
        "wood": {433.0: 1.5, 868.0: 2.0, 2400.0: 3.0, 5000.0: 4.0, 6000.0: 5.0},
        "glass": {433.0: 0.8, 868.0: 1.2, 2400.0: 2.0, 5000.0: 3.0, 6000.0: 4.0},
        "metal": {433.0: 8.0, 868.0: 10.0, 2400.0: 12.0, 5000.0: 18.0, 6000.0: 22.0},
        "steel": {433.0: 8.0, 868.0: 10.0, 2400.0: 12.0, 5000.0: 18.0, 6000.0: 22.0},
        "fire": {433.0: 7.0, 868.0: 9.0, 2400.0: 11.0, 5000.0: 16.0, 6000.0: 20.0},
    })
    default_window_attenuation_by_material_db: Dict[str, Dict[float, float]] = field(default_factory=lambda: {
        "default": {433.0: 0.8, 868.0: 1.2, 2400.0: 2.0, 5000.0: 3.0, 6000.0: 4.0},
        "glass": {433.0: 0.8, 868.0: 1.2, 2400.0: 2.0, 5000.0: 3.0, 6000.0: 4.0},
        "double": {433.0: 1.0, 868.0: 1.5, 2400.0: 2.5, 5000.0: 4.0, 6000.0: 5.0},
        "triple": {433.0: 1.2, 868.0: 1.8, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 6.0},
        "low-e": {433.0: 3.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 14.0, 6000.0: 18.0},
        "low e": {433.0: 3.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 14.0, 6000.0: 18.0},
        "metallised": {433.0: 4.0, 868.0: 6.0, 2400.0: 10.0, 5000.0: 16.0, 6000.0: 20.0},
        "metalized": {433.0: 4.0, 868.0: 6.0, 2400.0: 10.0, 5000.0: 16.0, 6000.0: 20.0},
    })
    # Generic IFC elements are available to the RF model by type. Building
    # fabric and structural categories receive conservative defaults; furniture,
    # MEP and unclassified items default to zero so a detailed IFC does not
    # accidentally accumulate unrealistic loss. The bulk attenuation manager can
    # activate or override any category/type across the complete project.
    default_ifc_element_attenuation_by_type_db: Dict[str, Dict[float, float]] = field(default_factory=lambda: {
        "default": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
        "slab": {433.0: 8.0, 868.0: 10.0, 2400.0: 12.0, 5000.0: 18.0, 6000.0: 22.0},
        "roof": {433.0: 5.0, 868.0: 7.0, 2400.0: 9.0, 5000.0: 13.0, 6000.0: 16.0},
        "column": {433.0: 3.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 12.0, 6000.0: 15.0},
        "beam": {433.0: 2.0, 868.0: 4.0, 2400.0: 6.0, 5000.0: 9.0, 6000.0: 12.0},
        "curtain_wall": {433.0: 1.0, 868.0: 2.0, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0},
        "covering": {433.0: 0.5, 868.0: 0.8, 2400.0: 1.0, 5000.0: 1.5, 6000.0: 2.0},
        "plate": {433.0: 2.0, 868.0: 3.0, 2400.0: 5.0, 5000.0: 8.0, 6000.0: 10.0},
        "member": {433.0: 1.0, 868.0: 2.0, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0},
        "concrete": {433.0: 5.0, 868.0: 7.0, 2400.0: 12.0, 5000.0: 16.0, 6000.0: 20.0},
        "brick": {433.0: 4.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 11.0, 6000.0: 14.0},
        "glass": {433.0: 1.0, 868.0: 2.0, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0},
        "metal": {433.0: 12.0, 868.0: 16.0, 2400.0: 20.0, 5000.0: 28.0, 6000.0: 35.0},
        "steel": {433.0: 12.0, 868.0: 16.0, 2400.0: 20.0, 5000.0: 28.0, 6000.0: 35.0},
        "furniture": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
        "equipment": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
        "distribution": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
        "proxy": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
        "transport": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
        "assembly": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
        "other": {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0},
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
    # RF rows are divided adaptively. rf_tile_rows is the preferred minimum
    # strip height for large grids; rf_tiles_per_worker supplies enough strips
    # for load balancing without repeatedly serialising the complete IFC model.
    rf_tile_rows: int = 16
    rf_tiles_per_worker: int = 2
    reuse_rf_process_pool: bool = True
    rf_worker_index_cache_entries: int = 2
    # Calculation profiles and adaptive/incremental RF acceleration.
    rf_calculation_profile: str = "balanced"  # fast, balanced, detailed
    enable_adaptive_rf_grid: bool = True
    adaptive_coarse_resolution_m: float = 3.0
    adaptive_gradient_threshold_db_per_m: float = 4.0
    adaptive_threshold_margin_db: float = 2.0
    adaptive_geometry_buffer_m: float = 1.5
    adaptive_ap_refine_radius_m: float = 6.0
    enable_per_ap_heatmap_cache: bool = True
    per_ap_heatmap_cache_entries: int = 192
    per_ap_heatmap_cache_mb: int = 768
    reuse_path_geometry_across_frequencies: bool = True
    precompute_reflection_candidates_per_tile: bool = True
    enable_tile_influence_pruning: bool = True
    tile_influence_margin_db: float = 8.0
    enable_tile_local_geometry: bool = True
    multipath_relative_power_cutoff_db: float = 30.0
    enable_numba_rf_kernels: bool = True
    use_shared_memory_rf_results: bool = True
    progressive_heatmap_updates: bool = True
    progressive_update_percent: int = 20
    heatmap_render_mode: str = "raster_contours"  # raster, raster_contours, contours
    interactive_preview_enabled: bool = True
    interactive_preview_delay_ms: int = 350
    interactive_preview_resolution_m: float = 3.0
    # When enabled and one or more shared planner boundaries exist, RSSI grid
    # points outside their union are not calculated, displayed or exported.
    ignore_results_outside_planner_boundaries: bool = False

    # Advanced propagation model. Reflections use a bounded image-source model
    # on the active floor. Higher-order rays are available by increasing
    # max_reflection_order; order 1 is the recommended interactive default.
    enable_multipath_reflections: bool = True
    max_reflection_order: int = 1
    max_reflection_surfaces: int = 6
    max_reflection_paths: int = 8
    reflection_search_radius_m: float = 18.0
    minimum_reflection_coefficient: float = 0.025
    enable_corner_diffraction: bool = True
    max_diffraction_paths: int = 3
    diffraction_search_radius_m: float = 5.0
    minimum_diffraction_loss_db: float = 6.0
    enable_small_scale_fading: bool = True
    small_scale_fading_sigma_db: float = 1.5
    fading_correlation_distance_m: float = 0.75
    fading_seed: int = 1729
    calculate_delay_spread: bool = True
    combined_ap_mode: str = "strongest"  # strongest or power_sum
    reflection_material_properties: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "default": {"relative_permittivity": 4.0, "conductivity_s_per_m": 0.02, "roughness_m": 0.003, "reflection_scale": 1.0},
        "concrete": {"relative_permittivity": 6.0, "conductivity_s_per_m": 0.05, "roughness_m": 0.006, "reflection_scale": 1.0},
        "brick": {"relative_permittivity": 4.4, "conductivity_s_per_m": 0.03, "roughness_m": 0.006, "reflection_scale": 0.95},
        "masonry": {"relative_permittivity": 5.0, "conductivity_s_per_m": 0.04, "roughness_m": 0.007, "reflection_scale": 0.95},
        "plasterboard": {"relative_permittivity": 2.5, "conductivity_s_per_m": 0.01, "roughness_m": 0.003, "reflection_scale": 0.85},
        "partition": {"relative_permittivity": 2.5, "conductivity_s_per_m": 0.01, "roughness_m": 0.003, "reflection_scale": 0.85},
        "glass": {"relative_permittivity": 6.5, "conductivity_s_per_m": 0.004, "roughness_m": 0.001, "reflection_scale": 1.0},
        "metal": {"relative_permittivity": 80.0, "conductivity_s_per_m": 1000000.0, "roughness_m": 0.001, "reflection_scale": 1.0},
        "steel": {"relative_permittivity": 80.0, "conductivity_s_per_m": 1000000.0, "roughness_m": 0.001, "reflection_scale": 1.0},
        "timber": {"relative_permittivity": 2.0, "conductivity_s_per_m": 0.01, "roughness_m": 0.004, "reflection_scale": 0.75},
        "wood": {"relative_permittivity": 2.0, "conductivity_s_per_m": 0.01, "roughness_m": 0.004, "reflection_scale": 0.75},
    })
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
        def _material_profile_dict(raw, fallback):
            if not isinstance(raw, dict):
                return fallback
            parsed_profiles = {}
            for material_name, profile in raw.items():
                if isinstance(profile, dict):
                    parsed_profiles[str(material_name).lower()] = {float(k): float(v) for k, v in profile.items()}
            return parsed_profiles or fallback

        settings.default_door_attenuation_by_material_db = _material_profile_dict(
            data.get("default_door_attenuation_by_material_db", data.get("door_attenuation_by_material_db", {})),
            settings.default_door_attenuation_by_material_db,
        )
        settings.default_window_attenuation_by_material_db = _material_profile_dict(
            data.get("default_window_attenuation_by_material_db", data.get("window_attenuation_by_material_db", {})),
            settings.default_window_attenuation_by_material_db,
        )
        settings.default_ifc_element_attenuation_by_type_db = _material_profile_dict(
            data.get("default_ifc_element_attenuation_by_type_db", data.get("ifc_element_attenuation_by_type_db", {})),
            settings.default_ifc_element_attenuation_by_type_db,
        )
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
        settings.rf_tiles_per_worker = max(1, min(8, int(data.get("rf_tiles_per_worker", 2))))
        settings.reuse_rf_process_pool = bool(data.get("reuse_rf_process_pool", True))
        settings.rf_worker_index_cache_entries = max(1, min(8, int(data.get("rf_worker_index_cache_entries", 2))))
        performance = data.get("rf_performance", data.get("performance", {}))
        if not isinstance(performance, dict):
            performance = {}
        settings.enable_ifc_multiprocessing = bool(performance.get("enable_ifc_multiprocessing", settings.enable_ifc_multiprocessing))
        settings.max_ifc_loader_processes = max(0, int(performance.get("max_ifc_loader_processes", settings.max_ifc_loader_processes)))
        settings.max_parallel_huge_ifc_processes = max(1, int(performance.get("max_parallel_huge_ifc_processes", settings.max_parallel_huge_ifc_processes)))
        settings.enable_rf_multiprocessing = bool(performance.get("enable_rf_multiprocessing", settings.enable_rf_multiprocessing))
        settings.max_rf_worker_processes = max(0, int(performance.get("max_rf_worker_processes", settings.max_rf_worker_processes)))
        settings.rf_multiprocessing_min_points = max(1, int(performance.get("rf_multiprocessing_min_points", settings.rf_multiprocessing_min_points)))
        settings.rf_tile_rows = max(1, int(performance.get("rf_tile_rows", settings.rf_tile_rows)))
        settings.rf_tiles_per_worker = max(1, min(8, int(performance.get("rf_tiles_per_worker", settings.rf_tiles_per_worker))))
        settings.reuse_rf_process_pool = bool(performance.get("reuse_rf_process_pool", settings.reuse_rf_process_pool))
        settings.rf_worker_index_cache_entries = max(1, min(8, int(performance.get("rf_worker_index_cache_entries", settings.rf_worker_index_cache_entries))))
        settings.contour_interpolation_factor = max(1, int(performance.get("contour_interpolation_factor", settings.contour_interpolation_factor)))
        profile = str(performance.get("calculation_profile", data.get("rf_calculation_profile", settings.rf_calculation_profile))).strip().lower()
        settings.rf_calculation_profile = profile if profile in {"fast", "balanced", "detailed"} else "balanced"
        settings.enable_adaptive_rf_grid = bool(performance.get("enable_adaptive_grid", data.get("enable_adaptive_rf_grid", settings.enable_adaptive_rf_grid)))
        settings.adaptive_coarse_resolution_m = max(0.25, float(performance.get("coarse_resolution_m", data.get("adaptive_coarse_resolution_m", settings.adaptive_coarse_resolution_m))))
        settings.adaptive_gradient_threshold_db_per_m = max(0.05, float(performance.get("gradient_threshold_db_per_m", settings.adaptive_gradient_threshold_db_per_m)))
        settings.adaptive_threshold_margin_db = max(0.1, float(performance.get("threshold_margin_db", settings.adaptive_threshold_margin_db)))
        settings.adaptive_geometry_buffer_m = max(0.0, float(performance.get("geometry_buffer_m", settings.adaptive_geometry_buffer_m)))
        settings.adaptive_ap_refine_radius_m = max(0.5, float(performance.get("ap_refine_radius_m", settings.adaptive_ap_refine_radius_m)))
        settings.enable_per_ap_heatmap_cache = bool(performance.get("enable_per_ap_cache", settings.enable_per_ap_heatmap_cache))
        settings.per_ap_heatmap_cache_entries = max(1, int(performance.get("per_ap_cache_entries", settings.per_ap_heatmap_cache_entries)))
        settings.per_ap_heatmap_cache_mb = max(32, int(performance.get("per_ap_cache_mb", settings.per_ap_heatmap_cache_mb)))
        settings.reuse_path_geometry_across_frequencies = bool(performance.get("reuse_path_geometry_across_frequencies", settings.reuse_path_geometry_across_frequencies))
        settings.precompute_reflection_candidates_per_tile = bool(performance.get("precompute_reflection_candidates_per_tile", settings.precompute_reflection_candidates_per_tile))
        settings.enable_tile_influence_pruning = bool(performance.get("enable_tile_influence_pruning", settings.enable_tile_influence_pruning))
        settings.tile_influence_margin_db = max(0.0, float(performance.get("tile_influence_margin_db", settings.tile_influence_margin_db)))
        settings.enable_tile_local_geometry = bool(performance.get("enable_tile_local_geometry", settings.enable_tile_local_geometry))
        settings.multipath_relative_power_cutoff_db = max(0.0, min(80.0, float(performance.get("multipath_relative_power_cutoff_db", settings.multipath_relative_power_cutoff_db))))
        settings.enable_numba_rf_kernels = bool(performance.get("enable_numba_kernels", settings.enable_numba_rf_kernels))
        settings.use_shared_memory_rf_results = bool(performance.get("use_shared_memory_results", settings.use_shared_memory_rf_results))
        settings.progressive_heatmap_updates = bool(performance.get("progressive_heatmap_updates", settings.progressive_heatmap_updates))
        settings.progressive_update_percent = max(5, min(100, int(performance.get("progressive_update_percent", settings.progressive_update_percent))))
        render_mode = str(performance.get("heatmap_render_mode", settings.heatmap_render_mode)).strip().lower()
        settings.heatmap_render_mode = render_mode if render_mode in {"raster", "raster_contours", "contours"} else "raster_contours"
        settings.interactive_preview_enabled = bool(performance.get("interactive_preview_enabled", settings.interactive_preview_enabled))
        settings.interactive_preview_delay_ms = max(50, int(performance.get("interactive_preview_delay_ms", settings.interactive_preview_delay_ms)))
        settings.interactive_preview_resolution_m = max(0.25, float(performance.get("interactive_preview_resolution_m", settings.interactive_preview_resolution_m)))
        propagation = data.get("propagation_model", data.get("advanced_propagation", {}))
        if not isinstance(propagation, dict):
            propagation = {}
        settings.ignore_results_outside_planner_boundaries = bool(
            propagation.get(
                "ignore_results_outside_planner_boundaries",
                data.get("ignore_results_outside_planner_boundaries", settings.ignore_results_outside_planner_boundaries),
            )
        )
        settings.enable_multipath_reflections = bool(propagation.get("enable_multipath_reflections", data.get("enable_multipath_reflections", settings.enable_multipath_reflections)))
        settings.max_reflection_order = max(0, min(3, int(propagation.get("max_reflection_order", data.get("max_reflection_order", settings.max_reflection_order)))))
        settings.max_reflection_surfaces = max(1, min(24, int(propagation.get("max_reflection_surfaces", settings.max_reflection_surfaces))))
        settings.max_reflection_paths = max(0, min(64, int(propagation.get("max_reflection_paths", settings.max_reflection_paths))))
        settings.reflection_search_radius_m = max(0.5, min(200.0, float(propagation.get("reflection_search_radius_m", settings.reflection_search_radius_m))))
        settings.minimum_reflection_coefficient = max(0.0, min(1.0, float(propagation.get("minimum_reflection_coefficient", settings.minimum_reflection_coefficient))))
        settings.enable_corner_diffraction = bool(propagation.get("enable_corner_diffraction", data.get("enable_corner_diffraction", settings.enable_corner_diffraction)))
        settings.max_diffraction_paths = max(0, min(32, int(propagation.get("max_diffraction_paths", settings.max_diffraction_paths))))
        settings.diffraction_search_radius_m = max(0.25, min(100.0, float(propagation.get("diffraction_search_radius_m", settings.diffraction_search_radius_m))))
        settings.minimum_diffraction_loss_db = max(0.0, min(45.0, float(propagation.get("minimum_diffraction_loss_db", settings.minimum_diffraction_loss_db))))
        settings.enable_small_scale_fading = bool(propagation.get("enable_small_scale_fading", data.get("enable_small_scale_fading", settings.enable_small_scale_fading)))
        settings.small_scale_fading_sigma_db = max(0.0, min(20.0, float(propagation.get("small_scale_fading_sigma_db", settings.small_scale_fading_sigma_db))))
        settings.fading_correlation_distance_m = max(0.05, min(20.0, float(propagation.get("fading_correlation_distance_m", settings.fading_correlation_distance_m))))
        settings.fading_seed = int(propagation.get("fading_seed", settings.fading_seed))
        settings.calculate_delay_spread = bool(propagation.get("calculate_delay_spread", settings.calculate_delay_spread))
        combination = str(propagation.get("combined_ap_mode", data.get("combined_ap_mode", settings.combined_ap_mode))).strip().lower()
        settings.combined_ap_mode = "power_sum" if combination in {"power_sum", "sum", "incoherent_power", "total_power"} else "strongest"
        raw_reflection_profiles = propagation.get("reflection_material_properties", data.get("reflection_material_properties", {}))
        if isinstance(raw_reflection_profiles, dict):
            parsed_reflection_profiles = {}
            for material_name, profile in raw_reflection_profiles.items():
                if not isinstance(profile, dict):
                    continue
                parsed_reflection_profiles[str(material_name).strip().lower()] = {
                    "relative_permittivity": float(profile.get("relative_permittivity", 4.0)),
                    "conductivity_s_per_m": float(profile.get("conductivity_s_per_m", 0.02)),
                    "roughness_m": float(profile.get("roughness_m", 0.003)),
                    "reflection_scale": float(profile.get("reflection_scale", 1.0)),
                }
            if parsed_reflection_profiles:
                settings.reflection_material_properties.update(parsed_reflection_profiles)
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

    def performance_model_dict(self) -> Dict[str, object]:
        return {
            "enable_ifc_multiprocessing": bool(self.enable_ifc_multiprocessing),
            "max_ifc_loader_processes": int(self.max_ifc_loader_processes),
            "max_parallel_huge_ifc_processes": int(self.max_parallel_huge_ifc_processes),
            "enable_rf_multiprocessing": bool(self.enable_rf_multiprocessing),
            "max_rf_worker_processes": int(self.max_rf_worker_processes),
            "rf_multiprocessing_min_points": int(self.rf_multiprocessing_min_points),
            "rf_tile_rows": int(self.rf_tile_rows),
            "rf_tiles_per_worker": int(self.rf_tiles_per_worker),
            "reuse_rf_process_pool": bool(self.reuse_rf_process_pool),
            "rf_worker_index_cache_entries": int(self.rf_worker_index_cache_entries),
            "contour_interpolation_factor": int(self.contour_interpolation_factor),
            "calculation_profile": str(self.rf_calculation_profile),
            "enable_adaptive_grid": bool(self.enable_adaptive_rf_grid),
            "coarse_resolution_m": float(self.adaptive_coarse_resolution_m),
            "gradient_threshold_db_per_m": float(self.adaptive_gradient_threshold_db_per_m),
            "threshold_margin_db": float(self.adaptive_threshold_margin_db),
            "geometry_buffer_m": float(self.adaptive_geometry_buffer_m),
            "ap_refine_radius_m": float(self.adaptive_ap_refine_radius_m),
            "enable_per_ap_cache": bool(self.enable_per_ap_heatmap_cache),
            "per_ap_cache_entries": int(self.per_ap_heatmap_cache_entries),
            "per_ap_cache_mb": int(self.per_ap_heatmap_cache_mb),
            "reuse_path_geometry_across_frequencies": bool(self.reuse_path_geometry_across_frequencies),
            "precompute_reflection_candidates_per_tile": bool(self.precompute_reflection_candidates_per_tile),
            "enable_tile_influence_pruning": bool(self.enable_tile_influence_pruning),
            "tile_influence_margin_db": float(self.tile_influence_margin_db),
            "enable_tile_local_geometry": bool(self.enable_tile_local_geometry),
            "multipath_relative_power_cutoff_db": float(self.multipath_relative_power_cutoff_db),
            "enable_numba_kernels": bool(self.enable_numba_rf_kernels),
            "use_shared_memory_results": bool(self.use_shared_memory_rf_results),
            "progressive_heatmap_updates": bool(self.progressive_heatmap_updates),
            "progressive_update_percent": int(self.progressive_update_percent),
            "heatmap_render_mode": str(self.heatmap_render_mode),
            "interactive_preview_enabled": bool(self.interactive_preview_enabled),
            "interactive_preview_delay_ms": int(self.interactive_preview_delay_ms),
            "interactive_preview_resolution_m": float(self.interactive_preview_resolution_m),
        }

    def propagation_model_dict(self) -> Dict[str, object]:
        return {
            "ignore_results_outside_planner_boundaries": bool(self.ignore_results_outside_planner_boundaries),
            "enable_multipath_reflections": bool(self.enable_multipath_reflections),
            "max_reflection_order": int(self.max_reflection_order),
            "max_reflection_surfaces": int(self.max_reflection_surfaces),
            "max_reflection_paths": int(self.max_reflection_paths),
            "reflection_search_radius_m": float(self.reflection_search_radius_m),
            "minimum_reflection_coefficient": float(self.minimum_reflection_coefficient),
            "enable_corner_diffraction": bool(self.enable_corner_diffraction),
            "max_diffraction_paths": int(self.max_diffraction_paths),
            "diffraction_search_radius_m": float(self.diffraction_search_radius_m),
            "minimum_diffraction_loss_db": float(self.minimum_diffraction_loss_db),
            "enable_small_scale_fading": bool(self.enable_small_scale_fading),
            "small_scale_fading_sigma_db": float(self.small_scale_fading_sigma_db),
            "fading_correlation_distance_m": float(self.fading_correlation_distance_m),
            "fading_seed": int(self.fading_seed),
            "calculate_delay_spread": bool(self.calculate_delay_spread),
            "combined_ap_mode": str(self.combined_ap_mode),
            "reflection_material_properties": {
                str(material): {
                    str(key): float(value) for key, value in dict(profile).items()
                }
                for material, profile in self.reflection_material_properties.items()
            },
        }

    def apply_performance_model_dict(self, data: Optional[Dict[str, object]]):
        if not isinstance(data, dict):
            return
        self.enable_ifc_multiprocessing = bool(data.get("enable_ifc_multiprocessing", self.enable_ifc_multiprocessing))
        self.max_ifc_loader_processes = max(0, int(data.get("max_ifc_loader_processes", self.max_ifc_loader_processes)))
        self.max_parallel_huge_ifc_processes = max(1, int(data.get("max_parallel_huge_ifc_processes", self.max_parallel_huge_ifc_processes)))
        self.enable_rf_multiprocessing = bool(data.get("enable_rf_multiprocessing", self.enable_rf_multiprocessing))
        self.max_rf_worker_processes = max(0, int(data.get("max_rf_worker_processes", self.max_rf_worker_processes)))
        self.rf_multiprocessing_min_points = max(1, int(data.get("rf_multiprocessing_min_points", self.rf_multiprocessing_min_points)))
        self.rf_tile_rows = max(1, int(data.get("rf_tile_rows", self.rf_tile_rows)))
        self.rf_tiles_per_worker = max(1, min(8, int(data.get("rf_tiles_per_worker", self.rf_tiles_per_worker))))
        self.reuse_rf_process_pool = bool(data.get("reuse_rf_process_pool", self.reuse_rf_process_pool))
        self.rf_worker_index_cache_entries = max(1, min(8, int(data.get("rf_worker_index_cache_entries", self.rf_worker_index_cache_entries))))
        self.contour_interpolation_factor = max(1, int(data.get("contour_interpolation_factor", self.contour_interpolation_factor)))
        profile = str(data.get("calculation_profile", self.rf_calculation_profile)).strip().lower()
        self.rf_calculation_profile = profile if profile in {"fast", "balanced", "detailed"} else "balanced"
        self.enable_adaptive_rf_grid = bool(data.get("enable_adaptive_grid", self.enable_adaptive_rf_grid))
        self.adaptive_coarse_resolution_m = max(0.25, float(data.get("coarse_resolution_m", self.adaptive_coarse_resolution_m)))
        self.adaptive_gradient_threshold_db_per_m = max(0.05, float(data.get("gradient_threshold_db_per_m", self.adaptive_gradient_threshold_db_per_m)))
        self.adaptive_threshold_margin_db = max(0.1, float(data.get("threshold_margin_db", self.adaptive_threshold_margin_db)))
        self.adaptive_geometry_buffer_m = max(0.0, float(data.get("geometry_buffer_m", self.adaptive_geometry_buffer_m)))
        self.adaptive_ap_refine_radius_m = max(0.5, float(data.get("ap_refine_radius_m", self.adaptive_ap_refine_radius_m)))
        self.enable_per_ap_heatmap_cache = bool(data.get("enable_per_ap_cache", self.enable_per_ap_heatmap_cache))
        self.per_ap_heatmap_cache_entries = max(1, int(data.get("per_ap_cache_entries", self.per_ap_heatmap_cache_entries)))
        self.per_ap_heatmap_cache_mb = max(32, int(data.get("per_ap_cache_mb", self.per_ap_heatmap_cache_mb)))
        self.reuse_path_geometry_across_frequencies = bool(data.get("reuse_path_geometry_across_frequencies", self.reuse_path_geometry_across_frequencies))
        self.precompute_reflection_candidates_per_tile = bool(data.get("precompute_reflection_candidates_per_tile", self.precompute_reflection_candidates_per_tile))
        self.enable_tile_influence_pruning = bool(data.get("enable_tile_influence_pruning", self.enable_tile_influence_pruning))
        self.tile_influence_margin_db = max(0.0, float(data.get("tile_influence_margin_db", self.tile_influence_margin_db)))
        self.enable_tile_local_geometry = bool(data.get("enable_tile_local_geometry", self.enable_tile_local_geometry))
        self.multipath_relative_power_cutoff_db = max(0.0, min(80.0, float(data.get("multipath_relative_power_cutoff_db", self.multipath_relative_power_cutoff_db))))
        self.enable_numba_rf_kernels = bool(data.get("enable_numba_kernels", self.enable_numba_rf_kernels))
        self.use_shared_memory_rf_results = bool(data.get("use_shared_memory_results", self.use_shared_memory_rf_results))
        self.progressive_heatmap_updates = bool(data.get("progressive_heatmap_updates", self.progressive_heatmap_updates))
        self.progressive_update_percent = max(5, min(100, int(data.get("progressive_update_percent", self.progressive_update_percent))))
        render_mode = str(data.get("heatmap_render_mode", self.heatmap_render_mode)).strip().lower()
        self.heatmap_render_mode = render_mode if render_mode in {"raster", "raster_contours", "contours"} else "raster_contours"
        self.interactive_preview_enabled = bool(data.get("interactive_preview_enabled", self.interactive_preview_enabled))
        self.interactive_preview_delay_ms = max(50, int(data.get("interactive_preview_delay_ms", self.interactive_preview_delay_ms)))
        self.interactive_preview_resolution_m = max(0.25, float(data.get("interactive_preview_resolution_m", self.interactive_preview_resolution_m)))

    def apply_propagation_model_dict(self, data: Optional[Dict[str, object]]):
        if not isinstance(data, dict):
            return
        self.ignore_results_outside_planner_boundaries = bool(
            data.get("ignore_results_outside_planner_boundaries", self.ignore_results_outside_planner_boundaries)
        )
        self.enable_multipath_reflections = bool(data.get("enable_multipath_reflections", self.enable_multipath_reflections))
        self.max_reflection_order = max(0, min(3, int(data.get("max_reflection_order", self.max_reflection_order))))
        self.max_reflection_surfaces = max(1, min(24, int(data.get("max_reflection_surfaces", self.max_reflection_surfaces))))
        self.max_reflection_paths = max(0, min(64, int(data.get("max_reflection_paths", self.max_reflection_paths))))
        self.reflection_search_radius_m = max(0.5, min(200.0, float(data.get("reflection_search_radius_m", self.reflection_search_radius_m))))
        self.minimum_reflection_coefficient = max(0.0, min(1.0, float(data.get("minimum_reflection_coefficient", self.minimum_reflection_coefficient))))
        self.enable_corner_diffraction = bool(data.get("enable_corner_diffraction", self.enable_corner_diffraction))
        self.max_diffraction_paths = max(0, min(32, int(data.get("max_diffraction_paths", self.max_diffraction_paths))))
        self.diffraction_search_radius_m = max(0.25, min(100.0, float(data.get("diffraction_search_radius_m", self.diffraction_search_radius_m))))
        self.minimum_diffraction_loss_db = max(0.0, min(45.0, float(data.get("minimum_diffraction_loss_db", self.minimum_diffraction_loss_db))))
        self.enable_small_scale_fading = bool(data.get("enable_small_scale_fading", self.enable_small_scale_fading))
        self.small_scale_fading_sigma_db = max(0.0, min(20.0, float(data.get("small_scale_fading_sigma_db", self.small_scale_fading_sigma_db))))
        self.fading_correlation_distance_m = max(0.05, min(20.0, float(data.get("fading_correlation_distance_m", self.fading_correlation_distance_m))))
        self.fading_seed = int(data.get("fading_seed", self.fading_seed))
        self.calculate_delay_spread = bool(data.get("calculate_delay_spread", self.calculate_delay_spread))
        combination = str(data.get("combined_ap_mode", self.combined_ap_mode)).strip().lower()
        self.combined_ap_mode = "power_sum" if combination in {"power_sum", "sum", "incoherent_power", "total_power"} else "strongest"
        raw_profiles = data.get("reflection_material_properties", {})
        if isinstance(raw_profiles, dict):
            for material, profile in raw_profiles.items():
                if not isinstance(profile, dict):
                    continue
                self.reflection_material_properties[str(material).strip().lower()] = {
                    "relative_permittivity": float(profile.get("relative_permittivity", 4.0)),
                    "conductivity_s_per_m": float(profile.get("conductivity_s_per_m", 0.02)),
                    "roughness_m": float(profile.get("roughness_m", 0.003)),
                    "reflection_scale": float(profile.get("reflection_scale", 1.0)),
                }

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
        self.ifc = ifcopenshell.open(str(path))
        try:
            import ifcopenshell.util.unit as ifc_unit_util
            self.unit_scale = float(ifc_unit_util.calculate_unit_scale(self.ifc) or 1.0)
        except Exception:
            self.unit_scale = 1.0
        self.project_external_walls_across_floors = bool(project_external_walls_across_floors)
        self.external_wall_keywords = [
            str(v).strip().lower() for v in (external_wall_keywords or [
                "external", "exterior", "outer", "facade", "façade", "curtain", "envelope", "perimeter"
            ]) if str(v).strip()
        ]
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)

    def load(self) -> Dict[str, FloorModel]:
        return self.load_filtered()

    def load_filtered(
        self,
        wall_guids: Optional[Iterable[str]] = None,
        space_guids: Optional[Iterable[str]] = None,
        element_guids: Optional[Iterable[str]] = None,
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
        wanted_elements = set(element_guids or [])
        filter_walls = bool(wall_guids is not None)
        filter_spaces = bool(space_guids is not None)
        filter_elements = bool(element_guids is not None)

        seen_wall_guids = set()
        wall_entities = []
        for wall in list(self.ifc.by_type("IfcWall")) + list(self.ifc.by_type("IfcWallStandardCase")):
            guid = getattr(wall, "GlobalId", "") or ""
            if guid in seen_wall_guids:
                continue
            seen_wall_guids.add(guid)
            if filter_walls and guid not in wanted_walls:
                continue
            wall_entities.append(wall)

        wall_geometry = self._plan_polygons_from_geometry_iterator(
            wall_entities,
            max_threads=(1 if filter_walls else min(8, os.cpu_count() or 1)),
        )
        for wall in wall_entities:
            guid = getattr(wall, "GlobalId", "") or ""
            source_floor = self._container_storey_name(wall)
            geom = wall_geometry.get(guid)
            if geom is None:
                geom = self._plan_polygon_from_geometry(wall)
            if geom is None:
                continue
            poly, z_min, z_max = geom
            if poly is None or poly.area <= 0:
                continue
            mat = self._material_name(wall)
            type_name = self._type_name(wall)
            assigned_floor = self._assigned_floor_name(wall, floors, z_min, z_max, source_floor)
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

        handled_wall_guids = set(seen_wall_guids)
        for element in self.ifc.by_type("IfcElement"):
            if element.is_a("IfcWall") or element.is_a("IfcWallStandardCase") or element.is_a("IfcOpeningElement"):
                continue
            guid = getattr(element, "GlobalId", "") or ""
            if not guid or guid in handled_wall_guids:
                continue
            if filter_elements and guid not in wanted_elements:
                continue
            source_floor = self._container_storey_name(element)
            geom = self._plan_polygon_from_element(element)
            if geom is None:
                continue
            poly, z_min, z_max = geom
            if poly is None or poly.area <= 0:
                continue
            assigned_floor = self._assigned_floor_name(element, floors, z_min, z_max, source_floor)
            mat = self._material_name(element)
            type_name = self._type_name(element)
            ifc_class = str(element.is_a() or "IfcElement")
            rf_category = self._rf_category_for_element(element)
            vertical_categories = {"column", "pile", "member", "stair", "ramp", "assembly", "proxy", "distribution", "transport"}
            if rf_category in vertical_categories and float(z_max) - float(z_min) > 2.8:
                floor_names = self._floor_names_for_z_span(floors, z_min, z_max, assigned_floor)
            else:
                floor_names = [assigned_floor]
            host_wall_guid = self._host_wall_guid_for_filling(element) if rf_category in {"door", "window"} else ""
            if rf_category in {"door", "window"}:
                element_profile = self._default_opening_attenuation_profile(rf_category, mat, type_name)
            else:
                element_profile = self._default_ifc_element_attenuation_profile(rf_category, mat, type_name, ifc_class)
            for floor_name in floor_names:
                floors.setdefault(floor_name, FloorModel(name=floor_name, elevation=0.0)).elements.append(
                    IFCElement2D(
                        guid=guid,
                        name=getattr(element, "Name", "") or type_name or "IFC element",
                        floor=floor_name,
                        source_file=self.path.name,
                        type_name=type_name,
                        material=mat,
                        polygon=poly,
                        z_min=z_min,
                        z_max=z_max,
                        source_storey=source_floor or "",
                        projected_to_floor=(floor_name != assigned_floor),
                        ifc_class=ifc_class,
                        rf_category=rf_category,
                        host_wall_guid=host_wall_guid,
                        attenuation_by_band_db=dict(element_profile),
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
            assigned_floor = self._assigned_floor_name(space, floors, z_min, z_max, source_floor)
            floor_names = [assigned_floor]
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
            out[name] = self._storey_elevation(st)
        return out

    def _storey_elevation(self, storey) -> float:
        return _ifc_storey_elevation_m(storey, dz=self.dz, unit_scale=self.unit_scale)

    def _container_storey_name(self, product) -> Optional[str]:
        for rel in getattr(product, "ContainedInStructure", []) or []:
            s = getattr(rel, "RelatingStructure", None)
            if s and s.is_a("IfcBuildingStorey"):
                return s.Name or s.GlobalId
        return None

    @staticmethod
    def _nearest_floor_name(floors: Dict[str, FloorModel], z: float) -> str:
        return min(floors.values(), key=lambda f: abs(f.elevation - z)).name

    @staticmethod
    def _local_placement_z(product) -> float:
        placement = getattr(product, "ObjectPlacement", None)
        rel = getattr(placement, "RelativePlacement", None)
        coords = getattr(getattr(rel, "Location", None), "Coordinates", None)
        try:
            return float(coords[2]) if coords and len(coords) > 2 else 0.0
        except Exception:
            return 0.0

    def _product_world_placement_z(self, product) -> Optional[float]:
        try:
            return float(_ifc_local_placement_summary(product).get("z", 0.0)) * self.unit_scale + self.dz
        except Exception:
            return None

    def _assigned_floor_name(
        self,
        product,
        floors: Dict[str, FloorModel],
        z_min: float,
        z_max: float,
        source_floor: Optional[str],
    ) -> str:
        """Choose the floor using host/storey elevation plus element placement offset.

        IFC containment alone is not reliable: some exports contain elements in
        a storey while their local placement has a vertical offset into another
        level. Geometry Z is also kept as a fallback for files where placement
        chains are incomplete.
        """
        if not floors:
            return source_floor or "Default"

        candidates: List[float] = []
        world_z = self._product_world_placement_z(product)
        source_host_z = None
        if source_floor in floors:
            source_host_z = float(floors[source_floor].elevation) + self._local_placement_z(product) * self.unit_scale
        if world_z is not None and math.isfinite(world_z):
            local_z = self._local_placement_z(product) * self.unit_scale
            if (
                source_host_z is not None
                and abs(float(world_z) - float(local_z)) <= 0.05
                and abs(float(source_host_z) - float(world_z)) > 0.50
            ):
                candidates.extend([source_host_z, world_z])
            else:
                candidates.append(world_z)
        if source_host_z is not None and source_host_z not in candidates:
            candidates.append(source_host_z)
        if math.isfinite(float(z_min)):
            candidates.append(float(z_min))
        if math.isfinite(float(z_max)):
            candidates.append(float(z_max))
        if not candidates:
            return source_floor if source_floor in floors else next(iter(floors))

        ordered = sorted(floors.values(), key=lambda floor: (floor.elevation, floor.name))

        def floor_for_z(value: float) -> str:
            return min(ordered, key=lambda floor: abs(float(floor.elevation) - float(value))).name

        # Placement is the strongest signal because it includes the host/storey
        # chain and any element offset from that host. Use geometry only when no
        # placement-derived candidate was available.
        return floor_for_z(candidates[0])

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
        return self._plan_polygon_from_vertices(verts)

    def _plan_polygons_from_geometry_iterator(self, products: List[object], max_threads: int = 1) -> Dict[str, Tuple[Polygon, float, float]]:
        if not products:
            return {}
        result: Dict[str, Tuple[Polygon, float, float]] = {}
        try:
            iterator = ifcopenshell.geom.iterator(
                self.settings,
                self.ifc,
                max(1, int(max_threads)),
                include=products,
            )
            if not iterator.initialize():
                return result
            while True:
                shape = iterator.get()
                guid = str(getattr(shape, "guid", "") or "")
                try:
                    verts = np.array(shape.geometry.verts, dtype=float).reshape((-1, 3))
                    if verts.size:
                        verts[:, 0] += self.dx
                        verts[:, 1] += self.dy
                        verts[:, 2] += self.dz
                        geom = self._plan_polygon_from_vertices(verts)
                        if geom is not None and guid:
                            result[guid] = geom
                except Exception:
                    pass
                if not iterator.next():
                    break
        except Exception:
            return {}
        return result

    @staticmethod
    def _plan_polygon_from_vertices(verts: np.ndarray) -> Optional[Tuple[Polygon, float, float]]:
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

    def _plan_polygon_from_element(self, product) -> Optional[Tuple[Polygon, float, float]]:
        """Return a plan footprint for imported context elements.

        Doors and windows in large architectural IFCs can have very detailed
        symbolic/3D representation geometry.  For RF/model context display, a
        placement-based panel footprint is enough and avoids minutes of OCC
        shape extraction.
        """
        fast = self._fast_panel_footprint(product)
        if fast is not None:
            return fast
        return self._plan_polygon_from_geometry(product)

    def _fast_panel_footprint(self, product) -> Optional[Tuple[Polygon, float, float]]:
        if not (product.is_a("IfcDoor") or product.is_a("IfcWindow")):
            return None
        try:
            width = float(getattr(product, "OverallWidth", 0.0) or 0.0) * self.unit_scale
            height = float(getattr(product, "OverallHeight", 0.0) or 0.0) * self.unit_scale
        except Exception:
            return None
        if width <= 0.01:
            return None
        depth = 0.10 if product.is_a("IfcWindow") else 0.16
        placement = _ifc_local_placement_summary(product)
        cx = float(placement.get("x", 0.0)) * self.unit_scale + self.dx
        cy = float(placement.get("y", 0.0)) * self.unit_scale + self.dy
        z_min = float(placement.get("z", 0.0)) * self.unit_scale + self.dz
        z_max = z_min + max(0.05, height)
        angle = math.radians(float(placement.get("rotation_from_x_deg", 0.0) or 0.0))
        ux, uy = math.cos(angle), math.sin(angle)
        vx, vy = -uy, ux
        half_w = width * 0.5
        half_d = depth * 0.5
        points = [
            (cx - ux * half_w - vx * half_d, cy - uy * half_w - vy * half_d),
            (cx + ux * half_w - vx * half_d, cy + uy * half_w - vy * half_d),
            (cx + ux * half_w + vx * half_d, cy + uy * half_w + vy * half_d),
            (cx - ux * half_w + vx * half_d, cy - uy * half_w + vy * half_d),
        ]
        polygon = Polygon(points)
        if polygon.is_empty or float(polygon.area) <= 1e-6:
            return None
        return polygon, z_min, z_max

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
    def _host_wall_guid_for_filling(product) -> str:
        """Return the wall GUID filled by an IfcDoor/IfcWindow when available."""
        try:
            for fill_relation in (getattr(product, "FillsVoids", None) or []):
                opening = getattr(fill_relation, "RelatingOpeningElement", None)
                if opening is None:
                    continue
                for void_relation in (getattr(opening, "VoidsElements", None) or []):
                    host = getattr(void_relation, "RelatingBuildingElement", None)
                    if host is None:
                        continue
                    if host.is_a("IfcWall") or host.is_a("IfcWallStandardCase"):
                        return str(getattr(host, "GlobalId", "") or "")
        except Exception:
            pass
        return ""

    @staticmethod
    def _rf_category_for_element(element) -> str:
        """Return a stable RF category for any imported IfcElement."""
        checks = (
            ("IfcDoor", "door"), ("IfcWindow", "window"), ("IfcSlab", "slab"),
            ("IfcRoof", "roof"), ("IfcColumn", "column"), ("IfcBeam", "beam"),
            ("IfcCurtainWall", "curtain_wall"), ("IfcCovering", "covering"),
            ("IfcPlate", "plate"), ("IfcMember", "member"), ("IfcPile", "pile"),
            ("IfcFooting", "footing"), ("IfcStair", "stair"), ("IfcRamp", "ramp"),
            ("IfcRailing", "railing"), ("IfcFurnishingElement", "furniture"),
            ("IfcBuildingElementProxy", "proxy"), ("IfcDistributionElement", "distribution"),
            ("IfcTransportElement", "transport"), ("IfcElementAssembly", "assembly"),
        )
        for ifc_name, category in checks:
            try:
                if element.is_a(ifc_name):
                    return category
            except Exception:
                continue
        class_name = str(getattr(element, "is_a", lambda: "IfcElement")() or "IfcElement")
        lowered = class_name.lower()
        if "equipment" in lowered or "terminal" in lowered or "device" in lowered:
            return "equipment"
        return "other"

    @staticmethod
    def _default_ifc_element_attenuation_profile(category: str, material: str, type_name: str, ifc_class: str = "") -> Dict[float, float]:
        """Conservative defaults for non-wall IFC element categories."""
        text = f"{category} {material} {type_name} {ifc_class}".lower()
        if category in {"distribution", "equipment", "furniture", "proxy", "transport", "assembly", "other"}:
            return {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0}
        if "metal" in text or "steel" in text:
            return {433.0: 12.0, 868.0: 16.0, 2400.0: 20.0, 5000.0: 28.0, 6000.0: 35.0}
        if "concrete" in text or "block" in text:
            return {433.0: 5.0, 868.0: 7.0, 2400.0: 12.0, 5000.0: 16.0, 6000.0: 20.0}
        if "brick" in text or "masonry" in text:
            return {433.0: 4.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 11.0, 6000.0: 14.0}
        if "glass" in text or category == "curtain_wall":
            return {433.0: 1.0, 868.0: 2.0, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0}
        category_profiles = {
            "slab": {433.0: 8.0, 868.0: 10.0, 2400.0: 12.0, 5000.0: 18.0, 6000.0: 22.0},
            "roof": {433.0: 5.0, 868.0: 7.0, 2400.0: 9.0, 5000.0: 13.0, 6000.0: 16.0},
            "column": {433.0: 3.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 12.0, 6000.0: 15.0},
            "beam": {433.0: 2.0, 868.0: 4.0, 2400.0: 6.0, 5000.0: 9.0, 6000.0: 12.0},
            "covering": {433.0: 0.5, 868.0: 0.8, 2400.0: 1.0, 5000.0: 1.5, 6000.0: 2.0},
            "plate": {433.0: 2.0, 868.0: 3.0, 2400.0: 5.0, 5000.0: 8.0, 6000.0: 10.0},
            "member": {433.0: 1.0, 868.0: 2.0, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 7.0},
        }
        return dict(category_profiles.get(category, {433.0: 0.0, 868.0: 0.0, 2400.0: 0.0, 5000.0: 0.0, 6000.0: 0.0}))

    @staticmethod
    def _default_opening_attenuation_profile(category: str, material: str, type_name: str) -> Dict[float, float]:
        """Planning assumptions for door/window penetration loss by RF band."""
        text = f"{material} {type_name}".lower()
        if category == "window":
            if any(key in text for key in ("low-e", "low e", "metallised", "metalized", "solar control", "coated")):
                return {433.0: 3.0, 868.0: 5.0, 2400.0: 8.0, 5000.0: 14.0, 6000.0: 18.0}
            if "triple" in text:
                return {433.0: 1.2, 868.0: 1.8, 2400.0: 3.0, 5000.0: 5.0, 6000.0: 6.0}
            if "double" in text:
                return {433.0: 1.0, 868.0: 1.5, 2400.0: 2.5, 5000.0: 4.0, 6000.0: 5.0}
            return {433.0: 0.8, 868.0: 1.2, 2400.0: 2.0, 5000.0: 3.0, 6000.0: 4.0}
        if category == "door":
            if any(key in text for key in ("steel", "metal", "fire", "security")):
                return {433.0: 8.0, 868.0: 10.0, 2400.0: 12.0, 5000.0: 18.0, 6000.0: 22.0}
            if "glass" in text or "glazed" in text:
                return {433.0: 0.8, 868.0: 1.2, 2400.0: 2.0, 5000.0: 3.0, 6000.0: 4.0}
            if any(key in text for key in ("timber", "wood")):
                return {433.0: 1.5, 868.0: 2.0, 2400.0: 3.0, 5000.0: 4.0, 6000.0: 5.0}
            return {433.0: 2.0, 868.0: 3.0, 2400.0: 4.0, 5000.0: 6.0, 6000.0: 8.0}
        return {}

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


def _wall_axis_from_polygon(polygon: Polygon) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float, float]]:
    """Return centre, unit long-axis, length and width for a rectangular wall footprint."""
    if polygon is None or polygon.is_empty:
        return None
    try:
        rectangle = polygon.minimum_rotated_rectangle
        coords = list(rectangle.exterior.coords)
    except Exception:
        return None
    if len(coords) < 5:
        return None
    edges: List[Tuple[float, Tuple[float, float]]] = []
    for first, second in zip(coords, coords[1:]):
        dx = float(second[0]) - float(first[0])
        dy = float(second[1]) - float(first[1])
        edge_length = math.hypot(dx, dy)
        if edge_length > 1e-9:
            edges.append((edge_length, (dx / edge_length, dy / edge_length)))
    if not edges:
        return None
    lengths = sorted(edge_length for edge_length, _ in edges)
    width = float(lengths[0])
    length = float(lengths[-1])
    if length <= 1e-9 or width <= 1e-9:
        return None
    axis = max(edges, key=lambda item: item[0])[1]
    centroid = rectangle.centroid
    return (float(centroid.x), float(centroid.y)), axis, length, width


def _close_wall_endpoint_gaps_on_floor(floor: FloorModel, deadline: Optional[float] = None) -> int:
    """Extend imported wall rectangles across small endpoint gaps on one floor."""
    walls = [
        wall for wall in getattr(floor, "walls", []) or []
        if not getattr(wall, "is_user_created", False)
        and wall.polygon is not None
        and not wall.polygon.is_empty
        and float(getattr(wall.polygon, "area", 0.0)) > 1e-6
    ]
    if len(walls) < 2:
        return 0

    axes: Dict[int, Tuple[Tuple[float, float], Tuple[float, float], float, float]] = {}
    widths: List[float] = []
    for index, wall in enumerate(walls):
        axis = _wall_axis_from_polygon(wall.polygon)
        if axis is None:
            continue
        axes[index] = axis
        width = axis[3]
        if 0.03 <= width <= 3.0:
            widths.append(width)
    if not axes:
        return 0

    typical_width = float(np.median(widths)) if widths else 0.20
    touch_tolerance = max(0.005, min(0.03, typical_width * 0.10))
    max_extension = max(0.08, min(0.60, typical_width * 2.5))
    extensions: Dict[int, List[float]] = {}
    wall_indices = list(axes.keys())
    try:
        from shapely.strtree import STRtree
        indexed_polygons = [walls[index].polygon for index in wall_indices]
        wall_tree = STRtree(indexed_polygons)
        geometry_to_index = {id(polygon): index for index, polygon in zip(wall_indices, indexed_polygons)}
    except Exception:
        wall_tree = None
        indexed_polygons = []
        geometry_to_index = {}

    def nearby_wall_indices(corridor) -> List[int]:
        if wall_tree is None:
            # Avoid locking the GUI on very large floors if spatial indexing is
            # unavailable for any reason.
            return wall_indices if len(wall_indices) <= 250 else []
        try:
            hits = wall_tree.query(corridor)
        except Exception:
            return []
        result: List[int] = []
        for hit in hits:
            if isinstance(hit, (int, np.integer)):
                pos = int(hit)
                if 0 <= pos < len(wall_indices):
                    result.append(wall_indices[pos])
            else:
                index = geometry_to_index.get(id(hit))
                if index is not None:
                    result.append(index)
        return result

    def endpoint_extension(index: int, sign: float) -> float:
        wall = walls[index]
        centre, unit, length, width = axes[index]
        ux, uy = unit
        half = length * 0.5
        ex = centre[0] + ux * half * sign
        ey = centre[1] + uy * half * sign
        direction = (ux * sign, uy * sign)
        endpoint = Point(ex, ey)
        cap = LineString([
            (ex - uy * width * 0.55, ey + ux * width * 0.55),
            (ex + uy * width * 0.55, ey - ux * width * 0.55),
        ])
        best_extension = 0.0
        best_distance = max_extension + 1.0
        corridor = LineString([
            (ex, ey),
            (ex + direction[0] * max_extension, ey + direction[1] * max_extension),
        ]).buffer(max(width * 0.60, typical_width * 0.35, 0.04), cap_style=2, join_style=2)

        for other_index in nearby_wall_indices(corridor):
            if other_index == index:
                continue
            other = walls[other_index]
            other_polygon = other.polygon
            try:
                if other_polygon.distance(cap) <= touch_tolerance or other_polygon.distance(endpoint) <= touch_tolerance:
                    continue
                if not other_polygon.intersects(corridor):
                    continue
                _, nearest = nearest_points(endpoint, other_polygon)
            except Exception:
                continue
            vx = float(nearest.x) - ex
            vy = float(nearest.y) - ey
            along = vx * direction[0] + vy * direction[1]
            if along <= touch_tolerance or along > max_extension:
                continue
            perpendicular = abs(vx * (-direction[1]) + vy * direction[0])
            if perpendicular > max(width * 0.75, typical_width * 0.50, 0.06):
                continue

            other_axis = axes.get(other_index)
            split_gap = False
            if other_axis is not None:
                other_unit = other_axis[1]
                if abs(unit[0] * other_unit[0] + unit[1] * other_unit[1]) >= 0.92:
                    split_gap = True
            extension = along * 0.5 if split_gap else along
            if extension < best_distance:
                best_distance = extension
                best_extension = extension
        return best_extension

    for index in axes:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        backward = endpoint_extension(index, -1.0)
        if deadline is not None and time.perf_counter() >= deadline:
            break
        forward = endpoint_extension(index, 1.0)
        if backward > touch_tolerance or forward > touch_tolerance:
            extensions[index] = [backward, forward]

    changed = 0
    for index, (backward, forward) in extensions.items():
        if deadline is not None and time.perf_counter() >= deadline:
            break
        centre, unit, length, width = axes[index]
        ux, uy = unit
        half = length * 0.5
        start = (centre[0] - ux * (half + backward), centre[1] - uy * (half + backward))
        end = (centre[0] + ux * (half + forward), centre[1] + uy * (half + forward))
        try:
            expanded = LineString([start, end]).buffer(width * 0.5, cap_style=2, join_style=2)
            if not expanded.is_valid:
                expanded = expanded.buffer(0)
            if expanded.is_empty or float(expanded.area) <= float(walls[index].polygon.area):
                continue
            walls[index].polygon = expanded
            changed += 1
        except Exception:
            continue
    return changed


def close_imported_wall_endpoint_gaps(floors: Dict[str, FloorModel], max_seconds: float = 4.0) -> int:
    """Close small endpoint gaps between imported wall footprints across all floors."""
    deadline = time.perf_counter() + max(0.25, float(max_seconds))
    total = 0
    for floor in floors.values():
        if time.perf_counter() >= deadline:
            break
        total += _close_wall_endpoint_gaps_on_floor(floor, deadline=deadline)
    return total


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
            for element in getattr(inc_floor, "elements", []):
                element.floor = key
                element.guid = f"{source_name}:{element.guid}"
                if getattr(element, "host_wall_guid", ""):
                    element.host_wall_guid = f"{source_name}:{element.host_wall_guid}"
                target[key].elements.append(element)
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
    @functools.lru_cache(maxsize=64)
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
    def _link_geometry_key(x: float, y: float, receiver_floor: FloorModel, ap: AccessPoint) -> tuple:
        return (
            str(ap.name), str(ap.floor), round(float(ap.x), 6), round(float(ap.y), 6),
            round(float(ap.mount_height_m), 4), round(float(ap.rx_height_m), 4),
            round(float(ap.azimuth_deg), 3), round(float(ap.downtilt_deg), 3),
            str(receiver_floor.name), round(float(x), 6), round(float(y), 6),
        )

    @staticmethod
    def _direct_link_geometry(
        x: float,
        y: float,
        receiver_floor: FloorModel,
        ap: AccessPoint,
        floors: Dict[str, FloorModel],
        include_inter_floor: bool,
        wall_indexes=None,
        opening_indexes=None,
        geometry_cache: Optional[Dict[object, object]] = None,
    ) -> Optional[Dict[str, object]]:
        key = ("direct",) + RFEngine._link_geometry_key(x, y, receiver_floor, ap)
        if geometry_cache is not None and key in geometry_cache:
            return geometry_cache[key]
        ap_floor = floors.get(ap.floor)
        if ap_floor is None or (ap.floor != receiver_floor.name and not include_inter_floor):
            return None
        horizontal_d = max(math.hypot(float(x) - ap.x, float(y) - ap.y), 0.1)
        ap_z = float(ap_floor.elevation) + float(ap.mount_height_m)
        rx_z = float(receiver_floor.elevation) + float(ap.rx_height_m)
        d_3d = max(math.hypot(horizontal_d, ap_z - rx_z), 1.0)
        bearing = math.degrees(math.atan2(float(y) - ap.y, float(x) - ap.x))
        elev_angle = math.degrees(math.atan2(rx_z - ap_z, horizontal_d))
        line = LineString([(ap.x, ap.y), (float(x), float(y))])
        crossed_walls: List[Wall2D] = []
        crossed_elements: List[IFCElement2D] = []
        crossed_slabs: List[IFCElement2D] = []
        checked_walls = set(); checked_elements = set()
        for path_floor in RFEngine.floors_between_inclusive(receiver_floor, ap_floor, floors):
            active_openings: List[IFCElement2D] = []
            for element in RFEngine._openings_intersecting_line(path_floor, line, opening_indexes):
                element_key = RFEngine._element_identity(element)
                if element_key in checked_elements or not element.polygon.intersects(line):
                    continue
                if not RFEngine._element_intersects_3d_path(element, line, ap_z, rx_z):
                    continue
                checked_elements.add(element_key)
                category = str(getattr(element, "rf_category", "other") or "other").lower()
                if category in {"door", "window"}:
                    active_openings.append(element); crossed_elements.append(element)
                elif category in {"slab", "roof"}:
                    crossed_slabs.append(element)
                else:
                    crossed_elements.append(element)
            for wall in RFEngine._walls_intersecting_line(path_floor, line, wall_indexes):
                wall_key = RFEngine._wall_identity(wall)
                if wall_key in checked_walls or not wall.polygon.intersects(line):
                    continue
                if any(RFEngine._opening_replaces_wall(opening, wall) for opening in active_openings):
                    continue
                crossed_walls.append(wall); checked_walls.add(wall_key)
        geometry = {
            "ap_floor": ap_floor,
            "horizontal_d": horizontal_d,
            "ap_z": ap_z,
            "rx_z": rx_z,
            "distance_3d": d_3d,
            "bearing": bearing,
            "elevation": elev_angle,
            "line": line,
            "walls": tuple(crossed_walls),
            "elements": tuple(crossed_elements),
            "slabs": tuple(crossed_slabs),
        }
        if geometry_cache is not None:
            geometry_cache[key] = geometry
        return geometry

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
        wall_indexes=None,
        opening_indexes=None,
        geometry_cache: Optional[Dict[object, object]] = None,
    ) -> float:
        """Calculate direct 3D RSSI while reusing frequency-independent link geometry."""
        radio = radio or ap.active_radios()[0]
        disconnected = float(getattr(heatmap_settings, "disconnected_rssi_dbm", -120.0) if heatmap_settings else -120.0)
        if not RFEngine.point_is_inside_radio_cutoff(x, y, receiver_floor, ap, floors, radio, heatmap_settings):
            return disconnected
        geometry = RFEngine._direct_link_geometry(
            x, y, receiver_floor, ap, floors, include_inter_floor,
            wall_indexes, opening_indexes, geometry_cache,
        )
        if geometry is None:
            return disconnected
        reference_loss = RFEngine.free_space_loss_db_at_1m(radio.frequency_mhz)
        path_loss = reference_loss + 10.0 * ap.path_loss_exponent * math.log10(float(geometry["distance_3d"]))
        az_rel = AntennaPattern._wrap_deg(float(geometry["bearing"]) - ap.azimuth_deg)
        elev_rel = float(geometry["elevation"]) + ap.downtilt_deg
        pattern_gain = 0.0
        if patterns:
            pattern = patterns.get(radio.antenna_pattern)
            if pattern:
                pattern_gain = pattern.gain_dbi(az_rel, elev_rel)
        wall_loss = sum(wall.attenuation_db_for_frequency(radio.frequency_mhz) for wall in geometry["walls"])
        element_loss = sum(element.attenuation_db_for_frequency(radio.frequency_mhz) for element in geometry["elements"])
        floor_loss = RFEngine.floor_penetration_loss_db(
            receiver_floor, geometry["ap_floor"], floors, radio.frequency_mhz, list(geometry["slabs"])
        )
        return (
            radio.tx_power_dbm + pattern_gain + float(getattr(radio, "antenna_gain_dbi", 0.0) or 0.0)
            - path_loss - wall_loss - element_loss - floor_loss
        )

    @staticmethod
    def _wall_identity(wall: Wall2D) -> str:
        return str(wall.guid or f"{wall.source_file}:{wall.name}:{wall.z_min:.3f}:{wall.z_max:.3f}")

    @staticmethod
    def _element_identity(element: IFCElement2D) -> str:
        return str(element.guid or f"{element.source_file}:{element.name}:{element.z_min:.3f}:{element.z_max:.3f}")

    @staticmethod
    def _reflection_surfaces_from_polygon(
        polygon,
        material: str,
        object_id: str,
        category: str,
    ) -> List[ReflectionSurface]:
        surfaces: List[ReflectionSurface] = []
        if polygon is None or getattr(polygon, "is_empty", True):
            return surfaces
        if getattr(polygon, "geom_type", "") != "Polygon":
            for part in getattr(polygon, "geoms", []):
                surfaces.extend(RFEngine._reflection_surfaces_from_polygon(
                    part, material, object_id, category
                ))
            return surfaces
        try:
            rings = [polygon.exterior]
            rings.extend(list(getattr(polygon, "interiors", [])))
            for ring in rings:
                coords = list(ring.coords)
                for a, b in zip(coords, coords[1:]):
                    start = (float(a[0]), float(a[1]))
                    end = (float(b[0]), float(b[1]))
                    if math.hypot(end[0] - start[0], end[1] - start[1]) < 0.25:
                        continue
                    surfaces.append(ReflectionSurface(
                        a=start,
                        b=end,
                        material=str(material or "default"),
                        object_id=str(object_id),
                        category=str(category or "wall"),
                    ))
        except Exception:
            return []
        return surfaces

    @staticmethod
    def _build_reflection_indexes(
        floors: Dict[str, FloorModel],
        heatmap_settings: Optional[HeatmapSettings],
    ) -> Dict[str, ReflectionIndex]:
        indexes: Dict[str, ReflectionIndex] = {}
        enabled = bool(getattr(heatmap_settings, "enable_multipath_reflections", False)) or bool(
            getattr(heatmap_settings, "enable_corner_diffraction", False)
        )
        if not enabled:
            return indexes
        for floor in floors.values():
            surfaces: List[ReflectionSurface] = []
            for wall in list(getattr(floor, "walls", []) or []):
                material = wall.rf_type_override or wall.material or wall.type_name or "default"
                surfaces.extend(RFEngine._reflection_surfaces_from_polygon(
                    wall.polygon,
                    material,
                    RFEngine._wall_identity(wall),
                    "wall",
                ))
            for element in list(getattr(floor, "elements", []) or []):
                category = str(getattr(element, "rf_category", "other") or "other").lower()
                if category in {"slab", "roof", "covering"}:
                    continue
                # Generic IFC objects only become reflecting obstacles when they
                # carry an RF profile or represent substantial structure.
                if not getattr(element, "is_rf_barrier", False) and category not in {
                    "column", "beam", "curtain_wall", "plate", "member"
                }:
                    continue
                material = element.rf_type_override or element.material or element.type_name or element.ifc_class or "default"
                surfaces.extend(RFEngine._reflection_surfaces_from_polygon(
                    element.polygon,
                    material,
                    RFEngine._element_identity(element),
                    category,
                ))
            if surfaces:
                indexes[floor.name] = ReflectionIndex(surfaces)
        return indexes

    @staticmethod
    def _path_penetration_loss_same_floor(
        points: List[Tuple[float, float]],
        floor: FloorModel,
        radio: APRadio,
        ap_z: float,
        rx_z: float,
        wall_indexes: Optional[Dict[str, Tuple[object, List[Wall2D], Dict[int, Wall2D]]]] = None,
        opening_indexes: Optional[Dict[str, Tuple[object, List[IFCElement2D], Dict[int, IFCElement2D]]]] = None,
        excluded_object_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[float, float]:
        if len(points) < 2:
            return 0.0, 0.0
        excluded = {str(value) for value in (excluded_object_ids or [])}
        segment_lengths = [
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points, points[1:])
        ]
        total_length = max(1e-9, sum(segment_lengths))
        cumulative = 0.0
        checked_walls = set()
        checked_elements = set()
        wall_loss = 0.0
        element_loss = 0.0

        for (start, end), segment_length in zip(zip(points, points[1:]), segment_lengths):
            if segment_length <= 1e-9:
                continue
            start_fraction = cumulative / total_length
            end_fraction = (cumulative + segment_length) / total_length
            segment_ap_z = float(ap_z) + (float(rx_z) - float(ap_z)) * start_fraction
            segment_rx_z = float(ap_z) + (float(rx_z) - float(ap_z)) * end_fraction
            cumulative += segment_length
            line = LineString([start, end])

            active_openings: List[IFCElement2D] = []
            for element in RFEngine._openings_intersecting_line(floor, line, opening_indexes):
                element_key = RFEngine._element_identity(element)
                if element_key in excluded or element_key in checked_elements:
                    continue
                if not element.polygon.intersects(line):
                    continue
                if not RFEngine._element_intersects_3d_path(element, line, segment_ap_z, segment_rx_z):
                    continue
                checked_elements.add(element_key)
                category = str(getattr(element, "rf_category", "other") or "other").lower()
                if category in {"door", "window"}:
                    active_openings.append(element)
                    element_loss += element.attenuation_db_for_frequency(radio.frequency_mhz)
                elif category not in {"slab", "roof"}:
                    element_loss += element.attenuation_db_for_frequency(radio.frequency_mhz)

            for wall in RFEngine._walls_intersecting_line(floor, line, wall_indexes):
                wall_key = RFEngine._wall_identity(wall)
                if wall_key in excluded or wall_key in checked_walls:
                    continue
                if not wall.polygon.intersects(line):
                    continue
                if not RFEngine._element_intersects_3d_path(
                    IFCElement2D(
                        guid=wall.guid,
                        name=wall.name,
                        floor=wall.floor,
                        source_file=wall.source_file,
                        type_name=wall.type_name,
                        material=wall.material,
                        polygon=wall.polygon,
                        z_min=wall.z_min,
                        z_max=wall.z_max,
                    ),
                    line,
                    segment_ap_z,
                    segment_rx_z,
                ):
                    continue
                if any(RFEngine._opening_replaces_wall(opening, wall) for opening in active_openings):
                    continue
                wall_loss += wall.attenuation_db_for_frequency(radio.frequency_mhz)
                checked_walls.add(wall_key)
        return wall_loss, element_loss

    @staticmethod
    def _direct_blocking_object_ids(
        source: Tuple[float, float],
        receiver: Tuple[float, float],
        floor: FloorModel,
        ap_z: float,
        rx_z: float,
        wall_indexes: Optional[Dict[str, Tuple[object, List[Wall2D], Dict[int, Wall2D]]]] = None,
    ) -> List[str]:
        line = LineString([source, receiver])
        blocked: List[str] = []
        for wall in RFEngine._walls_intersecting_line(floor, line, wall_indexes):
            if not wall.polygon.intersects(line):
                continue
            wall_as_element = IFCElement2D(
                guid=wall.guid,
                name=wall.name,
                floor=wall.floor,
                source_file=wall.source_file,
                type_name=wall.type_name,
                material=wall.material,
                polygon=wall.polygon,
                z_min=wall.z_min,
                z_max=wall.z_max,
            )
            if RFEngine._element_intersects_3d_path(wall_as_element, line, ap_z, rx_z):
                blocked.append(RFEngine._wall_identity(wall))
        return blocked

    @staticmethod
    def _departure_pattern_gain(
        ap: AccessPoint,
        radio: APRadio,
        patterns: Optional[Dict[str, AntennaPattern]],
        first_point: Tuple[float, float],
        total_horizontal_length: float,
        ap_z: float,
        rx_z: float,
    ) -> float:
        bearing = math.degrees(math.atan2(first_point[1] - ap.y, first_point[0] - ap.x))
        az_rel = AntennaPattern._wrap_deg(bearing - ap.azimuth_deg)
        elevation = math.degrees(math.atan2(float(rx_z) - float(ap_z), max(0.1, total_horizontal_length)))
        elev_rel = elevation + ap.downtilt_deg
        if not patterns:
            return 0.0
        pattern = patterns.get(radio.antenna_pattern)
        return pattern.gain_dbi(az_rel, elev_rel) if pattern else 0.0

    @staticmethod
    def propagation_at(
        x: float,
        y: float,
        receiver_floor: FloorModel,
        ap: AccessPoint,
        floors: Dict[str, FloorModel],
        patterns: Optional[Dict[str, AntennaPattern]] = None,
        radio: Optional[APRadio] = None,
        include_inter_floor: bool = True,
        heatmap_settings: Optional[HeatmapSettings] = None,
        wall_indexes: Optional[Dict[str, Tuple[object, List[Wall2D], Dict[int, Wall2D]]]] = None,
        opening_indexes: Optional[Dict[str, Tuple[object, List[IFCElement2D], Dict[int, IFCElement2D]]]] = None,
        reflection_indexes: Optional[Dict[str, ReflectionIndex]] = None,
        geometry_cache: Optional[Dict[object, object]] = None,
        reflection_index_override: Optional[ReflectionIndex] = None,
        reflection_sequences_override: Optional[Sequence[Sequence[ReflectionSurface]]] = None,
    ) -> PropagationSample:
        radio = radio or ap.active_radios()[0]
        disconnected = float(getattr(heatmap_settings, "disconnected_rssi_dbm", -120.0) if heatmap_settings else -120.0)
        ap_floor = floors.get(ap.floor)
        if ap_floor is None:
            return PropagationSample(disconnected, 0.0, 0, disconnected)
        if ap.floor != receiver_floor.name and not include_inter_floor:
            return PropagationSample(disconnected, 0.0, 0, disconnected)
        if not RFEngine.point_is_inside_radio_cutoff(
            x, y, receiver_floor, ap, floors, radio, heatmap_settings
        ):
            return PropagationSample(disconnected, 0.0, 0, disconnected)
        direct_rssi = RFEngine.rssi_at(
            x,
            y,
            receiver_floor,
            ap,
            floors,
            patterns,
            radio,
            include_inter_floor,
            heatmap_settings,
            wall_indexes,
            opening_indexes,
            geometry_cache,
        )
        source = (float(ap.x), float(ap.y))
        receiver = (float(x), float(y))
        horizontal_direct = max(0.1, math.hypot(receiver[0] - source[0], receiver[1] - source[1]))
        ap_z = float(ap_floor.elevation) + float(ap.mount_height_m)
        rx_z = float(receiver_floor.elevation) + float(ap.rx_height_m)
        direct_length_3d = max(1.0, math.hypot(horizontal_direct, ap_z - rx_z))
        path_powers: List[PathPower] = [PathPower(
            power_dbm=float(direct_rssi),
            length_m=direct_length_3d,
            phase_rad=propagation_phase_rad(direct_length_3d, radio.frequency_mhz),
            kind="direct",
        )]

        same_floor = ap_floor.name == receiver_floor.name
        index = reflection_index_override or (reflection_indexes or {}).get(receiver_floor.name)
        material_profiles = getattr(heatmap_settings, "reflection_material_properties", {}) or {}
        reference_loss = RFEngine.free_space_loss_db_at_1m(float(radio.frequency_mhz))
        radio_gain = float(getattr(radio, "antenna_gain_dbi", 0.0) or 0.0)
        if same_floor and index is not None and heatmap_settings is not None:
            if bool(getattr(heatmap_settings, "enable_multipath_reflections", False)):
                geometry_key = (
                    "reflections", id(index), id(reflection_sequences_override),
                    int(getattr(heatmap_settings, "max_reflection_order", 1)),
                    int(getattr(heatmap_settings, "max_reflection_surfaces", 6)),
                    int(getattr(heatmap_settings, "max_reflection_paths", 8)),
                    round(float(getattr(heatmap_settings, "reflection_search_radius_m", 18.0)), 3),
                ) + RFEngine._link_geometry_key(x, y, receiver_floor, ap)
                reflected_geometries = geometry_cache.get(geometry_key) if geometry_cache is not None else None
                if reflected_geometries is None:
                    reflected_geometries = generate_reflection_geometries(
                        source, receiver, index,
                        int(getattr(heatmap_settings, "max_reflection_order", 1)),
                        int(getattr(heatmap_settings, "max_reflection_surfaces", 6)),
                        int(getattr(heatmap_settings, "max_reflection_paths", 8)),
                        float(getattr(heatmap_settings, "reflection_search_radius_m", 18.0)),
                        reflection_sequences_override,
                    )
                    if geometry_cache is not None:
                        geometry_cache[geometry_key] = reflected_geometries
                reflected_paths = [
                    ray for ray in (
                        evaluate_reflection_geometry(
                            geometry, radio.frequency_mhz, material_profiles,
                            float(getattr(heatmap_settings, "minimum_reflection_coefficient", 0.025)),
                        )
                        for geometry in reflected_geometries
                    ) if ray is not None
                ]
                for ray in reflected_paths:
                    horizontal_length = max(0.1, float(ray.length_m))
                    distance_3d = max(1.0, math.hypot(horizontal_length, ap_z - rx_z))
                    path_loss = reference_loss + 10.0 * ap.path_loss_exponent * math.log10(distance_3d)
                    pattern_gain = RFEngine._departure_pattern_gain(
                        ap, radio, patterns, ray.points[1], horizontal_length, ap_z, rx_z
                    )
                    wall_loss, element_loss = RFEngine._path_penetration_loss_same_floor(
                        list(ray.points),
                        receiver_floor,
                        radio,
                        ap_z,
                        rx_z,
                        wall_indexes,
                        opening_indexes,
                        ray.interacted_object_ids,
                    )
                    coefficient_magnitude = max(1e-12, abs(ray.coefficient))
                    coefficient_gain_db = 20.0 * math.log10(coefficient_magnitude)
                    received = (
                        radio.tx_power_dbm
                        + pattern_gain
                        + radio_gain
                        - path_loss
                        - wall_loss
                        - element_loss
                        + coefficient_gain_db
                        - float(ray.extra_loss_db)
                    )
                    if received < disconnected - 60.0:
                        continue
                    path_powers.append(PathPower(
                        power_dbm=received,
                        length_m=distance_3d,
                        phase_rad=propagation_phase_rad(
                            distance_3d,
                            radio.frequency_mhz,
                            cmath.phase(ray.coefficient),
                        ),
                        kind=ray.kind,
                    ))

            if bool(getattr(heatmap_settings, "enable_corner_diffraction", False)):
                blocked_key = (
                    "direct_blockers", id(wall_indexes),
                ) + RFEngine._link_geometry_key(x, y, receiver_floor, ap)
                blocked_ids = geometry_cache.get(blocked_key) if geometry_cache is not None else None
                if blocked_ids is None:
                    blocked_ids = RFEngine._direct_blocking_object_ids(
                        source, receiver, receiver_floor, ap_z, rx_z, wall_indexes
                    )
                    if geometry_cache is not None:
                        geometry_cache[blocked_key] = blocked_ids
                if blocked_ids:
                    diffraction_key = (
                        "diffraction", id(index), tuple(sorted(blocked_ids)),
                        int(getattr(heatmap_settings, "max_diffraction_paths", 3)),
                        round(float(getattr(heatmap_settings, "diffraction_search_radius_m", 5.0)), 3),
                    ) + RFEngine._link_geometry_key(x, y, receiver_floor, ap)
                    diffraction_geometries = geometry_cache.get(diffraction_key) if geometry_cache is not None else None
                    if diffraction_geometries is None:
                        diffraction_geometries = generate_diffraction_geometries(
                            source, receiver, index,
                            int(getattr(heatmap_settings, "max_diffraction_paths", 3)),
                            float(getattr(heatmap_settings, "diffraction_search_radius_m", 5.0)),
                            blocked_ids,
                        )
                        if geometry_cache is not None:
                            geometry_cache[diffraction_key] = diffraction_geometries
                    diffracted_paths = [
                        evaluate_diffraction_geometry(
                            geometry, radio.frequency_mhz,
                            float(getattr(heatmap_settings, "minimum_diffraction_loss_db", 6.0)),
                        )
                        for geometry in diffraction_geometries
                    ]
                    for ray in diffracted_paths:
                        horizontal_length = max(0.1, float(ray.length_m))
                        distance_3d = max(1.0, math.hypot(horizontal_length, ap_z - rx_z))
                        path_loss = reference_loss + 10.0 * ap.path_loss_exponent * math.log10(distance_3d)
                        pattern_gain = RFEngine._departure_pattern_gain(
                            ap, radio, patterns, ray.points[1], horizontal_length, ap_z, rx_z
                        )
                        wall_loss, element_loss = RFEngine._path_penetration_loss_same_floor(
                            list(ray.points),
                            receiver_floor,
                            radio,
                            ap_z,
                            rx_z,
                            wall_indexes,
                            opening_indexes,
                            ray.interacted_object_ids,
                        )
                        received = (
                            radio.tx_power_dbm
                            + pattern_gain
                            + radio_gain
                            - path_loss
                            - wall_loss
                            - element_loss
                            - float(ray.extra_loss_db)
                        )
                        if received < disconnected - 60.0:
                            continue
                        path_powers.append(PathPower(
                            power_dbm=received,
                            length_m=distance_3d,
                            phase_rad=propagation_phase_rad(
                                distance_3d,
                                radio.frequency_mhz,
                                cmath.phase(ray.coefficient),
                            ),
                            kind=ray.kind,
                        ))

        relative_cutoff = float(getattr(heatmap_settings, "multipath_relative_power_cutoff_db", 30.0) or 30.0) if heatmap_settings is not None else 30.0
        if len(path_powers) > 1 and relative_cutoff >= 0.0:
            strongest_power = max(path.power_dbm for path in path_powers)
            retained = [path for path in path_powers if path.kind == "direct" or path.power_dbm >= strongest_power - relative_cutoff]
            path_powers = sorted(retained, key=lambda path: path.power_dbm, reverse=True)[: max(1, int(getattr(heatmap_settings, "max_reflection_paths", 8)) + int(getattr(heatmap_settings, "max_diffraction_paths", 3)) + 1)]

        if len(path_powers) == 1:
            combined_rssi = float(path_powers[0].power_dbm)
            compiled_delay_spread = 0.0
        else:
            combined_rssi, compiled_delay_spread = coherent_path_metrics(
                [path.power_dbm for path in path_powers],
                [path.phase_rad for path in path_powers],
                [path.length_m for path in path_powers],
                disconnected,
                bool(getattr(heatmap_settings, "enable_numba_rf_kernels", True)) if heatmap_settings is not None else True,
            )
        if heatmap_settings is not None and bool(getattr(heatmap_settings, "enable_small_scale_fading", False)):
            combined_rssi += deterministic_spatial_fading_db(
                x,
                y,
                float(getattr(heatmap_settings, "fading_correlation_distance_m", 0.75)),
                float(getattr(heatmap_settings, "small_scale_fading_sigma_db", 1.5)),
                int(getattr(heatmap_settings, "fading_seed", 1729)),
                stable_link_key(ap.name, radio.frequency_mhz, getattr(radio, "channel", "")),
            )
        combined_rssi = max(disconnected, float(combined_rssi))
        delay_spread = (
            compiled_delay_spread
            if heatmap_settings is not None and bool(getattr(heatmap_settings, "calculate_delay_spread", True))
            else 0.0
        )
        return PropagationSample(
            rssi_dbm=combined_rssi,
            delay_spread_ns=delay_spread,
            path_count=len(path_powers),
            direct_rssi_dbm=float(direct_rssi),
        )

    @staticmethod
    def combine_ap_samples(
        samples: List[PropagationSample],
        heatmap_settings: Optional[HeatmapSettings],
        disconnected: float,
    ) -> PropagationSample:
        if not samples:
            return PropagationSample(float(disconnected), 0.0, 0, float(disconnected))
        strongest = max(samples, key=lambda sample: sample.rssi_dbm)
        mode = str(getattr(heatmap_settings, "combined_ap_mode", "strongest") or "strongest").lower()
        if mode == "power_sum":
            rssi = incoherent_power_dbm([sample.rssi_dbm for sample in samples], disconnected)
            return PropagationSample(
                rssi_dbm=max(float(disconnected), rssi),
                delay_spread_ns=float(strongest.delay_spread_ns),
                path_count=sum(max(0, int(sample.path_count)) for sample in samples),
                direct_rssi_dbm=float(strongest.direct_rssi_dbm),
            )
        return strongest

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
    def floor_penetration_loss_db(
        receiver_floor: FloorModel,
        ap_floor: FloorModel,
        floors: Dict[str, FloorModel],
        frequency_mhz: float,
        slab_elements: Optional[List[IFCElement2D]] = None,
    ) -> float:
        if receiver_floor.name == ap_floor.name:
            return 0.0
        ordered = sorted(floors.values(), key=lambda f: (f.elevation, f.name))
        try:
            rx_i = next(i for i, f in enumerate(ordered) if f.name == receiver_floor.name)
            ap_i = next(i for i, f in enumerate(ordered) if f.name == ap_floor.name)
        except StopIteration:
            crossed = max(1, int(round(abs(receiver_floor.elevation - ap_floor.elevation) / 3.5)))
            return crossed * receiver_floor.slab_attenuation_db_for_frequency(frequency_mhz)
        lo, hi = sorted((rx_i, ap_i))
        crossed_boundaries = ordered[lo + 1:hi + 1] or [receiver_floor]
        candidates = list(slab_elements or [])
        total = 0.0
        used = set()
        for boundary in crossed_boundaries:
            boundary_z = float(boundary.elevation)
            matches = []
            for element in candidates:
                identity = (element.source_file, element.guid, element.floor)
                if identity in used:
                    continue
                z_min = min(float(element.z_min), float(element.z_max))
                z_max = max(float(element.z_min), float(element.z_max))
                centre = (z_min + z_max) / 2.0
                distance = 0.0 if z_min - 0.35 <= boundary_z <= z_max + 0.35 else abs(centre - boundary_z)
                if distance <= 1.25:
                    matches.append((distance, -element.attenuation_db_for_frequency(frequency_mhz), identity, element))
            if matches:
                _, _, identity, element = min(matches)
                used.add(identity)
                total += element.attenuation_db_for_frequency(frequency_mhz)
            else:
                total += boundary.slab_attenuation_db_for_frequency(frequency_mhz)
        return total

    @staticmethod
    def _active_radio_links(
        receiver_floor: FloorModel,
        aps: List[AccessPoint],
        include_inter_floor: bool,
    ) -> List[Tuple[AccessPoint, APRadio]]:
        links: List[Tuple[AccessPoint, APRadio]] = []
        for ap in aps:
            if not include_inter_floor and ap.floor != receiver_floor.name:
                continue
            for radio in ap.active_radios():
                links.append((ap, radio))
        return links

    @staticmethod
    def _build_wall_indexes(floors: Dict[str, FloorModel]) -> Dict[str, Tuple[object, List[Wall2D], Dict[int, Wall2D]]]:
        indexes: Dict[str, Tuple[object, List[Wall2D], Dict[int, Wall2D]]] = {}
        try:
            from shapely.strtree import STRtree
        except Exception:
            return indexes
        for floor in floors.values():
            walls = list(getattr(floor, "walls", []) or [])
            if not walls:
                continue
            try:
                tree = STRtree([wall.polygon for wall in walls])
                by_geometry_id = {id(wall.polygon): wall for wall in walls}
                indexes[floor.name] = (tree, walls, by_geometry_id)
            except Exception:
                continue
        return indexes

    @staticmethod
    def _walls_intersecting_line(
        floor: FloorModel,
        line: LineString,
        wall_indexes: Optional[Dict[str, Tuple[object, List[Wall2D], Dict[int, Wall2D]]]] = None,
    ) -> Iterable[Wall2D]:
        if wall_indexes:
            indexed = wall_indexes.get(floor.name)
            if indexed is not None:
                tree, walls, by_geometry_id = indexed
                try:
                    hits = tree.query(line)
                    result: List[Wall2D] = []
                    for hit in hits:
                        if isinstance(hit, (int, np.integer)):
                            idx = int(hit)
                            if 0 <= idx < len(walls):
                                result.append(walls[idx])
                        else:
                            wall = by_geometry_id.get(id(hit))
                            if wall is not None:
                                result.append(wall)
                    return result
                except Exception:
                    pass
        return list(getattr(floor, "walls", []) or [])

    @staticmethod
    def _build_opening_indexes(
        floors: Dict[str, FloorModel],
    ) -> Dict[str, Tuple[object, List[IFCElement2D], Dict[int, IFCElement2D]]]:
        indexes: Dict[str, Tuple[object, List[IFCElement2D], Dict[int, IFCElement2D]]] = {}
        try:
            from shapely.strtree import STRtree
        except Exception:
            return indexes
        for floor in floors.values():
            openings = [
                element for element in (getattr(floor, "elements", []) or [])
                if getattr(element, "is_rf_barrier", False)
                and element.polygon is not None
                and not element.polygon.is_empty
            ]
            if not openings:
                continue
            try:
                tree = STRtree([opening.polygon for opening in openings])
                by_geometry_id = {id(opening.polygon): opening for opening in openings}
                indexes[floor.name] = (tree, openings, by_geometry_id)
            except Exception:
                continue
        return indexes

    @staticmethod
    def _openings_intersecting_line(
        floor: FloorModel,
        line: LineString,
        opening_indexes: Optional[Dict[str, Tuple[object, List[IFCElement2D], Dict[int, IFCElement2D]]]] = None,
    ) -> Iterable[IFCElement2D]:
        if opening_indexes:
            indexed = opening_indexes.get(floor.name)
            if indexed is not None:
                tree, openings, by_geometry_id = indexed
                try:
                    hits = tree.query(line)
                    result: List[IFCElement2D] = []
                    for hit in hits:
                        if isinstance(hit, (int, np.integer)):
                            idx = int(hit)
                            if 0 <= idx < len(openings):
                                result.append(openings[idx])
                        else:
                            opening = by_geometry_id.get(id(hit))
                            if opening is not None:
                                result.append(opening)
                    return result
                except Exception:
                    pass
        return [
            element for element in (getattr(floor, "elements", []) or [])
            if getattr(element, "is_rf_barrier", False)
        ]

    @staticmethod
    def _element_intersects_3d_path(element: IFCElement2D, line: LineString, ap_z: float, rx_z: float) -> bool:
        """Return True when the link crosses the element footprint and z-span."""
        try:
            intersection = element.polygon.intersection(line)
            if intersection.is_empty:
                return False
            horizontal_length = float(line.length)
            if horizontal_length <= 1e-9:
                fractions = [0.0]
            else:
                points = []
                geom_type = getattr(intersection, "geom_type", "")
                if geom_type == "Point":
                    points = [intersection]
                elif geom_type in {"MultiPoint", "GeometryCollection"}:
                    points = [value for value in getattr(intersection, "geoms", []) if getattr(value, "geom_type", "") == "Point"]
                elif geom_type in {"LineString", "LinearRing"}:
                    coords = list(intersection.coords)
                    points = [Point(coords[0]), Point(coords[-1]), intersection.centroid] if coords else [intersection.centroid]
                elif geom_type == "MultiLineString":
                    for part in getattr(intersection, "geoms", []):
                        coords = list(part.coords)
                        if coords:
                            points.extend([Point(coords[0]), Point(coords[-1]), part.centroid])
                if not points:
                    points = [intersection.centroid]
                fractions = [max(0.0, min(1.0, float(line.project(point)) / horizontal_length)) for point in points]
            z_values = [float(ap_z) + fraction * (float(rx_z) - float(ap_z)) for fraction in fractions]
            path_min = min(z_values); path_max = max(z_values)
            element_min = min(float(element.z_min), float(element.z_max))
            element_max = max(float(element.z_min), float(element.z_max))
            tolerance = 0.12
            return path_max >= element_min - tolerance and path_min <= element_max + tolerance
        except Exception:
            # Retain 2D attenuation for exporters with unusable vertical metadata.
            return True

    @staticmethod
    def _opening_intersects_3d_path(opening: IFCElement2D, line: LineString, ap_z: float, rx_z: float) -> bool:
        return RFEngine._element_intersects_3d_path(opening, line, ap_z, rx_z)

    @staticmethod
    def _opening_replaces_wall(opening: IFCElement2D, wall: Wall2D) -> bool:
        host_guid = str(getattr(opening, "host_wall_guid", "") or "")
        if host_guid and host_guid == str(getattr(wall, "guid", "") or ""):
            return True
        if host_guid:
            return False
        try:
            # Fallback for IFCs that omit IfcRelFillsElement/IfcRelVoidsElement:
            # use a small tolerance because simplified panel footprints can sit
            # just proud of the host wall centreline.
            return opening.polygon.buffer(0.08).intersects(wall.polygon)
        except Exception:
            return False

    @staticmethod
    def _boundary_mask(
        xs: np.ndarray,
        ys: np.ndarray,
        boundary_geometry,
    ) -> Optional[np.ndarray]:
        """Return a boolean grid mask for points covered by the calculation boundary."""
        if boundary_geometry is None:
            return None
        try:
            geometry = boundary_geometry
            if geometry.is_empty:
                return None
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.is_empty:
                return None
            xs_array = np.asarray(xs, dtype=float)
            ys_array = np.asarray(ys, dtype=float)
            if shapely_intersects_xy is not None:
                try:
                    xx, yy = np.meshgrid(xs_array, ys_array)
                    return np.asarray(shapely_intersects_xy(geometry, xx, yy), dtype=bool)
                except Exception:
                    pass

            mask = np.zeros((len(ys_array), len(xs_array)), dtype=bool)
            prepared = prepare_geometry(geometry)
            minx, miny, maxx, maxy = geometry.bounds
            x_candidates = [
                (index, float(value)) for index, value in enumerate(xs_array)
                if minx - 1e-9 <= float(value) <= maxx + 1e-9
            ]
            for iy, y_value in enumerate(ys_array):
                y_float = float(y_value)
                if y_float < miny - 1e-9 or y_float > maxy + 1e-9:
                    continue
                for ix, x_float in x_candidates:
                    if prepared.covers(Point(x_float, y_float)):
                        mask[iy, ix] = True
            return mask
        except Exception:
            return None

    @staticmethod
    def _adaptive_rf_tile_rows(
        row_count: int,
        process_count: int,
        heatmap_settings: Optional[HeatmapSettings],
    ) -> int:
        """Choose a strip height that balances workers without tiny jobs."""
        row_count = max(1, int(row_count))
        process_count = max(1, min(int(process_count), row_count))
        preferred = max(1, int(getattr(heatmap_settings, "rf_tile_rows", 16) or 16))
        tiles_per_worker = max(1, int(getattr(heatmap_settings, "rf_tiles_per_worker", 2) or 2))
        desired_jobs = min(row_count, process_count * tiles_per_worker)
        adaptive = max(1, int(math.ceil(row_count / max(1, desired_jobs))))
        if row_count >= process_count * preferred:
            return max(preferred, adaptive)
        return max(1, int(math.ceil(row_count / process_count)))

    @staticmethod
    def _profiled_settings(
        settings: Optional[HeatmapSettings],
        profile_override: Optional[str] = None,
    ) -> Optional[HeatmapSettings]:
        if settings is None:
            return None
        profiled = replace(settings)
        profile = str(profile_override or getattr(settings, "rf_calculation_profile", "balanced")).strip().lower()
        if profile not in {"fast", "balanced", "detailed"}:
            profile = "balanced"
        profiled.rf_calculation_profile = profile
        if profile == "fast":
            profiled.enable_adaptive_rf_grid = True
            profiled.max_reflection_order = min(1, int(profiled.max_reflection_order))
            profiled.max_reflection_surfaces = min(4, int(profiled.max_reflection_surfaces))
            profiled.max_reflection_paths = min(4, int(profiled.max_reflection_paths))
            profiled.max_diffraction_paths = min(1, int(profiled.max_diffraction_paths))
            profiled.adaptive_ap_refine_radius_m = min(4.0, float(profiled.adaptive_ap_refine_radius_m))
            profiled.adaptive_geometry_buffer_m = min(0.5, float(profiled.adaptive_geometry_buffer_m))
            profiled.adaptive_threshold_margin_db = min(1.5, float(profiled.adaptive_threshold_margin_db))
            profiled.adaptive_gradient_threshold_db_per_m = max(6.0, float(profiled.adaptive_gradient_threshold_db_per_m))
            profiled.enable_small_scale_fading = False
            profiled.calculate_delay_spread = False
            profiled.multipath_relative_power_cutoff_db = min(20.0, float(profiled.multipath_relative_power_cutoff_db))
            profiled.heatmap_render_mode = "raster"
        elif profile == "balanced":
            profiled.max_reflection_order = min(2, int(profiled.max_reflection_order))
            profiled.max_diffraction_paths = min(2, int(profiled.max_diffraction_paths))
            profiled.multipath_relative_power_cutoff_db = min(30.0, float(profiled.multipath_relative_power_cutoff_db))
        else:
            # Detailed mode deliberately evaluates the complete uniform grid.
            profiled.enable_adaptive_rf_grid = False
        return profiled

    @staticmethod
    def _grid_for_floor(
        floor: FloorModel,
        aps: List[AccessPoint],
        resolution_m: float,
        calculation_boundary=None,
    ) -> Tuple[np.ndarray, np.ndarray, object]:
        minx, miny, maxx, maxy = RFEngine._floor_bounds(floor, aps)
        boundary = calculation_boundary
        if boundary is not None:
            try:
                if not boundary.is_valid:
                    boundary = boundary.buffer(0)
                if boundary.is_empty:
                    boundary = None
                else:
                    minx, miny, maxx, maxy = boundary.bounds
            except Exception:
                boundary = None
        resolution = max(0.05, float(resolution_m))
        xs = np.arange(minx, maxx + resolution * 0.5, resolution, dtype=float)
        ys = np.arange(miny, maxy + resolution * 0.5, resolution, dtype=float)
        if len(xs) < 2:
            xs = np.asarray([minx, max(maxx, minx + resolution)], dtype=float)
        if len(ys) < 2:
            ys = np.asarray([miny, max(maxy, miny + resolution)], dtype=float)
        return xs, ys, boundary

    @staticmethod
    def _geometry_revision(floors: Dict[str, FloorModel]) -> str:
        records = []
        for floor_name, model in sorted(floors.items()):
            objects = []
            for category, collection in (
                ("w", getattr(model, "walls", []) or []),
                ("s", getattr(model, "spaces", []) or []),
                ("e", getattr(model, "elements", []) or []),
            ):
                for obj in collection:
                    polygon = getattr(obj, "polygon", None)
                    try:
                        bounds = tuple(round(float(value), 4) for value in polygon.bounds)
                        area = round(float(polygon.area), 4)
                    except Exception:
                        bounds = (); area = 0.0
                    attenuation = tuple(sorted(
                        (round(float(key), 3), round(float(value), 4))
                        for key, value in dict(getattr(obj, "attenuation_by_band_db", {}) or {}).items()
                    ))
                    objects.append((
                        category, str(getattr(obj, "guid", "")), str(getattr(obj, "rf_type_override", "")),
                        bounds, area, round(float(getattr(obj, "z_min", 0.0)), 3),
                        round(float(getattr(obj, "z_max", 0.0)), 3), attenuation,
                    ))
            records.append((floor_name, round(float(model.elevation), 4), tuple(objects), tuple(sorted(model.slab_attenuation_by_band_db.items()))))
        return stable_digest(records)

    @staticmethod
    def _settings_revision(settings: Optional[HeatmapSettings]) -> str:
        if settings is None:
            return "none"
        return stable_digest({
            "profile": getattr(settings, "rf_calculation_profile", "balanced"),
            "cutoff": [getattr(settings, "enable_ap_cutoff_zones", True), getattr(settings, "ap_cutoff_radius_by_frequency_m", {})],
            "reflection": [
                settings.enable_multipath_reflections, settings.max_reflection_order,
                settings.max_reflection_surfaces, settings.max_reflection_paths,
                settings.reflection_search_radius_m, settings.minimum_reflection_coefficient,
                settings.reflection_material_properties,
            ],
            "diffraction": [settings.enable_corner_diffraction, settings.max_diffraction_paths,
                            settings.diffraction_search_radius_m, settings.minimum_diffraction_loss_db],
            "fading": [settings.enable_small_scale_fading, settings.small_scale_fading_sigma_db,
                       settings.fading_correlation_distance_m, settings.fading_seed],
            "delay": settings.calculate_delay_spread,
            "combined_ap_mode": getattr(settings, "combined_ap_mode", "strongest"),
            "disconnected": getattr(settings, "disconnected_rssi_dbm", -120.0),
            "path_cutoff": getattr(settings, "multipath_relative_power_cutoff_db", 30.0),
            "adaptive": [
                getattr(settings, "enable_adaptive_rf_grid", True),
                getattr(settings, "adaptive_coarse_resolution_m", 3.0),
                getattr(settings, "adaptive_gradient_threshold_db_per_m", 3.0),
                getattr(settings, "adaptive_threshold_margin_db", 5.0),
                getattr(settings, "adaptive_geometry_buffer_m", 1.25),
                getattr(settings, "adaptive_ap_refine_radius_m", 10.0),
                getattr(settings, "minimum_client_rssi_dbm", -82.0),
                getattr(settings, "auto_planner_settings", {}),
            ],
            "attenuation_defaults": [
                getattr(settings, "default_wall_attenuation_by_material_db", {}),
                getattr(settings, "default_door_attenuation_by_material_db", {}),
                getattr(settings, "default_window_attenuation_by_material_db", {}),
                getattr(settings, "default_ifc_element_attenuation_by_type_db", {}),
                getattr(settings, "default_floor_attenuation_by_frequency_db", {}),
            ],
        })

    @staticmethod
    def _pattern_revision(patterns: Optional[Dict[str, AntennaPattern]]) -> str:
        records = []
        for name, pattern in sorted((patterns or {}).items()):
            records.append((
                str(name), round(float(getattr(pattern, "peak_gain_dbi", 0.0)), 6),
                tuple((round(float(a), 6), round(float(g), 6)) for a, g in getattr(pattern, "azimuth_points", []) or []),
                tuple((round(float(a), 6), round(float(g), 6)) for a, g in getattr(pattern, "elevation_points", []) or []),
            ))
        return stable_digest(records)

    @staticmethod
    def _ap_revision(ap: AccessPoint) -> str:
        radios = []
        for radio in ap.active_radios():
            radios.append((
                str(radio.name), round(float(radio.frequency_mhz), 6), round(float(radio.tx_power_dbm), 4),
                round(float(getattr(radio, "antenna_gain_dbi", 0.0) or 0.0), 4), str(radio.antenna_pattern),
                str(getattr(radio, "channel", "")), round(float(getattr(radio, "cutoff_radius_m", 0.0) or 0.0), 3),
            ))
        return stable_digest((
            ap.name, ap.floor, round(float(ap.x), 5), round(float(ap.y), 5),
            round(float(ap.mount_height_m), 4), round(float(ap.rx_height_m), 4),
            round(float(ap.path_loss_exponent), 5), round(float(ap.azimuth_deg), 4),
            round(float(ap.downtilt_deg), 4), tuple(radios),
        ))

    @staticmethod
    def _boundary_revision(boundary) -> str:
        if boundary is None:
            return "none"
        try:
            return stable_digest((tuple(round(float(v), 5) for v in boundary.bounds), boundary.wkb_hex))
        except Exception:
            return stable_digest(str(boundary))

    @staticmethod
    def _tile_local_indexes(
        wall_indexes,
        opening_indexes,
        floor_names: Iterable[str],
        envelope,
    ):
        """Build compact STRtrees containing only objects in an AP-to-tile envelope."""
        local_walls = {}
        local_openings = {}
        for floor_name in floor_names:
            indexed = (wall_indexes or {}).get(floor_name)
            if indexed is not None:
                tree, walls, by_id = indexed
                subset = []
                try:
                    hits = tree.query(envelope)
                    for hit in hits:
                        if isinstance(hit, (int, np.integer)):
                            idx = int(hit)
                            if 0 <= idx < len(walls): subset.append(walls[idx])
                        else:
                            obj = by_id.get(id(hit))
                            if obj is not None: subset.append(obj)
                except Exception:
                    subset = list(walls)
                if subset:
                    polygons = [obj.polygon for obj in subset]
                    try:
                        local_walls[floor_name] = (STRtree(polygons), subset, {id(obj.polygon): obj for obj in subset})
                    except Exception:
                        pass
            indexed = (opening_indexes or {}).get(floor_name)
            if indexed is not None:
                tree, openings, by_id = indexed
                subset = []
                try:
                    hits = tree.query(envelope)
                    for hit in hits:
                        if isinstance(hit, (int, np.integer)):
                            idx = int(hit)
                            if 0 <= idx < len(openings): subset.append(openings[idx])
                        else:
                            obj = by_id.get(id(hit))
                            if obj is not None: subset.append(obj)
                except Exception:
                    subset = list(openings)
                if subset:
                    polygons = [obj.polygon for obj in subset]
                    try:
                        local_openings[floor_name] = (STRtree(polygons), subset, {id(obj.polygon): obj for obj in subset})
                    except Exception:
                        pass
        return local_walls or wall_indexes, local_openings or opening_indexes

    @staticmethod
    def _combine_ap_field_results(
        fields: List[SimulationResult],
        settings: Optional[HeatmapSettings],
        disconnected: float,
        elapsed: float,
        cache_hits: int,
        cache_misses: int,
    ) -> Optional[SimulationResult]:
        if not fields:
            return None
        stack = np.stack([result.rssi for result in fields], axis=0)
        finite_stack = np.where(np.isfinite(stack), stack, -np.inf)
        strongest_index = np.argmax(finite_stack, axis=0)
        strongest = np.take_along_axis(finite_stack, strongest_index[None, :, :], axis=0)[0]
        all_invalid = ~np.any(np.isfinite(stack), axis=0)
        mode = str(getattr(settings, "combined_ap_mode", "strongest") or "strongest").lower()
        if mode == "power_sum":
            powers = np.where(np.isfinite(stack), np.power(10.0, stack / 10.0), 0.0)
            total_power = np.sum(powers, axis=0)
            rssi = np.full_like(strongest, float(disconnected), dtype=float)
            positive = total_power > 0.0
            rssi[positive] = 10.0 * np.log10(total_power[positive])
        else:
            rssi = strongest
        rssi[all_invalid] = np.nan
        delay_stack = np.stack([
            result.delay_spread_ns if result.delay_spread_ns is not None else np.zeros_like(result.rssi)
            for result in fields
        ], axis=0)
        delay = np.take_along_axis(delay_stack, strongest_index[None, :, :], axis=0)[0]
        delay[all_invalid] = np.nan
        count_stack = np.stack([
            result.path_count if result.path_count is not None else np.ones_like(result.rssi, dtype=np.int16)
            for result in fields
        ], axis=0)
        if mode == "power_sum":
            count = np.sum(count_stack.astype(np.int32), axis=0).clip(0, 32767).astype(np.int16)
        else:
            count = np.take_along_axis(count_stack, strongest_index[None, :, :], axis=0)[0].astype(np.int16)
        count[all_invalid] = 0
        first = fields[0]
        worker_count = max(int(result.worker_processes) for result in fields)
        execution_modes = {result.execution_mode for result in fields}
        execution_mode = "cache" if cache_misses == 0 else ("multiprocess" if "multiprocess" in execution_modes else next(iter(execution_modes)))
        return SimulationResult(
            xs=first.xs.copy(), ys=first.ys.copy(), rssi=rssi,
            delay_spread_ns=delay, path_count=count,
            valid_mask=None if first.valid_mask is None else first.valid_mask.copy(),
            boundary_geometry=first.boundary_geometry,
            ignored_point_count=first.ignored_point_count,
            execution_mode=execution_mode,
            worker_processes=worker_count,
            elapsed_seconds=float(elapsed),
            performance_note=(
                f"combined {len(fields)} independently cached AP field(s); "
                f"{cache_hits} cache hit(s), {cache_misses} recalculated field(s)"
            ),
            approximate_points=max(int(result.approximate_points) for result in fields),
            exact_points=max(int(result.exact_points) for result in fields),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    @staticmethod
    def _simulate_groups_uniform(
        floor: FloorModel,
        floors: Dict[str, FloorModel],
        group_aps: Dict[object, List[AccessPoint]],
        resolution_m: float = 2.0,
        patterns: Optional[Dict[str, AntennaPattern]] = None,
        include_inter_floor: bool = True,
        heatmap_settings: Optional[HeatmapSettings] = None,
        progress_callback=None,
        calculation_boundary=None,
        evaluation_mask: Optional[np.ndarray] = None,
        grid_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        progressive_callback=None,
    ) -> Dict[object, SimulationResult]:
        """Uniform-grid exact evaluator with resident model and shared-memory workers."""
        started = time.perf_counter()
        if not floor.walls and not floor.spaces and not getattr(floor, "elements", []):
            return {}
        group_aps = {key: list(value) for key, value in group_aps.items() if value}
        if not group_aps:
            return {}
        all_aps = []
        seen = set()
        for aps in group_aps.values():
            for ap in aps:
                identity = (ap.name, ap.floor, float(ap.x), float(ap.y))
                if identity not in seen:
                    seen.add(identity); all_aps.append(ap)
        if not all_aps:
            return {}
        if grid_override is None:
            xs, ys, calculation_boundary = RFEngine._grid_for_floor(floor, all_aps, resolution_m, calculation_boundary)
        else:
            xs = np.asarray(grid_override[0], dtype=float); ys = np.asarray(grid_override[1], dtype=float)
        boundary_mask = RFEngine._boundary_mask(xs, ys, calculation_boundary)
        if boundary_mask is not None and not np.any(boundary_mask):
            return {}
        if evaluation_mask is None:
            work_mask = None if boundary_mask is None else boundary_mask.copy()
        else:
            work_mask = np.asarray(evaluation_mask, dtype=bool)
            if work_mask.shape != (len(ys), len(xs)):
                raise ValueError("Adaptive evaluation mask does not match RF grid")
            if boundary_mask is not None:
                work_mask &= boundary_mask
        disconnected = float(getattr(heatmap_settings, "disconnected_rssi_dbm", -120.0) if heatmap_settings else -120.0)
        radio_links_by_group = {
            key: RFEngine._active_radio_links(floor, aps, include_inter_floor)
            for key, aps in group_aps.items()
        }
        radio_links_by_group = {key: links for key, links in radio_links_by_group.items() if links}
        if not radio_links_by_group:
            return {}
        group_aps = {key: group_aps[key] for key in radio_links_by_group}
        group_order = list(group_aps.keys())
        rows, cols = len(ys), len(xs)
        if boundary_mask is None:
            initial_rssi = np.full((rows, cols), disconnected, dtype=float)
            initial_delay = np.zeros((rows, cols), dtype=float)
            ignored = 0
        else:
            initial_rssi = np.full((rows, cols), np.nan, dtype=float); initial_rssi[boundary_mask] = disconnected
            initial_delay = np.full((rows, cols), np.nan, dtype=float); initial_delay[boundary_mask] = 0.0
            ignored = int(boundary_mask.size - np.count_nonzero(boundary_mask))
        grids = {key: initial_rssi.copy() for key in group_order}
        delays = {key: initial_delay.copy() for key in group_order}
        counts = {key: np.zeros((rows, cols), dtype=np.int16) for key in group_order}
        exact_point_count = int(np.count_nonzero(work_mask)) if work_mask is not None else rows * cols
        total_links = sum(len(links) for links in radio_links_by_group.values())
        work_units = exact_point_count * max(1, total_links)
        use_mp = bool(getattr(heatmap_settings, "enable_rf_multiprocessing", False)) if heatmap_settings else False
        threshold = int(getattr(heatmap_settings, "rf_multiprocessing_min_points", 5000) if heatmap_settings else 5000)
        fallback_note = ""

        if use_mp and work_units >= threshold and rows > 1:
            requested = int(getattr(heatmap_settings, "max_rf_worker_processes", 0) or 0)
            process_count = min(_logical_process_count(requested), rows)
            tile_rows = RFEngine._adaptive_rf_tile_rows(rows, process_count, heatmap_settings)
            model_key = stable_digest((
                floor.name, RFEngine._geometry_revision(floors), RFEngine._settings_revision(heatmap_settings),
                bool(include_inter_floor), RFEngine._pattern_revision(patterns),
            ))
            context_path = _rf_get_worker_context_file(
                model_key, floor, floors, patterns, include_inter_floor, heatmap_settings,
                max(2, int(getattr(heatmap_settings, "rf_worker_index_cache_entries", 2) or 2)),
            )
            worker_cache_entries = max(1, int(getattr(heatmap_settings, "rf_worker_index_cache_entries", 2) or 2))
            use_shared = bool(getattr(heatmap_settings, "use_shared_memory_rf_results", True))
            shared_handles = []
            shared_specs = None
            try:
                if use_shared:
                    rssi_mem, rssi_stack, rssi_spec = create_shared_array((len(group_order), rows, cols), np.float64, np.nan)
                    delay_mem, delay_stack, delay_spec = create_shared_array((len(group_order), rows, cols), np.float64, np.nan)
                    count_mem, count_stack, count_spec = create_shared_array((len(group_order), rows, cols), np.int16, 0)
                    shared_handles = [rssi_mem, delay_mem, count_mem]
                    shared_specs = (rssi_spec, delay_spec, count_spec)
                    for group_index in range(len(group_order)):
                        rssi_stack[group_index, :, :] = initial_rssi
                        delay_stack[group_index, :, :] = initial_delay
                else:
                    rssi_stack = delay_stack = count_stack = None
                jobs = []
                for start_index in range(0, rows, tile_rows):
                    jobs.append((
                        model_key, context_path, xs, ys[start_index:start_index + tile_rows],
                        None if work_mask is None else work_mask[start_index:start_index + tile_rows, :],
                        start_index, disconnected, worker_cache_entries, group_aps, group_order, shared_specs,
                    ))
                process_count = min(process_count, max(1, len(jobs)))
                executor, owned = _get_rf_process_executor(
                    process_count, bool(getattr(heatmap_settings, "reuse_rf_process_pool", True))
                )
                futures = [executor.submit(_rf_grid_multi_tile_worker, job) for job in jobs]
                completed_rows = 0
                next_progressive = max(5, int(getattr(heatmap_settings, "progressive_update_percent", 20)))
                for future in concurrent.futures.as_completed(futures):
                    start_index, row_count, tile_results = future.result()
                    if not use_shared and tile_results is not None:
                        for key, (tile, delay_tile, count_tile) in tile_results.items():
                            grids[key][start_index:start_index + row_count, :] = tile
                            delays[key][start_index:start_index + row_count, :] = delay_tile
                            counts[key][start_index:start_index + row_count, :] = count_tile
                    completed_rows += int(row_count)
                    if progress_callback:
                        progress_callback(min(completed_rows, rows), rows)
                    percent = int(100.0 * completed_rows / max(1, rows))
                    if progressive_callback and bool(getattr(heatmap_settings, "progressive_heatmap_updates", True)) and percent >= next_progressive:
                        if use_shared:
                            preview = {
                                key: SimulationResult(
                                    xs=xs, ys=ys, rssi=rssi_stack[index].copy(),
                                    delay_spread_ns=delay_stack[index].copy(), path_count=count_stack[index].copy(),
                                    valid_mask=None if boundary_mask is None else boundary_mask.copy(),
                                    boundary_geometry=calculation_boundary, ignored_point_count=ignored,
                                    execution_mode="multiprocess-progressive", worker_processes=process_count,
                                    progressive_fraction=min(0.99, completed_rows / max(1, rows)),
                                ) for index, key in enumerate(group_order)
                            }
                        else:
                            preview = {
                                key: SimulationResult(
                                    xs=xs, ys=ys, rssi=grids[key].copy(), delay_spread_ns=delays[key].copy(),
                                    path_count=counts[key].copy(), valid_mask=None if boundary_mask is None else boundary_mask.copy(),
                                    boundary_geometry=calculation_boundary, ignored_point_count=ignored,
                                    execution_mode="multiprocess-progressive", worker_processes=process_count,
                                    progressive_fraction=min(0.99, completed_rows / max(1, rows)),
                                ) for key in group_order
                            }
                        progressive_callback(preview, completed_rows / max(1, rows))
                        next_progressive += max(5, int(getattr(heatmap_settings, "progressive_update_percent", 20)))
                if use_shared:
                    for index, key in enumerate(group_order):
                        grids[key] = rssi_stack[index].copy()
                        delays[key] = delay_stack[index].copy()
                        counts[key] = count_stack[index].copy()
                elapsed = time.perf_counter() - started
                note = (
                    f"resident static model; {len(jobs)} adaptive tile(s); shared-memory outputs "
                    f"{'enabled' if use_shared else 'disabled'}; frequency-independent path geometry reused"
                )
                return {
                    key: SimulationResult(
                        xs=xs, ys=ys, rssi=grids[key], delay_spread_ns=delays[key], path_count=counts[key],
                        valid_mask=None if boundary_mask is None else boundary_mask.copy(),
                        boundary_geometry=calculation_boundary, ignored_point_count=ignored,
                        execution_mode="multiprocess", worker_processes=process_count,
                        elapsed_seconds=elapsed, performance_note=note,
                        exact_points=exact_point_count,
                    ) for key in group_order
                }
            except Exception as exc:
                if "cancel" in str(exc).lower():
                    raise
                _shutdown_rf_process_executor(wait=False)
                fallback_note = f"RF multiprocessing failed ({type(exc).__name__}: {exc}); used single-process fallback."
            finally:
                for memory in shared_handles:
                    try: memory.close()
                    except Exception: pass
                    try: memory.unlink()
                    except Exception: pass
                if 'owned' in locals() and owned and 'executor' in locals():
                    try: executor.shutdown(wait=True)
                    except Exception: pass

        wall_indexes = RFEngine._build_wall_indexes(floors)
        opening_indexes = RFEngine._build_opening_indexes(floors)
        reflection_indexes = RFEngine._build_reflection_indexes(floors, heatmap_settings)
        if progress_callback:
            progress_callback(0, rows)
        for iy, yy in enumerate(ys):
            for ix, xx in enumerate(xs):
                if work_mask is not None and not bool(work_mask[iy, ix]):
                    continue
                geometry_cache = {} if bool(getattr(heatmap_settings, "reuse_path_geometry_across_frequencies", True)) else None
                for key in group_order:
                    samples = []
                    for ap, radio in radio_links_by_group[key]:
                        if RFEngine.point_is_inside_radio_cutoff(xx, yy, floor, ap, floors, radio, heatmap_settings):
                            samples.append(RFEngine.propagation_at(
                                float(xx), float(yy), floor, ap, floors, patterns, radio,
                                include_inter_floor, heatmap_settings, wall_indexes, opening_indexes,
                                reflection_indexes, geometry_cache,
                            ))
                    combined = RFEngine.combine_ap_samples(samples, heatmap_settings, disconnected)
                    grids[key][iy, ix] = combined.rssi_dbm
                    delays[key][iy, ix] = combined.delay_spread_ns
                    counts[key][iy, ix] = max(0, min(32767, int(combined.path_count)))
            if progress_callback:
                progress_callback(iy + 1, rows)
            if progressive_callback and bool(getattr(heatmap_settings, "progressive_heatmap_updates", True)):
                step = max(1, int(rows * max(5, int(getattr(heatmap_settings, "progressive_update_percent", 20))) / 100.0))
                if (iy + 1) % step == 0 and iy + 1 < rows:
                    progressive_callback({
                        key: SimulationResult(
                            xs=xs, ys=ys, rssi=grids[key].copy(), delay_spread_ns=delays[key].copy(),
                            path_count=counts[key].copy(), valid_mask=None if boundary_mask is None else boundary_mask.copy(),
                            boundary_geometry=calculation_boundary, ignored_point_count=ignored,
                            execution_mode="single-process-progressive", progressive_fraction=(iy + 1) / rows,
                        ) for key in group_order
                    }, (iy + 1) / rows)
        elapsed = time.perf_counter() - started
        note = fallback_note or "single-process exact grid; resident geometry and cross-frequency path cache enabled"
        return {
            key: SimulationResult(
                xs=xs, ys=ys, rssi=grids[key], delay_spread_ns=delays[key], path_count=counts[key],
                valid_mask=None if boundary_mask is None else boundary_mask.copy(),
                boundary_geometry=calculation_boundary, ignored_point_count=ignored,
                execution_mode="single-process-fallback" if fallback_note else "single-process",
                worker_processes=1, elapsed_seconds=elapsed, performance_note=note,
                exact_points=exact_point_count,
            ) for key in group_order
        }

    @staticmethod
    def _simulate_groups_adaptive(
        floor: FloorModel,
        floors: Dict[str, FloorModel],
        group_aps: Dict[object, List[AccessPoint]],
        resolution_m: float,
        patterns,
        include_inter_floor: bool,
        heatmap_settings: Optional[HeatmapSettings],
        progress_callback,
        calculation_boundary,
        progressive_callback=None,
    ) -> Dict[object, SimulationResult]:
        settings = heatmap_settings
        coarse_resolution = max(float(resolution_m) * 2.0, float(getattr(settings, "adaptive_coarse_resolution_m", 3.0)))
        if not bool(getattr(settings, "enable_adaptive_rf_grid", True)) or coarse_resolution <= float(resolution_m) * 1.15:
            return RFEngine._simulate_groups_uniform(
                floor, floors, group_aps, resolution_m, patterns, include_inter_floor,
                settings, progress_callback, calculation_boundary, progressive_callback=progressive_callback,
            )

        def coarse_progress(done, total):
            if progress_callback:
                progress_callback(int(250 * float(done) / max(1, float(total))), 1000)
        coarse = RFEngine._simulate_groups_uniform(
            floor, floors, group_aps, coarse_resolution, patterns, include_inter_floor,
            settings, coarse_progress, calculation_boundary,
        )
        if not coarse:
            return {}
        all_aps = [ap for aps in group_aps.values() for ap in aps]
        fine_xs, fine_ys, calculation_boundary = RFEngine._grid_for_floor(floor, all_aps, resolution_m, calculation_boundary)
        fine_valid = RFEngine._boundary_mask(fine_xs, fine_ys, calculation_boundary)
        interpolated = {}
        for key, result in coarse.items():
            interpolated[key] = SimulationResult(
                xs=fine_xs, ys=fine_ys,
                rssi=resample_regular_grid(result.xs, result.ys, result.rssi, fine_xs, fine_ys),
                delay_spread_ns=resample_regular_grid(result.xs, result.ys, result.delay_spread_ns, fine_xs, fine_ys) if result.delay_spread_ns is not None else None,
                path_count=nearest_resample_regular_grid(result.xs, result.ys, result.path_count, fine_xs, fine_ys).astype(np.int16) if result.path_count is not None else None,
                valid_mask=None if fine_valid is None else fine_valid.copy(),
                boundary_geometry=calculation_boundary, ignored_point_count=0 if fine_valid is None else int(fine_valid.size - np.count_nonzero(fine_valid)),
                execution_mode="adaptive-coarse", worker_processes=result.worker_processes,
                progressive_fraction=0.25,
            )
        if fine_valid is not None:
            for result in interpolated.values():
                result.rssi[~fine_valid] = np.nan
                if result.delay_spread_ns is not None:
                    result.delay_spread_ns[~fine_valid] = np.nan
                if result.path_count is not None:
                    result.path_count[~fine_valid] = 0
        if progressive_callback:
            progressive_callback(interpolated, 0.25)
        geometry_boxes = []
        for obj in list(getattr(floor, "walls", []) or []) + list(getattr(floor, "elements", []) or []):
            try: geometry_boxes.append(tuple(float(v) for v in obj.polygon.bounds))
            except Exception: pass
        ap_points = []
        for ap in all_aps:
            radii = [RFEngine.cutoff_radius_m_for_radio(radio, settings) for radio in ap.active_radios()]
            ap_points.append((ap.x, ap.y, max(radii or [0.0])))
        client_threshold = float(getattr(settings, "minimum_client_rssi_dbm", -82.0))
        planner_threshold = client_threshold
        try:
            planner_threshold = float((getattr(settings, "auto_planner_settings", {}) or {}).get("minimum_rssi_dbm", client_threshold))
        except Exception:
            planner_threshold = client_threshold
        # Adaptive sampling protects engineering decision thresholds rather than every
        # cosmetic isoline. Detailed/report mode uses the full uniform grid.
        thresholds = sorted({client_threshold, planner_threshold})
        stride = max(2, int(round(coarse_resolution / max(float(resolution_m), 1e-6))))
        refine_mask = adaptive_refinement_mask(
            fine_xs, fine_ys, [result.rssi for result in interpolated.values()], fine_valid,
            thresholds, stride,
            float(getattr(settings, "adaptive_gradient_threshold_db_per_m", 3.0)),
            float(getattr(settings, "adaptive_threshold_margin_db", 5.0)),
            geometry_boxes, float(getattr(settings, "adaptive_geometry_buffer_m", 1.25)),
            ap_points, float(getattr(settings, "adaptive_ap_refine_radius_m", 10.0)),
        )
        total_valid = int(np.count_nonzero(fine_valid)) if fine_valid is not None else refine_mask.size
        exact_points = int(np.count_nonzero(refine_mask))

        def fine_progress(done, total):
            if progress_callback:
                progress_callback(250 + int(750 * float(done) / max(1, float(total))), 1000)
        exact = RFEngine._simulate_groups_uniform(
            floor, floors, group_aps, resolution_m, patterns, include_inter_floor,
            settings, fine_progress, calculation_boundary, refine_mask, (fine_xs, fine_ys),
            progressive_callback=None,
        )
        final = {}
        for key, base in interpolated.items():
            detailed = exact.get(key)
            if detailed is None:
                final[key] = base; continue
            rssi = base.rssi.copy(); rssi[refine_mask] = detailed.rssi[refine_mask]
            delay = base.delay_spread_ns.copy() if base.delay_spread_ns is not None else np.zeros_like(rssi)
            if detailed.delay_spread_ns is not None: delay[refine_mask] = detailed.delay_spread_ns[refine_mask]
            count = base.path_count.copy() if base.path_count is not None else np.zeros_like(rssi, dtype=np.int16)
            if detailed.path_count is not None: count[refine_mask] = detailed.path_count[refine_mask]
            if fine_valid is not None:
                rssi[~fine_valid] = np.nan; delay[~fine_valid] = np.nan; count[~fine_valid] = 0
            final[key] = SimulationResult(
                xs=fine_xs, ys=fine_ys, rssi=rssi, delay_spread_ns=delay, path_count=count,
                valid_mask=None if fine_valid is None else fine_valid.copy(), boundary_geometry=calculation_boundary,
                ignored_point_count=base.ignored_point_count,
                execution_mode=f"adaptive-{detailed.execution_mode}", worker_processes=detailed.worker_processes,
                elapsed_seconds=float(base.elapsed_seconds) + float(detailed.elapsed_seconds),
                performance_note=(
                    f"adaptive coarse-to-fine grid: {exact_points:,} exact of {total_valid:,} valid points; "
                    f"{max(0, total_valid - exact_points):,} points interpolated from the coarse pass"
                ),
                approximate_points=max(0, total_valid - exact_points), exact_points=exact_points,
            )
        if progressive_callback:
            progressive_callback(final, 1.0)
        return final

    @staticmethod
    def _simulate_groups(
        floor: FloorModel,
        floors: Dict[str, FloorModel],
        group_aps: Dict[object, List[AccessPoint]],
        resolution_m: float = 2.0,
        patterns: Optional[Dict[str, AntennaPattern]] = None,
        include_inter_floor: bool = True,
        heatmap_settings: Optional[HeatmapSettings] = None,
        progress_callback=None,
        calculation_boundary=None,
        progressive_callback=None,
        profile_override: Optional[str] = None,
    ) -> Dict[object, SimulationResult]:
        """Incremental per-AP cache around adaptive/exact RF field evaluation."""
        started = time.perf_counter()
        settings = RFEngine._profiled_settings(heatmap_settings, profile_override)
        group_aps = {key: list(aps) for key, aps in group_aps.items() if aps}
        if not group_aps:
            return {}
        use_cache = bool(getattr(settings, "enable_per_ap_heatmap_cache", True))
        if not use_cache:
            return RFEngine._simulate_groups_adaptive(
                floor, floors, group_aps, resolution_m, patterns, include_inter_floor,
                settings, progress_callback, calculation_boundary, progressive_callback,
            )
        _RF_AP_FIELD_CACHE.configure(
            int(getattr(settings, "per_ap_heatmap_cache_entries", 192)),
            int(getattr(settings, "per_ap_heatmap_cache_mb", 768)) * 1024 * 1024,
        )
        model_revision = RFEngine._geometry_revision(floors)
        settings_revision = RFEngine._settings_revision(settings)
        pattern_revision = RFEngine._pattern_revision(patterns)
        boundary_revision = RFEngine._boundary_revision(calculation_boundary)
        cached_by_group: Dict[object, List[SimulationResult]] = {key: [] for key in group_aps}
        missing_groups = {}
        missing_cache_keys = {}
        hits = 0; misses = 0
        for group_key, aps in group_aps.items():
            for index, ap in enumerate(aps):
                cache_key = (
                    floor.name, model_revision, settings_revision, pattern_revision, boundary_revision,
                    bool(include_inter_floor), round(float(resolution_m), 5), RFEngine._ap_revision(ap),
                )
                cached = _RF_AP_FIELD_CACHE.get(cache_key)
                if cached is not None:
                    cached_by_group[group_key].append(cached); hits += 1
                else:
                    field_key = ("ap_field", group_key, index, RFEngine._ap_revision(ap))
                    missing_groups[field_key] = [ap]
                    missing_cache_keys[field_key] = (group_key, cache_key)
                    misses += 1
        if missing_groups:
            def missing_progress(partial_fields, fraction):
                if progressive_callback is None or not partial_fields:
                    return
                preview_by_group: Dict[object, List[SimulationResult]] = {
                    key: list(values) for key, values in cached_by_group.items()
                }
                for field_key, partial_result in partial_fields.items():
                    record = missing_cache_keys.get(field_key)
                    if record is None:
                        continue
                    preview_by_group[record[0]].append(partial_result)
                preview_combined = {}
                preview_elapsed = time.perf_counter() - started
                for preview_key, preview_values in preview_by_group.items():
                    preview_result = RFEngine._combine_ap_field_results(
                        preview_values, settings,
                        float(getattr(settings, "disconnected_rssi_dbm", -120.0)),
                        preview_elapsed, hits, misses,
                    )
                    if preview_result is not None:
                        preview_result.progressive_fraction = float(fraction)
                        preview_combined[preview_key] = preview_result
                if preview_combined:
                    progressive_callback(preview_combined, float(fraction))

            missing_results = RFEngine._simulate_groups_adaptive(
                floor, floors, missing_groups, resolution_m, patterns, include_inter_floor,
                settings, progress_callback, calculation_boundary,
                progressive_callback=missing_progress if progressive_callback is not None else None,
            )
            for field_key, result in missing_results.items():
                group_key, cache_key = missing_cache_keys[field_key]
                _RF_AP_FIELD_CACHE.put(cache_key, result)
                cached_by_group[group_key].append(result)
        elif progress_callback:
            progress_callback(1, 1)
        disconnected = float(getattr(settings, "disconnected_rssi_dbm", -120.0))
        elapsed = time.perf_counter() - started
        combined = {}
        for group_key, fields in cached_by_group.items():
            result = RFEngine._combine_ap_field_results(fields, settings, disconnected, elapsed, hits, misses)
            if result is not None:
                combined[group_key] = result
        if progressive_callback and combined:
            progressive_callback(combined, 1.0)
        return combined

    @staticmethod
    def simulate_frequencies(
        floor: FloorModel,
        floors: Dict[str, FloorModel],
        aps: List[AccessPoint],
        frequencies_mhz: Optional[Iterable[float]] = None,
        resolution_m: float = 2.0,
        patterns: Optional[Dict[str, AntennaPattern]] = None,
        include_inter_floor: bool = True,
        heatmap_settings: Optional[HeatmapSettings] = None,
        progress_callback=None,
        calculation_boundary=None,
        progressive_callback=None,
        profile_override: Optional[str] = None,
    ) -> Dict[float, SimulationResult]:
        """Calculate all requested frequencies in one shared process-pool pass."""
        requested = (
            sorted({float(value) for value in frequencies_mhz})
            if frequencies_mhz is not None
            else sorted({float(radio.frequency_mhz) for ap in aps for radio in ap.active_radios()})
        )
        groups: Dict[object, List[AccessPoint]] = {}
        for frequency in requested:
            aps_for_frequency: List[AccessPoint] = []
            for ap in aps:
                radios = [
                    radio for radio in ap.active_radios()
                    if getattr(radio, "enabled", True)
                    and abs(float(radio.frequency_mhz) - frequency) < 1e-6
                ]
                if radios:
                    aps_for_frequency.append(replace(ap, radios=radios))
            if aps_for_frequency:
                groups[float(frequency)] = aps_for_frequency
        results = RFEngine._simulate_groups(
            floor,
            floors,
            groups,
            resolution_m,
            patterns,
            include_inter_floor,
            heatmap_settings,
            progress_callback,
            calculation_boundary,
            progressive_callback,
            profile_override,
        )
        effective_settings = RFEngine._profiled_settings(heatmap_settings, profile_override)
        render_override = str(getattr(effective_settings, "heatmap_render_mode", "") or "")
        for value in results.values():
            value.render_mode_override = render_override
        return {float(key): value for key, value in results.items()}

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
        calculation_boundary=None,
        progressive_callback=None,
        profile_override: Optional[str] = None,
    ) -> Optional[SimulationResult]:
        """Compatibility wrapper preserving the original combined-radio result."""
        results = RFEngine._simulate_groups(
            floor,
            floors,
            {"combined": aps},
            resolution_m,
            patterns,
            include_inter_floor,
            heatmap_settings,
            progress_callback,
            calculation_boundary,
            progressive_callback,
            profile_override,
        )
        result = results.get("combined")
        if result is not None:
            effective_settings = RFEngine._profiled_settings(heatmap_settings, profile_override)
            result.render_mode_override = str(getattr(effective_settings, "heatmap_render_mode", "") or "")
        return result

    @staticmethod
    def _floor_bounds(floor: FloorModel, aps: List[AccessPoint]) -> Tuple[float, float, float, float]:
        bounds = [w.polygon.bounds for w in floor.walls] + [s.polygon.bounds for s in floor.spaces]
        bounds += [
            element.polygon.bounds for element in getattr(floor, "elements", [])
            if getattr(element, "polygon", None) is not None and not element.polygon.is_empty
        ]
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


def _ifc_storey_elevation_m(storey, dz: float = 0.0, unit_scale: float = 1.0) -> float:
    """Return a storey elevation using host placement, falling back to Elevation."""
    placement_z = None
    if getattr(storey, "ObjectPlacement", None) is not None:
        try:
            placement_z = float(_ifc_local_placement_summary(storey).get("z", 0.0))
        except Exception:
            placement_z = None
    raw_elevation = getattr(storey, "Elevation", None)
    try:
        elevation = float(raw_elevation) if raw_elevation is not None else None
    except Exception:
        elevation = None
    if placement_z is not None and math.isfinite(placement_z):
        scaled_placement = placement_z * float(unit_scale)
        scaled_elevation = elevation * float(unit_scale) if elevation is not None and math.isfinite(elevation) else None
        if scaled_elevation is not None and abs(scaled_placement) <= 0.05 and abs(scaled_elevation) > 0.05:
            return scaled_elevation + float(dz)
        return scaled_placement + float(dz)
    if elevation is not None and math.isfinite(elevation):
        return elevation * float(unit_scale) + float(dz)
    return float(dz)


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
    dz = float(args[3]) if len(args) > 3 else 0.0
    if ifcopenshell is None:
        raise RuntimeError("ifcopenshell is not installed. Run: pip install ifcopenshell")
    path = Path(path_str)
    model = ifcopenshell.open(str(path))
    try:
        import ifcopenshell.util.unit as ifc_unit_util
        unit_scale = float(ifc_unit_util.calculate_unit_scale(model) or 1.0)
    except Exception:
        unit_scale = 1.0
    storeys: Dict[str, float] = {}
    for st in model.by_type("IfcBuildingStorey"):
        name = getattr(st, "Name", None) or getattr(st, "GlobalId", None) or "Storey"
        storeys[str(name)] = _ifc_storey_elevation_m(st, dz=dz, unit_scale=unit_scale)
    seen = set()
    wall_guids: List[str] = []
    for wall in list(model.by_type("IfcWall")) + list(model.by_type("IfcWallStandardCase")):
        guid = getattr(wall, "GlobalId", "") or ""
        if guid and guid not in seen:
            seen.add(guid)
            wall_guids.append(guid)
    space_guids = [getattr(sp, "GlobalId", "") or "" for sp in model.by_type("IfcSpace")]
    space_guids = [g for g in space_guids if g]
    element_guids: List[str] = []
    for element in model.by_type("IfcElement"):
        if element.is_a("IfcWall") or element.is_a("IfcWallStandardCase") or element.is_a("IfcOpeningElement"):
            continue
        guid = getattr(element, "GlobalId", "") or ""
        if guid:
            element_guids.append(guid)
    origin_info = _extract_ifc_origin_information(model, path)
    return path_str, storeys, wall_guids, space_guids, element_guids, path.name, origin_info


def _load_ifc_geometry_chunk_in_process(args):
    """Extract geometry for one GlobalId chunk from a large IFC file."""
    (
        path_str, dx, dy, dz, project_external_walls, external_keywords,
        storeys, wall_guids, space_guids, element_guids, chunk_index, chunk_count,
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
        element_guids=list(element_guids or []),
        storeys_override={str(k): float(v) for k, v in dict(storeys or {}).items()},
    )
    return path_str, floors, path.name, int(chunk_index), int(chunk_count)


def _rf_write_worker_context_file(
    floor: FloorModel,
    floors: Dict[str, FloorModel],
    patterns: Optional[Dict[str, AntennaPattern]],
    include_inter_floor: bool,
    heatmap_settings: Optional[HeatmapSettings],
) -> str:
    """Serialise static model geometry once; AP/radio groups travel separately."""
    handle = tempfile.NamedTemporaryFile(prefix="rf_sim_context_", suffix=".pickle", delete=False)
    path = handle.name
    try:
        with handle:
            pickle.dump(
                (floor, floors, patterns, include_inter_floor, heatmap_settings),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise
    return path


def _rf_get_worker_context_file(
    context_key: str,
    floor: FloorModel,
    floors: Dict[str, FloorModel],
    patterns: Optional[Dict[str, AntennaPattern]],
    include_inter_floor: bool,
    heatmap_settings: Optional[HeatmapSettings],
    maximum_entries: int = 2,
) -> str:
    """Keep static floor snapshots available for workers between AP edits."""
    key = str(context_key)
    with _RF_CONTEXT_FILE_LOCK:
        existing = _RF_CONTEXT_FILE_CACHE.get(key)
        if existing and os.path.exists(existing):
            if key in _RF_CONTEXT_FILE_ORDER:
                _RF_CONTEXT_FILE_ORDER.remove(key)
            _RF_CONTEXT_FILE_ORDER.append(key)
            return existing
        path = _rf_write_worker_context_file(floor, floors, patterns, include_inter_floor, heatmap_settings)
        _RF_CONTEXT_FILE_CACHE[key] = path
        _RF_CONTEXT_FILE_ORDER.append(key)
        maximum = max(1, int(maximum_entries or 2))
        while len(_RF_CONTEXT_FILE_ORDER) > maximum:
            stale = _RF_CONTEXT_FILE_ORDER.pop(0)
            stale_path = _RF_CONTEXT_FILE_CACHE.pop(stale, None)
            if stale_path:
                try:
                    os.unlink(stale_path)
                except Exception:
                    pass
        return path


def _rf_worker_context(context_key: str, context_path: str, maximum_entries: int = 1):
    """Load a static model and build spatial indexes once per worker process."""
    key = str(context_key)
    cached = _RF_WORKER_CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached
    maximum = max(1, int(maximum_entries or 1))
    while len(_RF_WORKER_CONTEXT_ORDER) >= maximum:
        stale = _RF_WORKER_CONTEXT_ORDER.pop(0)
        _RF_WORKER_CONTEXT_CACHE.pop(stale, None)
    with open(context_path, "rb") as handle:
        floor, floors, patterns, include_inter_floor, heatmap_settings = pickle.load(handle)
    context = (
        floor,
        floors,
        patterns,
        include_inter_floor,
        heatmap_settings,
        RFEngine._build_wall_indexes(floors),
        RFEngine._build_opening_indexes(floors),
        RFEngine._build_reflection_indexes(floors, heatmap_settings),
    )
    _RF_WORKER_CONTEXT_CACHE[key] = context
    _RF_WORKER_CONTEXT_ORDER.append(key)
    return context


def _rf_grid_multi_tile_worker(args):
    """Calculate one strip for every RF group, optionally writing shared arrays."""
    (
        context_key,
        context_path,
        xs,
        ys_slice,
        valid_mask_slice,
        start_index,
        disconnected,
        worker_cache_entries,
        group_aps,
        group_order,
        shared_specs,
    ) = args
    (
        floor,
        floors,
        patterns,
        include_inter_floor,
        heatmap_settings,
        wall_indexes,
        opening_indexes,
        reflection_indexes,
    ) = _rf_worker_context(context_key, context_path, worker_cache_entries)
    xs = np.asarray(xs, dtype=float)
    ys_slice = np.asarray(ys_slice, dtype=float)
    valid_mask_slice = None if valid_mask_slice is None else np.asarray(valid_mask_slice, dtype=bool)
    radio_links_by_group = {
        key: RFEngine._active_radio_links(floor, group_aps.get(key, []), include_inter_floor)
        for key in group_order
    }

    shared_handles = []
    if shared_specs:
        rssi_handle, rssi_stack = attach_shared_array(shared_specs[0]); shared_handles.append(rssi_handle)
        delay_handle, delay_stack = attach_shared_array(shared_specs[1]); shared_handles.append(delay_handle)
        count_handle, count_stack = attach_shared_array(shared_specs[2]); shared_handles.append(count_handle)
        tile_results = None
    else:
        tile_results = {}
        rssi_stack = delay_stack = count_stack = None
        for group_index, key in enumerate(group_order):
            if valid_mask_slice is None:
                tile = np.full((len(ys_slice), len(xs)), float(disconnected), dtype=float)
                delay_tile = np.zeros((len(ys_slice), len(xs)), dtype=float)
            else:
                tile = np.full((len(ys_slice), len(xs)), np.nan, dtype=float)
                tile[valid_mask_slice] = float(disconnected)
                delay_tile = np.full((len(ys_slice), len(xs)), np.nan, dtype=float)
                delay_tile[valid_mask_slice] = 0.0
            tile_results[key] = (tile, delay_tile, np.zeros((len(ys_slice), len(xs)), dtype=np.int16))

    # Vectorised unobstructed upper-bound masks remove AP/radio links that cannot
    # contribute even before any Shapely intersection or ray work starts.
    influence_masks: Dict[Tuple[int, int], np.ndarray] = {}
    if bool(getattr(heatmap_settings, "enable_tile_influence_pruning", True)):
        for group_index, key in enumerate(group_order):
            for link_index, (ap, radio) in enumerate(radio_links_by_group.get(key, [])):
                ap_floor = floors.get(ap.floor)
                if ap_floor is None:
                    continue
                dz = (float(ap_floor.elevation) + float(ap.mount_height_m)) - (float(floor.elevation) + float(ap.rx_height_m))
                eirp = float(radio.tx_power_dbm) + float(getattr(radio, "antenna_gain_dbi", 0.0) or 0.0) + 12.0
                upper = best_case_rssi_grid(
                    xs, ys_slice, ap.x, ap.y, dz,
                    RFEngine.free_space_loss_db_at_1m(radio.frequency_mhz),
                    ap.path_loss_exponent, eirp,
                    bool(getattr(heatmap_settings, "enable_numba_rf_kernels", True)),
                )
                cutoff = float(disconnected) + float(getattr(heatmap_settings, "tile_influence_margin_db", 8.0) or 8.0)
                mask = upper >= cutoff
                radius = RFEngine.cutoff_radius_m_for_radio(radio, heatmap_settings)
                if radius > 0.0:
                    xx, yy = np.meshgrid(xs, ys_slice)
                    mask &= ((xx - ap.x) ** 2 + (yy - ap.y) ** 2 + dz * dz) <= radius * radius
                influence_masks[(group_index, link_index)] = mask

    tile_bounds = (
        float(xs[0]) if len(xs) else 0.0,
        float(ys_slice[0]) if len(ys_slice) else 0.0,
        float(xs[-1]) if len(xs) else 0.0,
        float(ys_slice[-1]) if len(ys_slice) else 0.0,
    )
    local_indexes_by_ap: Dict[Tuple[str, float, float], Tuple[dict, dict]] = {}
    if bool(getattr(heatmap_settings, "enable_tile_local_geometry", True)) and len(xs) and len(ys_slice):
        search_pad = max(
            1.0,
            float(getattr(heatmap_settings, "reflection_search_radius_m", 18.0)),
            float(getattr(heatmap_settings, "diffraction_search_radius_m", 5.0)),
        )
        for key in group_order:
            for ap, _radio in radio_links_by_group.get(key, []):
                ap_key = (str(ap.name), round(float(ap.x), 5), round(float(ap.y), 5))
                if ap_key in local_indexes_by_ap:
                    continue
                envelope = box(
                    min(tile_bounds[0], float(ap.x)) - search_pad,
                    min(tile_bounds[1], float(ap.y)) - search_pad,
                    max(tile_bounds[2], float(ap.x)) + search_pad,
                    max(tile_bounds[3], float(ap.y)) + search_pad,
                )
                local_indexes_by_ap[ap_key] = RFEngine._tile_local_indexes(
                    wall_indexes, opening_indexes, floors.keys(), envelope
                )

    reflection_subsets: Dict[Tuple[str, float, float], ReflectionIndex] = {}
    if bool(getattr(heatmap_settings, "precompute_reflection_candidates_per_tile", True)):
        base_index = reflection_indexes.get(floor.name)
        if base_index is not None:
            maximum = max(4, int(getattr(heatmap_settings, "max_reflection_surfaces", 6)) * 4)
            radius = float(getattr(heatmap_settings, "reflection_search_radius_m", 18.0))
            for key in group_order:
                for ap, _radio in radio_links_by_group.get(key, []):
                    ap_key = (str(ap.name), round(float(ap.x), 5), round(float(ap.y), 5))
                    if ap_key not in reflection_subsets:
                        reflection_subsets[ap_key] = base_index.subset_for_source_tile(
                            (float(ap.x), float(ap.y)), tile_bounds, radius, maximum
                        )

    reflection_sequences_by_ap: Dict[Tuple[str, float, float], Tuple[Tuple[ReflectionSurface, ...], ...]] = {}
    if bool(getattr(heatmap_settings, "precompute_reflection_candidates_per_tile", True)):
        tile_centre = ((tile_bounds[0] + tile_bounds[2]) * 0.5, (tile_bounds[1] + tile_bounds[3]) * 0.5)
        for key in group_order:
            for ap, _radio in radio_links_by_group.get(key, []):
                ap_key = (str(ap.name), round(float(ap.x), 5), round(float(ap.y), 5))
                if ap_key in reflection_sequences_by_ap:
                    continue
                sequences = precompute_reflection_sequences(
                    (float(ap.x), float(ap.y)), tile_centre, reflection_subsets.get(ap_key),
                    int(getattr(heatmap_settings, "max_reflection_order", 1)),
                    int(getattr(heatmap_settings, "max_reflection_surfaces", 6)),
                    float(getattr(heatmap_settings, "reflection_search_radius_m", 18.0)),
                )
                if sequences:
                    reflection_sequences_by_ap[ap_key] = sequences

    try:
        for iy, yy in enumerate(ys_slice):
            global_iy = int(start_index) + iy
            yy_value = float(yy)
            for ix, xx in enumerate(xs):
                if valid_mask_slice is not None and not bool(valid_mask_slice[iy, ix]):
                    continue
                xx_value = float(xx)
                geometry_cache: Optional[Dict[object, object]] = {} if bool(
                    getattr(heatmap_settings, "reuse_path_geometry_across_frequencies", True)
                ) else None
                for group_index, key in enumerate(group_order):
                    samples: List[PropagationSample] = []
                    for link_index, (ap, radio) in enumerate(radio_links_by_group.get(key, [])):
                        influence = influence_masks.get((group_index, link_index))
                        if influence is not None and not bool(influence[iy, ix]):
                            continue
                        if not RFEngine.point_is_inside_radio_cutoff(
                            xx_value, yy_value, floor, ap, floors, radio, heatmap_settings
                        ):
                            continue
                        ap_key = (str(ap.name), round(float(ap.x), 5), round(float(ap.y), 5))
                        local_wall_indexes, local_opening_indexes = local_indexes_by_ap.get(
                            ap_key, (wall_indexes, opening_indexes)
                        )
                        samples.append(RFEngine.propagation_at(
                            xx_value, yy_value, floor, ap, floors, patterns, radio,
                            include_inter_floor, heatmap_settings, local_wall_indexes,
                            local_opening_indexes, reflection_indexes, geometry_cache,
                            reflection_subsets.get(ap_key),
                            reflection_sequences_by_ap.get(ap_key),
                        ))
                    combined = RFEngine.combine_ap_samples(samples, heatmap_settings, float(disconnected))
                    if shared_specs:
                        rssi_stack[group_index, global_iy, ix] = combined.rssi_dbm
                        delay_stack[group_index, global_iy, ix] = combined.delay_spread_ns
                        count_stack[group_index, global_iy, ix] = max(0, min(32767, int(combined.path_count)))
                    else:
                        tile, delay_tile, count_tile = tile_results[key]
                        tile[iy, ix] = combined.rssi_dbm
                        delay_tile[iy, ix] = combined.delay_spread_ns
                        count_tile[iy, ix] = max(0, min(32767, int(combined.path_count)))
        return int(start_index), int(len(ys_slice)), tile_results
    finally:
        for handle in shared_handles:
            try:
                handle.close()
            except Exception:
                pass


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
Z_AP_RULER = 52
Z_AP_RULER_LABEL = 54
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


class IFCElementAttenuationDialog(WallAttenuationDialog):
    """Individual RF editor for any imported non-wall IFC element."""

    def __init__(self, parent, element: IFCElement2D, bands: List[float], profiles: Dict[str, Dict[float, float]]):
        super().__init__(parent, element, bands, profiles)
        self.setWindowTitle("IFC element RF type and attenuation")


class BulkIFCAttenuationDialog(QDialog):
    """Edit one RF profile per unique IFC category/type/material group."""

    def __init__(
        self,
        parent,
        groups: List[Dict[str, object]],
        bands: List[float],
        profiles: Dict[str, Dict[float, float]],
        current_floor_name: str = "",
    ):
        super().__init__(parent)
        self.groups = list(groups)
        self.bands = [float(value) for value in bands]
        self.profiles = dict(profiles)
        self.setWindowTitle("Bulk IFC attenuation by type")
        self.resize(1180, 680)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each row represents one unique IFC element type rather than every instance. "
            "Use Ctrl-click or Shift-click to highlight multiple rows, then apply one preset "
            "or a shared set of frequency attenuation values to the complete selection. "
            "The Apply checkbox is set automatically for edited rows. Walls, doors, windows, "
            "slabs, roofs, structural objects and all other imported IFC elements are available. "
            "Zero dB leaves an element as visual context only."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QHBoxLayout()
        self.current_floor_only = QCheckBox(
            f"Apply only to matching instances on {current_floor_name}" if current_floor_name else "Apply only to current floor"
        )
        controls.addWidget(self.current_floor_only)
        controls.addStretch(1)
        controls.addWidget(QLabel("Selection category"))
        self.category_combo = QComboBox()
        categories = sorted({str(group.get("category", "Other")) for group in self.groups})
        self.category_combo.addItem("All categories")
        self.category_combo.addItems(categories)
        controls.addWidget(self.category_combo)
        select_category = QPushButton("Select category")
        select_category.clicked.connect(self._select_category)
        controls.addWidget(select_category)
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        controls.addWidget(select_all)
        clear_all = QPushButton("Clear selection")
        clear_all.clicked.connect(lambda: self._set_all_checked(False))
        controls.addWidget(clear_all)
        layout.addLayout(controls)

        preset_controls = QHBoxLayout()
        preset_controls.addWidget(QLabel("Attenuation preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(self.profiles.keys()))
        preset_controls.addWidget(self.preset_combo, 1)
        preset_button = QPushButton("Apply preset to highlighted / checked rows")
        preset_button.clicked.connect(self._apply_preset_to_checked)
        preset_controls.addWidget(preset_button)
        zero_button = QPushButton("Set highlighted / checked rows to 0 dB")
        zero_button.clicked.connect(self._zero_checked)
        preset_controls.addWidget(zero_button)
        layout.addLayout(preset_controls)

        multi_frame = QFrame()
        multi_frame.setFrameShape(QFrame.StyledPanel)
        multi_layout = QVBoxLayout(multi_frame)
        multi_layout.setContentsMargins(8, 6, 8, 6)
        multi_header = QHBoxLayout()
        multi_title = QLabel("<b>Shared values for highlighted rows</b>")
        multi_header.addWidget(multi_title)
        multi_header.addStretch(1)
        self.multi_selection_label = QLabel("0 rows highlighted")
        multi_header.addWidget(self.multi_selection_label)
        multi_layout.addLayout(multi_header)

        type_layout = QHBoxLayout()
        self.multi_type_enabled = QCheckBox("Set RF type")
        type_layout.addWidget(self.multi_type_enabled)
        self.multi_type_combo = QComboBox()
        self.multi_type_combo.setEditable(True)
        known_types = sorted(
            {str(group.get("display_type", "")).strip() for group in self.groups if str(group.get("display_type", "")).strip()}
            | {str(name).split(" / ", 1)[-1] for name in self.profiles.keys()}
        )
        self.multi_type_combo.addItems(known_types)
        type_layout.addWidget(self.multi_type_combo, 1)
        multi_layout.addLayout(type_layout)

        band_grid = QGridLayout()
        self.multi_band_enabled: Dict[float, QCheckBox] = {}
        self.multi_band_values: Dict[float, QDoubleSpinBox] = {}
        for column, band in enumerate(self.bands):
            enabled = QCheckBox(self._frequency_label(band))
            enabled.setChecked(True)
            value = QDoubleSpinBox()
            value.setRange(-200.0, 500.0)
            value.setDecimals(3)
            value.setSingleStep(0.5)
            value.setSuffix(" dB")
            self.multi_band_enabled[float(band)] = enabled
            self.multi_band_values[float(band)] = value
            band_grid.addWidget(enabled, 0, column)
            band_grid.addWidget(value, 1, column)
        multi_layout.addLayout(band_grid)

        multi_buttons = QHBoxLayout()
        load_current = QPushButton("Load values from current row")
        load_current.clicked.connect(self._load_multi_values_from_current_row)
        multi_buttons.addWidget(load_current)
        multi_buttons.addStretch(1)
        apply_multi = QPushButton("Apply shared values to highlighted rows")
        apply_multi.clicked.connect(self._apply_multi_values_to_highlighted)
        multi_buttons.addWidget(apply_multi)
        multi_layout.addLayout(multi_buttons)
        layout.addWidget(multi_frame)

        headers = ["Apply", "Category", "IFC / RF type", "Material", "Instances", "Current floor"]
        headers += [self._frequency_label(value) for value in self.bands]
        headers += ["Source IFC files"]
        self.table = QTableWidget(len(self.groups), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        for row, group in enumerate(self.groups):
            apply_item = QTableWidgetItem("")
            apply_item.setFlags((apply_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            apply_item.setCheckState(Qt.Unchecked)
            apply_item.setData(Qt.UserRole, json.dumps(list(group.get("key", ())), separators=(",", ":")))
            self.table.setItem(row, 0, apply_item)
            values = [
                str(group.get("category", "Other")),
                str(group.get("display_type", "Unknown")),
                str(group.get("material", "")),
                str(group.get("count", 0)),
                str(group.get("current_floor_count", 0)),
            ]
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                if column != 2:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, column, item)
            profile = dict(group.get("attenuation", {}) or {})
            for offset, band in enumerate(self.bands):
                value = float(profile.get(float(band), 0.0))
                self.table.setItem(row, 6 + offset, QTableWidgetItem(f"{value:.3f}"))
            source_item = QTableWidgetItem(str(group.get("sources", "")))
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 6 + len(self.bands), source_item)
        self.table.resizeColumnsToContents()
        self.table.itemSelectionChanged.connect(self._update_multi_selection_label)
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _frequency_label(mhz: float) -> str:
        return f"{mhz / 1000:g} GHz dB" if mhz >= 1000.0 else f"{mhz:g} MHz dB"

    def _set_all_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)

    def _select_category(self):
        category = self.category_combo.currentText()
        for row in range(self.table.rowCount()):
            matches = category == "All categories" or self.table.item(row, 1).text() == category
            if matches:
                self.table.item(row, 0).setCheckState(Qt.Checked)

    def _profile_value(self, profile: Dict[float, float], band: float) -> float:
        if not profile:
            return 0.0
        if float(band) in profile:
            return float(profile[float(band)])
        keys = sorted(float(value) for value in profile)
        nearest = min(keys, key=lambda value: abs(value - float(band)))
        return float(profile[nearest])

    def _highlighted_rows(self) -> List[int]:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return []
        return sorted({index.row() for index in selection_model.selectedRows()})

    def _checked_rows(self) -> List[int]:
        return [
            row for row in range(self.table.rowCount())
            if self.table.item(row, 0).checkState() == Qt.Checked
        ]

    def _target_rows(self) -> List[int]:
        highlighted = self._highlighted_rows()
        return highlighted if highlighted else self._checked_rows()

    def _mark_rows_for_apply(self, rows: Iterable[int]):
        for row in rows:
            item = self.table.item(int(row), 0)
            if item is not None:
                item.setCheckState(Qt.Checked)

    def _update_multi_selection_label(self):
        count = len(self._highlighted_rows())
        self.multi_selection_label.setText(f"{count} row{'s' if count != 1 else ''} highlighted")

    def _load_multi_values_from_current_row(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No IFC type selected", "Highlight a row before loading its values.")
            return
        type_item = self.table.item(row, 2)
        if type_item is not None:
            self.multi_type_combo.setCurrentText(type_item.text())
            self.multi_type_enabled.setChecked(True)
        for offset, band in enumerate(self.bands):
            item = self.table.item(row, 6 + offset)
            try:
                value = float(item.text()) if item is not None else 0.0
            except Exception:
                value = 0.0
            self.multi_band_values[float(band)].setValue(value)
            self.multi_band_enabled[float(band)].setChecked(True)

    def _apply_multi_values_to_highlighted(self):
        rows = self._highlighted_rows()
        if not rows:
            QMessageBox.information(
                self,
                "No IFC types highlighted",
                "Use Ctrl-click or Shift-click to highlight two or more IFC type rows first.",
            )
            return
        set_type = self.multi_type_enabled.isChecked()
        rf_type = self.multi_type_combo.currentText().strip()
        active_bands = [
            float(band) for band in self.bands
            if self.multi_band_enabled[float(band)].isChecked()
        ]
        if not set_type and not active_bands:
            QMessageBox.information(self, "No values enabled", "Enable RF type or at least one frequency value to apply.")
            return
        for row in rows:
            if set_type:
                self.table.item(row, 2).setText(rf_type)
            for offset, band in enumerate(self.bands):
                if float(band) not in active_bands:
                    continue
                self.table.item(row, 6 + offset).setText(
                    f"{self.multi_band_values[float(band)].value():.3f}"
                )
        self._mark_rows_for_apply(rows)
        self.multi_selection_label.setText(
            f"Applied shared values to {len(rows)} highlighted row{'s' if len(rows) != 1 else ''}"
        )

    def _apply_preset_to_checked(self):
        rows = self._target_rows()
        if not rows:
            QMessageBox.information(self, "No IFC types selected", "Highlight rows or tick their Apply boxes first.")
            return
        profile = self.profiles.get(self.preset_combo.currentText(), self.profiles.get("default", {}))
        preset_name = self.preset_combo.currentText()
        for row in rows:
            self.table.item(row, 2).setText(preset_name.split(" / ", 1)[-1])
            for offset, band in enumerate(self.bands):
                self.table.item(row, 6 + offset).setText(f"{self._profile_value(profile, band):.3f}")
        self._mark_rows_for_apply(rows)

    def _zero_checked(self):
        rows = self._target_rows()
        if not rows:
            QMessageBox.information(self, "No IFC types selected", "Highlight rows or tick their Apply boxes first.")
            return
        for row in rows:
            for offset in range(len(self.bands)):
                self.table.item(row, 6 + offset).setText("0.000")
        self._mark_rows_for_apply(rows)

    def values(self) -> Tuple[bool, List[Dict[str, object]]]:
        changes = []
        for row in range(self.table.rowCount()):
            apply_item = self.table.item(row, 0)
            if apply_item.checkState() != Qt.Checked:
                continue
            try:
                key = tuple(json.loads(str(apply_item.data(Qt.UserRole))))
            except Exception:
                continue
            attenuation = {}
            for offset, band in enumerate(self.bands):
                try:
                    attenuation[float(band)] = float(self.table.item(row, 6 + offset).text())
                except Exception:
                    attenuation[float(band)] = 0.0
            changes.append({
                "key": key,
                "rf_type": self.table.item(row, 2).text().strip(),
                "attenuation": attenuation,
            })
        return bool(self.current_floor_only.isChecked()), changes


class BulkAccessPointDialog(QDialog):
    """Apply selected physical and radio parameters to many APs at once."""

    def __init__(self, parent, selected_count: int, current_floor_count: int, total_count: int, pattern_names: List[str]):
        super().__init__(parent)
        self.setWindowTitle("Bulk access point parameters")
        self.resize(720, 720)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Choose the AP scope, tick only the parameters to change, then apply. "
            "Unticked values remain unchanged. Radio values can target every radio, "
            "enabled radios, the first radio, or the radio closest to a chosen frequency."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        scope_form = QFormLayout()
        self.scope_combo = QComboBox()
        self.scope_combo.addItem(f"Selected access points ({selected_count})", "selected")
        self.scope_combo.addItem(f"Current floor ({current_floor_count})", "floor")
        self.scope_combo.addItem(f"All floors ({total_count})", "all")
        self.scope_combo.setCurrentIndex(0 if selected_count else 1)
        scope_form.addRow("Apply to", self.scope_combo)
        self.radio_target_combo = QComboBox()
        self.radio_target_combo.addItem("All radios", "all")
        self.radio_target_combo.addItem("Enabled radios", "enabled")
        self.radio_target_combo.addItem("First radio only", "first")
        self.radio_target_combo.addItem("Radio nearest frequency", "frequency")
        scope_form.addRow("Radio target", self.radio_target_combo)
        self.target_frequency = QDoubleSpinBox()
        self.target_frequency.setRange(1.0, 100000.0)
        self.target_frequency.setValue(5000.0)
        self.target_frequency.setSuffix(" MHz")
        scope_form.addRow("Target frequency", self.target_frequency)
        layout.addLayout(scope_form)

        self.fields: Dict[str, Tuple[QCheckBox, QWidget]] = {}
        form = QFormLayout()

        def add_field(key: str, label: str, widget: QWidget):
            check = QCheckBox(label)
            check.setToolTip("Tick to overwrite this parameter on every AP in the selected scope.")
            self.fields[key] = (check, widget)
            form.addRow(check, widget)

        ap_type = QComboBox(); ap_type.addItems(list(AP_TYPE_PRESETS.keys()))
        add_field("ap_type", "AP type / symbol", ap_type)
        radio_profile = QComboBox(); radio_profile.addItem("Project default radios"); radio_profile.addItems(list(RADIO_PROFILE_PRESETS.keys()))
        add_field("radio_profile", "Replace radio profile", radio_profile)
        mount = QDoubleSpinBox(); mount.setRange(0.1, 50.0); mount.setValue(2.7); mount.setSuffix(" m")
        add_field("mount_height_m", "Mount height", mount)
        rx = QDoubleSpinBox(); rx.setRange(0.1, 10.0); rx.setValue(1.2); rx.setSuffix(" m")
        add_field("rx_height_m", "Receiver height", rx)
        azimuth = QDoubleSpinBox(); azimuth.setRange(-360.0, 360.0); azimuth.setSuffix("°")
        add_field("azimuth_deg", "Azimuth", azimuth)
        downtilt = QDoubleSpinBox(); downtilt.setRange(-90.0, 90.0); downtilt.setSuffix("°")
        add_field("downtilt_deg", "Downtilt", downtilt)
        ple = QDoubleSpinBox(); ple.setRange(1.0, 8.0); ple.setValue(2.2); ple.setDecimals(3)
        add_field("path_loss_exponent", "Path-loss exponent", ple)
        clients = QSpinBox(); clients.setRange(1, 100000); clients.setValue(50)
        add_field("max_clients", "Clients per AP", clients)
        enabled = QComboBox(); enabled.addItem("Enabled", True); enabled.addItem("Disabled", False)
        add_field("radio_enabled", "Radio enabled state", enabled)
        tx = QDoubleSpinBox(); tx.setRange(-50.0, 100.0); tx.setValue(20.0); tx.setSuffix(" dBm")
        add_field("tx_power_dbm", "Radio transmit power", tx)
        gain = QDoubleSpinBox(); gain.setRange(-50.0, 100.0); gain.setSuffix(" dBi")
        add_field("antenna_gain_dbi", "Additional antenna gain", gain)
        frequency = QDoubleSpinBox(); frequency.setRange(1.0, 100000.0); frequency.setValue(5000.0); frequency.setSuffix(" MHz")
        add_field("frequency_mhz", "Radio frequency", frequency)
        pattern = QComboBox(); pattern.addItems(pattern_names)
        add_field("antenna_pattern", "Antenna pattern", pattern)
        channel = QComboBox(); channel.setEditable(True); channel.addItems(["1", "6", "11", "36", "40", "44", "48", "5", "21", "37"])
        add_field("channel", "Channel", channel)
        width = QDoubleSpinBox(); width.setRange(0.01, 1000.0); width.setValue(40.0); width.setSuffix(" MHz")
        add_field("channel_width_mhz", "Channel width", width)
        occupancy = QDoubleSpinBox(); occupancy.setRange(0.0, 100.0); occupancy.setValue(20.0); occupancy.setSuffix(" %")
        add_field("spectrum_occupancy_percent", "Spectrum occupancy", occupancy)
        cutoff = QDoubleSpinBox(); cutoff.setRange(0.0, 10000.0); cutoff.setValue(0.0); cutoff.setSuffix(" m")
        add_field("cutoff_radius_m", "Radio cut-off radius", cutoff)

        field_widget = QWidget(); field_widget.setLayout(form)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(field_widget)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "scope": self.scope_combo.currentData(),
            "radio_target": self.radio_target_combo.currentData(),
            "target_frequency_mhz": float(self.target_frequency.value()),
            "changes": {},
        }
        for key, (check, widget) in self.fields.items():
            if not check.isChecked():
                continue
            if isinstance(widget, QDoubleSpinBox):
                value = float(widget.value())
            elif isinstance(widget, QSpinBox):
                value = int(widget.value())
            elif isinstance(widget, QComboBox):
                value = widget.currentData() if key == "radio_enabled" else widget.currentText()
            else:
                continue
            result["changes"][key] = value
        return result


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
        self.handover_enabled = QCheckBox("Require overlapping service from a second AP for handover")
        self.handover_enabled.setChecked(settings.handover_enabled)
        self.handover_target = QDoubleSpinBox(); self.handover_target.setRange(0.0, 100.0); self.handover_target.setSuffix(" %"); self.handover_target.setValue(settings.target_handover_percent)
        self.handover_margin = QDoubleSpinBox(); self.handover_margin.setRange(0.0, 30.0); self.handover_margin.setSuffix(" dB"); self.handover_margin.setValue(settings.handover_margin_db)
        self.handover_channels = QCheckBox("Prefer non-overlapping channels where AP handover coverage overlaps")
        self.handover_channels.setChecked(settings.prefer_non_overlapping_handover_channels)
        self.handover_enabled.toggled.connect(self.handover_target.setEnabled)
        self.handover_enabled.toggled.connect(self.handover_margin.setEnabled)
        self.handover_enabled.toggled.connect(self.handover_channels.setEnabled)
        self.handover_target.setEnabled(settings.handover_enabled)
        self.handover_margin.setEnabled(settings.handover_enabled)
        self.handover_channels.setEnabled(settings.handover_enabled)
        self.sample_spacing = QDoubleSpinBox(); self.sample_spacing.setRange(0.5, 25.0); self.sample_spacing.setSuffix(" m"); self.sample_spacing.setValue(settings.sample_spacing_m)
        self.candidate_spacing = QDoubleSpinBox(); self.candidate_spacing.setRange(1.0, 50.0); self.candidate_spacing.setSuffix(" m"); self.candidate_spacing.setValue(settings.candidate_spacing_m)
        self.minimum_spacing = QDoubleSpinBox(); self.minimum_spacing.setRange(0.0, 100.0); self.minimum_spacing.setSuffix(" m"); self.minimum_spacing.setValue(settings.minimum_ap_spacing_m)
        self.maximum_aps = QSpinBox(); self.maximum_aps.setRange(1, 10_000); self.maximum_aps.setValue(settings.maximum_aps)
        self.area_mode = QComboBox()
        self.area_mode.addItem("Automatic — eligible spaces, then shared boundaries, then walls", "auto")
        self.area_mode.addItem("Eligible IFC/manual/inferred spaces only", "spaces")
        self.area_mode.addItem("Shared planner boundaries only", "boundaries")
        self.area_mode.addItem("Infer floor footprint from IFC walls", "walls")
        area_index = self.area_mode.findData(settings.planning_area_mode)
        self.area_mode.setCurrentIndex(max(0, area_index))
        self.wall_margin = QDoubleSpinBox(); self.wall_margin.setRange(0.0, 100.0); self.wall_margin.setDecimals(2); self.wall_margin.setSuffix(" m"); self.wall_margin.setValue(settings.wall_footprint_margin_m)
        self.use_inferred_spaces = QCheckBox("Use inferred simulator spaces as predictive AP planning locations")
        self.use_inferred_spaces.setChecked(settings.use_inferred_spaces)
        self.use_inferred_spaces.setToolTip(
            "When enabled, wall-derived inferred spaces contribute coverage samples and AP candidate positions. "
            "Disable this to restrict space-based prediction to imported IFC and manually drawn spaces."
        )
        self.expected_clients = QSpinBox(); self.expected_clients.setRange(0, 1000000); self.expected_clients.setValue(settings.expected_clients)
        self.clients_per_ap = QSpinBox(); self.clients_per_ap.setRange(1, 10000); self.clients_per_ap.setValue(settings.clients_per_ap)
        self.keep_existing = QCheckBox("Count and retain manually placed APs on this floor"); self.keep_existing.setChecked(settings.keep_existing_aps)
        self.remove_planned = QCheckBox("Replace APs created by the previous planner run"); self.remove_planned.setChecked(settings.remove_previous_planned_aps)
        form.addRow("Target floor coverage", self.target)
        form.addRow("Frequency coverage rule", self.coverage_mode)
        form.addRow(self.handover_enabled)
        form.addRow("Target handover overlap", self.handover_target)
        form.addRow("Secondary AP RSSI allowance", self.handover_margin)
        form.addRow(self.handover_channels)
        form.addRow("Coverage sample spacing", self.sample_spacing)
        form.addRow("Candidate AP spacing", self.candidate_spacing)
        form.addRow("Minimum AP separation", self.minimum_spacing)
        form.addRow("Maximum planned APs", self.maximum_aps)
        form.addRow("Planning area source", self.area_mode)
        form.addRow(self.use_inferred_spaces)
        form.addRow("Inferred wall-footprint margin", self.wall_margin)
        form.addRow("Expected connected clients", self.expected_clients)
        form.addRow("Maximum clients per AP", self.clients_per_ap)
        form.addRow(self.keep_existing)
        form.addRow(self.remove_planned)
        layout.addLayout(form)

        note = QLabel("AP positions are selected from simulated RSSI values: the planner prioritises the samples with the largest dB shortfall instead of relying only on geometric spacing. When handover is enabled, a sample contributes to the handover target only when a second distinct AP reaches the configured secondary threshold. The second AP may use a different channel; channel allocation prefers non-overlapping channels in shared coverage areas. Spectrum occupancy reduces effective client capacity. Inferred spaces are optional predictive locations and can be excluded without deleting them. Shared rectangular and polygon planner boundaries apply to every IFC floor and remain a hard placement limit. The outer-boundary suggestion tool can trace and preview the outermost IFC wall chain before it is accepted.")
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
            handover_enabled=self.handover_enabled.isChecked(),
            target_handover_percent=float(self.handover_target.value()),
            handover_margin_db=float(self.handover_margin.value()),
            prefer_non_overlapping_handover_channels=self.handover_channels.isChecked(),
            sample_spacing_m=float(self.sample_spacing.value()), candidate_spacing_m=float(self.candidate_spacing.value()),
            minimum_ap_spacing_m=float(self.minimum_spacing.value()), maximum_aps=int(self.maximum_aps.value()),
            planning_area_mode=str(self.area_mode.currentData() or "auto"),
            wall_footprint_margin_m=float(self.wall_margin.value()),
            use_inferred_spaces=self.use_inferred_spaces.isChecked(),
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


class RFPerformanceSettingsDialog(QDialog):
    """Configure RF/IFC multiprocessing, adaptive sampling, caching and rendering."""

    def __init__(
        self,
        parent,
        settings: HeatmapSettings,
        has_planner_boundaries: bool = False,
        settings_path: Optional[Path] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("RF and IFC performance settings")
        self.resize(780, 720)
        self._has_planner_boundaries = bool(has_planner_boundaries)
        self._settings_path = Path(settings_path) if settings_path else Path(__file__).with_name("rf_heatmap_settings.json")

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Tune worker counts, adaptive sampling, caches and heatmap rendering for the available CPU and memory. "
            "A value of 0 for a worker count means automatic. On hybrid laptop processors, fewer workers can be faster "
            "than using every logical thread. Changes apply immediately to the current project when OK is pressed."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        laptop = QPushButton("Intel i7-1255U / laptop")
        laptop.setToolTip("Four RF/IFC workers, boundary-limited raster output, smaller caches and lighter working propagation.")
        laptop.clicked.connect(self._apply_laptop_preset)
        automatic = QPushButton("Automatic / workstation")
        automatic.setToolTip("Automatic worker counts and the application balanced defaults.")
        automatic.clicked.connect(self._apply_automatic_preset)
        fast = QPushButton("Fast interactive")
        fast.clicked.connect(self._apply_fast_preset)
        preset_row.addWidget(laptop)
        preset_row.addWidget(automatic)
        preset_row.addWidget(fast)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_workers_tab(settings)
        self._build_sampling_tab(settings)
        self._build_rendering_tab(settings)
        self._build_propagation_cost_tab(settings)

        self.save_global_defaults = QCheckBox(
            f"Also save these values as global defaults in {self._settings_path.name}"
        )
        self.save_global_defaults.setChecked(False)
        self.save_global_defaults.setToolTip(str(self._settings_path))
        layout.addWidget(self.save_global_defaults)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for widget in (
            self.profile,
            self.rf_workers,
            self.ifc_workers,
            self.boundary_filter,
            self.render_mode,
            self.progressive_updates,
            self.delay_spread,
            self.fading,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._update_summary)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._update_summary)
            else:
                widget.toggled.connect(self._update_summary)
        self._connect_enabled_state()
        self._update_enabled_state()
        self._update_summary()

    @staticmethod
    def _page_with_form() -> Tuple[QWidget, QFormLayout]:
        page = QWidget()
        outer = QVBoxLayout(page)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        outer.addWidget(scroll)
        return page, form

    @staticmethod
    def _automatic_spin(maximum: int = 64) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, maximum)
        spin.setSpecialValueText("Automatic")
        return spin

    def _build_workers_tab(self, settings: HeatmapSettings):
        page, form = self._page_with_form()
        self.profile = QComboBox()
        self.profile.addItem("Fast preview", "fast")
        self.profile.addItem("Balanced", "balanced")
        self.profile.addItem("Detailed / report", "detailed")
        self.profile.setCurrentIndex(max(0, self.profile.findData(str(settings.rf_calculation_profile))))
        form.addRow("Calculation profile", self.profile)

        self.enable_rf_mp = QCheckBox("Enable multiprocessing for RSSI calculation")
        self.enable_rf_mp.setChecked(bool(settings.enable_rf_multiprocessing))
        form.addRow(self.enable_rf_mp)
        self.rf_workers = self._automatic_spin(64)
        self.rf_workers.setValue(int(settings.max_rf_worker_processes))
        self.rf_workers.setToolTip("0 uses the logical processor count. Four is a good starting point on an i7-1255U.")
        form.addRow("Maximum RF worker processes", self.rf_workers)
        self.rf_min_points = QSpinBox()
        self.rf_min_points.setRange(1, 100_000_000)
        self.rf_min_points.setValue(int(settings.rf_multiprocessing_min_points))
        form.addRow("Minimum point-radio work for multiprocessing", self.rf_min_points)
        self.tile_rows = QSpinBox()
        self.tile_rows.setRange(1, 512)
        self.tile_rows.setValue(int(settings.rf_tile_rows))
        form.addRow("Preferred minimum RF tile rows", self.tile_rows)
        self.tiles_per_worker = QSpinBox()
        self.tiles_per_worker.setRange(1, 8)
        self.tiles_per_worker.setValue(int(settings.rf_tiles_per_worker))
        form.addRow("RF tiles per worker", self.tiles_per_worker)
        self.reuse_pool = QCheckBox("Keep the RF process pool alive between calculations")
        self.reuse_pool.setChecked(bool(settings.reuse_rf_process_pool))
        form.addRow(self.reuse_pool)
        self.worker_index_cache = QSpinBox()
        self.worker_index_cache.setRange(1, 8)
        self.worker_index_cache.setValue(int(settings.rf_worker_index_cache_entries))
        form.addRow("Resident floor-index revisions per worker", self.worker_index_cache)

        self.enable_ifc_mp = QCheckBox("Enable multiprocessing while loading IFC files")
        self.enable_ifc_mp.setChecked(bool(settings.enable_ifc_multiprocessing))
        form.addRow(self.enable_ifc_mp)
        self.ifc_workers = self._automatic_spin(64)
        self.ifc_workers.setValue(int(settings.max_ifc_loader_processes))
        form.addRow("Maximum IFC loader processes", self.ifc_workers)
        self.huge_ifc_workers = QSpinBox()
        self.huge_ifc_workers.setRange(1, 8)
        self.huge_ifc_workers.setValue(int(settings.max_parallel_huge_ifc_processes))
        self.huge_ifc_workers.setToolTip("Keep this at 1 for very large IFC files to avoid multiplying memory use.")
        form.addRow("Parallel huge-IFC processes", self.huge_ifc_workers)

        self.boundary_filter = QCheckBox("Limit RSSI calculation and results to accepted planner boundaries")
        self.boundary_filter.setChecked(bool(settings.ignore_results_outside_planner_boundaries))
        form.addRow(self.boundary_filter)
        self.boundary_note = QLabel(
            "Accepted boundaries are present." if self._has_planner_boundaries else
            "No planner boundaries are currently saved in this project. This option will take effect after a boundary is drawn or accepted."
        )
        self.boundary_note.setWordWrap(True)
        form.addRow("Boundary status", self.boundary_note)
        self.tabs.addTab(page, "Workers and scope")

    def _build_sampling_tab(self, settings: HeatmapSettings):
        page, form = self._page_with_form()
        self.adaptive_grid = QCheckBox("Use adaptive coarse-to-fine sampling")
        self.adaptive_grid.setChecked(bool(settings.enable_adaptive_rf_grid))
        form.addRow(self.adaptive_grid)
        self.coarse_resolution = QDoubleSpinBox()
        self.coarse_resolution.setRange(0.25, 20.0)
        self.coarse_resolution.setDecimals(2)
        self.coarse_resolution.setValue(float(settings.adaptive_coarse_resolution_m))
        self.coarse_resolution.setSuffix(" m")
        form.addRow("Coarse-grid spacing", self.coarse_resolution)
        self.gradient_threshold = QDoubleSpinBox()
        self.gradient_threshold.setRange(0.05, 50.0)
        self.gradient_threshold.setDecimals(2)
        self.gradient_threshold.setValue(float(settings.adaptive_gradient_threshold_db_per_m))
        self.gradient_threshold.setSuffix(" dB/m")
        form.addRow("RSSI gradient refinement threshold", self.gradient_threshold)
        self.threshold_margin = QDoubleSpinBox()
        self.threshold_margin.setRange(0.1, 30.0)
        self.threshold_margin.setValue(float(settings.adaptive_threshold_margin_db))
        self.threshold_margin.setSuffix(" dB")
        form.addRow("Coverage-threshold refinement margin", self.threshold_margin)
        self.geometry_buffer = QDoubleSpinBox()
        self.geometry_buffer.setRange(0.0, 30.0)
        self.geometry_buffer.setValue(float(settings.adaptive_geometry_buffer_m))
        self.geometry_buffer.setSuffix(" m")
        form.addRow("Geometry refinement buffer", self.geometry_buffer)
        self.ap_refine_radius = QDoubleSpinBox()
        self.ap_refine_radius.setRange(0.5, 100.0)
        self.ap_refine_radius.setValue(float(settings.adaptive_ap_refine_radius_m))
        self.ap_refine_radius.setSuffix(" m")
        form.addRow("AP refinement radius", self.ap_refine_radius)

        self.per_ap_cache = QCheckBox("Cache each AP/frequency field for incremental recalculation")
        self.per_ap_cache.setChecked(bool(settings.enable_per_ap_heatmap_cache))
        form.addRow(self.per_ap_cache)
        self.cache_entries = QSpinBox()
        self.cache_entries.setRange(1, 10_000)
        self.cache_entries.setValue(int(settings.per_ap_heatmap_cache_entries))
        form.addRow("Maximum AP-field cache entries", self.cache_entries)
        self.cache_mb = QSpinBox()
        self.cache_mb.setRange(32, 32_768)
        self.cache_mb.setValue(int(settings.per_ap_heatmap_cache_mb))
        self.cache_mb.setSuffix(" MB")
        form.addRow("Maximum AP-field cache memory", self.cache_mb)
        self.reuse_path_geometry = QCheckBox("Reuse frequency-independent ray geometry across bands")
        self.reuse_path_geometry.setChecked(bool(settings.reuse_path_geometry_across_frequencies))
        form.addRow(self.reuse_path_geometry)
        self.precompute_reflections = QCheckBox("Precompute reflection candidates per AP/tile")
        self.precompute_reflections.setChecked(bool(settings.precompute_reflection_candidates_per_tile))
        form.addRow(self.precompute_reflections)
        self.tile_influence = QCheckBox("Reject APs that cannot influence an entire tile")
        self.tile_influence.setChecked(bool(settings.enable_tile_influence_pruning))
        form.addRow(self.tile_influence)
        self.tile_influence_margin = QDoubleSpinBox()
        self.tile_influence_margin.setRange(0.0, 40.0)
        self.tile_influence_margin.setValue(float(settings.tile_influence_margin_db))
        self.tile_influence_margin.setSuffix(" dB")
        form.addRow("Tile influence safety margin", self.tile_influence_margin)
        self.tile_local_geometry = QCheckBox("Use tile-local wall/opening/reflection geometry")
        self.tile_local_geometry.setChecked(bool(settings.enable_tile_local_geometry))
        form.addRow(self.tile_local_geometry)
        self.numba_kernels = QCheckBox("Use compiled Numba kernels when available")
        self.numba_kernels.setChecked(bool(settings.enable_numba_rf_kernels))
        form.addRow(self.numba_kernels)
        self.shared_memory = QCheckBox("Use shared-memory worker result arrays")
        self.shared_memory.setChecked(bool(settings.use_shared_memory_rf_results))
        form.addRow(self.shared_memory)
        self.tabs.addTab(page, "Sampling and caches")

    def _build_rendering_tab(self, settings: HeatmapSettings):
        page, form = self._page_with_form()
        self.render_mode = QComboBox()
        self.render_mode.addItem("Raster heatmap only (fastest)", "raster")
        self.render_mode.addItem("Raster heatmap with isolines", "raster_contours")
        self.render_mode.addItem("Filled contour polygons (slowest)", "contours")
        self.render_mode.setCurrentIndex(max(0, self.render_mode.findData(str(settings.heatmap_render_mode))))
        form.addRow("Heatmap rendering", self.render_mode)
        self.contour_interpolation = QSpinBox()
        self.contour_interpolation.setRange(1, 16)
        self.contour_interpolation.setValue(int(settings.contour_interpolation_factor))
        form.addRow("Contour interpolation factor", self.contour_interpolation)
        self.progressive_updates = QCheckBox("Redraw coarse and completed tile results during calculation")
        self.progressive_updates.setChecked(bool(settings.progressive_heatmap_updates))
        form.addRow(self.progressive_updates)
        self.progressive_percent = QSpinBox()
        self.progressive_percent.setRange(5, 100)
        self.progressive_percent.setValue(int(settings.progressive_update_percent))
        self.progressive_percent.setSuffix(" %")
        form.addRow("Progressive update interval", self.progressive_percent)
        self.interactive_preview = QCheckBox("Calculate a background preview after AP movement")
        self.interactive_preview.setChecked(bool(settings.interactive_preview_enabled))
        form.addRow(self.interactive_preview)
        self.preview_delay = QSpinBox()
        self.preview_delay.setRange(50, 10_000)
        self.preview_delay.setValue(int(settings.interactive_preview_delay_ms))
        self.preview_delay.setSuffix(" ms")
        form.addRow("AP-move preview delay", self.preview_delay)
        self.preview_resolution = QDoubleSpinBox()
        self.preview_resolution.setRange(0.25, 20.0)
        self.preview_resolution.setDecimals(2)
        self.preview_resolution.setValue(float(settings.interactive_preview_resolution_m))
        self.preview_resolution.setSuffix(" m")
        form.addRow("AP-move preview spacing", self.preview_resolution)
        self.tabs.addTab(page, "Rendering and preview")

    def _build_propagation_cost_tab(self, settings: HeatmapSettings):
        page, form = self._page_with_form()
        note = QLabel(
            "These controls change RF fidelity as well as speed. Use lighter values while placing APs and restore the Detailed profile for final reports."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.reflections = QCheckBox("Enable multipath reflections")
        self.reflections.setChecked(bool(settings.enable_multipath_reflections))
        form.addRow(self.reflections)
        self.reflection_order = QSpinBox()
        self.reflection_order.setRange(0, 3)
        self.reflection_order.setValue(int(settings.max_reflection_order))
        form.addRow("Maximum reflection order", self.reflection_order)
        self.reflection_surfaces = QSpinBox()
        self.reflection_surfaces.setRange(1, 24)
        self.reflection_surfaces.setValue(int(settings.max_reflection_surfaces))
        form.addRow("Candidate reflection surfaces", self.reflection_surfaces)
        self.reflection_paths = QSpinBox()
        self.reflection_paths.setRange(0, 64)
        self.reflection_paths.setValue(int(settings.max_reflection_paths))
        form.addRow("Retained reflection paths", self.reflection_paths)
        self.diffraction = QCheckBox("Enable corner diffraction")
        self.diffraction.setChecked(bool(settings.enable_corner_diffraction))
        form.addRow(self.diffraction)
        self.diffraction_paths = QSpinBox()
        self.diffraction_paths.setRange(0, 32)
        self.diffraction_paths.setValue(int(settings.max_diffraction_paths))
        form.addRow("Retained diffraction paths", self.diffraction_paths)
        self.fading = QCheckBox("Enable deterministic small-scale fading")
        self.fading.setChecked(bool(settings.enable_small_scale_fading))
        form.addRow(self.fading)
        self.delay_spread = QCheckBox("Calculate RMS delay spread")
        self.delay_spread.setChecked(bool(settings.calculate_delay_spread))
        form.addRow(self.delay_spread)
        self.relative_path_cutoff = QDoubleSpinBox()
        self.relative_path_cutoff.setRange(0.0, 80.0)
        self.relative_path_cutoff.setValue(float(settings.multipath_relative_power_cutoff_db))
        self.relative_path_cutoff.setSuffix(" dB")
        form.addRow("Discard paths below strongest by", self.relative_path_cutoff)
        self.tabs.addTab(page, "Propagation cost")

    def _connect_enabled_state(self):
        self.enable_rf_mp.toggled.connect(self._update_enabled_state)
        self.enable_ifc_mp.toggled.connect(self._update_enabled_state)
        self.adaptive_grid.toggled.connect(self._update_enabled_state)
        self.per_ap_cache.toggled.connect(self._update_enabled_state)
        self.tile_influence.toggled.connect(self._update_enabled_state)
        self.progressive_updates.toggled.connect(self._update_enabled_state)
        self.interactive_preview.toggled.connect(self._update_enabled_state)
        self.reflections.toggled.connect(self._update_enabled_state)
        self.diffraction.toggled.connect(self._update_enabled_state)

    def _update_enabled_state(self, *_):
        for widget in (self.rf_workers, self.rf_min_points, self.tile_rows, self.tiles_per_worker, self.reuse_pool, self.worker_index_cache):
            widget.setEnabled(self.enable_rf_mp.isChecked())
        for widget in (self.ifc_workers, self.huge_ifc_workers):
            widget.setEnabled(self.enable_ifc_mp.isChecked())
        for widget in (self.coarse_resolution, self.gradient_threshold, self.threshold_margin, self.geometry_buffer, self.ap_refine_radius):
            widget.setEnabled(self.adaptive_grid.isChecked())
        for widget in (self.cache_entries, self.cache_mb):
            widget.setEnabled(self.per_ap_cache.isChecked())
        self.tile_influence_margin.setEnabled(self.tile_influence.isChecked())
        self.progressive_percent.setEnabled(self.progressive_updates.isChecked())
        for widget in (self.preview_delay, self.preview_resolution):
            widget.setEnabled(self.interactive_preview.isChecked())
        for widget in (self.reflection_order, self.reflection_surfaces, self.reflection_paths):
            widget.setEnabled(self.reflections.isChecked())
        self.diffraction_paths.setEnabled(self.diffraction.isChecked())
        self._update_summary()

    def _apply_laptop_preset(self):
        self.profile.setCurrentIndex(max(0, self.profile.findData("balanced")))
        self.enable_rf_mp.setChecked(True)
        self.rf_workers.setValue(4)
        self.enable_ifc_mp.setChecked(True)
        self.ifc_workers.setValue(4)
        self.huge_ifc_workers.setValue(1)
        self.boundary_filter.setChecked(True)
        self.adaptive_grid.setChecked(True)
        self.coarse_resolution.setValue(4.0)
        self.per_ap_cache.setChecked(True)
        self.cache_entries.setValue(96)
        self.cache_mb.setValue(384)
        self.render_mode.setCurrentIndex(max(0, self.render_mode.findData("raster")))
        self.contour_interpolation.setValue(2)
        self.progressive_updates.setChecked(False)
        self.interactive_preview.setChecked(True)
        self.preview_resolution.setValue(4.0)
        self.reflections.setChecked(True)
        self.reflection_order.setValue(1)
        self.reflection_surfaces.setValue(4)
        self.reflection_paths.setValue(4)
        self.diffraction.setChecked(True)
        self.diffraction_paths.setValue(1)
        self.fading.setChecked(False)
        self.delay_spread.setChecked(False)
        self.relative_path_cutoff.setValue(25.0)
        self._update_enabled_state()

    def _apply_automatic_preset(self):
        defaults = HeatmapSettings.default()
        self.profile.setCurrentIndex(max(0, self.profile.findData("balanced")))
        self.enable_rf_mp.setChecked(True)
        self.rf_workers.setValue(0)
        self.rf_min_points.setValue(defaults.rf_multiprocessing_min_points)
        self.tile_rows.setValue(defaults.rf_tile_rows)
        self.tiles_per_worker.setValue(defaults.rf_tiles_per_worker)
        self.reuse_pool.setChecked(True)
        self.worker_index_cache.setValue(defaults.rf_worker_index_cache_entries)
        self.enable_ifc_mp.setChecked(True)
        self.ifc_workers.setValue(0)
        self.huge_ifc_workers.setValue(1)
        self.adaptive_grid.setChecked(True)
        self.coarse_resolution.setValue(3.0)
        self.per_ap_cache.setChecked(True)
        self.cache_entries.setValue(192)
        self.cache_mb.setValue(768)
        self.render_mode.setCurrentIndex(max(0, self.render_mode.findData("raster_contours")))
        self.contour_interpolation.setValue(4)
        self.progressive_updates.setChecked(True)
        self.reflections.setChecked(True)
        self.reflection_order.setValue(1)
        self.reflection_surfaces.setValue(6)
        self.reflection_paths.setValue(8)
        self.diffraction.setChecked(True)
        self.diffraction_paths.setValue(3)
        self.fading.setChecked(True)
        self.delay_spread.setChecked(True)
        self.relative_path_cutoff.setValue(30.0)
        self._update_enabled_state()

    def _apply_fast_preset(self):
        self._apply_laptop_preset()
        self.profile.setCurrentIndex(max(0, self.profile.findData("fast")))
        self.coarse_resolution.setValue(5.0)
        self.reflection_paths.setValue(2)
        self.diffraction.setChecked(False)
        self.relative_path_cutoff.setValue(20.0)
        self.preview_resolution.setValue(5.0)
        self._update_enabled_state()

    def _update_summary(self, *_):
        rf_workers = "automatic" if self.rf_workers.value() == 0 else str(self.rf_workers.value())
        ifc_workers = "automatic" if self.ifc_workers.value() == 0 else str(self.ifc_workers.value())
        boundary = "on" if self.boundary_filter.isChecked() else "off"
        warning = ""
        if self.boundary_filter.isChecked() and not self._has_planner_boundaries:
            warning = " Boundary limiting is selected, but this project currently contains no planner boundaries."
        self.summary.setText(
            f"Current selection: {self.profile.currentText()}, RF workers {rf_workers}, IFC workers {ifc_workers}, "
            f"{self.render_mode.currentText().lower()}, boundary filter {boundary}." + warning
        )

    @property
    def global_settings_path(self) -> Path:
        return self._settings_path

    def apply_to(self, settings: HeatmapSettings):
        settings.rf_calculation_profile = str(self.profile.currentData() or "balanced")
        settings.enable_rf_multiprocessing = self.enable_rf_mp.isChecked()
        settings.max_rf_worker_processes = int(self.rf_workers.value())
        settings.rf_multiprocessing_min_points = int(self.rf_min_points.value())
        settings.rf_tile_rows = int(self.tile_rows.value())
        settings.rf_tiles_per_worker = int(self.tiles_per_worker.value())
        settings.reuse_rf_process_pool = self.reuse_pool.isChecked()
        settings.rf_worker_index_cache_entries = int(self.worker_index_cache.value())
        settings.enable_ifc_multiprocessing = self.enable_ifc_mp.isChecked()
        settings.max_ifc_loader_processes = int(self.ifc_workers.value())
        settings.max_parallel_huge_ifc_processes = int(self.huge_ifc_workers.value())
        settings.ignore_results_outside_planner_boundaries = self.boundary_filter.isChecked()
        settings.enable_adaptive_rf_grid = self.adaptive_grid.isChecked()
        settings.adaptive_coarse_resolution_m = float(self.coarse_resolution.value())
        settings.adaptive_gradient_threshold_db_per_m = float(self.gradient_threshold.value())
        settings.adaptive_threshold_margin_db = float(self.threshold_margin.value())
        settings.adaptive_geometry_buffer_m = float(self.geometry_buffer.value())
        settings.adaptive_ap_refine_radius_m = float(self.ap_refine_radius.value())
        settings.enable_per_ap_heatmap_cache = self.per_ap_cache.isChecked()
        settings.per_ap_heatmap_cache_entries = int(self.cache_entries.value())
        settings.per_ap_heatmap_cache_mb = int(self.cache_mb.value())
        settings.reuse_path_geometry_across_frequencies = self.reuse_path_geometry.isChecked()
        settings.precompute_reflection_candidates_per_tile = self.precompute_reflections.isChecked()
        settings.enable_tile_influence_pruning = self.tile_influence.isChecked()
        settings.tile_influence_margin_db = float(self.tile_influence_margin.value())
        settings.enable_tile_local_geometry = self.tile_local_geometry.isChecked()
        settings.enable_numba_rf_kernels = self.numba_kernels.isChecked()
        settings.use_shared_memory_rf_results = self.shared_memory.isChecked()
        settings.heatmap_render_mode = str(self.render_mode.currentData() or "raster")
        settings.contour_interpolation_factor = int(self.contour_interpolation.value())
        settings.progressive_heatmap_updates = self.progressive_updates.isChecked()
        settings.progressive_update_percent = int(self.progressive_percent.value())
        settings.interactive_preview_enabled = self.interactive_preview.isChecked()
        settings.interactive_preview_delay_ms = int(self.preview_delay.value())
        settings.interactive_preview_resolution_m = float(self.preview_resolution.value())
        settings.enable_multipath_reflections = self.reflections.isChecked()
        settings.max_reflection_order = int(self.reflection_order.value())
        settings.max_reflection_surfaces = int(self.reflection_surfaces.value())
        settings.max_reflection_paths = int(self.reflection_paths.value())
        settings.enable_corner_diffraction = self.diffraction.isChecked()
        settings.max_diffraction_paths = int(self.diffraction_paths.value())
        settings.enable_small_scale_fading = self.fading.isChecked()
        settings.calculate_delay_spread = self.delay_spread.isChecked()
        settings.multipath_relative_power_cutoff_db = float(self.relative_path_cutoff.value())


class PropagationSettingsDialog(QDialog):
    """Configure bounded ray tracing, fading, delay spread and AP combination."""

    def __init__(self, parent, settings: HeatmapSettings):
        super().__init__(parent)
        self.setWindowTitle("RF propagation model")
        self.resize(680, 650)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "The direct 3D path-loss and IFC penetration model always remains active. "
            "These controls add coherent reflected/diffracted rays, deterministic small-scale fading, "
            "delay-spread calculation and optional received-power summation across APs. Higher reflection "
            "orders increase calculation time rapidly."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        self.calculation_profile = QComboBox()
        self.calculation_profile.addItem("Fast preview", "fast")
        self.calculation_profile.addItem("Balanced", "balanced")
        self.calculation_profile.addItem("Detailed / report", "detailed")
        profile_index = self.calculation_profile.findData(str(settings.rf_calculation_profile))
        self.calculation_profile.setCurrentIndex(max(0, profile_index))
        self.calculation_profile.setToolTip(
            "Fast uses a coarser adaptive pass and lighter multipath; Balanced retains normal study detail; "
            "Detailed uses the requested grid and full configured propagation options."
        )
        form.addRow("Calculation profile", self.calculation_profile)

        self.adaptive_grid = QCheckBox("Use adaptive coarse-to-fine RF sampling")
        self.adaptive_grid.setChecked(bool(settings.enable_adaptive_rf_grid))
        form.addRow(self.adaptive_grid)

        self.coarse_resolution = QDoubleSpinBox()
        self.coarse_resolution.setRange(0.25, 20.0)
        self.coarse_resolution.setDecimals(2)
        self.coarse_resolution.setValue(float(settings.adaptive_coarse_resolution_m))
        self.coarse_resolution.setSuffix(" m")
        form.addRow("Adaptive coarse spacing", self.coarse_resolution)

        self.per_ap_cache = QCheckBox("Cache each AP field for incremental recalculation")
        self.per_ap_cache.setChecked(bool(settings.enable_per_ap_heatmap_cache))
        form.addRow(self.per_ap_cache)

        self.shared_memory = QCheckBox("Use shared-memory worker output arrays")
        self.shared_memory.setChecked(bool(settings.use_shared_memory_rf_results))
        form.addRow(self.shared_memory)

        self.numba_kernels = QCheckBox("Use optional compiled numerical kernels when Numba is installed")
        self.numba_kernels.setChecked(bool(settings.enable_numba_rf_kernels))
        form.addRow(self.numba_kernels)

        self.progressive_updates = QCheckBox("Show progressive coarse/fine heatmap updates")
        self.progressive_updates.setChecked(bool(settings.progressive_heatmap_updates))
        form.addRow(self.progressive_updates)

        self.render_mode = QComboBox()
        self.render_mode.addItem("Raster heatmap with contour lines", "raster_contours")
        self.render_mode.addItem("Raster heatmap only (fastest)", "raster")
        self.render_mode.addItem("Filled contour polygons", "contours")
        render_index = self.render_mode.findData(str(settings.heatmap_render_mode))
        self.render_mode.setCurrentIndex(max(0, render_index))
        form.addRow("Heatmap rendering", self.render_mode)

        self.relative_path_cutoff = QDoubleSpinBox()
        self.relative_path_cutoff.setRange(0.0, 80.0)
        self.relative_path_cutoff.setDecimals(1)
        self.relative_path_cutoff.setValue(float(settings.multipath_relative_power_cutoff_db))
        self.relative_path_cutoff.setSuffix(" dB")
        self.relative_path_cutoff.setToolTip(
            "Discard reflected/diffracted paths this far below the strongest path before coherent summation."
        )
        form.addRow("Multipath power pruning", self.relative_path_cutoff)

        self.interactive_preview = QCheckBox("Recalculate a fast preview after AP movement")
        self.interactive_preview.setChecked(bool(settings.interactive_preview_enabled))
        form.addRow(self.interactive_preview)

        self.reflections = QCheckBox("Enable wall and IFC-element reflections")
        self.reflections.setChecked(bool(settings.enable_multipath_reflections))
        form.addRow(self.reflections)

        self.reflection_order = QSpinBox()
        self.reflection_order.setRange(0, 3)
        self.reflection_order.setValue(int(settings.max_reflection_order))
        self.reflection_order.setToolTip("1 = first-order reflections; 2 or 3 enables higher-order image-source paths.")
        form.addRow("Maximum reflection order", self.reflection_order)

        self.reflection_surfaces = QSpinBox()
        self.reflection_surfaces.setRange(1, 24)
        self.reflection_surfaces.setValue(int(settings.max_reflection_surfaces))
        form.addRow("Candidate surfaces per link", self.reflection_surfaces)

        self.reflection_paths = QSpinBox()
        self.reflection_paths.setRange(0, 64)
        self.reflection_paths.setValue(int(settings.max_reflection_paths))
        form.addRow("Retained reflection paths", self.reflection_paths)

        self.reflection_radius = QDoubleSpinBox()
        self.reflection_radius.setRange(0.5, 200.0)
        self.reflection_radius.setValue(float(settings.reflection_search_radius_m))
        self.reflection_radius.setSuffix(" m")
        form.addRow("Reflection search radius", self.reflection_radius)

        self.minimum_reflection = QDoubleSpinBox()
        self.minimum_reflection.setRange(0.0, 1.0)
        self.minimum_reflection.setDecimals(3)
        self.minimum_reflection.setSingleStep(0.005)
        self.minimum_reflection.setValue(float(settings.minimum_reflection_coefficient))
        form.addRow("Minimum reflection coefficient", self.minimum_reflection)

        self.diffraction = QCheckBox("Enable corner diffraction when the direct path is blocked")
        self.diffraction.setChecked(bool(settings.enable_corner_diffraction))
        form.addRow(self.diffraction)

        self.diffraction_paths = QSpinBox()
        self.diffraction_paths.setRange(0, 32)
        self.diffraction_paths.setValue(int(settings.max_diffraction_paths))
        form.addRow("Retained diffraction paths", self.diffraction_paths)

        self.diffraction_radius = QDoubleSpinBox()
        self.diffraction_radius.setRange(0.25, 100.0)
        self.diffraction_radius.setValue(float(settings.diffraction_search_radius_m))
        self.diffraction_radius.setSuffix(" m")
        form.addRow("Corner search distance", self.diffraction_radius)

        self.diffraction_loss = QDoubleSpinBox()
        self.diffraction_loss.setRange(0.0, 45.0)
        self.diffraction_loss.setValue(float(settings.minimum_diffraction_loss_db))
        self.diffraction_loss.setSuffix(" dB")
        form.addRow("Minimum diffraction loss", self.diffraction_loss)

        self.fading = QCheckBox("Enable deterministic spatial small-scale fading")
        self.fading.setChecked(bool(settings.enable_small_scale_fading))
        form.addRow(self.fading)

        self.fading_sigma = QDoubleSpinBox()
        self.fading_sigma.setRange(0.0, 20.0)
        self.fading_sigma.setValue(float(settings.small_scale_fading_sigma_db))
        self.fading_sigma.setSuffix(" dB")
        form.addRow("Residual fading strength", self.fading_sigma)

        self.fading_distance = QDoubleSpinBox()
        self.fading_distance.setRange(0.05, 20.0)
        self.fading_distance.setValue(float(settings.fading_correlation_distance_m))
        self.fading_distance.setSuffix(" m")
        form.addRow("Fading correlation distance", self.fading_distance)

        self.fading_seed = QSpinBox()
        self.fading_seed.setRange(-2_000_000_000, 2_000_000_000)
        self.fading_seed.setValue(int(settings.fading_seed))
        form.addRow("Fading seed", self.fading_seed)

        self.delay_spread = QCheckBox("Calculate RMS delay spread for each heatmap point")
        self.delay_spread.setChecked(bool(settings.calculate_delay_spread))
        form.addRow(self.delay_spread)

        self.ap_combination = QComboBox()
        self.ap_combination.addItem("Strongest serving AP (coverage RSSI)", "strongest")
        self.ap_combination.addItem("Incoherent power sum (total received RF power)", "power_sum")
        index = self.ap_combination.findData(str(settings.combined_ap_mode))
        self.ap_combination.setCurrentIndex(max(0, index))
        form.addRow("Multiple-AP combination", self.ap_combination)

        self.boundary_filter = QCheckBox("Ignore RSSI results outside shared planner boundaries")
        self.boundary_filter.setChecked(bool(settings.ignore_results_outside_planner_boundaries))
        self.boundary_filter.setToolTip(
            "When shared rectangular or polygon boundaries exist, points outside their union are skipped during calculation and omitted from heatmaps, hover results, statistics, CSV and PDF output."
        )
        form.addRow(self.boundary_filter)

        form_widget = QWidget()
        form_widget.setLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, 1)

        performance = QLabel(
            "Recommended interactive setting: reflection order 1, six candidate surfaces and eight retained paths. "
            "Use order 2 or 3 for final studies at a coarser heatmap resolution. Material reflection properties "
            "remain editable in rf_heatmap_settings.json."
        )
        performance.setWordWrap(True)
        layout.addWidget(performance)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.reflections.toggled.connect(self._update_enabled_state)
        self.diffraction.toggled.connect(self._update_enabled_state)
        self.fading.toggled.connect(self._update_enabled_state)
        self._update_enabled_state()

    def _update_enabled_state(self, *_):
        reflection_enabled = self.reflections.isChecked()
        for widget in (
            self.reflection_order,
            self.reflection_surfaces,
            self.reflection_paths,
            self.reflection_radius,
            self.minimum_reflection,
        ):
            widget.setEnabled(reflection_enabled)
        diffraction_enabled = self.diffraction.isChecked()
        for widget in (self.diffraction_paths, self.diffraction_radius, self.diffraction_loss):
            widget.setEnabled(diffraction_enabled)
        fading_enabled = self.fading.isChecked()
        for widget in (self.fading_sigma, self.fading_distance, self.fading_seed):
            widget.setEnabled(fading_enabled)

    def apply_to(self, settings: HeatmapSettings):
        settings.rf_calculation_profile = str(self.calculation_profile.currentData() or "balanced")
        settings.enable_adaptive_rf_grid = self.adaptive_grid.isChecked()
        settings.adaptive_coarse_resolution_m = float(self.coarse_resolution.value())
        settings.enable_per_ap_heatmap_cache = self.per_ap_cache.isChecked()
        settings.use_shared_memory_rf_results = self.shared_memory.isChecked()
        settings.enable_numba_rf_kernels = self.numba_kernels.isChecked()
        settings.progressive_heatmap_updates = self.progressive_updates.isChecked()
        settings.heatmap_render_mode = str(self.render_mode.currentData() or "raster_contours")
        settings.multipath_relative_power_cutoff_db = float(self.relative_path_cutoff.value())
        settings.interactive_preview_enabled = self.interactive_preview.isChecked()
        settings.enable_multipath_reflections = self.reflections.isChecked()
        settings.max_reflection_order = int(self.reflection_order.value())
        settings.max_reflection_surfaces = int(self.reflection_surfaces.value())
        settings.max_reflection_paths = int(self.reflection_paths.value())
        settings.reflection_search_radius_m = float(self.reflection_radius.value())
        settings.minimum_reflection_coefficient = float(self.minimum_reflection.value())
        settings.enable_corner_diffraction = self.diffraction.isChecked()
        settings.max_diffraction_paths = int(self.diffraction_paths.value())
        settings.diffraction_search_radius_m = float(self.diffraction_radius.value())
        settings.minimum_diffraction_loss_db = float(self.diffraction_loss.value())
        settings.enable_small_scale_fading = self.fading.isChecked()
        settings.small_scale_fading_sigma_db = float(self.fading_sigma.value())
        settings.fading_correlation_distance_m = float(self.fading_distance.value())
        settings.fading_seed = int(self.fading_seed.value())
        settings.calculate_delay_spread = self.delay_spread.isChecked()
        settings.combined_ap_mode = str(self.ap_combination.currentData() or "strongest")
        settings.ignore_results_outside_planner_boundaries = self.boundary_filter.isChecked()


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
        delete_hint = "delete this user-created RF wall" if wall.is_user_created else "remove this imported IFC wall from the simulator model"
        self.setToolTip(f"{wall.label}\nRight-click to inspect, edit RF attenuation or {delete_hint}")

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
        bulk_action = menu.addAction("Bulk attenuation by IFC type…")
        delete_action = None
        if self.wall.is_user_created:
            menu.addSeparator()
            delete_action = menu.addAction("Delete user-created RF wall")
        else:
            menu.addSeparator()
            delete_action = menu.addAction("Remove imported IFC wall from this RF model")
        chosen = menu.exec(event.screenPos())
        if chosen == edit_action:
            self.main.edit_wall_rf_properties(self.wall)
        elif rotate_ifc_action is not None and chosen == rotate_ifc_action:
            self.main.rotate_ifc_to_align_wall_with_x_axis(self.wall)
        elif chosen == reset_action:
            self.main.reset_wall_rf_properties(self.wall)
        elif chosen == bulk_action:
            self.main.show_bulk_ifc_attenuation()
        elif delete_action is not None and chosen == delete_action:
            if self.wall.is_user_created:
                self.main.delete_user_wall(self.wall)
            else:
                self.main.delete_imported_wall(self.wall)
        event.accept()


class SpaceGraphicsItem(QGraphicsPolygonItem):
    """Imported or user-created RF/planning space."""

    def __init__(self, main, space: Space2D, polygon: QPolygonF, pen: QPen, brush: QBrush):
        super().__init__(polygon)
        self.main = main
        self.space = space
        self.setPen(pen)
        self.setBrush(brush)
        self.setZValue(Z_IFC_SPACE_FILL)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        source = "User-created simulator space" if space.is_user_created else "Imported IFC space"
        action = "delete this user-created space" if space.is_user_created else "remove this imported IFC space from the simulator model"
        selection = "Selected for AP placement" if space.ap_planning_selected else "Click to select for AP placement"
        self.setToolTip(f"{space.name}\n{source}\n{selection}\nRight-click to {action}.")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.main.toggle_space_ap_planning_selection(self.space)
            event.accept()
            return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        self.setSelected(True)
        toggle_action = menu.addAction(
            "Remove from AP placement spaces" if self.space.ap_planning_selected else "Select for AP placement"
        )
        menu.addSeparator()
        if self.space.is_user_created:
            delete_action = menu.addAction("Delete user-created space")
        else:
            delete_action = menu.addAction("Remove imported IFC space from this RF model")
        chosen = menu.exec(event.screenPos())
        if chosen == toggle_action:
            self.main.toggle_space_ap_planning_selection(self.space)
        elif chosen == delete_action:
            if self.space.is_user_created:
                self.main.delete_user_space(self.space)
            else:
                self.main.delete_imported_space(self.space)
        event.accept()


class InferredSpaceGraphicsItem(QGraphicsPolygonItem):
    """RF/planning space inferred from wall and boundary geometry."""

    def __init__(self, main, space: Space2D, polygon: QPolygonF, pen: QPen, brush: QBrush):
        super().__init__(polygon)
        self.main = main
        self.space = space
        self.setPen(pen)
        self.setBrush(brush)
        self.setZValue(Z_IFC_SPACE_OUTLINE + 1)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        note = space.assumption_note or "Created from wall boundaries and the external planning baseline."
        selection = "Selected for AP placement" if space.ap_planning_selected else "Click to select for AP placement"
        self.setToolTip(
            f"{space.name}\nInferred simulator space - source IFC is unchanged.\n{note}\n"
            f"{selection}\nRight-click to delete this inferred space."
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if getattr(self.main, "inferred_space_interaction_mode", False):
                # In the dedicated interaction mode, ordinary Qt selection is
                # more useful than toggling AP-planning membership. This makes
                # Ctrl/Shift multi-selection and rubber-band deletion reliable.
                super().mousePressEvent(event)
                self.main.statusBar().showMessage(
                    f"Selected inferred space '{self.space.name or self.space.guid}'. "
                    "Press Delete to remove selected inferred spaces or right-click for actions."
                )
                return
            self.main.toggle_space_ap_planning_selection(self.space)
            event.accept()
            return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        self.setSelected(True)
        toggle_action = menu.addAction(
            "Remove from AP placement spaces" if self.space.ap_planning_selected else "Select for AP placement"
        )
        menu.addSeparator()
        delete_action = menu.addAction("Delete inferred space")
        chosen = menu.exec(event.screenPos())
        if chosen == toggle_action:
            self.main.toggle_space_ap_planning_selection(self.space)
        elif chosen == delete_action:
            self.main.delete_inferred_space(self.space)
        event.accept()


class IFCElementGraphicsItem(QGraphicsPolygonItem):
    """Imported non-wall IFC element shown as model context."""

    def __init__(self, main, element: IFCElement2D, polygon: QPolygonF, pen: QPen, brush: QBrush):
        super().__init__(polygon)
        self.main = main
        self.element = element
        self.setPen(pen)
        self.setBrush(brush)
        self.setZValue(Z_IFC_WALL + 1 if element.is_rf_opening else Z_IFC_SPACE_OUTLINE)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        material = f"\nMaterial: {element.material}" if element.material else ""
        losses = ", ".join(
            f"{frequency:g} MHz: {loss:g} dB"
            for frequency, loss in sorted(element.attenuation_by_band_db.items())
        )
        if element.is_rf_opening:
            behaviour = "When crossed, this opening loss replaces its host wall loss."
        elif element.is_rf_barrier:
            behaviour = "When the 3D radio path crosses this element, its loss is added."
        else:
            behaviour = "This type currently has 0 dB loss and is visual context only."
        rf_note = f"\nRF category: {element.rf_category}\nAttenuation: {losses or '0 dB'}\n{behaviour}"
        self.setToolTip(
            f"{element.name or element.guid}\n{element.ifc_class} | {element.rf_type_override or element.type_name}{material}{rf_note}\n"
            "Right-click to edit attenuation, open the bulk type manager, or remove this IFC element."
        )

    def contextMenuEvent(self, event):
        menu = QMenu()
        self.setSelected(True)
        inferred_space_actions = []
        for space in self.main.inferred_spaces_at_scene_pos(event.scenePos()):
            label = f"Delete inferred space '{space.name}'" if space.name else "Delete inferred space"
            inferred_space_actions.append((menu.addAction(label), space))
        if inferred_space_actions:
            menu.addSeparator()
        edit_action = menu.addAction("Edit IFC type attenuation…")
        reset_action = menu.addAction("Reset attenuation from IFC type/material")
        bulk_action = menu.addAction("Bulk attenuation by IFC type…")
        menu.addSeparator()
        delete_action = menu.addAction("Remove imported IFC element from this RF model")
        chosen = menu.exec(event.screenPos())
        for action, space in inferred_space_actions:
            if chosen == action:
                self.main.delete_inferred_space(space)
                event.accept()
                return
        if chosen == edit_action:
            self.main.edit_ifc_element_rf_properties(self.element)
        elif chosen == reset_action:
            self.main.reset_ifc_element_rf_properties(self.element)
        elif chosen == bulk_action:
            self.main.show_bulk_ifc_attenuation()
        elif chosen == delete_action:
            self.main.delete_imported_element(self.element)
        event.accept()


class PlannerBoundaryGraphicsItem(QGraphicsPolygonItem):
    """Selectable outline for a planner boundary shared across all floors."""

    def __init__(self, main, boundary: PlannerBoundary2D):
        coords = list(boundary.polygon.exterior.coords)
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in coords])
        super().__init__(polygon)
        self.main = main
        self.boundary = boundary
        pen = QPen(QColor("#00A6D6"), 1.5)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.setZValue(Z_TEXT + 2)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        boundary_kind = "Polygon" if boundary.shape_type == "polygon" else "Rectangular"
        self.setToolTip(
            f"{boundary.name}\n{boundary_kind} boundary shared by all IFC floors\n"
            "Predictive AP candidates cannot be placed outside the combined planner boundaries.\n"
            "Right-click to delete."
        )

    def contextMenuEvent(self, event):
        menu = QMenu()
        self.setSelected(True)
        delete_action = menu.addAction("Delete planner boundary")
        chosen = menu.exec(event.screenPos())
        if chosen == delete_action:
            self.main.delete_planner_boundary(self.boundary)
        event.accept()


class APArrayPlacementDialog(QDialog):
    """Configure a rectangular AP array before the user clicks its origin."""

    def __init__(self, parent=None, initial: Optional[Dict[str, object]] = None):
        super().__init__(parent)
        self.setWindowTitle("Access point array placement")
        initial = dict(initial or {})
        form = QFormLayout(self)

        self.axial_count = QSpinBox()
        self.axial_count.setRange(1, 100)
        self.axial_count.setValue(int(initial.get("axial_count", 3)))
        self.transverse_count = QSpinBox()
        self.transverse_count.setRange(1, 100)
        self.transverse_count.setValue(int(initial.get("transverse_count", 2)))
        self.axial_spacing = QDoubleSpinBox()
        self.axial_spacing.setRange(0.1, 1000.0)
        self.axial_spacing.setDecimals(2)
        self.axial_spacing.setValue(float(initial.get("axial_spacing_m", 8.0)))
        self.axial_spacing.setSuffix(" m")
        self.transverse_spacing = QDoubleSpinBox()
        self.transverse_spacing.setRange(0.1, 1000.0)
        self.transverse_spacing.setDecimals(2)
        self.transverse_spacing.setValue(float(initial.get("transverse_spacing_m", 8.0)))
        self.transverse_spacing.setSuffix(" m")
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-180.0, 180.0)
        self.angle.setDecimals(1)
        self.angle.setValue(float(initial.get("angle_deg", 0.0)))
        self.angle.setSuffix("°")
        self.centered = QCheckBox("Treat the clicked point as the centre of the array")
        self.centered.setChecked(bool(initial.get("centered", False)))
        self.stagger = QCheckBox("Stagger alternate transverse rows by half the axial distance")
        self.stagger.setChecked(bool(initial.get("stagger", False)))

        form.addRow("APs along axial direction", self.axial_count)
        form.addRow("APs along transverse direction", self.transverse_count)
        form.addRow("Axial distance", self.axial_spacing)
        form.addRow("Transverse distance", self.transverse_spacing)
        form.addRow("Axial angle", self.angle)
        form.addRow(self.centered)
        form.addRow(self.stagger)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Dict[str, object]:
        return {
            "axial_count": int(self.axial_count.value()),
            "transverse_count": int(self.transverse_count.value()),
            "axial_spacing_m": float(self.axial_spacing.value()),
            "transverse_spacing_m": float(self.transverse_spacing.value()),
            "angle_deg": float(self.angle.value()),
            "centered": bool(self.centered.isChecked()),
            "stagger": bool(self.stagger.isChecked()),
        }


class APSpacePlacementDialog(QDialog):
    """Configure automatic AP placement inside IFC, inferred or user spaces."""

    def __init__(self, parent=None, has_selected_spaces: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Place access points from spaces")
        form = QFormLayout(self)
        self.scope = QComboBox()
        if has_selected_spaces:
            self.scope.addItem("Only spaces selected for AP planning", "selected")
        self.scope.addItem("All spaces on the current floor", "all")
        self.strategy = QComboBox()
        self.strategy.addItem("One AP at a safe point inside each space", "one")
        self.strategy.addItem("Fill each space using an axial/transverse grid", "grid")
        self.axial_spacing = QDoubleSpinBox()
        self.axial_spacing.setRange(0.5, 500.0)
        self.axial_spacing.setValue(8.0)
        self.axial_spacing.setSuffix(" m")
        self.transverse_spacing = QDoubleSpinBox()
        self.transverse_spacing.setRange(0.5, 500.0)
        self.transverse_spacing.setValue(8.0)
        self.transverse_spacing.setSuffix(" m")
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-180.0, 180.0)
        self.angle.setValue(0.0)
        self.angle.setSuffix("°")
        self.inset = QDoubleSpinBox()
        self.inset.setRange(0.0, 50.0)
        self.inset.setValue(0.5)
        self.inset.setSuffix(" m")
        self.minimum_spacing = QDoubleSpinBox()
        self.minimum_spacing.setRange(0.0, 100.0)
        self.minimum_spacing.setValue(1.0)
        self.minimum_spacing.setSuffix(" m")
        form.addRow("Spaces", self.scope)
        form.addRow("Placement method", self.strategy)
        form.addRow("Axial distance", self.axial_spacing)
        form.addRow("Transverse distance", self.transverse_spacing)
        form.addRow("Grid angle", self.angle)
        form.addRow("Keep away from space edge", self.inset)
        form.addRow("Minimum distance from another AP", self.minimum_spacing)
        hint = QLabel("Grid controls are ignored when one AP per space is selected. Points are always checked against the usable space polygon.")
        hint.setWordWrap(True)
        form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Dict[str, object]:
        return {
            "scope": str(self.scope.currentData()),
            "strategy": str(self.strategy.currentData()),
            "axial_spacing_m": float(self.axial_spacing.value()),
            "transverse_spacing_m": float(self.transverse_spacing.value()),
            "angle_deg": float(self.angle.value()),
            "inset_m": float(self.inset.value()),
            "minimum_spacing_m": float(self.minimum_spacing.value()),
        }


def _access_point_symbol_path(ap_type: str, radius: float) -> QPainterPath:
    definition = AP_TYPE_PRESETS.get(ap_type, AP_TYPE_PRESETS["Ceiling AP"])
    symbol = definition.get("symbol", "circle_cross")
    r = float(radius)
    path = QPainterPath()
    if symbol == "square":
        path.addRect(-r, -r, 2.0 * r, 2.0 * r)
        path.moveTo(-r, 0.0); path.lineTo(r, 0.0)
    elif symbol == "triangle":
        path.moveTo(r, 0.0); path.lineTo(-0.75 * r, 0.85 * r); path.lineTo(-0.75 * r, -0.85 * r); path.closeSubpath()
    elif symbol == "hexagon":
        points = [QPointF(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in range(0, 360, 60)]
        path.moveTo(points[0])
        for point in points[1:]: path.lineTo(point)
        path.closeSubpath()
    elif symbol == "octagon":
        points = [QPointF(r * math.cos(math.radians(22.5 + a)), r * math.sin(math.radians(22.5 + a))) for a in range(0, 360, 45)]
        path.moveTo(points[0])
        for point in points[1:]: path.lineTo(point)
        path.closeSubpath()
        path.moveTo(-0.45 * r, 0.0); path.lineTo(0.45 * r, 0.0)
    elif symbol == "diamond":
        path.moveTo(0.0, r); path.lineTo(r, 0.0); path.lineTo(0.0, -r); path.lineTo(-r, 0.0); path.closeSubpath()
        path.addEllipse(-0.18 * r, -0.18 * r, 0.36 * r, 0.36 * r)
    elif symbol == "double_circle":
        path.addEllipse(-r, -r, 2.0 * r, 2.0 * r)
        path.addEllipse(-0.5 * r, -0.5 * r, r, r)
    else:
        path.addEllipse(-r, -r, 2.0 * r, 2.0 * r)
        path.moveTo(-0.65 * r, 0.0); path.lineTo(0.65 * r, 0.0)
        path.moveTo(0.0, -0.65 * r); path.lineTo(0.0, 0.65 * r)
    return path


class AccessPointGraphicsItem(QGraphicsPathItem):
    def __init__(self, main, ap: AccessPoint, radius: float, colour: QColor):
        super().__init__(_access_point_symbol_path(ap.ap_type, radius))
        self.main = main
        self.ap = ap
        self.radius = radius
        self._drag_press_scene: Optional[QPointF] = None
        self._drag_items: List[Tuple[QGraphicsItem, QPointF]] = []
        self._dragged = False
        self._toggle_selection_on_click = False
        # Labels and AP cutoff circles are separate scene items.  Keep lightweight
        # followers with the AP so live dragging does not require rebuilding the
        # complete IFC scene on mouse release.
        self._move_followers: List[QGraphicsItem] = []
        self.setPos(float(ap.x), float(ap.y))
        self.setBrush(QBrush(colour))
        pen = QPen(main._theme_colours()["ap_outline"], 0.2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(Z_AP)
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setCursor(Qt.OpenHandCursor)
        self._refresh_tooltip()

    def _refresh_tooltip(self):
        radio_summary = ", ".join(
            f"{r.frequency_mhz:g} MHz ch {r.channel or 'auto'} / {r.channel_width_mhz:g} MHz"
            for r in self.ap.active_radios()
        )
        self.setToolTip(
            f"{self.ap.name}{' (predicted)' if self.ap.planned else ''}\n"
            f"Type: {self.ap.ap_type}\nRadio profile: {self.ap.radio_profile}\n"
            f"{radio_summary}\nClients/AP: {self.ap.max_clients}\n"
            "Select several APs with Shift/Ctrl or a selection window, then drag any selected AP to move the group. "
            "Hold Shift while dragging to constrain movement; right-click for actions."
        )

    def _selected_ap_items(self) -> List["AccessPointGraphicsItem"]:
        scene = self.scene()
        if scene is None:
            return [self]
        selected = [
            item for item in scene.selectedItems()
            if isinstance(item, AccessPointGraphicsItem)
        ]
        return selected or [self]

    def add_move_follower(self, item: QGraphicsItem):
        """Move an auxiliary scene item by the same delta as this AP symbol."""
        if item is not None and item not in self._move_followers:
            self._move_followers.append(item)

    def set_scene_position_with_followers(self, position: QPointF):
        current = QPointF(self.pos())
        delta = QPointF(position) - current
        self.setPos(position)
        if delta.manhattanLength() <= 1e-12:
            return
        for follower in list(self._move_followers):
            try:
                follower.setPos(follower.pos() + delta)
            except RuntimeError:
                # The scene may have been rebuilt while an old item reference was
                # pending.  Drop deleted Qt objects without interrupting dragging.
                try:
                    self._move_followers.remove(follower)
                except ValueError:
                    pass

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        scene = self.scene()
        modifiers = event.modifiers()
        control = bool(modifiers & Qt.ControlModifier)
        additive = control or bool(modifiers & Qt.ShiftModifier)
        was_selected = self.isSelected()

        # Preserve an existing multi-selection when a selected AP is grabbed.
        # Ctrl/Shift can add an AP to the group; Ctrl-click without dragging can
        # still remove an already selected AP on release.
        if not was_selected:
            if scene is not None and not additive:
                scene.clearSelection()
            self.setSelected(True)
        self._toggle_selection_on_click = bool(control and was_selected)

        selected_items = self._selected_ap_items()
        if self not in selected_items:
            selected_items.append(self)
        self._drag_items = [(item, QPointF(item.pos())) for item in selected_items]
        self._drag_press_scene = QPointF(event.scenePos())
        self._dragged = False
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_press_scene is None or not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return

        delta = event.scenePos() - self._drag_press_scene
        if bool(event.modifiers() & Qt.ShiftModifier):
            if abs(delta.x()) >= abs(delta.y()):
                delta.setY(0.0)
            else:
                delta.setX(0.0)

        for item, origin in self._drag_items:
            if isinstance(item, AccessPointGraphicsItem):
                item.set_scene_position_with_followers(origin + delta)
            else:
                item.setPos(origin + delta)
        if getattr(self.main, "ap_ruler_enabled", False):
            self.main.refresh_access_point_rulers()
        if delta.manhattanLength() > 1e-6:
            self._dragged = True
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._drag_press_scene is None:
            super().mouseReleaseEvent(event)
            return

        self.setCursor(Qt.OpenHandCursor)
        moved_aps: List[AccessPoint] = []
        total_delta = QPointF(0.0, 0.0)
        for item, origin in self._drag_items:
            if not isinstance(item, AccessPointGraphicsItem):
                continue
            scene_pos = item.scenePos()
            if (scene_pos - origin).manhattanLength() > 1e-9:
                moved_aps.append(item.ap)
                total_delta = scene_pos - origin
            item.ap.x = float(scene_pos.x())
            item.ap.y = float(scene_pos.y())

        if self._toggle_selection_on_click and not self._dragged:
            self.setSelected(False)

        self._drag_press_scene = None
        self._drag_items = []
        self._toggle_selection_on_click = False
        if moved_aps:
            self.main.commit_access_point_group_move(moved_aps, total_delta)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.main.focus_ap_in_table(self.ap)
        event.accept()

    def contextMenuEvent(self, event):
        # Right-clicking an unselected AP makes it the operation target. If it is
        # already part of a multi-selection, preserve the complete selection.
        if not self.isSelected():
            scene = self.scene()
            if scene is not None:
                scene.clearSelection()
            self.setSelected(True)

        menu = QMenu()
        focus_action = menu.addAction("Edit in access point table")
        duplicate_action = menu.addAction("Duplicate access point")
        copy_action = menu.addAction("Copy selected access point(s)    Ctrl+C")
        paste_action = menu.addAction("Paste access point(s)    Ctrl+V")
        type_menu = menu.addMenu("Change access point type")
        type_actions = {type_menu.addAction(name): name for name in AP_TYPE_PRESETS}
        profile_menu = menu.addMenu("Apply radio profile")
        profile_names = ["Project default radios"] + list(RADIO_PROFILE_PRESETS.keys())
        profile_actions = {profile_menu.addAction(name): name for name in profile_names}
        menu.addSeparator()
        delete_action = menu.addAction("Delete access point")
        chosen = menu.exec(event.screenPos())
        if chosen == focus_action:
            self.main.focus_ap_in_table(self.ap)
        elif chosen == duplicate_action:
            self.main.duplicate_access_point(self.ap)
        elif chosen == copy_action:
            self.main.copy_selected_access_points()
        elif chosen == paste_action:
            self.main.paste_access_points()
        elif chosen == delete_action:
            self.main.delete_access_point(self.ap)
        elif chosen in type_actions:
            self.ap.ap_type = type_actions[chosen]
            self.main._invalidate_interactive_preview_requests()
            self.main.last_result = None
            self.main.draw_floor()
            self.main.populate_ap_table()
        elif chosen in profile_actions:
            self.main.apply_radio_profile_to_ap(self.ap, profile_actions[chosen])
        event.accept()

# ----------------------------- GUI -----------------------------

class PlanView(QGraphicsView):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        # Left-drag is reserved for rubber-band selection so overlapping/stacked
        # inferred spaces and IFC items can be selected and removed with Delete.
        # Middle mouse still provides pan/hand dragging.
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)
        # Keep cosmetic pens and device-independent text crisp on high-DPI screens.
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, False)
        self.setMouseTracking(True)
        self.scale(1, -1)  # IFC Y-up style plan view
        self._middle_panning = False
        self._last_pan_pos = None
        self._press_pos = None
        self._rssi_hover_label = QLabel(self.viewport())
        self._rssi_hover_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._rssi_hover_label.setStyleSheet(
            "QLabel { background: rgba(20, 24, 30, 225); color: white; "
            "border: 1px solid rgba(255,255,255,120); border-radius: 4px; padding: 4px 7px; }"
        )
        self._rssi_hover_label.hide()

    def show_rssi_hover(self, text: str, viewport_pos):
        self._rssi_hover_label.setText(text)
        self._rssi_hover_label.adjustSize()
        margin = 8
        x = int(viewport_pos.x()) + 16
        y = int(viewport_pos.y()) + 16
        x = max(margin, min(x, self.viewport().width() - self._rssi_hover_label.width() - margin))
        y = max(margin, min(y, self.viewport().height() - self._rssi_hover_label.height() - margin))
        self._rssi_hover_label.move(x, y)
        self._rssi_hover_label.show()
        self._rssi_hover_label.raise_()

    def hide_rssi_hover(self):
        self._rssi_hover_label.hide()

    def leaveEvent(self, event):
        self.hide_rssi_hover()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        viewport_pos = event.position().toPoint()
        scene_hover_pos = self.mapToScene(viewport_pos)
        if self._middle_panning:
            self.hide_rssi_hover()
        else:
            self.main.update_rssi_hover_readout(scene_hover_pos, viewport_pos)
        shift_constrain = bool(event.modifiers() & Qt.ShiftModifier)
        if getattr(self.main, "space_draw_mode", False) and bool(getattr(self.main, "_space_polygon_points", [])):
            pos = self.mapToScene(event.position().toPoint())
            self.main.show_space_preview(pos, shift_constrain=shift_constrain)

        if getattr(self.main, "boundary_draw_mode", False):
            has_rectangle_start = getattr(self.main, "_boundary_draw_start", None) is not None
            has_polygon_points = bool(getattr(self.main, "_boundary_polygon_points", []))
            if has_rectangle_start or has_polygon_points:
                pos = self.mapToScene(event.position().toPoint())
                self.main.show_planner_boundary_preview(pos, shift_constrain=shift_constrain)

        if getattr(self.main, "wall_draw_mode", False) and getattr(self.main, "_wall_draw_start", None) is not None:
            pos = self.mapToScene(event.position().toPoint())
            start = self.main._wall_draw_start
            snap, snapped = self.main.nearest_ifc_connection_point(
                pos, straight_from=start if shift_constrain else None
            )
            if shift_constrain and not snapped:
                snap = self.main.axis_constrained_point(start, pos)
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
        self._press_pos = event.position().toPoint()
        if getattr(self.main, "space_draw_mode", False):
            if event.button() == Qt.RightButton:
                if len(getattr(self.main, "_space_polygon_points", [])) >= 3:
                    self.main.finish_user_space()
                else:
                    self.main.cancel_space_drawing()
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                shift_constrain = bool(event.modifiers() & Qt.ShiftModifier)
                self.main.capture_space_point(pos, shift_constrain=shift_constrain)
                event.accept()
                return

        if getattr(self.main, "boundary_draw_mode", False):
            if event.button() == Qt.RightButton:
                if (
                    getattr(self.main, "boundary_draw_shape", "rectangle") == "polygon"
                    and len(getattr(self.main, "_boundary_polygon_points", [])) >= 3
                ):
                    self.main.finish_planner_polygon_boundary()
                else:
                    self.main.cancel_planner_boundary_drawing()
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                shift_constrain = bool(event.modifiers() & Qt.ShiftModifier)
                self.main.capture_planner_boundary_point(pos, shift_constrain=shift_constrain)
                event.accept()
                return

        if getattr(self.main, "wall_draw_mode", False):
            if event.button() == Qt.RightButton:
                self.main.cancel_user_wall_drawing()
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                shift_constrain = bool(event.modifiers() & Qt.ShiftModifier)
                self.main.capture_user_wall_point(pos, shift_constrain=shift_constrain)
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

        if getattr(self.main, "ap_placement_mode", ""):
            if event.button() == Qt.RightButton:
                self.main.cancel_ap_placement()
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                self.main.handle_ap_placement_click(pos)
                event.accept()
                return

        if event.button() == Qt.LeftButton:
            selectable_items = self.main.selectable_scene_items_at_view_pos(self, event.position().toPoint())
            if len(selectable_items) > 1 and not (event.modifiers() & Qt.ShiftModifier):
                chosen = self.main.choose_scene_item_from_overlap(selectable_items, event.globalPosition().toPoint())
                if chosen is not None:
                    self.main.activate_scene_item_selection(chosen, toggle_ap_space=False)
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

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            if self.main.copy_selected_access_points():
                event.accept()
                return
        if event.matches(QKeySequence.StandardKey.Paste):
            if self.main.paste_access_points():
                event.accept()
                return
        if event.key() == Qt.Key_Escape and getattr(self.main, "ap_placement_mode", ""):
            self.main.cancel_ap_placement()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and getattr(self.main, "inferred_space_interaction_mode", False):
            action = getattr(self.main, "inferred_space_interaction_action", None)
            if action is not None:
                action.setChecked(False)
            else:
                self.main.toggle_inferred_space_interaction_mode(False)
            event.accept()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.main.delete_selected_scene_items():
                event.accept()
                return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.main.floor is None:
            return
        if (
            getattr(self.main, "alignment_pick_mode", None)
            or getattr(self.main, "wall_draw_mode", False)
            or getattr(self.main, "space_draw_mode", False)
            or getattr(self.main, "boundary_draw_mode", False)
            or getattr(self.main, "ap_interaction_mode", False)
            or getattr(self.main, "inferred_space_interaction_mode", False)
            or getattr(self.main, "ap_placement_mode", "")
        ):
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
        self._interactive_preview_running = False
        self._interactive_preview_timer = QTimer(self)
        self._interactive_preview_timer.setSingleShot(True)
        self._interactive_preview_timer.timeout.connect(self._run_interactive_rf_preview)
        # RF preview orchestration runs on a background thread.  The RF engine may
        # still use its process pool, but waiting for worker tiles never blocks the
        # Qt event loop.  Generation numbers coalesce rapid AP moves and prevent a
        # stale preview from replacing a newer layout.
        self._interactive_preview_thread: Optional[threading.Thread] = None
        self._interactive_preview_future: Optional[concurrent.futures.Future] = None
        self._interactive_preview_poll_timer = QTimer(self)
        self._interactive_preview_poll_timer.setInterval(50)
        self._interactive_preview_poll_timer.timeout.connect(self._poll_interactive_rf_preview)
        self._interactive_preview_generation = 0
        self._interactive_preview_job_generation = 0
        self._interactive_preview_job_floor = ""
        self._interactive_preview_pending = False
        self._interactive_preview_profile = "fast"
        self._interactive_preview_cold_start = False
        self._rssi_result_stale = False
        self.auto_planner_settings = AutoPlannerSettings.from_dict(self.heatmap_settings.auto_planner_settings)
        self.heatmap_settings_path: Optional[Path] = None
        self.ifc_origin_info: Dict[str, Dict[str, object]] = {}
        self.view_rotation_deg: float = 0.0
        self.ap_interaction_mode: bool = False
        self.inferred_space_interaction_mode: bool = False
        self.ap_placement_mode: str = ""
        self.ap_ruler_enabled: bool = False
        self._ap_ruler_items: List[QGraphicsItem] = []
        self._updating_ap_rulers: bool = False
        self._drawing_floor: bool = False
        self._ap_clipboard_signature: str = ""
        self._ap_paste_generation: int = 0
        self._pending_ap_selection_ids = set()
        self._array_placement_settings: Dict[str, object] = {
            "axial_count": 3, "transverse_count": 2,
            "axial_spacing_m": 8.0, "transverse_spacing_m": 8.0,
            "angle_deg": 0.0, "centered": False, "stagger": False,
        }
        self.wall_draw_mode: bool = False
        self._wall_draw_start: Optional[QPointF] = None
        self._wall_preview_items: List[QGraphicsItem] = []
        self.space_draw_mode: bool = False
        self._space_polygon_points: List[QPointF] = []
        self._space_preview_items: List[QGraphicsItem] = []
        self.excluded_ifc_elements: List[Dict[str, str]] = []
        self.planner_boundaries: List[PlannerBoundary2D] = []
        self.boundary_draw_mode: bool = False
        self.boundary_draw_shape: str = "rectangle"
        self._boundary_draw_start: Optional[QPointF] = None
        self._boundary_polygon_points: List[QPointF] = []
        self._boundary_preview_items: List[QGraphicsItem] = []
        self._suggested_boundary_preview_items: List[QGraphicsItem] = []
        self._suggested_space_preview_items: List[QGraphicsItem] = []
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
        self.view.scene().selectionChanged.connect(self._ap_scene_selection_changed)
        self.rssi_legend = QLabel()
        self.rssi_legend.setWordWrap(True)
        self.rssi_legend.setMinimumHeight(72)
        self.rssi_legend.setTextFormat(Qt.RichText)
        self._apply_theme_styles()
        self.floor_combo = QComboBox()
        self.wall_table = QTableWidget(0, 0)
        self._configure_wall_table_headers()
        self.wall_table.itemChanged.connect(self._wall_table_changed)

        self.ap_table = QTableWidget(0, 17)
        self.ap_table.setHorizontalHeaderLabels([
            "AP", "Type", "Radio", "Enabled", "Floor", "X", "Y", "Pattern", "Azimuth", "Downtilt",
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
        self.ap_type_combo = QComboBox()
        for ap_type, definition in AP_TYPE_PRESETS.items():
            self.ap_type_combo.addItem(ap_type)
            self.ap_type_combo.setItemData(self.ap_type_combo.count() - 1, definition.get("description", ""), Qt.ToolTipRole)
        self.ap_type_combo.currentTextChanged.connect(self._new_ap_type_changed)
        self.radio_profile_combo = QComboBox()
        self.radio_profile_combo.addItem("Project default radios")
        self.radio_profile_combo.addItems(list(RADIO_PROFILE_PRESETS.keys()))
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
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("Floor", self.floor_combo)
        form.addRow("RSSI view frequency", self.rssi_view_frequency)
        form.addRow("Grid resolution", self.resolution)
        form.addRow("AP TX power", self.tx_power)
        form.addRow("Frequency / Wi-Fi band", self.freq)
        form.addRow("New AP type", self.ap_type_combo)
        form.addRow("New AP radio profile", self.radio_profile_combo)
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
        instruction = QLabel("Use the Access point tools ribbon to enter AP-only interaction, toggle horizontal/vertical AP rulers, use single or array placement, or place automatically from spaces. Use Walls and boundaries > Inferred space interaction to isolate inferred spaces for selection and deletion. Window-select or Shift/Ctrl-select APs for group movement. Ctrl+C and Ctrl+V copy and paste selected AP layouts. Legacy double-click placement remains available outside dedicated interaction modes.")
        instruction.setWordWrap(True)
        form.addRow(instruction)

        for field in (
            self.floor_combo, self.rssi_view_frequency, self.ap_type_combo, self.radio_profile_combo, self.pattern_combo,
            self.resolution, self.tx_power, self.freq, self.azimuth,
            self.downtilt, self.mount_height, self.rx_height, self.ple,
            self.min_client_rssi, self.slab_att_24, self.slab_att_5, self.slab_att_6,
        ):
            field.setMinimumWidth(170)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setWidget(controls)

        ap_panel = QWidget()
        ap_panel_layout = QVBoxLayout(ap_panel)
        ap_panel_layout.setContentsMargins(4, 4, 4, 4)
        ap_help = QLabel("Configure radios, channels, antenna patterns and client capacity for each access point.")
        ap_help.setWordWrap(True)
        ap_panel_layout.addWidget(ap_help)
        ap_panel_layout.addWidget(self.ap_table, 1)

        wall_panel = QWidget()
        wall_panel_layout = QVBoxLayout(wall_panel)
        wall_panel_layout.setContentsMargins(4, 4, 4, 4)
        wall_help = QLabel("One row is shown per unique IFC attenuation type. Repeated wall and element instances are condensed and edits apply to every matching instance.")
        wall_help.setWordWrap(True)
        wall_panel_layout.addWidget(wall_help)
        wall_panel_layout.addWidget(self.wall_table, 1)

        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setDocumentMode(True)
        self.inspector_tabs.addTab(settings_scroll, "Settings")
        self.inspector_tabs.addTab(ap_panel, "Access points")
        self.inspector_tabs.addTab(wall_panel, "Attenuation types")

        side = QWidget()
        side.setMinimumWidth(390)
        side.setMaximumWidth(720)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 4, 4, 4)
        side_layout.addWidget(self.inspector_tabs, 1)

        self.ap_table.setAlternatingRowColors(True)
        self.wall_table.setAlternatingRowColors(True)
        self.ap_table.verticalHeader().setVisible(False)
        self.wall_table.verticalHeader().setVisible(False)

        model_panel = QWidget()
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(0)
        model_layout.addWidget(self.view, 1)
        model_layout.addWidget(self.rssi_legend, 0)

        split = QSplitter()
        split.setChildrenCollapsible(False)
        split.addWidget(model_panel)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 2)
        split.setSizes([1020, 430])
        self._main_splitter = split
        self._update_rssi_legend()

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
        self.export_pdf_action = QAction("Export floor PDF", self)
        self.export_pdf_action.triggered.connect(self.export_floor_pdf)
        self.clear_ap_action = QAction("Clear APs", self)
        self.clear_ap_action.triggered.connect(self.clear_aps)
        self.ap_interaction_action = QAction("AP interaction mode", self)
        self.ap_interaction_action.setCheckable(True)
        self.ap_interaction_action.toggled.connect(self.toggle_ap_interaction_mode)
        self.inferred_space_interaction_action = QAction("Inferred space interaction mode", self)
        self.inferred_space_interaction_action.setCheckable(True)
        self.inferred_space_interaction_action.toggled.connect(self.toggle_inferred_space_interaction_mode)
        self.ap_ruler_action = QAction("Access point ruler", self)
        self.ap_ruler_action.setCheckable(True)
        self.ap_ruler_action.toggled.connect(self.toggle_ap_ruler)
        self.place_ap_action = QAction("Place access points", self)
        self.place_ap_action.setCheckable(True)
        self.place_ap_action.toggled.connect(self.toggle_single_ap_placement)
        self.array_ap_action = QAction("Place AP array", self)
        self.array_ap_action.triggered.connect(self.start_ap_array_placement)
        self.space_ap_action = QAction("Place APs from spaces", self)
        self.space_ap_action.triggered.connect(self.show_space_ap_placement_dialog)
        self.bulk_ap_action = QAction("Bulk AP parameters", self)
        self.bulk_ap_action.triggered.connect(self.show_bulk_ap_parameters)
        self.cancel_ap_tool_action = QAction("Cancel AP tool", self)
        self.cancel_ap_tool_action.triggered.connect(self.cancel_ap_placement)
        self.copy_ap_action = QAction("Copy access points", self)
        self.copy_ap_action.triggered.connect(self.copy_selected_access_points)
        self.paste_ap_action = QAction("Paste access points", self)
        self.paste_ap_action.triggered.connect(self.paste_access_points)
        self.load_pattern_action = QAction("Load pattern CSV", self)
        self.load_pattern_action.triggered.connect(self.load_pattern_csv)
        self.load_heatmap_settings_action = QAction("Load heatmap settings", self)
        self.load_heatmap_settings_action.triggered.connect(self.load_heatmap_settings)
        self.propagation_settings_action = QAction("Propagation model", self)
        self.propagation_settings_action.triggered.connect(self.show_propagation_settings)
        self.performance_settings_action = QAction("Performance settings", self)
        self.performance_settings_action.triggered.connect(self.show_performance_settings)
        self.boundary_result_filter_action = QAction("Limit RSSI to boundaries", self)
        self.boundary_result_filter_action.setCheckable(True)
        self.boundary_result_filter_action.setChecked(
            bool(self.heatmap_settings.ignore_results_outside_planner_boundaries)
        )
        self.boundary_result_filter_action.toggled.connect(self.toggle_boundary_result_filter)
        self.planner_settings_action = QAction("Planner settings", self)
        self.planner_settings_action.triggered.connect(self.show_auto_planner_settings)
        self.predict_aps_action = QAction("Predict AP locations", self)
        self.predict_aps_action.triggered.connect(self.run_auto_planner)
        self.draw_wall_action = QAction("Draw RF wall", self)
        self.draw_wall_action.setCheckable(True)
        self.draw_wall_action.toggled.connect(self.toggle_wall_draw_mode)
        self.bulk_attenuation_action = QAction("Bulk IFC attenuation", self)
        self.bulk_attenuation_action.triggered.connect(self.show_bulk_ifc_attenuation)
        self.draw_boundary_action = QAction("Draw rectangular boundary", self)
        self.draw_boundary_action.setCheckable(True)
        self.draw_boundary_action.toggled.connect(
            lambda enabled: self.toggle_planner_boundary_draw_mode(enabled, "rectangle")
        )
        self.draw_polygon_boundary_action = QAction("Draw polygon boundary", self)
        self.draw_polygon_boundary_action.setCheckable(True)
        self.draw_polygon_boundary_action.toggled.connect(
            lambda enabled: self.toggle_planner_boundary_draw_mode(enabled, "polygon")
        )
        self.suggest_external_boundary_action = QAction("Suggest external boundary", self)
        self.suggest_external_boundary_action.triggered.connect(self.suggest_external_planner_boundary)
        self.draw_space_action = QAction("Draw space", self)
        self.draw_space_action.setCheckable(True)
        self.draw_space_action.toggled.connect(self.toggle_space_draw_mode)
        self.create_spaces_action = QAction("Create spaces from walls", self)
        self.create_spaces_action.triggered.connect(self.suggest_spaces_from_wall_boundaries)
        self.select_ap_spaces_action = QAction("Choose AP spaces", self)
        self.select_ap_spaces_action.triggered.connect(self.show_ap_space_selection_dialog)
        self.clear_inferred_spaces_action = QAction("Clear inferred spaces", self)
        self.clear_inferred_spaces_action.triggered.connect(self.clear_inferred_spaces)
        self.clear_boundaries_action = QAction("Clear planner boundaries", self)
        self.clear_boundaries_action.triggered.connect(self.clear_planner_boundaries)
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
        self._configure_ribbon_actions()
        for shortcut_action in (self.copy_ap_action, self.paste_ap_action):
            shortcut_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            self.view.addAction(shortcut_action)
        self.ribbon = self._build_ribbon()

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.ribbon, 0)
        central_layout.addWidget(split, 1)
        self.setCentralWidget(central)
        self._apply_theme_styles()

        self.floor_combo.currentTextChanged.connect(self.select_floor)
        self.include_inter_floor.stateChanged.connect(lambda *_: self.draw_floor())

    def selectable_scene_items_at_view_pos(self, view: QGraphicsView, view_pos) -> List[QGraphicsItem]:
        selectable_types = (
            AccessPointGraphicsItem,
            PlannerBoundaryGraphicsItem,
            IFCElementGraphicsItem,
            InferredSpaceGraphicsItem,
            SpaceGraphicsItem,
            WallGraphicsItem,
        )
        seen = set()
        selectable: List[QGraphicsItem] = []
        for item in view.items(view_pos):
            if self.ap_interaction_mode and not isinstance(item, AccessPointGraphicsItem):
                continue
            if self.inferred_space_interaction_mode and not isinstance(item, InferredSpaceGraphicsItem):
                continue
            if not isinstance(item, selectable_types):
                continue
            if not (item.flags() & QGraphicsItem.ItemIsSelectable):
                continue
            key = id(item)
            if key in seen:
                continue
            seen.add(key)
            selectable.append(item)
        return selectable

    def _scene_item_selection_label(self, item: QGraphicsItem) -> str:
        if isinstance(item, AccessPointGraphicsItem):
            return f"Access point: {item.ap.name}"
        if isinstance(item, PlannerBoundaryGraphicsItem):
            return f"Planner boundary: {item.boundary.name}"
        if isinstance(item, InferredSpaceGraphicsItem):
            return f"Inferred space: {item.space.name or item.space.guid}"
        if isinstance(item, SpaceGraphicsItem):
            source = "User space" if item.space.is_user_created else "IFC space"
            return f"{source}: {item.space.name or item.space.guid}"
        if isinstance(item, WallGraphicsItem):
            source = "User RF wall" if item.wall.is_user_created else "IFC wall"
            return f"{source}: {item.wall.label}"
        if isinstance(item, IFCElementGraphicsItem):
            name = item.element.name or item.element.guid
            return f"IFC element: {item.element.type_name} - {name}"
        return item.toolTip() or type(item).__name__

    def choose_scene_item_from_overlap(self, items: List[QGraphicsItem], screen_pos: QPointF) -> Optional[QGraphicsItem]:
        dialog = QDialog(self)
        dialog.setWindowTitle("Select item")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        label = QLabel("Multiple selectable items overlap here. Choose the item to select.")
        label.setWordWrap(True)
        layout.addWidget(label)

        list_widget = QListWidget(dialog)
        list_widget.setMinimumWidth(460)
        for item in items:
            list_widget.addItem(QListWidgetItem(self._scene_item_selection_label(item)))
        list_widget.setCurrentRow(0)
        list_widget.itemDoubleClicked.connect(lambda *_: dialog.accept())
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.move(screen_pos)

        if dialog.exec() != QDialog.Accepted:
            return None
        row = list_widget.currentRow()
        if row < 0 or row >= len(items):
            return None
        return items[row]

    def activate_scene_item_selection(self, item: QGraphicsItem, toggle_ap_space: bool = False):
        scene = self.view.scene() if getattr(self, "view", None) else None
        if scene is not None:
            scene.clearSelection()
        item.setSelected(True)
        if isinstance(item, (InferredSpaceGraphicsItem, SpaceGraphicsItem)):
            if toggle_ap_space:
                self.toggle_space_ap_planning_selection(item.space)
            else:
                source = "Inferred space" if isinstance(item, InferredSpaceGraphicsItem) else ("User space" if item.space.is_user_created else "IFC space")
                self.statusBar().showMessage(f"Selected {source.lower()} '{item.space.name or item.space.guid}'. Press Delete to remove it, or right-click for AP placement options.")
            return
        if isinstance(item, AccessPointGraphicsItem):
            self.statusBar().showMessage(f"Selected access point '{item.ap.name}'")
            return
        if isinstance(item, PlannerBoundaryGraphicsItem):
            self.statusBar().showMessage(f"Selected planner boundary '{item.boundary.name}'. Right-click for actions.")
            return
        if isinstance(item, WallGraphicsItem):
            self.statusBar().showMessage(f"Selected wall '{item.wall.label}'. Right-click for actions.")
            return
        if isinstance(item, IFCElementGraphicsItem):
            name = item.element.name or item.element.guid
            self.statusBar().showMessage(f"Selected IFC element '{name}'. Right-click for actions.")

    def delete_selected_scene_items(self) -> bool:
        scene = self.view.scene() if getattr(self, "view", None) else None
        if scene is None:
            return False
        selected_items = [
            item for item in scene.selectedItems()
            if isinstance(item, (
                AccessPointGraphicsItem,
                PlannerBoundaryGraphicsItem,
                IFCElementGraphicsItem,
                InferredSpaceGraphicsItem,
                SpaceGraphicsItem,
                WallGraphicsItem,
            ))
        ]
        if not selected_items:
            self.statusBar().showMessage("Select an AP, wall, space, IFC element or boundary before pressing Delete")
            return False

        # A rubber-band/selection-window can select several overlapping graphics
        # items. Delete all unique underlying model objects in a stable order,
        # rather than just the topmost item, so inferred spaces can be removed
        # even when IFC elements are drawn above them.
        deleted_labels: List[str] = []
        seen = set()
        for item in sorted(selected_items, key=lambda candidate: candidate.zValue(), reverse=True):
            if isinstance(item, AccessPointGraphicsItem):
                key = ("ap", id(item.ap))
                if key in seen:
                    continue
                seen.add(key)
                self.aps = [ap for ap in self.aps if ap is not item.ap]
                deleted_labels.append(f"access point '{item.ap.name}'")
                continue
            if isinstance(item, PlannerBoundaryGraphicsItem):
                key = ("boundary", id(item.boundary))
                if key in seen:
                    continue
                seen.add(key)
                self.planner_boundaries = [candidate for candidate in self.planner_boundaries if candidate is not item.boundary]
                deleted_labels.append(f"planner boundary '{item.boundary.name}'")
                continue
            if isinstance(item, InferredSpaceGraphicsItem):
                key = ("inferred_space", id(item.space))
                if key in seen:
                    continue
                seen.add(key)
                floor = self.floors.get(item.space.floor)
                if floor is not None and item.space.is_inferred:
                    floor.spaces = [candidate for candidate in floor.spaces if candidate is not item.space]
                    deleted_labels.append(f"inferred space '{item.space.name}'")
                continue
            if isinstance(item, SpaceGraphicsItem):
                key = ("space", id(item.space))
                if key in seen:
                    continue
                seen.add(key)
                floor = self.floors.get(item.space.floor)
                if floor is not None:
                    floor.spaces = [candidate for candidate in floor.spaces if candidate is not item.space]
                    if item.space.is_user_created:
                        deleted_labels.append(f"user space '{item.space.name}'")
                    else:
                        self._remember_ifc_exclusion("space", item.space.source_file, item.space.guid)
                        deleted_labels.append(f"IFC space '{item.space.name or item.space.guid}'")
                continue
            if isinstance(item, WallGraphicsItem):
                key = ("wall", id(item.wall))
                if key in seen:
                    continue
                seen.add(key)
                floor = self.floors.get(item.wall.floor)
                if floor is not None:
                    floor.walls = [candidate for candidate in floor.walls if candidate is not item.wall]
                    if item.wall.is_user_created:
                        deleted_labels.append(f"user wall '{item.wall.label}'")
                    else:
                        self._remember_ifc_exclusion("wall", item.wall.source_file, item.wall.guid)
                        deleted_labels.append(f"IFC wall '{item.wall.label}'")
                continue
            if isinstance(item, IFCElementGraphicsItem):
                key = ("element", id(item.element))
                if key in seen:
                    continue
                seen.add(key)
                floor = self.floors.get(item.element.floor)
                if floor is not None:
                    floor.elements = [candidate for candidate in floor.elements if candidate is not item.element]
                    self._remember_ifc_exclusion("element", item.element.source_file, item.element.guid)
                    deleted_labels.append(f"IFC element '{item.element.name or item.element.guid}'")

        if not deleted_labels:
            return False
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        self.populate_ap_table()
        self.populate_wall_table()
        if len(deleted_labels) == 1:
            self.statusBar().showMessage(f"Deleted {deleted_labels[0]}")
        else:
            self.statusBar().showMessage(f"Deleted {len(deleted_labels)} selected items")
        return True

    # ----------------------------- Ribbon interface -----------------------------

    def _standard_icon(self, standard_name: str) -> QIcon:
        standard = getattr(QStyle.StandardPixmap, standard_name, None)
        if standard is None:
            return QIcon()
        return self.style().standardIcon(standard)

    def _configure_ribbon_actions(self):
        metadata = {
            "open_action": (
                "Open IFC models", "Replace the current project with one or more IFC models.",
                "SP_DialogOpenButton", "Ctrl+O"
            ),
            "add_action": (
                "Add IFC models", "Append one or more IFC models to the current project.",
                "SP_FileDialogNewFolder", "Ctrl+Shift+O"
            ),
            "open_dxf_action": (
                "Open DXF overlay", "Load a DXF drawing as a visual alignment reference.",
                "SP_DirOpenIcon", ""
            ),
            "align_ifc_action": (
                "Align IFC to DXF", "Open the interactive IFC-to-DXF alignment dialog.",
                "SP_BrowserReload", ""
            ),
            "two_point_align_action": (
                "Two-point alignment", "Align IFC and DXF using two corresponding snapped points.",
                "SP_DialogApplyButton", ""
            ),
            "clear_dxf_action": (
                "Clear DXF", "Remove the current DXF overlay without changing the IFC model.",
                "SP_DialogDiscardButton", ""
            ),
            "sim_action": (
                "Simulate RSSI", "Calculate RF coverage for all applicable frequencies and redraw the selected result.",
                "SP_MediaPlay", "F5"
            ),
            "export_action": (
                "Export CSV", "Export the calculated RF sample results to CSV.",
                "SP_DialogSaveButton", "Ctrl+E"
            ),
            "export_pdf_action": (
                "Export floor PDF", "Calculate the selected RSSI frequency for every floor and export one scaled floor-plan heatmap page per floor, including the RSSI legend and propagation summary.",
                "SP_FileDialogContentsView", "Ctrl+Shift+E"
            ),
            "clear_ap_action": (
                "Clear access points", "Remove every access point from the current floor.",
                "SP_TrashIcon", ""
            ),
            "ap_interaction_action": (
                "AP interaction", "Make access points the only selectable/movable plan objects. Window-select or Shift/Ctrl-select several APs, then drag any selected AP to move the group; hold Shift to constrain movement.",
                "SP_ArrowCursor", "Ctrl+Alt+A"
            ),
            "inferred_space_interaction_action": (
                "Inferred space interaction", "Make inferred spaces the only selectable plan objects. Click or window-select one or more inferred spaces, press Delete to remove them, or right-click for AP-planning and deletion actions.",
                "SP_ArrowCursor", "Ctrl+Alt+I"
            ),
            "ap_ruler_action": (
                "AP ruler", "Toggle horizontal and vertical dimension rulers between access points. Selected APs are prioritised; with no selection a minimal connected ruler set is shown for the current floor.",
                "SP_FileDialogDetailedView", "Ctrl+Alt+R"
            ),
            "place_ap_action": (
                "Single placement", "Enter continuous single-click access point placement mode. Right-click or press the cancel tool to finish.",
                "SP_FileDialogNewFolder", "Ctrl+Alt+P"
            ),
            "array_ap_action": (
                "Array placement", "Place a configurable AP array using independent axial and transverse distances.",
                "SP_FileDialogListView", ""
            ),
            "space_ap_action": (
                "Place from spaces", "Automatically place one or more APs inside IFC, inferred or manually drawn spaces while remaining in the manual workflow.",
                "SP_DialogApplyButton", ""
            ),
            "bulk_ap_action": (
                "Bulk AP parameters", "Apply selected physical and radio parameters to selected APs, the current floor, or the complete project.",
                "SP_FileDialogDetailedView", "Ctrl+Alt+B"
            ),
            "cancel_ap_tool_action": (
                "Cancel AP tool", "Exit the active access point placement tool without deleting placed APs.",
                "SP_DialogCancelButton", "Escape"
            ),
            "copy_ap_action": (
                "Copy APs", "Copy every selected access point, including its type, radios, channels and relative position.",
                "SP_FileIcon", "Ctrl+C"
            ),
            "paste_ap_action": (
                "Paste APs", "Paste copied access points onto the current floor with a small cascade offset while preserving their layout.",
                "SP_DialogApplyButton", "Ctrl+V"
            ),
            "load_pattern_action": (
                "Load antenna pattern", "Import a directional antenna pattern from a CSV file.",
                "SP_FileIcon", ""
            ),
            "load_heatmap_settings_action": (
                "Load display settings", "Load RF heatmap, colour, IFC and rendering settings from JSON.",
                "SP_ComputerIcon", ""
            ),
            "propagation_settings_action": (
                "Propagation model", "Configure coherent reflections, higher-order rays, corner diffraction, small-scale fading, delay spread and multiple-AP power combination.",
                "SP_FileDialogDetailedView", "Ctrl+Alt+M"
            ),
            "performance_settings_action": (
                "Performance settings", "Configure RF and IFC worker counts, adaptive sampling, AP-field caches, raster rendering, previews and propagation-cost limits. Includes an i7-1255U/laptop preset.",
                "SP_ComputerIcon", "Ctrl+Alt+Shift+P"
            ),
            "boundary_result_filter_action": (
                "Limit RSSI to boundaries", "Skip and hide RSSI samples outside the union of accepted shared planner boundaries. The filter also applies to hover values, CSV statistics and floor PDF exports.",
                "SP_DialogApplyButton", "Ctrl+Alt+L"
            ),
            "planner_settings_action": (
                "Planner settings", "Configure coverage targets, radios, client capacity, channels and spectrum occupancy.",
                "SP_FileDialogDetailedView", ""
            ),
            "predict_aps_action": (
                "Predict AP locations", "Automatically place and configure access points within the permitted planning area.",
                "SP_DialogApplyButton", "Ctrl+P"
            ),
            "draw_wall_action": (
                "Draw RF wall", "Draw a connected RF-blocking wall where the IFC model has missing geometry. Hold Shift to constrain it horizontal or vertical.",
                "SP_FileDialogContentsView", ""
            ),
            "bulk_attenuation_action": (
                "Bulk IFC attenuation", "Edit attenuation once per unique IFC wall/element type and apply it to every matching instance.",
                "SP_FileDialogDetailedView", "Ctrl+Alt+T"
            ),
            "draw_boundary_action": (
                "Rectangle boundary", "Draw a rectangular hard boundary shared by all IFC floors.",
                "SP_TitleBarMaxButton", ""
            ),
            "draw_polygon_boundary_action": (
                "Polygon boundary", "Draw a polygon hard boundary shared by all IFC floors; hold Shift for horizontal/vertical segments and right-click to finish.",
                "SP_DriveNetIcon", ""
            ),
            "suggest_external_boundary_action": (
                "Suggest outer boundary", "Trace the outermost combined IFC and manually placed RF wall chain, preview it on the plan and accept it as a shared planner boundary.",
                "SP_FileDialogListView", ""
            ),
            "create_spaces_action": (
                "Create spaces from walls", "Infer room spaces from wall boundaries. Missing facade walls are closed using the accepted planner boundary or a suggested outer baseline.",
                "SP_FileDialogDetailedView", ""
            ),
            "draw_space_action": (
                "Draw space", "Create a simulator space on the selected floor by clicking polygon vertices; hold Shift for horizontal or vertical segments and right-click to finish.",
                "SP_FileDialogNewFolder", ""
            ),
            "select_ap_spaces_action": (
                "AP spaces", "Choose the spaces where predictive AP placement is allowed. Selected spaces are highlighted on the plan.",
                "SP_DialogApplyButton", ""
            ),
            "clear_inferred_spaces_action": (
                "Clear inferred spaces", "Remove inferred wall-derived spaces from the selected floor without changing IFC or manually drawn spaces.",
                "SP_TrashIcon", ""
            ),
            "clear_boundaries_action": (
                "Clear boundaries", "Remove every rectangular and polygon planner boundary.",
                "SP_TrashIcon", ""
            ),
            "ifc_origin_action": (
                "IFC origin and orientation", "Inspect insertion points, site coordinates, CRS, map conversion and True North.",
                "SP_MessageBoxInformation", ""
            ),
            "rotate_left_action": (
                "Rotate view left", "Rotate the display 15 degrees counter-clockwise without changing model coordinates.",
                "SP_ArrowBack", "Alt+Left"
            ),
            "rotate_right_action": (
                "Rotate view right", "Rotate the display 15 degrees clockwise without changing model coordinates.",
                "SP_ArrowForward", "Alt+Right"
            ),
            "reset_rotation_action": (
                "Reset view rotation", "Return the display to the unrotated IFC model orientation.",
                "SP_BrowserReload", "Alt+Home"
            ),
            "save_plan_action": (
                "Save RF plan", "Save access points, channels, wall overrides, boundaries and alignment settings.",
                "SP_DialogSaveButton", "Ctrl+S"
            ),
            "load_plan_action": (
                "Load RF plan", "Load a previously saved RF plan and restore its project settings.",
                "SP_DialogOpenButton", "Ctrl+L"
            ),
        }
        for attribute, (label, tooltip, icon_name, shortcut) in metadata.items():
            action = getattr(self, attribute)
            action.setText(label)
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
            action.setIcon(self._standard_icon(icon_name))
            if shortcut:
                action.setShortcut(shortcut)

    def _make_ribbon_button(self, action: QAction) -> QToolButton:
        button = QToolButton()
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setIconSize(QSize(28, 28))
        button.setAutoRaise(False)
        button.setMinimumSize(92, 68)
        button.setMaximumHeight(72)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.setFocusPolicy(Qt.NoFocus)
        return button

    def _make_ribbon_group(self, title: str, action_names: List[str]) -> QWidget:
        group = QWidget()
        group.setObjectName("RibbonGroup")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 2)
        group_layout.setSpacing(1)

        box = QFrame()
        box.setObjectName("RibbonGroupBox")
        box.setFrameShape(QFrame.StyledPanel)
        button_layout = QGridLayout(box)
        button_layout.setContentsMargins(4, 4, 4, 4)
        button_layout.setHorizontalSpacing(3)
        button_layout.setVerticalSpacing(0)
        for column, action_name in enumerate(action_names):
            action = getattr(self, action_name)
            button_layout.addWidget(self._make_ribbon_button(action), 0, column)
            button_layout.setColumnStretch(column, 1)
        group_layout.addWidget(box, 1)

        caption = QLabel(title)
        caption.setObjectName("RibbonGroupCaption")
        caption.setAlignment(Qt.AlignCenter)
        caption.setMinimumHeight(18)
        group_layout.addWidget(caption, 0)
        group.setMinimumWidth(max(112, len(action_names) * 96 + 8))
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return group

    def _make_ribbon_page(self, groups: List[Tuple[str, List[str]]]) -> QScrollArea:
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(4)
        for title, action_names in groups:
            group = self._make_ribbon_group(title, action_names)
            layout.addWidget(group, max(1, len(action_names)))
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("RibbonPage")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _build_ribbon(self) -> QTabWidget:
        ribbon = QTabWidget()
        ribbon.setObjectName("MainRibbon")
        ribbon.setDocumentMode(False)
        ribbon.setTabPosition(QTabWidget.North)
        ribbon.setMovable(False)
        ribbon.setMinimumHeight(128)
        ribbon.setMaximumHeight(134)

        ribbon.addTab(self._make_ribbon_page([
            ("Project", ["open_action", "add_action", "load_plan_action", "save_plan_action"]),
            ("Planning workflow", ["planner_settings_action", "predict_aps_action", "sim_action"]),
            ("Results and cleanup", ["export_action", "export_pdf_action", "clear_ap_action"]),
        ]), "Home")
        ribbon.addTab(self._make_ribbon_page([
            ("Interaction", ["ap_interaction_action", "ap_ruler_action", "copy_ap_action", "paste_ap_action", "bulk_ap_action", "cancel_ap_tool_action"]),
            ("Manual placement", ["place_ap_action", "array_ap_action"]),
            ("Space-assisted placement", ["space_ap_action", "select_ap_spaces_action"]),
        ]), "Access points")
        ribbon.addTab(self._make_ribbon_page([
            ("IFC information", ["ifc_origin_action"]),
            ("DXF and alignment", ["open_dxf_action", "align_ifc_action", "two_point_align_action", "clear_dxf_action"]),
            ("View orientation", ["rotate_left_action", "rotate_right_action", "reset_rotation_action"]),
        ]), "Model and view")
        ribbon.addTab(self._make_ribbon_page([
            ("RF obstructions", ["draw_wall_action", "bulk_attenuation_action"]),
            ("Space creation", ["draw_space_action", "create_spaces_action"]),
            ("Space interaction", ["inferred_space_interaction_action", "select_ap_spaces_action", "clear_inferred_spaces_action"]),
            ("Permitted planning area", ["draw_boundary_action", "draw_polygon_boundary_action", "suggest_external_boundary_action", "clear_boundaries_action"]),
        ]), "Walls and boundaries")
        ribbon.addTab(self._make_ribbon_page([
            ("Antenna data", ["load_pattern_action"]),
            ("Propagation and display", ["propagation_settings_action", "performance_settings_action", "boundary_result_filter_action", "load_heatmap_settings_action"]),
            ("Analysis", ["sim_action", "export_action", "export_pdf_action"]),
        ]), "Radio and analysis")
        return ribbon

    # ----------------------------- Access point interaction and placement -----------------------------

    def toggle_ap_ruler(self, enabled: bool):
        """Show or hide horizontal/vertical spacing dimensions between APs."""
        self.ap_ruler_enabled = bool(enabled)
        self.refresh_access_point_rulers()
        if enabled:
            self.statusBar().showMessage(
                "AP ruler enabled: select two or more APs to dimension that layout; select one AP for its nearest neighbour; clear selection to dimension the current floor."
            )
        else:
            self.statusBar().showMessage("AP ruler hidden")

    def _ap_scene_selection_changed(self):
        if not self.ap_ruler_enabled or self._drawing_floor or self._updating_ap_rulers:
            return
        self.refresh_access_point_rulers()

    def _clear_access_point_rulers(self, scene: Optional[QGraphicsScene] = None):
        scene = scene or (self.view.scene() if getattr(self, "view", None) else None)
        items = list(getattr(self, "_ap_ruler_items", []))
        self._ap_ruler_items = []
        if scene is None:
            return
        for item in items:
            try:
                if item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                # scene.clear() may already have deleted the wrapped C++ item.
                pass

    def _minimum_spanning_ap_pairs(
        self, aps: List[AccessPoint], positions: Dict[int, QPointF]
    ) -> List[Tuple[AccessPoint, AccessPoint]]:
        """Return a compact connected set of AP pairs rather than every O(n²) pair."""
        ordered = [ap for ap in aps if id(ap) in positions]
        if len(ordered) < 2:
            return []
        if len(ordered) == 2:
            return [(ordered[0], ordered[1])]

        visited = {0}
        pairs: List[Tuple[AccessPoint, AccessPoint]] = []
        while len(visited) < len(ordered):
            best = None
            for source_index in sorted(visited):
                source = positions[id(ordered[source_index])]
                for target_index, target_ap in enumerate(ordered):
                    if target_index in visited:
                        continue
                    target = positions[id(target_ap)]
                    dx = float(target.x() - source.x())
                    dy = float(target.y() - source.y())
                    candidate = (dx * dx + dy * dy, source_index, target_index)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, source_index, target_index = best
            pairs.append((ordered[source_index], ordered[target_index]))
            visited.add(target_index)
        return pairs

    def _add_ap_ruler_path(
        self, scene: QGraphicsScene, path: QPainterPath, colour: QColor, tooltip: str
    ) -> QGraphicsPathItem:
        item = QGraphicsPathItem(path)
        pen = QPen(colour, 1.25)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.NoBrush))
        item.setZValue(Z_AP_RULER)
        item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setToolTip(tooltip)
        scene.addItem(item)
        self._ap_ruler_items.append(item)
        return item

    def _draw_ap_ruler_pair(
        self, scene: QGraphicsScene, first: AccessPoint, second: AccessPoint,
        first_pos: QPointF, second_pos: QPointF, lane: int
    ):
        x1, y1 = float(first_pos.x()), float(first_pos.y())
        x2, y2 = float(second_pos.x()), float(second_pos.y())
        horizontal = abs(x2 - x1)
        vertical = abs(y2 - y1)
        if horizontal < 1e-9 and vertical < 1e-9:
            return

        lane_offset = 1.25 + 0.45 * float(lane % 5)
        tick = 0.22
        horizontal_colour = QColor("#00A6D6")
        vertical_colour = QColor("#F59E0B")
        tooltip = f"{first.name} to {second.name}: horizontal {horizontal:.2f} m, vertical {vertical:.2f} m"

        if horizontal >= 1e-9:
            y_dim = min(y1, y2) - lane_offset
            path = QPainterPath()
            path.moveTo(x1, y1)
            path.lineTo(x1, y_dim)
            path.moveTo(x2, y2)
            path.lineTo(x2, y_dim)
            path.moveTo(x1, y_dim)
            path.lineTo(x2, y_dim)
            path.moveTo(x1, y_dim - tick)
            path.lineTo(x1, y_dim + tick)
            path.moveTo(x2, y_dim - tick)
            path.lineTo(x2, y_dim + tick)
            self._add_ap_ruler_path(scene, path, horizontal_colour, tooltip)
            label = self._add_upright_text(
                scene, f"H {horizontal:.2f} m", (x1 + x2) * 0.5, y_dim - 0.38,
                horizontal_colour, max(2, int(self.heatmap_settings.ap_label_font_size)),
                Z_AP_RULER_LABEL, bold=True,
            )
            label.setToolTip(tooltip)
            self._ap_ruler_items.append(label)

        if vertical >= 1e-9:
            x_dim = max(x1, x2) + lane_offset
            path = QPainterPath()
            path.moveTo(x1, y1)
            path.lineTo(x_dim, y1)
            path.moveTo(x2, y2)
            path.lineTo(x_dim, y2)
            path.moveTo(x_dim, y1)
            path.lineTo(x_dim, y2)
            path.moveTo(x_dim - tick, y1)
            path.lineTo(x_dim + tick, y1)
            path.moveTo(x_dim - tick, y2)
            path.lineTo(x_dim + tick, y2)
            self._add_ap_ruler_path(scene, path, vertical_colour, tooltip)
            label = self._add_upright_text(
                scene, f"V {vertical:.2f} m", x_dim + 0.48, (y1 + y2) * 0.5,
                vertical_colour, max(2, int(self.heatmap_settings.ap_label_font_size)),
                Z_AP_RULER_LABEL, bold=True,
            )
            label.setToolTip(tooltip)
            self._ap_ruler_items.append(label)

    def refresh_access_point_rulers(self):
        """Rebuild the AP ruler overlay using current graphics-item positions."""
        if self._drawing_floor or self._updating_ap_rulers:
            return
        scene = self.view.scene() if getattr(self, "view", None) else None
        if scene is None:
            return

        self._updating_ap_rulers = True
        try:
            self._clear_access_point_rulers(scene)
            if not self.ap_ruler_enabled or not self.floor:
                return

            ap_items = {
                id(item.ap): item
                for item in scene.items()
                if isinstance(item, AccessPointGraphicsItem) and item.ap.floor == self.floor.name
            }
            floor_aps = [ap for ap in self.aps if ap.floor == self.floor.name and id(ap) in ap_items]
            if len(floor_aps) < 2:
                return

            selected = [
                ap for ap in floor_aps
                if ap_items[id(ap)].isSelected()
            ]
            positions = {
                id(ap): QPointF(ap_items[id(ap)].scenePos())
                for ap in floor_aps
            }

            if len(selected) >= 2:
                ruler_aps = selected
            elif len(selected) == 1:
                anchor = selected[0]
                anchor_pos = positions[id(anchor)]
                neighbours = [ap for ap in floor_aps if ap is not anchor]
                nearest = min(
                    neighbours,
                    key=lambda ap: (
                        (positions[id(ap)].x() - anchor_pos.x()) ** 2
                        + (positions[id(ap)].y() - anchor_pos.y()) ** 2,
                        ap.name,
                    ),
                )
                ruler_aps = [anchor, nearest]
            else:
                ruler_aps = floor_aps

            for lane, (first, second) in enumerate(self._minimum_spanning_ap_pairs(ruler_aps, positions)):
                self._draw_ap_ruler_pair(
                    scene, first, second, positions[id(first)], positions[id(second)], lane
                )
        finally:
            self._updating_ap_rulers = False

    def _new_ap_type_changed(self, ap_type: str):
        definition = AP_TYPE_PRESETS.get(str(ap_type), {})
        pattern = str(definition.get("pattern", ""))
        if pattern and pattern in self.antenna_patterns:
            self.pattern_combo.setCurrentText(pattern)

    def _set_interaction_action_checked(self, action_name: str, checked: bool):
        action = getattr(self, action_name, None)
        if action is None:
            return
        action.blockSignals(True)
        action.setChecked(bool(checked))
        action.blockSignals(False)

    def _cancel_other_drawing_tools_for_ap(self):
        if getattr(self, "wall_draw_mode", False):
            self.cancel_user_wall_drawing()
        if getattr(self, "space_draw_mode", False):
            self.cancel_space_drawing(show_status=False)
        if getattr(self, "boundary_draw_mode", False):
            self.cancel_planner_boundary_drawing(show_status=False)
        if self.inferred_space_interaction_mode:
            self.inferred_space_interaction_mode = False
            self._set_interaction_action_checked("inferred_space_interaction_action", False)

    def _disable_ap_tools_for_geometry_mode(self):
        if self.ap_placement_mode:
            self.cancel_ap_placement(show_status=False)
        changed = False
        if self.ap_interaction_mode:
            self.ap_interaction_mode = False
            self._set_interaction_action_checked("ap_interaction_action", False)
            changed = True
        if self.inferred_space_interaction_mode:
            self.inferred_space_interaction_mode = False
            self._set_interaction_action_checked("inferred_space_interaction_action", False)
            changed = True
        if changed:
            self._preserve_view_on_redraw = True
            self.draw_floor()

    def toggle_ap_interaction_mode(self, enabled: bool):
        self.ap_interaction_mode = bool(enabled)
        if enabled:
            self._cancel_other_drawing_tools_for_ap()
            self.statusBar().showMessage(
                "AP interaction mode: only access points can be selected. Drag to move; hold Shift for axial movement; right-click for AP actions."
            )
        else:
            if self.ap_placement_mode:
                self.cancel_ap_placement(show_status=False)
            self.statusBar().showMessage("Normal plan interaction restored")
        self._preserve_view_on_redraw = True
        self.draw_floor()

    def toggle_inferred_space_interaction_mode(self, enabled: bool):
        enabled = bool(enabled)
        if enabled and not self.floor:
            QMessageBox.information(
                self,
                "No floor selected",
                "Load an IFC and select a floor before entering inferred-space interaction mode.",
            )
            self._set_interaction_action_checked("inferred_space_interaction_action", False)
            return

        if enabled:
            if self.ap_placement_mode:
                self.cancel_ap_placement(show_status=False)
            if self.ap_interaction_mode:
                self.ap_interaction_mode = False
                self._set_interaction_action_checked("ap_interaction_action", False)
            if getattr(self, "wall_draw_mode", False):
                self.cancel_user_wall_drawing()
            if getattr(self, "space_draw_mode", False):
                self.cancel_space_drawing(show_status=False)
            if getattr(self, "boundary_draw_mode", False):
                self.cancel_planner_boundary_drawing(show_status=False)
            self.inferred_space_interaction_mode = True
            self.statusBar().showMessage(
                "Inferred-space interaction mode: only inferred spaces can be selected. "
                "Click or window-select, use Ctrl/Shift for multiple selection, press Delete to remove, or right-click for actions."
            )
        else:
            self.inferred_space_interaction_mode = False
            self.statusBar().showMessage("Normal plan interaction restored")
        self._preserve_view_on_redraw = True
        self.draw_floor()

    def _ensure_ap_interaction_enabled(self):
        if self.ap_interaction_mode:
            return
        self.ap_interaction_mode = True
        action = getattr(self, "ap_interaction_action", None)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self._cancel_other_drawing_tools_for_ap()
        self._preserve_view_on_redraw = True
        self.draw_floor()

    def toggle_single_ap_placement(self, enabled: bool):
        if enabled:
            if not self.floor:
                QMessageBox.information(self, "No floor selected", "Load a model and select a floor before placing access points.")
                self.place_ap_action.blockSignals(True)
                self.place_ap_action.setChecked(False)
                self.place_ap_action.blockSignals(False)
                return
            self._ensure_ap_interaction_enabled()
            self.ap_placement_mode = "single"
            self.view.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage("Single AP placement: left-click to place repeatedly; right-click or Cancel AP tool to finish.")
        elif self.ap_placement_mode == "single":
            self.cancel_ap_placement(show_status=False)

    def cancel_ap_placement(self, show_status: bool = True):
        self.ap_placement_mode = ""
        action = getattr(self, "place_ap_action", None)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)
        self.view.setCursor(Qt.ArrowCursor)
        if show_status:
            self.statusBar().showMessage("Access point placement tool cancelled")

    def start_ap_array_placement(self):
        if not self.floor:
            QMessageBox.information(self, "No floor selected", "Load a model and select a floor before placing an AP array.")
            return
        dialog = APArrayPlacementDialog(self, self._array_placement_settings)
        if dialog.exec() != QDialog.Accepted:
            return
        self._array_placement_settings = dialog.values()
        self.cancel_ap_placement(show_status=False)
        self._ensure_ap_interaction_enabled()
        self.ap_placement_mode = "array"
        self.view.setCursor(Qt.CrossCursor)
        values = self._array_placement_settings
        self.statusBar().showMessage(
            f"AP array ready: {values['axial_count']} × {values['transverse_count']}, "
            f"{values['axial_spacing_m']:g} m axial / {values['transverse_spacing_m']:g} m transverse. Click the origin; right-click to cancel."
        )

    def handle_ap_placement_click(self, pos: QPointF):
        if self.ap_placement_mode == "single":
            ap = self.add_ap(float(pos.x()), float(pos.y()))
            if ap is not None:
                self.statusBar().showMessage(
                    f"Placed {ap.name} at ({ap.x:.2f}, {ap.y:.2f}). Continue clicking or right-click to finish."
                )
            return
        if self.ap_placement_mode == "array":
            self.place_ap_array(float(pos.x()), float(pos.y()), self._array_placement_settings)
            self.cancel_ap_placement(show_status=False)

    def place_ap_array(self, origin_x: float, origin_y: float, settings: Dict[str, object]):
        axial_count = max(1, int(settings.get("axial_count", 1)))
        transverse_count = max(1, int(settings.get("transverse_count", 1)))
        axial_spacing = max(0.01, float(settings.get("axial_spacing_m", 8.0)))
        transverse_spacing = max(0.01, float(settings.get("transverse_spacing_m", 8.0)))
        angle = math.radians(float(settings.get("angle_deg", 0.0)))
        axial = (math.cos(angle), math.sin(angle))
        transverse = (-math.sin(angle), math.cos(angle))
        centered = bool(settings.get("centered", False))
        stagger = bool(settings.get("stagger", False))
        axial_origin = -0.5 * (axial_count - 1) * axial_spacing if centered else 0.0
        transverse_origin = -0.5 * (transverse_count - 1) * transverse_spacing if centered else 0.0
        total = axial_count * transverse_count
        if total > 10_000:
            QMessageBox.warning(self, "AP array too large", "An AP array is limited to 10,000 access points per operation.")
            return
        created: List[AccessPoint] = []
        for transverse_index in range(transverse_count):
            stagger_offset = 0.5 * axial_spacing if stagger and transverse_index % 2 else 0.0
            for axial_index in range(axial_count):
                along = axial_origin + axial_index * axial_spacing + stagger_offset
                across = transverse_origin + transverse_index * transverse_spacing
                x = origin_x + axial[0] * along + transverse[0] * across
                y = origin_y + axial[1] * along + transverse[1] * across
                ap = self.add_ap(x, y, redraw=False)
                if ap is not None:
                    created.append(ap)
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.draw_floor()
        self.populate_ap_table()
        self.statusBar().showMessage(f"Placed {len(created)} access points in an axial/transverse array")

    def show_space_ap_placement_dialog(self):
        if not self.floor:
            QMessageBox.information(self, "No floor selected", "Load a model and select a floor before placing access points from spaces.")
            return
        if not self.floor.spaces:
            QMessageBox.information(
                self, "No spaces available",
                "This floor has no IFC, inferred or manually drawn spaces. Create or draw spaces first."
            )
            return
        selected = [space for space in self.floor.spaces if space.ap_planning_selected]
        self.cancel_ap_placement(show_status=False)
        dialog = APSpacePlacementDialog(self, bool(selected))
        if dialog.exec() != QDialog.Accepted:
            return
        self.place_aps_from_spaces(dialog.values())

    @staticmethod
    def _point_far_enough(x: float, y: float, points: List[Tuple[float, float]], minimum_spacing: float) -> bool:
        if minimum_spacing <= 0.0:
            return True
        minimum_sq = minimum_spacing * minimum_spacing
        return all((x - px) ** 2 + (y - py) ** 2 >= minimum_sq for px, py in points)

    def _grid_points_inside_space(
        self, polygon: Polygon, axial_spacing: float, transverse_spacing: float, angle_deg: float
    ) -> List[Tuple[float, float]]:
        centre = polygon.representative_point()
        rotated = shapely_rotate(polygon, -float(angle_deg), origin=(float(centre.x), float(centre.y)))
        minx, miny, maxx, maxy = rotated.bounds
        points: List[Tuple[float, float]] = []
        maximum_points = 10_000
        x = minx + 0.5 * axial_spacing
        while x <= maxx + 1e-9:
            y = miny + 0.5 * transverse_spacing
            while y <= maxy + 1e-9:
                candidate = Point(float(x), float(y))
                if rotated.covers(candidate):
                    restored = shapely_rotate(candidate, float(angle_deg), origin=(float(centre.x), float(centre.y)))
                    points.append((float(restored.x), float(restored.y)))
                    if len(points) >= maximum_points:
                        return points
                y += transverse_spacing
            x += axial_spacing
        return points

    def place_aps_from_spaces(self, settings: Dict[str, object]):
        if not self.floor:
            return
        scope = str(settings.get("scope", "all"))
        spaces = [space for space in self.floor.spaces if scope != "selected" or space.ap_planning_selected]
        if not spaces and scope == "selected":
            QMessageBox.information(self, "No selected spaces", "No spaces are selected for AP planning on this floor.")
            return
        strategy = str(settings.get("strategy", "one"))
        axial_spacing = max(0.1, float(settings.get("axial_spacing_m", 8.0)))
        transverse_spacing = max(0.1, float(settings.get("transverse_spacing_m", 8.0)))
        angle_deg = float(settings.get("angle_deg", 0.0))
        inset = max(0.0, float(settings.get("inset_m", 0.0)))
        minimum_spacing = max(0.0, float(settings.get("minimum_spacing_m", 0.0)))
        occupied = [(float(ap.x), float(ap.y)) for ap in self.aps if ap.floor == self.floor.name]
        spacing_grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        spacing_cell = max(minimum_spacing, 0.001)
        if minimum_spacing > 0.0:
            for px, py in occupied:
                spacing_grid.setdefault((math.floor(px / spacing_cell), math.floor(py / spacing_cell)), []).append((px, py))

        def point_available(x: float, y: float) -> bool:
            if minimum_spacing <= 0.0:
                return True
            cell_x = math.floor(x / spacing_cell)
            cell_y = math.floor(y / spacing_cell)
            minimum_sq = minimum_spacing * minimum_spacing
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    for px, py in spacing_grid.get((cell_x + offset_x, cell_y + offset_y), []):
                        if (x - px) ** 2 + (y - py) ** 2 < minimum_sq:
                            return False
            return True

        def remember_point(x: float, y: float):
            occupied.append((x, y))
            if minimum_spacing > 0.0:
                spacing_grid.setdefault((math.floor(x / spacing_cell), math.floor(y / spacing_cell)), []).append((x, y))

        created: List[AccessPoint] = []
        placement_limit = 10_000
        skipped_spaces = 0
        successful_spaces = 0
        processed_spaces = 0
        limit_reached = False
        for space in spaces:
            processed_spaces += 1
            polygon = space.polygon
            if inset > 0.0:
                inset_polygon = polygon.buffer(-inset)
                if not inset_polygon.is_empty:
                    if inset_polygon.geom_type == "Polygon":
                        polygon = inset_polygon
                    else:
                        parts = [part for part in getattr(inset_polygon, "geoms", []) if part.geom_type == "Polygon"]
                        if parts:
                            polygon = max(parts, key=lambda part: float(part.area))
            if polygon.is_empty:
                skipped_spaces += 1
                continue
            if strategy == "grid":
                candidates = self._grid_points_inside_space(polygon, axial_spacing, transverse_spacing, angle_deg)
                if not candidates:
                    representative = polygon.representative_point()
                    candidates = [(float(representative.x), float(representative.y))]
            else:
                representative = polygon.representative_point()
                candidates = [(float(representative.x), float(representative.y))]
            accepted_in_space = 0
            for x, y in candidates:
                if not polygon.covers(Point(x, y)):
                    continue
                if not point_available(x, y):
                    continue
                ap = self.add_ap(x, y, redraw=False)
                if ap is not None:
                    created.append(ap)
                    remember_point(x, y)
                    accepted_in_space += 1
                    if len(created) >= placement_limit:
                        limit_reached = True
                        break
            if accepted_in_space == 0:
                skipped_spaces += 1
            else:
                successful_spaces += 1
            if limit_reached:
                break
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self._ensure_ap_interaction_enabled()
        self.draw_floor()
        self.populate_ap_table()
        QMessageBox.information(
            self, "Space-assisted AP placement",
            f"Placed {len(created)} access point(s) inside {successful_spaces} space(s)."
            + (f"\n{skipped_spaces} processed space(s) produced no new point because of geometry or minimum-spacing constraints." if skipped_spaces else "")
            + (f"\nPlacement stopped at the safety limit of {placement_limit:,} APs after processing {processed_spaces} of {len(spaces)} spaces." if limit_reached else "")
        )

    def selected_access_points(self) -> List[AccessPoint]:
        scene = self.view.scene() if getattr(self, "view", None) else None
        selected_ids = set()
        if scene is not None:
            selected_ids = {
                id(item.ap) for item in scene.selectedItems()
                if isinstance(item, AccessPointGraphicsItem)
            }
        return [ap for ap in self.aps if id(ap) in selected_ids]

    def _access_point_clipboard_record(self, ap: AccessPoint) -> Dict[str, object]:
        return {
            "name": ap.name,
            "x": float(ap.x),
            "y": float(ap.y),
            "floor": ap.floor,
            "tx_power_dbm": float(ap.tx_power_dbm),
            "frequency_mhz": float(ap.frequency_mhz),
            "reference_loss_db_at_1m": float(ap.reference_loss_db_at_1m),
            "path_loss_exponent": float(ap.path_loss_exponent),
            "antenna_pattern": ap.antenna_pattern,
            "azimuth_deg": float(ap.azimuth_deg),
            "downtilt_deg": float(ap.downtilt_deg),
            "mount_height_m": float(ap.mount_height_m),
            "rx_height_m": float(ap.rx_height_m),
            "ap_type": ap.ap_type,
            "radio_profile": ap.radio_profile,
            "max_clients": int(ap.max_clients),
            "planned": bool(ap.planned),
            "radios": [self._radio_to_dict(radio) for radio in ap.radios],
        }

    def copy_selected_access_points(self) -> bool:
        selected = self.selected_access_points()
        if not selected:
            self.statusBar().showMessage("Select one or more access points before copying")
            return False
        payload = {
            "format": "rf-access-point-clipboard",
            "version": 1,
            "access_points": [self._access_point_clipboard_record(ap) for ap in selected],
        }
        text = json.dumps(payload, separators=(",", ":"))
        QApplication.clipboard().setText(text)
        self._ap_clipboard_signature = text
        self._ap_paste_generation = 0
        self.statusBar().showMessage(f"Copied {len(selected)} access point(s). Press Ctrl+V to paste.")
        return True

    def paste_access_points(self) -> bool:
        if not self.floor:
            self.statusBar().showMessage("Select a floor before pasting access points")
            return False
        text = QApplication.clipboard().text().strip()
        if not text:
            self.statusBar().showMessage("The clipboard is empty")
            return False
        try:
            payload = json.loads(text)
        except Exception:
            self.statusBar().showMessage("The clipboard does not contain copied RF access points")
            return False
        if not isinstance(payload, dict) or payload.get("format") != "rf-access-point-clipboard":
            self.statusBar().showMessage("The clipboard does not contain copied RF access points")
            return False
        records = payload.get("access_points", [])
        if not isinstance(records, list) or not records:
            self.statusBar().showMessage("The copied access point set is empty")
            return False

        if text != self._ap_clipboard_signature:
            self._ap_clipboard_signature = text
            self._ap_paste_generation = 0
        self._ap_paste_generation += 1
        cascade_offset = float(self._ap_paste_generation)

        created: List[AccessPoint] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                radios = [
                    self._radio_from_dict(value)
                    for value in record.get("radios", [])
                    if isinstance(value, dict)
                ]
                ap_type = str(record.get("ap_type", "Ceiling AP"))
                if ap_type not in AP_TYPE_PRESETS:
                    ap_type = "Ceiling AP"
                pasted = AccessPoint(
                    name=self._next_ap_name(),
                    x=float(record.get("x", 0.0)) + cascade_offset,
                    y=float(record.get("y", 0.0)) + cascade_offset,
                    floor=self.floor.name,
                    tx_power_dbm=float(record.get("tx_power_dbm", 20.0)),
                    frequency_mhz=float(record.get("frequency_mhz", 2400.0)),
                    reference_loss_db_at_1m=float(record.get("reference_loss_db_at_1m", 40.0)),
                    path_loss_exponent=float(record.get("path_loss_exponent", 2.2)),
                    antenna_pattern=str(record.get("antenna_pattern", "Omni ceiling AP")),
                    azimuth_deg=float(record.get("azimuth_deg", 0.0)),
                    downtilt_deg=float(record.get("downtilt_deg", 0.0)),
                    mount_height_m=float(record.get("mount_height_m", 2.7)),
                    rx_height_m=float(record.get("rx_height_m", 1.2)),
                    ap_type=ap_type,
                    radio_profile=str(record.get("radio_profile", "Project default radios")),
                    radios=radios,
                    max_clients=max(1, int(record.get("max_clients", 50))),
                    planned=False,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            self.aps.append(pasted)
            created.append(pasted)

        if not created:
            self.statusBar().showMessage("No valid access points could be pasted")
            return False
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self._rssi_result_stale = False
        self._refresh_rssi_frequency_dropdown()
        self._pending_ap_selection_ids = {id(ap) for ap in created}
        self._preserve_view_on_redraw = True
        self.draw_floor()
        self.populate_ap_table()
        self.statusBar().showMessage(
            f"Pasted {len(created)} access point(s) onto {self.floor.name}; the pasted group remains selected."
        )
        return True

    def _update_ap_table_positions(self, moved_aps: List[AccessPoint]):
        """Update only X/Y cells after a drag instead of rebuilding every AP row."""
        if not moved_aps or not hasattr(self, "ap_table"):
            return
        by_name = {ap.name: ap for ap in moved_aps}
        self.ap_table.blockSignals(True)
        try:
            for row in range(self.ap_table.rowCount()):
                identity_item = self.ap_table.item(row, 0)
                if identity_item is None:
                    continue
                ap = by_name.get(str(identity_item.data(Qt.UserRole) or identity_item.text()))
                if ap is None:
                    continue
                x_item = self.ap_table.item(row, 5)
                y_item = self.ap_table.item(row, 6)
                if x_item is not None:
                    x_item.setText(f"{float(ap.x):.2f}")
                if y_item is not None:
                    y_item.setText(f"{float(ap.y):.2f}")
        finally:
            self.ap_table.blockSignals(False)

    @staticmethod
    def _snapshot_rf_access_points(aps: List[AccessPoint]) -> List[AccessPoint]:
        """Create an RF-only AP snapshot safe for a background calculation."""
        return [
            replace(ap, radios=[replace(radio) for radio in ap.radios])
            for ap in aps
        ]

    def commit_access_point_group_move(self, moved_aps: List[AccessPoint], delta: QPointF):
        unique: List[AccessPoint] = []
        seen = set()
        for ap in moved_aps:
            if id(ap) in seen:
                continue
            seen.add(id(ap))
            unique.append(ap)
        if not unique:
            return

        # The AP symbol, boresight, label and cutoff rings have already moved in
        # the existing scene.  Do not reconstruct every IFC object just to commit
        # new coordinates.  Keep the previous heatmap visible but suppress its
        # hover values until the asynchronous replacement is ready.
        self._rssi_result_stale = bool(self.last_result is not None)
        self._update_ap_table_positions(unique)
        self.refresh_access_point_rulers()
        if self.view.scene() is not None:
            self.view.scene().update()
        self.statusBar().showMessage(
            f"Moved {len(unique)} access point(s) by ΔX {float(delta.x()):.2f} m, ΔY {float(delta.y()):.2f} m. "
            "RSSI preview queued in the background."
        )
        self._schedule_interactive_rf_preview()

    def _schedule_interactive_rf_preview(self):
        if not bool(getattr(self.heatmap_settings, "interactive_preview_enabled", True)):
            return
        if self.floor is None or not self.aps:
            return
        self._interactive_preview_generation += 1
        self._interactive_preview_pending = True
        delay = max(50, int(getattr(self.heatmap_settings, "interactive_preview_delay_ms", 350)))
        self._interactive_preview_timer.start(delay)

    def _invalidate_interactive_preview_requests(self):
        """Discard queued/running preview output without blocking for its worker."""
        if not hasattr(self, "_interactive_preview_generation"):
            return
        self._interactive_preview_generation += 1
        self._interactive_preview_pending = False
        self._interactive_preview_timer.stop()
        future = self._interactive_preview_future
        if future is not None and not future.running():
            future.cancel()

    def _run_interactive_rf_preview(self):
        if self.floor is None or not self.aps:
            return
        future = self._interactive_preview_future
        if future is not None and not future.done():
            # A rapid second drag should not start another complete calculation.
            # The generation marker makes the current output stale; the latest
            # request starts as soon as the single background slot is free.
            self._interactive_preview_pending = True
            self.statusBar().showMessage("AP moved again; coalescing the RSSI preview update...")
            return

        active_freqs = sorted({
            float(radio.frequency_mhz)
            for ap in self.aps for radio in ap.active_radios()
        })
        if not active_freqs:
            return

        generation = int(self._interactive_preview_generation)
        floor = self.floor
        aps_snapshot = self._snapshot_rf_access_points(self.aps)
        settings_snapshot = copy.deepcopy(self.heatmap_settings)
        patterns_snapshot = copy.deepcopy(self.antenna_patterns)
        resolution = max(
            float(self.resolution.value()),
            float(getattr(settings_snapshot, "interactive_preview_resolution_m", 3.0)),
        )
        had_current_fields = bool(self.rssi_results_by_frequency)
        profile_override = str(getattr(settings_snapshot, "rf_calculation_profile", "balanced")) if had_current_fields else "fast"
        cold_start = not had_current_fields
        if cold_start and len(aps_snapshot) >= 8:
            # Immediately after predictive placement no per-AP heatmap fields
            # exist.  A bounded direct/penetration preview avoids ray tracing all
            # planned APs before the user can continue editing.  The explicit
            # Calculate RSSI command still uses the configured propagation model.
            settings_snapshot.enable_multipath_reflections = False
            settings_snapshot.enable_corner_diffraction = False
            settings_snapshot.enable_small_scale_fading = False
            settings_snapshot.calculate_delay_spread = False
            settings_snapshot.enable_adaptive_rf_grid = True
            settings_snapshot.adaptive_coarse_resolution_m = max(
                float(settings_snapshot.adaptive_coarse_resolution_m), resolution, 4.0
            )
            settings_snapshot.heatmap_render_mode = "raster"

        self._interactive_preview_running = True
        self._interactive_preview_pending = False
        self._interactive_preview_job_generation = generation
        self._interactive_preview_job_floor = str(floor.name)
        self._interactive_preview_profile = profile_override
        self._interactive_preview_cold_start = cold_start
        self.statusBar().showMessage(
            "Building a rapid RSSI preview in the background..."
            if cold_start else
            "Recalculating only changed AP field(s) in the background..."
        )
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._interactive_preview_future = future
        calculation_boundary = self._rssi_calculation_boundary()
        include_inter_floor = self.include_inter_floor.isChecked()

        def calculate_preview():
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = RFEngine.simulate_frequencies(
                    floor,
                    self.floors,
                    aps_snapshot,
                    active_freqs,
                    resolution,
                    patterns_snapshot,
                    include_inter_floor,
                    settings_snapshot,
                    None,
                    calculation_boundary,
                    None,
                    profile_override,
                )
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        # A daemon coordinator keeps the Qt thread responsive while the existing
        # process pool performs the numerical work.  Daemon lifetime also means a
        # long preview cannot prevent the application from closing.
        self._interactive_preview_thread = threading.Thread(
            target=calculate_preview,
            name="rf-interactive-preview",
            daemon=True,
        )
        self._interactive_preview_thread.start()
        self._interactive_preview_poll_timer.start()

    def _poll_interactive_rf_preview(self):
        future = self._interactive_preview_future
        if future is None:
            self._interactive_preview_poll_timer.stop()
            self._interactive_preview_running = False
            return
        if not future.done():
            return

        self._interactive_preview_poll_timer.stop()
        job_generation = int(self._interactive_preview_job_generation)
        job_floor = str(self._interactive_preview_job_floor)
        cold_start = bool(self._interactive_preview_cold_start)
        self._interactive_preview_future = None
        self._interactive_preview_running = False
        try:
            results = future.result()
        except concurrent.futures.CancelledError:
            results = {}
        except Exception as exc:
            results = {}
            if job_generation == self._interactive_preview_generation:
                self.statusBar().showMessage(f"Background RSSI preview skipped: {exc}")

        current_floor_name = str(self.floor.name) if self.floor is not None else ""
        current_request = (
            job_generation == self._interactive_preview_generation
            and job_floor == current_floor_name
        )
        if current_request and results:
            converted = {float(key): value for key, value in results.items()}
            self.rssi_results_by_frequency = converted
            selected = self._selected_rssi_view_frequency()
            self.last_result = converted.get(selected) or converted[sorted(converted)[0]]
            self._rssi_result_stale = False
            scene = self.view.scene()
            self._pending_ap_selection_ids = {
                id(item.ap) for item in (scene.selectedItems() if scene is not None else [])
                if isinstance(item, AccessPointGraphicsItem)
            }
            self._preserve_view_on_redraw = True
            self.draw_floor()
            misses = max((int(result.cache_misses) for result in converted.values()), default=0)
            hits = max((int(result.cache_hits) for result in converted.values()), default=0)
            if cold_start:
                message = (
                    "Rapid post-planner RSSI preview updated in the background. "
                    "Run Calculate RSSI for the configured propagation study."
                )
            else:
                message = f"RSSI preview updated: {misses} AP field(s) recalculated, {hits} reused."
            self.statusBar().showMessage(message)

        # If the AP moved again while this job was running, immediately start the
        # newest coalesced request.  No stale result is ever applied to the scene.
        if self._interactive_preview_pending:
            self._interactive_preview_pending = False
            self._interactive_preview_timer.stop()
            QTimer.singleShot(0, self._run_interactive_rf_preview)

    def focus_ap_in_table(self, ap: AccessPoint):
        self.inspector_tabs.setCurrentIndex(1)
        for row in range(self.ap_table.rowCount()):
            item = self.ap_table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == ap.name:
                self.ap_table.selectRow(row)
                self.ap_table.scrollToItem(item)
                self.ap_table.setCurrentItem(item)
                return

    def duplicate_access_point(self, ap: AccessPoint):
        duplicate = replace(
            ap,
            name=self._next_ap_name(),
            x=float(ap.x) + 1.0,
            y=float(ap.y) + 1.0,
            planned=False,
            radios=[replace(radio) for radio in ap.radios],
        )
        self.aps.append(duplicate)
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.draw_floor()
        self.populate_ap_table()
        self.statusBar().showMessage(f"Duplicated {ap.name} as {duplicate.name}")

    def delete_access_point(self, ap: AccessPoint):
        self.aps = [candidate for candidate in self.aps if candidate is not ap]
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.draw_floor()
        self.populate_ap_table()
        self.statusBar().showMessage(f"Deleted {ap.name}")

    def apply_radio_profile_to_ap(self, ap: AccessPoint, profile_name: str):
        ap.radios = self._radios_for_profile(profile_name, ap.ap_type)
        ap.radio_profile = str(profile_name)
        if ap.radios:
            ap.tx_power_dbm = float(ap.radios[0].tx_power_dbm)
            ap.frequency_mhz = float(ap.radios[0].frequency_mhz)
            ap.antenna_pattern = ap.radios[0].antenna_pattern
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self._rssi_result_stale = False
        self._refresh_rssi_frequency_dropdown()
        self.draw_floor()
        self.populate_ap_table()
        self.statusBar().showMessage(f"Applied {profile_name} to {ap.name}")

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
        if enabled:
            self._disable_ap_tools_for_geometry_mode()
        if enabled and getattr(self, "boundary_draw_mode", False):
            self.cancel_planner_boundary_drawing()
        if enabled and getattr(self, "space_draw_mode", False):
            self.cancel_space_drawing()
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
            self.statusBar().showMessage("Draw RF wall: click an existing IFC wall/space edge for the first endpoint. Hold Shift for a horizontal or vertical wall. Right-click to cancel.")
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

    def _clear_space_preview(self):
        scene = self.view.scene()
        for item in list(getattr(self, "_space_preview_items", [])):
            try:
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self._space_preview_items = []

    def toggle_space_draw_mode(self, enabled: bool):
        if enabled:
            self._disable_ap_tools_for_geometry_mode()
        if enabled and getattr(self, "wall_draw_mode", False):
            self.cancel_user_wall_drawing()
        if enabled and getattr(self, "boundary_draw_mode", False):
            self.cancel_planner_boundary_drawing(show_status=False)
        self.space_draw_mode = bool(enabled)
        self._space_polygon_points = []
        self._clear_space_preview()
        if enabled:
            if not self.floor:
                QMessageBox.information(self, "No floor selected", "Load an IFC and select a floor before drawing a space.")
                self.cancel_space_drawing(show_status=False)
                return
            self.view.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage(
                "Draw space: left-click each vertex; hold Shift to constrain the next segment horizontal or vertical; right-click or click the first vertex to finish."
            )
        else:
            self.view.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("Space drawing stopped")

    def cancel_space_drawing(self, show_status: bool = True):
        self._space_polygon_points = []
        self._clear_space_preview()
        self.space_draw_mode = False
        action = getattr(self, "draw_space_action", None)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)
        self.view.setCursor(Qt.ArrowCursor)
        if show_status:
            self.statusBar().showMessage("Space drawing cancelled")

    @staticmethod
    def axis_constrained_point(start: QPointF, candidate: QPointF) -> QPointF:
        """Constrain a drawing point to the dominant horizontal/vertical axis."""
        sx, sy = float(start.x()), float(start.y())
        cx, cy = float(candidate.x()), float(candidate.y())
        if abs(cx - sx) >= abs(cy - sy):
            return QPointF(cx, sy)
        return QPointF(sx, cy)

    def _clear_boundary_preview(self):
        scene = self.view.scene()
        for item in list(getattr(self, "_boundary_preview_items", [])):
            try:
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self._boundary_preview_items = []

    def toggle_planner_boundary_draw_mode(self, enabled: bool, shape: str = "rectangle"):
        if enabled:
            self._disable_ap_tools_for_geometry_mode()
        shape = "polygon" if str(shape).lower() == "polygon" else "rectangle"
        if not enabled and (
            not self.boundary_draw_mode or self.boundary_draw_shape != shape
        ):
            return
        if enabled and getattr(self, "wall_draw_mode", False):
            self.cancel_user_wall_drawing()
        if enabled and getattr(self, "space_draw_mode", False):
            self.cancel_space_drawing()
        if enabled and self.boundary_draw_mode and self.boundary_draw_shape != shape:
            self.cancel_planner_boundary_drawing(show_status=False)

        self.boundary_draw_mode = bool(enabled)
        self.boundary_draw_shape = shape
        self._boundary_draw_start = None
        self._boundary_polygon_points = []
        self._clear_boundary_preview()

        if enabled:
            if not self.floor:
                QMessageBox.information(
                    self,
                    "No floor selected",
                    "Load an IFC and select a floor before drawing a shared planner boundary.",
                )
                self.cancel_planner_boundary_drawing(show_status=False)
                return
            other_action = (
                getattr(self, "draw_polygon_boundary_action", None)
                if shape == "rectangle"
                else getattr(self, "draw_boundary_action", None)
            )
            if other_action is not None:
                other_action.blockSignals(True)
                other_action.setChecked(False)
                other_action.blockSignals(False)
            self.view.setCursor(Qt.CrossCursor)
            if shape == "polygon":
                self.statusBar().showMessage(
                    "Draw shared polygon boundary: left-click each vertex; hold Shift to constrain the next segment horizontal or vertical; click the first vertex or right-click after at least three vertices to finish."
                )
            else:
                self.statusBar().showMessage(
                    "Draw shared rectangular boundary: click two opposite corners. Right-click to cancel."
                )
        else:
            self.view.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("Planner boundary drawing stopped")

    def cancel_planner_boundary_drawing(self, show_status: bool = True):
        self._boundary_draw_start = None
        self._boundary_polygon_points = []
        self._clear_boundary_preview()
        self.boundary_draw_mode = False
        for action_name in ("draw_boundary_action", "draw_polygon_boundary_action"):
            action = getattr(self, action_name, None)
            if action is not None:
                action.blockSignals(True)
                action.setChecked(False)
                action.blockSignals(False)
        self.view.setCursor(Qt.ArrowCursor)
        if show_status:
            self.statusBar().showMessage("Planner boundary drawing cancelled")

    def show_planner_boundary_preview(self, end_point: QPointF, shift_constrain: bool = False):
        self._clear_boundary_preview()
        if self.view.scene() is None:
            return
        pen = QPen(QColor("#00A6D6"), 1.5)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)

        if self.boundary_draw_shape == "polygon":
            points = list(self._boundary_polygon_points)
            if not points:
                return
            preview_point = QPointF(float(end_point.x()), float(end_point.y()))
            if shift_constrain:
                preview_point = self.axis_constrained_point(points[-1], preview_point)
            points.append(preview_point)
            polygon = QPolygonF(points)
            preview = QGraphicsPolygonItem(polygon)
            preview.setPen(pen)
            fill = QColor("#00A6D6")
            fill.setAlpha(24 if len(points) >= 3 else 0)
            preview.setBrush(QBrush(fill) if fill.alpha() else QBrush(Qt.NoBrush))
            preview.setZValue(Z_AP_LABEL + 20)
            self.view.scene().addItem(preview)
            items: List[QGraphicsItem] = [preview]
            marker_pen = QPen(QColor("#00A6D6"), 1.0)
            marker_pen.setCosmetic(True)
            marker_brush = QBrush(QColor("#FFFFFF"))
            for vertex in self._boundary_polygon_points:
                marker = self.view.scene().addEllipse(
                    float(vertex.x()) - 0.08,
                    float(vertex.y()) - 0.08,
                    0.16,
                    0.16,
                    marker_pen,
                    marker_brush,
                )
                marker.setZValue(Z_AP_LABEL + 21)
                items.append(marker)
            self._boundary_preview_items = items
            return

        if self._boundary_draw_start is None:
            return
        start = self._boundary_draw_start
        minx, maxx = sorted((float(start.x()), float(end_point.x())))
        miny, maxy = sorted((float(start.y()), float(end_point.y())))
        item = self.view.scene().addRect(
            minx, miny, maxx - minx, maxy - miny, pen, QBrush(Qt.NoBrush)
        )
        item.setZValue(Z_AP_LABEL + 20)
        self._boundary_preview_items = [item]

    @staticmethod
    def _planner_points_are_close(first: QPointF, second: QPointF, tolerance_m: float = 0.25) -> bool:
        return math.hypot(
            float(first.x()) - float(second.x()),
            float(first.y()) - float(second.y()),
        ) <= float(tolerance_m)

    def show_space_preview(self, end_point: QPointF, shift_constrain: bool = False):
        self._clear_space_preview()
        scene = self.view.scene()
        if scene is None or not self._space_polygon_points:
            return
        points = list(self._space_polygon_points)
        preview_point = QPointF(float(end_point.x()), float(end_point.y()))
        if shift_constrain:
            preview_point = self.axis_constrained_point(points[-1], preview_point)
        points.append(preview_point)
        polygon = QPolygonF(points)
        fill = QColor("#2D9CDB")
        fill.setAlpha(45 if len(points) >= 3 else 0)
        pen = QPen(QColor("#2D9CDB"), 1.5)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        preview = QGraphicsPolygonItem(polygon)
        preview.setPen(pen)
        preview.setBrush(QBrush(fill) if fill.alpha() else QBrush(Qt.NoBrush))
        preview.setZValue(Z_AP_LABEL + 24)
        scene.addItem(preview)
        items: List[QGraphicsItem] = [preview]
        marker_pen = QPen(QColor("#2D9CDB"), 1.0)
        marker_pen.setCosmetic(True)
        marker_brush = QBrush(QColor("#FFFFFF"))
        for vertex in self._space_polygon_points:
            marker = scene.addEllipse(
                float(vertex.x()) - 0.08,
                float(vertex.y()) - 0.08,
                0.16,
                0.16,
                marker_pen,
                marker_brush,
            )
            marker.setZValue(Z_AP_LABEL + 25)
            items.append(marker)
        self._space_preview_items = items

    def capture_space_point(self, scene_pos: QPointF, shift_constrain: bool = False):
        if not self.floor:
            return
        raw_point = QPointF(float(scene_pos.x()), float(scene_pos.y()))
        if (
            len(self._space_polygon_points) >= 3
            and self._planner_points_are_close(self._space_polygon_points[0], raw_point)
        ):
            self.finish_user_space()
            return
        point = raw_point
        if shift_constrain and self._space_polygon_points:
            point = self.axis_constrained_point(self._space_polygon_points[-1], raw_point)
        if (
            self._space_polygon_points
            and self._planner_points_are_close(self._space_polygon_points[-1], point, 0.05)
        ):
            self.statusBar().showMessage("Space vertices must be at least 0.05 m apart.")
            return
        self._space_polygon_points.append(point)
        self.show_space_preview(point)
        count = len(self._space_polygon_points)
        if count < 3:
            self.statusBar().showMessage(f"Space vertex {count} captured. Add at least {3 - count} more vertex/vertices.")
        else:
            self.statusBar().showMessage("Space polygon ready. Right-click or click the first vertex to finish.")

    def finish_user_space(self):
        if not self.floor:
            return
        points = list(self._space_polygon_points)
        if len(points) < 3:
            self.statusBar().showMessage("A space requires at least three vertices.")
            return
        polygon = Polygon([(float(point.x()), float(point.y())) for point in points])
        if not polygon.is_valid:
            self.statusBar().showMessage("Space polygon is self-intersecting or otherwise invalid. Adjust the vertices or cancel drawing.")
            return
        if polygon.is_empty or float(polygon.area) < 0.05:
            self.statusBar().showMessage("Space area must be at least 0.05 m2.")
            return
        index = 1 + sum(1 for space in self.floor.spaces if space.is_user_created)
        self.floor.spaces.append(Space2D(
            guid=f"user-space-{uuid.uuid4().hex}",
            name=f"User Space {index}",
            floor=self.floor.name,
            source_file="RF simulator user space",
            polygon=polygon,
            z_min=float(self.floor.elevation),
            z_max=float(self.floor.elevation + 3.0),
            source_storey=self.floor.name,
            is_user_created=True,
            assumption_note="Drawn manually in the RF simulator; source IFC is unchanged.",
        ))
        self._space_polygon_points = []
        self._clear_space_preview()
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        self.statusBar().showMessage(
            f"Created User Space {index} on '{self.floor.name}'. Continue drawing another space or toggle the tool off."
        )

    def capture_planner_boundary_point(self, scene_pos: QPointF, shift_constrain: bool = False):
        if not self.floor:
            return
        raw_point = QPointF(float(scene_pos.x()), float(scene_pos.y()))
        point = raw_point

        if self.boundary_draw_shape == "polygon":
            if (
                len(self._boundary_polygon_points) >= 3
                and self._planner_points_are_close(self._boundary_polygon_points[0], raw_point)
            ):
                self.finish_planner_polygon_boundary()
                return
            if shift_constrain and self._boundary_polygon_points:
                point = self.axis_constrained_point(self._boundary_polygon_points[-1], raw_point)
            if (
                self._boundary_polygon_points
                and self._planner_points_are_close(self._boundary_polygon_points[-1], point, 0.05)
            ):
                self.statusBar().showMessage("Polygon vertices must be at least 0.05 m apart.")
                return
            self._boundary_polygon_points.append(point)
            self.show_planner_boundary_preview(point)
            count = len(self._boundary_polygon_points)
            if count < 3:
                self.statusBar().showMessage(
                    f"Polygon vertex {count} captured. Add at least {3 - count} more vertex/vertices."
                )
            else:
                self.statusBar().showMessage(
                    f"Polygon vertex {count} captured. Hold Shift for horizontal/vertical segments; right-click or click the first vertex to finish."
                )
            return

        if self._boundary_draw_start is None:
            self._boundary_draw_start = point
            self.show_planner_boundary_preview(point)
            self.statusBar().showMessage(
                "First rectangular-boundary corner captured. Click the opposite corner."
            )
            return
        start = self._boundary_draw_start
        minx, maxx = sorted((float(start.x()), float(point.x())))
        miny, maxy = sorted((float(start.y()), float(point.y())))
        if maxx - minx < 0.10 or maxy - miny < 0.10:
            self.statusBar().showMessage(
                "Planner boundary must be at least 0.10 m wide and high."
            )
            return
        boundary = PlannerBoundary2D(
            guid=f"planner-boundary-{uuid.uuid4().hex}",
            name=f"Planner boundary {len(self.planner_boundaries) + 1}",
            polygon=box(minx, miny, maxx, maxy),
            shape_type="rectangle",
        )
        self.planner_boundaries.append(boundary)
        self._boundary_draw_start = None
        self._clear_boundary_preview()
        self._clear_rssi_results()
        self.draw_floor()
        self.statusBar().showMessage(
            f"Created {boundary.name}; it now constrains AP planning on all IFC floors. "
            "Click two more corners to add another rectangle, or right-click to finish."
        )

    def finish_planner_polygon_boundary(self):
        points = list(self._boundary_polygon_points)
        if len(points) < 3:
            self.statusBar().showMessage("A polygon boundary requires at least three vertices.")
            return
        coordinates = [(float(point.x()), float(point.y())) for point in points]
        polygon = Polygon(coordinates)
        if polygon.is_empty or float(polygon.area) < 0.01:
            self.statusBar().showMessage("Polygon boundary area must be at least 0.01 m².")
            return
        if not polygon.is_valid:
            self.statusBar().showMessage(
                "Polygon boundary is self-intersecting or otherwise invalid. Adjust the vertices or cancel drawing."
            )
            return
        boundary = PlannerBoundary2D(
            guid=f"planner-boundary-{uuid.uuid4().hex}",
            name=f"Planner boundary {len(self.planner_boundaries) + 1}",
            polygon=polygon,
            shape_type="polygon",
        )
        self.planner_boundaries.append(boundary)
        self._boundary_polygon_points = []
        self._clear_boundary_preview()
        self._clear_rssi_results()
        self.draw_floor()
        self.statusBar().showMessage(
            f"Created {boundary.name} with {len(coordinates)} vertices; it constrains AP planning on all IFC floors. "
            "Left-click to start another polygon or toggle the tool off."
        )

    def _clear_suggested_boundary_preview(self):
        scene = self.view.scene()
        for item in list(getattr(self, "_suggested_boundary_preview_items", [])):
            try:
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self._suggested_boundary_preview_items = []

    def _show_suggested_boundary_preview(self, polygons: List[Polygon]):
        """Overlay the proposed outer-wall chain without committing it."""
        self._clear_suggested_boundary_preview()
        scene = self.view.scene()
        if scene is None:
            return

        items: List[QGraphicsItem] = []
        outline_colour = QColor("#D000FF")
        fill_colour = QColor("#D000FF")
        fill_colour.setAlpha(35)
        pen = QPen(outline_colour, 2.2)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashDotLine)
        marker_pen = QPen(outline_colour, 1.2)
        marker_pen.setCosmetic(True)
        marker_brush = QBrush(QColor("#FFFFFF"))

        for index, polygon in enumerate(polygons, start=1):
            coords = list(polygon.exterior.coords)
            qpolygon = QPolygonF([QPointF(float(x), float(y)) for x, y in coords])
            preview = QGraphicsPolygonItem(qpolygon)
            preview.setPen(pen)
            preview.setBrush(QBrush(fill_colour))
            preview.setAcceptedMouseButtons(Qt.NoButton)
            preview.setZValue(Z_AP_LABEL + 40)
            scene.addItem(preview)
            items.append(preview)

            # Show the chained vertices so the user can inspect where wall
            # segments have been bridged before accepting the suggestion.
            vertices = coords[:-1]
            step = max(1, int(math.ceil(len(vertices) / 250.0)))
            for x, y in vertices[::step]:
                marker = scene.addEllipse(
                    float(x) - 0.08,
                    float(y) - 0.08,
                    0.16,
                    0.16,
                    marker_pen,
                    marker_brush,
                )
                marker.setAcceptedMouseButtons(Qt.NoButton)
                marker.setZValue(Z_AP_LABEL + 41)
                items.append(marker)

            representative = polygon.representative_point()
            label = self._add_upright_text(
                scene,
                f"Suggested outer-wall chain {index}",
                float(representative.x),
                float(representative.y),
                outline_colour,
                max(3, int(self.heatmap_settings.space_label_font_size)),
                Z_AP_LABEL + 42,
                bold=True,
            )
            items.append(label)

        self._suggested_boundary_preview_items = items

    @staticmethod
    def _bounds_area(polygons: List[Polygon]) -> float:
        if not polygons:
            return 0.0
        minx = min(float(polygon.bounds[0]) for polygon in polygons)
        miny = min(float(polygon.bounds[1]) for polygon in polygons)
        maxx = max(float(polygon.bounds[2]) for polygon in polygons)
        maxy = max(float(polygon.bounds[3]) for polygon in polygons)
        return max(0.0, maxx - minx) * max(0.0, maxy - miny)

    def _wall_polygons_for_boundary_suggestion(self) -> Tuple[List[Polygon], str, int]:
        """Return unique IFC and RF wall polygons for tracing the outer chain.

        All IFC walls are used rather than trusting an ``IsExternal`` flag,
        because many authoring exports omit or only partially populate that
        property. Manually placed RF walls are always included as well, so a
        user can close missing facade sections before requesting a boundary
        suggestion. Internal walls remain inside the union and therefore do
        not alter the extracted outer ring.
        """
        if not self.floor:
            return [], "", 0

        unique = set()
        ifc_walls: List[Wall2D] = []
        manual_rf_walls: List[Wall2D] = []
        generated_rf_walls: List[Wall2D] = []
        for wall in self.floor.walls:
            polygon = getattr(wall, "polygon", None)
            if polygon is None or polygon.is_empty or float(polygon.area) <= 1e-6:
                continue
            key = (str(wall.source_file), str(wall.guid), bytes(polygon.wkb))
            if key in unique:
                continue
            unique.add(key)
            if not wall.is_user_created:
                ifc_walls.append(wall)
            elif str(getattr(wall, "source_file", "")) == "RF simulator external baseline":
                generated_rf_walls.append(wall)
            else:
                manual_rf_walls.append(wall)

        selected = ifc_walls + manual_rf_walls + generated_rf_walls
        source_parts: List[str] = []
        if ifc_walls:
            source_parts.append(f"{len(ifc_walls)} IFC wall{'s' if len(ifc_walls) != 1 else ''}")
        if manual_rf_walls:
            source_parts.append(
                f"{len(manual_rf_walls)} manually placed RF wall"
                f"{'s' if len(manual_rf_walls) != 1 else ''}"
            )
        if generated_rf_walls:
            source_parts.append(
                f"{len(generated_rf_walls)} generated RF wall"
                f"{'s' if len(generated_rf_walls) != 1 else ''}"
            )
        if not source_parts:
            source_label = "no usable walls"
        elif len(source_parts) == 1:
            source_label = source_parts[0]
        else:
            source_label = ", ".join(source_parts[:-1]) + " and " + source_parts[-1]
        return [wall.polygon for wall in selected], source_label, len(selected)

    def suggest_external_planner_boundary(self):
        """Preview and optionally accept an outer-wall-derived shared boundary."""
        if not self.floor:
            QMessageBox.information(
                self,
                "No floor selected",
                "Load an IFC model and select a floor before suggesting an external boundary.",
            )
            return

        if getattr(self, "wall_draw_mode", False):
            self.cancel_user_wall_drawing()
        if getattr(self, "boundary_draw_mode", False):
            self.cancel_planner_boundary_drawing(show_status=False)

        wall_polygons, source_label, source_count = self._wall_polygons_for_boundary_suggestion()
        if not wall_polygons:
            QMessageBox.information(
                self,
                "No wall geometry",
                "The selected floor has no usable IFC or manually placed RF wall geometry from which to form an external chain.",
            )
            return

        default_gap = estimate_outer_wall_gap_tolerance(wall_polygons)
        dialog = QDialog(self)
        dialog.setWindowTitle("Suggested external planner boundary")
        dialog.setModal(True)
        dialog.resize(590, 300)
        layout = QVBoxLayout(dialog)

        explanation = QLabel(
            f"The magenta chain is derived from {source_label} on "
            f"'{self.floor.name}'. If accepted, the resulting polygon boundary is shared by all IFC floors. "
            "Increase the bridge distance where doors or missing wall segments leave the chain open."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form_widget = QWidget(dialog)
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 4, 0, 4)
        gap_spin = QDoubleSpinBox(form_widget)
        gap_spin.setRange(0.05, 10.0)
        gap_spin.setDecimals(2)
        gap_spin.setSingleStep(0.10)
        gap_spin.setValue(float(default_gap))
        gap_spin.setSuffix(" m")
        gap_spin.setToolTip(
            "Maximum opening bridged while joining wall segments. Increase for large doors or incomplete IFC wall chains; "
            "reduce it to avoid joining detached buildings."
        )
        form.addRow("Maximum wall-gap bridge", gap_spin)
        layout.addWidget(form_widget)

        summary = QLabel()
        summary.setWordWrap(True)
        summary.setFrameShape(QFrame.StyledPanel)
        summary.setMinimumHeight(72)
        layout.addWidget(summary)

        replace_existing = QCheckBox("Replace existing planner boundaries when accepted")
        replace_existing.setChecked(False)
        layout.addWidget(replace_existing)

        controls = QHBoxLayout()
        update_button = QPushButton("Update preview")
        update_button.setIcon(self._standard_icon("SP_BrowserReload"))
        controls.addWidget(update_button)
        controls.addStretch(1)
        accept_button = QPushButton("Accept suggestion")
        accept_button.setIcon(self._standard_icon("SP_DialogApplyButton"))
        cancel_button = QPushButton("Cancel")
        cancel_button.setIcon(self._standard_icon("SP_DialogCancelButton"))
        controls.addWidget(accept_button)
        controls.addWidget(cancel_button)
        layout.addLayout(controls)

        current_polygons: List[Polygon] = []
        current_metadata: Dict[str, object] = {}

        def update_preview():
            nonlocal current_polygons, current_metadata
            self.statusBar().showMessage("Tracing the outermost IFC and RF wall chain...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                current_polygons, current_metadata = suggest_external_boundary_polygons(
                    wall_polygons,
                    gap_tolerance_m=float(gap_spin.value()),
                )
            finally:
                QApplication.restoreOverrideCursor()

            self._show_suggested_boundary_preview(current_polygons)
            warnings = list(current_metadata.get("warnings", []) or [])
            if current_polygons:
                area = float(current_metadata.get("total_area_m2", 0.0) or 0.0)
                vertices = int(current_metadata.get("vertex_count", 0) or 0)
                chain_count = len(current_polygons)
                message = (
                    f"Preview: {chain_count} outer chain{'s' if chain_count != 1 else ''}, "
                    f"{vertices} vertices and {area:,.1f} m² total permitted area."
                )
                if warnings:
                    message += "\n\nReview: " + " ".join(str(value) for value in warnings)
                summary.setText(message)
                accept_button.setEnabled(True)
                self.statusBar().showMessage(
                    "External boundary suggestion previewed in magenta. Accept it or adjust the wall-gap bridge."
                )
            else:
                message = "No closed external chain was formed at this bridge distance."
                if warnings:
                    message += "\n\n" + " ".join(str(value) for value in warnings)
                summary.setText(message)
                accept_button.setEnabled(False)
                self.statusBar().showMessage(
                    "No closed outer-wall chain was found; increase the wall-gap bridge and update the preview."
                )

        update_button.clicked.connect(update_preview)
        # The preview is recalculated explicitly with the Update preview button.
        accept_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        update_preview()
        result = dialog.exec()
        self._clear_suggested_boundary_preview()

        if result != QDialog.Accepted or not current_polygons:
            self.statusBar().showMessage("External boundary suggestion cancelled")
            return

        if replace_existing.isChecked():
            self.planner_boundaries.clear()

        for offset, polygon in enumerate(current_polygons):
            chain_suffix = f" {offset + 1}" if len(current_polygons) > 1 else ""
            self.planner_boundaries.append(PlannerBoundary2D(
                guid=f"planner-boundary-{uuid.uuid4().hex}",
                name=f"Suggested external boundary{chain_suffix}",
                polygon=polygon,
                shape_type="polygon",
            ))

        self._clear_rssi_results()
        self.draw_floor()
        created = len(current_polygons)
        self.statusBar().showMessage(
            f"Accepted {created} suggested external planner boundar{'y' if created == 1 else 'ies'}; "
            "the permitted area now applies to all IFC floors."
        )

    def _clear_suggested_space_preview(self):
        scene = self.view.scene()
        for item in list(getattr(self, "_suggested_space_preview_items", [])):
            try:
                if scene is not None and item.scene() is scene:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self._suggested_space_preview_items = []

    def _show_suggested_space_preview(
        self, polygons: List[Polygon], external_wall_polygons: Optional[List[Polygon]] = None
    ):
        """Overlay inferred spaces and optional missing-facade RF walls."""
        self._clear_suggested_space_preview()
        scene = self.view.scene()
        if scene is None:
            return
        items: List[QGraphicsItem] = []
        outline = QColor("#00B894")
        fill = QColor("#00B894")
        fill.setAlpha(55)
        pen = QPen(outline, 1.6)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        for index, polygon in enumerate(polygons, start=1):
            coords = list(polygon.exterior.coords)
            qpolygon = QPolygonF([QPointF(float(x), float(y)) for x, y in coords])
            item = QGraphicsPolygonItem(qpolygon)
            item.setPen(pen)
            item.setBrush(QBrush(fill))
            item.setAcceptedMouseButtons(Qt.NoButton)
            item.setZValue(Z_AP_LABEL + 30)
            scene.addItem(item)
            items.append(item)
            point = polygon.representative_point()
            label = self._add_upright_text(
                scene,
                f"Proposed space {index}\n{float(polygon.area):,.1f} m²",
                float(point.x),
                float(point.y),
                outline,
                max(3, int(self.heatmap_settings.space_label_font_size)),
                Z_AP_LABEL + 31,
                bold=True,
            )
            items.append(label)

        wall_outline = QColor("#FF7A00")
        wall_fill = QColor("#FF7A00")
        wall_fill.setAlpha(105)
        wall_pen = QPen(wall_outline, 2.0)
        wall_pen.setCosmetic(True)
        for polygon in external_wall_polygons or []:
            coords = list(polygon.exterior.coords)
            qpolygon = QPolygonF([QPointF(float(x), float(y)) for x, y in coords])
            item = QGraphicsPolygonItem(qpolygon)
            item.setPen(wall_pen)
            item.setBrush(QBrush(wall_fill))
            item.setAcceptedMouseButtons(Qt.NoButton)
            item.setZValue(Z_AP_LABEL + 33)
            scene.addItem(item)
            items.append(item)
        self._suggested_space_preview_items = items

    @staticmethod
    def _ifc_element_blocks_space_inference(element: IFCElement2D) -> bool:
        text = f"{element.type_name} {element.name} {element.material}".lower()
        blocking_tokens = (
            "curtainwall", "curtain wall", "stair", "stairflight",
            "ramp", "railing", "member", "column", "beam", "plate",
            "footing", "roof", "covering",
        )
        pass_through_tokens = (
            "door", "window", "opening", "slab", "floor", "ceiling", "deck",
            "proxy", "buildingelementproxy", "generic model", "generic_model",
        )
        return any(token in text for token in blocking_tokens) and not any(token in text for token in pass_through_tokens)

    def _space_inference_barrier_polygons(self) -> Tuple[List[Polygon], Dict[str, int]]:
        if not self.floor:
            return [], {"wall_count": 0, "element_count": 0}
        polygons: List[Polygon] = []
        seen = set()
        wall_count = 0
        element_count = 0

        def add_polygon(polygon: Polygon, source_key: str) -> bool:
            if polygon is None or polygon.is_empty or float(polygon.area) <= 1e-6:
                return False
            key = (source_key, bytes(polygon.wkb))
            if key in seen:
                return False
            seen.add(key)
            polygons.append(polygon)
            return True

        for wall in self.floor.walls:
            # Previously inferred facade segments are regenerated from the
            # current baseline and must not mask a changed boundary.
            if str(getattr(wall, "source_file", "")) == "RF simulator external baseline":
                continue
            if add_polygon(getattr(wall, "polygon", None), f"wall:{wall.source_file}:{wall.guid}"):
                wall_count += 1

        for element in getattr(self.floor, "elements", []):
            if not self._ifc_element_blocks_space_inference(element):
                continue
            if add_polygon(getattr(element, "polygon", None), f"element:{element.source_file}:{element.guid}"):
                element_count += 1

        return polygons, {"wall_count": wall_count, "element_count": element_count}

    def _space_inference_wall_polygons(self) -> List[Polygon]:
        polygons, _ = self._space_inference_barrier_polygons()
        return polygons

    def suggest_spaces_from_wall_boundaries(self):
        """Preview and create RF/planning spaces from wall-enclosed floor areas."""
        if not self.floor:
            QMessageBox.information(
                self,
                "No floor selected",
                "Load an IFC model and select a floor before creating spaces from walls.",
            )
            return
        if getattr(self, "wall_draw_mode", False):
            self.cancel_user_wall_drawing()
        if getattr(self, "boundary_draw_mode", False):
            self.cancel_planner_boundary_drawing(show_status=False)

        wall_polygons, barrier_counts = self._space_inference_barrier_polygons()
        if not wall_polygons:
            QMessageBox.information(
                self,
                "No wall geometry",
                "The selected floor has no usable IFC/RF wall or blocking element geometry from which to create spaces.",
            )
            return

        accepted_baselines = [
            boundary.polygon for boundary in self.planner_boundaries
            if boundary.polygon is not None and not boundary.polygon.is_empty
        ]
        existing_ifc_spaces = sum(1 for space in self.floor.spaces if not space.is_inferred and not space.is_user_created)
        default_gap = estimate_space_gap_tolerance(wall_polygons)

        dialog = QDialog(self)
        dialog.setWindowTitle("Create spaces from wall boundaries")
        dialog.setModal(True)
        dialog.resize(660, 420)
        layout = QVBoxLayout(dialog)

        baseline_text = (
            f"{len(accepted_baselines)} accepted shared planner boundary polygon(s) will be used as the external wall baseline."
            if accepted_baselines else
            "No shared planner boundary exists. The outermost-wall algorithm will suggest an external baseline and preview it in magenta."
        )
        explanation = QLabel(
            "The green polygons are proposed simulator spaces. Small wall gaps and ordinary door openings are bridged before the free floor area is split. "
            "Imported curtain walls, stairs, railings, structural members, plates and roofs are also used as barriers where they appear on this floor; floor slabs, ceiling elements and generic model/proxy elements are ignored. "
            "Where facade walls are absent, the external planner boundary is treated as a virtual outside wall. "
            "The source IFC is never modified.\n\n" + baseline_text
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form_widget = QWidget(dialog)
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 4, 0, 4)
        gap_spin = QDoubleSpinBox(form_widget)
        gap_spin.setRange(0.05, 5.0)
        gap_spin.setDecimals(2)
        gap_spin.setSingleStep(0.10)
        gap_spin.setValue(float(default_gap))
        gap_spin.setSuffix(" m")
        gap_spin.setToolTip(
            "Maximum doorway or modelling gap bridged when forming closed room cells. Reduce this if adjacent rooms merge; increase it if doorway gaps leave rooms connected."
        )
        form.addRow("Wall/door gap bridge", gap_spin)
        minimum_area_spin = QDoubleSpinBox(form_widget)
        minimum_area_spin.setRange(0.05, 1000.0)
        minimum_area_spin.setDecimals(2)
        minimum_area_spin.setSingleStep(0.5)
        minimum_area_spin.setValue(2.0)
        minimum_area_spin.setSuffix(" m²")
        minimum_area_spin.setToolTip("Enclosed fragments smaller than this area are ignored as wall pockets or modelling artefacts.")
        form.addRow("Minimum space area", minimum_area_spin)
        layout.addWidget(form_widget)

        replace_inferred = QCheckBox("Replace existing inferred spaces on this floor when accepted")
        replace_inferred.setChecked(True)
        layout.addWidget(replace_inferred)
        create_baseline = QCheckBox("Create the suggested external baseline as shared planner boundaries")
        create_baseline.setChecked(not bool(accepted_baselines))
        create_baseline.setEnabled(not bool(accepted_baselines))
        create_baseline.setToolTip(
            "Stores the inferred exterior chain as the same shared boundary used by AP planning on every IFC floor."
        )
        layout.addWidget(create_baseline)
        create_external_walls = QCheckBox("Create missing external RF walls along the baseline")
        create_external_walls.setChecked(True)
        create_external_walls.setToolTip(
            "Adds RF-only wall geometry only where the accepted or suggested exterior baseline is not already covered by an IFC/RF wall. Existing facade walls are not duplicated."
        )
        layout.addWidget(create_external_walls)
        replace_external_walls = QCheckBox("Replace external RF walls previously inferred by this tool")
        replace_external_walls.setChecked(True)
        replace_external_walls.setToolTip(
            "Removes only facade segments previously generated from a space-inference baseline before rebuilding them. Manually drawn RF walls are retained."
        )
        layout.addWidget(replace_external_walls)

        summary = QLabel()
        summary.setWordWrap(True)
        summary.setFrameShape(QFrame.StyledPanel)
        summary.setMinimumHeight(90)
        layout.addWidget(summary)

        controls = QHBoxLayout()
        update_button = QPushButton("Update preview")
        update_button.setIcon(self._standard_icon("SP_BrowserReload"))
        controls.addWidget(update_button)
        controls.addStretch(1)
        accept_button = QPushButton("Accept spaces")
        accept_button.setIcon(self._standard_icon("SP_DialogApplyButton"))
        cancel_button = QPushButton("Cancel")
        cancel_button.setIcon(self._standard_icon("SP_DialogCancelButton"))
        controls.addWidget(accept_button)
        controls.addWidget(cancel_button)
        layout.addLayout(controls)

        current_spaces: List[Polygon] = []
        current_external_wall_parts: List[Polygon] = []
        current_metadata: Dict[str, object] = {}

        def update_preview():
            nonlocal current_spaces, current_external_wall_parts, current_metadata
            self.statusBar().showMessage("Creating candidate spaces from wall boundaries...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                current_spaces, current_metadata = infer_space_polygons(
                    wall_polygons,
                    external_boundary_polygons=accepted_baselines or None,
                    gap_tolerance_m=float(gap_spin.value()),
                    minimum_area_m2=float(minimum_area_spin.value()),
                )
            finally:
                QApplication.restoreOverrideCursor()

            inferred_baselines = list(current_metadata.get("external_boundaries", []) or [])
            used_suggested = bool(current_metadata.get("used_suggested_external_boundary", False))
            if used_suggested:
                self._show_suggested_boundary_preview(inferred_baselines)
            else:
                self._clear_suggested_boundary_preview()
            active_preview_baselines = accepted_baselines or inferred_baselines
            current_external_wall_parts = (
                missing_external_wall_polygons(
                    wall_polygons,
                    active_preview_baselines,
                    wall_thickness_m=0.20,
                    existing_wall_tolerance_m=max(0.25, float(gap_spin.value()) * 0.35),
                )
                if create_external_walls.isChecked() and active_preview_baselines else []
            )
            self._show_suggested_space_preview(current_spaces, current_external_wall_parts)

            warnings = list(current_metadata.get("warnings", []) or [])
            if current_spaces:
                area = float(current_metadata.get("total_area_m2", 0.0) or 0.0)
                message = (
                    f"Preview: {len(current_spaces)} proposed space{'s' if len(current_spaces) != 1 else ''}, "
                    f"covering {area:,.1f} m² on '{self.floor.name}'."
                )
                if barrier_counts.get("element_count", 0):
                    wall_count = barrier_counts.get("wall_count", 0)
                    element_count = barrier_counts.get("element_count", 0)
                    message += (
                        f" Inference used {wall_count} wall/RF barrier"
                        f"{'s' if wall_count != 1 else ''} and {element_count} imported IFC element barrier"
                        f"{'s' if element_count != 1 else ''}."
                    )
                if existing_ifc_spaces:
                    message += f" The floor already contains {existing_ifc_spaces} IFC space(s); these will be retained."
                if used_suggested:
                    message += " The magenta external baseline was inferred using the same outer-wall chain method as planner boundary creation."
                if current_external_wall_parts:
                    message += (
                        f" {len(current_external_wall_parts)} uncovered facade segment"
                        f"{'s are' if len(current_external_wall_parts) != 1 else ' is'} shown in orange and will be added as RF-only external walls."
                    )
                if warnings:
                    message += "\n\nReview: " + " ".join(str(value) for value in warnings)
                summary.setText(message)
                accept_button.setEnabled(True)
                self.statusBar().showMessage("Space proposal previewed in green. Review the assumptions before accepting.")
            else:
                message = "No usable room spaces were formed with the current assumptions."
                if warnings:
                    message += "\n\n" + " ".join(str(value) for value in warnings)
                summary.setText(message)
                accept_button.setEnabled(False)
                self.statusBar().showMessage("No spaces were formed; adjust the gap bridge, minimum area, or external boundary.")

        update_button.clicked.connect(update_preview)
        create_external_walls.toggled.connect(lambda *_: update_preview())
        accept_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        update_preview()
        result = dialog.exec()
        self._clear_suggested_space_preview()
        self._clear_suggested_boundary_preview()

        if result != QDialog.Accepted or not current_spaces:
            self.statusBar().showMessage("Space creation cancelled")
            return

        used_suggested = bool(current_metadata.get("used_suggested_external_boundary", False))
        inferred_baselines = list(current_metadata.get("external_boundaries", []) or [])
        if used_suggested and create_baseline.isChecked():
            for offset, polygon in enumerate(inferred_baselines):
                suffix = f" {offset + 1}" if len(inferred_baselines) > 1 else ""
                self.planner_boundaries.append(PlannerBoundary2D(
                    guid=f"planner-boundary-{uuid.uuid4().hex}",
                    name=f"Space inference external boundary{suffix}",
                    polygon=polygon,
                    shape_type="polygon",
                ))

        active_baselines = accepted_baselines or inferred_baselines
        removed_external_walls = 0
        if replace_external_walls.isChecked():
            retained_walls = []
            for wall in self.floor.walls:
                if str(getattr(wall, "source_file", "")) == "RF simulator external baseline":
                    removed_external_walls += 1
                else:
                    retained_walls.append(wall)
            self.floor.walls = retained_walls

        created_external_walls = 0
        if create_external_walls.isChecked() and active_baselines:
            for polygon in current_external_wall_parts:
                wall = Wall2D(
                    guid=f"user-rf-wall-{uuid.uuid4().hex}",
                    name="Inferred missing external RF wall",
                    floor=self.floor.name,
                    source_file="RF simulator external baseline",
                    type_name="external wall",
                    material="masonry external wall",
                    polygon=polygon,
                    z_min=float(self.floor.elevation),
                    z_max=float(self.floor.elevation + 3.0),
                    source_storey=self.floor.name,
                    rf_type_override="external wall",
                    rf_customised=True,
                    is_user_created=True,
                    user_wall_thickness_m=0.20,
                )
                wall.attenuation_by_band_db = self._profile_for_wall_from_settings(wall)
                self.floor.walls.append(wall)
                created_external_walls += 1

        if replace_inferred.isChecked():
            self.floor.spaces = [space for space in self.floor.spaces if not space.is_inferred]

        starting_index = 1 + sum(1 for space in self.floor.spaces if space.is_inferred)
        assumption_note = (
            f"Generated from wall boundaries using a {float(gap_spin.value()):.2f} m gap bridge and "
            + (
                "an inferred outer-wall baseline."
                if used_suggested else
                "the accepted shared planner boundary as the external wall."
            )
        )
        if barrier_counts.get("element_count", 0):
            element_count = barrier_counts.get("element_count", 0)
            assumption_note += (
                f" Included {element_count} imported IFC element barrier"
                f"{'s' if element_count != 1 else ''}."
            )
        for offset, polygon in enumerate(current_spaces):
            self.floor.spaces.append(Space2D(
                guid=f"inferred-space-{uuid.uuid4().hex}",
                name=f"Inferred Space {starting_index + offset}",
                floor=self.floor.name,
                source_file="RF simulator inferred space",
                polygon=polygon,
                z_min=float(self.floor.elevation),
                z_max=float(self.floor.elevation + 3.0),
                source_storey=self.floor.name,
                is_inferred=True,
                assumption_note=assumption_note,
            ))

        self._clear_rssi_results()
        self.draw_floor()
        created = len(current_spaces)
        wall_message = (
            f" Added {created_external_walls} missing external RF wall segment{'s' if created_external_walls != 1 else ''} along the baseline."
            if created_external_walls else ""
        )
        if removed_external_walls and not created_external_walls:
            wall_message += f" Removed {removed_external_walls} previously inferred external RF wall segment{'s' if removed_external_walls != 1 else ''}."
        self.statusBar().showMessage(
            f"Created {created} inferred space{'s' if created != 1 else ''} on '{self.floor.name}'."
            + wall_message
            + " The source IFC was not changed."
        )

    def delete_inferred_space(self, space: Space2D):
        floor = self.floors.get(space.floor)
        if floor is None or not space.is_inferred:
            return
        floor.spaces = [candidate for candidate in floor.spaces if candidate is not space]
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        self.statusBar().showMessage(f"Deleted inferred space '{space.name}'")

    def inferred_spaces_at_scene_pos(self, scene_pos: QPointF) -> List[Space2D]:
        if not self.floor:
            return []
        point = Point(float(scene_pos.x()), float(scene_pos.y()))
        matches: List[Space2D] = []
        for space in self.floor.spaces:
            if not getattr(space, "is_inferred", False):
                continue
            polygon = getattr(space, "polygon", None)
            if polygon is None or polygon.is_empty:
                continue
            try:
                if polygon.covers(point):
                    matches.append(space)
            except Exception:
                continue
        return matches

    def clear_inferred_spaces(self):
        if not self.floor:
            self.statusBar().showMessage("No floor selected")
            return
        count = sum(1 for space in self.floor.spaces if space.is_inferred)
        if count == 0:
            self.statusBar().showMessage("No inferred spaces exist on the selected floor")
            return
        answer = QMessageBox.question(
            self,
            "Clear inferred spaces",
            f"Delete {count} inferred space{'s' if count != 1 else ''} from '{self.floor.name}'? IFC spaces will be retained.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.floor.spaces = [space for space in self.floor.spaces if not space.is_inferred]
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        self.statusBar().showMessage(f"Cleared {count} inferred space{'s' if count != 1 else ''}")

    def delete_planner_boundary(self, boundary: PlannerBoundary2D):
        self.planner_boundaries = [
            candidate for candidate in self.planner_boundaries if candidate is not boundary
        ]
        self._clear_rssi_results()
        self.draw_floor()
        self.statusBar().showMessage("Deleted shared planner boundary")

    def clear_planner_boundaries(self):
        if not self.planner_boundaries:
            self.statusBar().showMessage("No shared planner boundaries to clear")
            return
        answer = QMessageBox.question(
            self,
            "Clear planner boundaries",
            "Delete all rectangular and polygon planner boundaries shared by every IFC floor?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.planner_boundaries.clear()
        self._clear_rssi_results()
        self.draw_floor()
        self.statusBar().showMessage("Cleared all shared planner boundaries")

    def nearest_ifc_connection_point(
        self,
        scene_pos: QPointF,
        straight_from: Optional[QPointF] = None,
    ) -> Tuple[QPointF, bool]:
        """Snap to existing IFC/user geometry, optionally on a straight axis.

        When ``straight_from`` is supplied (the Shift-key drawing mode), only
        intersections on the dominant horizontal or vertical axis through the
        first point are accepted. This keeps the new wall exactly straight
        while retaining the requirement that its second endpoint connects to
        existing IFC, inferred-space or manually drawn RF geometry.
        """
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
        objects = list(self.floor.walls) + list(self.floor.spaces)

        if straight_from is not None:
            constrained = self.axis_constrained_point(straight_from, scene_pos)
            horizontal = abs(sx - float(straight_from.x())) >= abs(sy - float(straight_from.y()))
            bounds = [obj.polygon.bounds for obj in objects if getattr(obj, "polygon", None) is not None and not obj.polygon.is_empty]
            if bounds:
                minx = min(float(item[0]) for item in bounds)
                miny = min(float(item[1]) for item in bounds)
                maxx = max(float(item[2]) for item in bounds)
                maxy = max(float(item[3]) for item in bounds)
            else:
                minx = min(float(straight_from.x()), sx) - 10.0
                miny = min(float(straight_from.y()), sy) - 10.0
                maxx = max(float(straight_from.x()), sx) + 10.0
                maxy = max(float(straight_from.y()), sy) + 10.0
            span = max(maxx - minx, maxy - miny, 10.0)
            margin = max(10.0, span * 0.25)
            if horizontal:
                axis_line = LineString([
                    (minx - margin, float(straight_from.y())),
                    (maxx + margin, float(straight_from.y())),
                ])
            else:
                axis_line = LineString([
                    (float(straight_from.x()), miny - margin),
                    (float(straight_from.x()), maxy + margin),
                ])

            best = constrained
            best_distance = snap_radius

            def consider_geometry(geometry):
                nonlocal best, best_distance
                if geometry is None or geometry.is_empty:
                    return
                geometry_type = str(getattr(geometry, "geom_type", ""))
                if geometry_type == "Point":
                    candidates = [geometry]
                elif geometry_type in {"LineString", "LinearRing"}:
                    candidates = [geometry.interpolate(geometry.project(source))]
                elif hasattr(geometry, "geoms"):
                    for part in geometry.geoms:
                        consider_geometry(part)
                    return
                else:
                    candidates = []
                for candidate in candidates:
                    distance = float(candidate.distance(source))
                    if distance <= best_distance:
                        best = QPointF(float(candidate.x), float(candidate.y))
                        best_distance = distance

            for obj in objects:
                try:
                    consider_geometry(obj.polygon.boundary.intersection(axis_line))
                except Exception:
                    continue
            return best, best_distance <= snap_radius

        best = QPointF(sx, sy)
        best_distance = snap_radius
        for obj in objects:
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

    def capture_user_wall_point(self, scene_pos: QPointF, shift_constrain: bool = False):
        if not self.floor:
            return
        straight_from = self._wall_draw_start if shift_constrain and self._wall_draw_start is not None else None
        snap, snapped = self.nearest_ifc_connection_point(scene_pos, straight_from=straight_from)
        if not snapped:
            if straight_from is not None:
                self.statusBar().showMessage("No existing wall or space boundary intersects the Shift-constrained horizontal/vertical line near the cursor.")
            else:
                self.statusBar().showMessage("RF wall endpoints must connect to an existing IFC wall or space boundary. Click closer to an edge.")
            return
        if self._wall_draw_start is None:
            self._wall_draw_start = snap
            self.show_user_wall_preview(snap)
            self.statusBar().showMessage("First wall endpoint captured. Click another existing IFC element/edge; hold Shift to constrain the wall horizontal or vertical.")
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
        self._invalidate_interactive_preview_requests()
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
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.populate_wall_table(); self.draw_floor()
        self.statusBar().showMessage(f"Updated RF attenuation for {wall.name or wall.guid}")

    @staticmethod
    def _normalise_attenuation_type(value: object) -> str:
        return " ".join(str(value or "").strip().casefold().split()) or "unknown"

    def _attenuation_group_key(self, obj) -> Tuple[str, str, str, str]:
        """Stable key that condenses repeated IFC instances into one type row."""
        if isinstance(obj, Wall2D):
            category = "wall"
            ifc_class = "IfcWall"
            if obj.is_user_created:
                type_name = obj.rf_type_override or obj.type_name or obj.material or "User RF wall"
            else:
                # Keep grouping stable after an RF override by using the authored
                # IFC type/material rather than the editable RF type text.
                type_name = obj.type_name or obj.material or obj.name or "Wall"
            material = obj.material or ""
        else:
            category = str(getattr(obj, "rf_category", "other") or "other")
            ifc_class = str(getattr(obj, "ifc_class", "IfcElement") or "IfcElement")
            type_name = getattr(obj, "type_name", "") or getattr(obj, "material", "") or getattr(obj, "name", "") or ifc_class
            material = getattr(obj, "material", "") or ""
        return (
            self._normalise_attenuation_type(category),
            self._normalise_attenuation_type(ifc_class),
            self._normalise_attenuation_type(type_name),
            self._normalise_attenuation_type(material),
        )

    def _all_attenuation_objects(self, current_floor_only: bool = False):
        floors = [self.floor] if current_floor_only and self.floor is not None else list(self.floors.values())
        for floor in floors:
            if floor is None:
                continue
            for wall in floor.walls:
                yield wall
            for element in getattr(floor, "elements", []):
                yield element

    def _objects_for_attenuation_group(self, key: Tuple[str, ...], current_floor_only: bool = False) -> List[object]:
        wanted = tuple(str(value) for value in key)
        return [
            obj for obj in self._all_attenuation_objects(current_floor_only=current_floor_only)
            if self._attenuation_group_key(obj) == wanted
        ]

    @staticmethod
    def _attenuation_object_identity(obj) -> Tuple[str, str]:
        return (str(getattr(obj, "source_file", "")), str(getattr(obj, "guid", "")))

    def _attenuation_groups(self) -> List[Dict[str, object]]:
        grouped: Dict[Tuple[str, str, str, str], List[object]] = {}
        for obj in self._all_attenuation_objects(current_floor_only=False):
            grouped.setdefault(self._attenuation_group_key(obj), []).append(obj)
        current_name = self.floor.name if self.floor else ""
        result: List[Dict[str, object]] = []
        for key, objects in grouped.items():
            representative = objects[0]
            physical_ids = {self._attenuation_object_identity(value) for value in objects}
            current_ids = {
                self._attenuation_object_identity(value)
                for value in objects if str(getattr(value, "floor", "")) == current_name
            }
            if isinstance(representative, Wall2D):
                category = "Wall"
                display_type = representative.rf_type_override or representative.type_name or representative.material or "Wall"
                material = representative.material or ""
            else:
                category = str(representative.rf_category or representative.ifc_class or "Other").replace("_", " ").title()
                display_type = representative.rf_type_override or representative.type_name or representative.material or representative.ifc_class
                material = representative.material or ""
            result.append({
                "key": key,
                "category": category,
                "display_type": display_type,
                "material": material,
                "count": len(physical_ids),
                "current_floor_count": len(current_ids),
                "attenuation": {float(band): representative.attenuation_db_for_frequency(float(band)) for band in self._frequency_bands()},
                "sources": ", ".join(sorted({str(getattr(value, "source_file", "")) for value in objects if getattr(value, "source_file", "")})),
            })
        return sorted(result, key=lambda value: (
            str(value.get("category", "")), str(value.get("display_type", "")), str(value.get("material", ""))
        ))

    def _combined_attenuation_presets(self) -> Dict[str, Dict[float, float]]:
        presets: Dict[str, Dict[float, float]] = {}
        sources = (
            ("wall", self.heatmap_settings.default_wall_attenuation_by_material_db),
            ("door", self.heatmap_settings.default_door_attenuation_by_material_db),
            ("window", self.heatmap_settings.default_window_attenuation_by_material_db),
            ("ifc", self.heatmap_settings.default_ifc_element_attenuation_by_type_db),
        )
        for prefix, profiles in sources:
            for name, profile in profiles.items():
                label = name if name == "default" and "default" not in presets else f"{prefix} / {name}"
                presets[label] = {float(key): float(value) for key, value in profile.items()}
        return presets

    def show_bulk_ifc_attenuation(self):
        groups = self._attenuation_groups()
        if not groups:
            QMessageBox.information(self, "No IFC attenuation types", "Load an IFC model before editing attenuation types.")
            return
        dialog = BulkIFCAttenuationDialog(
            self, groups, self._frequency_bands(), self._combined_attenuation_presets(),
            self.floor.name if self.floor else "",
        )
        if dialog.exec() != QDialog.Accepted:
            return
        current_floor_only, changes = dialog.values()
        changed_objects = 0
        changed_types = 0
        for change in changes:
            objects = self._objects_for_attenuation_group(tuple(change.get("key", ())), current_floor_only)
            if not objects:
                continue
            rf_type = str(change.get("rf_type", "")).strip()
            attenuation = {float(key): float(value) for key, value in dict(change.get("attenuation", {})).items()}
            for obj in objects:
                obj.rf_type_override = rf_type
                obj.attenuation_by_band_db.update(attenuation)
                obj.rf_customised = True
                changed_objects += 1
            changed_types += 1
        if not changed_types:
            self.statusBar().showMessage("No IFC attenuation type rows were selected")
            return
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.populate_wall_table()
        self.draw_floor()
        scope = self.floor.name if current_floor_only and self.floor else "all floors"
        self.statusBar().showMessage(
            f"Updated {changed_types} IFC attenuation type(s), affecting {changed_objects} visible/project instances on {scope}"
        )

    def edit_ifc_element_rf_properties(self, element: IFCElement2D):
        if element.rf_category == "door":
            profiles = self.heatmap_settings.default_door_attenuation_by_material_db
        elif element.rf_category == "window":
            profiles = self.heatmap_settings.default_window_attenuation_by_material_db
        else:
            profiles = self.heatmap_settings.default_ifc_element_attenuation_by_type_db
        dialog = IFCElementAttenuationDialog(self, element, self._frequency_bands(), profiles)
        if dialog.exec() != QDialog.Accepted:
            return
        rf_type, attenuation = dialog.values()
        for instance in [
            candidate for floor in self.floors.values() for candidate in getattr(floor, "elements", [])
            if candidate.guid == element.guid and candidate.source_file == element.source_file
        ] or [element]:
            instance.rf_type_override = rf_type
            instance.attenuation_by_band_db.update({float(key): float(value) for key, value in attenuation.items()})
            instance.rf_customised = True
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.populate_wall_table()
        self.draw_floor()
        self.statusBar().showMessage(f"Updated RF attenuation for {element.name or element.guid}")

    def reset_ifc_element_rf_properties(self, element: IFCElement2D):
        for instance in [
            candidate for floor in self.floors.values() for candidate in getattr(floor, "elements", [])
            if candidate.guid == element.guid and candidate.source_file == element.source_file
        ] or [element]:
            instance.rf_customised = False
            instance.rf_type_override = ""
            instance.attenuation_by_band_db = self._profile_for_ifc_element_from_settings(instance)
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.populate_wall_table()
        self.draw_floor()
        self.statusBar().showMessage(f"Reset RF attenuation for {element.name or element.guid}")

    @staticmethod
    def _ensure_ap_radio_list(ap: AccessPoint):
        if not ap.radios:
            ap.radios = [APRadio(
                name="Radio-1", frequency_mhz=float(ap.frequency_mhz), tx_power_dbm=float(ap.tx_power_dbm),
                antenna_pattern=ap.antenna_pattern, enabled=True,
            )]

    def _bulk_target_radios(self, ap: AccessPoint, mode: str, target_frequency_mhz: float) -> List[APRadio]:
        self._ensure_ap_radio_list(ap)
        if mode == "first":
            return ap.radios[:1]
        if mode == "enabled":
            return [radio for radio in ap.radios if radio.enabled]
        if mode == "frequency":
            return [min(ap.radios, key=lambda radio: abs(float(radio.frequency_mhz) - float(target_frequency_mhz)))]
        return list(ap.radios)

    def show_bulk_ap_parameters(self):
        selected = self.selected_access_points()
        current_floor_aps = [ap for ap in self.aps if self.floor is not None and ap.floor == self.floor.name]
        dialog = BulkAccessPointDialog(
            self, len(selected), len(current_floor_aps), len(self.aps), sorted(self.antenna_patterns.keys())
        )
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        changes = dict(values.get("changes", {}))
        if not changes:
            self.statusBar().showMessage("No AP parameter fields were selected")
            return
        scope = values.get("scope")
        if scope == "selected":
            targets = selected
        elif scope == "all":
            targets = list(self.aps)
        else:
            targets = current_floor_aps
        if not targets:
            QMessageBox.information(self, "No access points", "The selected bulk-edit scope contains no access points.")
            return

        physical_keys = {"ap_type", "mount_height_m", "rx_height_m", "azimuth_deg", "downtilt_deg", "path_loss_exponent", "max_clients"}
        radio_keys = {
            "radio_enabled", "tx_power_dbm", "antenna_gain_dbi", "frequency_mhz", "antenna_pattern",
            "channel", "channel_width_mhz", "spectrum_occupancy_percent", "cutoff_radius_m",
        }
        radio_mode = str(values.get("radio_target", "all"))
        target_frequency = float(values.get("target_frequency_mhz", 5000.0))
        for ap in targets:
            if "ap_type" in changes:
                ap.ap_type = str(changes["ap_type"])
            if "radio_profile" in changes:
                ap.radio_profile = str(changes["radio_profile"])
                ap.radios = self._radios_for_profile(ap.radio_profile, ap.ap_type)
            for key in physical_keys:
                if key in changes:
                    setattr(ap, key, changes[key])
            selected_radios = self._bulk_target_radios(ap, radio_mode, target_frequency)
            for radio in selected_radios:
                if "radio_enabled" in changes:
                    radio.enabled = bool(changes["radio_enabled"])
                for key in radio_keys - {"radio_enabled"}:
                    if key in changes:
                        setattr(radio, key, changes[key])
            if any(key in changes for key in radio_keys) and "radio_profile" not in changes:
                ap.radio_profile = "Custom"
            if ap.radios:
                ap.tx_power_dbm = float(ap.radios[0].tx_power_dbm)
                ap.frequency_mhz = float(ap.radios[0].frequency_mhz)
                ap.antenna_pattern = ap.radios[0].antenna_pattern

        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self._rssi_result_stale = False
        self._refresh_rssi_frequency_dropdown()
        self._pending_ap_selection_ids = {id(ap) for ap in targets if ap.floor == (self.floor.name if self.floor else "")}
        self.populate_ap_table()
        self.draw_floor()
        self.statusBar().showMessage(f"Applied {len(changes)} parameter field(s) to {len(targets)} access point(s)")

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
            self._invalidate_interactive_preview_requests()
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
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.populate_wall_table(); self.draw_floor()

    def delete_user_wall(self, wall: Wall2D):
        if not wall.is_user_created:
            return
        floor = self.floors.get(wall.floor)
        if floor is not None:
            floor.walls = [candidate for candidate in floor.walls if candidate is not wall]
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.populate_wall_table(); self.draw_floor()

    @staticmethod
    def _ifc_exclusion_record(kind: str, source_file: str, guid: str, floor: str = "") -> Dict[str, str]:
        return {
            "kind": str(kind),
            "source_file": str(source_file),
            "guid": str(guid),
            "floor": str(floor or ""),
        }

    def _remember_ifc_exclusion(self, kind: str, source_file: str, guid: str, floor: str = ""):
        record = self._ifc_exclusion_record(kind, source_file, guid, floor)
        key = (record["kind"], record["source_file"], record["guid"], record["floor"])
        existing = {
            (
                str(item.get("kind", "")),
                str(item.get("source_file", "")),
                str(item.get("guid", "")),
                str(item.get("floor", "")),
            )
            for item in getattr(self, "excluded_ifc_elements", [])
            if isinstance(item, dict)
        }
        if key not in existing:
            self.excluded_ifc_elements.append(record)

    @staticmethod
    def _matches_ifc_exclusion(candidate, record: Dict[str, str], kind: str) -> bool:
        if str(record.get("kind", "")) != kind:
            return False
        if str(getattr(candidate, "guid", "")) != str(record.get("guid", "")):
            return False
        if str(getattr(candidate, "source_file", "")) != str(record.get("source_file", "")):
            return False
        floor = str(record.get("floor", ""))
        return not floor or str(getattr(candidate, "floor", "")) == floor

    def _apply_ifc_exclusions(self):
        records = [item for item in getattr(self, "excluded_ifc_elements", []) if isinstance(item, dict)]
        if not records:
            return
        for floor in self.floors.values():
            floor.walls = [
                wall for wall in floor.walls
                if not any(self._matches_ifc_exclusion(wall, record, "wall") for record in records)
            ]
            floor.spaces = [
                space for space in floor.spaces
                if not any(self._matches_ifc_exclusion(space, record, "space") for record in records)
            ]
            floor.elements = [
                element for element in getattr(floor, "elements", [])
                if not any(self._matches_ifc_exclusion(element, record, "element") for record in records)
            ]

    def delete_imported_wall(self, wall: Wall2D):
        if wall.is_user_created:
            return
        answer = QMessageBox.question(
            self,
            "Remove imported IFC wall",
            f"Remove imported IFC wall '{wall.name or wall.guid}' from this RF model? The source IFC file will not be changed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._remember_ifc_exclusion("wall", wall.source_file, wall.guid)
        removed = 0
        for floor in self.floors.values():
            before = len(floor.walls)
            floor.walls = [
                candidate for candidate in floor.walls
                if not (
                    candidate.guid == wall.guid
                    and candidate.source_file == wall.source_file
                    and not candidate.is_user_created
                )
            ]
            removed += before - len(floor.walls)
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.populate_wall_table()
        self.draw_floor()
        self.statusBar().showMessage(f"Removed {removed} imported IFC wall instance{'s' if removed != 1 else ''} from the RF model")

    def delete_user_space(self, space: Space2D):
        if not space.is_user_created:
            return
        floor = self.floors.get(space.floor)
        if floor is not None:
            floor.spaces = [candidate for candidate in floor.spaces if candidate is not space]
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        self.statusBar().showMessage(f"Deleted user-created space '{space.name}'")

    def delete_imported_space(self, space: Space2D):
        if space.is_user_created or space.is_inferred:
            return
        answer = QMessageBox.question(
            self,
            "Remove imported IFC space",
            f"Remove imported IFC space '{space.name or space.guid}' from this RF model? The source IFC file will not be changed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._remember_ifc_exclusion("space", space.source_file, space.guid)
        removed = 0
        for floor in self.floors.values():
            before = len(floor.spaces)
            floor.spaces = [
                candidate for candidate in floor.spaces
                if not (
                    candidate.guid == space.guid
                    and candidate.source_file == space.source_file
                    and not candidate.is_user_created
                    and not candidate.is_inferred
                )
            ]
            removed += before - len(floor.spaces)
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        self.statusBar().showMessage(f"Removed {removed} imported IFC space instance{'s' if removed != 1 else ''} from the RF model")

    def delete_imported_element(self, element: IFCElement2D):
        answer = QMessageBox.question(
            self,
            "Remove imported IFC element",
            f"Remove imported IFC element '{element.name or element.guid}' from this RF model? The source IFC file will not be changed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._remember_ifc_exclusion("element", element.source_file, element.guid)
        removed = 0
        for floor in self.floors.values():
            before = len(getattr(floor, "elements", []))
            floor.elements = [
                candidate for candidate in getattr(floor, "elements", [])
                if not (
                    candidate.guid == element.guid
                    and candidate.source_file == element.source_file
                )
            ]
            removed += before - len(floor.elements)
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        self.statusBar().showMessage(f"Removed {removed} imported IFC element instance{'s' if removed != 1 else ''} from the RF model")

    def _space_ref(self, space: Space2D) -> Tuple[str, str, str]:
        return (str(space.source_file), str(space.guid), str(space.floor))

    def _selected_ap_planning_spaces(self) -> List[Space2D]:
        if not self.floor:
            return []
        return [
            space for space in self.floor.spaces
            if space.ap_planning_selected and space.polygon is not None and not space.polygon.is_empty
        ]

    def toggle_space_ap_planning_selection(self, space: Space2D):
        floor = self.floors.get(space.floor)
        if floor is None:
            return
        target_ref = self._space_ref(space)
        selected = not bool(space.ap_planning_selected)
        for candidate in floor.spaces:
            if self._space_ref(candidate) == target_ref:
                candidate.ap_planning_selected = selected
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        count = len(self._selected_ap_planning_spaces())
        if selected:
            self.statusBar().showMessage(
                f"Selected '{space.name}' for AP placement. Prediction will use {count} selected space{'s' if count != 1 else ''}."
            )
        else:
            self.statusBar().showMessage(
                f"Removed '{space.name}' from AP placement spaces. {count} selected space{'s' if count != 1 else ''} remain."
            )

    def show_ap_space_selection_dialog(self):
        if not self.floor:
            QMessageBox.information(self, "No floor selected", "Load an IFC and select a floor before choosing AP placement spaces.")
            return
        spaces = [
            space for space in self.floor.spaces
            if space.polygon is not None and not space.polygon.is_empty
        ]
        if not spaces:
            QMessageBox.information(
                self,
                "No spaces available",
                "The selected floor has no IFC, inferred, or manually drawn spaces to choose from.",
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose AP placement spaces")
        dialog.setModal(True)
        dialog.resize(520, 420)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "Checked spaces are used as the AP prediction area on the selected floor. "
            "If none are checked, the planner uses the normal planning-area settings."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        box_widget = QWidget(scroll)
        box_layout = QVBoxLayout(box_widget)
        checks: List[Tuple[Space2D, QCheckBox]] = []
        for space in spaces:
            kind = "inferred" if space.is_inferred else ("user" if space.is_user_created else "IFC")
            check = QCheckBox(f"{space.name or space.guid[:8]} [{kind}] - {float(space.polygon.area):,.1f} m2")
            check.setChecked(bool(space.ap_planning_selected))
            box_layout.addWidget(check)
            checks.append((space, check))
        box_layout.addStretch(1)
        scroll.setWidget(box_widget)
        layout.addWidget(scroll, 1)

        controls = QHBoxLayout()
        select_all = QPushButton("Select all")
        clear_all = QPushButton("Clear")
        controls.addWidget(select_all)
        controls.addWidget(clear_all)
        controls.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        controls.addWidget(buttons)
        layout.addLayout(controls)
        select_all.clicked.connect(lambda: [check.setChecked(True) for _, check in checks])
        clear_all.clicked.connect(lambda: [check.setChecked(False) for _, check in checks])
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QDialog.Accepted:
            return
        for space, check in checks:
            space.ap_planning_selected = bool(check.isChecked())
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.draw_floor()
        count = len(self._selected_ap_planning_spaces())
        self.statusBar().showMessage(
            f"Configured {count} AP placement space{'s' if count != 1 else ''} for '{self.floor.name}'."
        )

    # ----------------------------- Predictive AP planning -----------------------------

    def _rssi_calculation_boundary(self):
        if not bool(self.heatmap_settings.ignore_results_outside_planner_boundaries):
            return None
        return self._planner_boundary_area()

    def _clear_rssi_results(self):
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}

    def toggle_boundary_result_filter(self, enabled: bool):
        self.heatmap_settings.ignore_results_outside_planner_boundaries = bool(enabled)
        self._clear_rssi_results()
        self._update_rssi_legend()
        self.draw_floor()
        if enabled and not self.planner_boundaries:
            self.statusBar().showMessage(
                "Boundary RSSI filter enabled; create or accept a shared planner boundary for it to take effect."
            )
        elif enabled:
            self.statusBar().showMessage(
                "Boundary RSSI filter enabled: samples outside the shared planner-boundary union will be ignored."
            )
        else:
            self.statusBar().showMessage("Boundary RSSI filter disabled")

    def _save_performance_defaults(self, path: Path):
        """Update only performance/propagation keys while preserving the rest of the JSON file."""
        path = Path(path)
        data: Dict[str, object] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                data = loaded
        data.update({
            "enable_ifc_multiprocessing": bool(self.heatmap_settings.enable_ifc_multiprocessing),
            "max_ifc_loader_processes": int(self.heatmap_settings.max_ifc_loader_processes),
            "max_parallel_huge_ifc_processes": int(self.heatmap_settings.max_parallel_huge_ifc_processes),
            "enable_rf_multiprocessing": bool(self.heatmap_settings.enable_rf_multiprocessing),
            "max_rf_worker_processes": int(self.heatmap_settings.max_rf_worker_processes),
            "rf_multiprocessing_min_points": int(self.heatmap_settings.rf_multiprocessing_min_points),
            "rf_tile_rows": int(self.heatmap_settings.rf_tile_rows),
            "rf_tiles_per_worker": int(self.heatmap_settings.rf_tiles_per_worker),
            "reuse_rf_process_pool": bool(self.heatmap_settings.reuse_rf_process_pool),
            "rf_worker_index_cache_entries": int(self.heatmap_settings.rf_worker_index_cache_entries),
            "contour_interpolation_factor": int(self.heatmap_settings.contour_interpolation_factor),
            "ignore_results_outside_planner_boundaries": bool(self.heatmap_settings.ignore_results_outside_planner_boundaries),
        })
        data["rf_performance"] = self.heatmap_settings.performance_model_dict()
        existing_propagation = data.get("propagation_model", {})
        if not isinstance(existing_propagation, dict):
            existing_propagation = {}
        existing_propagation.update(self.heatmap_settings.propagation_model_dict())
        data["propagation_model"] = existing_propagation
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        self.heatmap_settings_path = path

    def show_performance_settings(self):
        before = self.heatmap_settings.performance_model_dict()
        before_propagation = self.heatmap_settings.propagation_model_dict()
        dialog = RFPerformanceSettingsDialog(
            self,
            self.heatmap_settings,
            has_planner_boundaries=bool(self.planner_boundaries),
            settings_path=self.heatmap_settings_path,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        dialog.apply_to(self.heatmap_settings)
        after = self.heatmap_settings.performance_model_dict()
        after_propagation = self.heatmap_settings.propagation_model_dict()

        worker_keys = {
            "enable_rf_multiprocessing", "max_rf_worker_processes", "reuse_rf_process_pool",
            "rf_worker_index_cache_entries", "rf_tile_rows", "rf_tiles_per_worker",
        }
        if any(before.get(key) != after.get(key) for key in worker_keys):
            _shutdown_rf_process_executor(wait=False)
        _RF_AP_FIELD_CACHE.clear()
        self.boundary_result_filter_action.blockSignals(True)
        self.boundary_result_filter_action.setChecked(
            bool(self.heatmap_settings.ignore_results_outside_planner_boundaries)
        )
        self.boundary_result_filter_action.blockSignals(False)
        self._clear_rssi_results()
        self._update_rssi_legend()
        self.draw_floor()

        saved_text = ""
        if dialog.save_global_defaults.isChecked():
            try:
                self._save_performance_defaults(dialog.global_settings_path)
                saved_text = f" Global defaults saved to {dialog.global_settings_path.name}."
            except Exception as exc:
                QMessageBox.warning(self, "Performance defaults not saved", str(exc))
        changed = before != after or before_propagation != after_propagation
        workers = int(self.heatmap_settings.max_rf_worker_processes)
        worker_text = "automatic" if workers == 0 else str(workers)
        boundary_note = ""
        if self.heatmap_settings.ignore_results_outside_planner_boundaries and not self.planner_boundaries:
            boundary_note = " Boundary limiting is enabled but will not take effect until a planner boundary exists."
        self.statusBar().showMessage(
            f"Performance settings {'updated' if changed else 'unchanged'}: "
            f"{self.heatmap_settings.rf_calculation_profile} profile, {worker_text} RF workers, "
            f"{self.heatmap_settings.heatmap_render_mode} rendering.{boundary_note}{saved_text}"
        )

    def show_propagation_settings(self):
        dialog = PropagationSettingsDialog(self, self.heatmap_settings)
        if dialog.exec() != QDialog.Accepted:
            return
        dialog.apply_to(self.heatmap_settings)
        self.boundary_result_filter_action.blockSignals(True)
        self.boundary_result_filter_action.setChecked(
            bool(self.heatmap_settings.ignore_results_outside_planner_boundaries)
        )
        self.boundary_result_filter_action.blockSignals(False)
        self._clear_rssi_results()
        self._update_rssi_legend()
        self.draw_floor()
        order = int(self.heatmap_settings.max_reflection_order)
        self.statusBar().showMessage(
            f"RF propagation model updated: reflection order {order}, "
            f"diffraction {'on' if self.heatmap_settings.enable_corner_diffraction else 'off'}, "
            f"fading {'on' if self.heatmap_settings.enable_small_scale_fading else 'off'}"
        )

    def show_auto_planner_settings(self):
        dlg = AutoPlannerSettingsDialog(self, self.auto_planner_settings, list(self.antenna_patterns.keys()))
        if dlg.exec() == QDialog.Accepted:
            self.auto_planner_settings = dlg._validated_settings
            self.heatmap_settings.auto_planner_settings = self.auto_planner_settings.to_dict()
            self._apply_frequency_settings_to_model(replace_existing=False)
            self._refresh_rssi_frequency_dropdown()
            self.populate_wall_table()
            self.statusBar().showMessage("Predictive AP planner settings updated")

    def _planner_boundary_area(self):
        polygons = [
            boundary.polygon for boundary in self.planner_boundaries
            if boundary.polygon is not None and not boundary.polygon.is_empty
        ]
        if not polygons:
            return None
        try:
            area = unary_union(polygons)
            if not area.is_valid:
                area = area.buffer(0)
            return None if area.is_empty else area
        except Exception:
            return None

    def _planner_floor_area(self):
        """Return the area sampled by the predictive planner.

        Shared user boundaries apply to every IFC floor. Whenever at least
        one exists, their union is a hard clipping mask: no candidate or sample
        point can be generated outside it. In automatic mode, IFC spaces are
        preferred; if spaces are absent, the shared boundaries become the
        planning area; wall-derived geometry is used only as the final fallback.
        """
        if not self.floor:
            return None
        settings = self.auto_planner_settings
        mode = str(getattr(settings, "planning_area_mode", "auto") or "auto").lower()
        boundary_area = self._planner_boundary_area()
        boundary_count = len(self.planner_boundaries)
        boundary_description = f"{boundary_count} shared planner {'boundary' if boundary_count == 1 else 'boundaries'}"
        use_inferred_spaces = bool(getattr(settings, "use_inferred_spaces", True))
        selected_space_objects = [
            space for space in self._selected_ap_planning_spaces()
            if use_inferred_spaces or not space.is_inferred
        ]
        selected_spaces = [
            space.polygon for space in selected_space_objects
            if space.polygon is not None and not space.polygon.is_empty
        ]
        space_objects = [
            space for space in self.floor.spaces
            if (use_inferred_spaces or not space.is_inferred)
            and space.polygon is not None
            and not space.polygon.is_empty
        ]
        spaces = [space.polygon for space in space_objects]

        def constrained(area, source_label: str):
            if area is None or area.is_empty:
                return None
            if boundary_area is not None:
                try:
                    area = area.intersection(boundary_area)
                except Exception:
                    return None
                if not area.is_valid:
                    area = area.buffer(0)
                if area.is_empty:
                    self._planner_area_source_label = (
                        f"{source_label}; no overlap with {boundary_description}"
                    )
                    return None
                source_label += f", clipped by {boundary_description}"
            self._planner_area_source_label = source_label
            return area

        if selected_spaces:
            try:
                selected_area = unary_union(selected_spaces)
                if not selected_area.is_valid:
                    selected_area = selected_area.buffer(0)
                if not selected_area.is_empty:
                    inferred_selected = sum(1 for space in selected_space_objects if space.is_inferred)
                    source_label = f"{len(selected_spaces)} selected AP placement space{'s' if len(selected_spaces) != 1 else ''}"
                    if inferred_selected:
                        source_label += f" including {inferred_selected} inferred"
                    return constrained(selected_area, source_label)
            except Exception:
                self._planner_area_source_label = "selected AP placement spaces could not be combined"
                return None

        if mode == "boundaries":
            if boundary_area is None:
                self._planner_area_source_label = "shared planner boundaries only (none drawn)"
                return None
            self._planner_area_source_label = boundary_description
            return boundary_area

        if mode in {"auto", "spaces"} and spaces:
            try:
                area = unary_union(spaces)
                if not area.is_valid:
                    area = area.buffer(0)
                if not area.is_empty:
                    inferred_count = sum(1 for space in space_objects if space.is_inferred)
                    manual_count = sum(1 for space in space_objects if space.is_user_created)
                    if inferred_count:
                        source_label = f"{len(spaces)} eligible space footprint(s), including {inferred_count} inferred"
                    elif manual_count:
                        source_label = f"{len(spaces)} IFC/manual space footprint(s)"
                    else:
                        source_label = f"{len(spaces)} IFC space footprint(s)"
                    return constrained(area, source_label)
            except Exception:
                if mode == "spaces":
                    return None
        if mode == "spaces":
            excluded_note = "; inferred spaces are disabled" if not use_inferred_spaces else ""
            self._planner_area_source_label = f"eligible spaces only (none available{excluded_note})"
            return None

        # When spaces are unavailable, user boundaries are an explicit and safer
        # planning extent than an automatically inferred convex/concave hull.
        if mode == "auto" and boundary_area is not None:
            self._planner_area_source_label = boundary_description
            return boundary_area

        wall_polygons = [
            wall.polygon for wall in self.floor.walls
            if wall.polygon is not None and not wall.polygon.is_empty
        ]
        if not wall_polygons:
            self._planner_area_source_label = "no IFC spaces, shared planner boundaries, or wall geometry"
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
            return constrained(
                area, f"footprint inferred from {len(wall_polygons)} IFC/RF wall polygon(s)"
            )
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

    def _planner_ap_rssi_values(self, ap: AccessPoint, requirements: List[PlannerRadioRequirement], samples: List[Tuple[float, float]], wall_tree, walls: List[Wall2D]) -> List[np.ndarray]:
        """Return the strongest RSSI from one physical AP for each required band.

        Multiple radios in the same band still count as one AP for handover. This
        prevents two radios fitted to one chassis from being mistaken for two
        independently located handover cells.
        """
        disconnected = float(self.heatmap_settings.disconnected_rssi_dbm)
        values_by_radio: List[np.ndarray] = []
        for requirement in requirements:
            matching = [
                radio for radio in ap.active_radios()
                if abs(float(radio.frequency_mhz) - float(requirement.frequency_mhz)) < 1.0
            ]
            values = np.full(len(samples), disconnected, dtype=np.float32)
            for radio in matching:
                radio_values = np.fromiter(
                    (self._planner_rssi(x, y, ap, radio, wall_tree, walls) for x, y in samples),
                    dtype=np.float32, count=len(samples),
                )
                np.maximum(values, radio_values, out=values)
            values_by_radio.append(values)
        return values_by_radio

    def _planner_ap_masks(self, ap: AccessPoint, requirements: List[PlannerRadioRequirement], samples: List[Tuple[float, float]], wall_tree, walls: List[Wall2D]) -> List[np.ndarray]:
        values_by_radio = self._planner_ap_rssi_values(ap, requirements, samples, wall_tree, walls)
        return [
            values >= float(requirement.minimum_rssi_dbm)
            for requirement, values in zip(requirements, values_by_radio)
        ]

    @staticmethod
    def _planner_insert_rssi(
        strongest: np.ndarray, second_strongest: np.ndarray, candidate: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Insert one AP's RSSI into the strongest/second-strongest arrays."""
        candidate = np.asarray(candidate, dtype=np.float32)
        stronger = candidate >= strongest
        new_strongest = np.where(stronger, candidate, strongest)
        new_second = np.where(stronger, strongest, np.maximum(second_strongest, candidate))
        return new_strongest.astype(np.float32, copy=False), new_second.astype(np.float32, copy=False)

    @staticmethod
    def _planner_rssi_deficit(
        values_by_radio: List[np.ndarray], requirements: List[PlannerRadioRequirement],
        mode: str, threshold_margin_db: float = 0.0,
    ) -> np.ndarray:
        """Return per-sample dB shortfall against the configured RSSI targets."""
        if not values_by_radio:
            return np.zeros(0, dtype=np.float32)
        deficits = np.vstack([
            np.maximum(
                0.0,
                float(requirement.minimum_rssi_dbm) - float(threshold_margin_db) - values,
            )
            for requirement, values in zip(requirements, values_by_radio)
        ]).astype(np.float32, copy=False)
        if str(mode).lower() == "any":
            return np.min(deficits, axis=0)
        return np.sum(deficits, axis=0)

    def _planner_top_rssi_masks(
        self, strongest_by_radio: List[np.ndarray], second_by_radio: List[np.ndarray],
        requirements: List[PlannerRadioRequirement], settings: AutoPlannerSettings, sample_count: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        coverage_by_radio = [
            strongest >= float(requirement.minimum_rssi_dbm)
            for requirement, strongest in zip(requirements, strongest_by_radio)
        ]
        secondary_by_radio = [
            second >= float(requirement.minimum_rssi_dbm) - float(settings.handover_margin_db)
            for requirement, second in zip(requirements, second_by_radio)
        ]
        overall = self._combine_frequency_masks(coverage_by_radio, settings.coverage_mode, sample_count)
        handover = self._combine_frequency_masks(secondary_by_radio, settings.coverage_mode, sample_count)
        # Handover is meaningful only where the primary service requirement is met.
        handover = handover & overall
        return coverage_by_radio, secondary_by_radio, overall, handover

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

    def _assign_planner_channels(
        self, new_aps: List[AccessPoint], requirements: List[PlannerRadioRequirement],
        samples: Optional[List[Tuple[float, float]]] = None, wall_tree=None,
        walls: Optional[List[Wall2D]] = None,
    ):
        """Assign channels, strongly avoiding co-channel reuse in handover overlap.

        Handover coverage deliberately counts distinct APs irrespective of channel.
        This assignment pass then prefers non-overlapping channels between APs
        whose usable RF cells overlap, allowing clients to roam between channels.
        """
        if not new_aps:
            return
        walls = walls or []
        new_ids = {id(ap) for ap in new_aps}
        target_floor = new_aps[0].floor
        assigned = [
            ap for ap in self.aps
            if id(ap) not in new_ids and ap.floor == target_floor
        ]
        overlap_mask_cache: Dict[Tuple[int, int], np.ndarray] = {}

        def overlap_mask(ap: AccessPoint, req_index: int) -> Optional[np.ndarray]:
            if not samples:
                return None
            key = (id(ap), req_index)
            cached = overlap_mask_cache.get(key)
            if cached is not None:
                return cached
            requirement = requirements[req_index]
            values = self._planner_ap_rssi_values(
                ap, [requirement], samples, wall_tree, walls
            )[0]
            threshold = float(requirement.minimum_rssi_dbm) - float(self.auto_planner_settings.handover_margin_db)
            cached = values >= threshold
            overlap_mask_cache[key] = cached
            return cached

        for ap in new_aps:
            for req_index, (req, radio) in enumerate(zip(requirements, ap.radios)):
                channels = list(req.channels) or [""]
                best_channel = channels[0]
                best_cost = float("inf")
                radius = req.cutoff_radius_m or float(
                    self.heatmap_settings.ap_cutoff_radius_by_frequency_m.get(req.frequency_mhz, 35.0)
                )
                candidate_mask = overlap_mask(ap, req_index)
                for channel in channels:
                    cost = 0.0
                    for other in assigned:
                        distance = math.hypot(ap.x - other.x, ap.y - other.y)
                        if distance > max(radius * 2.0, 15.0):
                            continue
                        for other_radio in other.active_radios():
                            if abs(other_radio.frequency_mhz - req.frequency_mhz) >= 1.0 or not str(other_radio.channel):
                                continue
                            spectral_overlap = self._channel_overlap_fraction(
                                req.frequency_mhz, str(channel), req.channel_width_mhz,
                                str(other_radio.channel), float(other_radio.channel_width_mhz),
                            )
                            if spectral_overlap <= 0.0:
                                continue
                            occupancy_weight = 1.0 + req.spectrum_occupancy_percent / 100.0
                            spatial_weight = 1.0 / max(distance, 1.0)
                            if (
                                self.auto_planner_settings.prefer_non_overlapping_handover_channels
                                and candidate_mask is not None
                            ):
                                other_mask = overlap_mask(other, req_index)
                                if other_mask is not None:
                                    shared = int(np.count_nonzero(candidate_mask & other_mask))
                                    if shared > 0:
                                        union = int(np.count_nonzero(candidate_mask | other_mask))
                                        shared_fraction = shared / max(1, union)
                                        # Shared usable cells dominate simple distance: these
                                        # are exactly the cells in which a client may roam.
                                        spatial_weight += 20.0 * shared_fraction
                            cost += spectral_overlap * occupancy_weight * spatial_weight
                    if cost < best_cost:
                        best_cost = cost
                        best_channel = channel
                radio.channel = str(best_channel)
                overlap_mask_cache.pop((id(ap), req_index), None)
            assigned.append(ap)

    def run_auto_planner(self):
        self._invalidate_interactive_preview_requests()
        self._rssi_result_stale = False
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
                inferred_hint = (
                    " Enable 'Use inferred simulator spaces as predictive AP planning locations' if inferred spaces should be included."
                    if not bool(getattr(settings, "use_inferred_spaces", True))
                    else ""
                )
                message = (
                    "This floor contains no usable eligible space geometry. Change Planning area source to "
                    "Automatic, draw shared planner boundaries, infer the footprint from walls, or create/select spaces."
                    + inferred_hint
                )
            elif settings.planning_area_mode == "boundaries":
                message = (
                    "No shared planner boundaries have been drawn. Use 'Draw rectangular boundary' or 'Draw polygon boundary' "
                    "and define the permitted area; the boundaries will apply to every IFC floor."
                )
            else:
                message = (
                    "No usable planning area was found. Draw one or more shared rectangular or polygon planner boundaries "
                    "or provide IFC space/wall geometry."
                )
            QMessageBox.warning(self, "No plannable area", message)
            return
        walls = list(self.floor.walls)
        try:
            from shapely.strtree import STRtree
            wall_tree = STRtree([wall.polygon for wall in walls]) if walls else None
        except Exception:
            wall_tree = None

        samples = self._planner_grid_points(area, settings.sample_spacing_m, wall_tree, walls, 2500)
        candidate_location_limit = min(10_000, max(450, int(settings.maximum_aps) * 2))
        candidates = self._planner_grid_points(
            area, settings.candidate_spacing_m, wall_tree, walls, candidate_location_limit
        )
        # Room representative points supplement the global grid, but boundaries
        # remain a hard limit even when IfcSpace geometry is present.
        for space in self.floor.spaces:
            if space.is_inferred and not bool(getattr(settings, "use_inferred_spaces", True)):
                continue
            try:
                point = space.polygon.representative_point()
                candidate = (float(point.x), float(point.y))
                try:
                    inside_area = area.covers(point)
                except Exception:
                    inside_area = area.contains(point)
                if (
                    inside_area
                    and candidate not in candidates
                    and not self._planner_point_is_blocked(point, wall_tree, walls)
                ):
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

        progress = QProgressDialog("Building RSSI-driven AP coverage...", "Cancel", 0, max(1, settings.maximum_aps), self)
        progress.setWindowTitle("Predictive AP planner")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        disconnected = float(self.heatmap_settings.disconnected_rssi_dbm)
        strongest_by_radio = [np.full(len(samples), disconnected, dtype=np.float32) for _ in requirements]
        second_by_radio = [np.full(len(samples), disconnected, dtype=np.float32) for _ in requirements]
        for ap in existing:
            ap_values = self._planner_ap_rssi_values(ap, requirements, samples, wall_tree, walls)
            updated = [
                self._planner_insert_rssi(strongest, second, values)
                for strongest, second, values in zip(strongest_by_radio, second_by_radio, ap_values)
            ]
            strongest_by_radio = [pair[0] for pair in updated]
            second_by_radio = [pair[1] for pair in updated]

        try:
            def make_candidate_ap(x: float, y: float, azimuth: float) -> AccessPoint:
                radios = [APRadio(
                    name=req.name, frequency_mhz=req.frequency_mhz, tx_power_dbm=req.tx_power_dbm,
                    antenna_pattern=req.antenna_pattern, enabled=True, cutoff_radius_m=req.cutoff_radius_m,
                    antenna_gain_dbi=req.antenna_gain_dbi, channel="", channel_width_mhz=req.channel_width_mhz,
                    spectrum_occupancy_percent=req.spectrum_occupancy_percent,
                ) for req in requirements]
                return AccessPoint(
                    name="candidate", x=x, y=y, floor=self.floor.name, radios=radios,
                    ap_type="Directional AP" if directional else "Ceiling AP", radio_profile="Predictive planner",
                    path_loss_exponent=float(self.ple.value()), azimuth_deg=azimuth,
                    mount_height_m=float(self.mount_height.value()), rx_height_m=float(self.rx_height.value()),
                    max_clients=settings.clients_per_ap, planned=True,
                )

            candidate_cache: Dict[int, Tuple[AccessPoint, List[np.ndarray]]] = {}

            def candidate_record(index: int) -> Tuple[AccessPoint, List[np.ndarray]]:
                cached = candidate_cache.get(index)
                if cached is not None:
                    return cached
                x, y, azimuth = candidate_specs[index]
                temp_ap = make_candidate_ap(float(x), float(y), float(azimuth))
                values = [
                    candidate_values.astype(np.float16)
                    for candidate_values in self._planner_ap_rssi_values(temp_ap, requirements, samples, wall_tree, walls)
                ]
                cached = (temp_ap, values)
                candidate_cache[index] = cached
                return cached

            coverage_by_radio, secondary_by_radio, overall, handover_overall = self._planner_top_rssi_masks(
                strongest_by_radio, second_by_radio, requirements, settings, len(samples)
            )
            coverage_deficit = self._planner_rssi_deficit(
                strongest_by_radio, requirements, settings.coverage_mode
            )
            handover_deficit = self._planner_rssi_deficit(
                second_by_radio, requirements, settings.coverage_mode, settings.handover_margin_db
            )
            target_fraction = settings.target_coverage_percent / 100.0
            handover_target_fraction = settings.target_handover_percent / 100.0
            average_occupancy = sum(req.spectrum_occupancy_percent for req in requirements) / len(requirements)
            effective_clients = max(1.0, settings.clients_per_ap * max(0.05, 1.0 - average_occupancy / 100.0))
            capacity_ap_count = int(math.ceil(settings.expected_clients / effective_clients)) if settings.expected_clients > 0 else 0
            selected: List[Tuple[AccessPoint, List[np.ndarray]]] = []
            positions = [(ap.x, ap.y) for ap in existing]
            remaining = set(range(len(candidate_specs)))
            sample_points = np.asarray(samples, dtype=np.float32)

            def coverage_fraction_now() -> float:
                return float(np.count_nonzero(overall)) / max(1, len(overall))

            def handover_fraction_now() -> float:
                return float(np.count_nonzero(handover_overall)) / max(1, len(handover_overall))

            def candidate_spacing_ok(candidate_ap: AccessPoint) -> Tuple[bool, float]:
                if not positions:
                    return True, float(settings.minimum_ap_spacing_m)
                min_distance = min(
                    math.hypot(candidate_ap.x - px, candidate_ap.y - py)
                    for px, py in positions
                )
                return min_distance + 1e-9 >= settings.minimum_ap_spacing_m, min_distance

            def nearest_position_distance(points: np.ndarray) -> np.ndarray:
                if points.size == 0:
                    return np.zeros(0, dtype=np.float32)
                if not positions:
                    return np.full(len(points), float("inf"), dtype=np.float32)
                distances = np.full(len(points), np.inf, dtype=np.float32)
                for px, py in positions:
                    dx = points[:, 0] - float(px)
                    dy = points[:, 1] - float(py)
                    distances = np.minimum(distances, np.sqrt(dx * dx + dy * dy))
                return distances

            def focus_sample_indices(coverage_needed: bool, handover_needed: bool, capacity_needed: bool) -> List[int]:
                if coverage_needed:
                    metric = np.asarray(coverage_deficit, dtype=np.float32)
                elif handover_needed:
                    metric = np.asarray(handover_deficit, dtype=np.float32)
                elif capacity_needed:
                    metric = nearest_position_distance(sample_points)
                else:
                    metric = np.zeros(len(samples), dtype=np.float32)
                positive = np.flatnonzero(metric > 1e-6)
                if positive.size == 0:
                    positive = np.arange(len(samples))
                order = positive[np.argsort(metric[positive])[::-1]]
                focus = [int(i) for i in order[:12]]
                if positions and len(focus) < 18:
                    uncovered = np.flatnonzero(~overall) if coverage_needed else np.arange(len(samples))
                    if uncovered.size:
                        distances = nearest_position_distance(sample_points[uncovered])
                        far_order = uncovered[np.argsort(distances)[::-1]]
                        for idx in far_order[:12]:
                            value = int(idx)
                            if value not in focus:
                                focus.append(value)
                return focus[:18]

            def nearby_candidate_indices(focus_indices: List[int], limit: int = 600) -> List[int]:
                if not remaining:
                    return []
                if not focus_indices:
                    return list(remaining)[:limit]
                focus = sample_points[focus_indices]
                scored: List[Tuple[float, int]] = []
                for index in remaining:
                    x, y, _ = candidate_specs[index]
                    dx = focus[:, 0] - float(x)
                    dy = focus[:, 1] - float(y)
                    distance = float(np.min(np.sqrt(dx * dx + dy * dy)))
                    scored.append((distance, index))
                scored.sort(key=lambda item: item[0])
                return [index for _, index in scored[:limit]]

            def insert_candidate_state(candidate_rssi: List[np.ndarray]):
                inserted = [
                    self._planner_insert_rssi(strongest, second, values)
                    for strongest, second, values in zip(strongest_by_radio, second_by_radio, candidate_rssi)
                ]
                proposed_strongest = [pair[0] for pair in inserted]
                proposed_second = [pair[1] for pair in inserted]
                proposed_coverage_by_radio, proposed_secondary_by_radio, proposed_overall, proposed_handover = self._planner_top_rssi_masks(
                    proposed_strongest, proposed_second, requirements, settings, len(samples)
                )
                proposed_coverage_deficit = self._planner_rssi_deficit(
                    proposed_strongest, requirements, settings.coverage_mode
                )
                proposed_handover_deficit = self._planner_rssi_deficit(
                    proposed_second, requirements, settings.coverage_mode, settings.handover_margin_db
                )
                return (
                    proposed_strongest, proposed_second,
                    proposed_coverage_by_radio, proposed_secondary_by_radio,
                    proposed_overall, proposed_handover,
                    proposed_coverage_deficit, proposed_handover_deficit,
                )

            def score_candidate(candidate_ap: AccessPoint, candidate_rssi: List[np.ndarray], min_distance: float, coverage_needed: bool, handover_needed: bool, capacity_needed: bool):
                state = insert_candidate_state(candidate_rssi)
                (
                    proposed_strongest, _proposed_second,
                    _proposed_coverage_by_radio, _proposed_secondary_by_radio,
                    proposed_overall, proposed_handover,
                    proposed_coverage_deficit, proposed_handover_deficit,
                ) = state
                coverage_db_gain = float(np.sum(coverage_deficit - proposed_coverage_deficit))
                newly_covered = int(np.count_nonzero(proposed_overall & ~overall))
                handover_db_gain = float(np.sum(handover_deficit - proposed_handover_deficit))
                newly_handover = int(np.count_nonzero(proposed_handover & ~handover_overall))
                if (coverage_needed or handover_needed) and coverage_db_gain <= 1e-6 and handover_db_gain <= 1e-6:
                    return -float("inf"), state
                score = 0.0
                if coverage_needed:
                    score += coverage_db_gain * 20.0 + newly_covered * 250.0
                else:
                    score += coverage_db_gain * 2.0
                if handover_needed:
                    score += handover_db_gain * 12.0 + newly_handover * 160.0
                elif settings.handover_enabled:
                    score += handover_db_gain * 1.0
                if capacity_needed and not coverage_needed and not handover_needed:
                    mean_margin = float(np.mean(np.maximum(0.0, proposed_strongest[0] - strongest_by_radio[0])))
                    score += min_distance * 10.0 + mean_margin
                else:
                    score += min_distance * 0.01
                return score, state

            def apply_selected(index: int, candidate_ap: AccessPoint, candidate_rssi: List[np.ndarray], state):
                nonlocal strongest_by_radio, second_by_radio, coverage_by_radio, secondary_by_radio
                nonlocal overall, handover_overall, coverage_deficit, handover_deficit
                selected.append((candidate_ap, candidate_rssi))
                positions.append((candidate_ap.x, candidate_ap.y))
                (
                    strongest_by_radio, second_by_radio, coverage_by_radio, secondary_by_radio,
                    overall, handover_overall, coverage_deficit, handover_deficit,
                ) = state
                remaining.discard(index)

            while remaining and len(selected) < settings.maximum_aps:
                if progress.wasCanceled():
                    raise RuntimeError("Predictive AP planning cancelled")
                coverage_fraction = coverage_fraction_now()
                handover_fraction = handover_fraction_now()
                total_capacity_aps = len(existing) + len(selected)
                coverage_needed = coverage_fraction + 1e-12 < target_fraction
                handover_needed = (
                    settings.handover_enabled
                    and handover_fraction + 1e-12 < handover_target_fraction
                )
                capacity_needed = total_capacity_aps < capacity_ap_count
                if not coverage_needed and not handover_needed and not capacity_needed:
                    break

                if not positions:
                    try:
                        start_point = area.representative_point()
                        sx, sy = float(start_point.x), float(start_point.y)
                    except Exception:
                        sx = float(np.mean(sample_points[:, 0])); sy = float(np.mean(sample_points[:, 1]))
                    seed_index = min(
                        remaining,
                        key=lambda index: math.hypot(float(candidate_specs[index][0]) - sx, float(candidate_specs[index][1]) - sy),
                    )
                    candidate_ap, candidate_rssi = candidate_record(seed_index)
                    state = insert_candidate_state(candidate_rssi)
                    apply_selected(seed_index, candidate_ap, candidate_rssi, state)
                    progress.setLabelText("Placed the first AP and calculated RSSI extremities...")
                    progress.setValue(len(selected))
                    QApplication.processEvents()
                    continue

                best_index = None
                best_score = -float("inf")
                best_state = None
                best_record = None
                focus_indices = focus_sample_indices(coverage_needed, handover_needed, capacity_needed)
                shortlist = nearby_candidate_indices(focus_indices)
                if not shortlist:
                    shortlist = list(remaining)[:180]
                for scan_index, record_index in enumerate(shortlist, start=1):
                    if progress.wasCanceled():
                        raise RuntimeError("Predictive AP planning cancelled")
                    if scan_index == 1 or scan_index % 20 == 0 or scan_index == len(shortlist):
                        progress.setLabelText(
                            f"AP {len(selected) + 1}: checking {scan_index}/{len(shortlist)} locations near weakest RSSI samples..."
                        )
                        progress.setValue(len(selected))
                        QApplication.processEvents()
                    candidate_ap, candidate_rssi = candidate_record(record_index)
                    spacing_ok, min_distance = candidate_spacing_ok(candidate_ap)
                    if not spacing_ok:
                        continue
                    score, state = score_candidate(candidate_ap, candidate_rssi, min_distance, coverage_needed, handover_needed, capacity_needed)
                    if score > best_score:
                        best_score = score
                        best_index = record_index
                        best_state = state
                        best_record = (candidate_ap, candidate_rssi)

                if best_index is None or best_state is None or best_record is None:
                    break
                candidate_ap, candidate_rssi = best_record
                apply_selected(best_index, candidate_ap, candidate_rssi, best_state)
                progress.setLabelText(
                    f"Selected {len(selected)} AP(s): coverage {coverage_fraction_now() * 100.0:.1f}%"
                )
                progress.setValue(len(selected))
                QApplication.processEvents()

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
                self.aps.append(candidate_ap)
                new_aps.append(candidate_ap)
            self._assign_planner_channels(
                new_aps, requirements, samples=samples, wall_tree=wall_tree, walls=walls
            )

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

        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self._rssi_result_stale = False
        self._refresh_rssi_frequency_dropdown()
        self.populate_ap_table()
        self.draw_floor()
        overall_pct = 100.0 * float(np.count_nonzero(overall)) / max(1, len(overall))
        handover_pct = 100.0 * float(np.count_nonzero(handover_overall)) / max(1, len(handover_overall))
        band_lines = []
        for req, coverage_mask, handover_mask in zip(requirements, coverage_by_radio, secondary_by_radio):
            pct = 100.0 * float(np.count_nonzero(coverage_mask)) / max(1, len(coverage_mask))
            overlap_pct = 100.0 * float(np.count_nonzero(handover_mask & coverage_mask)) / max(1, len(coverage_mask))
            band_lines.append(
                f"{req.name} ({req.frequency_mhz:g} MHz): {pct:.1f}% at ≥ {req.minimum_rssi_dbm:g} dBm; "
                f"{overlap_pct:.1f}% with a second AP at ≥ {req.minimum_rssi_dbm - settings.handover_margin_db:g} dBm"
            )
        available_capacity = int((len(existing) + len(new_aps)) * effective_clients)
        warnings = []
        if overall_pct + 1e-6 < settings.target_coverage_percent:
            warnings.append("The RSSI coverage target was not fully achievable with the configured AP limit, spacing, radio patterns and wall losses.")
        if settings.handover_enabled and handover_pct + 1e-6 < settings.target_handover_percent:
            warnings.append("The dual-AP handover overlap target was not fully achievable. Reduce minimum AP spacing, increase the AP limit, or review transmit power and wall attenuation.")
        if available_capacity < settings.expected_clients:
            warnings.append(f"The estimated effective capacity ({available_capacity}) is below the expected {settings.expected_clients} clients.")
        warning = ("\n\n" + "\n".join(warnings)) if warnings else ""
        handover_summary = (
            f"Handover overlap (second distinct AP, channel-independent): {handover_pct:.1f}% "
            f"against a {settings.target_handover_percent:.1f}% target.\n"
            if settings.handover_enabled else
            "Handover overlap requirement: disabled.\n"
        )
        QMessageBox.information(
            self, "Predictive AP plan complete",
            f"Added {len(new_aps)} RSSI-predicted AP(s) on {self.floor.name}.\n"
            f"Planning area: {getattr(self, '_planner_area_source_label', 'selected floor geometry')}.\n"
            f"Overall coverage ({'every band' if settings.coverage_mode == 'all' else 'any band'}): {overall_pct:.1f}%\n"
            + handover_summary
            + f"Effective client capacity after {average_occupancy:.1f}% average spectrum occupancy: approximately {available_capacity} clients.\n\n"
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
                "ap_type": ap.ap_type, "radio_profile": ap.radio_profile,
                "max_clients": ap.max_clients, "planned": ap.planned,
                "radios": [self._radio_to_dict(radio) for radio in ap.radios],
            })
        user_walls = []
        overrides = []
        element_overrides = []
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
            for element in getattr(floor, "elements", []):
                if element.rf_customised:
                    element_overrides.append({
                        "guid": element.guid,
                        "source_file": element.source_file,
                        "floor": element.floor,
                        "ifc_class": element.ifc_class,
                        "rf_category": element.rf_category,
                        "rf_type_override": element.rf_type_override,
                        "rf_customised": True,
                        "attenuation_by_band_db": {
                            str(key): float(value) for key, value in element.attenuation_by_band_db.items()
                        },
                    })
        planner_boundaries = [
            {
                "guid": boundary.guid,
                "name": boundary.name,
                "shape_type": boundary.shape_type,
                "polygon": [
                    [float(x), float(y)] for x, y in boundary.polygon.exterior.coords
                ],
            }
            for boundary in self.planner_boundaries
        ]
        inferred_spaces = []
        user_spaces = []
        selected_ap_spaces = []
        for floor in self.floors.values():
            for space in floor.spaces:
                if space.ap_planning_selected:
                    selected_ap_spaces.append({
                        "source_file": space.source_file,
                        "guid": space.guid,
                        "floor": space.floor,
                    })
                if not (space.is_inferred or space.is_user_created):
                    continue
                item = {
                    "guid": space.guid,
                    "name": space.name,
                    "floor": space.floor,
                    "source_file": space.source_file,
                    "polygon": [[float(x), float(y)] for x, y in space.polygon.exterior.coords],
                    "z_min": space.z_min,
                    "z_max": space.z_max,
                    "source_storey": space.source_storey,
                    "assumption_note": space.assumption_note,
                    "ap_planning_selected": space.ap_planning_selected,
                }
                if space.is_user_created:
                    user_spaces.append(item)
                else:
                    inferred_spaces.append(item)
        return {
            "format": "rf-attenuation-plan", "version": 7,
            "ifc_paths": [str(path) for path in self.loaded_ifc_paths],
            "selected_floor": self.floor.name if self.floor else "", "view_rotation_deg": self.view_rotation_deg,
            "ifc_alignment": {"dx": self.ifc_alignment.dx, "dy": self.ifc_alignment.dy, "rotation_deg": self.ifc_alignment.rotation_deg, "scale": self.ifc_alignment.scale},
            "auto_planner_settings": self.auto_planner_settings.to_dict(),
            "propagation_model": self.heatmap_settings.propagation_model_dict(),
            "rf_performance": self.heatmap_settings.performance_model_dict(),
            "planner_boundaries": planner_boundaries,
            "excluded_ifc_elements": [
                self._ifc_exclusion_record(
                    item.get("kind", ""), item.get("source_file", ""), item.get("guid", ""), item.get("floor", "")
                )
                for item in getattr(self, "excluded_ifc_elements", [])
                if isinstance(item, dict) and item.get("kind") and item.get("source_file") and item.get("guid")
            ],
            "selected_ap_spaces": selected_ap_spaces,
            "user_spaces": user_spaces,
            "inferred_spaces": inferred_spaces,
            "access_points": aps, "user_walls": user_walls, "wall_overrides": overrides,
            "element_overrides": element_overrides,
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
        # Remove prior simulator-created geometry before restoring the saved plan.
        for floor in self.floors.values():
            floor.walls = [wall for wall in floor.walls if not wall.is_user_created]
            floor.spaces = [space for space in floor.spaces if not space.is_inferred and not space.is_user_created]
            for wall in floor.walls:
                if wall.rf_original_polygon is not None:
                    wall.polygon = wall.rf_original_polygon
                wall.rf_original_polygon = None
                wall.rf_geometry_customised = False
                wall.rf_customised = False; wall.rf_type_override = ""
                wall.attenuation_by_band_db = self._profile_for_wall_from_settings(wall)
            for element in getattr(floor, "elements", []):
                element.rf_customised = False
                element.rf_type_override = ""
                element.attenuation_by_band_db = self._profile_for_ifc_element_from_settings(element)
        self.excluded_ifc_elements = [
            self._ifc_exclusion_record(
                item.get("kind", ""), item.get("source_file", ""), item.get("guid", ""), item.get("floor", "")
            )
            for item in data.get("excluded_ifc_elements", [])
            if isinstance(item, dict) and item.get("kind") in {"wall", "space", "element"} and item.get("source_file") and item.get("guid")
        ]
        self._apply_ifc_exclusions()
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
        element_override_exact = {
            (str(item.get("source_file", "")), str(item.get("guid", "")), str(item.get("floor", ""))): item
            for item in data.get("element_overrides", []) if isinstance(item, dict) and str(item.get("floor", ""))
        }
        element_override_legacy = {
            (str(item.get("source_file", "")), str(item.get("guid", ""))): item
            for item in data.get("element_overrides", []) if isinstance(item, dict)
        }
        for floor in self.floors.values():
            for element in getattr(floor, "elements", []):
                item = element_override_exact.get((element.source_file, element.guid, element.floor))
                if item is None:
                    item = element_override_legacy.get((element.source_file, element.guid))
                if not item:
                    continue
                element.rf_type_override = str(item.get("rf_type_override", ""))
                element.rf_customised = bool(item.get("rf_customised", True))
                element.attenuation_by_band_db.update({
                    float(key): float(value)
                    for key, value in dict(item.get("attenuation_by_band_db", {})).items()
                })

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
        for item in data.get("user_spaces", []):
            if not isinstance(item, dict):
                continue
            floor = self.floors.get(str(item.get("floor", "")))
            coords = item.get("polygon", [])
            if floor is None or not isinstance(coords, list) or len(coords) < 3:
                continue
            try:
                polygon = Polygon([(float(value[0]), float(value[1])) for value in coords])
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if polygon.is_empty:
                    continue
                if polygon.geom_type != "Polygon":
                    parts = list(getattr(polygon, "geoms", []))
                    if not parts:
                        continue
                    polygon = max(parts, key=lambda value: float(value.area))
                floor.spaces.append(Space2D(
                    guid=str(item.get("guid", f"user-space-{uuid.uuid4().hex}")),
                    name=str(item.get("name", "User Space")),
                    floor=floor.name,
                    source_file=str(item.get("source_file", "RF simulator user space")),
                    polygon=polygon,
                    z_min=float(item.get("z_min", floor.elevation)),
                    z_max=float(item.get("z_max", floor.elevation + 3.0)),
                    source_storey=str(item.get("source_storey", floor.name)),
                    is_user_created=True,
                    ap_planning_selected=bool(item.get("ap_planning_selected", False)),
                    assumption_note=str(item.get("assumption_note", "Restored user-created space")),
                ))
            except Exception:
                continue
        for item in data.get("inferred_spaces", []):
            if not isinstance(item, dict):
                continue
            floor = self.floors.get(str(item.get("floor", "")))
            coords = item.get("polygon", [])
            if floor is None or not isinstance(coords, list) or len(coords) < 3:
                continue
            try:
                polygon = Polygon([(float(value[0]), float(value[1])) for value in coords])
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if polygon.is_empty:
                    continue
                if polygon.geom_type != "Polygon":
                    parts = list(getattr(polygon, "geoms", []))
                    if not parts:
                        continue
                    polygon = max(parts, key=lambda value: float(value.area))
                floor.spaces.append(Space2D(
                    guid=str(item.get("guid", f"inferred-space-{uuid.uuid4().hex}")),
                    name=str(item.get("name", "Inferred Space")),
                    floor=floor.name,
                    source_file=str(item.get("source_file", "RF simulator inferred space")),
                    polygon=polygon,
                    z_min=float(item.get("z_min", floor.elevation)),
                    z_max=float(item.get("z_max", floor.elevation + 3.0)),
                    source_storey=str(item.get("source_storey", floor.name)),
                    is_inferred=True,
                    ap_planning_selected=bool(item.get("ap_planning_selected", False)),
                    assumption_note=str(item.get("assumption_note", "Restored inferred space")),
                ))
            except Exception:
                continue

        selected_refs = {
            (
                str(item.get("source_file", "")),
                str(item.get("guid", "")),
                str(item.get("floor", "")),
            )
            for item in data.get("selected_ap_spaces", [])
            if isinstance(item, dict) and item.get("source_file") and item.get("guid")
        }
        if selected_refs:
            for floor in self.floors.values():
                for space in floor.spaces:
                    ref = (str(space.source_file), str(space.guid), str(space.floor))
                    if ref in selected_refs:
                        space.ap_planning_selected = True

        self.planner_boundaries = []
        for item in data.get("planner_boundaries", []):
            if not isinstance(item, dict):
                continue
            coords = item.get("polygon", [])
            if not isinstance(coords, list) or len(coords) < 3:
                continue
            try:
                polygon = Polygon([(float(value[0]), float(value[1])) for value in coords])
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if polygon.is_empty:
                    continue
                if polygon.geom_type != "Polygon":
                    polygons = list(getattr(polygon, "geoms", []))
                    if not polygons:
                        continue
                    polygon = max(polygons, key=lambda value: float(value.area))
                if float(polygon.area) < 0.01:
                    continue
                shape_type = str(item.get("shape_type", "polygon")).strip().lower()
                if shape_type not in {"rectangle", "polygon"}:
                    shape_type = "polygon"
                self.planner_boundaries.append(PlannerBoundary2D(
                    guid=str(item.get("guid", f"planner-boundary-{uuid.uuid4().hex}")),
                    name=str(item.get("name", f"Planner boundary {len(self.planner_boundaries) + 1}")),
                    polygon=polygon,
                    shape_type=shape_type,
                ))
            except Exception:
                continue

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
                rx_height_m=float(item.get("rx_height_m", 1.2)),
                ap_type=str(item.get("ap_type", "Ceiling AP")), radio_profile=str(item.get("radio_profile", "Project default radios")),
                radios=radios, max_clients=int(item.get("max_clients", 50)), planned=bool(item.get("planned", False)),
            ))
        self.auto_planner_settings = AutoPlannerSettings.from_dict(data.get("auto_planner_settings", {}))
        self.heatmap_settings.apply_propagation_model_dict(data.get("propagation_model", {}))
        self.heatmap_settings.apply_performance_model_dict(data.get("rf_performance", {}))
        self.boundary_result_filter_action.blockSignals(True)
        self.boundary_result_filter_action.setChecked(
            bool(self.heatmap_settings.ignore_results_outside_planner_boundaries)
        )
        self.boundary_result_filter_action.blockSignals(False)
        selected_floor = str(data.get("selected_floor", ""))
        if selected_floor in self.floors:
            self.floor_combo.setCurrentText(selected_floor)
        self.reset_view_rotation()
        saved_rotation = float(data.get("view_rotation_deg", 0.0))
        if abs(saved_rotation) > 1e-9:
            self.rotate_view(saved_rotation)
        self._invalidate_interactive_preview_requests()
        self.last_result = None; self.rssi_results_by_frequency = {}
        self._refresh_rssi_frequency_dropdown()
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
            self._invalidate_interactive_preview_requests()
            self.last_result = None
        self.draw_floor()

    @staticmethod
    def _bilinear_grid_value(
        result: SimulationResult,
        grid,
        x: float,
        y: float,
    ) -> Optional[float]:
        xs = np.asarray(result.xs, dtype=float)
        ys = np.asarray(result.ys, dtype=float)
        values_grid = np.asarray(grid, dtype=float)
        if xs.size == 0 or ys.size == 0 or values_grid.shape != (ys.size, xs.size):
            return None
        if x < float(xs[0]) or x > float(xs[-1]) or y < float(ys[0]) or y > float(ys[-1]):
            return None

        if xs.size == 1 or ys.size == 1:
            ix = int(np.argmin(np.abs(xs - x)))
            iy = int(np.argmin(np.abs(ys - y)))
            value = float(values_grid[iy, ix])
            return value if math.isfinite(value) else None

        right = int(np.searchsorted(xs, x, side="right"))
        top = int(np.searchsorted(ys, y, side="right"))
        ix0 = max(0, min(right - 1, xs.size - 2))
        iy0 = max(0, min(top - 1, ys.size - 2))
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        x0, x1 = float(xs[ix0]), float(xs[ix1])
        y0, y1 = float(ys[iy0]), float(ys[iy1])
        tx = 0.0 if abs(x1 - x0) <= 1e-12 else (float(x) - x0) / (x1 - x0)
        ty = 0.0 if abs(y1 - y0) <= 1e-12 else (float(y) - y0) / (y1 - y0)
        values = np.asarray([
            values_grid[iy0, ix0], values_grid[iy0, ix1], values_grid[iy1, ix0], values_grid[iy1, ix1]
        ], dtype=float)
        if not np.all(np.isfinite(values)):
            finite = values[np.isfinite(values)]
            return float(finite[0]) if finite.size else None
        bottom = float(values[0]) * (1.0 - tx) + float(values[1]) * tx
        upper = float(values[2]) * (1.0 - tx) + float(values[3]) * tx
        return bottom * (1.0 - ty) + upper * ty

    @staticmethod
    def _bilinear_heatmap_value(result: SimulationResult, x: float, y: float) -> Optional[float]:
        boundary = getattr(result, "boundary_geometry", None)
        if boundary is not None:
            try:
                if not boundary.covers(Point(float(x), float(y))):
                    return None
            except Exception:
                pass
        return MainWindow._bilinear_grid_value(result, result.rssi, x, y)

    @staticmethod
    def _interpolated_heatmap_grid(result: SimulationResult, factor: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        factor = max(1, int(factor))
        cache_key = ("interpolated_grid", factor)
        cached = result.render_cache.get(cache_key)
        if cached is not None:
            return cached

        xs = np.asarray(result.xs, dtype=float)
        ys = np.asarray(result.ys, dtype=float)
        grid = np.asarray(result.rssi, dtype=float)
        if factor <= 1 or xs.size < 2 or ys.size < 2:
            value = (xs, ys, grid)
            result.render_cache[cache_key] = value
            return value

        fine_xs = np.linspace(float(xs[0]), float(xs[-1]), (len(xs) - 1) * factor + 1)
        fine_ys = np.linspace(float(ys[0]), float(ys[-1]), (len(ys) - 1) * factor + 1)
        # The source grid is regular, so bilinear subdivision can be performed
        # entirely with NumPy. Boundary-filtered grids carry NaNs outside the
        # permitted area; interpolate finite values and then apply an exact mask
        # at the finer coordinates so contours never leak beyond the boundary.
        t = np.arange(factor, dtype=float) / float(factor)

        def interpolate(source: np.ndarray) -> np.ndarray:
            x_segments = (
                source[:, :-1, None] * (1.0 - t)[None, None, :]
                + source[:, 1:, None] * t[None, None, :]
            )
            x_interpolated = np.concatenate(
                [x_segments.reshape(source.shape[0], -1), source[:, -1:]], axis=1
            )
            y_segments = (
                x_interpolated[:-1, :, None] * (1.0 - t)[None, None, :]
                + x_interpolated[1:, :, None] * t[None, None, :]
            )
            return np.concatenate(
                [
                    y_segments.transpose(0, 2, 1).reshape(-1, x_interpolated.shape[1]),
                    x_interpolated[-1:, :],
                ],
                axis=0,
            )

        finite_source = np.isfinite(grid)
        if np.all(finite_source):
            z = interpolate(grid)
        else:
            weights = interpolate(finite_source.astype(float))
            values = interpolate(np.where(finite_source, grid, 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(weights > 1e-12, values / weights, np.nan)
            fine_mask = RFEngine._boundary_mask(
                fine_xs, fine_ys, getattr(result, "boundary_geometry", None)
            )
            if fine_mask is not None:
                z = np.where(fine_mask, z, np.nan)
        value = (fine_xs, fine_ys, z)
        result.render_cache[cache_key] = value
        return value

    def update_rssi_hover_readout(self, scene_pos: QPointF, viewport_pos):
        if bool(getattr(self, "_rssi_result_stale", False)):
            self.view.hide_rssi_hover()
            return
        result = getattr(self, "last_result", None)
        frequency = self._selected_rssi_view_frequency()
        if result is None or frequency is None or not self.floor:
            self.view.hide_rssi_hover()
            return
        value = self._bilinear_heatmap_value(result, float(scene_pos.x()), float(scene_pos.y()))
        if value is None:
            self.view.hide_rssi_hover()
            return
        frequency_text = f"{frequency / 1000.0:g} GHz" if frequency >= 1000.0 else f"{frequency:g} MHz"
        disconnect = float(getattr(self.heatmap_settings, "disconnected_rssi_dbm", -120.0))
        rssi_text = f"≤ {disconnect:.0f} dBm" if value <= disconnect + 0.05 else f"{value:.1f} dBm"
        details = [f"{frequency_text}: {rssi_text}"]
        delay_grid = getattr(result, "delay_spread_ns", None)
        if delay_grid is not None:
            delay = self._bilinear_grid_value(
                result, delay_grid, float(scene_pos.x()), float(scene_pos.y())
            )
            if delay is not None and math.isfinite(delay):
                details.append(f"RMS delay spread: {delay:.1f} ns")
        count_grid = getattr(result, "path_count", None)
        if count_grid is not None:
            count = self._bilinear_grid_value(
                result, count_grid, float(scene_pos.x()), float(scene_pos.y())
            )
            if count is not None and math.isfinite(count):
                details.append(f"Contributing paths: {max(0, int(round(count)))}")
        details.append(f"X {scene_pos.x():.2f} m   Y {scene_pos.y():.2f} m")
        self.view.show_rssi_hover("\n".join(details), viewport_pos)

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
        if hasattr(self, "ribbon"):
            dark = bool(getattr(self, "dark_theme", False))
            ribbon_bg = "#2F3338" if dark else "#F3F5F7"
            group_bg = "#383D43" if dark else "#FFFFFF"
            button_bg = "#434950" if dark else "#FAFBFC"
            hover_bg = "#505861" if dark else "#E8F1FB"
            checked_bg = "#405A72" if dark else "#D6E9FB"
            text_colour = "#F2F2F2" if dark else "#202124"
            border = "#555C64" if dark else "#C8CDD3"
            self.ribbon.setStyleSheet(
                "QTabWidget#MainRibbon::pane { border: 0; border-bottom: 1px solid %(border)s; background: %(ribbon_bg)s; }"
                "QTabWidget#MainRibbon QTabBar::tab { min-width: 120px; padding: 6px 16px; color: %(text)s; }"
                "QTabWidget#MainRibbon QTabBar::tab:selected { background: %(group_bg)s; border: 1px solid %(border)s; border-bottom: 0; }"
                "QScrollArea#RibbonPage { background: %(ribbon_bg)s; }"
                "QWidget#RibbonGroup { background: transparent; }"
                "QFrame#RibbonGroupBox { background: %(group_bg)s; border: 1px solid %(border)s; border-radius: 3px; }"
                "QLabel#RibbonGroupCaption { color: %(text)s; font-size: 10px; padding-top: 1px; }"
                "QToolButton { background: %(button_bg)s; color: %(text)s; border: 1px solid transparent; border-radius: 3px; padding: 3px 5px; }"
                "QToolButton:hover { background: %(hover_bg)s; border-color: %(border)s; }"
                "QToolButton:pressed, QToolButton:checked { background: %(checked_bg)s; border-color: %(border)s; }"
                % {
                    "ribbon_bg": ribbon_bg,
                    "group_bg": group_bg,
                    "button_bg": button_bg,
                    "hover_bg": hover_bg,
                    "checked_bg": checked_bg,
                    "text": text_colour,
                    "border": border,
                }
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

        For project alignment use Model and view > Two-point alignment in the ribbon.
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
            "Use Model and view > Two-point alignment in the ribbon. The workflow "
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
            for element in getattr(floor, "elements", []):
                element.polygon = affine_transform(element.polygon, delta)
        for boundary in self.planner_boundaries:
            boundary.polygon = affine_transform(boundary.polygon, delta)
        for ap in self.aps:
            a, b, d, e, xoff, yoff = delta
            old_x, old_y = float(ap.x), float(ap.y)
            ap.x = a * old_x + b * old_y + xoff
            ap.y = d * old_x + e * old_y + yoff
        composed = AlignmentTransform._compose(delta, self.ifc_alignment.matrix())
        self.ifc_alignment = AlignmentTransform.from_matrix(composed)
        self._invalidate_interactive_preview_requests()
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
            if getattr(self, "space_draw_mode", False) or bool(getattr(self, "_space_polygon_points", [])):
                self.cancel_space_drawing()
            if (
                getattr(self, "boundary_draw_mode", False)
                or getattr(self, "_boundary_draw_start", None) is not None
                or bool(getattr(self, "_boundary_polygon_points", []))
            ):
                self.cancel_planner_boundary_drawing()
            if abs(float(getattr(self, "view_rotation_deg", 0.0))) > 1e-9:
                self.reset_view_rotation()
            self.floors = {}
            self.loaded_ifc_paths = []
            self.ifc_origin_info = {}
            self.aps.clear()
            self.planner_boundaries.clear()
            self.excluded_ifc_elements.clear()
            self._invalidate_interactive_preview_requests()
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
                            path_str, storeys, wall_guids, space_guids, element_guids, source_name, origin_info = _index_ifc_file_for_chunking(job)
                            self.ifc_origin_info[self._ifc_path_key(path)] = origin_info
                            records = [("wall", g) for g in wall_guids] + [("space", g) for g in space_guids] + [("element", g) for g in element_guids]
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
                                chunk_element_guids = [g for kind, g in chunk if kind == "element"]
                                chunk_job = (
                                    path_str, job[1], job[2], job[3], job[4], job[5],
                                    storeys, chunk_wall_guids, chunk_space_guids, chunk_element_guids, idx, len(chunks),
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
        closed_wall_count = close_imported_wall_endpoint_gaps(self.floors)
        self._refresh_floor_combo()
        total_walls = sum(len(f.walls) for f in self.floors.values())
        total_spaces = sum(len(f.spaces) for f in self.floors.values())
        total_elements = sum(len(getattr(f, "elements", [])) for f in self.floors.values())
        msg = f"Loaded {len(self.loaded_ifc_paths)} IFC file(s), {len(self.floors)} floor(s), {total_walls} wall(s), {total_spaces} space(s), {total_elements} other element(s)"
        if closed_wall_count:
            msg += f". Closed small endpoint gaps on {closed_wall_count} wall(s)."
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
        self.export_pdf_action.setEnabled(not loading)
        self.clear_ap_action.setEnabled(not loading)
        self.load_pattern_action.setEnabled(not loading)
        for action_name in (
            "planner_settings_action", "propagation_settings_action", "predict_aps_action", "bulk_ap_action", "draw_wall_action", "bulk_attenuation_action", "draw_space_action", "select_ap_spaces_action", "inferred_space_interaction_action",
            "draw_boundary_action", "draw_polygon_boundary_action", "suggest_external_boundary_action",
            "create_spaces_action", "clear_inferred_spaces_action", "clear_boundaries_action", "ifc_origin_action",
            "rotate_left_action", "rotate_right_action", "reset_rotation_action", "save_plan_action", "load_plan_action",
        ):
            action = getattr(self, action_name, None)
            if action is not None:
                action.setEnabled(not loading)
        self.floor_combo.setEnabled(not loading)

    def select_floor(self, name: str):
        self._invalidate_interactive_preview_requests()
        self._rssi_result_stale = False
        if getattr(self, "wall_draw_mode", False) or getattr(self, "_wall_draw_start", None) is not None:
            self.cancel_user_wall_drawing()
        if getattr(self, "space_draw_mode", False) or bool(getattr(self, "_space_polygon_points", [])):
            self.cancel_space_drawing()
        if (
            getattr(self, "boundary_draw_mode", False)
            or getattr(self, "_boundary_draw_start", None) is not None
            or bool(getattr(self, "_boundary_polygon_points", []))
        ):
            self.cancel_planner_boundary_drawing()
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
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self._load_slab_attenuation_to_ui()
        self.draw_floor()
        self.populate_ap_table()
        self.populate_wall_table()

    def _radios_for_profile(self, profile_name: str, ap_type: str) -> List[APRadio]:
        profile_name = str(profile_name or "Project default radios")
        if profile_name == "Project default radios":
            definitions = list(self.heatmap_settings.default_ap_radios or [])
        else:
            definitions = list(RADIO_PROFILE_PRESETS.get(profile_name, []))
        type_pattern = str(AP_TYPE_PRESETS.get(ap_type, AP_TYPE_PRESETS["Ceiling AP"]).get("pattern", self.pattern_combo.currentText()))
        radios: List[APRadio] = []
        for index, radio_def in enumerate(definitions):
            if not isinstance(radio_def, dict):
                continue
            pattern = str(radio_def.get("antenna_pattern", type_pattern))
            if pattern not in self.antenna_patterns:
                pattern = type_pattern if type_pattern in self.antenna_patterns else self.pattern_combo.currentText()
            radios.append(APRadio(
                name=str(radio_def.get("name", f"Radio-{index + 1}")),
                frequency_mhz=float(radio_def.get("frequency_mhz", self.freq.value())),
                tx_power_dbm=float(radio_def.get("tx_power_dbm", self.tx_power.value())),
                antenna_pattern=pattern,
                enabled=bool(radio_def.get("enabled", True)),
                cutoff_radius_m=float(radio_def.get("cutoff_radius_m", 0.0)),
                antenna_gain_dbi=float(radio_def.get("antenna_gain_dbi", 0.0)),
                channel=str(radio_def.get("channel", "")),
                channel_width_mhz=float(radio_def.get("channel_width_mhz", 20.0)),
                spectrum_occupancy_percent=float(radio_def.get("spectrum_occupancy_percent", 0.0)),
            ))
        if not radios:
            radios = [APRadio(
                name="Radio-1", frequency_mhz=float(self.freq.value()), tx_power_dbm=float(self.tx_power.value()),
                antenna_pattern=type_pattern if type_pattern in self.antenna_patterns else self.pattern_combo.currentText(), enabled=True,
            )]
        return radios

    def add_ap(
        self, x: float, y: float, *, redraw: bool = True,
        ap_type: Optional[str] = None, radio_profile: Optional[str] = None,
        planned: bool = False,
    ) -> Optional[AccessPoint]:
        if not self.floor:
            return None
        selected_type = str(ap_type or self.ap_type_combo.currentText() or "Ceiling AP")
        if selected_type not in AP_TYPE_PRESETS:
            selected_type = "Ceiling AP"
        selected_profile = str(radio_profile or self.radio_profile_combo.currentText() or "Project default radios")
        default_radios = self._radios_for_profile(selected_profile, selected_type)
        first_radio = default_radios[0]
        ap = AccessPoint(
            name=self._next_ap_name(), x=float(x), y=float(y), floor=self.floor.name,
            tx_power_dbm=float(first_radio.tx_power_dbm), frequency_mhz=float(first_radio.frequency_mhz),
            path_loss_exponent=float(self.ple.value()), antenna_pattern=first_radio.antenna_pattern,
            azimuth_deg=float(self.azimuth.value()), downtilt_deg=float(self.downtilt.value()),
            mount_height_m=float(self.mount_height.value()), rx_height_m=float(self.rx_height.value()),
            ap_type=selected_type, radio_profile=selected_profile,
            radios=default_radios, max_clients=int(self.auto_planner_settings.clients_per_ap), planned=bool(planned),
        )
        self.aps.append(ap)
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self._rssi_result_stale = False
        self._refresh_rssi_frequency_dropdown()
        if redraw:
            self.draw_floor()
            self.populate_ap_table()
        return ap

    def clear_aps(self):
        self.aps = [a for a in self.aps if not self.floor or a.floor != self.floor.name]
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self._rssi_result_stale = False
        self._refresh_rssi_frequency_dropdown()
        self.draw_floor()
        self.populate_ap_table()

    def draw_floor(self):
        scene = self.view.scene()
        self.view.hide_rssi_hover()
        self._drawing_floor = True
        scene.clear()
        self._ap_ruler_items = []
        self._ifc_snap_marker_items = []
        self._wall_preview_items = []
        self._space_preview_items = []
        self._boundary_preview_items = []
        self._suggested_boundary_preview_items = []
        self._suggested_space_preview_items = []
        if not self.floor:
            self._drawing_floor = False
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
            if space.ap_planning_selected:
                pen_colour = QColor("#F59E0B")
                fill_colour = QColor("#FDE68A")
            elif space.is_inferred:
                pen_colour = QColor("#00A77B")
                fill_colour = QColor("#7FE0C2")
            elif space.is_user_created:
                pen_colour = QColor("#2D9CDB")
                fill_colour = QColor("#93C5FD")
            else:
                pen_colour = colours["space_pen"]
                fill_colour = QColor(colours["space_fill"])
            if space.ap_planning_selected:
                fill_colour.setAlpha(115)
            elif space.is_inferred:
                fill_colour.setAlpha(70)
            elif space.is_user_created:
                fill_colour.setAlpha(65)
            pen = QPen(pen_colour, 0.20 if space.ap_planning_selected else (0.16 if (space.is_inferred or space.is_user_created) else 0.12))
            pen.setCosmetic(True)
            if space.is_inferred:
                pen.setStyle(Qt.DashLine)
                item = InferredSpaceGraphicsItem(self, space, poly, pen, QBrush(fill_colour))
            elif space.is_user_created:
                pen.setStyle(Qt.DashLine)
                item = SpaceGraphicsItem(self, space, poly, pen, QBrush(fill_colour))
            else:
                item = SpaceGraphicsItem(self, space, poly, pen, QBrush(fill_colour))
            scene.addItem(item)

            if space.name:
                centroid = space.polygon.representative_point()
                cx, cy = self._point_xy(centroid)
                label = str(space.name)
                if space.is_inferred:
                    label += " (inferred)"
                elif space.is_user_created:
                    label += " (user)"
                if space.ap_planning_selected:
                    label += " [AP]"
                self._add_upright_text(
                    scene, label, cx, cy,
                    QColor("#B45309") if space.ap_planning_selected else (QColor("#007A5A") if space.is_inferred else (QColor("#1D4ED8") if space.is_user_created else colours["space_text"])),
                    self.heatmap_settings.space_label_font_size, Z_TEXT, bold=True
                )

        generic_element_pen_colour = QColor("#526D82") if not getattr(self, "dark_theme", False) else QColor("#88A6BC")
        generic_element_fill_colour = QColor("#B6C6D2") if not getattr(self, "dark_theme", False) else QColor("#364854")
        generic_element_fill_colour.setAlpha(80)
        for element in getattr(self.floor, "elements", []):
            try:
                coords = list(element.polygon.exterior.coords)
            except Exception:
                continue
            if len(coords) < 3:
                continue
            if element.rf_category == "door":
                element_pen_colour = QColor("#A85D00") if not getattr(self, "dark_theme", False) else QColor("#FFB454")
                element_fill_colour = QColor("#E5A04B")
                element_fill_colour.setAlpha(175)
            elif element.rf_category == "window":
                element_pen_colour = QColor("#007EA8") if not getattr(self, "dark_theme", False) else QColor("#65D5FF")
                element_fill_colour = QColor("#72C7E7")
                element_fill_colour.setAlpha(165)
            elif element.is_rf_barrier:
                element_pen_colour = QColor("#7C3AED") if not getattr(self, "dark_theme", False) else QColor("#C4B5FD")
                element_fill_colour = QColor("#A78BFA")
                element_fill_colour.setAlpha(105)
            else:
                element_pen_colour = QColor(generic_element_pen_colour)
                element_fill_colour = QColor(generic_element_fill_colour)
            element_pen = QPen(element_pen_colour, 0.11 if element.is_rf_barrier else 0.08)
            element_pen.setCosmetic(True)
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in coords])
            item = IFCElementGraphicsItem(self, element, poly, element_pen, QBrush(element_fill_colour))
            scene.addItem(item)

        for boundary in self.planner_boundaries:
            item = PlannerBoundaryGraphicsItem(self, boundary)
            scene.addItem(item)
            minx, miny, maxx, maxy = boundary.polygon.bounds
            self._add_upright_text(
                scene,
                f"{boundary.name} (all floors)",
                float(minx),
                float(maxy),
                QColor("#00A6D6"),
                max(2, int(self.heatmap_settings.space_label_font_size)),
                Z_TEXT + 3,
                bold=True,
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
        pending_ap_selection_ids = set(getattr(self, "_pending_ap_selection_ids", set()))
        visible_aps = [a for a in self.aps if a.floor == self.floor.name or self.include_inter_floor.isChecked()]
        for ap in visible_aps:
            same_floor = ap.floor == self.floor.name
            radius = 0.75 if same_floor else 0.45
            colour = colours["ap_same_floor"] if same_floor else colours["ap_other_floor"]

            if same_floor:
                dot = AccessPointGraphicsItem(self, ap, radius, colour)
                scene.addItem(dot)
                if id(ap) in pending_ap_selection_ids:
                    dot.setSelected(True)
            else:
                dot = QGraphicsPathItem(_access_point_symbol_path(ap.ap_type, radius))
                dot.setPos(float(ap.x), float(ap.y))
                dot.setBrush(QBrush(colour))
                other_pen = QPen(colours["ap_outline"], 0.2)
                other_pen.setCosmetic(True)
                dot.setPen(other_pen)
                dot.setAcceptedMouseButtons(Qt.NoButton)
                dot.setZValue(Z_AP - 5)
                scene.addItem(dot)

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
                    cut_item.setAcceptedMouseButtons(Qt.NoButton)
                    scene.addItem(cut_item)
                    dot.add_move_follower(cut_item)

            # Keep the boresight arrow parented to the AP symbol so it follows live drag movement.
            length = 5.0 if same_floor else 3.0
            ang = math.radians(ap.azimuth_deg)
            arrow_path = QPainterPath(QPointF(0.0, 0.0))
            arrow_path.lineTo(length * math.cos(ang), length * math.sin(ang))
            arrow = QGraphicsPathItem(arrow_path, dot)
            arrow_pen = QPen(colour, 0.25)
            arrow_pen.setCosmetic(True)
            arrow.setPen(arrow_pen)
            arrow.setBrush(QBrush(Qt.NoBrush))
            arrow.setAcceptedMouseButtons(Qt.NoButton)
            arrow.setZValue(1.0)
            type_short = AP_TYPE_PRESETS.get(ap.ap_type, AP_TYPE_PRESETS["Ceiling AP"]).get("short", "AP")
            label = f"{ap.name} [{type_short}]" if same_floor else f"{ap.floor}: {ap.name} [{type_short}]"
            label_item = self._add_upright_text(
                scene, label, ap.x + radius + 0.25, ap.y + radius + 0.25,
                colour, self.heatmap_settings.ap_label_font_size, Z_AP_LABEL, bold=same_floor,
            )
            if same_floor:
                dot.add_move_follower(label_item)

        self._pending_ap_selection_ids = set()
        self._drawing_floor = False
        self.refresh_access_point_rulers()

        if self.ap_interaction_mode:
            for item in scene.items():
                if isinstance(item, AccessPointGraphicsItem):
                    continue
                item.setAcceptedMouseButtons(Qt.NoButton)
                if item.flags() & QGraphicsItem.ItemIsSelectable:
                    item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        elif self.inferred_space_interaction_mode:
            for item in scene.items():
                if isinstance(item, InferredSpaceGraphicsItem):
                    continue
                item.setAcceptedMouseButtons(Qt.NoButton)
                if item.flags() & QGraphicsItem.ItemIsSelectable:
                    item.setFlag(QGraphicsItem.ItemIsSelectable, False)

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

    @staticmethod
    def _geometry_to_qpath(geometry) -> QPainterPath:
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        if geometry is None:
            return path
        polygons = [geometry] if getattr(geometry, "geom_type", "") == "Polygon" else list(getattr(geometry, "geoms", []))
        for polygon in polygons:
            if getattr(polygon, "geom_type", "") != "Polygon":
                continue
            for ring in [polygon.exterior, *list(polygon.interiors)]:
                coords = list(ring.coords)
                if len(coords) < 3:
                    continue
                path.moveTo(QPointF(float(coords[0][0]), float(coords[0][1])))
                for x_value, y_value in coords[1:]:
                    path.lineTo(QPointF(float(x_value), float(y_value)))
                path.closeSubpath()
        return path

    def _draw_heatmap_raster(self, result: SimulationResult, xs: np.ndarray, ys: np.ndarray, z: np.ndarray):
        """Render the heatmap as one RGBA pixmap instead of thousands of fill polygons."""
        if len(xs) < 2 or len(ys) < 2 or z.size == 0:
            return
        cache_key = (
            "heatmap_raster", tuple((zone.name, zone.min_dbm, zone.max_dbm, zone.colour, zone.alpha) for zone in self.heatmap_settings.zones),
            tuple(z.shape), bool(getattr(self, "dark_theme", False)),
        )
        image = result.render_cache.get(cache_key)
        if image is None:
            rgba = np.zeros((z.shape[0], z.shape[1], 4), dtype=np.uint8)
            finite = np.isfinite(z)
            for zone in self.heatmap_settings.zones:
                mask = finite & (z >= float(zone.min_dbm)) & (z < float(zone.max_dbm))
                if not np.any(mask):
                    continue
                colour = QColor(zone.colour)
                if not colour.isValid():
                    colour = QColor("#555555")
                rgba[mask, 0] = int(colour.red())
                rgba[mask, 1] = int(colour.green())
                rgba[mask, 2] = int(colour.blue())
                rgba[mask, 3] = max(0, min(255, int(zone.alpha)))
            # Cover values above/below the explicitly bounded zones using the configured end colours.
            if np.any(finite):
                uncovered = finite & (rgba[:, :, 3] == 0)
                if np.any(uncovered):
                    high = max(self.heatmap_settings.zones, key=lambda zone: zone.max_dbm)
                    low = min(self.heatmap_settings.zones, key=lambda zone: zone.min_dbm)
                    high_mask = uncovered & (z >= float(high.max_dbm))
                    low_mask = uncovered & ~high_mask
                    for mask, zone in ((high_mask, high), (low_mask, low)):
                        if not np.any(mask):
                            continue
                        colour = QColor(zone.colour)
                        if not colour.isValid():
                            colour = QColor("#555555")
                        rgba[mask, 0] = int(colour.red())
                        rgba[mask, 1] = int(colour.green())
                        rgba[mask, 2] = int(colour.blue())
                        rgba[mask, 3] = max(0, min(255, int(zone.alpha)))
            image_format = getattr(QImage, "Format_RGBA8888", None)
            if image_format is None:
                image_format = QImage.Format.Format_RGBA8888
            image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], image_format).copy()
            result.render_cache[cache_key] = image
        pixmap = QPixmap.fromImage(image)
        item = QGraphicsPixmapItem(pixmap)
        cell_x = abs(float(xs[-1] - xs[0])) / max(1, len(xs) - 1)
        cell_y = abs(float(ys[-1] - ys[0])) / max(1, len(ys) - 1)
        item.setTransform(QTransform.fromScale(max(cell_x, 1e-9), max(cell_y, 1e-9)))
        item.setPos(float(xs[0]) - cell_x * 0.5, float(ys[0]) - cell_y * 0.5)
        item.setZValue(Z_HEATMAP_FILL)
        try:
            item.setTransformationMode(Qt.SmoothTransformation)
        except Exception:
            pass
        item.setAcceptedMouseButtons(Qt.NoButton)
        self.view.scene().addItem(item)

    def _draw_heatmap(self, result: SimulationResult):
        """Draw a cached raster heatmap and optional isolines/sample annotations."""
        scene = self.view.scene()
        colours = self._theme_colours()
        if len(result.xs) < 2 or len(result.ys) < 2:
            return
        render_mode = str(
            getattr(result, "render_mode_override", "")
            or getattr(self.heatmap_settings, "heatmap_render_mode", "raster_contours")
            or "raster_contours"
        ).lower()
        if render_mode not in {"raster", "raster_contours", "contours"}:
            render_mode = "raster_contours"
        if render_mode != "raster" and contourpy is None:
            QMessageBox.warning(self, "Missing dependency", "contourpy is required for contour lines. Run: pip install contourpy")
            render_mode = "raster"

        grid = np.asarray(result.rssi, dtype=float)
        rows, cols = grid.shape
        if rows < 2 or cols < 2:
            return

        factor = max(1, int(getattr(self.heatmap_settings, "contour_interpolation_factor", 4)))
        if render_mode in {"raster", "raster_contours"}:
            # The pixmap uses the native RF grid and lets Qt interpolate it on screen;
            # only optional contour lines need the denser numerical display grid.
            self._draw_heatmap_raster(result, np.asarray(result.xs), np.asarray(result.ys), grid)
        if render_mode == "raster":
            return
        xs, ys, z = self._interpolated_heatmap_grid(result, factor)
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

        contour_key = (
            "contour_geometry",
            render_mode,
            factor,
            tuple(levels),
            round(data_min, 6),
            round(data_max, 6),
        )
        contour_geometry = result.render_cache.get(contour_key)
        if contour_geometry is None:
            try:
                cg = contourpy.contour_generator(
                    x=xs,
                    y=ys,
                    z=np.ma.masked_invalid(z),
                    name="serial",
                    line_type=contourpy.LineType.Separate,
                    fill_type=contourpy.FillType.OuterOffset,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Contour generation failed", str(exc))
                return

            filled_bands = []
            if render_mode == "contours":
                bounds = [fill_low] + sorted(levels) + [fill_high]
                for lower, upper in zip(bounds[:-1], bounds[1:]):
                    if upper <= data_min or lower >= data_max:
                        continue
                    lower_c = max(lower, fill_low)
                    upper_c = min(upper, fill_high)
                    if upper_c <= lower_c:
                        continue
                    try:
                        polygons, offsets = cg.filled(lower_c, upper_c)
                    except Exception:
                        continue
                    filled_bands.append((polygons, offsets, (lower + upper) / 2.0))

            line_groups = {}
            for level in levels:
                if level < data_min or level > data_max:
                    continue
                try:
                    line_groups[level] = cg.lines(level)
                except Exception:
                    line_groups[level] = []
            contour_geometry = (filled_bands, line_groups)
            result.render_cache[contour_key] = contour_geometry

        filled_bands, line_groups = contour_geometry
        boundary_clip_path = self._geometry_to_qpath(getattr(result, "boundary_geometry", None))

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
                    for x_value, y_value in ring[1:]:
                        path.lineTo(QPointF(float(x_value), float(y_value)))
                    path.closeSubpath()
            return path

        for polygons, offsets, colour_ref in filled_bands:
            path = path_from_outer_offset(polygons, offsets)
            if not boundary_clip_path.isEmpty():
                path = path.intersected(boundary_clip_path)
            if path.isEmpty():
                continue
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(self.heatmap_settings.colour_for_rssi(colour_ref)))
            item.setPen(QPen(Qt.NoPen))
            item.setZValue(Z_HEATMAP_FILL)
            scene.addItem(item)

        contour_font_size = max(1, int(self.heatmap_settings.contour_label_font_size))
        contour_label_limit = 3
        contour_min_spacing = 35.0
        line_width = max(0.01, float(getattr(self.heatmap_settings, "contour_line_width", 1.25)))
        contour_line_cosmetic = bool(getattr(self.heatmap_settings, "contour_line_cosmetic", True))
        if contour_line_cosmetic:
            line_width = max(1.0, line_width)

        for level in levels:
            lines = line_groups.get(level, [])
            if not lines:
                continue
            line_colour = self.heatmap_settings.contour_line_qcolour(level, bool(getattr(self, "dark_theme", False)))
            pen = QPen(line_colour, line_width)
            pen.setCosmetic(contour_line_cosmetic)
            pen.setStyle(Qt.SolidLine)

            combined_path = QPainterPath()
            label_positions: List[Tuple[float, float]] = []
            labels_on_level = 0
            for line in lines:
                pts = np.asarray(line, dtype=float)
                if len(pts) < 2:
                    continue
                combined_path.moveTo(QPointF(float(pts[0, 0]), float(pts[0, 1])))
                for x_value, y_value in pts[1:]:
                    combined_path.lineTo(QPointF(float(x_value), float(y_value)))

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
                        fraction = (target - acc) / length
                        x_value = pts[i, 0] + (pts[i + 1, 0] - pts[i, 0]) * fraction
                        y_value = pts[i, 1] + (pts[i + 1, 1] - pts[i, 1]) * fraction
                        angle = math.degrees(math.atan2(seg[i, 1], seg[i, 0]))
                        chosen = (float(x_value), float(y_value), angle)
                        break
                    acc += float(length)
                if chosen is None:
                    continue
                x_value, y_value, angle = chosen
                if any(math.hypot(x_value - px, y_value - py) < contour_min_spacing for px, py in label_positions):
                    continue
                self._add_upright_text(
                    scene, f"{level:.0f} dBm", x_value, y_value, colours["contour_text"],
                    contour_font_size, Z_TEXT, bold=True, rotation_deg=angle
                )
                label_positions.append((x_value, y_value))
                labels_on_level += 1

            if not combined_path.isEmpty():
                item = QGraphicsPathItem(combined_path)
                item.setPen(pen)
                item.setBrush(QBrush(Qt.NoBrush))
                item.setZValue(Z_CONTOUR_LINE)
                scene.addItem(item)

        if render_mode != "contours":
            return

        stride_x = max(1, int(self.heatmap_settings.sample_stride_x))
        stride_y = max(1, int(self.heatmap_settings.sample_stride_y))
        sample_font_size = max(1, int(self.heatmap_settings.sample_label_font_size))
        cross_size = max(0.01, float(self.heatmap_settings.sample_cross_size))
        sample_colour = colours["sample_text"]
        sample_cross_colour = colours["sample_cross"]
        sample_pen = QPen(sample_cross_colour, max(1.0, float(getattr(self.heatmap_settings, "sample_cross_line_width", 1.0))))
        sample_pen.setCosmetic(True)
        cross_path = QPainterPath()
        for iy in range(0, rows, stride_y):
            for ix in range(0, cols, stride_x):
                rssi = float(grid[iy, ix])
                if not math.isfinite(rssi):
                    continue
                x_value = float(result.xs[ix])
                y_value = float(result.ys[iy])
                cross_path.moveTo(x_value - cross_size, y_value)
                cross_path.lineTo(x_value + cross_size, y_value)
                cross_path.moveTo(x_value, y_value - cross_size)
                cross_path.lineTo(x_value, y_value + cross_size)
                self._add_upright_text(
                    scene, f"{rssi:.0f} dBm", x_value + cross_size * 1.8, y_value + cross_size * 1.2,
                    sample_colour, sample_font_size, Z_TEXT
                )
        if not cross_path.isEmpty():
            cross_item = QGraphicsPathItem(cross_path)
            cross_item.setPen(sample_pen)
            cross_item.setBrush(QBrush(Qt.NoBrush))
            cross_item.setZValue(Z_SAMPLE_MARK)
            scene.addItem(cross_item)

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
        reflection_text = (
            f"reflections order {self.heatmap_settings.max_reflection_order}"
            if self.heatmap_settings.enable_multipath_reflections
            else "reflections off"
        )
        diffraction_text = "diffraction on" if self.heatmap_settings.enable_corner_diffraction else "diffraction off"
        fading_text = "fading on" if self.heatmap_settings.enable_small_scale_fading else "fading off"
        combination_text = "power sum" if self.heatmap_settings.combined_ap_mode == "power_sum" else "strongest AP"
        boundary_text = (
            "boundary filter on"
            if self.heatmap_settings.ignore_results_outside_planner_boundaries
            else "boundary filter off"
        )
        pieces = [
            f"<b>RSSI isolines</b> &nbsp; "
            f"<span style='color:{legend_text};'>Bands: {band_text} dBm. "
            f"Clients disconnect below {self.heatmap_settings.minimum_client_rssi_dbm:.0f} dBm. <br/> "
            f"Propagation: {reflection_text}, {diffraction_text}, {fading_text}, {combination_text}, {boundary_text}; "
            f"pattern files: {pattern_count}</span>"
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
            self.boundary_result_filter_action.blockSignals(True)
            self.boundary_result_filter_action.setChecked(
                bool(self.heatmap_settings.ignore_results_outside_planner_boundaries)
            )
            self.boundary_result_filter_action.blockSignals(False)
            self._clear_rssi_results()
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
        headers = ["Category", "IFC / RF type", "Material", "Instances floor/project"]
        headers += [self._frequency_label(band) for band in bands]
        headers += ["Source IFC files"]
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

    def _profile_for_ifc_element_from_settings(self, element: IFCElement2D) -> Dict[float, float]:
        if element.rf_category == "door":
            profiles = self.heatmap_settings.default_door_attenuation_by_material_db
        elif element.rf_category == "window":
            profiles = self.heatmap_settings.default_window_attenuation_by_material_db
        else:
            profiles = self.heatmap_settings.default_ifc_element_attenuation_by_type_db
            if not element.rf_customised and element.rf_category in profiles:
                return dict(profiles[element.rf_category])
        override = element.rf_type_override if element.rf_customised else ""
        text = f"{override} {element.rf_category} {element.ifc_class} {element.material} {element.type_name} {element.name}".lower()
        # Prefer the longest matching key so a precise type/material wins over a
        # broad category such as 'metal' or 'default'.
        for key in sorted((k for k in profiles if k != "default"), key=len, reverse=True):
            if key.lower() in text:
                return dict(profiles[key])
        return dict(profiles.get("default", {}))

    # Backwards-compatible name retained for older call sites/extensions.
    def _profile_for_opening_from_settings(self, element: IFCElement2D) -> Dict[float, float]:
        return self._profile_for_ifc_element_from_settings(element)

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
            for element in getattr(floor, "elements", []):
                profile = self._profile_for_ifc_element_from_settings(element)
                for band in bands:
                    if replace_existing or not element.rf_customised or band not in element.attenuation_by_band_db:
                        fallback = element.attenuation_db_for_frequency(band)
                        element.attenuation_by_band_db[band] = float(profile.get(band, fallback))
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
                    ap.ap_type,
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
            if item.column() == 0:
                new_name = item.text().strip()
                if new_name and not any(candidate is not ap and candidate.name == new_name for candidate in self.aps):
                    ap.name = new_name
            elif item.column() == 1:
                if item.text() in AP_TYPE_PRESETS:
                    ap.ap_type = item.text()
            elif item.column() == 2:
                radio.name = item.text().strip() or radio.name
            elif item.column() == 3:
                radio.enabled = item.text().strip().lower() in {"yes", "y", "true", "1", "on", "enabled"}
            elif item.column() == 4:
                if item.text() in self.floors:
                    ap.floor = item.text()
            elif item.column() == 5:
                ap.x = float(item.text())
            elif item.column() == 6:
                ap.y = float(item.text())
            elif item.column() == 7:
                if item.text() in self.antenna_patterns:
                    radio.antenna_pattern = item.text()
            elif item.column() == 8:
                ap.azimuth_deg = float(item.text())
            elif item.column() == 9:
                ap.downtilt_deg = float(item.text())
            elif item.column() == 10:
                radio.tx_power_dbm = float(item.text())
            elif item.column() == 11:
                radio.antenna_gain_dbi = float(item.text())
            elif item.column() == 12:
                radio.frequency_mhz = float(item.text())
                self._refresh_rssi_frequency_dropdown()
            elif item.column() == 13:
                radio.channel = item.text().strip()
            elif item.column() == 14:
                radio.channel_width_mhz = max(0.01, float(item.text()))
            elif item.column() == 15:
                radio.spectrum_occupancy_percent = max(0.0, min(100.0, float(item.text())))
            elif item.column() == 16:
                ap.max_clients = max(1, int(float(item.text())))
            if item.column() in {2, 3, 7, 10, 11, 12, 13, 14, 15}:
                ap.radio_profile = "Custom"
            # Keep legacy AP fields in sync with the first radio for older code/export.
            if ap.radios:
                ap.tx_power_dbm = float(ap.radios[0].tx_power_dbm)
                ap.frequency_mhz = float(ap.radios[0].frequency_mhz)
                ap.antenna_pattern = ap.radios[0].antenna_pattern
        except (ValueError, IndexError):
            return
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.draw_floor()
        self.populate_ap_table()

    def populate_wall_table(self):
        self.wall_table.blockSignals(True)
        self._configure_wall_table_headers()
        self.wall_table.setRowCount(0)
        if not self.floor:
            self.wall_table.blockSignals(False)
            return
        bands = self._frequency_bands()
        for group in self._attenuation_groups():
            if int(group.get("current_floor_count", 0)) <= 0:
                continue
            row = self.wall_table.rowCount()
            self.wall_table.insertRow(row)
            key_text = json.dumps(list(group.get("key", ())), separators=(",", ":"))
            category_item = QTableWidgetItem(str(group.get("category", "Other")))
            category_item.setData(Qt.UserRole, key_text)
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsEditable)
            self.wall_table.setItem(row, 0, category_item)
            type_item = QTableWidgetItem(str(group.get("display_type", "Unknown")))
            type_item.setData(Qt.UserRole, key_text)
            self.wall_table.setItem(row, 1, type_item)
            material_item = QTableWidgetItem(str(group.get("material", "")))
            material_item.setData(Qt.UserRole, key_text)
            material_item.setFlags(material_item.flags() & ~Qt.ItemIsEditable)
            self.wall_table.setItem(row, 2, material_item)
            count_item = QTableWidgetItem(f"{group.get('current_floor_count', 0)}/{group.get('count', 0)}")
            count_item.setData(Qt.UserRole, key_text)
            count_item.setFlags(count_item.flags() & ~Qt.ItemIsEditable)
            self.wall_table.setItem(row, 3, count_item)
            attenuation = dict(group.get("attenuation", {}) or {})
            for offset, band in enumerate(bands):
                item = QTableWidgetItem(f"{float(attenuation.get(float(band), 0.0)):.3f}")
                item.setData(Qt.UserRole, key_text)
                item.setData(Qt.UserRole + 1, float(band))
                self.wall_table.setItem(row, 4 + offset, item)
            source_item = QTableWidgetItem(str(group.get("sources", "")))
            source_item.setData(Qt.UserRole, key_text)
            source_item.setFlags(source_item.flags() & ~Qt.ItemIsEditable)
            self.wall_table.setItem(row, 4 + len(bands), source_item)
        self.wall_table.resizeColumnsToContents()
        self.wall_table.blockSignals(False)

    def _wall_table_changed(self, item: QTableWidgetItem):
        if not self.floor:
            return
        try:
            key = tuple(json.loads(str(item.data(Qt.UserRole))))
        except Exception:
            return
        objects = self._objects_for_attenuation_group(key, current_floor_only=False)
        if not objects:
            return
        bands = self._frequency_bands()
        if item.column() == 1:
            rf_type = item.text().strip()
            if not rf_type:
                return
            for obj in objects:
                obj.rf_type_override = rf_type
                obj.rf_customised = True
        elif 4 <= item.column() < 4 + len(bands):
            try:
                value = float(item.text())
                band = float(item.data(Qt.UserRole + 1))
            except Exception:
                return
            for obj in objects:
                obj.attenuation_by_band_db[band] = value
                obj.rf_customised = True
        else:
            return
        self._invalidate_interactive_preview_requests()
        self.last_result = None
        self.rssi_results_by_frequency = {}
        self.draw_floor()
        self.statusBar().showMessage(f"Updated {len(objects)} matching IFC attenuation instance(s) across the project")

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
        self._invalidate_interactive_preview_requests()
        self._rssi_result_stale = False
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

        progressive_state = {"fraction": 0.0}

        def update_progressive_heatmap(partial_results, fraction):
            if not bool(getattr(self.heatmap_settings, "progressive_heatmap_updates", True)):
                return
            fraction = max(0.0, min(1.0, float(fraction)))
            if fraction < 1.0 and fraction - progressive_state["fraction"] < 0.10:
                return
            progressive_state["fraction"] = fraction
            converted = {float(key): value for key, value in partial_results.items()}
            if not converted:
                return
            for value in converted.values():
                value.render_mode_override = "raster"
            self.rssi_results_by_frequency = converted
            selected = self._selected_rssi_view_frequency()
            self.last_result = converted.get(selected) or converted[sorted(converted)[0]]
            self._preserve_view_on_redraw = True
            self.draw_floor()
            progress.setLabelText(f"Calculating RSSI heatmap... {int(fraction * 100)}% field available")
            QApplication.processEvents()

        self.rssi_results_by_frequency = {}

        active_freqs = sorted(
            {float(r.frequency_mhz) for ap in self.aps for r in ap.active_radios()}
        )

        if not active_freqs:
            progress.close()
            QMessageBox.information(
                self,
                "No active radios",
                "No enabled AP radios are available for RSSI calculation.",
            )
            return

        self.statusBar().showMessage(
            f"Calculating {len(active_freqs)} RSSI band(s) with shared worker geometry..."
        )
        try:
            self.rssi_results_by_frequency = RFEngine.simulate_frequencies(
                self.floor,
                self.floors,
                self.aps,
                active_freqs,
                self.resolution.value(),
                self.antenna_patterns,
                include_inter_floor=self.include_inter_floor.isChecked(),
                heatmap_settings=self.heatmap_settings,
                progress_callback=update_progress,
                calculation_boundary=self._rssi_calculation_boundary(),
                progressive_callback=update_progressive_heatmap,
            )
        except Exception as exc:
            progress.close()
            if "cancel" in str(exc).lower():
                self.statusBar().showMessage("RSSI calculation cancelled")
                return
            QMessageBox.warning(self, "RSSI calculation failed", str(exc))
            self.statusBar().showMessage("RSSI calculation failed")
            return
        finally:
            progress.close()

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
            self._invalidate_interactive_preview_requests()
            self.last_result = None

        self._rssi_result_stale = False
        self.statusBar().showMessage("Rendering RSSI heatmap...")
        QApplication.processEvents()
        self.draw_floor()
        if self.last_result is not None:
            mode = self.last_result.execution_mode
            workers = max(1, int(self.last_result.worker_processes))
            elapsed = max(float(result.elapsed_seconds) for result in self.rssi_results_by_frequency.values())
            ignored = max(int(result.ignored_point_count) for result in self.rssi_results_by_frequency.values())
            boundary_text = (
                f" {ignored} grid point{'s' if ignored != 1 else ''} outside the shared boundary were ignored."
                if ignored else ""
            )
            exact_points = max(int(result.exact_points) for result in self.rssi_results_by_frequency.values())
            approximate_points = max(int(result.approximate_points) for result in self.rssi_results_by_frequency.values())
            cache_hits = max(int(result.cache_hits) for result in self.rssi_results_by_frequency.values())
            cache_misses = max(int(result.cache_misses) for result in self.rssi_results_by_frequency.values())
            adaptive_text = (
                f" {exact_points:,} exact and {approximate_points:,} interpolated points."
                if approximate_points else ""
            )
            cache_text = (
                f" AP-field cache: {cache_hits} hit(s), {cache_misses} recalculated."
                if cache_hits or cache_misses else ""
            )
            self.statusBar().showMessage(
                f"RSSI calculation complete in {elapsed:.1f} s using {mode} "
                f"({workers} worker{'s' if workers != 1 else ''})."
                f"{adaptive_text}{cache_text}{boundary_text} "
                f"Hover over the heatmap to inspect RSSI, path count and RMS delay spread."
            )
            fallback = next((
                result.performance_note for result in self.rssi_results_by_frequency.values()
                if result.execution_mode == "single-process-fallback"
            ), "")
            if fallback:
                QMessageBox.warning(self, "RF multiprocessing fallback", fallback)

    def closeEvent(self, event):
        self._invalidate_interactive_preview_requests()
        self._interactive_preview_poll_timer.stop()
        self._interactive_preview_thread = None
        self._shutdown_ifc_process_executor()
        _shutdown_rf_process_executor(wait=False)
        super().closeEvent(event)

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

    @staticmethod
    def _pdf_frequency_label(frequency_mhz: float) -> str:
        frequency_mhz = float(frequency_mhz)
        if frequency_mhz >= 1000.0:
            return f"{frequency_mhz / 1000.0:g} GHz"
        return f"{frequency_mhz:g} MHz"

    def _draw_pdf_rssi_legend(
        self,
        painter: QPainter,
        rect: QRectF,
        frequency_mhz: float,
        floor: FloorModel,
        result: Optional[SimulationResult],
        frequency_ap_count: int,
        floor_ap_count: int,
    ):
        painter.save()
        painter.setPen(QPen(QColor("#6B7280"), 1.0))
        painter.setBrush(QBrush(QColor("#F8FAFC")))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        pad = max(12.0, rect.width() * 0.045)
        x = rect.left() + pad
        y = rect.top() + pad
        width = max(40.0, rect.width() - (2.0 * pad))

        title_font = QFont(painter.font())
        title_font.setBold(True)
        title_font.setPointSize(12)
        painter.setFont(title_font)
        painter.setPen(QColor("#111827"))
        painter.drawText(QRectF(x, y, width, 34.0), Qt.AlignLeft | Qt.AlignVCenter, "RSSI legend")
        y += 38.0

        body_font = QFont(painter.font())
        body_font.setBold(False)
        body_font.setPointSize(8)
        painter.setFont(body_font)

        zone_row_height = 30.0
        swatch_width = 34.0
        for zone in self.heatmap_settings.zones:
            if y + zone_row_height > rect.bottom() - 175.0:
                break
            colour = QColor(zone.colour)
            if not colour.isValid():
                colour = QColor("#555555")
            colour.setAlpha(255)
            swatch = QRectF(x, y + 4.0, swatch_width, zone_row_height - 8.0)
            painter.fillRect(swatch, colour)
            painter.setPen(QPen(QColor("#4B5563"), 0.8))
            painter.drawRect(swatch)
            painter.setPen(QColor("#111827"))
            label = f"{zone.name}: {zone.min_dbm:.0f} to {zone.max_dbm:.0f} dBm"
            painter.drawText(
                QRectF(x + swatch_width + 9.0, y, width - swatch_width - 9.0, zone_row_height),
                Qt.AlignLeft | Qt.AlignVCenter,
                label,
            )
            y += zone_row_height

        y += 8.0
        painter.setPen(QPen(QColor("#CBD5E1"), 0.8))
        painter.drawLine(QPointF(x, y), QPointF(x + width, y))
        y += 12.0
        painter.setPen(QColor("#1F2937"))

        reflection_text = (
            f"order {self.heatmap_settings.max_reflection_order} reflections"
            if self.heatmap_settings.enable_multipath_reflections
            else "reflections off"
        )
        propagation_lines = [
            f"Floor: {floor.name}",
            f"Frequency: {self._pdf_frequency_label(frequency_mhz)}",
            f"Resolution: {float(self.resolution.value()):g} m",
            f"APs on floor: {floor_ap_count}",
            f"APs on frequency: {frequency_ap_count}",
            f"Inter-floor RF: {'included' if self.include_inter_floor.isChecked() else 'excluded'}",
            f"Model: {reflection_text}",
            f"Diffraction: {'on' if self.heatmap_settings.enable_corner_diffraction else 'off'}",
            f"Small-scale fading: {'on' if self.heatmap_settings.enable_small_scale_fading else 'off'}",
            f"AP combination: {'power sum' if self.heatmap_settings.combined_ap_mode == 'power_sum' else 'strongest AP'}",
            f"Boundary result filter: {'applied' if result is not None and result.boundary_geometry is not None else 'not applied'}",
        ]

        if result is not None:
            values = np.asarray(result.rssi, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                threshold = float(self.heatmap_settings.minimum_client_rssi_dbm)
                covered = float(np.count_nonzero(finite >= threshold)) * 100.0 / float(finite.size)
                propagation_lines.extend([
                    f"RSSI min / mean / max: {float(np.min(finite)):.1f} / {float(np.mean(finite)):.1f} / {float(np.max(finite)):.1f} dBm",
                    f"Area above {threshold:.0f} dBm: {covered:.1f}%",
                ])
        else:
            propagation_lines.append("RSSI result: unavailable")

        painter.drawText(
            QRectF(x, y, width, max(20.0, rect.bottom() - y - pad)),
            Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
            "\n".join(propagation_lines),
        )
        painter.restore()

    def export_floor_pdf(self):
        if not self.floors:
            QMessageBox.information(self, "No floors", "Load an IFC model before exporting an RSSI floor PDF.")
            return
        if not self.aps:
            QMessageBox.information(self, "No APs", "Place or predict at least one access point before exporting an RSSI floor PDF.")
            return

        self._sync_slab_attenuation_from_ui()
        for ap in self.aps:
            ap.path_loss_exponent = float(self.ple.value())
            ap.rx_height_m = float(self.rx_height.value())

        active_frequencies = sorted({
            float(radio.frequency_mhz)
            for ap in self.aps
            for radio in ap.active_radios()
            if getattr(radio, "enabled", True)
        })
        if not active_frequencies:
            QMessageBox.information(self, "No active radios", "No enabled AP radios are available for PDF export.")
            return

        frequency_mhz = self._selected_rssi_view_frequency()
        if frequency_mhz not in active_frequencies:
            frequency_mhz = active_frequencies[0]

        aps_for_frequency: List[AccessPoint] = []
        for ap in self.aps:
            radios = [
                radio for radio in ap.active_radios()
                if getattr(radio, "enabled", True)
                and abs(float(radio.frequency_mhz) - float(frequency_mhz)) < 1e-6
            ]
            if radios:
                aps_for_frequency.append(replace(ap, radios=radios))

        if not aps_for_frequency:
            QMessageBox.information(
                self,
                "No APs on selected frequency",
                f"No enabled access-point radios use {self._pdf_frequency_label(frequency_mhz)}.",
            )
            return

        frequency_name = f"{float(frequency_mhz):g}".replace(".", "_")
        default_name = f"rssi_floor_report_{frequency_name}MHz.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export RSSI floor report",
            default_name,
            "PDF files (*.pdf)",
        )
        if not path:
            return
        target_path = Path(path)
        if target_path.suffix.lower() != ".pdf":
            target_path = target_path.with_suffix(".pdf")
        temporary_path = target_path.with_name(f".{target_path.stem}.{uuid.uuid4().hex}.tmp.pdf")

        floor_items = sorted(self.floors.items(), key=lambda item: (item[1].elevation, item[0]))
        progress = QProgressDialog(
            "Calculating RSSI for all floors...",
            "Cancel",
            0,
            max(1, len(floor_items) * 110),
            self,
        )
        progress.setWindowTitle("Export RSSI floor PDF")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        results_by_floor: Dict[str, Optional[SimulationResult]] = {}
        try:
            for floor_index, (floor_name, floor) in enumerate(floor_items):
                progress.setLabelText(
                    f"Calculating {self._pdf_frequency_label(frequency_mhz)} RSSI for {floor_name} "
                    f"({floor_index + 1}/{len(floor_items)})..."
                )

                def update_progress(done, total, base=floor_index * 100):
                    if total > 0:
                        progress.setValue(base + int((float(done) / float(total)) * 100.0))
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        raise RuntimeError("RSSI floor PDF export cancelled")

                floor_results = RFEngine.simulate_frequencies(
                    floor,
                    self.floors,
                    aps_for_frequency,
                    [float(frequency_mhz)],
                    self.resolution.value(),
                    self.antenna_patterns,
                    include_inter_floor=self.include_inter_floor.isChecked(),
                    heatmap_settings=self.heatmap_settings,
                    progress_callback=update_progress,
                    calculation_boundary=self._rssi_calculation_boundary(),
                )
                results_by_floor[floor_name] = floor_results.get(float(frequency_mhz))
                progress.setValue((floor_index + 1) * 100)

            progress.setLabelText("Rendering floor pages to PDF...")
            QApplication.processEvents()
            if progress.wasCanceled():
                raise RuntimeError("RSSI floor PDF export cancelled")

            writer = QPdfWriter(str(temporary_path))
            writer.setCreator("RF Attenuation Simulator")
            writer.setTitle(
                f"RSSI floor report - {self._pdf_frequency_label(frequency_mhz)}"
            )
            writer.setResolution(150)
            writer.setPageSize(QPageSize(QPageSize.PageSizeId.A3))
            writer.setPageOrientation(QPageLayout.Orientation.Landscape)
            writer.setPageMargins(QMarginsF(8.0, 8.0, 8.0, 8.0), QPageLayout.Unit.Millimeter)

            painter = QPainter(writer)
            if not painter.isActive():
                raise RuntimeError("Qt could not initialise the PDF painter.")
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            original_floor = self.floor
            original_last_result = self.last_result
            original_results = self.rssi_results_by_frequency
            original_preserve = self._preserve_view_on_redraw
            original_ruler = self.ap_ruler_enabled
            original_transform = self.view.transform()
            original_h = self.view.horizontalScrollBar().value()
            original_v = self.view.verticalScrollBar().value()

            self.setUpdatesEnabled(False)
            self.ap_ruler_enabled = False
            try:
                for page_index, (floor_name, floor) in enumerate(floor_items):
                    if page_index > 0 and not writer.newPage():
                        raise RuntimeError(f"Could not create PDF page {page_index + 1}.")
                    if progress.wasCanceled():
                        raise RuntimeError("RSSI floor PDF export cancelled")

                    result = results_by_floor.get(floor_name)
                    self.floor = floor
                    self.last_result = result
                    self._preserve_view_on_redraw = True
                    self.draw_floor()
                    scene = self.view.scene()
                    scene.clearSelection()

                    page_rect = QRectF(writer.pageLayout().paintRectPixels(writer.resolution()))
                    painter.fillRect(page_rect, QColor("#FFFFFF"))
                    margin = max(20.0, page_rect.width() * 0.012)
                    header_height = max(70.0, page_rect.height() * 0.065)
                    footer_height = max(34.0, page_rect.height() * 0.025)
                    legend_width = max(320.0, page_rect.width() * 0.205)
                    plan_rect = QRectF(
                        page_rect.left() + margin,
                        page_rect.top() + header_height,
                        page_rect.width() - legend_width - (3.0 * margin),
                        page_rect.height() - header_height - footer_height - margin,
                    )
                    legend_rect = QRectF(
                        plan_rect.right() + margin,
                        plan_rect.top(),
                        legend_width,
                        plan_rect.height(),
                    )

                    title_font = QFont(painter.font())
                    title_font.setBold(True)
                    title_font.setPointSize(17)
                    painter.setFont(title_font)
                    painter.setPen(QColor("#111827"))
                    painter.drawText(
                        QRectF(
                            page_rect.left() + margin,
                            page_rect.top() + 4.0,
                            page_rect.width() - (2.0 * margin),
                            header_height * 0.58,
                        ),
                        Qt.AlignLeft | Qt.AlignVCenter,
                        f"RF Attenuation Simulator - {floor_name}",
                    )

                    subtitle_font = QFont(painter.font())
                    subtitle_font.setBold(False)
                    subtitle_font.setPointSize(9)
                    painter.setFont(subtitle_font)
                    subtitle = (
                        f"RSSI heatmap at {self._pdf_frequency_label(frequency_mhz)} | "
                        f"Floor elevation {float(floor.elevation):g} m | "
                        f"Grid resolution {float(self.resolution.value()):g} m"
                    )
                    painter.setPen(QColor("#4B5563"))
                    painter.drawText(
                        QRectF(
                            page_rect.left() + margin,
                            page_rect.top() + (header_height * 0.52),
                            page_rect.width() - (2.0 * margin),
                            header_height * 0.42,
                        ),
                        Qt.AlignLeft | Qt.AlignTop,
                        subtitle,
                    )

                    painter.fillRect(plan_rect, self._theme_colours()["background"])
                    source_rect = scene.itemsBoundingRect()
                    if result is not None and result.xs.size and result.ys.size:
                        result_rect = QRectF(
                            float(result.xs[0]),
                            float(result.ys[0]),
                            max(0.01, float(result.xs[-1] - result.xs[0])),
                            max(0.01, float(result.ys[-1] - result.ys[0])),
                        )
                        source_rect = source_rect.united(result_rect)
                    if source_rect.isEmpty():
                        source_rect = QRectF(0.0, 0.0, 1.0, 1.0)
                    pad = max(source_rect.width(), source_rect.height()) * 0.025
                    source_rect = source_rect.adjusted(-pad, -pad, pad, pad)

                    painter.save()
                    painter.setClipRect(plan_rect)
                    scene.render(painter, plan_rect, source_rect, Qt.KeepAspectRatio)
                    painter.restore()
                    painter.setPen(QPen(QColor("#94A3B8"), 1.0))
                    painter.setBrush(QBrush(Qt.NoBrush))
                    painter.drawRect(plan_rect)

                    if result is None:
                        warning_font = QFont(painter.font())
                        warning_font.setBold(True)
                        warning_font.setPointSize(13)
                        painter.setFont(warning_font)
                        painter.setPen(QColor("#B91C1C"))
                        painter.drawText(
                            plan_rect,
                            Qt.AlignCenter | Qt.TextWordWrap,
                            "No RSSI grid was produced for this floor.",
                        )

                    floor_ap_count = sum(
                        1 for ap in aps_for_frequency if ap.floor == floor_name
                    )
                    self._draw_pdf_rssi_legend(
                        painter,
                        legend_rect,
                        frequency_mhz,
                        floor,
                        result,
                        len(aps_for_frequency),
                        floor_ap_count,
                    )

                    footer_font = QFont(painter.font())
                    footer_font.setPointSize(8)
                    painter.setFont(footer_font)
                    painter.setPen(QColor("#64748B"))
                    painter.drawText(
                        QRectF(
                            page_rect.left() + margin,
                            page_rect.bottom() - footer_height,
                            page_rect.width() - (2.0 * margin),
                            footer_height,
                        ),
                        Qt.AlignRight | Qt.AlignVCenter,
                        f"Page {page_index + 1} of {len(floor_items)}",
                    )
                    progress.setValue((len(floor_items) * 100) + ((page_index + 1) * 10))
                    QApplication.processEvents()
            finally:
                painter.end()
                self.floor = original_floor
                self.last_result = original_last_result
                self.rssi_results_by_frequency = original_results
                self.ap_ruler_enabled = original_ruler
                self._preserve_view_on_redraw = True
                self.draw_floor()
                self.view.setTransform(original_transform)
                self.view.horizontalScrollBar().setValue(original_h)
                self.view.verticalScrollBar().setValue(original_v)
                self._preserve_view_on_redraw = original_preserve
                self.setUpdatesEnabled(True)
                self.update()

            del writer
            os.replace(str(temporary_path), str(target_path))
            progress.setValue(progress.maximum())
            self.statusBar().showMessage(
                f"Exported {len(floor_items)} RSSI floor page(s) to {target_path}"
            )
            QMessageBox.information(
                self,
                "RSSI floor PDF exported",
                f"Exported {len(floor_items)} floor page(s) at "
                f"{self._pdf_frequency_label(frequency_mhz)} to:\n{target_path}",
            )
        except Exception as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
            if "cancelled" in str(exc).lower():
                self.statusBar().showMessage("RSSI floor PDF export cancelled")
            else:
                QMessageBox.warning(self, "PDF export failed", str(exc))
        finally:
            progress.close()

    def export_csv(self):
        if not self.last_result:
            QMessageBox.information(self, "No result", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export RSSI CSV", "rssi_heatmap.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["floor", "x", "y", "simulation_mode", "active_radio_frequencies_mhz", "rssi_dbm", "rms_delay_spread_ns", "contributing_path_count", "rssi_zone", "client_connected", "disconnect_threshold_dbm", "ap_count", "include_inter_floor", "slab_loss_24_db", "slab_loss_5_db", "slab_loss_6_db", "patterns_used"])
            for iy, y in enumerate(self.last_result.ys):
                for ix, x in enumerate(self.last_result.xs):
                    rssi = float(self.last_result.rssi[iy, ix])
                    if not math.isfinite(rssi):
                        continue
                    zone = self.heatmap_settings.zone_for_rssi(rssi)
                    active_radios = [r for a in self.aps for r in a.active_radios()]
                    delay = float(self.last_result.delay_spread_ns[iy, ix]) if self.last_result.delay_spread_ns is not None else 0.0
                    path_count = int(self.last_result.path_count[iy, ix]) if self.last_result.path_count is not None else 1
                    writer.writerow([self.floor.name if self.floor else "", x, y, self.heatmap_settings.combined_ap_mode, ";".join(str(int(r.frequency_mhz)) for r in active_radios), rssi, delay, path_count, zone.name, rssi >= self.heatmap_settings.minimum_client_rssi_dbm, self.heatmap_settings.minimum_client_rssi_dbm, len(self.aps), self.include_inter_floor.isChecked(), float(self.slab_att_24.value()), float(self.slab_att_5.value()), float(self.slab_att_6.value()), ";".join(sorted({r.antenna_pattern for r in active_radios}))])


def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(lambda: _shutdown_rf_process_executor(wait=False))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
