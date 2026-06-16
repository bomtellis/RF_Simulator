"""
DXF pre-alignment workflow for RF Attenuation Simulator.

Adds a separate DXF alignment window before inserting the DXF into the IFC scene.
Workflow:
    1. User selects two IFC reference points in the main IFC view, with snap.
    2. User opens a DXF in a separate alignment dialog.
    3. User selects two matching DXF points, with endpoint/corner snap.
    4. The module calculates unit conversion, rotation, scale and translation.
    5. The corrected DXF overlay can then be inserted into the IFC scene.

This module is intentionally standalone so the main rf_simulator.py remains smaller.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, hypot, radians, degrees
from pathlib import Path
from typing import Iterable, Optional

try:
    import ezdxf
except Exception:  # pragma: no cover
    ezdxf = None

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen, QBrush, QTransform
from PySide6.QtWidgets import (
    QDialog, QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget, QCheckBox, QDoubleSpinBox,
)


DXF_INSUNITS_TO_METRES = {
    0: 1.0,          # unitless, assume metres unless overridden
    1: 0.0254,       # inches
    2: 0.3048,       # feet
    3: 1609.344,     # miles
    4: 0.001,        # millimetres
    5: 0.01,         # centimetres
    6: 1.0,          # metres
    7: 1000.0,       # kilometres
    8: 2.54e-8,      # microinches
    9: 2.54e-5,      # mils
    10: 0.9144,      # yards
    11: 1e-10,       # angstroms
    12: 1e-9,        # nanometres
    13: 1e-6,        # microns
    14: 0.1,         # decimetres
    15: 10.0,        # decametres
    16: 100.0,       # hectometres
    17: 1e9,         # gigametres
}


@dataclass
class SimilarityTransform2D:
    scale: float = 1.0
    rotation_rad: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        c = cos(self.rotation_rad)
        s = sin(self.rotation_rad)
        return (
            self.scale * (c * x - s * y) + self.tx,
            self.scale * (s * x + c * y) + self.ty,
        )

    def to_qtransform(self) -> QTransform:
        c = cos(self.rotation_rad)
        s = sin(self.rotation_rad)
        return QTransform(
            self.scale * c, self.scale * s, 0.0,
            -self.scale * s, self.scale * c, 0.0,
            self.tx, self.ty, 1.0,
        )


def two_point_transform(
    src_a: QPointF, src_b: QPointF,
    dst_a: QPointF, dst_b: QPointF,
    source_unit_to_ifc_unit: float = 1.0,
) -> SimilarityTransform2D:
    """Return transform mapping source/DXF points onto destination/IFC points."""
    sx1, sy1 = src_a.x() * source_unit_to_ifc_unit, src_a.y() * source_unit_to_ifc_unit
    sx2, sy2 = src_b.x() * source_unit_to_ifc_unit, src_b.y() * source_unit_to_ifc_unit
    dx1, dy1 = dst_a.x(), dst_a.y()
    dx2, dy2 = dst_b.x(), dst_b.y()

    svx, svy = sx2 - sx1, sy2 - sy1
    dvx, dvy = dx2 - dx1, dy2 - dy1
    slen = hypot(svx, svy)
    dlen = hypot(dvx, dvy)
    if slen <= 1e-9:
        raise ValueError("The two DXF alignment points are too close together.")

    scale = dlen / slen
    rot = atan2(dvy, dvx) - atan2(svy, svx)
    c = cos(rot)
    s = sin(rot)
    tx = dx1 - scale * (c * sx1 - s * sy1)
    ty = dy1 - scale * (s * sx1 + c * sy1)
    return SimilarityTransform2D(scale=scale, rotation_rad=rot, tx=tx, ty=ty)


class SnapGraphicsView(QGraphicsView):
    pointPicked = Signal(QPointF)

    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None):
        super().__init__(scene, parent)
        self.snap_enabled = True
        self.snap_radius_px = 14
        self.snap_points: list[QPointF] = []
        self._middle_panning = False
        self._last_pan_pos = None

        self._snap_marker = QGraphicsEllipseItem(-0.1, -0.1, 0.2, 0.2)
        self._snap_marker.setPen(QPen(QColor("#00AEEF"), 0))
        self._snap_marker.setBrush(QBrush(QColor(0, 174, 239, 120)))
        self._snap_marker.setZValue(10_000)
        scene.addItem(self._snap_marker)
        self._snap_marker.hide()

        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(1, -1)

    def ensure_snap_marker(self):
        try:
            self._snap_marker.setVisible(self._snap_marker.isVisible())
            return
        except RuntimeError:
            pass

        self._snap_marker = QGraphicsEllipseItem(-0.1, -0.1, 0.2, 0.2)
        self._snap_marker.setPen(QPen(QColor("#00AEEF"), 0))
        self._snap_marker.setBrush(QBrush(QColor(0, 174, 239, 120)))
        self._snap_marker.setZValue(10_000)
        self.scene().addItem(self._snap_marker)
        self._snap_marker.hide()

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def set_snap_points(self, points: Iterable[QPointF]):
        self.snap_points = list(points)

    def _nearest_snap(self, scene_pos: QPointF) -> QPointF:
        if not self.snap_enabled or not self.snap_points:
            return scene_pos
        px_radius_scene = self.mapToScene(self.snap_radius_px, 0).x() - self.mapToScene(0, 0).x()
        radius = abs(px_radius_scene)
        best = None
        best_d = None
        for p in self.snap_points:
            d = hypot(p.x() - scene_pos.x(), p.y() - scene_pos.y())
            if d <= radius and (best_d is None or d < best_d):
                best = p
                best_d = d
        return best if best is not None else scene_pos

    def mouseMoveEvent(self, event):
        if self._middle_panning and self._last_pan_pos is not None:
            delta = event.position().toPoint() - self._last_pan_pos
            self._last_pan_pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        self.ensure_snap_marker()
        pos = self.mapToScene(event.position().toPoint())
        snap = self._nearest_snap(pos)
        self._snap_marker.setPos(snap)
        self._snap_marker.show()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._middle_panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self.pointPicked.emit(self._nearest_snap(pos))
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


def _add_text(scene: QGraphicsScene, text: str, x: float, y: float, colour: str = "#0078d4"):
    item = QGraphicsSimpleTextItem(text)
    item.setBrush(QBrush(QColor(colour)))
    item.setPos(x, y)
    item.setScale(0.05)
    item.setZValue(10_001)
    scene.addItem(item)
    return item


def read_dxf_geometry(path: str | Path) -> tuple[list[QPainterPath], list[QPointF], float]:
    """Read basic DXF line/polyline/circle geometry and return paths, snap points and unit scale to metres."""
    if ezdxf is None:
        raise RuntimeError("ezdxf is required for DXF alignment. Install with: pip install ezdxf")
    doc = ezdxf.readfile(str(path))
    insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    unit_to_m = DXF_INSUNITS_TO_METRES.get(insunits, 1.0)
    msp = doc.modelspace()
    paths: list[QPainterPath] = []
    snap_points: list[QPointF] = []

    def pt2(p):
        return QPointF(float(p[0]), float(p[1]))

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
        typ = e.dxftype()
        try:
            if typ == "LINE":
                a = pt2(e.dxf.start)
                b = pt2(e.dxf.end)
                p = QPainterPath(a)
                p.lineTo(b)
                paths.append(p)
                snap_points.extend([a, b])
            elif typ in ("LWPOLYLINE", "POLYLINE"):
                pts = []
                if typ == "LWPOLYLINE":
                    pts = [QPointF(float(x), float(y)) for x, y, *_ in e.get_points()]
                else:
                    pts = [pt2(v.dxf.location) for v in e.vertices]
                if len(pts) >= 2:
                    p = QPainterPath(pts[0])
                    for q in pts[1:]:
                        p.lineTo(q)
                    if getattr(e, "closed", False):
                        p.closeSubpath()
                    paths.append(p)
                    snap_points.extend(pts)
            elif typ == "CIRCLE":
                c = pt2(e.dxf.center)
                r = float(e.dxf.radius)
                p = QPainterPath()
                p.addEllipse(c, r, r)
                paths.append(p)
                snap_points.extend([QPointF(c.x() + r, c.y()), QPointF(c.x() - r, c.y()), QPointF(c.x(), c.y() + r), QPointF(c.x(), c.y() - r), c])
            elif typ == "ARC":
                c = pt2(e.dxf.center)
                r = float(e.dxf.radius)
                p = QPainterPath()
                p.addEllipse(c, r, r)  # preview full circle for snap/reference simplicity
                paths.append(p)
                snap_points.append(c)
            elif typ == "SPLINE":
                pts = [pt2(p) for p in e.flattening(0.05)]
                if len(pts) >= 2:
                    p = QPainterPath(pts[0])
                    for q in pts[1:]:
                        p.lineTo(q)
                    paths.append(p)
                    snap_points.extend([pts[0], pts[-1]])

            elif typ == "ELLIPSE":
                pts = [pt2(p) for p in e.flattening(0.05)]
                if len(pts) >= 2:
                    p = QPainterPath(pts[0])
                    for q in pts[1:]:
                        p.lineTo(q)
                    paths.append(p)
                    snap_points.extend([pts[0], pts[-1]])
        except Exception:
            continue
    return paths, snap_points, unit_to_m


class DxfPreAlignDialog(QDialog):
    """Separate DXF window used before inserting corrected DXF into the IFC scene."""

    alignmentReady = Signal(object)

    def __init__(self, dxf_path: str, ifc_point_a: QPointF, ifc_point_b: QPointF, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Align DXF to selected IFC points")
        self.resize(1100, 800)
        self.dxf_path = dxf_path
        self.ifc_a = ifc_point_a
        self.ifc_b = ifc_point_b
        self.dxf_points: list[QPointF] = []
        self.transform_result: Optional[SimilarityTransform2D] = None

        self.scene = QGraphicsScene(self)
        self.view = SnapGraphicsView(self.scene, self)
        self.view.pointPicked.connect(self._picked)

        self.status = QLabel("Pick DXF point 1 matching IFC point 1. Middle mouse drag pans the DXF view.")
        self.snap_check = QCheckBox("Snap to DXF endpoints/corners")
        self.snap_check.setChecked(True)
        self.snap_check.toggled.connect(lambda v: setattr(self.view, "snap_enabled", bool(v)))

        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-360.0, 360.0)
        self.rotation_spin.setDecimals(4)
        self.rotation_spin.setSingleStep(0.25)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setToolTip("Manual DXF rotation correction applied before the two-point alignment calculation.")
        self.rotation_spin.valueChanged.connect(self._manual_rotation_changed)

        self.rotate_left_btn = QPushButton("⟲ 90°")
        self.rotate_right_btn = QPushButton("90° ⟳")
        self.rotate_left_btn.clicked.connect(lambda: self.rotation_spin.setValue(self.rotation_spin.value() - 90.0))
        self.rotate_right_btn.clicked.connect(lambda: self.rotation_spin.setValue(self.rotation_spin.value() + 90.0))

        self.apply_btn = QPushButton("Apply corrected DXF to IFC scene")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        self.clear_btn = QPushButton("Clear picked DXF points")
        self.clear_btn.clicked.connect(self._clear)

        top = QHBoxLayout()
        top.addWidget(self.status, 1)
        top.addWidget(self.snap_check)
        top.addWidget(QLabel("DXF rotation"))
        top.addWidget(self.rotation_spin)
        top.addWidget(self.rotate_left_btn)
        top.addWidget(self.rotate_right_btn)
        top.addWidget(self.clear_btn)
        top.addWidget(self.apply_btn)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.view, 1)

        self._load_dxf()

    def _rotation_rad(self) -> float:
        return radians(float(self.rotation_spin.value()))

    def _rotate_point_for_alignment(self, point: QPointF) -> QPointF:
        """Rotate raw DXF point around the raw DXF scene centre before alignment."""
        return self._rotate_about_centre(point, self._rotation_rad())

    def _unrotate_point_from_preview(self, point: QPointF) -> QPointF:
        """Convert a clicked/rotated preview point back to raw DXF coordinates."""
        return self._rotate_about_centre(point, -self._rotation_rad())

    def _rotate_about_centre(self, point: QPointF, angle: float) -> QPointF:
        c = getattr(self, "_rotation_centre", QPointF(0.0, 0.0))
        ca = cos(angle)
        sa = sin(angle)
        x = point.x() - c.x()
        y = point.y() - c.y()
        return QPointF(c.x() + ca * x - sa * y, c.y() + sa * x + ca * y)

    def _manual_rotation_changed(self, *_):
        self._redraw_dxf_geometry()
        self._recalculate_if_ready()

    def _load_dxf(self):
        self._raw_paths, self._raw_snaps, self.unit_to_m = read_dxf_geometry(self.dxf_path)
        raw_rect = QRectF()
        for p in self._raw_paths:
            raw_rect = raw_rect.united(p.boundingRect()) if not raw_rect.isNull() else QRectF(p.boundingRect())
        self._rotation_centre = raw_rect.center() if not raw_rect.isNull() else QPointF(0.0, 0.0)
        self._redraw_dxf_geometry(first_fit=True)
        self.status.setText(
            f"DXF loaded. Unit scale to IFC metres: {self.unit_to_m:g}. "
            f"Middle mouse drag pans. Pick DXF point 1."
        )

    def _transformed_preview_path(self, path: QPainterPath) -> QPainterPath:
        t = QTransform()
        c = self._rotation_centre
        t.translate(c.x(), c.y())
        t.rotate(float(self.rotation_spin.value()))
        t.translate(-c.x(), -c.y())
        return t.map(path)

    def _redraw_dxf_geometry(self, first_fit: bool = False):
        self.scene.clear()
        self.view.ensure_snap_marker()
        pen = QPen(QColor("#8a8a8a"), 0)
        pen.setCosmetic(True)
        for p in getattr(self, "_raw_paths", []):
            item = self.scene.addPath(self._transformed_preview_path(p), pen)
            item.setZValue(1)

        rotated_snaps = [self._rotate_point_for_alignment(p) for p in getattr(self, "_raw_snaps", [])]
        self.view.set_snap_points(rotated_snaps)

        # Recreate picked DXF markers after view rotation changes.
        for i, p in enumerate(getattr(self, "dxf_points", []), start=1):
            rp = self._rotate_point_for_alignment(p)
            r = 0.15
            marker = self.scene.addEllipse(rp.x() - r, rp.y() - r, 2*r, 2*r, QPen(QColor("#ff9900"), 0), QBrush(QColor("#ff9900")))
            marker.setZValue(9999)
            _add_text(self.scene, f"DXF {i}", rp.x(), rp.y(), "#ff9900")

        if first_fit:
            self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-5, -5, 5, 5))
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _picked(self, p: QPointF):
        if len(self.dxf_points) >= 2:
            return

        # The view is showing the manually rotated DXF preview. Store the raw
        # DXF coordinate so later rotation changes can be applied consistently.
        raw_p = self._unrotate_point_from_preview(p)

        if self.dxf_points:
            first = self.dxf_points[0]
            d = hypot(raw_p.x() - first.x(), raw_p.y() - first.y())

            # Use a scene/model tolerance, but keep it very small so normal CAD
            # points close together still work.
            duplicate_tol = max(1e-6, abs(self.unit_to_m) * 1e-6)

            if d <= duplicate_tol:
                self.status.setText(
                    "Second DXF point is the same as point 1. "
                    "Pick a different DXF endpoint/corner further away."
                )
                return

        self.dxf_points.append(raw_p)

        self._redraw_dxf_geometry()

        if len(self.dxf_points) == 1:
            self.status.setText("Pick DXF point 2 matching IFC point 2")
        else:
            try:
                self._recalculate_if_ready()
            except ValueError as exc:
                self.status.setText(str(exc))
                self.dxf_points.pop()
                self._redraw_dxf_geometry()
                self.apply_btn.setEnabled(False)

    def _recalculate_if_ready(self):
        if len(getattr(self, "dxf_points", [])) != 2:
            self.apply_btn.setEnabled(False)
            return

        # Apply manual rotation to the raw picked points before calculating
        # the two-point transform. Then compose the resulting similarity so it
        # maps unrotated raw DXF geometry directly into IFC coordinates.
        rotated_a = self._rotate_point_for_alignment(self.dxf_points[0])
        rotated_b = self._rotate_point_for_alignment(self.dxf_points[1])
        try:
            base = two_point_transform(rotated_a, rotated_b, self.ifc_a, self.ifc_b, self.unit_to_m)
        except ValueError as exc:
            self.transform_result = None
            self.apply_btn.setEnabled(False)
            self.status.setText(str(exc))
            return

        # Compose: raw DXF -> unit scale + manual rotation around centre -> base alignment.
        c = self._rotation_centre
        theta = self._rotation_rad()
        ca = cos(theta)
        sa = sin(theta)
        unit = float(self.unit_to_m)

        # Manual rotation about raw centre, including unit conversion.
        # x_r = unit * (cx + ca*(x-cx) - sa*(y-cy))
        # y_r = unit * (cy + sa*(x-cx) + ca*(y-cy))
        m_scale = unit
        r_a = m_scale * ca
        r_b = -m_scale * sa
        r_d = m_scale * sa
        r_e = m_scale * ca
        r_tx = m_scale * (c.x() - ca * c.x() + sa * c.y())
        r_ty = m_scale * (c.y() - sa * c.x() - ca * c.y())

        b_ca = cos(base.rotation_rad)
        b_sa = sin(base.rotation_rad)
        b_s = base.scale
        b_a = b_s * b_ca
        b_b = -b_s * b_sa
        b_d = b_s * b_sa
        b_e = b_s * b_ca

        # Compose base * manual
        a = b_a * r_a + b_b * r_d
        bb = b_a * r_b + b_b * r_e
        d = b_d * r_a + b_e * r_d
        e = b_d * r_b + b_e * r_e
        tx = b_a * r_tx + b_b * r_ty + base.tx
        ty = b_d * r_tx + b_e * r_ty + base.ty

        # Convert affine matrix back to SimilarityTransform2D.
        total_scale = hypot(a, d)
        total_rot = atan2(d, a)
        self.transform_result = SimilarityTransform2D(scale=total_scale, rotation_rad=total_rot, tx=tx, ty=ty)
        # Store full matrix as an optional attribute for exact mapping if needed.
        self.transform_result.affine_matrix = (a, bb, d, e, tx, ty)

        self.status.setText(
            f"Ready. Manual rotation={self.rotation_spin.value():.4f}°. "
            f"Final scale={total_scale:g}, rotation={degrees(total_rot):.4f}°, "
            f"offset=({tx:.3f}, {ty:.3f})"
        )
        self.apply_btn.setEnabled(True)

    def _clear(self):
        self.dxf_points.clear()
        self.apply_btn.setEnabled(False)
        self.transform_result = None
        self.scene.clear()
        self._load_dxf()

    def _apply(self):
        if self.transform_result is not None:
            self.alignmentReady.emit(self.transform_result)
            self.accept()
