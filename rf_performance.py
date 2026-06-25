"""Performance helpers for the RF Attenuation Simulator.

This module is intentionally Qt-free so it can be imported by multiprocessing
workers.  It provides bounded caches, optional Numba kernels, adaptive sampling,
array resampling and compact shared-memory descriptors.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Optional acceleration; the simulator remains fully functional without it.
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    njit = None
    prange = range


if njit is not None:
    @njit(cache=True, nogil=True, parallel=True, fastmath=True)
    def _best_case_rssi_kernel(
        xs: np.ndarray,
        ys: np.ndarray,
        ap_x: float,
        ap_y: float,
        dz: float,
        reference_loss: float,
        exponent: float,
        eirp_dbm: float,
    ) -> np.ndarray:
        rows = ys.size
        cols = xs.size
        output = np.empty(rows * cols, dtype=np.float64)
        for gid in prange(rows * cols):
            iy = gid // cols
            ix = gid - iy * cols
            dy = ys[iy] - ap_y
            dx = xs[ix] - ap_x
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d < 1.0:
                d = 1.0
            output[gid] = eirp_dbm - reference_loss - 10.0 * exponent * math.log10(d)
        return output.reshape((rows, cols))
else:
    _best_case_rssi_kernel = None


def best_case_rssi_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    ap_x: float,
    ap_y: float,
    dz: float,
    reference_loss: float,
    exponent: float,
    eirp_dbm: float,
    use_numba: bool = True,
) -> np.ndarray:
    """Return unobstructed upper-bound RSSI for an AP over a rectangular grid."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if use_numba and _best_case_rssi_kernel is not None:
        return _best_case_rssi_kernel(xs, ys, float(ap_x), float(ap_y), float(dz), float(reference_loss), float(exponent), float(eirp_dbm))
    xx, yy = np.meshgrid(xs, ys)
    distance = np.sqrt((xx - float(ap_x)) ** 2 + (yy - float(ap_y)) ** 2 + float(dz) ** 2)
    distance = np.maximum(distance, 1.0)
    return float(eirp_dbm) - float(reference_loss) - 10.0 * float(exponent) * np.log10(distance)


if njit is not None:
    @njit(cache=True, nogil=True)
    def _coherent_metrics_kernel(
        powers_dbm: np.ndarray, phases_rad: np.ndarray, lengths_m: np.ndarray, floor_dbm: float
    ) -> Tuple[float, float]:
        real_field = 0.0
        imag_field = 0.0
        total_power = 0.0
        weighted_delay = 0.0
        for index in range(powers_dbm.size):
            power_mw = 10.0 ** (powers_dbm[index] / 10.0)
            amplitude = math.sqrt(power_mw)
            real_field += amplitude * math.cos(phases_rad[index])
            imag_field += amplitude * math.sin(phases_rad[index])
            total_power += power_mw
            weighted_delay += power_mw * lengths_m[index] / 299_792_458.0
        field_power = real_field * real_field + imag_field * imag_field
        if field_power <= 0.0:
            rssi = floor_dbm
        else:
            rssi = max(floor_dbm, 10.0 * math.log10(field_power))
        if total_power <= 0.0 or powers_dbm.size <= 1:
            return rssi, 0.0
        mean_delay = weighted_delay / total_power
        variance = 0.0
        for index in range(powers_dbm.size):
            power_mw = 10.0 ** (powers_dbm[index] / 10.0)
            delay = lengths_m[index] / 299_792_458.0
            delta = delay - mean_delay
            variance += power_mw * delta * delta
        variance /= total_power
        return rssi, math.sqrt(max(0.0, variance)) * 1e9
else:
    _coherent_metrics_kernel = None


def coherent_path_metrics(
    powers_dbm: Sequence[float],
    phases_rad: Sequence[float],
    lengths_m: Sequence[float],
    floor_dbm: float = -200.0,
    use_numba: bool = True,
) -> Tuple[float, float]:
    """Coherently combine path fields and compute RMS delay spread in one compiled pass."""
    powers = np.asarray(powers_dbm, dtype=np.float64)
    phases = np.asarray(phases_rad, dtype=np.float64)
    lengths = np.asarray(lengths_m, dtype=np.float64)
    if powers.size == 0:
        return float(floor_dbm), 0.0
    if use_numba and _coherent_metrics_kernel is not None:
        rssi, delay = _coherent_metrics_kernel(powers, phases, lengths, float(floor_dbm))
        return float(rssi), float(delay)
    power_mw = np.power(10.0, powers / 10.0)
    fields = np.sqrt(power_mw) * np.exp(1j * phases)
    total_field_power = float(abs(np.sum(fields)) ** 2)
    rssi = float(floor_dbm) if total_field_power <= 0.0 else max(float(floor_dbm), 10.0 * math.log10(total_field_power))
    total_power = float(np.sum(power_mw))
    if powers.size <= 1 or total_power <= 0.0:
        return rssi, 0.0
    delays = lengths / 299_792_458.0
    mean_delay = float(np.sum(power_mw * delays) / total_power)
    variance = float(np.sum(power_mw * (delays - mean_delay) ** 2) / total_power)
    return rssi, math.sqrt(max(0.0, variance)) * 1e9


def stable_digest(value: Any) -> str:
    """Produce a compact deterministic digest for JSON-like model metadata."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


class BoundedLRU:
    """Thread-safe LRU with both item and approximate-byte limits."""

    def __init__(self, maximum_items: int = 128, maximum_bytes: int = 512 * 1024 * 1024):
        self.maximum_items = max(1, int(maximum_items))
        self.maximum_bytes = max(1, int(maximum_bytes))
        self._items: "OrderedDict[Any, Tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def estimate_bytes(value: Any) -> int:
        total = 0
        seen = set()
        stack = [value]
        while stack:
            item = stack.pop()
            identity = id(item)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(item, np.ndarray):
                total += int(item.nbytes)
            elif isinstance(item, Mapping):
                stack.extend(item.keys()); stack.extend(item.values())
            elif isinstance(item, (tuple, list, set)):
                stack.extend(item)
            elif hasattr(item, "__dict__"):
                stack.extend(vars(item).values())
                total += 128
            else:
                total += 128
        return max(1, total)

    def get(self, key: Any) -> Any:
        with self._lock:
            record = self._items.pop(key, None)
            if record is None:
                return None
            self._items[key] = record
            return record[0]

    def put(self, key: Any, value: Any) -> None:
        size = self.estimate_bytes(value)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._items[key] = (value, size)
            self._bytes += size
            while len(self._items) > self.maximum_items or self._bytes > self.maximum_bytes:
                _, (_, removed_size) = self._items.popitem(last=False)
                self._bytes -= removed_size

    def clear(self) -> None:
        with self._lock:
            self._items.clear(); self._bytes = 0

    def configure(self, maximum_items: int, maximum_bytes: int) -> None:
        with self._lock:
            self.maximum_items = max(1, int(maximum_items))
            self.maximum_bytes = max(1, int(maximum_bytes))
            while len(self._items) > self.maximum_items or self._bytes > self.maximum_bytes:
                _, (_, removed_size) = self._items.popitem(last=False)
                self._bytes -= removed_size

    def stats(self) -> Tuple[int, int]:
        with self._lock:
            return len(self._items), self._bytes


def resample_regular_grid(
    source_xs: np.ndarray,
    source_ys: np.ndarray,
    source_values: np.ndarray,
    target_xs: np.ndarray,
    target_ys: np.ndarray,
) -> np.ndarray:
    """Bilinearly resample a regular grid using two vectorised 1-D passes."""
    source_xs = np.asarray(source_xs, dtype=float)
    source_ys = np.asarray(source_ys, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_xs = np.asarray(target_xs, dtype=float)
    target_ys = np.asarray(target_ys, dtype=float)
    if source_values.size == 0:
        return np.full((len(target_ys), len(target_xs)), np.nan, dtype=float)
    if np.array_equal(source_xs, target_xs) and np.array_equal(source_ys, target_ys):
        return source_values.copy()

    # Interpolate finite values and finite weights separately so boundary NaNs do
    # not bleed across holes or outside planning polygons.
    finite = np.isfinite(source_values)
    values = np.where(finite, source_values, 0.0)
    weights = finite.astype(float)
    x_values = np.vstack([np.interp(target_xs, source_xs, row) for row in values])
    x_weights = np.vstack([np.interp(target_xs, source_xs, row) for row in weights])
    result_values = np.vstack([
        np.interp(target_ys, source_ys, x_values[:, ix]) for ix in range(x_values.shape[1])
    ]).T
    result_weights = np.vstack([
        np.interp(target_ys, source_ys, x_weights[:, ix]) for ix in range(x_weights.shape[1])
    ]).T
    result = np.full_like(result_values, np.nan, dtype=float)
    valid = result_weights > 1e-8
    result[valid] = result_values[valid] / result_weights[valid]
    return result


def nearest_resample_regular_grid(
    source_xs: np.ndarray,
    source_ys: np.ndarray,
    source_values: np.ndarray,
    target_xs: np.ndarray,
    target_ys: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour resampling for path-count and categorical grids."""
    source_xs = np.asarray(source_xs, dtype=float)
    source_ys = np.asarray(source_ys, dtype=float)
    target_xs = np.asarray(target_xs, dtype=float)
    target_ys = np.asarray(target_ys, dtype=float)
    x_index = np.abs(source_xs[None, :] - target_xs[:, None]).argmin(axis=1)
    y_index = np.abs(source_ys[None, :] - target_ys[:, None]).argmin(axis=1)
    return np.asarray(source_values)[np.ix_(y_index, x_index)].copy()


def dilate_boolean(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Small dependency-free binary dilation used by adaptive refinement."""
    mask = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or mask.size == 0:
        return mask.copy()
    output = mask.copy()
    for _ in range(radius):
        padded = np.pad(output, 1, mode="constant", constant_values=False)
        neighbours = np.zeros_like(output)
        for dy in range(3):
            for dx in range(3):
                neighbours |= padded[dy:dy + output.shape[0], dx:dx + output.shape[1]]
        output = neighbours
    return output


def adaptive_refinement_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    coarse_values: Sequence[np.ndarray],
    valid_mask: Optional[np.ndarray],
    thresholds_dbm: Sequence[float],
    coarse_stride: int,
    gradient_threshold_db_per_m: float,
    threshold_margin_db: float,
    geometry_boxes: Iterable[Tuple[float, float, float, float]],
    geometry_buffer_m: float,
    ap_points: Iterable[Tuple[float, float, float]],
    ap_refine_radius_m: float,
) -> np.ndarray:
    """Build the fine-grid points that need exact ray evaluation."""
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    rows, cols = len(ys), len(xs)
    mask = np.zeros((rows, cols), dtype=bool)
    stride = max(1, int(coarse_stride))
    mask[::stride, ::stride] = True

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
    else:
        valid = np.ones_like(mask)

    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
    for values in coarse_values:
        array = np.asarray(values, dtype=float)
        finite = np.isfinite(array)
        if not np.any(finite):
            continue
        safe = np.where(finite, array, np.nan)
        gy, gx = np.gradient(safe, max(dy, 1e-6), max(dx, 1e-6))
        gradient = np.hypot(np.nan_to_num(gx), np.nan_to_num(gy))
        mask |= gradient >= max(0.01, float(gradient_threshold_db_per_m))
        for threshold in thresholds_dbm:
            mask |= finite & (np.abs(array - float(threshold)) <= max(0.1, float(threshold_margin_db)))

    for minx, miny, maxx, maxy in geometry_boxes:
        x_sel = (xs >= float(minx) - geometry_buffer_m) & (xs <= float(maxx) + geometry_buffer_m)
        y_sel = (ys >= float(miny) - geometry_buffer_m) & (ys <= float(maxy) + geometry_buffer_m)
        if np.any(x_sel) and np.any(y_sel):
            mask[np.ix_(y_sel, x_sel)] = True

    xx, yy = np.meshgrid(xs, ys)
    for ap_x, ap_y, radius in ap_points:
        requested_radius = max(0.5, float(ap_refine_radius_m))
        cutoff_radius = float(radius)
        effective_radius = min(requested_radius, cutoff_radius) if cutoff_radius > 0.0 else requested_radius
        mask |= (xx - float(ap_x)) ** 2 + (yy - float(ap_y)) ** 2 <= effective_radius ** 2

    # Refine the one-cell boundary of valid polygons and holes.
    padded = np.pad(valid, 1, mode="edge")
    edge = np.zeros_like(valid)
    for dy_i in range(3):
        for dx_i in range(3):
            edge |= valid != padded[dy_i:dy_i + rows, dx_i:dx_i + cols]
    mask |= edge
    mask = dilate_boolean(mask, 1)
    return mask & valid


@dataclass(frozen=True)
class SharedArraySpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str


def create_shared_array(shape: Sequence[int], dtype: Any, fill_value: Optional[float] = None):
    dtype_obj = np.dtype(dtype)
    shape_tuple = tuple(int(v) for v in shape)
    size = int(np.prod(shape_tuple, dtype=np.int64)) * dtype_obj.itemsize
    memory = shared_memory.SharedMemory(create=True, size=max(1, size))
    array = np.ndarray(shape_tuple, dtype=dtype_obj, buffer=memory.buf)
    if fill_value is not None:
        array.fill(fill_value)
    return memory, array, SharedArraySpec(memory.name, shape_tuple, dtype_obj.str)


def attach_shared_array(spec: SharedArraySpec):
    memory = shared_memory.SharedMemory(name=spec.name)
    array = np.ndarray(tuple(spec.shape), dtype=np.dtype(spec.dtype), buffer=memory.buf)
    return memory, array
