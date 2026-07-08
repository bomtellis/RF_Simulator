"""Optional GPU acceleration for the RF simulator.

NVIDIA devices use Numba-CUDA JIT kernels.  Intel and AMD devices may use the
existing OpenCL fallback when PyOpenCL and a vendor runtime are installed.
The public helper names are retained for compatibility with older RF plans and
application code, but CUDA is always preferred when a usable NVIDIA GPU exists.
"""
from __future__ import annotations

import math
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ITU-R P.525-5 Eq. (6): Lbf = 32.4 + 20log10(f_MHz) + 20log10(d_km).
# At d = 1 m = 0.001 km this becomes 20log10(f_MHz) - 27.6.
P525_ONE_METRE_MHZ_CONSTANT_DB = -27.6

try:
    from numba import float32, njit, prange  # type: ignore
except Exception:  # pragma: no cover
    float32 = None
    njit = None
    prange = range

try:  # Current NVIDIA-supported CUDA target for Numba.
    from numba import cuda  # type: ignore
except Exception:  # pragma: no cover - normal without numba-cuda/CUDA driver
    cuda = None

try:  # Optional Intel/AMD/cross-vendor fallback.
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover
    cl = None


class GPUExecutionError(RuntimeError):
    """Raised when a detected/selected GPU fails instead of silently using CPU."""


@dataclass(frozen=True)
class OpenCLDeviceInfo:
    """Backward-compatible generic device record used by the settings dialog."""

    platform: str
    vendor: str
    name: str
    device_type: str
    global_memory_mb: int
    compute_units: int
    supports_fp64: bool
    token: str = "auto"

    @property
    def label(self) -> str:
        memory = f", {self.global_memory_mb} MB" if self.global_memory_mb else ""
        return f"{self.vendor} {self.name} ({self.device_type}{memory})"


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "").strip()


def _cuda_device_records() -> List[OpenCLDeviceInfo]:
    if cuda is None:
        return []
    try:
        if not cuda.is_available():
            return []
        records: List[OpenCLDeviceInfo] = []
        for index, gpu in enumerate(cuda.gpus):
            try:
                with gpu:
                    context = cuda.current_context()
                    device = context.device
                    name = _decode_name(getattr(device, "name", f"CUDA device {index}"))
                    memory = int(getattr(device, "total_memory", 0) or 0) // (1024 * 1024)
                    cc = getattr(device, "compute_capability", (0, 0))
                    multiprocessors = int(getattr(device, "MULTIPROCESSOR_COUNT", 0) or 0)
                    records.append(OpenCLDeviceInfo(
                        platform="CUDA",
                        vendor="NVIDIA",
                        name=name,
                        device_type=f"CUDA GPU cc {cc[0]}.{cc[1]}",
                        global_memory_mb=max(0, memory),
                        compute_units=max(0, multiprocessors),
                        supports_fp64=True,
                        token=f"cuda:{index}",
                    ))
            except Exception:
                continue
        return records
    except Exception:
        return []


def _device_type_name(device: Any) -> str:
    if cl is None:
        return "unknown"
    value = int(getattr(device, "type", 0))
    if value & int(cl.device_type.GPU):
        return "OpenCL GPU"
    if value & int(cl.device_type.ACCELERATOR):
        return "OpenCL accelerator"
    if value & int(cl.device_type.CPU):
        return "OpenCL CPU"
    return "OpenCL other"


def _opencl_device_records(include_cpu: bool = False) -> List[OpenCLDeviceInfo]:
    if cl is None:
        return []
    output: List[OpenCLDeviceInfo] = []
    try:
        platforms = cl.get_platforms()
    except Exception:
        return []
    stable_index = 0
    for platform in platforms:
        try:
            devices = platform.get_devices()
        except Exception:
            continue
        for device in devices:
            dtype = _device_type_name(device)
            if dtype == "OpenCL CPU" and not include_cpu:
                continue
            extensions = str(getattr(device, "extensions", "") or "").lower()
            vendor = str(getattr(device, "vendor", "Unknown") or "Unknown").strip()
            name = str(getattr(device, "name", "OpenCL device") or "OpenCL device").strip()
            output.append(OpenCLDeviceInfo(
                platform=str(getattr(platform, "name", "OpenCL") or "OpenCL").strip(),
                vendor=vendor,
                name=name,
                device_type=dtype,
                global_memory_mb=max(0, int(getattr(device, "global_mem_size", 0) or 0) // (1024 * 1024)),
                compute_units=max(0, int(getattr(device, "max_compute_units", 0) or 0)),
                supports_fp64=("cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions),
                token=f"opencl:{stable_index}:{vendor} {name}",
            ))
            stable_index += 1
    return output


def discover_opencl_devices(include_cpu: bool = False) -> List[OpenCLDeviceInfo]:
    """Return CUDA devices first, followed by OpenCL alternatives.

    The legacy function name is kept so older simulator modules continue to
    import successfully.
    """
    records = _cuda_device_records()
    seen = {(record.vendor.lower(), record.name.lower()) for record in records}
    for record in _opencl_device_records(include_cpu=include_cpu):
        # Do not show the same NVIDIA adapter twice unless CUDA is unavailable.
        key = (record.vendor.lower(), record.name.lower())
        if "nvidia" in record.vendor.lower() and any("nvidia" in vendor for vendor, _ in seen):
            continue
        records.append(record)
    return records


# ----------------------------- Numba CUDA JIT -----------------------------
if cuda is not None:
    @cuda.jit(cache=True, fastmath=True)
    def _cuda_influence_mask(xs, ys, links, valid, output, cols, rows):
        total = cols * rows
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for gid in range(start, total, stride):
            if valid[gid] == 0:
                output[gid] = 0
                continue
            ix = gid % cols
            iy = gid // cols
            x = xs[ix]
            y = ys[iy]
            keep = 0
            link_count = links.shape[0]
            for link in range(link_count):
                dx = x - links[link, 0]
                dy = y - links[link, 1]
                dz = links[link, 2]
                distance2 = dx * dx + dy * dy + dz * dz
                radius2 = links[link, 7]
                if radius2 > 0.0 and distance2 > radius2:
                    continue
                if distance2 < 1.0:
                    distance2 = 1.0
                distance = math.sqrt(distance2)
                upper = links[link, 5] - links[link, 3] - 10.0 * links[link, 4] * math.log10(distance)
                if upper >= links[link, 6] - 0.25:
                    keep = 1
                    break
            output[gid] = keep

    @cuda.jit(cache=True, fastmath=True)
    def _cuda_strongest_index(stack, fields, points, indices, valid):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for point in range(start, points, stride):
            best = -3.4028235e38
            best_index = 0
            any_valid = 0
            for field in range(fields):
                value = stack[field, point]
                if math.isfinite(value):
                    any_valid = 1
                    if value > best:
                        best = value
                        best_index = field
            indices[point] = best_index
            valid[point] = any_valid

    @cuda.jit(cache=True, fastmath=True)
    def _cuda_resample_bilinear(source, source_cols, source_rows, source_x0, source_y0,
                                source_dx, source_dy, target_xs, target_ys,
                                target_cols, target_rows, output):
        total = target_cols * target_rows
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for gid in range(start, total, stride):
            tx = gid % target_cols
            ty = gid // target_cols
            fx = (target_xs[tx] - source_x0) / source_dx
            fy = (target_ys[ty] - source_y0) / source_dy
            if fx < 0.0:
                fx = 0.0
            elif fx > source_cols - 1:
                fx = source_cols - 1
            if fy < 0.0:
                fy = 0.0
            elif fy > source_rows - 1:
                fy = source_rows - 1
            x0 = int(math.floor(fx))
            y0 = int(math.floor(fy))
            x1 = min(x0 + 1, source_cols - 1)
            y1 = min(y0 + 1, source_rows - 1)
            wx = fx - x0
            wy = fy - y0
            value_sum = 0.0
            weight_sum = 0.0

            value = source[y0, x0]
            weight = (1.0 - wx) * (1.0 - wy)
            if math.isfinite(value):
                value_sum += value * weight
                weight_sum += weight
            value = source[y0, x1]
            weight = wx * (1.0 - wy)
            if math.isfinite(value):
                value_sum += value * weight
                weight_sum += weight
            value = source[y1, x0]
            weight = (1.0 - wx) * wy
            if math.isfinite(value):
                value_sum += value * weight
                weight_sum += weight
            value = source[y1, x1]
            weight = wx * wy
            if math.isfinite(value):
                value_sum += value * weight
                weight_sum += weight
            if weight_sum > 1.0e-8:
                output[gid] = value_sum / weight_sum
            else:
                output[gid] = math.nan

    @cuda.jit(cache=True)
    def _cuda_colourise(values, points, mins, maxs, colours, zone_count,
                        high_index, low_index, output):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for gid in range(start, points, stride):
            value = values[gid]
            if not math.isfinite(value):
                output[gid, 0] = 0
                output[gid, 1] = 0
                output[gid, 2] = 0
                output[gid, 3] = 0
                continue
            selected = -1
            for zone in range(zone_count):
                if value >= mins[zone] and value < maxs[zone]:
                    selected = zone
                    break
            if selected < 0:
                selected = high_index if value >= maxs[high_index] else low_index
            output[gid, 0] = colours[selected, 0]
            output[gid, 1] = colours[selected, 1]
            output[gid, 2] = colours[selected, 2]
            output[gid, 3] = colours[selected, 3]

    @cuda.jit(device=True, inline=True)
    def _cuda_line_box_z_hit(ax, ay, az, dx, dy, z_delta, min_x, max_x, min_y, max_y, z_min, z_max):
        t0 = 0.0
        t1 = 1.0
        eps = 1.0e-7
        if dx > -eps and dx < eps:
            if ax < min_x or ax > max_x:
                return 0
        else:
            inv_dx = 1.0 / dx
            tx1 = (min_x - ax) * inv_dx
            tx2 = (max_x - ax) * inv_dx
            if tx1 > tx2:
                tmp = tx1
                tx1 = tx2
                tx2 = tmp
            if tx1 > t0:
                t0 = tx1
            if tx2 < t1:
                t1 = tx2
            if t0 > t1:
                return 0
        if dy > -eps and dy < eps:
            if ay < min_y or ay > max_y:
                return 0
        else:
            inv_dy = 1.0 / dy
            ty1 = (min_y - ay) * inv_dy
            ty2 = (max_y - ay) * inv_dy
            if ty1 > ty2:
                tmp = ty1
                ty1 = ty2
                ty2 = tmp
            if ty1 > t0:
                t0 = ty1
            if ty2 < t1:
                t1 = ty2
            if t0 > t1:
                return 0
        if t1 < 0.0 or t0 > 1.0:
            return 0
        if t0 < 0.0:
            t0 = 0.0
        if t1 > 1.0:
            t1 = 1.0
        za = az + z_delta * t0
        zb = az + z_delta * t1
        zlo = za if za < zb else zb
        zhi = zb if zb > za else za
        return 1 if (zhi >= z_min and zlo <= z_max) else 0

    @cuda.jit(cache=True, fastmath=True)
    def _cuda_direct_rssi_paths(xs, ys, valid, ap_data, segments, segment_indices, segment_offsets,
                                receiver_z, terrain_z, terrain_enabled, ground_loss_db, terrain_clearance,
                                disconnected, start_gid, work_count, path_rssi, cols, rows):
        """One CUDA thread per AP/point path, ordered AP-major.

        Adjacent threads process adjacent points for the same AP.  This keeps
        segment-list reads coherent and makes both the path write and the later
        field reduction coalesced.
        """
        ap_count = ap_data.shape[0]
        path_total = work_count * ap_count
        local_start = cuda.grid(1)
        stride = cuda.gridsize(1)
        total = cols * rows
        for path_index in range(local_start, path_total, stride):
            ai = path_index // work_count
            local_gid = path_index - ai * work_count
            gid = start_gid + local_gid
            if gid >= total or valid[gid] == 0:
                path_rssi[path_index] = math.nan
                continue
            ix = gid % cols
            iy = gid // cols
            x = xs[ix]
            y = ys[iy]
            ax = ap_data[ai, 0]
            ay = ap_data[ai, 1]
            az = ap_data[ai, 2]
            z_delta = ap_data[ai, 3]
            base_dbm = ap_data[ai, 4]
            path_loss_factor = ap_data[ai, 5]
            cutoff2 = ap_data[ai, 6]
            dz2 = ap_data[ai, 7]
            if terrain_enabled != 0:
                rz = receiver_z[gid]
                z_delta = rz - az
                dz2 = z_delta * z_delta
            dx = x - ax
            dy = y - ay
            d2xy = dx * dx + dy * dy
            if cutoff2 > 0.0 and d2xy > cutoff2:
                path_rssi[path_index] = math.nan
                continue
            d3 = math.sqrt(d2xy + dz2)
            if d3 < 1.0:
                d3 = 1.0
            ray_min_x = ax if ax < x else x
            ray_max_x = x if x > ax else ax
            ray_min_y = ay if ay < y else y
            ray_max_y = y if y > ay else ay
            wall_loss = 0.0
            first_segment = segment_offsets[ai]
            last_segment = segment_offsets[ai + 1]
            for packed_index in range(first_segment, last_segment):
                si = segment_indices[packed_index]
                # Cheap AABB rejection prevents most barriers from reaching the
                # more expensive ray/segment intersection arithmetic.
                if (segments[si, 8] < ray_min_x - 1.0e-5 or segments[si, 7] > ray_max_x + 1.0e-5 or
                        segments[si, 10] < ray_min_y - 1.0e-5 or segments[si, 9] > ray_max_y + 1.0e-5):
                    continue
                if segments[si, 11] >= 0.5:
                    if _cuda_line_box_z_hit(
                        ax, ay, az, dx, dy, z_delta,
                        segments[si, 7], segments[si, 8], segments[si, 9], segments[si, 10],
                        segments[si, 4], segments[si, 5],
                    ) != 0:
                        wall_loss += segments[si, 6]
                    continue
                x1 = segments[si, 0]
                y1 = segments[si, 1]
                sdx = segments[si, 2]
                sdy = segments[si, 3]
                den = dx * sdy - dy * sdx
                if den > -1.0e-7 and den < 1.0e-7:
                    continue
                t = ((x1 - ax) * sdy - (y1 - ay) * sdx) / den
                u = ((x1 - ax) * dy - (y1 - ay) * dx) / den
                if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                    zhit = az + z_delta * t
                    if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                        wall_loss += segments[si, 6]
            if terrain_enabled != 0 and ground_loss_db > 0.0:
                spacing_x = abs(xs[1] - xs[0]) if cols > 1 else 2.0
                spacing_y = abs(ys[1] - ys[0]) if rows > 1 else 2.0
                spacing = spacing_x if spacing_x < spacing_y else spacing_y
                if spacing < 1.0:
                    spacing = 1.0
                steps = int(math.ceil(math.sqrt(d2xy) / spacing))
                if steps < 4:
                    steps = 4
                elif steps > 32:
                    steps = 32
                for step in range(1, steps):
                    tline = step / steps
                    sx = ax + dx * tline
                    sy = ay + dy * tline
                    tx = int(math.floor(((sx - xs[0]) / (xs[cols - 1] - xs[0]) * (cols - 1)) + 0.5)) if cols > 1 else 0
                    ty = int(math.floor(((sy - ys[0]) / (ys[rows - 1] - ys[0]) * (rows - 1)) + 0.5)) if rows > 1 else 0
                    if tx < 0:
                        tx = 0
                    elif tx >= cols:
                        tx = cols - 1
                    if ty < 0:
                        ty = 0
                    elif ty >= rows:
                        ty = rows - 1
                    los_z = az + z_delta * tline
                    if terrain_z[ty * cols + tx] >= los_z - terrain_clearance:
                        wall_loss += ground_loss_db
                        break
            rssi = base_dbm - path_loss_factor * math.log10(d3) - wall_loss
            if rssi < disconnected:
                rssi = disconnected
            path_rssi[path_index] = rssi

    @cuda.jit(cache=True, fastmath=True)
    def _cuda_direct_rssi_paths_block(xs, ys, valid, ap_data, segments, segment_indices, segment_offsets,
                                      receiver_z, terrain_z, terrain_enabled, ground_loss_db, terrain_clearance,
                                      disconnected, start_gid, work_count, path_rssi, cols, rows):
        """Use one CUDA block per AP/point path for barrier-heavy IFC models.

        All threads in the block cooperate on the path's barrier list and a
        shared-memory reduction combines their attenuation contributions. This
        moves the most expensive inner loop fully onto parallel CUDA lanes.
        """
        shared_loss = cuda.shared.array(shape=128, dtype=float32)
        tid = cuda.threadIdx.x
        ap_count = ap_data.shape[0]
        path_total = work_count * ap_count
        total = cols * rows
        for path_index in range(cuda.blockIdx.x, path_total, cuda.gridDim.x):
            ai = path_index // work_count
            local_gid = path_index - ai * work_count
            gid = start_gid + local_gid
            active = gid < total and valid[gid] != 0
            x = 0.0
            y = 0.0
            ax = 0.0
            ay = 0.0
            az = 0.0
            z_delta = 0.0
            base_dbm = 0.0
            path_loss_factor = 0.0
            cutoff2 = 0.0
            dz2 = 0.0
            dx = 0.0
            dy = 0.0
            d2xy = 0.0
            if active:
                ix = gid % cols
                iy = gid // cols
                x = xs[ix]
                y = ys[iy]
                ax = ap_data[ai, 0]
                ay = ap_data[ai, 1]
                az = ap_data[ai, 2]
                z_delta = ap_data[ai, 3]
                base_dbm = ap_data[ai, 4]
                path_loss_factor = ap_data[ai, 5]
                cutoff2 = ap_data[ai, 6]
                dz2 = ap_data[ai, 7]
                if terrain_enabled != 0:
                    rz = receiver_z[gid]
                    z_delta = rz - az
                    dz2 = z_delta * z_delta
                dx = x - ax
                dy = y - ay
                d2xy = dx * dx + dy * dy
                if cutoff2 > 0.0 and d2xy > cutoff2:
                    active = False
            partial_loss = 0.0
            if active:
                ray_min_x = ax if ax < x else x
                ray_max_x = x if x > ax else ax
                ray_min_y = ay if ay < y else y
                ray_max_y = y if y > ay else ay
                first_segment = segment_offsets[ai]
                last_segment = segment_offsets[ai + 1]
                for packed_index in range(first_segment + tid, last_segment, cuda.blockDim.x):
                    si = segment_indices[packed_index]
                    if (segments[si, 8] < ray_min_x - 1.0e-5 or segments[si, 7] > ray_max_x + 1.0e-5 or
                            segments[si, 10] < ray_min_y - 1.0e-5 or segments[si, 9] > ray_max_y + 1.0e-5):
                        continue
                    if segments[si, 11] >= 0.5:
                        if _cuda_line_box_z_hit(
                            ax, ay, az, dx, dy, z_delta,
                            segments[si, 7], segments[si, 8], segments[si, 9], segments[si, 10],
                            segments[si, 4], segments[si, 5],
                        ) != 0:
                            partial_loss += segments[si, 6]
                        continue
                    x1 = segments[si, 0]
                    y1 = segments[si, 1]
                    sdx = segments[si, 2]
                    sdy = segments[si, 3]
                    den = dx * sdy - dy * sdx
                    if den > -1.0e-7 and den < 1.0e-7:
                        continue
                    t = ((x1 - ax) * sdy - (y1 - ay) * sdx) / den
                    u = ((x1 - ax) * dy - (y1 - ay) * dx) / den
                    if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                        zhit = az + z_delta * t
                        if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                            partial_loss += segments[si, 6]
                if tid == 0 and ground_loss_db > 0.0 and terrain_enabled != 0:
                    spacing_x = abs(xs[1] - xs[0]) if cols > 1 else 2.0
                    spacing_y = abs(ys[1] - ys[0]) if rows > 1 else 2.0
                    spacing = spacing_x if spacing_x < spacing_y else spacing_y
                    if spacing < 1.0:
                        spacing = 1.0
                    steps = int(math.ceil(math.sqrt(d2xy) / spacing))
                    if steps < 4:
                        steps = 4
                    elif steps > 32:
                        steps = 32
                    for step in range(1, steps):
                        tline = step / steps
                        sx = ax + dx * tline
                        sy = ay + dy * tline
                        tx = int(math.floor(((sx - xs[0]) / (xs[cols - 1] - xs[0]) * (cols - 1)) + 0.5)) if cols > 1 else 0
                        ty = int(math.floor(((sy - ys[0]) / (ys[rows - 1] - ys[0]) * (rows - 1)) + 0.5)) if rows > 1 else 0
                        if tx < 0:
                            tx = 0
                        elif tx >= cols:
                            tx = cols - 1
                        if ty < 0:
                            ty = 0
                        elif ty >= rows:
                            ty = rows - 1
                        los_z = az + z_delta * tline
                        if terrain_z[ty * cols + tx] >= los_z - terrain_clearance:
                            partial_loss += ground_loss_db
                            break
            shared_loss[tid] = partial_loss
            cuda.syncthreads()
            stride = cuda.blockDim.x // 2
            while stride > 0:
                if tid < stride:
                    shared_loss[tid] += shared_loss[tid + stride]
                cuda.syncthreads()
                stride //= 2
            if tid == 0:
                if not active:
                    path_rssi[path_index] = math.nan
                else:
                    d3 = math.sqrt(d2xy + dz2)
                    if d3 < 1.0:
                        d3 = 1.0
                    rssi = base_dbm - path_loss_factor * math.log10(d3) - shared_loss[0]
                    if rssi < disconnected:
                        rssi = disconnected
                    path_rssi[path_index] = rssi
            cuda.syncthreads()

    @cuda.jit(cache=True, fastmath=True)
    def _cuda_reduce_rssi_paths(valid, path_rssi, ap_count, disconnected, combine_mode,
                                 start_gid, work_count, out_rssi, out_counts):
        local_start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for local_gid in range(local_start, work_count, stride):
            gid = start_gid + local_gid
            if valid[gid] == 0:
                out_rssi[gid] = math.nan
                out_counts[gid] = 0
                continue
            best = disconnected
            power_sum_mw = 0.0
            path_count = 0
            for ai in range(ap_count):
                rssi = path_rssi[ai * work_count + local_gid]
                if not math.isfinite(rssi):
                    continue
                path_count += 1
                if combine_mode == 1:
                    power_sum_mw += math.pow(10.0, rssi / 10.0)
                elif rssi > best:
                    best = rssi
            if combine_mode == 1 and power_sum_mw > 0.0:
                best = 10.0 * math.log10(power_sum_mw)
            if path_count == 0:
                best = disconnected
            out_rssi[gid] = best
            out_counts[gid] = path_count if combine_mode == 1 else (1 if path_count > 0 else 0)

    @cuda.jit(cache=True, fastmath=True)
    def _cuda_inspect_direct_paths(ap_data, segments, segment_indices, segment_offsets,
                                   x, y, disconnected, output):
        ap_count = ap_data.shape[0]
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for ai in range(start, ap_count, stride):
            ax = ap_data[ai, 0]
            ay = ap_data[ai, 1]
            az = ap_data[ai, 2]
            z_delta = ap_data[ai, 3]
            base_dbm = ap_data[ai, 4]
            path_loss_factor = ap_data[ai, 5]
            cutoff2 = ap_data[ai, 6]
            dz2 = ap_data[ai, 7]
            dx = x - ax
            dy = y - ay
            d2xy = dx * dx + dy * dy
            d3 = math.sqrt(d2xy + dz2)
            if d3 < 1.0:
                d3 = 1.0
            if cutoff2 > 0.0 and d2xy > cutoff2:
                output[ai, 0] = disconnected
                output[ai, 1] = d3
                output[ai, 2] = 0.0
                output[ai, 3] = 0.0
                continue
            ray_min_x = ax if ax < x else x
            ray_max_x = x if x > ax else ax
            ray_min_y = ay if ay < y else y
            ray_max_y = y if y > ay else ay
            wall_loss = 0.0
            first_segment = segment_offsets[ai]
            last_segment = segment_offsets[ai + 1]
            for packed_index in range(first_segment, last_segment):
                si = segment_indices[packed_index]
                if (segments[si, 8] < ray_min_x - 1.0e-5 or segments[si, 7] > ray_max_x + 1.0e-5 or
                        segments[si, 10] < ray_min_y - 1.0e-5 or segments[si, 9] > ray_max_y + 1.0e-5):
                    continue
                if segments[si, 11] >= 0.5:
                    if _cuda_line_box_z_hit(
                        ax, ay, az, dx, dy, z_delta,
                        segments[si, 7], segments[si, 8], segments[si, 9], segments[si, 10],
                        segments[si, 4], segments[si, 5],
                    ) != 0:
                        wall_loss += segments[si, 6]
                    continue
                x1 = segments[si, 0]
                y1 = segments[si, 1]
                sdx = segments[si, 2]
                sdy = segments[si, 3]
                den = dx * sdy - dy * sdx
                if den > -1.0e-7 and den < 1.0e-7:
                    continue
                t = ((x1 - ax) * sdy - (y1 - ay) * sdx) / den
                u = ((x1 - ax) * dy - (y1 - ay) * dx) / den
                if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                    zhit = az + z_delta * t
                    if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                        wall_loss += segments[si, 6]
            rssi = base_dbm - path_loss_factor * math.log10(d3) - wall_loss
            if rssi < disconnected:
                rssi = disconnected
            output[ai, 0] = rssi
            output[ai, 1] = d3
            output[ai, 2] = wall_loss
            output[ai, 3] = 1.0

else:  # pragma: no cover
    _cuda_influence_mask = None
    _cuda_strongest_index = None
    _cuda_resample_bilinear = None
    _cuda_colourise = None
    _cuda_direct_rssi_paths = None
    _cuda_direct_rssi_paths_block = None
    _cuda_reduce_rssi_paths = None
    _cuda_inspect_direct_paths = None


class NumbaCUDARFAccelerator:
    """Cached Numba-CUDA kernels for a selected NVIDIA adapter.

    Kernels use grid-stride loops and an occupancy-sized launch grid.  This
    keeps the GPU busy on large heatmaps and avoids the previous one-thread-per-
    point launch producing very small block grids.  Direct RSSI inputs that are
    common to every radio group are retained on the device between calls.
    """

    backend_name = "CUDA"

    def __init__(self, device_index: int = 0):
        if cuda is None or not cuda.is_available():
            raise RuntimeError("Numba CUDA is unavailable")
        records = _cuda_device_records()
        if not records:
            raise RuntimeError("No CUDA-capable NVIDIA GPU was found")
        self.device_index = max(0, min(int(device_index), len(records) - 1))
        self.info = records[self.device_index]
        self._lock = threading.RLock()
        # 128 threads limits register pressure in the barrier-heavy kernels and
        # still gives four complete warps per block.  The grid is capped to a
        # deep queue per SM; direct RSSI batches also use several CUDA streams
        # so bounded chunks can stay queued without reusing temporary buffers.
        self.threads_per_block = 128
        self._sm_count = max(1, int(self.info.compute_units or 1))
        self._blocks_per_sm = 24
        self._minimum_blocks = max(32, self._sm_count * 4)
        self._maximum_blocks = max(self._minimum_blocks, self._sm_count * self._blocks_per_sm)
        self._direct_grid_key = None
        self._direct_grid_refs = None
        self._direct_grid_buffers = None
        self._direct_output_capacity = 0
        self._direct_rssi_buffer = None
        self._direct_count_buffer = None
        self._direct_path_capacity = 0
        self._direct_path_buffer = None
        self._direct_model_cache = OrderedDict()
        self._direct_model_cache_bytes = 0
        self._direct_model_cache_limit = 512 * 1024 * 1024
        self.last_direct_stats = {}

    def _launch_shape(self, count: int, threads: Optional[int] = None) -> Tuple[int, int]:
        count = max(1, int(count))
        threads = max(32, int(threads or self.threads_per_block))
        natural_blocks = max(1, (count + threads - 1) // threads)
        return min(natural_blocks, self._maximum_blocks), threads

    def _block_launch_shape(self, path_count: int, threads: int) -> Tuple[int, int]:
        threads = 32 if threads <= 32 else (64 if threads <= 64 else 128)
        natural_blocks = max(1, int(path_count))
        return min(natural_blocks, self._maximum_blocks), threads

    def _ensure_direct_buffers(self, total_points: int, path_items: int, stream):
        if self._direct_output_capacity < total_points or self._direct_rssi_buffer is None:
            self._direct_rssi_buffer = cuda.device_array(total_points, dtype=np.float32, stream=stream)
            self._direct_count_buffer = cuda.device_array(total_points, dtype=np.int16, stream=stream)
            self._direct_output_capacity = total_points
        if self._direct_path_capacity < path_items or self._direct_path_buffer is None:
            self._direct_path_buffer = cuda.device_array(path_items, dtype=np.float32, stream=stream)
            self._direct_path_capacity = path_items
        return self._direct_rssi_buffer, self._direct_count_buffer, self._direct_path_buffer

    def _model_buffers(self, compact_aps, prepared_segments, segment_indices, segment_offsets, stream):
        key = (id(compact_aps), id(prepared_segments), id(segment_indices), id(segment_offsets))
        cached = self._direct_model_cache.get(key)
        if cached is not None:
            self._direct_model_cache.move_to_end(key)
            return cached[0]
        buffers = (
            cuda.to_device(compact_aps, stream=stream),
            cuda.to_device(prepared_segments, stream=stream),
            cuda.to_device(segment_indices, stream=stream),
            cuda.to_device(segment_offsets, stream=stream),
        )
        size_bytes = int(compact_aps.nbytes + prepared_segments.nbytes + segment_indices.nbytes + segment_offsets.nbytes)
        host_refs = (compact_aps, prepared_segments, segment_indices, segment_offsets)
        self._direct_model_cache[key] = (buffers, size_bytes, host_refs)
        self._direct_model_cache_bytes += size_bytes
        while len(self._direct_model_cache) > 4 or self._direct_model_cache_bytes > self._direct_model_cache_limit:
            _old_key, (_old_buffers, old_size, _old_host_refs) = self._direct_model_cache.popitem(last=False)
            self._direct_model_cache_bytes -= int(old_size)
        return buffers

    @staticmethod
    def _cancelled(settings: Any) -> bool:
        event = getattr(settings, "_cancel_event", None) if settings is not None else None
        return bool(event is not None and event.is_set())

    def _grid_buffers(self, xs: np.ndarray, ys: np.ndarray, valid_mask: Optional[np.ndarray], stream):
        valid_shape = (len(ys), len(xs))
        key = valid_shape
        refs = self._direct_grid_refs
        if (
            self._direct_grid_key == key
            and refs is not None
            and refs[0] is xs
            and refs[1] is ys
            and refs[2] is valid_mask
            and self._direct_grid_buffers is not None
        ):
            return self._direct_grid_buffers
        xs32 = np.ascontiguousarray(xs, dtype=np.float32)
        ys32 = np.ascontiguousarray(ys, dtype=np.float32)
        valid = np.ones(valid_shape, dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask, dtype=np.uint8)
        buffers = (
            cuda.to_device(xs32, stream=stream),
            cuda.to_device(ys32, stream=stream),
            cuda.to_device(valid.ravel(), stream=stream),
        )
        self._direct_grid_key = key
        self._direct_grid_refs = (xs, ys, valid_mask)
        self._direct_grid_buffers = buffers
        return buffers

    def influence_mask(self, xs: np.ndarray, ys: np.ndarray, links: np.ndarray,
                       valid_mask: Optional[np.ndarray]) -> np.ndarray:
        xs32 = np.ascontiguousarray(xs, dtype=np.float32)
        ys32 = np.ascontiguousarray(ys, dtype=np.float32)
        links32 = np.ascontiguousarray(links, dtype=np.float32).reshape(-1, 8)
        valid = np.ones((len(ys32), len(xs32)), dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask, dtype=np.uint8)
        total = valid.size
        with self._lock, cuda.gpus[self.device_index]:
            stream = cuda.stream()
            d_xs = cuda.to_device(xs32, stream=stream)
            d_ys = cuda.to_device(ys32, stream=stream)
            d_links = cuda.to_device(links32, stream=stream)
            d_valid = cuda.to_device(valid.ravel(), stream=stream)
            d_output = cuda.device_array(total, dtype=np.uint8, stream=stream)
            blocks, threads = self._launch_shape(total)
            _cuda_influence_mask[blocks, threads, stream](d_xs, d_ys, d_links, d_valid, d_output, len(xs32), len(ys32))
            output = d_output.copy_to_host(stream=stream)
            stream.synchronize()
        return output.reshape((len(ys32), len(xs32))).astype(bool)

    def strongest_indices(self, stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        source = np.ascontiguousarray(stack, dtype=np.float32)
        fields = int(source.shape[0])
        points = int(np.prod(source.shape[1:]))
        flat = source.reshape(fields, points)
        with self._lock, cuda.gpus[self.device_index]:
            stream = cuda.stream()
            d_source = cuda.to_device(flat, stream=stream)
            d_indices = cuda.device_array(points, dtype=np.int32, stream=stream)
            d_valid = cuda.device_array(points, dtype=np.uint8, stream=stream)
            blocks, threads = self._launch_shape(points)
            _cuda_strongest_index[blocks, threads, stream](d_source, fields, points, d_indices, d_valid)
            indices = d_indices.copy_to_host(stream=stream)
            valid = d_valid.copy_to_host(stream=stream)
            stream.synchronize()
        shape = source.shape[1:]
        return indices.reshape(shape), valid.reshape(shape).astype(bool)

    def resample(self, source_xs: np.ndarray, source_ys: np.ndarray, source_values: np.ndarray,
                 target_xs: np.ndarray, target_ys: np.ndarray) -> np.ndarray:
        source = np.ascontiguousarray(source_values, dtype=np.float32)
        tx = np.ascontiguousarray(target_xs, dtype=np.float32)
        ty = np.ascontiguousarray(target_ys, dtype=np.float32)
        output = np.empty(len(tx) * len(ty), dtype=np.float32)
        source_dx = float(source_xs[1] - source_xs[0]) if len(source_xs) > 1 else 1.0
        source_dy = float(source_ys[1] - source_ys[0]) if len(source_ys) > 1 else 1.0
        with self._lock, cuda.gpus[self.device_index]:
            stream = cuda.stream()
            d_source = cuda.to_device(source, stream=stream)
            d_tx = cuda.to_device(tx, stream=stream)
            d_ty = cuda.to_device(ty, stream=stream)
            d_output = cuda.device_array(output.size, dtype=np.float32, stream=stream)
            blocks, threads = self._launch_shape(output.size)
            _cuda_resample_bilinear[blocks, threads, stream](
                d_source, source.shape[1], source.shape[0], np.float32(source_xs[0]),
                np.float32(source_ys[0]), np.float32(source_dx), np.float32(source_dy),
                d_tx, d_ty, len(tx), len(ty), d_output,
            )
            output = d_output.copy_to_host(stream=stream)
            stream.synchronize()
        return output.reshape((len(ty), len(tx))).astype(np.float64)

    def colourise(self, values: np.ndarray, zones: Sequence[Tuple[float, float, int, int, int, int]],
                  blocking: bool = True) -> Optional[np.ndarray]:
        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            return None
        try:
            source = np.ascontiguousarray(values, dtype=np.float32)
            mins = np.ascontiguousarray([zone[0] for zone in zones], dtype=np.float32)
            maxs = np.ascontiguousarray([zone[1] for zone in zones], dtype=np.float32)
            colours = np.ascontiguousarray([[zone[2], zone[3], zone[4], zone[5]] for zone in zones], dtype=np.uint8)
            high_index = int(np.argmax(maxs))
            low_index = int(np.argmin(mins))
            with cuda.gpus[self.device_index]:
                stream = cuda.stream()
                d_source = cuda.to_device(source.ravel(), stream=stream)
                d_mins = cuda.to_device(mins, stream=stream)
                d_maxs = cuda.to_device(maxs, stream=stream)
                d_colours = cuda.to_device(colours, stream=stream)
                d_output = cuda.device_array((source.size, 4), dtype=np.uint8, stream=stream)
                blocks, threads = self._launch_shape(source.size)
                _cuda_colourise[blocks, threads, stream](
                    d_source, source.size, d_mins, d_maxs, d_colours, len(zones),
                    high_index, low_index, d_output,
                )
                output = d_output.copy_to_host(stream=stream)
                stream.synchronize()
            return output.reshape((*source.shape, 4))
        finally:
            self._lock.release()

    def direct_rssi_grid(self, xs: np.ndarray, ys: np.ndarray, ap_data: np.ndarray,
                         segments: np.ndarray, valid_mask: Optional[np.ndarray],
                         disconnected: float, combine_mode: int, settings: Any = None,
                         receiver_z_grid: Optional[np.ndarray] = None,
                         terrain_elevation_grid: Optional[np.ndarray] = None,
                         ground_loss_db: float = 0.0,
                         terrain_clearance_m: float = 0.5,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        compact_aps, prepared_segments, segment_indices, segment_offsets = _prepare_direct_inputs(
            ap_data, segments, np.float32
        )
        rows, cols = len(ys), len(xs)
        total = rows * cols
        if total <= 0:
            return np.empty((rows, cols), dtype=np.float32), np.zeros((rows, cols), dtype=np.int16)
        ap_count = int(compact_aps.shape[0])
        configured_chunk = max(65_536, int(_setting(settings, "cuda_chunk_points", 1_048_576) or 1_048_576))
        memory_fraction = max(0.10, min(0.75, float(_setting(settings, "cuda_memory_fraction", 0.45) or 0.45)))
        block_threshold = max(1, int(_setting(settings, "cuda_block_segment_threshold", 48) or 48))
        queue_depth = max(1, min(8, int(_setting(settings, "cuda_queue_depth", 3) or 3)))
        max_barrier_checks = max(10_000_000, int(_setting(
            settings, "cuda_max_barrier_checks_per_launch", 200_000_000
        ) or 200_000_000))
        blocks_per_sm = max(4, min(64, int(_setting(settings, "cuda_blocks_per_sm", 24) or 24)))
        packed_segment_count = int(segment_offsets[-1]) if len(segment_offsets) else 0
        average_segments = (packed_segment_count / ap_count) if ap_count else 0.0
        use_block_kernel = bool(average_segments >= block_threshold)
        cooperative_threads = 32 if average_segments < 64 else (64 if average_segments < 128 else 128)

        with self._lock, cuda.gpus[self.device_index]:
            self._blocks_per_sm = blocks_per_sm
            self._maximum_blocks = max(self._minimum_blocks, self._sm_count * self._blocks_per_sm)
            stream = cuda.stream()
            d_xs, d_ys, d_valid = self._grid_buffers(xs, ys, valid_mask, stream)
            d_aps, d_segments, d_segment_indices, d_offsets = self._model_buffers(
                compact_aps, prepared_segments, segment_indices, segment_offsets, stream
            )
            terrain_enabled = (
                receiver_z_grid is not None
                and terrain_elevation_grid is not None
                and float(ground_loss_db) > 0.0
            )
            if terrain_enabled:
                receiver_z = np.ascontiguousarray(receiver_z_grid, dtype=np.float32).reshape(rows, cols)
                terrain_z = np.ascontiguousarray(terrain_elevation_grid, dtype=np.float32).reshape(rows, cols)
            else:
                receiver_z = np.zeros((1,), dtype=np.float32)
                terrain_z = np.zeros((1,), dtype=np.float32)
            d_receiver_z = cuda.to_device(receiver_z.ravel(), stream=stream)
            d_terrain_z = cuda.to_device(terrain_z.ravel(), stream=stream)

            # Scale the point/AP temporary field to the currently free device
            # memory rather than an arbitrary 256 MB ceiling.  This cuts launch
            # and synchronisation overhead on high-memory NVIDIA cards while
            # reserving enough memory for the display driver and other stages.
            try:
                free_bytes, _total_bytes = cuda.current_context().get_memory_info()
                free_bytes = int(free_bytes)
            except Exception:
                free_bytes = int(self.info.global_memory_mb or 2048) * 1024 * 1024
            static_bytes = (
                int(compact_aps.nbytes) + int(prepared_segments.nbytes) +
                int(segment_indices.nbytes) + int(segment_offsets.nbytes) +
                int(total * (np.dtype(np.float32).itemsize + np.dtype(np.int16).itemsize))
            )
            reserve_bytes = max(256 * 1024 * 1024, static_bytes * 2)
            usable_bytes = max(32 * 1024 * 1024, free_bytes - reserve_bytes)
            temp_budget_bytes = max(32 * 1024 * 1024, int(usable_bytes * memory_fraction))
            temp_budget_bytes = min(temp_budget_bytes, 2 * 1024 * 1024 * 1024)
            max_points_by_temp = max(4096, temp_budget_bytes // max(4, ap_count * 4))
            # The saved chunk value is a cancellation/latency baseline.  Auto
            # scaling may enlarge it when memory permits so the GPU receives a
            # deep work queue and spends less time at Python synchronisation
            # boundaries.
            auto_target = max(configured_chunk, min(total, 2_097_152))
            max_points_by_cancel = total
            effective_queue_depth = queue_depth
            if use_block_kernel and average_segments > 0.0:
                # Bound one cooperative barrier kernel to a predictable amount
                # of work so Cancel remains responsive even on very large IFCs.
                # The chunk still contains tens of thousands of CUDA blocks on
                # normal plans, which is ample to saturate the GPU.
                max_points_by_cancel = max(1024, int(
                    max_barrier_checks / max(1.0, ap_count * average_segments)
                ))
                # Keep cooperative barrier chunks bounded for cancellation, but
                # still queue a small number on independent streams.  The old
                # single-stream path often under-fed the GPU on large IFCs when
                # each chunk contained fewer blocks than the adapter could keep
                # resident.
                effective_queue_depth = max(1, min(queue_depth, 4))
            # Each queued stream needs its own AP/point temporary field.
            # Divide the memory-derived point budget by the actual stream count
            # so queue depth improves utilisation without oversubscribing VRAM.
            max_points_by_temp_stream = max(4096, int(max_points_by_temp / max(1, effective_queue_depth)))
            chunk_points = max(1, min(total, auto_target, max_points_by_temp_stream, max_points_by_cancel))
            max_path_items = chunk_points * ap_count
            d_rssi, d_counts, d_path_rssi = self._ensure_direct_buffers(total, max_path_items, stream)
            stream.synchronize()

            streams = [stream]
            path_buffers = [d_path_rssi]
            if effective_queue_depth > 1:
                streams = [cuda.stream() for _ in range(effective_queue_depth)]
                path_buffers = [
                    cuda.device_array(max_path_items, dtype=np.float32, stream=batch_stream)
                    for batch_stream in streams
                ]

            queued_chunks = 0
            chunk_count = 0
            for start_gid in range(0, total, chunk_points):
                if self._cancelled(settings):
                    raise RuntimeError("RSSI calculation cancelled")
                work_count = min(chunk_points, total - start_gid)
                path_items = work_count * ap_count
                batch_index = chunk_count % len(streams)
                batch_stream = streams[batch_index]
                batch_path_rssi = path_buffers[batch_index]
                if use_block_kernel:
                    path_blocks, threads = self._block_launch_shape(path_items, cooperative_threads)
                    _cuda_direct_rssi_paths_block[path_blocks, threads, batch_stream](
                        d_xs, d_ys, d_valid, d_aps, d_segments, d_segment_indices, d_offsets,
                        d_receiver_z, d_terrain_z, np.int32(1 if terrain_enabled else 0),
                        np.float32(ground_loss_db), np.float32(terrain_clearance_m),
                        np.float32(disconnected), np.int64(start_gid), np.int64(work_count),
                        batch_path_rssi, np.int32(cols), np.int32(rows),
                    )
                else:
                    path_blocks, threads = self._launch_shape(path_items)
                    _cuda_direct_rssi_paths[path_blocks, threads, batch_stream](
                        d_xs, d_ys, d_valid, d_aps, d_segments, d_segment_indices, d_offsets,
                        d_receiver_z, d_terrain_z, np.int32(1 if terrain_enabled else 0),
                        np.float32(ground_loss_db), np.float32(terrain_clearance_m),
                        np.float32(disconnected), np.int64(start_gid), np.int64(work_count),
                        batch_path_rssi, np.int32(cols), np.int32(rows),
                    )
                reduce_blocks, reduce_threads = self._launch_shape(work_count)
                _cuda_reduce_rssi_paths[reduce_blocks, reduce_threads, batch_stream](
                    d_valid, batch_path_rssi, np.int32(ap_count), np.float32(disconnected),
                    np.int32(combine_mode), np.int64(start_gid), np.int64(work_count),
                    d_rssi, d_counts,
                )
                queued_chunks += 1
                chunk_count += 1
                is_last = (start_gid + work_count) >= total
                # Queue several dependent path/reduction pairs before waiting.
                # They use one stream, so the temporary buffer is reused safely
                # while the GPU remains continuously fed.
                if queued_chunks >= effective_queue_depth or is_last:
                    for batch_stream in streams:
                        batch_stream.synchronize()
                    queued_chunks = 0
                    if self._cancelled(settings):
                        raise RuntimeError("RSSI calculation cancelled")

            try:
                host_rssi = cuda.pinned_array(total, dtype=np.float32)
                host_counts = cuda.pinned_array(total, dtype=np.int16)
            except Exception:
                host_rssi = np.empty(total, dtype=np.float32)
                host_counts = np.empty(total, dtype=np.int16)
            d_rssi[:total].copy_to_host(host_rssi, stream=stream)
            d_counts[:total].copy_to_host(host_counts, stream=stream)
            stream.synchronize()
            if self._cancelled(settings):
                raise RuntimeError("RSSI calculation cancelled")
            rssi = np.asarray(host_rssi).copy()
            counts = np.asarray(host_counts).copy()
            self.last_direct_stats = {
                "kernel": "block-per-path" if use_block_kernel else "thread-per-path",
                "average_segments_per_ap": float(average_segments),
                "cooperative_threads": int(cooperative_threads if use_block_kernel else 1),
                "chunk_points": int(chunk_points),
                "chunks": int(chunk_count),
                "chunk_memory_budget_points": int(max_points_by_temp_stream),
                "queue_depth": int(effective_queue_depth),
                "streams": int(len(streams)),
                "max_barrier_checks_per_launch": int(max_barrier_checks),
                "path_items": int(total * ap_count),
                "temporary_mb": float(max_path_items * 4 / (1024 * 1024)),
            }
        return rssi.reshape((rows, cols)), counts.reshape((rows, cols))

    def inspect_direct_paths(self, receiver_x, receiver_y, ap_data, segments, disconnected, settings=None):
        compact_aps, prepared_segments, segment_indices, segment_offsets = _prepare_direct_inputs(
            ap_data, segments, np.float32
        )
        ap_count = int(compact_aps.shape[0])
        if ap_count <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        with self._lock, cuda.gpus[self.device_index]:
            stream = cuda.stream()
            d_aps = cuda.to_device(compact_aps, stream=stream)
            d_segments = cuda.to_device(prepared_segments, stream=stream)
            d_segment_indices = cuda.to_device(segment_indices, stream=stream)
            d_offsets = cuda.to_device(segment_offsets, stream=stream)
            d_output = cuda.device_array((ap_count, 4), dtype=np.float32, stream=stream)
            blocks, threads = self._launch_shape(ap_count)
            _cuda_inspect_direct_paths[blocks, threads, stream](
                d_aps, d_segments, d_segment_indices, d_offsets,
                np.float32(receiver_x), np.float32(receiver_y), np.float32(disconnected), d_output,
            )
            output = d_output.copy_to_host(stream=stream)
            stream.synchronize()
        return np.asarray(output, dtype=np.float32).reshape(ap_count, 4)


# ------------------------------ OpenCL fallback -----------------------------
_KERNEL_SOURCE = r"""
__kernel void influence_mask(__global const float *xs, __global const float *ys,
 __global const float *links, const int link_count, __global const uchar *valid,
 __global uchar *output, const int cols, const int rows) {
 const int total=cols*rows;
 for(int gid=get_global_id(0);gid<total;gid+=get_global_size(0)){
  if(valid[gid]==0){output[gid]=0;continue;} const int ix=gid%cols; const int iy=gid/cols;
  const float x=xs[ix], y=ys[iy]; uchar keep=0;
  for(int link=0;link<link_count;++link){const int base=link*8; const float dx=x-links[base];
   const float dy=y-links[base+1], dz=links[base+2]; float d2=dx*dx+dy*dy+dz*dz;
   const float r2=links[base+7]; if(r2>0.0f&&d2>r2)continue; d2=fmax(d2,1.0f);
   const float upper=links[base+5]-links[base+3]-10.0f*links[base+4]*log10(sqrt(d2));
   if(upper>=links[base+6]-0.25f){keep=1;break;}}
  output[gid]=keep;
 }
}
__kernel void strongest_index(__global const float *stack,const int fields,const int points,
 __global int *indices,__global uchar *valid){
 for(int p=get_global_id(0);p<points;p+=get_global_size(0)){
  float best=-INFINITY;int bi=0;uchar any=0;for(int f=0;f<fields;++f){const float v=stack[f*points+p];
  if(isfinite(v)){any=1;if(v>best){best=v;bi=f;}}}indices[p]=bi;valid[p]=any;
 }
}
__kernel void resample_bilinear(__global const float *source,const int sc,const int sr,
 const float x0,const float y0,const float dx,const float dy,__global const float *txs,
 __global const float *tys,const int tc,const int tr,__global float *out){
 const int total=tc*tr;
 for(int gid=get_global_id(0);gid<total;gid+=get_global_size(0)){
  const int tx=gid%tc,ty=gid/tc;float fx=clamp((txs[tx]-x0)/dx,0.0f,(float)(sc-1));
  float fy=clamp((tys[ty]-y0)/dy,0.0f,(float)(sr-1));const int xa=(int)floor(fx),ya=(int)floor(fy);
  const int xb=min(xa+1,sc-1),yb=min(ya+1,sr-1);const float wx=fx-xa,wy=fy-ya;
  const int ids[4]={ya*sc+xa,ya*sc+xb,yb*sc+xa,yb*sc+xb};const float ws[4]={(1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy};
  float vs=0,ww=0;for(int k=0;k<4;++k){const float v=source[ids[k]];if(isfinite(v)){vs+=v*ws[k];ww+=ws[k];}}
  out[gid]=ww>1e-8f?vs/ww:NAN;
 }
}
__kernel void colourise(__global const float *values,const int points,__global const float *mins,
 __global const float *maxs,__global const uchar4 *colours,const int count,const int hi,const int lo,
 __global uchar4 *out){
 for(int gid=get_global_id(0);gid<points;gid+=get_global_size(0)){
  const float v=values[gid];if(!isfinite(v)){out[gid]=(uchar4)(0,0,0,0);continue;}int selected=-1;
  for(int z=0;z<count;++z)if(v>=mins[z]&&v<maxs[z]){selected=z;break;}
  if(selected<0)selected=v>=maxs[hi]?hi:lo;out[gid]=colours[selected];
 }
}

inline int line_box_z_hit(const float ax,const float ay,const float az,const float dx,const float dy,const float z_delta,
 const float min_x,const float max_x,const float min_y,const float max_y,const float z_min,const float z_max){
 float t0=0.0f,t1=1.0f;
 if(fabs(dx)<1.0e-7f){if(ax<min_x||ax>max_x)return 0;}
 else{float tx1=(min_x-ax)/dx,tx2=(max_x-ax)/dx;if(tx1>tx2){float tmp=tx1;tx1=tx2;tx2=tmp;}
  if(tx1>t0)t0=tx1;if(tx2<t1)t1=tx2;if(t0>t1)return 0;}
 if(fabs(dy)<1.0e-7f){if(ay<min_y||ay>max_y)return 0;}
 else{float ty1=(min_y-ay)/dy,ty2=(max_y-ay)/dy;if(ty1>ty2){float tmp=ty1;ty1=ty2;ty2=tmp;}
  if(ty1>t0)t0=ty1;if(ty2<t1)t1=ty2;if(t0>t1)return 0;}
 if(t1<0.0f||t0>1.0f)return 0;if(t0<0.0f)t0=0.0f;if(t1>1.0f)t1=1.0f;
 const float za=az+z_delta*t0,zb=az+z_delta*t1;const float zlo=fmin(za,zb),zhi=fmax(za,zb);
 return (zhi>=z_min&&zlo<=z_max)?1:0;
}
__kernel void direct_rssi_paths(__global const float *xs,__global const float *ys,
 __global const uchar *valid,__global const float *aps,__global const float *segments,
 __global const int *segment_indices,__global const int *offsets,__global const float *receiver_z,
 __global const float *terrain_z,const int terrain_enabled,const float ground_loss_db,
 const float terrain_clearance,const int ap_count,const float disconnected,const int start_gid,
 const int work_count,__global float *path_rssi,const int cols,const int rows){
 const int total=cols*rows;const int path_total=work_count*ap_count;
 for(int path_index=get_global_id(0);path_index<path_total;path_index+=get_global_size(0)){
  const int ai=path_index/work_count;const int local_gid=path_index-ai*work_count;
  const int gid=start_gid+local_gid;if(gid>=total||valid[gid]==0){path_rssi[path_index]=NAN;continue;}
  const int ix=gid%cols,iy=gid/cols;const float x=xs[ix],y=ys[iy];const int ab=ai*8;
  const float ax=aps[ab],ay=aps[ab+1],az=aps[ab+2];
  float z_delta=aps[ab+3];
  const float base_dbm=aps[ab+4],path_loss_factor=aps[ab+5],cutoff2=aps[ab+6];
  float dz2=aps[ab+7];
  if(terrain_enabled!=0){z_delta=receiver_z[gid]-az;dz2=z_delta*z_delta;}
  const float dx=x-ax,dy=y-ay;const float d2xy=dx*dx+dy*dy;
  if(cutoff2>0.0f&&d2xy>cutoff2){path_rssi[path_index]=NAN;continue;}
  float d3=sqrt(d2xy+dz2);d3=fmax(d3,1.0f);float wall_loss=0.0f;
  const float ray_min_x=fmin(ax,x),ray_max_x=fmax(ax,x),ray_min_y=fmin(ay,y),ray_max_y=fmax(ay,y);
  for(int pi=offsets[ai];pi<offsets[ai+1];++pi){const int si=segment_indices[pi];const int sb=si*12;
   if(segments[sb+8]<ray_min_x-1.0e-5f||segments[sb+7]>ray_max_x+1.0e-5f||segments[sb+10]<ray_min_y-1.0e-5f||segments[sb+9]>ray_max_y+1.0e-5f)continue;
   if(segments[sb+11]>=0.5f){if(line_box_z_hit(ax,ay,az,dx,dy,z_delta,segments[sb+7],segments[sb+8],segments[sb+9],segments[sb+10],segments[sb+4],segments[sb+5]))wall_loss+=segments[sb+6];continue;}
   const float x1=segments[sb],y1=segments[sb+1],sdx=segments[sb+2],sdy=segments[sb+3];
   const float den=dx*sdy-dy*sdx;if(fabs(den)<1.0e-7f)continue;
   const float t=((x1-ax)*sdy-(y1-ay)*sdx)/den;
   const float u=((x1-ax)*dy-(y1-ay)*dx)/den;
   if(t>=0.0f&&t<=1.0f&&u>=0.0f&&u<=1.0f){const float zhit=az+z_delta*t;
    if(zhit>=segments[sb+4]&&zhit<=segments[sb+5])wall_loss+=segments[sb+6];}}
  if(terrain_enabled!=0&&ground_loss_db>0.0f){
   const float sx0=xs[0],sy0=ys[0],sx1=xs[cols-1],sy1=ys[rows-1];
   float spacing_x=cols>1?fabs(xs[1]-xs[0]):2.0f;float spacing_y=rows>1?fabs(ys[1]-ys[0]):2.0f;
   float spacing=fmin(spacing_x,spacing_y);if(spacing<1.0f)spacing=1.0f;
   int steps=(int)ceil(sqrt(d2xy)/spacing);if(steps<4)steps=4;else if(steps>32)steps=32;
   for(int step=1;step<steps;++step){
    const float tline=(float)step/(float)steps;const float sx=ax+dx*tline;const float sy=ay+dy*tline;
    int tx=cols>1?(int)floor(((sx-sx0)/(sx1-sx0)*(float)(cols-1))+0.5f):0;
    int ty=rows>1?(int)floor(((sy-sy0)/(sy1-sy0)*(float)(rows-1))+0.5f):0;
    if(tx<0)tx=0;else if(tx>=cols)tx=cols-1;if(ty<0)ty=0;else if(ty>=rows)ty=rows-1;
    const float los_z=az+z_delta*tline;if(terrain_z[ty*cols+tx]>=los_z-terrain_clearance){wall_loss+=ground_loss_db;break;}
   }
  }
  float rssi=base_dbm-path_loss_factor*log10(d3)-wall_loss;path_rssi[path_index]=fmax(rssi,disconnected);
 }
}

__kernel void reduce_rssi_paths(__global const uchar *valid,__global const float *path_rssi,
 const int ap_count,const float disconnected,const int combine_mode,const int start_gid,
 const int work_count,__global float *out_rssi,__global short *out_counts){
 for(int local_gid=get_global_id(0);local_gid<work_count;local_gid+=get_global_size(0)){
  const int gid=start_gid+local_gid;if(valid[gid]==0){out_rssi[gid]=NAN;out_counts[gid]=0;continue;}
  float best=disconnected,power_sum=0.0f;int path_count=0;
  for(int ai=0;ai<ap_count;++ai){const float rssi=path_rssi[ai*work_count+local_gid];if(!isfinite(rssi))continue;
   ++path_count;if(combine_mode==1)power_sum+=pow(10.0f,rssi/10.0f);else if(rssi>best)best=rssi;}
  if(combine_mode==1&&power_sum>0.0f)best=10.0f*log10(power_sum);if(path_count==0)best=disconnected;
  out_rssi[gid]=best;out_counts[gid]=(short)(combine_mode==1?path_count:(path_count>0?1:0));
 }
}
__kernel void inspect_direct_paths(__global const float *aps,__global const float *segments,
 __global const int *segment_indices,__global const int *offsets,const int ap_count,
 const float x,const float y,const float disconnected,__global float *out){
 for(int ai=get_global_id(0);ai<ap_count;ai+=get_global_size(0)){
  const int ab=ai*8;const float ax=aps[ab],ay=aps[ab+1],az=aps[ab+2],z_delta=aps[ab+3];
  const float base_dbm=aps[ab+4],path_loss_factor=aps[ab+5],cutoff2=aps[ab+6],dz2=aps[ab+7];
  const float dx=x-ax,dy=y-ay;const float d2xy=dx*dx+dy*dy;const int ob=ai*4;
  if(cutoff2>0.0f&&d2xy>cutoff2){out[ob]=disconnected;out[ob+1]=sqrt(fmax(d2xy+dz2,1.0f));out[ob+2]=0.0f;out[ob+3]=0.0f;continue;}
  const float d3=fmax(sqrt(d2xy+dz2),1.0f);float wall_loss=0.0f;
  const float ray_min_x=fmin(ax,x),ray_max_x=fmax(ax,x),ray_min_y=fmin(ay,y),ray_max_y=fmax(ay,y);
  for(int pi=offsets[ai];pi<offsets[ai+1];++pi){const int si=segment_indices[pi];const int sb=si*12;
   if(segments[sb+8]<ray_min_x-1.0e-5f||segments[sb+7]>ray_max_x+1.0e-5f||segments[sb+10]<ray_min_y-1.0e-5f||segments[sb+9]>ray_max_y+1.0e-5f)continue;
   if(segments[sb+11]>=0.5f){if(line_box_z_hit(ax,ay,az,dx,dy,z_delta,segments[sb+7],segments[sb+8],segments[sb+9],segments[sb+10],segments[sb+4],segments[sb+5]))wall_loss+=segments[sb+6];continue;}
   const float x1=segments[sb],y1=segments[sb+1],sdx=segments[sb+2],sdy=segments[sb+3];
   const float den=dx*sdy-dy*sdx;if(fabs(den)<1.0e-7f)continue;
   const float t=((x1-ax)*sdy-(y1-ay)*sdx)/den;
   const float u=((x1-ax)*dy-(y1-ay)*dx)/den;
   if(t>=0.0f&&t<=1.0f&&u>=0.0f&&u<=1.0f){const float zhit=az+z_delta*t;
    if(zhit>=segments[sb+4]&&zhit<=segments[sb+5])wall_loss+=segments[sb+6];}}
  float rssi=base_dbm-path_loss_factor*log10(d3)-wall_loss;
  if(rssi<disconnected)rssi=disconnected;
  out[ob]=rssi;out[ob+1]=d3;out[ob+2]=wall_loss;out[ob+3]=1.0f;
 }
}
"""


class OpenCLRFAccelerator:
    backend_name = "OpenCL"

    def __init__(self, preference: str = "auto", allow_cpu: bool = False):
        if cl is None:
            raise RuntimeError("PyOpenCL is not installed")
        self.preference = str(preference or "auto").strip().lower()
        self.allow_cpu = bool(allow_cpu)
        self._lock = threading.RLock()
        self.device = self._choose_device()
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, self.device)
        self.program = cl.Program(self.context, _KERNEL_SOURCE).build()
        self._influence_kernel = cl.Kernel(self.program, "influence_mask")
        self._strongest_kernel = cl.Kernel(self.program, "strongest_index")
        self._resample_kernel = cl.Kernel(self.program, "resample_bilinear")
        self._colour_kernel = cl.Kernel(self.program, "colourise")
        self._direct_path_kernel = cl.Kernel(self.program, "direct_rssi_paths")
        self._direct_reduce_kernel = cl.Kernel(self.program, "reduce_rssi_paths")
        self._inspect_direct_kernel = cl.Kernel(self.program, "inspect_direct_paths")
        platform = getattr(self.device, "platform", None)
        self.info = OpenCLDeviceInfo(
            platform=str(getattr(platform, "name", "OpenCL") or "OpenCL").strip(),
            vendor=str(getattr(self.device, "vendor", "Unknown") or "Unknown").strip(),
            name=str(getattr(self.device, "name", "OpenCL device") or "OpenCL device").strip(),
            device_type=_device_type_name(self.device),
            global_memory_mb=max(0, int(getattr(self.device, "global_mem_size", 0) or 0) // (1024 * 1024)),
            compute_units=max(0, int(getattr(self.device, "max_compute_units", 0) or 0)),
            supports_fp64=("cl_khr_fp64" in str(getattr(self.device, "extensions", "") or "").lower()),
            token="opencl:selected",
        )
        max_group = max(1, int(getattr(self.device, "max_work_group_size", 128) or 128))
        self.local_size = min(128, max_group)
        self._minimum_groups = max(16, int(self.info.compute_units or 1) * 4)

    def _choose_device(self):
        candidates = []
        stable_index = 0
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                dtype = _device_type_name(device)
                if dtype == "OpenCL CPU" and not self.allow_cpu:
                    continue
                vendor = str(getattr(device, "vendor", "") or "").strip()
                name = str(getattr(device, "name", "") or "").strip()
                token = f"opencl:{stable_index}:{vendor} {name}".lower()
                short_token = f"opencl:{stable_index}"
                text = " ".join([token, short_token, str(getattr(platform, "name", "")), vendor, name, dtype]).lower()
                preferred = self.preference in {"auto", "opencl"} or self.preference == short_token or self.preference in text
                priority = 0 if dtype == "OpenCL GPU" else (1 if "accelerator" in dtype else 2)
                candidates.append((0 if preferred else 1, priority, stable_index, -int(getattr(device, "global_mem_size", 0) or 0), device))
                stable_index += 1
        if not candidates:
            raise RuntimeError("No suitable OpenCL GPU/accelerator was found")
        candidates.sort(key=lambda item: item[:4])
        return candidates[0][4]

    def _launch_shape(self, count: int):
        count = max(1, int(count))
        groups = max(1, (count + self.local_size - 1) // self.local_size)
        if count >= self.local_size * 8:
            groups = max(groups, self._minimum_groups)
        return (groups * self.local_size,), (self.local_size,)

    @staticmethod
    def _cancelled(settings: Any) -> bool:
        event = getattr(settings, "_cancel_event", None) if settings is not None else None
        return bool(event is not None and event.is_set())

    def influence_mask(self, xs, ys, links, valid_mask):
        xs32=np.ascontiguousarray(xs,dtype=np.float32);ys32=np.ascontiguousarray(ys,dtype=np.float32)
        links32=np.ascontiguousarray(links,dtype=np.float32).reshape(-1,8)
        valid=np.ones((len(ys32),len(xs32)),dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask,dtype=np.uint8)
        output=np.empty(valid.size,dtype=np.uint8);mf=cl.mem_flags
        with self._lock:
            args=[cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=xs32),cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=ys32),cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=links32),np.int32(len(links32)),cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=valid.ravel()),cl.Buffer(self.context,mf.WRITE_ONLY,output.nbytes),np.int32(len(xs32)),np.int32(len(ys32))]
            self._influence_kernel.set_args(*args);global_size,local_size=self._launch_shape(output.size)
            cl.enqueue_nd_range_kernel(self.queue,self._influence_kernel,global_size,local_size)
            cl.enqueue_copy(self.queue,output,args[5]).wait()
        return output.reshape((len(ys32),len(xs32))).astype(bool)

    def strongest_indices(self, stack):
        source=np.ascontiguousarray(stack,dtype=np.float32);fields=int(source.shape[0]);points=int(np.prod(source.shape[1:]));indices=np.empty(points,dtype=np.int32);valid=np.empty(points,dtype=np.uint8);mf=cl.mem_flags
        with self._lock:
            src=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=source.ravel());idx=cl.Buffer(self.context,mf.WRITE_ONLY,indices.nbytes);vb=cl.Buffer(self.context,mf.WRITE_ONLY,valid.nbytes)
            self._strongest_kernel.set_args(src,np.int32(fields),np.int32(points),idx,vb);global_size,local_size=self._launch_shape(points)
            cl.enqueue_nd_range_kernel(self.queue,self._strongest_kernel,global_size,local_size);cl.enqueue_copy(self.queue,indices,idx);cl.enqueue_copy(self.queue,valid,vb).wait()
        shape=source.shape[1:];return indices.reshape(shape),valid.reshape(shape).astype(bool)

    def resample(self, source_xs, source_ys, source_values, target_xs, target_ys):
        source=np.ascontiguousarray(source_values,dtype=np.float32);tx=np.ascontiguousarray(target_xs,dtype=np.float32);ty=np.ascontiguousarray(target_ys,dtype=np.float32);output=np.empty((len(ty),len(tx)),dtype=np.float32);mf=cl.mem_flags
        dx=float(source_xs[1]-source_xs[0]) if len(source_xs)>1 else 1.;dy=float(source_ys[1]-source_ys[0]) if len(source_ys)>1 else 1.
        with self._lock:
            src=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=source.ravel());txb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=tx);tyb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=ty);out=cl.Buffer(self.context,mf.WRITE_ONLY,output.nbytes)
            self._resample_kernel.set_args(src,np.int32(source.shape[1]),np.int32(source.shape[0]),np.float32(source_xs[0]),np.float32(source_ys[0]),np.float32(dx),np.float32(dy),txb,tyb,np.int32(len(tx)),np.int32(len(ty)),out);global_size,local_size=self._launch_shape(output.size)
            cl.enqueue_nd_range_kernel(self.queue,self._resample_kernel,global_size,local_size);cl.enqueue_copy(self.queue,output,out).wait()
        return output.astype(np.float64)

    def colourise(self, values, zones, blocking=True):
        acquired=self._lock.acquire(blocking=blocking)
        if not acquired:return None
        try:
            source=np.ascontiguousarray(values,dtype=np.float32);mins=np.ascontiguousarray([z[0] for z in zones],dtype=np.float32);maxs=np.ascontiguousarray([z[1] for z in zones],dtype=np.float32);colours=np.ascontiguousarray([[z[2],z[3],z[4],z[5]] for z in zones],dtype=np.uint8);output=np.empty((source.size,4),dtype=np.uint8);mf=cl.mem_flags
            src=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=source.ravel());mb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=mins);xb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=maxs);cb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=colours);out=cl.Buffer(self.context,mf.WRITE_ONLY,output.nbytes)
            self._colour_kernel.set_args(src,np.int32(source.size),mb,xb,cb,np.int32(len(zones)),np.int32(int(np.argmax(maxs))),np.int32(int(np.argmin(mins))),out);global_size,local_size=self._launch_shape(source.size)
            cl.enqueue_nd_range_kernel(self.queue,self._colour_kernel,global_size,local_size);cl.enqueue_copy(self.queue,output,out).wait();return output.reshape((*source.shape,4))
        finally:self._lock.release()

    def direct_rssi_grid(self, xs, ys, ap_data, segments, valid_mask, disconnected, combine_mode, settings=None,
                         receiver_z_grid=None, terrain_elevation_grid=None, ground_loss_db=0.0,
                         terrain_clearance_m=0.5):
        compact_aps, prepared_segments, segment_indices, offsets = _prepare_direct_inputs(
            ap_data, segments, np.float32
        )
        xs32 = np.ascontiguousarray(xs, dtype=np.float32)
        ys32 = np.ascontiguousarray(ys, dtype=np.float32)
        rows, cols = len(ys32), len(xs32)
        total = rows * cols
        valid = np.ones((rows, cols), dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask, dtype=np.uint8)
        terrain_enabled = (
            receiver_z_grid is not None
            and terrain_elevation_grid is not None
            and float(ground_loss_db) > 0.0
        )
        if terrain_enabled:
            receiver_z = np.ascontiguousarray(receiver_z_grid, dtype=np.float32).reshape(rows, cols)
            terrain_z = np.ascontiguousarray(terrain_elevation_grid, dtype=np.float32).reshape(rows, cols)
        else:
            receiver_z = np.zeros((1,), dtype=np.float32)
            terrain_z = np.zeros((1,), dtype=np.float32)
        rssi = np.empty(total, dtype=np.float32)
        counts = np.empty(total, dtype=np.int16)
        mf = cl.mem_flags
        configured_chunk = max(65_536, int(_setting(settings, "cuda_chunk_points", 1_048_576) or 1_048_576))
        memory_fraction = max(0.10, min(0.75, float(_setting(settings, "cuda_memory_fraction", 0.45) or 0.45)))
        queue_depth = max(1, min(8, int(_setting(settings, "cuda_queue_depth", 3) or 3)))
        max_barrier_checks = max(10_000_000, int(_setting(
            settings, "cuda_max_barrier_checks_per_launch", 200_000_000
        ) or 200_000_000))
        ap_count = int(compact_aps.shape[0])
        packed_segment_count = int(offsets[-1]) if len(offsets) else 0
        average_segments = (packed_segment_count / ap_count) if ap_count else 0.0
        global_bytes = int(getattr(self.device, "global_mem_size", 0) or (self.info.global_memory_mb * 1024 * 1024) or (2 * 1024 * 1024 * 1024))
        static_bytes = int(xs32.nbytes + ys32.nbytes + valid.nbytes + compact_aps.nbytes + prepared_segments.nbytes + segment_indices.nbytes + offsets.nbytes + rssi.nbytes + counts.nbytes)
        usable_bytes = max(32 * 1024 * 1024, global_bytes - max(256 * 1024 * 1024, static_bytes * 2))
        temp_budget_bytes = max(32 * 1024 * 1024, min(2 * 1024 * 1024 * 1024, int(usable_bytes * memory_fraction)))
        max_points_by_temp = max(4096, temp_budget_bytes // max(4, ap_count * 4))
        auto_target = max(configured_chunk, min(total, 2_097_152))
        max_points_by_cancel = total
        effective_queue_depth = queue_depth
        if average_segments > 0.0:
            max_points_by_cancel = max(4096, int(
                max_barrier_checks / max(1.0, ap_count * average_segments)
            ))
            effective_queue_depth = 1
        chunk_points = max(1, min(total, auto_target, max_points_by_temp, max_points_by_cancel))
        with self._lock:
            xb = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xs32)
            yb = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ys32)
            vb = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=valid.ravel())
            ab = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=compact_aps.ravel())
            sb = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prepared_segments.ravel())
            sib = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=segment_indices)
            ob = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=offsets)
            rzb = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=receiver_z.ravel())
            tzb = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=terrain_z.ravel())
            rb = cl.Buffer(self.context, mf.WRITE_ONLY, rssi.nbytes)
            cb = cl.Buffer(self.context, mf.WRITE_ONLY, counts.nbytes)
            pathb = cl.Buffer(self.context, mf.READ_WRITE, chunk_points * ap_count * np.dtype(np.float32).itemsize)
            queued = 0
            for start_gid in range(0, total, chunk_points):
                if self._cancelled(settings):
                    raise RuntimeError("RSSI calculation cancelled")
                work = min(chunk_points, total - start_gid)
                path_items = work * ap_count
                self._direct_path_kernel.set_args(
                    xb, yb, vb, ab, sb, sib, ob, rzb, tzb,
                    np.int32(1 if terrain_enabled else 0), np.float32(ground_loss_db),
                    np.float32(terrain_clearance_m), np.int32(ap_count), np.float32(disconnected),
                    np.int32(start_gid), np.int32(work), pathb, np.int32(cols), np.int32(rows),
                )
                global_size, local_size = self._launch_shape(path_items)
                cl.enqueue_nd_range_kernel(self.queue, self._direct_path_kernel, global_size, local_size)
                self._direct_reduce_kernel.set_args(
                    vb, pathb, np.int32(ap_count), np.float32(disconnected), np.int32(combine_mode),
                    np.int32(start_gid), np.int32(work), rb, cb,
                )
                global_size, local_size = self._launch_shape(work)
                event = cl.enqueue_nd_range_kernel(self.queue, self._direct_reduce_kernel, global_size, local_size)
                queued += 1
                is_last = (start_gid + work) >= total
                if queued >= effective_queue_depth or is_last:
                    event.wait()
                    queued = 0
                    if self._cancelled(settings):
                        raise RuntimeError("RSSI calculation cancelled")
            cl.enqueue_copy(self.queue, rssi, rb)
            cl.enqueue_copy(self.queue, counts, cb).wait()
        return rssi.reshape((rows, cols)), counts.reshape((rows, cols))

    def inspect_direct_paths(self, receiver_x, receiver_y, ap_data, segments, disconnected, settings=None):
        compact_aps, prepared_segments, segment_indices, offsets = _prepare_direct_inputs(
            ap_data, segments, np.float32
        )
        ap_count = int(compact_aps.shape[0])
        if ap_count <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        output = np.empty((ap_count, 4), dtype=np.float32)
        mf = cl.mem_flags
        with self._lock:
            ab = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=compact_aps.ravel())
            sb = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prepared_segments.ravel())
            sib = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=segment_indices)
            ob = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=offsets)
            out = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
            self._inspect_direct_kernel.set_args(
                ab, sb, sib, ob, np.int32(ap_count), np.float32(receiver_x), np.float32(receiver_y),
                np.float32(disconnected), out,
            )
            global_size, local_size = self._launch_shape(ap_count)
            cl.enqueue_nd_range_kernel(self.queue, self._inspect_direct_kernel, global_size, local_size)
            cl.enqueue_copy(self.queue, output, out).wait()
        return output



# ------------------------- Direct RSSI grid helpers -------------------------
_PREPARED_INPUT_CACHE_LOCK = threading.RLock()
_PREPARED_INPUT_CACHE = OrderedDict()
_PREPARED_INPUT_CACHE_MAX = 8


def _prepared_array_signature(array: np.ndarray) -> Tuple[Any, ...]:
    contiguous = np.ascontiguousarray(array)
    if contiguous.size == 0:
        return (contiguous.shape, contiguous.dtype.str, contiguous.nbytes, 0, 0)
    raw = memoryview(contiguous).cast("B")
    # Two inexpensive checksums make stale/colliding geometry entries
    # vanishingly unlikely while remaining much cheaper than Python AP×segment
    # filtering on large IFC plans.
    return (
        contiguous.shape, contiguous.dtype.str, contiguous.nbytes,
        zlib.crc32(raw) & 0xFFFFFFFF, zlib.adler32(raw) & 0xFFFFFFFF,
    )


def _prepare_direct_inputs(ap_data: np.ndarray, segments: np.ndarray, dtype=np.float32):
    """Compact and cache AP/barrier constants for GPU and Numba kernels.

    Filtering is vectorised per AP instead of iterating every segment in
    Python.  Each segment also carries an XY bounding box, allowing the device
    kernels to reject barriers that cannot intersect the current ray before
    doing division-heavy intersection maths.
    """
    raw_ap_array = np.asarray(ap_data)
    column_count = raw_ap_array.shape[-1] if raw_ap_array.ndim > 1 else 10
    aps = np.ascontiguousarray(np.asarray(ap_data, dtype=np.float64).reshape(-1, column_count))
    if aps.shape[1] < 10:
        raise ValueError("Direct RSSI AP data must contain at least 10 columns")
    raw_segment_array = np.asarray(segments, dtype=np.float64)
    if raw_segment_array.size == 0:
        raw_segments = np.zeros((0, 8), dtype=np.float64)
    else:
        raw_segment_array = np.ascontiguousarray(raw_segment_array)
        if raw_segment_array.ndim == 1:
            if raw_segment_array.size % 8 == 0:
                raw_segments = raw_segment_array.reshape(-1, 8)
            else:
                raw_segments = raw_segment_array.reshape(-1, 7)
        else:
            raw_segments = raw_segment_array.reshape(-1, raw_segment_array.shape[-1])
        if raw_segments.shape[1] < 8:
            raw_segments = np.column_stack([raw_segments[:, :7], np.zeros(raw_segments.shape[0], dtype=np.float64)])
        elif raw_segments.shape[1] > 8:
            raw_segments = raw_segments[:, :8]
        raw_segments = np.ascontiguousarray(raw_segments, dtype=np.float64)
    dtype = np.dtype(dtype)
    cache_key = (dtype.str, _prepared_array_signature(aps), _prepared_array_signature(raw_segments))
    with _PREPARED_INPUT_CACHE_LOCK:
        cached = _PREPARED_INPUT_CACHE.get(cache_key)
        if cached is not None:
            _PREPARED_INPUT_CACHE.move_to_end(cache_key)
            return cached

    segment_count = raw_segments.shape[0]
    if segment_count:
        mode = raw_segments[:, 7]
        solid = mode >= 0.5
        min_x = np.minimum(raw_segments[:, 0], raw_segments[:, 2])
        max_x = np.maximum(raw_segments[:, 0], raw_segments[:, 2])
        min_y = np.minimum(raw_segments[:, 1], raw_segments[:, 3])
        max_y = np.maximum(raw_segments[:, 1], raw_segments[:, 3])
        prepared_segments = np.empty((segment_count, 12), dtype=dtype)
        prepared_segments[:, 0] = raw_segments[:, 0]
        prepared_segments[:, 1] = raw_segments[:, 1]
        prepared_segments[:, 2] = raw_segments[:, 2] - raw_segments[:, 0]
        prepared_segments[:, 3] = raw_segments[:, 3] - raw_segments[:, 1]
        prepared_segments[:, 4:7] = raw_segments[:, 4:7]
        prepared_segments[:, 7] = min_x
        prepared_segments[:, 8] = max_x
        prepared_segments[:, 9] = min_y
        prepared_segments[:, 10] = max_y
        prepared_segments[:, 11] = mode
        nonzero_loss = np.abs(raw_segments[:, 6]) > 1.0e-12
    else:
        # Some OpenCL runtimes reject zero-byte buffers. Offsets remain zero,
        # so this dummy row is never read.
        prepared_segments = np.zeros((1, 12), dtype=dtype)
        min_x = max_x = min_y = max_y = np.zeros(0, dtype=np.float64)
        nonzero_loss = np.zeros(0, dtype=bool)

    compact = np.empty((aps.shape[0], 8), dtype=dtype)
    offsets = np.zeros(aps.shape[0] + 1, dtype=np.int32)
    packed_parts = []
    packed_total = 0
    for ai in range(aps.shape[0]):
        ax, ay, az, rz, tx, gain, freq_mhz, ple, cutoff2, floor_loss = aps[ai, :10]
        fspl_1m = 20.0 * math.log10(max(freq_mhz, 1.0e-12)) + P525_ONE_METRE_MHZ_CONSTANT_DB
        compact[ai] = (
            ax, ay, az, rz - az, tx + gain - fspl_1m - floor_loss, 10.0 * ple, cutoff2,
            (rz - az) * (rz - az),
        )
        if segment_count:
            zlo = min(az, rz)
            zhi = max(az, rz)
            eligible = nonzero_loss & (raw_segments[:, 5] >= zlo) & (raw_segments[:, 4] <= zhi)
            radius2 = float(cutoff2)
            if radius2 > 0.0 and np.any(eligible):
                ddx = np.maximum(np.maximum(min_x - ax, ax - max_x), 0.0)
                ddy = np.maximum(np.maximum(min_y - ay, ay - max_y), 0.0)
                eligible &= (ddx * ddx + ddy * ddy) <= radius2
            indices = np.flatnonzero(eligible).astype(np.int32, copy=False)
            if indices.size:
                packed_parts.append(indices)
                packed_total += int(indices.size)
        offsets[ai + 1] = packed_total
    segment_indices = (
        np.ascontiguousarray(np.concatenate(packed_parts), dtype=np.int32)
        if packed_parts else np.zeros(1, dtype=np.int32)
    )
    result = (
        np.ascontiguousarray(compact),
        np.ascontiguousarray(prepared_segments),
        segment_indices,
        np.ascontiguousarray(offsets),
    )
    with _PREPARED_INPUT_CACHE_LOCK:
        _PREPARED_INPUT_CACHE[cache_key] = result
        _PREPARED_INPUT_CACHE.move_to_end(cache_key)
        while len(_PREPARED_INPUT_CACHE) > _PREPARED_INPUT_CACHE_MAX:
            _PREPARED_INPUT_CACHE.popitem(last=False)
    return result



if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _numba_direct_rssi_grid(xs, ys, valid, ap_data, segments, segment_indices, segment_offsets,
                                disconnected, combine_mode, start_gid, work_count, out, counts):
        cols = xs.shape[0]
        total = ys.shape[0] * cols
        # Flattening the grid gives prange enough independent iterations even
        # when the heatmap has only a small number of rows.
        for local_gid in prange(work_count):
            gid = start_gid + local_gid
            if gid >= total:
                continue
            iy = gid // cols
            ix = gid - iy * cols
            if valid[gid] == 0:
                out[gid] = np.nan
                counts[gid] = 0
                continue
            x = xs[ix]
            y = ys[iy]
            best = disconnected
            power_sum_mw = 0.0
            pc = 0
            for ai in range(ap_data.shape[0]):
                ax = ap_data[ai, 0]; ay = ap_data[ai, 1]; az = ap_data[ai, 2]; z_delta = ap_data[ai, 3]
                base_dbm = ap_data[ai, 4]; path_loss_factor = ap_data[ai, 5]; cutoff2 = ap_data[ai, 6]; dz2 = ap_data[ai, 7]
                dx = x - ax; dy = y - ay
                d2xy = dx * dx + dy * dy
                if cutoff2 > 0.0 and d2xy > cutoff2:
                    continue
                d3 = math.sqrt(d2xy + dz2)
                if d3 < 1.0:
                    d3 = 1.0
                ray_min_x = ax if ax < x else x
                ray_max_x = x if x > ax else ax
                ray_min_y = ay if ay < y else y
                ray_max_y = y if y > ay else ay
                wall_loss = 0.0
                for packed_index in range(segment_offsets[ai], segment_offsets[ai + 1]):
                    si = segment_indices[packed_index]
                    if (segments[si, 8] < ray_min_x - 1.0e-5 or segments[si, 7] > ray_max_x + 1.0e-5 or
                            segments[si, 10] < ray_min_y - 1.0e-5 or segments[si, 9] > ray_max_y + 1.0e-5):
                        continue
                    if segments[si, 11] >= 0.5:
                        t0 = 0.0
                        t1 = 1.0
                        hit_box = True
                        min_x = segments[si, 7]; max_x = segments[si, 8]
                        min_y = segments[si, 9]; max_y = segments[si, 10]
                        if dx > -1.0e-7 and dx < 1.0e-7:
                            if ax < min_x or ax > max_x:
                                hit_box = False
                        else:
                            inv_dx = 1.0 / dx
                            tx1 = (min_x - ax) * inv_dx
                            tx2 = (max_x - ax) * inv_dx
                            if tx1 > tx2:
                                tmp = tx1; tx1 = tx2; tx2 = tmp
                            if tx1 > t0:
                                t0 = tx1
                            if tx2 < t1:
                                t1 = tx2
                            if t0 > t1:
                                hit_box = False
                        if hit_box:
                            if dy > -1.0e-7 and dy < 1.0e-7:
                                if ay < min_y or ay > max_y:
                                    hit_box = False
                            else:
                                inv_dy = 1.0 / dy
                                ty1 = (min_y - ay) * inv_dy
                                ty2 = (max_y - ay) * inv_dy
                                if ty1 > ty2:
                                    tmp = ty1; ty1 = ty2; ty2 = tmp
                                if ty1 > t0:
                                    t0 = ty1
                                if ty2 < t1:
                                    t1 = ty2
                                if t0 > t1:
                                    hit_box = False
                        if hit_box and not (t1 < 0.0 or t0 > 1.0):
                            if t0 < 0.0:
                                t0 = 0.0
                            if t1 > 1.0:
                                t1 = 1.0
                            za = az + z_delta * t0
                            zb = az + z_delta * t1
                            zlo = za if za < zb else zb
                            zhi = zb if zb > za else za
                            if zhi >= segments[si, 4] and zlo <= segments[si, 5]:
                                wall_loss += segments[si, 6]
                        continue
                    x1 = segments[si, 0]; y1 = segments[si, 1]
                    sdx = segments[si, 2]; sdy = segments[si, 3]
                    den = (x - ax) * sdy - (y - ay) * sdx
                    if den > -1.0e-7 and den < 1.0e-7:
                        continue
                    t = ((x1 - ax) * sdy - (y1 - ay) * sdx) / den
                    u = ((x1 - ax) * (y - ay) - (y1 - ay) * (x - ax)) / den
                    if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                        zhit = az + z_delta * t
                        if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                            wall_loss += segments[si, 6]
                rssi = base_dbm - path_loss_factor * math.log10(d3) - wall_loss
                if rssi < disconnected:
                    rssi = disconnected
                pc += 1
                if combine_mode == 1:
                    power_sum_mw += math.pow(10.0, rssi / 10.0)
                elif rssi > best:
                    best = rssi
            if combine_mode == 1 and power_sum_mw > 0.0:
                best = 10.0 * math.log10(power_sum_mw)
            if pc == 0:
                best = disconnected
            out[gid] = best
            counts[gid] = pc if combine_mode == 1 else (1 if pc > 0 else 0)
else:
    _numba_direct_rssi_grid = None


def gpu_direct_rssi_grid(xs: np.ndarray, ys: np.ndarray, ap_data: np.ndarray, segments: np.ndarray,
                         valid_mask: Optional[np.ndarray], disconnected: float, combine_mode: int,
                         settings: Any, receiver_z_grid: Optional[np.ndarray] = None,
                         terrain_elevation_grid: Optional[np.ndarray] = None,
                         ground_loss_db: float = 0.0,
                         terrain_clearance_m: float = 0.5) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    if not _enabled(settings):
        return None
    ap_array = np.asarray(ap_data)
    if ap_array.size == 0:
        return None
    work_items = len(xs) * len(ys) * max(1, int(ap_array.shape[0] if ap_array.ndim > 1 else 1))
    if not _large_enough(settings, work_items):
        return None
    def _run_direct(backend):
        if not hasattr(backend, "direct_rssi_grid"):
            return None
        grids = backend.direct_rssi_grid(
            xs, ys, ap_data, segments, valid_mask, disconnected, combine_mode, settings,
            receiver_z_grid, terrain_elevation_grid, ground_loss_db, terrain_clearance_m
        )
        return grids, backend.info.label, backend.backend_name, dict(getattr(backend, "last_direct_stats", {}) or {})

    result = _execute_gpu(settings, "direct RSSI grid calculation", _run_direct)
    if result is None:
        return None
    (rssi, counts), label, backend_name, stats = result
    detail = ""
    if stats:
        kernel = str(stats.get("kernel", "CUDA"))
        chunks = int(stats.get("chunks", 1) or 1)
        chunk_points = int(stats.get("chunk_points", 0) or 0)
        cooperative_threads = int(stats.get("cooperative_threads", 1) or 1)
        streams = int(stats.get("streams", 1) or 1)
        detail = f"; {kernel}, {chunks} launch chunk(s), {chunk_points:,} points/chunk"
        if streams > 1:
            detail += f", {streams} streams"
        if cooperative_threads > 1:
            detail += f", {cooperative_threads} threads/path"
    return rssi, counts, f"{backend_name}: {label}{detail}"


def gpu_inspect_direct_paths(receiver_x: float, receiver_y: float, ap_data: np.ndarray, segments: np.ndarray,
                             disconnected: float, settings: Any) -> Optional[Tuple[np.ndarray, str]]:
    if not _enabled(settings):
        return None
    ap_array = np.asarray(ap_data)
    if ap_array.size == 0:
        return None

    def _run_inspection(backend):
        if not hasattr(backend, "inspect_direct_paths"):
            return None
        rows = backend.inspect_direct_paths(receiver_x, receiver_y, ap_data, segments, disconnected, settings)
        return rows, backend.info.label, backend.backend_name

    result = _execute_gpu(settings, "direct RSSI path inspection", _run_inspection)
    if result is None:
        return None
    rows, label, backend_name = result
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4), f"{backend_name}: {label}"


def numba_direct_rssi_grid(xs: np.ndarray, ys: np.ndarray, ap_data: np.ndarray, segments: np.ndarray,
                           valid_mask: Optional[np.ndarray], disconnected: float, combine_mode: int,
                           settings: Any = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if _numba_direct_rssi_grid is None or np.asarray(ap_data).size == 0:
        return None
    xs64 = np.ascontiguousarray(xs, dtype=np.float64)
    ys64 = np.ascontiguousarray(ys, dtype=np.float64)
    compact, prepared_segments, segment_indices, offsets = _prepare_direct_inputs(ap_data, segments, np.float64)
    total = len(ys64) * len(xs64)
    valid = np.ones(total, dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask, dtype=np.uint8).ravel()
    out = np.empty(total, dtype=np.float64)
    counts = np.zeros(total, dtype=np.int16)
    cancel_event = getattr(settings, "_cancel_event", None) if settings is not None else None
    # A single call is fastest when cancellation is not being monitored. When
    # it is, use bounded compiled chunks so Cancel is observed quickly even on
    # dense IFCs with many AP/segment checks.  There is no safe way to kill a
    # running Numba CPU kernel from the UI thread, so cooperative chunk sizing is
    # the cancellation mechanism.
    if cancel_event is None:
        chunk_points = total
    else:
        configured_chunk = max(4_096, int(_setting(settings, "numba_chunk_points", 65_536) or 65_536))
        ap_count = max(1, int(compact.shape[0]))
        packed_segment_count = int(offsets[-1]) if len(offsets) else 0
        average_segments = (packed_segment_count / ap_count) if ap_count else 0.0
        max_ops_between_cancel = max(250_000, int(_setting(
            settings, "numba_cancel_check_work_items", 2_000_000
        ) or 2_000_000))
        work_complexity = max(1.0, ap_count * max(1.0, average_segments))
        max_points_by_work = max(512, int(max_ops_between_cancel / work_complexity))
        chunk_points = max(512, min(total, configured_chunk, max_points_by_work))
    for start_gid in range(0, total, chunk_points):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("RSSI calculation cancelled")
        work_count = min(chunk_points, total - start_gid)
        _numba_direct_rssi_grid(
            xs64, ys64, valid, compact, prepared_segments, segment_indices, offsets, float(disconnected),
            int(combine_mode), int(start_gid), int(work_count), out, counts,
        )
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("RSSI calculation cancelled")
    return out.reshape((len(ys64), len(xs64))), counts.reshape((len(ys64), len(xs64)))


_BACKEND_LOCK = threading.RLock()
_BACKENDS = {}
_FAILURES = {}


class _PreferenceOverride:
    def __init__(self, settings: Any, preference: str):
        self._settings = settings
        self.opencl_device_preference = preference

    def __getattr__(self, name: str) -> Any:
        if self._settings is None:
            raise AttributeError(name)
        return getattr(self._settings, name)


def _setting(settings: Any, name: str, default: Any) -> Any:
    return getattr(settings, name, default) if settings is not None else default


def _enabled(settings: Any) -> bool:
    return bool(_setting(settings, "enable_opencl_gpu", True))


def _force(settings: Any) -> bool:
    return bool(_setting(settings, "force_gpu_when_available", True))


def _preference(settings: Any) -> str:
    return str(_setting(settings, "opencl_device_preference", "auto") or "auto").strip().lower()


def _cuda_index(preference: str) -> int:
    if preference.startswith("cuda:"):
        try:
            return max(0, int(preference.split(":", 1)[1]))
        except Exception:
            return 0
    return 0


def _strict_device_preference(settings: Any) -> bool:
    """Return True only when the caller explicitly wants no backend fallback.

    The default is deliberately False: if a saved CUDA preference is no longer
    usable, the simulator should still try an OpenCL GPU before dropping to CPU.
    """
    return bool(_setting(settings, "gpu_strict_device_preference", False))


def _backend_key(settings: Any) -> Tuple[str, bool, bool, bool]:
    return (
        _preference(settings),
        bool(_setting(settings, "opencl_allow_cpu_device", False)),
        _force(settings),
        _strict_device_preference(settings),
    )


def _cuda_requested(preference: str) -> bool:
    return preference == "auto" or preference == "cuda" or preference.startswith("cuda:") or "nvidia" in preference


def _opencl_requested(preference: str) -> bool:
    return preference == "auto" or preference == "opencl" or preference.startswith("opencl:") or not _cuda_requested(preference)


def _has_opencl_gpu() -> bool:
    return any("GPU" in record.device_type for record in _opencl_device_records(False))


def get_existing_opencl_backend(settings: Any):
    if not _enabled(settings):
        return None
    with _BACKEND_LOCK:
        return _BACKENDS.get(_backend_key(settings))


def get_opencl_backend(settings: Any):
    """Return CUDA first, then OpenCL GPU; CPU is used only if no GPU exists.

    A stale saved CUDA preference should not strand the calculation on CPU.
    Unless gpu_strict_device_preference is explicitly enabled, CUDA failure is
    followed by an OpenCL GPU probe.  OpenCL CPU devices remain excluded unless
    opencl_allow_cpu_device is set and no real GPU backend is available.
    """
    if not _enabled(settings):
        return None
    preference = _preference(settings)
    allow_cpu = bool(_setting(settings, "opencl_allow_cpu_device", False))
    strict_preference = _strict_device_preference(settings)
    key = _backend_key(settings)
    with _BACKEND_LOCK:
        if key in _BACKENDS:
            return _BACKENDS[key]
        if key in _FAILURES:
            return None
        errors = []
        if _cuda_requested(preference):
            try:
                backend = NumbaCUDARFAccelerator(_cuda_index(preference))
                _BACKENDS[key] = backend
                return backend
            except Exception as exc:
                errors.append(f"CUDA: {type(exc).__name__}: {exc}")
                if strict_preference and (preference.startswith("cuda") or preference == "cuda"):
                    _FAILURES[key] = "; ".join(errors)
                    return None
        # Try an OpenCL GPU whenever OpenCL is requested, or when CUDA was the
        # requested/preferred backend but was unavailable.  Passing 'auto' for a
        # CUDA fallback avoids treating a stale cuda:0 token as a non-match.
        attempt_opencl = _opencl_requested(preference) or (not strict_preference and _cuda_requested(preference))
        if attempt_opencl:
            opencl_preference = preference if _opencl_requested(preference) else "auto"
            try:
                backend = OpenCLRFAccelerator(opencl_preference, allow_cpu)
                # Never select an OpenCL CPU while any GPU device exists.
                if backend.info.device_type == "OpenCL CPU" and (len(_cuda_device_records()) or _has_opencl_gpu()):
                    if strict_preference:
                        raise RuntimeError("GPU exists; OpenCL CPU fallback refused")
                    errors.append("OpenCL CPU preference ignored because a GPU exists")
                    backend = OpenCLRFAccelerator("auto", False)
                    if backend.info.device_type == "OpenCL CPU":
                        raise RuntimeError("GPU exists; OpenCL CPU fallback refused")
                _BACKENDS[key] = backend
                return backend
            except Exception as exc:
                errors.append(f"OpenCL: {type(exc).__name__}: {exc}")
        _FAILURES[key] = "; ".join(errors) or "No compatible GPU runtime/device"
        return None


def _backend_signature(backend: Any) -> Tuple[str, str, str, str]:
    info = getattr(backend, "info", None)
    return (
        str(getattr(backend, "backend_name", "")),
        str(getattr(info, "platform", "")),
        str(getattr(info, "vendor", "")),
        str(getattr(info, "name", "")),
    )


def _gpu_backend_candidates(settings: Any):
    primary = get_opencl_backend(settings)
    seen = set()
    if primary is not None:
        seen.add(_backend_signature(primary))
        yield primary
    if _preference(settings) != "auto":
        return
    allow_cpu = bool(_setting(settings, "opencl_allow_cpu_device", False))
    for record in _opencl_device_records(include_cpu=allow_cpu):
        if record.device_type == "OpenCL CPU":
            continue
        backend = get_opencl_backend(_PreferenceOverride(settings, record.token.lower()))
        if backend is None:
            continue
        signature = _backend_signature(backend)
        if signature in seen:
            continue
        seen.add(signature)
        yield backend


def reset_opencl_backends() -> None:
    with _BACKEND_LOCK:
        _BACKENDS.clear()
        _FAILURES.clear()


def _preferred_device_info(settings: Any) -> Optional[OpenCLDeviceInfo]:
    preference = _preference(settings)
    devices = discover_opencl_devices(include_cpu=bool(_setting(settings, "opencl_allow_cpu_device", False)))
    if not devices:
        return None
    if preference == "auto":
        return devices[0]
    for device in devices:
        text = f"{device.token} {device.platform} {device.vendor} {device.name} {device.device_type}".lower()
        if preference == device.token.lower() or preference in text:
            return device
    return devices[0]


def opencl_status(settings: Any, initialize: bool = False) -> str:
    """Backward-compatible status function describing the active GPU backend."""
    if not _enabled(settings):
        return "GPU acceleration disabled; CPU path active"
    if not initialize:
        info = _preferred_device_info(settings)
        if info is None:
            return "No compatible GPU detected; CPU fallback active"
        mode = "forced" if _force(settings) else "workload-dependent"
        return f"GPU detected ({mode}): {info.label}"
    backend = get_opencl_backend(settings)
    if backend is None:
        message = _FAILURES.get(_backend_key(settings), "No compatible GPU runtime/device")
        return f"GPU unavailable ({message}); CPU fallback active"
    mode = "forced" if _force(settings) else "workload-dependent"
    return f"{backend.backend_name} active ({mode}): {backend.info.label}"


def _large_enough(settings: Any, work_items: int) -> bool:
    if _force(settings):
        return True
    return int(work_items) >= max(1, int(_setting(settings, "opencl_min_work_items", 100000) or 100000))


def _execute_gpu(settings: Any, operation: str, callback):
    failures = []
    last_error = None
    for backend in _gpu_backend_candidates(settings):
        try:
            result = callback(backend)
        except Exception as exc:
            last_error = exc
            failures.append(f"{backend.backend_name} GPU '{backend.info.name}': {type(exc).__name__}: {exc}")
            continue
        if result is not None:
            return result
    if _force(settings) and failures:
        detail = " | ".join(failures)
        raise GPUExecutionError(
            f"All available GPU backends failed during {operation}; "
            f"CPU fallback was not used because force-GPU mode is enabled: {detail}"
        ) from last_error
    return None


def gpu_influence_mask(xs: np.ndarray, ys: np.ndarray, links: np.ndarray,
                       valid_mask: Optional[np.ndarray], settings: Any) -> Optional[np.ndarray]:
    if not _force(settings) and not bool(_setting(settings, "opencl_accelerate_influence", True)):
        return None
    links_array = np.asarray(links)
    if links_array.size == 0 or not _large_enough(settings, len(xs) * len(ys) * max(1, len(links_array))):
        return None
    return _execute_gpu(settings, "whole-grid AP influence pruning", lambda backend: backend.influence_mask(xs, ys, links_array, valid_mask))


def gpu_strongest_indices(stack: np.ndarray, settings: Any) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    if not _force(settings) and not bool(_setting(settings, "opencl_accelerate_field_combine", True)):
        return None
    array = np.asarray(stack)
    if array.ndim < 2 or not _large_enough(settings, array.size):
        return None
    result = _execute_gpu(settings, "strongest-AP field aggregation", lambda backend: (backend.strongest_indices(array), backend.info.label, backend.backend_name))
    if result is None:
        return None
    (indices, valid), label, backend_name = result
    return indices, valid, f"{backend_name}: {label}"


def gpu_resample_regular_grid(source_xs: np.ndarray, source_ys: np.ndarray, source_values: np.ndarray,
                              target_xs: np.ndarray, target_ys: np.ndarray, settings: Any) -> Optional[np.ndarray]:
    if not _force(settings) and not bool(_setting(settings, "opencl_accelerate_resampling", True)):
        return None
    if len(source_xs) < 2 or len(source_ys) < 2 or not _large_enough(settings, len(target_xs) * len(target_ys)):
        return None
    return _execute_gpu(settings, "adaptive-grid resampling", lambda backend: backend.resample(source_xs, source_ys, source_values, target_xs, target_ys))


def gpu_colourise(values: np.ndarray, zones: Iterable[Any], settings: Any,
                  initialize: bool = False) -> Optional[Tuple[np.ndarray, str]]:
    if not _force(settings) and not bool(_setting(settings, "opencl_accelerate_raster", True)):
        return None
    array = np.asarray(values)
    if not _large_enough(settings, array.size):
        return None
    backend = get_opencl_backend(settings) if initialize else get_existing_opencl_backend(settings)
    if backend is None:
        return None
    prepared = []
    for zone in zones:
        colour = str(getattr(zone, "colour", "#555555") or "#555555").lstrip("#")
        if len(colour) != 6:
            colour = "555555"
        try:
            red, green, blue = int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)
        except Exception:
            red, green, blue = 85, 85, 85
        prepared.append((float(zone.min_dbm), float(zone.max_dbm), red, green, blue, max(0, min(255, int(getattr(zone, "alpha", 135))))))
    if not prepared:
        return None
    try:
        rgba = backend.colourise(array, prepared, blocking=initialize)
        return (rgba, f"{backend.backend_name}: {backend.info.label}") if rgba is not None else None
    except Exception:
        # Raster colourisation is presentation-only and must not freeze/crash Qt.
        return None
