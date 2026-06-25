"""Optional GPU acceleration for the RF simulator.

NVIDIA devices use Numba-CUDA JIT kernels.  Intel and AMD devices may use the
existing OpenCL fallback when PyOpenCL and a vendor runtime are installed.
The public helper names are retained for compatibility with older RF plans and
application code, but CUDA is always preferred when a usable NVIDIA GPU exists.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
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

    @cuda.jit(cache=True, fastmath=True)
    def _cuda_direct_rssi_paths(xs, ys, valid, ap_data, segments, segment_indices, segment_offsets,
                                disconnected, start_gid, work_count, path_rssi, cols, rows):
        ap_count = ap_data.shape[0]
        path_total = work_count * ap_count
        local_start = cuda.grid(1)
        stride = cuda.gridsize(1)
        total = cols * rows
        for path_index in range(local_start, path_total, stride):
            local_gid = path_index // ap_count
            ai = path_index - local_gid * ap_count
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
            rz = ap_data[ai, 3]
            base_dbm = ap_data[ai, 4]
            ple = ap_data[ai, 5]
            cutoff2 = ap_data[ai, 6]
            dz2 = ap_data[ai, 7]
            dx = x - ax
            dy = y - ay
            d2xy = dx * dx + dy * dy
            if cutoff2 > 0.0 and d2xy > cutoff2:
                path_rssi[path_index] = math.nan
                continue
            d3 = math.sqrt(d2xy + dz2)
            if d3 < 1.0:
                d3 = 1.0
            wall_loss = 0.0
            first_segment = segment_offsets[ai]
            last_segment = segment_offsets[ai + 1]
            for packed_index in range(first_segment, last_segment):
                si = segment_indices[packed_index]
                x1 = segments[si, 0]
                y1 = segments[si, 1]
                sdx = segments[si, 2]
                sdy = segments[si, 3]
                den = (x - ax) * sdy - (y - ay) * sdx
                if den > -1.0e-7 and den < 1.0e-7:
                    continue
                t = ((x1 - ax) * sdy - (y1 - ay) * sdx) / den
                u = ((x1 - ax) * (y - ay) - (y1 - ay) * (x - ax)) / den
                if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                    zhit = az + (rz - az) * t
                    if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                        wall_loss += segments[si, 6]
            rssi = base_dbm - 10.0 * ple * math.log10(d3) - wall_loss
            if rssi < disconnected:
                rssi = disconnected
            path_rssi[path_index] = rssi

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
            base = local_gid * ap_count
            for ai in range(ap_count):
                rssi = path_rssi[base + ai]
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
            out_counts[gid] = path_count

else:  # pragma: no cover
    _cuda_influence_mask = None
    _cuda_strongest_index = None
    _cuda_resample_bilinear = None
    _cuda_colourise = None
    _cuda_direct_rssi_paths = None
    _cuda_reduce_rssi_paths = None


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
        self.threads_per_block = 128
        self._minimum_blocks = max(128, int(self.info.compute_units or 1) * 4)
        self._direct_grid_key = None
        self._direct_grid_refs = None
        self._direct_grid_buffers = None

    def _launch_shape(self, count: int) -> Tuple[int, int]:
        count = max(1, int(count))
        threads = self.threads_per_block
        natural_blocks = max(1, (count + threads - 1) // threads)
        # Saturate the SMs for substantial work. Grid-stride kernels make this
        # safe even when the occupancy-sized grid contains more threads than
        # the current point count.
        if count >= threads * 8:
            natural_blocks = max(natural_blocks, self._minimum_blocks)
        return natural_blocks, threads

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
                         disconnected: float, combine_mode: int, settings: Any = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        compact_aps, prepared_segments, segment_indices, segment_offsets = _prepare_direct_inputs(ap_data, segments, np.float32)
        rows, cols = len(ys), len(xs)
        total = rows * cols
        if total <= 0:
            return np.empty((rows, cols), dtype=np.float32), np.zeros((rows, cols), dtype=np.int16)
        chunk_points = max(65_536, int(_setting(settings, "cuda_chunk_points", 262_144) or 262_144))
        with self._lock, cuda.gpus[self.device_index]:
            stream = cuda.stream()
            d_xs, d_ys, d_valid = self._grid_buffers(xs, ys, valid_mask, stream)
            d_aps = cuda.to_device(compact_aps, stream=stream)
            d_segments = cuda.to_device(prepared_segments, stream=stream)
            d_segment_indices = cuda.to_device(segment_indices, stream=stream)
            d_offsets = cuda.to_device(segment_offsets, stream=stream)
            d_rssi = cuda.device_array(total, dtype=np.float32, stream=stream)
            d_counts = cuda.device_array(total, dtype=np.int16, stream=stream)
            ap_count = int(compact_aps.shape[0])
            # Bound the temporary point-by-AP path field. The expensive RF path
            # kernel now runs one independent GPU work item per point/AP pair,
            # then a light reduction kernel combines the AP fields per point.
            temp_budget_bytes = max(32 * 1024 * 1024, min(256 * 1024 * 1024, int(self.info.global_memory_mb or 2048) * 1024 * 1024 // 8))
            max_points_by_temp = max(4096, temp_budget_bytes // max(4, ap_count * 4))
            chunk_points = min(chunk_points, max_points_by_temp)
            d_path_rssi = cuda.device_array(chunk_points * ap_count, dtype=np.float32, stream=stream)
            stream.synchronize()
            for start_gid in range(0, total, chunk_points):
                if self._cancelled(settings):
                    raise RuntimeError("RSSI calculation cancelled")
                work_count = min(chunk_points, total - start_gid)
                path_items = work_count * ap_count
                path_blocks, threads = self._launch_shape(path_items)
                _cuda_direct_rssi_paths[path_blocks, threads, stream](
                    d_xs, d_ys, d_valid, d_aps, d_segments, d_segment_indices, d_offsets,
                    np.float32(disconnected), np.int64(start_gid), np.int64(work_count),
                    d_path_rssi, np.int32(cols), np.int32(rows),
                )
                reduce_blocks, reduce_threads = self._launch_shape(work_count)
                _cuda_reduce_rssi_paths[reduce_blocks, reduce_threads, stream](
                    d_valid, d_path_rssi, np.int32(ap_count), np.float32(disconnected),
                    np.int32(combine_mode), np.int64(start_gid), np.int64(work_count),
                    d_rssi, d_counts,
                )
                # Synchronise at chunk boundaries so Cancel remains responsive
                # without re-uploading the static RF model for each chunk.
                stream.synchronize()
            if self._cancelled(settings):
                raise RuntimeError("RSSI calculation cancelled")
            rssi = d_rssi.copy_to_host(stream=stream)
            counts = d_counts.copy_to_host(stream=stream)
            stream.synchronize()
        return rssi.reshape((rows, cols)), counts.reshape((rows, cols))


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
__kernel void direct_rssi_paths(__global const float *xs,__global const float *ys,
 __global const uchar *valid,__global const float *aps,__global const float *segments,
 __global const int *segment_indices,__global const int *offsets,const int ap_count,
 const float disconnected,const int start_gid,const int work_count,__global float *path_rssi,
 const int cols,const int rows){
 const int total=cols*rows;const int path_total=work_count*ap_count;
 for(int path_index=get_global_id(0);path_index<path_total;path_index+=get_global_size(0)){
  const int local_gid=path_index/ap_count;const int ai=path_index-local_gid*ap_count;
  const int gid=start_gid+local_gid;if(gid>=total||valid[gid]==0){path_rssi[path_index]=NAN;continue;}
  const int ix=gid%cols,iy=gid/cols;const float x=xs[ix],y=ys[iy];const int ab=ai*8;
  const float ax=aps[ab],ay=aps[ab+1],az=aps[ab+2],rz=aps[ab+3];
  const float base_dbm=aps[ab+4],ple=aps[ab+5],cutoff2=aps[ab+6],dz2=aps[ab+7];
  const float dx=x-ax,dy=y-ay;const float d2xy=dx*dx+dy*dy;
  if(cutoff2>0.0f&&d2xy>cutoff2){path_rssi[path_index]=NAN;continue;}
  float d3=sqrt(d2xy+dz2);d3=fmax(d3,1.0f);float wall_loss=0.0f;
  for(int pi=offsets[ai];pi<offsets[ai+1];++pi){const int si=segment_indices[pi];const int sb=si*7;
   const float x1=segments[sb],y1=segments[sb+1],sdx=segments[sb+2],sdy=segments[sb+3];
   const float den=(x-ax)*sdy-(y-ay)*sdx;if(fabs(den)<1.0e-7f)continue;
   const float t=((x1-ax)*sdy-(y1-ay)*sdx)/den;
   const float u=((x1-ax)*(y-ay)-(y1-ay)*(x-ax))/den;
   if(t>=0.0f&&t<=1.0f&&u>=0.0f&&u<=1.0f){const float zhit=az+(rz-az)*t;
    if(zhit>=segments[sb+4]&&zhit<=segments[sb+5])wall_loss+=segments[sb+6];}}
  float rssi=base_dbm-10.0f*ple*log10(d3)-wall_loss;path_rssi[path_index]=fmax(rssi,disconnected);
 }
}
__kernel void reduce_rssi_paths(__global const uchar *valid,__global const float *path_rssi,
 const int ap_count,const float disconnected,const int combine_mode,const int start_gid,
 const int work_count,__global float *out_rssi,__global short *out_counts){
 for(int local_gid=get_global_id(0);local_gid<work_count;local_gid+=get_global_size(0)){
  const int gid=start_gid+local_gid;if(valid[gid]==0){out_rssi[gid]=NAN;out_counts[gid]=0;continue;}
  float best=disconnected,power_sum=0.0f;int path_count=0;const int base=local_gid*ap_count;
  for(int ai=0;ai<ap_count;++ai){const float rssi=path_rssi[base+ai];if(!isfinite(rssi))continue;
   ++path_count;if(combine_mode==1)power_sum+=pow(10.0f,rssi/10.0f);else if(rssi>best)best=rssi;}
  if(combine_mode==1&&power_sum>0.0f)best=10.0f*log10(power_sum);if(path_count==0)best=disconnected;
  out_rssi[gid]=best;out_counts[gid]=(short)path_count;
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
                text = " ".join([str(getattr(platform, "name", "")), str(getattr(device, "vendor", "")), str(getattr(device, "name", "")), dtype]).lower()
                preferred = self.preference in {"auto", "opencl"} or self.preference in text
                priority = 0 if dtype == "OpenCL GPU" else (1 if "accelerator" in dtype else 2)
                candidates.append((0 if preferred else 1, priority, -int(getattr(device, "global_mem_size", 0) or 0), stable_index, device))
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

    def direct_rssi_grid(self, xs, ys, ap_data, segments, valid_mask, disconnected, combine_mode, settings=None):
        compact_aps,prepared_segments,segment_indices,offsets=_prepare_direct_inputs(ap_data,segments,np.float32)
        xs32=np.ascontiguousarray(xs,dtype=np.float32);ys32=np.ascontiguousarray(ys,dtype=np.float32)
        rows,cols=len(ys32),len(xs32);total=rows*cols
        valid=np.ones((rows,cols),dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask,dtype=np.uint8)
        rssi=np.empty(total,dtype=np.float32);counts=np.empty(total,dtype=np.int16);mf=cl.mem_flags
        chunk_points=max(65_536,int(_setting(settings,"cuda_chunk_points",262_144) or 262_144))
        ap_count=int(compact_aps.shape[0])
        temp_budget_bytes=max(32*1024*1024,min(256*1024*1024,int(self.info.global_memory_mb or 2048)*1024*1024//8))
        max_points_by_temp=max(4096,temp_budget_bytes//max(4,ap_count*4));chunk_points=min(chunk_points,max_points_by_temp)
        with self._lock:
            xb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=xs32);yb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=ys32)
            vb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=valid.ravel());ab=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=compact_aps.ravel())
            sb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=prepared_segments.ravel());sib=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=segment_indices);ob=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=offsets)
            rb=cl.Buffer(self.context,mf.WRITE_ONLY,rssi.nbytes);cb=cl.Buffer(self.context,mf.WRITE_ONLY,counts.nbytes)
            pathb=cl.Buffer(self.context,mf.READ_WRITE,chunk_points*ap_count*np.dtype(np.float32).itemsize)
            for start_gid in range(0,total,chunk_points):
                if self._cancelled(settings):raise RuntimeError("RSSI calculation cancelled")
                work=min(chunk_points,total-start_gid);path_items=work*ap_count
                self._direct_path_kernel.set_args(xb,yb,vb,ab,sb,sib,ob,np.int32(ap_count),np.float32(disconnected),np.int32(start_gid),np.int32(work),pathb,np.int32(cols),np.int32(rows))
                global_size,local_size=self._launch_shape(path_items);cl.enqueue_nd_range_kernel(self.queue,self._direct_path_kernel,global_size,local_size)
                self._direct_reduce_kernel.set_args(vb,pathb,np.int32(ap_count),np.float32(disconnected),np.int32(combine_mode),np.int32(start_gid),np.int32(work),rb,cb)
                global_size,local_size=self._launch_shape(work);cl.enqueue_nd_range_kernel(self.queue,self._direct_reduce_kernel,global_size,local_size).wait()
            if self._cancelled(settings):raise RuntimeError("RSSI calculation cancelled")
            cl.enqueue_copy(self.queue,rssi,rb);cl.enqueue_copy(self.queue,counts,cb).wait()
        return rssi.reshape((rows,cols)),counts.reshape((rows,cols))


# ------------------------- Direct RSSI grid helpers -------------------------
def _prepare_direct_inputs(ap_data: np.ndarray, segments: np.ndarray, dtype=np.float32):
    """Compact AP constants and conservatively prefilter RF barrier segments.

    The compact AP layout stores the constant 1 m RSSI base and vertical
    distance square once, avoiding repeated frequency/floor calculations for
    every point. Barrier rows are stored only once; each AP gets a packed list
    of integer segment indices after safe Z-range and cutoff-bounds rejection.
    This avoids duplicating the complete IFC barrier array for every AP.
    """
    raw_aps = np.asarray(ap_data)
    aps = np.asarray(ap_data, dtype=np.float64).reshape(-1, raw_aps.shape[-1] if raw_aps.ndim > 1 else 10)
    if aps.shape[1] < 10:
        raise ValueError("Direct RSSI AP data must contain at least 10 columns")
    raw_segments = np.asarray(segments, dtype=np.float64).reshape(-1, 7)
    prepared_segments = np.asarray(raw_segments, dtype=dtype).copy()
    if prepared_segments.shape[0]:
        prepared_segments[:, 2] = raw_segments[:, 2] - raw_segments[:, 0]
        prepared_segments[:, 3] = raw_segments[:, 3] - raw_segments[:, 1]
    else:
        # CUDA accepts zero-sized arrays, but some OpenCL runtimes reject a
        # zero-byte buffer. Offsets remain zero so this dummy row is never read.
        prepared_segments = np.zeros((1, 7), dtype=dtype)
    compact = np.empty((aps.shape[0], 8), dtype=dtype)
    packed_indices = []
    offsets = np.zeros(aps.shape[0] + 1, dtype=np.int32)
    for ai in range(aps.shape[0]):
        ax, ay, az, rz, tx, gain, freq_mhz, ple, cutoff2, floor_loss = aps[ai, :10]
        fspl_1m = 20.0 * math.log10(max(freq_mhz, 1.0e-12)) - 27.55
        compact[ai] = (ax, ay, az, rz, tx + gain - fspl_1m - floor_loss, ple, cutoff2, (rz - az) * (rz - az))
        zlo = min(az, rz)
        zhi = max(az, rz)
        radius2 = float(cutoff2)
        for si, seg in enumerate(raw_segments):
            if seg[5] < zlo or seg[4] > zhi:
                continue
            if radius2 > 0.0:
                minx = min(seg[0], seg[2]); maxx = max(seg[0], seg[2])
                miny = min(seg[1], seg[3]); maxy = max(seg[1], seg[3])
                ddx = minx - ax if ax < minx else (ax - maxx if ax > maxx else 0.0)
                ddy = miny - ay if ay < miny else (ay - maxy if ay > maxy else 0.0)
                if ddx * ddx + ddy * ddy > radius2:
                    continue
            packed_indices.append(si)
        offsets[ai + 1] = len(packed_indices)
    segment_indices = np.asarray(packed_indices, dtype=np.int32) if packed_indices else np.zeros(1, dtype=np.int32)
    return (
        np.ascontiguousarray(compact),
        np.ascontiguousarray(prepared_segments),
        np.ascontiguousarray(segment_indices),
        offsets,
    )


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
                ax = ap_data[ai, 0]; ay = ap_data[ai, 1]; az = ap_data[ai, 2]; rz = ap_data[ai, 3]
                base_dbm = ap_data[ai, 4]; ple = ap_data[ai, 5]; cutoff2 = ap_data[ai, 6]; dz2 = ap_data[ai, 7]
                dx = x - ax; dy = y - ay
                d2xy = dx * dx + dy * dy
                if cutoff2 > 0.0 and d2xy > cutoff2:
                    continue
                d3 = math.sqrt(d2xy + dz2)
                if d3 < 1.0:
                    d3 = 1.0
                wall_loss = 0.0
                for packed_index in range(segment_offsets[ai], segment_offsets[ai + 1]):
                    si = segment_indices[packed_index]
                    x1 = segments[si, 0]; y1 = segments[si, 1]
                    sdx = segments[si, 2]; sdy = segments[si, 3]
                    den = (x - ax) * sdy - (y - ay) * sdx
                    if den > -1.0e-7 and den < 1.0e-7:
                        continue
                    t = ((x1 - ax) * sdy - (y1 - ay) * sdx) / den
                    u = ((x1 - ax) * (y - ay) - (y1 - ay) * (x - ax)) / den
                    if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                        zhit = az + (rz - az) * t
                        if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                            wall_loss += segments[si, 6]
                rssi = base_dbm - 10.0 * ple * math.log10(d3) - wall_loss
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
            counts[gid] = pc
else:
    _numba_direct_rssi_grid = None


def gpu_direct_rssi_grid(xs: np.ndarray, ys: np.ndarray, ap_data: np.ndarray, segments: np.ndarray,
                         valid_mask: Optional[np.ndarray], disconnected: float, combine_mode: int,
                         settings: Any) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    if not _enabled(settings):
        return None
    ap_array = np.asarray(ap_data)
    if ap_array.size == 0:
        return None
    work_items = len(xs) * len(ys) * max(1, int(ap_array.shape[0] if ap_array.ndim > 1 else 1))
    if not _large_enough(settings, work_items):
        return None
    result = _execute_gpu(
        settings,
        "direct RSSI grid calculation",
        lambda backend: (backend.direct_rssi_grid(xs, ys, ap_data, segments, valid_mask, disconnected, combine_mode, settings), backend.info.label, backend.backend_name)
        if hasattr(backend, "direct_rssi_grid") else None,
    )
    if result is None:
        return None
    (rssi, counts), label, backend_name = result
    return rssi, counts, f"{backend_name}: {label}"


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
    # it is, use large compiled chunks so Cancel is checked without reverting
    # to Python per-point work.
    chunk_points = total if cancel_event is None else max(65_536, int(_setting(settings, "numba_chunk_points", 262_144) or 262_144))
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


def _backend_key(settings: Any) -> Tuple[str, bool, bool]:
    return (_preference(settings), bool(_setting(settings, "opencl_allow_cpu_device", False)), _force(settings))


def _cuda_requested(preference: str) -> bool:
    return preference == "auto" or preference == "cuda" or preference.startswith("cuda:") or "nvidia" in preference


def _opencl_requested(preference: str) -> bool:
    return preference == "auto" or preference == "opencl" or preference.startswith("opencl:") or not _cuda_requested(preference)


def get_existing_opencl_backend(settings: Any):
    if not _enabled(settings):
        return None
    with _BACKEND_LOCK:
        return _BACKENDS.get(_backend_key(settings))


def get_opencl_backend(settings: Any):
    """Return CUDA first, then OpenCL GPU; CPU is used only if no GPU exists."""
    if not _enabled(settings):
        return None
    preference = _preference(settings)
    allow_cpu = bool(_setting(settings, "opencl_allow_cpu_device", False))
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
                if preference.startswith("cuda") or preference == "cuda":
                    _FAILURES[key] = "; ".join(errors)
                    return None
        if _opencl_requested(preference):
            try:
                backend = OpenCLRFAccelerator(preference, allow_cpu)
                # Never select an OpenCL CPU while any GPU device exists.
                if backend.info.device_type == "OpenCL CPU" and (len(_cuda_device_records()) or any("GPU" in d.device_type for d in _opencl_device_records(False))):
                    raise RuntimeError("GPU exists; OpenCL CPU fallback refused")
                _BACKENDS[key] = backend
                return backend
            except Exception as exc:
                errors.append(f"OpenCL: {type(exc).__name__}: {exc}")
        _FAILURES[key] = "; ".join(errors) or "No compatible GPU runtime/device"
        return None


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
    backend = get_opencl_backend(settings)
    if backend is None:
        return None
    try:
        return callback(backend)
    except Exception as exc:
        if _force(settings):
            raise GPUExecutionError(
                f"{backend.backend_name} GPU '{backend.info.name}' failed during {operation}; "
                f"CPU fallback was not used because force-GPU mode is enabled: {type(exc).__name__}: {exc}"
            ) from exc
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
