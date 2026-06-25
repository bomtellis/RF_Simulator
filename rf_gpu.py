"""Optional cross-vendor OpenCL acceleration for the RF simulator.

The backend deliberately accelerates dense, regular-array stages while the
irregular IFC/Shapely geometry and ray discovery remain on the CPU.  PyOpenCL
supports Intel, AMD and NVIDIA OpenCL runtimes.  Every public helper returns
``None`` on an unavailable/unsupported device so the existing NumPy/CPU path
remains authoritative.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency. Install with: pip install pyopencl
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover - normal on systems without OpenCL Python bindings
    cl = None


@dataclass(frozen=True)
class OpenCLDeviceInfo:
    platform: str
    vendor: str
    name: str
    device_type: str
    global_memory_mb: int
    compute_units: int
    supports_fp64: bool

    @property
    def label(self) -> str:
        return f"{self.vendor} {self.name} ({self.device_type}, {self.global_memory_mb} MB)"


def _device_type_name(device: Any) -> str:
    if cl is None:
        return "unknown"
    value = int(getattr(device, "type", 0))
    if value & int(cl.device_type.GPU):
        return "GPU"
    if value & int(cl.device_type.ACCELERATOR):
        return "accelerator"
    if value & int(cl.device_type.CPU):
        return "CPU"
    return "other"


def discover_opencl_devices(include_cpu: bool = False) -> List[OpenCLDeviceInfo]:
    """Return available OpenCL devices without raising when no runtime exists."""
    if cl is None:
        return []
    output: List[OpenCLDeviceInfo] = []
    try:
        platforms = cl.get_platforms()
    except Exception:
        return []
    for platform in platforms:
        try:
            devices = platform.get_devices()
        except Exception:
            continue
        for device in devices:
            dtype = _device_type_name(device)
            if dtype == "CPU" and not include_cpu:
                continue
            extensions = str(getattr(device, "extensions", "") or "").lower()
            output.append(OpenCLDeviceInfo(
                platform=str(getattr(platform, "name", "OpenCL") or "OpenCL").strip(),
                vendor=str(getattr(device, "vendor", "Unknown") or "Unknown").strip(),
                name=str(getattr(device, "name", "OpenCL device") or "OpenCL device").strip(),
                device_type=dtype,
                global_memory_mb=max(0, int(getattr(device, "global_mem_size", 0) or 0) // (1024 * 1024)),
                compute_units=max(0, int(getattr(device, "max_compute_units", 0) or 0)),
                supports_fp64=("cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions),
            ))
    return output


_KERNEL_SOURCE = r"""
__kernel void influence_mask(
    __global const float *xs,
    __global const float *ys,
    __global const float *links,
    const int link_count,
    __global const uchar *valid,
    __global uchar *output,
    const int cols,
    const int rows)
{
    const int gid = get_global_id(0);
    const int total = cols * rows;
    if (gid >= total) return;
    if (valid[gid] == 0) { output[gid] = 0; return; }
    const int ix = gid % cols;
    const int iy = gid / cols;
    const float x = xs[ix];
    const float y = ys[iy];
    uchar keep = 0;
    for (int link = 0; link < link_count; ++link) {
        const int base = link * 8;
        const float dx = x - links[base + 0];
        const float dy = y - links[base + 1];
        const float dz = links[base + 2];
        float distance2 = dx * dx + dy * dy + dz * dz;
        const float radius2 = links[base + 7];
        if (radius2 > 0.0f && distance2 > radius2) continue;
        distance2 = fmax(distance2, 1.0f);
        const float distance = sqrt(distance2);
        const float upper = links[base + 5] - links[base + 3]
            - 10.0f * links[base + 4] * log10(distance);
        // A small tolerance prevents float32 rounding from incorrectly pruning
        // a point that the authoritative CPU path would retain.
        if (upper >= links[base + 6] - 0.25f) { keep = 1; break; }
    }
    output[gid] = keep;
}

__kernel void strongest_index(
    __global const float *stack,
    const int fields,
    const int points,
    __global int *indices,
    __global uchar *valid)
{
    const int point = get_global_id(0);
    if (point >= points) return;
    float best = -INFINITY;
    int best_index = 0;
    uchar any = 0;
    for (int field = 0; field < fields; ++field) {
        const float value = stack[field * points + point];
        if (isfinite(value)) {
            any = 1;
            if (value > best) { best = value; best_index = field; }
        }
    }
    indices[point] = best_index;
    valid[point] = any;
}

__kernel void resample_bilinear(
    __global const float *source,
    const int source_cols,
    const int source_rows,
    const float source_x0,
    const float source_y0,
    const float source_dx,
    const float source_dy,
    __global const float *target_xs,
    __global const float *target_ys,
    const int target_cols,
    const int target_rows,
    __global float *output)
{
    const int gid = get_global_id(0);
    const int total = target_cols * target_rows;
    if (gid >= total) return;
    const int tx = gid % target_cols;
    const int ty = gid / target_cols;
    float fx = (target_xs[tx] - source_x0) / source_dx;
    float fy = (target_ys[ty] - source_y0) / source_dy;
    fx = clamp(fx, 0.0f, (float)(source_cols - 1));
    fy = clamp(fy, 0.0f, (float)(source_rows - 1));
    const int x0 = (int)floor(fx);
    const int y0 = (int)floor(fy);
    const int x1 = min(x0 + 1, source_cols - 1);
    const int y1 = min(y0 + 1, source_rows - 1);
    const float wx = fx - (float)x0;
    const float wy = fy - (float)y0;
    const int ids[4] = {y0 * source_cols + x0, y0 * source_cols + x1,
                        y1 * source_cols + x0, y1 * source_cols + x1};
    const float ws[4] = {(1.0f-wx)*(1.0f-wy), wx*(1.0f-wy),
                         (1.0f-wx)*wy, wx*wy};
    float value_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const float value = source[ids[k]];
        if (isfinite(value)) {
            value_sum += value * ws[k];
            weight_sum += ws[k];
        }
    }
    output[gid] = weight_sum > 1.0e-8f ? value_sum / weight_sum : NAN;
}

__kernel void colourise(
    __global const float *values,
    const int points,
    __global const float *mins,
    __global const float *maxs,
    __global const uchar4 *colours,
    const int zone_count,
    const int high_index,
    const int low_index,
    __global uchar4 *output)
{
    const int gid = get_global_id(0);
    if (gid >= points) return;
    const float value = values[gid];
    if (!isfinite(value)) { output[gid] = (uchar4)(0,0,0,0); return; }
    int selected = -1;
    for (int zone = 0; zone < zone_count; ++zone) {
        if (value >= mins[zone] && value < maxs[zone]) { selected = zone; break; }
    }
    if (selected < 0) selected = value >= maxs[high_index] ? high_index : low_index;
    output[gid] = colours[selected];
}
"""


class OpenCLRFAccelerator:
    """One lazily-created OpenCL context/queue used by the coordinator thread."""

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
        platform = getattr(self.device, "platform", None)
        self.info = OpenCLDeviceInfo(
            platform=str(getattr(platform, "name", "OpenCL") or "OpenCL").strip(),
            vendor=str(getattr(self.device, "vendor", "Unknown") or "Unknown").strip(),
            name=str(getattr(self.device, "name", "OpenCL device") or "OpenCL device").strip(),
            device_type=_device_type_name(self.device),
            global_memory_mb=max(0, int(getattr(self.device, "global_mem_size", 0) or 0) // (1024 * 1024)),
            compute_units=max(0, int(getattr(self.device, "max_compute_units", 0) or 0)),
            supports_fp64=("cl_khr_fp64" in str(getattr(self.device, "extensions", "") or "").lower()),
        )

    def _choose_device(self):
        candidates = []
        stable_index = 0
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                dtype = _device_type_name(device)
                if dtype == "CPU" and not self.allow_cpu:
                    continue
                priority = 0 if dtype == "GPU" else (1 if dtype == "accelerator" else 2)
                text = " ".join([
                    str(getattr(platform, "name", "")), str(getattr(device, "vendor", "")),
                    str(getattr(device, "name", "")), dtype,
                ]).lower()
                preferred = self.preference == "auto" or self.preference in text
                candidates.append((
                    0 if preferred else 1,
                    priority,
                    -int(getattr(device, "global_mem_size", 0) or 0),
                    stable_index,
                    device,
                ))
                stable_index += 1
        if not candidates:
            raise RuntimeError("No suitable OpenCL GPU/accelerator was found")
        candidates.sort(key=lambda item: item[:4])
        return candidates[0][4]

    def influence_mask(self, xs: np.ndarray, ys: np.ndarray, links: np.ndarray,
                       valid_mask: Optional[np.ndarray]) -> np.ndarray:
        xs32 = np.ascontiguousarray(xs, dtype=np.float32)
        ys32 = np.ascontiguousarray(ys, dtype=np.float32)
        links32 = np.ascontiguousarray(links, dtype=np.float32).reshape(-1, 8)
        valid = np.ones((len(ys32), len(xs32)), dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask, dtype=np.uint8)
        output = np.empty(valid.size, dtype=np.uint8)
        mf = cl.mem_flags
        with self._lock:
            xs_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xs32)
            ys_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ys32)
            links_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=links32)
            valid_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=valid.ravel())
            out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
            self.program.influence_mask(
                self.queue, (output.size,), None, xs_buf, ys_buf, links_buf,
                np.int32(len(links32)), valid_buf, out_buf, np.int32(len(xs32)), np.int32(len(ys32)),
            )
            cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output.reshape((len(ys32), len(xs32))).astype(bool)

    def strongest_indices(self, stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        source = np.ascontiguousarray(stack, dtype=np.float32)
        fields = int(source.shape[0])
        points = int(np.prod(source.shape[1:]))
        indices = np.empty(points, dtype=np.int32)
        valid = np.empty(points, dtype=np.uint8)
        mf = cl.mem_flags
        with self._lock:
            src_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source.ravel())
            idx_buf = cl.Buffer(self.context, mf.WRITE_ONLY, indices.nbytes)
            valid_buf = cl.Buffer(self.context, mf.WRITE_ONLY, valid.nbytes)
            self.program.strongest_index(
                self.queue, (points,), None, src_buf, np.int32(fields), np.int32(points), idx_buf, valid_buf,
            )
            cl.enqueue_copy(self.queue, indices, idx_buf)
            cl.enqueue_copy(self.queue, valid, valid_buf).wait()
        shape = source.shape[1:]
        return indices.reshape(shape), valid.reshape(shape).astype(bool)

    def resample(self, source_xs: np.ndarray, source_ys: np.ndarray, source_values: np.ndarray,
                 target_xs: np.ndarray, target_ys: np.ndarray) -> np.ndarray:
        source = np.ascontiguousarray(source_values, dtype=np.float32)
        tx = np.ascontiguousarray(target_xs, dtype=np.float32)
        ty = np.ascontiguousarray(target_ys, dtype=np.float32)
        output = np.empty((len(ty), len(tx)), dtype=np.float32)
        source_dx = float(source_xs[1] - source_xs[0]) if len(source_xs) > 1 else 1.0
        source_dy = float(source_ys[1] - source_ys[0]) if len(source_ys) > 1 else 1.0
        mf = cl.mem_flags
        with self._lock:
            src_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source.ravel())
            tx_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tx)
            ty_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ty)
            out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
            self.program.resample_bilinear(
                self.queue, (output.size,), None, src_buf, np.int32(source.shape[1]), np.int32(source.shape[0]),
                np.float32(source_xs[0]), np.float32(source_ys[0]), np.float32(source_dx), np.float32(source_dy),
                tx_buf, ty_buf, np.int32(len(tx)), np.int32(len(ty)), out_buf,
            )
            cl.enqueue_copy(self.queue, output, out_buf).wait()
        return output.astype(np.float64)

    def colourise(self, values: np.ndarray, zones: Sequence[Tuple[float, float, int, int, int, int]],
                  blocking: bool = True) -> Optional[np.ndarray]:
        source = np.ascontiguousarray(values, dtype=np.float32)
        mins = np.ascontiguousarray([zone[0] for zone in zones], dtype=np.float32)
        maxs = np.ascontiguousarray([zone[1] for zone in zones], dtype=np.float32)
        colours = np.ascontiguousarray([[zone[2], zone[3], zone[4], zone[5]] for zone in zones], dtype=np.uint8)
        output = np.empty((source.size, 4), dtype=np.uint8)
        high_index = int(np.argmax(maxs))
        low_index = int(np.argmin(mins))
        mf = cl.mem_flags
        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            return None
        try:
            src_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source.ravel())
            min_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mins)
            max_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maxs)
            colour_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=colours)
            out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
            self.program.colourise(
                self.queue, (source.size,), None, src_buf, np.int32(source.size), min_buf, max_buf,
                colour_buf, np.int32(len(zones)), np.int32(high_index), np.int32(low_index), out_buf,
            )
            cl.enqueue_copy(self.queue, output, out_buf).wait()
        finally:
            self._lock.release()
        return output.reshape((*source.shape, 4))


_BACKEND_LOCK = threading.RLock()
_BACKENDS = {}
_FAILURES = {}


def _setting(settings: Any, name: str, default: Any) -> Any:
    return getattr(settings, name, default) if settings is not None else default


def _backend_key(settings: Any) -> Tuple[str, bool]:
    preference = str(_setting(settings, "opencl_device_preference", "auto") or "auto")
    allow_cpu = bool(_setting(settings, "opencl_allow_cpu_device", False))
    return preference.strip().lower(), allow_cpu


def get_existing_opencl_backend(settings: Any) -> Optional[OpenCLRFAccelerator]:
    """Return an already-initialised backend without invoking a GPU driver."""
    if not bool(_setting(settings, "enable_opencl_gpu", True)) or cl is None:
        return None
    with _BACKEND_LOCK:
        return _BACKENDS.get(_backend_key(settings))


def get_opencl_backend(settings: Any) -> Optional[OpenCLRFAccelerator]:
    if not bool(_setting(settings, "enable_opencl_gpu", True)) or cl is None:
        return None
    preference, allow_cpu = _backend_key(settings)
    key = (preference, allow_cpu)
    with _BACKEND_LOCK:
        if key in _BACKENDS:
            return _BACKENDS[key]
        if key in _FAILURES:
            return None
        try:
            backend = OpenCLRFAccelerator(preference, allow_cpu)
        except Exception as exc:
            _FAILURES[key] = f"{type(exc).__name__}: {exc}"
            return None
        _BACKENDS[key] = backend
        return backend


def reset_opencl_backends() -> None:
    with _BACKEND_LOCK:
        _BACKENDS.clear()
        _FAILURES.clear()


def _preferred_device_info(settings: Any) -> Optional[OpenCLDeviceInfo]:
    preference = str(_setting(settings, "opencl_device_preference", "auto") or "auto").strip().lower()
    allow_cpu = bool(_setting(settings, "opencl_allow_cpu_device", False))
    devices = discover_opencl_devices(include_cpu=allow_cpu)
    if not devices:
        return None
    ranked = []
    for index, info in enumerate(devices):
        text = f"{info.platform} {info.vendor} {info.name} {info.device_type}".lower()
        preferred = preference == "auto" or preference in text
        priority = 0 if info.device_type == "GPU" else (1 if info.device_type == "accelerator" else 2)
        ranked.append((0 if preferred else 1, priority, -info.global_memory_mb, index, info))
    ranked.sort(key=lambda item: item[:4])
    return ranked[0][4]


def opencl_status(settings: Any, initialize: bool = False) -> str:
    """Describe OpenCL availability without compiling kernels unless requested.

    The performance dialog calls the discovery-only form so opening it cannot
    block the Qt thread while a vendor driver compiles the OpenCL program.  The
    calculation path requests ``initialize=True`` after work has already moved
    to the background coordinator.
    """
    if not bool(_setting(settings, "enable_opencl_gpu", True)):
        return "OpenCL acceleration disabled; CPU path active"
    if cl is None:
        return "PyOpenCL is not installed; CPU fallback active"
    if not initialize:
        info = _preferred_device_info(settings)
        return f"OpenCL device detected: {info.label}" if info is not None else "No matching OpenCL device; CPU fallback active"
    backend = get_opencl_backend(settings)
    if backend is None:
        preference = str(_setting(settings, "opencl_device_preference", "auto") or "auto").strip().lower()
        message = _FAILURES.get((preference, bool(_setting(settings, "opencl_allow_cpu_device", False))))
        return f"OpenCL unavailable ({message}); CPU fallback active" if message else "No matching OpenCL device; CPU fallback active"
    return f"OpenCL active: {backend.info.label}"


def _large_enough(settings: Any, work_items: int) -> bool:
    return int(work_items) >= max(1, int(_setting(settings, "opencl_min_work_items", 100000) or 0 ))


def gpu_influence_mask(xs: np.ndarray, ys: np.ndarray, links: np.ndarray,
                       valid_mask: Optional[np.ndarray], settings: Any) -> Optional[np.ndarray]:
    if not bool(_setting(settings, "opencl_accelerate_influence", True)):
        return None
    links_array = np.asarray(links)
    if links_array.size == 0 or not _large_enough(settings, len(xs) * len(ys) * max(1, len(links_array))):
        return None
    backend = get_opencl_backend(settings)
    if backend is None:
        return None
    try:
        return backend.influence_mask(xs, ys, links_array, valid_mask)
    except Exception:
        return None


def gpu_strongest_indices(stack: np.ndarray, settings: Any) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    if not bool(_setting(settings, "opencl_accelerate_field_combine", True)):
        return None
    array = np.asarray(stack)
    if array.ndim < 2 or not _large_enough(settings, array.size):
        return None
    backend = get_opencl_backend(settings)
    if backend is None:
        return None
    try:
        indices, valid = backend.strongest_indices(array)
        return indices, valid, backend.info.label
    except Exception:
        return None


def gpu_resample_regular_grid(source_xs: np.ndarray, source_ys: np.ndarray, source_values: np.ndarray,
                              target_xs: np.ndarray, target_ys: np.ndarray, settings: Any) -> Optional[np.ndarray]:
    if not bool(_setting(settings, "opencl_accelerate_resampling", True)):
        return None
    if len(source_xs) < 2 or len(source_ys) < 2 or not _large_enough(settings, len(target_xs) * len(target_ys)):
        return None
    backend = get_opencl_backend(settings)
    if backend is None:
        return None
    try:
        return backend.resample(source_xs, source_ys, source_values, target_xs, target_ys)
    except Exception:
        return None


def gpu_colourise(values: np.ndarray, zones: Iterable[Any], settings: Any,
                  initialize: bool = False) -> Optional[Tuple[np.ndarray, str]]:
    if not bool(_setting(settings, "opencl_accelerate_raster", True)):
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
        prepared.append((float(zone.min_dbm), float(zone.max_dbm), red, green, blue,
                         max(0, min(255, int(getattr(zone, "alpha", 135))))))
    if not prepared:
        return None
    try:
        rgba = backend.colourise(array, prepared, blocking=initialize)
        return (rgba, backend.info.label) if rgba is not None else None
    except Exception:
        return None
