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
        gid = cuda.grid(1)
        total = cols * rows
        if gid >= total:
            return
        if valid[gid] == 0:
            output[gid] = 0
            return
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
        point = cuda.grid(1)
        if point >= points:
            return
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
        gid = cuda.grid(1)
        total = target_cols * target_rows
        if gid >= total:
            return
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
        # Otherwise retain the host-initialised NaN.

    @cuda.jit(cache=True)
    def _cuda_colourise(values, points, mins, maxs, colours, zone_count,
                        high_index, low_index, output):
        gid = cuda.grid(1)
        if gid >= points:
            return
        value = values[gid]
        if not math.isfinite(value):
            output[gid, 0] = 0
            output[gid, 1] = 0
            output[gid, 2] = 0
            output[gid, 3] = 0
            return
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
    def _cuda_direct_rssi_grid(xs, ys, valid, ap_data, segments, disconnected, combine_mode, out_rssi, out_counts, cols, rows):
        gid = cuda.grid(1)
        total = cols * rows
        if gid >= total:
            return
        if valid[gid] == 0:
            out_rssi[gid] = math.nan
            out_counts[gid] = 0
            return
        ix = gid % cols
        iy = gid // cols
        x = xs[ix]
        y = ys[iy]
        best = disconnected
        power_sum_mw = 0.0
        path_count = 0
        ap_count = ap_data.shape[0]
        seg_count = segments.shape[0]
        for ai in range(ap_count):
            ax = ap_data[ai, 0]
            ay = ap_data[ai, 1]
            az = ap_data[ai, 2]
            rz = ap_data[ai, 3]
            tx = ap_data[ai, 4]
            gain = ap_data[ai, 5]
            freq_mhz = ap_data[ai, 6]
            ple = ap_data[ai, 7]
            cutoff2 = ap_data[ai, 8]
            floor_loss = ap_data[ai, 9]
            dx = x - ax
            dy = y - ay
            dz = rz - az
            d2xy = dx * dx + dy * dy
            if cutoff2 > 0.0 and d2xy > cutoff2:
                continue
            d3 = math.sqrt(d2xy + dz * dz)
            if d3 < 1.0:
                d3 = 1.0
            fspl_1m = 20.0 * math.log10(freq_mhz) - 27.55
            wall_loss = 0.0
            for si in range(seg_count):
                x1 = segments[si, 0]
                y1 = segments[si, 1]
                x2 = segments[si, 2]
                y2 = segments[si, 3]
                den = (x - ax) * (y2 - y1) - (y - ay) * (x2 - x1)
                if den > -1.0e-7 and den < 1.0e-7:
                    continue
                t = ((x1 - ax) * (y2 - y1) - (y1 - ay) * (x2 - x1)) / den
                u = ((x1 - ax) * (y - ay) - (y1 - ay) * (x - ax)) / den
                if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                    zhit = az + (rz - az) * t
                    if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                        wall_loss += segments[si, 6]
            rssi = tx + gain - fspl_1m - 10.0 * ple * math.log10(d3) - floor_loss - wall_loss
            if rssi < disconnected:
                rssi = disconnected
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
    _cuda_direct_rssi_grid = None


class NumbaCUDARFAccelerator:
    """Cached Numba-CUDA kernels for a selected NVIDIA adapter."""

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
        self.threads_per_block = 256

    def _launch_shape(self, count: int) -> Tuple[int, int]:
        threads = self.threads_per_block
        return (max(1, (int(count) + threads - 1) // threads), threads)

    def influence_mask(self, xs: np.ndarray, ys: np.ndarray, links: np.ndarray,
                       valid_mask: Optional[np.ndarray]) -> np.ndarray:
        xs32 = np.ascontiguousarray(xs, dtype=np.float32)
        ys32 = np.ascontiguousarray(ys, dtype=np.float32)
        links32 = np.ascontiguousarray(links, dtype=np.float32).reshape(-1, 8)
        valid = np.ones((len(ys32), len(xs32)), dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask, dtype=np.uint8)
        total = valid.size
        with self._lock, cuda.gpus[self.device_index]:
            d_xs = cuda.to_device(xs32)
            d_ys = cuda.to_device(ys32)
            d_links = cuda.to_device(links32)
            d_valid = cuda.to_device(valid.ravel())
            d_output = cuda.device_array(total, dtype=np.uint8)
            blocks, threads = self._launch_shape(total)
            _cuda_influence_mask[blocks, threads](d_xs, d_ys, d_links, d_valid, d_output, len(xs32), len(ys32))
            cuda.synchronize()
            output = d_output.copy_to_host()
        return output.reshape((len(ys32), len(xs32))).astype(bool)

    def strongest_indices(self, stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        source = np.ascontiguousarray(stack, dtype=np.float32)
        fields = int(source.shape[0])
        points = int(np.prod(source.shape[1:]))
        flat = source.reshape(fields, points)
        with self._lock, cuda.gpus[self.device_index]:
            d_source = cuda.to_device(flat)
            d_indices = cuda.device_array(points, dtype=np.int32)
            d_valid = cuda.device_array(points, dtype=np.uint8)
            blocks, threads = self._launch_shape(points)
            _cuda_strongest_index[blocks, threads](d_source, fields, points, d_indices, d_valid)
            cuda.synchronize()
            indices = d_indices.copy_to_host()
            valid = d_valid.copy_to_host()
        shape = source.shape[1:]
        return indices.reshape(shape), valid.reshape(shape).astype(bool)

    def resample(self, source_xs: np.ndarray, source_ys: np.ndarray, source_values: np.ndarray,
                 target_xs: np.ndarray, target_ys: np.ndarray) -> np.ndarray:
        source = np.ascontiguousarray(source_values, dtype=np.float32)
        tx = np.ascontiguousarray(target_xs, dtype=np.float32)
        ty = np.ascontiguousarray(target_ys, dtype=np.float32)
        output = np.full(len(tx) * len(ty), np.nan, dtype=np.float32)
        source_dx = float(source_xs[1] - source_xs[0]) if len(source_xs) > 1 else 1.0
        source_dy = float(source_ys[1] - source_ys[0]) if len(source_ys) > 1 else 1.0
        with self._lock, cuda.gpus[self.device_index]:
            d_source = cuda.to_device(source)
            d_tx = cuda.to_device(tx)
            d_ty = cuda.to_device(ty)
            d_output = cuda.to_device(output)
            blocks, threads = self._launch_shape(output.size)
            _cuda_resample_bilinear[blocks, threads](
                d_source, source.shape[1], source.shape[0], np.float32(source_xs[0]),
                np.float32(source_ys[0]), np.float32(source_dx), np.float32(source_dy),
                d_tx, d_ty, len(tx), len(ty), d_output,
            )
            cuda.synchronize()
            output = d_output.copy_to_host()
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
            output = np.empty((source.size, 4), dtype=np.uint8)
            high_index = int(np.argmax(maxs))
            low_index = int(np.argmin(mins))
            with cuda.gpus[self.device_index]:
                d_source = cuda.to_device(source.ravel())
                d_mins = cuda.to_device(mins)
                d_maxs = cuda.to_device(maxs)
                d_colours = cuda.to_device(colours)
                d_output = cuda.device_array(output.shape, dtype=np.uint8)
                blocks, threads = self._launch_shape(source.size)
                _cuda_colourise[blocks, threads](
                    d_source, source.size, d_mins, d_maxs, d_colours, len(zones),
                    high_index, low_index, d_output,
                )
                cuda.synchronize()
                output = d_output.copy_to_host()
            return output.reshape((*source.shape, 4))
        finally:
            self._lock.release()


# ------------------------------ OpenCL fallback -----------------------------
_KERNEL_SOURCE = r"""
__kernel void influence_mask(__global const float *xs, __global const float *ys,
 __global const float *links, const int link_count, __global const uchar *valid,
 __global uchar *output, const int cols, const int rows) {
 const int gid=get_global_id(0); const int total=cols*rows; if(gid>=total)return;
 if(valid[gid]==0){output[gid]=0;return;} const int ix=gid%cols; const int iy=gid/cols;
 const float x=xs[ix], y=ys[iy]; uchar keep=0;
 for(int link=0;link<link_count;++link){const int base=link*8; const float dx=x-links[base];
 const float dy=y-links[base+1], dz=links[base+2]; float d2=dx*dx+dy*dy+dz*dz;
 const float r2=links[base+7]; if(r2>0.0f&&d2>r2)continue; d2=fmax(d2,1.0f);
 const float upper=links[base+5]-links[base+3]-10.0f*links[base+4]*log10(sqrt(d2));
 if(upper>=links[base+6]-0.25f){keep=1;break;}} output[gid]=keep; }
__kernel void strongest_index(__global const float *stack,const int fields,const int points,
 __global int *indices,__global uchar *valid){const int p=get_global_id(0);if(p>=points)return;
 float best=-INFINITY;int bi=0;uchar any=0;for(int f=0;f<fields;++f){const float v=stack[f*points+p];
 if(isfinite(v)){any=1;if(v>best){best=v;bi=f;}}}indices[p]=bi;valid[p]=any;}
__kernel void resample_bilinear(__global const float *source,const int sc,const int sr,
 const float x0,const float y0,const float dx,const float dy,__global const float *txs,
 __global const float *tys,const int tc,const int tr,__global float *out){const int gid=get_global_id(0);
 if(gid>=tc*tr)return;const int tx=gid%tc,ty=gid/tc;float fx=clamp((txs[tx]-x0)/dx,0.0f,(float)(sc-1));
 float fy=clamp((tys[ty]-y0)/dy,0.0f,(float)(sr-1));const int xa=(int)floor(fx),ya=(int)floor(fy);
 const int xb=min(xa+1,sc-1),yb=min(ya+1,sr-1);const float wx=fx-xa,wy=fy-ya;
 const int ids[4]={ya*sc+xa,ya*sc+xb,yb*sc+xa,yb*sc+xb};const float ws[4]={(1-wx)*(1-wy),wx*(1-wy),(1-wx)*wy,wx*wy};
 float vs=0,ww=0;for(int k=0;k<4;++k){const float v=source[ids[k]];if(isfinite(v)){vs+=v*ws[k];ww+=ws[k];}}
 out[gid]=ww>1e-8f?vs/ww:NAN;}
__kernel void colourise(__global const float *values,const int points,__global const float *mins,
 __global const float *maxs,__global const uchar4 *colours,const int count,const int hi,const int lo,
 __global uchar4 *out){const int gid=get_global_id(0);if(gid>=points)return;const float v=values[gid];
 if(!isfinite(v)){out[gid]=(uchar4)(0,0,0,0);return;}int selected=-1;for(int z=0;z<count;++z)
 if(v>=mins[z]&&v<maxs[z]){selected=z;break;}if(selected<0)selected=v>=maxs[hi]?hi:lo;out[gid]=colours[selected];}
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
        # Retrieve each kernel exactly once. This prevents PyOpenCL's
        # RepeatedKernelRetrieval warning and its associated creation overhead.
        self._influence_kernel = cl.Kernel(self.program, "influence_mask")
        self._strongest_kernel = cl.Kernel(self.program, "strongest_index")
        self._resample_kernel = cl.Kernel(self.program, "resample_bilinear")
        self._colour_kernel = cl.Kernel(self.program, "colourise")
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

    def influence_mask(self, xs, ys, links, valid_mask):
        xs32=np.ascontiguousarray(xs,dtype=np.float32);ys32=np.ascontiguousarray(ys,dtype=np.float32)
        links32=np.ascontiguousarray(links,dtype=np.float32).reshape(-1,8)
        valid=np.ones((len(ys32),len(xs32)),dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask,dtype=np.uint8)
        output=np.empty(valid.size,dtype=np.uint8);mf=cl.mem_flags
        with self._lock:
            args=[cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=xs32),cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=ys32),cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=links32),np.int32(len(links32)),cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=valid.ravel()),cl.Buffer(self.context,mf.WRITE_ONLY,output.nbytes),np.int32(len(xs32)),np.int32(len(ys32))]
            self._influence_kernel.set_args(*args);cl.enqueue_nd_range_kernel(self.queue,self._influence_kernel,(output.size,),None)
            cl.enqueue_copy(self.queue,output,args[5]).wait()
        return output.reshape((len(ys32),len(xs32))).astype(bool)

    def strongest_indices(self, stack):
        source=np.ascontiguousarray(stack,dtype=np.float32);fields=int(source.shape[0]);points=int(np.prod(source.shape[1:]));indices=np.empty(points,dtype=np.int32);valid=np.empty(points,dtype=np.uint8);mf=cl.mem_flags
        with self._lock:
            src=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=source.ravel());idx=cl.Buffer(self.context,mf.WRITE_ONLY,indices.nbytes);vb=cl.Buffer(self.context,mf.WRITE_ONLY,valid.nbytes)
            self._strongest_kernel.set_args(src,np.int32(fields),np.int32(points),idx,vb);cl.enqueue_nd_range_kernel(self.queue,self._strongest_kernel,(points,),None);cl.enqueue_copy(self.queue,indices,idx);cl.enqueue_copy(self.queue,valid,vb).wait()
        shape=source.shape[1:];return indices.reshape(shape),valid.reshape(shape).astype(bool)

    def resample(self, source_xs, source_ys, source_values, target_xs, target_ys):
        source=np.ascontiguousarray(source_values,dtype=np.float32);tx=np.ascontiguousarray(target_xs,dtype=np.float32);ty=np.ascontiguousarray(target_ys,dtype=np.float32);output=np.empty((len(ty),len(tx)),dtype=np.float32);mf=cl.mem_flags
        dx=float(source_xs[1]-source_xs[0]) if len(source_xs)>1 else 1.;dy=float(source_ys[1]-source_ys[0]) if len(source_ys)>1 else 1.
        with self._lock:
            src=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=source.ravel());txb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=tx);tyb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=ty);out=cl.Buffer(self.context,mf.WRITE_ONLY,output.nbytes)
            self._resample_kernel.set_args(src,np.int32(source.shape[1]),np.int32(source.shape[0]),np.float32(source_xs[0]),np.float32(source_ys[0]),np.float32(dx),np.float32(dy),txb,tyb,np.int32(len(tx)),np.int32(len(ty)),out)
            cl.enqueue_nd_range_kernel(self.queue,self._resample_kernel,(output.size,),None);cl.enqueue_copy(self.queue,output,out).wait()
        return output.astype(np.float64)

    def colourise(self, values, zones, blocking=True):
        acquired=self._lock.acquire(blocking=blocking)
        if not acquired:return None
        try:
            source=np.ascontiguousarray(values,dtype=np.float32);mins=np.ascontiguousarray([z[0] for z in zones],dtype=np.float32);maxs=np.ascontiguousarray([z[1] for z in zones],dtype=np.float32);colours=np.ascontiguousarray([[z[2],z[3],z[4],z[5]] for z in zones],dtype=np.uint8);output=np.empty((source.size,4),dtype=np.uint8);mf=cl.mem_flags
            src=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=source.ravel());mb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=mins);xb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=maxs);cb=cl.Buffer(self.context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=colours);out=cl.Buffer(self.context,mf.WRITE_ONLY,output.nbytes)
            self._colour_kernel.set_args(src,np.int32(source.size),mb,xb,cb,np.int32(len(zones)),np.int32(int(np.argmax(maxs))),np.int32(int(np.argmin(mins))),out)
            cl.enqueue_nd_range_kernel(self.queue,self._colour_kernel,(source.size,),None);cl.enqueue_copy(self.queue,output,out).wait();return output.reshape((*source.shape,4))
        finally:self._lock.release()



# ------------------------- Direct RSSI grid helpers -------------------------
if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_direct_rssi_grid(xs, ys, valid, ap_data, segments, disconnected, combine_mode):
        rows = ys.shape[0]
        cols = xs.shape[0]
        out = np.empty((rows, cols), dtype=np.float64)
        counts = np.zeros((rows, cols), dtype=np.int16)
        for iy in prange(rows):
            for ix in range(cols):
                if valid[iy, ix] == 0:
                    out[iy, ix] = np.nan
                    counts[iy, ix] = 0
                    continue
                x = xs[ix]
                y = ys[iy]
                best = disconnected
                power_sum_mw = 0.0
                pc = 0
                for ai in range(ap_data.shape[0]):
                    ax = ap_data[ai, 0]; ay = ap_data[ai, 1]; az = ap_data[ai, 2]; rz = ap_data[ai, 3]
                    tx = ap_data[ai, 4]; gain = ap_data[ai, 5]; freq_mhz = ap_data[ai, 6]; ple = ap_data[ai, 7]
                    cutoff2 = ap_data[ai, 8]; floor_loss = ap_data[ai, 9]
                    dx = x - ax; dy = y - ay; dz = rz - az
                    d2xy = dx * dx + dy * dy
                    if cutoff2 > 0.0 and d2xy > cutoff2:
                        continue
                    d3 = math.sqrt(d2xy + dz * dz)
                    if d3 < 1.0:
                        d3 = 1.0
                    fspl_1m = 20.0 * math.log10(freq_mhz) - 27.55
                    wall_loss = 0.0
                    for si in range(segments.shape[0]):
                        x1 = segments[si, 0]; y1 = segments[si, 1]; x2 = segments[si, 2]; y2 = segments[si, 3]
                        den = (x - ax) * (y2 - y1) - (y - ay) * (x2 - x1)
                        if den > -1.0e-7 and den < 1.0e-7:
                            continue
                        t = ((x1 - ax) * (y2 - y1) - (y1 - ay) * (x2 - x1)) / den
                        u = ((x1 - ax) * (y - ay) - (y1 - ay) * (x - ax)) / den
                        if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
                            zhit = az + (rz - az) * t
                            if zhit >= segments[si, 4] and zhit <= segments[si, 5]:
                                wall_loss += segments[si, 6]
                    rssi = tx + gain - fspl_1m - 10.0 * ple * math.log10(d3) - floor_loss - wall_loss
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
                out[iy, ix] = best
                counts[iy, ix] = pc
        return out, counts
else:
    _numba_direct_rssi_grid = None


def gpu_direct_rssi_grid(xs: np.ndarray, ys: np.ndarray, ap_data: np.ndarray, segments: np.ndarray,
                         valid_mask: Optional[np.ndarray], disconnected: float, combine_mode: int,
                         settings: Any) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    if not _enabled(settings):
        return None
    if np.asarray(ap_data).size == 0:
        return None
    result = _execute_gpu(
        settings,
        "direct RSSI grid calculation",
        lambda backend: (backend.direct_rssi_grid(xs, ys, ap_data, segments, valid_mask, disconnected, combine_mode), backend.info.label, backend.backend_name)
        if hasattr(backend, "direct_rssi_grid") else None,
    )
    if result is None:
        return None
    (rssi, counts), label, backend_name = result
    return rssi, counts, f"{backend_name}: {label}"


def numba_direct_rssi_grid(xs: np.ndarray, ys: np.ndarray, ap_data: np.ndarray, segments: np.ndarray,
                           valid_mask: Optional[np.ndarray], disconnected: float, combine_mode: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if _numba_direct_rssi_grid is None or np.asarray(ap_data).size == 0:
        return None
    xs64 = np.ascontiguousarray(xs, dtype=np.float64)
    ys64 = np.ascontiguousarray(ys, dtype=np.float64)
    aps64 = np.ascontiguousarray(ap_data, dtype=np.float64).reshape(-1, 10)
    seg64 = np.ascontiguousarray(segments, dtype=np.float64).reshape(-1, 7)
    valid = np.ones((len(ys64), len(xs64)), dtype=np.uint8) if valid_mask is None else np.ascontiguousarray(valid_mask, dtype=np.uint8)
    return _numba_direct_rssi_grid(xs64, ys64, valid, aps64, seg64, float(disconnected), int(combine_mode))


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
