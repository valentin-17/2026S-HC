from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import cuda

from hc02.device import estimate_occupancy, get_device_info, require_cuda
from hc02.reporting import result_path, seconds_from_ms, write_csv
from hc02.timing import time_cuda_kernel

FLOAT_BYTES = 4
INT_BYTES = 4


@cuda.jit(fastmath=True)
def memory_coalesced_kernel(a, b, scale):
    i = cuda.grid(1)
    if i < b.size:
        b[i] = a[i] * scale


@cuda.jit(fastmath=True)
def memory_strided_kernel(a, b, stride, scale):
    i = cuda.grid(1)
    if i < b.size:
        src = i * stride
        b[i] = a[src] * scale


@cuda.jit(fastmath=True)
def memory_gather_kernel(a, indices, b, scale):
    i = cuda.grid(1)
    if i < b.size:
        b[i] = a[indices[i]] * scale


def _bandwidth_gbs(effective_bytes: int, elapsed_ms: float) -> float:
    return effective_bytes / seconds_from_ms(elapsed_ms) / 1e9


def _memory_metrics(
    logical_n: int,
    bytes_read: int,
    bytes_written: int,
    index_bytes_read: int,
    elapsed_ms: float,
    peak_bandwidth_gbs: float | None,
) -> dict[str, float | int | None]:
    effective_bytes = bytes_read + bytes_written + index_bytes_read
    bandwidth_gbs = _bandwidth_gbs(effective_bytes, elapsed_ms)
    useful_flops = logical_n
    return {
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "index_bytes_read": index_bytes_read,
        "effective_bytes": effective_bytes,
        "bandwidth_gbs": bandwidth_gbs,
        "bandwidth_gbs_est": bandwidth_gbs,
        "peak_bandwidth_gbs": peak_bandwidth_gbs,
        "peak_bandwidth_pct": bandwidth_gbs / peak_bandwidth_gbs * 100.0 if peak_bandwidth_gbs else None,
        "flops_per_byte_est": useful_flops / effective_bytes if effective_bytes else None,
    }


def run_memory_benchmark(
    sizes: list[int],
    block_sizes: list[int],
    strides: list[int],
    repeats: int,
    seed: int,
    output: Path | None = None,
    peak_bandwidth_gbs: float | None = None,
    max_input_elements: int = 67_108_864,
) -> Path:
    require_cuda()
    device_info = get_device_info()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    strides = sorted({max(1, stride) for stride in strides})
    max_stride = max(strides or [1])

    for requested_n in sizes:
        required_elements = requested_n * max_stride
        if required_elements <= max_input_elements:
            logical_n = requested_n
            input_elements = max(requested_n, required_elements)
        else:
            logical_n = max(1, max_input_elements // max_stride)
            input_elements = logical_n * max_stride

        host_a = rng.random(input_elements, dtype=np.float32)
        host_indices = rng.integers(0, input_elements, size=logical_n, dtype=np.int32)
        d_a = cuda.to_device(host_a)
        d_b = cuda.device_array(logical_n, dtype=np.float32)
        d_indices = cuda.to_device(host_indices)

        for block_size in block_sizes:
            grid_size = (logical_n + block_size - 1) // block_size
            occupancy = estimate_occupancy(block_size, device_info)

            elapsed_ms = time_cuda_kernel(
                lambda: memory_coalesced_kernel[grid_size, block_size](
                    d_a,
                    d_b,
                    np.float32(1.25),
                ),
                repeats=repeats,
            )
            rows.append(
                {
                    "suite": "memory",
                    "access_pattern": "coalesced",
                    "pattern": "coalesced",
                    "n": requested_n,
                    "logical_n": logical_n,
                    "input_elements": input_elements,
                    "block_size": block_size,
                    "stride": 1,
                    "elapsed_ms": elapsed_ms,
                    **_memory_metrics(
                        logical_n,
                        bytes_read=logical_n * FLOAT_BYTES,
                        bytes_written=logical_n * FLOAT_BYTES,
                        index_bytes_read=0,
                        elapsed_ms=elapsed_ms,
                        peak_bandwidth_gbs=peak_bandwidth_gbs,
                    ),
                    **occupancy,
                }
            )

            for stride in strides:
                elapsed_ms = time_cuda_kernel(
                    lambda stride=stride: memory_strided_kernel[grid_size, block_size](
                        d_a,
                        d_b,
                        stride,
                        np.float32(1.25),
                    ),
                    repeats=repeats,
                )
                rows.append(
                    {
                        "suite": "memory",
                        "access_pattern": "strided",
                        "pattern": "strided",
                        "n": requested_n,
                        "logical_n": logical_n,
                        "input_elements": input_elements,
                        "block_size": block_size,
                        "stride": stride,
                        "elapsed_ms": elapsed_ms,
                        **_memory_metrics(
                            logical_n,
                            bytes_read=logical_n * FLOAT_BYTES,
                            bytes_written=logical_n * FLOAT_BYTES,
                            index_bytes_read=0,
                            elapsed_ms=elapsed_ms,
                            peak_bandwidth_gbs=peak_bandwidth_gbs,
                        ),
                        **occupancy,
                    }
                )

            elapsed_ms = time_cuda_kernel(
                lambda: memory_gather_kernel[grid_size, block_size](
                    d_a,
                    d_indices,
                    d_b,
                    np.float32(1.25),
                ),
                repeats=repeats,
            )
            rows.append(
                {
                    "suite": "memory",
                    "access_pattern": "random_gather",
                    "pattern": "random_gather",
                    "n": requested_n,
                    "logical_n": logical_n,
                    "input_elements": input_elements,
                    "block_size": block_size,
                    "stride": 0,
                    "elapsed_ms": elapsed_ms,
                    **_memory_metrics(
                        logical_n,
                        bytes_read=logical_n * FLOAT_BYTES,
                        bytes_written=logical_n * FLOAT_BYTES,
                        index_bytes_read=logical_n * INT_BYTES,
                        elapsed_ms=elapsed_ms,
                        peak_bandwidth_gbs=peak_bandwidth_gbs,
                    ),
                    **occupancy,
                }
            )

    output = output or result_path("memory")
    write_csv(rows, output)
    return output
