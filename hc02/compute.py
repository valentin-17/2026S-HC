from __future__ import annotations

from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from numba import cuda, njit

from hc02.device import estimate_occupancy, get_device_info, require_cuda
from hc02.reporting import result_path, seconds_from_ms, write_csv
from hc02.timing import time_cuda_kernel

COMPUTE_FLOPS_PER_ITER = 8


@cuda.jit(fastmath=True)
def compute_uniform_kernel(x, out, iterations):
    i = cuda.grid(1)
    if i >= x.size:
        return

    v = x[i]
    c0 = np.float32(1.000001)
    c1 = np.float32(0.0000001)
    c2 = np.float32(0.00013)
    for _ in range(iterations):
        v = v * c0 + c2
        v = v - v * v * c1
        v = v * c0 - c2
    out[i] = v


@cuda.jit(fastmath=True)
def compute_divergent_kernel(x, out, iterations, divergence_degree):
    i = cuda.grid(1)
    if i >= x.size:
        return

    degree = divergence_degree
    if degree < 1:
        degree = 1
    if degree > 32:
        degree = 32

    lane = cuda.threadIdx.x % 32
    path = lane % degree
    v = x[i]

    for p in range(32):
        if p < degree:
            if path == p:
                bias = np.float32(p + 1) * np.float32(0.00001)
                c0 = np.float32(1.000001) + bias
                c1 = np.float32(0.0000001)
                for _ in range(iterations):
                    v = v * c0 + bias
                    v = v - v * v * c1
                    v = v * c0 - bias
    out[i] = v


@njit(fastmath=True)
def compute_cpu_uniform(x: np.ndarray, iterations: int) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.size):
        v = x[i]
        for _ in range(iterations):
            v = v * np.float32(1.000001) + np.float32(0.00013)
            v = v - v * v * np.float32(0.0000001)
            v = v * np.float32(1.000001) - np.float32(0.00013)
        out[i] = v
    return out


@njit(fastmath=True)
def compute_cpu_divergent(x: np.ndarray, iterations: int, divergence_degree: int) -> np.ndarray:
    out = np.empty_like(x)
    degree = max(1, min(32, divergence_degree))
    for i in range(x.size):
        path = i % degree
        v = x[i]
        for p in range(32):
            if p < degree:
                if path == p:
                    bias = np.float32(p + 1) * np.float32(0.00001)
                    c0 = np.float32(1.000001) + bias
                    c1 = np.float32(0.0000001)
                    for _ in range(iterations):
                        v = v * c0 + bias
                        v = v - v * v * c1
                        v = v * c0 - bias
        out[i] = v
    return out


def _time_cpu(call, repeats: int) -> float:
    call()
    samples_ms: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples_ms.append((perf_counter() - start) * 1000.0)
    return mean(samples_ms)


def _compute_metrics(
    n: int,
    iterations: int,
    elapsed_ms: float,
    threads_launched: int,
    uniform_elapsed_ms: float | None = None,
) -> dict[str, float | int | None]:
    seconds = seconds_from_ms(elapsed_ms)
    useful_ops = n * iterations * COMPUTE_FLOPS_PER_ITER
    throughput_gops = useful_ops / seconds / 1e9
    return {
        "threads_launched": threads_launched,
        "elements_per_thread": n / threads_launched if threads_launched else None,
        "useful_ops": useful_ops,
        "throughput_gops": throughput_gops,
        "gflops_est": throughput_gops,
        "slowdown_vs_uniform": elapsed_ms / uniform_elapsed_ms if uniform_elapsed_ms else 1.0,
    }


def _empty_occupancy() -> dict[str, None]:
    return {
        "resident_blocks_per_sm_est": None,
        "active_warps_per_sm_est": None,
        "max_warps_per_sm": None,
        "occupancy_est": None,
        "active_warps_measured": None,
        "max_warps_measured": None,
        "occupancy_measured": None,
    }


def run_compute_benchmark(
    sizes: list[int],
    block_sizes: list[int],
    divergence_degrees: list[int],
    iterations: int,
    repeats: int,
    output: Path | None = None,
    include_cpu: bool = False,
    cpu_sizes: list[int] | None = None,
) -> Path:
    require_cuda()
    device_info = get_device_info()
    rows: list[dict[str, object]] = []
    divergence_degrees = sorted({max(1, min(32, degree)) for degree in divergence_degrees})

    for n in sizes:
        host_x = np.linspace(0.1, 1.0, n, dtype=np.float32)
        d_x = cuda.to_device(host_x)
        d_out = cuda.device_array_like(d_x)

        for block_size in block_sizes:
            grid_size = (n + block_size - 1) // block_size
            threads_launched = grid_size * block_size
            occupancy = estimate_occupancy(block_size, device_info)

            elapsed_ms = time_cuda_kernel(
                lambda: compute_uniform_kernel[grid_size, block_size](d_x, d_out, iterations),
                repeats=repeats,
            )
            uniform_elapsed_ms = elapsed_ms
            rows.append(
                {
                    "suite": "compute",
                    "mode": "uniform",
                    "backend": "gpu",
                    "kernel": "uniform",
                    "n": n,
                    "block_size": block_size,
                    "divergence_degree": 1,
                    "iterations": iterations,
                    "elapsed_ms": elapsed_ms,
                    **_compute_metrics(n, iterations, elapsed_ms, threads_launched),
                    **occupancy,
                }
            )

            for degree in divergence_degrees:
                if degree == 1:
                    continue

                elapsed_ms = time_cuda_kernel(
                    lambda degree=degree: compute_divergent_kernel[grid_size, block_size](
                        d_x,
                        d_out,
                        iterations,
                        degree,
                    ),
                    repeats=repeats,
                )
                rows.append(
                    {
                        "suite": "compute",
                        "mode": "divergence-gpu",
                        "backend": "gpu",
                        "kernel": "divergent",
                        "n": n,
                        "block_size": block_size,
                        "divergence_degree": degree,
                        "iterations": iterations,
                        "elapsed_ms": elapsed_ms,
                        **_compute_metrics(
                            n,
                            iterations,
                            elapsed_ms,
                            threads_launched,
                            uniform_elapsed_ms=uniform_elapsed_ms,
                        ),
                        **occupancy,
                    }
                )

    if include_cpu:
        for n in cpu_sizes or sizes:
            host_x = np.linspace(0.1, 1.0, n, dtype=np.float32)

            elapsed_ms = _time_cpu(lambda: compute_cpu_uniform(host_x, iterations), repeats)
            uniform_elapsed_ms = elapsed_ms
            rows.append(
                {
                    "suite": "compute",
                    "mode": "uniform",
                    "backend": "cpu",
                    "kernel": "uniform",
                    "n": n,
                    "block_size": 0,
                    "divergence_degree": 1,
                    "iterations": iterations,
                    "elapsed_ms": elapsed_ms,
                    **_compute_metrics(n, iterations, elapsed_ms, n),
                    **_empty_occupancy(),
                }
            )

            for degree in divergence_degrees:
                if degree == 1:
                    continue

                elapsed_ms = _time_cpu(
                    lambda degree=degree: compute_cpu_divergent(host_x, iterations, degree),
                    repeats,
                )
                rows.append(
                    {
                        "suite": "compute",
                        "mode": "divergence-cpu",
                        "backend": "cpu",
                        "kernel": "divergent",
                        "n": n,
                        "block_size": 0,
                        "divergence_degree": degree,
                        "iterations": iterations,
                        "elapsed_ms": elapsed_ms,
                        **_compute_metrics(n, iterations, elapsed_ms, n, uniform_elapsed_ms=uniform_elapsed_ms),
                        **_empty_occupancy(),
                    }
                )

    output = output or result_path("compute")
    write_csv(rows, output)
    return output
