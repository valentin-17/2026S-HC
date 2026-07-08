from __future__ import annotations

from collections.abc import Callable
from statistics import mean

from numba import cuda


def time_cuda_kernel(
    launch: Callable[[], None],
    repeats: int,
    warmups: int = 1,
) -> float:
    for _ in range(warmups):
        launch()
    cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = cuda.event(timing=True)
        end = cuda.event(timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples_ms.append(cuda.event_elapsed_time(start, end))

    return mean(samples_ms)

