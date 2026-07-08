from __future__ import annotations

import json
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from numba import cuda


@dataclass(frozen=True)
class DeviceInfo:
    gpu_name: str
    compute_capability: tuple[int, int] | None
    sm_count: int | None
    warp_size: int
    max_threads_per_block: int | None
    max_threads_per_sm: int | None
    max_blocks_per_sm: int | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.compute_capability is not None:
            data["compute_capability"] = list(self.compute_capability)
        return data


def require_cuda() -> None:
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check the NVIDIA driver, CUDA runtime, and "
            "that the local environment was created with: uv sync."
        )


def _device_attr(device: object, name: str, default: int | None = None) -> int | None:
    return getattr(device, name, default)


def get_device_info() -> DeviceInfo:
    require_cuda()
    device = cuda.get_current_device()
    return DeviceInfo(
        gpu_name=str(getattr(device, "name", "unknown")),
        compute_capability=getattr(device, "compute_capability", None),
        sm_count=_device_attr(device, "MULTIPROCESSOR_COUNT"),
        warp_size=_device_attr(device, "WARP_SIZE", 32) or 32,
        max_threads_per_block=_device_attr(device, "MAX_THREADS_PER_BLOCK"),
        max_threads_per_sm=_device_attr(device, "MAX_THREADS_PER_MULTIPROCESSOR"),
        max_blocks_per_sm=_device_attr(device, "MAX_BLOCKS_PER_MULTIPROCESSOR"),
    )


def estimate_occupancy(block_size: int, info: DeviceInfo | None = None) -> dict[str, float | int | None]:
    info = info or get_device_info()
    warp_size = info.warp_size or 32
    max_threads_per_sm = info.max_threads_per_sm or 2048
    max_blocks_per_sm = info.max_blocks_per_sm or 32

    blocks_by_threads = max(1, max_threads_per_sm // block_size)
    resident_blocks = min(max_blocks_per_sm, blocks_by_threads)
    warps_per_block = (block_size + warp_size - 1) // warp_size
    active_warps = resident_blocks * warps_per_block
    max_warps = max_threads_per_sm // warp_size
    occupancy = active_warps / max_warps if max_warps else 0.0

    return {
        "resident_blocks_per_sm_est": resident_blocks,
        "active_warps_per_sm_est": active_warps,
        "max_warps_per_sm": max_warps,
        "occupancy_est": min(occupancy, 1.0),
        "active_warps_measured": None,
        "max_warps_measured": None,
        "occupancy_measured": None,
    }


def write_device_json(path: Path) -> None:
    info = get_device_info()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(info.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_device_info(json_output: bool = False) -> None:
    info = get_device_info()
    if json_output:
        json.dump(info.to_dict(), sys.stdout, indent=2, sort_keys=True)
        print()
        return

    print(f"device={info.gpu_name}")
    print(f"compute_capability={info.compute_capability}")
    print(f"sm_count={info.sm_count}")
    print(f"warp_size={info.warp_size}")
    print(f"max_threads_per_block={info.max_threads_per_block}")
    print(f"max_threads_per_sm={info.max_threads_per_sm}")
    print(f"max_blocks_per_sm={info.max_blocks_per_sm}")
