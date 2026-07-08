from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any


def plot_results(inputs: list[Path], output_dir: Path) -> list[Path]:
    pd, _ = _plotting_libs()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in _expand_inputs(inputs):
        frame = pd.read_csv(path)
        if "gflops_est" in frame.columns:
            written.extend(_plot_compute(frame, path, output_dir))
        if _bandwidth_column(frame) is not None:
            written.extend(_plot_memory(frame, path, output_dir))
    return written


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in inputs:
        value = str(path)
        if any(char in value for char in "*?["):
            matches = [Path(match) for match in glob(value, recursive=True)]
            if not matches:
                raise FileNotFoundError(f"No files matched pattern: {value}")
            expanded.extend(sorted(matches))
        else:
            expanded.append(path)
    return expanded


def _plot_compute(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    written: list[Path] = []
    written.extend(_plot_compute_throughput(frame, source, output_dir))
    written.extend(_plot_divergence_slowdown(frame, source, output_dir))
    written.extend(_plot_cpu_gpu_divergence(frame, source, output_dir))
    return written


def _plot_compute_throughput(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    subset = _filter_uniform(frame)
    if subset.empty:
        return []

    _, plt = _plotting_libs()
    fig, ax = plt.subplots()
    for label, group in subset.groupby(["backend", "block_size"], dropna=False):
        backend, block_size = label
        group = group.sort_values("n")
        ax.plot(group["n"], group["gflops_est"], marker="o", label=f"{backend}, block={block_size}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("n")
    ax.set_ylabel("GFLOP/s estimate")
    ax.set_title("Uniform compute throughput")
    _finish_axes(ax)
    return [_save(fig, output_dir / f"{source.stem}-compute-throughput.png")]


def _plot_divergence_slowdown(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    subset = _filter_divergent(frame)
    if subset.empty or "slowdown_vs_uniform" not in subset.columns:
        return []

    _, plt = _plotting_libs()
    fig, ax = plt.subplots()
    for label, group in subset.groupby(["backend", "block_size"], dropna=False):
        backend, block_size = label
        max_n = group["n"].max()
        group = group[group["n"] == max_n].sort_values("divergence_degree")
        ax.plot(
            group["divergence_degree"],
            group["slowdown_vs_uniform"],
            marker="o",
            label=f"{backend}, block={block_size}, n={max_n}",
        )

    ax.set_xlabel("divergence degree")
    ax.set_ylabel("slowdown vs uniform")
    ax.set_title("Warp divergence slowdown")
    _finish_axes(ax)
    return [_save(fig, output_dir / f"{source.stem}-divergence-slowdown.png")]


def _plot_cpu_gpu_divergence(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    subset = _filter_divergent(frame)
    if subset.empty or "slowdown_vs_uniform" not in subset.columns:
        return []
    if set(subset["backend"].dropna()) < {"cpu", "gpu"}:
        return []

    _, plt = _plotting_libs()
    fig, ax = plt.subplots()
    for backend, group in subset.groupby("backend"):
        max_n = group["n"].max()
        group = group[group["n"] == max_n]
        group = group.groupby("divergence_degree", as_index=False)["slowdown_vs_uniform"].median()
        group = group.sort_values("divergence_degree")
        ax.plot(group["divergence_degree"], group["slowdown_vs_uniform"], marker="o", label=backend)

    ax.set_xlabel("divergence degree")
    ax.set_ylabel("median slowdown vs uniform")
    ax.set_title("CPU vs GPU divergence sensitivity")
    _finish_axes(ax)
    return [_save(fig, output_dir / f"{source.stem}-cpu-vs-gpu-divergence.png")]


def _plot_memory(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    written: list[Path] = []
    written.extend(_plot_memory_bandwidth(frame, source, output_dir))
    written.extend(_plot_memory_block_size(frame, source, output_dir))
    written.extend(_plot_peak_percent(frame, source, output_dir))
    return written


def _plot_memory_bandwidth(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    bandwidth = _bandwidth_column(frame)
    if bandwidth is None:
        return []

    x_col = "logical_n" if "logical_n" in frame.columns else "n"
    pattern_col = _pattern_column(frame)
    _, plt = _plotting_libs()
    fig, ax = plt.subplots()
    for label, group in frame.groupby([pattern_col, "stride"], dropna=False):
        pattern, stride = label
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[bandwidth], marker="o", label=f"{pattern}, stride={stride}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel("effective GB/s")
    ax.set_title("Memory bandwidth by access pattern")
    _finish_axes(ax)
    return [_save(fig, output_dir / f"{source.stem}-memory-bandwidth-pattern.png")]


def _plot_memory_block_size(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    bandwidth = _bandwidth_column(frame)
    if bandwidth is None or "block_size" not in frame.columns:
        return []

    x_col = "logical_n" if "logical_n" in frame.columns else "n"
    pattern_col = _pattern_column(frame)
    max_n = frame[x_col].max()
    subset = frame[frame[x_col] == max_n]

    _, plt = _plotting_libs()
    fig, ax = plt.subplots()
    for pattern, group in subset.groupby(pattern_col):
        group = group.groupby("block_size", as_index=False)[bandwidth].median()
        group = group.sort_values("block_size")
        ax.plot(group["block_size"], group[bandwidth], marker="o", label=pattern)

    ax.set_xlabel("block size")
    ax.set_ylabel("median effective GB/s")
    ax.set_title(f"Memory bandwidth over block size, {x_col}={max_n}")
    _finish_axes(ax)
    return [_save(fig, output_dir / f"{source.stem}-memory-block-size.png")]


def _plot_peak_percent(frame: Any, source: Path, output_dir: Path) -> list[Path]:
    if "peak_bandwidth_pct" not in frame.columns or frame["peak_bandwidth_pct"].dropna().empty:
        return []

    x_col = "logical_n" if "logical_n" in frame.columns else "n"
    pattern_col = _pattern_column(frame)
    _, plt = _plotting_libs()
    fig, ax = plt.subplots()
    for pattern, group in frame.groupby(pattern_col):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group["peak_bandwidth_pct"], marker="o", label=pattern)

    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel("percent of peak bandwidth")
    ax.set_title("Measured bandwidth as percent of peak")
    _finish_axes(ax)
    return [_save(fig, output_dir / f"{source.stem}-memory-peak-percent.png")]


def _filter_uniform(frame: Any) -> Any:
    if "mode" in frame.columns:
        return frame[frame["mode"] == "uniform"]
    return frame[frame["kernel"] == "uniform"]


def _filter_divergent(frame: Any) -> Any:
    if "mode" in frame.columns:
        return frame[frame["mode"].astype(str).str.startswith("divergence")]
    return frame[frame["kernel"] == "divergent"]


def _bandwidth_column(frame: Any) -> str | None:
    if "bandwidth_gbs" in frame.columns:
        return "bandwidth_gbs"
    if "bandwidth_gbs_est" in frame.columns:
        return "bandwidth_gbs_est"
    return None


def _pattern_column(frame: Any) -> str:
    return "access_pattern" if "access_pattern" in frame.columns else "pattern"


def _finish_axes(ax) -> None:
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    ax.figure.tight_layout()


def _save(fig, output: Path) -> Path:
    _, plt = _plotting_libs()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _plotting_libs() -> tuple[Any, Any]:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return pd, plt
