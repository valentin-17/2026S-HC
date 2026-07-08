from __future__ import annotations

import argparse
from pathlib import Path

from hc02.reporting import make_run_dir, parse_int_list

SIZES = [2**power for power in range(16, 28)]
DIVERGENCE = [1, 2, 4, 8, 16, 32]
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
CPU_SIZES = [2**16, 2**18, 2**20]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hc02")
    subparsers = parser.add_subparsers(dest="command", required=True)

    device = subparsers.add_parser("device", help="Print CUDA device metadata.")
    device.add_argument("--json", action="store_true", help="Print device metadata as JSON.")

    compute = subparsers.add_parser("compute", help="Run compute and divergence benchmarks.")
    compute.add_argument("--sizes", default="1048576,4194304,16777216")
    compute.add_argument("--block-sizes", default="128,256,512")
    compute.add_argument("--divergence", default="1,2,4,8,16,32")
    compute.add_argument("--iterations", type=int, default=512)
    compute.add_argument("--repeats", type=int, default=5)
    compute.add_argument("--include-cpu", action="store_true")
    compute.add_argument("--cpu-sizes", default="65536,262144,1048576")
    compute.add_argument("--out", type=Path)

    memory = subparsers.add_parser("memory", help="Run memory access benchmarks.")
    memory.add_argument("--sizes", default="1048576,4194304,16777216")
    memory.add_argument("--block-sizes", default="64,128,256,512")
    memory.add_argument("--strides", default="1,2,4,8,16,32")
    memory.add_argument("--repeats", type=int, default=5)
    memory.add_argument("--seed", type=int, default=2026)
    memory.add_argument("--peak-bandwidth-gbs", type=float)
    memory.add_argument("--max-input-elements", type=int, default=536_870_912)
    memory.add_argument("--out", type=Path)

    all_cmd = subparsers.add_parser("all", help="Run both benchmark suites.")
    all_cmd.add_argument("--sizes", default="1048576,4194304,16777216")
    all_cmd.add_argument("--block-sizes", default="128,256,512")
    all_cmd.add_argument("--iterations", type=int, default=512)
    all_cmd.add_argument("--repeats", type=int, default=5)
    all_cmd.add_argument("--seed", type=int, default=2026)
    all_cmd.add_argument("--include-cpu", action="store_true")
    all_cmd.add_argument("--peak-bandwidth-gbs", type=float)
    all_cmd.add_argument("--max-input-elements", type=int, default=536_870_912)

    suite = subparsers.add_parser("run-suite", help="Run the recommended worksheet experiment suite.")
    suite.add_argument("--run-id")
    suite.add_argument("--iterations", type=int, default=512)
    suite.add_argument("--repeats", type=int, default=5)
    suite.add_argument("--seed", type=int, default=2026)
    suite.add_argument("--peak-bandwidth-gbs", type=float)
    suite.add_argument("--max-input-elements", type=int, default=536_870_912)
    suite.add_argument("--skip-cpu", action="store_true")

    plot = subparsers.add_parser("plot", help="Create PNG plots from benchmark CSV files.")
    plot.add_argument("csv", nargs="+", type=Path)
    plot.add_argument("--out-dir", type=Path, default=Path("src/hc02/results/plots"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "device":
        from hc02.device import print_device_info

        print_device_info(json_output=args.json)
        return

    if args.command == "compute":
        from hc02.compute import run_compute_benchmark

        output = run_compute_benchmark(
            sizes=parse_int_list(args.sizes),
            block_sizes=parse_int_list(args.block_sizes),
            divergence_degrees=parse_int_list(args.divergence),
            iterations=args.iterations,
            repeats=args.repeats,
            output=args.out,
            include_cpu=args.include_cpu,
            cpu_sizes=parse_int_list(args.cpu_sizes),
        )
        print(f"wrote {output}")
        return

    if args.command == "memory":
        from hc02.memory import run_memory_benchmark

        output = run_memory_benchmark(
            sizes=parse_int_list(args.sizes),
            block_sizes=parse_int_list(args.block_sizes),
            strides=parse_int_list(args.strides),
            repeats=args.repeats,
            seed=args.seed,
            output=args.out,
            peak_bandwidth_gbs=args.peak_bandwidth_gbs,
            max_input_elements=args.max_input_elements,
        )
        print(f"wrote {output}")
        return

    if args.command == "all":
        from hc02.compute import run_compute_benchmark
        from hc02.memory import run_memory_benchmark

        sizes = parse_int_list(args.sizes)
        block_sizes = parse_int_list(args.block_sizes)
        compute_output = run_compute_benchmark(
            sizes=sizes,
            block_sizes=block_sizes,
            divergence_degrees=[1, 2, 4, 8, 16, 32],
            iterations=args.iterations,
            repeats=args.repeats,
            include_cpu=args.include_cpu,
            cpu_sizes=[65536, 262144, 1048576],
        )
        memory_output = run_memory_benchmark(
            sizes=sizes,
            block_sizes=block_sizes,
            strides=[1, 2, 4, 8, 16, 32],
            repeats=args.repeats,
            seed=args.seed,
            peak_bandwidth_gbs=args.peak_bandwidth_gbs,
            max_input_elements=args.max_input_elements,
        )
        print(f"wrote {compute_output}")
        print(f"wrote {memory_output}")
        return

    if args.command == "run-suite":
        from hc02.compute import run_compute_benchmark
        from hc02.device import write_device_json
        from hc02.memory import run_memory_benchmark
        from hc02.plotting import plot_results

        run_dir = make_run_dir(args.run_id)
        device_output = run_dir / "device.json"
        compute_output = run_dir / "compute.csv"
        memory_output = run_dir / "memory.csv"

        write_device_json(device_output)
        run_compute_benchmark(
            sizes=SIZES,
            block_sizes=BLOCK_SIZES,
            divergence_degrees=DIVERGENCE,
            iterations=args.iterations,
            repeats=args.repeats,
            output=compute_output,
            include_cpu=not args.skip_cpu,
            cpu_sizes=CPU_SIZES,
        )
        run_memory_benchmark(
            sizes=SIZES,
            block_sizes=BLOCK_SIZES,
            strides=DIVERGENCE,
            repeats=args.repeats,
            seed=args.seed,
            output=memory_output,
            peak_bandwidth_gbs=args.peak_bandwidth_gbs,
            max_input_elements=args.max_input_elements,
        )
        plot_outputs = plot_results([compute_output, memory_output], run_dir / "plots")
        print(f"wrote {device_output}")
        print(f"wrote {compute_output}")
        print(f"wrote {memory_output}")
        for output in plot_outputs:
            print(f"wrote {output}")
        return

    if args.command == "plot":
        from hc02.plotting import plot_results

        outputs = plot_results(args.csv, args.out_dir)
        for output in outputs:
            print(f"wrote {output}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
