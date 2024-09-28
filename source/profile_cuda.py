import argparse

from my_lib.benchmark_monitor import BenchmarkMonitor
from my_lib.gpu import GPU
from my_lib.utils import collect_system_info, export_data

data: dict = {
    "arguments": {},  # The command line arguments given
    "system_info": {},  # System information
    "results_summary": {},  # The most important results
    "timeline": {},  # The sampled results across time
}


def collect_cmd_args() -> argparse.Namespace:
    """Collects the commnad line arguments"""

    parser = argparse.ArgumentParser(
        description="Compiles and times a CUDA program. Requires sudo to manage the GPU clocks.",
    )

    parser.add_argument(
        "cuda_file", type=str, help="The CUDA program to compile and time."
    )

    parser.add_argument(
        "--nvcc",
        type=str,
        required=True,
        help="The path of the nvcc compiler to be used while running in sudo. '$(which nvcc)' can be used to make it easier.",
    )

    parser.add_argument(
        "-N",
        type=int,
        default=5,
        help="The amount of times to run the program to get the average run time.",
    )

    parser.add_argument(
        "--graphics-clk",
        type=int,
        default=None,
        help="The graphics clock to use in MHz.",
    )

    parser.add_argument(
        "--memory-clk",
        type=int,
        default=None,
        help="The memory clock to use in MHz.",
    )

    parser.add_argument(
        "--sleep-time",
        type=float,
        default=1,
        help="The amount of seconds to sleep after changing GPU clocks speeds so the systems stabilizes.",
    )

    return parser.parse_args()


def main(data: dict):
    try:
        # Collect cmd line arguments
        args = collect_cmd_args()
        data["arguments"] = vars(args)

        # Init the GPU and compile the benchmark
        gpu = GPU(sleep_time=args.sleep_time)
        benchmark_monitor = BenchmarkMonitor(
            benchmark=args.cuda_file, gpu=gpu, nvcc_path=args.nvcc, N_runs=args.N
        )

        data["system_info"] = collect_system_info(gpu_name=gpu.name)

        # Set the GPU clocks
        if args.memory_clk is not None:
            gpu.memory_clk = args.memory_clk
        if args.graphics_clk is not None:
            gpu.graphics_clk = args.graphics_clk

        benchmark_monitor.run_and_monitor()

        export_data(data=data, benchmark_name=args.cuda_file)
    finally:
        del gpu


if __name__ == "__main__":
    main(data=data)
