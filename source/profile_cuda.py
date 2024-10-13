import argparse
import os
import sys
from datetime import datetime

from my_lib.benchmark_monitor import BenchmarkMonitor
from my_lib.gpu import GPU
from my_lib.utils import collect_system_info, export_data, are_there_other_users

# Set umask to 000 to allow full read, write, and execute for everyone
# avoiding the normal user not being able to modify the files created
# as this script needs sudo (root) to run
os.umask(0o000)

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

    ######### Required arguments #########

    parser.add_argument(
        "cuda_file", type=str, help="The CUDA program to compile and time."
    )

    parser.add_argument(
        "--nvcc",
        type=str,
        required=True,
        help="The path of the nvcc compiler to be used while running in sudo. '$(which nvcc)' can be used to make it easier.",
    )

    ######### Optional arguments #########

    parser.add_argument(
        "--nvml-sampling-freq",
        type=int,
        default=10,
        help="The NVML sampling frequency to use [Hz]",
    )

    parser.add_argument(
        "--N-runs",
        type=int,
        default=1,
        help="The amount of times to run the program to get the median GPU metrics",
    )

    parser.add_argument(
        "--graphics-clk",
        type=int,
        default=None,
        help="The graphics clock to use [MHz]",
    )

    parser.add_argument(
        "--memory-clk",
        type=int,
        default=None,
        help="The memory clock to use [MHz]",
    )

    parser.add_argument(
        "--sleep-time",
        type=float,
        default=1,
        help="The amount of time to sleep after changing GPU clocks speeds so the systems stabilizes [s]",
    )

    return parser.parse_args()


def main(data: dict):

    # Check if there are other users logged in
    if are_there_other_users():
        print(
            "Other users are already using this machine, stopping script to not interfere."
        )
        return

    # Collect cmd line arguments
    args = collect_cmd_args()
    data["arguments"] = vars(args)
    data["arguments"][
        "full_command"
    ] = f"sudo -E pipenv run python3 {' '.join(sys.argv)}"

    print(f"Starting to run the script at {datetime.now()}.")

    # Init the GPU and compile the benchmark
    with GPU(sleep_time=args.sleep_time) as gpu:
        try:
            benchmark_monitor = BenchmarkMonitor(
                benchmark=args.cuda_file,
                gpu=gpu,
                nvcc_path=args.nvcc,
                N_runs=args.N_runs,
                NVML_sampling_frequency=args.nvml_sampling_freq,
            )

            data["system_info"] = collect_system_info(gpu_name=gpu.name)

            # Set the GPU clocks
            if args.memory_clk is not None:
                gpu.memory_clk = args.memory_clk
            if args.graphics_clk is not None:
                gpu.graphics_clk = args.graphics_clk

            data["results_summary"], data["timeline"], figure = (
                benchmark_monitor.run_and_monitor()
            )

            export_data(data=data, figure=figure, benchmark_path=args.cuda_file)
        except KeyboardInterrupt:
            print("\nInterrupting...")
            return


if __name__ == "__main__":
    main(data=data)
