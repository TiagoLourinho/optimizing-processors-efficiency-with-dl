import argparse
import os
from datetime import datetime

from config import config
from my_lib.benchmark_monitor import BenchmarkMonitor
from my_lib.gpu import GPU
from my_lib.utils import are_there_other_users, collect_system_info, export_data

# Set umask to 000 to allow full read, write, and execute for everyone
# avoiding the normal user not being able to modify the files created
# as this script needs sudo (root) to run
os.umask(0o000)

data: dict = {
    "cuda_file": "",  # The CUDA file profilled
    "config": {},  # The config used to profile
    "system_info": {},  # System information
    "results": {
        "nvml": {},
        "ncu": {},
    },  # The results of the profile (NVML metrics and NCU metrics)
    "nvml_timeline": {},  # The NVML samples collected that form a timeline
}


def collect_cmd_args() -> argparse.Namespace:
    """Collects the commnad line arguments"""

    parser = argparse.ArgumentParser(
        description="Compiles and profiles a CUDA program using NVML and NCU. Requires sudo. Expects a 'config.py' defining the profiling behavior.",
    )

    ######### Required arguments #########

    parser.add_argument(
        "cuda_file", type=str, help="The CUDA program to compile and time."
    )

    ######### Optional arguments #########

    parser.add_argument(
        "-o",
        dest="output_filename",
        type=str,
        default=None,
        help="The output file name to use.",
    )

    return parser.parse_args()


def validate_config(config: dict):
    """Validates the current configuration (not extensively) and raises an error if invalid"""

    ########## Check required keys ##########
    required_keys = {
        "nvcc_path",
        "ncu_path",
        "gpu_sleep_time",
        "nvml_sampling_freq",
        "nvml_n_runs",
    }
    if not all([config[key] is not None for key in required_keys]):
        raise ValueError(
            f"Missing some required keys in config. Required keys: {required_keys}"
        )

    ########## Check types ##########
    allow_none = {
        "gpu_graphics_clk",
        "gpu_memory_clk",
        "ncu_set",
        "ncu_sections",
        "ncu_metrics",
    }
    types = {
        "nvcc_path": str,
        "ncu_path": str,
        "gpu_graphics_clk": int,
        "gpu_memory_clk": int,
        "gpu_sleep_time": int,
        "nvml_sampling_freq": int,
        "nvml_n_runs": int,
        "ncu_set": str,
        "ncu_sections": list,
        "ncu_metrics": list,
    }
    # If the value is None check if it is allowed, otherwise, check if it matches the expected type
    if not all(
        [
            isinstance(value, types[key]) if value is not None else key in allow_none
            for key, value in config.items()
        ]
    ):
        raise ValueError("A config parameter has an invalid type.")

    ########## Check NCU metrics ##########
    ncu_config = {"ncu_set", "ncu_sections", "ncu_metrics"}
    # Only ony "metric type" can be defined
    if not sum([config[key] is not None for key in ncu_config]) == 1:
        raise ValueError(
            "For NCU metrics, use either and only a set, a list of sections or a list of metrics."
        )


def main(data: dict, config: dict):

    # Check if there are other users logged in
    if are_there_other_users():
        print(
            "Other users are already using this machine, stopping script to not interfere."
        )
        return

    validate_config(config)

    # Collect cmd line arguments
    args = collect_cmd_args()
    data["cuda_file"] = args.cuda_file
    data["config"] = config

    print(f"Starting to run the script at {datetime.now()}.")

    # Init the GPU and compile the benchmark
    with GPU(sleep_time=config["gpu_sleep_time"]) as gpu:
        try:
            benchmark_monitor = BenchmarkMonitor(
                benchmark=args.cuda_file,
                gpu=gpu,
                nvcc_path=config["nvcc_path"],
                NVML_N_runs=config["nvml_n_runs"],
                NVML_sampling_frequency=config["nvml_sampling_freq"],
            )

            data["system_info"] = collect_system_info(gpu_name=gpu.name)

            # Set the GPU clocks
            if config["gpu_memory_clk"] is not None:
                gpu.memory_clk = config["gpu_memory_clk"]
            if config["gpu_graphics_clk"] is not None:
                gpu.graphics_clk = config["gpu_graphics_clk"]

            (
                data["results"]["nvml"],
                data["nvml_timeline"],
                figure,
                nvml_did_other_users_login,
            ) = benchmark_monitor.run_and_monitor()

            data["results"]["did_other_users_login"] = nvml_did_other_users_login

            export_data(
                data=data,
                figure=figure,
                benchmark_path=args.cuda_file,
                output_filename=args.output_filename,
            )
        except KeyboardInterrupt:
            print("\nInterrupting...")
            return


if __name__ == "__main__":
    main(data=data, config=config)
