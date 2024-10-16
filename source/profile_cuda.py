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

    get_key_config_dict = lambda required, type: tuple([required, type])

    keys_config = {
        "nvcc_path": get_key_config_dict(required=True, type=str),
        "ncu_path": get_key_config_dict(required=True, type=str),
        "ncu_sections_folder": get_key_config_dict(required=False, type=str),
        "gpu_graphics_clk": get_key_config_dict(required=False, type=int),
        "gpu_memory_clk": get_key_config_dict(required=False, type=int),
        "gpu_sleep_time": get_key_config_dict(required=True, type=int),
        "nvml_sampling_freq": get_key_config_dict(required=True, type=int),
        "nvml_n_runs": get_key_config_dict(required=True, type=int),
        "ncu_set": get_key_config_dict(required=True, type=str),
    }

    if len(keys_config) != len(config):
        raise ValueError("Config has too many / too few parameters.")

    for key, (required, type) in keys_config.items():

        if key not in config:
            raise ValueError(f"Missing config key {key}.")

        value = config[key]

        if value is None:
            if required:
                raise ValueError(f"Key {key} should be defined.")
        elif not isinstance(value, type):
            raise ValueError(f"Key {key} has invalid type.")


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
