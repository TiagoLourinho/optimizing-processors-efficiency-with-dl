import socket
from datetime import datetime
import numpy as np

import cpuinfo
import psutil


def reduce_clocks_list(original_clocks: list[int], N: int, default: int):
    """
    Given a list of clocks, returns a reduced list with the N closest clocks to the default value.

    If N > len(original_clocks), then all the clocks are returned.
    """

    # Sort the clocks by their absolute distance to the default
    sorted_clocks = sorted(original_clocks, key=lambda x: abs(x - default))

    # Return the first N clocks (or all of them if N is too large)
    reduced = sorted_clocks[:N]

    return sorted(reduced)


def collect_system_info(gpu_name: str) -> dict:
    """Collects system information"""

    machine = socket.gethostname()

    cpu = cpuinfo.get_cpu_info()["brand_raw"]

    ram = psutil.virtual_memory().total / 1024**3  # To GB

    time = str(datetime.now())

    return {
        "machine": machine,
        "cpu": cpu,
        "gpu": gpu_name,
        "ram": round(ram),
        "run_start_time": time,
    }


def are_there_other_users(running_ncu=False) -> bool:
    """Checks if there are other users using the machine"""

    usernames = [user.name for user in psutil.users()]

    # Using tmux no users appear online
    if len(usernames) == 0:
        return False

    # Check if all user instances belong to the same user
    first_user = usernames[0]
    for user in usernames:
        # Only 2 allowed options:
        # - All users are the same
        # - NCU is running and spawns a root user so also allow that
        if not (user == first_user or (running_ncu and user == "root")):
            return True

    return False


def validate_config(config: dict):
    """Validates the current configuration (not extensively) and raises an error if invalid"""

    get_key_config_dict = lambda required, type: tuple([required, type])

    keys_config = {
        "benchmarks_folder": get_key_config_dict(required=True, type=str),
        "nvcc_path": get_key_config_dict(required=True, type=str),
        "ncu_path": get_key_config_dict(required=True, type=str),
        "ncu_sections_folder": get_key_config_dict(required=False, type=str),
        "ncu_python_report_folder": get_key_config_dict(required=True, type=str),
        "n_closest_core_clocks": get_key_config_dict(required=True, type=int),
        "n_closest_mem_clocks": get_key_config_dict(required=True, type=int),
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


def calculate_nvml_scaling_factors(sample: dict, baselines: dict):
    """
    Given the nvml_metrics, it calculates the scaling factors with the
    reference being obtained at the default frequencies
    """

    assert (
        sample["benchmark_name"] in baselines
    ), f'Baseline for {sample["benchmark_name"]} not found'

    power_baseline = baselines[sample["benchmark_name"]]["average_POWER"]
    runtime_baseline = baselines[sample["benchmark_name"]]["median_run_time"]

    scaling_factors = {
        "runtime_scaling_factor": sample["nvml_metrics"]["median_run_time"]
        / runtime_baseline,
        "power_scaling_factor": sample["nvml_metrics"]["average_POWER"]
        / power_baseline,
    }

    return scaling_factors
