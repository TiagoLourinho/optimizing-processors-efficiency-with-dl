import socket
from datetime import datetime

import cpuinfo
import psutil


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
        "gpu_sleep_time": get_key_config_dict(required=True, type=float),
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


def get_nvml_scaling_factors_and_update_baselines(
    benchmark_name: str, nvml_metrics: dict, baselines: dict
):
    """
    Given the nvml_metrics, it calculates the scaling factors with the
    reference being obtained at the maximum frequencies possible

    Warning: Assumes that when the baselines aren't found on the dict,
    it means that the given nvml_metrics are the baselines
    """

    # If there isn't anything on the baselines, then this means that
    # these nvml metrics are the baseline, so update the dict
    if benchmark_name not in baselines:
        scaling_factors = {"runtime_scaling_factor": 1, "power_scaling_factor": 1}

        baselines[benchmark_name] = nvml_metrics
    else:
        power_baseline = baselines[benchmark_name]["average_POWER"]
        runtime_baseline = baselines[benchmark_name]["median_run_time"]

        scaling_factors = {
            "runtime_scaling_factor": nvml_metrics["median_run_time"]
            / runtime_baseline,
            "power_scaling_factor": nvml_metrics["average_POWER"] / power_baseline,
        }

    return scaling_factors
