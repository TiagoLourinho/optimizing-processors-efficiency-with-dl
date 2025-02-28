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

    # This script needs sudo, so the current user actually counts as 2 online users
    max_len = 2

    # NCU spawns another user "root", so account for that
    if running_ncu:
        max_len += 1

    return len(psutil.users()) > max_len


def validate_config(config: dict):
    """Validates the current configuration (not extensively) and raises an error if invalid"""

    get_key_config_dict = lambda required, type: tuple([required, type])

    keys_config = {
        "benchmarks_folder": get_key_config_dict(required=True, type=str),
        "nvcc_path": get_key_config_dict(required=True, type=str),
        "ncu_path": get_key_config_dict(required=True, type=str),
        "ncu_sections_folder": get_key_config_dict(required=False, type=str),
        "ncu_python_report_folder": get_key_config_dict(required=True, type=str),
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
