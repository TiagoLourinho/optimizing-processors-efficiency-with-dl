import socket
from datetime import datetime

import cpuinfo
import numpy as np
import psutil


def approximate_linear_space(
    values: list[int], min_val: int, max_val: int, count: int
) -> list[int]:
    """Finds a set of `count` values that are approximately evenly spaced"""

    # Ensure min_val and max_val are within the range of values
    min_val = max(min_val, min(values))
    max_val = min(max_val, max(values))

    # Filter values within the specified range and remove duplicates
    filtered = sorted(set([v for v in values if min_val <= v <= max_val]))

    # If count is greater than the available numbers, return only the available ones
    if count >= len(filtered):
        return filtered

    # Use numpy to generate linearly spaced positions
    ideal_positions = np.linspace(min_val, max_val, count)

    result = []
    for target in ideal_positions:
        # Find the closest value in filtered values
        closest_value = min(filtered, key=lambda x: abs(x - target))
        result.append(closest_value)
        filtered.remove(closest_value)  # Remove to avoid duplicates

    return sorted(result)


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


def are_there_other_users() -> bool:
    """Checks if there are other users using the machine"""

    usernames = [user.name for user in psutil.users()]

    # Using tmux no users appear online
    if len(usernames) == 0:
        return False

    # Check if all user instances belong to the same user
    first_user = usernames[0]
    for user in usernames:
        if not user == first_user:
            return True

    return False


def validate_config(config: dict):
    """Validates the current configuration (not extensively) and raises an error if invalid"""

    get_key_config_dict = lambda required, type: tuple([required, type])

    keys_config = {
        "benchmarks_folder": get_key_config_dict(required=True, type=str),
        "nvcc_path": get_key_config_dict(required=True, type=str),
        "graphics_levels": get_key_config_dict(required=True, type=dict),
        "memory_levels": get_key_config_dict(required=True, type=dict),
        "sampling_freq": get_key_config_dict(required=True, type=int),
        "n_runs": get_key_config_dict(required=True, type=int),
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


def closest_value(numbers: list[int], target: int) -> int:
    if not numbers:
        return None
    return min(numbers, key=lambda x: abs(x - target))


def get_ed2p(power: float, runtime: float):
    # ED²P = E * D² = P * D * D² = P * D³
    return power * (runtime**3)


def get_edp(power: float, runtime: float):
    # EDP = E * D = P * D * D = P * D²
    return power * (runtime**2)
