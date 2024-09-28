import json
import os
import socket
from datetime import datetime

import cpuinfo
import psutil


def export_data(data: dict, benchmark_name: str):
    """Writes the collected data to a JSON file"""

    results_folder = "results"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    full_path = os.path.join(
        results_folder, os.path.basename(benchmark_name).replace(".cu", ".json")
    )

    with open(full_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


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
