import json
import os
import shutil
import socket
from datetime import datetime
from typing import Any

import cpuinfo
import psutil

MatplotlibFigure = Any
""" 
Placeholder for the matplotlib figure class (not direclty imported because of C++ library incompatibilities)

Check: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
"""


def export_data(
    data: dict,
    figure: MatplotlibFigure,
    benchmark_path: str,
    output_filename: str | None,
):
    """Writes the collected data to a JSON file"""

    results_folder = "results"
    extension = os.path.splitext(benchmark_path)[1]
    benchmark_filename = os.path.basename(benchmark_path).removesuffix(extension)

    if output_filename is None:
        output_filename = benchmark_filename

    benchmark_folder = os.path.join(results_folder, output_filename)

    # Create folders if they don't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(benchmark_folder):
        os.makedirs(benchmark_folder)

    # Keep a copy of the original CUDA file and set default permissions for all users
    os.chmod(
        shutil.copy2(
            benchmark_path,
            os.path.join(benchmark_folder, f"copy_{output_filename}{extension}"),
        ),
        0o666,
    )

    # Execution plots
    figure.savefig(os.path.join(benchmark_folder, f"plots_{output_filename}.png"))

    # Results JSON
    full_path = os.path.join(benchmark_folder, f"results_{output_filename}.json")
    with open(full_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Results were saved in {benchmark_folder} folder.")


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
