import os
import subprocess
import time

from tqdm import tqdm

BIN_FOLDER = "bin"  # Where to put the cuda binaries
BINARY_CUDA_FILE = "temp.out"  # Name of the cuda binary created


def compile(cuda_file: str, nvcc_path: str):
    """Compiles the CUDA program (`cuda_file`)"""

    if not os.path.exists(BIN_FOLDER):
        os.makedirs(BIN_FOLDER)

    nvcc_command = [
        nvcc_path,
        cuda_file,
        "-o",
        os.path.join(BIN_FOLDER, BINARY_CUDA_FILE),
    ]

    run_command(command=nvcc_command)

    print(f"Successfully compiled {cuda_file}.\n")


def run_command(command: list[str]):
    """Runs a `command` using the subprocess library and checks for errors"""

    try:
        result = subprocess.run(command, check=True, capture_output=True)

        output = result.stdout.decode() + "\n\n" + result.stderr.decode()
        output = output.strip()

        if output:
            print(output)

    except subprocess.CalledProcessError as e:
        print(
            f"Error while running: {' '.join(command)}\n\n{e.stderr.decode().strip()}"
        )
        exit()


def run_and_time(N: int) -> float:
    """Runs the the binary CUDA file `N` times and returns the average run time"""

    times = []

    print("Starting to run and timing the CUDA file...\n")

    for _ in tqdm(range(N)):
        start = time.time()

        run_command([os.path.join(BIN_FOLDER, BINARY_CUDA_FILE)])

        end = time.time()

        times.append(end - start)

    print()

    return sum(times) / N


def cleanup():
    """Deletes temporary files"""

    full_path = os.path.join(BIN_FOLDER, BINARY_CUDA_FILE)

    if os.path.exists(full_path):
        os.remove(full_path)
