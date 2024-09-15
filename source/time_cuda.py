import argparse
import os
import subprocess
import time

from tqdm import tqdm

BINARY_CUDA_FILE = "temp.out"


def run_command(command: list[str]):
    """Runs a `command` using the subprocess library and checks for errors"""

    try:
        result = subprocess.run(command, check=True, capture_output=True)

        output = result.stdout.decode() + "\n\n" + result.stderr.decode()

        if output:
            print(output)

    except subprocess.CalledProcessError as e:
        print(f"Error while running: {' '.join(command)}\n\n{e.stderr.decode()}")
        exit()


def collect_cmd_args() -> argparse.Namespace:
    """Collects the commnad line arguments"""

    parser = argparse.ArgumentParser(
        description="Compiles and times a CUDA program.",
    )

    parser.add_argument(
        "cuda_file", type=str, help="The CUDA program to compile and time."
    )

    parser.add_argument(
        "-N",
        type=int,
        default=5,
        help="The amount of times to run the program to get the average run time.",
    )

    return parser.parse_args()


def compile(cuda_file: str):
    """Compiles the CUDA program (`cuda_file`)"""

    nvcc_command = ["nvcc", cuda_file, "-o", BINARY_CUDA_FILE]

    run_command(command=nvcc_command)

    print(f"Successfully compiled {cuda_file}.\n")


def run_and_time(N: int) -> float:
    """Runs the the binary CUDA file `N` times and returns the average run time"""

    times = []

    print("Starting to run and timing the CUDA file...\n")

    for _ in tqdm(range(N)):
        start = time.time()

        run_command([f"./{BINARY_CUDA_FILE}"])

        end = time.time()

        times.append(end - start)

    print()

    return sum(times) / N


def cleanup():
    """Deletes temporary files"""

    if os.path.exists(BINARY_CUDA_FILE):
        os.remove(BINARY_CUDA_FILE)


def main():
    try:
        args = collect_cmd_args()

        compile(cuda_file=args.cuda_file)

        average_time = run_and_time(N=args.N)

        print(
            f"{args.cuda_file} had an average run time of {round(average_time, 2)} s across {args.N} runs."
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
