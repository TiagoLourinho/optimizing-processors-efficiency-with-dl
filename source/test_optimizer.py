import subprocess
import os
import argparse

N_RUNS_PER_BENCHMARK = 3

EXECUTABLES_PATH = "tests/executables/"
PTX_PATH = "tests/ptx/"

BENCHMARK_ARGS = [
    "128",
    "256",
    "100000",
    "32",
    "1",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python_path",
        required=True,
        help="Path to the Python interpreter used to run optimizer.py",
    )
    args = parser.parse_args()

    python_executable = args.python_path

    for benchmark in sorted(os.listdir(EXECUTABLES_PATH)):
        for baseline in [True, False]:
            for run_idx in range(N_RUNS_PER_BENCHMARK):
                print(
                    f"===== Running benchmark: {benchmark}, "
                    f"{'Baseline' if baseline else 'Controlled'}, "
                    f"Run: {run_idx + 1}/{N_RUNS_PER_BENCHMARK} =====\n"
                )

                command = [
                    "sudo",
                    python_executable,
                    "source/optimizer.py",
                    "--ptx_path",
                    os.path.join(PTX_PATH, benchmark.replace(".out", ".ptx")),
                    "--exec_path",
                    os.path.join(EXECUTABLES_PATH, benchmark),
                ]

                if baseline:
                    command.append("--get_baseline")

                # Everything after "--" is passed to the benchmark
                command.append("--")
                command.extend(BENCHMARK_ARGS)

                try:
                    subprocess.run(command)
                except KeyboardInterrupt:
                    return  # Stop testing


if __name__ == "__main__":
    main()
