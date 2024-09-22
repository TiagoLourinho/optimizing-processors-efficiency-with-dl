import argparse

from my_lib.gpu import GPU
from my_lib.utils import cleanup, run_and_time, compile


def collect_cmd_args() -> argparse.Namespace:
    """Collects the commnad line arguments"""

    parser = argparse.ArgumentParser(
        description="Compiles and times a CUDA program. Requires sudo to manage the GPU clocks.",
    )

    parser.add_argument(
        "cuda_file", type=str, help="The CUDA program to compile and time."
    )

    parser.add_argument(
        "--nvcc",
        type=str,
        required=True,
        help="The path of the nvcc compiler to be used while running in sudo. '$(which nvcc)' can be used to make it easier.",
    )

    parser.add_argument(
        "-N",
        type=int,
        default=5,
        help="The amount of times to run the program to get the average run time.",
    )

    parser.add_argument(
        "--graphics-clk",
        type=int,
        default=None,
        help="The graphics clock to use.",
    )

    parser.add_argument(
        "--memory-clk",
        type=int,
        default=None,
        help="The memory clock to use.",
    )

    parser.add_argument(
        "--sleep-time",
        type=float,
        default=1,
        help="The amount of seconds to sleep after changing GPU clocks speeds so the systems stabilizes.",
    )

    return parser.parse_args()


def main():
    try:
        args = collect_cmd_args()

        gpu = GPU(sleep_time=args.sleep_time)

        # Lock the given clocks. If None just lock with the values the GPU was already using (default)
        gpu.memory_clk = (
            args.memory_clk if args.memory_clk is not None else gpu.memory_clk
        )
        gpu.graphics_clk = (
            args.graphics_clk if args.graphics_clk is not None else gpu.graphics_clk
        )

        compile(cuda_file=args.cuda_file, nvcc_path=args.nvcc)

        average_time = run_and_time(N=args.N)

        print(
            f"{args.cuda_file} had an average run time of {round(average_time, 2)}s across {args.N} runs."
        )
        print()
        print(
            f"The GPU was operating with memory_clk={gpu.memory_clk}MHz and graphics_clk={gpu.graphics_clk}MHZ."
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
