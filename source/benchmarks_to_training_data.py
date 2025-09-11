import json
import os
import time
from datetime import datetime
from subprocess import CalledProcessError, TimeoutExpired

from config import config
from dotenv import load_dotenv
from my_lib.benchmark_monitor import BenchmarkMonitor
from my_lib.compiler import Compiler
from my_lib.encoded_instruction import EncodedInstruction
from my_lib.gpu import GPU, GPUClockChangingError
from my_lib.PTX_parser import PTXParser
from my_lib.utils import (
    are_there_other_users,
    collect_system_info,
    approximate_linear_space,
    validate_config,
)

load_dotenv()

# Use or None to avoid having empty strings if the var isn't defined
paths = {
    "benchmarks_folder": os.getenv("BENCHMARKS_FOLDER") or None,
    "nvcc_path": os.getenv("NVCC_PATH") or None,
}

config["benchmarks_to_training_data"].update(paths)

# Set umask to 000 to allow full read, write, and execute for everyone
# avoiding the normal user not being able to modify the files created
# as this script needs sudo (root) to run
os.umask(0o000)

data: dict = {
    "config": {},  # The config used to profile
    "system_info": {},  # System information
    "models_info": {},  # Extra info needed in the pytorch side to create the models
    "ptxs": {},  # Contains the PTX of every considered benchmark
    "training_data": [],  # The traning data (each training sample contains the encoded ptx, the frequencies used and the nvml/cupti metrics)
}

PTX_PATH = "bin/ptx"
EXECUTABLES_PATH = "bin/executables"

BENCHMARK_ARGS_TO_TEST = [
    None,  # First try the default invocation
    [
        "128",
        "256",
        "1000",
        "32",
    ],  # Usage ./binary <num_blocks> <num_threads_per_block> <iterations>threads active per warp
    [
        "128",
        "256",
        "1000",
        "32",
        "1",
    ],  # Some benchmarks give the error message as the previous one but still expect one extra argument for the stride
]


def main(data: dict, config: dict):

    # Check if there are other users logged in
    if are_there_other_users():
        print(
            "Other users are already using this machine, stopping script to not interfere."
        )
        return

    validate_config(config)

    # Collect cmd line arguments
    data["config"] = config

    start_time = datetime.now()
    print("Script started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Init the GPU and compile the benchmark
    with GPU() as gpu:

        # Init
        gpu.reset_graphics_clk()
        gpu.reset_memory_clk()
        time.sleep(1)

        compiler = Compiler(
            nvcc_path=config["nvcc_path"],
            benchmarks_folder=config["benchmarks_folder"],
        )
        ptx_parser = PTXParser()
        with BenchmarkMonitor(
            gpu=gpu,
            sampling_frequency=config["sampling_freq"],
            n_runs=config["n_runs"],
            timeout_seconds=config["timeout_seconds"],
        ) as benchmark_monitor:

            try:

                data["system_info"] = collect_system_info(gpu_name=gpu.name)

                compiler.compile_and_obtain_ptx()

                # Convert the PTX to a sequence of vectors
                for ptx_file in os.listdir(PTX_PATH):
                    benchmark_name = ptx_file.replace(".ptx", "")

                    data["ptxs"][benchmark_name] = ptx_parser.parse(
                        os.path.join(PTX_PATH, ptx_file), convert_to_dicts=True
                    )

                total_compiled_benchmarks = len(os.listdir(EXECUTABLES_PATH))
                skipped_benchmarks = 0
                skipped_clock_configs = 0

                sample_mem_clocks = approximate_linear_space(
                    values=gpu.get_supported_memory_clocks(),
                    min_val=config["memory_levels"]["min"],
                    max_val=config["memory_levels"]["max"],
                    count=config["memory_levels"]["count"],
                )
                print("\nMemory clocks to sample on: ", sample_mem_clocks)
                for memory_clock in sample_mem_clocks:

                    sample_core_clocks = approximate_linear_space(
                        values=gpu.get_supported_graphics_clocks(
                            memory_clock=memory_clock
                        ),
                        min_val=config["graphics_levels"]["min"],
                        max_val=config["graphics_levels"]["max"],
                        count=config["graphics_levels"]["count"],
                    )

                    n_total_approximated_samples = (
                        len(sample_core_clocks)
                        * len(sample_mem_clocks)
                        * total_compiled_benchmarks
                    )

                    print("\nCore clocks to sample on: ", sample_core_clocks)
                    for graphics_clock in sample_core_clocks:
                        try:
                            gpu.memory_clk = memory_clock
                            gpu.graphics_clk = graphics_clock

                            # Sometimes the driver changes the graphics clock after changing the graphics clock
                            assert gpu.memory_clk == memory_clock
                            assert gpu.graphics_clk == graphics_clock
                        except (GPUClockChangingError, AssertionError):

                            # The driver sometimes doesn't let the GPU change to frequencies too high so just skip them
                            print(
                                f"\nCouldn't change to memory_clk={memory_clock} and graphics_clock={graphics_clock}, skipping."
                            )
                            skipped_clock_configs += 1
                            continue

                        for executable in os.listdir(EXECUTABLES_PATH):

                            benchmark_name = executable.replace(".out", "")
                            executable_path = os.path.join(EXECUTABLES_PATH, executable)

                            ran_successfully = False
                            for args in BENCHMARK_ARGS_TO_TEST:

                                print(f"\nRunning {benchmark_name} with args={args}...")

                                # Some benchmarks require extra arguments like files and etc that aren't provided, so:
                                # - The subprocess will return status 1 while running with NVML (CalledProcessError)
                                try:
                                    benchmark_metrics = benchmark_monitor.run_benchmark(
                                        benchmark_path=executable_path,
                                        benchmark_args=args,
                                    )

                                except TimeoutExpired as e:
                                    print(f"\n{str(e)}")
                                    break  # No need to try other args if it timed out

                                except CalledProcessError as e:
                                    print(f"\n{str(e)}")
                                    continue

                                # Save results
                                ran_successfully = True

                                data["training_data"].append(
                                    {
                                        "benchmark_name": benchmark_name,
                                        "memory_frequency": memory_clock,
                                        "graphics_frequency": graphics_clock,
                                        "benchmark_metrics": benchmark_metrics,
                                    }
                                )

                                # Print current status
                                now = datetime.now()
                                elapsed_time = now - start_time
                                hours, remainder = divmod(
                                    elapsed_time.total_seconds(), 3600
                                )
                                minutes, _ = divmod(remainder, 60)
                                print(
                                    f"\nTime: {now.strftime('%Y-%m-%d %H:%M:%S')} ({int(hours)}h:{int(minutes)}min since starting)\nMemory clk: {memory_clock}Hz | Graphics clk: {graphics_clock}Hz ({skipped_clock_configs} clock configs skipped)\nBenchmark: {benchmark_name} ({skipped_benchmarks}/{total_compiled_benchmarks} skipped)\nCollected samples: {len(data['training_data'])} (~{round(100*(len(data['training_data'])/n_total_approximated_samples))}%)"
                                )

                                break  # If these args worked, no need to test the following

                            # Delete the benchmark as it can't be profilled with any of the available args
                            if not ran_successfully:
                                print(
                                    f"\nSkipping {benchmark_name} as it couldn't be run with any of the available arguments..."
                                )
                                executable_path = os.path.join(
                                    EXECUTABLES_PATH, f"{benchmark_name}.out"
                                )
                                os.remove(executable_path)

                                ptx_path = os.path.join(
                                    PTX_PATH, f"{benchmark_name}.ptx"
                                )
                                os.remove(ptx_path)

                                del data["ptxs"][benchmark_name]

                                skipped_benchmarks += 1

                data["system_info"]["duration"] = f"{int(hours)}h:{int(minutes)}min"
                data["models_info"] = EncodedInstruction.get_enconding_info()
                data["models_info"]["n_cupti_metrics"] = len(
                    data["training_data"][0]["benchmark_metrics"]["cupti_metrics"]
                )
                data["models_info"]["n_nvml_metrics"] = len(
                    data["training_data"][0]["benchmark_metrics"]["nvml_metrics"]
                )

                training_data_file = "training_data.json"
                with open(training_data_file, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                    print(f"\nExported training data to {training_data_file}.")

            except KeyboardInterrupt:
                print("\nInterrupting...")
                return


if __name__ == "__main__":
    main(data=data, config=config["benchmarks_to_training_data"])
