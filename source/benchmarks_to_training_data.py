import json
import os
import time
from datetime import datetime
from subprocess import CalledProcessError

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
    reduce_clocks_list,
    validate_config,
)

load_dotenv()

# Use or None to avoid having empty strings if the var isn't defined
paths = {
    "benchmarks_folder": os.getenv("BENCHMARKS_FOLDER") or None,
    "nvcc_path": os.getenv("NVCC_PATH") or None,
    "nsys_path": os.getenv("NSYS_PATH") or None,
    "nsys_set": os.getenv("NSYS_SET") or None,
}

config["benchmarks_to_training_data"].update(paths)

# Set umask to 000 to allow full read, write, and execute for everyone
# avoiding the normal user not being able to modify the files created
# as this script needs sudo (root) to run
os.umask(0o000)

data: dict = {
    "config": {},  # The config used to profile
    "system_info": {},  # System information
    "did_other_users_login": False,  # Whether or not another user logged in during metrics collection
    "models_info": {},  # Extra info needed in the pytorch side to create the models
    "default_freqs": {},  # Keep track of the default frequencies found
    "ptxs": {},  # Contains the PTX of every considered benchmark
    "training_data": [],  # The traning data (each training sample contains the encoded ptx, the frequencies used and the nvml/nsys metrics)
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
    print("\nScript started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Init the GPU and compile the benchmark
    with GPU() as gpu:

        # Find default values
        gpu.reset_graphics_clk()
        gpu.reset_memory_clk()
        time.sleep(1)
        data["default_freqs"]["graphics"] = gpu.graphics_clk
        data["default_freqs"]["memory"] = gpu.memory_clk

        print(
            f'Defaults freqs: Core: {data["default_freqs"]["graphics"]} Mem: {data["default_freqs"]["memory"]}'
        )

        try:
            compiler = Compiler(
                nvcc_path=config["nvcc_path"],
                benchmarks_folder=config["benchmarks_folder"],
            )
            ptx_parser = PTXParser()
            benchmark_monitor = BenchmarkMonitor(
                gpu=gpu,
                nvml_n_runs=config["nvml_n_runs"],
                nvml_sampling_frequency=config["nvml_sampling_freq"],
                nsys_path=config["nsys_path"],
                nsys_set=config["nsys_set"],
            )

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

            sample_mem_clocks = reduce_clocks_list(
                original_clocks=gpu.get_supported_memory_clocks(),
                N=config["n_closest_mem_clocks"],
                default=data["default_freqs"]["memory"],
            )
            print("\nMemory clocks to sample on: ", sample_mem_clocks)
            for memory_clock in sample_mem_clocks:

                sample_core_clocks = reduce_clocks_list(
                    original_clocks=gpu.get_supported_graphics_clocks(
                        memory_clock=memory_clock
                    ),
                    N=config["n_closest_core_clocks"],
                    default=data["default_freqs"]["graphics"],
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
                            # - Either the subprocess will return status 1 while running with NVML (CalledProcessError)
                            # - Or if it still returns status 0, NSYS report won't be created (FileNotFoundError)
                            try:
                                (
                                    nvml_metrics,
                                    _,
                                    nvml_did_other_users_login,
                                ) = benchmark_monitor.run_nvml(
                                    benchmark_path=executable_path, benchmark_args=args
                                )

                                nsys_metrics, nsys_did_other_users_login = (
                                    benchmark_monitor.run_nsys(
                                        benchmark_path=executable_path,
                                        benchmarks_args=args,
                                    )
                                )

                            except (CalledProcessError, FileNotFoundError) as e:
                                print(f"\n{str(e)}")
                                continue

                            # Save results
                            ran_successfully = True

                            data["did_other_users_login"] = (
                                data["did_other_users_login"]
                                or nvml_did_other_users_login
                                or nsys_did_other_users_login
                            )

                            data["training_data"].append(
                                {
                                    "benchmark_name": benchmark_name,
                                    "memory_frequency": memory_clock,
                                    "graphics_frequency": graphics_clock,
                                    "nvml_metrics": nvml_metrics,
                                    "nsys_metrics": nsys_metrics,
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
                                f"\nTime: {now.strftime('%Y-%m-%d %H:%M:%S')} ({int(hours)}h:{int(minutes)}min since starting)\nMemory clk: {memory_clock}Hz | Graphics clk: {graphics_clock}Hz ({skipped_clock_configs} clock configs skipped)\nBenchmark: {benchmark_name} ({skipped_benchmarks}/{total_compiled_benchmarks} skipped)\nCollected samples: {len(data['training_data'])}"
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

                            ptx_path = os.path.join(PTX_PATH, f"{benchmark_name}.ptx")
                            os.remove(ptx_path)

                            del data["ptxs"][benchmark_name]

                            skipped_benchmarks += 1

            if data["did_other_users_login"]:
                print(
                    "\nWarning: Other users logged in during execution of the script. Script might need to run again.\n"
                )

            data["system_info"]["duration"] = f"{int(hours)}h:{int(minutes)}min"
            data["models_info"] = EncodedInstruction.get_enconding_info()
            data["models_info"]["n_nsys_metrics"] = len(
                data["training_data"][0]["nsys_metrics"]
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
