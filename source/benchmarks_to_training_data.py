import json
import os
import sys
import time
from datetime import datetime
from subprocess import CalledProcessError

from config import config
from dotenv import load_dotenv

# The ncu_report C++ libraries throw the following error
# when imported dynamically during BenchmarkMonitor initialization:
#
# terminate called after throwing an instance of 'std::bad_cast'
# what():  std::bad_cast
#
# To prevent this, import them at the beginning
sys.path.append(config["benchmarks_to_training_data"]["ncu_python_report_folder"])
from datetime import datetime

import ncu_report  # type: ignore
from my_lib.benchmark_monitor import BenchmarkMonitor
from my_lib.compiler import Compiler
from my_lib.encoded_instruction import EncodedInstruction
from my_lib.gpu import GPU, GPUClockChangingError
from my_lib.PTX_parser import PTXParser
from my_lib.utils import (
    are_there_other_users,
    collect_system_info,
    get_nvml_scaling_factors_and_update_baselines,
    reduce_clocks_list,
    validate_config,
)

# Set umask to 000 to allow full read, write, and execute for everyone
# avoiding the normal user not being able to modify the files created
# as this script needs sudo (root) to run
os.umask(0o000)

data: dict = {
    "config": {},  # The config used to profile
    "system_info": {},  # System information
    "did_other_users_login": False,  # Whether or not another user logged in during metrics collection
    "models_info": {},  # Extra info needed in the pytorch side to create the models
    "ptxs": {},  # Contains the PTX of every considered benchmark
    "training_data": [],  # The traning data (each training sample contains the encoded ptx, the frequencies used and the nvml/ncu metrics)
}

PTX_PATH = "bin/ptx"
EXECUTABLES_PATH = "bin/executables"


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
    with GPU(sleep_time=config["gpu_sleep_time"]) as gpu:
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
                ncu_path=config["ncu_path"],
                ncu_sections_folder=config["ncu_sections_folder"],
                ncu_report_lib=ncu_report,
                ncu_set=config["ncu_set"],
            )

            data["system_info"] = collect_system_info(gpu_name=gpu.name)

            compiler.compile_and_obtain_ptx()

            # Convert the PTX to a sequence of vectors
            for ptx_file in os.listdir(PTX_PATH):
                benchmark_name = ptx_file.replace(".ptx", "")

                data["ptxs"][benchmark_name] = ptx_parser.parse(
                    os.path.join(PTX_PATH, ptx_file), convert_to_dicts=True
                )

            # For each combination of graphics and memory clocks, run all benchmarks and collect the metrics
            # Start by the higher frequencies so that the baselines can be collected
            baselines = (
                dict()
            )  # For each benchmark, contains the reference runtime and average power
            total_compiled_benchmarks = len(os.listdir(EXECUTABLES_PATH))
            skipped_benchmarks = 0
            skipped_clock_configs = 0
            for memory_clock in sorted(gpu.get_supported_memory_clocks(), reverse=True):
                for graphics_clock in reduce_clocks_list(
                    gpu.get_supported_graphics_clocks(memory_clock=memory_clock),
                    N=config["n_core_clocks"],
                    reverse=True,
                ):
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

                        print(f"\nRunning {benchmark_name}...")

                        # Some benchmarks require extra arguments like files and etc that aren't provided, so:
                        # - Either the subprocess will return status 1 while running with NVML (CalledProcessError)
                        # - Or if it still returns status 0, NCU will produce a warning saying that no kernels were profilled
                        #   and the NCU report won't be created (FileNotFoundError)
                        try:
                            (
                                nvml_metrics,
                                _,
                                nvml_did_other_users_login,
                            ) = benchmark_monitor.run_nvml(
                                benchmark_path=executable_path
                            )

                            ncu_metrics, ncu_did_other_users_login = (
                                benchmark_monitor.run_ncu(
                                    benchmark_path=executable_path
                                )
                            )
                        except (CalledProcessError, FileNotFoundError) as e:
                            # Delete the benchmark as it can't be profilled
                            print(f"\nSkipping {benchmark_name}:\n{str(e)}")
                            executable_path = os.path.join(
                                EXECUTABLES_PATH, f"{benchmark_name}.out"
                            )
                            os.remove(executable_path)

                            ptx_path = os.path.join(PTX_PATH, f"{benchmark_name}.ptx")
                            os.remove(ptx_path)

                            del data["ptxs"][benchmark_name]

                            time.sleep(gpu.sleep_time)
                            skipped_benchmarks += 1
                            continue

                        data["did_other_users_login"] = (
                            data["did_other_users_login"]
                            or nvml_did_other_users_login
                            or ncu_did_other_users_login
                        )

                        data["training_data"].append(
                            {
                                "benchmark_name": benchmark_name,
                                "memory_frequency": memory_clock,
                                "graphics_frequency": graphics_clock,
                                "nvml_metrics": nvml_metrics,
                                "nvml_scaling_factors": get_nvml_scaling_factors_and_update_baselines(
                                    benchmark_name=benchmark_name,
                                    nvml_metrics=nvml_metrics,
                                    baselines=baselines,
                                ),
                                "ncu_metrics": ncu_metrics,
                            }
                        )

                        now = datetime.now()
                        elapsed_time = now - start_time
                        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                        minutes, _ = divmod(remainder, 60)

                        print(
                            f"\nTime: {now.strftime('%Y-%m-%d %H:%M:%S')} ({int(hours)}h:{int(minutes)}min since starting)\nMemory clk: {memory_clock}Hz | Graphics clk: {graphics_clock}Hz ({skipped_clock_configs} clock configs skipped)\nBenchmark: {benchmark_name} ({skipped_benchmarks}/{total_compiled_benchmarks} skipped)\nCollected samples: {len(data['training_data'])}"
                        )

                        time.sleep(gpu.sleep_time)

            if data["did_other_users_login"]:
                print(
                    "\nWarning: Other users logged in during execution of the script. Script might need to run again.\n"
                )

            data["models_info"] = EncodedInstruction.get_enconding_info()
            data["models_info"]["n_ncu_metrics"] = len(
                data["training_data"][0]["ncu_metrics"]
            )

            training_data_file = "training_data.json"
            with open(training_data_file, "w") as json_file:
                json.dump(data, json_file, indent=4)
                print(f"\nExported training data to {training_data_file}.")

        except KeyboardInterrupt:
            print("\nInterrupting...")
            return


if __name__ == "__main__":
    load_dotenv()

    paths = {
        "benchmarks_folder": os.getenv("BENCHMARKS_FOLDER"),
        "nvcc_path": os.getenv("NVCC_PATH"),
        "ncu_path": os.getenv("NCU_PATH"),
        "ncu_sections_folder": os.getenv("NCU_SECTIONS_FOLDER"),
        "ncu_python_report_folder": os.getenv("NCU_PYTHON_REPORT_FOLDER"),
    }

    config["benchmarks_to_training_data"].update(paths)

    main(data=data, config=config["benchmarks_to_training_data"])
