import json
import os
from datetime import datetime
import time

from config import config
from my_lib.benchmark_monitor import BenchmarkMonitor
from my_lib.compiler import Compiler
from my_lib.gpu import GPU
from my_lib.PTX_parser import PTXParser
from my_lib.utils import are_there_other_users, collect_system_info, validate_config
from my_lib.encoded_instruction import EncodedInstruction

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

    print(f"Starting to run the script at {datetime.now()}.")

    # Init the GPU and compile the benchmark
    with GPU(sleep_time=config["gpu_sleep_time"]) as gpu:
        try:
            compiler = Compiler(benchmarks_folder=config["benchmarks_folder"])
            ptx_parser = PTXParser()
            benchmark_monitor = BenchmarkMonitor(
                gpu=gpu,
                nvcc_path=config["nvcc_path"],
                nvml_n_runs=config["nvml_n_runs"],
                nvml_sampling_frequency=config["nvml_sampling_freq"],
                ncu_path=config["ncu_path"],
                ncu_sections_folder=config["ncu_sections_folder"],
                ncu_python_report_folder=config["ncu_python_report_folder"],
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
            for memory_clock in gpu.get_supported_memory_clocks():
                for graphics_clock in gpu.get_supported_graphics_clocks(
                    memory_clock=memory_clock
                ):
                    for executable in os.listdir(EXECUTABLES_PATH):

                        benchmark_name = executable.replace(".out", "")
                        executable_path = os.path.join(EXECUTABLES_PATH, executable)

                        (
                            nvml_metrics,
                            _,
                            _,
                            nvml_did_other_users_login,
                        ) = benchmark_monitor.run_nvml(benchmark_path=executable_path)

                        # Some benchmarks require extra arguments like files and etc
                        # So NCU will produce a warning saying that no kernels were profilled
                        # And the NCU report won't be created resulting in the FileNotFoundError
                        try:
                            ncu_metrics, ncu_did_other_users_login = (
                                benchmark_monitor.run_ncu(
                                    benchmark_path=executable_path
                                )
                            )
                        except FileNotFoundError:
                            # Delete the benchmark as it can't be profilled
                            print(
                                f"Removing {benchmark_name} as no kernels were profilled."
                            )
                            executable_path = os.path.join(
                                EXECUTABLES_PATH, f"{benchmark_name}.out"
                            )
                            os.remove(executable_path)

                            ptx_path = os.path.join(PTX_PATH, f"{benchmark_name}.ptx")
                            os.remove(ptx_path)

                            del data["ptxs"][benchmark_name]

                            time.sleep(gpu.sleep_time)
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
                                "ncu_metrics": ncu_metrics,
                            }
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
    main(data=data, config=config)
