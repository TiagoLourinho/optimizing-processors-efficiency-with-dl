config = {
    ###########################
    ########## Paths ##########
    ###########################
    "benchmarks_folder": "benchmarks",  # str - The folder where the CUDA benchmarks are located
    "nvcc_path": "/usr/local/cuda-12.6/bin/nvcc",  # str - The path of the NVCC compiler
    "ncu_path": "/usr/local/cuda-12.6/bin/ncu",  # str - The path of the NCU profiler
    "ncu_sections_folder": None,  # str - The path of the folder to search for NCU sections. If None, then the default path is used.
    "ncu_python_report_folder": "/opt/nvidia/nsight-compute/2024.3.2/extras/python/",  # str - The path of the folder where the NCU python report interface is located
    ################################
    ########## GPU config ##########
    ################################
    "gpu_sleep_time": 1,  # int [s] - The amount of time to sleep after changing GPU clocks speeds so the systems stabilizes.
    #################################
    ########## NVML config ##########
    #################################
    "nvml_sampling_freq": 100,  # int [Hz] - The NVML sampling frequency to use (maximum is 100 Hz).
    "nvml_n_runs": 3,  # int [count] - The amount of times to run the program to get the median GPU metrics.
    ################################
    ########## NCU config ##########
    ################################
    "ncu_set": "basic",  # str - The name of the set of metrics to collect using NCU.
}
