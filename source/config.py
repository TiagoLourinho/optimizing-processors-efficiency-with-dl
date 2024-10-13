# Config of the profile cuda tool behavior
config = {
    ######################################
    ########## Executable paths ##########
    ######################################
    "nvcc_path": "/usr/local/cuda-12.4/bin/nvcc",  # str - The path of the NVCC compiler
    "ncu_path": "/usr/local/cuda-12.4/bin/ncu",  # str - The path of the NCU profiler
    ################################
    ########## GPU config ##########
    ################################
    "gpu_graphics_clk": None,  # int [MHz] | None - The graphics clock to use. If None, then it is "free".
    "gpu_memory_clk": None,  # int [MHz] | None - The memory clock to use. If None, then it is "free".
    "gpu_sleep_time": 1,  # int [s] - The amount of time to sleep after changing GPU clocks speeds so the systems stabilizes.
    #################################
    ########## NVML config ##########
    #################################
    "nvml_sampling_freq": 10,  # int [Hz] - The NVML sampling frequency to use (maximum is 50 Hz).
    "nvml_n_runs": 1,  # int [count] - The amount of times to run the program to get the median GPU metrics.
    ##############################################################################
    ########## NCU config (only one of the following should be defined) ##########
    ##############################################################################
    "ncu_set": "basic",  # str - The name of the set of metrics to collect using NCU.
    "ncu_sections": None,  # list[str] - The list of sections of metrics to collect using NCU.
    "ncu_metrics": None,  # list[str] - The list of isolated metrics to collect using NCU.
}
