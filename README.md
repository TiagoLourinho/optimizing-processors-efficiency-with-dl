# Optimizing GPUs’ energy efficiency through voltage-frequency scaling

## Environment Versions

- OS: [Pop!\_OS 22.04 LTS](https://pop.system76.com/)
- Python: [3.10.17](https://www.python.org/downloads/release/python-31017/)
- NVIDIA Driver: [565.77](https://www.nvidia.com/en-us/geforce/drivers/results/237587/)
- CUDA Toolkit: [12.6.2](https://developer.nvidia.com/cuda-12-6-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

## Environment Setup

1. Install [conda](https://docs.anaconda.com/miniconda/install/).
2. On the project root folder, run `conda env create -f conda_env.yml`.
3. Activate the environment with `conda activate master_thesis_env`.
4. Use this environment to run any of the developed tools on the project root folder.

## Developed Tools

### 1 - Training Samples Collector

#### Objective

Scan through a folder with CUDA benchmarks and convert them to training samples suitable for training the models, namely it:

1 - Compiles the benchmarks and extracts the PTX.

2 - Runs the benchmark suite across all the defined frequency combinations and samples the runtime metrics.

3 - Aggregates the training samples that contain the benchmark name, operating frequencies, static information (PTX) and dynamic information (runtime metrics).

#### Input

A benchmark folder with CUDA applications.

#### Output

A JSON file called `training_data.json` like the sample version below.

```json
{
    "config": {
        "benchmarks_folder": "benchmarks",
        "nvcc_path": "/usr/local/cuda-12.9/bin/nvcc",
        "n_closest_core_clocks": 25,
        "n_closest_mem_clocks": 5,
        "sampling_freq": 100,
        "n_runs": 3
    },
    "system_info": {
        "machine": "test-machine",
        "cpu": "Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz",
        "gpu": "NVIDIA GeForce RTX 3080",
        "ram": 15,
        "run_start_time": "2025-05-30 13:32:41.823992",
        "duration": "85h:50min"
    },
    "models_info": {
        "categorical_sizes": [
            19,
            174,
            9,
            19,
            3
        ],
        "n_numerical_features": 5,
        "n_cupti_metrics": 14,
        "n_nvml_metrics": 3
    },
    "ptxs": {
        "gpuPTXModel_benchmarks_microbenchmarks_akhilarunkumar-GPURelease_data_movement_ept_fadd_l2d_64p_fadd_l2d_95_5_64p": {
            "_Z11init_memoryPPyS_iii": [
                {
                    "categorical": [
                        8,
                        81,
                        6,
                        8,
                        0
                    ],
                    "numerical": [
                        1,
                        1,
                        0,
                        0,
                        0
                    ]
                }
            ]
        }
    },
    "training_data": [
        {
            "benchmark_name": "gpuPTXModel_benchmarks_microbenchmarks_akhilarunkumar-GPURelease_compute_epi_mov_64p_asm",
            "memory_frequency": 9501,
            "graphics_frequency": 570,
            "benchmark_metrics": {
                "runtime": 0.22061873165269694,
                "nvml_metrics": {
                    "TEMPERATURE": 52.0,
                    "POWER": 94.47814814814815,
                    "GPU_UTILIZATION": 0.0
                },
                "cupti_metrics": {
                    "dramc__read_throughput.avg.pct_of_peak_sustained_elapsed": 0.010782665189821619,
                    "dramc__write_throughput.avg.pct_of_peak_sustained_elapsed": 0.14040860199118152,
                    "tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed": 100.0
                }
            }
        }
    ]
}
```

#### Usage

1. Compile the CUPTI script using `source/cupti/Makefile`. (`CUDA_INSTALL_PATH` env variable should be defined)

2. Adjust the tool behavior changing `source/config.py` (`benchmarks_to_training_data` key).

3. Run `sudo $(which python3) source/benchmarks_to_training_data.py`.

### 2 - Models Trainer

#### Objective

Use the training samples collected to train the models and obtain:

- The graphics frequency scaling factor predictor.
- The memory frequency scaling factor predictor.
- The PTX encoder. 

#### Input

The `training_data.json` outputted by the previous tool.

#### Output

The following files concerning the trained models and summaries are created:

- `graphics_predictor.pth`: The trained parameters of the graphics frequency predictor.

- `memory_predictor.pth`: The trained parameters of the memory frequency predictor.

- `ptx_encoder.pth`: The trained parameters of the PTX encoder.

- `standardizer.pkl`: The trained standardizer.

- `models_parameters.json`: The hyperparameters used to build the architecture of the models.

- `models_training_summary.json`: A summary of the training losses and metrics that can be used to produce plots.


#### Usage

1. Adjust the tool behavior changing `source/config.py` (`models_trainer` key).

3. Run `python3 source/models_trainer.py`.

### 3 - Optimizer

#### Objective

Use the trained models to dynamically change the operating frequencies of the GPU while an application is running in order to improve energy efficiency.

#### Inputs

The following files produced by the previous tool: `graphics_predictor.pth`, `memory_predictor.pth`, `ptx_encoder.pth`, `standardizer.pkl` and `models_parameters.json`.

#### Outputs

No file is produced. It runs the application and prints some energy related metrics.

#### Usage

1. Run `sudo $(which python3) source/optimizer.py --ptx_path <PTX_OF_THE_APPLICATION> --exec_path <CUDA_EXECUTABLE>`.

#### 4 - Plots Creator

#### Objective

Create specific plots regarding training, for example.

#### Inputs

With the plot type being `training_summary` it expects to have `models_training_summary.json` produced by the model's trainer tool.

#### Outputs

A file with the desired name with the r² and training losses.

#### Usage

1. Run `python3 source/create_plots.py --plot_type training_summary --input_file models_training_summary.json --output_file out.png`.

## Useful `nvidia-smi` commands

### Query commands

- `nvidia-smi --query-gpu clocks.mem,clocks.gr --format csv`: Displays the current used clocks.
- `nvidia-smi --query-supported-clocks <type> --format csv`: Displays the supported clocks, `type` can be `mem` (memory) or `gr` (graphics/GPU).

_Note: Add `-l <seconds>` to any query command to update the results every `seconds`._

### Persistence mode

- `sudo nvidia-smi -pm <mode>`: Sets the persistence mode of the GPU, `mode` can be `0` (disabled) or `1` (enabled).

### Change clocks

- `sudo nvidia-smi -lmc <clock>`: Sets the memory clock to `clock` MHz (should be one of the supported clocks).
- `sudo nvidia-smi -lgc <clock>`: Sets the GPU clock to `clock` MHz (should be one of the supported clocks).
- `sudo nvidia-smi -rmc`: Resets the memory clock to the default value.
- `sudo nvidia-smi -rgc`: Resets the GPU clock to the default value.