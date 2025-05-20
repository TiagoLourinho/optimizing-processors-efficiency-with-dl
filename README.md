# Optimizing the processors' energy efficiency using deep learning models based on voltage-frequency scaling

## Enviroment versions

- OS: [Pop!\_OS 22.04 LTS](https://pop.system76.com/)
- Python: [3.10.12](https://www.python.org/downloads/release/python-31012/)
- NVIDIA Driver: [565.77](https://www.nvidia.com/en-us/geforce/drivers/results/237587/)
- CUDA Toolkit: [12.6.2](https://developer.nvidia.com/cuda-12-6-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

## Usage

1. Install [conda](https://docs.anaconda.com/miniconda/install/).
2. On the project root folder run `conda env create -f conda_env.yml`.
3. Modify `config.py` to set the profilling behavior.
4. Activate the enviroment with `conda activate master_thesis_env`.
 
### Benchmarks to training data tool

To run the tool that converts the benchmark to the training data (PTX and metrics collected) first compile the CUPTI script using `source/cupti/Makefile` and then run:

`sudo <PYTHON-PATH> source/benchmarks_to_training_data.py`

_Note: PYTHON-PATH can be found using `which python3`._

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
