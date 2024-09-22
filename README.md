# Optimizing the processors' energy efficiency using deep learning models based on voltage-frequency scaling

## Enviroment versions

- OS: [Ubuntu 22.04.5 LTS](https://releases.ubuntu.com/jammy/)
- Python: [3.10.12](https://www.python.org/downloads/release/python-31012/)
- NVIDIA Driver: [550.107.02](https://www.nvidia.com/en-us/drivers/details/230357/)
- CUDA Toolkit: [12.4.1](https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)

## Usage

1. Install [pipenv](https://pipenv.pypa.io/en/latest/).
2. On the project root folder run `pipenv install`.
3. Now run for example: `sudo -E pipenv run python3 source/profile_cuda.py --nvcc $(which nvcc) -N 1 --graphics-clk 2040 --memory-clk 8001 --sleep-time 1 benchmarks/benchmark.cu`

_Note: Run `pipenv run python3 source/time_cuda.py -h` to see the command usage._

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
