# Optimizing the processors' energy efficiency using deep learning models based on voltage-frequency scaling

## Enviroment versions

- OS: [Pop!\_OS 22.04 LTS](https://pop.system76.com/)
- Python: [3.10.12](https://www.python.org/downloads/release/python-31012/)
- NVIDIA Driver: [560.35.03](https://www.nvidia.com/en-us/drivers/details/230918/)
- CUDA Toolkit: [12.6.2](https://developer.nvidia.com/cuda-12-6-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

## Usage

1. Install [pipenv](https://pipenv.pypa.io/en/latest/).
2. On the project root folder run `pipenv install`.
3. Modify `config.py` to set the profilling behavior.
4. Now run:

`sudo -E env PATH="$PATH" pipenv run python3 source/profile_cuda.py <PATH_TO_CUDA_BENCHMARK> <BENCHMARK_ARGS> -o <OUTPUT_FILENAME>`

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
