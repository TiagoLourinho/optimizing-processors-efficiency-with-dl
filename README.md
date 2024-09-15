# Optimizing the processors' energy efficiency using deep learning models based on voltage-frequency scaling

## Enviroment versions

- OS: [Ubuntu 24.04.1 LTS](https://releases.ubuntu.com/24.04/)
- Python: [3.12.3](https://www.python.org/downloads/release/python-3123/)
- NVIDIA Driver: [560.35.03](https://www.nvidia.com/en-us/drivers/details/230918/)
- CUDA Toolkit: [12.6.1](https://developer.nvidia.com/cuda-12-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network)

## Useful `nvidia-smi` commands

### Query commands

- `nvidia-smi --query-gpu clocks.mem,clocks.gr --format csv`: Displays the current used clocks.
- `nvidia-smi --query-supported-clocks <type> --format csv`: Displays the supported clocks, `type` can be `mem` (memory) or `gr` (graphics/GPU).

_Note: Add `-l <seconds>` to any query command to update the results every `seconds`._

### Persistence mode

- `sudo nvidia-smi -pm <mode>`: Sets the persistence mode of the GPU, `mode` can be `0` (disabled) or `1` (enabled).

### Change clocks

- `sudo nvidia-smi -lmc <freq>`: Sets the memory frequency to `freq` MHz (should be one of the supported frequencies).
- `sudo nvidia-smi -lgc <freq>`: Sets the GPU frequency to `freq` MHz (should be one of the supported frequencies).
- `sudo nvidia-smi -rmc`: Resets the memory clock to the default value.
- `sudo nvidia-smi -rgc`: Resets the GPU clock to the default value.
