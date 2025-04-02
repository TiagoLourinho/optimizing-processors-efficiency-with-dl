import time
from enum import Enum

from pynvml import *


class GPUQueries(Enum):
    """Defines the queryable information of the GPU"""

    GRAPHICS_CLOCK = "Graphics clock [MHz]"
    """  Graphics clock [MHz] """

    MEMORY_CLOCK = "Memory clock [MHz]"
    """ Memory clock [MHz] """

    TEMPERATURE = "Temperature [°C]"
    """ Temperature [°C] """

    POWER = "Power [W]"
    """ Power [W] """

    GPU_UTILIZATION = "GPU utilization [%]"
    """ GPU utilization [%] """


class GPUClockChangingError(Exception):
    """Custom exception for when the driver doesn't let the GPU change frequendcies"""

    pass


class GPU:
    """Interface to query and manage the GPU"""

    ######################################## Dunder methods ########################################

    def __init__(self, sleep_time: int):
        """Defines sleep time and locked flags"""

        # After changing the GPU clocks, wait a bit to let the system stabilize
        self.sleep_time = sleep_time  # s

        # To know whether or not the clocks are locked
        self.__is_graphics_clk_locked = False
        self.__is_memory_clk_locked = False

    def __enter__(self) -> "GPU":
        """Initializes nvml and gets the device handle"""

        print("Initializing GPU...")

        nvmlInit()

        if nvmlDeviceGetCount() != 1:
            raise Exception("Only 1 GPU is currently supported")

        self.__handle = nvmlDeviceGetHandleByIndex(index=0)

        # Enable persistence mode to avoid the overhead of loading the driver each time
        nvmlDeviceSetPersistenceMode(handle=self.__handle, mode=NVML_FEATURE_ENABLED)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restores default values and shutdown nvml"""

        print("Resetting GPU clocks and shutting down nvml...")

        self.reset_graphics_clk()
        self.reset_memory_clk()

        nvmlDeviceSetPersistenceMode(handle=self.__handle, mode=NVML_FEATURE_DISABLED)

        nvmlShutdown()

    ######################################## Properties ########################################

    @property
    def name(self) -> str:
        """Returns the GPU name"""

        return nvmlDeviceGetName(handle=self.__handle)

    ######################################## Clocks management and monitoring ########################################

    @property
    def graphics_clk(self) -> int:
        """Graphics clock [MHz]"""

        return nvmlDeviceGetClockInfo(handle=self.__handle, type=NVML_CLOCK_GRAPHICS)

    @property
    def memory_clk(self) -> int:
        """Memory clock [MHz]"""

        # Avoid clock mismatch (see `get_supported_memory_clocks` docstring for more info)

        supported_clocks = set(self.get_supported_memory_clocks())
        sanitized_supported_clocks = set(
            self.get_supported_memory_clocks(sanitize=True)
        )
        bugged_clocks = sanitized_supported_clocks - supported_clocks

        current_clock = nvmlDeviceGetClockInfo(
            handle=self.__handle, type=NVML_CLOCK_MEM
        )

        return current_clock + (1 if current_clock in bugged_clocks else 0)

    @graphics_clk.setter
    def graphics_clk(self, value: int):
        """Sets the current graphics clock to `value` MHz"""

        # If the memory clock is locked, then only consider the current memory clock to get the supported graphics clock
        # If not, use the maximum possible memory clock so it returns all possible graphics clocks
        if self.__is_memory_clk_locked:
            memory_clk = self.memory_clk
        else:
            memory_clk = max(self.get_supported_memory_clocks())

        supported_clocks = self.get_supported_graphics_clocks(memory_clock=memory_clk)

        if value not in supported_clocks:
            raise ValueError(
                f"A graphics clock of {value} MHz isn't supported when paired with a memory clock of {memory_clk} MHz, choose from:\n{supported_clocks}"
            )

        nvmlDeviceSetGpuLockedClocks(
            handle=self.__handle, minGpuClockMHz=value, maxGpuClockMHz=value
        )

        time.sleep(self.sleep_time)

        # NVML returns for example 7001 MHz in the supported clocks,
        # but then the value returned by the query is just 7000 MHz
        if abs(self.graphics_clk - value) > 1:
            raise GPUClockChangingError()

        self.__is_graphics_clk_locked = True

    @memory_clk.setter
    def memory_clk(self, value: int):
        """Sets the current memory clock to `value` MHz"""

        supported_clocks = self.get_supported_memory_clocks()

        if value not in supported_clocks:
            raise ValueError(
                f"A memory clock of {value} MHz isn't supported, choose from:\n{supported_clocks}"
            )

        nvmlDeviceSetMemoryLockedClocks(
            handle=self.__handle, minMemClockMHz=value, maxMemClockMHz=value
        )

        time.sleep(self.sleep_time)

        # NVML returns for example 7001 MHz in the supported clocks,
        # but then the value returned by the query is just 7000 MHz
        if abs(self.memory_clk - value) > 1:
            raise GPUClockChangingError()

        self.__is_memory_clk_locked = True

    def reset_graphics_clk(self):
        """Resets the graphics clock to the default value"""

        nvmlDeviceResetGpuLockedClocks(handle=self.__handle)
        time.sleep(self.sleep_time)

        self.__is_graphics_clk_locked = False

    def reset_memory_clk(self):
        """Resets the memory clock to the default value"""

        nvmlDeviceResetMemoryLockedClocks(handle=self.__handle)
        time.sleep(self.sleep_time)

        self.__is_memory_clk_locked = False

    def get_supported_graphics_clocks(self, memory_clock: int) -> list[int]:
        """Returns the supported graphics clocks [MHz]"""

        return sorted(
            nvmlDeviceGetSupportedGraphicsClocks(
                handle=self.__handle, memoryClockMHz=memory_clock
            )
        )

    def get_supported_memory_clocks(self, sanitize: bool = False) -> list[int]:
        """
        Returns the supported memory clocks [MHz]

        If `sanitize` is True the method "fixes" some clocks
        (useful when the list is going to be used for comparison with the current memory clock for example)

        Example:
            1 - NVML supported clock call returns the following memory clocks: [405, 810, 6001, 7001, 8001]
            2 - Setting the memory clock to 8001 and then querying it would return 8000, causing a mismatch

        So, with `sanitize=True`, the function would "sanitize" the clocks ending in 1 and return [405, 810, 6000, 7000, 8000]
        """

        supported_clocks = sorted(
            nvmlDeviceGetSupportedMemoryClocks(handle=self.__handle)
        )

        if sanitize:
            # Remove numbers ending in 1 to avoid mismatch
            supported_clocks = [
                clock if str(clock)[-1] != "1" else clock - 1
                for clock in supported_clocks
            ]

        return supported_clocks

    ######################################## Queries ########################################

    def query(self, query_type: GPUQueries) -> float | int:
        """Queries a value of the GPU"""

        match query_type:
            case GPUQueries.GRAPHICS_CLOCK:
                return self.graphics_clk
            case GPUQueries.MEMORY_CLOCK:
                return self.memory_clk
            case GPUQueries.TEMPERATURE:
                return nvmlDeviceGetTemperature(
                    handle=self.__handle, sensor=NVML_TEMPERATURE_GPU
                )
            case GPUQueries.POWER:
                return nvmlDeviceGetPowerUsage(handle=self.__handle) / 1000  # To watts
            case GPUQueries.GPU_UTILIZATION:
                return nvmlDeviceGetUtilizationRates(handle=self.__handle).gpu
            case _:
                raise ValueError(f"Invalid GPU query: {query_type}")
