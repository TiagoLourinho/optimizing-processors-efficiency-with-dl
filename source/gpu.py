import time

from pynvml import *


class GPU:
    """Interface to query and manage the GPU"""

    def __init__(self, sleep_time: int):
        """Initializes nvml and gets the device handle"""

        # After changing the GPU clocks, wait a bit to let the system stabilize
        self.__sleep_time = sleep_time  # s

        nvmlInit()

        if nvmlDeviceGetCount() != 1:
            raise Exception("Only 1 GPU is currently supported")

        self.__handle = nvmlDeviceGetHandleByIndex(index=0)

        # Enable persistence mode to avoid the overhead of loading the driver each time
        nvmlDeviceSetPersistenceMode(handle=self.__handle, mode=NVML_FEATURE_ENABLED)

    def __del__(self):
        """Restores default values and shutdown nvml"""

        self.reset_graphics_clk()
        self.reset_memory_clk()

        nvmlDeviceSetPersistenceMode(handle=self.__handle, mode=NVML_FEATURE_DISABLED)

        nvmlShutdown()

    @property
    def graphics_clk(self) -> int:
        """Returns the current graphics clock"""

        return nvmlDeviceGetClockInfo(handle=self.__handle, type=NVML_CLOCK_GRAPHICS)

    @property
    def memory_clk(self) -> int:
        """Returns the current memory clock"""

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
        """Sets the current graphics clock"""

        supported_clocks = self.get_supported_graphics_clocks(
            memory_clock=self.memory_clk
        )

        if value not in supported_clocks:
            raise ValueError(
                f"A graphics clock of {value}MHz isn't supported when paired with a memory clock of {self.memory_clk}MHz, choose from:\n{supported_clocks}"
            )

        nvmlDeviceSetGpuLockedClocks(
            handle=self.__handle, minGpuClockMHz=value, maxGpuClockMHz=value
        )

        time.sleep(self.__sleep_time)

        # NVML returns for example 7001 MHz in the supported clocks,
        # but then the value returned by the query is just 7000 MHz
        if abs(self.graphics_clk - value) > 1:
            raise ValueError(
                f"It wasn't possible to set the graphics clock to {value}MHz probably due to power/temperature limits."
            )

    @memory_clk.setter
    def memory_clk(self, value: int):
        """Sets the current memory clock"""

        supported_clocks = self.get_supported_memory_clocks()

        if value not in supported_clocks:
            raise ValueError(
                f"A memory clock of {value}MHz isn't supported, choose from:\n{supported_clocks}"
            )

        nvmlDeviceSetMemoryLockedClocks(
            handle=self.__handle, minMemClockMHz=value, maxMemClockMHz=value
        )

        time.sleep(self.__sleep_time)

        # NVML returns for example 7001 MHz in the supported clocks,
        # but then the value returned by the query is just 7000 MHz
        if abs(self.memory_clk - value) > 1:
            raise ValueError(
                f"It wasn't possible to set the memory clock to {value}MHz probably due to power/temperature limits."
            )

    def reset_graphics_clk(self):
        """Resets the graphics clock to the default value"""

        nvmlDeviceResetGpuLockedClocks(handle=self.__handle)
        time.sleep(self.__sleep_time)

    def reset_memory_clk(self):
        """Resets the memory clock to the default value"""

        nvmlDeviceResetMemoryLockedClocks(handle=self.__handle)
        time.sleep(self.__sleep_time)

    def get_supported_graphics_clocks(self, memory_clock: int) -> list[int]:
        """Returns the supported graphics clocks"""

        return sorted(
            nvmlDeviceGetSupportedGraphicsClocks(
                handle=self.__handle, memoryClockMHz=memory_clock
            )
        )

    def get_supported_memory_clocks(self, sanitize: bool = False) -> list[int]:
        """
        Returns the supported memory clocks

        If `sanitize` is True the method "fixes" some clocks
        (useful when the list is going to be used for comparison with the current memory clock for example)

        Example:
            1 - NVML supported clock call returns the following memory clocks: [405, 810, 6001, 7001, 8001]
            2 - Setting the memory clock to 8001 and then querying it would return 8000, causing a mismatch

        So, with `sanitize`=True, the function would "sanitize" the clocks ending in 1 and return [405, 810, 6000, 7000, 8000]
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
