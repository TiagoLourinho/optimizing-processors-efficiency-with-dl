from pynvml import *


class GPU:
    """Interface to query and manage the GPU"""

    def __init__(self):
        """Initializes nvml and gets the device handle"""

        nvmlInit()

        if nvmlDeviceGetCount() != 1:
            raise Exception("Only 1 GPU is currently supported")

        self.__handle = nvmlDeviceGetHandleByIndex(0)

    def __del__(self):
        """Shutdowns nvml"""

        nvmlShutdown()
