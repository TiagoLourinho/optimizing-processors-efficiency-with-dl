// Modified but based on the original pm_sampling sample from NVIDIA CUPTI

/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 *
 * This sample demonstrates the usage of the PM sampling feature in the CUDA Profiling Tools Interface (CUPTI).
 * There are two parts to this sample:
 * 1) Querying the available metrics for a chip and their properties.
 * 2) Collecting PM sampling data for a CUDA workload.
 *
 * The PM sampling feature allows users to collect sampling data at a specific interval for the CUDA workload
 * launched in the application.
 * For the continuous collection usecase, two separate threads are required:
 * 1. A main thread for launching the CUDA workload.
 * 2. A decode thread for decoding the collected data at a certain interval.
 *
 * The user is responsible for continuously calling the decode API, which frees up the hardware buffer for storing new data.
 *
 * In this sample, In the main thread the CUDA workload is launched. This workload is a simple vector addition implemented
 * in the `VectorAdd` kernel.
 *
 * The decode thread where we call the `DecodeCounterData` API. This API decodes the raw PM sampling data stored
 * in the hardware to a counter data image that the user has allocated.
 *
 */

#include <atomic>
#include <chrono>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <thread>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#include "pm_sampling.h"

#define SERVER_PORT 45535

std::atomic<bool> stopDecodeThread(false);

struct ConfigArgs
{
    int deviceIndex = 0;
    std::string chipName;
    uint64_t samplingInterval = 5000000;            // ns == 0.005s -> 200Hz (so double the sampling rate used in the python script with nvml)
    size_t hardwareBufferSize = 1024 * 1024 * 1024; // 1024MB
    uint64_t maxSamples = 10000;
    // Metrics from https://docs.nvidia.com/cupti/main/main.html#metrics-table
    // The ones commented out resulted in CUPTI_ERROR_INVALID_PARAMETER.
    std::vector<const char *> metrics =
        {
            "gpc__cycles_elapsed.avg.per_second",
            "sys__cycles_elapsed.avg.per_second",
            "gr__cycles_active.sum.pct_of_peak_sustained_elapsed",
            "gr__dispatch_count.avg.pct_of_peak_sustained_elapsed",
            "tpc__warps_inactive_sm_active_realtime.avg.pct_of_peak_sustained_elapsed",
            "tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed",
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_shared_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
            "dramc__read_throughput.avg.pct_of_peak_sustained_elapsed",
            "dramc__write_throughput.avg.pct_of_peak_sustained_elapsed",
            "pcie__read_bytes.avg.pct_of_peak_sustained_elapsed",
            "pcie__write_bytes.avg.pct_of_peak_sustained_elapsed",
            "nvlrx__bytes.avg.pct_of_peak_sustained_elapsed",
            "nvltx__bytes.avg.pct_of_peak_sustained_elapsed",
            "pcie__rx_requests_aperture_bar1_op_read.sum",
            "pcie__rx_requests_aperture_bar1_op_write.sum"};
};

void startServer(CuptiProfilerHost &pmSamplingHost);
void PmSamplingDeviceSupportStatus(CUdevice device);
int PmSamplingCollection(std::vector<uint8_t> &counterAvailibilityImage, ConfigArgs &args);
void DecodeCounterData(
    std::vector<uint8_t> &counterDataImage,
    std::vector<const char *> metricsList,
    CuptiPmSampling &cuptiPmSamplingTarget,
    CuptiProfilerHost &pmSamplingHost,
    CUptiResult &result);

int main()
{
    ConfigArgs args;
    DRIVER_API_CALL(cuInit(0));

    std::string chipName = args.chipName;
    std::vector<uint8_t> counterAvailibilityImage;
    if (args.deviceIndex >= 0)
    {
        CUdevice cuDevice;
        DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));
        PmSamplingDeviceSupportStatus(cuDevice);

        CuptiPmSampling::GetChipName(args.deviceIndex, chipName);
        CuptiPmSampling::GetCounterAvailabilityImage(args.deviceIndex, counterAvailibilityImage);
    }

    return PmSamplingCollection(counterAvailibilityImage, args);
}

/* Starts the server that will expose the CUPTI metrics to the python script */
void startServer(CuptiProfilerHost &pmSamplingHost)
{
    httplib::Server svr;

    svr.Get("/metrics", [&](const httplib::Request &, httplib::Response &res)
            {
        json j = pmSamplingHost.getMostRecentSamplerRangeAsJson();
       
        res.set_content(j.dump(), "application/json"); });

    svr.Get("/shutdown", [&](const httplib::Request &, httplib::Response &res)
            {
        std::cout << "Stopping server...\n";
        res.set_content("{\"message\": \"Shutting down...\"}", "application/json");

        // Shutdown must happen in a separate thread, or it'll deadlock
        std::thread([&svr]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // small delay to ensure response is sent
            svr.stop();
        }).detach(); });

    std::cout << "Starting server...\n";
    svr.listen("0.0.0.0", SERVER_PORT);
}

int PmSamplingCollection(std::vector<uint8_t> &counterAvailibilityImage, ConfigArgs &args)
{
    std::string chipName;
    CuptiPmSampling::GetChipName(args.deviceIndex, chipName);

    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    std::vector<uint8_t> configImage;
    CUPTI_API_CALL(pmSamplingHost.CreateConfigImage(args.metrics, configImage));

    CuptiPmSampling cuptiPmSamplingTarget;
    cuptiPmSamplingTarget.SetUp(args.deviceIndex);

    // 1. Enable PM sampling and set config for the PM sampling data collection.
    CUPTI_API_CALL(cuptiPmSamplingTarget.EnablePmSampling(args.deviceIndex));
    CUPTI_API_CALL(cuptiPmSamplingTarget.SetConfig(configImage, args.hardwareBufferSize, args.samplingInterval));

    // 2. Create counter data image
    std::vector<uint8_t> counterDataImage;
    CUPTI_API_CALL(cuptiPmSamplingTarget.CreateCounterDataImage(args.maxSamples, args.metrics, counterDataImage));

    CUptiResult threadFuncResult;
    // 3. Launch the decode thread
    std::thread decodeThread(DecodeCounterData, std::ref(counterDataImage), std::ref(args.metrics), std::ref(cuptiPmSamplingTarget), std::ref(pmSamplingHost), std::ref(threadFuncResult));

    auto joinDecodeThread = [&]()
    {
        stopDecodeThread = true;
        decodeThread.join();
        if (threadFuncResult != CUPTI_SUCCESS)
        {
            const char *errstr;
            cuptiGetResultString(threadFuncResult, &errstr);
            std::cerr << "DecodeCounterData Thread failed with error " << errstr << std::endl;
            return 1;
        }
        return 0;
    };

    // 4. Start the PM sampling
    CUPTI_API_CALL(cuptiPmSamplingTarget.StartPmSampling());
    stopDecodeThread = false;

    startServer(std::ref(pmSamplingHost));

    cudaError_t errResult = cudaDeviceSynchronize();
    if (errResult != cudaSuccess)
    {
        std::cerr << "DeviceSync Failed " << cudaGetErrorString(errResult) << std::endl;
        return joinDecodeThread();
    }

    // 5. Stop the PM sampling and join the decode thread
    CUPTI_API_CALL(cuptiPmSamplingTarget.StopPmSampling());
    joinDecodeThread();

    // 7. Clean up
    cuptiPmSamplingTarget.TearDown();
    pmSamplingHost.TearDown();

    return 0;
}

void DecodeCounterData(std::vector<uint8_t> &counterDataImage,
                       std::vector<const char *> metricsList,
                       CuptiPmSampling &cuptiPmSamplingTarget,
                       CuptiProfilerHost &pmSamplingHost,
                       CUptiResult &result)
{
    while (!stopDecodeThread)
    {
        const char *errstr;
        result = cuptiPmSamplingTarget.DecodePmSamplingData(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "DecodePmSamplingData failed with error " << errstr << std::endl;
            return;
        }

        CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo{CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
        counterDataInfo.pCounterDataImage = counterDataImage.data();
        counterDataInfo.counterDataImageSize = counterDataImage.size();
        result = cuptiPmSamplingGetCounterDataInfo(&counterDataInfo);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "cuptiPmSamplingGetCounterDataInfo failed with error " << errstr << std::endl;
            return;
        }

        for (size_t sampleIndex = 0; sampleIndex < counterDataInfo.numCompletedSamples; ++sampleIndex)
        {
            pmSamplingHost.EvaluateCounterData(cuptiPmSamplingTarget.GetPmSamplerObject(), sampleIndex, metricsList, counterDataImage);
        }
        result = cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage);
        if (result != CUPTI_SUCCESS)
        {
            cuptiGetResultString(result, &errstr);
            std::cerr << "ResetCounterDataImage failed with error " << errstr << std::endl;
            return;
        }
    }
}

void PmSamplingDeviceSupportStatus(CUdevice device)
{
    CUpti_Profiler_DeviceSupported_Params params = {CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
    params.cuDevice = device;
    params.api = CUPTI_PROFILER_PM_SAMPLING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << device << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }

        exit(EXIT_WAIVED);
    }
}
