// Modified but based on the original pm_sampling sample from NVIDIA CUPTI

//
// Copyright 2024 NVIDIA Corporation. All rights reserved
//

// System headers
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <mutex>

// CUPTI headers
#include "helper_cupti.h"
#include <cupti_target.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>

// External headers
#include <json.hpp>
#include <httplib.h>

using json = nlohmann::json;

struct SamplerRange
{
    uint64_t sampleIndex;
    uint64_t startTimestamp;
    uint64_t endTimestamp;
    std::map<std::string, double> metricValues;
};

class CuptiProfilerHost
{
    std::string m_chipName;
    SamplerRange m_mostRecentSamplerRange;
    /*  m_mostRecentSamplerRange will be accessed by 2 threads so a mutex is needed:
            - the main thread will read it to send it to the python script
            - the decode thread will write on it
    */
    std::mutex m_samplerRangeMutex;
    uint64_t m_samplesCollected = 0;
    CUpti_Profiler_Host_Object *m_pHostObject = nullptr;

public:
    CuptiProfilerHost() = default;
    ~CuptiProfilerHost() = default;

    void SetUp(std::string chipName, std::vector<uint8_t> &counterAvailibilityImage)
    {
        m_chipName = chipName;
        CUPTI_API_CALL(Initialize(counterAvailibilityImage));
    }

    void TearDown()
    {
        CUPTI_API_CALL(Deinitialize());
    }

    CUptiResult CreateConfigImage(std::vector<const char *> metricsList, std::vector<uint8_t> &configImage)
    {
        // Add metrics to config image
        {
            CUpti_Profiler_Host_ConfigAddMetrics_Params configAddMetricsParams{CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
            configAddMetricsParams.pHostObject = m_pHostObject;
            configAddMetricsParams.ppMetricNames = metricsList.data();
            configAddMetricsParams.numMetrics = metricsList.size();
            CUPTI_API_CALL(cuptiProfilerHostConfigAddMetrics(&configAddMetricsParams));
        }

        // Get Config image size and data
        {
            CUpti_Profiler_Host_GetConfigImageSize_Params getConfigImageSizeParams{CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
            getConfigImageSizeParams.pHostObject = m_pHostObject;
            CUPTI_API_CALL(cuptiProfilerHostGetConfigImageSize(&getConfigImageSizeParams));
            configImage.resize(getConfigImageSizeParams.configImageSize);

            CUpti_Profiler_Host_GetConfigImage_Params getConfigImageParams = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
            getConfigImageParams.pHostObject = m_pHostObject;
            getConfigImageParams.pConfigImage = configImage.data();
            getConfigImageParams.configImageSize = configImage.size();
            CUPTI_API_CALL(cuptiProfilerHostGetConfigImage(&getConfigImageParams));
        }

        // Get Num of Passes
        {
            CUpti_Profiler_Host_GetNumOfPasses_Params getNumOfPassesParam{CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
            getNumOfPassesParam.pConfigImage = configImage.data();
            getNumOfPassesParam.configImageSize = configImage.size();
            CUPTI_API_CALL(cuptiProfilerHostGetNumOfPasses(&getNumOfPassesParam));
            std::cout << "Num of Passes: " << getNumOfPassesParam.numOfPasses << "\n";
        }

        return CUPTI_SUCCESS;
    }

    CUptiResult EvaluateCounterData(CUpti_PmSampling_Object *pSamplingObject, size_t rangeIndex, std::vector<const char *> metricsList, std::vector<uint8_t> &counterDataImage)
    {
        std::lock_guard<std::mutex> lock(m_samplerRangeMutex);

        m_mostRecentSamplerRange = SamplerRange{};
        SamplerRange &samplerRange = m_mostRecentSamplerRange;

        CUpti_PmSampling_CounterData_GetSampleInfo_Params getSampleInfoParams = {CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
        getSampleInfoParams.pPmSamplingObject = pSamplingObject;
        getSampleInfoParams.pCounterDataImage = counterDataImage.data();
        getSampleInfoParams.counterDataImageSize = counterDataImage.size();
        getSampleInfoParams.sampleIndex = rangeIndex;
        CUPTI_API_CALL(cuptiPmSamplingCounterDataGetSampleInfo(&getSampleInfoParams));

        samplerRange.startTimestamp = getSampleInfoParams.startTimestamp;
        samplerRange.endTimestamp = getSampleInfoParams.endTimestamp;
        samplerRange.sampleIndex = m_samplesCollected;
        m_samplesCollected++;

        std::vector<double> metricValues(metricsList.size());
        CUpti_Profiler_Host_EvaluateToGpuValues_Params evalauateToGpuValuesParams{CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        evalauateToGpuValuesParams.pHostObject = m_pHostObject;
        evalauateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
        evalauateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
        evalauateToGpuValuesParams.ppMetricNames = metricsList.data();
        evalauateToGpuValuesParams.numMetrics = metricsList.size();
        evalauateToGpuValuesParams.rangeIndex = rangeIndex;
        evalauateToGpuValuesParams.pMetricValues = metricValues.data();
        CUPTI_API_CALL(cuptiProfilerHostEvaluateToGpuValues(&evalauateToGpuValuesParams));

        for (size_t i = 0; i < metricsList.size(); ++i)
        {
            samplerRange.metricValues[metricsList[i]] = metricValues[i];
        }

        return CUPTI_SUCCESS;
    }

    /* Returns the most recent sample in JSON format */
    json getMostRecentSamplerRangeAsJson()
    {
        std::lock_guard<std::mutex> lock(m_samplerRangeMutex);
        json j;

        j["sampleIndex"] = m_mostRecentSamplerRange.sampleIndex;
        j["startTimestamp"] = m_mostRecentSamplerRange.startTimestamp;
        j["endTimestamp"] = m_mostRecentSamplerRange.endTimestamp;
        j["metricValues"] = m_mostRecentSamplerRange.metricValues;

        return j;
    }

private:
    CUptiResult Initialize(std::vector<uint8_t> &counterAvailibilityImage)
    {
        CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
        hostInitializeParams.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
        hostInitializeParams.pChipName = m_chipName.c_str();
        hostInitializeParams.pCounterAvailabilityImage = counterAvailibilityImage.data();
        CUPTI_API_CALL(cuptiProfilerHostInitialize(&hostInitializeParams));
        m_pHostObject = hostInitializeParams.pHostObject;
        return CUPTI_SUCCESS;
    }

    CUptiResult Deinitialize()
    {
        CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        deinitializeParams.pHostObject = m_pHostObject;
        CUPTI_API_CALL(cuptiProfilerHostDeinitialize(&deinitializeParams));
        return CUPTI_SUCCESS;
    }
};

class CuptiPmSampling
{
    CUpti_PmSampling_Object *m_pmSamplerObject = nullptr;

public:
    CuptiPmSampling() = default;
    ~CuptiPmSampling() = default;

    void SetUp(int deviceIndex)
    {
        std::cout << "CUDA Device Number: " << deviceIndex << "\n";
        CUdevice cuDevice;
        DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceIndex));

        {
            int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
            DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
            DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
            std::cout << "Compute Capability of Device: " << computeCapabilityMajor << "." << computeCapabilityMinor << "\n";
            if (computeCapabilityMajor * 10 + computeCapabilityMinor < 75)
            {
                std::cerr << "Sample not supported as it requires compute capability >= 7.5\n";
                exit(0);
            }
        }

        CUPTI_API_CALL(InitializeProfiler());
    }

    void TearDown()
    {
        CUPTI_API_CALL(DeInitializeProfiler());
    }

    CUptiResult CreateCounterDataImage(uint64_t maxSamples, std::vector<const char *> metricsList, std::vector<uint8_t> &counterDataImage)
    {
        CUpti_PmSampling_GetCounterDataSize_Params getCounterDataSizeParams = {CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
        getCounterDataSizeParams.pPmSamplingObject = m_pmSamplerObject;
        getCounterDataSizeParams.numMetrics = metricsList.size();
        getCounterDataSizeParams.pMetricNames = metricsList.data();
        getCounterDataSizeParams.maxSamples = maxSamples;
        CUPTI_API_CALL(cuptiPmSamplingGetCounterDataSize(&getCounterDataSizeParams));

        counterDataImage.resize(getCounterDataSizeParams.counterDataSize);
        CUpti_PmSampling_CounterDataImage_Initialize_Params initializeParams{CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
        initializeParams.pPmSamplingObject = m_pmSamplerObject;
        initializeParams.counterDataSize = counterDataImage.size();
        initializeParams.pCounterData = counterDataImage.data();
        CUPTI_API_CALL(cuptiPmSamplingCounterDataImageInitialize(&initializeParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult ResetCounterDataImage(std::vector<uint8_t> &counterDataImage)
    {
        CUpti_PmSampling_CounterDataImage_Initialize_Params initializeParams{CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
        initializeParams.pPmSamplingObject = m_pmSamplerObject;
        initializeParams.counterDataSize = counterDataImage.size();
        initializeParams.pCounterData = counterDataImage.data();
        CUPTI_API_CALL(cuptiPmSamplingCounterDataImageInitialize(&initializeParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult EnablePmSampling(size_t devIndex)
    {
        CUpti_PmSampling_Enable_Params enableParams{CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
        enableParams.deviceIndex = devIndex;
        CUPTI_API_CALL(cuptiPmSamplingEnable(&enableParams));
        m_pmSamplerObject = enableParams.pPmSamplingObject;
        return CUPTI_SUCCESS;
    }

    CUptiResult DisablePmSampling()
    {
        CUpti_PmSampling_Disable_Params disableParams{CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
        disableParams.pPmSamplingObject = m_pmSamplerObject;
        CUPTI_API_CALL(cuptiPmSamplingDisable(&disableParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult SetConfig(std::vector<uint8_t> &configImage, size_t hardwareBufferSize, uint64_t samplingInterval)
    {
        CUpti_PmSampling_SetConfig_Params setConfigParams = {CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
        setConfigParams.pPmSamplingObject = m_pmSamplerObject;

        setConfigParams.configSize = configImage.size();
        setConfigParams.pConfig = configImage.data();

        setConfigParams.hardwareBufferSize = hardwareBufferSize;
        setConfigParams.samplingInterval = samplingInterval;

        setConfigParams.triggerMode = CUpti_PmSampling_TriggerMode::CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
        CUPTI_API_CALL(cuptiPmSamplingSetConfig(&setConfigParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult StartPmSampling()
    {
        CUpti_PmSampling_Start_Params startProfilingParams = {CUpti_PmSampling_Start_Params_STRUCT_SIZE};
        startProfilingParams.pPmSamplingObject = m_pmSamplerObject;
        CUPTI_API_CALL(cuptiPmSamplingStart(&startProfilingParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult StopPmSampling()
    {
        CUpti_PmSampling_Stop_Params stopProfilingParams = {CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
        stopProfilingParams.pPmSamplingObject = m_pmSamplerObject;
        CUPTI_API_CALL(cuptiPmSamplingStop(&stopProfilingParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult DecodePmSamplingData(std::vector<uint8_t> &counterDataImage)
    {
        CUpti_PmSampling_DecodeData_Params decodeDataParams = {CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
        decodeDataParams.pPmSamplingObject = m_pmSamplerObject;
        decodeDataParams.pCounterDataImage = counterDataImage.data();
        decodeDataParams.counterDataImageSize = counterDataImage.size();
        CUPTI_API_CALL(cuptiPmSamplingDecodeData(&decodeDataParams));
        return CUPTI_SUCCESS;
    }

    CUpti_PmSampling_Object *GetPmSamplerObject()
    {
        return m_pmSamplerObject;
    }

    static CUptiResult GetChipName(size_t deviceIndex, std::string &chipName)
    {
        CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        CUpti_Device_GetChipName_Params getChipNameParams = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
        getChipNameParams.deviceIndex = deviceIndex;
        CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
        chipName = getChipNameParams.pChipName;
        return CUPTI_SUCCESS;
    }

    static CUptiResult GetCounterAvailabilityImage(size_t deviceIndex, std::vector<uint8_t> &counterAvailibilityImage)
    {
        CUpti_PmSampling_GetCounterAvailability_Params getCounterAvailabilityParams{CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
        getCounterAvailabilityParams.deviceIndex = deviceIndex;
        CUPTI_API_CALL(cuptiPmSamplingGetCounterAvailability(&getCounterAvailabilityParams));

        counterAvailibilityImage.clear();
        counterAvailibilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);
        getCounterAvailabilityParams.pCounterAvailabilityImage = counterAvailibilityImage.data();
        CUPTI_API_CALL(cuptiPmSamplingGetCounterAvailability(&getCounterAvailabilityParams));
        return CUPTI_SUCCESS;
    }

private:
    CUptiResult InitializeProfiler()
    {
        CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
        return CUPTI_SUCCESS;
    }

    CUptiResult DeInitializeProfiler()
    {
        CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
        return CUPTI_SUCCESS;
    }
};
