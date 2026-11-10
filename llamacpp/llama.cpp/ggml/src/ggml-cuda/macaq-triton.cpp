#include "macaq-triton.h"
#include "ggml-impl.h"
#include <cuda.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace {
struct kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    unsigned int warps = 0, shared = 0, groups = 0;
    int pdl = 0;
};
using kernel_set = std::array<kernel, 4>;
thread_local kernel_set * active = nullptr;

int index(bool quantized, bool routed = false) {
    return (routed ? 2 : 0) + (quantized ? 1 : 0);
}

void check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char * message = nullptr;
        cuGetErrorString(result, &message);
        GGML_ABORT("MACaQ Triton launch failed: %s", message ? message : "unknown driver error");
    }
}
}

void ggml_macaq_triton_prepare() {
    active = nullptr;
    const char * directory = std::getenv("MACAQ_TRITON_KERNEL_DIR");
    const char * mode = std::getenv("MACAQ_TRITON_MODE");
    if (!directory || !*directory || (mode && std::string(mode) == "off")) {
        return;
    }
    CUcontext context = nullptr;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) {
        return;
    }
    static std::mutex mutex;
    static std::map<std::pair<CUcontext, std::string>, std::unique_ptr<kernel_set>> cache;
    std::lock_guard<std::mutex> guard(mutex);
    const auto key = std::make_pair(context, std::string(directory));
    auto found = cache.find(key);
    if (found != cache.end()) {
        active = found->second.get();
        return;
    }
    auto kernels = std::make_unique<kernel_set>();
    CUdevice device;
    int major = 0, minor = 0;
    check(cuCtxGetDevice(&device));
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    int loaded[2] = {0, 0};
    for (bool routed : {false, true}) {
        const int block = 64;
        for (bool quantized : {false, true}) {
            auto & entry = (*kernels)[index(quantized, routed)];
            const std::string stem = key.second + "/h" + std::to_string(block) + (routed ? "-moe" : "") + (quantized ? "-q8" : "-f32");
            std::ifstream info(stem + ".info");
            int abi = 0, arch = 0;
            std::string name;
            if (!(info >> abi >> arch >> entry.warps >> entry.shared >> entry.groups >> entry.pdl >> name) ||
                    abi != (routed ? 3 : 2) || arch != major * 10 + minor ||
                    (entry.pdl != 0 && entry.pdl != 1) || (entry.pdl && (arch < 90 || CUDA_VERSION < 12000)) ||
                    (entry.warps != 1 && entry.warps != 2 && entry.warps != 4 && entry.warps != 8) ||
                    entry.groups != entry.warps || entry.shared > 48 * 1024) {
                continue;
            }
            if (cuModuleLoad(&entry.module, (stem + ".cubin").c_str()) != CUDA_SUCCESS) {
                continue;
            }
            if (cuModuleGetFunction(&entry.function, entry.module, name.c_str()) != CUDA_SUCCESS) {
                entry.function = nullptr;
                cuModuleUnload(entry.module);
                entry.module = nullptr;
                continue;
            }
            loaded[routed ? 1 : 0]++;
        }
    }
    GGML_LOG_INFO("MACaQ Triton: loaded %d/2 dense and %d/2 routed kernels from %s\n", loaded[0], loaded[1], directory);
    active = kernels.get();
    cache.emplace(key, std::move(kernels));
}

bool ggml_macaq_triton_available(int block, bool quantized) {
    const char * mode = std::getenv("MACAQ_TRITON_MODE");
    return active && block == 64 &&
        !(quantized && mode && std::string(mode) == "f32") &&
        (*active)[index(quantized)].function;
}

static void launch(kernel & entry, void ** arguments, int width, int rows, int padded,
                   int block, bool quantized, void * stream) {
    const int extent = quantized ? padded : width;
    const unsigned int grid = (extent + block * entry.groups - 1) / (block * entry.groups);
    CUresult result;
    if (entry.pdl) {
#if CUDA_VERSION >= 12000
        CUlaunchAttribute attr{};
        attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
        CUlaunchConfig config{};
        config.gridDimX = grid; config.gridDimY = rows; config.gridDimZ = 1;
        config.blockDimX = 32 * entry.warps; config.blockDimY = 1; config.blockDimZ = 1;
        config.sharedMemBytes = entry.shared;
        config.hStream = (CUstream) stream;
        config.attrs = &attr; config.numAttrs = 1;
        result = cuLaunchKernelEx(&config, entry.function, arguments, nullptr);
#else
        GGML_ABORT("MACaQ dependent launch requires toolkit 12 or newer");
#endif
    } else {
        result = cuLaunchKernel(entry.function, grid, rows, 1, 32 * entry.warps, 1, 1,
                                entry.shared, (CUstream) stream, arguments, nullptr);
    }
    if (result != CUDA_SUCCESS) {
        GGML_LOG_ERROR("MACaQ launch: block=%d q8=%d width=%d rows=%d padded=%d grid=%u warps=%u shared=%u\n",
                       block, quantized, width, rows, padded, grid, entry.warps, entry.shared);
    }
    check(result);
}

void ggml_macaq_triton_launch(const void * x, const void * scales, void * y, void * packed,
                             int width, int rows, int padded, int block,
                             bool quantized, void * stream) {
    auto & entry = (*active)[index(quantized)];
    void * scratch = nullptr;
    void * arguments[] = {&x, &scales, &y, &packed, &width, &rows, &padded, &scratch, &scratch};
    launch(entry, arguments, width, rows, padded, block, quantized, stream);
}

bool ggml_macaq_triton_moe_available(bool quantized) {
    const char * mode = std::getenv("MACAQ_TRITON_MODE");
    return active && !(quantized && mode && std::string(mode) == "f32") &&
        (*active)[index(quantized, true)].function;
}

void ggml_macaq_triton_moe_launch(const void * x, const void * scales, const void * ids,
                                 void * y, void * packed, int width, int tokens, int routes,
                                 int input_routes, int padded, int id_stride, int experts,
                                 bool quantized, void * stream) {
    auto & entry = (*active)[index(quantized, true)];
    void * scratch = nullptr;
    void * arguments[] = {&x, &scales, &ids, &y, &packed, &width, &tokens, &routes,
                         &input_routes, &padded, &id_stride, &experts, &scratch, &scratch};
    launch(entry, arguments, width, tokens * routes, padded, 64, quantized, stream);
}
