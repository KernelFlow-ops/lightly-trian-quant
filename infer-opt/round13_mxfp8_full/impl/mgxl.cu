// Round 13: copy of round10_lowp_probe/impl/mg16l.cu (generic cubin launcher: cudaLibrary + cudaLaunchKernelExC on torch's current stream, optional PDL).
// Generic launcher for cubin kernels built by mgb.py (cudaLibrary API, launches on torch's current stream).
// Supports programmatic dependent launch (PDL) via the launch attribute.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

struct Kern { cudaLibrary_t lib; cudaKernel_t kern; size_t smem; };
static std::vector<Kern> g_k;

int64_t load(const std::string& path, const std::string& name, int64_t smem) {
    Kern k;
    cudaError_t e = cudaLibraryLoadFromFile(&k.lib, path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    TORCH_CHECK(e == cudaSuccess, "cudaLibraryLoadFromFile: ", cudaGetErrorString(e));
    e = cudaLibraryGetKernel(&k.kern, k.lib, name.c_str());
    TORCH_CHECK(e == cudaSuccess, "cudaLibraryGetKernel: ", cudaGetErrorString(e));
    int dev; cudaGetDevice(&dev);
    e = cudaKernelSetAttributeForDevice(k.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem, dev);
    TORCH_CHECK(e == cudaSuccess, "set smem attr: ", cudaGetErrorString(e));
    k.smem = smem;
    g_k.push_back(k);
    return (int64_t)g_k.size() - 1;
}

int64_t occupancy(int64_t h, int64_t threads) {
    int nb = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)g_k[h].kern, (int)threads, g_k[h].smem);
    if (e != cudaSuccess) { cudaGetLastError(); return -1; }
    return nb;
}

// params: raw bytes of the kernel's by-value parameter struct
void launch(int64_t h, int64_t gx, int64_t gy, int64_t gz, int64_t block, const py::bytes& params, bool pdl) {
    std::string buf = params;
    void* args[] = {(void*)buf.data()};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(gx, gy, gz); cfg.blockDim = dim3(block); cfg.dynamicSmemBytes = g_k[h].smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    cudaError_t e = cudaLaunchKernelExC(&cfg, (const void*)g_k[h].kern, args);
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load", &load);
    m.def("occupancy", &occupancy);
    m.def("launch", &launch);
}
