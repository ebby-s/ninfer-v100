// Opt-in real-artifact check for pipeline weight placement. When a Qwen3.8-27B NVFP4 artifact is
// available (NINFER_QWEN3_8_27B_NVFP4_WEIGHTS) and two CUDA devices are visible, loading with
// pipeline_size=2 must split the weight bytes across the two stage devices and publish the
// token-embedding replica on the non-primary stage. Generation execution is out of scope here;
// this protects the placement contract that pipeline execution builds on.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::size_t free_device_bytes(int device) {
    cudaSetDevice(device);
    std::size_t free_bytes = 0;
    std::size_t total      = 0;
    if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess) { return 0; }
    return free_bytes;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
    if (artifact == nullptr) {
        std::cout << "SKIP: NINFER_QWEN3_8_27B_NVFP4_WEIGHTS is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "SKIP: pipeline placement needs two visible CUDA devices\n";
        return 77;
    }

    const std::size_t stage0_before = free_device_bytes(0);
    const std::size_t stage1_before = free_device_bytes(1);

    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.device          = 0;
    options.pipeline_size   = 2;
    options.pipeline_devices = {0, 1};
    // Keep the default pools small: this check owns placement, not capacity.
    options.max_context     = 2048;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.max_concurrency = 1;
    // v1 pipeline contract: eager decode (per-stage graph capture is deferred).
    options.use_cuda_graph  = false;

    std::size_t stage0_during = 0;
    std::size_t stage1_during = 0;
    try {
        const ninfer::Engine engine(options);
        stage0_during = free_device_bytes(0);
        stage1_during = free_device_bytes(1);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: pipeline load threw: " << error.what() << '\n';
        return 1;
    }

    const auto stage0_used = stage0_before - stage0_during;
    const auto stage1_used = stage1_before - stage1_during;
    std::cout << "stage0 used " << (stage0_used >> 20) << " MiB, stage1 used "
              << (stage1_used >> 20) << " MiB\n";

    int failures = 0;
    const auto expect = [&](bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    };
    expect(stage1_used > (std::size_t{4} << 30),
           "stage 1 must hold its weight share plus the embedding replica (got " +
               std::to_string(stage1_used >> 20) + " MiB)");
    expect(stage0_used > (std::size_t{4} << 30),
           "stage 0 must hold its weight share plus runtime pools (got " +
               std::to_string(stage0_used >> 20) + " MiB)");
    // The embedding replica (~2.4 GiB bf16 for the 248320x5120 table) lives only on stage 1.
    expect(stage1_used > stage0_used / 2,
           "stage 1 carries a substantial share of the model");

    if (failures != 0) { return 1; }
    std::cout << "pipeline placement check passed\n";
    return 0;
}
