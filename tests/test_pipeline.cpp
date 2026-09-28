#include "core/pipeline.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_partition(int layer_count, int interval, int stages) {
    const ninfer::PipelineStagePartition partition =
        ninfer::PipelineStagePartition::make(layer_count, interval, stages);

    int full_total = 0;
    int layer_sum  = 0;
    for (int stage = 0; stage < stages; ++stage) {
        expect(partition.layers_in_stage(stage) > 0, "every stage owns at least one layer");
        expect(partition.full_attention_count(stage) >= 0, "non-negative full count");
        expect(partition.gdn_count(stage) >= 0, "non-negative GDN count");
        expect(partition.layers_in_stage(stage) ==
                   partition.full_attention_count(stage) + partition.gdn_count(stage),
               "GDN + full counts cover the stage range");
        layer_sum  += partition.layers_in_stage(stage);
        full_total += partition.full_attention_count(stage);
        for (int layer = partition.first_layer(stage);
             layer < partition.first_layer(stage) + partition.layers_in_stage(stage); ++layer) {
            expect(partition.stage_of_layer(layer) == stage, "stage_of_layer round-trips");
        }
        // Stage ranges are contiguous and ordered.
        if (stage + 1 < stages) {
            expect(partition.first_layer(stage) + partition.layers_in_stage(stage) ==
                       partition.first_layer(stage + 1),
                   "stage ranges are contiguous");
        }
    }
    expect(layer_sum == layer_count, "stages cover every layer exactly once");
    expect(full_total == layer_count / interval, "stages cover every full-attention layer once");

    // Full-attention layers are as even as possible across stages.
    const int full_per_stage = layer_count / interval / stages;
    const int remainder      = layer_count / interval % stages;
    for (int stage = 0; stage < stages; ++stage) {
        expect(partition.full_attention_count(stage) ==
                   full_per_stage + (stage < remainder ? 1 : 0),
               "full-attention layers split evenly");
    }
}

/// Fills a source buffer with a tag pattern, transports it, and verifies the destination bytes.
int check_transport_roundtrip() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }

    // Two devices exercise the pinned-staging path; one device exercises same-device ordering
    // with two independent contexts on that device.
    const int second = count > 1 ? 1 : 0;
    ninfer::DeviceContext from(0);
    ninfer::DeviceContext to(second);
    ninfer::PipelineTransport transport(from, to, 1 << 20);

    constexpr std::size_t kBytes = 64 * 1024;
    std::vector<std::uint32_t> host(kBytes / sizeof(std::uint32_t));
    for (std::size_t i = 0; i < host.size(); ++i) { host[i] = static_cast<std::uint32_t>(i); }

    void* source      = nullptr;
    void* destination = nullptr;
    if (cudaSetDevice(0) != cudaSuccess || cudaMalloc(&source, kBytes) != cudaSuccess) {
        std::cerr << "FAIL: source allocation\n";
        return 1;
    }
    if (cudaSetDevice(second) != cudaSuccess || cudaMalloc(&destination, kBytes) != cudaSuccess) {
        std::cerr << "FAIL: destination allocation\n";
        return 1;
    }
    cudaMemcpy(source, host.data(), kBytes, cudaMemcpyHostToDevice);
    cudaMemset(destination, 0, kBytes);

    // Transport across the two stage contexts, then verify on the host.
    transport.enqueue(source, destination, kBytes);
    cudaError_t sync = cudaSetDevice(second);
    if (sync == cudaSuccess) { sync = cudaDeviceSynchronize(); }
    if (sync != cudaSuccess) {
        std::cerr << "FAIL: device synchronize: " << cudaGetErrorString(sync) << '\n';
        return 1;
    }

    std::vector<std::uint32_t> echoed(host.size(), 0);
    cudaMemcpy(echoed.data(), destination, kBytes, cudaMemcpyDeviceToHost);
    expect(std::memcmp(host.data(), echoed.data(), kBytes) == 0,
           "transported bytes match the source pattern");

    // Oversized transfers must be rejected: one byte beyond the transport capacity.
    bool threw = false;
    try {
        transport.enqueue(source, destination, transport.capacity_bytes() + 1);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "oversized transfer throws");

    cudaFree(source);
    cudaSetDevice(second);
    cudaFree(destination);
    return 0;
}

} // namespace

int main() {
    // Single stage: identity partition.
    const auto single = ninfer::PipelineStagePartition::make(64, 4, 1);
    expect(single.stage_of_layer(0) == 0, "single stage owns layer 0");
    expect(single.stage_of_layer(63) == 0, "single stage owns the last layer");
    expect(single.layers_in_stage(0) == 64, "single stage owns every layer");
    expect(single.full_attention_count(0) == 16, "single stage owns every full layer");

    // Two stages over the 27B topology: balanced full-attention layers, contiguous ranges.
    check_partition(64, 4, 2);
    const auto two = ninfer::PipelineStagePartition::make(64, 4, 2);
    expect(two.first_layer(0) == 0 && two.layers_in_stage(0) == 32, "two-stage split at 32");
    expect(two.full_attention_base(1) == 8, "stage 1 begins at full ordinal 8");

    // The 35B-A3B topology and an uneven full split.
    check_partition(40, 4, 2);
    const auto forty = ninfer::PipelineStagePartition::make(40, 4, 2);
    expect(forty.layers_in_stage(0) == 20 && forty.layers_in_stage(1) == 20,
           "40-layer model splits 20/20");

    check_partition(64, 4, 4);
    const auto four = ninfer::PipelineStagePartition::make(64, 4, 4);
    expect(four.layers_in_stage(0) == 16, "four stages own 16 layers each");

    // Non-divisible tails: GDN-only trailing layers fold into the last stage.
    check_partition(66, 4, 2);
    check_partition(64, 4, 8);
    check_partition(7, 4, 1);

    // Invalid requests throw.
    for (const auto& [layers, interval, stages] :
         {std::tuple{0, 4, 1}, {64, 4, 0}, {64, 4, 17}, {3, 4, 2}}) {
        bool threw = false;
        try {
            ninfer::PipelineStagePartition::make(layers, interval, stages);
        } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "invalid partition request throws");
    }

    const int transport = check_transport_roundtrip();
    if (transport != 0) { return transport; }

    if (failures != 0) {
        std::cerr << failures << " pipeline check(s) failed\n";
        return 1;
    }
    std::cout << "pipeline tests passed\n";
    return 0;
}
