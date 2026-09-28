#pragma once

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {

// Contiguous layer partition for one pipeline stage group. Full-attention layers (every
// `full_attention_interval`-th layer) are distributed evenly across stages because they dominate
// KV payload bytes; GDN layers fill the remaining range. With one stage the partition is the
// identity over [0, layer_count).
struct PipelineStagePartition {
    int stages                 = 1;
    int layer_count            = 0;
    int full_attention_interval = 4;
    // Boundary table of size stages + 1: stage s owns global layers [first_layer(s),
    // first_layer(s + 1)). full_ordinals[s] is the global full-attention ordinal of the first
    // full-attention layer owned by stage s; full_ordinals[stages] is the total full count.
    std::vector<int> boundaries;
    std::vector<int> full_ordinals;

    [[nodiscard]] static PipelineStagePartition
    make(int layer_count, int full_attention_interval, int stages);

    [[nodiscard]] int stage_of_layer(int layer) const;
    [[nodiscard]] int first_layer(int stage) const;
    [[nodiscard]] int layers_in_stage(int stage) const;
    // Global full-attention-layer ordinal of the first full-attention layer owned by `stage`;
    // stages without a full-attention layer return the count of full layers before them.
    [[nodiscard]] int full_attention_base(int stage) const;
    [[nodiscard]] int full_attention_count(int stage) const;
    [[nodiscard]] int gdn_count(int stage) const;

private:
    void require_stage(int stage) const;
};

// Ordered set of pipeline stage devices. The stage-0 context is the Engine's primary device and
// keeps its existing meaning for every single-device code path.
class DeviceGroup {
public:
    // Device ids in stage order; every id must be distinct and valid.
    explicit DeviceGroup(std::vector<int> device_ids);

    [[nodiscard]] int size() const noexcept { return static_cast<int>(contexts_.size()); }
    [[nodiscard]] DeviceContext& at(int stage);
    [[nodiscard]] const DeviceContext& at(int stage) const;
    [[nodiscard]] DeviceContext& primary() { return at(0); }
    [[nodiscard]] const DeviceContext& primary() const { return at(0); }

    // All stages must expose the same compute capability: kernels are compiled per architecture
    // and a mixed group would fail at first launch with an opaque error.
    void require_uniform_compute_capability() const;

private:
    std::vector<std::unique_ptr<DeviceContext>> contexts_;
};

// One inter-stage activation channel. Copies traverse device -> pinned host -> device. All
// enqueue methods are asynchronous and stream-ordered: the source copy is enqueued on the source
// stage stream, and a completion event orders the destination copy on the destination stage
// stream. The staging area is reused across enqueues; correctness relies on the destination-side
// use of a channel being stream-ordered after the previous enqueue's destination-side use, which
// holds because every channel consumer runs its stages back-to-back on one host thread.
class PipelineTransport {
public:
    // `capacity_bytes` is the upper bound of any single transfer through this channel.
    PipelineTransport(const DeviceContext& from, const DeviceContext& to,
                      std::size_t capacity_bytes);
    ~PipelineTransport();
    PipelineTransport(const PipelineTransport&)            = delete;
    PipelineTransport& operator=(const PipelineTransport&) = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    // Copies `bytes` from `source` (stage `from`) to `destination` (stage `to`), ordered after
    // prior work on the source stream and ordered before subsequent destination-stream work.
    void enqueue(const void* source, void* destination, std::size_t bytes) const;

private:
    void require_capacity(std::size_t bytes) const;

    const DeviceContext& from_;
    const DeviceContext& to_;
    std::size_t capacity_bytes_ = 0;
    PinnedHostBuffer staging_;
    cudaEvent_t staged_ready_ = nullptr;
};

} // namespace ninfer
