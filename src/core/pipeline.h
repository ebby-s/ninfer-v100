#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"

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
    int stages                  = 1;
    int layer_count             = 0;
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
    // Global full-attention-layer ordinal of the first full-attention layer owned by `stage`.
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
    // Device ids in stage order; every id must be distinct and valid. Stage 0 owns fresh
    // contexts for the given ids.
    explicit DeviceGroup(std::vector<int> device_ids);
    // Engine-owned primary: stage 0 references the caller's context so every stage-0 kernel and
    // inter-stage transfer share one stream with the existing single-device code paths.
    DeviceGroup(DeviceContext& primary, std::vector<int> remaining_ids);

    [[nodiscard]] int size() const noexcept {
        return primary_ != nullptr ? static_cast<int>(contexts_.size()) + 1
                                   : static_cast<int>(contexts_.size());
    }
    [[nodiscard]] DeviceContext& at(int stage);
    [[nodiscard]] const DeviceContext& at(int stage) const;
    [[nodiscard]] DeviceContext& primary() { return at(0); }
    [[nodiscard]] const DeviceContext& primary() const { return at(0); }

    // All stages must expose the same compute capability: kernels are compiled per architecture
    // and a mixed group would fail at first launch with an opaque error.
    void require_uniform_compute_capability() const;

private:
    // Stage 0 aliases the Engine primary (never owned); stages 1.. own their contexts. Both
    // vectors index stage - 1 / stage respectively.
    DeviceContext* primary_ = nullptr;
    std::vector<std::unique_ptr<DeviceContext>> owned_;
    std::vector<std::unique_ptr<DeviceContext>> contexts_;  // stages 1.., mirrors owned_ for at()
};

// Whole-model pipeline execution context: the stage devices plus the layer partition. Code that
// is unaware of pipelines never sees this type; a null PipelineContext means single-stage
// execution on one device whose DeviceContext is the group's primary stage.
class PipelineContext {
public:
    // Stage 0 references `primary_device` (the Engine's own context) so boundary transfers share
    // the stream every stage-0 kernel uses; `remaining_ids` are stage-ordered and distinct.
    PipelineContext(DeviceContext& primary_device, std::vector<int> remaining_ids,
                    int layer_count, int full_attention_interval, bool embedding_replica);

    [[nodiscard]] const PipelineStagePartition& partition() const noexcept { return partition_; }
    [[nodiscard]] DeviceGroup& group() noexcept { return group_; }
    [[nodiscard]] const DeviceGroup& group() const noexcept { return group_; }
    [[nodiscard]] int stage_count() const noexcept { return group_.size(); }
    [[nodiscard]] DeviceContext& stage_device(int stage) { return group_.at(stage); }
    // Embedding replicas live on every non-primary stage when enabled: speculative decoding on
    // the last stage gathers draft-token embeddings locally instead of round-tripping the
    // primary device every draft step.
    [[nodiscard]] bool embedding_replica() const noexcept { return embedding_replica_; }
    [[nodiscard]] std::size_t embedding_replica_bytes() const noexcept {
        return embedding_replica_ ? embedding_replica_bytes_ : 0;
    }
    void set_embedding_replica_bytes(std::size_t bytes) { embedding_replica_bytes_ = bytes; }
    // Device pointer holding the embedding-table replica on `stage`; null when the stage holds
    // no replica (the primary stage uses the model's own embedding tensor).
    [[nodiscard]] void* embedding_replica_ptr(int stage) const;

    void publish_embedding_replica(int stage, DeviceBuffer buffer, Weight view);
    // A device-resident embedding replica with `view`'s metadata, valid for the stage the loader
    // published it to. The speculative draft loop gathers draft-token embeddings through this
    // view on its own stage device.
    [[nodiscard]] const Weight* embedding_replica_view(int stage) const;

private:
    DeviceGroup group_;
    PipelineStagePartition partition_;
    bool embedding_replica_              = false;
    std::size_t embedding_replica_bytes_ = 0;
    std::vector<DeviceBuffer> embedding_replicas_; // per stage; empty where no replica is held
    std::vector<Weight> embedding_replica_views_;  // parallel per-stage views
};

// One inter-stage activation channel. Copies traverse device -> pinned host -> device. All
// enqueue methods are asynchronous and stream-ordered: the source copy is enqueued on the source
// stage stream, and a completion event orders the destination copy on the destination stage
// stream. The staging area is reused across enqueues; correctness relies on the destination-side
// use of a channel being stream-ordered after the previous enqueue's destination-side use, which
// holds because every channel consumer runs its stages back-to-back on one host thread.
class PipelineTransport {
public:
    // `capacity_bytes` is the upper bound of any single transfer through this channel;
    // `slot_count` divides the staging area into `capacity_bytes / slot_count` private regions
    // addressed by the slot index (graph-captured rounds need one slot per staged copy).
    PipelineTransport(const DeviceContext& from, const DeviceContext& to,
                      std::size_t capacity_bytes, std::size_t slot_count = 1);
    ~PipelineTransport();
    PipelineTransport(const PipelineTransport&)            = delete;
    PipelineTransport& operator=(const PipelineTransport&) = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    // Copies `bytes` from `source` (stage `from`) to `destination` (stage `to`), ordered after
    // prior work on the source stream and ordered before subsequent destination-stream work.
    void enqueue(const void* source, void* destination, std::size_t bytes) const;

    // CUDA-graph capture support. A captured round crosses devices between two graphs, so the
    // copy splits into a source-side D2H (captured into the source graph) and a destination-side
    // H2D (captured into the destination graph); the event orchestration is host-side between
    // graph launches instead. `slot` selects a private staging region so every copy inside one
    // captured graph keeps its own staging area.
    void enqueue_stage_out(const void* source, std::size_t bytes, std::size_t slot) const;
    void enqueue_stage_in(void* destination, std::size_t bytes, std::size_t slot) const;
    [[nodiscard]] void* staging(std::size_t slot) const;

private:
    void require_capacity(std::size_t bytes) const;

    const DeviceContext& from_;
    const DeviceContext& to_;
    std::size_t capacity_bytes_ = 0;
    PinnedHostBuffer staging_;
    cudaEvent_t staged_ready_   = nullptr;  // source copy complete
    cudaEvent_t consumed_ready_ = nullptr;  // staging area fully read by the destination side
    mutable bool have_consumed_ = false;
    std::size_t slot_count_     = 1;
};

} // namespace ninfer
