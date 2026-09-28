#include "core/pipeline.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

namespace ninfer {
namespace {

[[noreturn]] void pipeline_error(const std::string& message) {
    throw std::invalid_argument("pipeline: " + message);
}

} // namespace

PipelineStagePartition PipelineStagePartition::make(int layer_count,
                                                    int full_attention_interval, int stages) {
    if (layer_count <= 0) { pipeline_error("layer count must be positive"); }
    if (full_attention_interval <= 0) {
        pipeline_error("full-attention interval must be positive");
    }
    if (stages <= 0) { pipeline_error("stage count must be positive"); }

    const int full_layers = layer_count / full_attention_interval;
    if (stages > std::max(full_layers, 1)) {
        pipeline_error("stage count exceeds full-attention layer count");
    }

    PipelineStagePartition partition;
    partition.stages                  = stages;
    partition.layer_count             = layer_count;
    partition.full_attention_interval = full_attention_interval;
    partition.boundaries.reserve(static_cast<std::size_t>(stages) + 1);
    partition.full_ordinals.reserve(static_cast<std::size_t>(stages) + 1);

    // Stage s owns full-attention ordinals [full_ordinals[s], full_ordinals[s + 1]); the first
    // owned layer follows the previous full layer, so ranges balance the KV-heavy attention
    // layers and keep GDN layers contiguous.
    const int base   = full_layers / stages;
    const int extra  = full_layers % stages;
    partition.full_ordinals.push_back(0);
    partition.boundaries.push_back(0);
    for (int stage = 0; stage < stages; ++stage) {
        const int end_ordinal = (stage + 1) * base + std::min(stage + 1, extra);
        partition.full_ordinals.push_back(end_ordinal);
        partition.boundaries.push_back(end_ordinal == full_layers
                                           ? layer_count
                                           : end_ordinal * full_attention_interval);
    }
    return partition;
}

int PipelineStagePartition::stage_of_layer(int layer) const {
    if (layer < 0 || layer >= layer_count) { pipeline_error("layer out of range"); }
    int stage = 0;
    while (stage + 1 < stages && boundaries[static_cast<std::size_t>(stage) + 1] <= layer) {
        ++stage;
    }
    return stage;
}

int PipelineStagePartition::first_layer(int stage) const {
    require_stage(stage);
    return boundaries[static_cast<std::size_t>(stage)];
}

int PipelineStagePartition::layers_in_stage(int stage) const {
    require_stage(stage);
    return boundaries[static_cast<std::size_t>(stage) + 1] -
           boundaries[static_cast<std::size_t>(stage)];
}

int PipelineStagePartition::full_attention_base(int stage) const {
    require_stage(stage);
    return full_ordinals[static_cast<std::size_t>(stage)];
}

int PipelineStagePartition::full_attention_count(int stage) const {
    require_stage(stage);
    return full_ordinals[static_cast<std::size_t>(stage) + 1] -
           full_ordinals[static_cast<std::size_t>(stage)];
}

int PipelineStagePartition::gdn_count(int stage) const {
    return layers_in_stage(stage) - full_attention_count(stage);
}

void PipelineStagePartition::require_stage(int stage) const {
    if (stage < 0 || stage >= stages) {
        pipeline_error("stage out of range (requested " + std::to_string(stage) + ", stages " +
                       std::to_string(stages) + ", layers " + std::to_string(layer_count) + ")");
    }
}

PipelineContext::PipelineContext(DeviceContext& primary_device, std::vector<int> remaining_ids,
                                 int layer_count, int full_attention_interval,
                                 bool embedding_replica)
    : group_(primary_device, std::move(remaining_ids)),
      partition_(PipelineStagePartition::make(layer_count, full_attention_interval,
                                              static_cast<int>(group_.size()))),
      embedding_replica_(embedding_replica) {
    group_.require_uniform_compute_capability();
    std::fprintf(stderr, "DBG PipelineContext: stages=%d remaining=%zu\n", partition_.stages,
                 remaining_ids.size());
    if (embedding_replica_ && partition_.stages < 2) {
        pipeline_error("embedding replica requires at least two stages");
    }
    if (embedding_replica_) { embedding_replicas_.resize(static_cast<std::size_t>(partition_.stages)); }
}

void* PipelineContext::embedding_replica_ptr(int stage) const {
    if (!embedding_replica_ || stage < 0 || stage >= partition_.stages) { return nullptr; }
    const DeviceBuffer& buffer =
        embedding_replicas_[static_cast<std::size_t>(stage)];
    return buffer.bytes != 0 ? buffer.p : nullptr;
}

void PipelineContext::publish_embedding_replica(int stage, DeviceBuffer buffer) {
    if (!embedding_replica_) { pipeline_error("embedding replica is disabled"); }
    if (stage <= 0 || stage >= partition_.stages) {
        pipeline_error("embedding replica stage must be a non-primary stage");
    }
    embedding_replicas_[static_cast<std::size_t>(stage)] = std::move(buffer);
}

DeviceGroup::DeviceGroup(std::vector<int> device_ids) {
    if (device_ids.empty()) { pipeline_error("device list is empty"); }
    std::vector<int> sorted = device_ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        pipeline_error("duplicate device id");
    }
    contexts_.reserve(device_ids.size());
    for (const int id : device_ids) {
        contexts_.push_back(std::make_unique<DeviceContext>(id));
    }
}

DeviceGroup::DeviceGroup(DeviceContext& primary, std::vector<int> remaining_ids) {
    primary_ = &primary;
    std::vector<int> sorted = remaining_ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        pipeline_error("duplicate device id");
    }
    owned_.reserve(sorted.size());
    contexts_.reserve(sorted.size());
    for (const int id : sorted) {
        contexts_.push_back(std::make_unique<DeviceContext>(id));
    }
}

DeviceContext& DeviceGroup::at(int stage) {
    if (stage < 0 || stage >= size()) {
        pipeline_error("stage out of range (DeviceGroup::at requested " + std::to_string(stage) +
                       ", size " + std::to_string(size()) + ")");
    }
    return stage == 0 ? *primary_ : *contexts_[static_cast<std::size_t>(stage) - 1];
}

const DeviceContext& DeviceGroup::at(int stage) const {
    if (stage < 0 || stage >= size()) { pipeline_error("stage out of range"); }
    return stage == 0 ? *primary_ : *contexts_[static_cast<std::size_t>(stage) - 1];
}

void DeviceGroup::require_uniform_compute_capability() const {
    const int capability = primary_->compute_capability();
    if (contexts_.empty()) { return; }
    for (const auto& context : contexts_) {
        if (context->compute_capability() != capability) {
            pipeline_error("pipeline stages mix compute capabilities (" +
                           std::to_string(capability) + " vs " +
                           std::to_string(context->compute_capability()) + ")");
        }
    }
}

PipelineTransport::PipelineTransport(const DeviceContext& from, const DeviceContext& to,
                                     std::size_t capacity_bytes)
    : from_(from), to_(to), capacity_bytes_(capacity_bytes), staging_(capacity_bytes) {
    if (capacity_bytes == 0) { pipeline_error("transport capacity must be positive"); }
    // The staging event is recorded on the source stream, so it must be created with the source
    // device current.
    from_.bind_to_current_thread();
    if (cudaEventCreateWithFlags(&staged_ready_, cudaEventDisableTiming) != cudaSuccess) {
        pipeline_error("failed to create staging event");
    }
    // The consumed event is recorded on the destination stream, so it belongs to the destination
    // device's context.
    to_.bind_to_current_thread();
    if (cudaEventCreateWithFlags(&consumed_ready_, cudaEventDisableTiming) != cudaSuccess) {
        pipeline_error("failed to create consumed event");
    }
}

PipelineTransport::~PipelineTransport() {
    from_.bind_to_current_thread_noexcept();
    if (staged_ready_ != nullptr) { cudaEventDestroy(staged_ready_); }
    to_.bind_to_current_thread_noexcept();
    if (consumed_ready_ != nullptr) { cudaEventDestroy(consumed_ready_); }
}

void PipelineTransport::require_capacity(std::size_t bytes) const {
    if (bytes > capacity_bytes_) {
        pipeline_error("transfer exceeds transport capacity (" + std::to_string(bytes) + " > " +
                       std::to_string(capacity_bytes_) + ")");
    }
}

void PipelineTransport::enqueue(const void* source, void* destination, std::size_t bytes) const {
    if (bytes == 0) { return; }
    require_capacity(bytes);

    from_.bind_to_current_thread();
    // The staging area is reused across enqueues; the source side must not overwrite it until the
    // previous destination-side copy has consumed it.
    if (have_consumed_) { CUDA_CHECK(cudaStreamWaitEvent(from_.stream, consumed_ready_, 0)); }
    if (from_.device == to_.device) {
        // Same-device stages (single-stage and tests): one D2D copy on the source stream.
        CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice,
                                   from_.stream));
    } else {
        // Cross-device stages: stage through pinned host memory. The two V100-PCIe slots on the
        // reference host expose no peer access; a peer-direct fast path can replace this branch
        // once hardware that reports cudaDeviceCanAccessPeer is the target.
        CUDA_CHECK(cudaMemcpyAsync(staging_.data(), source, bytes, cudaMemcpyDeviceToHost,
                                   from_.stream));
    }
    CUDA_CHECK(cudaEventRecord(staged_ready_, from_.stream));

    to_.bind_to_current_thread();
    CUDA_CHECK(cudaStreamWaitEvent(to_.stream, staged_ready_, 0));
    if (from_.device != to_.device) {
        CUDA_CHECK(cudaMemcpyAsync(destination, staging_.data(), bytes, cudaMemcpyHostToDevice,
                                   to_.stream));
        CUDA_CHECK(cudaEventRecord(consumed_ready_, to_.stream));
        have_consumed_ = true;
    }
}

} // namespace ninfer
