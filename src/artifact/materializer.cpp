#include "artifact/materializer.h"

#include "core/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;
constexpr int kMaximumPipelineStages    = 16;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

// One pinned staging slot. A chunk read into the slot fans out to one H2D copy per stage stream;
// the slot records one completion event per stream it fed and cannot be refilled until every
// event completes.
class Slot {
public:
    // One event per copy stream; each is created with its stream's device current because CUDA
    // events live on the creating device's context.
    explicit Slot(std::size_t bytes, const std::vector<DeviceContext*>& copy_devices)
        : buffer(bytes) {
        events_.reserve(copy_devices.size());
        for (DeviceContext* device : copy_devices) {
            device->bind_to_current_thread();
            cudaEvent_t event = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            events_.push_back(event);
        }
    }

    ~Slot() {
        if (pending) {
            for (cudaEvent_t event : events_) { (void)cudaEventSynchronize(event); }
        }
        for (cudaEvent_t event : events_) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            for (cudaEvent_t event : pending_events_) { (void)cudaEventSynchronize(event); }
            pending_events_.clear();
            pending = false;
        }
    }

    // Records completion of this slot's copies on `stream`. At most one event per stream per
    // fill; repeated records on the same stream are deduplicated by the caller's fan-out order.
    void record(cudaStream_t stream, int stream_index) {
        CUDA_CHECK(cudaEventRecord(events_[static_cast<std::size_t>(stream_index)], stream));
        if (std::find(pending_events_.begin(), pending_events_.end(),
                      events_[static_cast<std::size_t>(stream_index)]) == pending_events_.end()) {
            pending_events_.push_back(events_[static_cast<std::size_t>(stream_index)]);
        }
        pending = true;
    }

    PinnedHostBuffer buffer;
    bool pending = false;

private:
    std::vector<cudaEvent_t> events_;        // one per known copy stream
    std::vector<cudaEvent_t> pending_events_; // events recorded for the current fill
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
    int stage                  = 0;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (stage_arenas_.empty()) { throw ArtifactError("artifact has no device tensor backing"); }
    return *stage_arenas_.front();
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer) {
    return materialize(reader, plan, device, startup_observer, {}, {});
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer,
                                 const MaterializationStageMap& stage_map,
                                 const MaterializationStageDevice& stage_device) {
    const bool staged = static_cast<bool>(stage_map);
    if (staged != static_cast<bool>(stage_device)) {
        throw ArtifactError("pipeline placement requires both stage map and stage devices");
    }
    const StartupObserver no_startup_observer;
    const StartupObserver& startup =
        startup_observer == nullptr ? no_startup_observer : *startup_observer;
    std::uint64_t total = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
    }
    StartupPhaseScope materialize_phase(startup, StartupPhase::WeightsMaterialize,
                                        StartupProgressUnit::Bytes, total);
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }

    // Resolve per-object stages and per-stage subtotals. Without pipeline placement everything
    // belongs to the single primary stage.
    std::vector<int> object_stages;
    object_stages.reserve(plan.device_objects.size());
    std::vector<std::uint64_t> stage_subtotals;
    int stage_count = 1;
    if (staged) {
        for (const DeviceMaterialization& placement : plan.device_objects) {
            const int stage = stage_map(reader.objects().at(placement.object.index));
            if (stage < 0 || stage >= kMaximumPipelineStages) {
                throw ArtifactError("pipeline stage map returned an invalid stage");
            }
            stage_count = std::max(stage_count, stage + 1);
            object_stages.push_back(stage);
        }
        stage_subtotals.assign(static_cast<std::size_t>(stage_count), 0);
        for (std::size_t i = 0; i < plan.device_objects.size(); ++i) {
            const DeviceMaterialization& placement = plan.device_objects[i];
            const std::uint64_t aligned =
                align_up(placement.bytes, placement.alignment == 0 ? 1 : placement.alignment,
                         "artifact tensor aligned size overflows u64");
            stage_subtotals[static_cast<std::size_t>(object_stages[i])] =
                checked_add(stage_subtotals[static_cast<std::size_t>(object_stages[i])], aligned,
                            "stage tensor byte count overflows u64");
        }
    } else {
        object_stages.assign(plan.device_objects.size(), 0);
        stage_subtotals.assign(1, capacity);
    }

    // Allocate one backing arena per used stage on that stage's device. Without placement this
    // is the historical single cudaMalloc on `device`. Stage subtotals include each object's
    // alignment slack because arenas bump-allocate aligned spans.
    std::vector<std::unique_ptr<DeviceArena>> stage_arenas;
    stage_arenas.reserve(static_cast<std::size_t>(stage_count));
    for (int stage = 0; stage < stage_count; ++stage) {
        DeviceContext& target = staged ? stage_device(stage) : device;
        target.bind_to_current_thread();
        if (staged) {
            std::size_t free_bytes  = 0;
            std::size_t total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const auto required =
                static_cast<std::size_t>(stage_subtotals[static_cast<std::size_t>(stage)]);
            if (required > free_bytes) {
                throw ArtifactError("pipeline stage " + std::to_string(stage) + " requires " +
                                    std::to_string(required) +
                                    " weight bytes, but only " + std::to_string(free_bytes) +
                                    " are free on its device");
            }
        }
        stage_arenas.push_back(std::make_unique<DeviceArena>(
            static_cast<std::size_t>(stage_subtotals[static_cast<std::size_t>(stage)])));
    }
    out.stage_arenas_ = std::move(stage_arenas);
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size());
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    for (std::size_t i = 0; i < plan.device_objects.size(); ++i) {
        const DeviceMaterialization& placement = plan.device_objects[i];
        const int stage                        = object_stages[i];
        DeviceArena& arena                     = *out.stage_arenas_[static_cast<std::size_t>(stage)];
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        DeviceSpan storage =
            arena.alloc_bytes(static_cast<std::size_t>(placement.bytes),
                              static_cast<std::size_t>(placement.alignment));
        if (payload.data.size() != placement.bytes) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;
        ranges.push_back(CopyRange{
            .source_begin = payload.absolute_offset,
            .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                        "artifact tensor source range overflows u64"),
            .destination  = static_cast<std::byte*>(storage.data),
            .stage        = stage,
        });
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    const std::uint64_t staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;
    StartupPhaseScope staging_phase(startup, StartupPhase::WeightsStagingPin,
                                    StartupProgressUnit::Bytes, staging_bytes);
    device.bind_to_current_thread();
    std::vector<DeviceContext*> copy_devices;
    copy_devices.reserve(static_cast<std::size_t>(stage_count));
    for (int stage = 0; stage < stage_count; ++stage) {
        copy_devices.push_back(staged ? &stage_device(stage) : &device);
    }
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes, copy_devices));
    }
    staging_phase.complete(staging_bytes, staging_bytes);
    out.stats_.peak_staging_bytes = staging_bytes;
    materialize_phase.progress(0, total);

    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    const auto start       = std::chrono::steady_clock::now();
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            slot.wait();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

            // One fan-out per stage stream touched by this chunk; the slot records one event per
            // stream so refill waits every consumer.
            bool stream_touched[kMaximumPipelineStages] = {};
            while (next_range < ranges.size() && ranges[next_range].source_end <= source) {
                ++next_range;
            }
            std::size_t range_index = next_range;
            while (range_index < ranges.size() && ranges[range_index].source_begin < chunk_end) {
                const CopyRange& range         = ranges[range_index];
                DeviceContext& target = staged ? stage_device(range.stage) : device;
                const std::uint64_t copy_begin = std::max(source, range.source_begin);
                const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                if (copy_begin < copy_end) {
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    target.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(slot.buffer.data()) +
                            static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, target.transfer_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                    stream_touched[range.stage] = true;
                }
                if (range.source_end <= chunk_end) {
                    ++range_index;
                } else {
                    break;
                }
            }
            next_range = range_index;
            for (int stage = 0; stage < stage_count; ++stage) {
                if (stream_touched[stage]) {
                    DeviceContext& target = staged ? stage_device(stage) : device;
                    slot.record(target.transfer_stream, stage);
                }
            }
            slot.pending = true;

            if (copied != last_published && copied < total) {
                last_published = copied;
                materialize_phase.progress(copied, total);
            }
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    for (int stage = 0; stage < stage_count; ++stage) {
        DeviceContext& target = staged ? stage_device(stage) : device;
        target.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(target.transfer_stream));
    }
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    slots.clear();
    materialize_phase.complete(copied, total);
    return out;
}

} // namespace ninfer::artifact
