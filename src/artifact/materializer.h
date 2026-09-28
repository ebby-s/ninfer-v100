#pragma once

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::artifact {

struct MaterializationStats {
    std::uint64_t file_bytes              = 0;
    std::uint64_t h2d_bytes               = 0;
    std::uint64_t device_capacity_bytes   = 0;
    std::uint64_t retained_resource_bytes = 0;
    std::uint64_t peak_staging_bytes      = 0;
    std::size_t tensor_count              = 0;
    std::size_t resource_count            = 0;
    double upload_seconds                 = 0.0;
};

// Pipeline placement hooks. When both are supplied, each device tensor's bytes are materialized
// into the backing arena of `stage_map(descriptor)` on `stage_device(stage)`; otherwise every
// tensor lands on the single `device` arena. The stage map must return a nonnegative stage for
// every device object.
using MaterializationStageMap    = std::function<int(const ObjectDescriptor&)>;
using MaterializationStageDevice = std::function<DeviceContext&(int)>;

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    void* device_data(ObjectHandle handle) const;
    std::span<const std::byte> resource_bytes(ObjectHandle handle) const;
    std::vector<std::byte> take_resource_bytes(ObjectHandle handle);

    const MaterializationStats& stats() const noexcept { return stats_; }

    DeviceArena& device_arena();

private:
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                            DeviceContext&, const StartupObserver*);
    friend MaterializedArtifact materialize(const Reader&, const MaterializationPlan&,
                                            DeviceContext&, const StartupObserver*,
                                            const MaterializationStageMap&,
                                            const MaterializationStageDevice&);

    struct ObjectStorage {
        void* device = nullptr;
        std::vector<std::byte> resource;
    };

    // Stage-owned backing arenas; stage 0 is the primary arena returned by device_arena(). All
    // stay allocated for the artifact lifetime. In the single-device path this holds exactly one
    // arena.
    std::vector<std::unique_ptr<DeviceArena>> stage_arenas_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device,
                                 const StartupObserver* startup_observer = nullptr);

// Pipeline-parallel variant: tensor bytes land on the stage chosen by `stage_map`, uploaded on
// that stage's transfer stream from the same direct-I/O pipeline. `device` must be the primary
// stage's context and is also used for single-stage resources.
MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer,
                                 const MaterializationStageMap& stage_map,
                                 const MaterializationStageDevice& stage_device);

} // namespace ninfer::artifact
