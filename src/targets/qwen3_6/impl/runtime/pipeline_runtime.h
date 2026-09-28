#pragma once

// Pipeline-parallel execution resources for the non-primary stages. Owned by ProgramImplCore for
// the Program lifetime; handed to the model schedule so the stage switch can reach each stage's
// device, workspace, KV planes, and linear-attention state.

#include "core/arena.h"
#include "core/pipeline.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/state_image.h>

#include <memory>
#include <vector>

namespace ninfer::targets::qwen3_6 {

// One non-primary stage. The stage KV pool is a passive mirror: its allocation state never
// advances (the primary pool owns the page lifecycle), while content operations — page copies and
// table row writes — are forwarded by the primary pools' mirror hooks at identical physical
// indices. Linear-attention state is a full stage-local pool that receives indexed slot mutations
// through the primary pool's mirror hook and is read/written directly by the stage's GDN layers.
struct PipelineStageResources {
    std::unique_ptr<DeviceArena> persistent;   // stage backing owner
    std::unique_ptr<DeviceArena> workspace;    // stage per-layer temporaries
    std::unique_ptr<DecoderState> decoder;     // stage text KV planes + execution tables
    std::unique_ptr<StateImageDevicePool> state_images;  // stage GDN linear state
};

// Cross-stage activation channel set: one forward and one backward transport per stage boundary.
struct PipelineExecution {
    const PipelineContext* context = nullptr;
    std::vector<PipelineStageResources> stages;  // entry i serves stage i + 1
    std::vector<std::unique_ptr<PipelineTransport>> forward;   // boundary i: stage i -> i + 1
    std::vector<std::unique_ptr<PipelineTransport>> backward;  // boundary i: stage i + 1 -> stage i

    [[nodiscard]] int stage_count() const noexcept {
        return context != nullptr ? context->stage_count() : 1;
    }

    [[nodiscard]] DeviceContext& stage_device(int stage) const {
        return context->stage_device(stage);
    }

    [[nodiscard]] WorkspaceArena& stage_workspace(int stage) const {
        return *stages[static_cast<std::size_t>(stage) - 1].workspace;
    }

    [[nodiscard]] DecoderState& stage_decoder(int stage) const {
        return *stages[static_cast<std::size_t>(stage) - 1].decoder;
    }

    [[nodiscard]] StateImageDevicePool& stage_state_images(int stage) const {
        return *stages[static_cast<std::size_t>(stage) - 1].state_images;
    }
};

} // namespace ninfer::targets::qwen3_6
