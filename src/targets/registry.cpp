#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/pipeline.h"
#include "core/startup.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

// Free VRAM summed across the pipeline stage devices; materialize() applies the real per-stage
// budget checks during placement.
std::size_t pipeline_free_device_bytes(const std::vector<int>& stage_devices) {
    std::size_t free_total = 0;
    for (const int id : stage_devices) {
        cudaSetDevice(id);
        std::size_t free_bytes = 0;
        std::size_t total      = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
        free_total += free_bytes;
    }
    return free_total;
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// Validates pipeline options and returns the stage device-id list in stage order; empty means
// pipeline parallelism is disabled (single device).
std::vector<int> pipeline_stage_devices(const EngineOptions& options) {
    if (options.pipeline_size == 1) {
        if (!options.pipeline_devices.empty()) {
            throw std::invalid_argument(
                "Engine pipeline_devices must be empty when pipeline_size is 1");
        }
        return {};
    }
    if (options.pipeline_size < 1 || options.pipeline_size > 16) {
        throw std::invalid_argument("Engine pipeline_size must be in [1,16]");
    }
    std::vector<int> devices = options.pipeline_devices;
    if (devices.empty()) {
        devices.reserve(static_cast<std::size_t>(options.pipeline_size));
        for (int stage = 0; stage < options.pipeline_size; ++stage) {
            devices.push_back(options.device + stage);
        }
    }
    if (static_cast<int>(devices.size()) != options.pipeline_size) {
        throw std::invalid_argument(
            "Engine pipeline_devices must name exactly pipeline_size devices");
    }
    if (devices.front() != options.device) {
        throw std::invalid_argument(
            "Engine pipeline_devices must begin with the primary device");
    }
    return devices;
}

// Maps an artifact object name to the pipeline stage that owns its device bytes, following the
// registered text-backbone naming convention: text/layers/<L>/... follow the layer partition;
// endpoints and speculative heads live on the last stage; embeddings and vision live on stage 0.
[[nodiscard]] int stage_of_object_name(const PipelineContext& pipeline,
                                       std::string_view name) {
    const int last_stage = pipeline.stage_count() - 1;
    constexpr std::string_view kLayerPrefix = "text/layers/";
    if (name.starts_with(kLayerPrefix)) {
        const std::size_t digits_begin = kLayerPrefix.size();
        const std::size_t slash        = name.find('/', digits_begin);
        if (slash != std::string_view::npos) {
            const std::string_view digits = name.substr(digits_begin, slash - digits_begin);
            if (!digits.empty() &&
                std::all_of(digits.begin(), digits.end(),
                            [](char c) { return c >= '0' && c <= '9'; })) {
                const int layer = std::stoi(std::string(digits));
                if (layer < pipeline.partition().layer_count) {
                    return pipeline.partition().stage_of_layer(layer);
                }
            }
        }
    }
    if (name == "text/final_norm" || name == "text/output_head" || name == "text/draft_head" ||
        name == "text/draft_head_token_ids" || name.starts_with("mtp/")) {
        return last_stage;
    }
    return 0;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    StartupPhaseScope target_plan_phase(options.startup_observer, StartupPhase::TargetPlan);
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const runtime::ContextCostIdentity context_cost_identity{
        .hardware_class = runtime::context_cost_hardware_class(
            device.props.name, device.props.major, device.props.minor),
        .model_id   = identity.model_id,
        .weights_id = identity.weights_id,
    };
    runtime::ResolvedContextMachineCost context_cost = runtime::resolve_context_machine_cost(
        context_cost_identity, options.context_cost.preset_path);

    // Pipeline parallelism: one DeviceContext per stage over the target's text backbone
    // topology; a null pipeline means the historical single-device load.
    std::optional<PipelineContext> pipeline;
    const std::vector<int> stage_devices = pipeline_stage_devices(options);
    if (!stage_devices.empty()) {
        pipeline.emplace(stage_devices, Target::kTextLayerCount, Target::kFullAttentionInterval,
                         options.pipeline_embedding_replica);
        // v1 pipeline scope: plain generation with MTP. CUDA Graph decode capture, Vision input,
        // and masked-block speculative decoding cross stage boundaries and are rejected until
        // their per-stage plumbing lands.
        if (options.speculative.backend == SpeculativeBackend::DFlash2 ||
            options.speculative.backend == SpeculativeBackend::DFlash) {
            throw std::invalid_argument(
                "pipeline parallelism (--pp) does not support masked-block speculative decoding; "
                "use --spec none or --spec mtp");
        }
        if (options.enable_vision) {
            throw std::invalid_argument(
                "pipeline parallelism (--pp) does not support Vision input yet");
        }
    }
    PipelineContext* pipeline_ptr = pipeline.has_value() ? &pipeline.value() : nullptr;

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    std::size_t preflight_runtime_bytes = 0;
    if (pipeline_ptr) {
        const std::uint64_t stage_free = pipeline_free_device_bytes(stage_devices);
        const std::uint64_t weight_bytes = load_plan.materialization().device_capacity_bytes;
        if (weight_bytes > stage_free) {
            throw std::invalid_argument(
                "model weights require " + std::to_string(weight_bytes) +
                " bytes across the pipeline stage devices, but only " + std::to_string(stage_free) +
                " bytes are free before loading weights");
        }
        preflight_runtime_bytes = static_cast<std::size_t>(stage_free - weight_bytes);
    } else {
        preflight_runtime_bytes = runtime_bytes_after_planned_weights(
            load_plan.materialization().device_capacity_bytes);
    }
    (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);
    target_plan_phase.complete();

    auto materialized = artifact::materialize(
        reader, load_plan.materialization(), device, &options.startup_observer,
        [pipeline_ptr](const artifact::ObjectDescriptor& object) {
            return stage_of_object_name(*pipeline_ptr, artifact::object_name(object));
        },
        [pipeline_ptr](int stage) -> DeviceContext& {
            return pipeline_ptr->stage_device(stage);
        });
    const artifact::MaterializationStats stats = materialized.stats();
    // Staged materialization leaves the last stage current; every later Program allocation and
    // startup transaction owns the primary device.
    device.bind_to_current_thread();

    StartupPhaseScope target_finalize_phase(options.startup_observer, StartupPhase::TargetFinalize);
    auto model =
        Target::construct_loaded_model(std::move(load_plan), std::move(materialized), pipeline_ptr);
    device.synchronize();
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity(options.kv_capacity, curve, current_free_device_bytes());
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    target_finalize_phase.complete();

    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto loaded = std::make_unique<Loaded>(std::move(model), options);
    frontend_phase.complete();

    StartupPhaseScope program_phase(options.startup_observer, StartupPhase::ProgramInitialize);
    auto instance =
        std::make_unique<Instance>(std::move(loaded), capacity_resolution, std::move(sequence_plan),
                                   device, options.startup_observer);
    device.synchronize();
    program_phase.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    summary.context_cost         = context_cost.summary;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults,
                             .context_cost      = std::move(context_cost.model),
                             .pipeline          = pipeline.has_value()
                                                      ? std::make_unique<PipelineContext>(
                                                            std::move(pipeline.value()))
                                                      : nullptr};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device,
                                         const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device,
                                          startup_observer)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device,
                                               const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device,
                                             startup_observer)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    StartupPhaseScope inspect_phase(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect_phase.complete();
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
