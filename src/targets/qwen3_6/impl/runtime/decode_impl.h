#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <tuple>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_6::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, state.execution.pipeline_execution);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling, cache_positions,
                    ops::kSamplePurposeDecode, state.execution.work, state.execution.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

// Prepares the fixed last-stage graph mirrors once, from the last stage's workspace.
void prepare_ordinary_graph_mirrors(OrdinaryBatchContext& state, std::int32_t batch_size) {
    qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;
    qwen3_6::OrdinaryGraphMirrors& mirrors = pipeline.ordinary_mirrors;
    if (mirrors.prepared) { return; }
    WorkspaceArena& work = pipeline.stage_workspace(pipeline.stage_count() - 1);
    mirrors.cache = work.alloc(DType::I32, {batch_size});
    mirrors.rope = work.alloc(DType::I32, {batch_size});
    mirrors.kv_rows = work.alloc(DType::I32, {batch_size});
    mirrors.src_slots = work.alloc(DType::I32, {batch_size});
    mirrors.dst_slots = work.alloc(DType::I32, {batch_size});
    mirrors.x = work.alloc(DType::BF16, {TextConfig::hidden, batch_size});
    mirrors.hidden = work.alloc(DType::BF16, {TextConfig::hidden, batch_size});
    mirrors.logits = work.alloc(DType::BF16, {TextConfig::output_rows, batch_size});
    mirrors.prepared = true;
}

// Captures the ordinary decode round as two per-stage graphs. Graph 0 (primary device) runs the
// ingress upload, embedding, and the primary-side layers, then stages the boundary activation and
// control tensors into the forward transport. Graph 1 (last stage) receives them, runs the
// remaining layers plus final norm and lm_head, and stages the outputs back. The caller chains
// the two launches with events and runs the stage-0 tail (restore, scatter, sample, egress)
// eagerly.
void capture_ordinary_decode_graphs(OrdinaryBatchContext& state, std::int32_t batch_size,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    DecodeGraphDefinition& definition0,
                                    DecodeGraphDefinition& definition1) {
    prepare_ordinary_graph_mirrors(state, batch_size);
    qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;

    auto ingress_upload = [&state] {
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.ordinary->ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));
    };
    qwen3_6::OrdinaryDecodeState& ordinary = state.frame;
    auto segment_args = [&]() {
        return std::make_tuple(
            ordinary.tokens.slice(0, 0, batch_size),
            ordinary.cache_positions.slice(0, 0, batch_size),
            ordinary.rope_positions.slice(0, 0, batch_size),
            ordinary.text_kv_table_rows.slice(0, 0, batch_size),
            ordinary.state_source_slots.slice(0, 0, batch_size),
            ordinary.state_destination_slots.slice(0, 0, batch_size));
    };
    state.execution.device.bind_to_current_thread();
    definition0.capture(state.execution.device.stream, [&] {
        ingress_upload();
        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, state.execution.pipeline_execution);
        auto [tokens, cache_positions, rope_positions, kv_rows, src, dst] = segment_args();
        card.ordinary_decode_graph_segment(0, tokens, cache_positions, rope_positions, kv_rows,
                                           src, dst, envelope, batch_size);
    });
    DeviceContext& last_stage = pipeline.stage_device(pipeline.stage_count() - 1);
    last_stage.bind_to_current_thread();
    definition1.capture(last_stage.stream, [&] {
        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, nullptr, state.execution.pipeline_execution);
        auto [tokens, cache_positions, rope_positions, kv_rows, src, dst] = segment_args();
        card.ordinary_decode_graph_segment(1, tokens, cache_positions, rope_positions, kv_rows,
                                           src, dst, envelope, batch_size);
    });
    state.execution.device.bind_to_current_thread();
}

// Eager stage-0 tail after the last-stage graph: restore the shipped outputs into the caller
// tensors, publish continuation hidden, sample, and stage the egress for the host.
void ordinary_pipeline_tail(OrdinaryBatchContext& state, std::int32_t batch_size,
                            ops::CausalAttentionExecutionEnvelope envelope) {
    qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;
    cudaStream_t stream = state.execution.device.stream;
    qwen3_6::OrdinaryDecodeState& ordinary = state.frame;
    PipelineTransport& back = *pipeline.backward[pipeline.stage_count() - 2];
    Tensor hidden = ordinary.hidden.slice(1, 0, batch_size);
    Tensor logits = ordinary.logits.slice(1, 0, batch_size);
    Tensor sampled = ordinary.sampled_tokens.slice(0, 0, batch_size);
    Tensor cache_positions = ordinary.cache_positions.slice(0, 0, batch_size);
    Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
    CUDA_CHECK(cudaMemcpyAsync(hidden.data, back.staging(0), hidden.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(logits.data, back.staging(1), logits.bytes(),
                               cudaMemcpyHostToDevice, stream));
    ops::scatter(hidden, state_destinations, state.continuation_hidden_store, stream);
    ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling, cache_positions,
                ops::kSamplePurposeDecode, state.execution.work, stream);
    CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                               sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                               stream));
}

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

// Runs one captured pipeline round: graph 0 on the primary stream, graph 1 on the last stage's
// stream chained by events, then the eager stage-0 tail.
void ordinary_decode_graph_round(OrdinaryBatchContext& state, std::int32_t batch_size,
                                 ops::CausalAttentionExecutionEnvelope envelope,
                                 DecodeGraphExecutable* executable0,
                                 DecodeGraphExecutable* executable1) {
    qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;
    DeviceContext& stage0 = state.execution.device;
    DeviceContext& stage1 = pipeline.stage_device(pipeline.stage_count() - 1);
    if (pipeline.ordinary_round_events[0] == nullptr) {
        stage0.bind_to_current_thread();
        CUDA_CHECK(cudaEventCreateWithFlags(&pipeline.ordinary_round_events[0],
                                            cudaEventDisableTiming));
        stage1.bind_to_current_thread();
        CUDA_CHECK(cudaEventCreateWithFlags(&pipeline.ordinary_round_events[1],
                                            cudaEventDisableTiming));
    }
    cudaEvent_t stage0_done = pipeline.ordinary_round_events[0];
    cudaEvent_t stage1_done = pipeline.ordinary_round_events[1];

    executable0->launch(stage0.stream);
    stage0.bind_to_current_thread();
    CUDA_CHECK(cudaEventRecord(stage0_done, stage0.stream));
    stage1.bind_to_current_thread();
    CUDA_CHECK(cudaStreamWaitEvent(stage1.stream, stage0_done, 0));
    executable1->launch(stage1.stream);
    CUDA_CHECK(cudaEventRecord(stage1_done, stage1.stream));
    stage0.bind_to_current_thread();
    CUDA_CHECK(cudaStreamWaitEvent(stage0.stream, stage1_done, 0));
    ordinary_pipeline_tail(state, batch_size, envelope);
}


void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
