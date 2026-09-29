#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "core/nvtx.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    // Pipeline execution skips the prefill draft bridge: the bridge's MTP layer runs after the
    // backbone on the primary stream, but MTP weights and KV live on the last stage. Drafts are
    // rebuilt by the decode rounds themselves; verification keeps outputs exact.
    if (state.execution.pipeline_execution != nullptr) { return; }
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache,
                     state.execution.pipeline_execution);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size,
                           std::uint32_t verify_k, std::uint32_t proposal_k,
                           MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, verify_k, proposal_k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            verify_k == 0 || verify_k > kMtpLookupMaximumDrafts || proposal_k == 0 ||
            proposal_k > kMtpDecodeMaximumDrafts || proposal_k > verify_k) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(verify_k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache,
                         state.execution.pipeline_execution);
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets            = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents    = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts     = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows           = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor rope_deltas        = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents       = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids      = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden   = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden          = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden        = frame.next_hidden.slice(1, 0, batch_size);
        const std::int32_t ar_steps =
            static_cast<std::int32_t>(std::max(proposal_k - 1U, 1U));
        Tensor ar_positions =
            frame.ar_positions.slice(0, 0, batch_size).slice(1, 0, ar_steps);
        Tensor ar_rope_positions =
            frame.ar_rope_positions.slice(0, 0, batch_size).slice(1, 0, ar_steps);
        Tensor ar_valid_columns =
            frame.ar_valid_columns.slice(0, 0, batch_size).slice(1, 0, ar_steps);
        Tensor next_drafts        = frame.next_drafts.slice(0, 0, batch_size);

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = verify_ids,
                                     .cache_positions         = target_positions,
                                     .rope_positions          = target_rope,
                                     .valid_columns           = target_valid,
                                     .kv_table_rows           = text_rows,
                                     .state_source_slots      = state_sources,
                                     .state_destination_slots = state_destinations,
                                     .target_hidden           = target_hidden,
                                     .target_logits           = target_logits,
                                     .target_tokens           = target_tokens,
                                     .drafts                  = current_drafts,
                                     .current_extents         = current_extents,
                                     .frontiers               = frontiers,
                                     .anchors                 = anchors,
                                     .licensed_tokens         = licensed_tokens,
                                     .licensed_counts         = licensed_counts,
                                     .accepted_drafts         = accepted,
                                     .selected_hidden         = selected_hidden,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                 },
                                 envelopes.target_verify);
        }

        {
            nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                          static_cast<std::uint64_t>(proposal_k) * batch_size);
            ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                        licensed_counts, rope_deltas, alignment_ids, next_extents,
                                        ar_positions, ar_rope_positions, ar_valid_columns,
                                        static_cast<std::int32_t>(proposal_k),
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        state.execution.device.stream);
            card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions,
                                          target_rope, licensed_counts, mtp_rows, envelopes.batch,
                                          alignment_hidden);
            ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                    state.execution.device.stream);

            Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
            Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
            card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
            for (std::uint32_t step = 0; step + 1 < proposal_k; ++step) {
                Tensor previous =
                    next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                Tensor next =
                    next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
                Tensor position =
                    ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
                Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch    = previous.view({1, batch_size});
                Tensor hidden_batch      = ar_hidden.view({TextConfig::hidden, 1, batch_size});
                Tensor next_hidden_batch = next_hidden.view({TextConfig::hidden, 1, batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              mtp_rows, envelopes.ar[step], next_hidden_batch);
                card.mtp_propose_batch(next_hidden, proposal_logits, next);
                CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

// Acceptance and the draft phase on the last stage against the mirror frame. Shared by the eager
// pipeline round and the captured stage-1 graph.
static void mtp_last_stage_phase(MtpBatchContext& state, qwen3_6::MtpDecodeState& mirror,
                                 TextContext& card, std::int32_t batch_size,
                                 std::uint32_t verify_k, std::uint32_t proposal_k,
                                 const MtpCausalAttentionEnvelopes& envelopes) {
    qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;
    DeviceContext& stage1           = pipeline.stage_device(pipeline.stage_count() - 1);
    WorkspaceArena& stage1_work     = pipeline.last_workspace();
    const std::int32_t width        = static_cast<std::int32_t>(verify_k) + 1;
    const std::int32_t ar_steps     = static_cast<std::int32_t>(std::max(proposal_k - 1U, 1U));

    Tensor mirror_anchors      = mirror.anchors.slice(0, 0, batch_size);
    Tensor mirror_frontiers    = mirror.base_frontiers.slice(0, 0, batch_size);
    Tensor mirror_budgets      = mirror.remaining_budgets.slice(0, 0, batch_size);
    Tensor mirror_extents      = mirror.current_extents.slice(0, 0, batch_size);
    Tensor mirror_drafts       = mirror.current_drafts.slice(1, 0, batch_size);
    Tensor mirror_rope_deltas  = mirror.rope_deltas.slice(0, 0, batch_size);
    Tensor mirror_licensed     = mirror.licensed_tokens.slice(1, 0, batch_size);
    Tensor mirror_licensed_cnt = mirror.licensed_counts.slice(0, 0, batch_size);
    Tensor mirror_accepted     = mirror.accepted_drafts.slice(0, 0, batch_size);
    Tensor mirror_next_extents = mirror.next_extents.slice(0, 0, batch_size);
    Tensor mirror_selected     = mirror.target_continuation_hidden.slice(1, 0, batch_size);
    const auto* mirror_sampling = mirror.sampling;
    Tensor target_tokens        = mirror.target_argmax.slice(1, 0, batch_size);
    Tensor target_logits        = mirror.target_logits.slice(2, 0, batch_size);
    Tensor target_hidden        = mirror.target_hidden.slice(2, 0, batch_size);
    {
        nvtx::ScopedRange accept_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                       static_cast<std::uint64_t>(width) * batch_size);
        ops::speculative_accept_greedy_drafts(
            target_tokens, target_logits, mirror_drafts, mirror_extents, mirror_frontiers,
            mirror_anchors, mirror_licensed, mirror_licensed_cnt, mirror_accepted,
            TextConfig::token_domain, mirror_sampling, stage1_work, stage1.stream);
        ops::speculative_select_accepted_hidden(target_hidden, mirror_accepted, mirror_selected,
                                                stage1.stream);
    }

    {
        nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                      static_cast<std::uint64_t>(proposal_k) * batch_size);
        Tensor alignment_ids    = mirror.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden = mirror.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden        = mirror.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden      = mirror.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions =
            mirror.ar_positions.slice(0, 0, batch_size).slice(1, 0, ar_steps);
        Tensor ar_rope_positions =
            mirror.ar_rope_positions.slice(0, 0, batch_size).slice(1, 0, ar_steps);
        Tensor ar_valid_columns =
            mirror.ar_valid_columns.slice(0, 0, batch_size).slice(1, 0, ar_steps);
        Tensor next_drafts      = mirror.next_drafts.slice(0, 0, batch_size);
        Tensor proposal_logits  = mirror.proposal_logits.slice(1, 0, batch_size);
        Tensor mtp_rows         = mirror.mtp_kv_table_rows.slice(0, 0, batch_size);

        ops::mtp_prepare_next_round(mirror.verify_ids, mirror_anchors, mirror_accepted,
                                    mirror_frontiers, mirror_budgets, mirror_licensed_cnt,
                                    mirror_rope_deltas, mirror.alignment_ids, mirror.next_extents,
                                    mirror.ar_positions, mirror.ar_rope_positions,
                                    mirror.ar_valid_columns,
                                    static_cast<std::int32_t>(proposal_k),
                                    static_cast<std::int32_t>(state.text_cache.max_context()),
                                    stage1.stream);
        card.mtp_forward_decode_batch(
            alignment_ids, target_hidden, mirror.target_positions.slice(1, 0, batch_size),
            mirror.target_rope_positions.slice(1, 0, batch_size), mirror_licensed_cnt,
            mtp_rows, envelopes.batch, alignment_hidden);
        ops::speculative_select_accepted_hidden(alignment_hidden, mirror_accepted, ar_hidden,
                                                stage1.stream);

        Tensor draft0 = mirror.next_drafts.slice(1, 0, 1).view({batch_size});
        card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
        for (std::uint32_t step = 0; step + 1 < proposal_k; ++step) {
            Tensor previous =
                next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor next =
                next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
            Tensor position =
                ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
            Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                              .view({1, batch_size});
            Tensor valid = ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                               .view({batch_size});
            Tensor previous_batch    = previous.view({1, batch_size});
            Tensor hidden_batch      = ar_hidden.view({TextConfig::hidden, 1, batch_size});
            Tensor next_hidden_batch = next_hidden.view({TextConfig::hidden, 1, batch_size});
            card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                          mtp_rows, envelopes.ar[step], next_hidden_batch);
            card.mtp_propose_batch(next_hidden, proposal_logits, next);
            CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, stage1.stream));
        }
    }
}

// Pipeline variant of one MTP round. The primary side uploads the ingress and prepares the
// verify inputs; the verify traversal crosses to the last stage (its tail writes stage-local
// mirror outputs); acceptance, the draft phase, and the proposal run entirely on the last stage
// against the stage-local mirror frame; only the egress region and the selected continuation
// rows ship back before the host-visible D2H. Correctness is unchanged by this split: drafts are
// verification-gated.
auto mtp_decode_batch_pipeline_body(MtpBatchContext& state, std::int32_t batch_size,
                                    std::uint32_t verify_k, std::uint32_t proposal_k,
                                    MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, verify_k, proposal_k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            verify_k == 0 || verify_k > kMtpLookupMaximumDrafts || proposal_k == 0 ||
            proposal_k > kMtpDecodeMaximumDrafts || proposal_k > verify_k) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }
        qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;
        const int last_stage = pipeline.stage_count() - 1;
        qwen3_6::PipelineStageResources& resources = pipeline.stage_resources(last_stage);
        if (!resources.mtp_frame) {
            throw std::logic_error("pipeline stage has no MTP decode frame");
        }
        qwen3_6::MtpDecodeState& frame  = state.frame;
        qwen3_6::MtpDecodeState& mirror = *resources.mtp_frame;
        DeviceContext& stage0           = state.execution.device;
        DeviceContext& stage1           = pipeline.stage_device(last_stage);
        WorkspaceArena& stage1_work     = pipeline.last_workspace();
        PipelineTransport& forward      = *pipeline.forward[last_stage - 1];
        PipelineTransport& backward     = *pipeline.backward[last_stage - 1];
        const std::int32_t width        = static_cast<std::int32_t>(verify_k) + 1;
        const std::int32_t ar_steps     = static_cast<std::int32_t>(std::max(proposal_k - 1U, 1U));

        if (pipeline.mtp_rope_delta.data == nullptr) {
            stage1.bind_to_current_thread();
            pipeline.mtp_rope_delta =
                stage1_work.alloc(DType::I32, {static_cast<std::int32_t>(kMaximumConcurrency)});
        }

        // [primary] ingress upload and verify-input preparation.
        stage0.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   stage0.stream));
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_rope      = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor target_valid     = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources    = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor current_drafts   = frame.current_drafts.slice(1, 0, batch_size);
        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor current_extents  = frame.current_extents.slice(0, 0, batch_size);
        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions, stage0.stream);

        // [boundary] mirror the whole ingress region plus the prepared verify ids, target
        // positions, and RoPE delta; the egress region and continuation rows return later.
        forward.enqueue(frame.ingress.data, mirror.ingress.data, sizeof(qwen3_6::MtpDecodeIngress));
        forward.enqueue(frame.verify_ids.data, mirror.verify_ids.data, frame.verify_ids.bytes());
        forward.enqueue(frame.target_positions.data, mirror.target_positions.data,
                        frame.target_positions.bytes());
        forward.enqueue(state.execution.io.rope_delta.data, pipeline.mtp_rope_delta.data,
                        pipeline.mtp_rope_delta.bytes());
        // The copied sampling configs embed stage-0 penalty-count pointers; null them in the
        // mirror so acceptance reads no foreign addresses (penalties are inert on this stage).
        stage1.bind_to_current_thread();
        for (int row = 0; row < batch_size; ++row) {
            CUDA_CHECK(cudaMemsetAsync(
                static_cast<std::byte*>(mirror.ingress.data) +
                    offsetof(qwen3_6::MtpDecodeIngress, sampling) +
                    static_cast<std::size_t>(row) * sizeof(ops::SamplingConfig) +
                    offsetof(ops::SamplingConfig, token_counts),
                0, sizeof(std::int32_t*), stage1.stream));
        }

        // [traversal] verify crosses both stages; its tail writes the mirror outputs on the last
        // stage and leaves it current. The staging enqueues above left the last stage bound, so
        // re-bind the primary before its kernels launch.
        stage0.bind_to_current_thread();
        TextContext card(stage0, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache, state.execution.pipeline_execution);
        card.set_gdn_state_action(GdnStateAction::RecordForReplay, state.execution.replay_records);
        card.target_verify_batch(verify_ids, target_positions, target_rope, target_valid, text_rows,
                                 state_sources, envelopes.target_verify, mirror.target_hidden,
                                 mirror.target_logits, mirror.target_argmax);

        // [last stage] acceptance and the draft phase against the mirror frame.
        stage1.bind_to_current_thread();
        mtp_last_stage_phase(state, mirror, card, batch_size, verify_k, proposal_k, envelopes);

        // [return] ship the egress region and the selected continuation rows, then finish the
        // round on the primary: continuation scatter and the host-visible egress transfer.
        backward.enqueue(mirror.egress.data, frame.egress.data, sizeof(qwen3_6::MtpDecodeEgress));
        backward.enqueue(mirror.target_continuation_hidden.data, frame.target_continuation_hidden.data,
                         frame.target_continuation_hidden.bytes());
        card.finish_round();
        stage0.bind_to_current_thread();
        ops::scatter(frame.target_continuation_hidden.slice(1, 0, batch_size),
                     frame.state_destination_slots.slice(0, 0, batch_size),
                     state.continuation_hidden_store, stage0.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   stage0.stream));
    };
}

void capture_mtp_decode_graphs(MtpBatchContext& state, std::int32_t batch_size,
                               std::uint32_t verify_k, std::uint32_t proposal_k,
                               MtpCausalAttentionEnvelopes envelopes,
                               DecodeGraphDefinition& definition0,
                               DecodeGraphDefinition& definition1) {
    if (state.execution.pipeline_execution == nullptr) {
        throw std::logic_error("pipeline MTP graph capture requires a pipeline");
    }
    qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;
    const int last_stage                 = pipeline.stage_count() - 1;
    qwen3_6::PipelineStageResources& resources = pipeline.stage_resources(last_stage);
    if (!resources.mtp_frame) { throw std::logic_error("pipeline stage has no MTP frame"); }
    qwen3_6::MtpDecodeState& frame  = state.frame;
    qwen3_6::MtpDecodeState& mirror = *resources.mtp_frame;
    DeviceContext& stage0           = state.execution.device;
    DeviceContext& stage1           = pipeline.stage_device(last_stage);
    PipelineTransport& forward      = *pipeline.forward[last_stage - 1];
    PipelineTransport& backward     = *pipeline.backward[last_stage - 1];
    if (pipeline.mtp_rope_delta.data == nullptr) {
        stage1.bind_to_current_thread();
        pipeline.mtp_rope_delta =
            pipeline.last_workspace().alloc(DType::I32, {static_cast<std::int32_t>(kMaximumConcurrency)});
    }

    stage0.bind_to_current_thread();
    definition0.capture(stage0.stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   stage0.stream));
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_rope      = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor target_valid     = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources    = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor current_drafts   = frame.current_drafts.slice(1, 0, batch_size);
        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor current_extents  = frame.current_extents.slice(0, 0, batch_size);
        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions, stage0.stream);
        forward.enqueue_stage_out(frame.ingress.data, sizeof(qwen3_6::MtpDecodeIngress), 7);
        forward.enqueue_stage_out(frame.verify_ids.data, frame.verify_ids.bytes(), 8);
        forward.enqueue_stage_out(frame.target_positions.data, frame.target_positions.bytes(), 9);
        forward.enqueue_stage_out(state.execution.io.rope_delta.data, pipeline.mtp_rope_delta.bytes(),
                                  10);
        TextContext card(stage0, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache, state.execution.pipeline_execution);
        card.set_gdn_state_action(GdnStateAction::RecordForReplay, state.execution.replay_records);
        card.target_verify_graph_segment(0, verify_ids, target_positions, target_rope, target_valid,
                                         text_rows, state_sources, envelopes.target_verify,
                                         mirror.target_hidden, mirror.target_logits,
                                         mirror.target_argmax);
    });

    stage1.bind_to_current_thread();
    definition1.capture(stage1.stream, [&] {
        forward.enqueue_stage_in(mirror.ingress.data, sizeof(qwen3_6::MtpDecodeIngress), 7);
        forward.enqueue_stage_in(mirror.verify_ids.data, frame.verify_ids.bytes(), 8);
        forward.enqueue_stage_in(mirror.target_positions.data, frame.target_positions.bytes(), 9);
        forward.enqueue_stage_in(pipeline.mtp_rope_delta.data, pipeline.mtp_rope_delta.bytes(), 10);
        for (int row = 0; row < batch_size; ++row) {
            CUDA_CHECK(cudaMemsetAsync(
                static_cast<std::byte*>(mirror.ingress.data) +
                    offsetof(qwen3_6::MtpDecodeIngress, sampling) +
                    static_cast<std::size_t>(row) * sizeof(ops::SamplingConfig) +
                    offsetof(ops::SamplingConfig, token_counts),
                0, sizeof(std::int32_t*), stage1.stream));
        }
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_rope      = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor target_valid     = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources    = frame.state_source_slots.slice(0, 0, batch_size);
        TextContext card(stage0, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache, state.execution.pipeline_execution);
        card.set_gdn_state_action(GdnStateAction::RecordForReplay, state.execution.replay_records);
        card.target_verify_graph_segment(1, verify_ids, target_positions, target_rope, target_valid,
                                         text_rows, state_sources, envelopes.target_verify,
                                         mirror.target_hidden, mirror.target_logits,
                                         mirror.target_argmax);
        mtp_last_stage_phase(state, mirror, card, batch_size, verify_k, proposal_k, envelopes);
        backward.enqueue_stage_out(mirror.egress.data, sizeof(qwen3_6::MtpDecodeEgress), 0);
        backward.enqueue_stage_out(mirror.target_continuation_hidden.data,
                                   frame.target_continuation_hidden.bytes(), 1);
    });
    stage0.bind_to_current_thread();
}

void mtp_decode_graph_round(MtpBatchContext& state, std::int32_t batch_size,
                            std::uint32_t verify_k, std::uint32_t proposal_k,
                            MtpCausalAttentionEnvelopes envelopes,
                            DecodeGraphExecutable* executable0,
                            DecodeGraphExecutable* executable1) {
    (void)verify_k;
    (void)proposal_k;
    (void)envelopes;
    qwen3_6::PipelineExecution& pipeline = *state.execution.pipeline_execution;
    const int last_stage                 = pipeline.stage_count() - 1;
    qwen3_6::MtpDecodeState& frame       = state.frame;
    qwen3_6::PipelineStageResources& resources = pipeline.stage_resources(last_stage);
    qwen3_6::MtpDecodeState& mirror      = *resources.mtp_frame;
    DeviceContext& stage0                = state.execution.device;
    DeviceContext& stage1                = pipeline.stage_device(last_stage);
    PipelineTransport& backward          = *pipeline.backward[last_stage - 1];
    if (pipeline.ordinary_round_events[0] == nullptr) {
        stage0.bind_to_current_thread();
        CUDA_CHECK(cudaEventCreateWithFlags(&pipeline.ordinary_round_events[0],
                                            cudaEventDisableTiming));
        stage1.bind_to_current_thread();
        CUDA_CHECK(cudaEventCreateWithFlags(&pipeline.ordinary_round_events[1],
                                            cudaEventDisableTiming));
    }
    executable0->launch(stage0.stream);
    stage0.bind_to_current_thread();
    CUDA_CHECK(cudaEventRecord(pipeline.ordinary_round_events[0], stage0.stream));
    stage1.bind_to_current_thread();
    CUDA_CHECK(cudaStreamWaitEvent(stage1.stream, pipeline.ordinary_round_events[0], 0));
    executable1->launch(stage1.stream);
    CUDA_CHECK(cudaEventRecord(pipeline.ordinary_round_events[1], stage1.stream));
    stage0.bind_to_current_thread();
    CUDA_CHECK(cudaStreamWaitEvent(stage0.stream, pipeline.ordinary_round_events[1], 0));
    backward.enqueue_stage_in(frame.egress.data, sizeof(qwen3_6::MtpDecodeEgress), 0);
    backward.enqueue_stage_in(frame.target_continuation_hidden.data,
                              frame.target_continuation_hidden.bytes(), 1);
    ops::scatter(frame.target_continuation_hidden.slice(1, 0, batch_size),
                 frame.state_destination_slots.slice(0, 0, batch_size),
                 state.continuation_hidden_store, stage0.stream);
    CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                               sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                               stage0.stream));
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size,
                              std::uint32_t verify_k, std::uint32_t proposal_k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition) {
    if (state.execution.pipeline_execution != nullptr) {
        throw std::logic_error("pipeline MTP rounds run eagerly");
    }
    auto body = mtp_decode_batch_body(state, batch_size, verify_k, proposal_k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size,
                      std::uint32_t verify_k, std::uint32_t proposal_k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable,
                      DecodeGraphExecutable* executable1) {
    if (state.execution.pipeline_execution != nullptr) {
        if (executable != nullptr && executable1 != nullptr) {
            mtp_decode_graph_round(state, batch_size, verify_k, proposal_k, envelopes, executable,
                                   executable1);
            return;
        }
        auto body =
            mtp_decode_batch_pipeline_body(state, batch_size, verify_k, proposal_k, envelopes);
        body();
        return;
    }
    auto body = mtp_decode_batch_body(state, batch_size, verify_k, proposal_k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
