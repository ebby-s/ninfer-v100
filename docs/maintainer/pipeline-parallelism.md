# Pipeline parallelism across V100 stages — design and status

Status: active design for `feat/pipeline-parallel`. This document is the single authority for the
remaining execution work; delete it once the feature lands and the product docs carry the surface.

## Goal

Pipeline parallelism (`--pp N`) splits the text backbone across N same-architecture GPUs so KV
capacity becomes the sum of stage VRAM: longer contexts and more context slots, not speed. The
reference host pairs two V100-PCIE-32GB whose slots report no P2P access, so inter-stage traffic
stages through pinned host memory (~10 KB per decode token at hidden 5120; ~10 MB per 2048-token
prefill chunk).

## Landed

- `core/pipeline`: `PipelineStagePartition` (full-attention layers balanced evenly, GDN fills),
  `DeviceGroup` (uniform compute capability), `PipelineTransport` (stream-ordered pinned staging
  with per-direction completion events), `PipelineContext` (stage devices + partition + embedding
  replicas, Engine-lifetime owner).
- Weight placement: `materialize()` places each tensor on its stage via the registered naming
  convention (`text/layers/<L>/...` follow the partition; `text/final_norm`, `text/output_head`,
  `text/draft_head*`, `mtp/*` on the last stage; `text/token_embedding`, `vision/*` on stage 0),
  with per-stage arenas, per-stage free-VRAM checks, and a fan-out of the direct-I/O upload
  pipeline across stage transfer streams. The token-embedding table is replicated onto every
  non-primary stage (flag `--pp-no-embed-replica`) so MTP gathers draft embeddings locally.
- Surface: `--pp N`, `--pp-devices ID,...`, `--pp-no-embed-replica` on `ninfer` and `ninfer-serve`;
  startup rejects masked-block speculative decoding and Vision under `--pp>1`; CUDA Graph decode
  capture is disabled for `--pp>1` (eager decode) until per-stage capture lands.
- `ninfer_pipeline_test` (partition invariants, transport round-trip on one and two devices) and
  the opt-in `ninfer_qwen3_6_27b_pipeline_placement_real_test` (real artifact, two devices).
- Hardware-verified (2x V100-PCIE-32GB, qwen3.8-27b nvfp4): a `--pp 2` load places ~10.2 GiB on
  stage 0 and ~11.4 GiB on stage 1 (weight share + embedding replica + last-stage MTP KV) and
  completes Program startup. Device-context hygiene this exposed: staging-slot events are created
  per stage device, load-time FP8/NVFP4 QPN prepacks run under a weight-device guard, and
  construct_target/Engine/Program rebind the primary device after staged work (a leaked
  current-device state otherwise cudaMallocs the persistent pool on a non-primary stage).

## Remaining execution work

### 1. Per-stage decoder state (KV) and GDN state — LANDED (pools + mirrors)

`ProgramImplCore` now builds a `PipelineExecution` (targets/qwen3_6/impl/runtime/
pipeline_runtime.h) under `--pp>1`: each non-primary stage gets a persistent backing, a workspace
arena, a passive-mirror `DecoderState` (text KV planes + execution tables at the same logical
capacity with the stage's layer subsets), and a stage-local `StateImageDevicePool` (GDN linear
state). The core pools carry single-writer mirror hooks — `DeviceKVPagePool::set_mirror`,
`KVExecutionTablePool::set_mirror`, `LinearAttentionStatePool::set_mirror` — that forward content
mutations (page copies, table row writes, slot copies/zeroes) to identical physical indices and
row ids; deterministic free-run allocation keeps indices equal, so the primary alone owns
allocation state and the mirrors stay passive. All hooks forward transitively for N-stage chains.

Still open in this area: KV capacity resolution budgets only the primary device's free bytes
(`registry.cpp` `current_free_device_bytes()`); switch it to `min` across stages minus per-stage
fixed costs (weights already excluded, embedding replica on the last stage, stage workspaces) so
`--kv-capacity auto` cannot over-commit a non-primary stage.

### 2. Stage-aware TextContext execution — LANDED (one open defect)

Stage switching is implemented: `advance_stage` transports the hidden activation and mirrors the
per-round control tensors (positions, KV rows, slot tensors, valid columns, and the device
sampling-config array) into the entering stage's workspace, `run_layers` switches at the partition
boundary, `attn_mix`/`gdn_mix` select stage-local cache views and linear pools with local
ordinals, and the decode/prefill tails compute final norm + lm_head + sampling on the last stage
and ship outputs back through the backward transport. PP=1 is bit-identical to the historical path
(verified: greedy outputs unchanged).

RESOLVED: the sampler fault had two stacked causes - the SamplingConfig embeds a device pointer
to the stage-0 penalty counts (now nulled in the stage-local mirror; penalties are therefore
inert on non-primary stages), and the boundary transports raced: `DeviceGroup` constructed a
duplicate stage-0 DeviceContext whose stream differs from the Engine primary's, so boundary
copies were unordered against stage-0 compute. Stage 0 now references the Engine's own context,
and each transport records a consumed-side event its source stream waits on before reusing the
pinned staging area.

Status: WORKING with CUDA Graphs at decode parity. `--pp 2` generation runs end-to-end on
2x V100-PCIE-32GB with the qwen3.8-27b nvfp4 artifact, captured as two per-stage graphs per
topology (stage-0 forward segment, last-stage segment) chained per round by host-side events
over slotted pinned staging, with an eager stage-0 tail (restore, scatter, sample, egress).
Measured on a 200-token generation run: decode 29.3 tok/s (PP=2) vs 28.7 tok/s (PP=1) — parity,
not a gap; prefill 199.7 vs 199.4 tok/s. Greedy outputs are byte-identical to graphed PP=1
across short, creative, and multi-chunk prompts. A 262144-token KV capacity that cannot fit one
V100 (19.1 GB runtime needed, 13.4 GB available) loads and generates on two. The earlier
"~20% slower" reading was total-wall-clock polluted by a cold-cache artifact upload; warm-cache
loads match PP=1 (2.4 s / 7.8 GiB/s). Known v1 limitation: sampling presence/frequency penalties are inert on
non-primary stages (their counts array is stage-0 resident); reject penalty-bearing requests at
the serving layer if exactness under penalties is required. Open: MTP speculative decoding under
PP (the throughput leader on a single GPU), long-context output sweep, per-stage graph coverage
for the MTP round.

`TextContext` gains a trailing `qwen3_6::PipelineExecution*` constructor parameter (defaulted
null) and members `active_ctx_` (defaults `&ctx_`), `stage_work_` (defaults `&work_`),
`active_stage_`. The mechanical part is contained in `text_context_impl.h`:

- `ctx_.` (20 sites) resolves through `active_ctx_`; `work_` (76 sites, word-boundary —
  `workspace_recipe` is untouched) through `stage_work_`; the six `state_.` sites inside
  `gdn_mix` resolve through an `active_linear(global_gidx)` helper returning `{pool, local_layer}`
  with `local = global - (first_layer(stage) - full_attention_base(stage))`. `attn_mix` selects
  the stage cache view the same way:
  `pipeline_execution->stage_decoder(stage).text_kv.batch_layer_view(local_fidx)`.
- `advance_stage(Tensor& x)` runs at the partition boundary inside `run_layers`: enqueue the
  forward transport for the hidden tensor into a same-shape stage-local workspace tensor, rebind
  the caller's view by Tensor assignment (Tensor is a lightweight view), bind the next stage
  device, reset + adopt its workspace, advance `active_stage_`, and rebind the control mirrors.
- Per-round control tensors (`cache_positions`, `rope_positions`, `kv_table_rows`,
  `valid_columns`, linear-state slot tensors) mirror at the entry of each public call via forward
  transports into stage-local copies; the `active_*` members point at the copies. The envelope is
  a host struct (`CausalAttentionExecutionEnvelope` is two u32s) and needs no copy.
- Outputs ship back before returning: `logits` (last-stage lm_head) and any caller-visible hidden
  go through the backward transport into the caller-provided stage-0 tensors, so Program's
  scatter/sample/egress mechanics on stage 0 are unchanged. The embedding gather stays on
  stage 0; final norm and `lm_head` are placed on the last stage already.
- `advance_stage` must also reset the entering stage's workspace arena per call (nothing
  cross-call persists there; all long-lived state lives in the pools).
- Embedding gather stays on stage 0; final norm + `lm_head` + proposal head on the last stage.
- Prefill flows chunks through stages sequentially; the rewrite-checkpoint hidden is captured on
  the last stage and shipped back like any output.

### 3. MTP on the last stage — implemented, one correctness defect open

LANDS (all gated off at startup for `--pp>1` until the defect below is fixed):
- Stage-local MTP KV pool + stage decode frame (full `MtpDecodeState` mirror, layout-cloned from
  the primary), per-stage GDN replay records AND their accept-fold plans; the settle path executes
  the primary fold plus every stage fold.
- The pipeline round body: primary ingress upload + verify-input prep, whole-ingress/verify-ids/
  target-positions/RoPE-delta ships, the verify traversal with mirror outputs, then acceptance,
  `mtp_prepare_next_round`, MTP forward, proposal, and the draft loop entirely on the last stage
  against the mirror frame, with egress-region and continuation-row ships back. Draft-token
  embedding gathers through the replica view.
- The GDN replay-fold op's launcher now takes the record's own layer count for its grid (stage
  subsets fold correctly; head/channel geometries stay pinned) and `is_registered_fold_geometry`
  admits contiguous layer subsets.
- Startup validates `--pp>1` with any speculative backend as an explicit product gate.

OPEN DEFECT: from the second decode round on, the last stage's verify predicts a different token
than the reference for the same prefix (e.g. after "Tokyo" it predicts EOT instead of "."). Round
one is correct; PP1 MTP and plain PP2 decode are both correct, and the first divergence is the
anchor column of round two, so the stage-1 GDN/KV state restored after the round-one rollback is
suspect (the fold executes with the expected commit count on the stage stream). Next step: dump
the stage-1 GDN state and target-KV row contents after the fold and diff against an equivalent
PP1 run at the same frontier.

Landed (dormant while `--pp>1` rejects `--spec`; verified compile-clean with the full suite):
- Last-stage MTP KV pool planned inside the stage decoder with the primary's page-group count;
  the primary MTP pool/tables register it as a passive mirror, so Program's MTP page lifecycle
  mirrors content by identical physical indices exactly like the text KV.
- Stage-local GDN replay records (both the verify-record and MTP-lookup kinds) planned with the
  stage's GDN layer count before the arena sizing and constructed on the stage backing.
- The embedding replica is published as a full `Weight` view per non-primary stage (all device
  pointers rebased onto the replica buffer), reachable through
  `PipelineExecution::embedding_view(stage)`.

Remaining segmentation contract:
1. MTP attention on the last stage must read the STAGE-local MTP cache. Add
   `PagedKVCacheView` construction from a `PagedKVCache` + row INDEX (the mirror tables have no
   lease lifecycle), and route `mtp_kv_view(sequence)`/`MtpBatchContext.mtp_cache` to
   `pipeline_execution->stage_decoder(last)` when a pipeline is active. Replace
   `batch_mtp_kv_->batch_layer_view(0)` and `io_.backend_kv_table_row` uses in the MTP paths
   with the stage-local cache plus its block-table matrix, and use the replica embedding view in
   `mtp_forward_stem`/bridge sites (the canonical `embed_` stays for stage-0 segments).
2. `io_.mtp` round state (`position`, `draft_tokens`, `ar_hidden`, `target_positions`, proposal
   buffers) and the MTP decode frame tensors are primary-device allocations consumed by last-stage
   kernels. Mirror the frame into the last stage workspace (the same fixed-mirror pattern as
   `OrdinaryGraphMirrors`, one for the MTP round), and ship host-visible results back through the
   backward transport before the egress D2H: accepted ids, per-row accepted counts, continuation
   hidden rows, and the final draft window.
3. The acceptance/replay ops (`ops::mtp_round_*`, replay apply, continuation scatter) run where
   their operands live: acceptance on the last stage (its mirrors), continuation scatter on the
   primary after the ship-back, replay application per stage through the mirrored record pools.
4. Capture as `capture_mtp_decode_graphs` with the same two-graph structure (graph 0 up to the
   boundary, graph 1 through proposal + ship-back), per-round driver chained by the existing
   events, then flip the `--spec` gate for `--pp>1` and run the exactness battery
   (`--spec mtp --draft-tokens 3`, greedy, PP=1 vs PP=2 must match).

DFlash2 stays rejected (its draft taps intermediate target layers across the boundary); Vision
stays rejected (its prefill tap contract spans stages).

### 4. Per-stage CUDA Graph decode (deferred)

Capture the stage-0 segment and the stage-1 segment into per-device graphs; the boundary copy is
host-ordered between launches (launch stage-0 graph, destination stream waits the source event,
launch stage-1 graph, output staging eager). The tail ops (scatter, sample, egress D2H) stay on
stage 0 by shipping logits back, or move to the last stage in a follow-up that ships only tokens.

## Verification contract

- Exactness: greedy decode PP=1 vs PP=2 (`--pp-devices 1,2`) must produce identical token streams
  on a fixed prompt set including long contexts; both stages run identical kernels in identical
  order, so any divergence is a wiring or ordering bug.
- Capacity: a `--kv-capacity`/`--max-context` configuration that exceeds one V100 must load and
  serve on two; this is the product claim.
- Performance: record prefill/decode tok/s and TTFT PP=1 vs PP=2 (expect a per-token PCIe hop
  cost; capacity is the deliverable) and MTP acceptance-rate equality.
- Placement: the opt-in real test asserts both stages carry weight bytes and stage 1 carries the
  replica.

## Reference hardware note

The reference V100 pair reports `cudaDeviceCanAccessPeer = 0` in both directions (PCIe, cross
NUMA). `PipelineTransport` keeps a peer-direct branch point; hardware that reports peer access can
enable the fast path by testing capability at construction.
