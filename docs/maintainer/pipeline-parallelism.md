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

OPEN DEFECT: under `--pp 2`, the prefill-finalize `ops::sample` on the last stage faults with an
async illegal address (`sampling.cu:45`, multiblock partial-topk kernel). All kernel arguments are
verified stage-local mirrors; prime suspects are the sampler's `layout.bind(workspace)` scratch
overlapping the mirrors allocated earlier from the same stage arena (no scope isolation), or a
stale device pointer among the tail operands. Next step: one compute-sanitizer pass over a
`--pp 2` run to identify the faulting pointer, then fix accordingly.

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

### 3. MTP on the last stage (after plain decode verifies)

`mtp_forward_*` and the proposal head run entirely on the last stage: the backbone output hidden
is produced there and draft-token embeddings gather from the local replica. `target_verify`
re-enters stage 0 per accepted round. GDN replay records split per stage with local ordinals.
DFlash2 stays rejected (its draft taps intermediate target layers across the boundary); Vision
stays rejected (its features feed stage-0 embeddings but its prefill tap contract spans stages).

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
