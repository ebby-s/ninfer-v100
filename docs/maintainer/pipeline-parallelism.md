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

## Remaining execution work

### 1. Per-stage decoder state (KV) and GDN state

`ProgramImplCore` plans one `DecoderState` (text KV + optional MTP KV) and one
`StateImageDevicePool` (GDN linear state + continuation hidden) in the persistent backing on the
primary device. Split by stage:

- Extend the persistent layout planning (`layouts_impl.h` `plan_persistent_layout`) to produce one
  layout per stage with the stage's layer subsets: text KV planes =
  `partition.full_attention_count(stage)`; state-image linear layers = `partition.gdn_count(stage)`;
  MTP KV and MTP replay records only on the last stage. Capacity (`page_group_count`) is shared:
  every stage plans the same logical capacity, and per-stage byte budgets come from that stage's
  free VRAM minus its weights and workspaces. KV capacity resolution
  (`runtime::resolve_kv_capacity`) must budget against summed stage free bytes (the preflight in
  `registry.cpp` already does this for weights).
- Allocate one persistent `DeviceArena` per stage on its device; construct per-stage
  `DecoderState`/`StateImageDevicePool`. Stage 0 keeps the existing member roles so the
  single-device path is untouched.
- `LogicalKVPageStore` + `KVAddressSpaceStore` funnel the whole physical page lifecycle
  (`materialize`, `dematerialize`, `resize_reservation`, `copy_page`, `zero_pages`, `publish`).
  Add a lockstep stage-1 mirror: identical logical operations drive the stage-1 pool and tables in
  the same order. Logical addressing stays shared (row indices, page-group identity); physical
  placement and table contents differ per device. Free-run allocation order is deterministic, so
  both pools derive identical physical index sequences from the same operation stream.
- `StateImageStore` slot operations (`copy_slot`, `zero_slot`, hidden store) mirror per stage the
  same way; `set_linear_state_slots` in TextContext fans out to both stage pools with local layer
  indices.

### 2. Stage-aware TextContext execution

`TextContext` runs all layers on one `ctx_.stream`. Introduce an optional `PipelineContext*` +
per-stage resources (work arena, text cache, linear pool, transport channel, workspace mirrors):

- `run_layers` switches stage at the partition boundary: record the source event, transport the
  hidden tensor into the next stage's workspace mirror, continue with that stage's stream, cache
  views, and pools. Layer-index mapping: full ordinal `f` owns global layer `4f+3`; GDN ordinal
  `g` owns global layer `4*(g/3) + g%3`; each stage indexes its pools with local ordinals
  (`f - full_attention_base(stage)`, `g - gdn_base(stage)`).
- `attn_mix` selects `stage_cache(stage)->batch_layer_view(local_fidx)`; `gdn_mix` selects the
  stage's linear pool view with its local ordinal.
- Per-round control tensors (`RoundState io`, positions, kv table rows, valid columns, slot
  indices, envelope) mirror into stage-1 device buffers at the entry of each public call; they are
  KB-scale. Outputs (`logits`, hidden egress, sampled tokens) copy back to the caller-provided
  stage-0 tensors via the transport before returning, so Program's D2H egress mechanics are
  unchanged.
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
