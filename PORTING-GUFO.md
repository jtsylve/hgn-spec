# Porting to gufo

Status: planned against gufo commit `9cad139` (2026-09-24). Paths are relative
to the gufo repository root.

This guide applies the [porting guide](PORTING.md) to
[gufo](https://github.com/gufo-org/gufo). Read that guide first; this one says
where each step lands in gufo and which gufo parts it reuses. It follows
gufo's rules: quality comes first, a retained optimization becomes the default
path, and every layout change is an experiment with evidence. No HGN kernel
has run in gufo yet.

## Gap analysis

| Area | gufo today | HGN needs |
| --- | --- | --- |
| Container | GGUF only (`src/core/gguf_reader.*`). No tensor-source interface | An HGN reader |
| Residency | Flash-Next: one `hipMalloc` for each tensor, O_DIRECT upload through pinned staging (`src/core/hip/weight_upload.*`) | Either residency model works |
| Dense matrices | Q8_0 in the fused kernels; BF16, F16 and F32 in some roles | w4b: `q4c` and `bf16`. Its overlay: `q8g64` for `o_proj`. v2 and ht43: `ht`, plus `q6g64` (v2) or `q8g64` (ht43) for the hyper-connection matrices. GGUF imports: `q8s32` |
| Routed experts | Q4_K, Q5_K, Q5_1 and Q8_0 (vendored MMQ and MMVQ, F16 WMMA routes) | w4b: `q4c`. v2: `i4r`. ht43: `ht`. GGUF imports: `q4a` and `q5a` variant 1, `q5a` variant 0, `q8s32` |
| Rotation vectors | Not checked in this revision | `f16` vectors for every `ht` and `i4r` tensor of v2 and ht43 (see [Rotation vectors](TENSORS.md#rotation-vectors)) |
| Embedding | Q8_0, Q6_K, BF16, F16 or F32 rows | `q4c` |
| Q6_K | Rejected on the HIP path (`kernels/rocm/device_model.cpp`) | `q6k` in GGUF imports |
| MTP head | Sidecar GGUF, `nextn.*` and `blk.48.*`, shared Q8_0 | `mtp.*`: `q4c`, or `q8g64` from the overlay, the head file, v2 or ht43 |
| n-gram table | IQ4_NL or BF16, dequantized to F32 on the CPU (`ngram.*`) | `fp8g` (published, in w4b or in the [n-gram table file](CONTAINER.md#n-gram-table-file)), or `iq4nl`, `q8_0`, `q5_0` or `q5_1` |
| DeltaNet head order | Tiled: value head `h` reads key head `h mod 16` | Grouped: key head `h / 3` |
| Norm gains | GGUF convention: the 1 is already added | Zero-centred for the [listed norms](TENSORS.md#zero-centred-norm-gains) |
| `A_log` | GGUF `ssm_a = −exp(A_log)`, F32 | `A_log` as `bf16` |
| Vision | `mmproj` GGUF | HGN sidecar, `visual.*`, `bf16` |

The paths under `kernels/rocm/` and the files `ngram.*`, `weights.*`,
`config.*`, `executor.cpp` and `reference.cpp` are in
`src/models/qwen38_flash_next/`.

## The reader

Add `src/core/hgn_reader.{hpp,cpp}` beside the GGUF reader, as the
[porting guide](PORTING.md#a-reader) describes.

- Expose the mapped file as one region with a file descriptor, in the same
  form as the `GgufMappedRegion` of a GGUF shard. `hip::WeightUpload` then
  copies HGN payloads without change.
- Keep a store-code enum of its own. Do not add HGN codes to `GgmlType`.
- Put the size rules and CPU decoders in `src/core/quant`, beside the GGML
  dequantizers. The CPU oracle needs them.

Add a target `hgn_reader_test` with labels `cpu;core;hgn`. It is a cheap host
contract like `gguf_reader_test`, so add it to `gufo_pr_targets` in
`cmake/Checks.cmake`.

## Binding

`Binder::Get` (`weights.cpp`) checks the GGUF type and shape of a tensor and
returns a `TensorRef`. Add an HGN path that fills the same roles. Record the
store code and variant in `TensorRef`, next to the GGML type.

Split the fused tensors into the tensors gufo binds today: `ffn_gate_exps` and
`ffn_up_exps` from `mlp.experts.gate_up_proj.weight`, and `indexer.q_proj` and
`indexer.k_proj` from `self_attn.indexer.index_qk_proj.weight` (see
[the split table](PORTING.md#binding-tensors-to-model-roles)). Keeping the
fused expert form for one gated kernel is a later experiment.

gufo already joins `attn_qkv` with `attn_gate`, and `attn_q`, `attn_k` and
`attn_v`, at upload. HGN keeps these tensors separate, as GGUF does, so the
joins apply unchanged.

## Value conventions

gufo expects the GGUF conventions, so apply the
[conversions](PORTING.md#value-conventions) at upload:

- Norm gains: gufo's RMSNorm multiplies by the stored gain and never adds 1.
  Widen the zero-centred gains to F32 and add 1.0.
- `A_log`: gufo's recurrence uses `ssm_a`. Compute `ssm_a = −exp(A_log)` in
  `f32`.
- The router: HGN stores `mlp.gate.weight` as `bf16`. gufo narrows its router
  to F16, which is a separate rounding.

**Head order.** gufo's kernels read key head `h mod 16` for value head `h`:
four places in `kernels.hip.cpp` and one in `reference.cpp`. Reorder the HGN
tensors to the tiled order at upload rather than change them. That keeps one
head order in the model code.

## The transform stage

`hip::WeightUpload` reads with `pread` into pinned staging and copies to the
device asynchronously. Add a transform between the read and the copy for the
row reorders, the fused splits, the rebuilt GGUF blocks and the small-tensor
conversions (see [A transform stage](PORTING.md#a-transform-stage)).

## Storage support

| Store | First path in gufo | Performance path |
| --- | --- | --- |
| `q4c`, learned codebook | None | New `q4c` kernels, floating-point form. Required for every published checkpoint |
| `q4c`, Q4_0 or IQ4 codebook | Rebuild Q4_0 or IQ4_NL blocks. gufo has no Flash-Next kernels for these yet | New `q4c` kernels, integer form |
| `q8g64` | Decode to F16 at load | A new `q8g64` kernel |
| `q6g64` | Decode to F16 at load | A new `q6g64` kernel, as the `q8g64` one |
| `ht` | Decode the dense matrices to the original basis at load | New [ht kernels](PORTING.md#the-ht-kernels). Required for the ht43 experts |
| `i4r` | None | New [i4r kernels](PORTING.md#the-i4r-kernels). Required for v2 |
| `q8s32` | Rebuild Q8_0 blocks; the Q8_0 kernels run them | — |
| `q5a` variant 0, `q4a` and `q5a` variant 1 | Rebuild Q5_1, Q4_K and Q5_K blocks; the expert kernels run them | Planar affine kernels, as an experiment |
| `q6k` | Instantiate the vendored MMQ/MMVQ Q6_K code | — |
| `bf16` | `SmallGemm` for 1–8 rows, hipBLASLt for wide batches | — |
| `fp8g` table | A decoder in `ngram.*` | Raw-row upload |
| `iq4nl` table | The current `NgramTable`, with the HGN offset | Raw-row upload |
| `q8_0`, `q5_0`, `q5_1` tables | A GGUF row decoder in `ngram.*`, with the HGN offset | Raw-row upload |

v2 and ht43 hold no n-gram table. Read the table from the
[n-gram table file](CONTAINER.md#n-gram-table-file) for them.

### q4c in gufo

- **Integer form.** It matches gufo's decode arithmetic: Q8_1 activations and
  the `sudot4` of the MMVQ tier. Expand the 16 `int8` levels with the byte
  permute that the vendored IQ4_NL code uses (`mmq/vecdotq.hpp`). A 32-value
  `q4c` group is one Q8_1 block, so the scales line up.
- **Floating-point form.** Use an `f16` byte table with `v_dot2_f32_f16` and
  F16 activations. gufo's WMMA routes already narrow activations to F16.
- **Routed experts.** Keep gufo's MoE structure: grouping across the batch,
  wave64 tiles above 8 rows and the routed F16 WMMA route for prefill. Change
  only the weight decode.
- **Prefill.** The F16 WMMA route decodes Q8_0 into F16 while it stages the
  tile in LDS; decode a `q4c` tile the same way. The W8A8 integer WMMA route
  applies scales for each block of 32 along `K`, which is one `q4c` group.

### q8g64 in gufo

gufo's Q8_1 activation blocks carry `Σ x` for each block of 32. A `q8g64`
group of 64 takes two blocks: `scale × Σ (code × x) + bias × Σ x`.

### Affine formats in gufo

Rebuilt Q5_1, Q4_K and Q5_K blocks run on gufo's existing expert kernels with
no other change. gufo's `EXPERIMENTS.md` rejected a persistent private layout
for the SSM weights because decoding got slower. Try the planar affine layout
only as an experiment with the same promotion rule.

## Residency

gufo's Flash-Next loader copies each tensor into `hipMalloc` memory with
O_DIRECT reads. gufo's Qwen3.8 27B loader copies each shard into an anonymous
huge-page region and registers it (`src/models/qwen/hip/model_loader.cpp`).
Start with the Flash-Next upload path and the transform stage. The
[residency rules](PORTING.md#memory-residency) apply if the loader later maps
HGN files directly.

## The n-gram table

`NgramTable` (`ngram.*`) reads rows with O_DIRECT and decodes them to F32 on
the CPU. Raw-row upload with a GPU decode is a gufo experiment: it moves 90 or
160 bytes for each row instead of 640 (see
[The n-gram table](PORTING.md#the-n-gram-table)).

## MTP head

gufo binds the MTP head with the GGUF names of the
[MTP table](PORTING.md#mtp-head). It already splits `nextn.eh_proj` into
embedding and hidden halves at upload (`kernels/rocm/device_model.cpp`), so the
two HGN projections bind to the halves directly. The HGN head needs the `q4c`
expert kernels.

## The 27B checkpoint

gufo runs Qwen3.8 27B from GGUF, with the z-lab DFlash2 drafter. The
[27B formats](PORTING.md#the-27b-checkpoint) need new kernels in gufo. The
halogen 27B drafter is a different training, so gufo's DFlash2 path does not
apply to it without a separate qualification.

## Qualification

| Check | Target or CTest name | Labels |
| --- | --- | --- |
| Container | `hgn_reader_test` | `cpu;core;hgn` |
| Storage decode | `hgn_decode_test` | `cpu;core;quant` |
| Transform | `qwen38_flash_next.hgn_transform` | `qwen38_flash_next;cpu` |
| `q4c` kernels | `qwen38_flash_next.q4c_ops` | `qwen38_flash_next;hip;gpu;gfx1151` |
| `q8g64` kernel | `qwen38_flash_next.q8g64_ops` | `qwen38_flash_next;hip;gpu;gfx1151` |
| Model, exact | `qwen38_flash_next_session_test` with an HGN `--model` | model tests |
| Model, published | The same, with w4b and its overlay | model tests |
| Model, v2 and ht43 | The same, with v2 or ht43 and the n-gram table file | model tests |

The content of each check is in [Qualification](PORTING.md#qualification).
Use `tests/models/qwen38_flash_next/routed_wmma_ops_test.cpp` as the model for
the kernel tests. For the exact-model check, convert the UD-Q4_K_XL build that
gufo supports (see [the worked example](GGUF-IMPORT.md#worked-example)).

Measure the quality of each published checkpoint with the precedent in
`docs/models/qwen3.8-27b/QUALITY.md`. Record each retained or rejected layout
in `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` in the table form
`| Experiment | Decision / evidence |`.

## Milestones

The [milestones](PORTING.md#milestones) of the porting guide apply. In gufo,
milestone 2 uses the UD-Q4_K_XL build and must match the GGUF path's logits
bit for bit; milestone 3 also gives gufo IQ4_XS, IQ4_NL and IQ3_S expert
support, which it does not have today. Milestone 5, v2 and ht43, is later
work. It needs the rotated formats, which no gufo plan covers yet.
