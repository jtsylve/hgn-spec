# Porting to gufo

Status: planned against gufo commit `dea22ce` (2026-10-08). Paths are relative
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
| Model constants and tokenizer | GGUF metadata: `Config::FromGguf` (`config.*`), `QwenTokenizer::CreateFromGguf` and the chat template (`src/models/qwen/`) | Another source. HGN files hold neither (see [Model constants and tokenizer](#model-constants-and-tokenizer)) |
| Residency | Flash-Next: one `hipMalloc` for each tensor, O_DIRECT upload through pinned staging (`src/core/hip/weight_upload.*`) | Either residency model works |
| Dense matrices | Q8_0 in the fused kernels; BF16, F16 and F32 in some roles | w4b: `q4c` and `bf16`. Its overlay: `q8g64` for `o_proj`. v2 and ht43: `ht`, plus `q6g64` (v2) or `q8g64` (ht43) for the hyper-connection matrices. GGUF imports: `q8s32` |
| Small tensors | F32 only: norms, `ffn_gate_inp`, `ffn_gate_inp_shexp`, `ssm_conv1d`, `ssm_dt.bias`, `ple_conv1d`. F32, BF16 or Q8_0: `ssm_alpha`, `ssm_beta`, `hc_*_inject` | `bf16`. `q4c` for `shared_expert_gate`, `in_proj_a` and `in_proj_b`. `q4c`, `q6g64` or `q8g64` for `block_inject_weight` |
| Routed experts | Q4_K, Q5_K, Q5_1 and Q8_0 (vendored MMQ and MMVQ, F16 WMMA routes) | w4b: `q4c`. v2: `i4r`. ht43: `ht`. GGUF imports: `q4a` and `q5a` variant 1, `q5a` variant 0, `q8s32` |
| Rotation vectors | None in Flash-Next. The only 128-point Walsh–Hadamard transform is in the DeepSeek V4 Flash indexer (`src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_indexer.hip.hpp`) | `f16` vectors for every `ht` and `i4r` tensor of v2 and ht43 (see [Rotation vectors](TENSORS.md#rotation-vectors)) |
| Embedding | Q8_0, BF16, F16 or F32 rows | `q4c` |
| Q6_K | Bound for the embedding, `output` and experts, then rejected on the HIP path (`kernels/rocm/device_model.cpp`) | `q6k` in GGUF imports |
| MTP head | A second GGUF named by `--mtp-model`: `nextn.*` and `blk.48.*`, shared Q8_0 | `mtp.*` in the checkpoint or in the head file: `q4c`, or `q8g64` from the overlay, the head file, v2 or ht43 |
| n-gram table | IQ4_NL or BF16 in the trunk GGUF, dequantized to F32 on the CPU (`ngram.*`) | `fp8g` (published, in w4b or in the [n-gram table file](CONTAINER.md#n-gram-table-file)), or `iq4nl`, `q8_0`, `q5_0` or `q5_1` |
| DeltaNet head order | Tiled: value head `h` reads key head `h mod 16` | Grouped: key head `h / 3` |
| Norm gains | GGUF convention: the 1 is already added | Zero-centred for the [listed norms](TENSORS.md#zero-centred-norm-gains) |
| `A_log` | GGUF `ssm_a = −exp(A_log)`, F32 | `A_log` as `bf16` |
| Vision | `mmproj` GGUF | HGN sidecar, `visual.*`, `bf16` |
| Cache identity | SHA-256 of every byte of every GGUF shard (`src/core/gguf_identity.*`) | The same for each HGN file (see [Cache identity](#cache-identity)) |

The paths under `kernels/rocm/` and the files `engine.cpp`, `ngram.*`,
`weights.*`, `config.*` and `reference.cpp` are in
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
- Check every `ht` and `i4r` entry at open. Refuse duplicate names and a
  table that does not fit in the file.
- Open the companion files with the same reader: the overlay, the MTP head
  file, the n-gram table file and the vision sidecar.

Add a target `hgn_reader_test` with labels `cpu;core;hgn`. Use the minimal
file and the malformed copies of the porting guide as its fixtures. It is a
cheap host contract like `gguf_reader_test`, so add it to `gufo_pr_targets` in
`cmake/Checks.cmake`.

## Model constants and tokenizer

HGN files have no metadata and no tokenizer. `Model::Load` (`engine.cpp`)
takes both from the GGUF file:

- `Config::FromGguf` reads the `qwen4exp.*` keys.
- `QwenTokenizer::CreateFromGguf` reads the vocabulary and the merges.
- The chat template comes from `tokenizer.chat_template`
  (`src/models/qwen/chat_template.cpp`).

For an HGN checkpoint:

- Build the `Config` from the [model constants](TENSORS.md#model-constants).
  Take the three PLE arrays from the `i64` tensors of layer 1 (see
  [Metadata arrays](GGUF-IMPORT.md#metadata-arrays)).
- HGN does not give the RMS epsilon, the RoPE values, the context length, the
  indexer top-k, the compression ratio, or the n-gram size, heads for each
  n-gram and end token of the n-gram hash. Build them in, or read them from a
  GGUF file of the same model.
- Load the tokenizer with `QwenTokenizer::CreateFromVocabulary`, or from a
  GGUF file of the same model. Take the chat template from the same source.

## Binding

`Binder::Get` (`weights.cpp`) checks the GGUF type and shape of a tensor and
returns a `TensorRef`. Add an HGN path that fills the same roles. Record the
store code and variant in `TensorRef`, next to the GGML type.

- `TensorRef` lists dimensions in GGUF order: `cols` is the innermost. Bind an
  HGN matrix `[N, K]` as `rows = N`, `cols = K`, and an expert tensor
  `[512, R, K]` with `experts = 512`.
- `TensorRef` computes its size from the GGML block geometry. Take the size of
  an HGN payload from its table entry. After a transform, describe the device
  tensor by the type that the transform makes.
- Bind the [rotation vectors](TENSORS.md#rotation-vectors) of each `ht` and
  `i4r` tensor to the same role. `Binder::Get` compares the whole shape. For a
  rotation vector, compare only the element count.

Split the fused tensors into the tensors gufo binds today: `ffn_gate_exps` and
`ffn_up_exps` from `mlp.experts.gate_up_proj.weight`, and `indexer.q_proj` and
`indexer.k_proj` from `self_attn.indexer.index_qk_proj.weight` (see
[the split table](PORTING.md#binding-tensors-to-model-roles)). Split an `ht`
expert tensor by tiles and an `i4r` tensor by rows. Each part keeps the input
vector `su` and the `svh` values of its rows. Keeping the fused expert form
for one gated kernel is a later experiment.

gufo stacks `attn_qkv` with `attn_gate`, and `attn_q`, `attn_k` and `attn_v`,
at upload when all parts are Q8_0. HGN keeps these tensors separate, as GGUF
does, so the joins apply unchanged to rebuilt Q8_0 blocks. Tensors of other
stores stay separate.

## Value conventions

gufo expects the GGUF conventions, so apply the
[conversions](PORTING.md#value-conventions) at upload:

- Norm gains: gufo's RMSNorm multiplies by the stored gain and never adds 1.
  Widen the zero-centred gains to F32 and add 1.0.
- `A_log`: gufo's recurrence uses `ssm_a`. Compute `ssm_a = −exp(A_log)` in
  `f32`.
- `dt_bias` and the convolutions: gufo binds `ssm_dt.bias`, `ssm_conv1d` and
  `ple_conv1d` as F32 only. Widen the `bf16` values.
- The router: gufo stacks `ffn_gate_inp` and `ffn_gate_inp_shexp` into one F32
  tensor and narrows it to F16 (`Stack` in `kernels/rocm/device_model.cpp`).
  Widen the `bf16` router and decode the `q4c` `shared_expert_gate` to F32.
  The narrowing to F16 is a separate rounding.
- `in_proj_a` and `in_proj_b`: gufo stacks `ssm_alpha` and `ssm_beta` the same
  way, from F32 or Q8_0. Decode the `q4c` tensors to F32.
- `block_inject_weight`: gufo accepts F32, BF16 or Q8_0 for `hc_*_inject`, but
  not F16. Decode it to F32.

**Head order.** gufo's kernels read key head `h mod 16` for value head `h`:
four places in `kernels.hip.cpp` and one in `reference.cpp`. Reorder the HGN
tensors to the tiled order at upload rather than change them. That keeps one
head order in the model code. When the transform decodes an `ht` matrix,
reorder the decoded rows and columns. A path that keeps `ht` moves whole
tiles with their `svh` and `suh` values (see
[the porting guide](PORTING.md#value-conventions)).

## The transform stage

`hip::WeightUpload` reads with `pread` into pinned staging and copies to the
device asynchronously. Add a transform between the read and the copy for the
row reorders, the fused splits, the rebuilt GGUF blocks, the decodes at load
and the small-tensor conversions (see
[A transform stage](PORTING.md#a-transform-stage)).

`WeightUpload::Copy` cuts each tensor into 16 MiB tasks at byte boundaries,
and each task reads one range. A transform task needs a band of whole rows,
with one range from each plane of a planar store. A decode at load writes
more bytes than it reads: `ht` to F16 writes 4 times the bytes of a 4-bit
payload. Size each band by its output.

## Storage support

| Store | First path in gufo | Performance path |
| --- | --- | --- |
| `q4c`, learned codebook | Decode `embed_tokens` to BF16 at load, as halogen-flash does. Decode the [small tensors](#value-conventions) to F32. None for the other tensors | New `q4c` kernels, floating-point form. Required for w4b and for the MTP experts |
| `q4c`, Q4_0 or IQ4 codebook | Rebuild Q4_0 or IQ4_NL blocks. gufo has no Flash-Next kernels for these yet | New `q4c` kernels, integer form |
| `q4c`, IQ3_S codebook | None. gufo has no IQ3_S kernels | New `q4c` kernels, integer form |
| `q8g64` | Decode to F16 at load; to F32 for `block_inject_weight` | A new `q8g64` kernel |
| `q6g64` | Decode to F16 at load; to F32 for `block_inject_weight` | A new `q6g64` kernel, as the `q8g64` one |
| `ht` | Decode the dense matrices to the original basis at load, as F16 or BF16. The F16 and BF16 dense paths run them | New [ht kernels](PORTING.md#the-ht-kernels). Required for the ht43 experts |
| `i4r` | None | New [i4r kernels](PORTING.md#the-i4r-kernels). Required for v2 |
| `f16` rotation vectors | Read by the transform when it decodes to the original basis | Keep as F16, or widen to F32 |
| `q8s32` | Rebuild Q8_0 blocks; the Q8_0 kernels run them | — |
| `q5a` variant 0, `q4a` and `q5a` variant 1 | Rebuild Q5_1, Q4_K and Q5_K blocks; the expert kernels run them | Planar affine kernels, as an experiment |
| `q6k` | Instantiate the vendored MMQ/MMVQ Q6_K code | — |
| `bf16` | `SmallGemm` for 1–8 rows, hipBLASLt for wide batches. For the indexer shapes, gufo's `DenseBf16Gemm` can replace hipBLASLt and gives the same bits | — |
| `fp8g` table | A decoder in `ngram.*` | Raw-row upload |
| `iq4nl` table | The current `NgramTable`, with the HGN offset | Raw-row upload |
| `q8_0`, `q5_0`, `q5_1` tables | A GGUF row decoder in `ngram.*`, with the HGN offset | Raw-row upload |

v2 and ht43 hold no n-gram table. Read the table from the
[n-gram table file](CONTAINER.md#n-gram-table-file) for them (see
[The n-gram table](#the-n-gram-table)).

In v2 and ht43, the routed experts of the MTP head are the only `q4c` tensors
that need a kernel. The other `q4c` tensors decode at load.

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

### q8g64 and q6g64 in gufo

gufo's Q8_1 activation blocks carry `Σ x` for each block of 32. A `q8g64`
group of 64 takes two blocks: `scale × Σ (code × x) + bias × Σ x`. A `q6g64`
kernel uses the same form. It builds each code from the low plane and the
high plane.

### ht and i4r in gufo

- **Input rotation.** gufo narrows each input to F16 once and reuses it for
  every projection of that input. Each `ht` matrix has its own `suh`, so
  rotate and narrow the input once for each matrix. The routed experts of a
  layer share one `su`. Rotate each token row once, in the pass where the
  routed F16 route narrows it.
- **The transform.** The DeepSeek V4 Flash indexer transform does one F32 row
  for each 128-lane workgroup, with the butterflies in LDS. Use it as a test
  reference. For production, fuse the rotation into the narrowing pass.
- **Routed experts.** Keep gufo's grouping and buckets. Apply the gate and up
  output transform before SwiGLU. For `i4r`, apply the down output transform
  once, after the routing-weighted sum. For `ht`, each expert has its own
  down `svh`, so apply the transform and `svh` before the routing weights.
- **Prefill.** The routed F16 WMMA route decodes a weight tile into F16 in
  LDS. Decode an `i4r` tile the same way, with the rotated F16 rows as the
  other operand. For the `ht` dense matrices, start with the original-basis
  copies of the first path. Try a tile decode in LDS as an experiment.

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

Halogen-flash reads `ht` and `i4r` payloads only in place. gufo's upload path
copies them instead; the format does not need the in-place form. `hipMalloc`
memory meets the 64-byte alignment rule of `i4r`. v2 holds no n-gram table, so
all 62.11 GiB of it is resident, before the decodes at load add to it. The
n-gram table file stays out of GPU memory, as the table of a GGUF trunk does
in gufo today.

## The n-gram table

`NgramTable` (`ngram.*`) reads rows with O_DIRECT, keeps a cache of raw rows
and decodes rows to F32 on the CPU. The executor reads the rows of the next
prefill chunk while the current chunk runs. In a long prefill the gather thus
overlaps the forward pass.

`NgramTable::Open` takes a file descriptor, an offset and a GGML type.
`engine.cpp` passes the trunk GGUF and the offset of its table. For v2 and
ht43, add an option that names the n-gram table file, as `--mtp-model` names
the MTP file. Open the file with the HGN reader and pass its descriptor and
the table offset. Check that each n-gram head fits in the table (see
[N-gram table file](CONTAINER.md#n-gram-table-file)).

Raw-row upload with a GPU decode is a gufo experiment. It moves 90 to 170
bytes for each row instead of 640, and it removes the CPU decode (see
[The n-gram table](PORTING.md#the-n-gram-table)). Measure it against the
overlapped gather.

## MTP head

gufo binds the MTP head with the GGUF names of the
[MTP table](PORTING.md#mtp-head). It reads the head from a second GGUF file,
named by `--mtp-model`, with its own reader (`MtpWeights::Bind`). w4b, v2 and
ht43 hold the 31 `mtp.` tensors, so bind them from the checkpoint's reader.
Use the MTP head file only with a trunk that does not hold them.

gufo already splits `nextn.eh_proj` into embedding and hidden halves at upload
(`kernels/rocm/device_model.cpp`), so the two HGN projections bind to the
halves directly. The head's dense projections in v2 and ht43 are `q8g64`. The
HGN head needs the `q4c` expert kernels.

## Cache identity

gufo keys persistent continuations with a SHA-256 of every byte of every GGUF
shard (`src/core/gguf_identity.*`, called from
`src/cli/serve/inference_backend.cpp`). Hash each HGN file that the model reads
in the same way: the checkpoint, the overlay, the MTP head file and the n-gram
table file. Do not use the header identity, because w4b and both of its
overlays have the same identity. Do not use the XOR checksums (see
[Checksum](CONTAINER.md#checksum)).

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
| `q8g64` and `q6g64` kernels | `qwen38_flash_next.q8g64_ops`, `qwen38_flash_next.q6g64_ops` | `qwen38_flash_next;hip;gpu;gfx1151` |
| `ht` and `i4r` kernels | `qwen38_flash_next.ht_ops`, `qwen38_flash_next.i4r_ops` | `qwen38_flash_next;hip;gpu;gfx1151` |
| Rotated decode | `qwen38_flash_next_hgn_decode_probe`, built explicitly like `qwen38_flash_next_reference_probe` | — |
| Model, exact | `qwen38_flash_next_session_test` with an HGN `--model` | Built by `qwen38_flash_next_model_tests` |
| Model, published | The same, with w4b and its overlay | Built by `qwen38_flash_next_model_tests` |
| Model, v2 and ht43 | The same, with v2 or ht43 and the n-gram table file | Built by `qwen38_flash_next_model_tests` |

The content of each check is in [Qualification](PORTING.md#qualification).
Use `tests/models/qwen38_flash_next/routed_wmma_ops_test.cpp` as the model for
the kernel tests. For the exact-model check, convert the UD-Q4_K_XL build that
gufo supports (see [the worked example](GGUF-IMPORT.md#worked-example)). Give
it `ssm_a` from the GGUF file: `−exp(A_log)` matches it only within `f32`
rounding. `session_test` requires `--mtp-model`. w4b, v2 and ht43 hold the
MTP head, so the test needs a form that takes the head from the checkpoint.

Measure the quality of each published checkpoint with the precedent in the
quantization comparison of `docs/models/qwen3.8-27b/QUALITY.md`. gufo's
`AGENTS.md` keeps numerical quality tests in `tests/functional/`, outside
hosted CI. Record each retained or rejected layout in
`docs/models/qwen3.8-flash-next/EXPERIMENTS.md` in the table form
`| Experiment | Decision / evidence |`.

## Milestones

The [milestones](PORTING.md#milestones) of the porting guide apply. In gufo,
milestone 2 uses the UD-Q4_K_XL build and must match the GGUF path's logits
bit for bit; milestone 3 also gives gufo IQ4_XS, IQ4_NL and IQ3_S expert
support, which it does not have today. Milestone 5, v2 and ht43, needs the
`i4r` or `ht` expert kernels and the n-gram table file option. Its `ht` dense
matrices and its `q6g64` and `q8g64` tensors can decode at load first.
Without the MTP head, v2 and ht43 need no `q4c` kernel.
