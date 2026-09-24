# Porting guide

This guide explains how to add HGN checkpoints, or the HGN storage formats, to
an inference engine. It applies to any engine. Guides for specific engines
build on it:

| Engine | Guide |
| --- | --- |
| gufo | [Porting to gufo](PORTING-GUFO.md) |

The guide does not give measured speeds. The formats were designed for AMD
Strix Halo (`gfx1151`, RDNA 3.5), so the kernel notes name its instructions as
examples. Other GPUs have equivalent instructions. Most of the guide is about
Qwen3.8 Flash-Next, where halogen-flash's behavior is known.
[The 27B checkpoint](#the-27b-checkpoint) has its own section.

## Goals

There are two goals. They share most of the work.

1. **Read HGN checkpoints.** Load halogen's published Flash-Next checkpoint
   with its overlay, MTP head file and vision sidecar.
2. **Use HGN layouts for GGUF input.** Repack a GGUF tensor into an HGN layout
   at load when that layout is faster or when the engine has no kernel for the
   GGUF type. Example: IQ4_XS, IQ4_NL and IQ3_S experts all become `q4c` and
   need one kernel family.

An engine that already runs GGUF files should start with an HGN file converted
from a GGUF build that it already runs. That step needs no new kernel (see
[Milestones](#milestones)). The published checkpoint then needs `q4c` and
`q8g64` kernels.

## Integration

### A reader

- Implement the [reader procedure](CONTAINER.md#reader-procedure). Refuse
  overlapping payloads and misaligned quantized payloads at open time, not at
  bind time.
- Describe each tensor with name, store code, variant, rank, shape, file
  offset, size and checksum.
- Expose the mapped file in the same form as a mapped GGUF shard, if the
  engine has one. The engine's upload path can then copy HGN payloads without
  change.
- Keep a store-code enumeration of its own. Do not add HGN codes to the GGML
  type enumeration. The two numbering spaces overlap.
- Put the size rules and CPU decoders beside the engine's other dequantizers.
  A CPU reference needs them.

Test the reader with the [minimal file](TEST-VECTORS.md#a-complete-file) and
malformed copies of it: wrong magic, version 3, wrong file size, an entry past
the end, a duplicate name, a 96-byte name without a NUL, overlapping payloads
and a misaligned `q8s32` payload. The
[published-file facts](TEST-VECTORS.md#published-files) give an optional check
against the real headers. These tests need no GPU and no model files.

### Binding tensors to model roles

An engine binds each tensor to a role in its model code. For HGN:

1. Map each HGN name to the engine's role. For an engine that uses GGUF
   names, the map is the inverse of the [name map](GGUF-IMPORT.md#name-map).
2. Check the shape. HGN lists the outermost dimension first; GGUF lists it
   last.
3. Record the store code and variant with the tensor.
4. Apply the [value conventions](#value-conventions) during upload.

Split the fused tensors, or teach the kernels the fused form:

| HGN tensor | GGUF tensors | Split |
| --- | --- | --- |
| `mlp.experts.gate_up_proj.weight` | `ffn_gate_exps`, `ffn_up_exps` | For expert `e`: rows `1280e … 1280e + 639` are gate, rows `1280e + 640 … 1280e + 1279` are up |
| `self_attn.indexer.index_qk_proj.weight` | `indexer.q_proj`, `indexer.k_proj` | Rows 0–511 are query, rows 512–639 are key |

A row split copies whole rows. For `q4c` it copies the rows of both planes and
gives each part a copy of the codebook. The HGN row order of the fused expert
tensor also suits one gated kernel: an expert's gate row and up row are 640
rows apart in one allocation. Measure both forms before you keep one.

### Value conventions

GGUF-based engines expect the GGUF value conventions. Convert during upload.
All except the `A_log` conversion are exact.

| HGN tensor | GGUF convention | Conversion |
| --- | --- | --- |
| Zero-centred norm gains | Gain with the 1 added | Widen to `f32`, add 1.0 |
| `linear_attn.A_log` | `ssm_a = −exp(A_log)` | `ssm_a = −exp(A_log)` in `f32` |
| `linear_attn.dt_bias` | F32 | Widen `bf16` to `f32` |
| `linear_attn.conv1d.weight` `[10240, 1, 4]` | GGUF `[4, 10240]` | None beyond the type. The memory order is the same |
| DeltaNet value-head axes | Tiled order | See below |
| `mlp.shared_expert_gate.weight` `[1, 2560]` | `[2560]` | None. The bytes are the same |

For the published checkpoint, every conversion in this table gives the value
of the Unsloth GGUF file: the norms and `dt_bias` exactly, `ssm_a` within
`f32` rounding.

**DeltaNet head order.** HGN stores the value heads grouped: value head `h`
uses key head `h / 3` (see [the head order](TENSORS.md#deltanet-head-order)).
An engine written for GGUF reads key head `h mod 16`. There are two ways to
join them:

1. **Reorder at upload.** Tiled head `t` takes grouped head
   `h(t) = 3 × (t mod 16) + t / 16`. Move the units of the
   [head reorder table](GGUF-IMPORT.md#head-reorder) in this direction. Row
   bands and single values are simple copies. For the columns of `out_proj`,
   move whole 32-value groups: 16 code bytes and one scale for `q4c`, 32 codes
   and one scale for `q8s32`, 256 bytes for `bf16`.
2. **Change the kernels.** Read key head `h / 3` for a grouped checkpoint.
   This avoids the column moves, but gives the model two head orders. Every
   kernel and the CPU reference must take the order as a load-time constant.

The first way is simpler to qualify. It keeps one head order in the model code
and needs one tested conversion routine. An engine written for the original
checkpoint already uses the grouped order and needs neither.

### A transform stage

Most engines copy weight bytes unchanged from the file to the device. The
conversions of this guide need a transform between the read and the copy. The
transform works on whole rows:

- reorder or split rows
- rebuild GGUF blocks from HGN planes (see [Storage support](#storage-support))
- widen and offset small tensors

Most transforms read one row of each plane of a tensor. The planes of a large
tensor are far apart in the file, so read a band of rows from each plane
before the transform. Bands of 1 MiB of source worked well in halogen-flash's
importer.

## Storage support

Each store needs a path that works first and a path that is fast. The first
paths below assume an engine with GGUF kernels.

| Store | Where it occurs | First path | Performance path |
| --- | --- | --- | --- |
| `q4c` variant 2, learned codebook | Published checkpoint: all matrices, embedding, experts, MTP | None | The [q4c kernels](#the-q4c-kernels), floating-point form. Required |
| `q4c` variant 2, Q4_0 or IQ4 codebook | GGUF imports | Rebuild Q4_0 or IQ4_NL blocks | The `q4c` kernels, integer form |
| `q4c` variant 2, IQ3_S codebook | GGUF imports | None | The `q4c` kernels, integer form |
| `q8g64` | Quality overlay (`o_proj`), MTP head file, MTP part of both overlays | Decode to F16 at load (1.9 times the memory of these tensors) | A [q8g64 kernel](#the-q8g64-kernel) |
| `q8s32` | GGUF imports | Rebuild Q8_0 blocks | A planar 8-bit kernel |
| `q5a` variant 0 | GGUF imports | Rebuild Q5_1 blocks | A planar affine kernel |
| `q4a` variant 1 | GGUF imports | Rebuild Q4_K blocks | A planar affine kernel |
| `q5a` variant 1 | GGUF imports | Rebuild Q5_K blocks | A planar affine kernel |
| `q6k` | GGUF imports | The engine's Q6_K kernels, unchanged | — |
| `bf16` | Published checkpoint: norms, routers, small vectors, PLE projections, vision | Dense `bf16` kernels | — |
| `fp8g` table | Published checkpoint | A CPU decoder: `fp8(code) × scale` | Raw-row upload (see [The n-gram table](#the-n-gram-table)) |
| `iq4nl` table | GGUF imports | The engine's IQ4_NL table reader, with the HGN offset | Raw-row upload |

A rebuilt GGUF tensor has the same size as its HGN form. The rules are in
[Reverse conversion](GGUF-IMPORT.md#reverse-conversion). Each one was checked
to give the original GGUF bytes.

### The q4c kernels

`q4c` is the format of every matrix of the published checkpoint, and it is
the target of four GGUF types. It needs three kernels: a decode matrix-vector
product, the routed experts and a prefill matrix product.

**The codebook decides the arithmetic.** The published checkpoint has a
learned codebook for each tensor: sixteen `f32` levels between about −1 and 1,
with up to 24 significant bits. The GGUF importer's codebooks are small
integers. Check the codebook at load and pick the form:

| Codebook | Form | Weight arithmetic |
| --- | --- | --- |
| Every level is an integer from −128 to 127 | Integer | Levels as `int8`; exact |
| Any other codebook | Floating point | Levels rounded to `f16` (11 bits) or `bf16` (8 bits) once |

In the floating-point form, round the levels to `f16` rather than `bf16`
where the activations allow it. A learned level such as −0.0171 keeps three
more bits.

**Decode, 1 to 8 tokens.** Halogen-flash's design, with the reasons:

- **A byte table in LDS.** At the start, lane `b` of a 256-lane workgroup
  writes entry `b` of a 256-entry table: the level of `b & 15` in the low half
  and the level of `b >> 4` in the high half. The table is 1 KiB. One LDS read
  for each code byte then gives a packed pair of 16-bit levels, which is the
  operand of `v_dot2_f32_bf16` (halogen-flash) or `v_dot2_f32_f16`. The inner
  loop has no nibble arithmetic.
- **Code rows are contiguous.** A 32-value group is 16 code bytes, one 128-bit
  load. The group scale is a 2-byte load from the scale plane.
- **One scale multiply for each group.** Sum the 16 dot products of a group
  in `f32`, multiply by the group scale once, then add to the total
  (strategy 2 in [Numerics](NUMERICS.md#strategy-2-factor-out-the-scale)).
- **Scales in registers.** When `K / 32` is at most 8 times the lanes of a
  row, each lane needs at most 8 scales. Load them once before the loop.
- **Token count as a template argument.** Read each code byte once for all
  1 to 8 tokens of the step.
- **Row tiles from the shape.** Halogen-flash uses these tiles. Retune them
  with your own measurements:

  | Shape | Lanes for each output row | Rows for each 256-lane workgroup |
  | --- | ---: | ---: |
  | `N ≤ 8` | 256 | 1 |
  | `K ≥ 5120` and `N < 5128` | 64 | 4 |
  | `K > 2591` | 32 | 8 |
  | Otherwise | 16 | 16 |

  When a row spans more than one wave, each wave writes its partial sum to
  LDS after the table, and one lane for each row adds them.
- **Fused epilogues.** SiLU, sigmoid, `2 × sigmoid` for the hyper-connection
  inject gate, and the query/gate split of `q_proj` (rows 0–255 of each block
  of 512 go to the query, rows 256–511 to the gate).
- **Grouped launches.** Up to four matrices that read the same input run in
  one launch. A small descriptor array holds each matrix's pointers and its
  first workgroup index.

**The integer form** suits engines that quantize activations to 8 bits for
their decode kernels, as llama.cpp-style engines do with Q8_1 blocks:

1. Keep the 16 levels as `int8` in registers. Expand each code byte with a
   byte permute, as IQ4_NL kernels do.
2. Use the 8-bit activations and a signed-by-unsigned 4-way dot product.
3. Multiply the `int32` group sum by the group scale and the activation scale
   in `f32`.

A 32-value `q4c` group is one Q8_1 activation block, so the scales line up.
For the IQ4 codebook this is the IQ4_NL arithmetic of llama.cpp.

**Routed experts.** Keep the engine's existing MoE structure and change only
the weight decode:

- Expert `e`, gate row `u` is global row `1280e + u`; up row `u` is
  `1280e + 640 + u`. Down row `d` is `2560e + d`. Code rows are `K / 2` bytes
  apart. Scale rows are `pad(K / 16, 16)` bytes apart.
- One codebook serves all experts of a tensor, so the byte table is built once
  for each launch.
- In halogen-flash the decode path is one kernel for gate, up and SwiGLU, and
  one for down. The down kernel also applies the routing weights, adds the
  shared expert with its sigmoid gate and writes the layer output. The kernels
  read the selected expert ids on the device, so a decode step does not wait
  for the host.

**Prefill.** Decode inside the matrix kernel. Halogen-flash instead decodes
the whole matrix into a 256 MiB `bf16` buffer and then calls a vendor GEMM,
which writes and reads 2 extra bytes for each weight. Two better forms:

- A floating-point WMMA kernel that decodes a `q4c` tile into `f16` while it
  stages the tile in LDS, with the 16 levels in registers or LDS.
  `f16(scale × level)` rounds to 11 significant bits, which is finer than
  halogen-flash's `bf16`.
- An integer WMMA kernel with scales for each block of 32 along `K`. A `q4c`
  group is such a block. With an integer codebook, expand the codes to `int8`
  and apply the `f16` group scale for each block.

### The q8g64 kernel

The quality overlay stores the 12 `self_attn.o_proj` matrices as `q8g64`, and
the MTP head file and both overlays store the 18 dense MTP projections that
way. The overlay is where most of the checkpoint's quality lives, according to
its publisher, so a port should support it.

`q8g64` is affine: `value = code × scale + bias` for groups of 64 unsigned
codes. Factor the bias out of the dot product:

```text
Σ value × x = scale × Σ (code × x) + bias × Σ x
```

Q8_1 activation blocks already carry `Σ x` for each block of 32, so a group
of 64 takes two blocks. Use an unsigned-by-signed 8-bit dot product for
`Σ (code × x)`. A floating-point form works too: convert the codes, sum
`code × x` in `f32`, and keep `Σ x` for each group.

### Affine formats

The affine stores hold exactly the numbers of Q5_1, Q4_K and Q5_K. Rebuild the
GGUF blocks, and an engine's GGUF expert kernels run them with no other
change. This is the first path.

Halogen-flash's planar kernel for the affine experts is a possible
performance path:

- The LDS byte table holds the nibble values themselves:
  `{bf16(b & 15), bf16(b >> 4)}`.
- Each activation pair takes three `v_dot2_f32_bf16`: gate codes, up codes and
  a pair of `bf16` ones, which gives `Σ x`. The group result is
  `scale × Σ (c × x) + zero × Σ x`.
- For `q5a`, the high bit is a separate term: each set bit adds 16.0 to its
  value. A signed one-bit field extract gives 0 or −1, which masks the `bf16`
  constant 16.0. One more `v_dot2_f32_bf16` adds the term.
- All code loads are 128-bit and contiguous in a row. The scale records are
  in small separate planes.

Treat the planar layout as an experiment and keep it only if it measures
faster than the block layout.

Halogen-flash has no dense affine kernel. It expands dense `q4a` and `q5a`
matrices to `bf16` at load. A port should rebuild the GGUF blocks instead.

## Memory residency

Strix Halo shares one LPDDR5X memory between the CPU and the GPU. A
host-registered buffer usually has a device pointer equal to its host
pointer. On a discrete GPU the payloads are copied to device memory
unchanged.

An HGN payload is kernel-ready and 64-byte aligned, so on a unified-memory
system a tensor that needs no transform can stay in the file mapping.
Halogen-flash maps HGN files and registers the ranges that the model uses. A
tensor that needs a transform must be copied. Halogen-flash's rules for
registered memory apply to any of these choices:

- Register only the ranges that the model uses. Sort the page-aligned ranges
  and join the ranges that touch.
- Before registering, compare the total with the available memory. Pinned
  pages cannot be swapped out. A process that pins too much memory stalls the
  whole machine instead of failing. Halogen-flash keeps a 16 GiB floor.
- Register with the read-only flag first. Some drivers refuse it; then
  register without it.
- Mark an anonymous weight region with `MADV_HUGEPAGE`. 60 GB of 4 KiB pages
  is about 15 million page-table entries, and every decode step reads most of
  the weights.
- When you convert a GGUF file, drop each source range from the page cache
  after its last use (`POSIX_FADV_DONTNEED`). Convert the largest tensors
  first, on all threads. Take 1 MiB of source for each unit of work.

The published checkpoint is 115.55 GiB, but 47.7 GiB of that is the n-gram
table, which stays out of GPU memory. Its publisher gives about 68 GiB as the
resident need.

Start with the engine's existing upload path and a transform stage. Measure
load time and resident memory. Consider the file mapping later, for the
tensors that need no transform.

## The n-gram table

The published checkpoint stores the table as `fp8g`: 160-byte rows and one
`f32` scale at `pad(160 × rows, 64)` from the payload start. An `iq4nl` table
from a GGUF import holds the same bytes as the GGUF IQ4_NL table, 90 bytes for
each row, so an IQ4_NL table reader can read it with the HGN offset.

Keep the table out of GPU memory and move only the selected rows. Halogen-flash
uploads the raw quantized rows and decodes them on the GPU. That moves far
fewer bytes than decoding on the CPU and uploading `f32` rows:

| Rows moved to the device | Bytes for each row |
| --- | ---: |
| `f32`, decoded on the CPU | 640 |
| Raw `fp8g` | 160 |
| Raw `iq4nl` | 90 |

A 2048-token prompt selects 32,768 rows: 21 MB as `f32` against 5.2 MB or
2.9 MB raw. The CPU decode time is saved as well.

## MTP head

The published checkpoint holds the MTP head. The head file holds the same 31
tensors with `q8g64` dense projections, for use with a GGUF trunk. The GGUF
MTP file of the same model names them as follows:

| HGN name | GGUF tensor |
| --- | --- |
| `mtp.fc_embedding.weight` | First 2560 columns of `blk.48.nextn.eh_proj.weight` |
| `mtp.fc_hidden.weight` | Last 2560 columns of `blk.48.nextn.eh_proj.weight` |
| `mtp.pre_fc_norm_embedding.weight` | `blk.48.nextn.enorm.weight` |
| `mtp.pre_fc_norm_hidden.weight` | `blk.48.nextn.hnorm.weight` |
| `mtp.hyper_connection_mixer.hc_norm.weight` | `blk.48.nextn.hc_head_norm.weight` |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | `blk.48.nextn.hc_head_down.weight` |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | `blk.48.nextn.hc_head_up.weight` |
| `mtp.layers.0.SUFFIX` | `blk.48.` with the GGUF suffix of the [layer name map](GGUF-IMPORT.md#layer-names) |

GGUF packs the two input projections into one tensor: `fc_embedding`, then
`fc_hidden`, across each row. Decoded rows of the Unsloth MTP GGUF and of the
head file agree in that order (correlation 1.0). All MTP norm gains are
zero-centred: each is exactly the value of the Unsloth MTP GGUF minus 1. Add 1
at upload, as for the trunk. The head's routed experts are `q4c`, so the MTP
path needs the `q4c` expert kernels.

## Numerical strategy

The format fixes the value of every weight but not the rounding of a matrix
product (see [Numerics](NUMERICS.md#how-halogen-flash-uses-the-values)). Choose
a strategy, state it and qualify it:

- Rebuilt GGUF blocks give the arithmetic of the engine's GGUF path exactly.
  Take an HGN file converted with exact conversions only (Q8_0, Q5_1, Q4_K,
  Q5_K, Q4_0, IQ4_NL, Q6_K, BF16). It must give the same logits as its GGUF
  source, bit for bit, when `ssm_a` takes the GGUF values. This is the
  strongest test of a loader.
- `q4c` with an integer codebook and `q8g64` fit an integer-activation path.
- `q4c` with a learned codebook, which is every matrix of the published
  checkpoint, needs floating-point levels. Use an `f16` byte table with
  `v_dot2_f32_f16` and `f16` activations, or a `bf16` table with
  `v_dot2_f32_bf16`. Qualify the result with the model's quality gates.

## The 27B checkpoint

The published 27B HGN file (see [Qwen3.8 27B tensors](TENSORS-27B.md)) uses
three formats that the Flash-Next work does not cover. No analyzed engine
reads them; their rules come from the file and are checked in
[Numerics](NUMERICS.md#evidence-for-the-27b-formats).

| Format | Use in the file | Kernel work |
| --- | --- | --- |
| `fp8r` | Token-generation copies of attention, DeltaNet and 8 feed-forward layers; `lm_head` | A decode matrix-vector product. `gfx1151` has no 8-bit float dot product, so expand codes to `f16` with a 256-entry table (one `f16` for each code) and use `v_dot2_f32_f16`. Apply the `bf16` row scale once at the end |
| `q4c` variants 0 and 1 | Token-generation copies of 56 feed-forward layers (NVFP4); MTP and drafter (NF4) | The `q4c` kernels with row-interleaved scales. Variant 0 decodes one fp8 scale for each 16 values; variant 1 one `f16` scale for each 32 |
| `i4l` | Prompt-processing copies of every trunk matrix | Rotate each 256-value block of the activations with the orthonormal Hadamard transform (a fast Walsh–Hadamard transform: 8 butterfly stages), quantize them to `int8`, and run an integer WMMA product against the 4-bit codes. Apply the `f16` group scale and the activation scale for each block of 256 |

The two copies of a 27B matrix give a clear phase split: token generation
reads 8-bit or NVFP4 weights, and prompt processing reads rotated 4-bit
weights with integer matrix instructions. A port can take the same split, or
use one copy for both phases at the cost of speed in one of them.

The 27B MTP head maps to the GGUF `nextn.*` tensors as in the
[27B document](TENSORS-27B.md#mtp-head); `mtp.fc.weight` is byte-identical to
`nextn.eh_proj.weight`. The DFlash2 drafter is halogen's own training, and
its structure is not specified here.

## Qualification

| Check | Needs | Content |
| --- | --- | --- |
| Container | CPU | The minimal file and malformed copies |
| Storage decode | CPU | The [decode vectors](TEST-VECTORS.md), size rules, random decodes against an FP64 reference |
| Transform | CPU | Rebuilt GGUF bytes, the head reorder and its inverse, norm offsets, fused splits |
| `q4c` kernels | GPU | FP64 oracle, both codebook forms, every tile, 1 to 8 tokens, every epilogue |
| `q8g64` kernel | GPU | FP64 oracle, 1 to 8 tokens |
| Model, exact | GPU and a GGUF file | Bit-identical logits for an exact conversion of a GGUF file that the engine runs |
| Model, published | GPU and the published checkpoint | Measured quality, with and without the overlay |

For the published checkpoint, measure quality; do not assume it. Its
quantization is not any GGUF quantization. Compare with a BF16 reference by
mean KL divergence, total variation and top-1 agreement. Its publisher reports
a 5–9% perplexity loss without the overlay.

## Milestones

1. **Reader.** The HGN reader, its fixtures and the storage decoders. No GPU.
2. **Exact HGN files.** Load an HGN file made from a GGUF build that the
   engine already runs (see [the worked example](GGUF-IMPORT.md#worked-example)).
   Rebuild GGUF blocks, reorder heads and restore norm offsets at upload.
   Gate: the same logits as the GGUF path, bit for bit.
3. **`q4c`.** The decode, routed-expert and prefill kernels, in both codebook
   forms. This adds IQ4_XS, IQ4_NL and IQ3_S GGUF input through an in-memory
   repack.
4. **Published checkpoint.** The `fp8g` table, `q8g64`, the overlay rules and
   the MTP head. Gate: measured quality.
5. **Experiments.** Planar affine kernels, zero-copy residency and raw-row
   n-gram upload, each with its own evidence.
6. **27B.** `fp8r`, `q4c` variants 0 and 1, and the rotated `i4l` prefill
   path, if the 27B file is a goal.

## Pitfalls

- **Nibble order.** GGUF puts value `j` and value `j + 16` in one byte. HGN
  puts value `2j` and value `2j + 1` in one byte. Test with values that are
  different at every position; a symmetric test hides the error.
- **Head order.** A wrong DeltaNet head order still gives fluent text. Detect
  it with a per-layer comparison against the CPU reference.
- **Norm offsets.** A missing 1 breaks the model at once. A 1 added twice
  gives slowly degraded output.
- **`A_log` sign.** `A_log` is `log(−A)`. `ssm_a` is `A`, which is negative.
- **Learned codebooks.** The published codebooks are not integers. An integer
  kernel that rounds them gives wrong weights without an error. Check the
  codebook at load.
- **Expert variants.** Gate and up accept only the k-scale affine variant;
  down accepts only `q5a` in the pair form. Check at load.
- **Shared codebook.** A fused expert tensor has one codebook for all 512
  experts. Do not look for one codebook for each expert.
- **Alignment.** 128-bit loads need 16-byte-aligned payloads. Check at open.
- **Scale overflow.** A combined IQ4_XS or IQ3_S scale can overflow `f16`.
  Refuse a non-finite scale in the importer.
- **Rotated weights.** An `i4l` tensor without the activation rotation gives
  noise, not a small error.
