# Qwen3.8 Flash-Next tensors

This document lists every tensor of a Qwen3.8 Flash-Next checkpoint in HGN
naming. For each tensor it gives the shape, the store codes that halogen-flash
accepts, the stores of the published checkpoints and the convention of the
stored values.

## Model constants

HGN files carry no hyperparameters. A reader has these constants built in and
checks each tensor shape against them.

| Constant | Value |
| --- | --- |
| Trunk layers | 48, numbered 0 to 47 |
| Hidden size | 2560 |
| Residual streams | 4. The hyper-connection width is `4 × 2560 = 10240` |
| Hyper-connection bottleneck | 320 |
| Vocabulary | 248320 tokens |
| Layer type | Layer `L` is full attention when `L mod 4 = 3` (12 layers), else Gated DeltaNet (36 layers) |
| Full attention | 24 query heads of 256, each with a 256-wide output gate; 2 key-value heads of 256 |
| Sparse-attention indexer | 4 query heads of 128 and one key of 128 |
| Gated DeltaNet | 16 key heads of 128, 48 value heads of 128, convolution width 4 |
| Mixture of experts | 512 routed experts of width 640, 10 selected for each token, one shared expert of width 640 |
| Per-layer embedding (PLE) | On layer 1 only. 16 n-gram heads of 160 values |
| MTP head | One full-attention block with its own experts |

The GGUF form of these constants is in
[GGUF import](GGUF-IMPORT.md#preconditions).

## Name grammar

| Pattern | Meaning |
| --- | --- |
| `embed_tokens.weight`, `lm_head.weight` | Token embedding and output head |
| `hyper_connection_mixer.*` | The final hyper-connection mixer before the output head |
| `layers.L.SUBLAYER.*` | A tensor of trunk layer `L`, in decimal without leading zeros |
| `mtp.*` | A tensor of the MTP head |
| `visual.*` | A tensor of the vision sidecar |
| `BASE.suh`, `BASE.svh` | The [rotation vectors](#rotation-vectors) of the matrix `BASE.weight` |
| `layers.L.mlp.experts.su` | The shared input vector of the routed experts of layer `L` |

`SUBLAYER` is one of six names. The canonical order of the sublayers is:

| Index | Sublayer |
| ---: | --- |
| 0 | `ple` |
| 1 | `attn_hyper_connection` |
| 2 | `linear_attn` |
| 3 | `self_attn` |
| 4 | `mlp_hyper_connection` |
| 5 | `mlp` |

## Reader roles

Halogen-flash uses each tensor in one of four roles. The role decides which
store codes it accepts.

| Role | Treatment | Accepted stores |
| --- | --- | --- |
| Matrix | The kernels read the payload in place | `bf16`, `q4c` variant 2, `q8g64`, `q8s32`, `q6k`, `q6g64`, `ht`. Also `q4a` and `q5a`, which halogen-flash expands to `bf16` at load |
| Dense | Expanded to a device `bf16` copy at load | `bf16`, `q4c` variant 2, `q8g64`, `q8s32`, `q6k`, `q6g64`, `q4a`, `q5a` |
| Expert | Read in place by the routed-expert kernels | Given for each tensor below |
| Fixed | Exactly one layout | Given for each tensor below |

Halogen-flash has no kernel for dense `q4a` and `q5a` matrices. It expands
them to `bf16` at load, which costs about three times their memory. A port can
do better (see [the porting guide](PORTING.md#affine-formats)).

Rules for `q6g64`, `ht` and `i4r`:

- A `q6g64` matrix must start at a 16-byte boundary. Write variant 64.
- An `ht` matrix must be 2-D, and both dimensions must be multiples of 128.
  It needs its two [rotation vectors](#rotation-vectors). Halogen-flash reads
  an `ht` matrix only in place. It refuses an `ht` matrix that it cannot keep
  in registered memory.
- `ht` and `i4r` are not Dense stores. So `embed_tokens.weight`, the norm
  gains and the router cannot use them.

These Matrix tensors cannot be `ht`, because a dimension is not a multiple of
128: `input_mix_weight_down`, `input_mix_weight_up` and `block_inject_weight`
in every hyper-connection and mixer, `in_proj_a`, `in_proj_b` and
`shared_expert_gate`.

Halogen-flash computes `lm_head.weight` and `mlp.shared_expert_gate.weight`
with `f32` outputs. It has no `f32` path for `q8g64` and `q6g64`, so it
refuses these two stores for these two tensors.  `ht` is accepted for `lm_head.weight`.

## Published checkpoints

The Flash-Next repository publishes three checkpoints. They hold the same
tensors, except for the n-gram table and the rotation vectors.

| Checkpoint | Tensors | n-gram table | Rotation vectors | Overlay files |
| --- | ---: | --- | ---: | --- |
| `qwen38-flash-next-w4b.hgn` | 1,198 | In the file | 0 | Two |
| `qwen38-flash-next-v2.hgn` | 2,015 | In the [n-gram table file](CONTAINER.md#n-gram-table-file) | 818 | None |
| `qwen38-flash-next-ht43.hgn` | 2,015 | In the n-gram table file | 818 | None |

Its publisher names v2 as the default checkpoint, and ht43 as a smaller
checkpoint to opt in to. w4b still loads.

### w4b

`qwen38-flash-next-w4b.hgn` holds 1,198 tensors: the trunk, the n-gram table
and the complete MTP head. It uses four stores:

| Store | Tensors | Roles |
| --- | ---: | --- |
| `q4c` variant 2 | 841 | Every matrix, including `embed_tokens`, `lm_head`, the experts and the MTP head |
| `bf16` | 353 | Norm gains, routers, `A_log`, `dt_bias`, convolutions, the two PLE projections |
| `fp8g` | 1 | The n-gram table |
| `i64` | 3 | The PLE index vectors |

Its codebooks are learned for each tensor (see
[Storage formats](STORAGE.md#codebooks)). The [overlay
files](CONTAINER.md#overlay-file) replace 723 of its matrices and the 18
dense projections of the MTP head.

### v2

`qwen38-flash-next-v2.hgn` holds 2,015 tensors: the trunk, its rotation
vectors and the complete MTP head. It does not hold the n-gram table. It uses
eight stores:

| Store | Tensors | Use |
| --- | ---: | --- |
| `ht` variant `0x1208` (4 bits) | 313 | `lm_head`; `q_proj`, `k_proj`, `v_proj`, `o_proj` and `index_qk_proj`; `in_proj_qkv`, `in_proj_z` and `out_proj`; the three shared-expert projections |
| `q6g64` | 290 | `input_mix_weight_down`, `input_mix_weight_up` and `block_inject_weight` of every hyper-connection, and the two projections of the final mixer |
| `i4r` | 96 | The routed experts of every layer |
| `q4c` variant 2 | 124 | `embed_tokens`, `in_proj_a`, `in_proj_b`, `shared_expert_gate`, and the three `q4c` tensors of the MTP head |
| `q8g64` | 18 | The dense projections of the MTP head |
| `bf16` | 353 | As in w4b |
| `f16` | 818 | Rotation vectors |
| `i64` | 3 | The PLE index vectors |

Every `bf16`, `i64` and `q4c` entry of v2 is equal to the w4b entry of the
same name. The store, shape, size, checksum and variant are equal. Its 31 `mtp.`
tensors are equal to those of the [MTP head file](CONTAINER.md#mtp-head-file).

### ht43

`qwen38-flash-next-ht43.hgn` is v2 with two differences:

- The routed experts are `ht`: variant `0x1208` (4 bits) in the even layers
  and `0x1206` (3 bits) in the odd layers. They have rotation vectors for each
  expert (see [ht experts](STORAGE.md#ht-experts)).
- The hyper-connection and mixer projections are `q8g64`, not `q6g64`.

| Store | Tensors |
| --- | ---: |
| `ht` variant `0x1208` (4 bits) | 361 |
| `ht` variant `0x1206` (3 bits) | 48 |
| `q8g64` | 308 |
| `q4c` variant 2 | 124 |
| `bf16` | 353 |
| `f16` | 818 |
| `i64` | 3 |

## Global tensors

| Name | Shape | Role | Store in w4b | Notes |
| --- | --- | --- | --- | --- |
| `embed_tokens.weight` | `[248320, 2560]` | Dense | `q4c` | |
| `lm_head.weight` | `[248320, 2560]` | Matrix | `q4c` | Not `q8g64` or `q6g64`. `ht` in v2 and ht43 |
| `hyper_connection_mixer.hc_norm.weight` | `[10240]` | Dense | `bf16` | [Zero-centred](#zero-centred-norm-gains) |
| `hyper_connection_mixer.input_mix_weight_down.weight` | `[320, 10240]` | Matrix | `q4c` | |
| `hyper_connection_mixer.input_mix_weight_up.weight` | `[10240, 320]` | Matrix | `q4c` | |

## Tensors of every layer

Prefix `layers.L.` for `L` from 0 to 47. Unless the table says otherwise, the
store of a Matrix in w4b is `q4c` and of a Dense vector is `bf16`. The
[published checkpoints](#published-checkpoints) section gives the stores of v2
and ht43.

### Hyper-connections

Two sets, with `S` equal to `attn` and to `mlp`. Prefix
`layers.L.S_hyper_connection.`.

| Name | Shape | Role | Notes |
| --- | --- | --- | --- |
| `hc_norm.weight` | `[10240]` | Dense | Zero-centred. One 2560-wide gain for each stream |
| `input_mix_weight_down.weight` | `[320, 10240]` | Matrix | |
| `input_mix_weight_up.weight` | `[10240, 320]` | Matrix | |
| `block_inject_weight.weight` | `[4, 10240]` | Matrix | Halogen-flash can also make a dense copy for a fused kernel |

### Mixture of experts

Prefix `layers.L.mlp.`.

| Name | Shape | Role | Notes |
| --- | --- | --- | --- |
| `gate.weight` | `[512, 2560]` | Fixed: `bf16` | The router. Halogen-flash makes an exact `f32` copy, so it needs `bf16` |
| `shared_expert.gate_proj.weight` | `[640, 2560]` | Matrix | |
| `shared_expert.up_proj.weight` | `[640, 2560]` | Matrix | |
| `shared_expert.down_proj.weight` | `[2560, 640]` | Matrix | |
| `shared_expert_gate.weight` | `[1, 2560]` | Matrix | A one-row matrix, not a vector. Not `q8g64` or `q6g64` |
| `experts.gate_up_proj.weight` | `[512, 1280, 2560]` | Expert | See [Fused experts](#fused-experts) |
| `experts.down_proj.weight` | `[512, 2560, 640]` | Expert | See [Fused experts](#fused-experts) |

A router with another store works only when the operator turns off the exact
`f32` router copy. Use `bf16`.

### Fused experts

`experts.gate_up_proj.weight` holds the gate and up projections of all 512
experts in one tensor of `512 × 1280` rows:

- Rows `1280e` to `1280e + 639` are the gate projection of expert `e`.
- Rows `1280e + 640` to `1280e + 1279` are the up projection of expert `e`.

`experts.down_proj.weight` holds `512 × 2560` rows. Rows `2560e` to
`2560e + 2559` are the down projection of expert `e`.

A kernel finds the data of global row `R` with the ordinary rules of the store
(see [Storage formats](STORAGE.md)). For most stores, the codes and the scales
of an expert are not in one contiguous block: each plane holds the rows of
all experts. `ht` is different: the data of each expert is one contiguous
block, and its gate and up rows share tile bands (see
[ht experts](STORAGE.md#ht-experts)).

| Tensor | Accepted stores |
| --- | --- |
| `experts.gate_up_proj.weight` | `q4c` variant 2; `q4a` variant 1; `q5a` variant 1; `ht`; `i4r` |
| `experts.down_proj.weight` | `q4c` variant 2; `q8s32` with a payload size of exactly 901,775,360 bytes; `q5a` variant 0; `ht`; `i4r` |

These rules apply to `ht` and `i4r` experts:

- Both projections must use the same store. An `ht` pair can use a
  different code width in each projection.
- Both projections must stay in registered memory. Halogen-flash has no copy
  path for them.
- An `i4r` payload must start at a 64-byte boundary.
- The layer must hold the four [rotation vectors](#rotation-vectors) of its
  experts.

A reader must refuse all other combinations. The pair variant of the affine
formats has no gate-and-up kernel. The k-scale variant cannot hold the down
projection, because `K = 640` is not a multiple of 256.

Halogen-flash sizes its expert buffers from layer 0. Its behavior with
different expert stores in different layers is **not established**. A writer
should use one expert store in all 48 layers. The `ht` code width can differ
between layers, as in ht43.

### Gated DeltaNet

Layers with `L mod 4 ≠ 3`. Prefix `layers.L.linear_attn.`.

| Name | Shape | Role | Notes |
| --- | --- | --- | --- |
| `in_proj_qkv.weight` | `[10240, 2560]` | Matrix | Rows 0–2047 query, 2048–4095 key, 4096–10239 value. [Grouped order](#deltanet-head-order) |
| `in_proj_z.weight` | `[6144, 2560]` | Matrix | Output gate, 48 bands of 128 rows. Grouped order |
| `in_proj_a.weight` | `[48, 2560]` | Matrix | One row for each value head. Grouped order |
| `in_proj_b.weight` | `[48, 2560]` | Matrix | One row for each value head. Grouped order |
| `out_proj.weight` | `[2560, 6144]` | Matrix | 48 blocks of 128 columns. Grouped order |
| `conv1d.weight` | `[10240, 1, 4]` | Dense | Depthwise convolution, one 4-tap filter for each channel. Grouped order |
| `norm.weight` | `[128]` | Dense | A plain gain, not zero-centred |
| `A_log` | `[48]` | Fixed: `f32` or `bf16` | `log(−A)`, one value for each value head. Grouped order. Published as `bf16` |
| `dt_bias` | `[48]` | Fixed: `f32` or `bf16` | One value for each value head. Grouped order. Published as `bf16` |

Halogen-flash converts `A_log` and `dt_bias` to `f32`. It reads every store
that is not `f32` as `bf16`, so a writer must use `f32` or `bf16` for these
two.

### Full attention

Layers with `L mod 4 = 3`. Prefix `layers.L.self_attn.`.

| Name | Shape | Role | Notes |
| --- | --- | --- | --- |
| `q_proj.weight` | `[12288, 2560]` | Matrix | 24 blocks of 512 rows: 256 query rows, then 256 gate rows of the same head |
| `k_proj.weight` | `[512, 2560]` | Matrix | 2 heads of 256 |
| `v_proj.weight` | `[512, 2560]` | Matrix | 2 heads of 256 |
| `o_proj.weight` | `[2560, 6144]` | Matrix | `q8g64` in the quality overlay |
| `q_norm.weight` | `[256]` | Dense | Zero-centred |
| `k_norm.weight` | `[256]` | Dense | Zero-centred |
| `indexer.index_qk_proj.weight` | `[640, 2560]` | Matrix | Rows 0–511 indexer query (4 heads of 128), rows 512–639 indexer key |
| `indexer.q_layernorm.weight` | `[128]` | Dense | Zero-centred |
| `indexer.k_layernorm.weight` | `[128]` | Dense | Zero-centred |

Halogen-flash loads the three indexer tensors only when the configured context
is longer than the indexer budget. A complete file must still hold them.

## Per-layer embedding

The PLE tensors belong to layer 1 only. Prefix `layers.1.ple.`.

| Name | Shape | Role | Notes |
| --- | --- | --- | --- |
| `key_proj.weight` | `[10240, 2560]` | Matrix | Published as `bf16` |
| `value_proj.weight` | `[2560, 2560]` | Matrix | Published as `bf16` |
| `conv1d.weight` | `[10240, 1, 4]` | Fixed: `bf16` | The reader copies the 81,920 payload bytes without a store check |
| `norm_key.weight` | `[10240]` | Dense | Zero-centred |
| `norm_query.weight` | `[10240]` | Dense | Zero-centred |
| `norm_conv.weight` | `[10240]` | Dense | Zero-centred |
| `ple_embedding.layer_multipliers` | `[3]` | Fixed: `i64` | |
| `ple_embedding.ngram_heads_offsets` | `[16]` | Fixed: `i64` | |
| `ple_embedding.ngram_heads_vocab_sizes` | `[16]` | Fixed: `i64` | |
| `ngram_embedding.weight` | `[128, R, 160]` | Fixed: `fp8g`, `iq4nl`, `q8_0`, `q5_0` or `q5_1` | The n-gram table. See below |

The three `i64` vectors are the parameters of the n-gram hash. The hash itself
is outside this specification. The published values are in
[Test vectors](TEST-VECTORS.md#published-files).

### The n-gram table

`layers.1.ple.ngram_embedding.weight` has `128 × R` rows of 160 values. The
published table has `R = 2500012`, which is 320,001,536 rows and 51.2 × 10⁹
values: 51.2 GB as `fp8g`, 28.8 GB as `iq4nl`, 54.4 GB as `q8_0`, 35.2 GB
as `q5_0` and 38.4 GB as `q5_1`. For each token the model
selects 16 rows, one for each n-gram head, and joins them into one 2560-wide
vector.

The table is the only tensor that halogen-flash does not make GPU-visible:

- It stays in the operating system page cache, mapped read-only.
- For each forward pass, the host copies the selected rows, still quantized,
  into a pinned staging buffer. With many rows it uses several threads.
- The device dequantizes the staged rows to `bf16`.

For `iq4nl`, `q8_0`, `q5_0` and `q5_1`, halogen-flash checks only that
`shape[2]` is 160 and that the payload size is correct.

A complete checkpoint holds the table. A checkpoint without the table can
serve only with an [n-gram table file](CONTAINER.md#n-gram-table-file). The
v2 and ht43 checkpoints hold no table. A converter that makes a complete HGN
file must include the table.

## Rotation vectors

An `ht` or `i4r` matrix uses rotation vectors (see
[Storage formats](STORAGE.md#rotation-vectors)). Each vector is a Fixed
tensor of store `f16`. Its element count must be exact. Its rank and shape
are free.

| Matrix | Vectors, with their element counts |
| --- | --- |
| A 2-D `ht` matrix `BASE.weight` of shape `[N, K]` | `BASE.suh`: `K`. `BASE.svh`: `N` |
| `ht` routed experts of layer `L` | `experts.su`: 2,560. `experts.gate_up_proj.svh`: 655,360. `experts.down_proj.suh`: 327,680. `experts.down_proj.svh`: 1,310,720 |
| `i4r` routed experts of layer `L` | `experts.su`: 2,560. `experts.gate_up_proj.svh`: 1,280. `experts.down_proj.suh`: 640. `experts.down_proj.svh`: 2,560 |

The expert vectors have the prefix `layers.L.mlp.`. In the published files,
each vector has the shape `[count]`. The `ht` expert vectors are the
exception: they have the shape `[512, count / 512]`, except `experts.su`.

`mlp.shared_expert.gate_proj.suh` and `mlp.shared_expert.up_proj.suh` can be
absent. Halogen-flash then uses `mlp.experts.su` of the same layer. The
published files hold both vectors.

Halogen-flash looks for each vector in the overlay file first, then in the
checkpoint.

## Value conventions

### Zero-centred norm gains

These RMSNorm gains are stored as `gain − 1`. The effective gain is
`1 + stored value`:

- every `hc_norm.weight`, in the mixers and in both hyper-connections of
  every layer, trunk and MTP head
- `self_attn.q_norm.weight` and `self_attn.k_norm.weight`
- `self_attn.indexer.q_layernorm.weight` and
  `self_attn.indexer.k_layernorm.weight`
- `ple.norm_key.weight`, `ple.norm_query.weight` and `ple.norm_conv.weight`
- `mtp.pre_fc_norm_embedding.weight` and `mtp.pre_fc_norm_hidden.weight`

`linear_attn.norm.weight` is a plain gain. GGUF files store all of these gains
with the 1 already added (see [GGUF import](GGUF-IMPORT.md#norm-offsets)). For
the published MTP head, every norm gain is exactly the GGUF value of the
public Unsloth MTP file minus 1, which confirms the list.

### DeltaNet head order

The 48 value heads form 16 groups of 3. Each group shares one key head. HGN
stores the value heads **grouped**: value head `h` uses key head `h / 3`. This
is the order of the original training checkpoint.

GGUF files store the value heads **tiled**: value head `t` uses key head
`t mod 16`. HGN value head `h` is GGUF value head

```text
t(h) = h / 3 + 16 × (h mod 3)
```

In the published checkpoint, `dt_bias` of layers 0 and 1 is exactly the GGUF
`ssm_dt.bias` in this order, and `−exp(A_log)` is the GGUF `ssm_a` within
`f32` rounding.

The permutation applies to every value-head axis:

| Tensor | Axis | Unit that moves |
| --- | --- | --- |
| `in_proj_qkv.weight` | Rows 4096 to 10239 | Bands of 128 rows |
| `in_proj_z.weight` | All rows | Bands of 128 rows |
| `in_proj_a.weight`, `in_proj_b.weight` | All rows | Single rows |
| `out_proj.weight` | All columns | Blocks of 128 columns |
| `conv1d.weight` | Channels 4096 to 10239 | Bands of 128 channels |
| `A_log`, `dt_bias` | All values | Single values |

The query and key rows (0 to 4095) of `in_proj_qkv` do not move.
`linear_attn.norm.weight` has no head axis.

### A_log

HGN stores `A_log = log(−A)`, one value for each value head. The decay rate
of head `h` is `A = −exp(A_log)`. GGUF stores `A` itself.

### Shapes that differ from GGUF

- A depthwise convolution is `[channels, 1, width]`. GGUF has no middle
  dimension.
- `shared_expert_gate.weight` is `[1, 2560]`. GGUF stores it as a vector.

## MTP head

The MTP head is one full-attention block with its own mixers. It uses the
trunk's `embed_tokens` and `lm_head`. All three published checkpoints hold
it, and so does the separate MTP head file. The v2 and ht43 checkpoints hold
the tensors of the head file.

| Name | Shape | Role | Store: w4b / head file, v2 and ht43 |
| --- | --- | --- | --- |
| `mtp.fc_embedding.weight` | `[2560, 2560]` | Matrix | `q4c` / `q8g64` |
| `mtp.fc_hidden.weight` | `[2560, 2560]` | Matrix | `q4c` / `q8g64` |
| `mtp.pre_fc_norm_embedding.weight` | `[2560]` | Dense | `bf16` |
| `mtp.pre_fc_norm_hidden.weight` | `[10240]` | Dense | `bf16` |
| `mtp.hyper_connection_mixer.hc_norm.weight` | `[10240]` | Dense | `bf16` |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | `[320, 10240]` | Matrix | `q4c` / `q8g64` |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | `[10240, 320]` | Matrix | `q4c` / `q8g64` |
| `mtp.layers.0.self_attn.*` | as in the trunk | as in the trunk | Matrices `q4c` / `q8g64` |
| `mtp.layers.0.attn_hyper_connection.*` | as in the trunk | as in the trunk | Matrices `q4c` / `q8g64` |
| `mtp.layers.0.mlp_hyper_connection.*` | as in the trunk | as in the trunk | Matrices `q4c` / `q8g64` |
| `mtp.layers.0.mlp.*` | as in the trunk | as in the trunk, but see below | Experts and `shared_expert_gate` `q4c`; shared expert `q4c` / `q8g64` |

The MTP block uses the full-attention tensor set of the trunk, including the
indexer. Its routed experts are stricter than the trunk's:
`experts.gate_up_proj.weight` must be `q4c`, and `experts.down_proj.weight`
must be `q4c` or `q8s32`. They cannot be `ht` or `i4r`.

The Matrix tensors of the head accept the same stores as the trunk Matrix
tensors, `ht` and `q6g64` included. An `ht` matrix of the head needs its own
rotation vectors, with the `mtp.` prefix. The published files hold no
`mtp.layers.0.mlp.experts.su`. So an `ht` shared expert of the head needs its
own `.suh` vectors.

## Vision sidecar

All 333 tensors use `bf16`. The kernels read them in place. The shapes follow
the vision encoder of the original model:

| Name | Shape |
| --- | --- |
| `visual.patch_embed.proj.weight` | `[1152, 1536]` |
| `visual.patch_embed.proj.bias` | `[1152]` |
| `visual.pos_embed.weight` | `[2304, 1152]`, a 48 × 48 grid |
| `visual.blocks.B.norm1.weight`, `.bias` | `[1152]` |
| `visual.blocks.B.attn.qkv.weight` | `[3456, 1152]` |
| `visual.blocks.B.attn.qkv.bias` | `[3456]` |
| `visual.blocks.B.attn.proj.weight` | `[1152, 1152]` |
| `visual.blocks.B.attn.proj.bias` | `[1152]` |
| `visual.blocks.B.norm2.weight`, `.bias` | `[1152]` |
| `visual.blocks.B.mlp.linear_fc1.weight` | `[4304, 1152]` |
| `visual.blocks.B.mlp.linear_fc1.bias` | `[4304]` |
| `visual.blocks.B.mlp.linear_fc2.weight` | `[1152, 4304]` |
| `visual.blocks.B.mlp.linear_fc2.bias` | `[1152]` |
| `visual.merger.norm.weight`, `.bias` | `[1152]` |
| `visual.merger.linear_fc1.weight` | `[4608, 4608]` |
| `visual.merger.linear_fc1.bias` | `[4608]` |
| `visual.merger.linear_fc2.weight` | `[2560, 4608]` |
| `visual.merger.linear_fc2.bias` | `[2560]` |

`B` runs from 0 to 26. The patch embedding takes two frames of 3 × 16 × 16
pixels, which is 1536 values.
