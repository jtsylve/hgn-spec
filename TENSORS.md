# Qwen3.8 Flash-Next tensors

This document lists every tensor of a Qwen3.8 Flash-Next checkpoint in HGN
naming. For each tensor it gives the shape, the store codes that halogen-flash
accepts, the store of the published checkpoint and the convention of the
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
| Matrix | The kernels read the payload in place | `bf16`, `q4c` variant 2, `q8g64`, `q8s32`, `q6k`. Also `q4a` and `q5a`, which halogen-flash expands to `bf16` at load |
| Dense | Expanded to a device `bf16` copy at load | `bf16`, `q4c` variant 2, `q8g64`, `q8s32`, `q6k`, `q4a`, `q5a` |
| Expert | Read in place by the routed-expert kernels | Given for each tensor below |
| Fixed | Exactly one layout | Given for each tensor below |

Halogen-flash has no kernel for dense `q4a` and `q5a` matrices. It expands
them to `bf16` at load, which costs about three times their memory. A port can
do better (see [the porting guide](PORTING.md#affine-formats)).

## The published checkpoint

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

## Global tensors

| Name | Shape | Role | Published store | Notes |
| --- | --- | --- | --- | --- |
| `embed_tokens.weight` | `[248320, 2560]` | Dense | `q4c` | |
| `lm_head.weight` | `[248320, 2560]` | Matrix | `q4c` | |
| `hyper_connection_mixer.hc_norm.weight` | `[10240]` | Dense | `bf16` | [Zero-centred](#zero-centred-norm-gains) |
| `hyper_connection_mixer.input_mix_weight_down.weight` | `[320, 10240]` | Matrix | `q4c` | |
| `hyper_connection_mixer.input_mix_weight_up.weight` | `[10240, 320]` | Matrix | `q4c` | |

## Tensors of every layer

Prefix `layers.L.` for `L` from 0 to 47. Unless the table says otherwise, the
published store of a Matrix is `q4c` and of a Dense vector is `bf16`.

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
| `shared_expert_gate.weight` | `[1, 2560]` | Matrix | A one-row matrix, not a vector |
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
(see [Storage formats](STORAGE.md)). The codes and the scales of an expert are
not in one contiguous block: each plane holds the rows of all experts.

| Tensor | Accepted stores |
| --- | --- |
| `experts.gate_up_proj.weight` | `q4c` variant 2; `q4a` variant 1; `q5a` variant 1 |
| `experts.down_proj.weight` | `q4c` variant 2; `q8s32` with a payload size of exactly 901,775,360 bytes; `q5a` variant 0 |

A reader must refuse all other combinations. The pair variant of the affine
formats has no gate-and-up kernel. The k-scale variant cannot hold the down
projection, because `K = 640` is not a multiple of 256.

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
| `ngram_embedding.weight` | `[128, R, 160]` | Fixed: `iq4nl` or `fp8g` | The n-gram table. See below |

The three `i64` vectors are the parameters of the n-gram hash. The hash itself
is outside this specification. The published values are in
[Test vectors](TEST-VECTORS.md#published-files).

### The n-gram table

`layers.1.ple.ngram_embedding.weight` has `128 × R` rows of 160 values. The
published table has `R = 2500012`, which is 320,001,536 rows and 51.2 × 10⁹
values: 51.2 GB as `fp8g`, 28.8 GB as `iq4nl`. For each token the model
selects 16 rows, one for each n-gram head, and joins them into one 2560-wide
vector.

The table is the only tensor that halogen-flash does not make GPU-visible:

- It stays in the operating system page cache, mapped read-only.
- For each forward pass, the host copies the selected rows, still quantized,
  into a pinned staging buffer. With many rows it uses several threads.
- The device dequantizes the staged rows to `bf16`.

A converter that makes a complete HGN file must include the table. A file
without it cannot serve.

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
trunk's `embed_tokens` and `lm_head`. The published checkpoint holds it, and
so does the separate MTP head file.

| Name | Shape | Role | Store: checkpoint / head file |
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
must be `q4c` or `q8s32`.

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
