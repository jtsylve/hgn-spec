# Qwen3.8 27B tensors

This document describes the published Qwen3.8 27B checkpoint:
`qwen3.8-27b-p1w4d-d2.hgn`, 35,865,565,184 bytes, 1,352 tensors, identity
`Qwen3.8-27B-p1-d2`. The halogen engine for this model was not analyzed. All
statements here come from the file and from the public Unsloth GGUF of the
same model (GGUF architecture `qwen35`). Treat the reader roles of
[Qwen3.8 Flash-Next tensors](TENSORS.md) as specific to halogen-flash.

## Model constants

| Constant | Value |
| --- | --- |
| Trunk layers | 64, numbered 0 to 63 |
| Hidden size | 5120 |
| Vocabulary | 248320 tokens |
| Layer type | Layer `L` is full attention when `L mod 4 = 3` (16 layers), else Gated DeltaNet (48 layers) |
| Full attention | `q_proj` has 12288 rows: 24 query heads of 256 with a 256-wide output gate. 4 key-value heads of 256 |
| Gated DeltaNet | As in Flash-Next: 16 key heads of 128, 48 value heads of 128, convolution width 4 |
| Feed-forward | Dense SwiGLU of width 17408 in every layer |
| Drafters | An MTP head and a DFlash2 block drafter of 5 layers |

## Stores

| Store | Tensors | Use |
| --- | ---: | --- |
| `bf16` | 493 | Embedding, norm gains, DeltaNet vectors and small projections, drafter parts |
| `i4l` | 400 | Prompt-processing copies of the trunk matrices |
| `fp8r` | 233 | Token-generation copies of the attention and DeltaNet matrices, of the last 8 feed-forward layers, and `lm_head` |
| `q4c` variant 0 | 168 | Token-generation copies of the feed-forward matrices of layers 0 to 55 (NVFP4 values) |
| `q4c` variant 1 | 54 | MTP and drafter matrices (NF4 codebook) |
| `i32` | 2 | Drafter configuration and vocabulary map |
| `f32` | 2 | Drafter configuration |

The formats are in [Storage formats](STORAGE.md).

## Two copies of each trunk matrix

Every large trunk matrix has two copies. `NAME` serves token generation.
`NAME.i4l` serves prompt processing: a Hadamard-rotated 4-bit copy that
integer matrix kernels can use (see [i4l](STORAGE.md#i4l-rotated-4-bit-codes)).

| Matrix | Shape | `NAME` | `NAME.i4l` |
| --- | --- | --- | --- |
| `linear_attn.in_proj_qkv.weight` | `[10240, 5120]` | `fp8r` | Yes |
| `linear_attn.in_proj_z.weight` | `[6144, 5120]` | `fp8r` | Yes |
| `linear_attn.out_proj.weight` | `[5120, 6144]` | `fp8r` | Yes |
| `self_attn.q_proj.weight` | `[12288, 5120]` | `fp8r` | Yes |
| `self_attn.k_proj.weight`, `v_proj.weight` | `[1024, 5120]` | `fp8r` | Yes |
| `self_attn.o_proj.weight` | `[5120, 6144]` | `fp8r` | Yes |
| `mlp.gate_proj.weight`, `up_proj.weight`, layers 0–55 | `[17408, 5120]` | `q4c` variant 0 | Yes |
| `mlp.down_proj.weight`, layers 0–55 | `[5120, 17408]` | `q4c` variant 0 | Yes |
| `mlp.*_proj.weight`, layers 56–63 | as above | `fp8r` | Yes |
| `lm_head.weight` | `[248320, 5120]` | `fp8r` | No |

The two copies are independent quantizations of one matrix. Decoded with the
rules of this specification, they agree with a correlation of 0.98 to 0.99 on
every sampled pair.

`linear_attn.in_proj_a.weight` and `in_proj_b.weight` (`[48, 5120]`) are
`bf16` and have one copy.

## Trunk tensors

| Name | Shape | Store | Notes |
| --- | --- | --- | --- |
| `embed_tokens.weight` | `[248320, 5120]` | `bf16` | |
| `norm.weight` | `[5120]` | `bf16` | Final norm. Zero-centred |
| `layers.L.input_layernorm.weight` | `[5120]` | `bf16` | Zero-centred |
| `layers.L.post_attention_layernorm.weight` | `[5120]` | `bf16` | Zero-centred |
| `layers.L.linear_attn.A_log` | `[48]` | `bf16` | `log(−A)`. Grouped head order |
| `layers.L.linear_attn.dt_bias` | `[48]` | `bf16` | Grouped head order |
| `layers.L.linear_attn.conv1d.weight` | `[10240, 1, 4]` | `bf16` | Grouped head order in channels 4096 and above |
| `layers.L.linear_attn.norm.weight` | `[128]` | `bf16` | A plain gain |
| `layers.L.self_attn.q_norm.weight`, `k_norm.weight` | `[256]` | `bf16` | Zero-centred |

The matrices are in the previous section. The DeltaNet head order and the
zero-centred convention are those of [Qwen3.8 Flash-Next](TENSORS.md#value-conventions).
Each claim in the Notes column was checked against the Unsloth GGUF. A
zero-centred gain is exactly the GGUF value minus 1. A plain gain is exactly
the GGUF value. After the head permutation, `dt_bias` and `conv1d` are exactly
the GGUF values, and `−exp(A_log)` is the GGUF `ssm_a` within `f32` rounding.

## MTP head

| Name | Shape | Store | Notes |
| --- | --- | --- | --- |
| `mtp.fc.weight` | `[5120, 10240]` | `bf16` | Byte-identical to the GGUF `nextn.eh_proj.weight`, with the same column order |
| `mtp.pre_fc_norm_embedding.weight` | `[5120]` | `bf16` | Zero-centred |
| `mtp.pre_fc_norm_hidden.weight` | `[5120]` | `bf16` | Zero-centred |
| `mtp.norm.weight` | `[5120]` | `bf16` | Zero-centred |
| `mtp.layers.0.input_layernorm.weight` | `[5120]` | `bf16` | Zero-centred |
| `mtp.layers.0.post_attention_layernorm.weight` | `[5120]` | `bf16` | |
| `mtp.layers.0.self_attn.q_proj.weight` | `[12288, 5120]` | `q4c` variant 1 | |
| `mtp.layers.0.self_attn.k_proj.weight`, `v_proj.weight` | `[1024, 5120]` | `q4c` variant 1 | |
| `mtp.layers.0.self_attn.o_proj.weight` | `[5120, 6144]` | `q4c` variant 1 | |
| `mtp.layers.0.self_attn.q_norm.weight`, `k_norm.weight` | `[256]` | `bf16` | |
| `mtp.layers.0.mlp.gate_proj.weight`, `up_proj.weight` | `[17408, 5120]` | `q4c` variant 1 | |
| `mtp.layers.0.mlp.down_proj.weight` | `[5120, 17408]` | `q4c` variant 1 | |

The GGUF names are `blk.64.nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`,
`nextn.shared_head_norm` and `blk.64.*` for the block. The four norms marked
zero-centred are exactly the GGUF values minus 1.

## DFlash2 drafter

The drafter is halogen's own. Its structure is outside this specification.
The tensors and their stores:

| Name | Shape | Store |
| --- | --- | --- |
| `drafter.fc.weight` | `[5120, 25600]` | `q4c` variant 1 |
| `drafter.hidden_norm.weight`, `drafter.norm.weight` | `[5120]` | `bf16` |
| `drafter.layers.D.input_layernorm.weight`, `post_attention_layernorm.weight` | `[5120]` | `bf16` |
| `drafter.layers.D.attention_conv.base_kernel`, `mlp_conv.base_kernel` | `[2, 2, 5120]` | `bf16` |
| `drafter.layers.D.attention_conv.kernel_projection.weight`, `mlp_conv.kernel_projection.weight` | `[1280, 5120]` | `q4c` variant 1 |
| `drafter.layers.D.self_attn.q_proj.weight` | `[4096, 5120]` | `q4c` variant 1 |
| `drafter.layers.D.self_attn.k_proj.weight`, `v_proj.weight` | `[1024, 5120]` | `q4c` variant 1 |
| `drafter.layers.D.self_attn.o_proj.weight` | `[5120, 4096]` | `q4c` variant 1 |
| `drafter.layers.D.self_attn.q_norm.weight`, `k_norm.weight` | `[128]` | `bf16` |
| `drafter.layers.D.mlp.gate_proj.weight`, `up_proj.weight` | `[17408, 5120]` | `q4c` variant 1 |
| `drafter.layers.D.mlp.down_proj.weight` | `[5120, 17408]` | `q4c` variant 1 |
| `drafter.candidate_selector.hidden_projection.weight` | `[256, 5120]` | `bf16` |
| `drafter.candidate_selector.predecessor_codebook`, `successor_codebook` | `[248320, 256]` | `bf16` |
| `drafter.config` | `[21]` | `i32` |
| `drafter.config_f32` | `[2]` | `f32` |
| `drafter.alpha_seed` | `[8]` | `f32` |
| `draft_vocab_map` | `[98304]` | `i32` |
| `draft_lm_head.weight` | `[98304, 5120]` | `q4c` variant 1 |

`D` runs from 0 to 4. The published `drafter.config` is
`1, 5, 5120, 17408, 128, 32, 8, 8, 2, 16, 256, 16, 248070, 2048, 0, 5, 5, 19,
33, 47, 61`, and `drafter.config_f32` is `10000000, 1e-06`. Their meaning is
**not established**. Its publisher trained this drafter. Whether its
structure matches the public z-lab DFlash2 drafter is **not established**.

## GGUF names

The trunk tensors correspond to the Unsloth `qwen35` GGUF names:

| HGN | GGUF |
| --- | --- |
| `embed_tokens.weight` | `token_embd.weight` |
| `lm_head.weight` | `output.weight` |
| `norm.weight` | `output_norm.weight` |
| `layers.L.input_layernorm.weight` | `blk.L.attn_norm.weight` |
| `layers.L.post_attention_layernorm.weight` | `blk.L.post_attention_norm.weight` |
| `layers.L.mlp.gate_proj.weight` | `blk.L.ffn_gate.weight` |
| `layers.L.mlp.up_proj.weight` | `blk.L.ffn_up.weight` |
| `layers.L.mlp.down_proj.weight` | `blk.L.ffn_down.weight` |
| `layers.L.linear_attn.*`, `layers.L.self_attn.*` | As in the [Flash-Next layer names](GGUF-IMPORT.md#layer-names) |

The row order of `q_proj` (query and gate for each head) is **not
established** for this file.
