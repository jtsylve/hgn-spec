# GGUF import

Halogen-flash reads a Qwen3.8 Flash-Next GGUF file in two ways. It can convert
the file into memory at start-up, or it can write a complete HGN file. Both ways
use the rules in this document. The rules also show how to keep the HGN layouts
in an engine that loads GGUF files.

The importer follows one principle: **it refuses rather than guesses.** Three
things stop the import with a message: an unknown tensor name, a type without
an exact conversion and a shape that the kernels do not expect. A checkpoint
that loads but computes wrong values is harder to find than one that does not
load.

## Preconditions

The importer reads GGUF versions 2 and 3. It uses `general.alignment` for the
data section, with 32 as the default.

A path that ends in `-NNNNN-of-MMMMM.gguf`, with five digits in each group,
names a split model. The importer opens shards `00001` to `MMMMM` with the
same prefix. It refuses a tensor name that is in two shards.

The metadata must hold these values:

| Key | Required value |
| --- | --- |
| `general.architecture` | `qwen4exp` |
| `qwen4exp.ssm.group_count` | 16 |
| `qwen4exp.ssm.time_step_rank` | 48 |
| `qwen4exp.ssm.state_size` | 128 |
| `qwen4exp.block_count` | 48 |
| `qwen4exp.expert_count` | 512 |
| `qwen4exp.embedding_length` | 2560 |
| `qwen4exp.ple.layer_multipliers` | An integer array |
| `qwen4exp.ple.head_offsets` | An integer array |
| `qwen4exp.ple.head_vocab_sizes` | An integer array |

The importer reads no other metadata. It reports `general.file_type` but does
not use it. A tensor with more than four dimensions is an error.

## Type rules

Every tensor of the file is checked, before any payload is read.

| GGUF type | HGN store | Variant | Exact | `K` rule |
| --- | --- | ---: | --- | --- |
| BF16 | `bf16` | 0 | Yes, bytes copied | — |
| F32 | `bf16` | 0 | Yes, or refused (see [Float tensors](#float-tensors)) | — |
| F16 | `bf16` | 0 | Yes, or refused | — |
| Q8_0 | `q8s32` | 0 | Yes | multiple of 64 |
| Q4_0 | `q4c` | 2 | Yes | multiple of 32 |
| IQ4_NL | `q4c` | 2 | Yes | multiple of 32 |
| IQ4_XS | `q4c` | 2 | Codes yes; scales rounded to `f16` | multiple of 256 |
| IQ3_S | `q4c` | 2 | Codes yes; scales rounded to `f16` | multiple of 256 |
| Q5_1 | `q5a` | 0 | Yes | multiple of 32 |
| Q4_K | `q4a` | 1 | Yes | multiple of 256 |
| Q5_K | `q5a` | 1 | Yes | multiple of 256 |
| Q6_K | `q6k` | 0 | Yes, bytes copied | multiple of 256 |

These types are refused anywhere in the file, because they have no exact
conversion: Q4_1, Q5_0, Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S,
IQ1_M, TQ1_0, TQ2_0 and MXFP4. Q8_1, Q8_K, I8, I16, I32, I64 and F64 have no
rule and are also refused.

Some tensors have narrower rules:

| Tensor | Rule |
| --- | --- |
| Fused expert gate and up | Q4_0, IQ4_NL, IQ4_XS or IQ3_S give `q4c`. Q5_1, Q4_K or Q5_K give the affine form. Other types are refused. Both halves must have the same type |
| Indexer query and key | BF16 only |
| `ssm_a` | F32 only |
| `ssm_out` (`out_proj`) | Not Q6_K, Q5_1, Q4_K or Q5_K, because the [head reorder](#head-reorder) cannot move their blocks |
| `per_layer_token_embd` | IQ4_NL only, with GGUF dimensions `[160, 128 × R]` |

The importer accepts some types that the reader then refuses. For the expert
gate and up tensor, the reader accepts only the k-scale affine form. For the
expert down tensor, it accepts only `q5a` in the pair form (see
[Qwen3.8 Flash-Next tensors](TENSORS.md#fused-experts)). So a file with Q5_1 expert gate
and up projections converts, but does not load.

## Worked example

The Unsloth UD-Q4_K_XL build (revision `38bb39ee`, four shards, 1,224 tensors). 
Its types convert as follows. Every conversion in it is exact.

| GGUF tensors | GGUF type | HGN store |
| --- | --- | --- |
| `token_embd`, `output` | Q8_0 | `q8s32` |
| Dense projections: `attn_qkv`, `attn_gate`, `ssm_out`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_*_shexp`, `hc_*_down`, `hc_*_up`, `output_hc_down`, `output_hc_up`, `ple_key`, `ple_value` | Q8_0 | `q8s32` |
| `ffn_gate_exps` and `ffn_up_exps`, 47 layers | Q4_K | `q4a` variant 1 |
| `ffn_gate_exps` and `ffn_up_exps`, 1 layer | Q5_K | `q5a` variant 1 |
| `ffn_down_exps`, 43 layers | Q5_1 | `q5a` variant 0 |
| `ffn_down_exps`, 5 layers | Q8_0 | `q8s32` |
| `indexer.q_proj`, `indexer.k_proj` | BF16 | `bf16`, joined |
| `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, `ffn_gate_inp`, `ffn_gate_inp_shexp`, `hc_*_inject`, all norms | F32 | `bf16` |
| `ssm_a` | F32 | `f32` (`A_log`) |
| `per_layer_token_embd` | IQ4_NL, `[160, 320001536]` | The n-gram table, unchanged |

The dense projections are Q8_0 in this build, so the converted trunk is larger
than the native checkpoint, which uses 4-bit codes for them.

## Name map

Every GGUF tensor must map to an HGN name. An unmapped name is an error.

### Whole names

| GGUF name | HGN name |
| --- | --- |
| `token_embd.weight` | `embed_tokens.weight` |
| `output.weight` | `lm_head.weight` |
| `output_hc_down.weight` | `hyper_connection_mixer.input_mix_weight_down.weight` |
| `output_hc_up.weight` | `hyper_connection_mixer.input_mix_weight_up.weight` |
| `output_hc_norm.weight` | `hyper_connection_mixer.hc_norm.weight` |
| `per_layer_token_embd.weight` | `layers.1.ple.ngram_embedding.weight` |

### Layer names

`blk.L.SUFFIX` becomes `layers.L.NEW`:

| GGUF suffix | HGN suffix |
| --- | --- |
| `attn_qkv.weight` | `linear_attn.in_proj_qkv.weight` |
| `attn_gate.weight` | `linear_attn.in_proj_z.weight` |
| `ssm_alpha.weight` | `linear_attn.in_proj_a.weight` |
| `ssm_beta.weight` | `linear_attn.in_proj_b.weight` |
| `ssm_out.weight` | `linear_attn.out_proj.weight` |
| `ssm_conv1d.weight` | `linear_attn.conv1d.weight` |
| `ssm_norm.weight` | `linear_attn.norm.weight` |
| `ssm_a` | `linear_attn.A_log` |
| `ssm_dt.bias` | `linear_attn.dt_bias` |
| `attn_q.weight` | `self_attn.q_proj.weight` |
| `attn_k.weight` | `self_attn.k_proj.weight` |
| `attn_v.weight` | `self_attn.v_proj.weight` |
| `attn_output.weight` | `self_attn.o_proj.weight` |
| `attn_q_norm.weight` | `self_attn.q_norm.weight` |
| `attn_k_norm.weight` | `self_attn.k_norm.weight` |
| `indexer.q_norm.weight` | `self_attn.indexer.q_layernorm.weight` |
| `indexer.k_norm.weight` | `self_attn.indexer.k_layernorm.weight` |
| `ffn_gate_inp.weight` | `mlp.gate.weight` |
| `ffn_gate_inp_shexp.weight` | `mlp.shared_expert_gate.weight` |
| `ffn_gate_shexp.weight` | `mlp.shared_expert.gate_proj.weight` |
| `ffn_up_shexp.weight` | `mlp.shared_expert.up_proj.weight` |
| `ffn_down_shexp.weight` | `mlp.shared_expert.down_proj.weight` |
| `ffn_down_exps.weight` | `mlp.experts.down_proj.weight` |
| `hc_attn_down.weight` | `attn_hyper_connection.input_mix_weight_down.weight` |
| `hc_attn_up.weight` | `attn_hyper_connection.input_mix_weight_up.weight` |
| `hc_attn_inject.weight` | `attn_hyper_connection.block_inject_weight.weight` |
| `hc_attn_norm.weight` | `attn_hyper_connection.hc_norm.weight` |
| `hc_ffn_down.weight` | `mlp_hyper_connection.input_mix_weight_down.weight` |
| `hc_ffn_up.weight` | `mlp_hyper_connection.input_mix_weight_up.weight` |
| `hc_ffn_inject.weight` | `mlp_hyper_connection.block_inject_weight.weight` |
| `hc_ffn_norm.weight` | `mlp_hyper_connection.hc_norm.weight` |
| `ple_key.weight` | `ple.key_proj.weight` |
| `ple_value.weight` | `ple.value_proj.weight` |
| `ple_conv1d.weight` | `ple.conv1d.weight` |
| `ple_norm_key.weight` | `ple.norm_key.weight` |
| `ple_norm_query.weight` | `ple.norm_query.weight` |
| `ple_norm_conv.weight` | `ple.norm_conv.weight` |

### Fused pairs

Two pairs of GGUF tensors become one HGN tensor each. The importer plans the
pair when it meets either half. Both halves must exist.

| GGUF first half | GGUF second half | HGN name |
| --- | --- | --- |
| `ffn_gate_exps.weight` | `ffn_up_exps.weight` | `mlp.experts.gate_up_proj.weight` |
| `indexer.q_proj.weight` | `indexer.k_proj.weight` | `self_attn.indexer.index_qk_proj.weight` |

The expert pair is interleaved for each expert: 640 gate rows, then 640 up
rows (see [Qwen3.8 Flash-Next tensors](TENSORS.md#fused-experts)). The indexer pair is
joined: all query rows, then all key rows.

### Metadata arrays

Three metadata arrays become `i64` tensors of layer 1. Each element becomes
one signed 64-bit integer.

| GGUF key | HGN name |
| --- | --- |
| `qwen4exp.ple.layer_multipliers` | `layers.1.ple.ple_embedding.layer_multipliers` |
| `qwen4exp.ple.head_offsets` | `layers.1.ple.ple_embedding.ngram_heads_offsets` |
| `qwen4exp.ple.head_vocab_sizes` | `layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` |

### The MTP head

GGUF files carry no MTP tensors that halogen-flash reads. The `mtp.*` tensors
come from the MTP head file (see [Container](CONTAINER.md#mtp-head-file)) and
are copied unchanged.

## Shapes

GGUF lists dimensions with the fastest-changing one first. HGN lists them with
the slowest-changing one first. So the HGN shape is the GGUF dimension list in
reverse order. Then two tensors change shape:

- A depthwise convolution `[W, C]` in GGUF becomes `[C, 1, W]`.
- `ffn_gate_inp_shexp.weight` `[2560]` becomes `[1, 2560]`.

The n-gram table `[160, 128 × R]` becomes `[128, R, 160]`.

After the plan is complete, the importer compares every shape with the shape
in [Qwen3.8 Flash-Next tensors](TENSORS.md). It collects all differences and refuses the
file if there is one or more. It also refuses a name of 96 bytes or more.

## Conversions

### Q4_0 and IQ4_NL to q4c

A GGUF block of 32 values is 18 bytes: an `f16` scale `d`, then 16 bytes. Byte
`j` holds the code of value `j` in bits 0–3 and of value `j + 16` in bits 4–7.

For each block `g` of a row:

1. Copy `d`, bit for bit, to scale `g` of the row.
2. Take the 32 codes in value order. Pack them two to a byte in the `q4c`
   order: byte `m` of the group gets value `2m` in bits 0–3 and value
   `2m + 1` in bits 4–7.

The codebook is the Q4_0 codebook or the IQ4 codebook of
[Storage formats](STORAGE.md#q4c-4-bit-codebook). The Q4_0 codebook is
`c − 8`, so `value = d × (c − 8)` as in GGUF. The conversion is exact.

### IQ4_XS to q4c

A GGUF super-block of 256 values is 136 bytes:

| Offset | Size | Contents |
| --- | ---: | --- |
| 0 | 2 | `f16` super-block scale `d` |
| 2 | 2 | `u16` `scales_h`: 2 high bits for each of 8 sub-blocks |
| 4 | 4 | `scales_l`: 4 low bits for each of 8 sub-blocks |
| 8 | 128 | 8 sub-blocks of 16 code bytes, in the Q4_0 nibble order |

For super-block `s` of a row and its sub-block `j` from 0 to 7 (group
`8s + j` of the row):

1. Compute the 6-bit sub-block scale:

   ```text
   ls = ((scales_l[j / 2] >> (4 × (j mod 2))) & 15) | (((scales_h >> 2j) & 3) << 4)
   ```
2. Compute `(ls − 32) × d` in `f32`. The product is exact.
3. Round it to `f16`, to nearest with ties to even. Store it as the group
   scale.
4. Re-pack the 16 code bytes as for Q4_0.

The codebook is the IQ4 codebook. The codes do not change. The scale can
change by at most half an `f16` unit in the last place, a relative error of at
most 2⁻¹¹.

### IQ3_S to q4c

A GGUF super-block of 256 values is 110 bytes:

| Offset | Size | Contents |
| --- | ---: | --- |
| 0 | 2 | `f16` super-block scale `d` |
| 2 | 64 | `qs`: low 8 bits of 64 grid indices |
| 66 | 8 | `qh`: bit 8 of the grid indices, one byte for each sub-block |
| 74 | 32 | `signs`: one sign bit for each value |
| 106 | 4 | 4-bit sub-block scales |

The grid is the standard GGUF IQ3_S grid: 512 `u32` words. Each byte of a word
is an odd level from 1 to 15.

For super-block `s` of a row and its sub-block `sb` from 0 to 7 (group
`8s + sb` of the row):

1. `ls = (scales[sb / 2] >> (4 × (sb mod 2))) & 15`.
2. Compute `(2 × ls + 1) × d` in `f32`. Round it to `f16` as for IQ4_XS. Store
   it as the group scale.
3. For word `w` from 0 to 7 of the sub-block:
   1. `index = qs[8sb + w] | (((qh[sb] >> w) & 1) << 8)`.
   2. `word = grid[index]`.
   3. `sign = signs[4sb + w / 2]`.
   4. For byte `e` from 0 to 3 of the word, value `4w + e` of the sub-block
      has `level = ((word >> 8e) & 255) >> 1`, from 0 to 7.
   5. When bit `4 × (w mod 2) + e` of `sign` is 1, the code is `7 − level`.
      When it is 0, the code is `8 + level`.
4. Pack the 32 codes in the `q4c` order.

The codebook is the odd codebook `2c − 15`. Code `7 − level` decodes to
`−(2 × level + 1)` and code `8 + level` to `2 × level + 1`, which are the
negative and positive grid levels. The codes are exact. The scale changes as
for IQ4_XS.

### Q8_0 to q8s32

A GGUF block of 32 values is 34 bytes: an `f16` scale, then 32 signed codes.
For block `g` of a row, copy the 32 codes to row bytes `32g` to `32g + 31`.
Copy the scale to row bytes `K + 2g` and `K + 2g + 1`. Fill the rest of the
row with zeros. The conversion is exact.

### Q5_1 to q5a variant 0

A GGUF block of 32 values is 24 bytes:

| Offset | Size | Contents |
| --- | ---: | --- |
| 0 | 2 | `f16` scale `d` |
| 2 | 2 | `f16` minimum `m` |
| 4 | 4 | `u32` `qh`: bit `i` is bit 4 of the code of value `i` |
| 8 | 16 | Low 4 bits of the codes, in the Q4_0 nibble order |

For block `g` of a row:

1. Copy `d` and `m` to the plane-1 record of group `g`, in that order.
2. Copy `qh` unchanged to high-plane word `g`.
3. Re-pack the 16 code bytes as for Q4_0.

GGUF defines `value = code × d + m`. HGN defines `value = code × scale +
zero`. The conversion is exact.

### Q4_K and Q5_K to q4a and q5a variant 1

A GGUF Q4_K super-block of 256 values is 144 bytes. A Q5_K super-block is 176
bytes.

| Offset | Size | Contents |
| --- | ---: | --- |
| 0 | 2 | `f16` `d` |
| 2 | 2 | `f16` `dmin` |
| 4 | 12 | Eight 6-bit scales and eight 6-bit minimums, packed |
| 16 | 32 | Q5_K only: `qh`. Bit `j` of byte `l` is bit 4 of the code of value `32j + l` |
| 16 (Q4_K), 48 (Q5_K) | 128 | `qs`: low 4 bits of the codes |

For super-block `s` of a row:

1. Copy `d` and `dmin` to the plane-2 record of super-block `s`.
2. Unpack the 12 bytes `q[0..11]` into `sc[j]` and `m[j]` for `j` from 0 to 7:
   - For `j < 4`: `sc[j] = q[j] & 63` and `m[j] = q[j + 4] & 63`.
   - For `j ≥ 4`: `sc[j] = (q[j + 4] & 15) | ((q[j − 4] >> 6) << 4)` and
     `m[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)`.
3. Write `sc[j]`, then `m[j]`, as the plane-1 record of group `8s + j`.
4. For group `8s + j`, the 32 low code bits are in bytes `32 × (j / 2)` to
   `32 × (j / 2) + 31` of `qs`: bits 0–3 when `j` is even, bits 4–7 when `j`
   is odd. Value `l` of the group is in byte `l` of that range. Pack the 32
   codes in the `q4c` order.
5. For Q5_K, set bit `l` of high-plane word `8s + j` to bit `j` of `qh[l]`.

GGUF defines `value = d × sc × code − dmin × m`. HGN computes the same numbers
(see [Numerics](NUMERICS.md)). The conversion is exact.

### Q6_K and BF16

The bytes are copied unchanged. A head reorder moves whole rows or whole
column blocks (see [Head reorder](#head-reorder)).

### Float tensors

An F32 or F16 tensor becomes `bf16`. F16 values are first widened to `f32`.
Then:

1. Subtract 1.0 in `f32` from every value of a
   [norm gain with an offset](#norm-offsets).
2. Apply the [head reorder](#head-reorder) when the tensor has one.
3. Check that the low 16 bits of every `f32` value are zero. If one value
   fails, refuse the tensor. The message gives the count of failed values.
4. Store the upper 16 bits of each value.

The check in step 3 makes the conversion exact. A value that is not already
a `bf16` value shows a transform that the importer does not know, or an F16
weight that was never `bf16`. The importer does not round it.

### ssm_a to A_log

GGUF stores `A`, which is negative. HGN stores `log(−A)` as `f32`.

1. Refuse the tensor if any value is zero or positive. Check all values
   first, so that no partial result is written.
2. Apply the head reorder: HGN value `h` comes from GGUF value `t(h)`.
3. Compute `log(−A)` in `f32` arithmetic.

Math libraries can differ by one unit in the last place. The result is
therefore not guaranteed to be bit-identical between importers.

### Fused experts

For expert `e` from 0 to 511, rows `640e` to `640e + 639` of the GGUF gate
tensor become HGN rows `1280e` to `1280e + 639`. The same rows of the GGUF up
tensor become HGN rows `1280e + 640` to `1280e + 1279`. Each row is converted
with the rule of its type. One tensor has one codebook, so gate and up must
have the same type.

## Norm offsets

GGUF adds 1 to these RMSNorm gains. The importer subtracts 1 so that HGN holds
the zero-centred value (see
[Qwen3.8 Flash-Next tensors](TENSORS.md#zero-centred-norm-gains)). The importer selects the
tensors by the end of their HGN name:

- `hc_norm.weight`
- `self_attn.q_norm.weight`
- `self_attn.k_norm.weight`
- `q_layernorm.weight`
- `k_layernorm.weight`
- `ple.norm_key.weight`
- `ple.norm_query.weight`
- `ple.norm_conv.weight`

`linear_attn.norm.weight` is not in the list.

Halogen-flash subtracts the offset only when it converts an F32 or F16 tensor. A
norm gain stored as BF16 in the GGUF file keeps its offset. GGUF writers store
norm gains as F32, so this does not occur in known files. A new importer
should subtract the offset for every source type.

## Head reorder

GGUF stores the 48 DeltaNet value heads in tiled order. HGN stores them in
grouped order (see [Qwen3.8 Flash-Next tensors](TENSORS.md#deltanet-head-order)). HGN head
`h` takes GGUF head `t(h) = h / 3 + 16 × (h mod 3)`.

| Tensor | What moves | Unit |
| --- | --- | --- |
| `in_proj_qkv` | Rows 4096 to 10239 | Bands of 128 rows |
| `in_proj_z` | All 6144 rows | Bands of 128 rows |
| `in_proj_a`, `in_proj_b` | All 48 rows | Single rows |
| `out_proj` | All 6144 columns | Blocks of 128 columns |
| `conv1d` | Channels 4096 to 10239 | Bands of 128 channels |
| `A_log`, `dt_bias` | All 48 values | Single values |

A row move is a copy of whole rows, in any storage format. A column move
must move whole groups. A block of 128 columns is 4 groups of 32 values, so:

- `q4c`: for each row, move 4 groups of 16 code bytes and 4 `f16` scales for
  each head. Destination group `j` takes source group
  `(j mod 4) + 4 × t(j / 4)`.
- `q8s32`: for each row, move 4 groups of 32 codes and their 4 scales, with
  the same group rule.
- `bf16`: for each row, move 48 blocks of 256 bytes.

Q6_K and the K-quants group 256 values, so a 128-column block splits a group.
The importer refuses these types on `out_proj`. It also refuses Q5_1 there,
although a Q5_1 group is 32 values wide. A new importer can support Q5_1 on
`out_proj` by moving whole groups, as for `q4c`.

Halogen-flash 0.13.5 does not apply the reorder in these cases:

- `in_proj_a` and `in_proj_b` with GGUF type Q8_0
- `in_proj_qkv`, `in_proj_z` and `out_proj` with GGUF type F32 or F16
- `conv1d` and `dt_bias` with GGUF type BF16

Its output is then in the wrong order. A new importer must apply the reorder
for every type.

## Output order

The importer sorts the tensors by a five-part key and writes the table and the
payloads in that order:

| Group | Tensors | Then sorted by |
| ---: | --- | --- |
| 0 | `embed_tokens.weight` | — |
| 1 | `layers.L.*` | Layer number, then [sublayer index](TENSORS.md#name-grammar), then the rest of the name |
| 2 | `hyper_connection_mixer.*` | The name |
| 3 | `lm_head.weight` | — |
| 4 | `mtp.*` | Names without a layer first, by the name without `mtp.`; then `mtp.layers.L.*` by layer, sublayer index and the rest of the name |

"The rest of the name" is the part after `layers.L.`. Names compare as byte
strings. Two different tensors never have equal keys, so the order is total.

## Output files

The importer makes one of two outputs.

**In memory, at start-up.** The result holds every converted tensor. It does
not hold the n-gram table or the MTP head tensors. Halogen-flash maps the n-gram
table directly from the GGUF shard, and it opens the MTP head file as a
second checkpoint. The payloads follow the output order, each at a multiple of
64, in one anonymous memory region. The [repack
cache](CONTAINER.md#repack-cache) stores this result.

**A complete HGN file.** The file holds every converted tensor, the n-gram
table (unless the operator excludes it) and the MTP head tensors (when the
operator names a head file). The n-gram table bytes are copied unchanged. The
identity is `qwen3.8-flash-next-gguf` and the version is 2.

A complete file for serving must hold the n-gram table.

## Reproducibility

The same GGUF files and head file always give the same bytes. The importer
converts tensors in parallel, but the output order and every offset are
fixed before any payload is written.

Halogen-flash can print a digest listing instead of writing the file. The
listing has one SHA-256 line for the header and table together, one line for
each tensor payload (without padding) and the file size:

```text
<sha256 of header and table>  <bytes>  <header+table>
<sha256 of payload>  <bytes>  <tensor name>
...
file_bytes <total file size>
```

Two importers agree when their listings agree. Use this to test a new
importer against halogen-flash without a second copy of a 60 GB file.

## Reverse conversion

The affine formats, `q8s32` and some `q4c` tensors hold exactly the numbers of
a GGUF block. An engine that already has GGUF kernels can rebuild the GGUF
blocks from HGN payloads. Every rebuild below was checked to give the original
GGUF bytes.

| HGN form | GGUF type | Rebuild |
| --- | --- | --- |
| `q8s32` | Q8_0 | For each group, write the scale, then the 32 codes |
| `q5a` variant 0 | Q5_1 | Write the plane-1 record (`d`, `m`), the high-plane word, then the 16 code bytes in GGUF nibble order |
| `q4a` variant 1 | Q4_K | Write `d` and `dmin`, the packed scales, then `qs` |
| `q5a` variant 1 | Q5_K | As Q4_K, with `qh` between the packed scales and `qs` |
| `q4c`, Q4_0 codebook | Q4_0 | For each group, write the scale, then the 16 code bytes in GGUF nibble order |
| `q4c`, IQ4 codebook | IQ4_NL | As Q4_0 |

To pack the scales of a K-quant super-block, for `j` from 0 to 3:

```text
q[j]     = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6)
q[j + 4] = (m[j] & 63)  | ((m[j + 4] >> 4) << 6)
q[j + 8] = (sc[j + 4] & 15) | ((m[j + 4] & 15) << 4)
```

GGUF nibble order for a 32-value group: byte `l` of the 16 bytes gets the
code of value `l` in bits 0–3 and of value `l + 16` in bits 4–7. For a
K-quant super-block: byte `32c + l` of `qs` gets the code of value `64c + l`
in bits 0–3 and of value `64c + 32 + l` in bits 4–7.

A `q4c` tensor with another codebook has no GGUF equivalent. This includes
tensors converted from IQ3_S and every tensor of the published Flash-Next
checkpoint, which has learned codebooks. Such a tensor needs a `q4c` kernel.
