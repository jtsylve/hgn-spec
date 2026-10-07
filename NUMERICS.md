# Numerics

This document defines the number formats, the exact value of every stored
weight and the rounding that halogen-flash applies when it uses the weights. It
also gives the measured exactness of each GGUF conversion and the evidence for
the formats that come from published files.

## Number formats

| Format | Definition |
| --- | --- |
| `f32` | IEEE 754 binary32 |
| `f16` | IEEE 754 binary16, with subnormal values |
| `bf16` | The upper 16 bits of a binary32 value: 8 exponent bits, 7 mantissa bits |
| fp8 | 1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits, no infinity and no NaN (see [Storage formats](STORAGE.md#the-fp8-encoding)) |

## Rounding operations

**`f16` to `f32`** is exact.

**`f32` to `bf16`** rounds to nearest with ties to even. For the bit pattern
`u` of a finite `f32` value, the result is the upper 16 bits of
`u + 0x7FFF + ((u >> 16) & 1)`. Decoded weights and layer outputs never hold
NaN or infinity, so their handling is not part of this contract.

**`f32` to `f16`** rounds to nearest with ties to even, with subnormal
results. A value too large for `f16` becomes an infinity. The GGUF importer
uses this rounding for two values:

- the combined group scales of IQ4_XS and IQ3_S
- the zero of a Q5_0 block, `−16 × d`

The Q5_0 zero is exact when the importer accepts it. The importer refuses a
block whose zero is not a finite `f16`. Codebook 1 of `ht` uses the same
rounding for an exact value (see
[Storage formats](STORAGE.md#codebook)).

**`f32` to `bf16` in the importer** is not a rounding, with one exception. The
importer keeps the upper 16 bits only when the lower 16 bits are zero, and
refuses the tensor otherwise (see
[GGUF import](GGUF-IMPORT.md#float-tensors)). The exception is an affine
`out_proj` (GGUF Q5_0, Q5_1, Q4_K or Q5_K). The importer computes the `f32`
value of each weight and rounds it to `bf16` to nearest with ties to even
(see [GGUF import](GGUF-IMPORT.md#type-rules)).

## The value of a stored weight

The value of a stored weight is an `f32` number, except for the rotated stores
(see below). The table gives the operations and their order. Each operation is
one IEEE `f32` operation that rounds to nearest with ties to even. The last
column says whether the operation can round.

| Store | Value | Rounds |
| --- | --- | --- |
| `bf16`, `f16`, `f32` | the stored number | No |
| `q4c` variants 1 and 2 | `f16(scale) × codebook[c]` | Only for a codebook entry with more than 13 significant bits |
| `q4c` variant 0 | `fp8(scale) × codebook[c]` | Only for a codebook entry with more than 20 significant bits |
| `fp8r` | `fp8(code) × bf16(scale)` | No |
| `q8s32` | `code × f16(scale)` | No |
| `q8g64` | `code × f16(scale) + f16(bias)` | The sum only |
| `q6g64` | `c × f16(scale) + f16(bias)` | The sum only |
| `q6k` | `(f16(d) × scale) × (code − 32)` | No |
| `iq4nl` | `f16(d) × L[code]` | No |
| `q8_0` | `code × f16(d)` | No |
| `q5_0` | `f16(d) × (code − 16)` | No |
| `q5_1` | `code × f16(d) + f16(m)` | The sum only |
| `fp8g` | `fp8(code) × scale` | Yes |
| `q4a`, `q5a` variant 0 | `c × f16(scale) + f16(zero)` | The sum only |
| `q4a`, `q5a` variant 1 | `c × (f16(d) × sc) + (−(f16(dmin) × m))` | The sum only |

A product rounds only when its exact result has more than 24 significant
bits. An `f16` value has 11, a `bf16` value 8 and an fp8 value 4. The integer
codes and the importer's codebooks have at most 8. So most products are exact,
and the order of the operations does not matter for them. The learned
codebooks of the Flash-Next checkpoint and the NF4 and NVFP4 codebooks are
`f32` values with up to 24 significant bits, so their products round once.

`i4l`, `ht` and `i4r` are not in the table, because their values are sums.
Each store defines a stored number for each position:

- `i4l`: for a block of 256 positions, `u[j] = code × f16(scale)` is exact.
- `i4r`: `M = code × f16(scale)` is exact.
- `ht`: `M` is a value of codebook 1. The codebook rounds once to `f16` by
  definition, so `M` is an exact `f16` number.

The value is a rotated sum of these numbers. For `ht` and `i4r`, the rotation
vectors also multiply it (see
[the 128-point rotation](STORAGE.md#the-128-point-rotation)). The sum has 256
terms for `i4l` and 16,384 terms for `ht` and `i4r`, and it rounds in any
finite precision. Its exact value is defined. An implementation that needs
decoded values should sum in `f32` or wider and accept the rounding. Kernels
do not need the decoded values (see
[Storage formats](STORAGE.md#i4l-rotated-4-bit-codes)).

When the product is exact, a fused multiply-add gives the same result as a
multiply followed by an add. An implementation can use either.

The GGUF definitions of Q5_1, Q4_K and Q5_K compute the same operations. So
do the GGUF definitions of the table stores `q8_0`, `q5_0` and `q5_1`. The HGN
value of these tensors is therefore bit-identical to the GGUF value in `f32`.
A tensor converted from Q5_0 has the GGUF value too, because
`c × d + (−16 × d)` is exact. Only the sign of a zero value can differ: HGN
gives +0 where GGUF can give −0.

## How halogen-flash uses the values

The stored value is exact, but a matrix product needs more rounding decisions.
The format does not fix them. Halogen-flash uses three strategies. Strategies
1 and 2 use `bf16` activations. Strategy 3 uses rotated `f16` activations. All
three accumulate in `f32` and round each output to `bf16` (or keep it in `f32`
where the next operation needs `f32`).

### Strategy 1: round each weight

Each weight value is rounded to `bf16`. Then it is multiplied with a `bf16`
activation, and the products are summed in `f32`.

Halogen-flash uses this strategy:

- for every prompt (prefill) matrix product of a store that is not rotated: it
  decodes the whole weight into a `bf16` buffer and calls a vendor `bf16` GEMM
- for the decode-time kernels of `q8s32`, `q8g64`, `q6g64` and `q6k`
- for every dense copy: norm gains, the embedding table, the router and the
  expanded affine matrices
- for the rows of the n-gram table

A weight rounded to `bf16` keeps 8 significant bits. The result is exactly
what a `bf16` checkpoint of the same values would give.

### Strategy 2: factor out the scale

The group scale is applied to a partial sum, not to each weight.

For `q4c` decode-time kernels:

1. Round each of the 16 codebook entries to `bf16` once.
2. For each group of 32 values, sum `bf16(codebook[c]) × activation` in `f32`.
3. Multiply the group sum by the `f16` scale, in `f32`, and add it to the
   total.

For the affine expert kernels:

```text
sum over a group = scale × Σ (c × x) + zero × Σ x
```

The codes `c` are small integers, so they are exact in `bf16`. The kernel
computes `Σ x` from the same `bf16` activations as `Σ (c × x)`.

With an integer codebook, strategy 2 does not round any weight. It is at
least as accurate as strategy 1. The two strategies do not give bit-identical
results.

### Strategy 3: rotate the activations

`ht` and `i4r` keep the weights in a rotated basis (see
[the 128-point rotation](STORAGE.md#the-128-point-rotation)). Halogen-flash
rotates the activations and does not decode the weights. It uses this
strategy for the decode-time products of `ht` matrices and of the routed
experts of both stores:

1. Multiply each `bf16` activation by the input vector `su`, in `f32`. The
   product is exact.
2. Apply the 128-point transform in `f32`. Multiply each result by the `f32`
   number nearest to 1/√128.
3. Round the rotated activations to `f16`.
4. Multiply them by the stored numbers `M` in `f16`. Sum the products in
   `f32`.
5. Apply the output transform in `f32`, and multiply each result by the same
   constant.
6. Multiply each result by the output vector `sv`. Round it to `bf16`, or keep
   it in `f32`.

The stored numbers of `ht` are values of codebook 1, which are `f16` numbers.
No scale applies to them. The codes of `i4r` are exact in `f16`. The kernel
multiplies each sum over a group of 128 by the `f16` group scale, in `f32`.

For the routed experts, the input of the down projection takes steps 1 to 3
again. For `i4r`, all experts of a layer share the output vectors.
Halogen-flash therefore applies steps 5 and 6 once, after the router-weighted
sum of the selected experts.

For prompt processing, halogen-flash uses these forms:

- `ht`, with few rows: an `f16` matrix kernel on rotated `f16` activations.
- `ht`, with more rows: halogen-flash decodes the original-basis matrix into a
  `bf16` buffer, then calls a `bf16` GEMM. It computes each value in `f32` and
  rounds it to `bf16` once. An operator setting keeps these `bf16` copies
  resident, so that prompt processing does not decode them again.
- `ht`, with another operator setting: the stored numbers are rounded to
  `bf16`, and the activations are rotated into `bf16`. This rounding loses
  bits of the codebook values.
- `i4r` experts, with many rows: a batched kernel rounds each product
  `scale × code` to `bf16` or to `f16`. In the `bf16` form, the rotated
  activations are `bf16`. In the `f16` form, the kernel narrows them to `f16`
  with rounding toward zero.

The prompt-processing arithmetic of the `ht` experts is **not established**.

### Consequence for a port

Bit-identical output with halogen-flash is possible only when a port copies the
strategies and the reduction order of each kernel. That is not a goal of this
specification. A port should instead:

- decode every stored value exactly as the table above gives
- choose its own product strategy, state it and qualify it against an
  independent reference

The [porting guide](PORTING.md#numerical-strategy) discusses the choice, for
engines with floating-point and with integer activations.

## Conversion exactness

Each GGUF conversion rule in [GGUF import](GGUF-IMPORT.md) was implemented
from this specification. The test used random blocks for each GGUF type: 6
rows of 512 values, with finite random `f16` scales. The table rows used 6
rows of 160 values. The test converted the blocks to HGN and decoded them with
the rules above. It compared the result with the dequantizer of the `gguf`
Python package, version 0.19.0.

| GGUF type | HGN form | Maximum absolute difference | Maximum relative difference | Result |
| --- | --- | ---: | ---: | --- |
| Q4_0 | `q4c` | 0 | 0 | Exact |
| IQ4_NL | `q4c` | 0 | 0 | Exact |
| IQ4_XS | `q4c` | 5.4 × 10⁻² | 4.7 × 10⁻⁴ | Within the `f16` scale rounding (≤ 2⁻¹¹) |
| IQ3_S | `q4c` | 6.9 × 10⁻³ | 4.3 × 10⁻⁴ | Within the `f16` scale rounding (≤ 2⁻¹¹) |
| Q8_0 | `q8s32` | 0 | 0 | Exact |
| Q5_0 | `q5a` variant 0 | 0 | 0 | Exact. The sign of a zero value can differ |
| Q5_1 | `q5a` variant 0 | 0 | 0 | Exact |
| Q4_K | `q4a` variant 1 | 0 | 0 | Exact |
| Q5_K | `q5a` variant 1 | 0 | 0 | Exact |
| Q6_K | `q6k` | 0 | 0 | Exact |
| IQ4_NL table rows | `iq4nl` | 0 | 0 | Exact |
| Q8_0 table rows | `q8_0` | 0 | 0 | Exact |
| Q5_0 table rows | `q5_0` | 0 | 0 | Exact |
| Q5_1 table rows | `q5_1` | 0 | 0 | Exact |
| Q5_0, Q5_1, Q4_K or Q5_K on `out_proj` | `bf16` | — | 3.9 × 10⁻³ | Within the `bf16` rounding (≤ 2⁻⁸) |

For Q5_0, the test used only scales that the importer accepts (`|d|` below
4096). Of 96 Q5_0 blocks with unrestricted random scales, the importer rule
refused 10. The test of the affine `out_proj` used random scales over the
whole `f16` range, so only the relative difference is useful.

The same test found the IQ3_S grid of halogen-flash equal to the grid of
the `gguf` package. The [reverse conversions](GGUF-IMPORT.md#reverse-conversion)
gave the original GGUF bytes for Q8_0, Q5_1, Q4_K, Q5_K, Q4_0 and IQ4_NL.

The IQ4_XS and IQ3_S difference is not an importer error. The GGUF formats
multiply a sub-block scale by a super-block scale at decode time. `q4c`
stores one `f16` scale for each group, so the product must be rounded once.
The codes do not change.

The affine `out_proj` difference is the `bf16` rounding of each weight. The
`f32` value before that rounding is equal to the GGUF value.

## Evidence for the 27B formats

No analyzed engine reads `fp8r`, `i4l` or `q4c` variants 0 and 1. Their rules
come from the published 27B file. Four kinds of evidence support them:

1. **Sizes.** The size rule of each format gives the payload size of all 855
   tensors that use these formats.
2. **Twin tensors.** Each trunk matrix of the 27B file has two independent
   quantizations. Decoded with the rules of this specification, the two agree:

   | Matrix | Formats | Correlation | Amplitude ratio |
   | --- | --- | ---: | ---: |
   | `layers.0.linear_attn.in_proj_qkv` | `fp8r`, `i4l` | 0.9918 | 0.984 |
   | `layers.3.self_attn.o_proj` | `fp8r`, `i4l` | 0.9920 | 0.984 |
   | `layers.60.mlp.down_proj` | `fp8r`, `i4l` | 0.9919 | 0.983 |
   | `layers.0.mlp.gate_proj` | `q4c` variant 0, `i4l` | 0.9844 | 0.981 |
   | `layers.20.mlp.up_proj` | `q4c` variant 0, `i4l` | 0.9863 | 0.981 |

   Without the Hadamard rotation, `i4l` does not correlate with its twin at
   all (below 0.15 for every layout tried).
3. **Structure.** In sampled rows, every `q4c` variant 1 group of 32 holds a
   code of level ±1. Every variant 0 group of 16 holds a code of level ±6.
   Every `fp8r` row holds a code of magnitude 448. These are the
   absolute-maximum scaling rules of NF4, NVFP4 and fp8 with a row scale.
4. **Codebooks.** Variant 0 stores the FP4 E2M1 levels times one number, and
   variant 1 stores the standard NF4 levels.

## Evidence for the Flash-Next v2 formats

Halogen-flash reads `ht`, `i4r` and `q6g64`, so their rules come from
the analyzed reader. The published files `qwen38-flash-next-v2.hgn` and
`qwen38-flash-next-ht43.hgn` use them. Four kinds of evidence check the rules
against these files:

1. **Sizes.** The size rules give the payload size of all 8,427 entries of the
   nine published files (see [Test vectors](TEST-VECTORS.md#published-files)).
2. **Twin tensors.** The w4b checkpoint holds a `q4c` copy of each of these
   matrices. That copy is an independent quantization. Decoded with the rules
   of this specification, the two copies agree. Each `ht` and `i4r` row
   compares one block of 128 × 128 values:

   | Matrix (file) | Rows, columns | Formats | Correlation | Amplitude ratio |
   | --- | --- | --- | ---: | ---: |
   | `layers.0.mlp.shared_expert.gate_proj` (v2) | 0–127, 0–127 | `ht` 4-bit, `q4c` | 0.9899 | 0.998 |
   | `layers.0.mlp.shared_expert.gate_proj` (v2) | 0–127, 128–255 | `ht` 4-bit, `q4c` | 0.9906 | 0.994 |
   | `layers.0.mlp.experts.gate_up_proj` (ht43), expert 3 | 0–127, 0–127 | `ht` 4-bit, `q4c` | 0.9899 | 1.000 |
   | `layers.0.mlp.experts.gate_up_proj` (ht43), expert 3 | 640–767, 0–127 | `ht` 4-bit, `q4c` | 0.9897 | 0.999 |
   | `layers.0.mlp.experts.down_proj` (ht43), expert 3 | 0–127, 0–127 | `ht` 4-bit, `q4c` | 0.9925 | 0.999 |
   | `layers.1.mlp.experts.gate_up_proj` (ht43), expert 3 | 0–127, 0–127 | `ht` 3-bit, `q4c` | 0.9728 | 1.007 |
   | `layers.1.mlp.experts.gate_up_proj` (ht43), expert 3 | 640–767, 0–127 | `ht` 3-bit, `q4c` | 0.9722 | 1.009 |
   | `layers.0.mlp.experts.gate_up_proj` (v2), expert 3 | 0–127, 0–127 | `i4r`, `q4c` | 0.9906 | 1.004 |
   | `layers.0.mlp.experts.gate_up_proj` (v2), expert 3 | 640–767, 0–127 | `i4r`, `q4c` | 0.9901 | 1.003 |
   | `layers.0.attn_hyper_connection.input_mix_weight_down` (v2) | 0–3, all | `q6g64`, `q4c` | 0.9956 | 1.000 |

   Rows 0–127 of an expert are gate rows. Rows 640–767 are up rows.
3. **Controls.** A wrong rule gives no agreement:
   - Without the rotation, the `ht` stored numbers do not correlate with the
     twin (correlation below 0.004 in magnitude).
   - With the gate tiles and the up tiles exchanged, the `ht` expert
     correlation is −0.02.
   - With two's-complement codes in place of offset-binary codes, the `i4r`
     correlation is −0.57.
4. **Structure.** In the sampled rotation vectors, every input vector holds
   only +1 and −1. In the sampled `q6g64` rows, every group of 64 holds code 0
   and code 63. The published `i4r` codes use the full range from −8 to 7.

## Tolerances for tests

| Check | Tolerance |
| --- | --- |
| Decoding a payload | Bit-identical `f32` values. For `ht` and `i4r`, bit-identical stored numbers `M`, which include the `ht` codebook values |
| A rotated value (`i4l`, `ht`, `i4r`) | Compare with an FP64 sum of the exact stored numbers. The difference comes only from the finite precision of the sum |
| Converting from GGUF | Bit-identical bytes, except `A_log` |
| `A_log` conversion | One unit in the last place, because `log` differs between math libraries |
| A matrix-product kernel | Bounded by the chosen strategy. Compare with an FP64 product of the exact values |
| A complete model | The quality gates of the engine |

The affine `out_proj` conversion rounds twice by definition: first to `f32`,
then to `bf16`. An importer that keeps this order gives bit-identical bytes.
