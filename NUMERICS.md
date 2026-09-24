# Numerics

This document defines the number formats, the exact value of every stored
weight and the rounding that halogen-flash applies when it uses the weights. It
also gives the measured exactness of each GGUF conversion.

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
uses this rounding only for the combined group scales of IQ4_XS and IQ3_S.

**`f32` to `bf16` in the importer** is not a rounding. The importer keeps the
upper 16 bits only when the lower 16 bits are zero, and refuses the tensor
otherwise (see [GGUF import](GGUF-IMPORT.md#float-tensors)).

## The value of a stored weight

The value of every stored weight is an `f32` number. The table gives the
operations and their order. Each operation is one IEEE `f32` operation that
rounds to nearest with ties to even. The last column says whether the
operation can round.

| Store | Value | Rounds |
| --- | --- | --- |
| `bf16`, `f32` | the stored number | No |
| `q4c` variants 1 and 2 | `f16(scale) × codebook[c]` | Only for a codebook entry with more than 13 significant bits |
| `q4c` variant 0 | `fp8(scale) × codebook[c]` | Only for a codebook entry with more than 20 significant bits |
| `fp8r` | `fp8(code) × bf16(scale)` | No |
| `q8s32` | `code × f16(scale)` | No |
| `q8g64` | `code × f16(scale) + f16(bias)` | The sum only |
| `q6k` | `(f16(d) × scale) × (code − 32)` | No |
| `iq4nl` | `f16(d) × L[code]` | No |
| `fp8g` | `fp8(code) × scale` | Yes |
| `q4a`, `q5a` variant 0 | `c × f16(scale) + f16(zero)` | The sum only |
| `q4a`, `q5a` variant 1 | `c × (f16(d) × sc) + (−(f16(dmin) × m))` | The sum only |

A product rounds only when its exact result has more than 24 significant
bits. An `f16` value has 11, a `bf16` value 8 and an fp8 value 4. The integer
codes and the importer's codebooks have at most 8. So most products are exact,
and the order of the operations does not matter for them. The learned
codebooks of the Flash-Next checkpoint and the NF4 and NVFP4 codebooks are
`f32` values with up to 24 significant bits, so their products round once.

`i4l` is not in the table because its values are a sum. For a block of 256
positions, the stored numbers `u[j] = code × f16(scale)` are exact. The value
`(1/16) × Σ u[j] × h(i, j)` is a sum of 256 terms and rounds in any finite
precision. Its exact value is defined; an implementation that needs decoded
values should sum in `f32` or wider and accept rounding. Kernels do not need
the decoded values (see [Storage formats](STORAGE.md#i4l-rotated-4-bit-codes)).

When the product is exact, a fused multiply-add gives the same result as a
multiply followed by an add. An implementation can use either.

The GGUF definitions of Q5_1, Q4_K and Q5_K compute the same operations. The
HGN value of a converted tensor is therefore bit-identical to the GGUF value
in `f32`.

## How halogen-flash uses the values

The stored value is exact, but a matrix product needs more rounding decisions.
The format does not fix them. Halogen-flash uses two strategies. Both use `bf16`
activations, accumulate in `f32` and round each output to `bf16` (or keep it
in `f32` where the next operation needs `f32`).

### Strategy 1: round each weight

Each weight value is rounded to `bf16`. Then it is multiplied with a `bf16`
activation, and the products are summed in `f32`.

Halogen-flash uses this strategy:

- for every prompt (prefill) matrix product: it decodes the whole weight into
  a `bf16` buffer and calls a vendor `bf16` GEMM
- for the decode-time kernels of `q8s32`, `q8g64` and `q6k`
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

### Consequence for a port

Bit-identical output with halogen-flash is possible only when a port copies both
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
rows of 512 values, with finite random `f16` scales. It converted them to HGN,
decoded them with the rules above and compared the result with the
dequantizer of the `gguf` Python package, version 0.19.0.

| GGUF type | HGN form | Maximum absolute difference | Maximum relative difference | Result |
| --- | --- | ---: | ---: | --- |
| Q4_0 | `q4c` | 0 | 0 | Exact |
| IQ4_NL | `q4c` | 0 | 0 | Exact |
| IQ4_XS | `q4c` | 5.4 × 10⁻² | 4.7 × 10⁻⁴ | Within the `f16` scale rounding (≤ 2⁻¹¹) |
| IQ3_S | `q4c` | 6.9 × 10⁻³ | 4.3 × 10⁻⁴ | Within the `f16` scale rounding (≤ 2⁻¹¹) |
| Q8_0 | `q8s32` | 0 | 0 | Exact |
| Q5_1 | `q5a` variant 0 | 0 | 0 | Exact |
| Q4_K | `q4a` variant 1 | 0 | 0 | Exact |
| Q5_K | `q5a` variant 1 | 0 | 0 | Exact |
| Q6_K | `q6k` | 0 | 0 | Exact |
| IQ4_NL table rows | `iq4nl` | 0 | 0 | Exact |

The same test found the IQ3_S grid of halogen-flash 0.13.5 equal to the grid of
the `gguf` package. The [reverse conversions](GGUF-IMPORT.md#reverse-conversion)
gave the original GGUF bytes for Q8_0, Q5_1, Q4_K, Q5_K, Q4_0 and IQ4_NL.

The IQ4_XS and IQ3_S difference is not an importer error. The GGUF formats
multiply a sub-block scale by a super-block scale at decode time. `q4c`
stores one `f16` scale for each group, so the product must be rounded once.
The codes do not change.

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

## Tolerances for tests

| Check | Tolerance |
| --- | --- |
| Decoding a payload | Bit-identical `f32` values |
| Converting from GGUF | Bit-identical bytes, except `A_log` |
| `A_log` conversion | One unit in the last place, because `log` differs between math libraries |
| A matrix-product kernel | Bounded by the chosen strategy. Compare with an FP64 product of the exact values |
| A complete model | The quality gates of the engine |
