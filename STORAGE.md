# HGN storage formats

The store code of an entry selects the storage format of its payload. The
variant field selects a form of some formats. This document gives the byte
layout, the size rule and the decode rule of each format.
[Numerics](NUMERICS.md) gives the arithmetic in full detail.

## Store codes

| Code | Name | Read by halogen-flash | In published files | Use |
| ---: | --- | --- | --- | --- |
| 0 | `bf16` | Yes | Both models | Dense weights, norm gains, embeddings, vision |
| 1 | `f32` | Yes | 27B | Small vectors; `A_log` from a GGUF import |
| 2 | `f16` | Rotation vectors only | Flash-Next | [Rotation vectors](#rotation-vectors) of `ht` and `i4r` tensors |
| 3 | `i32` | No | 27B | Small integer vectors of the 27B drafter |
| 4 | `i64` | Yes | Flash-Next | Three PLE index vectors |
| 5 | `q4c` | Variant 2 only | Both models | 4-bit codes with a codebook (display name Q4C-P) |
| 6 | `fp8r` | No | 27B | 8-bit float codes with a scale for each row |
| 7 | `q8g64` | Yes | Flash-Next | 8-bit unsigned codes with an affine pair for each 64 values |
| 8 | `i4l` | No | 27B | 4-bit codes in a rotated basis, for prompt processing |
| 9 | — | — | None | Unassigned. Halogen-flash prints `?` for this code |
| 10 | `fp8g` | Yes | Flash-Next | n-gram table: 8-bit float codes with one global scale |
| 11 | `q8s32` | Yes | None | 8-bit signed codes with a scale for each 32 values |
| 12 | `q6k` | Yes | None | GGUF Q6_K super-blocks, unchanged |
| 13 | `iq4nl` | Yes | None | n-gram table: GGUF IQ4_NL blocks, unchanged |
| 14 | `q4a` | Yes | None | 4-bit affine codes in planes |
| 15 | `q5a` | Yes | None | 5-bit affine codes in planes |
| 16 | `ht` | Yes | Flash-Next | 3-bit or 4-bit trellis codes in a rotated basis |
| 17–19 | — | — | None | Unassigned. Halogen-flash prints `?` for these codes |
| 20 | `q8_0` | Yes | None | n-gram table: GGUF Q8_0 blocks, unchanged |
| 21 | `q5_0` | Yes | None | n-gram table: GGUF Q5_0 blocks, unchanged |
| 22 | `q5_1` | Yes | None | n-gram table: GGUF Q5_1 blocks, unchanged |
| 23 | `i4r` | Yes | Flash-Next | Routed experts: 4-bit codes in a rotated basis |
| 24 | `q6g64` | Yes | Flash-Next | 6-bit unsigned codes with an affine pair for each 64 values |

The names are the strings that halogen-flash prints in its messages.
`q8s32`, `q6k`, `iq4nl`, `q8_0`, `q5_0`, `q5_1`, `q4a` and `q5a` occur in
files that halogen-flash makes from GGUF files (see
[GGUF import](GGUF-IMPORT.md)). `ht`, `i4r` and `q6g64` occur in published
files. Halogen-flash reads them but does not write them. A reader must refuse
a code that it does not implement.



## Common rules

- A payload holds `N` rows of `K` values. Value `(r, k)` has row `r` in
  `[0, N)` and position `k` in `[0, K)`.
- A 4-bit code plane packs two codes into each byte. Byte `j` of a row holds
  the code of value `2j` in bits 0–3 and the code of value `2j + 1` in bits
  4–7. This is not the GGUF order. A GGUF 32-value block puts value `j` and
  value `j + 16` into one byte. `iq4nl`, `q5_0` and `q5_1` keep the GGUF
  order. `ht` uses a bitstream, not a code plane.
- All padding bytes are zero. This includes the tail of each scale row after
  its last record.
- A writer places each payload at a multiple of 64. Every plane of the planar
  formats then starts at a multiple of 64, and every scale row at a multiple
  of 16.

## Size summary

| Store | Variant | `K` must be a multiple of | Payload size |
| --- | ---: | ---: | --- |
| `bf16`, `f16` | 0 | — | `2 × N × K` |
| `f32`, `i32` | 0 | — | `4 × N × K` |
| `i64` | 0 | — | `8 × N × K` |
| `q4c` | 2 | 32 | `64 + pad(N × K / 2, 64) + N × pad(K / 16, 16)` |
| `q4c` | 0, 1 | 256 (observed) | `64 + N × (K / 2 + K / 16)` |
| `fp8r` | 0 | — | `N × K + 2 × N` |
| `i4l` | `0x10100` | 256 | `N × K / 2 + 2 × N × K / 256` |
| `q8s32` | 0 | 64 | `N × pad(K + K / 16, 16)` |
| `q8g64` | 0 | 64 | `N × (K + K / 16)` |
| `q6g64` | 64 | 64 | `N × pad(13 × K / 16, 16)` |
| `q6k` | 0 | 256 | `210 × N × K / 256` |
| `ht` | `0x1206` (`b` = 3) or `0x1208` (`b` = 4) | 128, and `N` a multiple of 128 | `N × K × b / 8` |
| `i4r` | 128 | `K` = 2560 or 640 | `N × K / 2 + 2 × N × K / 128` |
| `iq4nl` | 0 | `K` = 160 | `90 × N` |
| `q8_0`, `q5_0`, `q5_1` | 0 | `K` = 160 | `170 × N`, `110 × N`, `120 × N` |
| `fp8g` | 0 | `K` = 160 | `pad(160 × N, 64) + 4` |
| `q4a`, `q5a` | 0 or 1 | 32 (variant 0), 256 (variant 1) | See [q4a and q5a](#q4a-and-q5a-affine-codes) |

For `iq4nl`, `q8_0`, `q5_0`, `q5_1` and `fp8g`, `N` is the number of table
rows, `shape[0] × shape[1]`. `ht` and `i4r` also have shape rules (see
[ht](#ht-trellis-codes-in-a-rotated-basis) and
[i4r](#i4r-rotated-4-bit-expert-codes)). Every one of the 8,427 entries of
the nine published files has the size that this table gives.

## Dense formats

`bf16`, `f32`, `f16`, `i32` and `i64` hold `N × K` numbers in row-major
order, 2, 4, 2, 4 and 8 bytes each. `f16` numbers are IEEE 754 binary16.
`i32` and `i64` numbers are signed.

Halogen-flash reads `f16` only as [rotation vectors](#rotation-vectors). The
published files use `f16` only for rotation vectors.

## The fp8 encoding

Three formats use one 8-bit floating-point encoding: `fp8r`, `fp8g` and the
scales of `q4c` variant 0. It has a sign bit (bit 7), a 4-bit exponent `e`
(bits 3–6) and a 3-bit mantissa `m` (bits 0–2). The exponent bias is 7.

| Exponent | Magnitude |
| --- | --- |
| `e = 0` | `m × 2^−9` |
| `e = 1` to `15` | `(1 + m / 8) × 2^(e − 7)` |

The sign bit negates the magnitude. The encoding has no infinity and no NaN.
Code `0x7F` is 480 and code `0xFF` is −480. This is the only difference from
the OCP E4M3FN encoding, where those two codes are NaN. No sampled payload of
the published files uses them: the largest magnitude that the writers use is
448 (`0x7E`).

## q4c: 4-bit codebook

`q4c` stores 4-bit codes, a codebook of sixteen `f32` levels and one scale for
each group. The variant selects the layout of the codes and the scales.

| Variant | Layout | Scale | Group | Seen in |
| ---: | --- | --- | ---: | --- |
| 0 | Row-interleaved | fp8 (E4M3) | 16 | 27B feed-forward layers (NVFP4 values) |
| 1 | Row-interleaved | `f16` | 32 | 27B MTP head and drafter (NF4 codebook) |
| 2 | Planar | `f16` | 32 | Flash-Next; every GGUF import |

Every variant starts with the codebook: sixteen `f32` levels, level `c` at
offset `4c`. The value of a weight is

```text
value(r, k) = scale(r, group of k) × codebook[code(r, k)]
```

Halogen-flash reads only variant 2. It refuses variants 0 and 1.

### Variant 2: planar

| Region | Offset | Size | Contents |
| --- | --- | --- | --- |
| Codebook | 0 | 64 | Sixteen `f32` levels |
| Code plane | 64 | `N × K / 2` | Row `r` starts at `64 + r × K / 2` |
| Padding | `64 + N × K / 2` | to `S` | Zero |
| Scale plane | `S = 64 + pad(N × K / 2, 64)` | `N × R` | Row `r` starts at `S + r × R`, with `R = pad(K / 16, 16)` |

A scale row holds `K / 32` `f16` scales, one for each group of 32, then zero
bytes up to `R`. `K` must be a multiple of 32.

### Variants 0 and 1: row-interleaved

| Region | Offset | Size | Contents |
| --- | --- | --- | --- |
| Codebook | 0 | 64 | Sixteen `f32` levels |
| Rows | 64 | `N × (K / 2 + K / 16)` | Row `r` starts at `64 + r × (K / 2 + K / 16)` |

Each row holds `K / 2` bytes of codes, then `K / 16` bytes of scales:

- Variant 0: `K / 16` fp8 scale bytes, one for each group of 16.
- Variant 1: `K / 32` `f16` scales, one for each group of 32.

In the published file `K` is always a multiple of 256, so every row starts at
a 16-byte boundary. The rule for other `K` values (row padding) is **not
established**.

### Codebooks

The codebook can hold any sixteen finite values in any order. These occur:

| Source | Codebook, levels 0 to 15 | Scale of a group |
| --- | --- | --- |
| Flash-Next checkpoint (variant 2) | Learned for each tensor. Sorted, from about −1 to 1 | About the largest magnitude in the group |
| NF4 (variant 1) | −1, −0.69619, −0.52507, −0.39492, −0.28444, −0.18477, −0.09105, 0, 0.07958, 0.16093, 0.24611, 0.33792, 0.44071, 0.56262, 0.72296, 1 | The largest magnitude in the group |
| NVFP4 (variant 0) | `G ×` (0, 0.5, 1, 1.5, 2, 3, 4, 6, −0, −0.5, −1, −1.5, −2, −3, −4, −6), with a global scale `G` for each tensor | The largest magnitude in the group, divided by 6 |
| GGUF Q4_0 (variant 2) | −8, −7, …, 6, 7 (`c − 8`) | The GGUF block scale |
| GGUF IQ4_NL, IQ4_XS (variant 2) | −127, −104, −83, −65, −49, −35, −22, −10, 1, 13, 25, 38, 53, 69, 89, 113 | The GGUF block scale |
| GGUF IQ3_S (variant 2) | −15, −13, …, 13, 15 (`2c − 15`) | The GGUF block scale |

The NF4 values are given to 5 digits here. The file holds the standard NF4
levels in `f32`; see [Test vectors](TEST-VECTORS.md#q4c-variant-1). The
"scale of a group" column describes how the writers chose the scales. It is
not a reader requirement.

All rows of a tensor share one codebook. A fused expert tensor therefore
shares one codebook across all its experts.

Storage cost: 4.5 bits for each value, plus 64 bytes for each tensor.

## fp8r: 8-bit float with a row scale

The variant is 0. The payload has two planes and no padding:

| Region | Offset | Size | Contents |
| --- | --- | --- | --- |
| Code plane | 0 | `N × K` | One fp8 code for each value. Row `r` starts at `r × K` |
| Scale plane | `N × K` | `2 × N` | One `bf16` scale for each row |

```text
value(r, k) = fp8(code(r, k)) × scale(r)
```

In the published file each row's scale is its largest magnitude divided by
448, and every row uses code `0x7E` or `0xFE`. Storage cost: 8 bits for each
value, plus 2 bytes for each row.

## i4l: rotated 4-bit codes

`i4l` stores a 4-bit copy of a matrix in a rotated basis. The 27B checkpoint
uses it for prompt processing. Each `i4l` tensor is named after the matrix
that it copies, with `.i4l` added. The other copy, in `fp8r` or `q4c`, serves
token generation.

All published `i4l` tensors have the variant `0x00010100` (65792). The rules
below hold for that variant. The group size is 256, which is the low 16 bits
of the variant; the meaning of the other bits is **not established**.

| Region | Offset | Size | Contents |
| --- | --- | --- | --- |
| Code plane | 0 | `N × K / 2` | Signed 4-bit codes, two's complement, in the common nibble order. Row `r` starts at `r × K / 2` |
| Scale plane | `N × K / 2` | `2 × N × K / 256` | `f16` scales. Scale `b` of row `r` is at `N × K / 2 + 2 × (r × K / 256 + b)` |

The payload has no padding. The published codes are in `[−7, 7]`, and each
scale is about the largest magnitude of its group divided by 7.

The stored numbers are rotated. For row `r` and block `b` of 256 positions,
let `u[j] = code(r, 256b + j) × scale(r, b)` for `j` from 0 to 255. Then

```text
value(r, 256b + i) = (1 / 16) × Σ_j u[j] × h(i, j)
h(i, j) = +1 when popcount(i AND j) is even, −1 when it is odd
```

`h` is the 256 × 256 Sylvester Hadamard matrix. Divided by 16 it is
orthonormal and symmetric, so it is its own inverse. A matrix product does
not need the decoded values. Rotate each 256-value block of the activations
the same way, then use `u` directly:

```text
Σ_k value(r, k) × x[k] = Σ_b Σ_j u_b[j] × x'_b[j],   x'_b = (1/16) × h × x_b
```

The rotation spreads large weights across the block, which makes 4-bit codes
more accurate. Storage cost: 4.0625 bits for each value.

`i4l` is not [`i4r`](#i4r-rotated-4-bit-expert-codes). The two formats differ
in the group size, the code encoding and the rotation.

## q8s32: 8-bit symmetric

The variant is 0. `K` must be a multiple of 64. Each row is `R = pad(K + K /
16, 16)` bytes, and row `r` starts at `r × R`.

| Row offset | Size | Contents |
| --- | --- | --- |
| 0 | `K` | Signed 8-bit codes. Value `k` is byte `k` |
| `K` | `K / 16` | `K / 32` `f16` scales, one for each group of 32 |
| `K + K / 16` | to `R` | Zero |

```text
value(r, k) = code(r, k) × scale(r, k / 32)
```

The payload must start at a 16-byte boundary. Storage cost: 8.5 bits for each
value, plus the row padding.

## q8g64: 8-bit affine

The variant is 0. `K` must be a multiple of 64. Each row is `K + K / 16`
bytes with no padding, and row `r` starts at `r × (K + K / 16)`.

| Row offset | Size | Contents |
| --- | --- | --- |
| 0 | `K` | Unsigned 8-bit codes |
| `K` | `K / 16` | `K / 64` records of four bytes: `f16` scale, then `f16` bias |

```text
value(r, k) = code(r, k) × scale(r, k / 64) + bias(r, k / 64)
```

The payload must start at a 16-byte boundary. In the published files every
group of 64 uses both code 0 and code 255. So the bias is the smallest value
of the group, and the scale spans the group's range in 255 steps. Storage cost:
8.5 bits for each value.

## q6g64: 6-bit affine

The variant is 64. `K` must be a multiple of 64. Each row is `S = pad(13 × K
/ 16, 16)` bytes, and row `r` starts at `r × S`.

| Row offset | Size | Contents |
| --- | --- | --- |
| 0 | `K / 2` | Low 4 bits of each code, in the common nibble order |
| `K / 2` | `K / 4` | High 2 bits of each code. Byte `K / 2 + k / 4` holds the 2 bits of value `k` at bits `2 × (k mod 4)` and `2 × (k mod 4) + 1` |
| `3 × K / 4` | `K / 16` | `K / 64` records of four bytes: `f16` scale, then `f16` bias |
| `13 × K / 16` | to `S` | Zero |

```text
c(r, k)     = low(r, k) | (high(r, k) << 4)                 from 0 to 63
value(r, k) = c(r, k) × scale(r, k / 64) + bias(r, k / 64)
```

The payload must start at a 16-byte boundary. A row has padding only when `K`
is not a multiple of 256. For example, when `K` is 320, a row is 272 bytes
and holds 260 bytes of data. Halogen-flash does not read the padding bytes.

In the sampled groups of the published file, every group of 64 uses both
code 0 and code 63, as for `q8g64`. Storage cost: 6.5 bits for each value,
plus the row padding.

## q6k: GGUF Q6_K

The variant is 0. `K` must be a multiple of 256. The payload is the GGUF Q6_K
data of the tensor, unchanged: `K / 256` super-blocks of 210 bytes for each
row. Super-block `s` of row `r` starts at `210 × (r × K / 256 + s)`.

| Block offset | Size | Contents |
| --- | --- | --- |
| 0 | 128 | `ql`: the low 4 bits of each code |
| 128 | 64 | `qh`: the high 2 bits of each code |
| 192 | 16 | Sixteen signed 8-bit scales, one for each 16 values |
| 208 | 2 | `f16` super-block scale `d` |

For value `i` in `[0, 256)` of a super-block:

```text
h  = i / 128            half of the super-block
q  = (i mod 128) / 32   quarter of the half
l  = i mod 32
lo = (ql[64h + 32 × (q mod 2) + l] >> (4 × (q / 2))) & 15
hi = (qh[32h + l] >> (2q)) & 3
value = d × scale[i / 16] × ((lo | (hi << 4)) − 32)
```

This is the standard GGUF definition. Storage cost: 6.5625 bits for each
value.

## iq4nl: GGUF IQ4_NL table

This store is used only for the Flash-Next n-gram table. The variant is 0. The
shape is `[128, R, 160]`, so the table has `128 × R` rows of 160 values. The
payload is the GGUF IQ4_NL data, unchanged: 5 blocks of 18 bytes for each
row, 90 bytes for each row. Row `r` starts at `90r`.

| Block offset | Size | Contents |
| --- | --- | --- |
| 0 | 2 | `f16` scale `d` |
| 2 | 16 | Byte `j` holds the code of value `j` (bits 0–3) and of value `j + 16` (bits 4–7) |

```text
value = d × L[code]
L = −127, −104, −83, −65, −49, −35, −22, −10, 1, 13, 25, 38, 53, 69, 89, 113
```

This store keeps the GGUF nibble order. It is not the `q4c` order.

## fp8g: fp8 table with a global scale

This store is used only for the Flash-Next n-gram table. The variant is 0.
The shape is `[128, R, 160]`, so the table has `rows = 128 × R` rows of 160
values.

| Region | Offset | Size | Contents |
| --- | --- | --- | --- |
| Codes | 0 | `160 × rows` | One fp8 code for each value. Row `r` starts at `160r` |
| Padding | `160 × rows` | to `P` | Zero |
| Scale | `P = pad(160 × rows, 64)` | 4 | One `f32` scale for the whole table |

```text
value = fp8(code) × scale
```

The published table has `R = 2500012`: 320,001,536 rows, 51,200,245,764 bytes
and a scale of about 1.99 × 10⁻⁴. Other uses of `fp8g` are **not
established**.

## q4a and q5a: affine codes

These two stores hold 4-bit (`q4a`) or 5-bit (`q5a`) unsigned codes with an
affine decode rule. The variant selects how the scale and the zero are stored.
Only the GGUF importer writes them.

| Variant | Name | Per group of 32 | Per super-block | `K` rule | GGUF source |
| ---: | --- | --- | --- | --- | --- |
| 0 | pair | `f16` scale, `f16` zero | — | multiple of 32 | Q5_1 or Q5_0 (as `q5a`) |
| 1 | k-scale | `u8` scale factor `sc`, `u8` offset factor `m` | `f16` `d`, `f16` `dmin` | multiple of 256 | Q4_K (as `q4a`), Q5_K (as `q5a`) |

The variant must be 0 or 1. A reader must refuse other values.

### Planes

Let `B = pad(N × K / 2, 64)`.

| Plane | Start | Row stride | Row contents |
| --- | --- | --- | --- |
| Code plane | 0 | `K / 2` | 4-bit codes in the common nibble order |
| High-bit plane (`q5a` only) | `H = B` | `K / 8` | `K / 32` `u32` words. Bit `b` of word `g` is bit 4 of the code of value `32g + b` |
| Plane 1 | `P1` | `S1` | `K / 32` group records |
| Plane 2 (variant 1 only) | `P2 = pad(P1 + N × S1, 64)` | `S2 = pad(4 × K / 256, 16)` | `K / 256` records: `f16` `d`, then `f16` `dmin` |

`P1` is `B` for `q4a` and `B + pad(N × K / 8, 64)` for `q5a`.

| Variant | Plane 1 record | `S1` |
| ---: | --- | --- |
| 0 | `f16` scale, then `f16` zero (4 bytes) | `pad(4 × K / 32, 16)` |
| 1 | `u8` `sc`, then `u8` `m` (2 bytes) | `pad(2 × K / 32, 16)` |

The payload size is `P1 + N × S1` for variant 0 and `P2 + N × S2` for
variant 1. The bytes between the end of one plane and the start of the next
are zero.

### Decode rule

```text
c(r, k)     = code-plane nibble of (r, k)                       for q4a
c(r, k)     = code-plane nibble + 16 × high-plane bit of (r, k)  for q5a

variant 0:  value = c × scale(r, k/32) + zero(r, k/32)

variant 1:  scale = d(r, k/256) × sc(r, k/32)
            zero  = −(dmin(r, k/256) × m(r, k/32))
            value = c × scale + zero
```

`c` is 0 to 15 for `q4a` and 0 to 31 for `q5a`. `sc` and `m` are 0 to 63.

Requirements:

- `K` is a multiple of 32 for variant 0 and of 256 for variant 1.
- The payload size is equal to the size rule.
- The payload starts at a 16-byte boundary.

The numbers in these payloads are the numbers of the GGUF blocks, moved into
planes. A reader can rebuild the GGUF blocks without loss (see
[Reverse conversion](GGUF-IMPORT.md#reverse-conversion)).

| Store | Variant | Bits for each value |
| --- | ---: | ---: |
| `q4a` | 1 | 4.625 |
| `q5a` | 0 | 6.0 |
| `q5a` | 1 | 5.625 |

## The 128-point rotation

`ht` and `i4r` hold a matrix in a rotated basis. This section gives the
rotation that both stores use.

Let `M[r][c]` be the stored number at row `r` and column `c`. Each store
defines `M`. Let `sv` be the output vector, with `N` values, and `su` the
input vector, with `K` values. Let `h(i, j)` be the Sylvester Hadamard
function of [i4l](#i4l-rotated-4-bit-codes), for `i` and `j` from 0 to 127.
Then

```text
value(r, c) = sv[r] × su[c] × (1/128) × Σ_i Σ_j h(r mod 128, i)
              × M[128 × (r / 128) + i][128 × (c / 128) + j] × h(j, c mod 128)
```

The sums run over `i` and `j` from 0 to 127. Each block of 128 × 128 stored
numbers gives one block of 128 × 128 values. For one block, in matrix form:

```text
W = diag(sv) × Hn × M × Hn × diag(su),    Hn = h / √128
```

`Hn` is orthonormal and symmetric, so it is its own inverse. A matrix product
does not need the decoded values. To compute `y = W × x`:

1. Multiply each value of `x` by `su`.
2. Rotate each block of 128 values with `Hn`.
3. Multiply `M` by the result.
4. Rotate each block of 128 values of the result with `Hn`.
5. Multiply each value by `sv`.

### Rotation vectors

`su` and `sv` are separate tensors of store `f16` (code 2). The element count
of each vector must equal the count that its matrix needs. Its payload size
must be 2 × the count. Halogen-flash does not check the rank, the shape or the
variant of a vector. Writers should write variant 0.

For a 2-D matrix tensor named `BASE.weight`, the vectors are:

| Tensor | Values | Vector |
| --- | --- | --- |
| `BASE.suh` | `K` | Input vector `su` |
| `BASE.svh` | `N` | Output vector `sv` |

For example, `layers.0.linear_attn.in_proj_qkv.weight` has the vectors
`layers.0.linear_attn.in_proj_qkv.suh` and
`layers.0.linear_attn.in_proj_qkv.svh`. The fused experts use four vectors
for each layer. Their names are in [ht](#ht-experts) and
[i4r](#i4r-rotated-4-bit-expert-codes), and one more
exception is in [Qwen3.8 Flash-Next tensors](TENSORS.md#rotation-vectors).

In the sampled vectors of the published files, every input vector holds only
+1 and −1. An `ht` output vector holds a sign and a scale for each row. An
`i4r` output vector holds only +1 and −1. These are choices of the writer, not
reader requirements.

## ht: trellis codes in a rotated basis

`ht` keeps the stored numbers `M` of [the 128-point
rotation](#the-128-point-rotation) as a trellis code. The code has `b` bits
for each value, with `b` equal to 3 or 4. Each value is a function of a
16-bit state, which the bitstream gives. `ht` has no scales. The rotation
vectors hold the magnitudes.

### Variant

| Bits | Field | Required value |
| --- | --- | --- |
| 0–7 | Two times the code width `b` | 6 (`b` = 3) or 8 (`b` = 4) |
| 8–11 | Tile layout | 2 |
| 12–15 | Codebook | 1 |
| 16–31 | — | 0 |

So the variant must be `0x1206` or `0x1208`. Other values of the layout field
and of the codebook field are **not established**. A reader must refuse them.

### Shape and size

The shape must be one of these:

- 2-D, with `N` and `K` multiples of 128, from 128 to 2²⁴.
- 3-D, exactly `[512, 1280, 2560]` or `[512, 2560, 640]`. These are the fused
  experts.

The payload size is exactly `N × K × b / 8`. The payload has no padding.

### Tiles

A tile holds 16 × 16 values in `32 × b` bytes, as `8 × b` little-endian `u32`
words. A tile is 96 bytes when `b` is 3, and 128 bytes when `b` is 4.

Value `i` of a tile, from 0 to 255, is at:

```text
tile row    = (i / 8) mod 16
tile column = 8 × (i / 128) + (i mod 8)
```

So a tile is two halves of 16 rows and 8 columns. Each half is in row-major
order.

In a 2-D tensor, rows `128B` to `128B + 127` are band `B`. A band holds its
column tiles in order. A column tile holds its 8 row tiles in order. Value
`(r, c)` is in tile `t`, which starts at byte `32 × b × t`:

```text
t = ((r / 128) × (K / 16) + c / 16) × 8 + (r mod 128) / 16
```

In the tile, the value is at tile row `r mod 16` and tile column `c mod 16`.

### Trellis decode

Read a tile as one bitstream of `L = 256 × b` bits. Bit `n` of the stream is
bit `31 − (n mod 32)` of word `n / 32`. So the stream reads each word from its
most significant bit. The stream is circular: bit `n + L` is bit `n`.

The state of value `i` is a 16-bit number. Its bits are stream bits
`b × (i + 1) − 16` to `b × (i + 1) − 1`, in that order. The first of these
bits is the most significant bit of the state. Each value adds `b` new bits to the state.
The state of value 0 takes the last `16 − b` bits of the stream, then the
first `b` bits.

### Codebook

Codebook 1 is a fixed function of the 16-bit state `s`:

1. `x = (s × 0x83DCD12D) mod 2³²`.
2. `a` = the sum of the four bytes of `x`. `a` is from 0 to 1020.
3. `T = (1774 × a − 905216) / 2¹⁸`, rounded to `f16` to nearest with ties to
   even.

`T` is the stored number `M` at the position of the value. An implementation
can compute step 3 as one fused multiply-add in `f16`: `(1024 + a) × 0x1EEE +
0xC931`, where the two constants are `f16` bit patterns. The codebook has 913
different values, from −3.453125 to 3.34765625. Over all 65,536 states, the
mean is about 0 and the standard deviation is about 1.

### ht experts

For the fused experts, the data of expert `e` is one contiguous block of
`shape[1] × K × b / 8` bytes. The block of expert `e` starts at byte `e ×
shape[1] × K × b / 8`. Row `r` below is a row of the expert, from 0 to
`shape[1] − 1`.

- `experts.down_proj`, `[512, 2560, 640]`: the 2-D rule applies in the block,
  with `N = 2560` and `K = 640`.
- `experts.gate_up_proj`, `[512, 1280, 2560]`: rows 0 to 639 are the gate
  projection and rows 640 to 1279 are the up projection (see
  [Fused experts](TENSORS.md#fused-experts)). A band holds 128 gate rows and
  the 128 up rows at the same position. Value `(r, c)` is in tile `t`:

  ```text
  p = (r mod 640) / 128
  j = (r mod 128) / 16 + 8 × (r / 640)
  t = (p × (K / 16) + c / 16) × 16 + j
  ```

  In the tile, the value is at tile row `r mod 16` and tile column
  `c mod 16`.

Each expert is a separate matrix for the rotation. The experts of a layer use
four rotation vectors. Prefix the names with `layers.L.mlp.`:

| Tensor | Values | Vector |
| --- | ---: | --- |
| `experts.su` | 2,560 | `su` of gate and up, the same for all experts |
| `experts.gate_up_proj.svh` | 512 × 1,280 | `sv` of expert `e`: values `1280e` to `1280e + 1279` |
| `experts.down_proj.suh` | 512 × 640 | `su` of expert `e`: values `640e` to `640e + 639` |
| `experts.down_proj.svh` | 512 × 2,560 | `sv` of expert `e`: values `2560e` to `2560e + 2559` |

The tensor `experts.gate_up_proj.suh` does not exist.

Storage cost: `b` bits for each value, plus the rotation vectors.

## i4r: rotated 4-bit expert codes

`i4r` holds the fused routed experts in the basis of [the 128-point
rotation](#the-128-point-rotation). Only the two fused expert tensors use it.

The variant is 128, the group size. The shape must be `[512, 1280, 2560]` or
`[512, 2560, 640]`. `N = 512 × shape[1]` and `K = shape[2]`. Row `R` of the
tensor is row `R − e × shape[1]` of expert `e`, as for every fused expert
tensor.

| Region | Offset | Size | Contents |
| --- | --- | --- | --- |
| Code plane | 0 | `N × K / 2` | 4-bit codes in the common nibble order. Row `R` starts at `R × K / 2` |
| Scale plane | `N × K / 2` | `2 × N × K / 128` | `f16` scales. Scale `g` of row `R` is at `N × K / 2 + 2 × (R × K / 128 + g)` |

The payload has no padding. It must start at a 64-byte boundary.

The code is offset binary: `c = nibble − 8`, from −8 to 7. The stored number
is exact:

```text
M[R][k] = c(R, k) × scale(R, k / 128)
```

Each expert is a separate matrix for the rotation. Expert `e` uses rows
`e × shape[1]` to `e × shape[1] + shape[1] − 1` of `M`. All 512 experts of a
layer use the same four rotation vectors. Prefix the names with
`layers.L.mlp.`:

| Tensor | Values | Vector |
| --- | ---: | --- |
| `experts.su` | 2,560 | `su` of gate and up |
| `experts.gate_up_proj.svh` | 1,280 | `sv` of gate and up |
| `experts.down_proj.suh` | 640 | `su` of down |
| `experts.down_proj.svh` | 2,560 | `sv` of down |

The names are the names of the `ht` expert vectors. The counts are not the
same: an `ht` expert has its own vectors, but `i4r` experts share them.

`i4r` and `i4l` are different formats:

| | `i4l` | `i4r` |
| --- | --- | --- |
| Group | 256 | 128 |
| Code | Two's complement, −8 to 7 | Offset binary, −8 to 7 |
| Rotation | Input side only, 256-point, `1/16` | Both sides, 128-point, `1/√128` on each side |
| Rotation vectors | None | Four for each layer |

The published codes use the full range from −8 to 7. Storage cost: 4.125
bits for each value.

## q8_0, q5_0 and q5_1: GGUF table blocks

These three stores are used only for the Flash-Next n-gram table. The variant
is 0. The shape is `[128, R, 160]`, so the table has `rows = 128 × R` rows of
160 values. The payload is the GGUF data, unchanged: five blocks of 32 values
for each row. Row `t` starts at `t` × the row size. Value `v` of a row is
value `v mod 32` of block `v / 32`.

| Store | Block size | Row size | Block contents | Value |
| --- | ---: | ---: | --- | --- |
| `q8_0` | 34 | 170 | `f16` `d` at 0; 32 signed 8-bit codes at 2 | `code × d` |
| `q5_0` | 22 | 110 | `f16` `d` at 0; `u32` `qh` at 2; 16 code bytes at 6 | `d × (code − 16)` |
| `q5_1` | 24 | 120 | `f16` `d` at 0; `f16` `m` at 2; `u32` `qh` at 4; 16 code bytes at 8 | `code × d + m` |

The 5-bit code of value `j` of a `q5_0` or `q5_1` block:

- For `j` from 0 to 15, bits 0–3 are bits 0–3 of code byte `j`.
- For `j` from 16 to 31, bits 0–3 are bits 4–7 of code byte `j − 16`.
- Bit 4 is bit `j` of `qh`.

These are the standard GGUF definitions. The stores keep the GGUF nibble
order.

## Alignment summary

| Store | Payload alignment that halogen-flash checks | Alignment that writers provide |
| --- | --- | --- |
| `q8s32`, `q8g64`, `q6g64`, `q4a`, `q5a` | 16 bytes | 64 bytes |
| `i4r` | 64 bytes | 64 bytes |
| All others | none | 64 bytes |

A reader should check 16-byte alignment for all quantized stores, and 64-byte
alignment for `i4r`. The 64-byte payload offset makes this true for every
file that follows the [writer procedure](CONTAINER.md#writer-procedure).
