# Test vectors

These vectors come from a reference implementation that was written from this
specification. Each conversion vector was also checked against the
dequantizer of the `gguf` Python package (version 0.19.0). So were the vectors
of the GGUF table blocks. The `ht` tile vectors were also checked against the
decode arithmetic of halogen-flash. The rotation vector was also
checked with an independent matrix product. Hexadecimal byte strings list
bytes in file order, 32 bytes to a line. Decimal values are exact: each is
the decimal form of an `f32` value.

## A complete file

A file with three tensors. Its SHA-256 is
`f88df782c300afa52494f572bd9425d6bc7a7296d378bef5fc5e582181c2c54e`.

| Name | Store | Shape | Offset | Size | Checksum | Variant |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `example.bf16.weight` | 0 | `[2, 32]` | 640 | 128 | `0x01600160` | 0 |
| `example.q4c.weight` | 5 | `[1, 64]` | 768 | 144 | `0xF5003800` | 2 |
| `example.i64` | 4 | `[3]` | 960 | 24 | `0x00000001` | 0 |

- The header says version 2, 3 tensors, table offset `0x68`, data offset
  `0x280` (640) and file size 1024. The identity is `example`.
- The table ends at `0x68 + 3 × 0xA0 = 584`. The payload start is
  `pad(584, 64) = 640`.
- `example.bf16.weight` holds 64 evenly spaced values from −1 to 1, rounded to
  `bf16`.
- `example.q4c.weight` is the `q4c` payload of the next section.
- `example.i64` holds 3, 5 and 7.
- The `q4c` payload is 144 bytes and is padded to 192, so `example.i64`
  starts at 960.

```text
00000000  48 47 4e 31 02 00 00 00 03 00 00 00 00 00 00 00
00000010  68 00 00 00 00 00 00 00 80 02 00 00 00 00 00 00
00000020  00 04 00 00 00 00 00 00 65 78 61 6d 70 6c 65 00
00000030  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000040  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000050  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000060  00 00 00 00 00 00 00 00 65 78 61 6d 70 6c 65 2e
00000070  62 66 31 36 2e 77 65 69 67 68 74 00 00 00 00 00
00000080  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000090  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000000a0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000000b0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000000c0  00 00 00 00 00 00 00 00 00 00 00 00 02 00 00 00
000000d0  02 00 00 00 00 00 00 00 20 00 00 00 00 00 00 00
000000e0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000000f0  80 02 00 00 00 00 00 00 80 00 00 00 00 00 00 00
00000100  60 01 60 01 00 00 00 00 65 78 61 6d 70 6c 65 2e
00000110  71 34 63 2e 77 65 69 67 68 74 00 00 00 00 00 00
00000120  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000130  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000140  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000150  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000160  00 00 00 00 00 00 00 00 05 00 00 00 02 00 00 00
00000170  01 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00
00000180  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000190  00 03 00 00 00 00 00 00 90 00 00 00 00 00 00 00
000001a0  00 38 00 f5 02 00 00 00 65 78 61 6d 70 6c 65 2e
000001b0  69 36 34 00 00 00 00 00 00 00 00 00 00 00 00 00
000001c0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000001d0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000001e0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000001f0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000200  00 00 00 00 00 00 00 00 04 00 00 00 01 00 00 00
00000210  03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000220  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000230  c0 03 00 00 00 00 00 00 18 00 00 00 00 00 00 00
00000240  01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000250  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000260  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000270  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000280  80 bf 78 bf 70 bf 68 bf 5f bf 57 bf 4f bf 47 bf
00000290  3f bf 37 bf 2f bf 27 bf 1e bf 16 bf 0e bf 06 bf
000002a0  fc be ec be db be cb be bb be ab be 9a be 8a be
000002b0  74 be 53 be 33 be 12 be e4 bd a3 bd 43 bd 82 bc
000002c0  82 3c 43 3d a3 3d e4 3d 12 3e 33 3e 53 3e 74 3e
000002d0  8a 3e 9a 3e ab 3e bb 3e cb 3e db 3e ec 3e fc 3e
000002e0  06 3f 0e 3f 16 3f 1e 3f 27 3f 2f 3f 37 3f 3f 3f
000002f0  47 3f 4f 3f 57 3f 5f 3f 68 3f 70 3f 78 3f 80 3f
00000300  00 00 00 c1 00 00 e0 c0 00 00 c0 c0 00 00 a0 c0
00000310  00 00 80 c0 00 00 40 c0 00 00 00 c0 00 00 80 bf
00000320  00 00 00 00 00 00 80 3f 00 00 00 40 00 00 40 40
00000330  00 00 80 40 00 00 a0 40 00 00 c0 40 00 00 e0 40
00000340  83 2d c7 61 0b a5 4f e9 83 2d c7 61 0b a5 4f e9
00000350  83 2d c7 61 0b a5 4f e9 83 2d c7 61 0b a5 4f e9
00000360  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000370  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00000380  00 38 00 b4 00 00 00 00 00 00 00 00 00 00 00 00
00000390  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000003a0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000003b0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000003c0  03 00 00 00 00 00 00 00 05 00 00 00 00 00 00 00
000003d0  07 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000003e0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
000003f0  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

## Checksum

The payload `01 02 03 04 05 06 07 08 09 0a` has the words `0x04030201`,
`0x08070605` and `0x00000A09`. The checksum is `0x0C040E0D`.

## bf16 rounding

| `f32` value | `bf16` bits |
| --- | --- |
| 1.0 | `0x3F80` |
| 1.00390625 | `0x3F80` |
| 1.005859375 | `0x3F81` |
| 1.009765625 | `0x3F81` |
| -3.1415927410125732 | `0xC049` |
| 65504.0 | `0x4780` |

1.00390625 is halfway between two `bf16` values; the tie goes to the even
result `0x3F80`.

## fp8 codes

| Code | Value |
| --- | ---: |
| `0x00` | 0.0 |
| `0x01` | 0.001953125 |
| `0x07` | 0.013671875 |
| `0x08` | 0.015625 |
| `0x38` | 1.0 |
| `0x3F` | 1.875 |
| `0x40` | 2.0 |
| `0x77` | 240.0 |
| `0x7E` | 448.0 |
| `0x7F` | 480.0 |
| `0x80` | -0.0 |
| `0xB8` | -1.0 |
| `0xFF` | -480.0 |

## q4c variant 2 from Q4_0

One row of 64 values: two GGUF Q4_0 blocks with scales 0.5 and −0.25. The code
of value `i` is `(5i + 3) mod 16`.

GGUF row (36 bytes):

```text
00383388dd2277cc1166bb0055aaff4499ee00b43388dd2277cc1166bb0055aa
ff4499ee
```

HGN `q4c` payload, shape `[1, 64]`, 144 bytes: codebook at 0, codes at 64,
scale row at 128 (16 bytes, two scales and 12 zero bytes):

```text
000000c10000e0c00000c0c00000a0c0000080c0000040c0000000c0000080bf
000000000000803f0000004000004040000080400000a0400000c0400000e040
832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9
0000000000000000000000000000000000000000000000000000000000000000
003800b4000000000000000000000000
```

| Position | Value |
| ---: | ---: |
| 0 | -2.5 |
| 1 | 0.0 |
| 2 | 2.5 |
| 3 | -3.0 |
| 4 | -0.5 |
| 5 | 2.0 |
| 6 | -3.5 |
| 7 | -1.0 |

| Position | Value |
| ---: | ---: |
| 32 | 1.25 |
| 33 | -0.0 |
| 34 | -1.25 |
| 35 | 1.5 |
| 36 | 0.25 |
| 37 | -1.0 |
| 38 | 1.75 |
| 39 | 0.5 |

## IQ4_XS group scale

A super-block scale `d = 0.012298583984375` (`f16` bits `0x224C`) and a
6-bit sub-block scale `ls = 45` give the exact product `(45 − 32) × d =
0.159881591796875`. The `q4c` group scale is the `f16` value `0x311E`,
which is 0.159912109375. The relative change is 1.9 × 10⁻⁴.

## q4c variant 1

One row of 32 values with the NF4 codebook and one `f16` scale of 0.03125. The
code of value `i` is `5i mod 16`. Payload (64 + 16 + 2 bytes):

```text
000080bfb13932bf306b06bfa032cabe4da291be3f353dbe7178babd00000000
fffaa23de3ca243edd047c3e3a03ad3eb8a4e13eab07103fb313393f0000803f
50fa943ed8721cb650fa943ed8721cb60028
```

| Position | Value |
| ---: | ---: |
| 0 | -0.03125 |
| 1 | -0.005774169694632292 |
| 2 | 0.007691009435802698 |
| 3 | 0.03125 |
| 4 | -0.008888793177902699 |
| 5 | 0.005029068794101477 |
| 6 | 0.02259240113198757 |
| 7 | -0.012341171503067017 |

## q4c variant 0

One row of 32 values. The codebook is the E2M1 levels times 0.5. The scale
bytes are `0x38` (1.0) for values 0–15 and `0x40` (2.0) for values 16–31. The
code of value `i` is `5i mod 16`. Payload (64 + 16 + 2 bytes):

```text
000000000000803e0000003f0000403f0000803f0000c03f0000004000004040
00000000000080be000000bf000040bf000080bf0000c0bf000000c0000040c0
50fa943ed8721cb650fa943ed8721cb63840
```

| Position | Value |
| ---: | ---: |
| 0 | 0.0 |
| 1 | 1.5 |
| 2 | -0.5 |
| 3 | -3.0 |

| Position | Value |
| ---: | ---: |
| 16 | 0.0 |
| 17 | 3.0 |
| 18 | -1.0 |
| 19 | -6.0 |

## q8s32 from Q8_0

One row of 64 values: two GGUF Q8_0 blocks with scales 0.01 and 0.02 (as
`f16`). The code of value `i` is `((37i) mod 255) − 127`.

GGUF row (68 bytes):

```text
1f2181a6cbf0153a5f85aacff4193e6389aed3f81d42678db2d7fc21466b91b6
db001f25254a6f95badf04294e7399bee3082d52779dc2e70c31567ba1c6eb10
355a7fa5
```

HGN row (80 bytes: 64 codes, 2 scales, 12 zero bytes):

```text
81a6cbf0153a5f85aacff4193e6389aed3f81d42678db2d7fc21466b91b6db00
254a6f95badf04294e7399bee3082d52779dc2e70c31567ba1c6eb10355a7fa5
1f211f25000000000000000000000000
```

| Position | Value |
| ---: | ---: |
| 0 | -1.2702713012695312 |
| 1 | -0.9001922607421875 |
| 2 | -0.5301132202148438 |
| 3 | -0.1600341796875 |

| Position | Value |
| ---: | ---: |
| 32 | 0.7401580810546875 |
| 33 | 1.480316162109375 |
| 34 | 2.2204742431640625 |
| 35 | -2.1404571533203125 |

## q5a variant 0 from Q5_1

One GGUF Q5_1 block, `d = 0.125`, `m = −1.5`. The code of value `i` is
`(11i + 1) mod 32`.

GGUF block (24 bytes):

```text
003000be2469db9611cc7722dd8833ee9944ffaa5500bb66
```

HGN payload, shape `[1, 32]`, 144 bytes: code plane at 0, high-bit plane at
64, plane 1 at 128:

```text
c1278de349af056bc1278de349af056b00000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000
2469db9600000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000
003000be000000000000000000000000
```

| Position | Value |
| ---: | ---: |
| 0 | -1.375 |
| 1 | 0.0 |
| 2 | 1.375 |
| 3 | -1.25 |
| 4 | 0.125 |
| 5 | 1.5 |
| 6 | -1.125 |
| 7 | 0.25 |

## q5a variant 0 from Q5_0

One GGUF Q5_0 block, `d = 0.125`. The code of value `i` is `(11i + 1) mod 32`,
as in the Q5_1 vector above. The conversion is in
[GGUF import](GGUF-IMPORT.md#type-rules).

GGUF block (22 bytes):

```text
00302469db9611cc7722dd8833ee9944ffaa5500bb66
```

HGN payload, shape `[1, 32]`, 144 bytes. It is equal to the Q5_1 payload
above, except for the plane-1 record. The record holds `d`, then the zero
`−16 × d = −2.0` (`f16` bits `0xC000`):

```text
c1278de349af056bc1278de349af056b00000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000
2469db9600000000000000000000000000000000000000000000000000000000
0000000000000000000000000000000000000000000000000000000000000000
003000c0000000000000000000000000
```

| Position | Value |
| ---: | ---: |
| 0 | -1.875 |
| 1 | -0.5 |
| 2 | 0.875 |
| 3 | -1.75 |
| 4 | -0.375 |
| 5 | 1.0 |
| 6 | -1.625 |
| 7 | -0.25 |

The zero must be a finite `f16`. These scales show the limit:

| `d` bits | `d` | Zero bits | Result |
| --- | ---: | --- | --- |
| `0x6BFF` | 4094.0 | `0xFBFF` (−65504.0) | Accepted |
| `0x6C00` | 4096.0 | — | Refused |
| `0x7C00` | infinity | — | Refused |
| `0x0001` | 5.960464477539063e-08 | `0x8010` | Accepted |
| `0x0000` | 0.0 | `0x8000` (−0.0) | Accepted |
| `0x8000` | -0.0 | `0x0000` (+0.0) | Accepted |

## GGUF table blocks

The stores `q8_0`, `q5_0` and `q5_1` keep the GGUF blocks of the n-gram table
unchanged. A table row holds five blocks. These blocks are the first block of
a row:

| Store | Block | Values 0–3 |
| --- | --- | --- |
| `q8_0` | The first block of the [Q8_0 vector](#q8s32-from-q8_0): `1f2181a6…b6db00`, 34 bytes | -1.2702713012695312, -0.9001922607421875, -0.5301132202148438, -0.1600341796875 |
| `q5_0` | The block of the [Q5_0 vector](#q5a-variant-0-from-q5_0), 22 bytes | -1.875, -0.5, 0.875, -1.75 |
| `q5_1` | The block of the [Q5_1 vector](#q5a-variant-0-from-q5_1), 24 bytes | -1.375, 0.0, 1.375, -1.25 |

Each value is equal to the value of the `gguf` package for the same block.

## K-quant scale unpacking

The 12 packed bytes `8142c304458607c8192a3b4c` unpack to these pairs:

| Sub-block | `sc` | `m` |
| ---: | ---: | ---: |
| 0 | 1 | 5 |
| 1 | 2 | 6 |
| 2 | 3 | 7 |
| 3 | 4 | 8 |
| 4 | 41 | 17 |
| 5 | 26 | 34 |
| 6 | 59 | 3 |
| 7 | 12 | 52 |

As plane-1 bytes of a `q4a` or `q5a` variant 1 row:
`010502060307040829111a223b030c34`.

## Affine out_proj to bf16

An affine `out_proj` becomes `bf16` (see [GGUF import](GGUF-IMPORT.md#type-rules)).
This group of a Q5_1 block has `d = 0.0999755859375` (`f16` bits `0x2E66`)
and `m = −1.0` (`0xBC00`). The `f32` value is `code × d + m` with one
rounding. Then the value is rounded to `bf16`, to nearest with ties to even.

| Code | `f32` value | `bf16` bits |
| ---: | ---: | --- |
| 0 | -1.0 | `0xBF80` |
| 1 | -0.9000244140625 | `0xBF66` |
| 2 | -0.800048828125 | `0xBF4D` |
| 16 | 0.599609375 | `0x3F1A` |
| 31 | 2.0992431640625 | `0x4006` |

These `f32` values are exact, so the `bf16` rounding is the only rounding.
0.599609375 is halfway between `0x3F19` and `0x3F1A`; the tie goes to the even
result `0x3F1A`.

## fp8r

Two rows of 8 values. The row scales are the `bf16` values `0x3A00`
(4.8828125 × 10⁻⁴) and `0x3B80` (3.90625 × 10⁻³). Payload (16 codes and 2
scales):

```text
38b87e000140c4307e3c44bc08885010003a803b
```

| Position | Row 0 | Row 1 |
| ---: | ---: | ---: |
| 0 | 0.00048828125 | 1.75 |
| 1 | -0.00048828125 | 0.005859375 |
| 2 | 0.21875 | 0.01171875 |
| 3 | 0.0 | -0.005859375 |
| 4 | 9.5367431640625e-07 | 6.103515625e-05 |
| 5 | 0.0009765625 | -6.103515625e-05 |
| 6 | -0.00146484375 | 0.03125 |
| 7 | 0.000244140625 | 0.0001220703125 |

## i4l

One row of 256 values, variant `0x10100`. The code of value `j` is
`((7j) mod 15) − 7`, and the one `f16` scale is 0.0078125 (bits `0x2000`). The
payload is 128 code bytes and 2 scale bytes. Its first 16 bytes are
`09f7e6d5c4b3a291706f5e4d3c2b1a09`; its SHA-256 is
`6c850365446fcba809e1cdcb42f16218e78b01f31e3dff95eb41f3831d08123d`.

| Position | Stored `u` | Value after the rotation |
| ---: | ---: | ---: |
| 0 | -0.0546875 | -0.00341796875 |
| 1 | 0.0 | 0.02392578125 |
| 2 | 0.0546875 | -0.00341796875 |
| 3 | -0.0078125 | -0.00732421875 |
| 4 | 0.046875 | 0.00048828125 |
| 5 | -0.015625 | -0.00732421875 |
| 6 | 0.0390625 | -0.00732421875 |
| 7 | -0.0234375 | -0.00732421875 |

The SHA-256 of all 256 values as
little-endian `f32`, each rounded from the exact value to nearest, is
`5aaa41202e89b89ceb981fa54735eb0496af0445d28c9523fe76544fc34d30b2`.

## q6g64

One row of 64 values (`K = 64`). The code of value `k` is `(37k + 11) mod
64`. The record holds the scale 0.03125 (`f16` bits `0x2800`) and the bias
−1.0 (`0xBC00`). The row is `pad(52, 16) = 64` bytes: 32 bytes of low bits,
16 bytes of high bits, one record and 12 zero bytes.

```text
0ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc761
dc21b7d82176c82d768b1d728bdc6287002800bc000000000000000000000000
```

| Position | Code | Value |
| ---: | ---: | ---: |
| 0 | 11 | -0.65625 |
| 1 | 48 | 0.5 |
| 2 | 21 | -0.34375 |
| 3 | 58 | 0.8125 |
| 4 | 31 | -0.03125 |
| 5 | 4 | -0.875 |
| 6 | 41 | 0.28125 |
| 7 | 14 | -0.5625 |

Code 0 is at position 17, and code 63 is at position 36.

## ht codebook

The codebook value `T` for some 16-bit states:

| State | `a` | `T` | `f16` bits |
| --- | ---: | ---: | --- |
| `0x0000` | 0 | -3.453125 | `0xC2E8` |
| `0x0001` | 605 | 0.64111328125 | `0x3921` |
| `0x0002` | 444 | -0.448486328125 | `0xB72D` |
| `0x1234` | 425 | -0.5771484375 | `0xB89E` |
| `0x8000` | 382 | -0.8681640625 | `0xBAF2` |
| `0xABCD` | 316 | -1.314453125 | `0xBD42` |
| `0xFFFF` | 414 | -0.6513671875 | `0xB936` |

Over all 65,536 states, the codebook has 913 different values. The smallest is
−3.453125 and the largest is 3.34765625.

## ht tile

One tile with `b = 4`. Word `j` of the tile, from 0 to 31, is `(0x9E3779B9 ×
(j + 1)) mod 2³²`. Payload (128 bytes):

```text
b979379e72f36e3c2b6da6dae4e6dd789d60151756da4cb50f548453c8cdbbf1
8147f38f3ac12a2ef33a62ccacb4996a652ed1081ea808a7d7214045909b77e3
4915af81028fe61fbb081ebe7482555c2dfc8cfae675c4989feffb36586933d5
11e36a73ca5ca21183d6d9af3c50114ef5c948ecae43808a67bdb7282037efc6
```

| Value `i` | (row, column) | State | `T` | `f16` bits |
| ---: | --- | --- | ---: | --- |
| 0 | (0, 0) | `0x7209` | 1.48046875 | `0x3DEC` |
| 1 | (0, 1) | `0x209E` | 1.5888671875 | `0x3E5B` |
| 2 | (0, 2) | `0x09E3` | -0.32666015625 | `0xB53A` |
| 7 | (0, 7) | `0x79B9` | 0.09967041015625 | `0x2E61` |
| 8 | (1, 0) | `0x9B93` | 0.1064453125 | `0x2ED0` |
| 127 | (15, 7) | `0x9B90` | 1.642578125 | `0x3E92` |
| 128 | (0, 8) | `0xB908` | -1.341796875 | `0xBD5E` |
| 255 | (15, 15) | `0x3720` | 1.568359375 | `0x3E46` |

The state of value 0 takes the last 12 bits of the stream (`0x720`, from the
word `0xC6EF3720`) and the first 4 bits (`0x9`). The SHA-256 of the 256 `f16`
bit patterns, little-endian, in the order of the value index `i`, is
`32ae1812fa6a2dc63150621a31ec916757607f03eb9058d51923ab7b92a78d9b`.

The same words with `b = 3` give a tile of 24 words (96 bytes): the first 96
bytes above.

| Value `i` | (row, column) | State | `T` | `f16` bits |
| ---: | --- | --- | ---: | --- |
| 0 | (0, 0) | `0x4AC4` | -0.1236572265625 | `0xAFEA` |
| 1 | (0, 1) | `0x5627` | 1.7578125 | `0x3F08` |
| 2 | (0, 2) | `0xB13C` | -0.61083984375 | `0xB8E3` |
| 7 | (0, 7) | `0x3779` | -0.1981201171875 | `0xB257` |
| 8 | (1, 0) | `0xBBCD` | -0.1236572265625 | `0xAFEA` |
| 127 | (15, 7) | `0xB4AC` | -0.495849609375 | `0xB7EF` |
| 128 | (0, 8) | `0xA560` | 1.2431640625 | `0x3CF9` |
| 255 | (15, 15) | `0x6958` | 0.74267578125 | `0x39F1` |

The SHA-256 of its 256 `f16` bit patterns, in the same order, is
`415420d5d59d5f2b288f563473b597b207e94852290120291887286c5f00555b`.

## The 128-point rotation

One 2-D `ht` tensor of shape `[128, 128]`, variant `0x1208` (`b = 4`). The
payload is 8,192 bytes. Word `w` of the payload, from 0 to 2047, is
`(0x9E3779B9 × (w + 1)) mod 2³²`, so its first tile is the 4-bit tile above.
The SHA-256 of the payload is
`cfab929290971c665ca08cb3222f626e41bfb1b013837ced3b7178136b8f2ca1`.

The rotation vectors are exact `f16` values:

- `su[c]` is −1 when `(5c) mod 7` is less than 3, else +1.
- `sv[r]` is `(8 + (r mod 8)) / 512`, negated when `r mod 3` is 0. So
  `sv[0..3]` is −0.015625, 0.017578125, 0.01953125 and −0.021484375.

| (row, column) | Value |
| --- | ---: |
| (0, 0) | -0.0005001630634069443 |
| (0, 1) | -0.0008721128106117249 |
| (1, 0) | -0.021587662398815155 |
| (1, 1) | 0.006169011816382408 |
| (64, 64) | -0.002170378342270851 |
| (127, 127) | 0.003455530386418104 |

The SHA-256 of all 16,384 values as little-endian `f32`, in row-major order,
each rounded from the exact value to nearest with ties to even, is
`1a1f0f4bee42296506b55f20ae44b69a6b558ee16bf5c70ec9519fa52cbfc1b2`.

## i4r

One row of 128 values of an `i4r` tensor: one group. The nibble of value `k`
is `(5k + 3) mod 16`. The scale is 0.015625 (`f16` bits `0x2400`, bytes
`0024`). The 64 code bytes of the row:

```text
832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9
832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9
```

| Position | Nibble | Code | Stored number |
| ---: | ---: | ---: | ---: |
| 0 | 3 | -5 | -0.078125 |
| 1 | 8 | 0 | 0.0 |
| 2 | 13 | 5 | 0.078125 |
| 3 | 2 | -6 | -0.09375 |
| 4 | 7 | -1 | -0.015625 |
| 5 | 12 | 4 | 0.0625 |
| 6 | 1 | -7 | -0.109375 |
| 7 | 6 | -2 | -0.03125 |

The code is offset binary: nibble 8 is code 0, and nibble 0 is code −8. A
two's-complement reading gives wrong numbers without an error. The values of
the weights come from the [128-point
rotation](STORAGE.md#the-128-point-rotation) of these stored numbers.

## Head permutation

HGN value head `h` is GGUF value head `t(h)`:

| `h` | `t(h)` | `h` | `t(h)` | `h` | `t(h)` | `h` | `t(h)` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 12 | 4 | 24 | 8 | 36 | 12 |
| 1 | 16 | 13 | 20 | 25 | 24 | 37 | 28 |
| 2 | 32 | 14 | 36 | 26 | 40 | 38 | 44 |
| 3 | 1 | 15 | 5 | 27 | 9 | 39 | 13 |
| 4 | 17 | 16 | 21 | 28 | 25 | 40 | 29 |
| 5 | 33 | 17 | 37 | 29 | 41 | 41 | 45 |
| 6 | 2 | 18 | 6 | 30 | 10 | 42 | 14 |
| 7 | 18 | 19 | 22 | 31 | 26 | 43 | 30 |
| 8 | 34 | 20 | 38 | 32 | 42 | 44 | 46 |
| 9 | 3 | 21 | 7 | 33 | 11 | 45 | 15 |
| 10 | 19 | 22 | 23 | 34 | 27 | 46 | 31 |
| 11 | 35 | 23 | 39 | 35 | 43 | 47 | 47 |

## Tensor sizes

Payload sizes of the large Flash-Next tensors. The sizes marked "published"
occur in a published file.

| Tensor | Store | Shape | Bytes |
| --- | --- | --- | ---: |
| `experts.gate_up_proj` | `q4c` | `[512,1280,2560]` | 943,718,464 (published) |
| `experts.gate_up_proj` | `q4a` variant 1 | `[512,1280,2560]` | 975,175,680 |
| `experts.gate_up_proj` | `q5a` variant 0 | `[512,1280,2560]` | 1,258,291,200 |
| `experts.gate_up_proj` | `q5a` variant 1 | `[512,1280,2560]` | 1,184,890,880 |
| `experts.gate_up_proj` | `ht`, `b = 3` | `[512,1280,2560]` | 629,145,600 (published) |
| `experts.gate_up_proj` | `ht`, `b = 4` | `[512,1280,2560]` | 838,860,800 (published) |
| `experts.gate_up_proj` | `i4r` | `[512,1280,2560]` | 865,075,200 (published) |
| `experts.down_proj` | `q4c` | `[512,2560,640]` | 482,345,024 (published) |
| `experts.down_proj` | `q8s32` | `[512,2560,640]` | 901,775,360 |
| `experts.down_proj` | `q5a` variant 0 | `[512,2560,640]` | 629,145,600 |
| `experts.down_proj` | `ht`, `b = 3` | `[512,2560,640]` | 314,572,800 (published) |
| `experts.down_proj` | `ht`, `b = 4` | `[512,2560,640]` | 419,430,400 (published) |
| `experts.down_proj` | `i4r` | `[512,2560,640]` | 432,537,600 (published) |
| `linear_attn.in_proj_qkv` | `q4c` | `[10240,2560]` | 14,745,664 (published) |
| `linear_attn.in_proj_qkv` | `q8s32` | `[10240,2560]` | 27,852,800 |
| `linear_attn.in_proj_qkv` | `ht`, `b = 4` | `[10240,2560]` | 13,107,200 (published) |
| `linear_attn.out_proj` | `bf16` | `[2560,6144]` | 31,457,280 |
| `self_attn.q_proj` | `q4c` | `[12288,2560]` | 17,694,784 (published) |
| `self_attn.q_proj` | `ht`, `b = 4` | `[12288,2560]` | 15,728,640 (published) |
| `lm_head` | `q4c` | `[248320,2560]` | 357,580,864 (published) |
| `embed_tokens` | `q4c` | `[248320,2560]` | 357,580,864 (published) |
| `lm_head` | `q6k` | `[248320,2560]` | 521,472,000 |
| `lm_head` | `q8s32` | `[248320,2560]` | 675,430,400 |
| `lm_head` | `ht`, `b = 4` | `[248320,2560]` | 317,849,600 (published) |
| `self_attn.o_proj` | `q8g64` | `[2560,6144]` | 16,711,680 (published) |
| `input_mix_weight_down` | `q6g64` | `[320,10240]` | 2,662,400 (published) |
| `input_mix_weight_up` | `q6g64` | `[10240,320]` | 2,785,280 (published) |
| `block_inject_weight` | `q6g64` | `[4,10240]` | 33,280 (published) |
| `ngram_embedding`, `R` = 2,500,012 | `fp8g` | `[128,2500012,160]` | 51,200,245,764 (published) |
| `ngram_embedding`, `R` = 2,500,012 | `iq4nl` | `[128,2500012,160]` | 28,800,138,240 |
| `ngram_embedding`, `R` = 2,500,012 | `q8_0` | `[128,2500012,160]` | 54,400,261,120 |
| `ngram_embedding`, `R` = 2,500,012 | `q5_0` | `[128,2500012,160]` | 35,200,168,960 |
| `ngram_embedding`, `R` = 2,500,012 | `q5_1` | `[128,2500012,160]` | 38,400,184,320 |

`experts.down_proj` as `q5a` variant 0 comes from Q5_1 or Q5_0. A `bf16`
`out_proj` comes from an affine GGUF `out_proj`.

## Published files

Measured from the published repositories at these revisions:

| Repository | Revision |
| --- | --- |
| `peonist-ai/halogen-qwen3.8-flash-next` | `6e1aa1acda75bd8a3bb9043b1545f83187cb6ded` |
| `peonist-ai/halogen-qwen3.8-27b` | `800ec932fc4a83ddd18b83d293cfd147946e77f9` |



| File | Bytes | Entries | Identity | Payload start |
| --- | ---: | ---: | --- | ---: |
| `qwen38-flash-next-w4b.hgn` | 124,068,083,904 | 1,198 | `qwen3.8-flash-next` | 191,808 |
| `qwen38-flash-next-w4b.overlay.hgn` | 2,572,466,560 | 741 | `qwen3.8-flash-next` | 118,720 |
| `qwen38-flash-next-w4b.overlay-speed.hgn` | 2,478,095,488 | 741 | `qwen3.8-flash-next` | 118,720 |
| `qwen38-flash-next-mtp.hgn` | 1,523,566,720 | 31 | `qwen3.8-flash-next-mtp` | 5,120 |
| `qwen38-flash-next-vision.hgn` | 897,916,416 | 333 | `qwen3.8-flash-next-vision` | 53,440 |
| `qwen3.8-27b-p1w4d-d2.hgn` | 35,865,565,184 | 1,352 | `Qwen3.8-27B-p1-d2` | 216,448 |
| `qwen38-flash-next-v2.hgn` | 66,687,678,432 | 2,015 | `qwen3.8-flash-next-v2` | 322,560 |
| `qwen38-flash-next-ht43.hgn` | 57,618,253,792 | 2,015 | `qwen3.8-flash-next-ht43` | 322,560 |
| `qwen38-flash-next-ngram.hgn` | 51,200,246,144 | 1 | `qwen3.8-flash-next-ngram` | 320 |

Stores in each file:

| File | Store counts |
| --- | --- |
| `qwen38-flash-next-w4b.hgn` | `q4c` v2 841, `bf16` 353, `i64` 3, `fp8g` 1 |
| `qwen38-flash-next-w4b.overlay.hgn` | `q4c` v2 711, `q8g64` 30 |
| `qwen38-flash-next-w4b.overlay-speed.hgn` | `q4c` v2 723, `q8g64` 18 |
| `qwen38-flash-next-mtp.hgn` | `q8g64` 18, `bf16` 10, `q4c` v2 3 |
| `qwen38-flash-next-vision.hgn` | `bf16` 333 |
| `qwen3.8-27b-p1w4d-d2.hgn` | `bf16` 493, `i4l` 400, `fp8r` 233, `q4c` v0 168, `q4c` v1 54, `i32` 2, `f32` 2 |
| `qwen38-flash-next-v2.hgn` | `f16` 818, `bf16` 353, `ht` `0x1208` 313, `q6g64` 290, `q4c` v2 124, `i4r` 96, `q8g64` 18, `i64` 3 |
| `qwen38-flash-next-ht43.hgn` | `f16` 818, `ht` `0x1208` 361, `bf16` 353, `q8g64` 308, `q4c` v2 124, `ht` `0x1206` 48, `i64` 3 |
| `qwen38-flash-next-ngram.hgn` | `fp8g` 1 |

SHA-256 of the header and table (bytes 0 to the end of the table), for a
reader test that does not need the payloads:

| File | SHA-256 |
| --- | --- |
| `qwen38-flash-next-w4b.hgn` | `8ca62a2dbf682aeb4572289c545bf1694c022dc0f76d4b26f1cd1e59c073aad4` |
| `qwen38-flash-next-w4b.overlay.hgn` | `4526329b2c8e7263e89f9d10d5330da28f4db802432dc6072e052d52acbb51bb` |
| `qwen38-flash-next-w4b.overlay-speed.hgn` | `f9cbcf3b57124005b7dcdc1885eca67f61cc2bb5d80c964fb374eda26a2a7082` |
| `qwen38-flash-next-mtp.hgn` | `8bb91b381a490b8595ae9a3949ce4226b2c4ecbdcb54a28570091facd47d5a8f` |
| `qwen38-flash-next-vision.hgn` | `2cd308a09a4b69d46875cfa2d6167663d7c8ca26fd17738ad011874ce4d0e8fc` |
| `qwen3.8-27b-p1w4d-d2.hgn` | `788082dc2b4eb0f6a89daac2d928aaceed9a5afe48411baebbf6c1892402cd92` |
| `qwen38-flash-next-v2.hgn` | `6dd6e4fd5afe6b5c657a4c96f8398253a96cdb8820f8b1449a97350462705020` |
| `qwen38-flash-next-ht43.hgn` | `657223daf9d4675ef92cf37e254ca35fbe9355d4f0ea26ced4c9e73854aa0ca6` |
| `qwen38-flash-next-ngram.hgn` | `52a6dc997be8909b8814f1509b3a10a6e92f3ab299da0a87e78bf7a987f578fd` |

These facts hold for the nine files:

- Every one of the 8,427 entries has the size of the
  [size rule](STORAGE.md#size-summary) of its store. Every payload offset is
  a multiple of 64. In every file, the data offset field is equal to the
  payload start.
- Seven files end at their last payload, rounded up to 64. `v2` and `ht43`
  end at the end of their last payload, 32 bytes before the next multiple of
  64.
- The n-gram file holds one entry, `layers.1.ple.ngram_embedding.weight`.
  Its store, shape, size, checksum (`0xF682E517`) and variant are equal to
  those of the w4b table entry.
- `v2` and `ht43` hold each name of w4b except the n-gram table, and 818
  rotation vectors. They hold no n-gram table.
- In `v2` and in `ht43`, each `bf16`, `i64` and `q4c` entry (480 in each file)
  is equal to the w4b entry of the same name: store, shape, size, checksum and
  variant.
- The 31 `mtp.` entries of `v2` and of `ht43` are equal to the entries of the
  MTP head file.

The PLE index vectors of the Flash-Next checkpoints. The values were measured
in w4b. In `v2` and `ht43`, the entries of the three vectors, checksums
included, are equal to the w4b entries.

| Tensor | Values |
| --- | --- |
| `layers.1.ple.ple_embedding.layer_multipliers` | 23703573157769, 20109073645365, 8052911324071 |
| `layers.1.ple.ple_embedding.ngram_heads_offsets` | 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 |
| `layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` | 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 |

Each offset is the sum of the vocabulary sizes before it. The last offset
plus the last vocabulary size is 320,001,446. The table has
320,001,536 rows, so every head's rows fit inside it.
