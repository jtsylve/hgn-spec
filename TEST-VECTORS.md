# Test vectors

These vectors come from a reference implementation that was written from this
specification. Each conversion vector was also checked against the
dequantizer of the `gguf` Python package (version 0.19.0). Hexadecimal byte
strings list bytes in file order, 32 bytes to a line. Decimal values are
exact: each is the decimal form of an `f32` value.

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
occur in the published checkpoint or its overlay.

| Tensor | Store | Shape | Bytes |
| --- | --- | --- | ---: |
| `experts.gate_up_proj` | `q4c` | `[512,1280,2560]` | 943,718,464 (published) |
| `experts.gate_up_proj` | `q4a` variant 1 | `[512,1280,2560]` | 975,175,680 |
| `experts.gate_up_proj` | `q5a` variant 0 | `[512,1280,2560]` | 1,258,291,200 |
| `experts.gate_up_proj` | `q5a` variant 1 | `[512,1280,2560]` | 1,184,890,880 |
| `experts.down_proj` | `q4c` | `[512,2560,640]` | 482,345,024 (published) |
| `experts.down_proj` | `q8s32` | `[512,2560,640]` | 901,775,360 |
| `experts.down_proj` | `q5a` variant 0 | `[512,2560,640]` | 629,145,600 |
| `linear_attn.in_proj_qkv` | `q4c` | `[10240,2560]` | 14,745,664 (published) |
| `linear_attn.in_proj_qkv` | `q8s32` | `[10240,2560]` | 27,852,800 |
| `self_attn.q_proj` | `q4c` | `[12288,2560]` | 17,694,784 (published) |
| `lm_head` | `q4c` | `[248320,2560]` | 357,580,864 (published) |
| `embed_tokens` | `q4c` | `[248320,2560]` | 357,580,864 (published) |
| `lm_head` | `q6k` | `[248320,2560]` | 521,472,000 |
| `lm_head` | `q8s32` | `[248320,2560]` | 675,430,400 |
| `self_attn.o_proj` | `q8g64` | `[2560,6144]` | 16,711,680 (published) |

## Published files

Measured from the published repositories at these revisions:

| Repository | Revision |
| --- | --- |
| `peonist-ai/halogen-qwen3.8-flash-next` | `e053f488b120b99ed2525e6ac99f68c51b3b6179` |
| `peonist-ai/halogen-qwen3.8-27b` | `800ec932fc4a83ddd18b83d293cfd147946e77f9` |

| File | Bytes | Entries | Identity | Payload start |
| --- | ---: | ---: | --- | ---: |
| `qwen38-flash-next-w4b.hgn` | 124,068,083,904 | 1,198 | `qwen3.8-flash-next` | 191,808 |
| `qwen38-flash-next-w4b.overlay.hgn` | 2,572,466,560 | 741 | `qwen3.8-flash-next` | 118,720 |
| `qwen38-flash-next-w4b.overlay-speed.hgn` | 2,478,095,488 | 741 | `qwen3.8-flash-next` | 118,720 |
| `qwen38-flash-next-mtp.hgn` | 1,523,566,720 | 31 | `qwen3.8-flash-next-mtp` | 5,120 |
| `qwen38-flash-next-vision.hgn` | 897,916,416 | 333 | `qwen3.8-flash-next-vision` | 53,440 |
| `qwen3.8-27b-p1w4d-d2.hgn` | 35,865,565,184 | 1,352 | `Qwen3.8-27B-p1-d2` | 216,448 |

Stores in each file:

| File | Store counts |
| --- | --- |
| `qwen38-flash-next-w4b.hgn` | `q4c` v2 841, `bf16` 353, `i64` 3, `fp8g` 1 |
| `qwen38-flash-next-w4b.overlay.hgn` | `q4c` v2 711, `q8g64` 30 |
| `qwen38-flash-next-w4b.overlay-speed.hgn` | `q4c` v2 723, `q8g64` 18 |
| `qwen38-flash-next-mtp.hgn` | `q8g64` 18, `bf16` 10, `q4c` v2 3 |
| `qwen38-flash-next-vision.hgn` | `bf16` 333 |
| `qwen3.8-27b-p1w4d-d2.hgn` | `bf16` 493, `i4l` 400, `fp8r` 233, `q4c` v0 168, `q4c` v1 54, `i32` 2, `f32` 2 |

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

The PLE index vectors of the Flash-Next checkpoint:

| Tensor | Values |
| --- | --- |
| `layers.1.ple.ple_embedding.layer_multipliers` | 23703573157769, 20109073645365, 8052911324071 |
| `layers.1.ple.ple_embedding.ngram_heads_offsets` | 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 |
| `layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` | 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 |

Each offset is the sum of the vocabulary sizes before it. The last offset
plus the last vocabulary size is 320,001,446. The table has
320,001,536 rows, so every head's rows fit inside it.
