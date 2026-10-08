# Change log

All notable changes to this specification are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version
numbers follow the [versioning rules](README.md#versioning) of this
specification.

## [Unreleased]

### Changed

- [Porting to gufo](PORTING-GUFO.md): planned against gufo commit `dea22ce`.
  - The v2 and ht43 needs are mapped to gufo: the rotation vectors, the `ht`,
    `i4r` and `q6g64` paths, the n-gram table file and the MTP head in the
    checkpoint.
  - New sections: the model constants and tokenizer, which HGN files do not
    hold, and the cache identity.
  - gufo's F32-only small tensors and the conversions that they need.
  - The 16 MiB upload tasks, the next-chunk n-gram gather and the
    BF16 GEMM of the indexer shapes.
- [Attribution](ATTRIBUTION.md): the gufo pin is now commit `dea22ce`.

## [1.1.0] - 2026-10-07

This version describes the halogen-flash 0.16.2 reader and the published
files of Flash-Next revision `6e1aa1ac`. 

### Added

- [Storage formats](STORAGE.md): six store codes.
  - `ht` (16): 3-bit or 4-bit trellis codes with a computed codebook.
  - `q8_0` (20), `q5_0` (21) and `q5_1` (22): GGUF blocks for the n-gram
    table.
  - `i4r` (23): 4-bit routed-expert codes with a scale for each 128 values.
  - `q6g64` (24): 6-bit affine codes.
  - Codes 17 to 19 are unassigned.
- [Storage formats](STORAGE.md#the-128-point-rotation): the 128-point
  rotation that `ht` and `i4r` use, and the `f16`
  [rotation vectors](STORAGE.md#rotation-vectors) that go with it.
- [Container](CONTAINER.md#n-gram-table-file): the n-gram table file, a fifth
  kind of companion file.
- [Container](CONTAINER.md#reader-procedure): halogen-flash checks `ht` and
  `i4r` entries when it opens a file. It also reads the table before load to
  plan memory, and ignores a table of more than 2²⁰ entries.
- [Container](CONTAINER.md#overlay-file): the overlay rules for the new stores
  and for rotation vectors.
- [Qwen3.8 Flash-Next tensors](TENSORS.md#published-checkpoints): the v2 and
  ht43 checkpoints. Also the new stores in the reader roles and the fused
  experts, the [rotation vectors](TENSORS.md#rotation-vectors) of each
  matrix, and the new n-gram table stores.
- [GGUF import](GGUF-IMPORT.md): Q5_0 to `q5a` variant 0. The conversion of
  an affine `out_proj` to `bf16`, which is the one conversion that rounds.
  Q8_0, Q5_0 and Q5_1 n-gram tables.
- [Numerics](NUMERICS.md): values of the new stores, strategy 3 (rotated
  activations), new exactness rows, and
  [evidence for the Flash-Next v2 formats](NUMERICS.md#evidence-for-the-flash-next-v2-formats).
- [Porting guide](PORTING.md): kernel notes for `q6g64`, `ht` and `i4r`.
  Also v2 and ht43 in the goals, the storage table, residency, the milestones
  and the pitfalls. [Porting to gufo](PORTING-GUFO.md) lists the v2 needs as
  later work.
- [Test vectors](TEST-VECTORS.md): vectors for the new stores and
  conversions, and facts of the three new published files.

### Changed

- [Container](CONTAINER.md#file-layout): end padding of a file is optional.
  The v2 and ht43 files end at the end of their last payload.
- [Container](CONTAINER.md#repack-cache): the manifest names each shard by its
  file name, escaped for JSON.
- [GGUF import](GGUF-IMPORT.md#type-rules):
  - Q5_0 is no longer refused.
  - The expert down projection accepts only types that a kernel reads.
  - Q5_0 and Q5_1 are refused for expert gate and up.
  - A BF16 `ssm_a` is refused.
  - The [head reorder](GGUF-IMPORT.md#head-reorder) applies to every accepted
    type.
  - BF16 norm gains, `dt_bias`, convolutions and `shared_expert_gate` take
    the float path, so the importer also subtracts the offset from BF16 norm
    gains.
- [Qwen3.8 Flash-Next tensors](TENSORS.md#the-n-gram-table): a checkpoint
  without the n-gram table can serve with an n-gram table file.
- [Attribution](ATTRIBUTION.md): the engine pin is now image `0.16.2`, and the
  Flash-Next pin is now revision `6e1aa1ac`.

### Fixed

- [Container](CONTAINER.md#reader-procedure): halogen-flash does not check that
  the table fits in the file, and it keeps the first entry of a duplicate
  name. Version 1.0.1 said that it does both.
- [Qwen3.8 Flash-Next tensors](TENSORS.md#reader-roles): halogen-flash refuses
  `q8g64` for `lm_head` and `shared_expert_gate`.
- [GGUF import](GGUF-IMPORT.md): halogen-flash computes `log(−A)` in binary64
  and rounds the result to `f32`. Version 1.0.1 said `f32` arithmetic.
- [Storage formats](STORAGE.md#dense-formats): store code 2 (`f16`) is
  established as dense binary16. It was **not established** in 1.0.1.

These facts remain **not established** in this version:

- the difference between HGN file versions 1 and 2
- other values of the `ht` layout and codebook fields
- how the writers chose the `ht` codes, the rotation vectors and the `i4r`
  scales
- the behavior of halogen-flash with different expert stores in different
  layers
- the row padding of `q4c` variants 0 and 1 when `K` is not a multiple of 256
- the meaning of the `i4l` variant bits other than the group size
- any use of `fp8g` other than the n-gram table
- the row order of the 27B `q_proj` (query and gate for each head)
- the structure and configuration values of the 27B drafter

## [1.0.1] - 2026-09-24

### Fixed

- [GGUF import](GGUF-IMPORT.md#head-reorder): removed the list of source
  types for which halogen-flash 0.13.5 skips the head reorder. The list was
  not complete and had one error: halogen-flash refuses a BF16 `ssm_conv1d`,
  because the tensor fails the shape check. The rule for a new importer does
  not change.

## [1.0.0] - 2026-09-24

The first release. It describes halogen's published checkpoints and the
halogen-flash 0.13.5 reader.

### Added

- [Container](CONTAINER.md): the header, the tensor table, the XOR checksum,
  the reader and writer procedures, and the companion files (MTP head file,
  overlay files, vision sidecar and repack cache).
- [Storage formats](STORAGE.md): every store code that halogen-flash reads or
  that the published files use, with layout, size rule and decode rule:
  `bf16`, `f32`, `i32`, `i64`, `q4c` variants 0, 1 and 2, `fp8r`, `q8g64`,
  `i4l`, `fp8g`, `q8s32`, `q6k`, `iq4nl`, `q4a` and `q5a`, and the fp8
  encoding.
- [Qwen3.8 Flash-Next tensors](TENSORS.md): names, shapes, reader roles,
  accepted stores and value conventions, with the MTP head and the vision
  sidecar.
- [Qwen3.8 27B tensors](TENSORS-27B.md): the tensors of the published 27B
  checkpoint, its paired decode and prefill copies, and its value conventions.
- [GGUF import](GGUF-IMPORT.md): the conversion of a `qwen4exp` GGUF file,
  the name map, the head reorder, the norm offsets, the output order, reverse
  conversion to GGUF blocks and a worked example.
- [Numerics](NUMERICS.md): the value of every stored weight, the rounding
  rules, halogen-flash's matrix-product strategies, measured conversion
  exactness and the evidence for the 27B formats.
- [Porting guide](PORTING.md) for any engine, and
  [Porting to gufo](PORTING-GUFO.md).
- [Test vectors](TEST-VECTORS.md): a complete 1,024-byte file, vectors for
  each format and conversion, tensor sizes and facts of the published files.
- [Attribution](ATTRIBUTION.md), and the dual license `MIT OR CC-BY-4.0`.

These facts remain **not established** in this version:

- the difference between HGN file versions 1 and 2
- the layout of store code 2 (`f16`), which no published file uses
- the row padding of `q4c` variants 0 and 1 when `K` is not a multiple of 256
- the meaning of the `i4l` variant bits other than the group size
- any use of `fp8g` other than the n-gram table
- the row order of the 27B `q_proj` (query and gate for each head)
- the structure and configuration values of the 27B drafter

[Unreleased]: https://github.com/jtsylve/hgn-spec/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/jtsylve/hgn-spec/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/jtsylve/hgn-spec/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/jtsylve/hgn-spec/releases/tag/v1.0.0
