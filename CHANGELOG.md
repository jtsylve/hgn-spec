# Change log

All notable changes to this specification are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version
numbers follow the [versioning rules](README.md#versioning) of this
specification.

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

[1.0.0]: https://github.com/jtsylve/hgn-spec/releases/tag/v1.0.0
