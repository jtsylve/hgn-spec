# HGN checkpoint format

HGN is the weight file format of the halogen inference engines. An HGN file is
a small header, a fixed-size tensor table and 64-byte-aligned payloads. Each
payload is already in the layout that a GPU kernel reads. A reader maps the
file into memory and uses the payloads in place.

Two halogen engines use the format. They share the container and the store
codes. They use different subsets of the storage formats:

| Engine | Model | Published checkpoint |
| --- | --- | --- |
| halogen-flash | Qwen3.8 Flash-Next (GGUF architecture `qwen4exp`) | [`peonist-ai/halogen-qwen3.8-flash-next`](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next) |
| halogen | Qwen3.8 27B (GGUF architecture `qwen35`) | [`peonist-ai/halogen-qwen3.8-27b`](https://huggingface.co/peonist-ai/halogen-qwen3.8-27b) |

This document set is a clean-room specification. It describes bytes, values
and behavior. It does not describe or quote any implementation. It is written
for an engineer who adds HGN support, or the HGN storage formats, to another
inference engine.

## Documents

| Document | Contents |
| --- | --- |
| [Container](CONTAINER.md) | Header, tensor table, alignment, checksum, validation and the companion files |
| [Storage formats](STORAGE.md) | The store codes: byte layout, size rule and decode rule of each |
| [Qwen3.8 Flash-Next tensors](TENSORS.md) | Every tensor: name, shape, accepted formats and value conventions |
| [Qwen3.8 27B tensors](TENSORS-27B.md) | The tensors of the published 27B checkpoint |
| [GGUF import](GGUF-IMPORT.md) | How a `qwen4exp` GGUF becomes HGN tensors, and which conversions are exact |
| [Numerics](NUMERICS.md) | Number formats, decode arithmetic, rounding and exactness |
| [Porting guide](PORTING.md) | Integration plan, kernel design, memory residency and qualification for any engine |
| [Porting to gufo](PORTING-GUFO.md) | The porting guide applied to gufo |
| [Test vectors](TEST-VECTORS.md) | Byte-level examples, a complete small file and facts of the published files |
| [Attribution](ATTRIBUTION.md) | Sources, licenses, pinned revisions and trademarks |
| [Change log](CHANGELOG.md) | Changes in each version of this specification |

Read in this order to implement a reader: Container, Storage formats, one of
the tensor documents, Numerics. Read GGUF import to convert GGUF files or to
keep the HGN layouts for GGUF input. Read the porting guide for the engine
work, then the guide for your engine if there is one.

## Design summary

The format follows eight decisions. Each document gives the details.

1. **Kernel-ready payloads.** No payload needs a transform before a kernel
   reads it. Load time is the time to map and register memory.
2. **Planes of codes and scales.** A quantized payload keeps codes and scales
   apart, as whole planes or within each row. Code rows are contiguous and
   aligned, so kernels use 128-bit loads.
3. **A codebook in each 4-bit tensor.** The `q4c` format stores sixteen `f32`
   levels at the start of the payload. One decoder serves learned codebooks,
   NF4, NVFP4 and the GGUF types Q4_0, IQ4_NL, IQ4_XS and IQ3_S.
4. **One precision for each phase.** A tensor can have two copies: an 8-bit
   copy for token generation and a rotated 4-bit copy for prompt processing
   (see [Qwen3.8 27B tensors](TENSORS-27B.md)).
5. **No metadata.** The file has no key-value section. The engine has the
   model constants built in and checks every tensor against them. A 64-byte
   identity string names the checkpoint.
6. **Strict sizes.** A reader computes the payload size of every tensor from
   its shape and store code. It refuses the file when the size in the table is
   different.
7. **The original training conventions.** Norm gains are zero-centred and the
   DeltaNet value heads are grouped, as in the original checkpoints. GGUF files
   change both; HGN does not.
8. **Overlays and sidecars.** A small overlay file replaces selected tensors
   with better-quantized copies. The MTP head and the vision encoder can be
   separate files.

## Conventions

- All integers and floating-point numbers are little-endian.
- `f32` is IEEE 754 binary32, `f16` is IEEE 754 binary16. `bf16` is the upper
  16 bits of a binary32 value.
- Offsets and sizes are in bytes unless a unit is given.
- `/` is integer division that rounds toward zero. All operands are
  non-negative.
- `pad(x, a)` is `x` rounded up to a multiple of `a`.
- Shapes are row-major and list the outermost dimension first.
- `K` is the last (innermost) dimension of a tensor: the reduction dimension of
  a matrix product. `N` is the product of all other dimensions: the row count.
- "Must" marks a requirement for compatibility. "Should" marks a
  recommendation.
- **Not established** marks a fact that the evidence does not settle. Do not
  treat it as a requirement.

## Terms

| Term | Meaning |
| --- | --- |
| Entry | One 160-byte record in the tensor table |
| Payload | The bytes of one tensor, at the offset that its entry gives |
| Store code | The 32-bit number in an entry that selects a storage format |
| Variant | The 32-bit number in an entry that selects a form of a storage format |
| Row | `K` consecutive values of a tensor |
| Group | Consecutive values of one row that share a scale. The size depends on the format |
| Super-block | 256 consecutive values of one row |
| Plane | A region of a payload that holds one kind of data for all rows |
| Code | An integer that a decode rule maps to a value |
| Codebook | The sixteen `f32` levels that a 4-bit code selects in `q4c` |
| Value | The real number that a stored weight represents |
| Reader | Software that opens HGN files for inference |
| Writer | Software that produces HGN files |
| Importer | A writer that converts GGUF files |

## Scope

In scope:

- The HGN container and its companion files.
- All storage formats that the engines read or that the published files use.
- The tensor set, shapes and value conventions of both published models.
- The GGUF-to-HGN conversion rules of halogen-flash.
- Kernel and memory design that the formats were built for.

Out of scope:

- The model mathematics, except where a stored value depends on it.
- The n-gram hash of Qwen3.8 Flash-Next and the internals of the 27B drafter.
- The tokenizer. HGN files carry no tokenizer.
- Prompt-cache files, snapshots and other runtime state files.

## Versioning

This specification uses [semantic versioning](https://semver.org/):

- **Major** versions change a rule so that a reader or writer that followed
  the previous version is no longer correct.
- **Minor** versions add rules, formats, tensors or facts that were not
  established before, without changing an existing rule.
- **Patch** versions correct errors in examples or wording and clarify text
  without changing a rule.

The specification version is not the HGN file version. The file version is
the `version` field of the [header](CONTAINER.md#header).

## Evidence and confidence

Every rule has at least one of these sources. Each document says which.

- **The halogen-flash 0.13.5 reader.** Its observed behavior gives the
  container rules, the formats it reads, the tensor rules of Qwen3.8
  Flash-Next and the GGUF import.
- **The published files.** Their headers and tables (4,396 entries in six
  files) were checked against every container and size rule in this set. None
  failed. Payload samples were decoded and checked with structural tests.
- **Independent references.** Each GGUF conversion rule was implemented from
  this text and compared with the dequantizer of the `gguf` Python package.
  Value conventions were compared with the public Unsloth GGUF files of both
  models.

The 27B engine was not analyzed. The rules for the formats that only the 27B
checkpoint uses (`fp8r`, `i4l` and `q4c` variants 0 and 1) come from the
published file. Each decode rule reproduces a second, independent copy of the
same weight in that file (see [Numerics](NUMERICS.md#evidence-for-the-27b-formats)).

## Legal basis

The halogen license (section 3 of its end-user license agreement) permits
reverse engineering and welcomes interoperability research. It asks that
derived source or kernels are not redistributed as one's own work. This set
contains no halogen source and no halogen kernel code. It describes file
formats and observable behavior for interoperability. [Attribution](ATTRIBUTION.md)
lists every source, its license and its pinned revision.


## Document Revision

*Version 1.0.0, 2026-09-24.*

Describes halogen's published checkpoints and the
halogen-flash 0.13.5 reader. See the [change log](CHANGELOG.md).

Copyright (c) 2026 Joe T. Sylve, Ph.D. <joe.sylve@gmail.com>.

Dual-licensed under the [MIT License](LICENSE-MIT) or 
[CC BY 4.0](LICENSE-CC-BY-4.0), at your option; see [LICENSE](LICENSE)
and [ATTRIBUTION](ATTRIBUTION.md#license).


The latest version is maintained at
[github.com/jtsylve/hgn-spec](https://github.com/jtsylve/hgn-spec).
