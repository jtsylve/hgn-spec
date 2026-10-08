# Attribution

## Author

Copyright (c) 2026 Joe T. Sylve, Ph.D. \<joe.sylve@gmail.com\>

Joe T. Sylve is the author of this document set.

## License

This document set is dual-licensed (`SPDX-License-Identifier: MIT OR
CC-BY-4.0`). Use it under either license, at your option:

| License | Full text | Best fit |
| --- | --- | --- |
| MIT License | [LICENSE-MIT](LICENSE-MIT) | Software: code, fixtures and test vectors that implement or copy from this set. |
| Creative Commons Attribution 4.0 International (CC BY 4.0) | [LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0), [creativecommons.org](https://creativecommons.org/licenses/by/4.0/) | Documentation: republishing, quoting, adapting or translating the text |

[LICENSE](LICENSE) states the choice. Both licenses require attribution and
nothing more:

- **Under MIT**, keep the copyright notice and the permission notice with all
  copies or substantial portions, including formulas and test vectors copied
  into code.
- **Under CC BY 4.0**, give the author's name, the copyright notice, a
  reference to the license and a link to the source, and say whether you made
  changes. A suggested attribution: "HGN checkpoint format specification by
  Joe T. Sylve, Ph.D., licensed under CC BY 4.0", with a link to
  <https://github.com/jtsylve/hgn-spec>.

The latest version of this set is maintained at
[github.com/jtsylve/hgn-spec](https://github.com/jtsylve/hgn-spec).

When the set moves into another repository, keep `LICENSE`, `LICENSE-MIT` and
`LICENSE-CC-BY-4.0` beside it, or add the notice to that repository's `NOTICE`
file.

## Provenance of the text

This document set is original text. It contains no halogen source code and no
halogen kernel code. It describes a file format and observable behavior for
interoperability, and it restates public format definitions from the sources
below. Each source keeps its own license and copyright.

## The HGN format and its engines

| Component | Relationship | License | Pinned revision / version | Source |
| --- | --- | --- | --- | --- |
| halogen-flash engine | Originator of the HGN format. Its observed behavior is the source of the container rules, the halogen-flash reader rules and the GGUF import rules | Proprietary end-user license agreement, version 0.1; section 3 permits reverse engineering and welcomes interoperability research | Image `ghcr.io/peonist-ai/halogen-flash-server:0.16.2`; engine binary SHA-256 `ac123b7ff5134e527368fc0644598379bcd976630691d36e9a469d55bee23f3b` | [peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) |
| halogen engine (27B) | Uses the same container. Not analyzed | Proprietary; see its repository | Image `ghcr.io/peonist-ai/halogen:0.1.4` (named by the checkpoint) | [peonist-ai/halogen-server](https://github.com/peonist-ai/halogen-server) |
| Flash-Next HGN checkpoint | Published files. Headers, tables and sampled payloads checked against this specification | `apache-2.0` in the repository metadata; the publisher states that the base model's terms govern the weights | `6e1aa1acda75bd8a3bb9043b1545f83187cb6ded` | [peonist-ai/halogen-qwen3.8-flash-next](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next) |
| 27B HGN checkpoint | Published file. Source of the `fp8r`, `i4l` and `q4c` variant 0 and 1 rules | `apache-2.0` in the repository metadata; the publisher states that the base model's terms govern the weights | `800ec932fc4a83ddd18b83d293cfd147946e77f9` | [peonist-ai/halogen-qwen3.8-27b](https://huggingface.co/peonist-ai/halogen-qwen3.8-27b) |

Copyright in the halogen engines and in the HGN checkpoints belongs to
Peonist, LLC and to the authors of the base models. Their license asks that
derived source or kernels are not redistributed as one's own work; this set
redistributes neither.

## Models

| Component | Relationship | License | Pinned revision / version | Source |
| --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next | Base model of the Flash-Next checkpoint; original value conventions | `qwen-community-1.0` (as recorded in the GGUF metadata) | — | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Qwen3.8 27B | Base model of the 27B checkpoint | `Apache-2.0` | — | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Unsloth Qwen3.8 Flash-Next GGUF | Reference for the tensor types, shapes, norm offsets, head order and MTP norms | Base model terms | `38bb39ee97821de2c9009abb7e93950eec396e66` (UD-Q4_K_XL, MTP Q8_0) | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| Unsloth Qwen3.8 27B GGUF | Reference for the 27B value conventions and MTP projection | Base model terms | `4ca720788d1e01f1bff70c033e0d0028fd02e502` (UD-Q8_K_XL) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| Unsloth Qwen3.8 27B NVFP4 | Named by the 27B checkpoint as the source of its NVFP4 values | Base model terms | — | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) |

## Format definitions restated here

| Definition | Used for | License or status | Source |
| --- | --- | --- | --- |
| GGUF container and GGML block types (Q4_0, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S), the IQ4_NL levels and the IQ3_S grid | GGUF import, `q6k`, `iq4nl`, `q8_0`, `q5_0`, `q5_1`, reverse conversion | `MIT`, Copyright (c) 2023-2026 The ggml authors | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| `gguf` Python package, version 0.19.0 | Independent dequantizer for the conversion checks | `MIT`, The ggml authors | [gguf on PyPI](https://pypi.org/project/gguf/) |
| NF4 levels | `q4c` variant 1 codebook | Published in QLoRA (Dettmers et al., 2023); implementation `MIT` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| FP4 E2M1 and 8-bit E4M3 encodings | `q4c` variant 0, the fp8 encoding | OCP Microscaling Formats (MX) specification; FP8 formats per Micikevicius et al., 2022 | [Open Compute Project](https://www.opencompute.org/) |
| NVFP4 block structure | `q4c` variant 0 (16-value blocks, E4M3 block scales) | NVIDIA format description | [NVIDIA](https://developer.nvidia.com/) |
| Sylvester Hadamard matrix and the fast Walsh–Hadamard transform | `i4l`, `i4r`, `ht` | Public mathematics | — |
| IEEE 754 binary16 and binary32 | All floating-point fields | IEEE standard | [IEEE 754-2019](https://standards.ieee.org/ieee/754/6210/) |
| FIPS 180-4 SHA-256 | Digests in the test vectors and the digest listing | Public standard | [NIST FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final) |

## Engine-specific guides

| Component | Relationship | License | Pinned revision | Source |
| --- | --- | --- | --- | --- |
| gufo | Subject of [Porting to gufo](PORTING-GUFO.md). Its file paths and current behavior are cited there | `MIT`, Copyright (c) 2026 Gufo contributors | `dea22ceea20d07f95a5ecbb2b06ffad7b93d0548` | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| AMD RDNA 3.5 instruction set | Instruction names in the porting guides (`v_dot2_f32_bf16`, `v_dot2_f32_f16`, WMMA) | AMD documentation | — | [AMD GPUOpen](https://gpuopen.com/) |

## Trademarks

"halogen" is a trademark of Peonist, LLC (U.S. application pending), as its
publisher states. Other names of companies, products, models and hardware in
this set belong to their owners. This set uses all such names only to
identify the formats, engines, models and hardware that it describes. It does
not state or suggest endorsement by any owner.
