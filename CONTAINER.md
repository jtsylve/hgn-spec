# HGN container

An HGN file has three parts: a header, a tensor table and a payload region.
The file has no key-value metadata and no tokenizer. All numbers are
little-endian. Both halogen engines use the same container.

## File layout

| Offset | Size | Contents |
| --- | --- | --- |
| `0x00` | `0x68` | Header |
| `0x68` | `0xA0 × n` | Tensor table: `n` entries |
| `0x68 + 0xA0 × n` | to the payload start | Zero bytes |
| `pad(0x68 + 0xA0 × n, 64)` | to the end of the file | Payloads, each at a 64-byte boundary |

The payload start is always a multiple of 64. Each payload starts at a
multiple of 64. The bytes between the end of one payload and the start of the
next are zero. The file ends at the end of the last payload, rounded up to 64.

All six published files follow this layout exactly: their payloads are
contiguous, in table order, with no gap larger than the padding.

## Header

The header is 104 (`0x68`) bytes.

| Offset | Size | Type | Field | Rule |
| --- | ---: | --- | --- | --- |
| `0x00` | 4 | `u32` | magic | `0x314E4748`. In file order the bytes are `48 47 4E 31`, "HGN1" |
| `0x04` | 4 | `u32` | version | 1 or 2. The layout is the same for both. Writers write 2 |
| `0x08` | 8 | `u64` | tensor count | `n`, the number of table entries |
| `0x10` | 8 | `u64` | table offset | File offset of the first entry. Writers write `0x68` |
| `0x18` | 8 | `u64` | data offset | The payload start. Writers write it; readers ignore it |
| `0x20` | 8 | `u64` | file size | Must equal the size of the file |
| `0x28` | 64 | `char[64]` | identity | ASCII, padded with NUL bytes |

Every published file has version 2. The difference between version 1 and
version 2 is **not established**. A reader must accept both and parse them the
same way.

The identity names the checkpoint. Halogen-flash reports it to its clients.
These identities occur:

| Identity | File |
| --- | --- |
| `qwen3.8-flash-next` | The Flash-Next checkpoint and both of its overlay files |
| `qwen3.8-flash-next-mtp` | The Flash-Next MTP head file |
| `qwen3.8-flash-next-vision` | The Flash-Next vision sidecar |
| `qwen3.8-flash-next-gguf` | Every file that halogen-flash makes from a GGUF file |
| `Qwen3.8-27B-p1-d2` | The 27B checkpoint |

## Tensor table

Entry `i` starts at `table offset + 0xA0 × i`. An entry is 160 (`0xA0`) bytes.

| Offset | Size | Type | Field | Rule |
| --- | ---: | --- | --- | --- |
| `0x00` | 96 | `char[96]` | name | ASCII, 1 to 95 bytes, padded with NUL bytes. Unique in the file |
| `0x60` | 4 | `u32` | store code | Selects the storage format. See [Storage formats](STORAGE.md) |
| `0x64` | 4 | `u32` | rank | Number of dimensions, 1 to 4 |
| `0x68` | 32 | `i64[4]` | shape | Dimensions, outermost first. Slots at and after `rank` are zero |
| `0x88` | 8 | `u64` | offset | File offset of the payload. A multiple of 64 |
| `0x90` | 8 | `u64` | size | Payload size in bytes, without padding |
| `0x98` | 4 | `u32` | checksum | XOR fold of the payload. See [Checksum](#checksum) |
| `0x9C` | 4 | `u32` | variant | Store-specific variant number. Zero when the store has none |

The name has at most 95 bytes because a NUL must follow it. A writer must
refuse a longer name. It must not shorten it.

`K` is `shape[rank − 1]`. `N` is the product of `shape[0]` to
`shape[rank − 2]`, or 1 when `rank` is 1. Every size rule in
[Storage formats](STORAGE.md) uses these two numbers.

Writers must set the unused shape slots to zero. One reader check compares all
four slots of two entries (see [Overlay file](#overlay-file)).

## Checksum

The checksum is a 32-bit XOR fold of the payload bytes:

1. Split the payload into 4-byte words from offset 0.
2. If the last word is short, fill it with zero bytes at its end.
3. Read each word as a little-endian `u32`.
4. XOR all words. An empty payload has the checksum 0.

The padding after a payload is not part of the checksum. Sampled payloads of
the published files match their checksums.

Halogen-flash computes the checksum when it writes a file. It verifies the
checksum only when it copies tensors out of an MTP head file (see
[MTP head file](#mtp-head-file)). It does not verify checksums during a normal
load. A reader should verify checksums when it converts or copies tensors. It
can skip the check on the load path, because the check reads every byte.

The XOR fold detects damaged bytes. It does not detect two words that change
place. Do not use it as a content identity. Use SHA-256 of the payload for
that purpose.

## Reader procedure

1. Map the file read-only. Do the mapping at a page boundary, so that file
   alignment and memory alignment are equal.
2. Refuse the file in these cases:
   - It is shorter than `0x68` bytes.
   - The magic is wrong.
   - The version is not 1 or 2.
   - The file size field is not the size of the file.
3. Read `n` entries from the table offset. Refuse the file when the table does
   not fit in the file.
4. For each entry, take the name up to the first NUL byte. Refuse the entry
   when `offset + size` is larger than the file size.
5. Build a map from name to entry. Refuse duplicate names.
6. When the model binds a tensor, apply the checks of its store code: the size
   rule, the `K` rule, the variant and the alignment. The checks are in
   [Storage formats](STORAGE.md).

Halogen-flash does steps 1 to 5 when it opens the file. It does step 6 when the
model first uses each tensor. It does not check that payloads do not overlap.
A reader should check it.

A reader must not depend on the order of the entries.

## Writer procedure

1. Make the complete tensor list first: name, store code, shape, variant and
   payload size. Refuse the list if any size rule fails.
2. Compute the payload start, `pad(0x68 + 0xA0 × n, 64)`.
3. Give each payload the next free offset. Advance by `pad(size, 64)`.
4. Set the file size to the end of the last padded payload. Create the file at
   that size, so that all padding is zero.
5. Write each payload at its offset and compute its checksum.
6. Write the header and the table last, because the table holds the checksums.
7. Flush the file to storage. Give it its final name only after the flush.

Use the same entry order for the table and for the payloads. The same input
then gives the same bytes. The file holds no time stamp.

The canonical order that the GGUF importer uses is in
[GGUF import](GGUF-IMPORT.md#output-order).

## Companion files

A checkpoint can come with other HGN files. Halogen-flash reads four kinds.
All are optional for a reader that has a complete checkpoint.

### MTP head file

The multi-token-prediction (MTP) head can be a separate HGN file. Its
published name is `qwen38-flash-next-mtp.hgn`, identity
`qwen3.8-flash-next-mtp`, 31 tensors.

The published Flash-Next checkpoint already holds the same 31 `mtp.` tensors.
Halogen-flash needs the head file only when the trunk is a GGUF file, because
it does not read the MTP tensors of a GGUF file.

- The reader takes every tensor whose name starts with `mtp.`. It ignores
  every other tensor in the file.
- The file must hold at least one `mtp.` tensor.
- A name that is in the trunk and in the head file is an error.
- The routed experts of the head must use `q4c` for gate and up, and `q4c` or
  `q8s32` for down.
- A converter that writes a complete file copies each `mtp.` tensor verbatim:
  bytes, store code, variant and checksum. It verifies the checksum of each
  copied payload against the head file.

The published head file stores its 18 dense projections as `q8g64`, its
experts and shared-expert gate as `q4c` and its norms and router as `bf16`.
The tensor names are in
[Qwen3.8 Flash-Next tensors](TENSORS.md#mtp-head).

### Overlay file

An overlay file replaces some tensors of a base file. It lets a distributor
publish better-quantized copies of sensitive tensors without a new base file.

The default overlay of `name.hgn` is `name.overlay.hgn` in the same directory.
Halogen-flash uses it when it exists, unless an operator setting turns it off
or names a different file. An operator can also exclude overlay tensors by a
name pattern.

Rules for each overlay tensor:

- The base file must hold a tensor with the same name.
- The rank must be equal. All four shape slots must be equal.
- One of these must be true:
  - The store code, the payload size and the variant are equal.
  - The base store is `q4c` and the overlay store is `bf16`, with a payload
    size of `2 × numel`.
  - The base store is `q4c` and the overlay store is `q8g64`, with a payload
    size of `17 × numel / 16`.

The reader uses the overlay tensor in place of the base tensor. It does not
make the base tensor resident.

The Flash-Next repository publishes two overlays with 741 tensors each. Both
hold new copies of the 723 non-expert matrices of the trunk, all except the
embedding, and of the 18 dense projections of the MTP head. They differ in one
choice:

| File | `self_attn.o_proj` (12 tensors) | MTP dense projections (18 tensors) |
| --- | --- | --- |
| `qwen38-flash-next-w4b.overlay.hgn` | `q8g64` | `q8g64` |
| `qwen38-flash-next-w4b.overlay-speed.hgn` | `q4c` | `q8g64` |

### Vision sidecar

The vision encoder is a separate HGN file. Its published name is
`qwen38-flash-next-vision.hgn`, identity `qwen3.8-flash-next-vision`, 333
tensors. Every tensor name starts with `visual.`, and every tensor uses
`bf16`. A file without `visual.pos_embed.weight` is not a vision sidecar. The
tensor list is in [Qwen3.8 Flash-Next tensors](TENSORS.md#vision-sidecar).

### Repack cache

Halogen-flash can read a GGUF file directly. It then converts the tensors in
memory, with the rules in [GGUF import](GGUF-IMPORT.md). It can store the
result as a repack cache, so that the next start maps the cache and skips the
conversion.

The cache is two files in one directory:

- `STEM.hgn`: an HGN file, version 2, identity `qwen3.8-flash-next-gguf`.
- `STEM.hgn.json`: a manifest.

`STEM` is the file name of the first GGUF shard without `.gguf` and without a
split suffix of the form `-00001-of-00004`. The directory is the directory of
the first shard, or a directory that the operator selects.

The cache file holds every converted tensor. It does not hold the n-gram table
or the MTP head tensors. Its payloads are in the canonical order, and its
payload region is byte-identical to the in-memory result. So a payload offset
in the cache is the in-memory offset plus the payload start.

The manifest is JSON text with a fixed format. Halogen-flash compares it byte
for byte with the text that it computes for the current GGUF files. This is an
example with two shards:

```json
{
  "halogen_gguf_cache": 1,
  "entries": 1432,
  "arena_bytes": 61234567168,
  "shards": [
    {"name": "/models/example-00001-of-00002.gguf", "size": 49950000000, "mtime_ns": 1790000000000000000},
    {"name": "/models/example-00002-of-00002.gguf", "size": 12340000000, "mtime_ns": 1790000000000000000}
  ]
}
```

- Two spaces indent each line inside the object. Four spaces indent each shard
  line.
- A line break follows each line, the last one included.
- `entries` is the number of tensors in the cache file.
- `arena_bytes` is the size of the payload region: the sum of all payload
  sizes, each rounded up to 64.
- `name` is the shard path as the program opened it, with no JSON escaping.
- `size` is the shard size in bytes. `mtime_ns` is its modification time in
  nanoseconds.
- The numbers in the example are illustrations, not real values.

The cache is valid when three things are true:

- The manifest text is equal.
- The `.hgn` file exists.
- Its entries agree with a fresh conversion plan: the same count, and in
  order the same name, size and store code.

To write the cache safely:

1. Remove any `STEM.hgn.tmp` and `STEM.hgn.json.tmp` that an earlier run left.
2. Check that the directory has the file size plus 1 GiB free. If not, do not
   write the cache.
3. Write `STEM.hgn.tmp`. Flush it. Rename it to `STEM.hgn`.
4. Only then write `STEM.hgn.json.tmp` and rename it to `STEM.hgn.json`.

A manifest without its `.hgn` file then shows an interrupted write. A reader
treats a stale cache as absent.

### Raw override files

Halogen-flash has two diagnostic inputs outside the HGN container. A port does
not need them.

- A per-tensor override: a file named `TENSOR-NAME.bf16` that holds only the
  raw `bf16` values of the tensor, `2 × numel` bytes, with no header.
- An embedding override: raw `bf16` values of shape `[248320, 2560]`,
  exactly 1,271,398,400 bytes.
