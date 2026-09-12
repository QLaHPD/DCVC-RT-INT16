# UF INT16 runtime

The archive CLI supports prepared integer intra and temporal inference for HT-S,
HT-L and LD. It quantizes the released checkpoints; no retraining is required.
This is a local arithmetic design, not a specification from the UF paper.
FP16 remains the default; select integer encoding with `--runtime int16`.

The archive encoder retains decoded P-frame features without rendering pixels
it will not use. It still reconstructs the required reset frame. Decode/view
continue to reconstruct every frame. This optimization preserves prepared
identities, arithmetic and bitstream bytes. HT-L performance measurements are in
[the validation report](UF_INT16_VALIDATION.md).

The prior caches ordered tensor positions for the current shape, avoiding
repeated boolean-index synchronization. Pointwise integer CUDA convolutions
use wider spatial tiles or deeper reduction tiles to reuse data and reduce
barriers. Both preserve exact integer arithmetic and existing prepared files.
After updating from the earlier kernel, rebuild it in the UF environment:

```bash
MAX_JOBS=1 ./scripts/uf-python -m pip install --no-build-isolation ./src/int16/native
./scripts/uf-python scripts/check_uf_setup.py
```

The setup check reports kernel revision `tiled-v2`. No model re-preparation is
needed. Tile selection was measured on Jetson Orin; its speed benefit on other
CUDA GPUs remains to be measured.

## Commands

Build the entropy extension and `src/int16/native` as described in
[local setup](UF_LOCAL_SETUP.md), then:

```bash
./scripts/uf-python main.py prepare-int16 --model_structure hts
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/integer --runtime int16 --qi 36 --qp 30 --procs 1
./scripts/uf-python main.py verify --input runs/integer/Big_Buck_Bunny_720_10s_30MB.mp4.uf
./scripts/uf-python main.py decode --input runs/integer/Big_Buck_Bunny_720_10s_30MB.mp4.uf --output_file runs/integer-reconstruction.mkv
./scripts/uf-python main.py view --input runs/integer/Big_Buck_Bunny_720_10s_30MB.mp4.uf
```

Folder input, resolution/FPS, audio, cooperative claims and cleanup use the same
[archive commands](UF_ARCHIVE.md). Decode/verify/view/cleanup automatically select
the recorded runtime; conflicting explicit selections fail. Preparation also
runs automatically when a cache is missing. RT environment flags are ignored.

## Moving prepared models

Preparation prints both prepared paths and identities. Copy the same two files
and the entire `.uf` archive to another machine. Supply `--prepared_i PATH
--prepared_p PATH` there. Explicit prepared files are self-contained: decoding
does not need the original floating checkpoints. Build native extensions in the
destination UF environment. RT prepared models are incompatible.

The canonical prepared identity covers source checkpoint SHA256, variant,
arithmetic version, parameters, lookup tables and entropy CDFs. The manifest and
72-byte `UF16` bitstream preamble bind both prepared identities. A mismatched or
modified file fails validation. Copying exact prepared files avoids depending on
identical floating-point table generation on different hosts.

## Arithmetic contract

The initial contract uses signed INT16 features at scale 512 and signed INT16
weights at scale 8192. Convolution dot products accumulate in INT64, divide by
8192 with nearest rounding (ties away from zero), add feature-scale bias, then
saturate. Residual additions saturate instead of wrapping. Feature products use
INT64 intermediates and divide by 512. The input conversion is the integer form
of `byte / 255 - 0.5`.

UF's four-way interleaved WSiLU activation sum is implemented explicitly.
Nonlinear activations and entropy-scale selection use prepared lookup tables.
Checkpoint quantization and CDF/table
generation belong to preparation, not runtime floating-point neural inference.

Prepared artifacts bind source checkpoint hashes, arithmetic version, tensor
contents and entropy tables. Archive metadata and the integer bitstream header
bind those prepared identities. A mismatched runtime/model fails clearly.
Sharing prepared artifacts avoids relying on host-specific table regeneration.

The CUDA Tensor Core path splits each INT16 into unsigned low and signed high
bytes. For reduction length K <= 32768, low-low, combined cross and high-high
sums fit INT32 (bounds 65025K, 65280K, 16384K). Recombination happens in INT64
before rounding. Larger reductions and unsupported cases use generic exact
kernels. See the integer MMA forms in the
[NVIDIA PTX reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).

## Validation scope

See [measured results and exact hashes](UF_INT16_VALIDATION.md). The Tensor Core
encoder produced the same full-video bytes as the generic implementation in
85.82 seconds (about 3.85× faster). It remains slower than FP16.

The generic integer backend completed 300 frames of 1280×720, 30 FPS Bunny at
QI36/QP30, with full reconstruction-hash validation before publication:
452,408 bytes, 39 packets, 330.61 seconds encoding. This establishes integer
reconstruction consistency, not FP16 reconstruction or speed parity.

Validation covers:

- CPU scalar/reference arithmetic versus CUDA, including sums beyond INT32.
- Integer layer implementations for intra and UF temporal reconstruction.
- Integer latent quantization, prior prediction and entropy indexing.
- Persistent prepared artifacts with content validation and model identity.
- Explicit archive runtime selection and automatic matching decode/view paths.
- Real I/P sequences, temporal resets and partial final chunks.
- Fresh-process decode equality and repeated encode reproducibility.
- CPU/CUDA reference comparison; additional physical GPUs remain unverified until
  tested on those machines.
- Full Bunny archive test and quality comparison against the FP16 baseline.
- Fresh FP16 archive verification and archive storage protections.

The portable checker rejects floating tensors during neural inference and
records bitstream, reconstruction and temporal-state hashes:

```bash
./scripts/uf-python scripts/check_uf_int16.py --variant hts --cpu-reference --output runs/int16-reference.json
# On another GPU, with the same prepared files and copied reference report:
./scripts/uf-python scripts/check_uf_int16.py --variant hts --prepared_i IMAGE.int16.pt --prepared_p VIDEO.int16.pt --compare runs/int16-reference.json --output runs/int16-second-gpu.json
```

Additional physical GPU models remain untested. `--device cpu` selects the slow
integer reference backend for archive commands. Use one process on the 8 GB
Jetson; concurrent model instances can exhaust shared RAM.

The native integer extension is isolated as `uf_int16_cuda`; it does not replace
UF's FP16 extension or install anything into the RT environment.
