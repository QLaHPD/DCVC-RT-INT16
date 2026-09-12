# UF local project preparation

## Locations

Project: `/mnt/to_storage/TOOLS/DCVC-UF-INT16`

Environment: `/mnt/to_storage/miniconda/envs/dcvc-uf-int16`

Place the downloaded UF models in:

```text
/mnt/to_storage/TOOLS/DCVC-UF-INT16/checkpoints/
```

Video coding needs the UF image checkpoint plus the matching HT-S, HT-L or LD
video checkpoint. See [the checkpoint instructions](../checkpoints/README.md).
Do not copy RT checkpoints or their integer preparation caches here.

The supplied paper is copied locally to `papers/dcvc-uf.pdf` and ignored by Git.
The pinned CUTLASS v4.4.1 source is under `third_party/cutlass`, also ignored.
Environment, dependency and diagnostic logs are under `.local/logs`.

## Running commands

From the project directory:

```bash
./scripts/uf-python scripts/check_uf_setup.py
./scripts/uf-python test_video.py --help
```

The wrapper also works using its absolute path from another directory. It reads
its interpreter from the ignored `.local/python` file. On another machine, set
`UF_PYTHON` to that machine's dedicated UF Python executable or create that file.
The wrapper prevents inherited `PYTHONPATH` and RT INT16 environment flags from
selecting RT code or making the floating-point UF runtime appear integerized.
It puts compiler/runtime caches and temporary files under this checkout's `.local/`.

For an interactive shell, `conda activate dcvc-uf-int16` selects the environment.
Prefer the wrapper for commands when the shell has existing RT environment flags.

## Jetson compatibility status

The isolated environment was cloned from the working NVIDIA Jetson environment
so that its CUDA-enabled PyTorch remains available. The cloned **RT native
extensions were removed from the UF environment**; they must not be reused by UF.
The original RT environment was not changed. UF's Python requirements are installed
without upgrading/replacing NVIDIA PyTorch.

Local baseline:

- Python 3.10
- NVIDIA PyTorch 2.5.0a0, CUDA 12.6
- Jetson Orin, SM 8.7

Upstream UF documents Python 3.12, PyTorch 2.9.1 and CUDA 13.0 as its tested stack.
The local entropy-extension package now permits Python 3.10 and has built successfully.
CLI/training f-strings were adjusted to parse under Python 3.10 without changing behavior.
**Both UF native extensions are built and installed locally.** Their stream/guard
includes now use the narrow c10 headers, avoiding unused sparse/BLAS development
header dependencies on this Jetson. The full 300-frame Bunny test passed with HT-S FP16 on this older NVIDIA
stack, including fresh-process decoding and resume. See [results](UF_BUNNY_VALIDATION.md). Source imports and CUDA availability do not prove native
codec compatibility. No driver/toolkit upgrade or change to the RT installation
was made.

The setup checker verifies the native extensions. Use the stricter gate before
attempting bitstream tests:

```bash
./scripts/uf-python scripts/check_uf_setup.py --require-native
```

This fails when UF's own native modules are missing. It also
rejects extensions loaded from outside the selected environment and checks for
UF-specific proxy classes. A successful import gate still needs real checkpoint
encode/decode testing.

On a supported UF stack, the upstream build commands are:

```bash
./scripts/uf-python -m pip install --no-build-isolation ./src/cpp
MAX_JOBS=1 ./scripts/uf-python -m pip install --no-build-isolation ./src/layers/extensions/inference
```

The UF CUDA build now honors an explicit `MAX_JOBS`. Its automatic default scales
with available RAM instead of forcing eight compiler jobs on a small Jetson.
CUTLASS-heavy compilation can require substantial memory even with one job.

## Integer runtime

Build the entropy extension above, then the separate integer CUDA extension:

```bash
MAX_JOBS=1 ./scripts/uf-python -m pip install --no-build-isolation ./src/int16/native
./scripts/uf-python scripts/check_uf_setup.py --require-int16-native
./scripts/uf-python main.py prepare-int16 --model_structure hts
```

The integer implementation has its own arithmetic contract, prepared weights,
lookup tables and entropy CDFs. See [UF INT16](UF_INT16.md) for commands and
validation limits. Build native extensions on each target machine; transfer the
same prepared model files for portable decoding. RT prepared files and the RT
QAT trainer do not apply to UF.

UF code is maintained on `uf-int16-managed`. The RT branch and managed RT runtime
remain separate. Models, native builds, caches and test media stay local.
