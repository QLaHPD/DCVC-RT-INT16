# DCVC-UF INT16 development

Local development branch for integer inference and portable decoding in
[Microsoft's DCVC-UF](https://github.com/microsoft/DCVC).

**Current status:** the archive CLI supports FP16 (default) and a prepared INT16
runtime selected with `--runtime int16`. Both have completed a full 720p Bunny
encode/decode validation on Jetson. Integer arithmetic also has an independent
CPU reference; additional physical GPU models still need validation.
See [integer runtime and portability](docs/UF_INT16.md),
[archive commands](docs/UF_ARCHIVE.md), and [FP16 baseline](docs/UF_BUNNY_VALIDATION.md).
The [Pareto search](docs/UF_PARETO.md) finds nondominated size/PSNR settings
without retaining its temporary bitstreams.

## Local workspace

- Checkout: `/mnt/to_storage/TOOLS/DCVC-UF-INT16`
- Branch: `uf-int16-managed`
- Dedicated environment: `dcvc-uf-int16`
- Models: **[checkpoints/](checkpoints/README.md)**
- Setup instructions and compatibility status: **[docs/UF_LOCAL_SETUP.md](docs/UF_LOCAL_SETUP.md)**

```bash
./scripts/uf-python scripts/check_uf_setup.py
./scripts/uf-python test_video.py --help
```

The launcher selects the dedicated local interpreter and cache directories. It
removes inherited RT INT16 flags and Python import-path overrides for its child
process. No checkpoints are downloaded automatically.

## Source layout

| Location | Purpose |
|---|---|
| `src/models/image_model.py` | UF intra-frame model |
| `src/models/video_model_ht.py` | UF HT-S / HT-L models |
| `src/models/video_model_ld.py` | UF low-delay model |
| `src/int16/` | Prepared integer models, CPU reference and integer CUDA kernels |
| `main.py`, `src/archive/` | Archive CLI, runtime selection and storage workflow |
| `test_video.py` | Upstream UF evaluation and bitstream coding interface |
| `train_image.py`, `train_video.py`, `training.md` | Upstream UF float training |
| `DCVC-family/DCVC-RT/` | Inherited RT implementation, maintained separately |

Development starts from RT branch commit `468bd15`. This is a separate Git
worktree with its own branch and working files; it shares Git history with the RT
checkout. UF development is published on `uf-int16-managed`; RT development
remains on `int16-managed`.

Keep model files, native binaries, caches, datasets and outputs out of Git. The
existing RT runtime and its prepared INT16 files remain in their original location.
