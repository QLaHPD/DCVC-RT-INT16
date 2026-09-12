# DCVC-UF checkpoints

Place the upstream UF model files directly in this directory:

- `cvpr2026_image.pth.tar` — required image/I-frame model.
- `cvpr2026_video_hts.pth.tar` — HT-S video model.
- `cvpr2026_video_htl.pth.tar` — HT-L video model, if available.
- `cvpr2026_video_ld.pth.tar` — low-delay video model, if available.

The upstream CLI accepts explicit paths, so different supplied filenames can be
used without renaming. Choose `--model_structure hts`, `htl`, or `ld` to match the
video checkpoint. At least the image model and one corresponding video model
are needed for video compression.

Keep original UF checkpoints here. RT checkpoints and RT `.int16prep.pt` caches
are incompatible with UF. UF integerization and its prepared-cache format have
not been implemented yet. Model files in this directory are ignored by Git.
