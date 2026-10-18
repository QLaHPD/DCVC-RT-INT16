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

Keep original UF checkpoints or portable UF prepared `*.int16.pt` files here.
RT checkpoints and RT `.int16prep.pt` caches are incompatible with UF. Use
`main.py prepare-int16` to prepare UF checkpoints, or copy the exact prepared
files used by an existing channel. `join` locates local files by their saved
content identities; `--prepared_i` and `--prepared_p` can select other paths.
Prepared UF files include integer weights and entropy tables and can be used
without the original float checkpoints. Model files are ignored by Git.
