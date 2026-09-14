# UF INT16 HT-L calibration benchmark

[uf-int16-htl-144p-pareto.json](uf-int16-htl-144p-pareto.json) is the complete,
unchanged result of our earlier 2,000-trial search, completed September 13, 2026
UTC on the Jetson. All 2,000 trials succeeded. The published copy lives outside
the ignored `runs/` directory and ships with the code; models and test media do
not. Original local paths in the metadata record provenance and are not needed
to use the results.

The source was the 10-second, 300-frame Big Buck Bunny sample, resized from
1280×720 to **256×144**, at 30 FPS. Source SHA-256, prepared-model identities,
search seed, parameter domains and measurements are retained in the JSON.
PSNR is mean per-frame YUV420 PSNR with 6:1:1 Y:U:V weighting. Byte sizes are
video bitstreams only, excluding audio and metadata. Encoding seconds are the
codec's measured encoding duration, excluding decoding and download startup.

`encode`/`stream --target-psnr` searches every successful trial with measured
PSNR at least the target, choosing minimum encoding time, then minimum bytes
for a time tie, then trial number. Restricting selection to the 2D size/PSNR
frontier would discard potentially faster choices. This is a speed-first
selection, not simultaneous independent minimization of size and time.

For `--target-psnr 30`, trial **783** is selected:

| QI | QP | Reset interval | Intra period | Skip threshold | PSNR (dB) | Encode seconds | Bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 14 | 2 | -1 | 0.2 | 31.225371 | 3.992191 | 17,830 |

These are reference measurements, not a promise that arbitrary videos reach
the same PSNR or encode at the same speed. The selector does not retrain models,
measure input quality, or run an extra encode/decode pass.
