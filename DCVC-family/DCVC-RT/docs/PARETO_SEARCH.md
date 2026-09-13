# RT INT16 Pareto search

Run from `DCVC-family/DCVC-RT` in the `dcvc-rt-int16` environment:

```bash
DCVC_USE_INT16=1 DCVC_INT16_AUTOTUNE=1 DCVC_INT16_FAST_INPUT=1 python main.py pareto-search --input_file /path/to/Big_Buck_Bunny_720_10s_30MB.mp4 --output_json runs/rt-int16-bunny-144p-pareto.json --resolution 144 --trials 2000 --initial_samples 800 --cuda_idx 0
```

The first 800 unique configurations balance every QI/QP value from 0–63,
every reset interval from 1–60, and every intra period from 1–60 plus `-1`.
Unlike UF HT-L, RT codes one frame per packet and supports all these periods.
It preserves the production encoder's per-frame QP offsets, so the actual
packet QP can exceed the base QP (the released model has extra trained entries).
For all-intra encoding, P-frame QP and reset settings do not affect the result.

The additional search dimension is the existing encode `--force_zero_thres`
control, sampled at `0, 0.05, 0.10, 0.15, 0.20` by default. Search value zero
means skipping is disabled (normal encode default `None`). Set
`--skip_thresholds` to choose another discrete set. When reproducing a nonzero
threshold, supply the same `--force_zero_thres` to both RT encode and decode.

After the balanced phase, genetic crossover/mutation of current nondominated
configurations plus 15% global exploration targets minimum bitstream bytes and
maximum PSNR. Completed configuration tuples are never repeated. Adaptive
randomness is seeded by trial number, including after resume.

Source frames are decoded once to memory using bicubic resizing, exactly as in
the UF Pareto search, with the source frame rate preserved. This deliberately
differs from RT folder encoding's default fast-bilinear scaling. To reproduce a
search point with normal RT encoding, first resize the original with bicubic to
the search dimensions and pass that input at unchanged size. Audio is omitted.
PSNR is measured on the actual decoded 8-bit YUV420 output against those source
frames: mean per-frame PSNR, with 6:1:1 Y/U/V weighting. Source and raw-frame
hashes, checkpoint/prepared hashes and input geometry are stored in the JSON.

Each trial calls the production RT encoder and decoder, reusing initialized
models. Encoding time includes the production loop, synchronization, and any
first-use autotuning, but excludes source loading, model loading and decoding.
The JSON also records decode/trial durations and cumulative encoding/search
times. Timings are observations, not optimization objectives.

The progress bar reports trial count, frontier size, last PSNR and failures.
One JSON is atomically replaced after each trial. Temporary bitstreams and
production logs are deleted after each trial; no reconstruction videos are
retained. The same command resumes an interrupted matching JSON. Source/model
or search-setting mismatches are rejected. Codec errors are recorded and stop
the process so a broken CUDA context cannot consume the remaining trial budget.
