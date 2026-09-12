# UF size/quality Pareto search

`pareto-search` evaluates one source at a fixed resolution and model while
minimizing the compressed video bitstream and maximizing reconstructed PSNR.
It loads the source and INT16 HT model once, displays a `tqdm` progress bar, and
removes each temporary `.bin` after measuring it. The only retained search
artifact is the requested JSON file.

```bash
./scripts/uf-python main.py pareto-search --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_json runs/uf-int16-htl-144p-pareto.json --resolution 144 --model_structure htl --trials 2000 --initial_samples 800
```

The first 800 unique configurations use balanced categorical Latin sampling.
Every value of QI and QP from 0–63 appears 12 or 13 times in that phase; every
reset interval from 1–60 appears 13 or 14 times. The HT-L intra categories are
`-1, 1, 8, 16, 24, 32, 40, 48, 56`, each appearing 88 or 89 times. HT-S and
HT-L can place I-frames only between eight-frame temporal chunks; LD therefore
uses every period from 1–60. `-1` means no additional I-frame after the first.

The model also exposes an entropy skip threshold. The default search balances
`0, 0.05, 0.10, 0.15, 0.20`; pass `--skip_thresholds` to replace these discrete
values. Each threshold appears exactly 160 times in the initial 800-trial phase.
Normal archive encoding exposes the same control as `--skip_thres`.

The remaining 1,200 trials use genetic crossover and mutation among current
Pareto-front configurations, with random exploration to avoid collapsing into
one quality region. Configurations never repeat. The JSON is atomically replaced
after every trial and includes its configuration, video-byte count, frame-average
YUV420 PSNR, component PSNRs, encode/decode/trial duration, cumulative encoding
time, cumulative search time, status, and current Pareto trial IDs.

Rerunning the same command resumes an interrupted compatible JSON. Source hash,
prepared model identities, search space, seed, and trial counts must match.
Use `--no-resume` to refuse an existing output. PSNR is calculated per frame and
then averaged, using the conventional `6:1:1` Y:U:V weighting used by upstream
DCVC-UF evaluation. Audio is neither encoded nor included in the byte objective.
