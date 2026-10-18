# UF integer validation on Jetson Orin

## Shared-folder join implementation — 2026-10-18

Added `join --root ... --channels ID_OR_URL ...`, per-channel profile inheritance
for local encoding and streaming, preflight mismatch reporting, and cooperative
waiting for peer completions/expired leases. Resume checks count completed video
archives first without initializing encoder models. Streaming thumbnail events
are excluded from video totals; local image passes retain their own progress.

Review is limited to Python syntax parsing, `git diff --check` and manual
control-flow inspection, as requested. No automated tests, live join execution,
new model loads, downloads or performance measurements were performed for this
feature. Runtime and multi-machine acceptance remain unverified. The existing
Os Pingos encoding was left running; its saved metadata was read to confirm the
field names and canonical relative paths used by profile recovery.

## Resume check ordering update — 2026-10-17

Subsequent local INT16 runs check existing video archives in a separate first
phase. Each check updates progress; all checks finish before unfinished videos
are scheduled. The existing pipeline, artifact and original-file integrity
checks remain in place. Completed commits are validated on CPU without taking
an encoding claim or initializing encoder models; pending encoding retains
ownership checks. This ordering update applies on the next run.

The measurements below describe the earlier optimization snapshot, which
scheduled unfinished videos first. Its source snapshots and results are preserved.

## Persistent local encoding workers — 2026-10-17

Evidence, the `16af9cf` control snapshot, final Python sources and measurements
are in `runs/python-speed-20261017T190837Z-persistent/`, summarized by
`analysis.json`. Native libraries remain `packed-v9` and `lazy-skip-v2`.

Local INT16 encoding now reuses one checked model pair per worker. Loading is
lazy, so already-completed archives require no model initialization. Video
references are released between jobs; mask caches retain one resolution.
Failed workers are replaced. At most one extra input per worker is hashed and
probed on CPU while current encoding runs. A device/inode/size/mtime/ctime
fingerprint guards reuse of that preparation, and the full final source hash
check remains. The scheduler maps prepared checkpoints, keeps their checked
identities, and releases each tensor set before opening the next checkpoint.
Unfinished inputs start first on resume; completed inputs still receive all
existing archive/source integrity checks. FP16 and streaming retain their
existing worker lifecycle.

These CLI measurements use two workers on the same 1.02 GHz Orin, QI10/QP14,
reset 2, skip 0.2, 144p short edge and 4 FPS. Twelve moving FFV1 clips contain
1,100 frames total, several aspect ratios, and PCM source audio. Times include
startup, model loading, FFmpeg input, Opus mono 6 kbps audio and publication:

| Model | Fresh worker per video | Final persistent workers | End-to-end FPS before → after |
|---|---:|---:|---:|
| HT-L, median of two runs | 57.995 s | 22.780 s | 18.97 → 48.29 |
| HT-S, one run | 54.014 s | 20.154 s | 20.36 → 54.58 |

The HT-L short-video workload is 2.55× faster. The benefit depends on clip length
and startup's share of total runtime. Warmed long-video encoding remains approximately
123 FPS with two workers on the measured 513-frame fixture.

A 96-video, 8,800-frame run initialized models exactly twice, with each PID
encoding 48 videos. After initial warmup, worker RSS stayed at approximately
958–962 MiB across repeated size changes. The scheduler's retained RSS fell
from 566.0 to 392.9 MiB. Relative to model reuse alone, source preparation and
scheduler checkpoint changes reduced that batch from 101.483 to 97.641 s
(86.71 → 90.13 FPS). Median publication-to-next-`encode_started` time fell
from 138.6 to 4.9 ms; this metric ends before input decoding and the first
neural packet.

Every measured HT-L/HT-S video bitstream matched its fresh-worker control,
including all 96 videos in both longer batches. All twelve compared audio
outputs reconstructed to identical PCM hashes through pipes. No decoded media
files were saved. A completed twelve-video batch resumed in 4.715 s with zero
model loads. A mixed batch resumed twelve archives and encoded its one new
video with one model load; the new video was scheduled first.

Two isolated GPU scheduling candidates were measured for three repetitions:
releasing the gate before collecting entropy bytes, and splitting the gate
around entropy work. They measured 122.58 and 122.30 FPS against 122.66 FPS for
the existing schedule, with identical streams. The production gate is unchanged.

## Python checkpoint storage, partial output and seeking — 2026-10-16

These changes follow `packed-v9`. Controls, frozen source files, library
hashes and measurements are in `runs/python-speed-20261016-round7/`.
The CUDA revision remains `packed-v9` and entropy revision `lazy-skip-v2`.
Runtime arithmetic, prepared identities, archive fields and encoded bytes remain
unchanged. No new dependencies or downloaded models are required.

CUDA initialization maps modern prepared checkpoint files while copying each
network to the GPU. Legacy prepared serialization keeps ordinary loading, as
does CPU execution. Complete content hashing and identity checks remain in
place. File signatures are checked after validation and after device copies,
so a checkpoint changed during initialization is rejected. Hashing now reads
the contiguous NumPy buffer directly, avoiding a tensor-sized temporary byte
copy. A 34 MiB synthetic hash workload measured 40.75 → 32.53 ms, with identical
identities across integer dtypes, scalars, empty tensors and transposed views.

Before the direct-buffer hashing change, three alternating HT-L initialization
and 17-frame encode/decode probes measured these medians per worker:

| Measurement | Ordinary loading | Mapped loading |
|---|---:|---:|
| Initialization | 2.419 s | 2.276 s |
| Process RSS after initialization | 689.1 MiB | 481.1 MiB |
| Process RSS after encode/decode | 700.8 MiB | 537.8 MiB |
| Peak process RSS | 1123.1 MiB | 1120.9 MiB |

The retained memory reduction is approximately 163 MiB per worker after that
workload. Startup peak memory is essentially unchanged; CUDA weight allocation
and the number of model sets are unchanged.

Final partial chunks now reconstruct only their valid output frames. HT-S
retains its shared head pairs. A partial reset still reconstructs the padded
eighth frame for the next reference. Ordinary model calls continue to return a
complete chunk. The archive and viewer request partial output only when both
the model and layer implementations support it; saved implementations retain
their full-output path. At 256×144, QI10/QP14, reset 2, skip 0.2, the median of
seven alternating in-memory decode measurements after warmup was:

| Frames | HT-L decode FPS before → after | HT-S decode FPS before → after |
|---:|---:|---:|
| 3 | 29.60 → 35.78 | 35.61 → 43.02 |
| 10 | 57.32 → 65.06 | 71.02 → 80.59 |
| 20 | 80.60 → 85.34 | 101.03 → 106.64 |
| 34 | 86.04 → 90.83 | 109.32 → 115.47 |

These gains concern short clips with a partial final chunk. Their contribution
shrinks on long videos. Reset-disabled cases also matched every pixel hash and
improved 5.8–28.0% on these fixtures.

Viewer seeks decode intermediate P chunks into reference features without
rendering their pixels. Reset references are retained. Conversion and transfer
are performed only for the target chunk. Explicit `output_frames=0` is supported
for integer P decoding; intra decoding still requires its image output. Eight
forward/backward seeks through a 34-frame noisy fixture produced identical
pixel hashes and these median improvements over five repetitions:

| Mode | Reset disabled | Reset 2 |
|---|---:|---:|
| HT-L | 29.2% | 28.5% |
| HT-S | 38.0% | 31.2% |
| LD | 15.2% | 8.6% |

Additional experiments were left outside the runtime. Fusing prior quantization
passed 307 exact checks and improved its isolated operation, but full HT-L
encoding was effectively unchanged: -0.28% at 144p and +0.10% at 288p over five
alternating repetitions. Pinned transfers were flat at 144p-sized partitions
and slower for small transfers. Concatenation occupied approximately 0.33–0.35%
of measured encoding time. Deferred reconstruction weights saved 33 MiB of GPU
allocation while keeping a 231 MiB checkpoint mapping resident and increasing
peak process RSS by 68 MiB in the probe. Explicit heap trimming released only
about 2 MiB at startup. These changes were not added to the runtime.

The frozen optimization snapshot passed all **165 tests** and all **12 bidirectional
HT-L/HT-S/LD compatibility cases**, including the previous golden streams and
pixel hashes. The normal CLI encoded, verified and resumed both fixtures with
`--fps 4 --procs 2`, preserving 34/137 frames, encoded bytes and both originals.
Batch verification completed 24 coordinated GPU chunks without overlap and
removed its temporary lock directory. No decoded media was saved.

The final throughput check uses the same 1.02 GHz Orin, FFV1 input,
QI10/QP14, reset 2, skip 0.2, three encode repetitions and separate in-memory
decoding. The control restores the local `ffn-v7` implementation from the start
of round6; it precedes both `pipeline-v8` and `packed-v9`:

| Dimensions | Frames per worker | Workers | Encode FPS before → final | Decode FPS before → final |
|---|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 111.71 → 118.68 | 98.17 → 107.90 |
| 256×144 | 257 | 2, aggregate | 113.47 → 121.47 | 96.17 → 107.71 |
| 512×288 | 129 | 1 | 30.11 → 31.66 | 27.14 → 29.42 |
| 512×288 | 129 | 2, aggregate | 30.88 → 32.55 | 27.11 → 29.57 |
| 256×144 | 1025 | 2, aggregate | 117.11 → 125.66 | 96.71 → 108.26 |

Initialization and network downloads are excluded from these synthetic FPS
measurements. `audit-final.json`, `final-fingerprints.json` and `final-source/`
record the checked evidence and exact measured implementation. Runtime code and
reproduction instructions are included in this branch.

## Python packed storage between depth blocks — 2026-10-16

`packed-v9` additionally carries packed output between consecutive compatible
depth blocks. The final pointwise projection writes the same completed value
to its ordinary NCHW output and a temporary packed output in one pass. The next
block consumes the packed input directly. The cache is local to one sequential
call, uses the existing size bounds and is discarded at the end of that chain.
Other shapes, layouts and older extensions retain the original execution path.

The full-pipeline settings of the `pipeline-v8` comparison measured the
following medians of three encode repetitions. Decode
values are separate full in-memory verification passes:

| Dimensions | Workers | Encode FPS before → after | Decode FPS before → after |
|---|---:|---:|---:|
| 256×144 | 1 | 117.29 → 118.94 | 106.94 → 107.96 |
| 256×144 | 2, aggregate | 119.70 → 121.35 | 106.52 → 107.54 |
| 512×288 | 1 | 31.33 → 31.64 | 29.05 → 29.40 |
| 512×288 | 2, aggregate | 32.19 → 32.56 | 29.34 → 29.55 |

Streams and decoded hashes match `pipeline-v8`. The promoted implementation
passed all 151 tests and all 12 bidirectional HT-L/HT-S/LD compatibility cases.
The earlier evidence and native build logs are in the round6 run directory;
the installed `packed-v9` library and frozen promoted sources are also saved in
the round7 control snapshot.

## Python block storage and bounded startup memory — 2026-10-16

This round compares the preceding local `ffn-v7` implementation against
`pipeline-v8`, with its initializer, model, layers, arithmetic dispatcher,
archive wrapper and both native libraries restored for the control. Frozen
sources, library hashes and a completion audit are in
`runs/python-speed-20261016-round6/`. Runtime changes and reproduction
instructions are included in this branch; run artifacts remain local.

Canonical depth blocks retain channel-last storage between their internal
convolutions, eliminating three repeated input packs while preserving their
public NCHW input/output. Shape rules retain the existing implementation for
other blocks, layouts and alignment. The existing 16 MiB input-packing and
8 MiB receptive-field limits remain. Reset decoding reuses the last decoded
frame for its unshuffled reference instead of repeating the reconstruction
head. Output conversion and transfer operate on a complete chunk in memory;
single frames retain the existing conversion kernel.

Source conversion, replication to a multiple of 16 and 8x pixel unshuffle use
one tiled CUDA pass. Shared-memory tiling coalesces both reads and writes.
The original public `tensor()` method and older-library fallback retain their
INT16 behavior; the archive encode hook uses bytes when both model and layer
implementations support them. Quantization parameter views are reused.

The entropy extension starts only the requested workers and grows its pool
when parallelism increases, carrying installed tables to the new workers.
The single-stream masked decoder avoids Python mask/gather/scatter temporaries
and has vector-friendly all-active/all-skipped paths. Its revision is
`lazy-skip-v2`; all eight entropy parallelism settings match saved original
bytes. Checkpoint initialization releases each CPU checkpoint after copying
its network, before reading the next. No additional model copies, libraries,
arithmetic identities, prepared identities or bitstream fields were added.

Both existing extensions were rebuilt and installed offline with one compiler
job and CUDA architecture 8.7. The same Orin Nano at 1.02 GHz, moving FFV1
input, HT-L INT16, QI10/QP14, reset 2, intra -1, skip 0.2 and 30 source FPS
were used. Encode values are medians of three repetitions after a 17-frame
warmup, including FFmpeg input and stream writes, excluding initialization.
Decode values are separate full in-memory verification passes, after every
worker finishes encoding.

| Dimensions | Frames per worker | Workers | Encode FPS before → after | Decode FPS before → after |
|---|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 111.71 → 117.29 | 98.17 → 106.94 |
| 256×144 | 257 | 2, aggregate | 113.47 → 119.70 | 96.17 → 106.52 |
| 512×288 | 129 | 1 | 30.11 → 31.33 | 27.14 → 29.05 |
| 512×288 | 129 | 2, aggregate | 30.88 → 32.19 | 27.11 → 29.34 |

HT-L encoding improved 4.0–5.5% and decoding 7.0–10.8%. Peak RSS fell
approximately 73–95 MiB per worker. The sustained two-worker run, with 1,025
frames per worker, measured 117.11 → 124.02 encode FPS and 96.71 → 107.26
decode FPS, with identical streams and pixels. The other UF modes were also
measured with 129 frames per worker:

| Mode | Workers | Encode FPS before → after | Decode FPS before → after |
|---|---:|---:|---:|
| HT-S | 1 | 134.98 → 140.99 | 120.96 → 135.95 |
| HT-S | 2, aggregate | 132.63 → 139.77 | 110.98 → 125.76 |
| LD | 1 | 75.16 → 76.22 | 86.18 → 93.66 |
| LD | 2, aggregate | 77.91 → 78.42 | 89.26 → 95.89 |

All 148 tests passed without skips. All 12 cross-decoding cases match the
previous golden streams and pixels in both directions. The checker now
switches the complete model and layer classes, including helper methods and
cached native block callables. The CLI encoded, verified and resumed both
synthetic originals with `--fps 4 --procs 2`, preserving their previous streams
and 34/137 decoded frames. Batch verification completed 24 coordinated GPU
chunks without overlap and removed its temporary lock directory. No decoded
media was written. These synthetic FPS measurements exclude network and
model initialization; initialization is recorded separately in the reports.

Rebuild both existing extensions after updating:

```bash
MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.7 ./scripts/uf-python -m pip install --no-build-isolation --no-deps ./src/cpp ./src/int16/native
```

## Python FFN orientation and shared decoder scheduling — 2026-10-16

This round compares against published commit
[`b5205e6`](https://github.com/QLaHPD/DCVC-RT-INT16/commit/b5205e6b94ac8348116d7eb864983d245d70059e),
restoring its model, layer dispatcher, archive wrapper and `dispatch-v6` native
library. These runtime and benchmark changes were measured following that push.

Selected feed-forward expansion convolutions now use spatial positions as MMA
rows and output channels as columns. Each lane combines two individually
rounded, saturated and activated channels before one warp shuffle completes
the four-channel reduction. Shape rules retain the older orientation where
it measured faster, including the main 512-channel expansion at 144p.
The existing 16 MiB input-packing bound remains in force. Rejected tile,
occupancy and byte-packing experiments are retained only in the local run
directory. The native revision is `ffn-v7`; its arithmetic identity stays v1.

INT16 HT-L workers sharing a GPU now coordinate complete decode chunks as well
as encode chunks for frames with a short edge of at least 144 pixels. Batch
decoding resolves the runtime per archive and passes the same temporary GPU
gate to its image and video codecs. GPU completion precedes releasing the
gate. Lock paths remain absent from archive metadata and identities. CPU,
HT-S, LD and independent GPUs retain their existing scheduling. No additional
models or dependencies are required, and the extension was built and installed
with `MAX_JOBS=1` and `TORCH_CUDA_ARCH_LIST=8.7`.

Measurements use the same Orin Nano at 1.02 GHz, local moving FFV1 input,
HT-L INT16, QI10/QP14, reset 2, intra -1, skip 0.2 and 30 source FPS.
Encode throughput is the median of three repetitions after a 17-frame warmup,
including FFmpeg input and bitstream writes and excluding initialization.
Decode throughput is a separate full in-memory verification pass per worker
group. All workers finish timed encoding before verification starts.

| Dimensions | Frames per worker | Workers | Encode FPS before → after | Decode FPS before → after |
|---|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 111.44 → 111.64 | 96.68 → 98.21 |
| 256×144 | 257 | 2, aggregate | 112.91 → 113.33 | 87.64 → 96.06 |
| 512×288 | 129 | 1 | 29.48 → 30.18 | 26.42 → 27.23 |
| 512×288 | 129 | 2, aggregate | 29.78 → 30.84 | 23.95 → 27.06 |

Encoding is effectively unchanged at 144p and improved 2.3–3.6% at 288p.
Aggregate two-worker decoding improved 9.6% at 144p and 13.0% at 288p.
The sustained two-worker check, with 1,025 frames per worker and three encode
repetitions, measured 116.64 → 117.09 encode FPS (effectively unchanged) and
88.06 → 96.69 decode FPS (9.8%). HT-S and LD encoding also remained essentially
unchanged, with observed differences ranging from -0.6% to +0.7%. Peak RSS
remained approximately 1.19 GiB per HT-L worker at 144p and 1.23 GiB at 288p.

All 131 regression tests and all 12 HT-L/HT-S/LD cross-decoding cases passed.
The compatibility checker rebinds cached native block callables when switching
libraries and records both library hashes. Both decoding directions, measured
streams and decoded pixels match the preceding round's golden hashes.
Full-range CPU comparisons cover the new kernels, accumulation cancellation,
rounding boundaries, batches, tails, noncontiguous views and two-byte offsets.

The normal CLI encoded and verified two synthetic originals with `--fps 4
--procs 2`, producing 34 and 137 frames and resuming both without re-encoding.
The real batch coordinator additionally verified both archives entirely in
memory across 24 GPU chunks: their lock intervals did not overlap and the
temporary gate directory was removed afterward. Original inputs were retained;
no decoded media files were written. Reports, source/native fingerprints,
frozen controls and the completion audit are under
`runs/python-speed-20261016-round5/`. These synthetic results exclude download
throughput and model initialization.

## Python convolution dispatch and integer boundaries — 2026-10-16

This round compares against the preceding local `packed-v5` implementation,
saved with its source and native library in
`runs/python-speed-20261016-round4/baseline/`. It measures further gains on top
of the preceding optimization round.

Common pointwise projections now fetch two 32-element reduction fragments
together. Shape selection uses the measured 32/64-column tiles and a two-warp
block for small square projections. The four-channel activation reduction
retains the smaller register set where that is faster. Rejected kernel trials
and per-shape timings are retained in the local run directory. The existing
16 MiB input-packing and 8 MiB receptive-field limits remain in force.

Symbol quantization, symbol restoration and reciprocal calculation each use
one integer CUDA pass. Multiplication has scalar, full-tensor and per-channel
broadcast paths that avoid general index calculations. Each INT16 product and
its rounding offset fit INT32; convolution dot products still accumulate
exactly into INT64. All original rounding and saturation boundaries remain.
CPU tensors, other input dtypes and older native libraries retain their
previous operations.

Canonical depth/residual blocks can dispatch their five convolutions through
one native call, using references to the existing prepared weights. This
reduces Python dispatch; the GPU kernels and intermediate saturation points
remain separate. An isolated HT-L comparison measured 111.32 versus
111.36 FPS with the older Python dispatcher, so that change alone has no
detectable HT-L throughput benefit in this workload.

The native revision is `dispatch-v6`. The existing extension was rebuilt with
`MAX_JOBS=1` and `TORCH_CUDA_ARCH_LIST=8.7`. No dependencies, additional model
copies, arithmetic identities, prepared identities or bitstream fields were
added. GPU scheduling uses the preceding implementation's per-invocation
HT-L gate.

The same Orin Nano at 1.02 GHz, local moving FFV1 input, HT-L INT16,
QI10/QP14, reset 2, intra -1, skip 0.2 and 30 source FPS were used. Values
are medians of three repetitions after a 17-frame warmup. Timing includes
FFmpeg input and bitstream writes; initialization and verification are
separate. Both workers finish timed encoding before either starts decoding.

| Dimensions | Frames per worker | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 100.27 | 111.32 | 11.0% |
| 256×144 | 257 | 2, aggregate | 101.48 | 112.69 | 11.0% |
| 512×288 | 129 | 1 | 27.65 | 29.47 | 6.6% |
| 512×288 | 129 | 2, aggregate | 27.96 | 29.80 | 6.6% |

The sustained two-worker run used 1,025 frames per worker and three measured
repetitions. Aggregate throughput improved 104.74 → 116.40 FPS (11.1%),
again with identical streams and pixel hashes. Measured peak RSS was about
1.20 GiB per worker at 144p and 1.24 GiB at 288p. Verification throughput
improved 89.64 → 96.23 FPS at 144p and 25.17 → 26.50 FPS at 288p with one
worker; two-worker verification improved 81.07 → 87.51 and 22.79 → 24.02 FPS.

The same settings with 129 frames per worker also checked the other UF modes:

| Mode | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|
| HT-S | 1 | 123.96 | 134.29 | 8.3% |
| HT-S | 2, aggregate | 121.29 | 134.69 | 11.0% |
| LD | 1 | 66.01 | 75.28 | 14.0% |
| LD | 2, aggregate | 74.45 | 77.75 | 4.4% |

All measured streams and decoded pixel hashes match the saved implementation.
All 125 regression tests and all 12 HT-L/HT-S/LD cross-decoding cases passed.
The checks include full INT16 conversion ranges, every INT8 symbol value,
multiply broadcasts, noncontiguous views, reduction tails, maximal accumulation
and each residual saturation boundary. The normal CLI also encoded, verified
and resumed two local clips with `--fps 4 --procs 2` (34 and 137 frames),
matching the previous decoded hashes. Original synthetic inputs were retained.
Verification hashes decoded frames in memory and writes no decoded media.
Reports, native/source fingerprints and the completion audit are under
`runs/python-speed-20261016-round4/`. These synthetic measurements exclude
network throughput and model initialization.

## Python packed matrices and residual fusion — 2026-10-16

This round compares against published commit `878742e`, including its saved
Python model, layer dispatcher and `ffn-v4` CUDA library. Large pointwise
convolutions can pack INT16 input into channel-last order once and read matrix
fragments directly, avoiding repeated shared-memory transposes and block
barriers. Measured shape rules retain the shared-memory path for smaller
matrices and large square matrices where packing loses time. Each channel-last
packing buffer is bounded to 16 MiB; spatial receptive-field packing retains
its existing 8 MiB bound.

Activation lookup and residual additions now run in the convolution epilogue.
Convolution rounding/saturation and each residual saturation remain separate,
in their original order. A specialized 3×3 depthwise kernel avoids the general
grouped-convolution loops. Reconstruction is queued before waiting for the
existing CPU rANS worker, allowing those operations to overlap.

HT-L workers sharing a GPU now coordinate complete encode chunks within their
invocation. Both input workers remain active. This applies to INT16 frames
with a short edge of at least 144 pixels; LD and HT-S retain concurrent GPU
execution. Each invocation has its own temporary lock directory and a separate
lock per shared GPU. Locks are released when a worker exits, including abnormal
termination. Scheduling paths are absent from archive identities and metadata.

Verification also has a fused integer conversion to cropped YUV420 bytes,
preserving luma rounding and chroma averaging/flooring. Integer pixels require
no finite-value reduction. Decoded frames are hashed in memory and discarded;
verification writes no decoded media. CPU inference and older native libraries
retain the original operations. Prepared weights, entropy tables, arithmetic
identity and bitstream format are unchanged; no additional models or libraries
are required.

The native kernel revision is `packed-v5`. Rebuild the existing extension with
`MAX_JOBS=1` using the command below. The complete regression suite passed all
119 tests, including independent CPU comparisons for packed matrix tails,
batching, maximal accumulation/cancellation, residual saturation order,
noncontiguous views, two-byte storage offsets, depthwise borders and every
INT16 pixel value. CPU-only process checks cover archive/stream coordination,
namespace cleanup, worker death and GPU completion before releasing a slot.

Measured on the same Orin Nano at 1.02 GHz, with HT-L INT16, QI10/QP14,
reset 2, intra -1 and skip 0.2. Local moving FFV1 input uses 30 source FPS.
Values are medians of three repetitions after a 17-frame warmup. Timing
includes FFmpeg input and bitstream writes; initialization and verification
are excluded. A worker barrier now prevents verification from overlapping
another worker's final encode. The control restores the original archive
wrapper as well as its model, layers and CUDA library.

| Dimensions | Frames per worker | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 84.84 | 100.45 | 18.4% |
| 256×144 | 257 | 2, aggregate | 78.01 | 101.70 | 30.4% |
| 512×288 | 129 | 1 | 22.48 | 27.76 | 23.5% |
| 512×288 | 129 | 2, aggregate | 20.37 | 27.96 | 37.3% |

Without chunk coordination, the new kernels measured 92.72 aggregate FPS at
144p and 25.29 at 288p. Final peak RSS was approximately 1.22 GiB per worker
at 144p and 1.27 GiB at 288p. The original control and optimized decoder
verified 69.56 → 89.31 FPS at 144p and 20.15 → 25.22 FPS at 288p with one
worker; aggregate two-worker verification improved 65.13 → 81.03 and
18.24 → 22.76 FPS respectively.

The sustained two-worker check used 1,025 frames per worker and three measured
repetitions with the same verification barrier. Aggregate throughput improved
80.03 → 104.91 FPS (31.1%), with identical streams and decoded hashes.

The final implementation passed all 12 HT-L/HT-S/LD cross-decoding cases using
noisy synthetic input and the saved older native library in both directions.
Measured streams and pixel hashes match. The normal CLI also encoded and
verified two local files with `--fps 4 --procs 2` (34 and 137 frames), retained
both originals, and resumed both archives without re-encoding. Full local
profiles, candidates, frozen baselines, hashes and measurements are under
`runs/python-speed-20261016-round3/`. Real channel throughput also depends on
source decoding and network throughput.

## Python integer convolutions — 2026-10-16

Further optimization of the Python INT16 runtime combines pointwise convolution
with WSiLU and its four-channel reduction. Convolution rounding and saturation
still happen before each activation lookup, followed by the original sum and
saturation. Small spatial convolutions now materialize their receptive fields
once before the integer matrix kernel, with temporary storage capped at 8 MiB.
Pointwise tile selection was also adjusted using measured kernel timings.
Prepared weights, CDFs, arithmetic identity and bitstream format are unchanged.
CPU inference and older native extensions retain the original operations.

Same Orin Nano and 1.02 GHz GPU, same local moving FFV1 synthetic input and
HT-L INT16 settings as the preceding optimization round: QI10/QP14, reset 2, intra -1,
skip 0.2 and 30 source FPS. Each value is the median of three runs after a
17-frame warmup, including FFmpeg input and bitstream writes and excluding
model initialization and verification. The larger test resizes that local
source to 512×288. Two workers share the GPU and process separate copies.

| Dimensions | Frames per worker | Workers | Previous FPS | New FPS | Gain |
|---|---:|---:|---:|---:|---:|
| 256×144 | 257 | 1 | 80.64 | 84.64 | 4.9% |
| 256×144 | 257 | 2, aggregate | 74.28 | 77.98 | 5.0% |
| 512×288 | 129 | 1 | 20.74 | 22.47 | 8.3% |
| 512×288 | 129 | 2, aggregate | 19.03 | 20.37 | 7.0% |

The 144p control restored the saved previous Python model, layer dispatcher
and native library, reproducing 80.62 FPS and 74.20 aggregate FPS. The 288p
baseline also used that complete previous implementation. Native library and
Python source hashes are recorded by the benchmark. The fused activation
alone measured 80.95 / 74.65 FPS; the spatial packing and tile selection account
for most of this round's improvement.

All measured bitstreams and decoded YUV hashes match the previous version.
The 12 HT-L/HT-S/LD cross-decoding cases additionally restore the older layer
dispatcher and CUDA library, and match the saved original fixture hashes in
both decoding directions. They cover skip off/on, resets, repeated I-frames,
partial chunks and padding at changed dimensions. Full-range CUDA arithmetic
is checked against the independent CPU reference, including partial tiles,
batching, ties, overflow bounds and saturation between the fused operations.
All 107 regression tests passed. Two-worker archive CLI encoding and resume
also passed with `--fps 4`; both archives contain 34 frames. The raw synthetic
CLI decode outputs were hashed and removed. Benchmarks and cross-decoding
checks decode in memory and retain only hashes; use `main.py verify` for
subsequent archive checks without writing decoded media.

Rebuild the existing `uf_int16_cuda` extension with `MAX_JOBS=1` after updating.
The native kernel revision is `ffn-v4`; its arithmetic version stays v1.
Reproduction commands and complete comparison options are in the
[benchmark instructions](../benchmarks/README.md#local-python-throughput).
Local profiles, tile trials, frozen baseline source/library, stream fixtures
and raw measurements are under `runs/python-speed-20261016/`.

## Python fused spatial partitions — 2026-10-15

The Python INT16 runtime now uses fused CUDA partition operations for symbol
quantization, entropy indexes and latent reconstruction. Encoding transfers all
spatial partitions together, then removes skipped symbols on the CPU. Decoding
transfers indexes and skip flags together. The arithmetic ID, prepared model
identities, rANS tables and bitstream format remain unchanged. CPU inference
and older extensions without these operations retain the original path.

Measured on the local Orin Nano with its GPU at 1.02 GHz, using a local FFV1
`testsrc2` synthetic video: 257 frames, 256×144, 30 source FPS, HT-L INT16,
QI10/QP14, reset interval 2, intra period -1 and skip threshold 0.2.
Medians below use three repetitions after a 17-frame warmup. FPS includes
FFmpeg input, bounded prefetch and bitstream writes; it excludes model loading
and verification decoding. Two workers process separate copies concurrently
on the same GPU; their aggregate FPS uses the elapsed span across both workers.

| Workers | Before (FPS) | Fused partitions (FPS) | Improvement |
|---:|---:|---:|---:|
| 1 | 72.98 | 80.31 | 10.0% |
| 2, aggregate | 69.46 | 74.21 | 6.8% |

Every measured stream is byte-for-byte identical to its saved pre-change
baseline (10,891 bytes), and decoded YUV SHA-256 hashes match. The 12 additional
cross-decoding cases cover HT-L/HT-S/LD, skip off/on, frequent resets, repeated
I-frames, partial final chunks and changed input dimensions. For every case,
the new decoder reads the original implementation's stream and the original
decoder reads the new stream with identical pixels. All 103 regression tests
passed, including full-range arithmetic and independent CPU checks of the
new partition kernels.

A separate check decoded the saved 257-frame pre-change file directly with the
new decoder and matched its original pixel hash. Forcing the original path
also reproduced the original bitstream and pixels through the older-extension
fallback. The normal archive CLI encoded and batch-decoded two local synthetic
videos with two workers and `--fps 4`; both outputs contained 34 frames and had
identical decoded hashes. Synthetic sources have no YouTube `info.json`; use
`--allow-missing-metadata --keep-originals` when checking their archive inventory.
The baseline control rerun reproduced 72.63 FPS (one worker) and 69.52 FPS
(two workers), consistent with the initial baseline above.

Rebuild the existing extension to enable the fast path:

```bash
MAX_JOBS=1 ./scripts/uf-python -m pip install --no-build-isolation ./src/int16/native
```

See [the reproducible synthetic benchmark](../benchmarks/README.md#local-python-throughput).
Local raw measurements, pre-change model source and compatibility fixtures are
under `runs/python-speed-20261015/`. These FPS figures describe this workload
and hardware; download/network throughput and other resolutions were not part
of this measurement.

## HT-L 144p indexing and CUDA tiles — 2026-09-12

Compared with `07ffd08` using the same initialized HT-L model, QI36/QP30,
256×144, 300 Bunny frames and one worker. Three runs per mode alternated;
each encoded bitstream matched the original baseline byte for byte.

| Measurement | Previous code | Cached positions and tiled kernels |
|---|---:|---:|
| Full input/encoding pipeline, median FPS | 55.61 | **65.91** |
| Preloaded input, median codec FPS | 64.50 | **72.23** |

The full pipeline improves by **18.5%** and the preloaded codec by **12.0%**.
Full-pipeline defaults use two decoder threads in the previous version and
four here on this six-CPU Jetson; both use eight-frame bounded prefetch.
The new default divides available CPUs among workers, reserves one CPU per
worker for neural dispatch, and limits input decoder threads to 1–4.
Explicit `--input_threads` overrides this. FP16 defaults remain unchanged.
Encoding FPS excludes model loading and the subsequent validation decode.

The prior caches ordered flat mask positions for one shape, replacing repeated
boolean indexing with indexed gathers. Pointwise CUDA integer convolutions use
64-element reduction tiles or wider spatial tiles selected by shape. Generic
and non-pointwise paths retain their original behavior. Arithmetic, prepared
identities, artifact naming and entropy-symbol order are unchanged. The final
300-frame decode matches the original YUV hash below.

A separate CLI run using the new defaults encoded all 300 frames at 64.09 FPS
and passed full validation with identical compressed bytes and decoded pixels.
All 23 unit tests passed, including both tile paths, odd reduction tails,
multi-batch tensors, accumulator limits, mask partition order and CPU budgets.
HT-L, HT-S and LD passed CPU/CUDA checks against their earlier reference
reports, including reset states and feature-only encoder parity.
A 17-frame 720p HT-L CLI run with resets and I-frames every eight frames also
encoded and validated with the exact earlier bitstream and reconstruction
hashes. It emitted Jetson NvMap allocation warnings but completed successfully;
this short test does not establish memory headroom for multiple workers.

Rebuild `src/int16/native` after updating; the setup checker reports `tiled-v2`.
Existing prepared model files are reused. The indexing and input changes are
portable; tile heuristics were measured only on this Jetson Orin. Other GPUs
need their own performance and cross-device consistency checks.

CUDA graph prototypes added cold-start cost and memory pressure for a small
warm-run gain. An exact four-way INT8 BLAS decomposition was substantially
slower than the custom kernel. Neither prototype is part of the runtime.

## Earlier HT-L 144p optimization (`07ffd08`) — 2026-09-12

Same Bunny source, INT16 HT-L, QI36/QP30, 256×144 at 30 FPS, all 300 frames,
one process. Prepared model and arithmetic identities are unchanged.

| Full input/encoding pipeline | Median encode FPS (3 runs) |
|---|---:|
| Previous behavior: full P reconstruction, one input thread, no prefetch | 29.44 |
| Optimized defaults: required features/reset only, two input threads, bounded prefetch | **55.95** |

This is about **1.90× faster** in a six-run alternating comparison using the
same initialized model. Individual optimized runs measured 55.89–56.35 FPS.
Encoding FPS excludes model loading and the subsequent validation decode. The
earlier standalone CLI baseline measured 28.85 FPS; the optimized standalone CLI
measured 55.35 FPS. Four input threads with prefetch measured 57.79 FPS in one
separate run; that version used two by default to limit per-worker CPU usage.

Measurements separating the causes:

- With all input frames preloaded, skipping unused P-frame pixel reconstruction
  raised median codec throughput from 49.67 to 64.53 FPS (three runs per mode).
- Reading/resizing the input alone improved from 38.38 to 59.81 FPS with two
  decoder threads. Pixel hashes matched across tested decoder/filter settings.
- Bounded read-ahead then overlaps input work with GPU encoding. All generated
  full-video bitstreams match the original HT-L baseline exactly:
  `42e73b602cfc44fe5311224d119d043ed0bb6148d36b3021c5b581fa404cfbb1`.
- Full decoded YUV remains
  `5fca3d1b3961f2ca3678dc704f0e34e23c228dcb5f792c6cb32c90123de4444c`.

The default archive command enables these changes automatically:

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/optimized-htl --runtime int16 --model_structure htl --resolution 144 --qi 36 --qp 30 --procs 1
```

The arithmetic checker now compares feature-only encoding against full
reconstruction, including encoded bytes and reference/memory/context at resets.
Input tests cover frame order/content, repeated EOF, decoder failure, closing a
full prefetch queue, and resuming with changed performance settings. The changes
use standard CPU threading and skip unnecessary neural work; they are not
Jetson-specific, although these throughput measurements are from Jetson Orin.

FP16 keeps its original input defaults (one decoder thread, no prefetch). A
720p FP16 test with the larger input buffers exhausted this Jetson's memory;
with the original settings, all 300 frames encoded and validated with the exact
original FP16 bitstream. Faster input defaults therefore apply automatically
only to INT16. Early input cancellation stops the owned read-only FFmpeg child
immediately, avoiding its SIGTERM flush delay when the pipe is full.

Final validation: all 20 unit tests passed; HT-S, HT-L and LD passed CPU/CUDA
state checks against pre-optimization reference reports. A partial 17-frame
720p HT-L archive with resets, periodic I-frames and a padded final chunk also
encoded and validated successfully. Fixing early shutdown preserved its exact
bytes and pixels while removing the observed 10-second input-close delay.
The Bunny source remained unchanged. No native rebuild or model re-preparation
is needed for these optimizations.

The following baseline sections describe code before these optimizations
(published at `a938b27`).

## HT-L 144p baseline — 2026-09-12

HT-L is the focus for subsequent optimization. This run used INT16 Tensor Core
inference, QI36/QP30, reset interval 32, one process and all 300 Bunny frames.
1280×720 was resized to 256×144, preserving 30 FPS. Prepared caches were reused.

| Measurement | HT-L | Earlier HT-S baseline |
|---|---:|---:|
| Encoding loop | **28.85 FPS** / 10.400 s | 32.23 FPS / 9.308 s |
| Validation decode | **55.54 FPS** / 5.401 s | 76.43 FPS / 3.925 s |
| Bitstream | 28,583 bytes | 28,293 bytes |
| Frames / packets | 300 / 39 | 300 / 39 |

All frames passed validation. These are separate single-run stage timings;
encoding FPS excludes model loading/preparation and validation. No inference
changes were made for this benchmark.

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/uf-int16-htl-144p-baseline-20260912 --runtime int16 --model_structure htl --resolution 144 --qi 36 --qp 30 --procs 1
```

Validated raw YUV SHA256:
`5fca3d1b3961f2ca3678dc704f0e34e23c228dcb5f792c6cb32c90123de4444c`.

## 144p baseline — 2026-09-12

Current Tensor Core INT16 runtime, HT-S, QI36/QP30, reset interval 32,
one process, all 300 Bunny frames. Original 1280×720 resized to 256×144;
the source's 30 FPS was preserved. Existing prepared caches were reused.

| Measurement | Result |
|---|---:|
| Encoding loop | 9.308 s / **32.23 FPS** |
| Validation decode | 3.925 s / **76.43 FPS** |
| Bitstream | 28,293 bytes / 39 packets |
| Validation | All 300 frames passed |

These are single-run stage timings, excluding model loading/preparation from
encoding FPS and measuring validation separately. No inference changes were
made for this benchmark.

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/uf-int16-144p-baseline-20260912 --runtime int16 --model_structure hts --resolution 144 --qi 36 --qp 30 --procs 1
```

Code-inspection candidates for subsequent optimization:

1. Avoid reconstructing all P-frame pixels during encoding when only the
   reference features are needed; retain necessary reset-frame reconstruction.
   `IntegerModel.compress()` currently calls the full reconstruction path.
2. Capture suitable fixed-shape neural sections in CUDA graphs to reduce Python
   and launch overhead; keep entropy transfers outside capture where required.
3. Fuse prior masking, quantization and symbol/index packing, and reduce separate
   synchronous CPU transfers for the entropy coder.

These are candidates, not measured gains. Any change must preserve the existing
bitstream and reconstructed-state hashes.

## 720p validation

Test source: `Big_Buck_Bunny_720_10s_30MB.mp4`, 300 frames, 1280×720,
30 FPS. HT-S, QI36/QP30, reset interval 32, original resolution/FPS.
Source SHA256: `a2c61fa8bdb4381a28a06a4df53fdda35e35dd49082b99bf199e3125be38f3b9`.

| Measurement | FP16 baseline | INT16 generic | INT16 Tensor Core |
|---|---:|---:|---:|
| Bitstream bytes | 434,740 | 452,408 | 452,408 |
| Packets | 39 | 39 | 39 |
| Encode seconds | 11.418 | 330.610 | 85.819 |
| Average PSNR against source (dB) | 34.422331 | 34.253348 | 34.253348 |

The INT16 stream includes a 72-byte runtime/prepared-identity preamble. Its
roughly 4.1% larger size and 0.17 dB lower PSNR are measurements on this one clip,
not a general rate-distortion claim. Integerization changes the codec's numerical
behavior; it does not reproduce the floating model's exact reconstruction.

The Tensor Core decoder decoded the generic integer bitstream in 78.148 seconds
and produced exactly the same validated raw YUV hash:
`8326cf7aabe07ac4f5806cadb08ade4e79dce30373e506e70908f1573e1a1852`.
The exported FFV1 Matroska file was independently counted as 300 frames at 720p30.

The Tensor Core encoder also completed all 300 frames, including validation,
and produced a byte-identical bitstream to the generic encoder, SHA256:
`7b036185f86f7100c356dd9fc8f4b4f26152aa0917053e349159337c21e07cd0`.
Encoding is about 3.85× faster than the generic integer implementation, but
still substantially slower than the optimized FP16 baseline. These are separate
single-run measurements, not controlled statistical benchmarks.

Arithmetic tests cover independent scalar/CPU convolution, exact accumulation
beyond INT32, reduction lengths 32768 and 32769, cancellation, odd tiles,
unsaturated outputs, general/strided/grouped convolutions and fused integer
operations. Identity tests reject changed prepared contents, wrong source/model
identities, nonfinite checkpoint parameters and conflicting bitstream runtimes.
Archive tests cover atomic publication, interrupted validation, cooperative
claims, pipeline mismatch, resume and exact-path cleanup.

A separate real-model fixture passed folder encoding with Opus audio, periodic
I-frames, unchanged-resolution handling, resume without changing the bitstream
hash/mtime, CPU verification, forced-runtime rejection and live decoding through
ffplay's dummy SDL driver. Both encoding and decoding used explicit prepared
files with deliberately nonexistent floating-checkpoint paths. Cleanup preview
preserved the generated source; confirmed cleanup validated and removed only
that fixture. The user's Bunny source hash remained unchanged.

A two-frame Bunny archive encoded and validated on the CPU reference at 56×32
also passed fresh CUDA verification, including a padded final temporal chunk.
All 17 unit tests passed; the integer arithmetic subset was rerun successfully
after adding maximal low-byte accumulation at the Tensor Core reduction bound.

The HT-S consistency checker matched the pre-optimization bitstreams and all
reconstruction/reference/memory/context hashes on CUDA and the CPU reference,
including temporal resets. HT-L at 32×32 and LD at 48×48 also passed CUDA/CPU
bitstream reconstruction and temporal-state checks. The checker rejects
floating-point tensors during inference. Fresh FP16 verification still matches
the original 300-frame reconstruction hash:
`7287c2dc7c724900f215955f8d20bd378b22aefd32452be99c0232dba341689d`.
These tests ran in the dedicated UF environment with NVIDIA PyTorch
2.5.0a0+872d972e41.nv24.08 and CUDA 12.6. Other physical GPU models remain
unverified; run `scripts/check_uf_int16.py --compare` there with the same prepared
files. Native binaries must be rebuilt for the target environment.

Initial generic runs exhausted the 8 GB Jetson's shared RAM. Fused integer
elementwise kernels eliminated large intermediate tensors; the full generic
encode and validation then completed with one worker. Use `--procs 1` on this
machine. Test artifacts and models remain local and are excluded from Git.
