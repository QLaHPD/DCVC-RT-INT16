# CPU entropy optimization (2026-09-08)

Baseline: `9d82cb4` on Jetson Orin Nano, with the existing INT16 GPU optimizations.

## Changes

- Precompute `floor(2^32/frequency)` in each rANS table entry. After unchanged
  byte renormalization, a 64-bit integer multiply estimates the quotient. The
  estimate is exact or one too small; a remainder check corrects it. Frequency
  one is handled directly. This produces the same unsigned state as the original
  division and remainder operations. Unused zero-frequency table entries are
  not divided during initialization.
- Emit escape/bypass bits directly in their original reverse order, eliminating
  a temporary heap-allocated vector for every escaped symbol.
- Flush the second encoder only when two-coder mode is enabled, avoiding its
  unnecessary thread wakeup in ordinary single-coder mode.

These changes use portable C++ integer operations, with no Jetson-specific
intrinsics. They require rebuilding the CPU extension. No new runtime switch,
model conversion, bitstream version, or decoder migration is needed. Each
encoder table entry uses four additional bytes for its reciprocal. The existing
threading and GPU-to-CPU transfer protocol otherwise remain in place.

## Measurement

`benchmark_entropy_cpu.py` captures real CDFs, Y/Z symbols, coder mode, and output
bytes while the synthetic codec runs. It separately records GPU transfer calls
and native enqueue/finish calls. Transfer timings include GPU dependency waits;
they are not interpreted as CPU computation time.

CPU-only replay then registers the captured CDFs outside the timed region and
repeats reset, enqueue, flush, and byte retrieval. Every stream is checked against
the original bytes. CPU process time includes all threads in the process.

Initial unrestricted timings varied with the `schedutil` governor. For the
controlled comparison, both CPU policies temporarily used `performance` within
their existing maximum-frequency limits. A shell EXIT trap restored the original
governors; both were verified as `schedutil` afterward. Each variant ran five
trials, with 200 timed repetitions of the 40-frame corpus per trial. Three warmup
repetitions were excluded. Baseline/candidate ordering alternated between trials.

| Captured corpus | Metric per frame | Baseline median | Candidate median | Reduction |
| --- | --- | ---: | ---: | ---: |
| 176x96, one coder | CPU time | 0.1070 ms | 0.0727 ms | 32.1% |
| 176x96, one coder | Elapsed time | 0.1020 ms | 0.0740 ms | 27.4% |
| 320x180, two coders | CPU time | 0.3274 ms | 0.2310 ms | 29.5% |
| 320x180, two coders | Elapsed time | 0.1849 ms | 0.1355 ms | 26.7% |

All five paired trials favored the candidate in each corpus. The two-coder
corpus uses QP I/P 0/45, intra interval 16, and reset interval 8. The one-coder
corpus uses QP I/P 35/14 and reset interval 32.

These reductions apply to the CPU entropy stage. It is only a small fraction of
the whole GPU codec's frame time, so a similar percentage increase in overall
encoding FPS is not claimed. The figures are specific to this machine and these
captured symbol distributions. Asynchronous transfer batching and CPU/GPU overlap
were not changed by this patch.

## Exact-output checks

- The original division-based rANS update and the reciprocal update match in
  8,388,480 state/renormalization-byte cases spanning every frequency 1..65535,
  threshold boundaries, and pseudorandom 32-bit states.
- Twelve synthetic entropy streams exercise sparse escapes across all signed
  byte values, both signs of offsets, mixed Z/Y tasks, and repeated one/two-coder
  transitions. Streams match the old extension and decoded symbols match inputs.
- Both captured corpora match original bytes throughout all CPU replay trials.
- Full 40-frame codec runs at 176x96, 320x180, and padded 128x72 match baseline
  bitstreams, reference features, every decoded frame, and combined pixel hashes.
  The latter two use periodic I-frames, resets, and two coders. QP pairs are
  respectively 35/14, 0/45, and 63/0.
- All 47 existing unit tests pass.

## Reproduction

Activate the codec environment and build the CPU extension with `src/cpp/setup.py`
or use the repository's `build_native_extensions.sh` to rebuild both extensions.
Keep the old CPU library at a separate path for reference comparisons.

```bash
python benchmarks/benchmark_entropy_cpu.py \
  --capture /tmp/entropy.pkl --report /tmp/capture.json -- \
  --fixture /path/to/synthetic.mkv --frames 40 --output /tmp/reference_codec

python benchmarks/benchmark_entropy_cpu.py \
  --replay /tmp/entropy.pkl --cpp-extension /path/to/candidate.so \
  --iterations 200 --report /tmp/replay.json

g++ -std=c++17 -O3 benchmarks/check_entropy_arithmetic.cpp -o /tmp/check_entropy
/tmp/check_entropy

python benchmarks/check_entropy_streams.py --extension /path/to/old.so --save /tmp/streams.npz
python benchmarks/check_entropy_streams.py --extension /path/to/candidate.so --compare /tmp/streams.npz
```

Only replay trusted local captures: their pickle format is intended for development
benchmarks. Local evidence lives under `benchmarks/cpu_optimization_20260908/` on
the development machine. Captures, models, media, generated streams, and compiled
binaries are excluded from Git.
