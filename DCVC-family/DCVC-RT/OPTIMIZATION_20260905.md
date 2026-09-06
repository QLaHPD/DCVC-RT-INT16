Runtime deployment: `/mnt/to_storage/TOOLS/DCVC-RT-INT16-OPT-MANAGED`. Git sources live under `DCVC-family/DCVC-RT` in `/home/visilionosh/DCVC-RT-INT16`. Generated videos, reference tensors, and compiled libraries referenced below remain on the test host; only source, reusable benchmark scripts, and this summary are versioned. Generated logs, detailed results, and backups also remain local.

The optimized build raises measured throughput by **43% with one worker and 59% with three workers**, while retaining bit-exact encoded output and decoded reconstructions in the verification suite. RAM use is essentially unchanged. The verified extension is installed in the runtime deployment.

These measurements compare against the code and native library present in **this managed tree at the start of September 5, 2026**, which already contained the optimizations described in `OPTIMIZATION_REPORT.md`.

| Production encoder pipeline | Baseline | Updated |
| --- | ---: | ---: |
| One worker | 43.31 fps | 62.11 fps |
| Three workers, per worker | 14.70 fps | 23.38 fps |
| Three workers, aggregate | 44.05 fps | 69.81 fps |
| Peak RSS per worker with three workers | 1.271 GiB | 1.265 GiB |

Three repetitions of each configuration were run with alternating baseline/candidate order. Three-worker processes started behind a barrier, within 19 ms of one another. The per-worker numbers are medians of all worker observations; aggregate rates use total frames divided by the interval from the earliest start to latest finish. Baseline three-worker rates ranged from 14.63–14.85 fps; updated rates ranged from 23.23–23.71 fps.

The timing includes FFmpeg startup and decoding, prefetch, YUV conversion, device transfers, I/P encoding, entropy coding, and atomic output writing. It excludes Python imports and model loading. A generated 160-frame, 176×96, 24 fps `testsrc2` video was used at QP-I 35 / QP-P 14 and reset interval 32. The production frame-rate filter emits 159 frames. Audio is disabled. This is a synthetic workload on this Orin, so the absolute rates can differ with source content, resolution, audio, and system load.

Detailed results are retained locally under `benchmarks/optimization_20260905/`. All 24 worker output files have SHA-256 `b9290d479727c247d5a9d39f069a5cc52b27820d601e31177b86749316a14698`.

Two changes provide the gain:

- Dense stride-1 1×1 and padded 3×3 INT16 convolutions now use integer Tensor Cores on compute capability 8.0 or newer. The kernel splits each signed INT16 into an unsigned low byte and signed high byte. Four byte matrix products reconstruct the INT16 dot product modulo 2^32. The existing signed rounding, bias addition, and final clipping then apply. There is no reduced-precision quantization or floating-point accumulation. This follows the integer operand and fragment definitions in the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832). Unsupported convolution geometries retain their original dispatch. Older GPUs retain the scalar paths; a scalar implementation also handles builds containing only older PTX that are JIT-loaded on newer GPUs.
- Hyperlatent rounding, symbol conversion, rescaling, and clipping now share one CUDA kernel. This removes intermediate tensors, multiple launches, and the synchronous scalar upload performed by the previous rounding helper. The Python fallback remains available when an older extension is loaded.

The Tensor Core implementation allocates shared memory and registers, without persistent packed-weight copies or large global-memory workspaces. It therefore improves throughput without increasing model memory substantially.

Verification completed:

- All **138 convolution regression cases** match tensors captured from the starting library. Cases include full-range and smaller INT16 values, accumulator wraparound, bias/no bias, channel and spatial tails, multiple batches, non-default CUDA streams, grouped/depthwise/strided fallbacks, and noncontiguous inputs.
- The fused conversion matches **all 65,536 possible INT16 inputs**, including half-step rounding and positive clipping boundaries. Noncontiguous and empty inputs also pass.
- All **16 existing production-shape convolution benchmarks** match the saved baseline exactly using the installed extension.
- **304 frames across three generated clips** match complete encoded bytes, every encoded reference-state hash, and every decoded reconstruction hash: moving 176×96 at QP 35/14; moving 320×180 at QP 12/4 with periodic I frames and two entropy coders; and flat 128×72 at QP 63/55. These cover padding, QP shifts, and reference resets. The normal local extension loader was also checked with another complete 160-frame encode/decode comparison.
- A standalone `compute_75` PTX build was JIT-loaded on the Orin and verified against a CPU convolution calculation, exercising the older-PTX fallback for both 1×1 and 3×3.
- The existing **16 application unit tests** pass.

Detailed correctness logs and per-frame hashes are retained locally under `benchmarks/optimization_20260905/`. Physical execution on another GPU architecture was not available.

The changes are in `src/layers/extensions/inference/int16_mma.cuh`, `int_kernel.cu`, `def.h`, `bind.cpp`, `setup.py`, and `src/layers/cuda_inference.py`. Original source and library backups are retained locally under `benchmarks/optimization_20260905/baseline_source/` in the runtime deployment. The Git commit records the source changes.

Validated environment: `/mnt/to_storage/miniconda/envs/dcvc-rt-int16/bin/python`, PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA 12.6, Jetson Orin / compute capability 8.7. The local library is `src/layers/extensions/inference/inference_extensions_cuda.cpython-310-aarch64-linux-gnu.so`, SHA-256 `c8cc0b22d57fe0f537e2589da688319b3c20315b9166dcb48d01d2a1cb85bbbc`. The environment's shared site-packages extension was left unchanged; this project's loader selects the rebuilt local library first. Other platforms should rebuild from source.

The following command reproduces the repeated pipeline benchmark from the runtime deployment, using the saved local baseline and generated fixture. In another checkout, first build a baseline library before applying this change and generate a fixture with FFmpeg; these binary artifacts are intentionally not versioned.

```bash
/mnt/to_storage/miniconda/envs/dcvc-rt-int16/bin/python benchmarks/benchmark_parallel_codec.py \
  --baseline benchmarks/optimization_20260905/baseline_source/inference_extensions_cuda.cpython-310-aarch64-linux-gnu.so \
  --candidate src/layers/extensions/inference/inference_extensions_cuda.cpython-310-aarch64-linux-gnu.so \
  --fixture benchmarks/optimization_20260905/fixtures/testsrc2_176x96.mkv \
  --output benchmarks/pipeline_recheck_20260905 --pipeline
```

The benchmark calls the production `NeuralEncoder` pipeline directly and never invokes channel cleanup. All codec benchmark fixtures were generated with FFmpeg `lavfi` (`testsrc2` or `color`), stored under `benchmarks/optimization_20260905/fixtures/`. No benchmark read or wrote `/mnt/to_storage/DATA`. Checkpoints and their prepared states remain unchanged. Only temporary files created by this work or its tests were removed. Compilation used at most two jobs and benchmarking at most three model workers; no clocks or power settings were changed.
