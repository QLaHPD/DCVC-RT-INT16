# Training implementation and verification record

Implementation scope and local verification for configurable DCVC-RT training.
Usage and configuration are documented in [TRAINING.md](TRAINING.md).

## Implemented

- Versioned YAML/JSON configuration, strict architecture/stage validation, and
  paths resolved relative to the configuration file.
- Configurable image/video widths and repeated block counts, retaining the
  original topology and default checkpoint keys/shapes.
- Differentiable image/video forwards, entropy estimates and rate-distortion loss.
- Exact native INT16 forward with FP32 master weights and surrogate gradients,
  including canonical CPU parameter fusion, integer arithmetic and references.
- Images, frame sequences and one-time original-video preparation; source-disjoint
  manifests, deterministic sampling and checks for changed/missing source frames.
- Staged image/video/pair training, gradient accumulation, temporal gradient
  windows, activation checkpointing, single-node DDP and non-finite update handling.
- Atomic optimizer-boundary checkpoints, configuration/source identity checks,
  per-rank RNG state and exact resume with the original process count.
- Isolated actual-bitstream validation, reconstruction/reference parity checks,
  measured bitrate, PSNR and float-to-INT16 reconstruction comparisons.
- Atomic export of matched I/P bundles, architecture metadata and prepared-state
  bindings to checkpoint/architecture hashes. Existing CLI inference loads these
  architectures without changing `.bin` or `.dcvci` syntax.
- Unified CLI commands, smoke/from-scratch/fine-tuning recipes and user documentation.

## Verified locally (2026-09-11)

- **111 tests passed** in the final complete discovery run, including existing
  inference, scheduler and lifecycle regressions. Log retained locally at
  `benchmarks/results/training-regression-final.log`.
- **Five native CUDA QAT tests also passed separately**: integer accumulator wrap,
  fused operations, changed-weight refresh, I/P forward parity with gradients,
  feature refresh and stale prepared-cache invalidation.
- Default image/video checkpoint schema fingerprints match unmodified commit
  `0833356`. Smaller models execute locally; a larger configuration's parameter
  dimensions/block counts were checked without allocating its GPU parameters.
- Original-video preparation/cache reuse, source-disjoint deterministic sampling,
  changed-frame detection and relative manifest paths passed.
- Resumed I/P training tensors matched an uninterrupted run exactly. Activation
  checkpointing preserved float losses and parameter gradients exactly.
- Two CPU DDP ranks completed staged I/P training with accumulation. A NaN injected
  on one rank caused both ranks to skip the update without changing weights;
  subsequent training remained synchronized and checkpoints held both RNG states.
- A tiny CUDA I/P QAT run completed training, actual image/video validation and
  model export. A second run completed float-to-INT16 stages for both models with
  activation checkpointing. Its interrupted validation resumed from the saved
  optimizer boundary. Artifacts remain under `benchmarks/results/training-qat-*`
  and `benchmarks/results/training-mixed-*`.
- The exported trained bundle passed **65-frame** native validation at base QP
  **0 and 63**, with feature reset interval 32, exact QAT/native values and exact
  encoder/decoder reconstruction and reference state. Results remain at
  `benchmarks/results/training-long-validation/result.json`.
- Export failure left no partial published bundle; existing bundles were retained.
  Real `main.py export-model` and `main.py decode` runs produced the custom models,
  a PNG and the expected six-frame YUV output.
- Live-viewer workers decoded exported images and matched sequential video RGB
  frames while seeking forward and backward. The GUI window itself was not tested.
- All three supplied recipes validated; Python compilation and Git whitespace
  checks passed. Models, datasets, generated media and environments are ignored.

## Hardware and evidence limits

Native arithmetic/training was tested on this Jetson using its NVIDIA CUDA
PyTorch environment. That PyTorch build omits distributed support. DDP testing
therefore used an isolated CPU PyTorch 2.5.1 environment under
`benchmarks/results/ddp-test-env`, preserving the installed NVIDIA environment.

Physical multi-GPU NCCL execution and cross-machine/GPU reproducibility have not
been verified: the five-GPU computer is inaccessible from this machine. The
training guide includes the five-GPU launch command for testing after pulling.

These tests establish implementation behavior on small synthetic workloads.
They do not establish compression quality, convergence, full-size training
throughput or reproduction of the paper's rate-distortion results.
