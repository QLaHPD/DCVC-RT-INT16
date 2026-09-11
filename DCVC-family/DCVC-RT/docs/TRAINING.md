# Training DCVC-RT models

Training belongs to the same `main.py` interface as encoding and decoding. It
supports image/I-frame models, video/P-frame models, and an I-then-P pipeline.
Widths and repeated block counts are configurable; the default architecture
retains the existing checkpoint keys and tensor shapes.

These are experimental training recipes. They are not the authors' original
training implementation, and the example epoch counts do not establish that a
run will reproduce the paper's compression results. Small smoke models test the
pipeline, not compression quality.

## Installation and first run

Use the existing CUDA-enabled PyTorch environment and native extensions, then:

```bash
python -m pip install -r requirements-training.txt
python main.py train --config configs/train/smoke.yaml --check-config
```

Edit source locations in the YAML before preparing data. **Relative paths are
resolved against the config file**, not the shell's working directory.

```bash
python main.py prepare-training-data --config configs/train/smoke.yaml
python main.py train --config configs/train/smoke.yaml
```

Available starting points:

| Config | Purpose |
|---|---|
| `configs/train/smoke.yaml` | Small models, two optimizer steps per model, real INT16 validation |
| `configs/train/base.yaml` | General image/video training from scratch, float followed by INT16 |
| `configs/train/qat_finetune.yaml` | INT16-aware fine-tuning of the existing pretrained I/P checkpoints |

Training does not require `DCVC_USE_INT16=1`, `DCVC_INT16_AUTOTUNE=1`, or CUDA graph
flags. Arithmetic is selected explicitly by each stage's `mode: float` or
`mode: int16`. Inference flags continue to apply to normal encoding and decoding.

## Configuration

Unknown keys and incompatible architectures/stages are rejected before models
are allocated. The resolved configuration is saved in the run directory.

| Section | Controls |
|---|---|
| `task` | `image`, `video`, or `pair` |
| `model` | Image/video architecture and optional `init_image` / `init_video` checkpoints |
| `data` | Sources, manifest/cache destinations, split fraction, workers, batch per GPU, optional resize |
| `stages` | Model, arithmetic, epochs, LR range, crop, sequence length, temporal gradient window |
| `optimizer` | AdamW settings, gradient clipping/accumulation, activation checkpointing |
| `quantization` | Fixed runtime scales and optional zero-symbol threshold |
| `loss` | Lambda ranges, YUV/RGB mixing, channel and temporal distortion weights |
| `validation` | QPs, frequency, source/frame limits, actual-bitstream checks, feature-reset interval |
| `output` | Run directory and checkpoint/log frequencies |

Image defaults are 368 encoder/decoder channels, 256 latent channels, and 128
hyperprior channels. Video defaults are 256 feature channels, 320 reconstruction
channels, and 128 latent/hyperprior channels. Both expose the counts of their
existing repeated block stacks. Mandatory transitions and the codec topology
stay fixed. Image latents must divide into four groups and video latents into two.

For example, the following makes a smaller image model:

```yaml
model:
  image:
    channels: 192
    latent_channels: 128
    hyper_channels: 64
    encoder_blocks: 3
    decoder_blocks: 6
    fusion_blocks: 2
    spatial_blocks: 2
```

A changed architecture starts from scratch. Fine-tuning requires checkpoints
whose dimensions and block counts match exactly. No tensor slicing or partial
loading is performed. QP banks retain 64 base entries; video also retains its
eight extra entries for hierarchical offsets.

Stages are explicit. `crop_size` is `[height, width]` in multiples of 16;
`sequence_length` includes the initial I-frame. `temporal_gradient_length` counts
P-frames per backpropagation window. References carry forward within a clip but
their gradients stop at window boundaries; gradients are backpropagated and
freed after each window. `max_steps` optionally bounds a stage by optimizer-step
attempts, and `samples_per_epoch` can increase sampling from a small dataset.

`pair` finishes all image stages before training video. The image model is
frozen during video training. A video-only run requires `model.init_image`.

## Data preparation

`data.sources` accepts multiple entries:

```yaml
sources:
  - type: images
    path: /data/images
  - type: frame_sequences
    path: /data/vimeo/sequences
  - type: videos
    path: /data/original-videos
```

Image sources are discovered recursively. Frame-sequence sources group images
by directory and sort numeric filenames naturally; the repository's
`description.json` sequence/frames format is also supported. Each sequence must
have consistent dimensions. A stage uses only sequences long enough for its
configured length. Long-clip stages therefore require actual long sequences.

By default, original videos are decoded once by FFmpeg into cached YUV420 `.npz` frames.
Frames retain their stored orientation; container rotation metadata is not applied.
Set `data.video_loading: direct` to index videos without extracting a frame cache.
`prepare-training-data` then probes metadata in parallel and writes a small manifest;
training and validation seek into the source videos and decode only the requested
consecutive frames in DataLoader workers. `configs/train/tiny_vimeo.yaml` uses this
mode. Source videos must remain available and unchanged throughout the run.
The default `cache` mode retains the existing extracted-frame workflow.

Direct sampling seeks by timestamp using average FPS and container frame counts
(duration-derived counts when absent). Variable-rate media therefore uses approximate
frame positions, without resampling or duplicating decoded frames. A short tail retries
an earlier position and then the beginning deterministically; a video that cannot
provide the requested clip fails explicitly. Spatial transforms are shared across
the decoded clip. Direct mode trades disk space and upfront extraction time for
repeated CPU decoding; `data.num_workers` controls prefetch workers per GPU.

Cache preparation runs one decoder at a time, closes its streams, and publishes a
completed cache atomically. Cache identity includes the source file identity
and preprocessing dimensions. Completed caches are reused. Sources are retained.

The manifest records dimensions, ordered frame paths, source groups, split, and
frame file identities. Entire source groups are assigned to either training or
validation before temporal sampling. Set an entry's `split: train` or
`split: validation` to use a predefined split. At least one source is needed in
each split. Keep clips originating from one video together in one sequence;
preparation cannot infer that separately supplied copies came from the same video.

Rejected media are listed in the neighboring `.report.json`. Missing or changed
manifest frames abort training rather than silently changing different GPU
ranks' samples. Preparation is explicit; training does not download datasets.

Sampling uses shared spatial crops/flips across each clip, BT.709 YUV input, and
edge padding for undersized images. `resize_short_edge` is optional and preserves
aspect ratio; no resize is performed by default.

## INT16-aware training

The original DCVC-RT paper describes conversion after float training. This fork
adds quantization-aware training using the deployed integer arithmetic for
forward results and float surrogate gradients for optimization.

- Master weights and AdamW moments remain FP32.
- Features and biases use scale 512; weights use scale 8192.
- Forward computation includes integer rounding, clipping, accumulator behavior,
  lookup tables, fused operations, latent symbols and temporal reference state.
- The video encoder's fused parameters are computed canonically on CPU without
  changing the original trainable parameters.
- Quantized weights are recomputed from current parameters on each forward.
  Training never consumes a stale inference-preparation cache.
- Backward derivatives approximate the discontinuous integer computation;
  rounding uses straight-through gradients and clipping bounds the surrogate.

Exact forward computation does not make integer operations differentiable.
Training is more expensive than inference and may be slower than approximate
fake quantization. AMP and inference CUDA graphs are not used in this first
training implementation. CUDA and the native INT16 extension are required for
INT16 stages; they never silently fall back to float. `device: cpu` is available
for float correctness tests with `validation.actual_bitstreams: false`.

The objective is rate plus lambda-weighted distortion. Default distortion is
80% YUV MSE with channel weights `(6,1,1)/8`, plus 20% RGB MSE. Base QPs are
sampled uniformly and lambda is interpolated logarithmically. The video QP
offset pattern is `[0,8,0,4,0,4,0,4]`. Training bitrate is an entropy estimate;
validation reports actual file bytes separately, including headers.

## Multiple GPUs, checkpoints and resume

For five GPUs on one computer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 main.py train --config configs/train/base.yaml
```

`batch_per_gpu` is per process. Effective batch size is that value multiplied by
GPU count and `accumulation_steps`. Each GPU holds a complete model; GPU memory
is not pooled. Reduce crop/window/batch sizes or enable activation checkpointing
when memory is limited. This is synchronized DDP training, independent of the
encoder's `--shared-work` scheduler.

DDP requires a PyTorch build with distributed support (NCCL for CUDA). Some
Jetson NVIDIA wheels omit it; those builds still support single-GPU training.
`distributed_timeout_seconds` defaults to 21600 (six hours), allowing other ranks
to wait while rank zero validates actual bitstreams. Increase it for longer
validation jobs; it also controls how long a failed collective may take to abort.

Logs include UTC timestamps, stage, losses, estimated bitrate, throughput,
gradient norm, skipped non-finite updates, memory, and QAT saturation statistics.
Non-finite updates are skipped consistently across ranks. Ten consecutive
non-finite updates stop the run.

`last.pt` contains model(s), optimizer, schedule position, per-rank RNG state,
next batch position, manifest identity and configuration. Checkpoints are
published atomically after complete optimizer steps. Resume with the same
process count and training configuration:

```bash
torchrun --standalone --nproc_per_node=5 main.py train --config configs/train/base.yaml --resume runs/base/last.pt
```

Ctrl+C requests a stop at the next completed optimizer-step boundary. A second
trainer cannot write the same run directory. Dataset, architecture and training
changes require a new run initialized from weights, rather than an exact resume.

## Validation and export

Real validation temporarily releases training GPU parameters/moments and DDP
buckets, then starts an isolated CUDA process. It measures actual bitrate and
PSNR, compares float and INT16 reconstruction, verifies QAT/native equality, and
decodes the produced `.bin` / `.dcvci` files through the existing decoder. Video
checks include feature refreshes and reference-state equality across frames.
Validation costs additional time and disk space under the run directory.

Best checkpoints are kept per model (`best-image.pt`, `best-video.pt`), and
`best.pt` identifies the latest model whose best checkpoint was updated.

```bash
python main.py export-model --checkpoint runs/base/best-video.pt --output checkpoints/my_codec
```

A video training checkpoint contains the image weights actually used to train
its references. Export produces that matched pair, rather than substituting a
different independently selected image checkpoint. Exports contain:

- Unfused `image.pth.tar` and, when present, `video.pth.tar` with architecture metadata.
- Bound `.int16prep.pt` files containing integer weights, LUTs and entropy tables.
- `manifest.json` with checkpoint, architecture and prepared-file identities.

Use the new files with existing `--model_path_i` and `--model_path_p` arguments.
Encode/stream, thumbnail compression, decode and live viewers construct the
architecture from checkpoint metadata. Existing checkpoints without metadata
use the original architecture. Keep the prepared files with the checkpoints
when sharing models across computers. Stale prepared state for a new-format
checkpoint is regenerated instead of being used with changed weights.

The bitstream format is unchanged. A `.bin` or `.dcvci` still requires the
matching model files and INT16 runtime when decoding; model identity is not
embedded in the bitstream. Preserve model bundles needed by existing archives.

Training runs, prepared datasets, generated bitstreams and model weights are
local artifacts and must not be added to Git.
