# Cooperative encoding from multiple machines

The local `encode` command can use a shared output directory as a small distributed work queue. Start the same command on every machine and add `--shared-work`. Each instance claims only as many videos as it has free workers, reports its owner and frame progress in shared state, and waits until all peers finish the discovered queue.

The source root may have a different absolute path on each machine. The output root must refer to the same shared storage, and every machine must see the same channel IDs and source basenames. Use the same checkpoints and output-affecting options everywhere. The command hashes both checkpoints and stores a pipeline fingerprint per channel; incompatible peers stop with an error rather than mix outputs.

Because final artifact names omit the source-container suffix, two inputs such as `item.mkv` and `item.webm` represent one queue item. Discovery deterministically prefers MKV, then MP4, WebM, MOV, and AVI, prints a warning, and retains the unused variant.

Example for the Jetson:

```bash
DCVC_USE_INT16=1 python main.py encode \
  --base_root /mnt/to_storage/DATA/YOUTUBE \
  --output_root /mnt/to_storage/DATA/YOUTUBE \
  --channel_ids UCWKtHaeXVzUscYGcm0hEunw \
  --model_path_i checkpoints/cvpr2025_image.pth.tar \
  --model_path_p checkpoints/cvpr2025_video.pth.tar \
  --resolution 96 --fps 24 --qp_i 35 --qp_p 14 \
  --audio opus --opus_bitrate 6k --opus_channels mono \
  --procs 1 --cuda true --ui plain \
  --shared-work --shared-instance jetson
```

Run the equivalent command on the second computer with its local input/model paths and a different label, for example `--shared-instance desktop`. Both checkpoint files must have identical contents. FFmpeg hardware decoding (`--ff_hwaccel`), worker count, UI mode, and local paths may differ because they do not affect the pipeline fingerprint. Every peer must use the same neural execution mode, such as CUDA INT16.

Coordination state lives under `<channel>/.dcvc-shared-work/`. Atomic claim-directory creation prevents healthy peers from selecting the same basename. The owner updates its claim every ten seconds with frame count, FPS, and resolution. The default 900-second lease allows another peer to recover work after a machine stops updating. Increase `--shared-lease-seconds` on a storage system with unusually long outages or cache delays.

Workers write finished artifacts under `<channel>/.dcvc-stage/`. Only the process that still owns the claim can rename those files into their final `.bin` and `.opus` names. A completion marker is published after every required artifact exists. This fences a delayed worker after lease takeover and prevents it from overwriting the new owner's result.

Shared mode never deletes source videos and does not run the mutable channel-cleanup lifecycle. It also defers `--thumbnail_codec dcvc-intra`. Once every shared command has exited successfully, run the encode command once without `--shared-work` and include the thumbnail options if desired. Its normal resume scan will find all video outputs, encode or resume thumbnails, build the final channel inventory, validate the archive, and offer one cleanup choice for every validated original.

All participating encoders must support `--shared-work`. A currently running older/non-shared encoder has no claim, so let it finish before starting the shared queue. The shared filesystem must provide atomic directory creation and rename semantics; local POSIX filesystems and correctly configured NFS shares do, while unusual network or cloud mounts should be tested with a small fixture first.
