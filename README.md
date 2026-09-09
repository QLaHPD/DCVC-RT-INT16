# DCVC-RT INT16 Managed

This is a fork of [Microsoft/DCVC](https://github.com/microsoft/DCVC) focused on a deterministic INT16 runtime, bit-exact CUDA kernel optimizations, and a production-oriented video archive pipeline for DCVC-RT.

The maintained implementation is in **[DCVC-family/DCVC-RT](DCVC-family/DCVC-RT/README.md)**.

Highlights:

- INT16 feature and weight execution designed for deterministic cross-device coding
- Optimized bit-exact CUDA convolution kernels
- File-level multi-GPU encoding and decoding with one independent worker per GPU
- Direct yt-dlp/FFmpeg streaming without storing source video containers
- Unified encode, decode, viewer, and validated-cleanup CLI
- Multi-channel scheduling with a tmux-friendly terminal dashboard
- Non-blocking cleanup approvals and optional `--auto-delete`
- Persistent archive manifests, SHA-256 hashes, and restart-safe deletion auditing
- Portable native-extension build script for each target computer

## Blender Bunny configuration sweep

This exploratory rate-distortion sweep records an earlier search for the best encoder configurations on the Blender Bunny test video rendered at 144p (256×144). It compares 1,803 configurations over 300 frames and identifies 116 Pareto-optimal results. Lower file size and higher PSNR are preferred; open the standalone SVG to inspect each point's QP, intra-period, and reset-interval settings.

[![Pareto frontier of file size versus PSNR for the Blender Bunny 256×144 test](DCVC-family/DCVC-RT/docs/assets/pareto_frontier_blender_bunny_256x144.svg)](DCVC-family/DCVC-RT/docs/assets/pareto_frontier_blender_bunny_256x144.svg)

The rest of the repository is retained from upstream so its history, license, and the other DCVC-family implementations remain intact. The original upstream root README is preserved as [UPSTREAM_ROOT_README.md](UPSTREAM_ROOT_README.md).

## Attribution

- **Original DCVC/DCVC-RT research and implementation:** Microsoft Research and the authors listed in the [DCVC-RT paper](https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.html).
- **Fork owner and maintainer:** [Salatiel Jordão (@QLaHPD)](https://github.com/QLaHPD).
- **Primary implementation contributor for this fork:** **OpenAI Codex**, operating as an AI coding agent under the direction and review of the repository owner.

Codex is credited transparently as an implementation contributor; it is not represented as a GitHub account, legal copyright holder, or independent project maintainer. This fork is not affiliated with or endorsed by Microsoft or OpenAI.

## License

The upstream project is distributed under the [MIT License](LICENSE.txt). Existing file-level notices remain in place. Contributions in this fork are distributed under the same repository license.
