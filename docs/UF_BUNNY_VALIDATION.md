# Jetson UF archive validation — 2026-09-12

Tested locally using the dedicated UF environment, without INT16 or changes to
the RT installation. Models, source video, native binaries, and generated outputs
are not committed.

## Configuration and result

| Item | Result |
|---|---|
| Source | Big_Buck_Bunny_720_10s_30MB.mp4 |
| Source and reconstruction | 1280×720, 30 FPS, 300 frames, 10 seconds |
| Model / arithmetic | HT-S / FP16 |
| QI / QP | 36 / 30 |
| Reset interval / intra period | 32 / -1 |
| Workers | 1, Jetson Orin |
| Encoded video | 434,740 bytes (39 packets) |
| Video bitrate | 347.792 kbit/s |
| Encoding loop | 11.418 s, approximately 26.27 FPS |
| Fresh verification decode | 9.551 s |
| PSNR against original, FFmpeg YUV420 average | 34.422331 dB |
| Reconstructed media | FFV1 Matroska, 300 frames verified by FFprobe |
| Resume | 0 encoded, 1 resumed, 0 failed |

Encoding-loop time includes initial native warmup and input reading, but excludes
checkpoint loading, output validation and process startup. The full encode command
including validation took approximately 32.4 seconds. These are one-run archive
measurements, not a controlled steady-state benchmark. The supplied clip has no
audio; audio was tested separately on a generated one-second fixture.

A fresh decoder process and a separate decode-to-file process both reproduced the
archive's recorded raw YUV SHA256:

```text
7287c2dc7c724900f215955f8d20bd378b22aefd32452be99c0232dba341689d
```

The original video was retained and its SHA256 remained:

```text
a2c61fa8bdb4381a28a06a4df53fdda35e35dd49082b99bf199e3125be38f3b9
```

Checkpoint hashes (the weights themselves remain local):

- Image: `b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302`
- HT-S video: `565607d84943ee79072a5760d34a200d5724b4bb72c6bc24bfab41e8ae4f5f47`

## Reproduce

From the UF checkout:

```bash
./scripts/uf-python main.py encode --input_file Big_Buck_Bunny_720_10s_30MB.mp4 --output_root runs/bunny-qi36-qp30 --qi 36 --qp 30 --procs 1
./scripts/uf-python main.py verify --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.mp4.uf
./scripts/uf-python main.py decode --input runs/bunny-qi36-qp30/Big_Buck_Bunny_720_10s_30MB.mp4.uf --output_file runs/bunny-reconstructed.mkv
```

Decode refuses to overwrite an existing output file. Repeating encode resumes
only archives with matching settings, model hashes, source hash and artifact hashes.

## Additional checks

- Nine-frame 144p smoke encode and independent decode: passed.
- Folder/channel discovery, unchanged 256×144 dimensions, 30→8 FPS conversion,
  final partial HT-S chunk, mono 6k Opus encoding and decode validation: passed.
- Seven storage tests cover atomic publication, corruption detection, resume,
  peer claims, settings mismatch, duplicate stems, resizing and cleanup safeguards.
- Headless ffplay streaming smoke test: passed; no decoded media file created.
- Native cleanup verified and deleted only a generated fixture source.
- Two simultaneous 144p HT-S workers exceeded Jetson memory during validation:
  one archive completed; the failed job published no archive and retained its
  original. Retrying with one worker encoded the failed job and resumed the
  completed one without changing its bitstream hash or modification time.
  Use `--procs 1` on this device.
- No original Bunny video was deleted.

The native extensions built on Python 3.10.20, NVIDIA PyTorch
2.5.0a0+872d972e41.nv24.08, CUDA 12.6 and CUTLASS v4.4.1 (`4370102`).
This proves local HT-S operation, not bit-exact FP16 portability across GPUs.
