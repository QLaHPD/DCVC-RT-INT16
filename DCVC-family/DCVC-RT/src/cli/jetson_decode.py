"""Optional Jetson NVDEC bridge; preserve FFmpeg's scaling and FPS semantics.

Never use NVIDIA's FFmpeg decoder here: on L4T 36.4.7 it drops frames and
writes diagnostics into raw stdout. GStreamer supplies timestamped I420 via
an isolated descriptor instead. Unsupported inputs retain software decoding.
"""
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


class DecodeError(RuntimeError):
    pass


def jetson_source(path):
    if not Path(path).is_file() or not shutil.which("gst-launch-1.0"):
        raise DecodeError("Jetson decoding requires a local file and GStreamer")
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ], capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        streams = [s for s in data["streams"] if s.get("codec_type") == "video"]
        if len(streams) != 1:
            raise DecodeError("hardware path requires exactly one video stream")
        stream = streams[0]
        # Keep the validated scope narrow. Other codecs/layouts fall back.
        if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
            raise DecodeError("hardware path currently supports 8-bit H.264 only")
        if stream["width"] % 8 or stream["height"] % 2:
            raise DecodeError("input dimensions require software plane packing")
        if stream.get("field_order", "unknown") not in ("unknown", "progressive"):
            raise DecodeError("interlaced input requires software decoding")
        if stream.get("color_range", "unknown") not in ("unknown", "tv"):
            raise DecodeError("full-range input requires software decoding")
        if stream.get("side_data_list") or stream.get("tags", {}).get("rotate"):
            raise DecodeError("video side data requires software decoding")
        if data["format"].get("start_time") != stream.get("start_time"):
            raise DecodeError("different container/video start times require software decoding")
        tb = str(Fraction(stream["time_base"]))
        formats = data["format"]["format_name"].split(",")
        if "mov" in formats or "mp4" in formats:
            demux = "qtdemux"
        else:
            raise DecodeError("unsupported hardware input container")
        return demux, tb
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as exc:
        raise DecodeError(f"cannot probe hardware input: {exc}") from exc


class JetsonPipe:
    def __init__(self, decoder, ffmpeg, diagnostics):
        self.decoder = decoder
        self.ffmpeg = ffmpeg
        self.diagnostics = diagnostics
        self.stdout = ffmpeg.stdout
        self.pid = ffmpeg.pid

    def poll(self):
        a, b = self.ffmpeg.poll(), self.decoder.poll()
        return None if a is None or b is None else (a or b)

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        a = self.ffmpeg.wait(timeout=timeout)
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        b = self.decoder.wait(timeout=remaining)
        self.diagnostics.seek(0)
        diagnostic = self.diagnostics.read().decode("utf-8", errors="replace")
        if a or b or any(s in diagnostic for s in ("ERROR", "NvMapMem", "Failed")):
            raise DecodeError(f"Jetson decoder failed ({b}), FFmpeg ({a}): {diagnostic[-2000:]}")
        return 0

    def terminate(self):
        for proc in (self.ffmpeg, self.decoder):
            if proc.poll() is None:
                proc.terminate()

    def kill(self):
        for proc in (self.ffmpeg, self.decoder):
            if proc.poll() is None:
                proc.kill()

    def close(self):
        self.terminate()
        for proc in (self.ffmpeg, self.decoder):
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.stdout.close()
        self.diagnostics.close()


def start_jetson_pipe(path, ffmpeg_command, popen=subprocess.Popen):
    demux, time_base = jetson_source(path)
    read_fd, write_fd = os.pipe()
    diagnostics = tempfile.TemporaryFile()
    decoder = None
    try:
        gst = ["gst-launch-1.0", "-q", "filesrc", f"location={path}", "!",
               demux, "name=demux", "demux.video_0", "!", "queue", "!",
               "h264parse", "!", "nvv4l2decoder", "!", "nvvidconv", "!",
               "video/x-raw,format=I420", "!", "matroskamux",
               "streamable=true", "timecodescale=1", "!", "fdsink", f"fd={write_fd}"]
        decoder = popen(gst, stdout=diagnostics, stderr=diagnostics, pass_fds=(write_fd,))
        cmd = list(ffmpeg_command)
        cmd[cmd.index("-i") + 1] = f"/proc/self/fd/{read_fd}"
        cmd[cmd.index("-hwaccel") + 1] = "none"
        vf = cmd.index("-vf") + 1
        # Gst timestamps use nanoseconds; restore the source time base before
        # fps:round=down to avoid selecting neighbouring frames at boundaries.
        cmd[vf] = f"settb=expr={time_base}," + cmd[vf]
        ffmpeg = popen(cmd, stdout=subprocess.PIPE, stderr=diagnostics, pass_fds=(read_fd,))
        return JetsonPipe(decoder, ffmpeg, diagnostics)
    except Exception as exc:
        if decoder is not None:
            decoder.kill()
            decoder.wait()
        diagnostics.close()
        if isinstance(exc, OSError):
            raise DecodeError(f"cannot start Jetson decoder: {exc}") from exc
        raise
    finally:
        os.close(read_fd)
        os.close(write_fd)


def validate_decode(proc, frames):
    if not frames:
        raise DecodeError("decoder produced no frames")
    try:
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise DecodeError("decoder did not finish after end of video") from exc
    if rc:
        raise DecodeError(f"FFmpeg decoding failed with exit status {rc}")
