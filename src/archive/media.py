"""Bounded FFmpeg pipes and metadata helpers for UF archive inputs."""

import json
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile

import numpy as np


VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.mov', '.avi', '.m4v', '.ts'}


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                             '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    video = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
    if video is None:
        raise ValueError(f'No video stream: {path}')
    rate = video.get('avg_frame_rate') or video.get('r_frame_rate')
    if not rate or rate == '0/0':
        rate = video.get('r_frame_rate', '30/1')
    return {'width': int(video['width']), 'height': int(video['height']),
            'fps': float(Fraction(rate)), 'duration': float(data.get('format', {}).get('duration', 0)),
            'audio': any(s['codec_type'] == 'audio' for s in data['streams'])}


def dimensions(width, height, resolution):
    if resolution is None:
        return width + width % 2, height + height % 2
    scale = resolution / min(width, height)
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


class FrameReader:
    def __init__(self, source, width, height, original, fps=None):
        self.width, self.height = width, height
        filters = []
        if (width, height) != (original['width'], original['height']):
            filters.append(f'scale={width}:{height}:flags=bicubic')
        if fps is not None:
            filters.append(f'fps={fps}')
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-noautorotate',
                   '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn']
        if filters:
            command += ['-vf', ','.join(filters)]
        command += ['-pix_fmt', 'yuv420p', '-vsync', '0', '-f', 'rawvideo', 'pipe:1']
        self.errors = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.errors)
        self.ended = False

    def read(self):
        size = self.width * self.height * 3 // 2
        blocks, remaining = [], size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                if remaining != size:
                    raise ValueError('FFmpeg produced a truncated frame')
                code = self.process.wait()
                self.ended = True
                if code:
                    self.errors.seek(0)
                    raise RuntimeError(self.errors.read(8192).decode(errors='replace'))
                return None
            blocks.append(chunk)
            remaining -= len(chunk)
        values = np.frombuffer(b''.join(blocks), dtype=np.uint8)
        n = self.width * self.height
        y = values[:n].reshape(self.height, self.width)
        u = values[n:n + n//4].reshape(self.height//2, self.width//2)
        v = values[n + n//4:].reshape(self.height//2, self.width//2)
        return np.stack((y, u.repeat(2, 0).repeat(2, 1), v.repeat(2, 0).repeat(2, 1)))

    def close(self):
        # Release a producer blocked writing its next frame before waiting for it.
        self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def encode_audio(source, target, channels, bitrate, duration):
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(source), '-map', '0:a:0',
                    '-vn', '-c:a', 'libopus', '-ac', '1' if channels == 'mono' else '2',
                    '-b:a', bitrate, '-t', str(duration), '-f', 'opus', str(target)], check=True)


def verify_audio(path):
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
                    '-f', 'null', '-'], check=True)
