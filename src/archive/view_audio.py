"""Bounded FFmpeg/SDL2 audio playback and its consumed-sample media clock."""

import ctypes as C
import ctypes.util
import math
import shutil
import subprocess
import tempfile
import threading


RATE = 48000
CHANNELS = 2
BYTES_PER_SECOND = RATE * CHANNELS * 2


class AudioSpec(C.Structure):
    # SDL2 SDL_AudioSpec, using NULL callback for SDL_QueueAudio.
    _fields_ = [('freq', C.c_int), ('format', C.c_uint16), ('channels', C.c_uint8),
                ('silence', C.c_uint8), ('samples', C.c_uint16), ('padding', C.c_uint16),
                ('size', C.c_uint32), ('callback', C.c_void_p), ('userdata', C.c_void_p)]


class AudioDevice:
    def __init__(self):
        library = ctypes.util.find_library('SDL2-2.0') or ctypes.util.find_library('SDL2')
        if not library:
            raise RuntimeError('SDL2 audio library is unavailable')
        self.sdl = C.CDLL(library)
        signatures = {
            'SDL_InitSubSystem': ([C.c_uint32], C.c_int),
            'SDL_QuitSubSystem': ([C.c_uint32], None),
            'SDL_GetError': ([], C.c_char_p),
            'SDL_OpenAudioDevice': ([C.c_char_p, C.c_int, C.POINTER(AudioSpec),
                                     C.POINTER(AudioSpec), C.c_int], C.c_uint32),
            'SDL_CloseAudioDevice': ([C.c_uint32], None),
            'SDL_PauseAudioDevice': ([C.c_uint32, C.c_int], None),
            'SDL_ClearQueuedAudio': ([C.c_uint32], None),
            'SDL_GetQueuedAudioSize': ([C.c_uint32], C.c_uint32),
            'SDL_QueueAudio': ([C.c_uint32, C.c_void_p, C.c_uint32], C.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.sdl, name)
            function.argtypes, function.restype = arguments, result
        if self.sdl.SDL_InitSubSystem(0x10):  # SDL_INIT_AUDIO
            raise RuntimeError(self.error())
        desired = AudioSpec(freq=RATE, format=0x8010, channels=CHANNELS, samples=512)  # S16LE
        obtained = AudioSpec()
        self.handle = self.sdl.SDL_OpenAudioDevice(None, 0, C.byref(desired), C.byref(obtained), 0)
        if not self.handle:
            error = self.error()
            self.sdl.SDL_QuitSubSystem(0x10)
            raise RuntimeError(error)

    def error(self):
        return self.sdl.SDL_GetError().decode(errors='replace')

    def pause(self, paused=True):
        self.sdl.SDL_PauseAudioDevice(self.handle, int(paused))

    def queued_bytes(self):
        return self.sdl.SDL_GetQueuedAudioSize(self.handle)

    def queue(self, data):
        buffer = C.create_string_buffer(data)
        if self.sdl.SDL_QueueAudio(self.handle, buffer, len(data)):
            raise RuntimeError(self.error())

    def clear(self):
        self.sdl.SDL_ClearQueuedAudio(self.handle)

    def close(self):
        if self.handle:
            self.sdl.SDL_CloseAudioDevice(self.handle)
            self.handle = 0
            self.sdl.SDL_QuitSubSystem(0x10)


def tempo_filters(speed):
    """Preserve pitch when --fps changes playback speed, including <0.5x."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('Audio playback speed must be positive and finite')
    factors = []
    while speed < .5:
        factors.append(.5)
        speed /= .5
    while speed > 2:
        factors.append(2)
        speed /= 2
    factors.append(speed)
    return ','.join(f'atempo={factor:.12g}' for factor in factors)


class AudioPlayer:
    """A quarter-second PCM queue; position follows samples sent to hardware.

    Hardware buffering introduces a small device-dependent clock offset. Queue
    underruns freeze this clock instead of advancing video past missing audio.
    """
    def __init__(self, path):
        self.binary = shutil.which('ffmpeg')
        if not self.binary:
            raise RuntimeError('FFmpeg is unavailable')
        self.path = str(path)
        self.device = AudioDevice()
        self.lock = threading.RLock()
        self.process = self.thread = self.errors = None
        self.cancel = threading.Event()
        self.anchor, self.speed, self.total = 0., 1., 0
        self.error = None
        self.ready = self.done = False
        self.paused = True
        self.closed = False

    def play(self, position, speed=1.):
        if self.closed:
            raise RuntimeError('Audio player is closed')
        filters = tempo_filters(speed)
        if not math.isfinite(position) or position < 0:
            raise ValueError('Audio position must be nonnegative and finite')
        self.stop()
        self.anchor, self.speed = position, speed
        self.cancel = threading.Event()
        self.error, self.ready, self.done, self.paused = None, False, False, False
        self.errors = tempfile.TemporaryFile()
        command = [self.binary, '-v', 'error', '-nostdin', '-ss', str(position),
                   '-i', self.path, '-map', '0:a:0', '-vn', '-af', filters,
                   '-ac', str(CHANNELS), '-ar', str(RATE), '-f', 's16le', 'pipe:1']
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.errors)
            self.thread = threading.Thread(target=self._feed, name='uf-view-audio', daemon=True)
            self.thread.start()
        except BaseException:
            self.stop()
            raise

    def _feed(self):
        try:
            while not self.cancel.is_set():
                if self.device.queued_bytes() >= BYTES_PER_SECOND // 4:
                    self.cancel.wait(.01)
                    continue
                data = self.process.stdout.read(4096)
                if not data:
                    code = self.process.wait()
                    if code and not self.cancel.is_set():
                        self.errors.seek(0)
                        raise RuntimeError(self.errors.read(4096).decode(errors='replace'))
                    with self.lock:
                        self.done = True
                        self.ready = True
                        if not self.paused and not self.cancel.is_set():
                            self.device.pause(False)
                    return
                with self.lock:
                    if self.cancel.is_set():
                        return
                    self.device.queue(data)
                    self.total += len(data)
                    if not self.ready and self.total >= BYTES_PER_SECOND // 10:
                        self.ready = True
                        if not self.paused:
                            self.device.pause(False)
        except Exception as exc:
            if not self.cancel.is_set():
                self.error = f'{type(exc).__name__}: {exc}'

    def position(self):
        with self.lock:
            consumed = max(0, self.total - self.device.queued_bytes())
            return self.anchor + consumed / BYTES_PER_SECOND * self.speed

    def ended(self):
        with self.lock:
            return self.done and not self.device.queued_bytes()

    def pause(self):
        with self.lock:
            self.paused = True
            self.device.pause()

    def resume(self):
        with self.lock:
            self.paused = False
            if self.ready:
                self.device.pause(False)

    def stop(self):
        if self.closed:
            return
        self.cancel.set()
        self.device.pause()
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError('Audio decoder thread did not stop')
        if self.process is not None:
            self.process.stdout.close()
        if self.errors is not None:
            self.errors.close()
        with self.lock:
            self.device.clear()
            self.total = 0
        self.process = self.thread = self.errors = None
        self.paused = True

    def close(self):
        if self.closed:
            return
        self.stop()
        self.device.close()
        self.closed = True
