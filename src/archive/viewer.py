"""UF windowed playback using archived audio as the presentation clock."""

import queue
import time

from src.archive.rt_managed import managed_module
from src.archive.view_audio import AudioPlayer


class ViewerApp(managed_module('viewer').ViewerApp):
    def __init__(self, root, source, fps, start_frame, max_width, max_height, runtime_label,
                 audio=True):
        self.audio = None
        self.audio_error = None
        self.native_fps = source.metadata['video']['fps']
        self.speed = fps / self.native_fps
        self.audio_active = False
        self.clock_position = max(0, start_frame) / self.native_fps
        self.clock_time = None
        self.pending = False
        self.displayed_frame = None
        self.closed = False
        if audio and source.audio_path is not None:
            try:
                self.audio = AudioPlayer(source.audio_path)
            except Exception as exc:
                self.audio_error = str(exc)
        super().__init__(root, source, fps, start_frame, max_width, max_height, runtime_label)
        self.scale.bind('<ButtonPress-1>', lambda _event: self.pause())

    def request_frame(self, frame_index):
        self.pending = True
        super().request_frame(frame_index)

    def set_status(self, text):
        if self.audio_error:
            text += f' | Audio unavailable: {self.audio_error}'
        elif self.audio is not None:
            text += ' | Audio'
        super().set_status(text)

    def poll_results(self):
        if self.closed:
            return
        while True:
            try:
                kind, request_id, payload = self.worker.results.get_nowait()
            except queue.Empty:
                break
            if request_id != self.request_id:
                continue
            self.pending = False
            if kind == 'error':
                self.pause()
                self.set_status(payload)
            else:
                self.display_frame(payload)
        self.root.after(15, self.poll_results)

    def display_frame(self, frame):
        self.displayed_frame = frame.index
        super().display_frame(frame)

    def position(self):
        if self.audio_active:
            if self.audio.error:
                self.clock_position = self.audio.position()
                self.audio_error = self.audio.error
                self.audio.close()
                self.audio = None
                self.audio_active = False
                self.clock_time = time.monotonic() if self.playing else None
                self.set_status('Playing video')
            elif not self.audio.ended():
                return self.audio.position()
            else:
                # A slightly shorter audio track must not stall the final frame.
                self.clock_position = self.audio.position()
                self.audio_active = False
                self.clock_time = time.monotonic() if self.playing else None
        elapsed = 0 if self.clock_time is None else time.monotonic() - self.clock_time
        return self.clock_position + elapsed * self.speed

    def start_clock(self):
        self.clock_position = self.displayed_frame / self.native_fps
        self.clock_time = time.monotonic()
        if self.audio is not None:
            try:
                self.audio.play(self.clock_position, self.speed)
                self.audio_active = True
            except Exception as exc:
                self.audio_error = str(exc)
                self.audio.close()
                self.audio = None
                self.set_status('Playing video')

    def toggle_play(self):
        if self.playing:
            self.pause()
            return
        self.playing = True
        self.play_button.configure(text='Pause')
        if self.displayed_frame == self.source.index.frame_count - 1 and not self.pending:
            self.seek(0, playing=True)
        elif self.audio_active:
            self.audio.resume()
        elif self.clock_time is None:
            # Initial model loading / an outstanding seek finishes before sound starts.
            if not self.pending:
                self.start_clock()
        self.schedule_next_play_frame()

    def pause(self):
        if self.playing:
            self.clock_position = self.position()
        self.playing = False
        self.clock_time = None
        if self.audio is not None:
            self.audio.pause()
        self.play_button.configure(text='Play')
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def schedule_next_play_frame(self):
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
        if self.playing:
            if self.clock_time is None and not self.audio_active and not self.pending:
                self.start_clock()
            self.play_after_id = self.root.after(10, self.next_play_frame)

    def next_play_frame(self):
        self.play_after_id = None
        if not self.playing:
            return
        if self.pending:
            self.schedule_next_play_frame()
            return
        if self.clock_time is None and not self.audio_active:
            self.start_clock()
        position = self.position()
        total = self.source.index.frame_count
        if position >= total / self.native_fps:
            self.pause()
            if self.displayed_frame != total - 1:
                self.request_frame(total - 1)
            return
        target = min(total - 1, int(position * self.native_fps))
        if target > self.displayed_frame:
            # Decode lag drops presentation frames instead of accumulating A/V drift.
            self.request_frame(target)
        self.schedule_next_play_frame()

    def seek(self, frame_index, playing=False):
        self.pause()
        if self.audio is not None:
            self.audio.stop()
        self.audio_active = False
        self.clock_time = None
        self.playing = playing
        self.play_button.configure(text='Pause' if playing else 'Play')
        self.request_frame(frame_index)

    def previous_frame(self):
        self.seek(self.current_frame - 1)

    def next_frame(self):
        self.seek(self.current_frame + 1)

    def on_slider_release(self, _event):
        self.seek(int(self.scale.get()))

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.pause()
            if self.audio is not None:
                self.audio.close()
        finally:
            super().close()
