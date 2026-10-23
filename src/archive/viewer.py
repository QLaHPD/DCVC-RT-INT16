"""UF windowed playback using archived audio as the presentation clock."""

from functools import partial
import time

from src.archive.frame_source import raw_rgb
from src.archive.rt_managed import managed_module
from src.archive.view_audio import AudioPlayer
from src.archive.view_buffer import BufferedDecodeWorker


class ViewerApp(managed_module('viewer').ViewerApp):
    def __init__(self, root, source, fps, start_frame, max_width, max_height, runtime_label,
                 audio=True, decode_ahead=512, decode_history=512):
        self.audio = None
        self.audio_error = None
        self.native_fps = source.metadata['video']['fps']
        self.speed = fps / self.native_fps
        self.audio_active = False
        self.clock_position = max(0, start_frame) / self.native_fps
        self.clock_time = None
        self.pending = False
        self.clock_started = self.waiting_for_buffer = False
        self.seen_error = None
        self.displayed_frame = None
        self.closed = False
        if audio and source.audio_path is not None:
            try:
                self.audio = AudioPlayer(source.audio_path)
            except Exception as exc:
                self.audio_error = str(exc)
        super().__init__(root, source, fps, start_frame, max_width, max_height, runtime_label,
                         worker_factory=partial(BufferedDecodeWorker, capacity=decode_ahead, history=decode_history))
        self.scale.bind('<ButtonPress-1>', lambda _event: self.pause())
        self.buffer_label = self.tk.Label(self.status_label.master, anchor='e')
        self.buffer_label.pack(fill=self.tk.X)

    def request_frame(self, frame_index):
        self.pending = True
        self.seen_error = None
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
        status = self.worker.status(self.request_id)
        if status['error'] and status['error'] != self.seen_error:
            self.pause()
            self.seen_error = status['error']
            self.pending = False
            self.set_status(status['error'])
        elif self.pending:
            frame = self.worker.pop_due(self.request_id, self.current_frame)
            if frame is not None:
                self.pending = False
                self.display_frame(frame)
        self.buffer_label.configure(text=f"Previous: {status['history']}/{self.worker.history_capacity} | "
                                         f"Ahead: {status['size']}/{self.worker.capacity} frames")
        self.poll_after_id = self.root.after(10, self.poll_results)

    def display_frame(self, frame):
        if frame.rgb is None:
            frame.rgb = raw_rgb(frame.raw, frame.width, frame.height)
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
        self.clock_started = True
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
        if self.displayed_frame == self.source.index.frame_count - 1 and not self.pending:
            self.seek(0, playing=True)
            return
        if self.worker.status(self.request_id)['error']:
            return
        self.playing = self.waiting_for_buffer = True
        self.play_button.configure(text='Pause')
        self.ensure_clock()
        self.schedule_next_play_frame()

    def pause(self):
        if self.playing and self.clock_started and not self.waiting_for_buffer:
            self.clock_position = self.position()
        self.playing = False
        self.waiting_for_buffer = False
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
            self.play_after_id = self.root.after(5, self.next_play_frame)

    def ensure_clock(self):
        if self.pending:
            return False
        if self.waiting_for_buffer:
            status = self.worker.status(self.request_id)
            warm = min(8, self.worker.capacity,
                       self.source.index.frame_count - self.displayed_frame - 1)
            if status['size'] < warm and not status['eof']:
                self.set_status('Buffering video...')
                return False
            self.waiting_for_buffer = False
            if not self.clock_started:
                self.start_clock()
            else:
                self.clock_time = time.monotonic()
                if self.audio_active:
                    self.audio.resume()
        return True

    def buffer_underrun(self):
        self.clock_position = self.position()
        self.clock_time = None
        if self.audio is not None:
            self.audio.pause()
        self.waiting_for_buffer = True
        self.set_status('Buffering video...')

    def next_play_frame(self):
        self.play_after_id = None
        if not self.playing:
            return
        if not self.ensure_clock():
            self.schedule_next_play_frame()
            return
        position = self.position()
        total = self.source.index.frame_count
        target = min(total - 1, int(position * self.native_fps))
        if target > self.displayed_frame:
            frame = self.worker.pop_due(self.request_id, target)
            if frame is not None:
                self.display_frame(frame)
            if self.displayed_frame < target:
                # Freeze both clocks during genuine starvation, rather than
                # letting audio advance through a long run of missing pictures.
                self.buffer_underrun()
        if position >= total / self.native_fps and self.displayed_frame == total - 1:
            self.pause()
            return
        self.schedule_next_play_frame()

    def seek(self, frame_index, playing=False):
        self.pause()
        if self.audio is not None:
            self.audio.stop()
        self.audio_active = False
        self.clock_time = None
        self.clock_started = False
        self.playing = playing
        self.waiting_for_buffer = playing
        self.play_button.configure(text='Pause' if playing else 'Play')
        self.request_frame(frame_index)
        if playing:
            self.schedule_next_play_frame()

    def previous_frame(self):
        self.seek((self.displayed_frame if self.displayed_frame is not None else self.current_frame) - 1)

    def next_frame(self):
        self.seek((self.displayed_frame if self.displayed_frame is not None else self.current_frame) + 1)

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
