import ctypes.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.archive.view_audio import AudioPlayer, BYTES_PER_SECOND, tempo_filters


def until(predicate, timeout=3, pump=lambda: None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pump()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for playback')


class TempoTests(unittest.TestCase):
    def test_pitch_preserving_tempo_chain(self):
        for speed in (.01, .25, .5, 1, 2, 4, 100):
            factors = [float(part.split('=')[1]) for part in tempo_filters(speed).split(',')]
            self.assertTrue(all(.5 <= factor <= 2 for factor in factors))
            self.assertAlmostEqual(math.prod(factors), speed)
        for speed in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                tempo_filters(speed)

    def test_view_cli_audio_enabled_by_default_and_can_disable(self):
        from main import build_parser
        parser = build_parser()
        self.assertFalse(parser.parse_args(['view', 'clip.bin']).no_audio)
        self.assertTrue(parser.parse_args(['view', 'clip.bin', '--no-audio']).no_audio)
        for fps in (8, 15, 60):
            options = parser.parse_args(['view', 'clip.bin', '--fps', str(fps)])
            self.assertEqual((options.decode_ahead, options.decode_history), (512, 512))


@unittest.skipUnless(shutil.which('ffmpeg') and ctypes.util.find_library('SDL2-2.0'), 'Needs FFmpeg and SDL2')
class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'sound.opus'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
                        '-t', '1.2', '-c:a', 'libopus', str(self.path)], check=True)
        env = patch.dict(os.environ, SDL_AUDIODRIVER='dummy')
        env.start()
        self.addCleanup(env.stop)
        self.player = AudioPlayer(self.path)
        self.addCleanup(self.player.close)

    def test_play_pause_resume_clock_and_bounded_queue(self):
        player = self.player
        player.play(0)
        until(lambda: player.position() > .12)
        self.assertLessEqual(player.device.queued_bytes(), BYTES_PER_SECOND // 4 + 4096)
        player.pause()
        paused = player.position()
        time.sleep(.1)
        self.assertAlmostEqual(player.position(), paused, places=4)
        player.resume()
        until(lambda: player.position() > paused + .1)

    def test_seek_speed_and_short_audio_reach_eof(self):
        player = self.player
        player.play(0)
        until(lambda: player.position() > .1)
        player.play(.6, speed=2)
        self.assertGreaterEqual(player.position(), .6)
        until(player.ended)
        self.assertAlmostEqual(player.position(), 1.2, delta=.06)
        self.assertIsNone(player.error)

    def test_close_stops_decoder_and_thread_idempotently(self):
        player = self.player
        player.play(0)
        process, thread = player.process, player.thread
        until(lambda: player.ready)
        player.close()
        player.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(thread.is_alive())
        self.assertTrue(process.stdout.closed)
        self.assertEqual(player.device.handle, 0)

    def test_decoder_failure_is_reported_without_hanging(self):
        broken = self.path.with_name('broken.opus')
        broken.write_bytes(b'not an opus stream')
        self.player.path = str(broken)
        self.player.play(0)
        until(lambda: self.player.error is not None)
        self.assertIn('RuntimeError', self.player.error)

    def test_thread_start_failure_releases_child_and_pipes(self):
        with patch('src.archive.view_audio.threading.Thread.start', side_effect=RuntimeError('cannot start thread')):
            with self.assertRaisesRegex(RuntimeError, 'cannot start thread'):
                self.player.play(0)
        self.assertIsNone(self.player.process)
        self.assertIsNone(self.player.errors)
        self.assertIsNone(self.player.thread)


class FakeSource:
    def __init__(self, audio_path=None, frames=16):
        self.audio_path = audio_path
        self.index = SimpleNamespace(path=Path('clip.bin'), frame_count=frames)
        self.metadata = dict(video=dict(fps=8), pipeline=dict(runtime='int16'))
        self.closed = False
        self.delay = 0
        self.calls = []

    def seek(self, index, **_kwargs):
        self.calls.append(index)
        time.sleep(self.delay)
        return SimpleNamespace(index=index, width=8, height=8, frame_type='P', qp=39,
                               rgb=np.zeros((8, 8, 3), dtype=np.uint8), raw=bytes(64) + bytes([128])*32)

    def close(self):
        self.closed = True


class ViewerLifecycleTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DISPLAY'), 'Needs a Tk display')
    def test_shared_viewer_default_worker_remains_compatible(self):
        import tkinter as tk
        from src.archive.rt_managed import managed_module
        viewer = managed_module('viewer')
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        root.withdraw()
        app = viewer.ViewerApp(root, FakeSource(), 8, 0, 100, 100, 'test')
        try:
            self.assertIsInstance(app.worker, viewer.DecodeWorker)
            until(lambda: app.photo is not None, pump=root.update)
            app.toggle_play()
            until(lambda: app.current_frame > 0, pump=root.update)
        finally:
            app.close()
            app.worker.thread.join(timeout=2)
        self.assertFalse(app.worker.thread.is_alive())

    def test_keyboard_interrupt_closes_window_and_audio_owner(self):
        from src.archive.frame_source import run_view
        root = Mock()
        root.winfo_exists.return_value = True
        root.mainloop.side_effect = KeyboardInterrupt
        app = Mock()
        app.close.side_effect = root.destroy
        args = SimpleNamespace(input='clip.bin', recursive=False, fps=None, start_frame=0,
                               max_width=512, max_height=288, no_audio=False)
        with patch('tkinter.Tk', return_value=root), \
                patch('src.archive.frame_source.FrameSource', return_value=FakeSource()), \
                patch('src.archive.viewer.ViewerApp', return_value=app):
            with self.assertRaises(KeyboardInterrupt):
                run_view(args)
        app.close.assert_called_once()
        root.destroy.assert_called_once()


class FakeAudio:
    def __init__(self, _path):
        self.time = 0
        self.error = None
        self.calls = []
        self.finished = False

    def play(self, position, speed):
        self.time = position
        self.calls.append(('play', position, speed))

    def position(self):
        return self.time

    def ended(self):
        return self.finished

    def pause(self):
        self.calls.append(('pause',))

    def resume(self):
        self.calls.append(('resume',))

    def stop(self):
        self.calls.append(('stop',))

    def close(self):
        self.calls.append(('close',))


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Needs a Tk display')
class ViewerTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from src.archive.viewer import ViewerApp
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.source = FakeSource(Path('sound.opus'))
        self.audio_patch = patch('src.archive.viewer.AudioPlayer', FakeAudio)
        self.audio_patch.start()
        self.addCleanup(self.audio_patch.stop)
        self.app = ViewerApp(self.root, self.source, 8, 0, 100, 100, 'test')
        self.addCleanup(self.cleanup)
        until(lambda: self.app.displayed_frame == 0, pump=self.root.update)

    def cleanup(self):
        self.app.close()
        self.app.worker.thread.join(timeout=2)

    def test_play_follows_audio_clock_and_pause_resume(self):
        app = self.app
        app.toggle_play()
        app.audio.time = .75
        until(lambda: app.displayed_frame == 6, pump=self.root.update)
        app.toggle_play()
        self.assertFalse(app.playing)
        self.assertEqual(app.audio.calls[-1], ('pause',))
        app.toggle_play()
        self.assertIn(('resume',), app.audio.calls)

    def test_manual_seek_and_steps_reposition_audio_on_play(self):
        app = self.app
        app.toggle_play()
        app.seek(10)
        until(lambda: app.displayed_frame == 10, pump=self.root.update)
        self.assertFalse(app.playing)
        app.toggle_play()
        self.assertEqual(app.audio.calls[-1], ('play', 1.25, 1.))
        app.previous_frame()
        until(lambda: app.displayed_frame == 9, pump=self.root.update)
        self.assertFalse(app.playing)
        app.next_frame()
        until(lambda: app.displayed_frame == 10, pump=self.root.update)

    def test_short_audio_does_not_stall_video_end_and_replay(self):
        app = self.app
        app.toggle_play()
        app.audio.time = 1.9
        app.audio.finished = True
        until(lambda: not app.playing, pump=self.root.update)
        until(lambda: app.displayed_frame == 15, pump=self.root.update)
        app.audio.finished = False
        app.toggle_play()
        until(lambda: app.displayed_frame == 0, pump=self.root.update)
        self.assertTrue(app.playing)
        until(lambda: app.audio.calls[-1] == ('play', 0., 1.), pump=self.root.update)
        self.assertEqual(app.audio.calls[-1], ('play', 0., 1.))

    def test_play_waits_for_seek_before_starting_sound(self):
        app = self.app
        self.source.index.frame_count = 1200
        self.source.delay = .02
        app.seek(800)  # A cache miss: audio waits for the seek and warm-up.
        count = len([c for c in app.audio.calls if c[0] == 'play'])
        app.toggle_play()
        self.assertEqual(len([c for c in app.audio.calls if c[0] == 'play']), count)
        until(lambda: app.displayed_frame == 800, pump=self.root.update)
        until(lambda: app.audio.calls[-1] == ('play', 100., 1.), pump=self.root.update)
        self.assertEqual(app.audio.calls[-1], ('play', 100., 1.))

    def test_previous_and_next_reuse_decoded_frames(self):
        app = self.app
        until(lambda: app.worker.status(app.request_id)['eof'], pump=self.root.update)
        app.toggle_play()
        app.audio.time = 1.25
        until(lambda: app.displayed_frame == 10, pump=self.root.update)
        calls = list(self.source.calls)
        app.previous_frame()
        until(lambda: app.displayed_frame == 9, pump=self.root.update)
        self.assertFalse(app.playing)
        self.assertEqual(self.source.calls, calls)
        app.next_frame()
        until(lambda: app.displayed_frame == 10, pump=self.root.update)
        self.assertEqual(self.source.calls, calls)

    def test_prefetched_frames_wait_for_their_presentation_time(self):
        app = self.app
        until(lambda: app.worker.status(app.request_id)['eof'], pump=self.root.update)
        self.assertEqual(app.worker.status(app.request_id)['size'], 15)
        app.toggle_play()
        # A full buffer must not dump its frames onto the screen at decode speed.
        end = time.monotonic() + .08
        while time.monotonic() < end:
            self.root.update()
            time.sleep(.005)
        self.assertEqual(app.displayed_frame, 0)
        for index in (1, 2, 3):
            app.audio.time = index / app.native_fps
            until(lambda: app.displayed_frame == index, pump=self.root.update)

    def test_audio_failure_keeps_video_running(self):
        app = self.app
        audio = app.audio
        app.toggle_play()
        audio.error = 'audio test error'
        until(lambda: app.audio is None, pump=self.root.update)
        until(lambda: app.displayed_frame > 0, pump=self.root.update)
        self.assertTrue(app.playing)
        self.assertIn('audio test error', app.status_label.cget('text'))

    def test_no_audio_flag_does_not_open_device(self):
        from src.archive.viewer import ViewerApp
        self.app.close()
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        with patch('src.archive.viewer.AudioPlayer', side_effect=AssertionError('audio disabled')):
            self.app = ViewerApp(self.root, self.source, 8, 0, 100, 100, 'test', audio=False)
        until(lambda: self.app.displayed_frame == 0, pump=self.root.update)
        self.app.toggle_play()
        until(lambda: self.app.displayed_frame > 0, pump=self.root.update)

    def test_custom_fps_changes_audio_speed_and_media_clock(self):
        app = self.app
        app.fps = 4
        app.speed = app.fps / app.native_fps
        app.toggle_play()
        self.assertEqual(app.audio.calls[-1], ('play', 0., .5))
        app.audio.time = .5
        until(lambda: app.displayed_frame == 4, pump=self.root.update)

    def test_missing_audio_device_keeps_viewer_usable(self):
        from src.archive.viewer import ViewerApp
        self.app.close()
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        with patch('src.archive.viewer.AudioPlayer', side_effect=RuntimeError('No audio output device')):
            self.app = ViewerApp(self.root, self.source, 8, 0, 100, 100, 'test')
        until(lambda: self.app.displayed_frame == 0, pump=self.root.update)
        self.app.toggle_play()
        until(lambda: self.app.displayed_frame > 0, pump=self.root.update)
        self.assertIn('No audio output device', self.app.status_label.cget('text'))


if __name__ == '__main__':
    unittest.main()
