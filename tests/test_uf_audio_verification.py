import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.archive.media import verify_audio


@unittest.skipUnless(shutil.which('ffmpeg'), 'Needs FFmpeg')
class AudioVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self, frame_ms):
        path = self.root / f'audio-{frame_ms}.opus'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi',
                        '-i', 'sine=frequency=440:sample_rate=48000', '-t', '1.0625',
                        '-ac', '1', '-c:a', 'libopus', '-b:a', '6k',
                        '-frame_duration', str(frame_ms), str(path)], check=True)
        return path

    def test_trimmed_opus_verifies_without_dts_errors_or_file_changes(self):
        actual_run = subprocess.run
        for frame_ms in (20, 60):
            with self.subTest(frame_ms=frame_ms):
                path = self.fixture(frame_ms)
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                results = []

                def capture(command, **kwargs):
                    result = actual_run(command, capture_output=True, **kwargs)
                    results.append(result)
                    return result

                with patch('src.archive.media.subprocess.run', side_effect=capture):
                    verify_audio(path)
                self.assertEqual(results[0].returncode, 0)
                self.assertEqual(results[0].stderr, b'')
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_sample_timestamps_preserve_every_decoded_sample(self):
        path = self.fixture(60)
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0']
        tail = ['-c:a', 'pcm_s16le', '-f', 's16le', '-']
        original = subprocess.run(command + tail, check=True, capture_output=True)
        normalized = subprocess.run(command + ['-af', 'asetpts=N/SR/TB'] + tail,
                                    check=True, capture_output=True)
        self.assertEqual(len(normalized.stdout), 51000 * 2)
        self.assertEqual(normalized.stdout, original.stdout)
        self.assertEqual(normalized.stderr, b'')

    def test_invalid_audio_still_fails_verification(self):
        path = self.root / 'invalid.opus'
        path.write_bytes(b'invalid Opus input')
        actual_run = subprocess.run
        with patch('src.archive.media.subprocess.run',
                   side_effect=lambda command, **kwargs: actual_run(command, capture_output=True, **kwargs)):
            with self.assertRaises(subprocess.CalledProcessError):
                verify_audio(path)
