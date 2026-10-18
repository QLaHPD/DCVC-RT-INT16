import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser
from src.archive.coordinator import archived_job, check_completed, validate_completed
from src.archive.profiles import normalize_inputs
from src.archive.storage import identity, load_bundle, sha256
from src.archive.workflow import discover, encode_job


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source.mp4'
        self.source.write_bytes(b'original-video')
        self.config = {'codec': 'dcvc-uf', 'audio': 'opus'}
        self.parent = self.root / 'output'
        self.parent.mkdir()
        self.final = self.parent / 'source.uf.json'
        self.payloads = {'video.bin': b'compressed-video', 'audio.opus': b'compressed-audio'}
        self.record = dict(
            format='dcvc-uf-archive', version=1, layout='flat', archive_stem='source',
            pipeline=self.config, source=dict(path=str(self.source), relative_path=self.source.name,
                sha256=sha256(self.source), media={'audio': True}, **identity(self.source)),
            video={'frames': 4}, validation={'mode': 'artifact-hashes', 'decode_performed': False},
            artifacts={}, files={'video.bin': 'source.bin', 'audio.opus': 'source.opus'})
        for name, value in self.payloads.items():
            path = self.parent / self.record['files'][name]
            path.write_bytes(value)
            self.record['artifacts'][name] = {'size': len(value), 'sha256': sha256(path)}
        self.save()
        self.job = dict(source=str(self.source), relative=self.source.name, final=str(self.final),
                        legacy=str(self.parent / 'source.mp4.uf'), layout='flat',
                        lease_key='source.mp4.uf', allow_missing_original=True)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        events = patch('src.archive.coordinator.event')
        events.start()
        self.addCleanup(events.stop)

    def save(self):
        self.final.write_text(json.dumps(self.record))

    def status(self, mode='fast'):
        return validate_completed(dict(self.job, resume_check=mode), self.config)

    def worker_status(self, mode='fast'):
        with contextlib.redirect_stdout(io.StringIO()):
            return encode_job(dict(self.job, resume_check=mode), self.config, (), 'cpu', 'test')

    @contextlib.contextmanager
    def no_payload_reads(self):
        original_open = Path.open
        payloads = {self.source} | {self.parent / name for name in self.record['files'].values()}
        def guarded_open(path, *args, **kwargs):
            if path in payloads:
                raise AssertionError(f'Resume read payload: {path}')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_open), \
             patch('src.archive.storage.sha256', side_effect=AssertionError('Resume hashed payload')), \
             patch('src.archive.workflow.sha256', side_effect=AssertionError('Resume hashed original')):
            yield

    def test_fast_resume_reads_no_payloads_in_startup_peer_or_worker_checks(self):
        with self.no_payload_reads():
            self.assertEqual(validate_completed(self.job, self.config), 'resumed')
            self.assertEqual(check_completed([self.job], self.config, 1), ['resumed'])
            self.assertEqual(self.worker_status(), 'resumed')

    def test_legacy_bundle_also_resumes_without_payload_reads(self):
        self.final.unlink()
        legacy = Path(self.job['legacy'])
        legacy.mkdir()
        for name in self.record['files'].values():
            (self.parent / name).rename(legacy / name)
        self.parent = legacy
        self.final = legacy / 'manifest.json'
        for key in ('layout', 'archive_stem'):
            del self.record[key]
        self.save()
        with self.no_payload_reads():
            self.assertEqual(self.status(), 'resumed')
            self.assertEqual(self.worker_status(), 'resumed')

    def test_fast_rejects_changed_original_size_or_mtime(self):
        self.source.write_bytes(b'a longer original')
        self.assertEqual(self.status(), 'failed')
        self.assertEqual(self.worker_status(), 'failed')
        self.source.write_bytes(b'original-video')
        saved = self.record['source']['mtime_ns']
        os.utime(self.source, ns=(saved, saved + 1_000_000_000))
        self.assertEqual(self.status(), 'failed')
        self.assertEqual(self.worker_status(), 'failed')
        self.assertEqual(self.status('full'), 'resumed')
        self.assertEqual(self.worker_status('full'), 'resumed')

    def test_full_detects_original_content_change_with_same_size_and_mtime(self):
        self.source.write_bytes(b'X' * self.record['source']['size'])
        saved = self.record['source']['mtime_ns']
        os.utime(self.source, ns=(saved, saved))
        self.assertEqual(self.status(), 'resumed')
        self.assertEqual(self.status('full'), 'failed')
        self.assertEqual(self.worker_status('full'), 'failed')

    def test_fast_resume_accepts_whole_second_timestamps_in_either_direction(self):
        whole_second = 1_768_943_251_000_000_000
        fractional = whole_second + 823_494_434
        for saved, current in ((fractional, whole_second), (whole_second, fractional)):
            with self.subTest(saved=saved, current=current):
                self.record['source']['mtime_ns'] = saved
                os.utime(self.source, ns=(current, current))
                self.save()
                with self.no_payload_reads():
                    self.assertEqual(self.status(), 'resumed')
                    self.assertEqual(self.worker_status(), 'resumed')

    def test_fast_resume_rejects_distinct_seconds_or_fractional_timestamps(self):
        saved = 1_768_943_251_823_494_434
        self.record['source']['mtime_ns'] = saved
        self.save()
        for current in (saved + 1, 1_768_943_252_000_000_000):
            with self.subTest(current=current):
                os.utime(self.source, ns=(current, current))
                self.assertEqual(self.status(), 'failed')
                self.assertEqual(self.worker_status(), 'failed')

    def test_full_detects_same_size_artifact_corruption_and_reader_stays_strict(self):
        (self.parent / 'source.bin').write_bytes(b'X' * len(self.payloads['video.bin']))
        self.assertEqual(self.status(), 'resumed')
        self.assertEqual(self.status('full'), 'failed')
        self.assertEqual(self.worker_status('full'), 'failed')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            load_bundle(self.final)

    def test_fast_rejects_missing_truncated_or_symlinked_artifacts(self):
        path = self.parent / 'source.bin'
        for kind in ('missing', 'truncated', 'symlink'):
            with self.subTest(kind=kind):
                path.unlink(missing_ok=True)
                if kind == 'truncated':
                    path.write_bytes(b'x')
                elif kind == 'symlink':
                    other = self.root / 'other.bin'
                    other.write_bytes(self.payloads['video.bin'])
                    path.symlink_to(other)
                self.assertEqual(self.status(), 'failed')
                self.assertEqual(self.worker_status(), 'failed')

    def test_missing_original_is_supported_but_symlinks_are_rejected(self):
        self.source.unlink()
        with self.no_payload_reads():
            self.assertEqual(self.status(), 'resumed')
            self.assertEqual(self.worker_status(), 'resumed')
        self.source.symlink_to(self.root / 'absent.mp4')
        self.assertEqual(self.status(), 'failed')
        self.assertEqual(self.worker_status(), 'failed')

    def test_fast_preserves_manifest_and_pipeline_checks(self):
        original = copy.deepcopy(self.record)
        for kind in ('completion', 'pipeline', 'audio', 'video', 'path', 'digest'):
            with self.subTest(kind=kind):
                self.record = copy.deepcopy(original)
                if kind == 'completion':
                    self.record['validation'] = {}
                elif kind == 'pipeline':
                    self.record['pipeline'] = dict(self.config, qp_i=1)
                elif kind in ('audio', 'video'):
                    name = 'audio.opus' if kind == 'audio' else 'video.bin'
                    del self.record['artifacts'][name]
                    del self.record['files'][name]
                elif kind == 'path':
                    self.record['files']['video.bin'] = '../outside.bin'
                else:
                    self.record['artifacts']['video.bin']['sha256'] = 'invalid'
                self.save()
                self.assertEqual(self.status(), 'failed')

    def test_cli_defaults_and_policy_propagation(self):
        for command in ('join', 'encode', 'stream'):
            for mode in ('fast', 'full'):
                with self.subTest(command=command, mode=mode):
                    flags = [] if mode == 'fast' else ['--resume-check', 'full']
                    args = build_parser().parse_args([command, '--root', str(self.root), *flags])
                    self.assertEqual(args.resume_check, mode)
                    if command != 'stream':
                        normalize_inputs(args)
                        jobs = discover(args)
                        self.assertEqual(len(jobs), 1)
                        self.assertEqual(jobs[0]['resume_check'], mode)
                    args.base_root = str(self.root)
                    args.output_root = str(self.parent)
                    recovered = archived_job(self.final, self.record, args)
                    self.assertEqual(recovered['resume_check'], mode)


if __name__ == '__main__':
    unittest.main()
