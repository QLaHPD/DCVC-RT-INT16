import contextlib
import io
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from test_uf_archive import ArchiveTests as Fixtures, FakeCodec
from src.archive.storage import load_bundle, artifact_path, publish_flat
from src.archive.workflow import run_cleanup, run_decode
from src.archive.lifecycle import inventory, finish
from src.archive.rt_managed import dashboard_module


class FlatLifecycleTests(unittest.TestCase):
    setUp = Fixtures.setUp
    encode = Fixtures.encode

    def flat(self):
        legacy = self.final
        self.final = legacy.with_name('source.uf.json')
        self.job.update(final=str(self.final), layout='flat', legacy=str(legacy), lease_key=legacy.name)

    def options(self, **extra):
        values=dict(input=str(self.final), base_root=None, yes=True, model_path_i=None, model_path_p=None,
                    prepared_i=None, prepared_p=None, runtime='auto', device='cuda', cuda_idx=0,
                    output_root=str(self.final.parent), channel_ids=None, auto_delete=False,
                    keep_originals=False, cleanup_dry_run=False)
        return types.SimpleNamespace(**dict(values, **extra))

    def test_flat_encode_resume_and_decoder_find_sibling_files(self):
        self.flat()
        self.assertEqual(self.encode(),'encoded')
        self.assertTrue(self.final.is_file())
        self.assertTrue(self.final.with_name('source.bin').is_file())
        self.assertTrue(self.final.with_name('source.opus').is_file())
        self.assertFalse(self.final.with_name('source.mkv.uf').exists())
        root, data=load_bundle(self.final.with_name('source.bin'))
        self.assertEqual(data['files']['video.bin'],'source.bin')
        self.assertEqual(self.encode(),'resumed')
        with patch('src.archive.workflow.checked_codec',return_value=FakeCodec()):
            self.assertEqual(run_decode(self.options(input=str(root/'source.bin')),verify_only=True),0)

    def test_existing_opus_is_reused_without_reencoding(self):
        from src.archive.media import encode_audio
        from src.archive.storage import sha256
        self.flat(); self.final.parent.mkdir()
        opus=self.final.with_name('source.opus')
        encode_audio(self.source,opus,'mono','6k',1)
        before=sha256(opus)
        with patch('src.archive.workflow.encode_audio',side_effect=AssertionError('must reuse audio')):
            self.assertEqual(self.encode(),'encoded')
        self.assertEqual(sha256(opus),before)

    def test_inventory_remaps_another_machines_source(self):
        import shutil
        self.flat();self.assertEqual(self.encode(),'encoded')
        mount=self.root/'another-mount';mount.mkdir()
        shutil.copy2(self.source,mount/self.source.name)
        self.source.unlink()
        row=next(iter(inventory([self.final.parent],base_root=mount).values()))
        self.assertEqual(row['originals'],1)
        self.assertEqual(row['archives'][0]['source'],str(mount/self.source.name))

    def test_missing_metadata_and_unprocessed_sources_block_channel(self):
        from src.archive.lifecycle import channel_blockers
        self.flat();self.assertEqual(self.encode(),'encoded')
        row=next(iter(inventory([self.final.parent]).values()))
        self.assertTrue(channel_blockers(row,self.options(allow_missing_metadata=False)))
        self.source.with_name('another.mkv').write_bytes(b'pending')
        self.assertTrue(channel_blockers(row,self.options(allow_missing_metadata=True)))

    def test_flat_collision_never_overwrites(self):
        self.flat();self.final.parent.mkdir()
        other=self.final.with_name('source.bin');other.write_bytes(b'RT or unrelated output')
        self.assertEqual(self.encode(),'failed')
        self.assertEqual(other.read_bytes(),b'RT or unrelated output')
        self.assertFalse(self.final.exists());self.assertTrue(self.source.exists())

    def test_interrupted_publication_recovers_identical_payload(self):
        self.flat()
        original=publish_flat
        def fail(stage,manifest,metadata,pulse,key):
            # Simulate a crash immediately before committing the manifest.
            import os
            os.link(stage/'video.bin', Path(manifest).with_name('source.bin'))
            raise RuntimeError('interrupted publication')
        with patch('src.archive.workflow.publish_flat',side_effect=fail):
            self.assertEqual(self.encode(),'failed')
        self.assertFalse(self.final.exists())
        self.assertEqual(self.encode(),'encoded')
        load_bundle(self.final)

    def test_lost_owner_cannot_publish_manifest(self):
        self.flat()
        def fail(stage,manifest,metadata,pulse,key):
            pulse.check=lambda: (_ for _ in ()).throw(RuntimeError('lost lease'))
            return publish_flat(stage,manifest,metadata,pulse,key)
        with patch('src.archive.workflow.publish_flat',side_effect=fail):
            self.assertEqual(self.encode(),'failed')
        self.assertFalse(self.final.exists())
        self.assertFalse(self.final.with_name('source.bin').exists())

    def test_legacy_is_resumed_without_reencoding(self):
        self.assertEqual(self.encode(),'encoded')
        legacy=self.final
        self.flat()
        self.assertEqual(self.encode(),'resumed')
        self.assertFalse(self.final.exists());load_bundle(legacy)

    def test_plain_auto_delete_validates_and_deletes_only_original(self):
        self.flat();self.assertEqual(self.encode(),'encoded')
        other=self.source.with_name('unrelated.txt');other.write_text('keep')
        with patch('src.archive.workflow.checked_codec',return_value=FakeCodec()),contextlib.redirect_stdout(io.StringIO()):
            finish(self.options(auto_delete=True))
        self.assertFalse(self.source.exists());self.assertTrue(other.exists());load_bundle(self.final)
        self.assertTrue(self.final.with_name('source.cleanup.json').exists())
        row=next(iter(inventory([self.final.parent]).values()))
        self.assertEqual(row['originals'],0);self.assertEqual(row['absent'],1)

    def test_failed_channel_dry_run_and_changed_original_are_retained(self):
        self.flat();self.assertEqual(self.encode(),'encoded')
        with contextlib.redirect_stdout(io.StringIO()):
            finish(self.options(auto_delete=True),failed_channels=[str(self.final.parent)])
            finish(self.options(cleanup_dry_run=True))
        self.assertTrue(self.source.exists())
        self.source.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError,'differs'):
            run_cleanup(self.options())
        self.assertTrue(self.source.exists())

    def test_tui_waits_for_confirmation_even_on_completed_run(self):
        self.flat();self.assertEqual(self.encode(),'encoded')
        rows=inventory([self.final.parent])
        module=dashboard_module()
        with contextlib.redirect_stderr(io.StringIO()):
            dashboard=module.EncodeDashboard(1,1,self.final.parent,mode='plain')
        row=next(iter(rows.values()))
        dashboard.channels=[dict(channel_id=row['channel'],status='awaiting-approval',total=1)]
        class Keys:
            def __init__(self,keys):self.keys=iter(keys)
            def getch(self):return next(self.keys,-1)
        with patch.object(dashboard,'_draw'):
            dashboard._screen=Keys([ord('a'),ord('d')])
            self.assertEqual(dashboard.poll_actions(),[])
            self.assertEqual(dashboard.views[dashboard.view_index],'approvals')
            dashboard._screen=Keys([ord('n')]);self.assertEqual(dashboard.poll_actions(),[])
            dashboard._screen=Keys([ord('d'),ord('y')]);actions=dashboard.poll_actions()
        dashboard._screen=None;dashboard.close()
        self.assertEqual(actions[0].kind,'approve')
        action={'action':'delete','channel':actions[0].channel_id}
        class Commands:
            def __init__(self):self.items=iter([action, {'action':'quit'}]);self.calls=0
            def get(self,timeout):self.calls+=1;return next(self.items)
        commands=Commands()
        with patch('src.archive.workflow.checked_codec',return_value=FakeCodec()),contextlib.redirect_stdout(io.StringIO()):
            finish(self.options(),commands)
        self.assertEqual(commands.calls,2);self.assertFalse(self.source.exists())

    def test_streamed_archive_never_requests_local_deletion(self):
        self.flat();self.assertEqual(self.encode(),'encoded')
        data=json.loads(self.final.read_text());data['source']['kind']='youtube';data['source']['url']='https://www.youtube.com/watch?v=abcdefghijk'
        self.final.write_text(json.dumps(data))
        row=next(iter(inventory([self.final.parent]).values()))
        self.assertEqual(row['streamed'],1);self.assertEqual(row['originals'],0)
        with self.assertRaisesRegex(ValueError,'no stored original'):
            run_cleanup(self.options())
        self.assertTrue(self.source.exists())

    def test_retry_reuses_validated_opus_after_manifest_commit_interruption(self):
        import os
        self.flat()
        original_link = os.link
        def interrupt(source, target, *args, **kwargs):
            if Path(target) == self.final:
                raise OSError('simulated interruption before manifest commit')
            return original_link(source, target, *args, **kwargs)
        with patch('src.archive.storage.os.link', side_effect=interrupt):
            self.assertEqual(self.encode(), 'failed')
        audio = self.final.with_name('source.opus').read_bytes()
        self.assertFalse(self.final.exists())
        with patch('src.archive.workflow.make_codec', side_effect=AssertionError('must reuse validated stage')):
            self.assertEqual(self.encode(), 'encoded')
        self.assertEqual(self.final.with_name('source.opus').read_bytes(), audio)
        load_bundle(self.final)
        self.assertFalse(list(self.final.parent.glob('.*.pending')))

    def test_encode_cleanup_scope_excludes_other_originals(self):
        self.flat(); self.assertEqual(self.encode(), 'encoded')
        args = self.options(auto_delete=True)
        args._cleanup_sources = set()
        with contextlib.redirect_stdout(io.StringIO()):
            finish(args)
        self.assertTrue(self.source.exists())

    def test_peer_deletion_is_visible_and_second_cleanup_is_idempotent(self):
        self.flat(); self.assertEqual(self.encode(), 'encoded')
        with patch('src.archive.workflow.checked_codec', return_value=FakeCodec()), contextlib.redirect_stdout(io.StringIO()):
            run_cleanup(self.options())
        with patch('src.archive.workflow.checked_codec', side_effect=AssertionError('already absent')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_cleanup(self.options()), 0)
        self.assertEqual(next(iter(inventory([self.final.parent]).values()))['absent'], 1)


# The imported fixture class is not an additional test suite here.
del Fixtures

class DashboardQueueTests(unittest.TestCase):
    def test_spawned_worker_events_reach_dashboard_without_gpu(self):
        import multiprocessing
        import tempfile
        from src.archive.workflow import _job_entry
        context = multiprocessing.get_context('spawn')
        events=context.Queue();parent,child=context.Pipe(duplex=False)
        with tempfile.TemporaryDirectory() as folder:
            job=dict(source=str(Path(folder)/'missing.mkv'),relative='missing.mkv',final=str(Path(folder)/'out/missing.uf'))
            worker=context.Process(target=_job_entry,args=(child,job,{},(),0,'queue-test'),kwargs={'event_queue':events})
            try:
                worker.start();child.close()
                self.assertTrue(parent.poll(15));self.assertEqual(parent.recv(),'failed')
                message=events.get(timeout=5)
                self.assertEqual(message['event'],'encode_failed')
                worker.join(timeout=5);self.assertEqual(worker.exitcode,0)
            finally:
                if worker.is_alive():worker.terminate();worker.join()
                parent.close();child.close();events.close()

class RTFilenameTests(unittest.TestCase):
    setUp = FlatLifecycleTests.setUp
    encode = FlatLifecycleTests.encode
    flat = FlatLifecycleTests.flat
    options = FlatLifecycleTests.options
    # Inherit setup/encode helpers only; keep this test separate from old-flat compatibility tests.
    def test_rt_filename_and_plain_audio_use_source_id(self):
        self.flat(); self.job['filename_style']='rt'
        self.assertEqual(self.encode(),'encoded')
        video=self.final.with_name('source_48x32_qI36_qP30.bin')
        self.assertTrue(video.is_file())
        self.assertFalse(self.final.with_name('source.bin').exists())
        self.assertTrue(self.final.with_name('source.opus').exists())
        root,data=load_bundle(video)
        self.assertEqual(data['files']['video.bin'],video.name)
        self.assertEqual(self.encode(),'resumed')
        with patch('src.archive.workflow.checked_codec',return_value=FakeCodec()):
            self.assertEqual(run_decode(self.options(input=str(video)),verify_only=True),0)
        with self.assertRaisesRegex(ValueError,'does not belong'):
            load_bundle(self.final.with_name('source_96x64_qI36_qP30.bin'))
