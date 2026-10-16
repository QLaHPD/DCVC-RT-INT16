import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.int16.scheduling import EncodeGates, GpuGate


def _hold_gate(path, writer, release):
    with GpuGate(path):
        writer.send('locked')
        release.wait(30)
    writer.close()


def _wait_gate(path, writer):
    writer.send('waiting')
    with GpuGate(path):
        writer.send('locked')
    writer.close()


def _probe_worker(writer, job, config, paths, device, instance, event_queue=None):
    with GpuGate(job['gpu_gate']):
        start = time.monotonic()
        time.sleep(.05)
        record = dict(source=job['source'], start=start, end=time.monotonic(), gate=job['gpu_gate'])
        with open(job['probe_log'], 'a') as output:
            output.write(json.dumps(record)+'\n')
    writer.send('encoded')
    writer.close()


def _stream_probe_worker(writer, job, config, paths, device, instance, event_queue=None):
    from src.archive.stream_pipeline import open_descriptor, close_descriptor, prepared_pipe
    descriptor = job['prepared_download']
    open_descriptor(descriptor)
    try:
        with prepared_pipe(descriptor) as stream:
            assert stream.read() == b'payload'
        _probe_worker(writer, job, config, paths, device, instance, event_queue)
    finally:
        close_descriptor(descriptor)


class SchedulingTests(unittest.TestCase):
    def test_namespaces_only_coordinate_shared_htl_gpus(self):
        config = dict(runtime='int16', variant='htl')
        gates = EncodeGates(config, [0, 0, 1])
        other = EncodeGates(config, [0, 0])
        try:
            original = {'source': 'source'}
            attached = gates.job(original, 0)
            self.assertNotIn('gpu_gate', original)
            self.assertIn('gpu_gate', attached)
            self.assertEqual(gates.job(original, 1), original)
            self.assertNotEqual(attached['gpu_gate'], other.job(original, 0)['gpu_gate'])
            directory = Path(attached['gpu_gate']).parent
            self.assertTrue(directory.exists())
        finally:
            gates.close(); other.close()
        self.assertFalse(directory.exists())
        for config, devices in ((None, [0, 0]),
                                (dict(runtime='fp16', variant='htl'), [0, 0]),
                                (dict(runtime='int16', variant='ld'), [0, 0]),
                                (dict(runtime='int16', variant='hts'), [0, 0]),
                                (dict(runtime='int16', variant='htl'), [0, 1]),
                                (dict(runtime='int16', variant='htl'), ['cpu', 'cpu'])):
            scope = EncodeGates(config, devices)
            self.assertIsNone(scope.temporary)
            scope.close()

    def test_owner_death_releases_waiting_process(self):
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'gpu.lock')
            release = context.Event()
            parent, child = context.Pipe(duplex=False)
            waiting, writer = context.Pipe(duplex=False)
            holder = context.Process(target=_hold_gate, args=(path, child, release))
            waiter = context.Process(target=_wait_gate, args=(path, writer))
            holder.start(); child.close()
            try:
                self.assertTrue(parent.poll(10)); self.assertEqual(parent.recv(), 'locked')
                waiter.start(); writer.close()
                self.assertTrue(waiting.poll(10)); self.assertEqual(waiting.recv(), 'waiting')
                self.assertFalse(waiting.poll(.1))
                holder.terminate(); holder.join(timeout=10)
                self.assertTrue(waiting.poll(10)); self.assertEqual(waiting.recv(), 'locked')
                waiter.join(timeout=10)
                self.assertFalse(waiter.is_alive())
                self.assertEqual(waiter.exitcode, 0)
            finally:
                for process in (holder, waiter):
                    if process.pid is not None and process.is_alive():
                        process.terminate(); process.join(timeout=10)
                        if process.is_alive(): process.kill(); process.join()
                parent.close(); waiting.close(); writer.close()

    def test_model_holds_gate_through_gpu_completion_and_errors(self):
        import torch
        from src.int16.model import IntegerModel
        events = []
        class Gate:
            def __enter__(self): events.append('enter')
            def __exit__(self, *exc): events.append('exit')
        model = SimpleNamespace(gpu_gate=Gate(), device=torch.device('cuda:0'),
                                _compress=Mock(return_value='encoded'))
        x = torch.empty((1, 3, 144, 144), dtype=torch.int16)
        with patch('torch.cuda.synchronize', side_effect=lambda *_: events.append('sync')):
            self.assertEqual(IntegerModel.compress(model, x, 10), 'encoded')
            self.assertEqual(events, ['enter', 'sync', 'exit'])
            events.clear(); model._compress.side_effect = ValueError('encode failed')
            with self.assertRaisesRegex(ValueError, 'encode failed'):
                IntegerModel.compress(model, x, 10)
            self.assertEqual(events, ['enter', 'sync', 'exit'])
            events.clear(); model._compress.side_effect = None
            IntegerModel.compress(model, x[:, :, :128, :128], 10)
            self.assertEqual(events, [])
            model.device = torch.device('cpu')
            IntegerModel.compress(model, x, 10)
            self.assertEqual(events, [])

    def test_archive_workers_share_gate_and_cleanup_without_gpu(self):
        from src.archive.workflow import execute_jobs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root/'events.jsonl'
            jobs = [dict(source=str(root/name), final=str(root/(name+'.uf.json')), probe_log=str(log))
                    for name in ('first', 'second')]
            args = SimpleNamespace(procs=2, device='cuda', cuda_idx=[0], shared_instance=None)
            outcomes, failed = execute_jobs(args, jobs, dict(runtime='int16', variant='htl'), [],
                                            worker_entry=_probe_worker)
            self.assertEqual(outcomes, ['encoded', 'encoded'])
            self.assertFalse(failed)
            records = sorted((json.loads(line) for line in log.read_text().splitlines()), key=lambda r:r['start'])
            self.assertEqual(records[0]['gate'], records[1]['gate'])
            self.assertLessEqual(records[0]['end'], records[1]['start'])
            self.assertFalse(Path(records[0]['gate']).parent.exists())
            self.assertTrue(all('gpu_gate' not in job for job in jobs))

    def test_stream_workers_share_gate_without_gpu_or_network(self):
        from src.archive.stream_pipeline import execute_stream_jobs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'downloader'
            binary.write_text('#!'+sys.executable+'\nimport os\nos.write(1,b"payload")\n')
            binary.chmod(0o700)
            log = root/'events.jsonl'
            jobs = [dict(source=name, final=str(root/(name+'.uf.json')), probe_log=str(log),
                         remote=dict(binary=str(binary), url=name, video_format='v'))
                    for name in ('first', 'second')]
            args = SimpleNamespace(device='cuda', procs=2, cuda_idx=[0], shared_instance='test')
            with patch('src.archive.streaming.stream_jobs', return_value=iter(jobs)), \
                    patch('src.archive.workflow._job_entry', _stream_probe_worker):
                outcomes, failed = execute_stream_jobs(args, dict(runtime='int16', variant='htl'),
                                                       (), str(binary), {})
            self.assertEqual(outcomes, ['encoded', 'encoded'])
            self.assertFalse(failed)
            records = sorted((json.loads(line) for line in log.read_text().splitlines()), key=lambda r:r['start'])
            self.assertEqual(records[0]['gate'], records[1]['gate'])
            self.assertLessEqual(records[0]['end'], records[1]['start'])
            self.assertFalse(Path(records[0]['gate']).parent.exists())


if __name__ == '__main__':
    unittest.main()
