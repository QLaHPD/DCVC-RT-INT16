import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.archive.media import FrameReader
from src.archive.stream_pipeline import (BufferPolicy, PreparedDownload, near_download_end,
                                        open_descriptor, close_descriptor, prepared_pipe,
                                        execute_stream_jobs)


def _consume_worker(writer, job, config, paths, device, instance, event_queue=None):
    descriptor = job['prepared_download']
    try:
        open_descriptor(descriptor)
        with prepared_pipe(descriptor) as stream:
            payload = stream.read()
        policy = job['buffer_policy']
        policy.claim()
        policy.observe(300, 60)
        with open(job['log'], 'a') as output:
            output.write(json.dumps(dict(event='consume', source=job['source'], payload=payload.decode(),
                                         capacity=policy.capacity(), time=time.monotonic()))+'\n')
        time.sleep(1)
        with open(job['log'], 'a') as output:
            output.write(json.dumps(dict(event='worker_done', source=job['source'],time=time.monotonic()))+'\n')
        writer.send('encoded')
    except Exception:
        writer.send('failed')
    finally:
        close_descriptor(descriptor)
        writer.close()


class _FramesCodec:
    def encode(self, reader, path, qi, qp, reset, intra, limit, progress):
        count = 0
        while reader.read() is not None:
            count += 1
            progress(frames=count, fps=2)
        Path(path).write_bytes(b'encoded-'+str(count).encode())
        return dict(frames=count,packets=count,encode_seconds=count/2)


def _encode_fake_worker(*args, **kwargs):
    from src.archive.workflow import _job_entry
    with patch('src.archive.workflow.make_codec',return_value=_FramesCodec()):
        return _job_entry(*args, **kwargs)


class StreamPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = multiprocessing.get_context('spawn')

    def downloader(self):
        binary = self.root/'downloader'
        log = self.root/'events.jsonl'
        binary.write_text('#!'+sys.executable+'\n'+'''
import os,sys,json,time
url=sys.argv[-1]
with open('''+repr(str(log))+''','a') as output:
 output.write(json.dumps(dict(event='download_start',source=url,time=time.monotonic()))+'\\n')
if url=='block':
 while True: os.write(1,b'x'*65536)
print('UF_DOWNLOAD:80:100:800:1000:NA:downloading',file=sys.stderr,flush=True)
os.write(1,('payload-'+url).encode())
if url=='fail':sys.exit(5)
''')
        binary.chmod(0o700)
        return binary,log

    def test_calibration_happens_once_after_five_seconds(self):
        policy = BufferPolicy(self.context)
        self.assertEqual(policy.capacity(),8)
        policy.claim()
        self.assertFalse(policy.observe(240,60))
        self.assertTrue(policy.observe(301,60.1))
        self.assertEqual(policy.capacity(),121)
        self.assertFalse(policy.observe(1000,100))
        policy.release();policy.claim()
        self.assertEqual(policy.capacity(),121)
        self.assertFalse(policy.observe(500,10))

    def test_short_or_failed_first_video_allows_next_sample(self):
        policy=BufferPolicy(self.context);policy.claim()
        self.assertFalse(policy.observe(10,20))
        policy.release();policy.claim()
        self.assertTrue(policy.observe(100,20));self.assertEqual(policy.capacity(),40)

    def test_fragment_threshold_and_nonfragment_fallback(self):
        self.assertFalse(near_download_end('UF_DOWNLOAD:79:100:990:1000:NA:downloading'))
        self.assertTrue(near_download_end('UF_DOWNLOAD:80:100:800:1000:NA:downloading'))
        self.assertFalse(near_download_end('UF_DOWNLOAD:NA:NA:949:1000:NA:downloading'))
        self.assertTrue(near_download_end('UF_DOWNLOAD:NA:NA:950:NA:1000:downloading'))
        self.assertTrue(near_download_end('UF_DOWNLOAD:NA:NA:NA:NA:NA:finished'))
        self.assertFalse(near_download_end('unrelated diagnostic'))

    def test_prepared_pipe_reports_download_failure(self):
        binary,_=self.downloader()
        prepared=PreparedDownload(dict(binary=str(binary),url='fail',video_format='v'),self.context)
        descriptor=prepared.descriptor();open_descriptor(descriptor)
        try:
            with self.assertRaisesRegex(RuntimeError,'yt-dlp stream failed'):
                with prepared_pipe(descriptor) as stream:
                    self.assertEqual(stream.read(),b'payload-fail')
        finally:
            close_descriptor(descriptor);prepared.close()

    def test_cancel_unconsumed_prefetch_cleans_process_and_cookie_copy(self):
        binary,_=self.downloader()
        cookies=self.root/'cookies';cookies.write_text('original')
        prepared=PreparedDownload(dict(binary=str(binary),url='block',video_format='v',cookies=str(cookies)),self.context)
        name=prepared.cookie_path
        self.assertEqual(os.stat(name).st_mode & 0o777,0o660)
        started=time.monotonic();prepared.close()
        self.assertLess(time.monotonic()-started,5)
        self.assertIsNotNone(prepared.process.poll())
        self.assertFalse(Path(name).exists());self.assertEqual(cookies.read_text(),'original')

    def test_next_downloader_starts_before_current_worker_finishes(self):
        binary,log=self.downloader()
        jobs=[dict(source=name,final=str(self.root/(name+'.uf.json')),log=str(log),
                   remote=dict(binary=str(binary),url=name,video_format='v')) for name in ('one','two')]
        args=SimpleNamespace(device='cpu',procs=1,cuda_idx=[0],shared_instance='test')
        with patch('src.archive.streaming.stream_jobs',return_value=iter(jobs)),patch('src.archive.workflow._job_entry',_consume_worker):
            outcomes,failed=execute_stream_jobs(args,{},(),str(binary),{})
        self.assertEqual(outcomes,['encoded','encoded']);self.assertFalse(failed)
        events=[json.loads(line) for line in log.read_text().splitlines()]
        start=next(e['time'] for e in events if e['source']=='two' and e['event']=='download_start')
        finish=next(e['time'] for e in events if e['source']=='one' and e['event']=='worker_done')
        self.assertLess(start,finish)
        for e in events:
            if e['event']=='consume':
                self.assertEqual(e['payload'],'payload-'+e['source'])
                self.assertEqual(e['capacity'],120)

    def test_dynamic_frame_queue_preserves_pixels_and_eof(self):
        source=self.root/'fixture.mkv'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=256x144:rate=10',
                        '-frames:v','20','-c:v','ffv1',str(source)],check=True)
        original={'width':256,'height':144}
        sequences=[]
        for adaptive in (False,True):
            with FrameReader(source,256,144,original,prefetch_frames=8,adaptive_buffer=adaptive) as reader:
                if adaptive:
                    reader.prime();reader.set_prefetch_frames(140)
                    self.assertEqual(reader._queue.maxsize,140)
                frames=[]
                while (frame:=reader.read()) is not None:frames.append(frame.tobytes())
                sequences.append(frames)
        self.assertEqual(len(sequences[0]),20);self.assertEqual(sequences[0],sequences[1])

    def test_prefetched_pipe_runs_through_real_worker_and_ffmpeg(self):
        from src.archive.storage import load_bundle
        source=self.root/'fixture.mkv'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=256x144:rate=10',
                        '-frames:v','20','-c:v','ffv1',str(source)],check=True)
        binary=self.root/'container-downloader'
        binary.write_text('#!'+sys.executable+'\nimport sys,os\n'
                          'print("UF_DOWNLOAD:80:100:800:1000:NA:downloading",file=sys.stderr,flush=True)\n'
                          'with open(sys.argv[-1],"rb") as source:\n'
                          ' while chunk:=source.read(65536):os.write(1,chunk)\n')
        binary.chmod(0o700)
        remote=dict(binary=str(binary),url=str(source),video_format='v',
                    media=dict(width=256,height=144,fps=10,duration=2,audio=False),public_metadata={'id':'one'})
        job=dict(source='one',relative='one',final=str(self.root/'out/one.uf.json'),remote=remote,
                 layout='flat',filename_style='rt',lease_key='one.uf',input_threads=1)
        config=dict(codec='dcvc-uf',variant='htl',fps=10,resolution=144,qp_i=10,qp_p=14,
                    reset_interval=2,intra_period=-1,max_frames=None,audio='none')
        args=SimpleNamespace(device='cpu',procs=1,cuda_idx=[0],shared_instance='test')
        with patch('src.archive.streaming.stream_jobs',return_value=iter([job])),patch('src.archive.workflow._job_entry',_encode_fake_worker):
            outcomes,failed=execute_stream_jobs(args,config,(),str(binary),{})
        self.assertEqual(outcomes,['encoded']);self.assertFalse(failed)
        _,data=load_bundle(job['final'])
        self.assertEqual(data['video']['frames'],20)
        self.assertFalse(data['validation']['decode_performed'])
        self.assertEqual(job['buffer_policy'].capacity(),4)

    def test_metadata_request_can_be_cancelled(self):
        from src.archive.streaming import metadata
        binary=self.root/'slow';binary.write_text('#!/bin/sh\nsleep 30\n');binary.chmod(0o700)
        stop=threading.Event();timer=threading.Timer(.2,stop.set);timer.start()
        started=time.monotonic()
        try:
            with self.assertRaises(InterruptedError):metadata('url',str(binary),None,cancel_event=stop)
        finally:timer.cancel()
        self.assertLess(time.monotonic()-started,3)


if __name__=='__main__':unittest.main()
