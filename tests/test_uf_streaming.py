import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.archive.streaming import cookie_copy, select_formats, safe_id, download_pipe


class StreamingTests(unittest.TestCase):
    def test_cookie_copy_preserves_source_and_removes_working_copy(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cookies';p.write_text('original')
            with self.assertRaises(RuntimeError):
                with cookie_copy(p) as name:
                    self.assertEqual(os.stat(name).st_mode & 0o777, 0o660)
                    Path(name).write_text('changed');raise RuntimeError()
            self.assertEqual(p.read_text(),'original');self.assertFalse(Path(name).exists())

    def test_original_audio_wins_over_higher_bitrate_dub(self):
        v=dict(format_id='v',url='v',vcodec='vp9',height=720,width=1280,acodec='none')
        a=dict(format_id='it',url='a',vcodec='none',acodec='opus',language='it',format_note='original',abr=60)
        b=dict(a,format_id='en',language='en',format_note='dub',abr=200)
        self.assertEqual(select_formats({'formats':[v,a,b]})[1]['format_id'],'it')
        a['format_note']=''
        with self.assertRaises(ValueError):select_formats({'formats':[v,a,b]})

    def test_ids_cannot_escape_output_directory(self):
        with self.assertRaises(ValueError):safe_id('../escape')
        with self.assertRaises(ValueError):safe_id('channel_handle',channel=True)

    def test_failed_downloader_is_not_published_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'downloader';p.write_text('#!/bin/sh\nprintf partial\nexit 1\n');p.chmod(0o700)
            with self.assertRaisesRegex(RuntimeError,'stream failed'):
                with download_pipe(dict(binary=str(p),url='test'), 'v') as source:
                    self.assertEqual(source.read(),b'partial')

    def test_stream_cli_uses_encode_validation(self):
        import main
        with patch('sys.argv', ['main.py','stream','--source_urls','https://www.youtube.com/watch?v=abcdefghijk','--output_root','/tmp/test-uf','--runtime','int16']), patch('src.archive.dashboard.run',return_value=0) as run:
            self.assertEqual(main.main(),0)
            args=run.call_args.args[0]
            self.assertEqual(args.cuda_idx,[0])
            self.assertEqual(args.prefetch_frames,8)

    def test_source_height_cap_and_rt_stream_names(self):
        from src.archive.streaming import stream_jobs
        from main import build_parser
        channel = 'UC'+'a'*22
        video = dict(format_id='v',url='x',vcodec='vp9',height=720,width=1280,acodec='none')
        smaller = dict(video,format_id='small',height=360,width=640)
        info = dict(id='abcdefghijk',channel_id=channel,upload_date='20260102',timestamp=1767355200,formats=[video,smaller],chapters=[{'title':'Intro','start_time':0}],categories=['Food'])
        self.assertEqual(select_formats(info,480)[0]['format_id'],'small')
        with tempfile.TemporaryDirectory() as root:
            args=build_parser().parse_args(['stream','--source_urls','https://www.youtube.com/@example',
                                           '--output_root',root,'--procs','5'])
            with patch('src.archive.streaming.metadata',side_effect=[{'entries':[info],'_type':'playlist'},info]):
                jobs=list(stream_jobs(args,'yt-dlp',{'failed':0}))
            self.assertEqual(Path(jobs[0]['final']).name,'abcdefghijk_1767355200.uf.json')
            self.assertEqual(jobs[0]['remote']['video_format'],'small')
            self.assertEqual(jobs[0]['remote']['public_metadata'],info)
            self.assertEqual(Path(jobs[0]['final']).parent.name,channel)

    def test_unix_timestamp_fallback_is_utc_and_old_names_resume(self):
        import json
        from src.archive.streaming import upload_timestamp, existing_stream_archives
        self.assertEqual(upload_timestamp({'timestamp':1767355200,'upload_date':'20260101'}),1767355200)
        self.assertEqual(upload_timestamp({'release_timestamp':1767355200}),1767355200)
        self.assertEqual(upload_timestamp({'upload_date':'19700102'}),86400)
        with self.assertRaises(ValueError):upload_timestamp({})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'abcdefghijk_20260102.uf.json'
            url='https://www.youtube.com/watch?v=abcdefghijk'
            path.write_text(json.dumps({'format':'dcvc-uf-archive','source':{'url':url}}))
            before=path.read_bytes()
            self.assertEqual(existing_stream_archives(directory)[url],path)
            self.assertEqual(path.read_bytes(),before)

    def test_ffmpeg_decodes_streamed_container_without_source_file(self):
        import subprocess
        from src.archive.media import FrameReader
        producer = subprocess.Popen(['ffmpeg','-v','error','-f','lavfi','-i',
            'color=size=16x16:rate=2','-frames:v','2','-c:v','ffv1','-f','matroska','pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            with FrameReader('pipe:0',16,16,{'width':16,'height':16},input_pipe=producer.stdout) as reader:
                self.assertIsNotNone(reader.read())
                self.assertIsNotNone(reader.read())
                self.assertIsNone(reader.read())
            self.assertEqual(producer.wait(timeout=5),0)
        finally:
            producer.stdout.close()
            if producer.poll() is None:producer.kill();producer.wait()
