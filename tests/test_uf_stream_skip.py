import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser
from src.archive.streaming import stream_jobs


class StreamSkipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.channel = 'UC'+'a'*22
        self.folder = self.root/self.channel
        self.folder.mkdir()
        self.config = {'audio':'none'}
        self.video_id = 'abcdefghijk'
        self.url = 'https://www.youtube.com/watch?v='+self.video_id
        self.bin = self.folder/(self.video_id+'_123_256x144_qI10_qP14.bin')
        self.bin.write_bytes(b'complete')
        self.record = dict(format='dcvc-uf-archive', pipeline=self.config,
                           source={'url':self.url}, validation={'mode':'artifact-hashes','decode_performed':False},
                           artifacts={'video.bin':{'size':8,'sha256':'unused-for-fast-stat-check'}},
                           files={'video.bin':self.bin.name})
        self.manifest = self.folder/(self.video_id+'_123.uf.json')
        self.save()
        self.args = build_parser().parse_args(['stream','--source_urls','https://www.youtube.com/@example',
                                               '--output_root',str(self.root)])
        self.listing = {'_type':'playlist','channel_id':self.channel,'entries':[{'id':self.video_id}]}
        self.counts = {'failed':0}

    def save(self):
        self.manifest.write_text(json.dumps(self.record))

    def test_complete_output_needs_only_channel_listing_request(self):
        with patch('src.archive.streaming.metadata',return_value=self.listing) as fetch, patch('src.archive.streaming.event') as events:
            self.assertEqual(list(stream_jobs(self.args,'yt-dlp',self.counts,config=self.config)),[])
        self.assertEqual(fetch.call_count,1)
        self.assertEqual(self.counts['resumed'],1)
        self.assertFalse(any(call.args[0] in ('stream_resolving','already_encoded','stream_selected') for call in events.call_args_list))

    def test_missing_partial_or_mismatched_outputs_are_not_skipped(self):
        for kind in ('missing','truncated','no-manifest','no-completion','different-config','missing-audio'):
            with self.subTest(kind=kind):
                self.setUp()
                if kind=='missing': self.bin.unlink()
                elif kind=='truncated': self.bin.write_bytes(b'x')
                elif kind=='no-manifest': self.manifest.unlink()
                elif kind=='no-completion': self.record['validation']={};self.save()
                elif kind=='different-config': self.record['pipeline']={'audio':'opus'};self.save()
                else:
                    self.config['audio']='opus'
                    self.record['source']['media']={'audio':True}
                    self.save()
                with patch('src.archive.streaming.metadata',side_effect=[self.listing,RuntimeError('resolution attempted')]) as fetch:
                    list(stream_jobs(self.args,'yt-dlp',self.counts,config=self.config))
                self.assertEqual(fetch.call_count,2)
                self.assertEqual(self.counts['failed'],1)

    def test_tabs_inherit_channel_and_snapshot_is_reused(self):
        self.listing['entries']=[{'_type':'playlist','entries':[{'id':self.video_id}]}]
        self.args.source_urls *= 2
        from src.archive.streaming import existing_stream_archives
        with patch('src.archive.streaming.metadata',return_value=self.listing) as fetch, patch('src.archive.streaming.existing_stream_archives',wraps=existing_stream_archives) as scan:
            self.assertEqual(list(stream_jobs(self.args,'yt-dlp',self.counts,config=self.config)),[])
        self.assertEqual(scan.call_count,1)
        self.assertEqual(fetch.call_count,2)

    def test_skipped_video_does_not_consume_max_videos(self):
        self.args.max_videos=1
        self.listing['entries'].append({'id':'lmnopqrstuv'})
        with patch('src.archive.streaming.metadata',side_effect=[self.listing,RuntimeError('new video resolved')]) as fetch:
            list(stream_jobs(self.args,'yt-dlp',self.counts,config=self.config))
        self.assertEqual(fetch.call_count,2)
        self.assertEqual(self.counts['resumed'],1)


if __name__ == '__main__': unittest.main()
