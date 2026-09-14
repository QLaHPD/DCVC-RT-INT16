import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from src.archive.frame_source import FrameSource, packet_index
from src.archive.thumbnails import ImageReader, encode_images, read_image, decode_image, cleanup_image
from src.utils.stream_helper import write_sps, write_ip


class FakeNetwork:
    def __init__(self): self.calls=[]
    def clear_dpb(self): pass
    def add_ref_feature_from_frame(self,*args,**kwargs): pass
    def decompress(self,bits,sps,qp,ec,*args):
        self.calls.append(tuple(bits))
        values = list(bits)
        return {'x_hat': values[0] if len(values)==1 else values}


class FakeCodec:
    def __init__(self):
        self.image,self.video=FakeNetwork(),FakeNetwork()
        self.device=SimpleNamespace(type='cpu')
    def raw_frame(self,value,w,h):return bytes([value])*(w*h*3//2)
    def encode(self,reader,path,*args,**kwargs):
        reader.read()
        with open(path,'wb') as out:
            write_sps(out,dict(sps_id=0,width=reader.width,height=reader.height))
            write_ip(out,True,0,45,0,0,bytes([128]))
        return dict(frames=1,packets=1,encode_seconds=1)
    def decode(self,path,metadata):
        raw=self.raw_frame(128,metadata['width'],metadata['height'])
        return dict(decoded_yuv_sha256=hashlib.sha256(raw).hexdigest(),decoded_frames=1)


class ManagedMediaTests(unittest.TestCase):
    def setUp(self):
        environment=patch.dict('os.environ');environment.start();self.addCleanup(environment.stop)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.args=SimpleNamespace(runtime='auto',device='cpu',cuda_idx=0,model_path_i=None,model_path_p=None,
                                  prepared_i=None,prepared_p=None,base_root=None,yes=True)
        self.config=dict(runtime='fp16',variant='htl',qp_i=45,qp_p=45)

    def test_seek_handles_ht_chunks_partial_tail_and_backward_iframe(self):
        path=self.root/'movie.bin'
        with path.open('wb') as out:
            write_sps(out,dict(sps_id=0,width=2,height=2))
            for iframe,values in [(True,[10]),(False,range(11,19)),(True,[19]),(False,range(20,28))]:
                write_ip(out,iframe,0,10,0,0,bytes(values))
        data=dict(pipeline=self.config,video=dict(width=2,height=2,frames=13,packets=4))
        fake=FakeCodec()
        with patch('src.archive.storage.load_bundle',return_value=(self.root,data)),patch('src.archive.workflow.checked_codec',return_value=fake):
            # Legacy logical video path.
            path.rename(self.root/'video.bin')
            source=FrameSource(self.root/'video.bin',self.args)
            self.addCleanup(source.close)
            for index in [12,10,9,0,1,8,7,12]:
                self.assertEqual(source.seek(index).raw,bytes([10+index])*6)
            self.assertEqual(fake.image.calls[0],(19,)) # Jump starts at nearest I, skips first GOP.
            with self.assertRaises(IndexError):source.seek(13)

    def test_incomplete_index_and_extra_packet_rejected(self):
        for frames,packets in [(2,1),(1,2)]:
            out=io.BytesIO();write_sps(out,dict(sps_id=0,width=2,height=2));write_ip(out,True,0,0,0,0,b'x')
            with self.assertRaises(ValueError):
                packet_index(io.BufferedReader(io.BytesIO(out.getvalue())),dict(width=2,height=2,frames=frames,packets=packets),8)

    def test_thumbnail_resume_integrity_decode_and_exact_cleanup(self):
        from PIL import Image
        original=self.root/'thumb.jpg';Image.new('RGB',(7,5),'red').save(original)
        other=self.root/'keep.txt';other.write_text('keep')
        target=self.root/'out'/'thumb.jpg_qI45.dcvci'
        fake=FakeCodec()
        with patch('src.archive.workflow.make_codec',return_value=fake) as factory, patch.object(fake,'decode',side_effect=AssertionError('unexpected encode validation')):
            self.assertFalse(encode_images([(original,target)],self.config,(), 'cpu'))
            self.assertFalse(encode_images([(original,target)],self.config,(), 'cpu'))
            self.assertEqual(factory.call_count,1)
        metadata,_=read_image(target)
        self.assertEqual(metadata['validation'],{'mode':'artifact-hashes','decode_performed':False})
        self.assertEqual((metadata['image_width'],metadata['image_height']),(7,5))
        self.args.input=str(target);self.args.output_file=str(self.root/'decoded.png')
        with patch('src.archive.workflow.checked_codec',return_value=fake):
            decode_image(self.args)
            with Image.open(self.args.output_file) as decoded:
                self.assertEqual(decoded.size,(7,5))
            cleanup_image(self.args)
        self.assertFalse(original.exists());self.assertTrue(other.exists());self.assertTrue(target.exists())
        with target.open('ab') as stream:stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError,'modified'):read_image(target)

    def test_thumbnail_model_reused_and_changed_original_retained(self):
        from PIL import Image
        jobs=[]
        for name in ('a.png','b.png'):
            source=self.root/name;Image.new('RGB',(4,4)).save(source)
            jobs.append((source,self.root/(name+'_qI45.dcvci')))
        with patch('src.archive.workflow.make_codec',return_value=FakeCodec()) as factory:
            self.assertFalse(encode_images(jobs,self.config,(),'cpu'))
            self.assertEqual(factory.call_count,1)
            jobs[0][0].write_bytes(b'changed')
            self.assertTrue(encode_images(jobs,self.config,(),'cpu'))
        self.args.input=str(jobs[0][1])
        with self.assertRaisesRegex(ValueError,'changed'):cleanup_image(self.args)
        self.assertTrue(jobs[0][0].exists())

    def test_cli_rt_view_and_image_flags(self):
        from main import build_parser
        parser=build_parser()
        args=parser.parse_args(['view','movie.bin','--runtime','int16','--start_frame','9'])
        self.assertEqual(args.bin_path,'movie.bin');self.assertEqual(args.start_frame,9)
        args=parser.parse_args(['encode','--base_root','input','--output_root','output','--cuda','true','--thumbnail_codec','dcvc-intra'])
        self.assertEqual(args.thumbnail_qp,45)
        args=parser.parse_args(['decode','--input_folder','input','--output_folder','output','--input_kind','images'])
        self.assertEqual(args.input_kind,'images')


if __name__=='__main__':unittest.main()
