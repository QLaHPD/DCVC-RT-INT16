import io
from pathlib import Path
import tempfile
import unittest

import torch

from src.int16.ops import ARITHMETIC_ID, quantize_parameter
from src.int16.prepared import content_hash, load_prepared, resolve_prepared
from src.int16.codec import IntegerCodec, MAGIC
from src.archive.codec import Codec


class IntegerIdentityTests(unittest.TestCase):
    def test_prepared_identity_rejects_mutation_and_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'model.pt'
            data = {'format':'dcvc-uf-prepared-v1','arithmetic':ARITHMETIC_ID,
                    'source_sha256':'a'*64,'variant':'image',
                    'tensors':{'weight':torch.tensor([-32768,0,32767],dtype=torch.int16)}}
            data['identity'] = content_hash(data)
            torch.save(data,path)
            self.assertEqual(load_prepared(path,'a'*64,'image')['identity'],data['identity'])
            # Explicit prepared files do not require the original float checkpoint.
            _, standalone = resolve_prepared(Path(root)/'absent.pth', 'image', path,
                                               data['identity'], 'a'*64)
            self.assertEqual(standalone['identity'], data['identity'])
            with self.assertRaisesRegex(ValueError,'source checkpoint'):
                load_prepared(path,'b'*64,'image')
            with self.assertRaisesRegex(ValueError,'variant'):
                load_prepared(path,'a'*64,'hts')
            data['tensors']['weight'][1] = 1
            torch.save(data,path)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                load_prepared(path)

    def test_runtime_and_prepared_id_are_required_by_bitstream(self):
        integer = IntegerCodec.__new__(IntegerCodec)
        integer.preamble = MAGIC + bytes(range(64))
        stream = io.BufferedReader(io.BytesIO(integer.preamble+b'payload'))
        integer.read_preamble(stream)
        self.assertEqual(stream.read(),b'payload')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            integer.read_preamble(io.BytesIO(MAGIC+bytes(64)))
        with self.assertRaisesRegex(ValueError,'FP16 runtime'):
            Codec.__new__(Codec).read_preamble(io.BufferedReader(io.BytesIO(integer.preamble)))

    def test_non_finite_weights_are_rejected(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaisesRegex(ValueError,'non-finite'):
                quantize_parameter(torch.tensor([value]))
