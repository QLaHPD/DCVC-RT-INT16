import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.int16.ops import ARITHMETIC_ID, quantize_parameter
from src.int16.prepared import PreparedFileGuard, content_hash, load_prepared, prepare, resolve_prepared
from src.int16.codec import IntegerCodec, MAGIC
from src.archive.codec import Codec


class IntegerIdentityTests(unittest.TestCase):
    @staticmethod
    def prepared_data():
        data = {'format': 'dcvc-uf-prepared-v1', 'arithmetic': ARITHMETIC_ID,
                'source_sha256': 'a'*64, 'variant': 'image',
                'tensors': {'weight': torch.tensor([-32768, 0, 32767], dtype=torch.int16)}}
        data['identity'] = content_hash(data)
        return data

    def test_identity_hash_reads_complete_buffers_for_all_integer_layouts(self):
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            for values in (torch.empty((0, 2), dtype=dtype), torch.tensor(1, dtype=dtype),
                           torch.arange(48).reshape(6, 8).to(dtype).T):
                with self.subTest(dtype=dtype, shape=tuple(values.shape)):
                    data = self.prepared_data()
                    data['tensors'] = {'weight': values}
                    header = {key: data[key] for key in ('format', 'arithmetic', 'variant', 'source_sha256')}
                    descriptor = ['weight', str(dtype), list(values.shape)]
                    expected = hashlib.sha256()
                    expected.update(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
                    expected.update(json.dumps(descriptor, separators=(',', ':')).encode())
                    array = values.contiguous().numpy()
                    expected.update(array.astype(array.dtype.newbyteorder('<'), copy=False).tobytes())
                    self.assertEqual(content_hash(data), expected.hexdigest())

    def test_mapped_preparation_and_legacy_fallback_keep_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.prepared_data()
            for legacy in (False, True):
                path = Path(directory)/('legacy.pt' if legacy else 'zip.pt')
                torch.save(data, path, _use_new_zipfile_serialization=not legacy)
                guard = PreparedFileGuard()
                original = torch.load
                with patch('src.int16.prepared.torch.load', wraps=original) as load:
                    _, loaded = resolve_prepared(Path(directory)/'absent', 'image', path,
                        data['identity'], 'a'*64, memory_map=True, _guard=guard)
                self.assertEqual(loaded['identity'], data['identity'])
                self.assertTrue(torch.equal(loaded['tensors']['weight'], data['tensors']['weight']))
                self.assertEqual(load.call_args.kwargs.get('mmap', False), not legacy)
                self.assertEqual(len(guard.files), int(not legacy))
                guard.check()
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    load_prepared(path, expected_identity='b'*64, memory_map=True)

    def test_default_loading_has_independent_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            data = self.prepared_data()
            torch.save(data, path)
            original = torch.load
            with patch('src.int16.prepared.torch.load', wraps=original) as load:
                loaded = load_prepared(path)
            self.assertNotIn('mmap', load.call_args.kwargs)
            loaded['tensors']['weight'].add_(1)
            self.assertEqual(load_prepared(path)['identity'], data['identity'])

    def test_mapped_guard_rejects_changes_during_and_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            data = self.prepared_data()
            torch.save(data, path)
            def change_after_hash(value):
                identity = content_hash(value)
                with open(path, 'ab') as stream:
                    stream.write(b'changed')
                return identity
            with patch('src.int16.prepared.content_hash', side_effect=change_after_hash):
                with self.assertRaisesRegex(ValueError, 'changed while loading'):
                    load_prepared(path, memory_map=True)
            torch.save(data, path)
            guard = PreparedFileGuard()
            loaded = load_prepared(path, memory_map=True, _guard=guard)
            guard.check()
            path.rename(path.with_suffix('.old'))
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, 'changed while loading'):
                guard.check()
            self.assertTrue(torch.equal(loaded['tensors']['weight'], data['tensors']['weight']))
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'changed while loading'):
                guard.check()

    def test_prepare_existing_cache_maps_without_constructing_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'source'
            checkpoint.write_bytes(b'existing source')
            from src.int16.prepared import file_hash
            data = self.prepared_data()
            data['source_sha256'] = file_hash(checkpoint)
            data['identity'] = content_hash(data)
            path = Path(directory)/'model.pt'
            torch.save(data, path)
            guard = PreparedFileGuard()
            with patch('src.int16.prepared.make_model', side_effect=AssertionError('unexpected model')):
                actual, loaded = prepare(checkpoint, 'image', path, memory_map=True, _guard=guard)
            self.assertEqual(actual, path)
            self.assertEqual(loaded['identity'], data['identity'])
            self.assertEqual(len(guard.files), 1)

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
