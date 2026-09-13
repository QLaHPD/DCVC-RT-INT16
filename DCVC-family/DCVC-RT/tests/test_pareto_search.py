import contextlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.cli import pareto_search as search


class ParetoTests(unittest.TestCase):
    def test_initial_800_cover_every_requested_value_evenly(self):
        domains = {'qp_i': list(range(64)), 'qp_p': list(range(64)),
                   'intra_period': search.valid_intra_periods(), 'reset_interval': list(range(1, 61)),
                   'force_zero_thres': [0, .05, .1, .15, .2]}
        rows = search.uniform_configurations(domains, 800, 20260912)
        self.assertEqual(len({tuple(r.values()) for r in rows}), 800)
        self.assertEqual(rows, search.uniform_configurations(domains, 800, 20260912))
        self.assertEqual(domains['intra_period'], [-1, *range(1, 61)])
        for key, values in domains.items():
            counts = [sum(row[key] == v for row in rows) for v in values]
            self.assertGreater(min(counts), 0)
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_frontier_and_genetic_candidates(self):
        rows = [dict(trial=i+1, status='ok', bitstream_bytes=b, psnr_db=p, config={'q': i})
                for i, (b, p) in enumerate([(10, 20), (20, 30), (30, 25), (20, 29)])]
        self.assertEqual(search.pareto_trials(rows), [1, 2])
        seen = {(i,) for i in range(4)}
        for seed in range(20):
            row = search._adaptive_configuration({'q': list(range(64))}, rows, seen, random.Random(seed))
            self.assertNotIn((row['q'],), seen)
            self.assertIn(row['q'], range(64))

    def test_psnr_component_weighting_and_frame_count(self):
        sink = search.PsnrSink([bytes(12)], 4, 2)
        sink.write(bytes([1]*8 + [2]*2 + [3]*2))
        expected = sum(w * 10 * math.log10(255**2 / e**2)
                       for w, e in [(6, 1), (1, 2), (1, 3)]) / 8
        self.assertAlmostEqual(sink.result()['psnr_db'], expected)
        with self.assertRaises(ValueError):
            sink.write(bytes(12))

    def test_atomic_progress_resume_and_genetic_phase(self):
        class Evaluator:
            count = 0
            interrupt_at = 2
            def __init__(self, *args): pass
            def evaluate(self, config):
                if self.count == self.interrupt_at:
                    raise KeyboardInterrupt
                self.count += 1
                return dict(bitstream_bytes=100 + config['qp_i'], psnr_db=20 + config['qp_p'],
                            encode_seconds=1, decode_seconds=1, frames=1, packets=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('source', 'image', 'video', 'image.int16prep.pt', 'video.int16prep.pt'):
                (root/name).write_bytes(name.encode())
            args = SimpleNamespace(input_file=str(root/'source'), output_json=str(root/'result.json'),
                model_path_i=str(root/'image'), model_path_p=str(root/'video'), resolution=144,
                cuda_idx=0, initial_samples=2, trials=4, seed=123,
                skip_thresholds=[0, .1], resume=True)
            media = dict(width=4, height=2, original_width=4, original_height=2, frames=1)
            with patch('src.codec.frame_decoder.configure_decode_runtime'), \
                 patch.object(search, 'load_source', return_value=(bytes(12), [bytes(12)], media)), \
                 patch.object(search, 'RTTrial', Evaluator), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    search.run(args)
                first = json.loads((root/'result.json').read_text())
                self.assertEqual(first['completed'], 2)
                self.assertEqual(first['status'], 'interrupted')
                Evaluator.interrupt_at = -1
                search.run(args)
                final = json.loads((root/'result.json').read_text())
                self.assertEqual(final['results'][:2], first['results'])
                self.assertEqual(final['completed'], 4)
                self.assertEqual(final['total_encode_seconds'], 4)
                self.assertEqual(final['status'], 'complete')
                self.assertEqual([r['phase'] for r in final['results']], ['uniform']*2 + ['genetic']*2)
                self.assertEqual(len({json.dumps(r['config'], sort_keys=True) for r in final['results']}), 4)
                args.seed += 1
                with self.assertRaisesRegex(ValueError, 'do not match'):
                    search.run(args)


if __name__ == '__main__':
    unittest.main()
