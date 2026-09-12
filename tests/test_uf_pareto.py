import math
import random
import unittest

import numpy as np

from src.archive.pareto import (PsnrSink, _adaptive_configuration, pareto_trials, uniform_configurations,
                                valid_intra_periods)


class ParetoTests(unittest.TestCase):
    def test_valid_intra_periods_follow_temporal_chunk(self):
        self.assertEqual(valid_intra_periods('htl'), [-1, 1, 8, 16, 24, 32, 40, 48, 56])
        self.assertEqual(valid_intra_periods('hts'), valid_intra_periods('htl'))
        self.assertEqual(valid_intra_periods('ld')[:4], [-1, 1, 2, 3])
        self.assertEqual(valid_intra_periods('ld')[-1], 60)

    def test_uniform_samples_are_unique_balanced_and_reproducible(self):
        domains = {'a': list(range(4)), 'b': list(range(5)), 'c': [-1, 1, 8]}
        first = uniform_configurations(domains, 20, 7)
        self.assertEqual(first, uniform_configurations(domains, 20, 7))
        self.assertEqual(len({tuple(item.values()) for item in first}), 20)
        for name, values in domains.items():
            counts = [sum(item[name] == value for item in first) for value in values]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_pareto_frontier_minimizes_bytes_and_maximizes_psnr(self):
        results = [
            {'trial': 1, 'status': 'ok', 'bitstream_bytes': 10, 'psnr_db': 20},
            {'trial': 2, 'status': 'ok', 'bitstream_bytes': 20, 'psnr_db': 30},
            {'trial': 3, 'status': 'ok', 'bitstream_bytes': 15, 'psnr_db': 19},
            {'trial': 4, 'status': 'ok', 'bitstream_bytes': 20, 'psnr_db': 29},
            {'trial': 5, 'status': 'failed'},
            {'trial': 6, 'status': 'ok', 'bitstream_bytes': 30, 'psnr_db': 31},
        ]
        self.assertEqual(pareto_trials(results), [1, 2, 6])

    def test_genetic_candidate_stays_in_domain_and_is_unique(self):
        domains = {'qp_i': list(range(8)), 'qp_p': list(range(8)),
                   'intra_period': [-1, 1, 8], 'reset_interval': list(range(1, 9)),
                   'skip_threshold': [0.0, 0.1]}
        config = {name: values[0] for name, values in domains.items()}
        results = [{'trial': 1, 'status': 'ok', 'bitstream_bytes': 10,
                    'psnr_db': 20, 'config': config}]
        seen = {tuple(config[name] for name in domains)}
        candidate = _adaptive_configuration(domains, results, seen, random.Random(9))
        self.assertNotIn(tuple(candidate[name] for name in domains), seen)
        for name, value in candidate.items():
            self.assertIn(value, domains[name])

    def test_psnr_sink_uses_frame_average_yuv_611(self):
        width, height = 4, 2
        reference = bytes([0] * 12)
        decoded = bytes([1] * 8 + [2] * 2 + [3] * 2)
        sink = PsnrSink([reference], width, height)
        sink.write(decoded)
        result = sink.result()
        y = 10 * math.log10(255 * 255)
        u = 10 * math.log10(255 * 255 / 4)
        v = 10 * math.log10(255 * 255 / 9)
        self.assertAlmostEqual(result['psnr_y_db'], y)
        self.assertAlmostEqual(result['psnr_u_db'], u)
        self.assertAlmostEqual(result['psnr_v_db'], v)
        self.assertAlmostEqual(result['psnr_db'], (6*y + u + v) / 8)
        self.assertTrue(np.isfinite(list(result.values())).all())


if __name__ == '__main__':
    unittest.main()
