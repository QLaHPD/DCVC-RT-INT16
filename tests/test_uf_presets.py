import copy
import unittest
from unittest.mock import patch

from main import build_parser, main
from src.archive.presets import apply_target_psnr, select_trial


class PresetTests(unittest.TestCase):
    def parse(self, *options):
        return build_parser().parse_args(['encode', '--input_file', 'sample.mp4', '--output_root', 'out', *options])

    def test_manual_defaults_unchanged(self):
        args = self.parse()
        before = copy.deepcopy(vars(args))
        apply_target_psnr(args)
        self.assertEqual(vars(args), before)
        self.assertEqual((args.runtime, args.model_structure, args.qp_i, args.qp_p), ('fp16', 'hts', 36, 30))

    def test_published_target_30_matches_known_fastest_trial(self):
        args = self.parse('--target-psnr', '30')
        apply_target_psnr(args)
        self.assertEqual((args.runtime, args.model_structure, args.resolution), ('int16', 'htl', 144))
        self.assertEqual((args.qp_i, args.qp_p, args.reset_interval, args.intra_period, args.skip_thres),
                         (10, 14, 2, -1, .2))
        self.assertEqual(args.target_psnr_selection['trial'], 783)

    def test_explicit_aliases_constrain_selection_even_if_equal_to_default(self):
        args = self.parse('--target-psnr=30', '--qi=36', '--qp', '30')
        apply_target_psnr(args)
        self.assertEqual((args.qp_i, args.qp_p), (36, 30))
        self.assertGreaterEqual(args.target_psnr_selection['measured_psnr'], 30)

    def test_incompatible_benchmark_settings_fail(self):
        for option, value in [('runtime', 'fp16'), ('model_structure', 'hts'), ('resolution', '96')]:
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'requires'):
                apply_target_psnr(self.parse('--target-psnr', '30', '--'+option, value))

    def test_impossible_target_and_unmeasured_constraints_fail(self):
        for options in [('--target-psnr', '1000'), ('--target-psnr', '30', '--skip_thres', '0.123')]:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'No measured configuration'):
                apply_target_psnr(self.parse(*options))

    def test_time_then_bytes_then_trial_and_quality_threshold(self):
        rows = [dict(trial=i, status='ok', config={}, psnr_db=psnr, encode_seconds=seconds,
                     bitstream_bytes=size) for i, psnr, seconds, size in
                [(1, 29, 1, 1), (2, 30, 2, 20), (3, 31, 2, 10), (4, 35, 3, 1), (5, 31, 2, 10)]]
        rows += [dict(status='failed')]
        self.assertEqual(select_trial(rows, 30)['trial'], 3)
        for target in [float('nan'), float('inf'), -1]:
            with self.assertRaises(ValueError): select_trial(rows, target)

    def test_stream_alias_and_other_options_preserved(self):
        args = build_parser().parse_args(['stream', '--source_urls', 'https://example.com/video',
                                         '--output_root', 'out', '--target-psnr', '30',
                                         '--procs', '3', '--fps', '24', '--opus_bitrate', '6k'])
        apply_target_psnr(args)
        self.assertEqual((args.procs, args.fps, args.opus_bitrate), (3, 24, '6k'))

    def test_cli_applies_selection_before_runtime_defaults_and_dispatch(self):
        argv = ['main.py', 'stream', '--source_urls', 'https://example.com/video',
                '--output_root', 'out', '--target-psnr', '30']
        with patch('sys.argv', argv), patch('src.archive.dashboard.run', return_value=0) as run:
            self.assertEqual(main(), 0)
        args = run.call_args.args[0]
        self.assertEqual(args.target_psnr_selection['trial'], 783)
        self.assertEqual(args.prefetch_frames, 8)
        self.assertEqual(args.runtime, 'int16')


if __name__ == '__main__':
    unittest.main()
