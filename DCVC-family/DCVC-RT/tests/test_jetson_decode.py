import io
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.cli.jetson_decode import DecodeError, JetsonPipe, jetson_source, validate_decode


class JetsonDecodeTests(unittest.TestCase):
    def test_rejects_failed_empty_and_hung_decoder(self):
        for proc, frames in ((Mock(wait=Mock(return_value=1)), 1),
                             (Mock(), 0),
                             (Mock(wait=Mock(side_effect=subprocess.TimeoutExpired('ffmpeg', 30))), 1)):
            with self.subTest(frames=frames), self.assertRaises(DecodeError):
                validate_decode(proc, frames)

    def test_checks_both_children_and_allocation_errors(self):
        for dec_rc, ff_rc, diagnostic in ((1, 0, b''), (0, 1, b''),
                                          (0, 0, b'NvMapMemAllocInternalTagged error 12')):
            pipe = JetsonPipe(Mock(wait=Mock(return_value=dec_rc)),
                              Mock(wait=Mock(return_value=ff_rc)), io.BytesIO(diagnostic))
            with self.assertRaises(DecodeError):
                pipe.wait(timeout=1)

    def test_unsupported_layouts_fall_back(self):
        base = dict(codec_type='video', codec_name='h264', pix_fmt='yuv420p',
                    width=640, height=360, time_base='1/12288', field_order='progressive')
        with tempfile.NamedTemporaryFile() as fixture:
            for change in ({'width': 854}, {'pix_fmt': 'yuv420p10le'},
                           {'codec_name': 'vp9'}, {'field_order': 'tt'},
                           {'color_range': 'pc'}, {'side_data_list': [{'rotation': 90}]}):
                data = dict(streams=[dict(base, **change)], format={'format_name': 'mov,mp4'})
                with patch('src.cli.jetson_decode.shutil.which', return_value='/usr/bin/gst-launch-1.0'), \
                     patch('src.cli.jetson_decode.subprocess.run', return_value=Mock(stdout=json.dumps(data))), \
                     self.subTest(change=change), self.assertRaises(DecodeError):
                    jetson_source(fixture.name)

    def test_matroska_uses_software_to_preserve_final_frame_timing(self):
        data = dict(streams=[dict(codec_type='video', codec_name='h264', pix_fmt='yuv420p',
                                 width=640, height=360, time_base='1/1000')],
                    format={'format_name': 'matroska,webm'})
        with tempfile.NamedTemporaryFile() as fixture, \
             patch('src.cli.jetson_decode.shutil.which', return_value='/usr/bin/gst-launch-1.0'), \
             patch('src.cli.jetson_decode.subprocess.run', return_value=Mock(stdout=json.dumps(data))), \
             self.assertRaises(DecodeError):
            jetson_source(fixture.name)

    def test_truncated_planes_raise(self):
        from src.cli.encode_workflow import NeuralEncoder
        for raw in (b'x', b'x'*16, b'x'*21):
            with self.subTest(length=len(raw)), self.assertRaises(DecodeError):
                NeuralEncoder._read_yuv420_frame(Mock(stdout=io.BytesIO(raw)), 4, 4)
        self.assertIsNone(NeuralEncoder._read_yuv420_frame(Mock(stdout=io.BytesIO()), 4, 4))

    def test_failed_decode_cannot_replace_existing_bitstream(self):
        import torch
        from src.cli import encode_workflow as workflow
        encoder = workflow.NeuralEncoder.__new__(workflow.NeuralEncoder)
        encoder.cfg = workflow.EncoderCfg()
        encoder.device = torch.device('cpu')
        encoder.i_net = Mock()
        encoder.i_net.compress.return_value = {'x_hat': None, 'bit_stream': b'fake-entropy'}
        encoder.p_net = Mock()
        # One complete frame followed by an unsuccessful decoder exit must not
        # commit the otherwise valid frame that was encoded before the failure.
        proc = Mock(stdout=io.BytesIO(bytes(16 * 16 * 3 // 2)), wait=Mock(return_value=1))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'existing.bin'
            output.write_bytes(b'previous-completed-output')
            with self.assertRaises(DecodeError):
                encoder.encode_from_ffmpeg_rawpipe(proc, 16, 16, output, Path(directory),
                                                   'synthetic', finalize_mode='atomic')
            self.assertEqual(output.read_bytes(), b'previous-completed-output')

    def test_hardware_failure_retries_entire_file_in_software(self):
        from src.cli import encode_workflow as workflow
        hardware = Mock()
        software = Mock(stdout=io.BytesIO(b''))
        encoder = Mock()
        encoder.encode_from_ffmpeg_rawpipe.side_effect = [DecodeError('midstream failure'), None]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(workflow, 'probe_stream_props', return_value=(640, 360, 24)), \
             patch.object(workflow, 'start_jetson_pipe', return_value=hardware), \
             patch.object(workflow, 'popen_command', return_value=software):
            ok, _ = workflow.process_one_file(
                workflow.EncodeTask('synthetic', str(Path(directory)/'input.mp4'), True, False),
                queue.Queue(), 0, Path(directory)/'output', encoder,
                workflow.EncoderCfg(ff_hwaccel='jetson'), False, {}, 'atomic')
        self.assertTrue(ok)
        self.assertEqual(encoder.encode_from_ffmpeg_rawpipe.call_count, 2)
        hardware.close.assert_called_once()
        self.assertIs(encoder.encode_from_ffmpeg_rawpipe.call_args_list[1].args[0], software)


if __name__ == '__main__':
    unittest.main()
