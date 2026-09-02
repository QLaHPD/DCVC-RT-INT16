from __future__ import annotations

import queue
import unittest
from unittest.mock import patch

from src.cli.device_plan import build_worker_device_plan
from src.cli.decode_workflow import worker_entry as decode_worker_entry
from src.cli.encode_workflow import worker_entry


class WorkerDevicePlanTests(unittest.TestCase):
    def test_five_visible_gpus_default_to_five_workers(self):
        plan = build_worker_device_plan(
            use_cuda=True,
            requested_workers=None,
            requested_cuda_indices=None,
            cuda_available=True,
            cuda_device_count=5,
        )
        self.assertTrue(plan.using_cuda)
        self.assertEqual(plan.worker_count, 5)
        self.assertEqual(plan.cuda_indices, (0, 1, 2, 3, 4))

    def test_selected_gpus_default_to_one_worker_each(self):
        plan = build_worker_device_plan(
            use_cuda=True,
            requested_workers=None,
            requested_cuda_indices=[1, 3],
            cuda_available=True,
            cuda_device_count=5,
        )
        self.assertEqual(plan.cuda_indices, (1, 3))

    def test_explicit_extra_workers_are_assigned_round_robin(self):
        plan = build_worker_device_plan(
            use_cuda=True,
            requested_workers=5,
            requested_cuda_indices=[1, 3],
            cuda_available=True,
            cuda_device_count=5,
        )
        self.assertEqual(plan.cuda_indices, (1, 3, 1, 3, 1))

    def test_invalid_gpu_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "visible logical indices are 0..4"):
            build_worker_device_plan(
                use_cuda=True,
                requested_workers=None,
                requested_cuda_indices=[5],
                cuda_available=True,
                cuda_device_count=5,
            )

    def test_cpu_mode_defaults_to_one_worker(self):
        plan = build_worker_device_plan(
            use_cuda=False,
            requested_workers=None,
            requested_cuda_indices=None,
            cuda_available=True,
            cuda_device_count=5,
        )
        self.assertFalse(plan.using_cuda)
        self.assertEqual(plan.cuda_indices, (None,))

    def test_worker_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_worker_device_plan(
                use_cuda=True,
                requested_workers=0,
                requested_cuda_indices=None,
                cuda_available=True,
                cuda_device_count=5,
            )

    def test_encoder_worker_selects_assigned_device_before_model_load(self):
        task_queue = queue.Queue()
        progress_queue = queue.Queue()
        task_queue.put(None)

        with patch("src.cli.encode_workflow.signal.signal"), \
                patch("src.cli.encode_workflow.set_torch_env"), \
                patch("src.cli.encode_workflow.release_cuda"), \
                patch("src.cli.encode_workflow.torch.cuda.is_available", return_value=True), \
                patch("src.cli.encode_workflow.torch.cuda.set_device") as set_device, \
                patch("src.cli.encode_workflow.NeuralEncoder") as encoder:
            worker_entry(
                wid=2,
                task_q=task_queue,
                progress_q=progress_queue,
                stop_event=type("StopEvent", (), {"is_set": lambda self: False})(),
                output_root="/unused",
                model_i="image.pth.tar",
                model_p="video.pth.tar",
                enc_cfg_dict={},
                audio_mode="none",
                opus_params={},
                use_cuda=True,
                cuda_device_index=3,
                finalize_mode="atomic",
            )

        set_device.assert_called_once_with(3)
        self.assertEqual(str(encoder.call_args.args[0]), "cuda:3")
        self.assertEqual(progress_queue.get_nowait()["device"], "cuda:3")

    def test_decoder_worker_loads_models_on_assigned_device(self):
        task_queue = queue.Queue()
        progress_queue = queue.Queue()
        task_queue.put(None)

        with patch("src.cli.decode_workflow.signal.signal"), \
                patch(
                    "src.cli.decode_workflow.resolve_decode_device",
                    return_value="cuda:4",
                ) as resolve_device, \
                patch("src.cli.decode_workflow.load_decoder_models") as load_models:
            decode_worker_entry(
                wid=4,
                task_q=task_queue,
                progress_q=progress_queue,
                stop_event=type("StopEvent", (), {"is_set": lambda self: False})(),
                args_dict={
                    "model_path_i": "image.pth.tar",
                    "model_path_p": "video.pth.tar",
                    "force_zero_thres": None,
                },
                use_cuda=True,
                cuda_device_index=4,
            )

        resolve_device.assert_called_once_with(True, 4)
        self.assertEqual(load_models.call_args.args[2], "cuda:4")
        self.assertEqual(progress_queue.get_nowait()["device"], "cuda:4")


if __name__ == "__main__":
    unittest.main()
