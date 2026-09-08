from __future__ import annotations

import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.cli import encode_workflow as workflow


class ThreadProcess:
    def __init__(self, target, args, daemon=True):
        del daemon
        self._target = target
        self._args = args
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._target, args=self._args, daemon=True)
        self._thread.start()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def terminate(self):
        pass

    def kill(self):
        pass


class ThreadContext:
    @staticmethod
    def Queue():
        return queue.Queue()

    @staticmethod
    def Event():
        return threading.Event()

    Process = ThreadProcess


class QuietDashboard:
    def __init__(self, total, worker_count, storage_path, mode="plain"):
        del worker_count, storage_path, mode
        self.total = total
        self.done = 0

    def set_done(self, done):
        self.done = done

    def increment_done(self, step=1, elapsed_seconds=None):
        del elapsed_seconds
        self.done += step

    def set_worker_text(self, wid, text):
        del wid, text

    def clear_worker(self, wid):
        del wid

    def write(self, text):
        del text

    def close(self):
        pass


class FakeEncoder:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class SharedSchedulerTests(unittest.TestCase):
    def test_two_multiworker_instances_encode_each_video_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_channel = root / "input" / "CHANNEL"
            output_root = root / "output"
            output_channel = output_root / "CHANNEL"
            input_channel.mkdir(parents=True)
            output_channel.mkdir(parents=True)
            model_i = root / "image.model"
            model_p = root / "video.model"
            model_i.write_bytes(b"same-image-model")
            model_p.write_bytes(b"same-video-model")
            tasks = []
            for index in range(4):
                source = input_channel / f"video_{index}.mkv"
                source.write_bytes(b"fixture")
                tasks.append(workflow.EncodeTask("CHANNEL", str(source), True, False))
            discovery = workflow.ChannelEncodeDiscovery(
                channel_id="CHANNEL",
                pending_tasks=tasks,
                source_paths=[task.video_path for task in tasks],
                total_found=len(tasks),
                already_done=0,
            )
            encoded = []
            encoded_lock = threading.Lock()

            def fake_process(task, progress_q, wid, channel_out, encoder, enc_cfg,
                             audio_enable, opus_params, finalize_mode, defer_publish=False):
                del encoder, enc_cfg, audio_enable, opus_params, finalize_mode
                self.assertTrue(defer_publish)
                base = Path(task.video_path).stem
                progress_q.put({
                    "type": "worker_task_start", "wid": wid, "vid": base,
                    "channel_id": task.channel_id,
                })
                staged = workflow.shared_stage_path(channel_out, base, task.publish_token, ".bin")
                staged.write_bytes(f"encoded-{base}".encode())
                time.sleep(0.03)
                with encoded_lock:
                    encoded.append(base)
                final = channel_out / f"{base}_176x96_qI35_qP14.bin"
                progress_q.put({
                    "type": "worker_done", "wid": wid, "vid": base,
                    "channel_id": task.channel_id, "elapsed": 0.03,
                    "publish": [{"temporary": str(staged), "final": str(final)}],
                })
                return True, 0.03

            def make_args(name):
                return SimpleNamespace(
                    auto_delete=False,
                    cleanup_dry_run=False,
                    shared_lease_seconds=30.0,
                    shared_instance=name,
                    model_path_i=str(model_i),
                    model_path_p=str(model_p),
                    audio="none",
                    disk_finalize="atomic",
                    ui="plain",
                )

            device_plan = SimpleNamespace(
                worker_count=3,
                cuda_indices=(None, None, None),
                using_cuda=False,
            )
            enc_cfg = {
                "qp_i": 35, "qp_p": 14, "force_intra_period": -1,
                "reset_interval": 32, "resolution": 96, "pad_multiple": 16,
                "ff_color_matrix": "bt709", "ff_hwaccel": "none",
                "ffmpeg_prefetch": 8, "fps": 24, "force_zero_thres": None,
            }
            results = []

            def run_instance(name):
                results.append(workflow._run_shared_encode(
                    make_args(name), device_plan, output_root, [discovery], list(tasks),
                    len(tasks), 0, enc_cfg, {},
                ))

            with mock.patch.object(workflow, "get_context", return_value=ThreadContext()), \
                    mock.patch.object(workflow, "NeuralEncoder", FakeEncoder), \
                    mock.patch.object(workflow, "process_one_file", side_effect=fake_process), \
                    mock.patch.object(workflow, "EncodeDashboard", QuietDashboard), \
                    mock.patch.object(workflow.signal, "signal"), \
                    mock.patch.object(workflow, "set_torch_env"), \
                    mock.patch.object(workflow, "release_cuda"):
                first = threading.Thread(target=run_instance, args=("machine-a",))
                second = threading.Thread(target=run_instance, args=("machine-b",))
                first.start()
                second.start()
                first.join(timeout=10)
                second.join(timeout=10)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(results, [0, 0])
            self.assertEqual(sorted(encoded), [f"video_{index}" for index in range(4)])
            for index in range(4):
                output = output_channel / f"video_{index}_176x96_qI35_qP14.bin"
                self.assertEqual(output.read_bytes(), f"encoded-video_{index}".encode())


if __name__ == "__main__":
    unittest.main()
