#!/usr/bin/env python3
"""Check graph replay against eager INT16 features across inputs, QPs and shapes."""
import os
from pathlib import Path
import sys

os.environ["DCVC_USE_INT16"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.layers.int16_inference import get_prepared_feature_param_slice
from src.models.video_model import DMC
from src.utils.common import load_model_for_inference, set_torch_env


def main():
    set_torch_env()
    device = torch.device("cuda:0")
    model = load_model_for_inference(
        DMC(), str(ROOT / "checkpoints/cvpr2025_video.pth.tar"), device
    )
    runner = model._feature_graph
    default = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    count = 0
    with torch.no_grad():
        for stream in (default, side, default):
            # Deliberately reuse one model/cache across resolution and stream changes.
            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                for height, width in ((12, 22), (24, 40), (10, 16), (12, 22)):
                    previous_graph = None
                    for qp, value in ((0, 0), (14, -32768), (22, 32767), (63, None), (71, None)):
                        feature = torch.randint(-32768, 32768, (1, 256, height, width),
                                                dtype=torch.int16, device=device)
                        if value is not None:
                            feature.fill_(value)
                        quant = get_prepared_feature_param_slice(
                            model, "q_feature", model.q_feature, qp, qp + 1, device
                        )
                        expected = model.feature_extractor(feature, quant)
                        actual = runner(model.feature_extractor, feature, quant)
                        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
                        if previous_graph is not None:
                            assert runner.graph is previous_graph, "QP/input change recaptured"
                        previous_graph = runner.graph
                        count += 1
        # Simulate a persistent worker opening several videos, with resets and
        # changing frame QPs. The DPB must never retain graph-owned output buffers.
        for height, width in ((96, 176), (180, 320), (72, 128), (96, 176)):
            padded_h, padded_w = (height + 15) // 16 * 16, (width + 15) // 16 * 16
            frames = [torch.rand((1, 3, padded_h, padded_w), device=device).half()
                      for _ in range(6)]
            results = []
            for enabled in (False, True):
                model._use_feature_graph = enabled
                model.clear_dpb()
                model.set_curr_poc(0)
                model.add_ref_frame(None, torch.zeros_like(frames[0]))
                sequence = []
                last_qp = 0
                for index, (frame, qp) in enumerate(zip(frames, (0, 14, 22, 63, 71, 0))):
                    if index == 3:
                        model.prepare_feature_adaptor_i(last_qp)
                    encoded = model.compress(frame, qp)
                    sequence.append((encoded['bit_stream'], model.dpb[0].feature.cpu().clone()))
                    last_qp = qp
                results.append(sequence)
            for eager, graphed in zip(*results):
                assert eager[0] == graphed[0]
                assert torch.equal(eager[1], graphed[1])
        # Moving/repreparing/reloading a model must not retain graphs with old weights.
        model.to(device)
        assert runner.graph is None
        feature = torch.zeros((1, 256, 12, 22), dtype=torch.int16, device=device)
        runner(model.feature_extractor, feature, quant)
        model.load_state_dict(model.state_dict())
        assert runner.graph is None
        runner(model.feature_extractor, feature, quant)
        model.prepare_int16_inference()
        assert runner.graph is None
    print(f"PASS: {count} exact feature cases, 24 persistent-worker frames, "
          "QP reuse, shape/stream changes, model invalidation")


if __name__ == "__main__":
    main()
