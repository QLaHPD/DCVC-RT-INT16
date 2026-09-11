"""Bounded CUDA Graph replay for a pure inference stage with changing inputs."""

import torch


class InferenceGraph:
    """Keep one shape/stream graph; returned tensors live until the next call.

    The caller must consume outputs on the calling stream before replaying, and
    finish any other-stream consumers before the next call or clear. Clear this
    cache when replacing model weights. No CPU work or model state updates may
    be performed by the captured function.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        # Outputs are allocated on the capture stream but consumed on the replay
        # stream. Finish that work before releasing a private pool on invalidation.
        # This barrier is only for a shape/stream/model change, never steady replay.
        if getattr(self, "graph", None) is not None:
            self.stream.synchronize()
        self.key = None
        self.graph = None
        self.stream = None
        self.inputs = None
        self.outputs = None

    def __call__(self, function, *inputs):
        device = inputs[0].device
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            key = (stream.cuda_stream, tuple(
                (x.device, x.dtype, tuple(x.shape)) for x in inputs
            ))
            if self.key != key:
                # Bound retained graph memory across videos with different sizes.
                self.clear()
                self.inputs = tuple(x.clone().contiguous() for x in inputs)
                capture_stream = torch.cuda.Stream(device=device)
                capture_stream.wait_stream(stream)
                with torch.cuda.stream(capture_stream):
                    # Populate lazy kernel/LUT caches and autotuning before capture.
                    for _ in range(3):
                        function(*self.inputs)
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    self.outputs = function(*self.inputs)
                self.graph = graph
                self.stream = stream
                self.key = key
            else:
                for static, current in zip(self.inputs, inputs):
                    static.copy_(current)
            self.graph.replay()
            return self.outputs
