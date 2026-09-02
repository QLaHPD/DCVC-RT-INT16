#!/usr/bin/env python3
"""Microbenchmark and bit-exact regression checks for the INT16 1x1 CUDA kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CASES = (
    ("v_256_1024_12x22", 256, 1024, 12, 22),
    ("v_256_256_12x22", 256, 256, 12, 22),
    ("v_512_256_12x22", 512, 256, 12, 22),
    ("v_1024_256_12x22", 1024, 256, 12, 22),
    ("v_384_1536_6x11", 384, 1536, 6, 11),
    ("v_384_384_6x11", 384, 384, 6, 11),
    ("v_768_384_6x11", 768, 384, 6, 11),
    ("v_1536_384_6x11", 1536, 384, 6, 11),
    ("i_368_1472_12x22", 368, 1472, 12, 22),
    ("i_368_368_12x22", 368, 368, 12, 22),
    ("i_736_368_12x22", 736, 368, 12, 22),
    ("i_1472_368_12x22", 1472, 368, 12, 22),
    ("i_512_2048_6x11", 512, 2048, 6, 11),
    ("i_512_512_6x11", 512, 512, 6, 11),
    ("i_1024_512_6x11", 1024, 512, 6, 11),
    ("p_128_1024_3x3_6x11", 128, 1024, 6, 11, 3, 1, 1),
)


def unpack_case(case):
    name, in_channels, out_channels, height, width, *convolution = case
    kernel_size, stride, padding = convolution or (1, 1, 0)
    return name, in_channels, out_channels, height, width, kernel_size, stride, padding


def make_case(index: int, in_channels: int, out_channels: int, height: int, width: int,
              kernel_size: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xDC0C0000 + index)
    x = torch.randint(
        -4096, 4097, (1, in_channels, height, width), dtype=torch.int16, generator=generator
    ).cuda()
    weight = torch.randint(
        -22000, 22001, (out_channels, in_channels, kernel_size, kernel_size), dtype=torch.int16,
        generator=generator,
    ).cuda()
    bias = torch.randint(
        -32768, 32768, (out_channels,), dtype=torch.int16, generator=generator
    ).cuda()
    return x, weight, bias


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def load_extension(explicit_path: Path | None):
    if explicit_path is None:
        from src.layers.int16_inference import _INT16_EXT
        return _INT16_EXT

    path = explicit_path.resolve()
    spec = importlib.util.spec_from_file_location("inference_extensions_cuda", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference_extensions_cuda"] = module
    spec.loader.exec_module(module)
    return module


def run_case(extension, index: int, case, warmup: int, iterations: int):
    name, in_channels, out_channels, height, width, kernel_size, stride, padding = unpack_case(case)
    x, weight, bias = make_case(index, in_channels, out_channels, height, width, kernel_size)

    for _ in range(warmup):
        output = extension.conv2d_int16_cuda(
            x, weight, bias, stride, stride, padding, padding, 1
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = extension.conv2d_int16_cuda(
            x, weight, bias, stride, stride, padding, padding, 1
        )
    end.record()
    end.synchronize()

    milliseconds = start.elapsed_time(end) / iterations
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    macs = out_height * out_width * in_channels * out_channels * kernel_size * kernel_size
    gmac_per_second = macs / (milliseconds * 1.0e6)
    return output, milliseconds, gmac_per_second


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--save-reference", type=Path)
    parser.add_argument("--compare-reference", type=Path)
    args = parser.parse_args()

    if args.save_reference and args.compare_reference:
        parser.error("choose either --save-reference or --compare-reference")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")

    extension = load_extension(args.extension)
    print(f"extension={extension.__file__}")
    references = None
    if args.compare_reference:
        references = torch.load(args.compare_reference, map_location="cpu", weights_only=True)

    captured = {}
    mismatches = 0
    for index, case in enumerate(CASES):
        name = case[0]
        output, milliseconds, throughput = run_case(
            extension, index, case, args.warmup, args.iterations
        )
        output_cpu = output.cpu()
        digest = tensor_digest(output_cpu)
        status = ""
        if references is not None:
            expected = references[name]
            equal = torch.equal(output_cpu, expected)
            status = " exact=yes" if equal else " exact=NO"
            mismatches += int(not equal)
        captured[name] = output_cpu
        print(
            f"{name}: {milliseconds:.4f} ms, {throughput:.2f} GMAC/s, "
            f"sha256={digest}{status}"
        )

    if args.save_reference:
        args.save_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(captured, args.save_reference)
        print(f"saved_reference={args.save_reference}")

    if mismatches:
        print(f"mismatches={mismatches}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
