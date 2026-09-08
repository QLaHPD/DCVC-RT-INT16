"""Check exact residual fusion and measure MMA tile choices on the current GPU."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from benchmarks.benchmark_int16_conv import CASES, load_extension, make_case, unpack_case


def timed(fn, iterations):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    ext = load_extension(args.extension)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    generator = torch.Generator().manual_seed(20260907)
    stream = torch.cuda.Stream()
    cases_checked = 0
    with torch.cuda.stream(stream):
        # Full range, multiple batches, odd dimensions and noncontiguous inputs.
        for b, c, m, h, w in [(2, 3, 5, 3, 7), (1, 33, 65, 5, 13), (1, 256, 256, 12, 22)]:
            x = torch.randint(-32768, 32768, (b, c, h, w*2), generator=generator, dtype=torch.int16).cuda()[..., ::2]
            weight = torch.randint(-32768, 32768, (m, c, 1, 1), generator=generator, dtype=torch.int16).cuda()
            residual = torch.randint(-32768, 32768, (b, m, h, w*2), generator=generator, dtype=torch.int16).cuda()[..., ::2]
            for bias in [None, torch.randint(-32768, 32768, (m,), generator=generator, dtype=torch.int16).cuda()]:
                expected = ext.add_tensors_int16_cuda(ext.conv2d_int16_cuda(x, weight, bias, 1, 1, 0, 0, 1), residual)
                for tile in (16, 32, 64):
                    actual = ext.conv2d_int16_residual_cuda(x, weight, bias, residual, tile)
                    assert torch.equal(actual, expected), (b, c, m, h, w, tile)
                    cases_checked += 1
        # Convolution saturation followed by an opposite-signed residual must not
        # become a single clamp. Also exercise INT32 accumulator wraparound.
        for channels in (1, 2, 3, 32, 33):
            for value, residual_value in [(32767, -32768), (-32768, 32767)]:
                x = torch.full((1, channels, 1, 3), value, dtype=torch.int16, device='cuda')
                weight = torch.full((65, channels, 1, 1), 32767, dtype=torch.int16, device='cuda')
                residual = torch.full((1, 65, 1, 3), residual_value, dtype=torch.int16, device='cuda')
                expected = ext.add_tensors_int16_cuda(ext.conv2d_int16_cuda(x, weight, None, 1, 1, 0, 0, 1), residual)
                for tile in (16, 32, 64):
                    assert torch.equal(ext.conv2d_int16_residual_cuda(x, weight, None, residual, tile), expected)
                    cases_checked += 1
        results = []
        for index, case in enumerate(CASES):
            name, c, m, h, w, kernel, _, _ = unpack_case(case)
            if kernel != 1:
                continue
            x, weight, bias = make_case(index, c, m, h, w, kernel)
            residual = torch.randint(-32768, 32768, (1, m, h, w), generator=generator, dtype=torch.int16).cuda()
            baseline = lambda: ext.add_tensors_int16_cuda(ext.conv2d_int16_cuda(x, weight, bias, 1, 1, 0, 0, 1), residual)
            expected = baseline()
            for tile in (16, 32, 64):
                assert torch.equal(ext.conv2d_int16_residual_cuda(x, weight, bias, residual, tile), expected)
                cases_checked += 1
            timings = {'separate': timed(baseline, args.iterations)}
            for tile in (16, 32, 64):
                timings[str(tile)] = timed(lambda: ext.conv2d_int16_residual_cuda(x, weight, bias, residual, tile), args.iterations)
            results.append(dict(name=name, milliseconds=timings, fastest_tile=min((16, 32, 64), key=lambda t: timings[str(t)])))
            print(json.dumps(results[-1]), flush=True)
    stream.synchronize()
    # An untuned shape must be capturable without event timing/synchronization.
    from src.layers import int16_inference as ops
    ops.conv2d_int16_residual_cuda = ext.conv2d_int16_residual_cuda
    ops._RESIDUAL_TILES.clear()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tile = ops._residual_tile(x, weight, bias, residual)
        captured = ext.conv2d_int16_residual_cuda(x, weight, bias, residual, tile)
    assert tile == 32 and not ops._RESIDUAL_TILES
    for value in (-32768, 0, 32767):
        residual.fill_(value)
        graph.replay()
        expected = ext.add_tensors_int16_cuda(ext.conv2d_int16_cuda(x, weight, bias, 1, 1, 0, 0, 1), residual)
        assert torch.equal(captured, expected)
    selected = ops._residual_tile(x, weight, bias, residual)
    assert selected in (16, 32, 64) and len(ops._RESIDUAL_TILES) == 1
    assert ops._residual_tile(x, weight, bias, residual) == selected
    print('PASS: uncached graph capture, changed residuals, and autotune cache reuse')
    report = dict(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  exact_cases=cases_checked, graph_and_cache_exact=True, results=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {cases_checked} exact fusion/tile cases on a non-default stream')


if __name__ == '__main__':
    main()
