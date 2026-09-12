import unittest
import torch

from src.int16 import ops


class IntegerArithmeticTests(unittest.TestCase):
    def test_rounding_saturation_and_source_boundary(self):
        x = torch.tensor([-13, -12, -11, -4, 0, 4, 11, 12, 13])
        self.assertEqual(ops.divide_round(x, 8).tolist(), [-2, -2, -1, -1, 0, 1, 1, 2, 2])
        self.assertEqual(ops.add(torch.tensor([32767, -32768], dtype=torch.int16),
                                 torch.tensor([1, -1], dtype=torch.int16)).tolist(), [32767, -32768])
        self.assertEqual(ops.from_bytes(torch.tensor([0, 127, 128, 255], dtype=torch.uint8)).tolist(),
                         [-256, -1, 1, 256])

    def test_convolution_matches_scalar_integer_definition(self):
        torch.manual_seed(11)
        x = torch.randint(-1000, 1000, (1, 4, 5, 7), dtype=torch.int16)
        w = torch.randint(-800, 800, (6, 2, 3, 3), dtype=torch.int16)
        bias = torch.arange(6, dtype=torch.int16)
        result = ops.conv2d_reference(x, w, bias, stride=(2, 1), padding=(1, 1), groups=2)
        expected = torch.empty_like(result)
        for oc in range(6):
            for oy in range(3):
                for ox in range(7):
                    total = 0
                    for ic in range(2):
                        for ky in range(3):
                            for kx in range(3):
                                iy, ix = oy*2+ky-1, ox+kx-1
                                if 0 <= iy < 5 and 0 <= ix < 7:
                                    total += int(x[0, (oc//3)*2+ic, iy, ix]) * int(w[oc,ic,ky,kx])
                    value = (abs(total)+4096)//8192 * (-1 if total<0 else 1) + oc
                    expected[0,oc,oy,ox] = max(-32768,min(32767,value))
        self.assertTrue(torch.equal(result, expected))

    def test_wide_accumulator_does_not_wrap_int32(self):
        x = torch.full((1, 128, 1, 1), -32768, dtype=torch.int16)
        w = torch.full((1, 128, 1, 1), -32768, dtype=torch.int16)
        self.assertEqual(ops.conv2d_reference(x, w).item(), 32767)
        w.fill_(32767)
        self.assertEqual(ops.conv2d_reference(x, w).item(), -32768)


class CudaIntegerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA not available')
        try:
            import uf_int16_cuda
        except ImportError:
            raise unittest.SkipTest('UF integer CUDA extension not built')

    def test_fused_operations_match_reference(self):
        torch.manual_seed(29)
        x = torch.randint(-32768,32768,(2,12,5,7),dtype=torch.int16)
        lut = torch.arange(-32768,32768,dtype=torch.int16)
        byte_lut = (torch.arange(65536)%128).byte()
        for table in (lut, byte_lut):
            self.assertTrue(torch.equal(ops.lookup(x,table),ops.lookup(x.cuda(),table.cuda()).cpu()))
        self.assertTrue(torch.equal(ops.wsilu4(x,lut),ops.wsilu4(x.cuda(),lut.cuda()).cpu()))
        for shape in ((1,12,1,1),(2,12,5,7),(1,1,5,1)):
            scale = torch.randint(-32768,32768,shape,dtype=torch.int16)
            self.assertTrue(torch.equal(ops.multiply(x,scale),ops.multiply(x.cuda(),scale.cuda()).cpu()))
        # Mixed signs must cancel before saturation, not after each pairwise add.
        operands = [torch.full_like(x,v) for v in (32767,32767,-32768,-32768)]
        self.assertTrue(torch.equal(ops.add(*operands),ops.add(*(v.cuda() for v in operands)).cpu()))
        values = torch.arange(256,dtype=torch.uint8)
        self.assertTrue(torch.equal(ops.from_bytes(values),ops.from_bytes(values.cuda()).cpu()))

    def test_cuda_matches_cpu_including_overflow_and_odd_tiles(self):
        torch.manual_seed(19)
        cases = [(2, 35, 19, 3, 7, 1, 1, 0, 1), (1, 6, 9, 7, 9, 3, 2, 1, 3),
                 (1, 5, 5, 8, 11, 3, 1, 1, 5), (1, 128, 17, 1, 1, 1, 1, 0, 1)]
        for batch, ci, co, h, w, kernel, stride, pad, groups in cases:
            x = torch.randint(-32768, 32768, (batch, ci, h, w), dtype=torch.int16)
            weights = torch.randint(-32768, 32768, (co, ci//groups, kernel, kernel), dtype=torch.int16)
            bias = torch.randint(-1000, 1000, (co,), dtype=torch.int16)
            reference = ops.conv2d_reference(x, weights, bias, (stride, stride), (pad, pad), groups)
            actual = ops.conv2d(x.cuda(), weights.cuda(), bias.cuda(), (stride, stride), (pad, pad), groups)
            self.assertTrue(torch.equal(reference, actual.cpu()), str((ci, co, kernel, groups)))

    def test_mma_reduction_bounds_and_cancellation(self):
        for channels in (32768, 32769):
            x = torch.full((1, channels, 1, 1), 32767, dtype=torch.int16)
            w = torch.full((3, channels, 1, 1), 32767, dtype=torch.int16)
            w[1].fill_(-32768)
            w[2, :channels//2].fill_(-32767)
            if channels % 2:
                w[2, -1].zero_()
            expected = ops.conv2d_reference(x, w)
            actual = ops.conv2d(x.cuda(), w.cuda()).cpu()
            self.assertTrue(torch.equal(expected, actual), str(channels))
            self.assertEqual(actual.flatten().tolist(), [32767, -32768, 0])
            # Maximal unsigned low-byte accumulation must cancel exactly
            # against high-byte terms, leaving an unsaturated result.
            x.fill_(-1)
            w.fill_(-1)
            self.assertTrue(torch.equal(ops.conv2d_reference(x, w),
                                        ops.conv2d(x.cuda(), w.cuda()).cpu()))

    def test_mma_unsaturated_general_convolution(self):
        import uf_int16_cuda as native
        torch.manual_seed(83)
        for kernel, stride, pad in ((1, 1, 0), (3, 1, 1), (3, 2, 1)):
            x = torch.randint(-90, 91, (2, 35, 7, 11), dtype=torch.int16)
            w = torch.randint(-90, 91, (19, 35, kernel, kernel), dtype=torch.int16)
            b = torch.arange(-9, 10, dtype=torch.int16)
            reference = ops.conv2d_reference(x, w, b, (stride,stride), (pad,pad))
            args = (x.cuda(), w.cuda(), b.cuda(), stride, stride, pad, pad, 1)
            self.assertTrue(torch.equal(reference, native.conv2d(*args).cpu()))
            self.assertTrue(torch.equal(reference, native.conv2d_generic(*args).cpu()))

    def test_tiled_pointwise_matches_cpu_and_original_mma(self):
        import uf_int16_cuda as native
        torch.manual_seed(91)
        # Exercise both spatial tiling and deeper K tiling, with partial K/M
        # tiles, odd spatial tails and two batches.
        for shape, co in (((1,35,16,16),19), ((2,35,16,16),19),
                          ((1,67,3,7),71), ((1,128,1,1),17)):
            for bound in (90,32768):
                x=torch.randint(-bound,bound,shape,dtype=torch.int16)
                w=torch.randint(-bound,bound,(co,shape[1],1,1),dtype=torch.int16)
                b=torch.arange(co,dtype=torch.int16)
                reference=ops.conv2d_reference(x,w,b)
                args=(x.cuda(),w.cuda(),b.cuda(),1,1,0,0,1)
                self.assertTrue(torch.equal(reference,native.conv2d(*args).cpu()))
                self.assertTrue(torch.equal(reference,native.conv2d_baseline(*args).cpu()))


if __name__ == '__main__':
    unittest.main()
