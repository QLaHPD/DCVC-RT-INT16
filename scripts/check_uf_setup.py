"""Inspect local UF prerequisites without loading checkpoints or encoding media."""

import argparse
import importlib
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-native', action='store_true',
                        help='Fail unless the UF native runtime is ready too')
    parser.add_argument('--require-int16-native', action='store_true',
                        help='Also require the integer CUDA runtime and entropy decoder API')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Direct script execution otherwise places only scripts/ on sys.path.
    sys.path.insert(0, str(root))
    import torch
    import src.models.image_model
    import src.models.video_model_ht
    import src.models.video_model_ld

    print(f'UF checkout: {root}')
    print(f'Python: {sys.version.split()[0]} ({sys.executable})')
    print(f'PyTorch: {torch.__version__}; built for CUDA {torch.version.cuda}')
    cuda = torch.cuda.is_available()
    print(f'CUDA: {torch.cuda.get_device_name(0) if cuda else "unavailable"}')
    print('UF Python model imports: OK')
    print('Arithmetic: FP16 default; explicit --runtime int16 selects the prepared integer runtime')
    native_ready = cuda
    for module_name, symbols in (
        ('MLCodec_extensions_cpp', ('RansEncoder', 'RansDecoder')),
        ('inference_extensions_cuda', ('DMCIProxy', 'DMCHTSProxy', 'DMCHTLProxy', 'DMCLDProxy')),
    ):
        try:
            module = importlib.import_module(module_name)
            origin = Path(module.__file__).resolve()
            if not origin.is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError(f'extension comes from outside the UF environment: {origin}')
            missing = [name for name in symbols if not hasattr(module, name)]
            if missing:
                raise RuntimeError(f'missing UF API {missing}; possibly an RT extension')
            print(f'{module_name}: OK ({origin})')
        except (ImportError, RuntimeError, OSError) as exc:
            native_ready = False
            print(f'{module_name}: NOT READY ({exc})')
    integer_ready = cuda
    try:
        from src.int16.ops import ARITHMETIC_ID
        integer = importlib.import_module('uf_int16_cuda')
        origin = Path(integer.__file__).resolve()
        if not origin.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f'integer extension outside UF environment: {origin}')
        if getattr(integer, 'arithmetic_id', None) != ARITHMETIC_ID:
            raise RuntimeError('integer arithmetic version mismatch; rebuild extension')
        for symbol in ('conv2d', 'conv2d_generic', 'conv2d_baseline', 'add', 'multiply', 'lookup', 'from_bytes', 'wsilu4'):
            if not hasattr(integer, symbol):
                raise RuntimeError(f'missing integer API: {symbol}')
        entropy = importlib.import_module('MLCodec_extensions_cpp')
        if not hasattr(entropy.RansDecoder, 'get_decoded_tensor'):
            raise RuntimeError('rebuild entropy extension for integer decoder API')
        print(f'uf_int16_cuda: OK ({origin}); {ARITHMETIC_ID}')
        print(f'Integer kernel revision: {getattr(integer, "kernel_revision", "legacy")}')
    except (ImportError, RuntimeError, OSError) as exc:
        integer_ready = False
        print(f'uf_int16_cuda: NOT READY ({exc})')
    cutlass = root / 'third_party/cutlass'
    if (cutlass / 'include/cutlass/cutlass.h').exists():
        revision = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', '--short', 'HEAD'], text=True).strip()
        print(f'CUTLASS: {revision}')
    else:
        print('CUTLASS: missing; see docs/UF_LOCAL_SETUP.md')
    checkpoints = sorted(p.name for p in (root / 'checkpoints').iterdir()
                         if p.is_file() and p.name not in {'.gitkeep', 'README.md'})
    print(f'Checkpoint directory: {root / "checkpoints"}')
    print(f'Checkpoint files: {", ".join(checkpoints) if checkpoints else "awaiting user-provided models"}')
    print('Native import checks do not establish encode/decode or cross-device parity.')
    return 1 if ((args.require_native and not native_ready) or
                 (args.require_int16_native and not integer_ready)) else 0


if __name__ == '__main__':
    raise SystemExit(main())
