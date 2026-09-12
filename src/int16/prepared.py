"""Portable, content-identified checkpoint preparation for the UF integer codec."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .ops import ARITHMETIC_ID, FEATURE_SCALE, WEIGHT_SCALE, quantize_parameter


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_model(variant, meta=False):
    from src.models.image_model import DMCI
    from src.utils.common import ModelStructure
    def construct():
        if variant == 'image':
            return DMCI().eval()
        if variant == 'ld':
            from src.models.video_model_ld import DMC
            return DMC().eval()
        if variant in ('hts', 'htl'):
            from src.models.video_model_ht import DMC
            return DMC(ModelStructure(variant)).eval()
        raise ValueError(f'Unknown model variant: {variant}')
    if meta:
        with torch.device('meta'):
            return construct()
    return construct()


def content_hash(data):
    digest = hashlib.sha256()
    header = {key: data[key] for key in ('format', 'arithmetic', 'variant', 'source_sha256')}
    digest.update(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
    for name, tensor in sorted(data['tensors'].items()):
        if not isinstance(tensor, torch.Tensor) or tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f'Prepared value is not an integer tensor: {name}')
        descriptor = [name, str(tensor.dtype), list(tensor.shape)]
        digest.update(json.dumps(descriptor, separators=(',', ':')).encode())
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(values.astype(values.dtype.newbyteorder('<'), copy=False).tobytes())
    return digest.hexdigest()


def load_prepared(path, source_sha256=None, variant=None, expected_identity=None):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format') != 'dcvc-uf-prepared-v1' or data.get('arithmetic') != ARITHMETIC_ID:
        raise ValueError('Unsupported UF integer preparation format/arithmetic')
    if source_sha256 is not None and data['source_sha256'] != source_sha256:
        raise ValueError('Prepared model does not match source checkpoint')
    if variant is not None and data['variant'] != variant:
        raise ValueError('Prepared model variant mismatch')
    identity = content_hash(data)
    if data.get('identity') != identity or (expected_identity is not None and identity != expected_identity):
        raise ValueError('Prepared integer model/table identity mismatch; copy the matching prepared file')
    return data


def resolve_prepared(checkpoint, variant, path=None, expected_identity=None, expected_source=None):
    """An explicit prepared file is self-contained; float weights are only needed to create it."""
    if path is None and expected_source is not None:
        source = Path(checkpoint)
        candidate = source.parent / f'{source.name}.{ARITHMETIC_ID}.{expected_source[:16]}.int16.pt'
        if candidate.exists():
            path = candidate
    if path is not None and Path(path).exists():
        return Path(path), load_prepared(path, expected_source, variant, expected_identity)
    path, data = prepare(checkpoint, variant, path)
    if expected_identity is not None and data['identity'] != expected_identity:
        raise ValueError('Prepared identity mismatch; copy the prepared file used by the encoder')
    if expected_source is not None and data['source_sha256'] != expected_source:
        raise ValueError('Prepared checkpoint identity mismatch')
    return path, data


def prepare(checkpoint, variant, output=None):
    from src.utils.common import get_state_dict
    checkpoint = Path(checkpoint).resolve()
    source_sha = file_hash(checkpoint)
    output = Path(output) if output else checkpoint.parent / f'{checkpoint.name}.{ARITHMETIC_ID}.{source_sha[:16]}.int16.pt'
    if output.exists():
        return output, load_prepared(output, source_sha, variant)
    model = make_model(variant)
    model.load_state_dict(get_state_dict(checkpoint), strict=True)
    model.update(0)
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            tensors[name+'.weight'] = quantize_parameter(module.weight, WEIGHT_SCALE)
            if module.bias is not None:
                tensors[name+'.bias'] = quantize_parameter(module.bias)
    for name, value in model.named_parameters():
        if name.startswith('q_'):
            tensors[name] = quantize_parameter(value)
    values = torch.arange(-32768, 32768, dtype=torch.float64) / FEATURE_SCALE
    tensors['tables.wsilu'] = quantize_parameter(values * torch.sigmoid(4*values))
    gaussian = model.gaussian_encoder
    scales = values.clamp(gaussian.scale_min, gaussian.scale_max)
    scale_min = torch.tensor(gaussian.scale_min, dtype=torch.float64).log()
    scale_max = torch.tensor(gaussian.scale_max, dtype=torch.float64).log()
    indexes = ((scales.log()-scale_min) * ((gaussian.scale_level-1)/(scale_max-scale_min)))
    tensors['tables.scale_index'] = indexes.floor().clamp(0, gaussian.scale_level-1).byte()
    for name, estimator in (('z', model.bit_estimator_z), ('y', gaussian)):
        cdf, lengths = estimator.get_cdf_info()
        tensors[f'tables.{name}_cdf'] = torch.from_numpy(cdf.copy()).int()
        tensors[f'tables.{name}_lengths'] = torch.from_numpy(lengths.copy()).int()
    data = {'format': 'dcvc-uf-prepared-v1', 'arithmetic': ARITHMETIC_ID,
            'variant': variant, 'source_sha256': source_sha, 'tensors': tensors}
    data['identity'] = content_hash(data)
    if file_hash(checkpoint) != source_sha:
        raise ValueError('Checkpoint changed during integer preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+output.name, dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Concurrent preparation may race; an existing cache is never replaced.
            os.link(temporary, output)
        except FileExistsError:
            existing = load_prepared(output, source_sha, variant)
            if existing['identity'] != data['identity']:
                raise ValueError('Concurrent preparation produced a different identity')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output, load_prepared(output, source_sha, variant)
