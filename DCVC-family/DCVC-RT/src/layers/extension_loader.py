import importlib
import importlib.util
import sys
from pathlib import Path


def _has_required_attrs(module, required):
    return all(hasattr(module, attr) for attr in required)


def _iter_local_candidates():
    base_dir = Path(__file__).resolve().parent / "extensions" / "inference"
    yield from sorted(base_dir.glob("inference_extensions_cuda*.so"))
    for build_dir in sorted(base_dir.glob("build/lib.*")):
        yield from sorted(build_dir.glob("inference_extensions_cuda*.so"))


def load_inference_extensions(required=()):
    last_error = None

    for candidate in _iter_local_candidates():
        try:
            spec = importlib.util.spec_from_file_location("inference_extensions_cuda", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["inference_extensions_cuda"] = module
            spec.loader.exec_module(module)
            if _has_required_attrs(module, required):
                return module
            last_error = AttributeError(
                f"{candidate} is missing: {', '.join(required)}"
            )
        except Exception as exc:  # pylint: disable=W0718
            sys.modules.pop("inference_extensions_cuda", None)
            last_error = exc

    try:
        module = importlib.import_module("inference_extensions_cuda")
        if _has_required_attrs(module, required):
            return module
        last_error = AttributeError(
            f"imported inference_extensions_cuda is missing: {', '.join(required)}"
        )
    except Exception as exc:  # pylint: disable=W0718
        last_error = exc

    if last_error is None:
        missing = ", ".join(required) if required else "requested symbols"
        raise ImportError(f"unable to load inference_extensions_cuda with {missing}")
    raise last_error
