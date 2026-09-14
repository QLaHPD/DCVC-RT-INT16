"""Load the shared RT managed presentation layer without importing its codec."""
from pathlib import Path
import importlib
import sys
import types


def managed_module(module):
    # UF and RT share one versioned repository. Reuse the actual RT implementation
    # rather than maintaining a second lookalike dashboard/key map.
    name = 'src.cli'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[2] / 'DCVC-family/DCVC-RT/src/cli')]
        sys.modules[name] = package
    return importlib.import_module('src.cli.' + module)


def dashboard_module():
    return managed_module("dashboard")
