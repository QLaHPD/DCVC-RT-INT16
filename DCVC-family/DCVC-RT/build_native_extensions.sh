#!/usr/bin/env bash

# Build and install DCVC-RT's CPU entropy coder and CUDA inference kernels.
#
# Usage:
#   conda activate <env_name>
#   ./build_native_extensions.sh
#
# Optional environment overrides:
#   TORCH_CUDA_ARCH_LIST="8.6;8.7+PTX"  GPU architectures to include.
#   MAX_JOBS=2                             Parallel compiler jobs (default: 2).
#   CUDA_HOME=/path/to/cuda                CUDA toolkit containing bin/nvcc.

set -Eeuo pipefail

usage() {
    printf '%s\n' \
        "Usage: conda activate <env_name>; $0" \
        "" \
        "Builds and installs both required native extensions into the active Conda environment:" \
        "  - MLCodec_extensions_cpp (CPU entropy coding)" \
        "  - inference_extensions_cuda (CUDA and int16 kernels)" \
        "" \
        "Overrides: TORCH_CUDA_ARCH_LIST, MAX_JOBS, CUDA_HOME"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf '\n==> %s\n' "$*"
}

[[ -n "${CONDA_PREFIX:-}" ]] || fail "No Conda environment is active. Run: conda activate <env_name>"
command -v python >/dev/null 2>&1 || fail "python is not available after Conda activation"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cpp_dir="$script_dir/src/cpp"
cuda_dir="$script_dir/src/layers/extensions/inference"
[[ -f "$cpp_dir/setup.py" ]] || fail "Missing build definition: $cpp_dir/setup.py"
[[ -f "$cuda_dir/setup.py" ]] || fail "Missing build definition: $cuda_dir/setup.py"

conda_prefix_real="$(cd -- "$CONDA_PREFIX" && pwd -P)"
python_prefix="$({ python - <<'PY'
import os
import sys
print(os.path.realpath(sys.prefix))
PY
} 2>/dev/null)" || fail "Unable to inspect the active Python interpreter"
[[ "$python_prefix" == "$conda_prefix_real" ]] || fail \
    "python belongs to '$python_prefix', not the active Conda environment '$conda_prefix_real'"

python_executable="$({ python - <<'PY'
import os
import sys
print(os.path.realpath(sys.executable))
PY
} 2>/dev/null)"
case "$python_executable" in
    "$conda_prefix_real"/*) ;;
    *) fail "python executable is outside the active Conda environment: $python_executable" ;;
esac

command -v c++ >/dev/null 2>&1 || fail "A C++ compiler is required (install g++ or the Conda C++ compiler)"

if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    nvcc_path="$CUDA_HOME/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(command -v nvcc)"
    CUDA_HOME="$(cd -- "$(dirname -- "$nvcc_path")/.." && pwd -P)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME="/usr/local/cuda"
    nvcc_path="$CUDA_HOME/bin/nvcc"
else
    fail "nvcc was not found. Install a CUDA toolkit compatible with the PyTorch build, then retry."
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

[[ "${MAX_JOBS:-2}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

note "Active build environment"
printf 'Conda prefix : %s\n' "$conda_prefix_real"
printf 'Python       : %s\n' "$python_executable"
printf 'C++ compiler : %s\n' "$(command -v c++)"
printf 'NVCC         : %s\n' "$nvcc_path"
printf 'CUDA_HOME    : %s\n' "$CUDA_HOME"
printf 'MAX_JOBS     : %s\n' "$MAX_JOBS"
"$nvcc_path" --version | tail -n 1

note "Checking PyTorch and installing build helpers in the active environment"
python -m pip install --no-input setuptools wheel pybind11 ninja

torch_report="$({ python - <<'PY'
import torch
print(f"PyTorch {torch.__version__}")
print(f"PyTorch CUDA {torch.version.cuda}")
if torch.version.cuda is None:
    raise SystemExit("This PyTorch installation has no CUDA support")
PY
})" || fail "A CUDA-enabled PyTorch installation is required in the active environment"
printf '%s\n' "$torch_report"

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    detected_arches="$({ python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(3)

arches = sorted({torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())})
values = [f"{major}.{minor}" for major, minor in arches]
values[-1] += "+PTX"
print(";".join(values))
PY
    })" || fail \
        "No CUDA GPU is visible. Set TORCH_CUDA_ARCH_LIST explicitly when compiling without a visible GPU."
    export TORCH_CUDA_ARCH_LIST="$detected_arches"
fi
printf 'CUDA arches  : %s\n' "$TORCH_CUDA_ARCH_LIST"

torch_cuda_version="$(python - <<'PY'
import torch
print(torch.version.cuda)
PY
)"
nvcc_release="$({ "$nvcc_path" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1; })"
[[ -n "$nvcc_release" ]] || fail "Could not determine the NVCC release"
if [[ "${torch_cuda_version%%.*}" != "${nvcc_release%%.*}" ]]; then
    fail "PyTorch uses CUDA $torch_cuda_version but NVCC is CUDA $nvcc_release (major versions differ)"
fi
if [[ "$torch_cuda_version" != "$nvcc_release" ]]; then
    printf 'WARNING: PyTorch uses CUDA %s while NVCC is CUDA %s; PyTorch may warn about the minor mismatch.\n' \
        "$torch_cuda_version" "$nvcc_release" >&2
fi

wheel_dir="$(mktemp -d "${TMPDIR:-/tmp}/dcvc-native-wheels.XXXXXXXX")"
cleanup_temp() {
    rm -rf -- "$wheel_dir"
    if [[ -n "${staged_cuda:-}" && -f "$staged_cuda" ]]; then
        rm -f -- "$staged_cuda"
    fi
}
trap cleanup_temp EXIT

# These are generated build directories at fixed locations under this source tree.
# Removing them prevents a copied binary for another computer from being reused.
note "Removing stale generated build directories"
rm -rf -- "$cpp_dir/build" "$cuda_dir/build"

note "Building the CPU entropy-coder wheel"
python -m pip wheel \
    --no-input --no-cache-dir --no-deps --no-build-isolation \
    --wheel-dir "$wheel_dir" "$cpp_dir"

note "Building the CUDA/int16 inference wheel"
python -m pip wheel \
    --no-input --no-cache-dir --no-deps --no-build-isolation \
    --wheel-dir "$wheel_dir" "$cuda_dir"

shopt -s nullglob
wheels=("$wheel_dir"/*.whl)
shopt -u nullglob
(( ${#wheels[@]} == 2 )) || fail "Expected two extension wheels, found ${#wheels[@]}"

note "Installing freshly built wheels into $conda_prefix_real"
python -m pip install --no-input --no-deps --force-reinstall "${wheels[@]}"

installed_cuda="$({
    cd -- "$wheel_dir"
    env PYTHONPATH= python - <<'PY'
import importlib.util
import os
import torch

spec = importlib.util.find_spec("inference_extensions_cuda")
if spec is None or not spec.origin:
    raise SystemExit("installed CUDA extension was not found")
print(os.path.realpath(spec.origin))
PY
})" || fail "Could not locate the installed CUDA extension"
case "$installed_cuda" in
    "$conda_prefix_real"/*) ;;
    *) fail "Installed CUDA extension resolved outside the active environment: $installed_cuda" ;;
esac

# The project loader intentionally checks this source directory before site-packages.
# Replace copied/stale binaries only after the new wheel has built and installed successfully.
note "Refreshing the source-tree CUDA binary used by the project loader"
cuda_basename="$(basename -- "$installed_cuda")"
staged_cuda="$cuda_dir/.$cuda_basename.new.$$"
install -m 0755 -- "$installed_cuda" "$staged_cuda"
find "$cuda_dir" -maxdepth 1 -type f -name 'inference_extensions_cuda*.so' -delete
mv -f -- "$staged_cuda" "$cuda_dir/$cuda_basename"

note "Verifying installed modules and required int16 kernel symbols"
(
    cd -- "$wheel_dir"
    env PYTHONPATH= python - <<'PY'
import os
import torch
import MLCodec_extensions_cpp as entropy
import inference_extensions_cuda as cuda_ext

entropy_required = ("RansEncoder", "RansDecoder", "pmf_to_quantized_cdf")
cuda_required = (
    "conv2d_int16_cuda",
    "add_bias_int16_cuda",
    "add_tensors_int16_cuda",
    "mul_feature_scale_int16_cuda",
    "reciprocal_scale_int16_cuda",
    "apply_lut_int16_cuda",
    "process_with_mask_int16_cuda",
    "combine_for_reading_int16_cuda",
    "restore_y_parts_int16_cuda",
    "build_index_dec_int16_cuda",
    "build_index_enc_int16_cuda",
    "add_and_multiply_int16_cuda",
    "bias_quant_int16_cuda",
    "wsilu_chunk_add_int16_cuda",
    "bias_wsilu_depthwise_conv2d_int16_cuda",
    "bias_pixel_shuffle_2_int16_cuda",
    "bias_pixel_shuffle_8_int16_cuda",
)

missing_entropy = [name for name in entropy_required if not hasattr(entropy, name)]
missing_cuda = [name for name in cuda_required if not hasattr(cuda_ext, name)]
if missing_entropy or missing_cuda:
    raise SystemExit(
        f"extension verification failed; entropy missing={missing_entropy}, CUDA missing={missing_cuda}"
    )

print(f"Entropy module: {os.path.realpath(entropy.__file__)}")
print(f"CUDA module   : {os.path.realpath(cuda_ext.__file__)}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("All required native and int16 symbols are present.")
PY
)

note "Verifying the repository loader selects the freshly built CUDA binary"
(
    cd -- "$script_dir"
    DCVC_USE_INT16=1 python - <<'PY'
import os
from src.layers.extension_loader import load_inference_extensions

module = load_inference_extensions(required=("process_with_mask_int16_cuda",))
print(f"Repository CUDA module: {os.path.realpath(module.__file__)}")
PY
)

printf '\nBuild and installation completed successfully.\n'
printf 'Keep this script with the repository and rerun it after copying the tool to another computer.\n'
