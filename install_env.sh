#!/usr/bin/env bash
# Install the inference overlay without modifying the container's PyTorch/CUDA.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash install_env.sh [--help]

Requires Apptainer or Singularity and the NVIDIA PyTorch 24.09-py3 image.
Load your site's container runtime module first, if needed.

Environment variables:
  CONTAINER_IMAGE  Required: path to the existing .sif image (IMAGE also accepted).
  PKG_ENV         Required: a fresh directory for the package overlay.
  PKG_MARKER      Default: $PKG_ENV/.qwen3_atomic_env_ok
  WITH_GPU        Set to 1 to use --nv and require a visible CUDA GPU; default: 0.

The target must not already exist. No sudo, image download, or system package
changes are performed. Installation needs network access to the configured pip
index. The target is a flat PYTHONPATH overlay, not an activated virtualenv.

Export CONTAINER_IMAGE and PKG_ENV, then run: bash install_env.sh
Keep PKG_ENV exported when running the launchers.
For heuristic launchers, also export HEURISTIC_PKG_DIR="$PKG_ENV".

Scope: inference/data-construction dependencies, not the training environment.
Training additionally needs compatible trl/datasets versions, which were not
reported in the supplied environment inspection.
HELP
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${IMAGE:-}}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to an existing container image}"
: "${PKG_ENV:?Set PKG_ENV to a fresh package directory}"
PKG_MARKER="${PKG_MARKER:-$PKG_ENV/.qwen3_atomic_env_ok}"
WITH_GPU="${WITH_GPU:-0}"

[[ "$WITH_GPU" == 0 || "$WITH_GPU" == 1 ]] || {
  echo '[ERROR] WITH_GPU must be 0 or 1.' >&2; exit 2;
}
[[ -f "$CONTAINER_IMAGE" ]] || {
  echo '[ERROR] CONTAINER_IMAGE must point to an existing NVIDIA PyTorch 24.09-py3 image.' >&2
  exit 2
}
if command -v apptainer >/dev/null 2>&1; then
  CTR_CMD=apptainer
elif command -v singularity >/dev/null 2>&1; then
  CTR_CMD=singularity
else
  echo '[ERROR] Load or install Apptainer/Singularity first.' >&2
  exit 2
fi
[[ ! -e "$PKG_ENV" && ! -L "$PKG_ENV" ]] || {
  echo '[ERROR] PKG_ENV already exists. Choose a fresh target; nothing was changed.' >&2
  exit 2
}

# Resolve the output parent so a relative custom target also works in a container.
mkdir -p -- "$(dirname -- "$PKG_ENV")"
TARGET_PARENT="$(cd -- "$(dirname -- "$PKG_ENV")" && pwd -P)"
PKG_ENV="$TARGET_PARENT/$(basename -- "$PKG_ENV")"
PKG_MARKER_PARENT="$(dirname -- "$PKG_MARKER")"
STAGE="$(mktemp -d "$TARGET_PARENT/.inference-install.XXXXXXXX")"
cleanup() { rm -rf -- "$STAGE"; }
trap cleanup EXIT

CTR_ARGS=(exec --cleanenv)
[[ "$WITH_GPU" == 0 ]] || CTR_ARGS+=(--nv)

# Use a temporary system-site-packages venv for dependency resolution. Unlike
# pip --target with dependency resolution, this reuses the base torch stack.
# Only the newly installed Python packages are then copied to the flat overlay.
"$CTR_CMD" "${CTR_ARGS[@]}" \
  --bind "$TARGET_PARENT:$TARGET_PARENT" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH= \
  --env PYTHONUNBUFFERED=1 \
  "$CONTAINER_IMAGE" python - "$STAGE" "$WITH_GPU" <<'PY'
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

stage = Path(sys.argv[1])
require_gpu = sys.argv[2] == "1"
import torch

if sys.version_info[:2] != (3, 10):
    raise RuntimeError("Expected Python 3.10 from the NVIDIA 24.09-py3 image")
if not torch.__version__.startswith("2.5.0a0") or torch.version.cuda != "12.6":
    raise RuntimeError("Expected container PyTorch 2.5.0a0 with CUDA 12.6")
if require_gpu and not torch.cuda.is_available():
    raise RuntimeError("WITH_GPU=1 but CUDA is not available")

# Direct versions reported by the supplied inspection. Container-provided
# torchvision, Jinja2, einops, flash-attn and triton are intentionally reused.
requirements = [
    "transformers==4.51.0", "tokenizers==0.21.4", "safetensors==0.7.0",
    "huggingface-hub==0.36.2", "accelerate==1.13.0", "numpy==1.26.4",
    "sentencepiece==0.2.0", "protobuf==4.25.9", "tqdm==4.67.3",
    "regex==2026.4.4", "packaging==26.2", "filelock==3.29.0",
    "PyYAML==6.0.3", "requests==2.33.1", "httpx==0.28.1",
    "openai==2.33.0", "pydantic==2.13.3", "psutil==7.2.2", "peft==0.11.1",
    # Required by heuristic_v3.py, but not inspected in the supplied version log.
    "pysbd==0.3.4",
]

# Prevent the resolver from upgrading the container's compiled GPU packages.
protected = {}
for dist in metadata.distributions():
    name = dist.metadata.get("Name", "").lower().replace("_", "-")
    if name in {"torch", "torchvision", "torchaudio", "triton", "flash-attn"} or name.startswith("nvidia-"):
        protected[name] = dist.version
constraints = stage / "container-constraints.txt"
constraints.write_text("".join(f"{name}==={version}\n" for name, version in sorted(protected.items())))

temporary_venv = stage / "resolver"
venv.create(temporary_venv, system_site_packages=True, with_pip=True)
python = str(temporary_venv / "bin/python")
env = dict(os.environ, PIP_REQUIRE_VIRTUALENV="true")
subprocess.run([
    python, "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
    "--no-cache-dir", "--constraint", str(constraints), *requirements,
], check=True, env=env)

site = Path(subprocess.check_output([
    python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))",
], text=True, env=env).strip())
for dist in metadata.distributions(path=[str(site)]):
    name = dist.metadata.get("Name", "").lower().replace("_", "-")
    if name in protected or name.startswith("nvidia-"):
        raise RuntimeError(f"Refusing to shadow container GPU package: {name}")

overlay = stage / "overlay"
shutil.copytree(site, overlay, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
env.update(PYTHONPATH=str(overlay), PYTHONNOUSERSITE="1")
smoke = r'''
import importlib
import importlib.metadata as metadata
import json
import sys
import torch

expected = json.loads(sys.argv[1])
for requirement in expected:
    name, version = requirement.split("==", 1)
    actual = metadata.version(name)
    if actual != version:
        raise RuntimeError(f"{name}: expected {version}, got {actual}")
    print(f"[OK] {name}=={actual}")
for module in (
    "transformers", "tokenizers", "safetensors", "huggingface_hub",
    "accelerate", "numpy", "sentencepiece", "google.protobuf", "tqdm",
    "regex", "packaging", "filelock", "yaml", "requests", "httpx",
    "openai", "pydantic", "psutil", "peft", "pysbd",
):
    importlib.import_module(module)
print(f"[OK] Container torch {torch.__version__}; CUDA build {torch.version.cuda}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
'''
import json
subprocess.run([sys.executable, "-c", smoke, json.dumps(requirements)], check=True, env=env)
(overlay / "inference-requirements.txt").write_text("\n".join(requirements) + "\n")
print("[OK] Inference overlay passed package/version import checks.")
PY

# Publish only after successful installation and imports; never overwrite an env.
[[ ! -e "$PKG_ENV" && ! -L "$PKG_ENV" ]] || {
  echo '[ERROR] Target appeared during installation; refusing to overwrite it.' >&2
  exit 2
}
mv -- "$STAGE/overlay" "$PKG_ENV"
mkdir -p -- "$PKG_MARKER_PARENT"
touch -- "$PKG_MARKER"
echo '[DONE] Inference overlay installed.'
echo '[INFO] Keep PKG_ENV exported for all launchers.'
