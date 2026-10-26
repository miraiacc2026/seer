#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Runtime / container
# ============================================================

# Load the container runtime in the caller environment.

# ============================================================
# Qwen3-8B query-SFT query-structure inference for ASQA
# ============================================================

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the container image file}"
WORKDIR="${WORKDIR:-$RUNNER_DIR}"
PKG_ENV="${PKG_ENV:-}"

SRC_JSON="${SRC_JSON:?Set SRC_JSON to the input dataset file}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to the query output directory}"

BACKEND="${BACKEND:-local}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"

# ASQA eval: indices 0-947 inclusive.
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-947}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1200}"

# Select a query-SFT model and its compatible tokenizer explicitly.
MODEL="${MODEL:-}"
BASE_TOKENIZER="${BASE_TOKENIZER:-$MODEL}"
if [[ "$BACKEND" == "local" ]]; then
  : "${MODEL:?Set MODEL to the local Qwen3 model directory}"
elif [[ "$BACKEND" == "openai" ]]; then
  : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to the API endpoint}"
fi

# ============================================================
# Caches
# ============================================================

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MYPROJ/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MYPROJ/xdg_cache}"
export HF_HOME="${HF_HOME:-$MYPROJ/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MYPROJ/hf_home/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$MYPROJ/hf_home/hub}"

mkdir -p \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_HUB_CACHE" \
  "$MYPROJ/.logs" \
  "$OUT_DIR"

if command -v apptainer >/dev/null 2>&1; then
  CTR_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CTR_CMD="singularity"
else
  echo "❌ Neither apptainer nor singularity found in PATH" >&2
  exit 99
fi

# ============================================================
# Checks
# ============================================================

[[ -f "$CONTAINER_IMAGE" ]] || { echo "❌ Missing image: $CONTAINER_IMAGE" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { echo "❌ Missing WORKDIR: $WORKDIR" >&2; exit 3; }
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "❌ Missing PKG_ENV: $PKG_ENV" >&2; exit 4; }
[[ -f "$SRC_JSON" ]] || { echo "❌ Missing SRC_JSON: $SRC_JSON" >&2; exit 5; }
[[ -f "$WORKDIR/query_range.py" ]] || { echo "❌ Missing query_range.py: $WORKDIR/query_range.py" >&2; exit 6; }
[[ -f "$WORKDIR/query.py" ]] || { echo "❌ Missing query.py: $WORKDIR/query.py" >&2; exit 7; }

if [[ "$BACKEND" == "local" ]]; then
  [[ -d "$MODEL" ]] || { echo "❌ Missing MODEL dir: $MODEL" >&2; exit 8; }
  [[ -d "$BASE_TOKENIZER" ]] || { echo "❌ Missing BASE_TOKENIZER dir: $BASE_TOKENIZER" >&2; exit 9; }
fi

if [[ "$BACKEND" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY is required when BACKEND=openai" >&2
  exit 10
fi

echo "[INFO] Container cmd      : $CTR_CMD"
echo "[INFO] Image              : $CONTAINER_IMAGE"
echo "[INFO] Workdir            : $WORKDIR"
echo "[INFO] PKG_ENV            : $PKG_ENV"
echo "[INFO] BACKEND            : $BACKEND"
echo "[INFO] MODEL              : $MODEL"
echo "[INFO] BASE_TOKENIZER     : $BASE_TOKENIZER"
echo "[INFO] SRC_JSON           : $SRC_JSON"
echo "[INFO] OUT_DIR            : $OUT_DIR"
echo "[INFO] START_IDX          : $START_IDX"
echo "[INFO] END_IDX            : $END_IDX  # inclusive"
echo "[INFO] BATCH_SIZE         : $BATCH_SIZE"
echo "[INFO] MAX_OUTPUT_TOKENS  : $MAX_OUTPUT_TOKENS"

# Use the runtime's default certificate configuration.

# Bind configured locations, including overrides outside the cache root.
SOURCE_ROOT="$(cd -- "$WORKDIR/.." && pwd -P)"
BIND_ARGS=()
bind_configured_path() {
  local name="$1" path="${!1:-}" directory
  [[ -n "$path" ]] || return 0
  [[ "$path" == /* ]] || path="$PWD/$path"
  printf -v "$name" '%s' "$path"
  directory="$path"
  while [[ ! -d "$directory" ]]; do directory="$(dirname -- "$directory")"; done
  BIND_ARGS+=(--bind "$directory:$directory")
}
for name in SOURCE_ROOT WORKDIR PKG_ENV SRC_JSON OUT_DIR PIP_CACHE_DIR XDG_CACHE_HOME \
  HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE; do
  bind_configured_path "$name"
done
if [[ "$BACKEND" == "local" ]]; then
  bind_configured_path MODEL
  bind_configured_path BASE_TOKENIZER
fi

"$CTR_CMD" exec --nv \
  "${BIND_ARGS[@]}" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_MAX_WORKERS=1 \
  --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env MODEL="$MODEL" \
  --env BASE_TOKENIZER="$BASE_TOKENIZER" \
  --env BACKEND="$BACKEND" \
  --env OPENAI_MODEL="$OPENAI_MODEL" \
  --env MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
  --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --env OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  --env WORKDIR="$WORKDIR" \
  --env START_IDX="$START_IDX" \
  --env END_IDX="$END_IDX" \
  --env BATCH_SIZE="$BATCH_SIZE" \
  --env OUT_DIR="$OUT_DIR" \
  --env SRC_JSON="$SRC_JSON" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env PKG_ENV="$PKG_ENV" \
  "$CONTAINER_IMAGE" /bin/bash -lc '
    set -euo pipefail

    cd "$WORKDIR"
    unset PYTHONPATH
    export PYTHONPATH="$PKG_ENV"

    echo "[INFO] inside container pwd=$(pwd)"
    echo "[INFO] python:"
    python - <<PY
import sys
print(sys.executable)
print(sys.path[:5])
PY

    echo "[INFO] sanity check: source data length"
    python - <<PY
import json, os
path = os.environ["SRC_JSON"]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print("SRC_JSON:", path)
print("num_examples:", len(data))
print("first_keys:", list(data[0].keys()) if isinstance(data, list) and data else "N/A")
PY

    echo "[INFO] Running query_range.py ..."
    python query_range.py
    echo "[INFO] Done."
  '
