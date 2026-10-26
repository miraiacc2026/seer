#!/usr/bin/env bash
# Rerank R3 SFT - Qwen3-8B - 2 GPU FSDP
set -euo pipefail

# ============================================================
# Global paths
# ============================================================

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"
WORKDIR="${WORKDIR:-$RUNNER_DIR}"
SCRIPT="${SCRIPT:-$WORKDIR/train_rerank_fsdp.py}"

# ============================================================
# Qwen3 / package environment
# ============================================================

PKG_ENV="${PKG_ENV:-}"
PKG_MARKER="${PKG_MARKER:-}"

PKG_TRAIN="${PKG_TRAIN:-}"
PKG_OPENAI="${PKG_OPENAI:-}"
PKG_JSON="${PKG_JSON:-}"
PKG_ST="${PKG_ST:-}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"

# ============================================================
# Model / Data
# ============================================================

MODEL="${MODEL:?Set MODEL to the local Qwen3 model directory}"
TOKENIZER_ID="${TOKENIZER_ID:-$MODEL}"

DATA_PATH="${DATA_PATH:?Set DATA_PATH to the training JSON file}"

OUTPUT_ROOT="${OUTPUT_ROOT:-}"

DATE_HOUR="$(date +%Y%m%d_%H)"
RUN_NAME="${RUN_NAME:-qwen3_rerank_sft_${DATE_HOUR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:?Set OUTPUT_DIR or OUTPUT_ROOT}/$RUN_NAME}"

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ============================================================
# Shared caches
# ============================================================

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MYPROJ/pip_cache_qwen3}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MYPROJ/xdg_cache_qwen3}"
export HF_HOME="${HF_HOME:-$MYPROJ/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MYPROJ/hf_datasets_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MYPROJ/triton_cache}"
export TMPDIR="${TMPDIR:-$MYPROJ/tmp}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR"

# ============================================================
# Distributed settings
# ============================================================

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29519}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ============================================================
# Training hyperparameters
# ============================================================

SEED="${SEED:-42}"
MAX_EXAMPLES="${MAX_EXAMPLES:--1}"

MAX_LEN="${MAX_LEN:-4096}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"

# Approximately half an epoch for 3489 training samples
# Global batch = 1 x 2 x 8 = 16

SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-109}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}"

PACKING="${PACKING:-0}"

GRAD_CKPT="${GRAD_CKPT:-1}"
FORCE_GRAD_CKPT="${FORCE_GRAD_CKPT:-0}"

ATTN_IMPL="${ATTN_IMPL:-eager}"
FSDP_WRAP_CLS="${FSDP_WRAP_CLS:-Qwen3DecoderLayer}"

# ============================================================
# Load modules
# ============================================================

# Load the container runtime in the caller environment.

if command -v apptainer >/dev/null 2>&1; then
  CTR_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CTR_CMD="singularity"
else
  echo "[ERROR] Neither apptainer nor singularity found" >&2
  exit 99
fi

# ============================================================
# Sanity checks
# ============================================================

[[ -f "$IMAGE" ]] || {
  echo "[ERROR] Missing image: $IMAGE" >&2
  exit 2
}

[[ -d "$WORKDIR" ]] || {
  echo "[ERROR] Missing WORKDIR: $WORKDIR" >&2
  exit 3
}

[[ -f "$WORKDIR/$SCRIPT" || -f "$SCRIPT" ]] || {
  echo "[ERROR] Missing training script: $SCRIPT" >&2
  exit 4
}

[[ -f "$DATA_PATH" ]] || {
  echo "[ERROR] Missing training JSON: $DATA_PATH" >&2
  exit 5
}

[[ -d "$MODEL" ]] || {
  echo "[ERROR] Missing model: $MODEL" >&2
  exit 6
}

[[ -d "$TOKENIZER_ID" ]] || {
  echo "[ERROR] Missing tokenizer: $TOKENIZER_ID" >&2
  exit 7
}

[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || {
  echo "[ERROR] Missing PKG_ENV: $PKG_ENV" >&2
  exit 8
}

[[ -z "$PKG_MARKER" || -f "$PKG_MARKER" ]] || {
  echo "[ERROR] Missing PKG marker: $PKG_MARKER" >&2
  exit 9
}

if [[ -n "$PKG_TRAIN" && ! -d "$PKG_TRAIN" ]]; then
  echo "[ERROR] Missing PKG_TRAIN: $PKG_TRAIN" >&2
  exit 10
fi

# ============================================================
# Training information
# ============================================================

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

GLOBAL_BATCH=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))

echo "[INFO] Container       : $CTR_CMD"
echo "[INFO] Image           : $IMAGE"
echo "[INFO] Script          : $SCRIPT"
echo "[INFO] Data            : $DATA_PATH"
echo "[INFO] Model           : $MODEL"
echo "[INFO] Tokenizer       : $TOKENIZER_ID"
echo "[INFO] Output          : $OUTPUT_DIR"
echo "[INFO] Log             : $LOG_FILE"
echo "[INFO] NPROC_PER_NODE  : $NPROC_PER_NODE"
echo "[INFO] MAX_EXAMPLES    : $MAX_EXAMPLES"
echo "[INFO] NUM_EPOCHS      : $NUM_EPOCHS"
echo "[INFO] BATCH_SIZE      : $BATCH_SIZE"
echo "[INFO] GRAD_ACCUM      : $GRAD_ACCUM"
echo "[INFO] GLOBAL_BATCH    : $GLOBAL_BATCH"
echo "[INFO] LR              : $LR"
echo "[INFO] MAX_LEN         : $MAX_LEN"
echo "[INFO] SAVE_STRATEGY   : $SAVE_STRATEGY"
echo "[INFO] SAVE_STEPS      : $SAVE_STEPS"
echo "[INFO] SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT"
echo "[INFO] GRAD_CKPT       : $GRAD_CKPT"
echo "[INFO] FORCE_GRAD_CKPT : $FORCE_GRAD_CKPT"
echo "[INFO] ATTN_IMPL       : $ATTN_IMPL"

# Use the runtime's default certificate configuration.

# ============================================================
# Apptainer bindings
# ============================================================

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
if [[ "$SCRIPT" != /* && -f "$WORKDIR/$SCRIPT" ]]; then SCRIPT="$WORKDIR/$SCRIPT"; fi
SOURCE_ROOT="$(cd -- "$(dirname -- "$SCRIPT")/.." && pwd -P)"
for name in SOURCE_ROOT WORKDIR SCRIPT DATA_PATH MODEL TOKENIZER_ID OUTPUT_DIR LOG_DIR LOG_FILE \
  PKG_ENV PKG_TRAIN PKG_OPENAI PKG_JSON PKG_ST PIP_CACHE_DIR XDG_CACHE_HOME \
  HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRITON_CACHE_DIR TMPDIR; do
  bind_configured_path "$name"
done
if [[ -n "$EXTRA_PYTHONPATH" ]]; then
  IFS=: read -r -a EXTRA_DIRS <<< "$EXTRA_PYTHONPATH"
  for i in "${!EXTRA_DIRS[@]}"; do
    extra_path="${EXTRA_DIRS[$i]}"
    bind_configured_path extra_path
    EXTRA_DIRS[$i]="$extra_path"
  done
  EXTRA_PYTHONPATH="$(IFS=:; echo "${EXTRA_DIRS[*]}")"
fi

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  BIND_ARGS+=(--bind "$SLURM_TMPDIR:$SLURM_TMPDIR")
fi

# ============================================================
# Execute container
# ============================================================

"$CTR_CMD" exec --nv \
  "${BIND_ARGS[@]}" \
  --env SLURM_TMPDIR="${SLURM_TMPDIR:-}" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_MAX_WORKERS=1 \
  --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TMPDIR="$TMPDIR" \
  --env TMP="$TMPDIR" \
  --env TEMP="$TMPDIR" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  --env CUDA_DEVICE_MAX_CONNECTIONS="$CUDA_DEVICE_MAX_CONNECTIONS" \
  --env TOKENIZERS_PARALLELISM=false \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING="$TORCH_NCCL_ASYNC_ERROR_HANDLING" \
  --env NCCL_DEBUG="$NCCL_DEBUG" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env WORKDIR="$WORKDIR" \
  --env SCRIPT="$SCRIPT" \
  --env DATA_PATH="$DATA_PATH" \
  --env MODEL="$MODEL" \
  --env TOKENIZER_ID="$TOKENIZER_ID" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env RUN_NAME="$RUN_NAME" \
  --env LOG_FILE="$LOG_FILE" \
  --env PKG_ENV="$PKG_ENV" \
  --env PKG_TRAIN="$PKG_TRAIN" \
  --env PKG_OPENAI="$PKG_OPENAI" \
  --env PKG_JSON="$PKG_JSON" \
  --env PKG_ST="$PKG_ST" \
  --env EXTRA_PYTHONPATH="$EXTRA_PYTHONPATH" \
  --env NNODES="$NNODES" \
  --env NODE_RANK="$NODE_RANK" \
  --env MASTER_ADDR="$MASTER_ADDR" \
  --env MASTER_PORT="$MASTER_PORT" \
  --env NPROC_PER_NODE="$NPROC_PER_NODE" \
  --env SEED="$SEED" \
  --env MAX_EXAMPLES="$MAX_EXAMPLES" \
  --env MAX_LEN="$MAX_LEN" \
  --env NUM_EPOCHS="$NUM_EPOCHS" \
  --env BATCH_SIZE="$BATCH_SIZE" \
  --env GRAD_ACCUM="$GRAD_ACCUM" \
  --env LR="$LR" \
  --env LOGGING_STEPS="$LOGGING_STEPS" \
  --env SAVE_STRATEGY="$SAVE_STRATEGY" \
  --env SAVE_STEPS="$SAVE_STEPS" \
  --env SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
  --env PACKING="$PACKING" \
  --env GRAD_CKPT="$GRAD_CKPT" \
  --env FORCE_GRAD_CKPT="$FORCE_GRAD_CKPT" \
  --env ATTN_IMPL="$ATTN_IMPL" \
  --env FSDP_WRAP_CLS="$FSDP_WRAP_CLS" \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail

    cd "$WORKDIR"

    # --------------------------------------------------------
    # Resolve training script
    # --------------------------------------------------------

    if [[ -f "$SCRIPT" ]]; then
      SCRIPT_PATH="$SCRIPT"
    elif [[ -f "$WORKDIR/$SCRIPT" ]]; then
      SCRIPT_PATH="$WORKDIR/$SCRIPT"
    else
      echo "[ERROR] Training script not found: $SCRIPT" >&2
      exit 11
    fi

    # --------------------------------------------------------
    # Python package paths
    # --------------------------------------------------------

    PY_PATHS=()

    [[ -n "${PKG_ENV:-}" && -d "$PKG_ENV" ]] && PY_PATHS+=("$PKG_ENV")
    [[ -n "${PKG_TRAIN:-}" && -d "$PKG_TRAIN" ]] && PY_PATHS+=("$PKG_TRAIN")
    [[ -n "${PKG_OPENAI:-}" && -d "$PKG_OPENAI" ]] && PY_PATHS+=("$PKG_OPENAI")
    [[ -n "${PKG_JSON:-}" && -d "$PKG_JSON" ]] && PY_PATHS+=("$PKG_JSON")
    [[ -n "${PKG_ST:-}" && -d "$PKG_ST" ]] && PY_PATHS+=("$PKG_ST")

    if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
      IFS=":" read -r -a EXTRA_ARR <<< "$EXTRA_PYTHONPATH"
      for p in "${EXTRA_ARR[@]}"; do
        [[ -n "$p" && -d "$p" ]] && PY_PATHS+=("$p")
      done
    fi

    export PYTHONPATH="$(IFS=:; echo "${PY_PATHS[*]}")"

    echo "[DEBUG] pwd=$PWD"
    echo "[DEBUG] PYTHONPATH=$PYTHONPATH"
    echo "[DEBUG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[DEBUG] SLURM_TMPDIR=${SLURM_TMPDIR:-}"

    # --------------------------------------------------------
    # Python environment sanity check
    # --------------------------------------------------------

    python - <<PY
import os
import torch

mods = [
    "transformers",
    "trl",
    "datasets",
    "accelerate",
    "peft",
    "tokenizers",
]

print("python ok")
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())

for m in mods:
    mod = __import__(m)
    print(
        f"{m:12s}",
        getattr(mod, "__version__", "unknown"),
        getattr(mod, "__file__", "<builtin>")
    )

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() >= int(os.environ.get("NPROC_PER_NODE", "1")), "Insufficient GPUs"

import transformers
assert transformers.__version__ == "4.51.0", (
    f"Expected transformers==4.51.0, got {transformers.__version__}"
)

print("[OK] Environment sanity passed")
PY

    # --------------------------------------------------------
    # Additional training arguments
    # --------------------------------------------------------

    EXTRA_ARGS=()

    [[ "${PACKING:-0}" == "1" ]] && EXTRA_ARGS+=(--packing)
    [[ "${GRAD_CKPT:-0}" == "1" ]] && EXTRA_ARGS+=(--grad_ckpt)
    [[ "${FORCE_GRAD_CKPT:-0}" == "1" ]] && EXTRA_ARGS+=(--force_grad_ckpt)

    # --------------------------------------------------------
    # Launch FSDP training
    # --------------------------------------------------------

    echo "[INFO] ===== Launch torchrun ====="

    torchrun \
      --nnodes="$NNODES" \
      --node_rank="$NODE_RANK" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      --nproc_per_node="$NPROC_PER_NODE" \
      "$SCRIPT_PATH" \
      --data_path "$DATA_PATH" \
      --model_id "$MODEL" \
      --tokenizer_id "$TOKENIZER_ID" \
      --output_dir "$OUTPUT_DIR" \
      --seed "$SEED" \
      --max_examples "$MAX_EXAMPLES" \
      --max_len "$MAX_LEN" \
      --num_epochs "$NUM_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --grad_accum "$GRAD_ACCUM" \
      --lr "$LR" \
      --logging_steps "$LOGGING_STEPS" \
      --save_strategy "$SAVE_STRATEGY" \
      --save_steps "$SAVE_STEPS" \
      --save_total_limit "$SAVE_TOTAL_LIMIT" \
      --attn_impl "$ATTN_IMPL" \
      --fsdp_wrap_cls "$FSDP_WRAP_CLS" \
      "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$LOG_FILE"
  '

echo "[INFO] Qwen3 Rerank R3 SFT finished successfully"
