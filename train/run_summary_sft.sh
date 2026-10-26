#!/usr/bin/env bash
# summary SFT FSDP launcher via Apptainer for Qwen3-8B
set -euo pipefail

# ============================================================
# Global paths
# ============================================================
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"
WORKDIR="${WORKDIR:-$RUNNER_DIR}"

# ============================================================
# Layered package dirs
# IMPORTANT:
# PKG_ENV must come before PKG_TRAIN so Qwen3 transformers==4.51.0 wins.
# ============================================================
PKG_ENV="${PKG_ENV:-}"
PKG_MARKER="${PKG_MARKER:-}"

PKG_TRAIN="${PKG_TRAIN:-}"
PKG_OPENAI="${PKG_OPENAI:-}"
PKG_JSON="${PKG_JSON:-}"
PKG_ST="${PKG_ST:-}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"

# ============================================================
# Model / data / script
# ============================================================
MODEL="${MODEL:?Set MODEL to the local Qwen3 model directory}"
TOKENIZER_ID="${TOKENIZER_ID:-$MODEL}"

DATA_PATH="${DATA_PATH:?Set DATA_PATH to the training data}"
SCRIPT="${SCRIPT:-$WORKDIR/train_summary_fsdp.py}"

OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-qwen3_summary_sft_${DATE_TAG}}"
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

# allocator / runtime stability
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TOKENIZERS_PARALLELISM=false

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
MASTER_PORT="${MASTER_PORT:-29617}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ============================================================
# Training hyperparameters
# Qwen3 summary SFT full run
# ============================================================
SEED="${SEED:-42}"
MAX_EXAMPLES="${MAX_EXAMPLES:--1}"

MAX_LEN="${MAX_LEN:-4096}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# Save about twice per epoch for 3,489 TRAIN samples at global batch 16:
# 3489 / (1 * 2 * 8) / 2 = 109.03 optimizer steps per half epoch.
# Trainer save_strategy=steps saves intermediate checkpoints; the Python script
# also performs its existing final trainer.save_model() after training.
# Recalculate SAVE_STEPS if train size, batch, world size or accumulation changes.
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-109}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

PACKING="${PACKING:-0}"

# For a multi-GPU FSDP setup, keep this off unless explicitly forced.
GRAD_CKPT="${GRAD_CKPT:-0}"
FORCE_GRAD_CKPT="${FORCE_GRAD_CKPT:-0}"

# Keep the existing eager attention default.
ATTN_IMPL="${ATTN_IMPL:-eager}"

# train_summary_fsdp.py can auto-detect this, but pass it explicitly for Qwen3.
FSDP_WRAP_CLS="${FSDP_WRAP_CLS:-Qwen3DecoderLayer}"

# ============================================================
# Environment modules
# ============================================================
# Load the container runtime in the caller environment.

if command -v apptainer >/dev/null 2>&1; then
  CTR_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CTR_CMD="singularity"
else
  echo "❌ Neither apptainer nor singularity found in PATH" >&2
  exit 99
fi

# ============================================================
# Sanity checks
# ============================================================
[[ -f "$IMAGE" ]] || { echo "❌ Missing image: $IMAGE" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { echo "❌ Missing WORKDIR: $WORKDIR" >&2; exit 3; }
[[ -f "$DATA_PATH" || -d "$DATA_PATH" ]] || { echo "❌ Missing DATA_PATH: $DATA_PATH" >&2; exit 4; }
[[ -d "$MODEL" ]] || { echo "❌ Missing MODEL dir: $MODEL" >&2; exit 5; }
[[ -f "$WORKDIR/$SCRIPT" || -f "$SCRIPT" ]] || { echo "❌ Missing SCRIPT: $SCRIPT" >&2; exit 6; }
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "❌ Missing PKG_ENV: $PKG_ENV" >&2; exit 7; }
[[ -z "$PKG_MARKER" || -f "$PKG_MARKER" ]] || { echo "❌ Missing PKG marker: $PKG_MARKER" >&2; exit 8; }

if [[ -n "$PKG_TRAIN" && ! -d "$PKG_TRAIN" ]]; then
  echo "❌ Missing PKG_TRAIN: $PKG_TRAIN" >&2
  exit 9
fi

if [[ -n "$TOKENIZER_ID" && ! -d "$TOKENIZER_ID" ]]; then
  echo "❌ TOKENIZER_ID was provided but not found as a local dir: $TOKENIZER_ID" >&2
  exit 10
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"
GLOBAL_BATCH=$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))

echo "[INFO] Container cmd        : $CTR_CMD"
echo "[INFO] Image                : $IMAGE"
echo "[INFO] Workdir              : $WORKDIR"
echo "[INFO] Script               : $SCRIPT"
echo "[INFO] Data path            : $DATA_PATH"
echo "[INFO] Model                : $MODEL"
echo "[INFO] Tokenizer            : ${TOKENIZER_ID:-<use model_id>}"
echo "[INFO] PKG_ENV              : $PKG_ENV"
echo "[INFO] PKG_TRAIN            : $PKG_TRAIN"
echo "[INFO] Run name             : $RUN_NAME"
echo "[INFO] Output dir           : $OUTPUT_DIR"
echo "[INFO] Log file             : $LOG_FILE"
echo "[INFO] CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "[INFO] NPROC_PER_NODE       : $NPROC_PER_NODE"
echo "[INFO] MAX_LEN              : $MAX_LEN"
echo "[INFO] NUM_EPOCHS           : $NUM_EPOCHS"
echo "[INFO] BATCH_SIZE           : $BATCH_SIZE"
echo "[INFO] GRAD_ACCUM           : $GRAD_ACCUM"
echo "[INFO] GLOBAL_BATCH         : $GLOBAL_BATCH"
echo "[INFO] LR                   : $LR"
echo "[INFO] LOGGING_STEPS        : $LOGGING_STEPS"
echo "[INFO] SAVE_STRATEGY        : $SAVE_STRATEGY"
echo "[INFO] SAVE_TOTAL_LIMIT     : $SAVE_TOTAL_LIMIT"
echo "[INFO] GRAD_CKPT            : $GRAD_CKPT"
echo "[INFO] FORCE_GRAD_CKPT      : $FORCE_GRAD_CKPT"
echo "[INFO] ATTN_IMPL            : $ATTN_IMPL"
echo "[INFO] FSDP_WRAP_CLS        : $FSDP_WRAP_CLS"
echo "[INFO] MAX_EXAMPLES         : $MAX_EXAMPLES"
echo "[INFO] MASTER_PORT          : $MASTER_PORT"

# Use the runtime's default certificate configuration.

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
  --env NCCL_DEBUG="$NCCL_DEBUG" \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING="$TORCH_NCCL_ASYNC_ERROR_HANDLING" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
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

    if [[ -f "$SCRIPT" ]]; then
      SCRIPT_PATH="$SCRIPT"
    elif [[ -f "$WORKDIR/$SCRIPT" ]]; then
      SCRIPT_PATH="$WORKDIR/$SCRIPT"
    else
      echo "[ERROR] Training script not found: $SCRIPT" >&2
      exit 11
    fi

    PY_PATHS=()
    [[ -n "${PKG_ENV:-}"    && -d "$PKG_ENV"    ]] && PY_PATHS+=("$PKG_ENV")
    [[ -n "${PKG_TRAIN:-}"  && -d "$PKG_TRAIN"  ]] && PY_PATHS+=("$PKG_TRAIN")
    [[ -n "${PKG_OPENAI:-}" && -d "$PKG_OPENAI" ]] && PY_PATHS+=("$PKG_OPENAI")
    [[ -n "${PKG_JSON:-}"   && -d "$PKG_JSON"   ]] && PY_PATHS+=("$PKG_JSON")
    [[ -n "${PKG_ST:-}"     && -d "$PKG_ST"     ]] && PY_PATHS+=("$PKG_ST")

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

    python - <<PY
import os
import torch

mods = ["transformers", "trl", "datasets", "accelerate", "peft", "tokenizers"]

print("python ok")
print("cuda available:", torch.cuda.is_available())
print("cuda count    :", torch.cuda.device_count())

for m in mods:
    mod = __import__(m)
    print(f"{m:12s}", getattr(mod, "__version__", "unknown"), getattr(mod, "__file__", "<builtin>"))

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() >= int(os.environ.get("NPROC_PER_NODE", "1")), "Not enough visible GPUs"

import transformers
assert transformers.__version__ == "4.51.0", f"Expected transformers==4.51.0, got {transformers.__version__}"

print("[OK] sanity passed")
PY

    EXTRA_ARGS=()
    [[ "${PACKING:-0}" == "1" ]] && EXTRA_ARGS+=(--packing)
    [[ "${GRAD_CKPT:-0}" == "1" ]] && EXTRA_ARGS+=(--grad_ckpt)
    [[ "${FORCE_GRAD_CKPT:-0}" == "1" ]] && EXTRA_ARGS+=(--force_grad_ckpt)
    [[ -n "${TOKENIZER_ID:-}" ]] && EXTRA_ARGS+=(--tokenizer_id "$TOKENIZER_ID")

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

echo "[INFO] ✅ Qwen3 summary SFT finished"
