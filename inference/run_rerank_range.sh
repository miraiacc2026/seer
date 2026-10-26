#!/usr/bin/env bash
set -euo pipefail

# The host Python is used only for wall-clock timestamps and JSON metrics.
HOST_PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$HOST_PYTHON" ]] || { echo "Host python3/python is required for timing." >&2; exit 98; }
SHELL_STARTED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"


RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"
WORKDIR="${WORKDIR:-$RUNNER_DIR}"

# Qwen3 env
PKG_ENV="${PKG_ENV:-}"
PKG_MARKER="${PKG_MARKER:-}"

START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-947}"                                   # [WWW] ASQA dev = 948
IDX_LIST_FILE="${IDX_LIST_FILE:-}"

# Pipeline default paths
BENCH_ROOT="${BENCH_ROOT:-}"
CANDIDATE_DIR="${CANDIDATE_DIR:-}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to the rerank output directory}"

# New raw-passage mode.
# USE_RAW_PASSAGES=0: original behavior, read candidate_pool_{idx}.json from CANDIDATE_DIR.
# USE_RAW_PASSAGES=1: read SRC_JSON, take top MAX_INPUT_PASSAGES docs, split sentences, then rerank.
USE_RAW_PASSAGES="${USE_RAW_PASSAGES:-0}"
SRC_JSON="${SRC_JSON:-}"
if [[ "$USE_RAW_PASSAGES" == "1" ]]; then
  : "${SRC_JSON:?Set SRC_JSON for raw-passage mode}"
else
  : "${CANDIDATE_DIR:?Set CANDIDATE_DIR for candidate-pool mode}"
fi
MAX_INPUT_PASSAGES="${MAX_INPUT_PASSAGES:-5}"
RAW_QSTRUCT_DIR="${RAW_QSTRUCT_DIR:-}"
MIN_SENT_CHARS="${MIN_SENT_CHARS:-20}"

BACKEND="${BACKEND:-local}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.2}"

# Select the trained reranker, or a vanilla Qwen3 model for A7, explicitly.
MODEL="${MODEL:-}"
BASE_TOKENIZER="${BASE_TOKENIZER:-$MODEL}"

# [WWW] One prompt per query, matching the R3 teacher data used for SFT.
RERANK_BATCH="${RERANK_BATCH:-18}"
# Number of rerank prompts generated together on the GPU.
RERANK_INFER_BATCH="${RERANK_INFER_BATCH:-3}"
# structured: use the stored query structure; original: use only the raw query.
QUERY_MODE="${QUERY_MODE:-original}"                        # [WWW] A4 default; A1 uses structured
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2600}"              # [WWW] 18 labels per prompt

# In candidate-pool mode, this truncates candidate_pool.
# In raw-passage mode:
#   MAX_INPUT_CANDIDATES=0 means use all sentences from top MAX_INPUT_PASSAGES docs.
#   MAX_INPUT_CANDIDATES>0 means use only first N sentences.
MAX_INPUT_CANDIDATES="${MAX_INPUT_CANDIDATES:-18}"

TOPK_CORE="${TOPK_CORE:-5}"
TOPK_SUPPORT="${TOPK_SUPPORT:-8}"
MAX_PACKED_SENTS="${MAX_PACKED_SENTS:-14}"
# [WWW] Only effective if rerank_topk_evidence.py reads it (R3 pack_final_evidence).
MIN_PACKED_SENTS="${MIN_PACKED_SENTS:-3}"

DISABLE_QWEN_THINKING="${DISABLE_QWEN_THINKING:-1}"
ATTN_IMPL="${ATTN_IMPL:-eager}"

SKIP_EXISTING="${SKIP_EXISTING:-0}"

SCRIPT="${SCRIPT:-$WORKDIR/rerank_range.py}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# [WWW] Fail fast on settings that silently change the experiment.
if [[ "$BACKEND" == "local" && -z "$MODEL" ]]; then
  echo "❌ MODEL is required (new rerank checkpoint, or Qwen3-8B for A7)." >&2
  exit 13
fi
if [[ "$BACKEND" == "openai" ]]; then
  : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to the API endpoint}"
fi
if [[ "$USE_RAW_PASSAGES" != "1" && "$RERANK_BATCH" -lt "$MAX_INPUT_CANDIDATES" ]]; then
  echo "[WARN] RERANK_BATCH=$RERANK_BATCH < MAX_INPUT_CANDIDATES=$MAX_INPUT_CANDIDATES:" \
       "candidates will be split into several prompts, unlike the SFT data." >&2
fi

# All result JSON, logs and metrics share this output root.
# RUN_ID is unique by default; explicit reuse is rejected to avoid mixed metrics.
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
RUN_ID="${RUN_ID:-$("$HOST_PYTHON" -c 'from datetime import datetime, timezone; import uuid; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])')}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid RUN_ID: use letters, digits, _, . or -." >&2
  exit 11
fi
RUN_DIR="$OUT_DIR/metrics/$RUN_ID"
mkdir -p "$OUT_DIR/metrics"
mkdir "$RUN_DIR" || { echo "RUN_ID already exists: $RUN_DIR" >&2; exit 12; }
export RUN_ID OUT_DIR
exec 3>&1 4>&2
LOGGER_PID=""
LOG_PIPE="$RUN_DIR/.rerank_range_log.pipe"

write_shell_summary() {
  local exit_code=$?
  trap - EXIT
  set +e
  "$HOST_PYTHON" - "$RUN_DIR" "$exit_code" "$SHELL_STARTED_EPOCH" \
    "${CONTAINER_STARTED_EPOCH:-}" "${CONTAINER_FINISHED_EPOCH:-}" \
    "${CONTAINER_EXIT_CODE:-}" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
exit_code = int(sys.argv[2])
started = float(sys.argv[3])
finished = time.time()
container_start = float(sys.argv[4]) if sys.argv[4] else None
container_end = float(sys.argv[5]) if sys.argv[5] else finished
container_code = int(sys.argv[6]) if sys.argv[6] else None
summary = {
    "run_id": run_dir.name,
    "stage": "rerank",
    "status": "completed" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "container_exit_code": container_code,
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "out_dir": str(run_dir.parent.parent),
    "latency_seconds": {
        "shell_total_seconds": finished - started,
        "setup_seconds": (container_start if container_start is not None else finished) - started,
        "container_seconds": container_end - container_start if container_start is not None else None,
    },
    "python_summary": str(run_dir / "rerank_range_summary.json"),
    "log": str(run_dir / "rerank_range.log"),
    "measurement_notes": [
        "Shell total includes setup, modules, container startup, sanity checks and Python execution.",
        "Container latency includes startup, sanity checks and Python execution.",
        "Nested times overlap; do not add shell, container and Python stage times together.",
        "Timing stops before shell-summary writing and shell teardown.",
        "Shell timings use wall-clock epoch seconds; Python timings use a monotonic clock.",
    ],
}
path = run_dir / "rerank_range_summary.json"
if path.exists():
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        summary["python_status"] = result.get("status")
        for key in ("input_tokens", "output_tokens", "total_tokens", "token_counts_complete"):
            summary[key] = result.get(key)
    except (OSError, ValueError) as exc:
        summary["python_summary_read_error"] = str(exc)
destination = run_dir / "rerank_range_shell_summary.json"
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"[METRICS] shell_total_seconds={finished - started:.6f} exit_code={exit_code}")
print(f"[METRICS] {destination}")
PY
  local metrics_exit_code=$?
  if [[ "$metrics_exit_code" -ne 0 ]]; then
    echo "[WARN] Could not save shell summary (status=$metrics_exit_code)." >&2
  fi
  exec 1>&3 2>&4
  if [[ -n "$LOGGER_PID" ]]; then
    wait "$LOGGER_PID" || echo "[WARN] Log writer failed." >&2
  fi
  rm -f "$LOG_PIPE"
  exec 3>&- 4>&-
  exit "$exit_code"
}
trap write_shell_summary EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Stream to console and log without relying on /dev/fd process substitution.
mkfifo "$LOG_PIPE"
tee "$RUN_DIR/rerank_range.log" <"$LOG_PIPE" >&3 &
LOGGER_PID=$!
exec >"$LOG_PIPE" 2>&1
rm -f "$LOG_PIPE"
echo "[INFO] RUN_ID             : $RUN_ID"
echo "[INFO] Metrics directory  : $RUN_DIR"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MYPROJ/pip_cache_qwen3}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MYPROJ/xdg_cache_qwen3}"
export HF_HOME="${HF_HOME:-$MYPROJ/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MYPROJ/hf_datasets_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MYPROJ/triton_cache}"
export TMPDIR="${TMPDIR:-$MYPROJ/tmp}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$OUT_DIR" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR"

# Load the container runtime in the caller environment.

if command -v apptainer >/dev/null 2>&1; then
  CTR_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CTR_CMD="singularity"
else
  echo "❌ Neither apptainer nor singularity found in PATH" >&2
  exit 99
fi

[[ -f "$IMAGE" ]] || { echo "❌ Missing image: $IMAGE" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { echo "❌ Missing WORKDIR: $WORKDIR" >&2; exit 3; }
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "❌ Missing PKG_ENV: $PKG_ENV" >&2; exit 4; }
[[ -z "$PKG_MARKER" || -f "$PKG_MARKER" ]] || { echo "❌ Missing PKG marker: $PKG_MARKER" >&2; exit 5; }
[[ -f "$SCRIPT" ]] || { echo "❌ Missing SCRIPT: $SCRIPT" >&2; exit 7; }

if [[ "$USE_RAW_PASSAGES" == "1" ]]; then
  [[ -f "$SRC_JSON" ]] || { echo "❌ USE_RAW_PASSAGES=1 but missing SRC_JSON: $SRC_JSON" >&2; exit 6; }
else
  [[ -d "$CANDIDATE_DIR" ]] || { echo "❌ Missing CANDIDATE_DIR: $CANDIDATE_DIR" >&2; exit 6; }
fi

if [[ -n "$IDX_LIST_FILE" ]]; then
  [[ -f "$IDX_LIST_FILE" ]] || { echo "❌ Missing IDX_LIST_FILE: $IDX_LIST_FILE" >&2; exit 10; }
fi

if [[ "$BACKEND" == "local" ]]; then
  [[ -d "$MODEL" ]] || { echo "❌ Missing MODEL dir: $MODEL" >&2; exit 8; }
  [[ -d "$BASE_TOKENIZER" ]] || { echo "❌ Missing BASE_TOKENIZER dir: $BASE_TOKENIZER" >&2; exit 9; }
fi

if [[ "$BACKEND" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY is required when BACKEND=openai" >&2
  exit 11
fi

SOURCE_ROOT="$(cd -- "$(dirname -- "$SCRIPT")/.." && pwd -P)"
BIND_PATHS=()
bind_configured_path() {
  local name="$1" path="${!1:-}" directory
  [[ -n "$path" ]] || return 0
  [[ "$path" == /* ]] || path="$PWD/$path"
  printf -v "$name" '%s' "$path"
  directory="$path"
  while [[ ! -d "$directory" ]]; do directory="$(dirname -- "$directory")"; done
  BIND_PATHS+=(--bind "$directory:$directory")
}
for name in SOURCE_ROOT WORKDIR SCRIPT PKG_ENV OUT_DIR IDX_LIST_FILE RAW_QSTRUCT_DIR \
  PIP_CACHE_DIR XDG_CACHE_HOME HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE \
  TRITON_CACHE_DIR TMPDIR; do
  bind_configured_path "$name"
done
if [[ "$USE_RAW_PASSAGES" == "1" ]]; then
  bind_configured_path SRC_JSON
else
  bind_configured_path CANDIDATE_DIR
fi
if [[ "$BACKEND" == "local" ]]; then
  bind_configured_path MODEL
  bind_configured_path BASE_TOKENIZER
fi
if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
  BIND_PATHS+=( --bind "$SLURM_TMPDIR:$SLURM_TMPDIR" --env SLURM_TMPDIR="$SLURM_TMPDIR" )
fi

echo "[INFO] IMAGE=$IMAGE"
echo "[INFO] WORKDIR=$WORKDIR"
echo "[INFO] PKG_ENV=$PKG_ENV"
echo "[INFO] SCRIPT=$SCRIPT"
echo "[INFO] START_IDX=$START_IDX"
echo "[INFO] END_IDX=$END_IDX"
echo "[INFO] IDX_LIST_FILE=${IDX_LIST_FILE:-<range mode>}"
echo "[INFO] BENCH_ROOT=$BENCH_ROOT"
echo "[INFO] CANDIDATE_DIR=$CANDIDATE_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] USE_RAW_PASSAGES=$USE_RAW_PASSAGES"
echo "[INFO] SRC_JSON=${SRC_JSON:-<empty>}"
echo "[INFO] MAX_INPUT_PASSAGES=$MAX_INPUT_PASSAGES"
echo "[INFO] RAW_QSTRUCT_DIR=${RAW_QSTRUCT_DIR:-<empty>}"
echo "[INFO] MIN_SENT_CHARS=$MIN_SENT_CHARS"
echo "[INFO] BACKEND=$BACKEND"
echo "[INFO] OPENAI_MODEL=$OPENAI_MODEL"
echo "[INFO] MODEL=$MODEL"
echo "[INFO] BASE_TOKENIZER=$BASE_TOKENIZER"
echo "[INFO] RERANK_BATCH=$RERANK_BATCH"
echo "[INFO] RERANK_INFER_BATCH=$RERANK_INFER_BATCH"
echo "[INFO] QUERY_MODE=$QUERY_MODE"
echo "[INFO] MAX_OUTPUT_TOKENS=$MAX_OUTPUT_TOKENS"
echo "[INFO] MAX_INPUT_CANDIDATES=$MAX_INPUT_CANDIDATES"
echo "[INFO] TOPK_CORE=$TOPK_CORE"
echo "[INFO] TOPK_SUPPORT=$TOPK_SUPPORT"
echo "[INFO] MAX_PACKED_SENTS=$MAX_PACKED_SENTS"
echo "[INFO] MIN_PACKED_SENTS=$MIN_PACKED_SENTS"
echo "[INFO] DISABLE_QWEN_THINKING=$DISABLE_QWEN_THINKING"
echo "[INFO] ATTN_IMPL=$ATTN_IMPL"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_TMPDIR=${SLURM_TMPDIR:-<empty>}"

# Use the runtime's default certificate configuration.

CONTAINER_STARTED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"
set +e
"$CTR_CMD" exec --nv \
  "${BIND_PATHS[@]}" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TMPDIR="$TMPDIR" \
  --env TMP="$TMPDIR" \
  --env TEMP="$TMPDIR" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_MAX_WORKERS=1 \
  --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env WORKDIR="$WORKDIR" \
  --env PKG_ENV="$PKG_ENV" \
  --env SCRIPT="$SCRIPT" \
  --env START_IDX="$START_IDX" \
  --env END_IDX="$END_IDX" \
  --env IDX_LIST_FILE="$IDX_LIST_FILE" \
  --env BENCH_ROOT="$BENCH_ROOT" \
  --env CANDIDATE_DIR="$CANDIDATE_DIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env RUN_ID="$RUN_ID" \
  --env SKIP_EXISTING="$SKIP_EXISTING" \
  --env USE_RAW_PASSAGES="$USE_RAW_PASSAGES" \
  --env SRC_JSON="$SRC_JSON" \
  --env MAX_INPUT_PASSAGES="$MAX_INPUT_PASSAGES" \
  --env RAW_QSTRUCT_DIR="$RAW_QSTRUCT_DIR" \
  --env MIN_SENT_CHARS="$MIN_SENT_CHARS" \
  --env BACKEND="$BACKEND" \
  --env OPENAI_MODEL="$OPENAI_MODEL" \
  --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --env OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  --env MODEL="$MODEL" \
  --env BASE_TOKENIZER="$BASE_TOKENIZER" \
  --env RERANK_BATCH="$RERANK_BATCH" \
  --env RERANK_INFER_BATCH="$RERANK_INFER_BATCH" \
  --env QUERY_MODE="$QUERY_MODE" \
  --env MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
  --env MAX_INPUT_CANDIDATES="$MAX_INPUT_CANDIDATES" \
  --env TOPK_CORE="$TOPK_CORE" \
  --env TOPK_SUPPORT="$TOPK_SUPPORT" \
  --env MAX_PACKED_SENTS="$MAX_PACKED_SENTS" \
  --env MIN_PACKED_SENTS="$MIN_PACKED_SENTS" \
  --env DISABLE_QWEN_THINKING="$DISABLE_QWEN_THINKING" \
  --env ATTN_IMPL="$ATTN_IMPL" \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$WORKDIR"

    unset PYTHONSTARTUP PYTHONHOME PYTHONPATH
    export PYTHONPATH="$PKG_ENV"

    echo "[INFO] inside container pwd=$(pwd)"
    echo "[INFO] inside container script=$SCRIPT"
    echo "[INFO] inside container pythonpath=$PYTHONPATH"
    echo "[INFO] inside container CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[INFO] inside container IDX_LIST_FILE=${IDX_LIST_FILE:-<range mode>}"
    echo "[INFO] inside container USE_RAW_PASSAGES=$USE_RAW_PASSAGES"
    echo "[INFO] inside container SRC_JSON=${SRC_JSON:-<empty>}"

    python - <<PY
import os
import torch
import transformers
import tokenizers

print("python ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
print("transformers:", transformers.__version__, transformers.__file__)
print("tokenizers:", tokenizers.__version__, tokenizers.__file__)
print("MODEL:", os.environ.get("MODEL"))
print("BASE_TOKENIZER:", os.environ.get("BASE_TOKENIZER"))
print("PKG_ENV:", os.environ.get("PKG_ENV"))
print("QUERY_MODE:", os.environ.get("QUERY_MODE"))
print("RERANK_BATCH:", os.environ.get("RERANK_BATCH"))
print("DISABLE_QWEN_THINKING:", os.environ.get("DISABLE_QWEN_THINKING"))
print("USE_RAW_PASSAGES:", os.environ.get("USE_RAW_PASSAGES"))
print("SRC_JSON:", os.environ.get("SRC_JSON"))
print("MAX_INPUT_PASSAGES:", os.environ.get("MAX_INPUT_PASSAGES"))
print("MAX_INPUT_CANDIDATES:", os.environ.get("MAX_INPUT_CANDIDATES"))
print("RERANK_INFER_BATCH:", os.environ.get("RERANK_INFER_BATCH"))
print("IDX_LIST_FILE:", os.environ.get("IDX_LIST_FILE"))
try:
    import pysbd
    print("pysbd: ok", pysbd.__file__)
except Exception as e:
    print("pysbd: missing/fallback", repr(e))
PY

    python -u "$SCRIPT"
  '

CONTAINER_EXIT_CODE=$?
CONTAINER_FINISHED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"
set -e
if [[ "$CONTAINER_EXIT_CODE" -eq 0 ]]; then
  echo "[OK] rerank range finished: $OUT_DIR"
fi
exit "$CONTAINER_EXIT_CODE"
