#!/usr/bin/env bash
set -euo pipefail

# Canonical Apptainer/Qwen3 package environment, from reference launcher.
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"

BATCHDIR="${BATCHDIR:-$RUNNER_DIR}"

# Single Qwen3 package environment: no legacy per-library package directories.
PKG_ENV="${PKG_ENV:-}"
PKG_MARKER="${PKG_MARKER:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
[[ -d "$BATCHDIR" ]] || { echo "❌ Missing BATCHDIR: $BATCHDIR" >&2; exit 3; }
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "❌ Missing PKG_ENV: $PKG_ENV" >&2; exit 5; }
[[ -z "$PKG_MARKER" || -f "$PKG_MARKER" ]] || { echo "❌ Missing PKG marker: $PKG_MARKER" >&2; exit 6; }

# Teacher-data Pipeline: original mode consumes TopK original query only; structured mode also uses Query Structure.
TEST_TEACHER_ROOT="${TEST_TEACHER_ROOT:-}"
QUERY_STRUCT_DIR="${QUERY_STRUCT_DIR:-}"
TOPK_DIR="${TOPK_DIR:?Set TOPK_DIR to the evidence input directory}"
LOCAL_STRUCTURE_DIR="${LOCAL_STRUCTURE_DIR:-}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:?Set SUMMARY_OUT_DIR to the summary output directory}"
BATCH_WORKDIR="${BATCH_WORKDIR:-$SUMMARY_OUT_DIR/jobs}"
TSV_PATH="${TSV_PATH:-$BATCH_WORKDIR/latest_batches.tsv}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
# Configure Chat and native Batch endpoints independently.
PROXY_BASE_URL="${PROXY_BASE_URL:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
API_SUPPORTS_BATCH="${API_SUPPORTS_BATCH:-0}"
case "$API_SUPPORTS_BATCH" in
  0|1) ;;
  *) echo "[ERROR] API_SUPPORTS_BATCH must be 0 or 1" >&2; exit 2 ;;
esac
CHAT_WORKERS="${CHAT_WORKERS:-4}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-2}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1200}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-24h}"
QUERY_MODE="${QUERY_MODE:-original}"
USE_PLAN="${USE_PLAN:-1}"
MAX_CORE="${MAX_CORE:-5}"
MAX_SUPPORT="${MAX_SUPPORT:-8}"
MAX_PACKED="${MAX_PACKED:-14}"
# Existing summary files may belong to a previous structured run.
# Set OVERWRITE=1 explicitly to replace them during consume_tsv/consume_local.
OVERWRITE="${OVERWRITE:-0}"
if [[ "$QUERY_MODE" != "original" && "$QUERY_MODE" != "structured" ]]; then
  echo "[ERROR] QUERY_MODE must be original or structured: $QUERY_MODE" >&2
  exit 2
fi
if [[ "$QUERY_MODE" == "structured" && "${1:-}" != "single" ]]; then
  : "${QUERY_STRUCT_DIR:?Set QUERY_STRUCT_DIR for structured mode}"
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "[ERROR] OVERWRITE must be 0 or 1: $OVERWRITE" >&2
  exit 2
fi

# single-mode inputs must cross the host/container boundary explicitly.
QUERY_STRUCT="${QUERY_STRUCT:-}"
TOPK_JSON="${TOPK_JSON:-}"
LOCAL_STRUCTURE="${LOCAL_STRUCTURE:-}"
OUT_JSON="${OUT_JSON:-}"

PYFILE="${PYFILE:-$BATCHDIR/asqa_summary_batch.py}"
mkdir -p "$BATCH_WORKDIR" "$SUMMARY_OUT_DIR"

if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
  BIND_TMP=( --bind "$SLURM_TMPDIR:$SLURM_TMPDIR" --env SLURM_TMPDIR="$SLURM_TMPDIR" )
else
  BIND_TMP=()
fi

CMD="${1:-}"
shift || true

echo "[INFO] Container cmd       : $CTR_CMD"
echo "[INFO] Image               : $IMAGE"
echo "[INFO] PKG_ENV             : $PKG_ENV"
echo "[INFO] PYFILE              : $PYFILE"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_TMPDIR        : ${SLURM_TMPDIR:-<empty>}"
echo "[INFO] BATCHDIR            : $BATCHDIR"
echo "[INFO] TEST_TEACHER_ROOT   : $TEST_TEACHER_ROOT"
echo "[INFO] QUERY_STRUCT_DIR    : $QUERY_STRUCT_DIR"
echo "[INFO] TOPK_DIR            : $TOPK_DIR"
echo "[INFO] LOCAL_STRUCTURE_DIR : ${LOCAL_STRUCTURE_DIR:-<none>}"
echo "[INFO] SUMMARY_OUT_DIR     : $SUMMARY_OUT_DIR"
echo "[INFO] BATCH_WORKDIR       : $BATCH_WORKDIR"
echo "[INFO] CMD                 : $CMD"
echo "[INFO] OPENAI_MODEL        : $OPENAI_MODEL"
echo "[INFO] PROXY_BASE_URL     : $PROXY_BASE_URL"
echo "[INFO] CHAT_WORKERS        : $CHAT_WORKERS"
echo "[INFO] MAX_OUTPUT_TOKENS   : $MAX_OUTPUT_TOKENS"
echo "[INFO] COMPLETION_WINDOW   : $COMPLETION_WINDOW"
echo "[INFO] QUERY_MODE=$QUERY_MODE USE_PLAN=$USE_PLAN MAX_CORE=$MAX_CORE MAX_SUPPORT=$MAX_SUPPORT MAX_PACKED=$MAX_PACKED OVERWRITE=$OVERWRITE"
echo "[INFO] START_IDX=${START_IDX:-} END_IDX=${END_IDX:-} CHUNK_SIZE=${CHUNK_SIZE:-}"
echo "[INFO] JOB_NAME=${JOB_NAME:-} BATCH_ID=${BATCH_ID:-} PATTERN=${PATTERN:-} TSV_PATH=${TSV_PATH:-}"

# Use the runtime's default certificate configuration.
SOURCE_ROOT="$(cd -- "$(dirname -- "$PYFILE")/.." && pwd -P)"
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
for name in SOURCE_ROOT BATCHDIR PKG_ENV PYFILE QUERY_STRUCT_DIR TOPK_DIR LOCAL_STRUCTURE_DIR \
  SUMMARY_OUT_DIR BATCH_WORKDIR TSV_PATH QUERY_STRUCT TOPK_JSON LOCAL_STRUCTURE \
  OUT_JSON RAW_OUT_JSONL FAILED_IDS_OUT PIP_CACHE_DIR XDG_CACHE_HOME HF_HOME \
  HF_HUB_CACHE HF_DATASETS_CACHE TRITON_CACHE_DIR TMPDIR; do
  bind_configured_path "$name"
done

"$CTR_CMD" exec --nv \
  "${BIND_ARGS[@]}" \
  ${BIND_TMP[@]+"${BIND_TMP[@]}"} \
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
  --env PKG_ENV="$PKG_ENV" \
  --env BATCHDIR="$BATCHDIR" \
  --env TEST_TEACHER_ROOT="$TEST_TEACHER_ROOT" \
  --env PYFILE="$PYFILE" \
  --env QUERY_STRUCT_DIR="$QUERY_STRUCT_DIR" \
  --env TOPK_DIR="$TOPK_DIR" \
  --env LOCAL_STRUCTURE_DIR="$LOCAL_STRUCTURE_DIR" \
  --env SUMMARY_OUT_DIR="$SUMMARY_OUT_DIR" \
  --env BATCH_WORKDIR="$BATCH_WORKDIR" \
  --env QUERY_MODE="$QUERY_MODE" \
  --env OVERWRITE="$OVERWRITE" \
  --env USE_PLAN="$USE_PLAN" \
  --env MAX_CORE="$MAX_CORE" \
  --env MAX_SUPPORT="$MAX_SUPPORT" \
  --env MAX_PACKED="$MAX_PACKED" \
  --env CMD="$CMD" \
  --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --env PROXY_API_KEY="${PROXY_API_KEY:-}" \
  --env PROXY_BASE_URL="$PROXY_BASE_URL" \
  --env OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  --env API_SUPPORTS_BATCH="$API_SUPPORTS_BATCH" \
  --env CHAT_WORKERS="$CHAT_WORKERS" \
  --env OPENAI_TIMEOUT="$OPENAI_TIMEOUT" \
  --env OPENAI_MAX_RETRIES="$OPENAI_MAX_RETRIES" \
  --env OPENAI_MODEL="$OPENAI_MODEL" \
  --env MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
  --env COMPLETION_WINDOW="$COMPLETION_WINDOW" \
  --env START_IDX="${START_IDX:-}" \
  --env END_IDX="${END_IDX:-}" \
  --env CHUNK_SIZE="${CHUNK_SIZE:-}" \
  --env JOB_NAME="${JOB_NAME:-}" \
  --env TSV_PATH="${TSV_PATH:-}" \
  --env RAW_OUT_JSONL="${RAW_OUT_JSONL:-}" \
  --env FAILED_IDS_OUT="${FAILED_IDS_OUT:-}" \
  --env BATCH_ID="${BATCH_ID:-}" \
  --env PATTERN="${PATTERN:-}" \
  --env QUERY_STRUCT="$QUERY_STRUCT" \
  --env TOPK_JSON="$TOPK_JSON" \
  --env LOCAL_STRUCTURE="$LOCAL_STRUCTURE" \
  --env OUT_JSON="$OUT_JSON" \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$BATCHDIR"

    # Use only the Qwen3 package path; do not inherit legacy/site Python paths.
    unset PYTHONSTARTUP PYTHONHOME PYTHONPATH
    export PYTHONPATH="$PKG_ENV"

    [[ -f "$PYFILE" ]] || { echo "[ERROR] Missing python file: $PYFILE" >&2; exit 12; }

    echo "[INFO] inside container pwd=$(pwd)"
    echo "[INFO] inside container PYTHONPATH=$PYTHONPATH"
    echo "[INFO] inside container PYFILE=$PYFILE"
    echo "[INFO] inside container CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    python - <<PY
import importlib.util
import os
import sys
print("python:", sys.version.split()[0], sys.executable)
for pkg in ("openai", "torch", "transformers", "tokenizers"):
    spec = importlib.util.find_spec(pkg)
    print(pkg + ":", spec.origin if spec is not None else "MISSING")
print("PKG_ENV:", os.environ["PKG_ENV"])
PY

    # Native Batch is opt-in and independent of the endpoint hostname.
    case "$CMD" in
      submit_split|status|status_from_tsv|status_tsv|consume|consume_from_tsv|consume_tsv)
        if [[ "$API_SUPPORTS_BATCH" != "1" ]]; then
          echo "[ERROR] Native Batch is disabled; use run_chat or set API_SUPPORTS_BATCH=1 for a capable endpoint." >&2
          exit 21
        fi
        : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to the native Batch API endpoint}"
        [[ -n "${OPENAI_API_KEY:-}" ]] || {
          echo "[ERROR] OPENAI_API_KEY required for native Batch commands" >&2; exit 10;
        }
        ;;
      run_chat|single)
        : "${PROXY_BASE_URL:?Set PROXY_BASE_URL to the Chat API endpoint}"
        [[ -n "${PROXY_API_KEY:-}" ]] || {
          echo "[ERROR] PROXY_API_KEY is required for $CMD" >&2; exit 10;
        }
        ;;
    esac

    case "$CMD" in
      run_chat)
        START_IDX="${START_IDX:-0}"
        END_IDX="${END_IDX:-10}"
        JOB_NAME="${JOB_NAME:-asqa_train_summary_s3_${START_IDX}_${END_IDX}}"
        [[ -d "$TOPK_DIR" ]] || { echo "[ERROR] TOPK_DIR missing: $TOPK_DIR" >&2; exit 13; }
        overwrite_flags=()
        [[ "$OVERWRITE" == "1" ]] && overwrite_flags+=(--overwrite)
        python "$PYFILE" run_chat \
          --query_struct_dir "$QUERY_STRUCT_DIR" \
          --topk_dir "$TOPK_DIR" \
          --jobs_dir "$BATCH_WORKDIR" \
          --summary_out_dir "$SUMMARY_OUT_DIR" \
          --start_idx "$START_IDX" --end_idx "$END_IDX" \
          --job_name "$JOB_NAME" --workers "$CHAT_WORKERS" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --query_mode "$QUERY_MODE" --use_plan "$USE_PLAN" \
          --max_core "$MAX_CORE" --max_support "$MAX_SUPPORT" \
          --max_packed "$MAX_PACKED" \
          "${overwrite_flags[@]}"
        ;;
      build_split)
        START_IDX="${START_IDX:-}"
        END_IDX="${END_IDX:-}"
        CHUNK_SIZE="${CHUNK_SIZE:-50}"
        JOB_NAME="${JOB_NAME:-asqa_summary_batch}"

        if [[ -z "$START_IDX" || -z "$END_IDX" ]]; then
          echo "Usage: START_IDX=... END_IDX=... [CHUNK_SIZE=50] [JOB_NAME=...] bash asqa_summary_batch.sh build_split" >&2
          exit 2
        fi

        python "$PYFILE" build_split \
          --query_struct_dir "$QUERY_STRUCT_DIR" \
          --topk_dir "$TOPK_DIR" \
          --local_structure_dir "$LOCAL_STRUCTURE_DIR" \
          --out_dir "$BATCH_WORKDIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --chunk_size "$CHUNK_SIZE" \
          --job_name "$JOB_NAME" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --query_mode "$QUERY_MODE" \
          --use_plan "$USE_PLAN" \
          --max_core "$MAX_CORE" \
          --max_support "$MAX_SUPPORT" \
          --max_packed "$MAX_PACKED"

        echo "[INFO] build_split done under: $BATCH_WORKDIR"
        ;;

      submit_split)
        START_IDX="${START_IDX:-}"
        END_IDX="${END_IDX:-}"
        CHUNK_SIZE="${CHUNK_SIZE:-50}"
        JOB_NAME="${JOB_NAME:-asqa_summary_batch}"

        if [[ -z "$START_IDX" || -z "$END_IDX" ]]; then
          echo "Usage: START_IDX=... END_IDX=... [CHUNK_SIZE=50] [JOB_NAME=...] bash asqa_summary_batch.sh submit_split" >&2
          exit 2
        fi

        python "$PYFILE" submit_split \
          --query_struct_dir "$QUERY_STRUCT_DIR" \
          --topk_dir "$TOPK_DIR" \
          --local_structure_dir "$LOCAL_STRUCTURE_DIR" \
          --out_dir "$BATCH_WORKDIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --chunk_size "$CHUNK_SIZE" \
          --job_name "$JOB_NAME" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --completion_window "$COMPLETION_WINDOW" \
          --query_mode "$QUERY_MODE" \
          --use_plan "$USE_PLAN" \
          --max_core "$MAX_CORE" \
          --max_support "$MAX_SUPPORT" \
          --max_packed "$MAX_PACKED"
        ;;

      status)
        BATCH_ID="${BATCH_ID:-}"
        if [[ -z "$BATCH_ID" ]]; then
          echo "Usage: BATCH_ID=<batch_id> bash asqa_summary_batch.sh status" >&2
          exit 2
        fi

        python "$PYFILE" status --batch_id "$BATCH_ID"
        ;;

      status_from_tsv|status_tsv)
        python "$PYFILE" status_tsv --tsv "$TSV_PATH"
        ;;

      consume)
        BATCH_ID="${BATCH_ID:-}"
        START_IDX="${START_IDX:-}"
        END_IDX="${END_IDX:-}"
        JOB_NAME="${JOB_NAME:-asqa_summary_batch}"

        if [[ -z "$BATCH_ID" || -z "$START_IDX" || -z "$END_IDX" ]]; then
          echo "Usage: START_IDX=... END_IDX=... BATCH_ID=<batch_id> [JOB_NAME=...] bash asqa_summary_batch.sh consume" >&2
          exit 2
        fi

        RAW_OUT_JSONL="${RAW_OUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.output.jsonl}"
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/${JOB_NAME}.failed_ids.txt}"

        python "$PYFILE" consume \
          --batch_id "$BATCH_ID" \
          --query_struct_dir "$QUERY_STRUCT_DIR" \
          --topk_dir "$TOPK_DIR" \
          --local_structure_dir "$LOCAL_STRUCTURE_DIR" \
          --summary_out_dir "$SUMMARY_OUT_DIR" \
          --raw_out_jsonl "$RAW_OUT_JSONL" \
          --failed_ids_out "$FAILED_IDS_OUT" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --query_mode "$QUERY_MODE" \
          --use_plan "$USE_PLAN"
        ;;

      consume_from_tsv|consume_tsv)
        overwrite_args=()
        if [[ "$OVERWRITE" == "1" ]]; then
          overwrite_args+=(--overwrite)
        fi
        python "$PYFILE" consume_tsv \
          --tsv "$TSV_PATH" \
          --jobs_dir "$BATCH_WORKDIR" \
          --query_struct_dir "$QUERY_STRUCT_DIR" \
          --topk_dir "$TOPK_DIR" \
          --local_structure_dir "$LOCAL_STRUCTURE_DIR" \
          --summary_out_dir "$SUMMARY_OUT_DIR" \
          --query_mode "$QUERY_MODE" \
          --use_plan "$USE_PLAN" \
          "${overwrite_args[@]}"
        ;;

      consume_local)
        overwrite_args=()
        if [[ "$OVERWRITE" == "1" ]]; then
          overwrite_args+=(--overwrite)
        fi
        START_IDX="${START_IDX:-}"
        END_IDX="${END_IDX:-}"
        PATTERN="${PATTERN:-batch_*.output.jsonl}"
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/consume_local.failed_ids.txt}"

        if [[ -z "$START_IDX" || -z "$END_IDX" ]]; then
          echo "Usage: START_IDX=... END_IDX=... [PATTERN=batch_*.output.jsonl] bash asqa_summary_batch.sh consume_local" >&2
          exit 2
        fi

        python "$PYFILE" consume_local \
          --jobs_dir "$BATCH_WORKDIR" \
          --pattern "$PATTERN" \
          --query_struct_dir "$QUERY_STRUCT_DIR" \
          --topk_dir "$TOPK_DIR" \
          --local_structure_dir "$LOCAL_STRUCTURE_DIR" \
          --summary_out_dir "$SUMMARY_OUT_DIR" \
          --failed_ids_out "$FAILED_IDS_OUT" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --query_mode "$QUERY_MODE" \
          --use_plan "$USE_PLAN" \
          "${overwrite_args[@]}"
        ;;

      single)
        QUERY_STRUCT="${QUERY_STRUCT:-}"
        TOPK_JSON="${TOPK_JSON:-}"
        LOCAL_STRUCTURE="${LOCAL_STRUCTURE:-}"
        OUT_JSON="${OUT_JSON:-}"

        if [[ -z "$TOPK_JSON" || ( "$QUERY_MODE" == "structured" && -z "$QUERY_STRUCT" ) ]]; then
          echo "Usage: QUERY_MODE=original TOPK_JSON=... [QUERY_STRUCT=optional-fallback] [OUT_JSON=...] bash asqa_summary_batch.sh single" >&2
          exit 2
        fi

        python "$PYFILE" single \
          --backend proxy \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --query_struct "$QUERY_STRUCT" \
          --topk "$TOPK_JSON" \
          --local_structure "$LOCAL_STRUCTURE" \
          --out "$OUT_JSON" \
          --query_mode "$QUERY_MODE" \
          --use_plan "$USE_PLAN" \
          --max_core "$MAX_CORE" \
          --max_support "$MAX_SUPPORT" \
          --max_packed "$MAX_PACKED"
        ;;

      *)
        cat >&2 <<EOF
Usage:
  START_IDX=0 END_IDX=10 CHAT_WORKERS=4 JOB_NAME=asqa_train_summary_s3_0_10 bash asqa_summary_batch.sh run_chat
  START_IDX=0 END_IDX=4353 CHAT_WORKERS=4 JOB_NAME=asqa_train_summary_s3_0_4353 bash asqa_summary_batch.sh run_chat
  OVERWRITE=1 QUERY_MODE=original bash asqa_summary_batch.sh consume_tsv
  TSV_PATH=... bash asqa_summary_batch.sh status_tsv
  TSV_PATH=... bash asqa_summary_batch.sh consume_tsv
EOF
        exit 2
        ;;
    esac
  ' "$@"
