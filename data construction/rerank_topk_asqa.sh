#!/usr/bin/env bash
set -euo pipefail

# ASQA Train R3 / A4: GPT-4o original-query rerank, with V3 batching.
# Unified Qwen3 package environment (reference: rerank_range launcher).
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"

BATCHDIR="${BATCHDIR:-$RUNNER_DIR}"
PYFILE="${PYFILE:-$BATCHDIR/rerank_topk_asqa.py}"

PKG_ENV="${PKG_ENV:-}"
PKG_MARKER="${PKG_MARKER:-}"

# ASQA Train (4353 samples): Candidate Pool from heuristic_v3.
# Input candidate pool is configurable; do NOT copy or mutate it.
# All generated artifacts are rooted in caller-configured output locations.
CANDIDATE_DIR="${CANDIDATE_DIR:?Set CANDIDATE_DIR to the heuristic candidate pool}"
TEST_RERANK_ROOT="${TEST_RERANK_ROOT:-}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-${TEST_RERANK_ROOT:?Set FINAL_OUT_DIR or TEST_RERANK_ROOT}/topk}"
BATCH_WORKDIR="${BATCH_WORKDIR:-${TEST_RERANK_ROOT:-$FINAL_OUT_DIR}/jobs}"

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
# Raised with RERANK_BATCH: 18 labelled candidates need roughly 2x the tokens of
# 6, and a truncated response is what produces missing labels.
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2600}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-24h}"

# R3 / A4: original query only in GPT-4o Rerank Prompt.
# One prompt per query: "core" is a comparative judgement over the whole pool,
# so chunking it into groups of 6 made the teacher systematically conservative.
RERANK_BATCH="${RERANK_BATCH:-18}"
MAX_INPUT_CANDIDATES="${MAX_INPUT_CANDIDATES:-18}"
QUERY_MODE="${QUERY_MODE:-original}"
RESUBMIT="${RESUBMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
TOPK_CORE="${TOPK_CORE:-5}"
TOPK_SUPPORT="${TOPK_SUPPORT:-8}"
MAX_PACKED_SENTS="${MAX_PACKED_SENTS:-14}"
# Floor on packed evidence; 0 restores the previous all-drop behaviour.
MIN_PACKED_SENTS="${MIN_PACKED_SENTS:-3}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Match the reference script's cache and runtime settings.
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
  "$BATCH_WORKDIR" "$FINAL_OUT_DIR" \
  "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$TRITON_CACHE_DIR" "$TMPDIR"

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
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "❌ Missing PKG_ENV: $PKG_ENV" >&2; exit 4; }
[[ -z "$PKG_MARKER" || -f "$PKG_MARKER" ]] || { echo "❌ Missing PKG marker: $PKG_MARKER" >&2; exit 5; }

if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
  BIND_TMP=( --bind "$SLURM_TMPDIR:$SLURM_TMPDIR" --env SLURM_TMPDIR="$SLURM_TMPDIR" )
else
  BIND_TMP=()
fi

CMD="${1:-}"
shift || true

echo "[INFO] Container cmd       : $CTR_CMD"
echo "[INFO] Image               : $IMAGE"
echo "[INFO] BATCHDIR            : $BATCHDIR"
echo "[INFO] PKG_ENV             : $PKG_ENV"
echo "[INFO] CANDIDATE_DIR       : $CANDIDATE_DIR"
echo "[INFO] TEST_RERANK_ROOT    : $TEST_RERANK_ROOT"
echo "[INFO] FINAL_OUT_DIR       : $FINAL_OUT_DIR"
echo "[INFO] BATCH_WORKDIR       : $BATCH_WORKDIR"
echo "[INFO] CMD                 : $CMD"
echo "[INFO] OPENAI_MODEL        : $OPENAI_MODEL"
echo "[INFO] PROXY_BASE_URL     : $PROXY_BASE_URL"
echo "[INFO] CHAT_WORKERS        : $CHAT_WORKERS"
echo "[INFO] MAX_OUTPUT_TOKENS   : $MAX_OUTPUT_TOKENS"
echo "[INFO] COMPLETION_WINDOW   : $COMPLETION_WINDOW"
echo "[INFO] RERANK_BATCH        : $RERANK_BATCH"
echo "[INFO] MAX_INPUT_CANDIDATES: $MAX_INPUT_CANDIDATES"
echo "[INFO] QUERY_MODE          : $QUERY_MODE"
echo "[INFO] TOPK_CORE           : $TOPK_CORE"
echo "[INFO] TOPK_SUPPORT        : $TOPK_SUPPORT"
echo "[INFO] MAX_PACKED_SENTS    : $MAX_PACKED_SENTS"
echo "[INFO] MIN_PACKED_SENTS    : $MIN_PACKED_SENTS"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_TMPDIR        : ${SLURM_TMPDIR:-<empty>}"
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
for name in SOURCE_ROOT BATCHDIR PYFILE PKG_ENV CANDIDATE_DIR FINAL_OUT_DIR BATCH_WORKDIR \
  TSV_PATH RAW_OUT_JSONL FAILED_IDS_OUT PIP_CACHE_DIR XDG_CACHE_HOME HF_HOME \
  HF_HUB_CACHE HF_DATASETS_CACHE TRITON_CACHE_DIR TMPDIR; do
  bind_configured_path "$name"
done

"$CTR_CMD" exec --nv \
  "${BIND_ARGS[@]}" \
  ${BIND_TMP[@]+"${BIND_TMP[@]}"} \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env PKG_ENV="$PKG_ENV" \
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
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --env PROXY_API_KEY="${PROXY_API_KEY:-}" \
  --env OPENAI_MODEL="$OPENAI_MODEL" \
  --env PROXY_BASE_URL="$PROXY_BASE_URL" \
  --env OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  --env API_SUPPORTS_BATCH="$API_SUPPORTS_BATCH" \
  --env CHAT_WORKERS="$CHAT_WORKERS" \
  --env OPENAI_TIMEOUT="$OPENAI_TIMEOUT" \
  --env OPENAI_MAX_RETRIES="$OPENAI_MAX_RETRIES" \
  --env MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
  --env COMPLETION_WINDOW="$COMPLETION_WINDOW" \
  --env CANDIDATE_DIR="$CANDIDATE_DIR" \
  --env FINAL_OUT_DIR="$FINAL_OUT_DIR" \
  --env BATCH_WORKDIR="$BATCH_WORKDIR" \
  --env BATCHDIR="$BATCHDIR" \
  --env PYFILE="$PYFILE" \
  --env TEST_RERANK_ROOT="$TEST_RERANK_ROOT" \
  --env CMD="$CMD" \
  --env START_IDX="${START_IDX:-}" \
  --env END_IDX="${END_IDX:-}" \
  --env CHUNK_SIZE="${CHUNK_SIZE:-}" \
  --env JOB_NAME="${JOB_NAME:-}" \
  --env TSV_PATH="${TSV_PATH:-}" \
  --env RAW_OUT_JSONL="${RAW_OUT_JSONL:-}" \
  --env FAILED_IDS_OUT="${FAILED_IDS_OUT:-}" \
  --env BATCH_ID="${BATCH_ID:-}" \
  --env PATTERN="${PATTERN:-}" \
  --env RERANK_BATCH="$RERANK_BATCH" \
  --env MAX_INPUT_CANDIDATES="$MAX_INPUT_CANDIDATES" \
  --env QUERY_MODE="$QUERY_MODE" \
  --env RESUBMIT="$RESUBMIT" \
  --env OVERWRITE="$OVERWRITE" \
  --env ALLOW_PARTIAL="$ALLOW_PARTIAL" \
  --env TOPK_CORE="$TOPK_CORE" \
  --env TOPK_SUPPORT="$TOPK_SUPPORT" \
  --env MAX_PACKED_SENTS="$MAX_PACKED_SENTS" \
  --env MIN_PACKED_SENTS="$MIN_PACKED_SENTS" \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$BATCHDIR"

    unset PYTHONSTARTUP PYTHONHOME PYTHONPATH
    export PYTHONPATH="$PKG_ENV"

    echo "[INFO] inside container pwd=$(pwd)"
    echo "[INFO] inside container PYTHONPATH=$PYTHONPATH"
    echo "[INFO] inside container CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    if [[ ! -f "$PYFILE" ]]; then
      echo "[ERROR] Missing python file: $PYFILE" >&2
      exit 12
    fi

    echo "[INFO] pwd=$(pwd)"
    echo "[INFO] python file:"
    ls -l "$PYFILE"

    JOB_NAME="${JOB_NAME:-asqa_rerank}"
    CHUNK_SIZE="${CHUNK_SIZE:-50}"
    TSV_PATH="${TSV_PATH:-$BATCH_WORKDIR/${JOB_NAME}.batches.tsv}"
    START_IDX="${START_IDX:-}"
    END_IDX="${END_IDX:-}"
    # Native Batch is opt-in and independent of the endpoint hostname.
    case "$CMD" in
      submit_split|status|status_tsv|status_from_tsv|consume_tsv|consume_from_tsv|consume)
        if [[ "$API_SUPPORTS_BATCH" != "1" ]]; then
          echo "[ERROR] Native Batch is disabled; use run_chat or set API_SUPPORTS_BATCH=1 for a capable endpoint." >&2
          exit 21
        fi
        : "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to the native Batch API endpoint}"
        [[ -n "${OPENAI_API_KEY:-}" ]] || {
          echo "[ERROR] OPENAI_API_KEY missing for command: $CMD" >&2; exit 10;
        }
        ;;
      run_chat)
        : "${PROXY_BASE_URL:?Set PROXY_BASE_URL to the Chat API endpoint}"
        [[ -n "${PROXY_API_KEY:-}" ]] || {
          echo "[ERROR] PROXY_API_KEY missing for run_chat" >&2; exit 10;
        }
        ;;
    esac

    rerank_options=(
      --openai_model "$OPENAI_MODEL"
      --rerank_batch "$RERANK_BATCH"
      --max_input_candidates "$MAX_INPUT_CANDIDATES"
      --query_mode "$QUERY_MODE"
      --max_output_tokens "$MAX_OUTPUT_TOKENS"
    )
    pack_options=(
      --topk_core "$TOPK_CORE"
      --topk_support "$TOPK_SUPPORT"
      --max_packed_sents "$MAX_PACKED_SENTS"
    )
    partial_options=()
    [[ "$ALLOW_PARTIAL" == "1" ]] && partial_options+=(--allow_partial)
    overwrite_options=()
    [[ "$OVERWRITE" == "1" ]] && overwrite_options+=(--overwrite)

    case "$CMD" in
      run_chat)
        [[ -d "$CANDIDATE_DIR" ]] || {
          echo "[ERROR] Missing candidate dir: $CANDIDATE_DIR" >&2; exit 13;
        }
        START_IDX="${START_IDX:-0}"
        END_IDX="${END_IDX:-10}"
        JOB_NAME="${JOB_NAME:-asqa_train_r3_proxy_${START_IDX}_${END_IDX}}"
        python "$PYFILE" run_chat \
          --candidate_dir "$CANDIDATE_DIR" \
          --jobs_dir "$BATCH_WORKDIR" \
          --final_out_dir "$FINAL_OUT_DIR" \
          --start_idx "$START_IDX" --end_idx "$END_IDX" \
          --job_name "$JOB_NAME" --workers "$CHAT_WORKERS" \
          "${rerank_options[@]}" "${pack_options[@]}" "${overwrite_options[@]}"
        ;;
      build_split|submit_split)
        if [[ ! -d "$CANDIDATE_DIR" ]]; then
          echo "[ERROR] Candidate input directory does not exist: $CANDIDATE_DIR" >&2
          exit 13
        fi
        if ! find "$CANDIDATE_DIR" -maxdepth 1 -type f -name 'candidate_pool_*.json' -print -quit | grep -q .; then
          echo "[ERROR] No candidate_pool_*.json files in: $CANDIDATE_DIR" >&2
          exit 14
        fi
        if [[ -z "$START_IDX" || -z "$END_IDX" ]]; then
          echo "Usage: START_IDX=0 END_IDX=50 [CHUNK_SIZE=50] bash rerank_topk_asqa.sh submit_split" >&2
          exit 2
        fi
        shared=(
          --candidate_dir "$CANDIDATE_DIR" --out_dir "$BATCH_WORKDIR"
          --start_idx "$START_IDX" --end_idx "$END_IDX"
          --chunk_size "$CHUNK_SIZE" --job_name "$JOB_NAME"
          "${rerank_options[@]}"
        )
        if [[ "$CMD" == "submit_split" ]]; then
          resubmit_options=()
          [[ "$RESUBMIT" == "1" ]] && resubmit_options+=(--resubmit)
          python "$PYFILE" submit_split "${shared[@]}" \
            --completion_window "$COMPLETION_WINDOW" --tsv "$TSV_PATH" \
            "${resubmit_options[@]}"
          echo "[DONE] Unified Batch registry: $TSV_PATH"
        else
          python "$PYFILE" build_split "${shared[@]}"
        fi
        ;;

      status)
        [[ -n "${BATCH_ID:-}" ]] || { echo "Set BATCH_ID" >&2; exit 2; }
        python "$PYFILE" status --batch_id "$BATCH_ID"
        ;;

      status_tsv|status_from_tsv)
        echo "[INFO] Checking every batch in $TSV_PATH"
        python "$PYFILE" status_from_tsv --tsv "$TSV_PATH"
        ;;

      consume_tsv|consume_from_tsv)
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/${JOB_NAME}.consume.failed_ids.txt}"
        echo "[INFO] Consuming all completed batches from $TSV_PATH"
        python "$PYFILE" consume_from_tsv \
          --tsv "$TSV_PATH" --jobs_dir "$BATCH_WORKDIR" \
          --candidate_dir "$CANDIDATE_DIR" --final_out_dir "$FINAL_OUT_DIR" \
          --failed_ids_out "$FAILED_IDS_OUT" \
          "${rerank_options[@]}" "${pack_options[@]}" \
          "${partial_options[@]}" "${overwrite_options[@]}"
        ;;

      consume)
        [[ -n "${BATCH_ID:-}" && -n "$START_IDX" && -n "$END_IDX" ]] || {
          echo "Set BATCH_ID, START_IDX, END_IDX" >&2; exit 2;
        }
        RAW_OUT_JSONL="${RAW_OUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.${BATCH_ID}.output.jsonl}"
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/${JOB_NAME}.${BATCH_ID}.failed_ids.txt}"
        python "$PYFILE" consume --batch_id "$BATCH_ID" \
          --candidate_dir "$CANDIDATE_DIR" --final_out_dir "$FINAL_OUT_DIR" \
          --raw_out_jsonl "$RAW_OUT_JSONL" --failed_ids_out "$FAILED_IDS_OUT" \
          --start_idx "$START_IDX" --end_idx "$END_IDX" \
          "${rerank_options[@]}" "${pack_options[@]}" "${partial_options[@]}"
        ;;

      consume_local)
        [[ -n "$START_IDX" && -n "$END_IDX" ]] || {
          echo "Set START_IDX and END_IDX" >&2; exit 2;
        }
        PATTERN="${PATTERN:-*.output.jsonl}"
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/consume_local.failed_ids.txt}"
        python "$PYFILE" consume_local \
          --jobs_dir "$BATCH_WORKDIR" --pattern "$PATTERN" \
          --candidate_dir "$CANDIDATE_DIR" --final_out_dir "$FINAL_OUT_DIR" \
          --failed_ids_out "$FAILED_IDS_OUT" \
          --start_idx "$START_IDX" --end_idx "$END_IDX" \
          "${rerank_options[@]}" "${pack_options[@]}" \
          "${partial_options[@]}" "${overwrite_options[@]}"
        ;;

      *)
        echo "Commands: run_chat | build_split | submit_split | status | status_tsv | consume_tsv | consume | consume_local" >&2
        echo "Defaults: TEST_RERANK_ROOT=$TEST_RERANK_ROOT TSV_PATH=$TSV_PATH" >&2
        exit 2
        ;;
    esac
  ' "$@"
