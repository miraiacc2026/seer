#!/usr/bin/env bash
set -euo pipefail

# ASQA TRAIN Query teacher-data via Proxy Chat Completions. Eval outputs stay isolated.
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"

BATCHDIR="${BATCHDIR:-$RUNNER_DIR}"
ENTDIR="${ENTDIR:-}"

# Qwen3 environment: legacy per-package PYTHONPATH injection removed.
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
  "$TMPDIR" \
  "$MYPROJ/.logs"

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
[[ -z "$ENTDIR" || -d "$ENTDIR" ]] || { echo "❌ Missing ENTDIR: $ENTDIR" >&2; exit 4; }
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "❌ Missing PKG_ENV: $PKG_ENV" >&2; exit 5; }
[[ -z "$PKG_MARKER" || -f "$PKG_MARKER" ]] || { echo "❌ Missing PKG marker: $PKG_MARKER" >&2; exit 6; }

SRC_JSON="${SRC_JSON:?Set SRC_JSON to the input dataset file}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1400}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-24h}"
CHAT_WORKERS="${CHAT_WORKERS:-3}"
# Configure the endpoint explicitly; capabilities never depend on its hostname.
API_BASE_URL="${API_BASE_URL:-${OPENAI_BASE_URL:-${PROXY_BASE_URL:-}}}"
API_MODE="${API_MODE:-}"
PROXY_BASE_URL="${PROXY_BASE_URL:-}"
API_SUPPORTS_BATCH="${API_SUPPORTS_BATCH:-0}"
case "$API_SUPPORTS_BATCH" in
  0|1) ;;
  *) echo "[ERROR] API_SUPPORTS_BATCH must be 0 or 1" >&2; exit 2 ;;
esac
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-2}"

# Separate training outputs from eval outputs to avoid index collisions.
TEST_QUERY_ROOT="${TEST_QUERY_ROOT:-}"
OUT_DIR="${OUT_DIR:-${TEST_QUERY_ROOT:?Set OUT_DIR or TEST_QUERY_ROOT}/query_struct}"
BATCH_WORKDIR="${BATCH_WORKDIR:-$OUT_DIR/jobs}"
TSV_PATH="${TSV_PATH:-$BATCH_WORKDIR/latest_batches.tsv}"

PY_SCRIPT="${PY_SCRIPT:-$BATCHDIR/asqa_query_batch.py}"
BASH_SCRIPT="${BASH_SCRIPT:-$BATCHDIR/asqa_query_batch.sh}"

mkdir -p "$BATCH_WORKDIR" "$OUT_DIR"

if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
  BIND_TMP=( --bind "$SLURM_TMPDIR:$SLURM_TMPDIR" --env SLURM_TMPDIR="$SLURM_TMPDIR" )
else
  BIND_TMP=()
fi

CMD="${1:-}"
shift || true

echo "[INFO] Container cmd : $CTR_CMD"
echo "[INFO] Image         : $IMAGE"
echo "[INFO] PKG_ENV       : $PKG_ENV"
echo "[INFO] PKG_MARKER    : $PKG_MARKER"
echo "[INFO] CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_TMPDIR  : ${SLURM_TMPDIR:-<empty>}"
echo "[INFO] BATCHDIR      : $BATCHDIR"
echo "[INFO] ENTDIR        : $ENTDIR"
echo "[INFO] SRC_JSON      : $SRC_JSON"
echo "[INFO] TEST_QUERY_ROOT: $TEST_QUERY_ROOT"
echo "[INFO] BATCH_WORKDIR : $BATCH_WORKDIR"
echo "[INFO] OUT_DIR       : $OUT_DIR"
echo "[INFO] PY_SCRIPT     : $PY_SCRIPT"
echo "[INFO] BASH_SCRIPT   : $BASH_SCRIPT"
echo "[INFO] CMD           : $CMD"
echo "[INFO] OPENAI_MODEL  : $OPENAI_MODEL"
echo "[INFO] API_BASE_URL  : ${API_BASE_URL:-<not configured>}"
echo "[INFO] CHAT_WORKERS  : $CHAT_WORKERS"
echo "[INFO] MAX_OUTPUT_TOKENS: $MAX_OUTPUT_TOKENS"
echo "[INFO] COMPLETION_WINDOW: $COMPLETION_WINDOW"
echo "[INFO] START_IDX=${START_IDX:-} END_IDX=${END_IDX:-} CHUNK_SIZE=${CHUNK_SIZE:-}"
echo "[INFO] JOB_NAME=${JOB_NAME:-} INPUT_JSONL=${INPUT_JSONL:-} BATCH_ID=${BATCH_ID:-}"
echo "[INFO] PATTERN=${PATTERN:-}"
echo "[INFO] TSV_PATH=${TSV_PATH:-}"

# Use the runtime's default certificate configuration.
SOURCE_ROOT="$(cd -- "$(dirname -- "$PY_SCRIPT")/.." && pwd -P)"
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
for name in SOURCE_ROOT BATCHDIR ENTDIR PKG_ENV PY_SCRIPT BASH_SCRIPT SRC_JSON OUT_DIR \
  BATCH_WORKDIR TSV_PATH INPUT_JSONL META_JSON RAW_OUT_JSONL FAILED_IDS_OUT \
  FALLBACK_IDS_OUT PIP_CACHE_DIR XDG_CACHE_HOME HF_HOME HF_HUB_CACHE \
  HF_DATASETS_CACHE TRITON_CACHE_DIR TMPDIR; do
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
  --env ENTDIR="$ENTDIR" \
  --env TEST_QUERY_ROOT="$TEST_QUERY_ROOT" \
  --env BATCH_WORKDIR="$BATCH_WORKDIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env PY_SCRIPT="$PY_SCRIPT" \
  --env BASH_SCRIPT="$BASH_SCRIPT" \
  --env CMD="$CMD" \
  --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --env PROXY_API_KEY="${PROXY_API_KEY:-}" \
  --env OPENAI_BASE_URL="$API_BASE_URL" \
  --env PROXY_BASE_URL="$PROXY_BASE_URL" \
  --env API_MODE="$API_MODE" \
  --env API_SUPPORTS_BATCH="$API_SUPPORTS_BATCH" \
  --env OPENAI_TIMEOUT="$OPENAI_TIMEOUT" \
  --env OPENAI_MAX_RETRIES="$OPENAI_MAX_RETRIES" \
  --env CHAT_WORKERS="$CHAT_WORKERS" \
  --env OPENAI_MODEL="$OPENAI_MODEL" \
  --env MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
  --env COMPLETION_WINDOW="$COMPLETION_WINDOW" \
  --env SRC_JSON="$SRC_JSON" \
  --env START_IDX="${START_IDX:-}" \
  --env END_IDX="${END_IDX:-}" \
  --env CHUNK_SIZE="${CHUNK_SIZE:-}" \
  --env JOB_NAME="${JOB_NAME:-}" \
  --env INPUT_JSONL="${INPUT_JSONL:-}" \
  --env META_JSON="${META_JSON:-}" \
  --env RAW_OUT_JSONL="${RAW_OUT_JSONL:-}" \
  --env FAILED_IDS_OUT="${FAILED_IDS_OUT:-}" \
  --env FALLBACK_IDS_OUT="${FALLBACK_IDS_OUT:-}" \
  --env BATCH_ID="${BATCH_ID:-}" \
  --env PATTERN="${PATTERN:-}" \
  --env TSV_PATH="${TSV_PATH:-}" \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$BATCHDIR"

    # Match the canonical Qwen3 loader: do not inherit host/site Python paths.
    unset PYTHONSTARTUP PYTHONHOME PYTHONPATH
    export PYTHONPATH="$PKG_ENV"

    echo "[INFO] inside container pwd=$(pwd)"
    echo "[INFO] inside container PYTHONPATH=$PYTHONPATH"
    echo "[INFO] inside container CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    # Import diagnostics: expose which environment provides the Batch API SDK.
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

    if [[ ! -f "$PY_SCRIPT" ]]; then
      echo "[ERROR] Missing python file: $PY_SCRIPT" >&2
      exit 12
    fi

    echo "[INFO] pwd=$(pwd)"
    echo "[INFO] python script:"
    ls -l "$PY_SCRIPT"

    needs_key=0
    case "$CMD" in
      submit|submit_split|status|consume|run_split|status_tsv|consume_tsv|run_chat|probe_batch) needs_key=1 ;;
      *) needs_key=0 ;;
    esac
    if [[ "$needs_key" == "1" ]]; then
      : "${OPENAI_BASE_URL:?Set API_BASE_URL or OPENAI_BASE_URL to the API endpoint}"
      if [[ "$API_MODE" == "proxy" || ( -n "$PROXY_BASE_URL" && "${OPENAI_BASE_URL%/}" == "${PROXY_BASE_URL%/}" ) ]]; then
        : "${PROXY_BASE_URL:?Set PROXY_BASE_URL to the proxy API endpoint}"
        : "${PROXY_API_KEY:?Set PROXY_API_KEY for the configured proxy endpoint}"
      else
        : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the configured API endpoint}"
      fi
    fi

    # Native Batch requires an explicit capability declaration.
    if [[ "$API_SUPPORTS_BATCH" != "1" ]]; then
      case "$CMD" in
        submit|submit_split|run_split|status|status_tsv|consume|consume_tsv)
          echo "[ERROR] Native Batch is disabled; use run_chat or set API_SUPPORTS_BATCH=1 for a capable endpoint." >&2
          exit 21 ;;
      esac
    fi

    if [[ ( "$CMD" == "build" || "$CMD" == "build_split" || "$CMD" == "run_split" || "$CMD" == "submit_split" || "$CMD" == "run_chat" ) && ! -f "$SRC_JSON" ]]; then
      echo "[ERROR] SRC_JSON not found: $SRC_JSON" >&2
      exit 11
    fi

    case "$CMD" in
      validate_input)
        python "$PY_SCRIPT" validate_input --src "$SRC_JSON"
        ;;

      probe_batch)
        # Read-only API check. GET /batches works only if provider supports listing.
        python "$PY_SCRIPT" probe_batch
        ;;

      run_chat)
        # Direct parallel Chat Completions requests (not native Batch pricing).
        START_IDX="${START_IDX:-0}"
        END_IDX="${END_IDX:-10}"
        JOB_NAME="${JOB_NAME:-asqa_query_proxy_chat_${START_IDX}_${END_IDX}}"
        python "$PY_SCRIPT" run_chat \
          --src "$SRC_JSON" \
          --out_dir "$OUT_DIR" \
          --jobs_dir "$BATCH_WORKDIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --job_name "$JOB_NAME" \
          --workers "$CHAT_WORKERS" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS"
        ;;

      build)
        START_IDX="${START_IDX:-0}"
        END_IDX="${END_IDX:--1}"
        JOB_NAME="${JOB_NAME:-asqa_query_batch_${START_IDX}_${END_IDX}}"
        INPUT_JSONL="${INPUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.jsonl}"

        python "$PY_SCRIPT" build \
          --src "$SRC_JSON" \
          --out_jsonl "$INPUT_JSONL" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS"

        echo "[INFO] build done: $INPUT_JSONL"
        [[ -f "$INPUT_JSONL" ]] && ls -lh "$INPUT_JSONL"
        ;;

      build_split)
        START_IDX="${START_IDX:-0}"
        END_IDX="${END_IDX:--1}"
        CHUNK_SIZE="${CHUNK_SIZE:-50}"
        JOB_NAME="${JOB_NAME:-asqa_query_batch_${START_IDX}_${END_IDX}}"

        python "$PY_SCRIPT" build_split \
          --src "$SRC_JSON" \
          --out_dir "$BATCH_WORKDIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --chunk_size "$CHUNK_SIZE" \
          --job_name "$JOB_NAME" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS"

        echo "[INFO] build_split done under: $BATCH_WORKDIR"
        ;;

      submit)
        JOB_NAME="${JOB_NAME:-asqa_query_batch}"
        INPUT_JSONL="${INPUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.jsonl}"
        META_JSON="${META_JSON:-$BATCH_WORKDIR/${JOB_NAME}.meta.json}"

        python "$PY_SCRIPT" submit \
          --input_jsonl "$INPUT_JSONL" \
          --openai_model "$OPENAI_MODEL" \
          --completion_window "$COMPLETION_WINDOW" \
          --metadata_name "$JOB_NAME" \
          --batch_meta_out "$META_JSON"
        ;;

      submit_split|run_split)
        # ONE COMMAND: build_split -> submit each job -> write a unified TSV.
        # Re-running reuses existing batch IDs if checksums agree.
        START_IDX="${START_IDX:-0}"
        END_IDX="${END_IDX:--1}"
        CHUNK_SIZE="${CHUNK_SIZE:-50}"
        JOB_NAME="${JOB_NAME:-asqa_query_batch_${START_IDX}_${END_IDX}}"

        python "$PY_SCRIPT" submit_split \
          --src "$SRC_JSON" \
          --out_dir "$BATCH_WORKDIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --chunk_size "$CHUNK_SIZE" \
          --job_name "$JOB_NAME" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --completion_window "$COMPLETION_WINDOW"
        ;;

      status)
        BATCH_ID="${BATCH_ID:-}"
        if [[ -z "$BATCH_ID" ]]; then
          echo "Usage: BATCH_ID=<batch_id> bash asqa_query_batch.sh status" >&2
          exit 2
        fi
        python "$PY_SCRIPT" status --batch_id "$BATCH_ID"
        ;;

      status_tsv)
        # If TSV_PATH is not supplied, use the latest manifest.
        python "$PY_SCRIPT" status_tsv --tsv "$TSV_PATH"
        ;;

      consume)
        BATCH_ID="${BATCH_ID:-}"
        if [[ -z "$BATCH_ID" ]]; then
          echo "Usage: START_IDX=... END_IDX=... BATCH_ID=<batch_id> bash asqa_query_batch.sh consume" >&2
          exit 2
        fi

        JOB_NAME="${JOB_NAME:-asqa_query_batch}"
        RAW_OUT_JSONL="${RAW_OUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.output.jsonl}"
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/${JOB_NAME}.failed_ids.txt}"
        FALLBACK_IDS_OUT="${FALLBACK_IDS_OUT:-$BATCH_WORKDIR/${JOB_NAME}.fallback_ids.txt}"

        python "$PY_SCRIPT" consume \
          --batch_id "$BATCH_ID" \
          --src "$SRC_JSON" \
          --out_dir "$OUT_DIR" \
          --raw_out_jsonl "$RAW_OUT_JSONL" \
          --failed_ids_out "$FAILED_IDS_OUT" \
          --fallback_ids_out "$FALLBACK_IDS_OUT"
        ;;

      consume_tsv)
        # Decode ALL completed Batches in TSV. Pending ones are left untouched.
        python "$PY_SCRIPT" consume_tsv \
          --tsv "$TSV_PATH" \
          --src "$SRC_JSON" \
          --out_dir "$OUT_DIR" \
          --jobs_dir "$BATCH_WORKDIR"
        ;;

      consume_local|consume_latest)
        # Both commands read previously downloaded Batch API output JSONL locally.
        # The current Python module does NOT expose a consume_local CLI command;
        # use its public parse/post-filter functions to guarantee identical labels.
        PATTERN="${PATTERN:-batch_*.output.jsonl}"
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/${CMD}.failed_ids.txt}"
        FALLBACK_IDS_OUT="${FALLBACK_IDS_OUT:-$BATCH_WORKDIR/${CMD}.fallback_ids.txt}"
        START_IDX="${START_IDX:-0}"
        END_IDX="${END_IDX:--1}"
        export PATTERN FAILED_IDS_OUT FALLBACK_IDS_OUT START_IDX END_IDX

        python - <<PY
import glob
import json
import os
import re
from pathlib import Path

from asqa_query_batch import (
    _load_dataset,
    _write_query_struct_json,
    build_minimal_query_struct,
    extract_output_text_from_batch_line,
    post_filter_query_struct,
    preprocess,
    safe_parse_json,
)

jobs = Path(os.environ["BATCH_WORKDIR"]).expanduser()
pattern = os.environ.get("PATTERN", "batch_*.output.jsonl")
src = os.environ["SRC_JSON"]
out_dir = Path(os.environ["OUT_DIR"]).expanduser()
failed_out = Path(os.environ["FAILED_IDS_OUT"]).expanduser()
fallback_out = Path(os.environ["FALLBACK_IDS_OUT"]).expanduser()
start_idx = int(os.environ.get("START_IDX", "0"))
end_idx = int(os.environ.get("END_IDX", "-1"))
cmd_mode = os.environ.get("CMD", "consume_local")

files = [Path(p) for p in glob.glob(str(jobs / pattern))]
if not files:
    raise FileNotFoundError(f"No downloaded output files match: {jobs / pattern}")
if cmd_mode == "consume_latest":
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
else:
    files.sort()

data = _load_dataset(src)
out_dir.mkdir(parents=True, exist_ok=True)
failed_out.parent.mkdir(parents=True, exist_ok=True)
fallback_out.parent.mkdir(parents=True, exist_ok=True)

ok = 0
skipped_existing = 0
fallback_count = 0
failed_rows = []
fallback_rows = []
seen_ids = set()

for fp in files:
    for lineno, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            match = re.fullmatch(r"query-(\d+)", cid, flags=re.I)
            if not match:
                raise ValueError(f"Invalid custom_id: {cid}")
            idx = int(match.group(1))
            if idx < start_idx or (end_idx >= 0 and idx >= end_idx):
                continue
            if idx in seen_ids:
                continue
            if idx >= len(data):
                raise IndexError(f"sample_idx={idx} outside dataset length={len(data)}")

            final_file = out_dir / f"query_{idx}.json"
            if final_file.exists() and final_file.stat().st_size > 0:
                skipped_existing += 1
                seen_ids.add(idx)
                continue

            status = obj.get("response", {}).get("status_code", 200)
            if obj.get("error") or (isinstance(status, int) and status >= 400):
                err = obj.get("error") or status
                raise RuntimeError(f"Batch API error: {err}")

            query, _, _ = preprocess(data[idx])
            if not query:
                raise ValueError(f"Empty query for sample_idx={idx}")

            try:
                raw = extract_output_text_from_batch_line(obj)
                parsed = safe_parse_json(raw)
                qstruct = post_filter_query_struct(parsed, query)
            except Exception as e:
                qstruct = build_minimal_query_struct(query)
                fallback_count += 1
                fallback_rows.append(f"{cid}\t{e}")
                print(f"[FALLBACK] {cid}: {e}")

            _write_query_struct_json(final_file, idx, query, qstruct)
            seen_ids.add(idx)
            ok += 1
        except Exception as e:
            failed_rows.append(f"{fp.name}:{lineno}\t{e}")
            print(f"[FAILED] {fp.name}:{lineno}: {e}")

failed_out.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
fallback_out.write_text("\n".join(fallback_rows) + ("\n" if fallback_rows else ""), encoding="utf-8")
print(json.dumps({
    "mode": cmd_mode,
    "src": src,
    "jobs_dir": str(jobs),
    "pattern": pattern,
    "out_dir": str(out_dir),
    "num_output_files": len(files),
    "ok": ok,
    "skipped_existing": skipped_existing,
    "fallback": fallback_count,
    "failed": len(failed_rows),
    "failed_ids_out": str(failed_out),
    "fallback_ids_out": str(fallback_out),
}, ensure_ascii=False, indent=2))
PY
        ;;

      *)
        cat >&2 <<EOF
Usage:
  bash asqa_query_batch.sh validate_input  # local schema check, no API charge
  bash asqa_query_batch.sh probe_batch  # read-only, checks Batch listing support
  START_IDX=0 END_IDX=10 bash asqa_query_batch.sh run_chat  # 10 direct chat calls
  bash asqa_query_batch.sh build
  bash asqa_query_batch.sh build_split
  bash asqa_query_batch.sh submit
  bash asqa_query_batch.sh submit_split  # builds, submits, writes TSV
  bash asqa_query_batch.sh run_split
  BATCH_ID=<batch_id> bash asqa_query_batch.sh status
  bash asqa_query_batch.sh status_tsv  # defaults to jobs/latest_batches.tsv
  START_IDX=... END_IDX=... BATCH_ID=<batch_id> bash asqa_query_batch.sh consume
  bash asqa_query_batch.sh consume_tsv  # defaults to jobs/latest_batches.tsv
  START_IDX=... END_IDX=... bash asqa_query_batch.sh consume_local
  START_IDX=... END_IDX=... bash asqa_query_batch.sh consume_latest

Examples:

  START_IDX=0 END_IDX=200 CHUNK_SIZE=50 JOB_NAME=asqa_query_0_200 \
  bash asqa_query_batch.sh build_split

  START_IDX=0 END_IDX=200 CHUNK_SIZE=50 JOB_NAME=asqa_query_0_200 \
  bash asqa_query_batch.sh submit_split  # builds, submits, writes TSV

  START_IDX=0 END_IDX=200 CHUNK_SIZE=50 JOB_NAME=asqa_query_0_200 \
  bash asqa_query_batch.sh run_split

  BATCH_ID=batch_xxx bash asqa_query_batch.sh status

  bash asqa_query_batch.sh status_tsv

  START_IDX=0 END_IDX=200 BATCH_ID=batch_xxx JOB_NAME=asqa_query_0_200 \
  bash asqa_query_batch.sh consume

  bash asqa_query_batch.sh consume_tsv

  START_IDX=0 END_IDX=3000 bash asqa_query_batch.sh consume_latest
  PATTERN="batch_*.output.jsonl" START_IDX=0 END_IDX=3000 \
  bash asqa_query_batch.sh consume_latest
EOF
        exit 2
        ;;
    esac
  ' "$@"
