#!/usr/bin/env bash
set -euo pipefail

# The host Python is used only for wall-clock timestamps and JSON metrics.
HOST_PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$HOST_PYTHON" ]] || { echo "Host python3/python is required for timing." >&2; exit 98; }
SHELL_STARTED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"


# ============================================================
# Global paths
# ============================================================
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"
# Resolve Python sources relative to this launcher, regardless of the caller's cwd.
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKDIR="${WORKDIR:-$RUNNER_DIR}"

# Use Qwen3 env for transformers + tokenizers.
# Do NOT use heuristic env.
PKG_ENV="${PKG_ENV:-}"
PKG_MARKER="${PKG_MARKER:-}"

# ============================================================
# WWW 2027 / ASQA dev (948 items) — A1–A5 summary variants
#   BENCH_ROOT/query_structure/query_*.json
#   BENCH_ROOT/topk_A1|A2|A3/topk_*.json
#   BENCH_ROOT/summary_A1|A2|A3|A4|A5/summary_*.json
# A4 reuses A1 evidence but disables planning.
# A5 reuses A3 evidence and runs vanilla Qwen3-8B.
# Explicit TOPK_DIR, OUT_DIR, MODEL etc. still override defaults.
# ============================================================
BENCH_ROOT="${BENCH_ROOT:?Set BENCH_ROOT to the experiment artifact root}"
QUERY_DIR="${QUERY_DIR:-$BENCH_ROOT/query_structure}"
VARIANT="${VARIANT:-A1}"
case "$VARIANT" in
  A1) DEFAULT_TOPK_VARIANT=A1; DEFAULT_USE_PLAN=1 ;;
  A2) DEFAULT_TOPK_VARIANT=A2; DEFAULT_USE_PLAN=1 ;;
  A3) DEFAULT_TOPK_VARIANT=A3; DEFAULT_USE_PLAN=1 ;;
  A4) DEFAULT_TOPK_VARIANT=A1; DEFAULT_USE_PLAN=0 ;;
  A5) DEFAULT_TOPK_VARIANT=A3; DEFAULT_USE_PLAN=1 ;;
  *) echo "VARIANT must be A1, A2, A3, A4 or A5" >&2; exit 15 ;;
esac
TOPK_DIR="${TOPK_DIR:-$BENCH_ROOT/topk_$DEFAULT_TOPK_VARIANT}"
SUMMARY_RUN_NAME="${SUMMARY_RUN_NAME:-summary_$VARIANT}"
OUT_DIR="${OUT_DIR:-$BENCH_ROOT/summary_$VARIANT}"

START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-947}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
QUERY_MODE="${QUERY_MODE:-original}"
USE_PLAN="${USE_PLAN:-$DEFAULT_USE_PLAN}"
case "$QUERY_MODE" in
  structured|original) ;;
  *) echo "QUERY_MODE must be structured or original" >&2; exit 15 ;;
esac
case "$USE_PLAN" in
  0|1) ;;
  *) echo "USE_PLAN must be 1 or 0" >&2; exit 15 ;;
esac

# ============================================================
# Select an S3 model for A1–A4 or a vanilla Qwen3 model for A5 explicitly.
# ============================================================
MODEL="${MODEL:?Set MODEL to the summary model directory}"
BASE_TOKENIZER="${BASE_TOKENIZER:-$MODEL}"

# Keep original inference budgets/batch settings unchanged.
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-600}"
MAX_CORE="${MAX_CORE:-5}"
MAX_SUPPORT="${MAX_SUPPORT:-5}"
MAX_PACKED="${MAX_PACKED:-18}"
SUMMARY_BATCH_SIZE="${SUMMARY_BATCH_SIZE:-4}"
DISABLE_QWEN_THINKING="${DISABLE_QWEN_THINKING:-1}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT="${SCRIPT:-$WORKDIR/summary_range.py}"

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
export RUN_ID OUT_DIR QUERY_MODE USE_PLAN
exec 3>&1 4>&2
LOGGER_PID=""
LOG_PIPE="$RUN_DIR/.summary_range_log.pipe"

write_shell_summary() {
  local exit_code=$?
  trap - EXIT
  set +e
  "$HOST_PYTHON" - "$RUN_DIR" "$exit_code" "$SHELL_STARTED_EPOCH" \
    "${CONTAINER_STARTED_EPOCH:-}" "${CONTAINER_FINISHED_EPOCH:-}" \
    "${CONTAINER_EXIT_CODE:-}" <<'PY'
import json
import os
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
    "stage": "summary",
    "query_mode": os.environ["QUERY_MODE"],
    "use_plan": os.environ["USE_PLAN"] == "1",
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
    "python_summary": str(run_dir / "summary_range_summary.json"),
    "log": str(run_dir / "summary_range.log"),
    "measurement_notes": [
        "Shell total includes setup, modules, container startup, sanity checks and Python execution.",
        "Container latency includes startup, sanity checks and Python execution.",
        "Nested times overlap; do not add shell, container and Python stage times together.",
        "Timing stops before shell-summary writing and shell teardown.",
        "Shell timings use wall-clock epoch seconds; Python timings use a monotonic clock.",
    ],
}
path = run_dir / "summary_range_summary.json"
if path.exists():
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        summary["python_status"] = result.get("status")
        for key in ("input_tokens", "output_tokens", "total_tokens", "token_counts_complete"):
            summary[key] = result.get(key)
    except (OSError, ValueError) as exc:
        summary["python_summary_read_error"] = str(exc)
destination = run_dir / "summary_range_shell_summary.json"
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
tee "$RUN_DIR/summary_range.log" <"$LOG_PIPE" >&3 &
LOGGER_PID=$!
exec >"$LOG_PIPE" 2>&1
rm -f "$LOG_PIPE"
echo "[INFO] RUN_ID             : $RUN_ID"
echo "[INFO] Metrics directory  : $RUN_DIR"

# ============================================================
# Caches
# ============================================================
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

# ============================================================
# Modules / container
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
# Host-side checks
# ============================================================
[[ -f "$IMAGE" ]] || { echo "❌ Missing image: $IMAGE" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { echo "❌ Missing WORKDIR: $WORKDIR" >&2; exit 3; }
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "❌ Missing PKG_ENV: $PKG_ENV" >&2; exit 4; }
[[ -z "$PKG_MARKER" || -f "$PKG_MARKER" ]] || { echo "❌ Missing PKG marker: $PKG_MARKER" >&2; exit 5; }

[[ -d "$BENCH_ROOT" ]] || { echo "❌ Missing BENCH_ROOT: $BENCH_ROOT" >&2; exit 6; }
if [[ "$QUERY_MODE" == "structured" ]]; then
  [[ -d "$QUERY_DIR" ]] || { echo "❌ Missing QUERY_DIR: $QUERY_DIR" >&2; exit 7; }
fi
[[ -d "$TOPK_DIR" ]] || { echo "❌ Missing TOPK_DIR: $TOPK_DIR" >&2; exit 8; }

[[ -d "$MODEL" ]] || { echo "❌ Missing MODEL dir: $MODEL" >&2; exit 9; }
[[ -d "$BASE_TOKENIZER" ]] || { echo "❌ Missing BASE_TOKENIZER dir: $BASE_TOKENIZER" >&2; exit 10; }
[[ -f "$SCRIPT" ]] || { echo "❌ Missing SCRIPT: $SCRIPT" >&2; exit 11; }
SUMMARY_MODULE_DIR="$(cd -- "$(dirname -- "$SCRIPT")" && pwd -P)"
SCRIPT="$SUMMARY_MODULE_DIR/$(basename -- "$SCRIPT")"
[[ -f "$SUMMARY_MODULE_DIR/summary.py" ]] || {
  echo "❌ Missing summary.py: $SUMMARY_MODULE_DIR/summary.py" >&2
  echo "Place summary.py alongside summary_range.py." >&2
  exit 14
}

QUERY_COUNT=0
if [[ -d "$QUERY_DIR" ]]; then
  QUERY_COUNT="$(find "$QUERY_DIR" -maxdepth 1 -name 'query_*.json' | wc -l)"
fi
TOPK_COUNT="$(find "$TOPK_DIR" -maxdepth 1 -name 'topk_*.json' | wc -l)"

if [[ "$QUERY_MODE" == "structured" && "$QUERY_COUNT" -eq 0 ]]; then
  echo "❌ QUERY_DIR has 0 query_*.json files: $QUERY_DIR" >&2
  exit 12
fi

if [[ "$TOPK_COUNT" -eq 0 ]]; then
  echo "❌ TOPK_DIR has 0 topk_*.json files: $TOPK_DIR" >&2
  exit 13
fi

SOURCE_ROOT="$(cd -- "$SUMMARY_MODULE_DIR/.." && pwd -P)"
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
for name in SOURCE_ROOT WORKDIR SCRIPT SUMMARY_MODULE_DIR PKG_ENV BENCH_ROOT QUERY_DIR \
  TOPK_DIR OUT_DIR MODEL BASE_TOKENIZER SAMPLE_LIST_PATH PIP_CACHE_DIR \
  XDG_CACHE_HOME HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRITON_CACHE_DIR TMPDIR; do
  bind_configured_path "$name"
done

if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
  BIND_ARGS+=(--bind "$SLURM_TMPDIR:$SLURM_TMPDIR")
fi

# Use the runtime's default certificate configuration.

echo "[INFO] Container cmd         : $CTR_CMD"
echo "[INFO] Image                 : $IMAGE"
echo "[INFO] Workdir               : $WORKDIR"
echo "[INFO] PKG_ENV               : $PKG_ENV"
echo "[INFO] Script                : $SCRIPT"
echo "[INFO] Summary module        : $SUMMARY_MODULE_DIR/summary.py"
echo "[INFO] VARIANT               : $VARIANT"
echo "[INFO] BENCH_ROOT            : $BENCH_ROOT"
echo "[INFO] QUERY_DIR             : $QUERY_DIR"
echo "[INFO] TOPK_DIR              : $TOPK_DIR"
echo "[INFO] OUT_DIR               : $OUT_DIR"
echo "[INFO] QUERY_COUNT           : $QUERY_COUNT"
echo "[INFO] TOPK_COUNT            : $TOPK_COUNT"
echo "[INFO] START_IDX             : $START_IDX"
echo "[INFO] END_IDX               : $END_IDX"
echo "[INFO] SKIP_EXISTING         : $SKIP_EXISTING"
echo "[INFO] MODEL                 : $MODEL"
echo "[INFO] BASE_TOKENIZER        : $BASE_TOKENIZER"
echo "[INFO] MAX_OUTPUT_TOKENS     : $MAX_OUTPUT_TOKENS"
echo "[INFO] MAX_CORE              : $MAX_CORE"
echo "[INFO] MAX_SUPPORT           : $MAX_SUPPORT"
echo "[INFO] MAX_PACKED            : $MAX_PACKED"
echo "[INFO] SUMMARY_BATCH_SIZE    : $SUMMARY_BATCH_SIZE"
echo "[INFO] QUERY_MODE            : $QUERY_MODE"
echo "[INFO] USE_PLAN              : $USE_PLAN"
echo "[INFO] DISABLE_QWEN_THINKING : $DISABLE_QWEN_THINKING"
echo "[INFO] ATTN_IMPL             : $ATTN_IMPL"
echo "[INFO] CUDA_VISIBLE_DEVICES  : $CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_TMPDIR          : ${SLURM_TMPDIR:-<empty>}"

CONTAINER_STARTED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"
set +e
"$CTR_CMD" exec --nv \
  "${BIND_ARGS[@]}" \
  --env SLURM_TMPDIR="${SLURM_TMPDIR:-}" \
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
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env WORKDIR="$WORKDIR" \
  --env PKG_ENV="$PKG_ENV" \
  --env SCRIPT="$SCRIPT" \
  --env SUMMARY_MODULE_DIR="$SUMMARY_MODULE_DIR" \
  --env BENCH_ROOT="$BENCH_ROOT" \
  --env QUERY_DIR="$QUERY_DIR" \
  --env TOPK_DIR="$TOPK_DIR" \
  --env OUT_DIR="$OUT_DIR" \
  --env RUN_ID="$RUN_ID" \
  --env SUMMARY_RUN_NAME="$SUMMARY_RUN_NAME" \
  --env START_IDX="$START_IDX" \
  --env END_IDX="$END_IDX" \
  --env SAMPLE_LIST_PATH="${SAMPLE_LIST_PATH:-}" \
  --env SKIP_EXISTING="$SKIP_EXISTING" \
  --env MODEL="$MODEL" \
  --env BASE_TOKENIZER="$BASE_TOKENIZER" \
  --env MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
  --env MAX_CORE="$MAX_CORE" \
  --env MAX_SUPPORT="$MAX_SUPPORT" \
  --env MAX_PACKED="$MAX_PACKED" \
  --env SUMMARY_BATCH_SIZE="$SUMMARY_BATCH_SIZE" \
  --env QUERY_MODE="$QUERY_MODE" \
  --env USE_PLAN="$USE_PLAN" \
  --env DISABLE_QWEN_THINKING="$DISABLE_QWEN_THINKING" \
  --env ATTN_IMPL="$ATTN_IMPL" \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$WORKDIR"

    unset PYTHONSTARTUP PYTHONHOME PYTHONPATH
    export PYTHONPATH="$SUMMARY_MODULE_DIR:$WORKDIR${PKG_ENV:+:$PKG_ENV}"

    echo "[INFO] inside container"
    echo "[INFO] pwd=$PWD"
    echo "[INFO] PYTHONPATH=$PYTHONPATH"
    echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[INFO] BENCH_ROOT=$BENCH_ROOT"
    echo "[INFO] QUERY_DIR=$QUERY_DIR"
    echo "[INFO] TOPK_DIR=$TOPK_DIR"
    echo "[INFO] OUT_DIR=$OUT_DIR"
    echo "[INFO] MODEL=$MODEL"
    echo "[INFO] BASE_TOKENIZER=$BASE_TOKENIZER"
    echo "[INFO] SUMMARY_BATCH_SIZE=$SUMMARY_BATCH_SIZE"
    echo "[INFO] QUERY_MODE=$QUERY_MODE"
    echo "[INFO] USE_PLAN=$USE_PLAN"
    echo "[INFO] DISABLE_QWEN_THINKING=$DISABLE_QWEN_THINKING"
    echo "[INFO] ATTN_IMPL=$ATTN_IMPL"

    python - <<PY
import os
import torch
import transformers
import tokenizers
from pathlib import Path
import summary
import inspect
from summary import load_local_model, build_summary_prompt, build_evidence_lines, parse_summary_json

print("python ok")
print("summary module:", summary.__file__)
if not {"query_mode", "use_plan"}.issubset(inspect.signature(build_summary_prompt).parameters):
    raise RuntimeError("Update summary.py alongside summary_range.py: missing ablation prompt options")
if "use_plan" not in inspect.signature(parse_summary_json).parameters:
    raise RuntimeError("Update summary.py alongside summary_range.py: missing no-plan parser")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
print("transformers:", transformers.__version__, transformers.__file__)
print("tokenizers:", tokenizers.__version__, tokenizers.__file__)
print("MODEL:", os.environ.get("MODEL"))
print("BASE_TOKENIZER:", os.environ.get("BASE_TOKENIZER"))
print("PKG_ENV:", os.environ.get("PKG_ENV"))
print("BENCH_ROOT:", os.environ.get("BENCH_ROOT"))
print("QUERY_DIR:", os.environ.get("QUERY_DIR"))
print("TOPK_DIR:", os.environ.get("TOPK_DIR"))
print("OUT_DIR:", os.environ.get("OUT_DIR"))
print("SUMMARY_BATCH_SIZE:", os.environ.get("SUMMARY_BATCH_SIZE"))
print("QUERY_MODE:", os.environ.get("QUERY_MODE"))
print("USE_PLAN:", os.environ.get("USE_PLAN"))

query_dir = Path(os.environ["QUERY_DIR"])
topk_dir = Path(os.environ["TOPK_DIR"])
print("query files:", len(list(query_dir.glob("query_*.json"))))
print("topk files:", len(list(topk_dir.glob("topk_*.json"))))
PY

    python -u "$SCRIPT"
  '

CONTAINER_EXIT_CODE=$?
CONTAINER_FINISHED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"
set -e
if [[ "$CONTAINER_EXIT_CODE" -eq 0 ]]; then
  echo "[OK] summary range finished: $OUT_DIR"
fi
exit "$CONTAINER_EXIT_CODE"
