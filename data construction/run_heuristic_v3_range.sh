#!/usr/bin/env bash
# Run heuristic_range_v3.py through Apptainer, recording latency only.
set -euo pipefail

HOST_PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$HOST_PYTHON" ]] || { echo "[ERR] host Python is required for timing" >&2; exit 98; }
SHELL_STARTED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"

# Use the script's own directory, including after copying it to another server.
RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"
WORKDIR="${WORKDIR:-$RUNNER_DIR}"
SCRIPT="${SCRIPT:-$WORKDIR/heuristic_range_v3.py}"
HEURISTIC_PKG_DIR="${HEURISTIC_PKG_DIR:-}"
HEURISTIC_MARKER="${HEURISTIC_MARKER:-}"

START_IDX="${START_IDX:-0}"
# END_IDX is inclusive. ASQA train has 4,353 records (indices 0..4352).
END_IDX="${END_IDX:-4352}"
# ASQA TRAIN teacher-data pipeline: 4,353 train questions with GTR top-5 docs.
TEST_TEACHER_ROOT="${TEST_TEACHER_ROOT:-}"
SRC_JSON="${SRC_JSON:?Set SRC_JSON to the input dataset file}"
QSTRUCT_DIR="${QSTRUCT_DIR:?Set QSTRUCT_DIR to the query-stage output directory}"
OUT_DIR_WAS_SET="${OUT_DIR:+1}"
# First 10 and the remaining 4,343 use this SAME production directory (no smoke directory).
OUT_DIR="${OUT_DIR:-${TEST_TEACHER_ROOT:?Set OUT_DIR or TEST_TEACHER_ROOT}/asqa_train_heuristic/candidate_pool}"
# Validate all requested query_<idx>.json against the exact source record before GPU work.
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
RERANK_MODEL="${RERANK_MODEL:?Set RERANK_MODEL to the local reranker model directory}"

# Same candidate-selection settings as the supplied Bash.
CANDIDATES="${CANDIDATES:-120}"
FINAL_SLOTS="${FINAL_SLOTS:-18}"
MIN_CORE="${MIN_CORE:-5}"
MIN_PARALLEL="${MIN_PARALLEL:-8}"
CORE_DOC_CAP="${CORE_DOC_CAP:-3}"
MAX_INPUT_PASSAGES="${MAX_INPUT_PASSAGES:-5}"
FINAL_TOPK_PASSAGES="${FINAL_TOPK_PASSAGES:-5}"
SIM_WINDOW="${SIM_WINDOW:-1}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-5}"
RERANK_MAX_LENGTH="${RERANK_MAX_LENGTH:-${MAX_LENGTH:-2048}}"
SURVIVAL_MIN="${SURVIVAL_MIN:-0.18}"
NOISE_FOR_DROP="${NOISE_FOR_DROP:-0.30}"
MIN_TOTAL_KEEP="${MIN_TOTAL_KEEP:-0.05}"
# Reuse successful initial 0..9 outputs when expanding to 0..4352.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Defaults preserve historical runs. Use fixed for BOTH A1 and A2 comparisons.
CANDIDATE_MODE="${CANDIDATE_MODE:-full}"
BUDGET_MODE="${BUDGET_MODE:-legacy}"
if [[ -z "$OUT_DIR_WAS_SET" && ( "$CANDIDATE_MODE" != "full" || "$BUDGET_MODE" != "legacy" ) ]]; then
  OUT_DIR="${OUT_DIR}_${CANDIDATE_MODE}_${BUDGET_MODE}"
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"
RUN_ID="${RUN_ID:-$("$HOST_PYTHON" -c 'from datetime import datetime, timezone; import uuid; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])')}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "[ERR] invalid RUN_ID" >&2; exit 11; }
RUN_DIR="$OUT_DIR/metrics/$RUN_ID"
mkdir -p "$OUT_DIR/metrics"
mkdir "$RUN_DIR" || { echo "[ERR] RUN_ID already exists: $RUN_DIR" >&2; exit 12; }
export RUN_ID OUT_DIR

exec 3>&1 4>&2
LOGGER_PID=""
LOG_PIPE="$RUN_DIR/.heuristic_log.pipe"
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

directory = Path(sys.argv[1])
exit_code, started, finished = int(sys.argv[2]), float(sys.argv[3]), time.time()
container_start = float(sys.argv[4]) if sys.argv[4] else None
container_end = float(sys.argv[5]) if sys.argv[5] else finished
summary = {
    "run_id": directory.name,
    "stage": "heuristic_candidate_construction",
    "status": "completed" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "container_exit_code": int(sys.argv[6]) if sys.argv[6] else None,
    "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "out_dir": str(directory.parent.parent),
    "latency_seconds": {
        "shell_total_seconds": finished - started,
        "setup_seconds": (container_start if container_start is not None else finished) - started,
        "container_seconds": container_end - container_start if container_start is not None else None,
    },
    "python_summary": str(directory / "heuristic_range_summary.json"),
    "log": str(directory / "heuristic_range.log"),
    "measurement_notes": [
        "Shell total includes setup, modules, container startup, checks and Python execution.",
        "Container time includes startup, sanity imports and Python execution.",
        "Shell uses wall-clock epoch timestamps; Python uses a monotonic clock.",
        "Nested times overlap. Do not add shell, container, Python and sample times together.",
        "Shell timing excludes final shell-summary writing and logger teardown.",
    ],
}
python_path = directory / "heuristic_range_summary.json"
if python_path.is_file():
    try:
        result = json.loads(python_path.read_text(encoding="utf-8"))
        summary["python_status"] = result.get("status")
        summary["python_latency_seconds"] = result.get("latency_seconds")
        for key in ("samples_saved", "samples_skipped", "samples_missing", "samples_failed"):
            summary[key] = result.get(key)
    except (OSError, ValueError) as exc:
        summary["python_summary_read_error"] = str(exc)
destination = directory / "heuristic_range_shell_summary.json"
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"[LATENCY] shell_total_seconds={finished - started:.6f} exit_code={exit_code}")
print(f"[METRICS] {destination}")
PY
  local metrics_code=$?
  if [[ "$metrics_code" -ne 0 ]]; then
    echo "[WARN] shell summary could not be saved (status=$metrics_code)" >&2
  fi
  exec 1>&3 2>&4
  if [[ -n "$LOGGER_PID" ]]; then
    wait "$LOGGER_PID" || echo "[WARN] log writer failed" >&2
  fi
  rm -f "$LOG_PIPE"
  exec 3>&- 4>&-
  exit "$exit_code"
}
trap write_shell_summary EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkfifo "$LOG_PIPE"
tee "$RUN_DIR/heuristic_range.log" <"$LOG_PIPE" >&3 &
LOGGER_PID=$!
exec >"$LOG_PIPE" 2>&1
rm -f "$LOG_PIPE"
echo "[INFO] RUN_ID=$RUN_ID"
echo "[INFO] TEST_TEACHER_ROOT=$TEST_TEACHER_ROOT"
echo "[INFO] Metrics directory=$RUN_DIR"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MYPROJ/pip_cache_qwen3}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MYPROJ/xdg_cache_qwen3}"
export HF_HOME="${HF_HOME:-$MYPROJ/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SLURM_TMPDIR="${SLURM_TMPDIR:-$MYPROJ/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$SLURM_TMPDIR"

# Load the container runtime in the caller environment.
if command -v apptainer >/dev/null 2>&1; then
  CTR_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CTR_CMD="singularity"
else
  echo "[ERR] missing apptainer/singularity" >&2
  exit 99
fi

[[ -f "$IMAGE" ]] || { echo "[ERR] missing image: $IMAGE" >&2; exit 2; }
[[ -d "$WORKDIR" ]] || { echo "[ERR] missing WORKDIR: $WORKDIR" >&2; exit 3; }
[[ -z "$HEURISTIC_PKG_DIR" || -d "$HEURISTIC_PKG_DIR" ]] || { echo "[ERR] missing environment: $HEURISTIC_PKG_DIR" >&2; exit 4; }
[[ -z "$HEURISTIC_MARKER" || -f "$HEURISTIC_MARKER" ]] || { echo "[ERR] missing environment marker: $HEURISTIC_MARKER" >&2; exit 4; }
[[ -f "$SRC_JSON" ]] || { echo "[ERR] missing SRC_JSON: $SRC_JSON" >&2; exit 5; }
[[ -d "$QSTRUCT_DIR" ]] || { echo "[ERR] missing QSTRUCT_DIR: $QSTRUCT_DIR" >&2; exit 6; }
[[ -f "$SCRIPT" ]] || { echo "[ERR] missing range script: $SCRIPT" >&2; exit 7; }
[[ -d "$RERANK_MODEL" ]] || { echo "[ERR] missing model: $RERANK_MODEL" >&2; exit 9; }
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT")" && pwd -P)"
SCRIPT="$SCRIPT_DIR/$(basename "$SCRIPT")"
[[ -f "$SCRIPT_DIR/heuristic_v3.py" ]] || { echo "[ERR] missing heuristic_v3.py in $SCRIPT_DIR" >&2; exit 8; }
[[ -f "$SCRIPT_DIR/qwen_relevance.py" ]] || { echo "[ERR] missing qwen_relevance.py in $SCRIPT_DIR" >&2; exit 8; }

# Preflight uses only the host Python standard library; no model load or GPU needed.
if [[ "$VALIDATE_INPUTS" == "1" ]]; then
  "$HOST_PYTHON" - "$SRC_JSON" "$QSTRUCT_DIR" "$START_IDX" "$END_IDX" <<'PY_VALIDATE'
import json
import re
import sys
from pathlib import Path

source, qdir = Path(sys.argv[1]), Path(sys.argv[2])
start, end = int(sys.argv[3]), int(sys.argv[4])
records = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(records, list) or not (0 <= start <= end < len(records)):
    raise SystemExit(f"[PREFLIGHT ERROR] Invalid inclusive index range {start}..{end}; dataset length={len(records) if isinstance(records, list) else 'not a list'}")

problems = []
for i in range(start, end + 1):
    path = qdir / f"query_{i}.json"
    if not path.is_file():
        problems.append(f"{i}: missing {path}")
        continue
    try:
        q = json.loads(path.read_text(encoding="utf-8"))
        source_record = records[i]
        if not isinstance(q, dict) or not isinstance(q.get("query_struct"), dict):
            raise ValueError("query_struct missing or not a JSON object")
        if not isinstance(source_record, dict):
            raise ValueError("dataset record is not an object")
        source_query = (source_record.get("query") or source_record.get("question") or "").strip()
        stored_query = (q.get("query") or "").strip()
        norm = lambda x: re.sub(r"\s+", " ", x)
        if not source_query or norm(stored_query) != norm(source_query):
            raise ValueError(f"query mismatch: data={source_query[:90]!r}, saved={stored_query[:90]!r}")
        if q.get("sample_idx", i) != i:
            raise ValueError(f"sample_idx mismatch: {q.get('sample_idx')!r}")
    except (OSError, ValueError, TypeError) as exc:
        problems.append(f"{i}: {exc}")

if problems:
    for error in problems[:20]:
        print(f"[PREFLIGHT ERROR] {error}", file=sys.stderr)
    raise SystemExit(f"[PREFLIGHT ERROR] {len(problems)} / {end-start+1} query structures missing or incompatible; not starting GPU work")
print(f"[PREFLIGHT OK] {end-start+1} aligned samples ({start}..{end}), dataset={source}, qstruct={qdir}")
PY_VALIDATE
fi

# Canonical paths also make externally supplied data/output locations bindable.
WORKDIR="$(cd "$WORKDIR" && pwd -P)"
QSTRUCT_DIR="$(cd "$QSTRUCT_DIR" && pwd -P)"
SOURCE_DIR="$(cd "$(dirname "$SRC_JSON")" && pwd -P)"
SRC_JSON="$SOURCE_DIR/$(basename "$SRC_JSON")"
RERANK_MODEL="$(cd "$RERANK_MODEL" && pwd -P)"
if [[ -n "$HEURISTIC_PKG_DIR" ]]; then
  HEURISTIC_PKG_DIR="$(cd "$HEURISTIC_PKG_DIR" && pwd -P)"
fi
SOURCE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
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
for name in SOURCE_ROOT WORKDIR SCRIPT SCRIPT_DIR HEURISTIC_PKG_DIR SOURCE_DIR QSTRUCT_DIR \
  OUT_DIR RERANK_MODEL SLURM_TMPDIR PIP_CACHE_DIR XDG_CACHE_HOME HF_HOME \
  HF_DATASETS_CACHE HF_HUB_CACHE; do
  bind_configured_path "$name"
done
ENV_ARGS=()
for name in WORKDIR SCRIPT SCRIPT_DIR HEURISTIC_PKG_DIR START_IDX END_IDX \
  SRC_JSON QSTRUCT_DIR OUT_DIR RUN_ID RERANK_MODEL CANDIDATES FINAL_SLOTS \
  MIN_CORE MIN_PARALLEL CORE_DOC_CAP MAX_INPUT_PASSAGES FINAL_TOPK_PASSAGES \
  SIM_WINDOW RERANK_BATCH_SIZE RERANK_MAX_LENGTH SURVIVAL_MIN NOISE_FOR_DROP \
  MIN_TOTAL_KEEP SKIP_EXISTING SLURM_TMPDIR PIP_CACHE_DIR XDG_CACHE_HOME \
  HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE CANDIDATE_MODE BUDGET_MODE; do
  ENV_ARGS+=(--env "$name=${!name}")
  echo "[INFO] $name=${!name}"
done
if [[ -n "${RERANK_INSTRUCTION:-}" ]]; then
  ENV_ARGS+=(--env "RERANK_INSTRUCTION=$RERANK_INSTRUCTION")
fi
# Use the runtime's default certificate configuration.

CONTAINER_STARTED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"
set +e
"$CTR_CMD" exec --nv "${BIND_ARGS[@]}" "${ENV_ARGS[@]}" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_MAX_WORKERS=1 \
  --env HF_HUB_ENABLE_HF_TRANSFER=0 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 --env TOKENIZERS_PARALLELISM=false \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$WORKDIR"
    unset PYTHONSTARTUP PYTHONHOME PYTHONPATH
    export PYTHONPATH="$SCRIPT_DIR:$WORKDIR${HEURISTIC_PKG_DIR:+:$HEURISTIC_PKG_DIR}"
    echo "[INFO] inside container pwd=$(pwd)"
    python - <<PY
import inspect, sys, transformers
from heuristic_v3 import build_candidate_pool
print("python:", sys.executable)
print("transformers:", transformers.__version__)
print("heuristic source:", inspect.getfile(build_candidate_pool))
PY
    python -u "$SCRIPT"
  '
CONTAINER_EXIT_CODE=$?
CONTAINER_FINISHED_EPOCH="$("$HOST_PYTHON" -c 'import time; print(time.time())')"
set -e
exit "$CONTAINER_EXIT_CODE"
