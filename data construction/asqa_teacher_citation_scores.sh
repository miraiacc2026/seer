#!/usr/bin/env bash
set -euo pipefail
# Run the unchanged ALCE eval.py's per-item TRUE-NLI citation logic.
# No modifications to the original eval.py / eval.sh / teacher JSON files.
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MYPROJ="${MYPROJ:?Set MYPROJ to a writable cache root}"
IMAGE="${IMAGE:?Set IMAGE to the container image file}"
PKG_ENV="${PKG_ENV:-}"
BATCHDIR="${BATCHDIR:-$RUNNER_DIR}"
ALCE_DIR="${ALCE_DIR:?Set ALCE_DIR to the ALCE source directory}"
FILTER_PY="${FILTER_PY:-$BATCHDIR/asqa_teacher_filter_v2.py}"
INPUT_JSON="${INPUT_JSON:?Set INPUT_JSON to the teacher evaluation JSON file}"
SCORES_JSONL="${SCORES_JSONL:-${OUTPUT_ROOT:?Set SCORES_JSONL or OUTPUT_ROOT}/asqa_train_teacher_s3_citation_scores.jsonl}"
AUTOAIS_MODEL="${AUTOAIS_MODEL:?Set AUTOAIS_MODEL to the local TRUE-NLI model directory}"
# QA and MAUVE are unused by score-citations; forward only caller overrides.
QA_MODEL="${QA_MODEL:-}"
MAUVE_MODEL="${MAUVE_MODEL:-}"
NLTK_DATA="${NLTK_DATA:?Set NLTK_DATA to the NLTK resource directory}"
AT_MOST_CITATIONS="${AT_MOST_CITATIONS:-3}"
LIMIT="${LIMIT:-}"

for file in "$IMAGE" "$FILTER_PY" "$INPUT_JSON" "$ALCE_DIR/eval.py"; do
  [[ -f "$file" ]] || { echo "[ERROR] Missing file: $file" >&2; exit 2; }
done
[[ -z "$PKG_ENV" || -d "$PKG_ENV" ]] || { echo "[ERROR] Missing PKG_ENV: $PKG_ENV" >&2; exit 2; }
[[ -d "$AUTOAIS_MODEL" ]] || {
  echo "[ERROR] Missing NLI model dir: $AUTOAIS_MODEL" >&2; exit 2;
}
[[ -d "$NLTK_DATA" ]] || { echo "[ERROR] Missing NLTK_DATA: $NLTK_DATA" >&2; exit 2; }
# Load the container runtime in the caller environment.
if command -v apptainer >/dev/null 2>&1; then
  CTR=apptainer
elif command -v singularity >/dev/null 2>&1; then
  CTR=singularity
else
  echo '[ERROR] apptainer/singularity not found' >&2
  exit 2
fi
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MYPROJ/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MYPROJ/xdg_cache}"
export HF_HOME="${HF_HOME:-$MYPROJ/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" \
  "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$(dirname "$SCORES_JSONL")"

# Bind each configured location, including overrides outside the cache root.
SOURCE_ROOT="$(cd -- "$(dirname -- "$FILTER_PY")/.." && pwd -P)"
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
for name in SOURCE_ROOT FILTER_PY INPUT_JSON SCORES_JSONL ALCE_DIR PKG_ENV AUTOAIS_MODEL \
  QA_MODEL MAUVE_MODEL NLTK_DATA PIP_CACHE_DIR XDG_CACHE_HOME HF_HOME \
  HF_DATASETS_CACHE HF_HUB_CACHE; do
  bind_configured_path "$name"
done

echo "[INFO] INPUT_JSON       : $INPUT_JSON"
echo "[INFO] SCORES_JSONL     : $SCORES_JSONL"
echo "[INFO] ALCE eval.py     : $ALCE_DIR/eval.py"
echo "[INFO] AT_MOST_CITATIONS: $AT_MOST_CITATIONS"
echo "[INFO] LIMIT             : ${LIMIT:-none}"

"$CTR" exec --nv \
  "${BIND_ARGS[@]}" \
  --env FILTER_PY="$FILTER_PY" \
  --env INPUT_JSON="$INPUT_JSON" \
  --env SCORES_JSONL="$SCORES_JSONL" \
  --env ALCE_DIR="$ALCE_DIR" \
  --env PKG_ENV="$PKG_ENV" \
  --env AT_MOST_CITATIONS="$AT_MOST_CITATIONS" \
  --env LIMIT="$LIMIT" \
  --env AUTOAIS_MODEL="$AUTOAIS_MODEL" \
  --env QA_MODEL="$QA_MODEL" \
  --env MAUVE_MODEL="$MAUVE_MODEL" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_MAX_WORKERS=1 \
  --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env NLTK_DATA="$NLTK_DATA" \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$ALCE_DIR"
    unset PYTHONSTARTUP PYTHONHOME PYTHONPATH
    export PYTHONPATH="$ALCE_DIR${PKG_ENV:+:$PKG_ENV}"
    extra=()
    [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")
    python "$FILTER_PY" score-citations \
      --input-json "$INPUT_JSON" \
      --eval-py "$ALCE_DIR/eval.py" \
      --scores-jsonl "$SCORES_JSONL" \
      --at-most-citations "$AT_MOST_CITATIONS" \
      "${extra[@]}"
  '
