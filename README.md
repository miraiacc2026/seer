# SEER: Source Code (Anonymized for Review)

This repository contains the core pipeline code for SEER, a three-stage
evidence pipeline for citation-grounded question answering:

1. **Query**: turn the question into a structured query (entities, relations, scope, retrieval views).
2. **Rerank**: build a high-recall sentence candidate pool with a heuristic, then label candidates as core, support or drop with a rerank model.
3. **Summary**: write a plan and a final cited answer from the selected evidence.

Datasets, model checkpoints, credentials and experiment outputs are not included.
All locations are supplied by the caller. No script has a built-in host path.

## Layout

```
data construction/   Teacher data generation, filtering and gold-bundle collection
train/               SFT data builders and FSDP training for the three stages
inference/           Range runners for each stage and the evaluation merger
data/                Training sets and the MedCite test set (see "Data and checkpoints")
checkpoints/         Fine-tuned Query, Rerank and Summary checkpoints
seer_config.py       Shared helpers: required-path checks, API selection, lazy imports
install_env.sh       Installs the pinned Python package overlay (see ENVIRONMENT.md)
ENVIRONMENT.md       Observed software environment and package versions
```

## Requirements

- Apptainer or Singularity, and an NVIDIA PyTorch 24.09-py3 container image (Python 3.10, PyTorch 2.5, CUDA 12.6).
- One or more NVIDIA GPUs. Training scripts default to 2 GPUs (`NPROC_PER_NODE=2`).
- Local copies of Qwen3-8B (all three stages) and Qwen3-Reranker-4B (heuristic relevance scoring).
- The ALCE benchmark source (`eval.py`) for citation scoring and evaluation.
- An OpenAI-compatible API only for teacher-data generation in `data construction/`.

Python packages are installed into a separate directory that the launchers
put on `PYTHONPATH`. PyTorch and CUDA come from the container. Load your
container runtime first, then:

```bash
export CONTAINER_IMAGE=/path/to/nv-pytorch_24.09-py3.sif
export PKG_ENV=/path/to/new/package_dir   # must not already exist
bash install_env.sh
```

Pass `PKG_ENV` to the launchers afterwards. Details are in `ENVIRONMENT.md`.

## Configuration conventions

- Every launcher requires `MYPROJ` (a writable cache root) and the container image (`IMAGE`, or `CONTAINER_IMAGE` for `query_range.sh`).
- Input, output and model paths are environment variables, or CLI arguments for the Python tools. A missing required path stops the run before anything is written.
- Hyperparameters (batch size, token budgets, top-k, learning rate, epochs) are exposed as environment variables with defaults in each launcher.
- Sample ranges are inclusive: `START_IDX`, `END_IDX`.
- API endpoints come from `OPENAI_BASE_URL` or `PROXY_BASE_URL`. Credentials are `OPENAI_API_KEY` and `PROXY_API_KEY`, kept separate. Native Batch commands also need `API_SUPPORTS_BATCH=1`. Use `run_chat` for endpoints without Batch.

## Data and checkpoints

Entry points for the released data and models. Paths are relative to the
repository root; export them as absolute paths for the launchers.

### Data (`data/`)

The training data is hosted in an anonymous Google Drive folder:
<https://drive.google.com/drive/folders/1pnprTfDkICvcxVKw8MTcb1B6os216gbd>

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1pnprTfDkICvcxVKw8MTcb1B6os216gbd" -O data/train
```

| Path | Content | Used by |
|---|---|---|
| `data/train/asqa_query_s3_train.json`, `..._val.json` | Query-stage SFT set and validation split | `train/run_query_sft.sh` (`DATA_PATH`) |
| `data/train/asqa_r3_rerank_train.json`, `..._val.json` | Rerank-stage SFT set and validation split | `train/run_rerank_sft.sh` (`DATA_PATH`) |
| `data/train/asqa_s3_summary_train.json`, `..._val.json` | Summary-stage SFT set and validation split | `train/run_summary_sft.sh` (`DATA_PATH`) |
| `data/medcite/` | MedCite test set | `SRC_JSON` of the inference runners |

The three training files are the outputs of `train/build_*_sft_*.py`, in the
`{"examples": [...]}` format. The inference runners read the test set as a JSON
list (or an object with a `data` or `examples` list) of records. Each record
needs a `question` (or `query`) and its retrieved passages in `docs`.

```bash
export SRC_JSON=$PWD/data/medcite/<test_file>.json
export DATA_PATH=$PWD/data/train/asqa_query_s3_train.json
```

### Checkpoints (`checkpoints/`)

| Path | Stage | Used by |
|---|---|---|
| `checkpoints/query/` | Query structuring | `MODEL` in `inference/query_range.sh` |
| `checkpoints/rerank/` | Evidence reranking | `MODEL` in `inference/run_rerank_range.sh` |
| `checkpoints/summary/` | Plan and final summary | `MODEL` in `inference/run_summary_range.sh` |

All three are fine-tuned from Qwen3-8B. The inference runners take the
tokenizer from `BASE_TOKENIZER`, which defaults to `MODEL`. Set it to a local
Qwen3-8B directory only if a checkpoint directory has no tokenizer files.
Set `MODEL` to the checkpoint directory of the stage being run:

```bash
export MODEL=$PWD/checkpoints/query
export BASE_TOKENIZER=/path/to/Qwen3-8B   # optional
```

Qwen3-8B and Qwen3-Reranker-4B are not redistributed here.

## Pipeline

### 1. Teacher data (`data construction/`)

Run in this order. Each stage writes to a new directory and never modifies its inputs.

| Step | Script | Purpose |
|---|---|---|
| Query | `asqa_query_batch.sh` | Query-structure labels from the teacher model |
| Candidates | `run_heuristic_v3_range.sh` | Heuristic sentence candidate pool |
| Rerank | `rerank_topk_asqa.sh` | Core/support/drop labels from the teacher model |
| Summary | `asqa_summary_batch.sh` | Plan and final summary from the teacher model |
| Filter | `asqa_teacher_citation_scores.sh`, then `asqa_teacher_filter_v2.py filter` | NLI citation scoring and staged filtering |
| Collect | `collect_asqa_gold_data.py` | Bundle the filtered samples by original `sample_idx` |

`prepare_asqa_train_query_input.py` and `merge_asqa_train_teacher_audit_eval.py` are helpers for input conversion and audit evaluation.

### 2. Training (`train/`)

Build SFT data for each stage, then train:

```bash
python train/build_rerank_sft_r3.py  --gold-root <bundle> --output-dir <sft_dir>
python train/build_query_sft_s3.py   --gold-root <bundle> --output-dir <query_sft_dir> --sft-root <sft_dir>
python train/build_summary_sft_s3.py --gold-root <bundle> --output-dir <summary_sft_dir> \
  --rerank-split-dir <sft_dir>

MODEL=<qwen3_8b_dir> DATA_PATH=<train.json> OUTPUT_ROOT=<ckpt_dir> \
  MYPROJ=<cache_root> IMAGE=<image.sif> bash train/run_query_sft.sh
# likewise run_rerank_sft.sh, run_summary_sft.sh
```

All three stages share the same validation split. Training uses FSDP via the
container and can be launched on one node with `NPROC_PER_NODE`.

### 3. Inference (`inference/`)

Run the stages in order; each reads the previous stage's output directory.

```bash
# Query
MODEL=<query_ckpt> BASE_TOKENIZER=<qwen3_8b_dir> SRC_JSON=<dev.json> OUT_DIR=<query_dir> \
  MYPROJ=<cache_root> CONTAINER_IMAGE=<image.sif> bash inference/query_range.sh

# Candidate pool
SRC_JSON=<dev.json> QSTRUCT_DIR=<query_dir> OUT_DIR=<pool_dir> RERANK_MODEL=<reranker_dir> \
  MYPROJ=<cache_root> IMAGE=<image.sif> bash inference/run_heuristic_v3_range.sh

# Rerank
CANDIDATE_DIR=<pool_dir> MODEL=<rerank_ckpt> OUT_DIR=<topk_dir> \
  MYPROJ=<cache_root> IMAGE=<image.sif> bash inference/run_rerank_range.sh

# Summary (VARIANT selects the ablation setting, A1 to A5)
BENCH_ROOT=<artifact_root> MODEL=<summary_ckpt> VARIANT=A1 \
  MYPROJ=<cache_root> IMAGE=<image.sif> bash inference/run_summary_range.sh

# Merge into ALCE format
python inference/merge_asqa_summary_eval.py --help
```

Each runner records timing and token-count metrics under `<OUT_DIR>/metrics/`.
The merger fails on missing predictions by default; `--allow-partial` produces a diagnostic, non-comparable subset.

## Notes

- `python <script> --help` works for every Python entry point without GPU packages installed.
- Model-dependent imports are lazy, so configuration errors appear before models load.
- Evaluation uses the unmodified ALCE `eval.py`.
- Package versions beyond those in `ENVIRONMENT.md` are resolved by pip and are not pinned.
- The scripts were syntax-checked and the launchers were exercised against a mocked container runtime. End-to-end runs on GPU hardware were not repeated after anonymization.
