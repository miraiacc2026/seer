#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the heuristic over a range, loading the Qwen relevance model once."""
import time

SCRIPT_STARTED = time.perf_counter()
SCRIPT_STARTED_EPOCH = time.time()

import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, env_path
torch = LazyModule("torch")

from heuristic_v3 import build_candidate_pool, load_qwen_reranker, preprocess

IMPORT_SECONDS = time.perf_counter() - SCRIPT_STARTED


def synchronize_cuda():
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


@contextmanager
def timed(values, name, cuda=False):
    """Include completed GPU work and retain elapsed time on failure."""
    if values is None:
        yield
        return
    if cuda:
        synchronize_cuda()
    started = time.perf_counter()
    try:
        yield
    finally:
        try:
            if cuda:
                synchronize_cuda()
        finally:
            values[name] = time.perf_counter() - started


def start_metrics(out_dir):
    run_id = os.environ.get("RUN_ID") or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("RUN_ID must contain only letters, digits, _, . and -")
    directory = out_dir.resolve() / "metrics" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    sample_path = directory / "heuristic_range_samples.jsonl"
    if (directory / "heuristic_range_summary.json").exists():
        raise FileExistsError(f"RUN_ID already used: {directory}")
    sample_path.touch(exist_ok=False)
    print(f"[METRICS] {directory}")
    return directory, sample_path


def finish_metrics(directory, config, samples, latency, status, error):
    saved = [row for row in samples if row["status"] == "saved"]
    summary = {
        "run_id": directory.name, "stage": "heuristic_candidate_construction",
        "status": status, "error": error, "config": config,
        "started_at_utc": datetime.fromtimestamp(SCRIPT_STARTED_EPOCH, timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_requested": config["end_idx"] - config["start_idx"] + 1,
        "samples_recorded": len(samples),
        "samples_saved": len(saved),
        "samples_skipped": sum(row["status"] == "skipped" for row in samples),
        "samples_missing": sum(row["status"] == "missing" for row in samples),
        "samples_failed": sum(row["status"] == "failed" for row in samples),
        "latency_seconds": {
            **latency,
            "samples_total_seconds": sum(row["latency_seconds"]["sample_total_seconds"] for row in samples),
            "candidate_construction_total_seconds": sum(
                row["latency_seconds"].get("candidate_construction_seconds", 0.0) for row in samples
            ),
            "mean_saved_sample_seconds": (
                sum(row["latency_seconds"]["sample_total_seconds"] for row in saved) / len(saved)
                if saved else None
            ),
            "mean_saved_candidate_construction_seconds": (
                sum(row["latency_seconds"]["candidate_construction_seconds"] for row in saved) / len(saved)
                if saved else None
            ),
            "python_stage_seconds": time.perf_counter() - SCRIPT_STARTED,
        },
        "measurement_notes": [
            "All durations are seconds. Python uses a monotonic clock.",
            "Model loading and candidate construction synchronize all visible CUDA devices.",
            "Candidate construction includes sentence preparation, Qwen relevance scoring, heuristic scoring and selection.",
            "Sample total includes query-file loading, preprocessing, candidate construction and output writing.",
            "Successful-sample means exclude missing, skipped and failed samples.",
            "Failed operations retain elapsed time; operations not reached have no timing field.",
            "Python stage includes range-module imports but excludes interpreter startup/shutdown and final summary writing.",
            "Nested durations overlap and must not be added together.",
        ],
    }
    path = directory / "heuristic_range_summary.json"
    temporary = path.with_suffix(".json.tmp")
    save_json(summary, temporary)
    temporary.replace(path)
    print(
        f"[SUMMARY] status={status} saved={len(saved)} skipped={summary['samples_skipped']} "
        f"missing={summary['samples_missing']} failed={summary['samples_failed']}"
    )
    print(f"[LATENCY] model_load_seconds={latency.get('model_load_seconds')}")
    print(f"[LATENCY] candidate_construction_total_seconds={summary['latency_seconds']['candidate_construction_total_seconds']:.6f}")
    print(f"[LATENCY] python_stage_seconds={summary['latency_seconds']['python_stage_seconds']:.6f}")
    print(f"[METRICS] {path}")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def check_existing_mode(path, options):
    """Do not overwrite or reuse another experimental condition's output."""
    if not path.is_file():
        return
    metadata = read_json(path).get("candidate_meta", {})
    previous = (metadata.get("candidate_mode", "full"), metadata.get("budget_mode", "legacy"))
    requested = (options["candidate_mode"], options["budget_mode"])
    if previous != requested:
        raise ValueError(
            f"Existing output uses {previous}, requested {requested}: {path}. "
            "Use a separate OUT_DIR (or OUT_JSON for single-sample runs)."
        )


def read_options(single=False):
    """Preserve existing environment variables and the two entry points' defaults."""
    definitions = [
        ("max_candidates", "CANDIDATES", 120, int),
        ("final_slots", "FINAL_SLOTS", 18, int),
        ("min_core", "MIN_CORE", 5, int),
        ("min_parallel", "MIN_PARALLEL", 8, int),
        ("core_doc_cap", "CORE_DOC_CAP", 3, int),
        ("max_input_passages", "MAX_INPUT_PASSAGES", 8 if single else 5, int),
        ("final_topk_passages", "FINAL_TOPK_PASSAGES", 5, int),
        ("sim_window", "SIM_WINDOW", 1, int),
        ("batch_size", "EMB_BATCH_SIZE" if single else "RERANK_BATCH_SIZE", 64 if single else 8, int),
        ("survival_min", "SURVIVAL_MIN", 0.18, float),
        ("noise_for_drop", "NOISE_FOR_DROP", 0.35 if single else 0.30, float),
        ("min_total_keep", "MIN_TOTAL_KEEP", 0.05, float),
        ("candidate_mode", "CANDIDATE_MODE", "full", str),
        ("budget_mode", "BUDGET_MODE", "legacy", str),
    ]
    options = {key: cast(os.environ.get(env, str(default))) for key, env, default, cast in definitions}
    if not single:
        os.environ["RERANK_MAX_LENGTH"] = os.environ.get(
            "RERANK_MAX_LENGTH", os.environ.get("MAX_LENGTH", "2048")
        )
    return options


def candidate_record(sample_idx, record, query_structure, reranker, options, single=False, latency=None):
    with timed(latency, "preprocess_seconds"):
        query, documents, _ = preprocess(record)
        docs = {"query": query, "documents": [
            {"doc_id": d["doc_id"], "doc_text": d["text"]} for d in documents
        ]}
    with timed(latency, "candidate_construction_seconds", cuda=True):
        result = build_candidate_pool(docs, query_structure, reranker, **options)
    # Retain diagnostic field names used by old output consumers.
    metadata = {key: value for key, value in options.items() if key != "batch_size"}
    metadata.update({
        "candidate_mode": options["candidate_mode"],
        "budget_mode": options["budget_mode"],
        "query_structure_used_for_construction": options["candidate_mode"] == "full",
        "candidate_types": ["core", "parallel", "drop"],
        "parallel_definition": "parallel = max(answer_priority, context_priority)",
        "drop_definition": "remaining non-selected candidates from top-k passages after core and parallel selection",
        "reranker_model_for_sim": os.environ.get("RERANK_MODEL", ""),
    })
    metadata["qwen_batch_size" if single else "rerank_batch_size"] = options["batch_size"]
    if not single:
        metadata["rerank_max_length"] = os.environ.get("RERANK_MAX_LENGTH", "")
    return {
        "sample_idx": sample_idx, "query": query,
        "query_struct": query_structure.get("query_struct", {}),
        **result, "candidate_meta": metadata,
    }


def main(single=False):
    import argparse
    argparse.ArgumentParser(description="Candidate construction. Required env: SRC_JSON, RERANK_MODEL; QSTRUCT_JSON/OUT_JSON (single) or QSTRUCT_DIR/OUT_DIR (range).").parse_args()
    options = read_options(single)
    start = int(os.environ.get("SAMPLE_IDX" if single else "START_IDX", "0"))
    end = start if single else int(os.environ.get("END_IDX", "0"))
    source = env_path("SRC_JSON", "file")
    query_dir = env_path("QSTRUCT_DIR", "dir") if not single else None
    out_dir = env_path("OUT_DIR", "output") if not single else None
    model_path = str(env_path("RERANK_MODEL", "dir"))
    skip_existing = os.environ.get("SKIP_EXISTING", "0").strip().lower() in {"1", "true", "yes", "y"}
    single_query = env_path("QSTRUCT_JSON", "file") if single else None
    single_output = env_path("OUT_JSON") if single else None
    report_dir = single_output.parent if single else out_dir
    directory, sample_path = start_metrics(report_dir)
    config = {
        "start_idx": start, "end_idx": end, "src_json": str(source),
        "qstruct_dir": str(query_dir), "out_dir": str(report_dir),
        "model": model_path, **options, "skip_existing": skip_existing,
        "rerank_max_length": os.environ.get("RERANK_MAX_LENGTH", "1024"),
    }
    if single:
        config.update(qstruct_json=str(single_query), out_json=str(single_output))
    print(json.dumps(config, ensure_ascii=False, indent=2))
    samples, missing, failed = [], [], []
    latency = {"imports_seconds": IMPORT_SECONDS}
    status, error = "completed", None
    try:
        with timed(latency, "data_load_seconds"):
            records = read_json(source)
        if not isinstance(records, list) or not 0 <= start <= end < len(records):
            raise ValueError("Require a JSON list and 0 <= START_IDX <= END_IDX < len(data)")
        if not single and not query_dir.is_dir():
            raise FileNotFoundError(f"Missing QSTRUCT_DIR: {query_dir}")
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Missing RERANK_MODEL: {model_path}")
        if options["batch_size"] <= 0:
            raise ValueError("Reranker batch size must be positive")
        if options["candidate_mode"] not in {"full", "no_structure"}:
            raise ValueError("CANDIDATE_MODE must be full or no_structure")
        if options["budget_mode"] not in {"legacy", "fixed"}:
            raise ValueError("BUDGET_MODE must be legacy or fixed")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config["device"] = device
        with timed(latency, "model_load_seconds", cuda=True):
            reranker = load_qwen_reranker(model_path, device)
        with timed(latency, "processing_seconds"):
            for index in range(start, end + 1):
                query_path = single_query if single else query_dir / f"query_{index}.json"
                out_path = single_output if single else out_dir / f"candidate_pool_{index}.json"
                sample_started = time.perf_counter()
                row = {
                    "run_id": directory.name, "stage": "heuristic_candidate_construction",
                    "candidate_mode": options["candidate_mode"], "budget_mode": options["budget_mode"],
                    "sample_idx": index, "status": "failed", "error": None,
                    "query_path": str(query_path), "output_path": str(out_path),
                    "candidate_count": None, "latency_seconds": {},
                }
                try:
                    check_existing_mode(out_path, options)
                    if skip_existing and out_path.is_file() and not single:
                        row["status"] = "skipped"
                        print(f"[SKIP] sample_idx={index} {out_path}")
                        continue
                    if not query_path.is_file():
                        row.update(status="missing", error=f"Missing query structure: {query_path}")
                        missing.append({"sample_idx": index, "missing_qstruct": str(query_path)})
                        print(f"[MISS] sample_idx={index} {query_path}")
                        continue
                    with timed(row["latency_seconds"], "query_load_seconds"):
                        query_structure = read_json(query_path)
                    result = candidate_record(
                        index, records[index], query_structure, reranker, options,
                        single=single, latency=row["latency_seconds"],
                    )
                    with timed(row["latency_seconds"], "output_write_seconds"):
                        save_json(result, out_path)
                    row.update(status="saved", candidate_count=len(result["candidate_pool"]))
                    if "candidate_budget" in result:
                        row["candidate_budget"] = result["candidate_budget"]
                    print(f"[OK] sample_idx={index} candidates={row['candidate_count']} -> {out_path}")
                    if "candidate_budget" in result:
                        print(f"[BUDGET] sample_idx={index} {json.dumps(result['candidate_budget'])}")
                except Exception as exc:
                    row["error"] = repr(exc)
                    failed.append({"sample_idx": index, "error": repr(exc)})
                    print(f"[FAIL] sample_idx={index} {exc!r}")
                except BaseException as exc:
                    row["error"] = repr(exc)
                    raise
                finally:
                    row["latency_seconds"]["sample_total_seconds"] = time.perf_counter() - sample_started
                    samples.append(row)
                    with sample_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(
                        f"[LATENCY] sample_idx={index} status={row['status']} "
                        f"sample_seconds={row['latency_seconds']['sample_total_seconds']:.6f}"
                    )
        if missing:
            save_json(missing, report_dir / "missing_qstruct_indices.json")
        if failed:
            save_json(failed, report_dir / "failed_items.json")
        if missing or failed:
            status = "completed_with_errors"
        return 1 if missing or failed else 0
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        finish_metrics(directory, config, samples, latency, status, error)


if __name__ == "__main__":
    raise SystemExit(main())
