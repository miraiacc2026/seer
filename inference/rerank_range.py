#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

SCRIPT_STARTED = time.perf_counter()
SCRIPT_STARTED_UTC = time.time()

import os
import re
import json
from pathlib import Path
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from functools import wraps
import importlib
import uuid
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, lazy_symbol, env_path, required_path, api_configuration
torch = LazyModule("torch")
AutoModelForCausalLM = lazy_symbol("transformers", "AutoModelForCausalLM")

try:
    import pysbd
except Exception:
    pysbd = None

from rerank_topk_evidence import (
    load_tokenizer_safely,
    normalize_query_struct,
    rerank_candidates,
    pack_final_evidence,
    _make_slim_row_with_meta,
    BASE_TOKENIZER_DEFAULT,
    DISABLE_QWEN_THINKING,
    MIN_PACKED_SENTS,
)



IMPORT_SECONDS = time.perf_counter() - SCRIPT_STARTED


@contextmanager
def timed(values, key):
    started = time.perf_counter()
    try:
        yield
    finally:
        values[key] = time.perf_counter() - started


def synchronize_cuda():
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def complete_sum(values):
    return sum(values) if all(value is not None for value in values) else None


def add_token_totals(row):
    row["total_tokens"] = (
        row["input_tokens"] + row["output_tokens"]
        if row["input_tokens"] is not None and row["output_tokens"] is not None else None
    )
    row["token_counts_complete"] = row["total_tokens"] is not None


def token_id_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def count_generated_tokens(ids, eos_token_id):
    # Count the actual generation, including EOS, before text/think removal.
    # Padding inserted after EOS is not output. A pad ID generated before EOS
    # (or at max_length without EOS) is itself a generated token.
    eos_ids = set(token_id_list(eos_token_id))
    for index, token_id in enumerate(ids):
        if token_id in eos_ids:
            return index + 1
    return len(ids)


def effective_generation_option(model, args, kwargs, name):
    if name in kwargs:
        return kwargs[name]
    config = kwargs.get("generation_config")
    if config is None and len(args) > 1:
        config = args[1]
    if config is None:
        config = getattr(model, "generation_config", None)
    return getattr(config, name, None)


def capture_input_tokens(model, args, kwargs, row):
    ids = kwargs.get("input_ids")
    if ids is None:
        ids = kwargs.get("inputs")
    if ids is None and args:
        ids = args[0]
    if ids is None or len(ids.shape) != 2:
        raise ValueError("generate input_ids unavailable; input count is unknown")
    row["input_batch_size"] = int(ids.shape[0])
    width = int(ids.shape[1])
    row["padded_input_tokens"] = int(ids.shape[0]) * width
    mask = kwargs.get("attention_mask")
    if mask is not None:
        counts = [int(n) for n in mask.sum(dim=1).tolist()]
    else:
        pad_id = effective_generation_option(model, args, kwargs, "pad_token_id")
        pad_ids = set(token_id_list(pad_id))
        rows = ids.tolist()
        # With no mask, PAD==EOS cannot distinguish padding from prompt EOS.
        counts = [
            None if pad_ids.intersection(tokens) else len(tokens)
            for tokens in rows
        ]
        if any(n is None for n in counts):
            row["measurement_errors"].append("No attention_mask and pad IDs in prompt: input count unknown")
    row["input_tokens_per_sequence"] = counts
    row["input_tokens"] = complete_sum(counts)
    row["known_input_tokens"] = sum(n for n in counts if n is not None)
    return width


class RunMetrics:
    def __init__(self, out_dir, config):
        self.run_id = os.environ.get("RUN_ID") or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("RUN_ID must contain only letters, digits, _, . or -")
        self.directory = out_dir / "metrics" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = {
            kind: self.directory / f"rerank_range_{kind}.jsonl" for kind in ("samples", "calls")
        }
        if any(path.exists() for path in self.paths.values()):
            raise FileExistsError(f"RUN_ID already used: {self.directory}")
        for path in self.paths.values():
            path.touch(exist_ok=False)
        self.config = config
        self.timings = {"imports_seconds": IMPORT_SECONDS}
        self.calls, self.samples = [], []
        self.current_sample = None
        self.instrumentation_notes = []

    def append(self, kind, row):
        row = dict(row, run_id=self.run_id, stage="rerank")
        with self.paths[kind].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (self.calls if kind == "calls" else self.samples).append(row)

    @contextmanager
    def sample(self, sample_idx, out_json):
        started = time.perf_counter()
        first_call = len(self.calls)
        row = {
            "sample_idx": sample_idx, "output_path": str(out_json),
            "status": "failed", "error": None, "model_work_requested": False,
            "num_candidates": None, "latency_seconds": {},
        }
        self.current_sample = row
        try:
            yield row
        except BaseException as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            calls = self.calls[first_call:]
            row["model_calls"] = len(calls)
            row["model_calls_failed"] = sum(c["status"] != "completed" for c in calls)
            # Positive candidate count with no observed call must never look like free inference.
            unobserved = row["model_work_requested"] and not calls
            row["model_calls_observed"] = not unobserved
            for kind in ("input", "output"):
                row[f"{kind}_tokens"] = (
                    None if unobserved else complete_sum([c[f"{kind}_tokens"] for c in calls])
                )
                row[f"known_{kind}_tokens"] = sum(c[f"known_{kind}_tokens"] for c in calls)
            add_token_totals(row)
            row["latency_seconds"]["model_calls_seconds"] = sum(c["generation_seconds"] for c in calls)
            row["latency_seconds"]["sample_seconds"] = time.perf_counter() - started
            self.append("samples", row)
            self.current_sample = None

    def observe(self, operation, backend, model=None, args=(), kwargs=None):
        kwargs = kwargs or {}
        row = {
            "call_id": len(self.calls),
            "sample_idx": self.current_sample["sample_idx"] if self.current_sample else None,
            "backend": backend, "status": "failed", "error": None,
            "input_tokens": None, "output_tokens": None,
            "known_input_tokens": 0, "known_output_tokens": 0,
            "measurement_errors": [],
        }
        width = None
        counting_started = time.perf_counter()
        if backend == "local":
            try:
                width = capture_input_tokens(model, args, kwargs, row)
            except Exception as exc:
                row["measurement_errors"].append(f"input counting: {exc}")
            try:
                synchronize_cuda()
            except Exception as exc:
                row["measurement_errors"].append(f"pre-generation CUDA sync: {exc}")
        row["pre_generation_measurement_seconds"] = time.perf_counter() - counting_started
        started = time.perf_counter()
        response = None
        try:
            response = operation()
            row["status"] = "completed"
            return response
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if backend == "local":
                try:
                    synchronize_cuda()
                except Exception as exc:
                    row["measurement_errors"].append(f"post-generation CUDA sync: {exc}")
            row["generation_seconds"] = time.perf_counter() - started
            counting_started = time.perf_counter()
            if row["status"] == "completed":
                try:
                    if backend == "local":
                        sequences = getattr(response, "sequences", response)
                        if width is None:
                            raise ValueError("prompt width unavailable; output count is unknown")
                        if len(sequences.shape) != 2 or int(sequences.shape[1]) < width:
                            raise ValueError("unexpected generate output shape")
                        eos = effective_generation_option(model, args, kwargs, "eos_token_id")
                        counts = [
                            count_generated_tokens(tokens, eos)
                            for tokens in sequences[:, width:].tolist()
                        ]
                        row["output_sequences"] = len(counts)
                        row["output_tokens_per_sequence"] = counts
                        row["output_tokens"] = sum(counts)
                        row["known_output_tokens"] = sum(counts)
                    else:
                        usage = getattr(response, "usage", None)
                        if usage is None:
                            raise ValueError("response.usage unavailable (including streaming responses)")
                        for kind, field in (("input", "prompt_tokens"), ("output", "completion_tokens")):
                            value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
                            if value is None:
                                value = usage.get(f"{kind}_tokens") if isinstance(usage, dict) else getattr(usage, f"{kind}_tokens", None)
                            if value is not None:
                                row[f"{kind}_tokens"] = int(value)
                                row[f"known_{kind}_tokens"] = int(value)
                except Exception as exc:
                    row["measurement_errors"].append(f"output/usage counting: {exc}")
            row["post_generation_measurement_seconds"] = time.perf_counter() - counting_started
            add_token_totals(row)
            self.append("calls", row)

    def finish(self, status, error=None):
        saved = sum(s["status"] == "completed" for s in self.samples)
        generation_seconds = sum(c["generation_seconds"] for c in self.calls)
        summary = {
            "run_id": self.run_id, "stage": "rerank", "status": status, "error": error,
            "started_at_utc": datetime.fromtimestamp(SCRIPT_STARTED_UTC, timezone.utc).isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config, "metrics_dir": str(self.directory),
            "samples_requested": self.config.get("num_sample_indices"),
            "samples_recorded": len(self.samples), "samples_saved": saved,
            "samples_skipped_existing": sum(s["status"] == "skipped_existing" for s in self.samples),
            "samples_missing": sum(s["status"] == "missing" for s in self.samples),
            "samples_failed": sum(s["status"] == "failed" for s in self.samples),
            "samples_with_unobserved_model_calls": sum(not s["model_calls_observed"] for s in self.samples),
            "model_calls": len(self.calls),
            "model_calls_failed": sum(c["status"] != "completed" for c in self.calls),
            "latency_seconds": dict(
                self.timings, model_calls_total_seconds=generation_seconds,
                python_stage_seconds=time.perf_counter() - SCRIPT_STARTED,
            ),
            "mean_sample_seconds": (
                sum(s["latency_seconds"]["sample_seconds"] for s in self.samples if s["status"] == "completed") / saved
                if saved else None
            ),
            "mean_model_call_seconds": generation_seconds / len(self.calls) if self.calls else None,
            "instrumentation_notes": self.instrumentation_notes,
            "measurement_notes": [
                "Token cost means counts, not money. Input counts include full prompts/templates and exclude padding via attention_mask.",
                "Output counts use actual generated IDs before decoding/think removal and include first EOS; padding after EOS is excluded.",
                "RERANK_BATCH is candidates per prompt. Local prompts are grouped by RERANK_INFER_BATCH for one batched generate call; every observed generate/API call is counted.",
                "Local input_tokens count the tokenized inputs supplied to generate; a failed call's unreturned output tokens are unknown.",
                "Unknown counts are null; known_* fields are measured subtotals only.",
                "Skipped/missing samples use zero tokens in this run; previous runs are not included.",
                "Sample rerank_seconds includes prompt construction, tokenization, model calls and parsing inside rerank_candidates.",
                "Local generation latency uses CUDA synchronization. Counting overhead is recorded separately in calls.",
                "Python stage includes imports through summary construction, excluding interpreter startup/shutdown and summary writing.",
                "Nested latency measurements overlap; do not add sample, rerank, generation, Python and shell times.",
                "API counts require response.usage; SDK-internal retries or streaming usage not returned here are unobservable.",
            ],
        }
        for kind in ("input", "output"):
            summary[f"{kind}_tokens"] = complete_sum([s[f"{kind}_tokens"] for s in self.samples])
            summary[f"known_{kind}_tokens"] = sum(c[f"known_{kind}_tokens"] for c in self.calls)
        summary["known_total_tokens"] = summary["known_input_tokens"] + summary["known_output_tokens"]
        add_token_totals(summary)
        summary["output_tokens_per_generation_second"] = (
            summary["output_tokens"] / generation_seconds
            if summary["output_tokens"] is not None and generation_seconds > 0 else None
        )
        atomic_json(self.directory / "rerank_range_summary.json", summary)
        print("========== rerank_range summary ==========")
        print(f"status={status} saved={saved} failed={summary['samples_failed']} skipped={summary['samples_skipped_existing']} missing={summary['samples_missing']}")
        print(f"input_tokens={summary['input_tokens']} output_tokens={summary['output_tokens']} total_tokens={summary['total_tokens']}")
        print(f"python_stage_seconds={summary['latency_seconds']['python_stage_seconds']:.6f}")
        print(f"metrics_dir={self.directory}")


@contextmanager
def observe_local_generation(model, metrics):
    # Preserve the actual model object and the dependency's inference settings.
    original = model.generate
    had_instance_method = "generate" in vars(model)
    @wraps(original)
    def measured_generate(*args, **kwargs):
        return metrics.observe(
            lambda: original(*args, **kwargs), "local", model=model, args=args, kwargs=kwargs
        )
    model.generate = measured_generate
    try:
        yield
    finally:
        if had_instance_method:
            model.generate = original
        else:
            delattr(model, "generate")


@contextmanager
def observe_api_generation(metrics):
    # Hook public SDK resources: rerank_candidates may construct its own client.
    with ExitStack() as stack:
        for module_name, class_name in (
            ("openai.resources.chat.completions", "Completions"),
            ("openai.resources.responses", "Responses"),
        ):
            try:
                resource = getattr(importlib.import_module(module_name), class_name)
                original = resource.create
            except (ImportError, AttributeError) as exc:
                metrics.instrumentation_notes.append(f"{module_name} hook unavailable: {exc}")
                continue
            def make_wrapper(original_method):
                @wraps(original_method)
                def measured_create(client, *args, **kwargs):
                    return metrics.observe(
                        lambda: original_method(client, *args, **kwargs), "openai"
                    )
                return measured_create
            resource.create = make_wrapper(original)
            stack.callback(setattr, resource, "create", original)
        yield


# ============================================================
# Model loading
# ============================================================

def load_local_model_once(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = load_tokenizer_safely()
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[INFO] tokenizer loaded from: {BASE_TOKENIZER_DEFAULT}")

    attn_impl = os.environ.get("ATTN_IMPL", "").strip()

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if device == "cuda" else None,
        "local_files_only": True,
        "trust_remote_code": True,
    }

    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    model.eval()

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    print(f"[INFO] model loaded once from: {model_path}")
    print(f"[INFO] torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"[INFO] torch.cuda.device_count={torch.cuda.device_count()}")
    print(f"[INFO] DISABLE_QWEN_THINKING={DISABLE_QWEN_THINKING}")
    print(f"[INFO] ATTN_IMPL={attn_impl if attn_impl else '<model default>'}")

    return tokenizer, model


# ============================================================
# Index loading
# ============================================================

def load_sample_indices():
    idx_list_file = os.environ.get("IDX_LIST_FILE", "").strip()

    if idx_list_file:
        idx_path = Path(idx_list_file).expanduser()
        if not idx_path.is_file():
            raise FileNotFoundError(f"IDX_LIST_FILE not found: {idx_path}")

        sample_indices = []
        seen = set()

        for line in idx_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # allow either plain "123" or "123\t..."
            x = int(line.split()[0])
            if x not in seen:
                sample_indices.append(x)
                seen.add(x)

        if not sample_indices:
            raise ValueError(f"IDX_LIST_FILE is empty: {idx_path}")

        start_idx = min(sample_indices)
        end_idx = max(sample_indices)

    else:
        start_idx = int(os.environ.get("START_IDX", "0"))
        end_idx = int(os.environ.get("END_IDX", "0"))
        if end_idx < start_idx:
            raise ValueError(f"END_IDX < START_IDX: {end_idx} < {start_idx}")
        sample_indices = list(range(start_idx, end_idx + 1))

    return idx_list_file, start_idx, end_idx, sample_indices


# ============================================================
# Raw passage -> sentence candidate pool
# Used when USE_RAW_PASSAGES=1.
# This bypasses heuristic candidate construction.
# ============================================================

_BULLET_RE = re.compile(r"[•●▪◦·]+")

if pysbd is not None:
    _SEG = pysbd.Segmenter(language="en", clean=False)
else:
    _SEG = None


def clean_sentence_text(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sent_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    text = _BULLET_RE.sub(". ", text)

    if _SEG is not None:
        sents = _SEG.segment(text)
    else:
        # Fallback if pysbd is unavailable.
        sents = re.split(r"(?<=[.!?])\s+", text)

    out = []
    for s in sents:
        s = clean_sentence_text(s)
        if s:
            out.append(s)
    return out


def load_raw_records(src_json: str) -> List[Dict[str, Any]]:
    src = Path(src_json).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"SRC_JSON not found: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data.get("examples"), list):
            return data["examples"]

    raise ValueError(f"Unsupported SRC_JSON root type: {type(data).__name__}")


def extract_query_and_docs_from_raw_record(
    record: Dict[str, Any],
    max_input_passages: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not isinstance(record, dict):
        raise ValueError("raw record must be dict")

    query = ""
    for k in ["query", "question"]:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            query = v.strip()
            break

    docs = record.get("docs")
    if not isinstance(docs, list):
        docs = record.get("passages")
    if not isinstance(docs, list):
        docs = record.get("ctxs")
    if not isinstance(docs, list):
        docs = record.get("contexts")
    if not isinstance(docs, list):
        docs = record.get("documents")
    if not isinstance(docs, list):
        docs = []

    if max_input_passages > 0:
        docs = docs[:max_input_passages]

    documents = []
    for i, d in enumerate(docs, start=1):
        if isinstance(d, dict):
            text = (
                d.get("text")
                or d.get("content")
                or d.get("passage")
                or d.get("summary")
                or d.get("extraction")
                or ""
            )
            title = d.get("title", "")
            raw_id = d.get("id", "")
            score = d.get("score", None)
        elif isinstance(d, str):
            text = d
            title = ""
            raw_id = ""
            score = None
        else:
            continue

        text = str(text).strip()
        if not text:
            continue

        documents.append({
            "doc_id": i,
            "title": title,
            "raw_id": raw_id,
            "score": score,
            "text": text,
        })

    if not query:
        raise ValueError("failed to extract query/question from raw record")

    return query, documents


def load_raw_query_struct(raw_qstruct_dir: str, sample_idx: int) -> Dict[str, Any]:
    if not raw_qstruct_dir:
        return {}

    qpath = Path(raw_qstruct_dir).expanduser() / f"query_{sample_idx}.json"
    if not qpath.is_file():
        return {}

    try:
        obj = json.loads(qpath.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("query_struct"), dict):
            return obj["query_struct"]
    except Exception as e:
        print(f"[WARN] failed to load query_struct for sample_idx={sample_idx}: {repr(e)}")

    return {}


def build_raw_passage_candidate_obj(
    records: List[Dict[str, Any]],
    sample_idx: int,
    max_input_passages: int,
    raw_qstruct_dir: str = "",
    min_sent_chars: int = 20,
) -> Dict[str, Any]:
    if sample_idx < 0 or sample_idx >= len(records):
        raise IndexError(f"sample_idx={sample_idx} out of range, len={len(records)}")

    record = records[sample_idx]

    query, documents = extract_query_and_docs_from_raw_record(
        record=record,
        max_input_passages=max_input_passages,
    )

    candidate_pool = []
    for doc in documents:
        sents = sent_split(doc["text"])
        sent_id = 0

        for s in sents:
            s = clean_sentence_text(s)
            if not s:
                continue
            if min_sent_chars > 0 and len(s) < min_sent_chars:
                continue

            sent_id += 1
            candidate_pool.append({
                "doc_id": int(doc["doc_id"]),
                "sent_id": int(sent_id),
                "sentence": s,
                "doc_title": doc.get("title", ""),
                "raw_doc_id": doc.get("raw_id", ""),
                "doc_score": doc.get("score", None),
                "score": {
                    "source": "raw_passage_full_sentence_pool"
                },
            })

    qstruct = load_raw_query_struct(raw_qstruct_dir, sample_idx)

    return {
        "sample_idx": sample_idx,
        "sample_id": record.get("sample_id", None),
        "query": query,
        "query_struct": qstruct,
        "candidate_pool": candidate_pool,
        "raw_meta": {
            "source": "raw_passages",
            "max_input_passages": max_input_passages,
            "num_docs": len(documents),
            "num_candidate_sentences": len(candidate_pool),
            "raw_qstruct_dir": raw_qstruct_dir or None,
            "min_sent_chars": min_sent_chars,
        },
    }


# ============================================================
# One sample processing
# ============================================================

def process_one_sample(
    sample_idx: int,
    candidate_json: Path,
    out_json: Path,
    backend: str,
    openai_model: str,
    tokenizer,
    local_model,
    rerank_batch: int,
    max_output_tokens: int,
    topk_core: int,
    topk_support: int,
    max_packed_sents: int,
    max_input_candidates: int,
    skip_existing: bool,
    use_raw_passages: bool = False,
    raw_records: Optional[List[Dict[str, Any]]] = None,
    max_input_passages: int = 5,
    raw_qstruct_dir: str = "",
    min_sent_chars: int = 20,
    query_mode: str = "structured",
    sample_metrics: Optional[Dict[str, Any]] = None,
) -> bool:
    sample_metrics = sample_metrics if sample_metrics is not None else {"latency_seconds": {}}
    latency = sample_metrics["latency_seconds"]
    if skip_existing and out_json.exists():
        sample_metrics["status"] = "skipped_existing"
        print(f"[SKIP] sample_idx={sample_idx} existing output: {out_json}")
        return True

    with timed(latency, "candidate_load_or_build_seconds"):
        if use_raw_passages:
            if raw_records is None:
                sample_metrics["error"] = "raw_records is None"
                print(f"[FAIL] sample_idx={sample_idx} raw_records is None")
                return False

            try:
                obj = build_raw_passage_candidate_obj(
                    records=raw_records,
                    sample_idx=sample_idx,
                    max_input_passages=max_input_passages,
                    raw_qstruct_dir=raw_qstruct_dir,
                    min_sent_chars=min_sent_chars,
                )
            except Exception as e:
                sample_metrics["error"] = f"{type(e).__name__}: {e}"
                print(f"[FAIL] sample_idx={sample_idx} failed to build raw candidate pool: {repr(e)}")
                return False

        else:
            if not candidate_json.exists():
                sample_metrics.update(status="missing", error="candidate JSON not found")
                print(f"[MISS] sample_idx={sample_idx} missing candidate json: {candidate_json}")
                return False

            try:
                obj = json.loads(candidate_json.read_text(encoding="utf-8"))
            except Exception as e:
                sample_metrics["error"] = f"{type(e).__name__}: {e}"
                print(f"[FAIL] sample_idx={sample_idx} cannot read candidate json: {repr(e)}")
                return False

        if not isinstance(obj, dict):
            sample_metrics["error"] = "candidate object is not a dict"
            print(f"[FAIL] sample_idx={sample_idx} candidate object is not a dict")
            return False

        if "query" not in obj:
            sample_metrics["error"] = "missing query"
            print(f"[FAIL] sample_idx={sample_idx} missing key: query")
            return False

        if "candidate_pool" not in obj or not isinstance(obj["candidate_pool"], list):
            sample_metrics["error"] = "missing/invalid candidate_pool"
            print(f"[FAIL] sample_idx={sample_idx} missing/invalid key: candidate_pool")
            return False

    query = obj["query"]
    with timed(latency, "query_normalization_seconds"):
        if query_mode == "original":
            # Keep the original query only. The structured fields remain in
            # the source JSON for traceability but are not sent to reranking.
            qstruct = {}
        else:
            qstruct = normalize_query_struct(obj.get("query_struct", {}), query)

    candidate_pool = obj["candidate_pool"]

    # max_input_candidates <= 0 means use full sentence pool.
    if max_input_candidates > 0:
        candidate_pool = candidate_pool[:max_input_candidates]

    print(
        f"[START] sample_idx={sample_idx} "
        f"use_raw_passages={use_raw_passages} "
        f"num_candidates={len(candidate_pool)} "
        f"candidate_json={candidate_json if not use_raw_passages else '<raw_passages>'}"
    )

    sample_metrics["num_candidates"] = len(candidate_pool)
    sample_metrics["model_work_requested"] = bool(candidate_pool)

    # Used only by debug logic inside rerank_topk_evidence.py.
    os.environ["SAMPLE_IDX"] = str(sample_idx)

    try:
        with timed(latency, "rerank_seconds"):
            labeled_rows = rerank_candidates(
                query=query,
                qstruct=qstruct,
                candidate_pool=candidate_pool,
                backend=backend,
                openai_model=openai_model,
                tokenizer=tokenizer,
                local_model=local_model,
                batch_size=rerank_batch,
                max_output_tokens=max_output_tokens,
                query_mode=query_mode,
            )
    except Exception as e:
        sample_metrics["error"] = f"{type(e).__name__}: {e}"
        print(f"[FAIL] sample_idx={sample_idx} rerank failed: {repr(e)}")
        return False

    with timed(latency, "evidence_pack_seconds"):
        core_rows, support_rows, drop_rows, packed = pack_final_evidence(
            labeled_rows=labeled_rows,
            topk_core=topk_core,
            topk_support=topk_support,
            max_packed_sents=max_packed_sents,
            min_packed_sents=MIN_PACKED_SENTS,
        )

    out = {
        "sample_idx": obj.get("sample_idx", sample_idx),
        "sample_id": obj.get("sample_id", None),
        "query": query,
        "query_struct": obj.get("query_struct", {}),
        "candidate_json": str(candidate_json) if not use_raw_passages else None,
        "raw_meta": obj.get("raw_meta", {}),
        "core_evidence": [_make_slim_row_with_meta(r, "core") for r in core_rows],
        "support_evidence": [_make_slim_row_with_meta(r, "support") for r in support_rows],
        "dropped": [_make_slim_row_with_meta(r, "drop") for r in drop_rows[:200]],
        "packed_evidence": packed,
        "rerank_meta": {
            "backend": backend,
            "openai_model": openai_model if backend == "openai" else None,
            "local_model": os.environ.get("MODEL") if backend == "local" else None,
            "base_tokenizer": BASE_TOKENIZER_DEFAULT if backend == "local" else None,
            "disable_qwen_thinking": DISABLE_QWEN_THINKING if backend == "local" else None,
            "use_raw_passages": use_raw_passages,
            "num_candidates": len(candidate_pool),
            "topk_core": topk_core,
            "topk_support": topk_support,
            "max_packed_sents": max_packed_sents,
            "min_packed_sents": MIN_PACKED_SENTS,
            "rerank_batch": rerank_batch,
            "rerank_infer_batch": int(os.environ.get("RERANK_INFER_BATCH", "3")),
            "query_mode": query_mode,
            "max_output_tokens": max_output_tokens,
            "max_input_candidates": max_input_candidates,
            "max_input_passages": max_input_passages if use_raw_passages else None,
            "min_sent_chars": min_sent_chars if use_raw_passages else None,
            "raw_qstruct_dir": raw_qstruct_dir if use_raw_passages and raw_qstruct_dir else None,
        },
    }

    with timed(latency, "output_write_seconds"):
        out_json.parent.mkdir(parents=True, exist_ok=True)
        tmp_json = out_json.with_suffix(out_json.suffix + ".tmp")
        tmp_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_json.replace(out_json)

    print(
        f"[OK] sample_idx={sample_idx} "
        f"core={len(core_rows)} support={len(support_rows)} drop={len(drop_rows)} "
        f"packed={len(packed)} -> {out_json}"
    )

    sample_metrics["status"] = "completed"
    return True


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    argparse.ArgumentParser(description="Rerank range. Required env: OUT_DIR; CANDIDATE_DIR unless USE_RAW_PASSAGES=1; MODEL for local inference.").parse_args()
    bench_root = os.environ.get("BENCH_ROOT", "").strip() or None
    candidate_dir = env_path("CANDIDATE_DIR", "dir") if os.environ.get("USE_RAW_PASSAGES", "0") != "1" else None
    out_dir = env_path("OUT_DIR", "output")

    backend = os.environ.get("BACKEND", "local").strip().lower()
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    query_mode = os.environ.get("QUERY_MODE", "structured").strip().lower()
    if query_mode not in {"structured", "original"}:
        raise ValueError("QUERY_MODE must be 'structured' or 'original'")

    # RERANK_BATCH = number of candidate sentences inside one rerank prompt.
    # RERANK_INFER_BATCH = number of prompts sent to local generate together.
    rerank_batch = int(os.environ.get("RERANK_BATCH", "6"))
    rerank_infer_batch = int(os.environ.get("RERANK_INFER_BATCH", "3"))
    max_output_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "1400"))

    topk_core = int(os.environ.get("TOPK_CORE", "5"))
    topk_support = int(os.environ.get("TOPK_SUPPORT", "8"))
    max_packed_sents = int(os.environ.get("MAX_PACKED_SENTS", "14"))

    max_input_candidates = int(os.environ.get("MAX_INPUT_CANDIDATES", "18"))
    skip_existing = os.environ.get("SKIP_EXISTING", "0") == "1"

    use_raw_passages = os.environ.get("USE_RAW_PASSAGES", "0") == "1"
    src_json = os.environ.get("SRC_JSON", "").strip()
    max_input_passages = int(os.environ.get("MAX_INPUT_PASSAGES", "5"))
    raw_qstruct_dir = os.environ.get("RAW_QSTRUCT_DIR", "").strip()
    min_sent_chars = int(os.environ.get("MIN_SENT_CHARS", "20"))

    if backend == "local":
        env_path("MODEL", "dir")
        required_path(BASE_TOKENIZER_DEFAULT, "BASE_TOKENIZER / MODEL", "dir")
    elif backend == "openai":
        api_configuration("openai")
    if use_raw_passages:
        src_json = str(env_path("SRC_JSON", "file"))
        if raw_qstruct_dir:
            required_path(raw_qstruct_dir, "RAW_QSTRUCT_DIR", "dir")

    config = {
        "backend": backend, "openai_model": openai_model,
        "model": os.environ.get("MODEL", ""), "base_tokenizer": BASE_TOKENIZER_DEFAULT,
        "candidate_dir": str(candidate_dir), "out_dir": str(out_dir),
        "rerank_batch": rerank_batch, "rerank_infer_batch": rerank_infer_batch,
        "query_mode": query_mode,
        "max_output_tokens": max_output_tokens,
        "topk_core": topk_core, "topk_support": topk_support,
        "max_packed_sents": max_packed_sents, "min_packed_sents": MIN_PACKED_SENTS,
        "max_input_candidates": max_input_candidates,
        "skip_existing": skip_existing, "use_raw_passages": use_raw_passages,
        "src_json": src_json, "max_input_passages": max_input_passages,
        "raw_qstruct_dir": raw_qstruct_dir, "min_sent_chars": min_sent_chars,
        "disable_qwen_thinking": DISABLE_QWEN_THINKING,
        "attn_impl": os.environ.get("ATTN_IMPL", ""),
    }
    metrics = RunMetrics(out_dir, config)
    status, error = "completed", None
    print(f"[INFO] RUN_ID={metrics.run_id} metrics_dir={metrics.directory}")
    try:
        with timed(metrics.timings, "index_load_seconds"):
            idx_list_file, start_idx, end_idx, sample_indices = load_sample_indices()
            if any(index < 0 for index in sample_indices):
                raise ValueError("Sample indices must be >= 0")
        config.update(
            idx_list_file=idx_list_file, start_idx=start_idx,
            end_idx_inclusive=end_idx, num_sample_indices=len(sample_indices),
        )
        if rerank_batch <= 0 or rerank_infer_batch <= 0 or max_output_tokens <= 0:
            raise ValueError("RERANK_BATCH, RERANK_INFER_BATCH and MAX_OUTPUT_TOKENS must be > 0")
        if backend not in ("local", "openai"):
            raise ValueError(f"Unsupported BACKEND: {backend}")
        print(f"[INFO] IDX_LIST_FILE={idx_list_file if idx_list_file else '<range mode>'}")
        print(f"[INFO] START_IDX={start_idx}")
        print(f"[INFO] END_IDX={end_idx}")
        print(f"[INFO] NUM_SAMPLE_INDICES={len(sample_indices)}")
        print(f"[INFO] FIRST_SAMPLE_INDICES={sample_indices[:20]}")
        print(f"[INFO] BENCH_ROOT={bench_root}")
        print(f"[INFO] CANDIDATE_DIR={candidate_dir}")
        print(f"[INFO] OUT_DIR={out_dir}")
        print(f"[INFO] BACKEND={backend}")
        print(f"[INFO] RERANK_BATCH={rerank_batch}")
        print(f"[INFO] RERANK_INFER_BATCH={rerank_infer_batch}")
        print(f"[INFO] QUERY_MODE={query_mode}")
        print(f"[INFO] MAX_OUTPUT_TOKENS={max_output_tokens}")
        print(f"[INFO] MAX_INPUT_CANDIDATES={max_input_candidates}")
        print(f"[INFO] SKIP_EXISTING={skip_existing}")
        print(f"[INFO] USE_RAW_PASSAGES={use_raw_passages}")
        print(f"[INFO] SRC_JSON={src_json if src_json else '<empty>'}")
        print(f"[INFO] MAX_INPUT_PASSAGES={max_input_passages}")
        print(f"[INFO] RAW_QSTRUCT_DIR={raw_qstruct_dir if raw_qstruct_dir else '<empty>'}")
        print(f"[INFO] MIN_SENT_CHARS={min_sent_chars}")
        print(f"[INFO] pysbd_available={pysbd is not None}")
        tokenizer = local_model = None
        if backend == "local":
            model_path = os.environ.get("MODEL", "").strip()
            if not model_path:
                raise ValueError("BACKEND=local but MODEL is empty")
            model_path_obj = Path(model_path).expanduser()
            if not model_path_obj.is_dir():
                raise FileNotFoundError(f"Local MODEL dir not found: {model_path_obj}")
            with timed(metrics.timings, "model_and_tokenizer_load_seconds"):
                tokenizer, local_model = load_local_model_once(str(model_path_obj))
                synchronize_cuda()
        else:
            api_configuration("openai")

        raw_records = None
        if use_raw_passages:
            if not src_json:
                raise ValueError("USE_RAW_PASSAGES=1 but SRC_JSON is empty")
            with timed(metrics.timings, "raw_data_load_seconds"):
                raw_records = load_raw_records(src_json)
            print(f"[INFO] loaded raw records: {len(raw_records)} from {src_json}")

        observer = (
            observe_local_generation(local_model, metrics)
            if backend == "local" else observe_api_generation(metrics)
        )
        with observer, timed(metrics.timings, "processing_seconds"):
            for sample_idx in sample_indices:
                candidate_json = candidate_dir / f"candidate_pool_{sample_idx}.json"
                out_json = out_dir / f"topk_{sample_idx}.json"
                with metrics.sample(sample_idx, out_json) as sample_metrics:
                    process_one_sample(
                        sample_idx=sample_idx, candidate_json=candidate_json, out_json=out_json,
                        backend=backend, openai_model=openai_model, tokenizer=tokenizer,
                        local_model=local_model, rerank_batch=rerank_batch,
                        max_output_tokens=max_output_tokens, topk_core=topk_core,
                        topk_support=topk_support, max_packed_sents=max_packed_sents,
                        max_input_candidates=max_input_candidates, skip_existing=skip_existing,
                        use_raw_passages=use_raw_passages, raw_records=raw_records,
                        max_input_passages=max_input_passages, raw_qstruct_dir=raw_qstruct_dir,
                        min_sent_chars=min_sent_chars, query_mode=query_mode,
                        sample_metrics=sample_metrics,
                    )
        if any(s["status"] in ("failed", "missing") for s in metrics.samples) or any(
            c["status"] != "completed" for c in metrics.calls
        ):
            status = "completed_with_failures_or_missing"
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metrics.finish(status, error)


if __name__ == "__main__":
    main()
