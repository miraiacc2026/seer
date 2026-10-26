#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

SCRIPT_STARTED = time.perf_counter()

import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import uuid
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, lazy_symbol, env_path, api_client, api_configuration
torch = LazyModule("torch")
AutoModelForCausalLM = lazy_symbol("transformers", "AutoModelForCausalLM")
AutoTokenizer = lazy_symbol("transformers", "AutoTokenizer")

from query import (
    load_records,
    extract_query_from_record,
    parse_query_once,
    post_filter_query_struct,
    build_minimal_query_struct,
    write_query_struct_json,
    safe_parse_json,
)


IMPORT_SECONDS = time.perf_counter() - SCRIPT_STARTED

THINK_RE = re.compile(r"(?is)<think>.*?</think>")


def build_query_messages(query: str) -> List[Dict[str, str]]:
    system = (
        "You are a query-understanding module for evidence retrieval in a RAG pipeline. "
        "Do not answer the question. Output compact JSON only. "
        "Prefer high-recall, query-grounded retrieval signals."
    )

    user = f"""
Return JSON ONLY with this schema:

{{
  "anchor_entities": [],
  "anchor_aliases": {{}},
  "anchor_terms": [],
  "relation_terms": [],
  "relation_synonym": [],
  "relation_expansion": [],
  "relation_markers": [],
  "scope_terms": [],
  "attributes": [
    {{
      "attribute": "",
      "bridge_triggers": [],
      "hypothesis_triggers": [],
      "time_or_scope": []
    }}
  ],
  "retrieval_control": {{
    "relation_markers": [],
    "answerability_markers": [],
    "parallel_cues": []
  }},
  "retrieval_views": [
    {{"view_name": "literal", "text": ""}},
    {{"view_name": "entity_expanded", "text": ""}},
    {{"view_name": "semantic", "text": ""}}
  ]
}}

Rules:
1. Do not answer the question.
2. Do not invent a predicted answer entity.
3. anchor_entities must be query-grounded.
4. anchor_aliases keys must exactly match anchor_entities.
5. Avoid generic filler like "is to", "affect", "related to".
6. relation_* fields should describe what answer-bearing evidence looks like.
7. scope_terms only from explicit query constraints.
8. Keep everything compact.

Example:
Query: "the fundamental purpose of recording instruments that affect real property is to"

Output:
{{
  "anchor_entities": ["recording instruments", "real property"],
  "anchor_aliases": {{
    "recording instruments": ["recorded instruments", "instrument recording"],
    "real property": ["real estate", "realty"]
  }},
  "anchor_terms": ["fundamental purpose", "recording instruments", "real property"],
  "relation_terms": ["legal purpose", "function"],
  "relation_synonym": ["serve to", "designed to"],
  "relation_expansion": ["legal rationale for recording instruments", "function of recording documents in land records"],
  "relation_markers": ["purpose", "function", "notice", "priority", "protect"],
  "scope_terms": [],
  "attributes": [
    {{
      "attribute": "legal purpose of recording instruments",
      "bridge_triggers": ["purpose", "function", "notice", "priority"],
      "hypothesis_triggers": ["protect", "establish", "provide"],
      "time_or_scope": []
    }}
  ],
  "retrieval_control": {{
    "relation_markers": ["purpose", "function", "notice", "priority"],
    "answerability_markers": ["serves to", "used to", "designed to"],
    "parallel_cues": ["however", "also", "in contrast"]
  }},
  "retrieval_views": [
    {{"view_name": "literal", "text": "the fundamental purpose of recording instruments that affect real property is to"}},
    {{"view_name": "entity_expanded", "text": "the fundamental purpose of recording instruments affecting real estate is to"}},
    {{"view_name": "semantic", "text": "Find the legal purpose or function of recording instruments affecting real property in land records."}}
  ]
}}

Query: {query}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_local_model_once(model_path: str, tokenizer_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
        trust_remote_code=True,
    )

    model.eval()

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    print(f"[INFO] tokenizer loaded once from: {tokenizer_name}")
    print(f"[INFO] model loaded once from: {model_path}")
    print(f"[INFO] device={device}")
    return tokenizer, model


def make_prompt_text(tokenizer, query: str) -> str:
    messages = build_query_messages(query)

    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


@contextmanager
def timed(values, key):
    """Record elapsed wall time even when the operation raises."""
    started = time.perf_counter()
    try:
        yield
    finally:
        values[key] = time.perf_counter() - started


def synchronize_cuda():
    # Covers models split across GPUs by device_map="auto".
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)


def count_generated_tokens(token_ids, eos_token_id, pad_token_id):
    """Include the first EOS; exclude padding after EOS and trailing non-EOS pads."""
    if eos_token_id is None:
        eos_ids = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = set(eos_token_id)
    else:
        eos_ids = {eos_token_id}
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return index + 1
    end = len(token_ids)
    if pad_token_id is not None and pad_token_id not in eos_ids:
        while end and token_ids[end - 1] == pad_token_id:
            end -= 1
    return end


def complete_sum(values):
    return sum(values) if all(v is not None for v in values) else None


def token_total(input_tokens, output_tokens):
    if input_tokens is None or output_tokens is None:
        return None
    return input_tokens + output_tokens


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class RunMetrics:
    def __init__(self, out_dir, config):
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.run_id = os.environ.get("RUN_ID") or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_" + uuid.uuid4().hex[:8]
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("RUN_ID must contain only letters, digits, _, . or -")
        self.directory = out_dir / "metrics" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_path = self.directory / "query_range_samples.jsonl"
        self.batch_path = self.directory / "query_range_batches.jsonl"
        # Refuse to mix metrics from two runs with the same RUN_ID.
        if self.sample_path.exists() or self.batch_path.exists():
            raise FileExistsError(f"RUN_ID already used: {self.directory}")
        self.sample_path.touch(exist_ok=False)
        self.batch_path.touch(exist_ok=False)
        self.samples = []
        self.batches = []
        self.timings = {"imports_seconds": IMPORT_SECONDS}
        self.config = config

    def append(self, kind, value):
        value = dict(value, run_id=self.run_id, stage="query")
        path = self.sample_path if kind == "sample" else self.batch_path
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        (self.samples if kind == "sample" else self.batches).append(value)

    def finish(self, status, error=None):
        input_counts = [b["input_tokens"] for b in self.batches]
        output_counts = [b["output_tokens"] for b in self.batches]
        input_tokens = complete_sum(input_counts)
        output_tokens = complete_sum(output_counts)
        known_input = sum(b["known_input_tokens"] for b in self.batches)
        known_output = sum(b["known_output_tokens"] for b in self.batches)
        generation_seconds = sum(b.get("generation_seconds") or 0 for b in self.batches)
        batch_seconds = sum(b["batch_latency_seconds"] for b in self.batches)
        saved = sum(s.get("saved", False) for s in self.samples)
        fallback = sum(s.get("fallback", False) for s in self.samples)
        summary = {
            "run_id": self.run_id,
            "stage": "query",
            "status": status,
            "error": error,
            "started_at_utc": self.started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "metrics_dir": str(self.directory),
            "samples_saved": saved,
            "samples_parsed": sum(s["status"] == "parsed" for s in self.samples),
            "samples_fallback": fallback,
            "samples_missing_query": sum(s["status"] == "missing_query" for s in self.samples),
            "batches": len(self.batches),
            "batches_generation_failed": sum(bool(b.get("generation_error")) for b in self.batches),
            "token_counts_complete": input_tokens is not None and output_tokens is not None,
            "batches_with_incomplete_token_counts": sum(
                not b["token_counts_complete"] for b in self.batches
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": token_total(input_tokens, output_tokens),
            "known_input_tokens": known_input,
            "known_output_tokens": known_output,
            "known_total_tokens": known_input + known_output,
            "latency_seconds": dict(
                self.timings,
                generation_total_seconds=generation_seconds,
                batches_total_seconds=batch_seconds,
                python_stage_seconds=time.perf_counter() - SCRIPT_STARTED,
            ),
            "mean_batch_latency_seconds": batch_seconds / len(self.batches) if self.batches else None,
            "mean_generation_seconds_amortized": (
                generation_seconds / sum(b["batch_size"] for b in self.batches)
                if self.batches else None
            ),
            "output_tokens_per_generation_second": (
                output_tokens / generation_seconds
                if output_tokens is not None and generation_seconds > 0 else None
            ),
            "measurement_notes": [
                "Token cost means token counts only; there are no monetary costs.",
                "Local input tokens include the full prompt, few-shot example and chat template; padding is excluded.",
                "Local output tokens are counted before decoding/think removal, including the first EOS; padding after EOS is excluded.",
                "Input counts describe tokenized prompts; failed generation may have consumed an unknown number of output tokens.",
                "Unknown counts are null. known_* fields are measured subtotals, not estimates.",
                "Local batch generation latency includes CUDA synchronization; amortized latency is batch latency divided by batch size.",
                "Python stage includes imports through summary construction, excluding interpreter startup/shutdown and summary-file writing.",
                "Nested timings overlap: processing includes batches; batch latency includes generation and postprocessing.",
                "API counts use response.usage only; unreported SDK retry/server work cannot be measured here.",
            ],
        }
        atomic_json(self.directory / "query_range_summary.json", summary)
        print("========== query_range summary ==========")
        print(f"saved={saved} fallback={fallback} status={status}")
        print(f"input_tokens={input_tokens} output_tokens={output_tokens} total_tokens={summary['total_tokens']}")
        print(f"python_stage_seconds={summary['latency_seconds']['python_stage_seconds']:.6f}")
        print(f"metrics_dir={self.directory}")


def generate_llama_batch(queries, tokenizer, model, max_new_tokens, stats):
    with timed(stats, "tokenization_seconds"):
        prompts = [make_prompt_text(tokenizer, q) for q in queries]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        stats["input_tokens_per_sample"] = [
            int(n) for n in inputs["attention_mask"].sum(dim=1).tolist()
        ]
        stats["padded_input_tokens"] = int(inputs["input_ids"].numel())

    with timed(stats, "transfer_seconds"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        synchronize_cuda()
    input_len = inputs["input_ids"].shape[1]
    stats["generation_attempted"] = True
    with timed(stats, "generation_seconds"):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            synchronize_cuda()

    with timed(stats, "decode_seconds"):
        gen_ids = output_ids[:, input_len:]
        rows = gen_ids.tolist()
        if len(rows) != len(queries):
            raise ValueError("Model returned an unexpected number of sequences")
        stats["output_tokens_per_sample"] = [
            count_generated_tokens(row, tokenizer.eos_token_id, tokenizer.pad_token_id)
            for row in rows
        ]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [THINK_RE.sub("", text or "").strip() for text in texts]


def new_batch_stats(batch_id, batch_items, backend):
    return {
        "batch_id": batch_id,
        "backend": backend,
        "sample_indices": [x["sample_idx"] for x in batch_items],
        "batch_size": len(batch_items),
        "generation_attempted": False,
        "generation_error": None,
        "input_tokens_per_sample": [None] * len(batch_items),
        "output_tokens_per_sample": [None] * len(batch_items),
        "tokenization_seconds": None,
        "transfer_seconds": None,
        "generation_seconds": None,
        "decode_seconds": None,
    }


def finish_batch(metrics, stats, samples, batch_started):
    stats["batch_latency_seconds"] = time.perf_counter() - batch_started
    stats["input_tokens"] = complete_sum(stats["input_tokens_per_sample"])
    stats["output_tokens"] = complete_sum(stats["output_tokens_per_sample"])
    stats["total_tokens"] = token_total(stats["input_tokens"], stats["output_tokens"])
    stats["token_counts_complete"] = stats["total_tokens"] is not None
    for kind in ("input", "output"):
        counts = (
            [call[f"{kind}_tokens"] for call in stats["api_calls"]]
            if "api_calls" in stats else stats[f"{kind}_tokens_per_sample"]
        )
        stats[f"known_{kind}_tokens"] = sum(n for n in counts if n is not None)
    stats["samples_saved"] = sum(s.get("saved", False) for s in samples)
    stats["samples_fallback"] = sum(s.get("fallback", False) for s in samples)
    metrics.append("batch", stats)
    for sample in samples:
        sample["batch_latency_seconds_amortized"] = stats["batch_latency_seconds"] / stats["batch_size"]
        sample["generation_seconds_amortized"] = (
            stats["generation_seconds"] / stats["batch_size"]
            if stats["generation_seconds"] is not None else None
        )
        metrics.append("sample", sample)


def save_sample(item, raw, out_dir, input_tokens, output_tokens, batch_id, samples, parsed=False):
    started = time.perf_counter()
    sample_idx, query = item["sample_idx"], item["query"]
    result = {
        "sample_idx": sample_idx, "batch_id": batch_id,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": token_total(input_tokens, output_tokens),
        "token_counts_complete": input_tokens is not None and output_tokens is not None,
        "fallback": False, "saved": False, "status": "parsed", "error": None,
    }
    samples.append(result)
    try:
        try:
            if parsed:
                raw_struct = raw
            else:
                if not raw or not raw.strip():
                    raise ValueError("empty generation")
                raw_struct = safe_parse_json(raw)
            query_struct = post_filter_query_struct(raw_struct, query)
        except Exception as exc:
            result.update(status="fallback", fallback=True, error=str(exc))
            query_struct = build_minimal_query_struct(query)
        out_path = out_dir / f"query_{sample_idx}.json"
        write_query_struct_json(out_path, sample_idx, query, query_struct)
        result["saved"] = True
        result["output_path"] = str(out_path)
        print(f"[SAVED] sample_idx={sample_idx} status={result['status']} -> {out_path}")
    except BaseException as exc:
        result.update(status="write_error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["postprocess_write_seconds"] = time.perf_counter() - started
    return result


class UsageTrackingClient:
    """Forward the existing query.py API calls, observing actual response usage."""
    def __init__(self, target, calls):
        self._target = target
        self._calls = calls

    def __getattr__(self, name):
        target = getattr(self._target, name)
        if name in ("chat", "completions", "responses"):
            return UsageTrackingClient(target, self._calls)
        if name != "create" or not callable(target):
            return target

        def create(*args, **kwargs):
            call = {"input_tokens": None, "output_tokens": None}
            self._calls.append(call)
            response = target(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                def field(primary, alternate):
                    if isinstance(usage, dict):
                        value = usage.get(primary, usage.get(alternate))
                    else:
                        value = getattr(usage, primary, None)
                        if value is None:
                            value = getattr(usage, alternate, None)
                    return int(value) if value is not None else None
                call["input_tokens"] = field("prompt_tokens", "input_tokens")
                call["output_tokens"] = field("completion_tokens", "output_tokens")
            return response
        return create


def process_local_batch(batch_items, out_dir, tokenizer, model, max_output_tokens, metrics, batch_id):
    batch_started = time.perf_counter()
    stats = new_batch_stats(batch_id, batch_items, "local")
    samples = []
    print(f"[BATCH] sample_idx={stats['sample_indices']} batch_size={len(batch_items)}")
    try:
        try:
            raw_texts = generate_llama_batch(
                [x["query"] for x in batch_items], tokenizer, model, max_output_tokens, stats
            )
        except Exception as exc:
            stats["generation_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[BATCH-FAIL] fallback whole batch: {exc}")
            raw_texts = [""] * len(batch_items)
        with timed(stats, "postprocess_write_seconds"):
            for index, (item, raw) in enumerate(zip(batch_items, raw_texts)):
                save_sample(
                    item, raw, out_dir,
                    stats["input_tokens_per_sample"][index],
                    stats["output_tokens_per_sample"][index], batch_id, samples,
                )
    finally:
        finish_batch(metrics, stats, samples, batch_started)


def process_openai_one(item, out_dir, client, openai_model, max_output_tokens, metrics, batch_id):
    batch_started = time.perf_counter()
    stats = new_batch_stats(batch_id, [item], "openai")
    calls, samples = [], []
    try:
        try:
            stats["generation_attempted"] = True
            with timed(stats, "generation_seconds"):
                raw_struct = parse_query_once(
                    query=item["query"],
                    client=UsageTrackingClient(client, calls),
                    backend="openai", openai_model=openai_model,
                    max_new_tokens=max_output_tokens,
                )
            parsed = True
        except Exception as exc:
            stats["generation_error"] = f"{type(exc).__name__}: {exc}"
            raw_struct, parsed = "", False
        stats["api_calls"] = calls
        stats["input_tokens_per_sample"] = [
            complete_sum([c["input_tokens"] for c in calls]) if calls else None
        ]
        stats["output_tokens_per_sample"] = [
            complete_sum([c["output_tokens"] for c in calls]) if calls else None
        ]
        with timed(stats, "postprocess_write_seconds"):
            save_sample(
                item, raw_struct, out_dir, stats["input_tokens_per_sample"][0],
                stats["output_tokens_per_sample"][0], batch_id, samples, parsed=parsed,
            )
    finally:
        finish_batch(metrics, stats, samples, batch_started)


def main():
    import argparse
    argparse.ArgumentParser(description="Query range. Required env: SRC_JSON, OUT_DIR; MODEL and BASE_TOKENIZER for local inference.").parse_args()
    start_idx = int(os.environ.get("START_IDX", "0"))
    end_idx = int(os.environ.get("END_IDX", "0"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    src_json = env_path("SRC_JSON", "file").resolve()
    out_dir = env_path("OUT_DIR", "output").resolve()
    backend = os.environ.get("BACKEND", "local").strip().lower()
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    max_output_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "1200"))
    model_path = os.environ.get("MODEL", "")
    tokenizer_name = os.environ.get("BASE_TOKENIZER", "")
    if backend == "local":
        model_path = str(env_path("MODEL", "dir"))
        tokenizer_name = str(env_path("BASE_TOKENIZER", "dir"))
    elif backend == "openai":
        api_configuration("openai")
    config = {
        "start_idx": start_idx, "end_idx_inclusive": end_idx,
        "batch_size": batch_size, "src_json": str(src_json), "out_dir": str(out_dir),
        "backend": backend, "openai_model": openai_model,
        "max_output_tokens": max_output_tokens,
        "model": model_path, "base_tokenizer": tokenizer_name,
    }
    metrics = RunMetrics(out_dir, config)
    print("========== query_range config ==========")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"RUN_ID={metrics.run_id}")
    status, error = "completed", None
    try:
        if batch_size <= 0 or max_output_tokens <= 0:
            raise ValueError("BATCH_SIZE and MAX_OUTPUT_TOKENS must be > 0")
        if start_idx < 0 or end_idx < start_idx:
            raise ValueError("Require 0 <= START_IDX <= END_IDX (inclusive)")
        if backend not in ("local", "openai"):
            raise ValueError(f"Unsupported BACKEND: {backend}")
        with timed(metrics.timings, "data_load_seconds"):
            if not src_json.is_file():
                raise FileNotFoundError(f"SRC_JSON not found: {src_json}")
            records = load_records(src_json)
        if end_idx >= len(records):
            raise IndexError(f"END_IDX={end_idx} out of range, len(records)={len(records)}")
        work_items = []
        with timed(metrics.timings, "query_preparation_seconds"):
            for sample_idx in range(start_idx, end_idx + 1):
                query = extract_query_from_record(records[sample_idx])
                if not query:
                    metrics.append("sample", {
                        "sample_idx": sample_idx, "batch_id": None, "status": "missing_query",
                        "saved": False, "fallback": False,
                        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "token_counts_complete": True,
                    })
                    print(f"[SKIP] sample_idx={sample_idx} missing query")
                else:
                    work_items.append({"sample_idx": sample_idx, "query": query})
        print(f"[INFO] prepared items={len(work_items)}")
        if work_items and backend == "local":
            if not model_path or not Path(model_path).is_dir():
                raise FileNotFoundError(f"Local MODEL dir not found: {model_path}")
            if not tokenizer_name or not Path(tokenizer_name).is_dir():
                raise FileNotFoundError(f"BASE_TOKENIZER dir not found: {tokenizer_name}")
            with timed(metrics.timings, "model_and_tokenizer_load_seconds"):
                tokenizer, local_model = load_local_model_once(model_path, tokenizer_name)
                synchronize_cuda()
            with timed(metrics.timings, "processing_seconds"):
                for batch_id, index in enumerate(range(0, len(work_items), batch_size)):
                    process_local_batch(
                        work_items[index:index + batch_size], out_dir, tokenizer, local_model,
                        max_output_tokens, metrics, batch_id,
                    )
        elif work_items:
            with timed(metrics.timings, "api_client_setup_seconds"):
                client = api_client("openai")
            try:
                with timed(metrics.timings, "processing_seconds"):
                    for batch_id, item in enumerate(work_items):
                        process_openai_one(
                            item, out_dir, client, openai_model, max_output_tokens, metrics, batch_id
                        )
            finally:
                client.close()
        if any(s.get("fallback") or s["status"] == "missing_query" for s in metrics.samples):
            status = "completed_with_fallback_or_skips"
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metrics.finish(status, error)


if __name__ == "__main__":
    main()
