#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
SCRIPT_STARTED = time.perf_counter()
SCRIPT_STARTED_EPOCH = time.time()

import os
import re
import json
import hashlib
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, env_path, required_path
torch = LazyModule("torch")

from summary import (
    load_local_model,
    build_summary_prompt,
    build_evidence_lines,
    parse_summary_json,
)

IMPORT_SECONDS = time.perf_counter() - SCRIPT_STARTED

TEXT_KEYS = [
    "sent_content",
    "sentence",
    "text",
    "content",
    "evidence",
    "passage",
    "doc_text",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_sample_indices(path: Path) -> List[int]:
    """Read one sample index per line; blank lines and # comments are ignored."""
    if not path.is_file():
        raise FileNotFoundError(f"SAMPLE_LIST_PATH not found: {path}")
    indices = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in re.split(r"[\s,]+", line):
            if not token:
                continue
            try:
                idx = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid sample index at {path}:{line_no}: {token!r}") from exc
            if idx < 0:
                raise ValueError(f"Sample indices must be >= 0; got {idx} at {path}:{line_no}")
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices


def get_model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def render_chat_text_safe(tokenizer, messages) -> str:
    kwargs = dict(
        add_generation_prompt=True,
        tokenize=False,
    )

    if os.environ.get("DISABLE_QWEN_THINKING", "1") == "1":
        try:
            return tokenizer.apply_chat_template(
                conversation=messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            pass

    try:
        return tokenizer.apply_chat_template(
            conversation=messages,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            **kwargs,
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


def generate_batch_local_chat(
    prompt_pairs: List[Tuple[str, str]], tokenizer, model,
    max_new_tokens: int, stats=None,
) -> List[str]:
    if not prompt_pairs:
        return []
    if stats is None:
        stats = {}

    with timed(stats, "tokenization_seconds"):
        rendered_prompts = []
        for system_prompt, user_prompt in prompt_pairs:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            rendered_prompts.append(render_chat_text_safe(tokenizer, messages))
        enc = tokenizer(
            rendered_prompts, return_tensors="pt", padding=True, truncation=False,
        )
        stats["input_tokens_per_sample"] = [
            int(n) for n in enc["attention_mask"].sum(dim=1).tolist()
        ]
        stats["padded_input_tokens"] = int(enc["input_ids"].numel())

    with timed(stats, "transfer_seconds"):
        model_device = get_model_input_device(model)
        enc = {k: v.to(model_device) for k, v in enc.items()}
        synchronize_cuda()

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stats["generation_attempted"] = True
    with timed(stats, "generation_seconds"):
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
        finally:
            synchronize_cuda()

    with timed(stats, "decode_seconds"):
        prompt_len = enc["input_ids"].shape[-1]
        generated_rows = output_ids[:, prompt_len:].tolist()
        if len(generated_rows) != len(prompt_pairs):
            raise ValueError("Model returned an unexpected number of sequences")
        stats["output_tokens_per_sample"] = [
            count_generated_tokens(row, eos_token_id, pad_token_id)
            for row in generated_rows
        ]
        outputs = []
        for gen_ids in generated_rows:
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
            if "</think>" in text:
                text = text.split("</think>", 1)[-1].strip()
            outputs.append(text)
        return outputs


def norm_row(row: Any, idx: int) -> Dict[str, Any] | None:
    if not isinstance(row, dict):
        if isinstance(row, str) and row.strip():
            return {
                "doc_id": idx + 1,
                "sent_id": idx,
                "sentence": row.strip(),
            }
        return None

    if isinstance(row.get("top_sentence"), dict):
        ts = row["top_sentence"]
        text = ""
        for k in TEXT_KEYS:
            if isinstance(ts.get(k), str) and ts[k].strip():
                text = ts[k].strip()
                break

        if text:
            return {
                "doc_id": row.get("doc_id", ts.get("doc_id", idx + 1)),
                "sent_id": ts.get("sent_id", idx),
                "sentence": text,
            }

    text = ""
    for k in TEXT_KEYS:
        if isinstance(row.get(k), str) and row[k].strip():
            text = row[k].strip()
            break

    if not text:
        return None

    return {
        "doc_id": row.get("doc_id", row.get("document_id", row.get("id", idx + 1))),
        "sent_id": row.get("sent_id", row.get("sentence_id", idx)),
        "sentence": text,
    }


def collect_rows(obj: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
    out = []
    seen = set()

    for key in keys:
        rows = obj.get(key)
        if not isinstance(rows, list):
            continue

        for i, row in enumerate(rows):
            nr = norm_row(row, i)
            if nr is None:
                continue

            dedup_key = (str(nr.get("doc_id")), nr.get("sentence", "").lower())
            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            out.append(nr)

    return out


def normalize_topk_for_summary(topk_obj: Dict[str, Any], max_packed: int) -> Dict[str, Any]:
    if not isinstance(topk_obj, dict):
        raise ValueError("topk object must be dict")

    query = topk_obj.get("query", "")

    core = collect_rows(topk_obj, ["core_evidence"])
    support = collect_rows(topk_obj, ["support_evidence"])

    packed = collect_rows(
        topk_obj,
        [
            "packed_evidence",
            "reranked_topk_evidence",
            "topk_evidence",
            "reranked_evidence",
            "topk",
            "evidence",
            "candidate_pool",
        ],
    )

    if not packed:
        packed = collect_rows(topk_obj, ["topk_passages", "working_topk_passages"])

    if not packed:
        raise ValueError(
            "No usable packed evidence found in topk json. "
            f"Available keys: {list(topk_obj.keys())}"
        )

    return {
        "query": query,
        "core_evidence": core,
        "support_evidence": support,
        "packed_evidence": packed[:max_packed],
    }


def prepare_one_sample(
    sample_idx: int,
    query_dir: Path,
    topk_dir: Path,
    out_dir: Path,
    max_core: int,
    max_support: int,
    max_packed: int,
    skip_existing: bool,
    debug_dir: Path | None = None,
    stats: Dict[str, Any] | None = None,
    query_mode: str = "structured",
    use_plan: bool = True,
) -> Tuple[str, Dict[str, Any] | None]:
    query_path = query_dir / f"query_{sample_idx}.json" if query_dir is not None else None
    topk_path = topk_dir / f"topk_{sample_idx}.json"
    out_path = out_dir / f"summary_{sample_idx}.json"

    if query_mode == "structured" and not query_path.exists():
        print(f"[MISS] sample_idx={sample_idx} missing query: {query_path}")
        return "missing", None

    if not topk_path.exists():
        print(f"[MISS] sample_idx={sample_idx} missing topk: {topk_path}")
        return "missing", None

    try:
        if query_mode not in {"structured", "original"} or not isinstance(use_plan, bool):
            raise ValueError("Require QUERY_MODE=structured|original and boolean use_plan")
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            try:
                existing_obj = read_json(out_path)
                existing_meta = existing_obj.get("summary_meta", {}) if isinstance(existing_obj, dict) else {}
                existing_mode = (existing_meta.get("query_mode", "structured"),
                                 existing_meta.get("use_plan", True))
                existing_valid = (
                    isinstance(existing_obj, dict)
                    and existing_obj.get("sample_idx") == sample_idx
                    and isinstance(existing_obj.get("final_summary"), str)
                    and bool(existing_obj["final_summary"].strip())
                )
                if existing_valid and existing_mode == (query_mode, use_plan):
                    print(f"[SKIP] sample_idx={sample_idx} valid existing: {out_path}")
                    return "skipped", None
                print(f"[RECOMPUTE] sample_idx={sample_idx} existing output invalid or modes="
                      f"{existing_mode}; requested={(query_mode, use_plan)}")
            except Exception as exc:
                print(f"[RECOMPUTE] sample_idx={sample_idx} unreadable existing output: {exc!r}")

        topk_raw = read_json(topk_path)
        query = str(topk_raw.get("query") or "").strip()
        query_source = "topk.query" if query else "query_file.query"
        used_query_path = None
        qstruct = {}
        # Original-query mode needs no query_*.json when topk contains query.
        # Never fall back to the query_struct embedded in the reranker output.
        if query_mode == "structured" or not query:
            if query_path is None:
                raise ValueError("QUERY_DIR is required when topk.query is absent")
            query_obj = read_json(query_path)
            used_query_path = str(query_path)
            if not query:
                query = str(query_obj.get("query") or "").strip()
            if query_mode == "structured":
                qstruct = query_obj.get("query_struct", {})

        if not query:
            raise ValueError("empty query")
        if query_mode == "structured" and (not isinstance(qstruct, dict) or not qstruct):
            raise ValueError("empty query_struct")

        topk_obj = normalize_topk_for_summary(topk_raw, max_packed=max_packed)
        topk_obj["query"] = query

        evidence_lines = build_evidence_lines(
            topk_obj=topk_obj,
            max_core=max_core,
            max_support=max_support,
            max_packed=max_packed,
        )

        system_prompt, user_prompt = build_summary_prompt(
            query=query,
            qstruct=qstruct,
            evidence_lines=evidence_lines,
            query_mode=query_mode,
            use_plan=use_plan,
        )

        payload = {
            "sample_idx": sample_idx,
            "query": query,
            "query_path": used_query_path,
            "query_source": query_source,
            "query_mode": query_mode,
            "use_plan": use_plan,
            "topk_path": str(topk_path),
            "out_path": str(out_path),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "evidence_lines": evidence_lines,
            "evidence_sha256": hashlib.sha256(
                json.dumps(evidence_lines, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        }

        print(f"[READY] sample_idx={sample_idx} evidence_lines={len(evidence_lines)}")
        return "ready", payload

    except Exception as e:
        print(f"[FAIL] sample_idx={sample_idx} prepare error: {repr(e)}")
        if stats is not None:
            stats.update(error=f"{type(e).__name__}: {e}", failure_stage="preparation")
        debug_dir = debug_dir if debug_dir is not None else out_dir / "summary_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"summary_{sample_idx}.prepare_error.txt").write_text(str(e), encoding="utf-8")
        return "failed", None


def write_one_output(payload: Dict[str, Any], raw_text: str, out_dir: Path,
                     debug_dir: Path | None = None, stats: Dict[str, Any] | None = None) -> str:
    sample_idx = int(payload["sample_idx"])
    out_path = Path(payload["out_path"])

    try:
        parsed = parse_summary_json(raw_text, use_plan=payload.get("use_plan", True))

        out_obj = {
            "sample_idx": sample_idx,
            "query": payload["query"],
            "plan_style": parsed["plan_style"],
            "plan": parsed["plan"],
            "final_summary": parsed["final_summary"],
            "summary_meta": {
                "query_path": payload["query_path"],
                "topk_path": payload["topk_path"],
                "evidence_lines": len(payload["evidence_lines"]),
                "evidence_sha256": payload["evidence_sha256"],
                "query_source": payload["query_source"],
                "query_mode": payload["query_mode"],
                "use_plan": payload["use_plan"],
            },
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out_path, out_obj)

        print(
            f"[OK] sample_idx={sample_idx} "
            f"plan_items={len(parsed['plan'])} "
            f"-> {out_path}"
        )

        return "success"

    except Exception as e:
        print(f"[FAIL] sample_idx={sample_idx} parse/write error: {repr(e)}")
        if stats is not None:
            stats.update(error=f"{type(e).__name__}: {e}", failure_stage="parse_or_write")

        debug_dir = debug_dir if debug_dir is not None else out_dir / "summary_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        (debug_dir / f"summary_{sample_idx}.raw.txt").write_text(raw_text or "", encoding="utf-8")
        (debug_dir / f"summary_{sample_idx}.error.txt").write_text(str(e), encoding="utf-8")

        return "failed"


class RunMetrics:
    def __init__(self, out_dir, config):
        self.run_id = os.environ.get("RUN_ID") or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_" + uuid.uuid4().hex[:8]
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("RUN_ID must contain only letters, digits, _, . or -")
        self.directory = out_dir / "metrics" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_path = self.directory / "summary_range_samples.jsonl"
        self.batch_path = self.directory / "summary_range_batches.jsonl"
        if self.sample_path.exists() or self.batch_path.exists():
            raise FileExistsError(f"RUN_ID already used: {self.directory}")
        self.sample_path.touch(exist_ok=False)
        self.batch_path.touch(exist_ok=False)
        self.debug_dir = self.directory / "debug"
        self.samples, self.batches = [], []
        self.timings = {"imports_seconds": IMPORT_SECONDS}
        self.config = config

    def append(self, kind, value):
        value = dict(value, stage="summary", run_id=self.run_id,
                     query_mode=self.config["query_mode"], use_plan=self.config["use_plan"])
        path = self.sample_path if kind == "sample" else self.batch_path
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        (self.samples if kind == "sample" else self.batches).append(value)

    def finish(self, status, error=None):
        input_tokens = complete_sum([b["input_tokens"] for b in self.batches])
        output_tokens = complete_sum([b["output_tokens"] for b in self.batches])
        known_input = sum(
            n for b in self.batches for n in b["input_tokens_per_sample"] if n is not None
        )
        known_output = sum(
            n for b in self.batches for n in b["output_tokens_per_sample"] if n is not None
        )
        counts = {
            key: sum(s["status"] == key for s in self.samples)
            for key in ("success", "failed", "missing", "skipped", "not_processed")
        }
        counts["ready"] = sum(b["batch_size"] for b in self.batches)
        counts["requested"] = self.config.get(
            "requested_count",
            self.config["end_idx_inclusive"] - self.config["start_idx"] + 1,
        )
        generation_seconds = sum(b["generation_seconds"] or 0 for b in self.batches)
        batch_seconds = sum(b["batch_latency_seconds"] for b in self.batches)
        if status == "completed" and (counts["failed"] or counts["missing"]):
            status = "completed_with_failures_or_missing"
        summary = {
            "run_id": self.run_id, "stage": "summary", "status": status, "error": error,
            "started_at_utc": datetime.fromtimestamp(SCRIPT_STARTED_EPOCH, timezone.utc).isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config, "metrics_dir": str(self.directory),
            "counts": counts,
            "samples_saved": counts["success"],
            "samples_failed": counts["failed"],
            "samples_missing": counts["missing"],
            "samples_skipped": counts["skipped"],
            "batches": len(self.batches),
            "batches_generation_failed": sum(bool(b.get("generation_error")) for b in self.batches),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": token_total(input_tokens, output_tokens),
            "known_input_tokens": known_input, "known_output_tokens": known_output,
            "known_total_tokens": known_input + known_output,
            "token_counts_complete": input_tokens is not None and output_tokens is not None,
            "batches_with_incomplete_token_counts": sum(
                not b["token_counts_complete"] for b in self.batches
            ),
            "latency_seconds": dict(
                self.timings,
                preparation_total_seconds=sum(s["preparation_seconds"] for s in self.samples),
                generation_total_seconds=generation_seconds,
                batches_total_seconds=batch_seconds,
                python_stage_seconds=time.perf_counter() - SCRIPT_STARTED,
            ),
            "mean_batch_latency_seconds": batch_seconds / len(self.batches) if self.batches else None,
            "mean_generation_seconds_amortized": (
                generation_seconds / counts["ready"] if counts["ready"] else None
            ),
            "output_tokens_per_generation_second": (
                output_tokens / generation_seconds
                if output_tokens is not None and generation_seconds > 0 else None
            ),
            "measurement_notes": [
                "Token cost means counts only, not money.",
                "Input tokens include the full summary prompt, evidence and chat template; attention_mask excludes padding.",
                "Output tokens are counted before decoding/think removal and include the first EOS; padding after EOS is excluded.",
                "Input counts describe prepared tokenized prompts. Output counts may be unknown if generation fails.",
                "Unknown counts are null; known_* fields report measured subtotals.",
                "Skipped/missing/preparation-failed samples make no model call and contribute zero tokens in this run.",
                "Token totals include successful generations even when parsing or saving subsequently fails.",
                "Generation timing synchronizes all visible CUDA devices. Amortized timings are batch totals divided by batch size.",
                "Batch latency excludes sample preparation and includes tokenization, transfer, generation, decoding and output handling.",
                "Nested timings overlap: processing includes preparation and batches; do not add nested timings together.",
                "Python stage includes imports through summary construction, excluding interpreter startup/shutdown and summary-file writing.",
            ],
        }
        atomic_json(self.directory / "summary_range_summary.json", summary)
        print("========== summary_range summary ==========")
        for key, value in counts.items():
            print(f"{key}={value}")
        print(f"input_tokens={input_tokens} output_tokens={output_tokens} total_tokens={summary['total_tokens']}")
        print(f"python_stage_seconds={summary['latency_seconds']['python_stage_seconds']:.6f}")
        print(f"metrics_dir={self.directory}")
        return summary


def process_summary_batch(batch, tokenizer, model, max_output_tokens, out_dir, metrics):
    batch_started = time.perf_counter()
    batch_id = len(metrics.batches)
    stats = {
        "batch_id": batch_id, "backend": "local", "batch_size": len(batch),
        "sample_indices": [x["sample_idx"] for x in batch],
        "generation_attempted": False, "generation_error": None,
        "input_tokens_per_sample": [None] * len(batch),
        "output_tokens_per_sample": [None] * len(batch),
        "tokenization_seconds": None, "transfer_seconds": None,
        "generation_seconds": None, "decode_seconds": None,
    }
    samples = [
        {
            "sample_idx": x["sample_idx"], "batch_id": batch_id,
            "status": "not_processed", "saved": False,
            "error": None, "failure_stage": None,
            "preparation_seconds": x["_preparation_seconds"],
            "postprocess_write_seconds": 0.0,
            "evidence_lines": len(x["evidence_lines"]),
            "output_path": x["out_path"],
        } for x in batch
    ]
    print(f"[BATCH] batch_size={len(batch)} indices={stats['sample_indices']}")
    try:
        try:
            raw_outputs = generate_batch_local_chat(
                [(x["system_prompt"], x["user_prompt"]) for x in batch],
                tokenizer, model, max_output_tokens, stats,
            )
        except Exception as exc:
            stats["generation_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] whole batch inference error: {exc!r}")
            for sample in samples:
                sample.update(status="failed", error=stats["generation_error"], failure_stage="inference")
            with timed(stats, "postprocess_write_seconds"):
                metrics.debug_dir.mkdir(parents=True, exist_ok=True)
                for sample in samples:
                    (metrics.debug_dir / f"summary_{sample['sample_idx']}.batch_error.txt").write_text(
                        stats["generation_error"], encoding="utf-8"
                    )
        else:
            with timed(stats, "postprocess_write_seconds"):
                for payload, raw_text, sample in zip(batch, raw_outputs, samples):
                    with timed(sample, "postprocess_write_seconds"):
                        sample["status"] = write_one_output(
                            payload, raw_text, out_dir, debug_dir=metrics.debug_dir, stats=sample
                        )
                        sample["saved"] = sample["status"] == "success"
    except BaseException as exc:
        stats["fatal_error"] = f"{type(exc).__name__}: {exc}"
        for sample in samples:
            if sample["status"] == "not_processed":
                sample.update(status="failed", error=stats["fatal_error"], failure_stage="interrupted")
        raise
    finally:
        stats["batch_latency_seconds"] = time.perf_counter() - batch_started
        stats["input_tokens"] = complete_sum(stats["input_tokens_per_sample"])
        stats["output_tokens"] = complete_sum(stats["output_tokens_per_sample"])
        stats["total_tokens"] = token_total(stats["input_tokens"], stats["output_tokens"])
        stats["token_counts_complete"] = stats["total_tokens"] is not None
        stats["samples_saved"] = sum(s["saved"] for s in samples)
        stats["samples_failed"] = sum(s["status"] == "failed" for s in samples)
        metrics.append("batch", stats)
        for index, sample in enumerate(samples):
            sample["input_tokens"] = stats["input_tokens_per_sample"][index]
            sample["output_tokens"] = stats["output_tokens_per_sample"][index]
            sample["total_tokens"] = token_total(sample["input_tokens"], sample["output_tokens"])
            sample["token_counts_complete"] = sample["total_tokens"] is not None
            sample["batch_latency_seconds_amortized"] = stats["batch_latency_seconds"] / len(batch)
            sample["generation_seconds_amortized"] = (
                stats["generation_seconds"] / len(batch)
                if stats["generation_seconds"] is not None else None
            )
            metrics.append("sample", sample)


def main():
    import argparse
    argparse.ArgumentParser(description="Summary range. Required env: TOPK_DIR, OUT_DIR, MODEL; QUERY_DIR when QUERY_MODE=structured. BASE_TOKENIZER defaults to configured MODEL.").parse_args()
    start_idx = int(os.environ.get("START_IDX", "0"))
    end_idx = int(os.environ.get("END_IDX", "0"))
    sample_list_raw = os.environ.get("SAMPLE_LIST_PATH", "").strip()
    sample_list_path = Path(sample_list_raw).expanduser() if sample_list_raw else None
    sample_indices = read_sample_indices(sample_list_path) if sample_list_path else list(range(start_idx, end_idx + 1))
    if sample_list_path and not sample_indices:
        raise ValueError(f"SAMPLE_LIST_PATH contains no indices: {sample_list_path}")

    bench_root = os.environ.get("BENCH_ROOT", "").strip() or None
    query_mode = os.environ.get("QUERY_MODE", "structured").strip().lower()
    query_dir = env_path("QUERY_DIR", "dir") if query_mode == "structured" or os.environ.get("QUERY_DIR", "").strip() else None
    topk_dir = env_path("TOPK_DIR", "dir")

    summary_run_name = os.environ.get("SUMMARY_RUN_NAME", "qwen3_summary")

    out_dir = env_path("OUT_DIR", "output")
    model_name = str(env_path("MODEL", "dir"))
    tokenizer_name = str(required_path(os.environ.get("BASE_TOKENIZER") or model_name, "BASE_TOKENIZER / MODEL", "dir"))

    max_output_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "600"))
    max_core = int(os.environ.get("MAX_CORE", "5"))
    max_support = int(os.environ.get("MAX_SUPPORT", "5"))
    max_packed = int(os.environ.get("MAX_PACKED", "18"))

    summary_batch_size = int(os.environ.get("SUMMARY_BATCH_SIZE", "4"))
    if summary_batch_size <= 0:
        summary_batch_size = 1

    skip_existing = os.environ.get("SKIP_EXISTING", "0") == "1"
    query_mode = os.environ.get("QUERY_MODE", "structured").strip().lower()
    use_plan_setting = os.environ.get("USE_PLAN", "1").strip()
    if query_mode not in {"structured", "original"}:
        raise ValueError("QUERY_MODE must be structured or original")
    if use_plan_setting not in {"0", "1"}:
        raise ValueError("USE_PLAN must be 1 or 0")
    use_plan = use_plan_setting == "1"

    print(f"[INFO] SAMPLE_LIST_PATH={sample_list_path if sample_list_path else '<range mode>'}")
    print(f"[INFO] REQUESTED_INDICES_COUNT={len(sample_indices)}")
    if sample_list_path:
        print(f"[INFO] REQUESTED_INDICES_PREVIEW={sample_indices[:20]}")
    else:
        print(f"[INFO] START_IDX={start_idx}")
        print(f"[INFO] END_IDX={end_idx}")
    print(f"[INFO] BENCH_ROOT={bench_root}")
    print(f"[INFO] QUERY_DIR={query_dir}")
    print(f"[INFO] TOPK_DIR={topk_dir}")
    print(f"[INFO] OUT_DIR={out_dir}")
    print(f"[INFO] MODEL={model_name}")
    print(f"[INFO] BASE_TOKENIZER={tokenizer_name}")
    print(f"[INFO] MAX_OUTPUT_TOKENS={max_output_tokens}")
    print(f"[INFO] MAX_CORE={max_core}")
    print(f"[INFO] MAX_SUPPORT={max_support}")
    print(f"[INFO] MAX_PACKED={max_packed}")
    print(f"[INFO] SUMMARY_BATCH_SIZE={summary_batch_size}")
    print(f"[INFO] SKIP_EXISTING={skip_existing}")
    print(f"[INFO] QUERY_MODE={query_mode}")
    print(f"[INFO] USE_PLAN={int(use_plan)}")

    config = {
        "start_idx": min(sample_indices) if sample_list_path else start_idx,
        "end_idx_inclusive": max(sample_indices) if sample_list_path else end_idx,
        "sample_list_path": str(sample_list_path) if sample_list_path else None,
        "requested_count": len(sample_indices),
        "requested_indices": sample_indices if sample_list_path else None,
        "bench_root": str(bench_root), "query_dir": str(query_dir),
        "topk_dir": str(topk_dir), "out_dir": str(out_dir),
        "summary_run_name": summary_run_name,
        "model": model_name, "base_tokenizer": tokenizer_name,
        "max_output_tokens": max_output_tokens,
        "max_core": max_core, "max_support": max_support, "max_packed": max_packed,
        "summary_batch_size": summary_batch_size, "skip_existing": skip_existing,
        "query_mode": query_mode, "use_plan": use_plan,
        "disable_qwen_thinking": os.environ.get("DISABLE_QWEN_THINKING", "1"),
        "attn_impl": os.environ.get("ATTN_IMPL", "eager"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    metrics = RunMetrics(out_dir, config)
    print(f"[INFO] RUN_ID={metrics.run_id}")
    status, error = "completed", None
    pending = []
    try:
        if not sample_list_path and (start_idx < 0 or end_idx < start_idx):
            raise ValueError("Require 0 <= START_IDX <= END_IDX (inclusive)")
        if max_output_tokens <= 0:
            raise ValueError("MAX_OUTPUT_TOKENS must be > 0")
        if min(max_core, max_support, max_packed) < 0:
            raise ValueError("MAX_CORE, MAX_SUPPORT and MAX_PACKED must be >= 0")
        with timed(metrics.timings, "validation_seconds"):
            required_directories = [
                ("TOPK_DIR", topk_dir),
                ("MODEL", Path(model_name).expanduser()),
                ("BASE_TOKENIZER", Path(tokenizer_name).expanduser()),
            ]
            if query_mode == "structured":
                required_directories.append(("QUERY_DIR", query_dir))
            for name, path in required_directories:
                if not path.is_dir():
                    raise FileNotFoundError(f"{name} directory not found: {path}")

        with timed(metrics.timings, "model_and_tokenizer_load_seconds"):
            tokenizer, model = load_local_model(
                model_name=model_name, tokenizer_name=tokenizer_name,
            )
            synchronize_cuda()

        with timed(metrics.timings, "processing_seconds"):
            for sample_idx in sample_indices:
                preparation = {}
                with timed(preparation, "preparation_seconds"):
                    sample_status, payload = prepare_one_sample(
                        sample_idx, query_dir, topk_dir, out_dir,
                        max_core, max_support, max_packed, skip_existing,
                        debug_dir=metrics.debug_dir, stats=preparation,
                        query_mode=query_mode, use_plan=use_plan,
                    )
                if sample_status == "ready" and payload is not None:
                    payload["_preparation_seconds"] = preparation["preparation_seconds"]
                    pending.append(payload)
                    if len(pending) >= summary_batch_size:
                        batch, pending = pending, []
                        process_summary_batch(
                            batch, tokenizer, model, max_output_tokens, out_dir, metrics
                        )
                else:
                    metrics.append("sample", dict(
                        preparation, sample_idx=sample_idx, batch_id=None,
                        status=sample_status, saved=False,
                        input_tokens=0, output_tokens=0, total_tokens=0,
                        token_counts_complete=True,
                        generation_seconds_amortized=0.0,
                        batch_latency_seconds_amortized=0.0,
                        postprocess_write_seconds=0.0,
                    ))
            if pending:
                batch, pending = pending, []
                process_summary_batch(
                    batch, tokenizer, model, max_output_tokens, out_dir, metrics
                )
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Prepared but unsubmitted samples consumed no generation tokens.
        for payload in pending:
            metrics.append("sample", {
                "sample_idx": payload["sample_idx"], "batch_id": None,
                "status": "not_processed", "saved": False,
                "preparation_seconds": payload["_preparation_seconds"],
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "token_counts_complete": True,
            })
        report = metrics.finish(status, error)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
