#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASQA query-structure teacher data generation over OpenAI-compatible APIs.

Proxy GPT-4o Chat Completions is verified. Proxy native Batch API returned 404 on GET /batches. Use run_chat for Chat
Completions requests (no native Batch billing).
Prompt and post-filter remain aligned with the original query.py.
No torch / transformers imports.
"""
import ast
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import shutil
from types import SimpleNamespace
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import api_client, api_configuration, configure_paths, required_path, require_batch, validate_optional_paths

DEFAULT_SRC = os.environ.get("SRC_JSON", "")
DEFAULT_JOBS = os.environ.get("JOBS_DIR", "")
DEFAULT_OUTPUT = os.environ.get("OUT_DIR", "")
TSV_COLUMNS = (
    "jsonl", "meta", "batch_id", "status", "input_file_id",
    "src_json", "src_sha256", "jsonl_sha256", "openai_model",
    "start_idx", "end_idx",
)

WORD_RE = re.compile(r"[a-z0-9]+", re.I)
CUST_RE = re.compile(r"^query-(\d+)$", re.I)

GENERIC_RELATION_TERMS = {
    "is", "are", "was", "were", "be", "been", "being",
    "is to", "are to", "was to", "were to",
    "do", "does", "did", "can", "could", "may", "might", "would", "should",
    "kind", "kind of", "type", "type of", "sort of",
    "related", "related to", "affect", "affects", "affected", "affecting",
    "use", "used", "used to", "using",
    "thing", "things", "something", "someone",
}

GENERIC_ANCHOR_TERMS = GENERIC_RELATION_TERMS | {
    "what", "which", "who", "when", "where", "why", "how",
    "purpose", "function",
}

# ASQA eval/train export accepts question/passages; also tolerate query/docs.
# Only the question goes into the Query Structure prompt, not the documents.
def preprocess(record):
    """Return (query, five-document-or-other list, unused target)."""
    if not isinstance(record, dict):
        raise TypeError(f"Expected dictionary sample, got {type(record).__name__}")
    raw_query = record.get("question") or record.get("query") or ""
    if not isinstance(raw_query, str):
        raise TypeError(f"Expected string question/query, got {type(raw_query).__name__}")
    query = raw_query.strip()
    passages = record.get("passages")
    if passages is None:
        passages = record.get("docs", [])
    documents = []
    if isinstance(passages, list):
        for i, doc in enumerate(passages, start=1):
            if isinstance(doc, str):
                title, text = "", doc.strip()
            elif isinstance(doc, dict):
                title = str(doc.get("title") or "").strip()
                text = str(doc.get("text") or "").strip()
            else:
                continue
            if title or text:
                documents.append({"doc_id": i, "title": title, "text": text})
    return query, documents, ""


# Query Structure normalization (verbatim from current query.py)
def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def norm_key(s: str) -> str:
    return norm_ws(s).lower()


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def dedup_keep_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items:
        if not isinstance(x, str):
            continue
        xx = norm_ws(x)
        if not xx:
            continue
        k = norm_key(xx)
        if k in seen:
            continue
        seen.add(k)
        out.append(xx)
    return out


def is_query_grounded_phrase(phrase: str, original_query: str) -> bool:
    p = norm_ws(phrase)
    q = norm_ws(original_query)
    if not p or not q:
        return False
    if norm_key(p) in norm_key(q):
        return True

    p_toks = set(tokenize(p))
    q_toks = set(tokenize(q))
    if not p_toks:
        return False
    overlap = len(p_toks & q_toks) / max(1, len(p_toks))
    return overlap >= 0.6


def is_generic_term(text: str, generic_bank: set) -> bool:
    t = norm_key(text)
    if not t:
        return True
    if t in generic_bank:
        return True
    toks = tokenize(t)
    return len(toks) == 1 and toks[0] in generic_bank


def clean_list(
    items: Any,
    *,
    max_items: int,
    max_words: int = 8,
    generic_bank: set | None = None,
    original_query: str | None = None,
) -> List[str]:
    if not isinstance(items, list):
        return []

    out = []
    for x in items:
        if not isinstance(x, str):
            continue
        xx = norm_ws(x)
        if not xx:
            continue
        if len(xx.split()) > max_words:
            continue
        if generic_bank is not None and is_generic_term(xx, generic_bank):
            continue
        if original_query is not None and not is_query_grounded_phrase(xx, original_query):
            continue
        out.append(xx)

    return dedup_keep_order(out)[:max_items]


def clean_alias_map(alias_map: Any, anchor_entities: List[str]) -> Dict[str, List[str]]:
    if not isinstance(alias_map, dict):
        return {}

    anchor_lookup = {norm_key(x): x for x in anchor_entities}
    out = {}

    for k, vals in alias_map.items():
        if not isinstance(k, str):
            continue
        kk = norm_key(k)
        if kk not in anchor_lookup:
            continue

        canon = anchor_lookup[kk]
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue

        cleaned = clean_list(vals, max_items=6, max_words=8)
        cleaned = [v for v in cleaned if norm_key(v) != norm_key(canon)]
        out[canon] = cleaned

    return out


def build_minimal_query_struct(query: str) -> dict:
    q = norm_ws(query)
    return {
        "anchor_entities": [],
        "anchor_aliases": {},
        "anchor_terms": [],
        "relation_terms": [],
        "relation_synonym": [],
        "relation_expansion": [],
        "relation_markers": [],
        "scope_terms": [],
        "attributes": [],
        "retrieval_control": {
            "relation_markers": [],
            "answerability_markers": [],
            "parallel_cues": [],
        },
        "retrieval_views": [
            {"view_name": "literal", "text": q},
            {"view_name": "entity_expanded", "text": q},
            {"view_name": "semantic", "text": f"Find evidence to answer: {q}"},
        ],
    }


def safe_parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty model output.")

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")

    json_str = raw[start:end + 1]
    candidates = [
        json_str,
        json_str.replace("```json", "").replace("```", "").strip(),
        re.sub(r",\s*([}\]])", r"\1", json_str),
    ]

    last_err = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception as e:
            last_err = e

    try:
        obj = ast.literal_eval(candidates[-1])
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        last_err = e

    raise ValueError(f"Failed to parse JSON: {last_err}")


def default_views(
    query: str,
    anchor_entities: List[str],
    anchor_aliases: Dict[str, List[str]],
    relation_terms: List[str],
) -> List[Dict[str, str]]:
    alias_bits = []
    for ent in anchor_entities:
        aliases = anchor_aliases.get(ent, [])
        if aliases:
            alias_bits.append(f"{ent} ({', '.join(aliases[:2])})")

    entity_expanded = norm_ws(query)
    if alias_bits:
        entity_expanded += " Anchors: " + "; ".join(alias_bits)
    if relation_terms:
        entity_expanded += " Relation focus: " + ", ".join(relation_terms[:3])

    semantic = norm_ws(query)
    if relation_terms:
        semantic = norm_ws(f"Find evidence about {', '.join(relation_terms[:3])}: {query}")

    return [
        {"view_name": "literal", "text": norm_ws(query)},
        {"view_name": "entity_expanded", "text": entity_expanded},
        {"view_name": "semantic", "text": semantic},
    ]


def post_filter_query_struct(qs: dict, original_query: str) -> dict:
    if not isinstance(qs, dict):
        qs = {}

    anchor_entities = clean_list(qs.get("anchor_entities", []), max_items=8, max_words=8)
    anchor_aliases = clean_alias_map(qs.get("anchor_aliases", {}), anchor_entities)

    anchor_terms = clean_list(
        qs.get("anchor_terms", []),
        max_items=8,
        max_words=8,
        generic_bank=GENERIC_ANCHOR_TERMS,
        original_query=original_query,
    )
    anchor_terms = [t for t in anchor_terms if norm_key(t) not in {norm_key(x) for x in anchor_entities}]

    relation_terms = clean_list(qs.get("relation_terms", []), max_items=8, max_words=8, generic_bank=GENERIC_RELATION_TERMS)
    relation_synonym = clean_list(qs.get("relation_synonym", []), max_items=8, max_words=8, generic_bank=GENERIC_RELATION_TERMS)
    relation_expansion = clean_list(qs.get("relation_expansion", []), max_items=8, max_words=8, generic_bank=GENERIC_RELATION_TERMS)
    relation_markers = clean_list(qs.get("relation_markers", []), max_items=8, max_words=6, generic_bank=GENERIC_RELATION_TERMS)

    scope_terms = clean_list(
        qs.get("scope_terms", []),
        max_items=8,
        max_words=8,
        original_query=original_query,
    )

    attrs = []
    for item in qs.get("attributes", []) if isinstance(qs.get("attributes", []), list) else []:
        if not isinstance(item, dict):
            continue
        attribute = clean_list([item.get("attribute", "")], max_items=1, max_words=8, generic_bank=GENERIC_RELATION_TERMS)
        bridge = clean_list(item.get("bridge_triggers", []), max_items=6, max_words=6, generic_bank=GENERIC_RELATION_TERMS)
        hyp = clean_list(item.get("hypothesis_triggers", []), max_items=6, max_words=6, generic_bank=GENERIC_RELATION_TERMS)
        tos = clean_list(item.get("time_or_scope", []), max_items=4, max_words=8)
        if attribute or bridge or hyp or tos:
            attrs.append({
                "attribute": attribute[0] if attribute else "",
                "bridge_triggers": bridge,
                "hypothesis_triggers": hyp,
                "time_or_scope": tos,
            })

    rc_raw = qs.get("retrieval_control", {})
    if not isinstance(rc_raw, dict):
        rc_raw = {}

    retrieval_control = {
        "relation_markers": clean_list(
            rc_raw.get("relation_markers", []) or relation_markers,
            max_items=8,
            max_words=6,
            generic_bank=GENERIC_RELATION_TERMS,
        ),
        "answerability_markers": clean_list(
            rc_raw.get("answerability_markers", []),
            max_items=8,
            max_words=8,
        ),
        "parallel_cues": clean_list(
            rc_raw.get("parallel_cues", []),
            max_items=8,
            max_words=8,
        ),
    }

    relation_union = dedup_keep_order(
        relation_terms
        + relation_synonym
        + relation_expansion
        + relation_markers
        + retrieval_control["relation_markers"]
    )

    raw_views = qs.get("retrieval_views", [])
    cleaned_views = {}
    if isinstance(raw_views, list):
        for item in raw_views:
            if not isinstance(item, dict):
                continue
            name = norm_key(str(item.get("view_name", "")))
            text = norm_ws(str(item.get("text", "")))
            if name in {"literal", "entity_expanded", "semantic"} and text:
                cleaned_views[name] = {"view_name": name, "text": " ".join(text.split()[:48])}

    defaults = {v["view_name"]: v for v in default_views(original_query, anchor_entities, anchor_aliases, relation_union)}
    retrieval_views = [cleaned_views.get(name, defaults[name]) for name in ["literal", "entity_expanded", "semantic"]]

    return {
        "anchor_entities": anchor_entities,
        "anchor_aliases": anchor_aliases,
        "anchor_terms": anchor_terms,
        "relation_terms": relation_terms,
        "relation_synonym": relation_synonym,
        "relation_expansion": relation_expansion,
        "relation_markers": relation_markers,
        "scope_terms": scope_terms,
        "attributes": attrs[:4],
        "retrieval_control": retrieval_control,
        "retrieval_views": retrieval_views,
    }


# Inference-identical system/user prompt; batch transport differs only.
def build_messages(query: str) -> List[Dict[str, str]]:
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


    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# Existing output format and OpenAI Batch transport
def _write_query_struct_json(out_path: Path, sample_idx: int, query: str, query_struct: dict) -> None:
    payload = {
        "sample_idx": sample_idx,
        "query": query,
        "query_struct": query_struct,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _api_base_url() -> str:
    return api_configuration()[1]


def _require_api_key() -> str:
    # Proxy credentials must be explicitly supplied; never leak a real OpenAI key
    # to the third-party gateway just because OPENAI_API_KEY is in the shell.
    return api_configuration()[0]


def _get_client():
    return api_client()


def extract_output_text_from_batch_line(obj: dict) -> str:
    # Proxy's verified endpoint uses Chat Completions; official Batch may wrap
    # either Chat Completions or Responses in response.body.
    response = obj.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = "\n".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ).strip()
            if text:
                return text
        raise ValueError(f"Empty/refused Chat Completions output: {msg.get('refusal')}")

    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()

    output = body.get("output", [])
    texts: List[str] = []
    if isinstance(output, list):
        for item in output:
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    t = c.get("text", "")
                    if c.get("type") in ("output_text", "text") and t:
                        texts.append(t)

    text = "\n".join(texts).strip()
    if not text:
        raise ValueError("Empty model output in batch result.")
    return text


def _load_dataset(src: str) -> List[dict]:
    p = Path(src).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"dataset json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset json must be a JSON list.")
    return data


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _ensure_nonempty_file(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(f"{label} not created: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} created but empty: {path}")


def _submit_one_jsonl(input_jsonl: Path, openai_model: str, completion_window: str, metadata_name: str) -> dict:
    require_batch()
    _require_api_key()
    if completion_window != "24h":
        raise ValueError("Batch completion_window currently supports '24h' only.")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    client = _get_client()

    with input_jsonl.open("rb") as f:
        up = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={
            "name": metadata_name,
            "task": "query_expansion",
            "model": openai_model,
        },
    )

    return {
        "input_jsonl": str(input_jsonl),
        "input_file_id": up.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": "/v1/chat/completions",
        "completion_window": completion_window,
    }


def cmd_build(args):
    src = Path(args.src).expanduser()
    out_jsonl = Path(args.out_jsonl).expanduser()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    data = _load_dataset(str(src))
    start_idx = int(args.start_idx)
    end_idx = len(data) if int(args.end_idx) < 0 else min(int(args.end_idx), len(data))

    written = 0
    skipped = 0

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for sample_idx in range(start_idx, end_idx):
            record = data[sample_idx]
            query, _documents, _target_output = preprocess(record)
            if not query:
                skipped += 1
                continue

            req = {
                "custom_id": f"query-{sample_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.openai_model,
                    "messages": build_messages(query),
                    "max_tokens": int(args.max_output_tokens),
                },
            }
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")
            written += 1

    _ensure_nonempty_file(out_jsonl, "jsonl")

    print(json.dumps({
        "src": str(src),
        "out_jsonl": str(out_jsonl),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "written": written,
        "skipped": skipped,
        "openai_model": args.openai_model,
        "max_output_tokens": int(args.max_output_tokens),
        "exists": out_jsonl.exists(),
        "size_bytes": out_jsonl.stat().st_size if out_jsonl.exists() else 0,
    }, ensure_ascii=False, indent=2))


def cmd_build_split(args):
    src = Path(args.src).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_dataset(str(src))
    start_idx = int(args.start_idx)
    end_idx = len(data) if int(args.end_idx) < 0 else min(int(args.end_idx), len(data))
    chunk_size = int(args.chunk_size)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    total_written = 0
    total_skipped = 0
    files = []

    for chunk_start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_idx)
        out_jsonl = out_dir / f"{args.job_name}_{chunk_start:04d}_{chunk_end:04d}.jsonl"

        written = 0
        skipped = 0

        with out_jsonl.open("w", encoding="utf-8") as fout:
            for sample_idx in range(chunk_start, chunk_end):
                record = data[sample_idx]
                query, _documents, _target_output = preprocess(record)
                if not query:
                    skipped += 1
                    continue

                req = {
                    "custom_id": f"query-{sample_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.openai_model,
                        "messages": build_messages(query),
                        "max_tokens": int(args.max_output_tokens),
                    },
                }
                fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                written += 1

        if written > 0:
            _ensure_nonempty_file(out_jsonl, "split jsonl")
            files.append(str(out_jsonl))

        total_written += written
        total_skipped += skipped

    if total_written == 0:
        raise RuntimeError(
            f"build_split wrote 0 requests. Check src/query extraction. src={src} start={start_idx} end={end_idx}"
        )

    payload = {
        "src": str(src),
        "out_dir": str(out_dir),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "chunk_size": chunk_size,
        "num_files": len(files),
        "written": total_written,
        "skipped": total_skipped,
        "files": files,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_submit(args):
    payload = _submit_one_jsonl(
        input_jsonl=Path(args.input_jsonl).expanduser(),
        openai_model=args.openai_model,
        completion_window=args.completion_window,
        metadata_name=args.metadata_name,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.batch_meta_out:
        meta_path = Path(args.batch_meta_out).expanduser()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {meta_path}")


def cmd_status(args):
    _require_api_key()
    client = _get_client()
    batch = client.batches.retrieve(args.batch_id)

    rc = getattr(batch, "request_counts", None)
    if rc is not None:
        try:
            rc = {"total": rc.total, "completed": rc.completed, "failed": rc.failed}
        except Exception:
            rc = str(rc)

    payload = {
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": rc,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_consume(args):
    _require_api_key()

    client = _get_client()
    batch = client.batches.retrieve(args.batch_id)

    print(f"[INFO] batch_id={batch.id}")
    print(f"[INFO] status={batch.status}")

    if batch.status != "completed":
        raise RuntimeError(f"Batch is not completed. Current status: {batch.status}")

    ofid = getattr(batch, "output_file_id", None)
    if not ofid:
        raise RuntimeError("Batch completed but output_file_id is missing.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_out_path: Optional[Path] = Path(args.raw_out_jsonl).expanduser() if args.raw_out_jsonl else None
    if raw_out_path:
        raw_out_path.parent.mkdir(parents=True, exist_ok=True)

    src_json = args.src or os.environ.get("SRC_JSON", DEFAULT_SRC)
    data = _load_dataset(src_json)

    text = client.files.content(ofid).text

    if raw_out_path:
        raw_out_path.write_text(text, encoding="utf-8")
        print(f"[saved raw batch output] {raw_out_path}")

    ok = 0
    failed = 0
    fallback = 0
    failed_rows: List[str] = []
    fallback_rows: List[str] = []
    seen_ids = set()
    expected_ids = set(getattr(args, "expected_ids", []) or [])

    start_i = int(getattr(args, "start_idx", None) if getattr(args, "start_idx", None) is not None else os.environ.get("START_IDX", "0"))
    end_i = int(getattr(args, "end_idx", None) if getattr(args, "end_idx", None) is not None else os.environ.get("END_IDX", "-1"))

    for line in text.splitlines():
        if not line.strip():
            continue

        custom_id = "<unparseable>"
        try:
            obj = json.loads(line)
            custom_id = obj.get("custom_id", "")
            m = CUST_RE.match(custom_id)
            if not m:
                raise ValueError(f"Bad custom_id: {custom_id}")
            sample_idx = int(m.group(1))

            if sample_idx < start_i:
                continue
            if end_i >= 0 and sample_idx >= end_i:
                continue
            if expected_ids and sample_idx not in expected_ids:
                raise ValueError(f"Unexpected sample index from API: {sample_idx}")
            if sample_idx in seen_ids:
                raise ValueError(f"Duplicate custom_id in Batch response: {custom_id}")
            seen_ids.add(sample_idx)

            record = data[sample_idx]
            query, _documents, _target_output = preprocess(record)
            if not query:
                raise ValueError("Empty query from preprocess()")

            # Match the inference script's parse -> post_filter -> fallback path.
            # Do not turn transport/API failures into apparently valid teacher labels.
            if obj.get("error") or obj.get("response", {}).get("status_code", 200) >= 400:
                raise RuntimeError(f"API error: {obj.get('error') or obj.get('response', {}).get('status_code')}")
            try:
                raw_text = extract_output_text_from_batch_line(obj)
                parsed = safe_parse_json(raw_text)
                query_struct = post_filter_query_struct(parsed, query)
            except Exception as parse_error:
                query_struct = build_minimal_query_struct(query)
                fallback += 1
                fallback_rows.append(f"{custom_id}\t{parse_error}")
                print(f"[FALLBACK] {custom_id}: {parse_error}")

            out_path = out_dir / f"query_{sample_idx}.json"
            if out_path.exists():
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if existing.get("query") != query:
                    raise ValueError(f"Existing output query differs for index {sample_idx}: {out_path}")
            _write_query_struct_json(out_path, sample_idx, query, query_struct)
            ok += 1

        except Exception as e:
            failed += 1
            failed_rows.append(f"{custom_id}\t{e}")
            print(f"[FAILED] {custom_id}: {e}")

    missing_ids = sorted(expected_ids - seen_ids)
    for missing in missing_ids:
        failed += 1
        failed_rows.append(f"query-{missing}\tmissing_from_batch_output")
        print(f"[FAILED] query-{missing}: missing_from_batch_output")

    summary = {
        "batch_id": batch.id,
        "status": batch.status,
        "ok": ok,
        "failed": failed,
        "fallback": fallback,
        "missing": len(missing_ids),
        "out_dir": str(out_dir),
        "start_idx": start_i,
        "end_idx": end_i,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.fallback_ids_out:
        fallback_path = Path(args.fallback_ids_out).expanduser()
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.write_text("\n".join(fallback_rows) + ("\n" if fallback_rows else ""), encoding="utf-8")
        print(f"[saved fallback ids] {fallback_path}")

    if args.failed_ids_out:
        fail_path = Path(args.failed_ids_out).expanduser()
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
        print(f"[saved failed ids] {fail_path}")
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            digest.update(buf)
    return digest.hexdigest()


def _read_tsv(tsv: Path) -> List[dict]:
    if not tsv.exists():
        return []
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or not {"jsonl", "batch_id"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid batch TSV header: {tsv}")
        return [dict(row) for row in reader]


def _write_tsv(tsv: Path, rows: List[dict]):
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp = tsv.with_name(tsv.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(tsv)


def _effective_tsv(name: str) -> Path:
    if name:
        path = Path(name).expanduser()
    else:
        path = required_path(DEFAULT_JOBS, "JOBS_DIR or --tsv") / "latest_batches.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Batch TSV not found: {path}; run submit_split first or set TSV_PATH")
    return path


def _row_from_metadata(meta: dict, jp: Path, meta_path: Path) -> dict:
    return {
        "jsonl": str(jp.resolve()),
        "meta": str(meta_path.resolve()),
        "batch_id": meta["batch_id"],
        "status": meta.get("status", ""),
        "input_file_id": meta.get("input_file_id", ""),
        "src_json": meta.get("src_json", ""),
        "src_sha256": meta.get("src_sha256", ""),
        "jsonl_sha256": meta.get("jsonl_sha256", ""),
        "openai_model": meta.get("openai_model", ""),
        "start_idx": str(meta.get("start_idx", "")),
        "end_idx": str(meta.get("end_idx", "")),
    }


def _assert_previous_submission_matches(old: dict, jp: Path, src: Path, src_hash: str, model: str):
    """Prevent accidental repeated paid Batch submissions or stale-job reuse."""
    if not old.get("batch_id"):
        return
    # Legacy metadata lacks fingerprints, so there is no safe way to confirm
    # which original dataset / prompt was submitted: do not silently reuse it.
    if not old.get("jsonl_sha256") or not old.get("src_sha256"):
        raise RuntimeError(
            f"Existing submission lacks source/request checksums: {jp}. "
            "Use a NEW JOB_NAME for this run; existing batch IDs remain usable. "
            "Do not resubmit the same requests unintentionally."
        )
    if old.get("jsonl_sha256") != _sha256(jp):
        raise RuntimeError(f"Existing submitted JSONL differs from current build: {jp}; use a new JOB_NAME")
    if old.get("src_sha256") != src_hash or Path(old.get("src_json", "")).expanduser() != src:
        raise RuntimeError(f"Existing submission belongs to another dataset: {jp}; use a new JOB_NAME")
    if old.get("openai_model") != model:
        raise RuntimeError(f"Existing submission uses a different model: {jp}; use a new JOB_NAME")


def cmd_submit_split(args):
    """Build -> submit -> save per-job & latest batch ID TSV, all in one command.

    Re-running this command reuses already-submitted Batch IDs. It never silently
    incurs a second submission for the same input file.
    """
    _require_api_key()
    if args.completion_window != "24h":
        raise ValueError("Batch completion_window currently supports '24h' only")

    src = Path(args.src).expanduser().resolve()
    jobs = Path(args.out_dir).expanduser().resolve()
    jobs.mkdir(parents=True, exist_ok=True)
    data = _load_dataset(str(src))
    start_idx = int(args.start_idx)
    end_idx = len(data) if int(args.end_idx) < 0 else min(int(args.end_idx), len(data))
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0 or start_idx < 0 or end_idx <= start_idx:
        raise ValueError("Require CHUNK_SIZE > 0 and 0 <= START_IDX < END_IDX")
    if not re.fullmatch(r"[\w.-]+", args.job_name):
        raise ValueError("JOB_NAME must contain only letters, digits, _, - or .")
    src_hash = _sha256(src)

    # Generate exactly the same query requests as the existing build_split path.
    cmd_build_split(SimpleNamespace(
        src=str(src), out_dir=str(jobs), start_idx=start_idx, end_idx=end_idx,
        chunk_size=chunk_size, job_name=args.job_name,
        openai_model=args.openai_model, max_output_tokens=args.max_output_tokens,
    ))
    segments = []
    for chunk_start in range(start_idx, end_idx, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end_idx)
        jp = jobs / f"{args.job_name}_{chunk_start:04d}_{chunk_end:04d}.jsonl"
        if jp.exists() and jp.stat().st_size:
            segments.append((jp, chunk_start, chunk_end))
    if not segments:
        raise RuntimeError("No non-empty Batch request files were generated")

    manifest = jobs / f"{args.job_name}.batches.tsv"
    existing_rows = {str(Path(r["jsonl"]).expanduser().resolve()): r for r in _read_tsv(manifest)}
    planned = []
    for jp, first, last in segments:
        meta_path = jp.with_suffix(".meta.json")
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else existing_rows.get(str(jp), {})
        _assert_previous_submission_matches(old, jp, src, src_hash, args.openai_model)
        planned.append((jp, first, last, meta_path, old))

    rows = []
    latest = jobs / "latest_batches.tsv"
    for jp, first, last, meta_path, old in planned:
        if old.get("batch_id"):
            print(f"[REUSE] batch_id={old['batch_id']} for {jp.name}; no duplicate API submission")
            payload = dict(old)
        else:
            payload = _submit_one_jsonl(
                input_jsonl=jp, openai_model=args.openai_model,
                completion_window=args.completion_window, metadata_name=jp.stem,
            )
            print(f"[SUBMITTED] {jp.name}: {payload['batch_id']} ({payload['status']})")
        payload.update({
            "src_json": str(src), "src_sha256": src_hash,
            "jsonl_sha256": _sha256(jp), "openai_model": args.openai_model,
            "start_idx": first, "end_idx": last,
        })
        # A successful API call must be persisted BEFORE sending the next request.
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(_row_from_metadata(payload, jp, meta_path))
        _write_tsv(manifest, rows)
        shutil.copyfile(manifest, latest)
    print(f"[BATCH_TSV] {manifest}")
    print(f"[LATEST_TSV] {latest}")
    print(f"[SUMMARY] batches={len(rows)} new={sum(not p[4].get('batch_id') for p in planned)} reused={sum(bool(p[4].get('batch_id')) for p in planned)}")


def cmd_run_split(args):
    # Backward-compatible alias; both commands now have identical safe behavior.
    return cmd_submit_split(args)


def cmd_status_tsv(args):
    _require_api_key()
    tsv = _effective_tsv(args.tsv)
    rows = _read_tsv(tsv)
    if not rows:
        raise ValueError(f"Empty TSV: {tsv}")
    client = _get_client()
    counts = {}
    for row in rows:
        bid = row["batch_id"]
        try:
            batch = client.batches.retrieve(bid)
            status = str(batch.status)
            row["status"] = status
            rc = getattr(batch, "request_counts", None)
            details = ({"total": rc.total, "completed": rc.completed, "failed": rc.failed} if rc else {})
            print(json.dumps({"batch_id": bid, "jsonl": row.get("jsonl"), "status": status, "request_counts": details}, ensure_ascii=False))
        except Exception as e:
            status = "FETCH_ERROR"
            print(f"[STATUS_ERROR] {bid}: {e}")
        counts[status] = counts.get(status, 0) + 1
    _write_tsv(tsv, rows)
    print(f"[STATUS_TSV] {tsv} -> {counts}")


def cmd_consume_tsv(args):
    """Download and decode every completed Batch from one TSV in one command."""
    _require_api_key()
    tsv = _effective_tsv(args.tsv)
    rows = _read_tsv(tsv)
    if not rows:
        raise ValueError(f"Empty TSV: {tsv}")
    client = _get_client()
    out_dir = Path(args.out_dir).expanduser()
    jobs = Path(args.jobs_dir).expanduser() if args.jobs_dir else tsv.parent
    jobs.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"consumed": 0, "pending": 0, "failed": 0}
    src_override = Path(args.src).expanduser().resolve() if args.src else None

    for row in rows:
        bid = row["batch_id"]
        try:
            row_src = Path(row.get("src_json") or DEFAULT_SRC).expanduser().resolve()
            if src_override is not None and row_src != src_override:
                raise ValueError(f"TSV source {row_src} differs from --src={src_override}")
            if row.get("src_sha256") and _sha256(row_src) != row["src_sha256"]:
                raise ValueError(f"Input dataset changed since submission: {row_src}")
            jp = Path(row["jsonl"]).expanduser()
            if row.get("jsonl_sha256") and _sha256(jp) != row["jsonl_sha256"]:
                raise ValueError(f"Input JSONL changed since submission: {jp}")
            batch = client.batches.retrieve(bid)
            row["status"] = str(batch.status)
            if batch.status != "completed":
                print(f"[PENDING] {bid}: status={batch.status}")
                counts["pending"] += 1
                continue

            # Limit the consumer to the index range actually submitted in this row.
            # The source dataset must match the one used at build time.
            first = int(row["start_idx"]) if row.get("start_idx") else 0
            last = int(row["end_idx"]) if row.get("end_idx") else -1
            stem = jp.stem
            expected_ids = set()
            for request_line in jp.read_text(encoding="utf-8").splitlines():
                if request_line.strip():
                    cid = json.loads(request_line)["custom_id"]
                    m = CUST_RE.fullmatch(cid)
                    if not m:
                        raise ValueError(f"Malformed request custom_id: {cid} in {jp}")
                    expected_ids.add(int(m.group(1)))
            consumed_args = SimpleNamespace(
                expected_ids=expected_ids,
                batch_id=bid, src=str(row_src), out_dir=str(out_dir),
                raw_out_jsonl=str(jobs / f"{stem}.output.jsonl"),
                failed_ids_out=str(jobs / f"{stem}.failed_ids.txt"),
                fallback_ids_out=str(jobs / f"{stem}.fallback_ids.txt"),
                start_idx=first, end_idx=last,
            )
            result = cmd_consume(consumed_args)
            if result["failed"]:
                raise RuntimeError(f"{result['failed']} failed/missing request outputs in {bid}; inspect {stem}.failed_ids.txt")
            counts["consumed"] += 1
            print(f"[CONSUMED] {bid} -> {out_dir}")
        except Exception as e:
            counts["failed"] += 1
            print(f"[CONSUME_ERROR] {bid}: {e}")
        finally:
            _write_tsv(tsv, rows)
    print(f"[CONSUME_TSV] {tsv}: {counts}")
    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} batches could not be consumed; inspect log above")



def cmd_validate_input(args):
    """Inspect the train input before incurring any external API charges."""
    data = _load_dataset(args.src)
    if not data:
        raise ValueError("Empty training JSON list")
    key_formats = Counter()
    docs_count = Counter()
    empty_indices = []
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise TypeError(f"Index {i} is not an object")
        key_formats[("question" if "question" in record else "query" if "query" in record else "MISSING")] += 1
        q, docs, _ = preprocess(record)
        docs_count[len(docs)] += 1
        if not q:
            empty_indices.append(i)
    print(json.dumps({
        "src": str(Path(args.src).expanduser().resolve()),
        "total_items": len(data),
        "query_key_distribution": dict(key_formats),
        "document_count_distribution": {str(k): v for k, v in sorted(docs_count.items())},
        "empty_query_count": len(empty_indices),
        "empty_query_indices_first_20": empty_indices[:20],
        "first_two_queries": [preprocess(x)[0] for x in data[:2]],
        "api_calls": 0,
    }, ensure_ascii=False, indent=2))
    if empty_indices:
        raise ValueError(f"{len(empty_indices)} samples have empty queries")


def cmd_probe_batch(args):
    """Read-only feature check. 200/list isn't a guarantee uploads/create also work."""
    client = _get_client()
    try:
        batch_page = client.batches.list(limit=1)
        print(json.dumps({
            "batch_list_accessible": True,
            "note": "Read-only GET /batches succeeded; Files upload and batch creation still unverified.",
            "items_returned": len(getattr(batch_page, "data", [])),
        }, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({
            "batch_list_accessible": False,
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "note": "Do not submit native Batch until Proxy confirms API support. Use run_chat.",
        }, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc


def _read_valid_output(path: Path, sample_idx: int, query: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        qs = obj.get("query_struct")
        return (
            obj.get("sample_idx") == sample_idx
            and obj.get("query") == query
            and isinstance(qs, dict)
            and isinstance(qs.get("retrieval_views"), list)
            and len(qs["retrieval_views"]) == 3
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def cmd_run_chat(args):
    """Proxy Chat Completions concurrent *requests*, NOT native discounted Batch.

    Per-sample checkpoints allow restart without recharging successful requests.
    Parse errors are failures, NOT silent fallback teacher labels.
    """
    _require_api_key()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be between 1 and 16")
    if not re.fullmatch(r"[\w.-]+", args.job_name):
        raise ValueError("JOB_NAME must contain only letters, digits, _, - or .")
    data = _load_dataset(args.src)
    start = args.start_idx
    end = len(data) if args.end_idx < 0 else min(args.end_idx, len(data))
    if start < 0 or end <= start:
        raise ValueError("Require 0 <= start_idx < end_idx")
    out_dir = Path(args.out_dir).expanduser()
    jobs_dir = Path(args.jobs_dir).expanduser() / args.job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = []
    for i in range(start, end):
        query, _, _ = preprocess(data[i])
        if not query:
            raise ValueError(f"Empty question at sample_idx={i}")
        dest = out_dir / f"query_{i}.json"
        if not args.overwrite and _read_valid_output(dest, i, query):
            skipped.append(i)
        else:
            if dest.exists() and not args.overwrite:
                raise RuntimeError(f"Existing invalid/mismatched teacher file; won't overwrite: {dest}")
            pending.append((i, query))

    print(f"[CHAT RUN] requested={end-start} pending={len(pending)} existing_valid={len(skipped)} "
          f"workers={args.workers} model={args.openai_model} range=[{start},{end})", flush=True)
    print("[MODE] Chat Completions concurrent calls; NOT native OpenAI Batch billing", flush=True)

    # A client is shared: OpenAI Python SDK/httpx supports concurrent calls.
    client = _get_client()
    def process(item):
        idx, query = item
        resp = client.chat.completions.create(
            model=args.openai_model,
            messages=build_messages(query),
            max_tokens=args.max_output_tokens,
            stream=False,
        )
        # Store original response before parsing to enable diagnosis of invalid JSON.
        model_dump = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else dict(resp)
        raw_file = jobs_dir / f"query_{idx}.raw.json"
        _atomic_write_json(raw_file, model_dump)
        choices = model_dump.get("choices") or []
        if not choices:
            raise ValueError(f"Missing model choices; inspect {raw_file}")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise ValueError(f"Truncated JSON (finish_reason=length); inspect {raw_file}")
        raw_text = extract_output_text_from_batch_line({
            "response": {"status_code": 200, "body": model_dump}
        })
        parsed = safe_parse_json(raw_text)
        qstruct = post_filter_query_struct(parsed, query)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("Parsed teacher JSON must be a non-empty object")
        if not qstruct.get("anchor_entities") and not qstruct.get("relation_terms") and not qstruct.get("attributes"):
            raise ValueError(f"Query Structure appears empty; inspect {raw_file}")
        _atomic_write_json(out_dir / f"query_{idx}.json", {
            "sample_idx": idx, "query": query, "query_struct": qstruct
        })
        usage = model_dump.get("usage") or {}
        return {"sample_idx": idx, "model": model_dump.get("model"), "usage": usage}

    succeeded = []
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(process, item): item[0] for item in pending}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                item_result = future.result()
                succeeded.append(item_result)
                print(f"[OK] query-{idx} model={item_result['model']}", flush=True)
            except Exception as exc:
                failed.append((idx, str(exc)))
                print(f"[FAILED] query-{idx}: {type(exc).__name__}: {exc}", flush=True)

    final_valid = [i for i in range(start, end) if _read_valid_output(
        out_dir / f"query_{i}.json", i, preprocess(data[i])[0]
    )]
    final_missing = sorted(set(range(start, end)) - set(final_valid))
    manifest_path = jobs_dir / "run_chat.summary.json"
    summary = {
        "mode": "chat_completions_concurrent_not_native_batch",
        "model": args.openai_model,
        "src": str(Path(args.src).expanduser()),
        "start_idx": start,
        "end_idx": end,
        "requested": end - start,
        "succeeded_this_run": len(succeeded),
        "final_valid": len(final_valid),
        "final_missing": final_missing,
        "skipped_existing": skipped,
        "failed": [{"sample_idx": i, "error": msg} for i, msg in sorted(failed)],
        "success_details": sorted(succeeded, key=lambda x: x["sample_idx"]),
        "returned_model_distribution_this_run": dict(sorted(Counter(
            row.get("model") or "UNKNOWN" for row in succeeded
        ).items())),
        "output_dir": str(out_dir),
        "raw_dir": str(jobs_dir),
    }
    _atomic_write_json(manifest_path, summary)
    print(json.dumps({
        "requested": end-start, "succeeded_this_run": len(succeeded),
        "final_valid": len(final_valid), "final_missing": final_missing,
        "skipped_existing": len(skipped), "failed": len(failed),
        "manifest": str(manifest_path), "output_dir": str(out_dir),
        "returned_model_distribution_this_run": summary["returned_model_distribution_this_run"],
    }, ensure_ascii=False, indent=2), flush=True)
    if failed or final_missing:
        raise RuntimeError(
            f"{len(failed)} requests failed; {len(final_missing)} requested files missing. "
            "Rerun the same job to retry only missing samples."
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_validate = sub.add_parser("validate_input", help="Validate train JSON locally, no API calls")
    ap_validate.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap_validate.set_defaults(func=cmd_validate_input)

    sub.add_parser("probe_batch", help="Read-only GET /batches capability check").set_defaults(func=cmd_probe_batch)

    ap_chat = sub.add_parser("run_chat", help="Concurrent Proxy Chat Completions (not native Batch)")
    ap_chat.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap_chat.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT)
    ap_chat.add_argument("--jobs_dir", type=str, default=DEFAULT_JOBS)
    ap_chat.add_argument("--start_idx", type=int, default=0)
    ap_chat.add_argument("--end_idx", type=int, default=-1)
    ap_chat.add_argument("--job_name", type=str, default="asqa_query_proxy_chat")
    ap_chat.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_chat.add_argument("--max_output_tokens", type=int, default=1400)
    ap_chat.add_argument("--workers", type=int, default=3)
    ap_chat.add_argument("--overwrite", action="store_true")
    ap_chat.set_defaults(func=cmd_run_chat)

    ap_build = sub.add_parser("build")
    ap_build.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap_build.add_argument("--out_jsonl", type=str, required=True)
    ap_build.add_argument("--start_idx", type=int, default=0)
    ap_build.add_argument("--end_idx", type=int, default=-1)
    ap_build.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_build.add_argument("--max_output_tokens", type=int, default=1400)
    ap_build.set_defaults(func=cmd_build)

    ap_build_split = sub.add_parser("build_split")
    ap_build_split.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap_build_split.add_argument("--out_dir", type=str, required=True)
    ap_build_split.add_argument("--start_idx", type=int, default=0)
    ap_build_split.add_argument("--end_idx", type=int, default=-1)
    ap_build_split.add_argument("--chunk_size", type=int, default=50)
    ap_build_split.add_argument("--job_name", type=str, default="asqa_query_batch")
    ap_build_split.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_build_split.add_argument("--max_output_tokens", type=int, default=1400)
    ap_build_split.set_defaults(func=cmd_build_split)

    ap_submit = sub.add_parser("submit")
    ap_submit.add_argument("--input_jsonl", type=str, required=True)
    ap_submit.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_submit.add_argument("--completion_window", type=str, default="24h")
    ap_submit.add_argument("--metadata_name", type=str, default="asqa-query-expansion-batch")
    ap_submit.add_argument("--batch_meta_out", type=str, default="")
    ap_submit.set_defaults(func=cmd_submit)

    ap_status = sub.add_parser("status")
    ap_status.add_argument("--batch_id", type=str, required=True)
    ap_status.set_defaults(func=cmd_status)

    ap_consume = sub.add_parser("consume")
    ap_consume.add_argument("--batch_id", type=str, required=True)
    ap_consume.add_argument("--src", type=str, default="", help="ASQA dataset used for build; defaults to SRC_JSON or DEFAULT_SRC")
    ap_consume.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT)
    ap_consume.add_argument("--raw_out_jsonl", type=str, default="")
    ap_consume.add_argument("--failed_ids_out", type=str, default="")
    ap_consume.add_argument("--fallback_ids_out", type=str, default="")
    ap_consume.set_defaults(func=cmd_consume)

    ap_run = sub.add_parser("run_split")
    ap_run.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap_run.add_argument("--out_dir", type=str, default=DEFAULT_JOBS)
    ap_run.add_argument("--start_idx", type=int, default=0)
    ap_run.add_argument("--end_idx", type=int, default=-1)
    ap_run.add_argument("--chunk_size", type=int, default=50)
    ap_run.add_argument("--job_name", type=str, default="asqa_query_batch")
    ap_run.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_run.add_argument("--max_output_tokens", type=int, default=1400)
    ap_run.add_argument("--completion_window", type=str, default="24h")
    ap_run.set_defaults(func=cmd_run_split)

    ap_submit_split = sub.add_parser("submit_split", help="Build, submit, and record all Batch IDs in one TSV")
    ap_submit_split.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap_submit_split.add_argument("--out_dir", type=str, default=DEFAULT_JOBS)
    ap_submit_split.add_argument("--start_idx", type=int, default=0)
    ap_submit_split.add_argument("--end_idx", type=int, default=-1)
    ap_submit_split.add_argument("--chunk_size", type=int, default=50)
    ap_submit_split.add_argument("--job_name", type=str, default="asqa_query_batch")
    ap_submit_split.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_submit_split.add_argument("--max_output_tokens", type=int, default=1400)
    ap_submit_split.add_argument("--completion_window", type=str, default="24h")
    ap_submit_split.set_defaults(func=cmd_submit_split)

    ap_status_tsv = sub.add_parser("status_tsv", help="Check all Batch IDs from a TSV")
    ap_status_tsv.add_argument("--tsv", type=str, default="")
    ap_status_tsv.set_defaults(func=cmd_status_tsv)

    ap_consume_tsv = sub.add_parser("consume_tsv", help="Decode all completed Batch IDs from a TSV")
    ap_consume_tsv.add_argument("--tsv", type=str, default="")
    ap_consume_tsv.add_argument("--src", type=str, default="", help="Optional source-path consistency check")
    ap_consume_tsv.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT)
    ap_consume_tsv.add_argument("--jobs_dir", type=str, default="")
    ap_consume_tsv.set_defaults(func=cmd_consume_tsv)

    args = ap.parse_args()
    validate_optional_paths(args)
    if args.cmd in {"probe_batch", "submit", "status", "consume", "run_split", "submit_split", "status_tsv", "consume_tsv"}:
        require_batch()
    if hasattr(args, "src") and args.cmd != "consume_tsv":
        args.src = str(required_path(args.src or os.environ.get("SRC_JSON"), "--src / SRC_JSON", "file"))
    configure_paths(args, ("out_dir", "out_jsonl", "input_jsonl"))
    if args.cmd == "run_chat":
        configure_paths(args, ("jobs_dir",))
    if hasattr(args, "tsv") and not args.tsv:
        args.tsv = os.environ.get("TSV_PATH") or os.environ.get("TSV", "")
        if not args.tsv:
            args.tsv = str(required_path(os.environ.get("JOBS_DIR"), "--tsv / TSV or JOBS_DIR") / "latest_batches.tsv")
    args.func(args)


if __name__ == "__main__":
    main()
