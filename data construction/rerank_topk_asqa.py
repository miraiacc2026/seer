#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import re
import json
import time
import argparse
import csv
import hashlib
import fcntl
import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, DefaultDict
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import api_client, api_configuration, configure_paths, require_batch, validate_optional_paths


# Minimum number of sentences handed downstream. An empty or near-empty packed
# set caps answer coverage at zero, so we refill from the highest-scoring dropped
# candidates. Set MIN_PACKED_SENTS=0 to restore the previous behaviour.
MIN_PACKED_SENTS = int(os.environ.get("MIN_PACKED_SENTS", "3"))


# ============================================================
# OpenAI helpers
# ============================================================
def _get_client():
    return api_client("openai")


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _ensure_nonempty_file(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(f"{label} not created: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} created but empty: {path}")


def _submit_one_jsonl(input_jsonl: Path, openai_model: str, completion_window: str, metadata_name: str) -> dict:
    require_batch()
    api_configuration("openai")
    if completion_window != "24h":
        raise ValueError("Batch completion_window currently supports '24h' only.")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")

    client = _get_client()

    with input_jsonl.open("rb") as f:
        up = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata={
            "name": metadata_name,
            "task": "rerank_topk_evidence",
            "model": openai_model,
        },
    )

    return {
        "input_jsonl": str(input_jsonl),
        "input_file_id": up.id,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": "/v1/responses",
        "completion_window": completion_window,
    }


def extract_output_text_from_batch_line(obj: dict) -> str:
    body = obj.get("response", {}).get("body", {})

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


# ============================================================
# Inference-aligned prompt, normalization, parsing, and packing
# Pure Python functions reproduced from rerank_topk_evidence.py
# ============================================================
def strip_qwen_thinking(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()

    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    if "<think>" in text:
        text = text.split("<think>", 1)[0].strip()

    return text.strip()


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _norm_key(s: str) -> str:
    return _norm_ws(s).lower()


def _dedup_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        if not isinstance(x, str):
            continue
        xx = _norm_ws(x)
        if not xx:
            continue
        k = _norm_key(xx)
        if k in seen:
            continue
        seen.add(k)
        out.append(xx)
    return out


def _clean_alias_map(alias_map: Any, anchor_entities: List[str]) -> Dict[str, List[str]]:
    if not isinstance(alias_map, dict):
        return {}

    anchor_lookup = {_norm_key(x): x for x in anchor_entities}
    out: Dict[str, List[str]] = {}

    for k, vals in alias_map.items():
        if not isinstance(k, str):
            continue
        kk = _norm_key(k)
        if kk not in anchor_lookup:
            continue

        canon = anchor_lookup[kk]
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue

        cleaned = _dedup_keep_order([v for v in vals if isinstance(v, str)])
        cleaned = [v for v in cleaned if _norm_key(v) != _norm_key(canon)]
        out[canon] = cleaned[:6]

    return out


def _minimal_query_struct(query: str) -> Dict[str, Any]:
    q = _norm_ws(query)
    return {
        "anchor_entities": [],
        "anchor_aliases": {},
        "anchor_terms": [],
        "relation_terms": [],
        "scope_terms": [],
        "retrieval_views": [
            {"view_name": "literal", "text": q},
            {"view_name": "entity_expanded", "text": q},
            {"view_name": "semantic", "text": q},
        ],
    }


def normalize_query_struct(raw_qs: Any, query: str) -> Dict[str, Any]:
    if not isinstance(raw_qs, dict):
        return _minimal_query_struct(query)

    anchor_entities = raw_qs.get("anchor_entities") or raw_qs.get("core_entities") or []
    if not isinstance(anchor_entities, list):
        anchor_entities = []
    anchor_entities = _dedup_keep_order([x for x in anchor_entities if isinstance(x, str)])[:8]

    anchor_aliases = raw_qs.get("anchor_aliases") or raw_qs.get("entity_aliases") or {}
    anchor_aliases = _clean_alias_map(anchor_aliases, anchor_entities)

    anchor_terms = raw_qs.get("anchor_terms") or []
    if not isinstance(anchor_terms, list):
        anchor_terms = []
    anchor_terms = _dedup_keep_order([x for x in anchor_terms if isinstance(x, str)])[:8]

    relation_terms = []
    for k in [
        "relation_terms",
        "relation_synonym",
        "relation_expansion",
        "relation_markers",
    ]:
        vals = raw_qs.get(k) or []
        if isinstance(vals, list):
            relation_terms.extend([x for x in vals if isinstance(x, str)])

    attrs = raw_qs.get("attributes") or []
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            attr = a.get("attribute", "")
            if isinstance(attr, str) and attr.strip():
                relation_terms.append(attr)
            bt = a.get("bridge_triggers") or []
            ht = a.get("hypothesis_triggers") or []
            if isinstance(bt, list):
                relation_terms.extend([x for x in bt if isinstance(x, str)])
            if isinstance(ht, list):
                relation_terms.extend([x for x in ht if isinstance(x, str)])

    rc = raw_qs.get("retrieval_control") or {}
    if isinstance(rc, dict):
        for k in ["relation_markers", "answerability_markers"]:
            vals = rc.get(k) or []
            if isinstance(vals, list):
                relation_terms.extend([x for x in vals if isinstance(x, str)])

    relation_terms = _dedup_keep_order(relation_terms)[:12]

    scope_terms = raw_qs.get("scope_terms") or []
    if not isinstance(scope_terms, list):
        scope_terms = []
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, dict):
                tos = a.get("time_or_scope") or []
                if isinstance(tos, list):
                    scope_terms.extend([x for x in tos if isinstance(x, str)])
    scope_terms = _dedup_keep_order(scope_terms)[:8]

    retrieval_views = raw_qs.get("retrieval_views") or []
    cleaned_views = {}
    if isinstance(retrieval_views, list):
        for item in retrieval_views:
            if not isinstance(item, dict):
                continue
            name = _norm_key(str(item.get("view_name", "")))
            text = _norm_ws(str(item.get("text", "")))
            if name in {"literal", "entity_expanded", "semantic"} and text:
                cleaned_views[name] = {"view_name": name, "text": " ".join(text.split()[:48])}

    defaults = {
        "literal": {"view_name": "literal", "text": _norm_ws(query)},
        "entity_expanded": {"view_name": "entity_expanded", "text": _norm_ws(query)},
        "semantic": {"view_name": "semantic", "text": _norm_ws(query)},
    }

    retrieval_views = [
        cleaned_views.get("literal", defaults["literal"]),
        cleaned_views.get("entity_expanded", defaults["entity_expanded"]),
        cleaned_views.get("semantic", defaults["semantic"]),
    ]

    return {
        "anchor_entities": anchor_entities,
        "anchor_aliases": anchor_aliases,
        "anchor_terms": anchor_terms,
        "relation_terms": relation_terms,
        "scope_terms": scope_terms,
        "retrieval_views": retrieval_views,
    }


def _build_original_query_prompt(query: str, candidates: List[Dict[str, Any]]):
    """Build a rerank prompt that exposes only the original question."""
    cand_lines = []
    for idx, c in enumerate(candidates, start=1):
        cand_lines.append(
            f"CAND_ID: {idx}\n"
            f"DOC_ID: {c['doc_id']}\n"
            f"SENT_ID: {c['sent_id']}\n"
            f"TEXT: {c['sentence']}\n"
        )

    system = """
You are an evidence reranker for retrieval-augmented QA.

Your job is NOT to answer the question.
Your job is to select a compact but SUFFICIENT evidence set that supports a short,
citation-grounded answer covering every valid answer to the question.

Use only the original question and the candidate sentences.
Prefer factual precision and answer coverage over brevity.

Many questions are ambiguous and have SEVERAL equally valid answers, for example
different entities, dates, scopes, editions, or jurisdictions. Each distinct valid
answer needs its own retained sentence. Dropping an answer is a worse error than
retaining one extra sentence.

Bucket meanings:
- "core": directly supports one valid answer to the question
- "support": contributes a distinct factual detail, date, qualification, scope,
  jurisdiction, or alternative answer that core evidence does not already cover.
  Support need not stand alone as a complete answer, but must be grounded in the
  candidate sentence, relevant to the original query, and safe to cite.
- "drop": weak, noisy, fragmentary, context-dependent, near-duplicate, or unsafe

Return JSON only.
""".strip()

    user = f"""
    Original query:
    {query}

    Task:
    For each candidate sentence, assign:
    - "core"
    - "support"
    - "drop"

    Important rules:
    1. Prefer explicit answer-bearing sentences for core.
    2. Do not mark a sentence core merely because it shares words with the query.
    3. Use support for any distinct, relevant factual contribution: an alternative
       answer, date, qualification, scope, jurisdiction, or useful context not
       already covered by core. A support sentence need NOT be a stand-alone
       answer; it must still state a verifiable fact grounded in its text.
    4. Reject fabricated, clearly irrelevant, or near-duplicate sentences. Do
       NOT treat DIFFERENT valid answers or complementary facts as duplicates.
    5. If the question admits several valid answers, retain evidence for EACH
       answer branch rather than only the first or most common one.
    6. Count candidates that mention the question's subject OR contain a plausible
       answer, distinct qualification, or useful factual context for that subject.
       Let N be this count. You MUST assign at least min(6, N) distinct
       candidates to core or support, with no more than 5 core and 8 support.
       This is a minimum coverage constraint, not a preference. Exclude wholly
       irrelevant sentences from N; never fabricate evidence.
    7. Prefer core for direct answers. If no candidate directly answers the query
       but relevant partial evidence exists, retain the best partial evidence
       as support rather than marking all candidates drop.

    Return STRICT JSON only:
    {{
      "results": [
        {{
          "doc_id": 1,
          "sent_id": 2,
          "bucket": "core|support|drop",
          "score": 0.0,
          "reason": "short phrase"
        }}
      ]
    }}

    Every candidate below must appear exactly once in "results".
    Keep each "reason" under 8 words.

    BEFORE RETURNING, SILENTLY VERIFY ALL THREE CONDITIONS:
    (a) Every candidate appears exactly once, with its original doc_id/sent_id.
    (b) core + support count is at least min(6, N), as defined in rule 6,
        and core <= 5, support <= 8.
    (c) Every distinct supported answer branch has at least one core/support row.
    If (b) or (c) fails, revise the best remaining relevant drop candidates to
    support, then re-check. Never relabel an irrelevant sentence just for count.
    Return only the completed JSON; do not include the checks in the output.

    Candidates:
    {chr(10).join(cand_lines)}
    """.strip()
    return system, user


def build_rerank_prompt(
    query: str,
    qstruct: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    query_mode: str = "structured",
):
    query_mode = (query_mode or "structured").strip().lower()
    if query_mode == "original":
        return _build_original_query_prompt(query, candidates)
    if query_mode != "structured":
        raise ValueError("query_mode must be 'structured' or 'original'")

    qstruct = normalize_query_struct(qstruct, query)

    anchor_entities = qstruct.get("anchor_entities", [])
    anchor_aliases = qstruct.get("anchor_aliases", {})
    anchor_terms = qstruct.get("anchor_terms", [])
    relation_terms = qstruct.get("relation_terms", [])
    scope_terms = qstruct.get("scope_terms", [])
    retrieval_views = qstruct.get("retrieval_views", [])

    cand_lines = []
    for idx, c in enumerate(candidates, start=1):
        cand_lines.append(
            f"CAND_ID: {idx}\n"
            f"DOC_ID: {c['doc_id']}\n"
            f"SENT_ID: {c['sent_id']}\n"
            f"TEXT: {c['sentence']}\n"
        )

    rv_lines = []
    for v in retrieval_views:
        if isinstance(v, dict):
            rv_lines.append(f"- {v.get('view_name', '')}: {v.get('text', '')}")

    alias_lines = []
    for k, vals in (anchor_aliases or {}).items():
        alias_lines.append(f"- {k}: {vals}")

    system = """
You are an evidence reranker for retrieval-augmented QA.

Your job is NOT to answer the question.
Your job is to select a compact but sufficient evidence set that supports a short, citation-grounded answer.

Optimize for BOTH:
1. factual precision
2. factual coverage

Use the query signals carefully:
- anchor_entities / anchor_aliases help disambiguate the target entity or concept
- anchor_terms reflect the explicit ask of the question
- relation_terms describe what answer-bearing evidence should look like
- scope_terms describe explicit constraints
- retrieval_views are semantic rewrites for matching intent

Important:
- relation_terms are only auxiliary guidance.
- Do NOT keep a sentence merely because it matches generic relation words like "record", "top", "most", "performed", or "written".
- Prefer sentences that answer the full query, not just a narrower local record, partial statistic, or loosely related mention.
- When the query asks for an original creator / source / identity / purpose / definition / highest global record,
  prefer globally answer-bearing evidence over local, partial, or derivative evidence.
- Besides the main question, you may also keep sentences that directly support a closely related valid sub-answer implied by the question.
- In particular, if a sentence clearly supports a likely short sub-question in the same answer family, it can be kept as support.
- But topical relevance alone is not enough: the sentence must still support a concrete factual claim that could be cited later.

Bucket meanings:
- "core": directly supports the main answer
- "support": useful secondary evidence for a short independent follow-up sentence
- "drop": weak, noisy, fragmentary, indirect, redundant, or unsafe

Scoring:
- 0.90-1.00 = very strong direct answer evidence
- 0.75-0.89 = strong central evidence, likely core
- 0.60-0.74 = useful support evidence
- 0.40-0.59 = weak support / likely drop
- 0.00-0.39 = drop

Return JSON only.
""".strip()

    user = f"""
    Query:
    {query}
    
    Anchor entities:
    {anchor_entities}
    
    Anchor aliases:
    {chr(10).join(alias_lines) if alias_lines else "[]"}
    
    Anchor terms:
    {anchor_terms}
    
    Relation terms:
    {relation_terms}
    
    Scope terms:
    {scope_terms}
    
    Retrieval views:
    {chr(10).join(rv_lines) if rv_lines else "[]"}
    
    Task:
    For each candidate sentence, assign:
    - "core"
    - "support"
    - "drop"
    
    Important rules:
    1. Prefer explicit answer-bearing sentences for core.
    2. If a sentence only supports a narrower local fact, derivative version, cover version, subset record, team/league record, or partial comparison, do NOT mark it core unless the query itself asks for that narrower scope.
    3. A support sentence may support either the main answer or a closely related short sub-answer that would naturally appear in the same gold answer family.
    4. If a sentence clearly supports a likely sub-question implied by the query, you may keep it as support even if it is not the single best direct answer sentence.
    5. A support sentence must still justify a short stand-alone cited sentence later.
    6. A sentence that is relevant but incomplete, overly local, ambiguous, or context-dependent should often be drop.
    7. Do not reward lexical overlap alone.
    8. Avoid keeping many near-duplicates.
    9. Keep enough evidence for coverage, but not noisy filler.
    
    Return STRICT JSON only:
    {{
      "results": [
        {{
          "doc_id": 1,
          "sent_id": 2,
          "bucket": "core|support|drop",
          "score": 0.0,
          "reason": "short phrase"
        }}
      ]
    }}
    
    Candidates:
    {chr(10).join(cand_lines)}
    """.strip()

    return system, user


_RESULT_OBJ_RE = re.compile(
    r'\{[^{}]*"doc_id"\s*:\s*\d+'
    r'[^{}]*"sent_id"\s*:\s*\d+'
    r'[^{}]*"bucket"\s*:\s*"(?:core|support|drop)"'
    r'[^{}]*"score"\s*:\s*[-+]?\d*\.?\d+'
    r'[^{}]*"reason"\s*:\s*"[^"]*"'
    r'[^{}]*\}',
    flags=re.DOTALL,
)


def _salvage_partial_results(raw_text: str) -> List[Dict[str, Any]]:
    results = []
    for m in _RESULT_OBJ_RE.finditer(raw_text or ""):
        block = m.group(0)
        try:
            item = json.loads(block)
            if isinstance(item, dict):
                results.append(item)
        except Exception:
            continue
    return results


def parse_rerank_json(raw_text: str) -> Dict[str, Any]:
    raw_text = strip_qwen_thinking(raw_text)

    # 1) normal parse
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) try outer block
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) try common trailing comma repair
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if m:
        block = m.group(0)
        repaired = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 4) ast literal fallback
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            obj = ast.literal_eval(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 5) salvage partial items from truncated output
    salvaged = _salvage_partial_results(raw_text)
    if salvaged:
        return {"results": salvaged}

    raise ValueError(f"Invalid rerank JSON:\n{raw_text}")


def _make_slim_row_with_meta(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    out = {
        "doc_id": int(row.get("doc_id", -1)),
        "sent_id": int(row.get("sent_id", -1)),
        "sent_content": row.get("sentence", "") or "",
        "source": source,
        "bucket": row.get("bucket", ""),
        "llm_score": float(row.get("llm_score", 0.0)),
    }
    sc = row.get("score", {}) or {}
    if "total" in sc:
        out["heuristic_score"] = float(sc["total"])
    return out


def pack_final_evidence(
    labeled_rows: List[Dict[str, Any]],
    topk_core: int = 5,
    topk_support: int = 8,
    max_packed_sents: int = 14,
    min_packed_sents: Optional[int] = None,
):
    """Pack core + support evidence, with a floor against all-drop teacher output.

    The floor exists because an empty or near-empty packed set caps downstream
    answer coverage at zero by construction. Fallback rows keep their original
    "drop" bucket and are tagged source="fallback" so they stay auditable and
    can be excluded from reranker supervision if desired.
    """
    if min_packed_sents is None:
        min_packed_sents = MIN_PACKED_SENTS

    core_rows = [r for r in labeled_rows if r.get("bucket") == "core"]
    support_rows = [r for r in labeled_rows if r.get("bucket") == "support"]
    drop_rows = [r for r in labeled_rows if r.get("bucket") == "drop"]

    core_rows.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)
    support_rows.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)
    drop_rows.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)

    core_rows = core_rows[:topk_core]
    support_rows = support_rows[:topk_support]

    packed = []
    seen = set()

    for group_name, rows in [("core", core_rows), ("support", support_rows)]:
        for r in rows:
            if len(packed) >= max_packed_sents:
                break

            k = (int(r["doc_id"]), int(r["sent_id"]))
            if k in seen:
                continue

            seen.add(k)
            packed.append(_make_slim_row_with_meta(r, source=group_name))

    floor = min(int(min_packed_sents), int(max_packed_sents))
    if len(packed) < floor and drop_rows:
        # Teacher scores are often all zero in this case, so break ties on the
        # heuristic score that produced the candidate pool.
        fallback = sorted(
            drop_rows,
            key=lambda r: (float(r.get("llm_score", 0.0)),
                           float((r.get("score") or {}).get("total", 0.0))),
            reverse=True,
        )
        for r in fallback:
            if len(packed) >= floor:
                break
            k = (int(r["doc_id"]), int(r["sent_id"]))
            if k in seen:
                continue
            seen.add(k)
            packed.append(_make_slim_row_with_meta(r, source="fallback"))

    return core_rows, support_rows, drop_rows, packed


# ============================================================
# Custom id helpers
# ============================================================
CUST_RE = re.compile(r"^rerank-(\d+)-chunk-(\d+)$", re.I)
_JSONL_RANGE_RE = re.compile(r"_(\d{4,})_(\d{4,})\.jsonl$", re.I)


def make_custom_id(sample_idx: int, chunk_idx: int) -> str:
    return f"rerank-{sample_idx}-chunk-{chunk_idx}"


def parse_custom_id(custom_id: str) -> Tuple[int, int]:
    m = CUST_RE.match(custom_id or "")
    if not m:
        raise ValueError(f"Bad custom_id: {custom_id}")
    return int(m.group(1)), int(m.group(2))


def _infer_range_from_jsonl_path(jsonl_path: str) -> Tuple[int, int]:
    name = Path(jsonl_path).name
    m = _JSONL_RANGE_RE.search(name)
    if not m:
        raise ValueError(f"cannot infer start/end idx from jsonl filename: {jsonl_path}")
    return int(m.group(1)), int(m.group(2))


# ============================================================
# Durable, single-TSV Batch registry
# ============================================================
TSV_FIELDS = [
    "jsonl", "meta", "batch_id", "status", "input_file_id",
    "sha256", "candidate_dir", "model", "rerank_batch",
    "max_input_candidates", "query_mode", "max_output_tokens",
    "submitted_at", "status_updated_at",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_path(out_dir: Path, job_name: str, tsv: str = "") -> Path:
    return Path(tsv).expanduser().resolve() if tsv else (out_dir / f"{job_name}.batches.tsv").resolve()


@contextmanager
def _registry_lock(tsv_path: Path):
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(tsv_path) + ".lock")
    with lock_path.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read_batches_tsv(tsv_path: Path) -> List[Dict[str, str]]:
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Batch TSV not found: {tsv_path}")
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "batch_id" not in reader.fieldnames or "jsonl" not in reader.fieldnames:
            raise ValueError(f"Invalid Batch TSV header: {tsv_path}")
        return [dict(row) for row in reader if row.get("batch_id")]


def _save_batches_tsv(tsv_path: Path, rows: List[Dict[str, str]]) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement: never lose the mapping when a job or SSH session fails.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=tsv_path.parent,
        prefix=tsv_path.name + ".tmp.", delete=False,
    ) as f:
        tmp_path = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(tsv_path)


def _latest_rows_per_jsonl(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # The registry retains ALL batch IDs. Recovery uses only the latest attempt
    # for a given JSONL, avoiding stale labels from resubmissions.
    seen = set()
    kept = []
    for row in reversed(rows):
        key = str(Path(row["jsonl"]).expanduser().resolve())
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return list(reversed(kept))


def _row_config(row: Dict[str, str], args):
    def number(field, default):
        return int(row[field]) if row.get(field) else int(default)
    return {
        "candidate_dir": Path(row.get("candidate_dir") or args.candidate_dir).expanduser(),
        "openai_model": row.get("model") or args.openai_model,
        "rerank_batch": number("rerank_batch", args.rerank_batch),
        "max_input_candidates": number("max_input_candidates", args.max_input_candidates),
        "query_mode": row.get("query_mode") or args.query_mode,
        "max_output_tokens": number("max_output_tokens", args.max_output_tokens),
    }


# ============================================================
# Candidate loading
# ============================================================
def load_candidate_json(candidate_path: Path) -> dict:
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate json not found: {candidate_path}")
    with candidate_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"candidate json must be object: {candidate_path}")
    # Match the inference entry point: query_struct is optional.
    for k in ["query", "candidate_pool"]:
        if k not in obj:
            raise ValueError(f"candidate json missing key '{k}': {candidate_path}")
    if not isinstance(obj["candidate_pool"], list):
        raise ValueError(f"candidate_pool must be list: {candidate_path}")
    return obj


def resolve_candidate_path(sample_idx: int, candidate_dir: Path) -> Path:
    return candidate_dir / f"candidate_pool_{sample_idx}.json"


def iter_sample_indices(start_idx: int, end_idx: int) -> List[int]:
    if end_idx < 0:
        raise ValueError("END_IDX must be >= 0 for deterministic batch build.")
    if end_idx <= start_idx:
        raise ValueError(f"Require END_IDX > START_IDX, got start={start_idx}, end={end_idx}")
    return list(range(start_idx, end_idx))


# ============================================================
# Build batch requests
# ============================================================
def build_one_request(
    sample_idx: int,
    chunk_idx: int,
    query: str,
    qstruct: Dict[str, Any],
    batch_candidates: List[Dict[str, Any]],
    openai_model: str,
    max_output_tokens: int,
    query_mode: str = "structured",
) -> Dict[str, Any]:
    # Inference main normalizes before passing qstruct to build_rerank_prompt;
    # build_rerank_prompt also normalizes in structured mode. Match both.
    qstruct = normalize_query_struct(qstruct, query)
    system_prompt, user_prompt = build_rerank_prompt(
        query, qstruct, batch_candidates, query_mode=query_mode,
    )
    return {
        "custom_id": make_custom_id(sample_idx, chunk_idx),
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": openai_model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": int(max_output_tokens),
        },
    }


def chunk_candidates(candidate_pool: List[Dict[str, Any]], rerank_batch: int) -> List[List[Dict[str, Any]]]:
    if rerank_batch <= 0:
        raise ValueError("rerank_batch must be > 0")
    return [candidate_pool[i:i + rerank_batch] for i in range(0, len(candidate_pool), rerank_batch)]


def cmd_build_split(args):
    candidate_dir = Path(args.candidate_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    chunk_size = int(args.chunk_size)
    rerank_batch = int(args.rerank_batch)
    max_input_candidates = int(args.max_input_candidates)
    query_mode = args.query_mode

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if rerank_batch <= 0:
        raise ValueError("rerank_batch must be > 0")
    if max_input_candidates < 0:
        raise ValueError("max_input_candidates must be >= 0")
    if query_mode not in {"structured", "original"}:
        raise ValueError("query_mode must be 'structured' or 'original'")

    sample_indices = iter_sample_indices(start_idx, end_idx)
    total_requests = 0
    total_samples = 0
    total_missing = 0
    files = []

    for split_start_pos in range(0, len(sample_indices), chunk_size):
        split_indices = sample_indices[split_start_pos: split_start_pos + chunk_size]
        split_start_idx = split_indices[0]
        split_end_idx = split_indices[-1] + 1
        out_jsonl = out_dir / f"{args.job_name}_{split_start_idx:04d}_{split_end_idx:04d}.jsonl"

        written = 0
        missing = 0
        covered_samples = 0

        with out_jsonl.open("w", encoding="utf-8") as fout:
            for sample_idx in split_indices:
                candidate_path = resolve_candidate_path(sample_idx, candidate_dir)
                if not candidate_path.exists():
                    print(f"[WARN] missing candidate json: {candidate_path}")
                    missing += 1
                    continue

                obj = load_candidate_json(candidate_path)
                query = obj["query"]
                qstruct = obj.get("query_struct", {})
                candidate_pool = obj["candidate_pool"][:max_input_candidates]

                batches = chunk_candidates(candidate_pool, rerank_batch)
                for chunk_idx, batch_candidates in enumerate(batches):
                    req = build_one_request(
                        sample_idx=sample_idx,
                        chunk_idx=chunk_idx,
                        query=query,
                        qstruct=qstruct,
                        batch_candidates=batch_candidates,
                        openai_model=args.openai_model,
                        max_output_tokens=int(args.max_output_tokens),
                        query_mode=query_mode,
                    )
                    fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                    written += 1
                covered_samples += 1

        if written > 0:
            _ensure_nonempty_file(out_jsonl, "rerank split jsonl")
            files.append(str(out_jsonl))
        else:
            # Prevent a stale job file from being submitted after a rebuild.
            out_jsonl.unlink(missing_ok=True)

        total_requests += written
        total_samples += covered_samples
        total_missing += missing

    if total_requests == 0:
        raise RuntimeError(
            f"build_split wrote 0 requests. Check candidate_dir/start/end. "
            f"candidate_dir={candidate_dir} start={start_idx} end={end_idx}"
        )

    payload = {
        "candidate_dir": str(candidate_dir),
        "out_dir": str(out_dir),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "chunk_size": chunk_size,
        "rerank_batch": rerank_batch,
        "max_input_candidates": max_input_candidates,
        "query_mode": query_mode,
        "num_files": len(files),
        "num_samples_covered": total_samples,
        "num_missing_candidate_json": total_missing,
        "num_requests": total_requests,
        "files": files,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_submit_split(args):
    """Build requested shards, submit each exactly once by default, and persist every ID."""
    api_configuration("openai")
    if args.completion_window != "24h":
        raise ValueError("Batch completion_window currently supports '24h' only.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = _registry_path(out_dir, args.job_name, args.tsv)

    # Serialize build+submit for this TSV. An interruption leaves already-submitted
    # IDs on disk; another invocation resumes remaining shards without duplication.
    with _registry_lock(tsv_path):
        cmd_build_split(args)
        rows = _read_batches_tsv(tsv_path) if tsv_path.exists() else []
        # Do not silently create overlapping shards under the same registry if
        # CHUNK_SIZE or START/END boundaries are changed between runs.
        active = _latest_rows_per_jsonl(rows)
        for first in range(int(args.start_idx), int(args.end_idx), int(args.chunk_size)):
            last = min(first + int(args.chunk_size), int(args.end_idx))
            new_jp = (out_dir / f"{args.job_name}_{first:04d}_{last:04d}.jsonl").resolve()
            if not new_jp.is_file() or new_jp.stat().st_size == 0:
                continue
            for prior in active:
                old_jp = Path(prior["jsonl"]).expanduser().resolve()
                if old_jp == new_jp:
                    continue
                a, b = _infer_range_from_jsonl_path(prior["jsonl"])
                if first < b and a < last:
                    raise RuntimeError(
                        f"Overlapping shard ranges in registry: {new_jp.name} and {old_jp.name}. "
                        "Use a new JOB_NAME (and separate output root) when changing shard boundaries."
                    )
        submitted, skipped = 0, 0
        for first in range(int(args.start_idx), int(args.end_idx), int(args.chunk_size)):
            last = min(first + int(args.chunk_size), int(args.end_idx))
            jp = (out_dir / f"{args.job_name}_{first:04d}_{last:04d}.jsonl").resolve()
            if not jp.is_file() or jp.stat().st_size == 0:
                print(f"[WARN] skip empty/missing JSONL: {jp}")
                continue
            digest = _file_sha256(jp)
            prev = [r for r in rows if Path(r["jsonl"]).expanduser().resolve() == jp]
            if prev and not args.resubmit:
                latest = prev[-1]
                if latest.get("sha256") and latest["sha256"] != digest:
                    raise RuntimeError(
                        f"JSONL changed after submission: {jp}. Do not overwrite old jobs. "
                        "Use a different JOB_NAME or --resubmit explicitly."
                    )
                skipped += 1
                print(f"[SKIP already submitted] {jp.name} -> {latest['batch_id']} ({latest.get('status')})")
                continue

            payload = _submit_one_jsonl(jp, args.openai_model, args.completion_window, jp.stem)
            # Preserve a unique meta for every attempt, including forced resubmission.
            meta_path = out_dir / f"{jp.stem}.{payload['batch_id']}.meta.json"
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            row = {
                "jsonl": str(jp), "meta": str(meta_path),
                "batch_id": payload["batch_id"], "status": payload["status"],
                "input_file_id": payload["input_file_id"], "sha256": digest,
                "candidate_dir": str(Path(args.candidate_dir).expanduser().resolve()),
                "model": args.openai_model, "rerank_batch": str(args.rerank_batch),
                "max_input_candidates": str(args.max_input_candidates),
                "query_mode": args.query_mode, "max_output_tokens": str(args.max_output_tokens),
                "submitted_at": _now_tag(), "status_updated_at": "",
            }
            rows.append(row)
            _save_batches_tsv(tsv_path, rows)  # immediately persist after EACH API submission
            print(f"[SUBMITTED] {jp.name} -> {payload['batch_id']}")
            submitted += 1

    print(json.dumps({"tsv": str(tsv_path), "submitted": submitted,
                      "skipped_duplicate": skipped, "total_registered": len(rows),
                      "active_shards": len(_latest_rows_per_jsonl(rows))},
                     ensure_ascii=False, indent=2))


def cmd_status(args):
    api_configuration("openai")
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


def cmd_status_from_tsv(args):
    api_configuration("openai")
    client = _get_client()
    tsv_path = Path(args.tsv).expanduser().resolve()
    with _registry_lock(tsv_path):
        rows = _read_batches_tsv(tsv_path)
        results = []
        for row in rows:
            try:
                batch = client.batches.retrieve(row["batch_id"])
                rc = getattr(batch, "request_counts", None)
                row["status"] = batch.status
                row["status_updated_at"] = _now_tag()
                results.append({
                    "batch_id": batch.id, "jsonl": row["jsonl"],
                    "status": batch.status, "output_file_id": getattr(batch, "output_file_id", None),
                    "error_file_id": getattr(batch, "error_file_id", None),
                    "total": getattr(rc, "total", None),
                    "completed": getattr(rc, "completed", None),
                    "failed": getattr(rc, "failed", None),
                })
            except Exception as e:
                results.append({"batch_id": row["batch_id"], "jsonl": row["jsonl"],
                                "status": "ERROR_CHECK", "error": str(e)})
        _save_batches_tsv(tsv_path, rows)
    print(json.dumps({"tsv": str(tsv_path), "num_batches": len(results),
                      "results": results}, ensure_ascii=False, indent=2))


# ============================================================
# Consume and reconstruct final topk_{idx}.json
# ============================================================
def collect_raw_results_for_one_sample(
    candidate_pool: List[Dict[str, Any]],
    line_results: Dict[int, str],
    rerank_batch: int,
) -> List[Dict[str, Any]]:
    """Mirror inference rerank_candidates postprocessing for batch responses."""
    labeled: List[Dict[str, Any]] = []

    for chunk_idx, idx in enumerate(range(0, len(candidate_pool), rerank_batch)):
        batch = candidate_pool[idx:idx + rerank_batch]
        raw = strip_qwen_thinking(line_results.get(chunk_idx, ""))
        try:
            parsed = parse_rerank_json(raw)
            results = parsed.get("results", [])
            if not isinstance(results, list):
                results = []
        except Exception as e:
            print(f"[WARN] parse_rerank_json failed at batch starting idx={idx}: {e}")
            results = []

        if not results:
            salvaged = _salvage_partial_results(raw)
            if salvaged:
                print(f"[WARN] using salvaged partial results at batch idx={idx}: {len(salvaged)} items")
                results = salvaged

        result_map = {}
        for r in results:
            try:
                k = (int(r["doc_id"]), int(r["sent_id"]))
                bucket = str(r.get("bucket", "drop")).strip().lower()
                if bucket not in {"core", "support", "drop"}:
                    bucket = "drop"
                score = float(r.get("score", 0.0))
                reason = str(r.get("reason", "")).strip()
                result_map[k] = {
                    "bucket": bucket,
                    "llm_score": max(0.0, min(1.0, score)),
                    "reason": reason,
                }
            except Exception:
                continue

        for c in batch:
            k = (int(c["doc_id"]), int(c["sent_id"]))
            rr = result_map.get(k, {"bucket": "drop", "llm_score": 0.0, "reason": "missing_from_model_output"})
            labeled.append({
                **c,
                "bucket": rr["bucket"],
                "llm_score": rr["llm_score"],
                "reason": rr["reason"],
            })

    labeled.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)
    return labeled


def write_final_topk_json(
    sample_idx: int,
    candidate_path: Path,
    obj: dict,
    line_results: Dict[int, str],
    rerank_batch: int,
    topk_core: int,
    topk_support: int,
    max_packed_sents: int,
    out_dir: Path,
    openai_model: str,
    max_input_candidates: int = 18,
    query_mode: str = "structured",
    max_output_tokens: int = 1400,
    allow_partial: bool = False,
    backend: str = "openai_batch",
):
    if rerank_batch <= 0:
        raise ValueError("rerank_batch must be > 0")
    if max_input_candidates < 0:
        raise ValueError("max_input_candidates must be >= 0")
    query = obj["query"]
    candidate_pool = obj["candidate_pool"][:max_input_candidates]
    num_chunks = len(chunk_candidates(candidate_pool, rerank_batch))
    missing_chunks = [i for i in range(num_chunks) if i not in line_results or not line_results[i]]
    if missing_chunks and not allow_partial:
        # A missing API response is a failed batch job, not an evidence "drop".
        # The inference parser itself defaults missing *labels within* a response
        # to drop; keep that behavior but do not silently hide missing responses.
        raise ValueError(f"Missing batch response chunks for sample {sample_idx}: {missing_chunks}")

    labeled_rows = collect_raw_results_for_one_sample(
        candidate_pool=candidate_pool,
        line_results=line_results,
        rerank_batch=rerank_batch,
    )

    core_rows, support_rows, drop_rows, packed = pack_final_evidence(
        labeled_rows=labeled_rows,
        topk_core=topk_core,
        topk_support=topk_support,
        max_packed_sents=max_packed_sents,
    )

    out = {
        "sample_idx": obj.get("sample_idx", sample_idx),
        "query": query,
        "query_struct": obj.get("query_struct", {}),
        "candidate_json": str(candidate_path),
        "core_evidence": [_make_slim_row_with_meta(r, "core") for r in core_rows],
        "support_evidence": [_make_slim_row_with_meta(r, "support") for r in support_rows],
        "dropped": [_make_slim_row_with_meta(r, "drop") for r in drop_rows[:200]],
        "packed_evidence": packed,
        "rerank_meta": {
            "backend": backend,
            "openai_model": openai_model,
            "local_model": None,
            "base_tokenizer": None,
            "disable_qwen_thinking": None,
            "num_candidates": len(candidate_pool),
            "topk_core": topk_core,
            "topk_support": topk_support,
            "max_packed_sents": max_packed_sents,
            "rerank_batch": rerank_batch,
            "rerank_infer_batch": None,
            "query_mode": query_mode,
            "max_output_tokens": max_output_tokens,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"topk_{sample_idx}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def cmd_consume(args):
    api_configuration("openai")

    client = _get_client()
    batch = client.batches.retrieve(args.batch_id)

    print(f"[INFO] batch_id={batch.id}")
    print(f"[INFO] status={batch.status}")

    if batch.status != "completed":
        raise RuntimeError(f"Batch is not completed. Current status: {batch.status}")

    ofid = getattr(batch, "output_file_id", None)
    if not ofid:
        raise RuntimeError("Batch completed but output_file_id is missing.")

    raw_out_path: Optional[Path] = Path(args.raw_out_jsonl).expanduser() if args.raw_out_jsonl else None
    if raw_out_path:
        raw_out_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_dir = Path(args.candidate_dir).expanduser()
    out_dir = Path(args.final_out_dir).expanduser()
    failed_out: Optional[Path] = Path(args.failed_ids_out).expanduser() if args.failed_ids_out else None
    if failed_out:
        failed_out.parent.mkdir(parents=True, exist_ok=True)

    start_i = int(args.start_idx)
    end_i = int(args.end_idx)
    rerank_batch = int(args.rerank_batch)

    text = client.files.content(ofid).text

    if raw_out_path:
        raw_out_path.write_text(text, encoding="utf-8")
        print(f"[saved raw batch output] {raw_out_path}")

    grouped: DefaultDict[int, Dict[int, str]] = defaultdict(dict)
    failed = 0
    failed_rows: List[str] = []

    for line in text.splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj.get("custom_id", "")

        try:
            sample_idx, chunk_idx = parse_custom_id(custom_id)

            if sample_idx < start_i:
                continue
            if end_i >= 0 and sample_idx >= end_i:
                continue

            raw_text = extract_output_text_from_batch_line(obj)
            grouped[sample_idx][chunk_idx] = raw_text

        except Exception as e:
            failed += 1
            failed_rows.append(f"{custom_id}\t{e}")
            print(f"[FAILED-PARSE] {custom_id}: {e}")

    ok = 0
    missing_candidate = 0

    target_indices = iter_sample_indices(start_i, end_i)

    for sample_idx in target_indices:
        try:
            candidate_path = resolve_candidate_path(sample_idx, candidate_dir)
            if not candidate_path.exists():
                missing_candidate += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing candidate json: {candidate_path}")
                print(f"[WARN] missing candidate json: {candidate_path}")
                continue

            cand_obj = load_candidate_json(candidate_path)
            line_results = grouped.get(sample_idx, {})
            out_path = write_final_topk_json(
                sample_idx=sample_idx,
                candidate_path=candidate_path,
                obj=cand_obj,
                line_results=line_results,
                rerank_batch=rerank_batch,
                topk_core=int(args.topk_core),
                topk_support=int(args.topk_support),
                max_packed_sents=int(args.max_packed_sents),
                out_dir=out_dir,
                openai_model=args.openai_model,
                max_input_candidates=int(args.max_input_candidates),
                query_mode=args.query_mode,
                max_output_tokens=int(args.max_output_tokens),
                allow_partial=args.allow_partial,
            )
            print(f"[OK] wrote {out_path}")
            ok += 1

        except Exception as e:
            failed += 1
            failed_rows.append(f"sample-{sample_idx}\t{e}")
            print(f"[FAILED-WRITE] sample-{sample_idx}: {e}")

    summary = {
        "batch_id": batch.id,
        "status": batch.status,
        "ok": ok,
        "failed": failed,
        "missing_candidate_json": missing_candidate,
        "final_out_dir": str(out_dir),
        "start_idx": start_i,
        "end_idx": end_i,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failed_out:
        failed_out.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
        print(f"[saved failed ids] {failed_out}")


def cmd_consume_from_tsv(args):
    """One command: download every completed shard, reconstruct topk samples."""
    api_configuration("openai")
    client = _get_client()
    tsv_path = Path(args.tsv).expanduser().resolve()
    rows = _latest_rows_per_jsonl(_read_batches_tsv(tsv_path))
    final_out_dir = Path(args.final_out_dir).expanduser().resolve()
    jobs_dir = Path(args.jobs_dir).expanduser().resolve()
    final_out_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Fail early if two different shards include the same sample indices.
    claimed = set()
    for row in rows:
        start, end = _infer_range_from_jsonl_path(row["jsonl"])
        overlap = claimed.intersection(range(start, end))
        if overlap:
            raise RuntimeError(f"Overlapping samples across TSV shards: {sorted(overlap)[:8]}")
        claimed.update(range(start, end))

    consumed = 0
    written = 0
    skipped_existing = 0
    skipped_not_completed = 0
    failed_rows = []
    for row in rows:
        batch_id, jsonl_path = row["batch_id"], row["jsonl"]
        first, last = _infer_range_from_jsonl_path(jsonl_path)
        cfg = _row_config(row, args)
        try:
            batch = client.batches.retrieve(batch_id)
            if batch.status != "completed":
                skipped_not_completed += 1
                print(f"[SKIP] {batch_id} status={batch.status} jsonl={jsonl_path}")
                continue
            output_file_id = getattr(batch, "output_file_id", None)
            if not output_file_id:
                failed_rows.append(f"{batch_id}\tmissing_output_file_id")
                continue
            raw_text = client.files.content(output_file_id).text
            stem = Path(jsonl_path).stem
            raw_out_path = jobs_dir / f"{stem}.{batch_id}.output.jsonl"
            raw_out_path.write_text(raw_text, encoding="utf-8")
            print(f"[raw] {raw_out_path}")
            # Save API request-error details separately, for selective regeneration.
            err_id = getattr(batch, "error_file_id", None)
            if err_id:
                (jobs_dir / f"{stem}.{batch_id}.errors.jsonl").write_text(
                    client.files.content(err_id).text, encoding="utf-8"
                )
            grouped: DefaultDict[int, Dict[int, str]] = defaultdict(dict)
            batch_failures = []
            for lineno, line in enumerate(raw_text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    idx, chunk_idx = parse_custom_id(obj.get("custom_id", ""))
                    if not first <= idx < last:
                        raise ValueError(f"sample idx {idx} outside shard [{first},{last})")
                    resp = obj.get("response") or {}
                    if obj.get("error") or int(resp.get("status_code", 200)) >= 400:
                        raise RuntimeError(f"API failed: {obj.get('error') or resp.get('status_code')}")
                    raw = extract_output_text_from_batch_line(obj)
                    if chunk_idx in grouped[idx]:
                        raise RuntimeError("duplicate response custom_id")
                    grouped[idx][chunk_idx] = raw
                except Exception as e:
                    batch_failures.append(f"{batch_id}\tline-{lineno}\tparse_error\t{e}")
            for idx in range(first, last):
                try:
                    candidate_path = resolve_candidate_path(idx, cfg["candidate_dir"])
                    if not candidate_path.is_file():
                        # The builder might have skipped missing inputs; do not fabricate labels.
                        raise FileNotFoundError(f"missing candidate json: {candidate_path}")
                    out_path = final_out_dir / f"topk_{idx}.json"
                    if out_path.exists() and out_path.stat().st_size and not args.overwrite:
                        skipped_existing += 1
                        continue
                    cand_obj = load_candidate_json(candidate_path)
                    out_path = write_final_topk_json(
                        sample_idx=idx, candidate_path=candidate_path, obj=cand_obj,
                        line_results=grouped.get(idx, {}),
                        rerank_batch=cfg["rerank_batch"],
                        topk_core=int(args.topk_core), topk_support=int(args.topk_support),
                        max_packed_sents=int(args.max_packed_sents), out_dir=final_out_dir,
                        openai_model=cfg["openai_model"],
                        max_input_candidates=cfg["max_input_candidates"],
                        query_mode=cfg["query_mode"],
                        max_output_tokens=cfg["max_output_tokens"],
                        allow_partial=args.allow_partial,
                    )
                    written += 1
                    print(f"[OK] {out_path}")
                except Exception as e:
                    batch_failures.append(f"{batch_id}\tsample-{idx}\twrite_error\t{e}")
            failed_rows.extend(batch_failures)
            failures_path = jobs_dir / f"{stem}.{batch_id}.failed_ids.txt"
            failures_path.write_text("\n".join(batch_failures) + ("\n" if batch_failures else ""), encoding="utf-8")
            consumed += 1
            print(f"[CONSUMED] {batch_id}; per-batch errors={len(batch_failures)}")
        except Exception as e:
            failed_rows.append(f"{batch_id}\tconsume_error\t{e}")
            print(f"[FAILED] {batch_id}: {e}")

    failed_out = Path(args.failed_ids_out).expanduser() if args.failed_ids_out else jobs_dir / "consume_from_tsv.failed_ids.txt"
    failed_out.parent.mkdir(parents=True, exist_ok=True)
    failed_out.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
    print(json.dumps({
        "tsv": str(tsv_path), "active_batches": len(rows), "consumed": consumed,
        "skipped_not_completed": skipped_not_completed,
        "topk_written": written, "skipped_existing": skipped_existing,
        "errors": len(failed_rows), "failed_ids_out": str(failed_out),
        "candidate_dir_fallback": str(args.candidate_dir),
        "final_out_dir": str(final_out_dir),
    }, ensure_ascii=False, indent=2))


# ============================================================
# Consume local output jsonl
# ============================================================
def cmd_consume_local(args):
    import glob

    candidate_dir = Path(args.candidate_dir).expanduser()
    jobs_dir = Path(args.jobs_dir).expanduser()
    pattern = args.pattern
    out_dir = Path(args.final_out_dir).expanduser()
    failed_out = Path(args.failed_ids_out).expanduser()

    start_i = int(args.start_idx)
    end_i = int(args.end_idx)

    paths = [Path(p) for p in glob.glob(str(jobs_dir / pattern))]
    if not paths:
        raise FileNotFoundError(f"no files match: {jobs_dir / pattern}")

    out_dir.mkdir(parents=True, exist_ok=True)
    failed_out.parent.mkdir(parents=True, exist_ok=True)

    grouped: DefaultDict[int, Dict[int, str]] = defaultdict(dict)
    failed_rows: List[str] = []
    parse_failed = 0

    for fp in sorted(paths):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")

            try:
                sample_idx, chunk_idx = parse_custom_id(cid)
                if sample_idx < start_i:
                    continue
                if end_i >= 0 and sample_idx >= end_i:
                    continue
                raw = extract_output_text_from_batch_line(obj)
                grouped[sample_idx][chunk_idx] = raw
            except Exception as e:
                parse_failed += 1
                failed_rows.append(f"{cid}\t{e}")

    ok = 0
    write_failed = 0
    missing_candidate = 0

    for sample_idx in iter_sample_indices(start_i, end_i):
        try:
            candidate_path = resolve_candidate_path(sample_idx, candidate_dir)
            if not candidate_path.exists():
                missing_candidate += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing candidate json: {candidate_path}")
                continue

            cand_obj = load_candidate_json(candidate_path)
            out_path = out_dir / f"topk_{sample_idx}.json"
            if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
                print(f"[SKIP] existing topk output (use --overwrite to replace): {out_path}")
                continue

            out_path = write_final_topk_json(
                sample_idx=sample_idx,
                candidate_path=candidate_path,
                obj=cand_obj,
                line_results=grouped.get(sample_idx, {}),
                rerank_batch=int(args.rerank_batch),
                topk_core=int(args.topk_core),
                topk_support=int(args.topk_support),
                max_packed_sents=int(args.max_packed_sents),
                out_dir=out_dir,
                openai_model=args.openai_model,
                max_input_candidates=int(args.max_input_candidates),
                query_mode=args.query_mode,
                max_output_tokens=int(args.max_output_tokens),
                allow_partial=args.allow_partial,
            )
            print(f"[OK] wrote {out_path}")
            ok += 1

        except Exception as e:
            write_failed += 1
            failed_rows.append(f"sample-{sample_idx}\t{e}")

    failed_out.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")

    print(json.dumps({
        "jobs_dir": str(jobs_dir),
        "pattern": pattern,
        "final_out_dir": str(out_dir),
        "start_idx": start_i,
        "end_idx": end_i,
        "num_output_files": len(paths),
        "ok": ok,
        "parse_failed": parse_failed,
        "write_failed": write_failed,
        "missing_candidate_json": missing_candidate,
        "failed_out": str(failed_out),
    }, indent=2, ensure_ascii=False))


# ============================================================
# Proxy Chat Completions transport, reusing the original R3 prompt,
# parser, normalization, chunking and evidence packer UNCHANGED.
# The provider does not support OpenAI native Batch.
# ============================================================
def _get_proxy_chat_client():
    return api_client("proxy")


def _write_atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def cmd_run_chat(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import Counter

    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    out_dir = Path(args.final_out_dir).expanduser().resolve()
    jobs_dir = Path(args.jobs_dir).expanduser().resolve()
    run_dir = jobs_dir / args.job_name
    raw_dir = run_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    start, end = int(args.start_idx), int(args.end_idx)
    indices = iter_sample_indices(start, end)  # same [start, end) as Batch R3
    if int(args.workers) <= 0:
        raise ValueError("workers must be > 0")

    settings = {
        "candidate_dir": str(candidate_dir),
        "final_out_dir": str(out_dir),
        "model_requested": args.openai_model,
        "api_base_url": api_configuration("proxy")[1],
        "query_mode": args.query_mode,
        "rerank_batch": int(args.rerank_batch),
        "max_input_candidates": int(args.max_input_candidates),
        "max_output_tokens": int(args.max_output_tokens),
        "topk_core": int(args.topk_core),
        "topk_support": int(args.topk_support),
        "max_packed_sents": int(args.max_packed_sents),
        "min_packed_sents": MIN_PACKED_SENTS,
    }
    config_path = run_dir / "run_chat.config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != settings:
            raise RuntimeError(
                "Existing job uses different R3 parameters. Use a different JOB_NAME "
                f"or preserve settings: {config_path}"
            )
    else:
        _write_atomic_json(config_path, settings)

    pending = []
    existing_valid = 0
    for i in indices:
        path = out_dir / f"topk_{i}.json"
        if path.is_file() and path.stat().st_size and not args.overwrite:
            obj = json.loads(path.read_text(encoding="utf-8"))
            meta = obj.get("rerank_meta") or {}
            expected = {
                "query_mode": args.query_mode,
                "rerank_batch": int(args.rerank_batch),
                "num_candidates": None,
                "max_output_tokens": int(args.max_output_tokens),
                "topk_core": int(args.topk_core),
                "topk_support": int(args.topk_support),
                "max_packed_sents": int(args.max_packed_sents),
            }
            expected.pop("num_candidates")
            if obj.get("sample_idx") != i or any(meta.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f"Existing topk_{i}.json is incompatible with R3 settings: {path}")
            existing_valid += 1
        else:
            pending.append(i)

    print(f"[CHAT RUN] requested={len(indices)} pending={len(pending)} "
          f"existing_valid={existing_valid} workers={args.workers} "
          f"model={args.openai_model} query_mode={args.query_mode} "
          f"range=[{start},{end})", flush=True)
    print("[MODE] Proxy Chat Completions (NOT native Batch billing); R3 prompt/packing unchanged", flush=True)

    thread_state = __import__("threading").local()

    def worker(i):
        # Each worker uses its own OpenAI SDK client.
        client = getattr(thread_state, "client", None)
        candidate_path = resolve_candidate_path(i, candidate_dir)
        obj = load_candidate_json(candidate_path)
        if obj.get("sample_idx", i) != i:
            raise ValueError(f"candidate_pool_{i}.json sample_idx mismatch")
        candidates = obj["candidate_pool"][:int(args.max_input_candidates)]
        chunks = chunk_candidates(candidates, int(args.rerank_batch))
        line_results = {}
        versions = Counter()
        for chunk_index, chunk in enumerate(chunks):
            request = build_one_request(
                sample_idx=i, chunk_idx=chunk_index, query=obj["query"],
                qstruct=obj.get("query_struct", {}), batch_candidates=chunk,
                openai_model=args.openai_model,
                max_output_tokens=int(args.max_output_tokens),
                query_mode=args.query_mode,
            )
            # Only translate API transport: Responses.input -> Chat.messages,
            # Responses.max_output_tokens -> Chat.max_tokens. Content is identical.
            payload = request["body"]
            prompt_digest = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            raw_file = raw_dir / f"rerank-{i}-chunk-{chunk_index}.json"
            cached = None
            if raw_file.exists() and not args.overwrite:
                cached = json.loads(raw_file.read_text(encoding="utf-8"))
                if cached.get("prompt_sha256") != prompt_digest:
                    cached = None
            if cached:
                text = cached["output_text"]
                returned_model = cached.get("returned_model") or "unknown"
            else:
                if client is None:
                    client = _get_proxy_chat_client()
                    thread_state.client = client
                response = client.chat.completions.create(
                    model=payload["model"],
                    messages=payload["input"],
                    max_tokens=payload["max_output_tokens"],
                )
                choices = response.choices or []
                text = (choices[0].message.content or "") if choices else ""
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"Empty response for rerank-{i}-chunk-{chunk_index}")
                returned_model = getattr(response, "model", None) or "unknown"
                _write_atomic_json(raw_file, {
                    "custom_id": request["custom_id"],
                    "prompt_sha256": prompt_digest,
                    "requested_model": args.openai_model,
                    "returned_model": returned_model,
                    "output_text": text,
                    "response": response.model_dump() if hasattr(response, "model_dump") else None,
                })
            line_results[chunk_index] = text
            versions[returned_model] += 1

        out_path = write_final_topk_json(
            sample_idx=i, candidate_path=candidate_path, obj=obj,
            line_results=line_results,
            rerank_batch=int(args.rerank_batch),
            topk_core=int(args.topk_core),
            topk_support=int(args.topk_support),
            max_packed_sents=int(args.max_packed_sents),
            out_dir=out_dir,
            openai_model=args.openai_model,
            max_input_candidates=int(args.max_input_candidates),
            query_mode=args.query_mode,
            max_output_tokens=int(args.max_output_tokens),
            allow_partial=False,
            backend="proxy_chat",
        )
        return i, out_path, versions

    succeeded, failed = 0, {}
    model_versions = Counter()
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(worker, i): i for i in pending}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, path, versions = future.result()
                succeeded += 1
                model_versions.update(versions)
                print(f"[OK] rerank-{idx} {dict(versions)} -> {path}", flush=True)
            except Exception as exc:
                failed[idx] = f"{type(exc).__name__}: {exc}"
                print(f"[FAILED] rerank-{idx}: {failed[idx]}", flush=True)

    missing = [i for i in indices if not (out_dir / f"topk_{i}.json").is_file()]
    summary = {
        "requested": len(indices),
        "succeeded_this_run": succeeded,
        "skipped_existing": existing_valid,
        "final_valid": len(indices) - len(missing),
        "final_missing": missing,
        "failed": failed,
        "returned_model_distribution_this_run": dict(model_versions),
        "final_out_dir": str(out_dir),
        "job_dir": str(run_dir),
        "settings": settings,
    }
    _write_atomic_json(run_dir / "run_chat.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failed or missing:
        raise RuntimeError(f"Proxy rerank incomplete: failed={len(failed)}, missing={len(missing)}. Retry same job to resume.")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build_split = sub.add_parser("build_split")
    ap_build_split.add_argument("--candidate_dir", type=str, default=None)
    ap_build_split.add_argument("--out_dir", type=str, required=True)
    ap_build_split.add_argument("--start_idx", type=int, required=True)
    ap_build_split.add_argument("--end_idx", type=int, required=True)
    ap_build_split.add_argument("--chunk_size", type=int, default=50)
    ap_build_split.add_argument("--job_name", type=str, default="rerank_asqa_topk_batch")
    ap_build_split.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_build_split.add_argument("--max_output_tokens", type=int, default=1400)
    ap_build_split.add_argument("--rerank_batch", type=int, default=6)
    ap_build_split.add_argument("--max_input_candidates", type=int, default=18)
    ap_build_split.add_argument("--query_mode", type=str, choices=["structured", "original"], default="structured")
    ap_build_split.set_defaults(func=cmd_build_split)

    ap_submit_split = sub.add_parser("submit_split")
    ap_submit_split.add_argument("--candidate_dir", type=str, default=None)
    ap_submit_split.add_argument("--out_dir", type=str, required=True)
    ap_submit_split.add_argument("--start_idx", type=int, required=True)
    ap_submit_split.add_argument("--end_idx", type=int, required=True)
    ap_submit_split.add_argument("--chunk_size", type=int, default=50)
    ap_submit_split.add_argument("--job_name", type=str, default="rerank_asqa_topk_batch")
    ap_submit_split.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_submit_split.add_argument("--max_output_tokens", type=int, default=1400)
    ap_submit_split.add_argument("--completion_window", type=str, default="24h")
    ap_submit_split.add_argument("--rerank_batch", type=int, default=6)
    ap_submit_split.add_argument("--max_input_candidates", type=int, default=18)
    ap_submit_split.add_argument("--query_mode", type=str, choices=["structured", "original"], default="structured")
    ap_submit_split.add_argument("--tsv", type=str, default="", help="Stable registry path; default: out_dir/JOB_NAME.batches.tsv")
    ap_submit_split.add_argument("--resubmit", action="store_true", help="Explicitly resubmit already registered shards")
    ap_submit_split.set_defaults(func=cmd_submit_split)

    ap_status = sub.add_parser("status")
    ap_status.add_argument("--batch_id", type=str, required=True)
    ap_status.set_defaults(func=cmd_status)

    ap_status_tsv = sub.add_parser("status_from_tsv")
    ap_status_tsv.add_argument("--tsv", type=str, required=True)
    ap_status_tsv.set_defaults(func=cmd_status_from_tsv)

    ap_consume = sub.add_parser("consume")
    ap_consume.add_argument("--batch_id", type=str, required=True)
    ap_consume.add_argument("--candidate_dir", type=str, default=None)
    ap_consume.add_argument("--final_out_dir", type=str, default=None)
    ap_consume.add_argument("--raw_out_jsonl", type=str, default="")
    ap_consume.add_argument("--failed_ids_out", type=str, default="")
    ap_consume.add_argument("--start_idx", type=int, required=True)
    ap_consume.add_argument("--end_idx", type=int, required=True)
    ap_consume.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_consume.add_argument("--rerank_batch", type=int, default=6)
    ap_consume.add_argument("--max_input_candidates", type=int, default=18)
    ap_consume.add_argument("--query_mode", type=str, choices=["structured", "original"], default="structured")
    ap_consume.add_argument("--max_output_tokens", type=int, default=1400)
    ap_consume.add_argument("--allow_partial", action="store_true", help="Allow missing response chunks to default to drop; unsafe for SFT labels")
    ap_consume.add_argument("--topk_core", type=int, default=5)
    ap_consume.add_argument("--topk_support", type=int, default=8)
    ap_consume.add_argument("--max_packed_sents", type=int, default=14)
    ap_consume.set_defaults(func=cmd_consume)

    ap_consume_tsv = sub.add_parser("consume_from_tsv")
    ap_consume_tsv.add_argument("--tsv", type=str, required=True)
    ap_consume_tsv.add_argument("--jobs_dir", type=str, default=None)
    ap_consume_tsv.add_argument("--candidate_dir", type=str, default=None)
    ap_consume_tsv.add_argument("--final_out_dir", type=str, default=None)
    ap_consume_tsv.add_argument("--failed_ids_out", type=str, default="")
    ap_consume_tsv.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_consume_tsv.add_argument("--rerank_batch", type=int, default=6)
    ap_consume_tsv.add_argument("--max_input_candidates", type=int, default=18)
    ap_consume_tsv.add_argument("--query_mode", type=str, choices=["structured", "original"], default="structured")
    ap_consume_tsv.add_argument("--max_output_tokens", type=int, default=1400)
    ap_consume_tsv.add_argument("--allow_partial", action="store_true", help="Allow missing response chunks to default to drop; unsafe for SFT labels")
    ap_consume_tsv.add_argument("--topk_core", type=int, default=5)
    ap_consume_tsv.add_argument("--topk_support", type=int, default=8)
    ap_consume_tsv.add_argument("--max_packed_sents", type=int, default=14)
    ap_consume_tsv.add_argument("--overwrite", action="store_true", help="Overwrite existing topk outputs")
    ap_consume_tsv.set_defaults(func=cmd_consume_from_tsv)

    # Convenience aliases matching the Shell wrapper.
    sub._name_parser_map["status_tsv"] = ap_status_tsv
    sub._name_parser_map["consume_tsv"] = ap_consume_tsv

    ap_consume_local = sub.add_parser("consume_local")
    ap_consume_local.add_argument("--jobs_dir", type=str, required=True)
    ap_consume_local.add_argument("--pattern", type=str, default="batch_*.output.jsonl")
    ap_consume_local.add_argument("--candidate_dir", type=str, default=None)
    ap_consume_local.add_argument("--final_out_dir", type=str, default=None)
    ap_consume_local.add_argument("--failed_ids_out", type=str, required=True)
    ap_consume_local.add_argument("--start_idx", type=int, required=True)
    ap_consume_local.add_argument("--end_idx", type=int, required=True)
    ap_consume_local.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_consume_local.add_argument("--rerank_batch", type=int, default=6)
    ap_consume_local.add_argument("--max_input_candidates", type=int, default=18)
    ap_consume_local.add_argument("--query_mode", type=str, choices=["structured", "original"], default="structured")
    ap_consume_local.add_argument("--max_output_tokens", type=int, default=1400)
    ap_consume_local.add_argument("--allow_partial", action="store_true", help="Allow missing response chunks to default to drop; unsafe for SFT labels")
    ap_consume_local.add_argument("--overwrite", action="store_true", help="Overwrite existing topk files during local re-consumption")
    ap_consume_local.add_argument("--topk_core", type=int, default=5)
    ap_consume_local.add_argument("--topk_support", type=int, default=8)
    ap_consume_local.add_argument("--max_packed_sents", type=int, default=14)
    ap_consume_local.set_defaults(func=cmd_consume_local)

    ap_chat = sub.add_parser("run_chat", help="Proxy concurrent Chat Completions, R3 prompt unchanged")
    ap_chat.add_argument("--candidate_dir", type=str, required=True)
    ap_chat.add_argument("--jobs_dir", type=str, required=True)
    ap_chat.add_argument("--final_out_dir", type=str, required=True)
    ap_chat.add_argument("--start_idx", type=int, required=True)
    ap_chat.add_argument("--end_idx", type=int, required=True)
    ap_chat.add_argument("--job_name", type=str, required=True)
    ap_chat.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_chat.add_argument("--max_output_tokens", type=int, default=2600)
    ap_chat.add_argument("--rerank_batch", type=int, default=18)
    ap_chat.add_argument("--max_input_candidates", type=int, default=18)
    ap_chat.add_argument("--query_mode", choices=["structured", "original"], default="original")
    ap_chat.add_argument("--workers", type=int, default=4)
    ap_chat.add_argument("--topk_core", type=int, default=5)
    ap_chat.add_argument("--topk_support", type=int, default=8)
    ap_chat.add_argument("--max_packed_sents", type=int, default=14)
    ap_chat.add_argument("--overwrite", action="store_true")
    ap_chat.set_defaults(func=cmd_run_chat)

    args = ap.parse_args()
    validate_optional_paths(args)
    if args.cmd in {"submit_split", "status", "status_from_tsv", "status_tsv", "consume", "consume_from_tsv", "consume_tsv"}:
        require_batch()
    configure_paths(args, ("candidate_dir", "out_dir", "final_out_dir", "jobs_dir"))
    args.func(args)


if __name__ == "__main__":
    main()
