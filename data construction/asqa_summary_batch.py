#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import ast
import time
import argparse
import csv
import hashlib
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import api_client, api_configuration, configure_paths, required_path, require_batch, validate_optional_paths



# ============================================================
# Config
# ============================================================
BACKEND = os.environ.get("BACKEND", "openai").strip().lower()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
LOCAL_MODEL_NAME = os.environ.get("MODEL", "")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "600"))
QUERY_MODE_DEFAULT = os.environ.get("QUERY_MODE", "structured").strip().lower()
if QUERY_MODE_DEFAULT not in {"structured", "original"}:
    raise ValueError("QUERY_MODE must be 'structured' or 'original'")

DEFAULT_QUERY_STRUCT_DIR = os.environ.get("QUERY_STRUCT_DIR", "")
DEFAULT_TOPK_DIR = os.environ.get("TOPK_DIR", "")
DEFAULT_JOBS_DIR = os.environ.get("JOBS_DIR", "")
DEFAULT_SUMMARY_DIR = os.environ.get("SUMMARY_OUT_DIR", "")
TSV_FIELDS = [
    "jsonl", "meta", "batch_id", "status", "input_file_id", "jsonl_sha256",
    "query_struct_dir", "topk_dir", "query_mode", "use_plan", "openai_model",
    "max_output_tokens", "max_core", "max_support", "max_packed", "start_idx", "end_idx",
]

CIT_RE = re.compile(r"\[(\d+)\]")
CUST_RE = re.compile(r"^summary-(\d+)$", re.I)


class EmptyFinalSummaryError(ValueError):
    """A valid model response with no seed summary; safe to skip during consume."""
_JSONL_RANGE_RE = re.compile(r"_(\d{4,})_(\d{4,})\.jsonl$", re.I)


# ============================================================
# OpenAI / Batch helpers
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
            "task": "summary_generation",
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
# Proxy transport (Chat Completions); S3 prompt/parser unchanged.
# Proxy does not provide OpenAI native Batch API.
# ============================================================
def _get_proxy_chat_client():
    return api_client("proxy")


def _write_atomic_json_proxy(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _proxy_chat_completion(client, system_prompt, user_prompt, model, max_output_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=int(max_output_tokens),
    )
    choice = (response.choices or [None])[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty Proxy Chat Completions output")
    # Don't accept truncated JSON/plan as a valid teacher response.
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise ValueError("Proxy output truncated (finish_reason=length)")
    return content.strip(), (getattr(response, "model", None) or "unknown"), response


def generate_proxy_chat(system_prompt: str, user_prompt: str, model_name: str, max_output_tokens: int) -> str:
    text, _, _ = _proxy_chat_completion(
        _get_proxy_chat_client(), system_prompt, user_prompt, model_name, max_output_tokens
    )
    return text

# ============================================================
# Summary S3: S1 plus relation coverage and answer-phrase preservation.
# Evidence rendering is unchanged; all cleaned plan items are saved.
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


def _dedup_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        if not isinstance(x, str):
            continue
        xx = _norm_ws(x)
        if not xx:
            continue
        k = xx.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(xx)
    return out


def _safe_list(obj: Any) -> List[Any]:
    return obj if isinstance(obj, list) else []


def _safe_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def extract_query_guidance(qstruct: Dict[str, Any]) -> Dict[str, Any]:
    qstruct = _safe_dict(qstruct)

    anchor_entities = _dedup_keep_order(_safe_list(qstruct.get("anchor_entities")))
    anchor_terms = _dedup_keep_order(_safe_list(qstruct.get("anchor_terms")))
    scope_terms = _dedup_keep_order(_safe_list(qstruct.get("scope_terms")))

    relation_terms = []
    relation_terms.extend(_safe_list(qstruct.get("relation_terms")))
    relation_terms.extend(_safe_list(qstruct.get("relation_synonym")))
    relation_terms.extend(_safe_list(qstruct.get("relation_expansion")))
    relation_terms.extend(_safe_list(qstruct.get("relation_markers")))

    attrs = _safe_list(qstruct.get("attributes"))
    for a in attrs:
        if not isinstance(a, dict):
            continue
        attr = _norm_ws(a.get("attribute", ""))
        if attr:
            relation_terms.append(attr)
        relation_terms.extend(_safe_list(a.get("bridge_triggers")))
        relation_terms.extend(_safe_list(a.get("hypothesis_triggers")))

    rc = _safe_dict(qstruct.get("retrieval_control"))
    relation_terms.extend(_safe_list(rc.get("relation_markers")))
    answerability_markers = _dedup_keep_order(_safe_list(rc.get("answerability_markers")))
    parallel_cues = _dedup_keep_order(_safe_list(rc.get("parallel_cues")))

    relation_terms = _dedup_keep_order(relation_terms)

    alias_map = _safe_dict(qstruct.get("anchor_aliases"))
    alias_lines = []
    for k, vals in alias_map.items():
        vals = _safe_list(vals)
        vals = [v for v in vals if isinstance(v, str) and _norm_ws(v)]
        if vals:
            alias_lines.append(f"- {k}: {vals}")

    retrieval_views = []
    for v in _safe_list(qstruct.get("retrieval_views")):
        if isinstance(v, dict):
            name = _norm_ws(v.get("view_name", ""))
            text = _norm_ws(v.get("text", ""))
            if name and text:
                retrieval_views.append(f"- {name}: {text}")

    return {
        "anchor_entities": anchor_entities,
        "anchor_terms": anchor_terms,
        "scope_terms": scope_terms,
        "relation_terms": relation_terms,
        "answerability_markers": answerability_markers,
        "parallel_cues": parallel_cues,
        "alias_lines": alias_lines,
        "retrieval_views": retrieval_views,
    }


def _extract_sent_text(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    return _norm_ws(
        row.get("sent_content", "")
        or row.get("sentence", "")
        or row.get("text", "")
    )


def load_topk_json(topk_path: Path) -> dict:
    if not topk_path.exists():
        raise FileNotFoundError(f"topk json not found: {topk_path}")

    with topk_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"topk json must be object: {topk_path}")

    # The original query can fall back to query_<idx>.json when topk.query
    # is missing. Do not require a query_struct field in original mode.
    return obj


def build_evidence_lines(topk_obj: Dict[str, Any], max_core: int, max_support: int, max_packed: int) -> List[str]:
    lines = []
    core_rows = _safe_list(topk_obj.get("core_evidence"))[:max_core]
    support_rows = _safe_list(topk_obj.get("support_evidence"))[:max_support]
    packed_rows = _safe_list(topk_obj.get("packed_evidence"))[:max_packed]

    used = set()

    def add_rows(rows: List[Dict[str, Any]], source_name: str):
        nonlocal lines, used
        for r in rows:
            if not isinstance(r, dict):
                continue
            doc_id = r.get("doc_id", -1)
            sent = _extract_sent_text(r)
            if not sent:
                continue
            key = (doc_id, sent.lower(), source_name)
            if key in used:
                continue
            used.add(key)
            lines.append(f"[{source_name}] DOC_ID={doc_id} TEXT={sent}")

    add_rows(core_rows, "core")
    add_rows(support_rows, "support")
    add_rows(packed_rows, "packed")
    return lines


def build_summary_prompt(
    query: str,
    qstruct: Dict[str, Any],
    evidence_lines: List[str],
    query_mode: str = "structured",
    use_plan: bool = True,
):
    if query_mode not in {"structured", "original"}:
        raise ValueError("query_mode must be structured or original")
    if not isinstance(use_plan, bool):
        raise ValueError("use_plan must be a bool")

    # Do not normalize or render structured fields in original-query mode.
    guidance = ""
    if query_mode == "structured":
        qg = extract_query_guidance(qstruct)
        guidance = f"""Query guidance:
Anchor entities:
{qg["anchor_entities"]}

Anchor aliases:
{chr(10).join(qg["alias_lines"]) if qg["alias_lines"] else "[]"}

Anchor terms:
{qg["anchor_terms"]}

Relation terms:
{qg["relation_terms"]}

Scope terms:
{qg["scope_terms"]}

Answerability markers:
{qg["answerability_markers"]}

Parallel cues:
{qg["parallel_cues"]}

Retrieval views:
{chr(10).join(qg["retrieval_views"]) if qg["retrieval_views"] else "[]"}

"""

    # Fictional example, shared by plan/no-plan modes. No evaluation answers
    # or gold subquestions are used when constructing the prompt.
    example_query = "When was the Lark Museum built?"
    example_evidence = (
        "[core] DOC_ID=2 TEXT=Construction on the Lark Museum began in March 2007, "
        "and the museum opened on June 18, 2009."
    )
    example_summary = (
        "Construction on the Lark Museum began in March 2007 [2]. "
        "The museum opened on June 18, 2009 [2]."
    )

    plan_purpose = ""
    plan_requirements = "- Write final_summary directly without an attribution plan or intermediate claim list.\n"
    decision_rule = ""
    output_keys = '"final_summary"'
    example_output = {"final_summary": example_summary}
    if use_plan:
        plan_purpose = "- The purpose of plan is to preserve each distinct supported answer relation and attach it to supporting DOC_IDs.\n"
        plan_requirements = """- plan_style must be exactly one of:
  - "direct": use when the evidence supports one answer relation, including a list sharing that relation
  - "parallel": use when the evidence supports two or more distinct answer relations, interpretations, scopes, or event stages
- plan must be a list of claim objects
- each claim object must contain:
  - "claim": one short factual sentence or clause with its answer phrase and distinguishing qualifiers
  - "citations": a list of supporting DOC_ID strings, such as ["5"] or ["5","1"]
- citations must be non-empty
- include each distinct supported answer relation in plan, even if another claim already cites the same document
- preserve every planned answer relation, answer phrase, and qualifier in final_summary
- keep a coordinated list answering one relation together in one claim; do not create a separate claim for each list member
"""
        decision_rule = """
Decision rule:
- When in doubt between "direct" and "parallel", prefer "parallel" if more than one supported answer branch exists.
"""
        output_keys = '"plan_style", "plan", "final_summary"'
        example_output = {
            "plan_style": "parallel",
            "plan": [
                {
                    "claim": "Construction on the Lark Museum began in March 2007.",
                    "citations": ["2"],
                },
                {
                    "claim": "The museum opened on June 18, 2009.",
                    "citations": ["2"],
                },
            ],
            "final_summary": example_summary,
        }
    output_shape = json.dumps(example_output, ensure_ascii=False, indent=2)

    system = f"""
You are a high-recall seed-summary {'planner' if use_plan else 'writer'} for a retrieval-augmented QA system.

Your job is to compress the evidence into a compact but coverage-preserving backbone for a later expansion stage.

Hard rules:
1. Use ONLY the provided evidence.
2. Do NOT invent facts.
3. Preserve all distinct, answer-relevant, supported branches.
4. Do NOT collapse competing or alternative supported answer branches into one.
5. If two supported branches differ by time, version, location, role, event stage, authority, counting scheme, scope, interpretation, or standard, keep both with their distinguishing qualifiers.
6. This stage is for preserving supported answer structure, not for aggressive pruning.
7. Prefer coverage-preserving compression over overly minimal summaries.
8. Use only DOC_ID citations. Do NOT invent or use sentence ids.
9. Output JSON only.
""".strip()

    user = f"""
Formatting example (fictional; its facts and DOC_ID are NOT evidence for the actual query below):
Example query: {example_query}
Example evidence: {example_evidence}
Example output:
{output_shape}

Use only the actual evidence below when answering the actual query. The example shows how one evidence sentence can support two different answer relations; it does not prescribe the number of claims.

Query:
{query}

{guidance}Evidence:
{chr(10).join(evidence_lines) if evidence_lines else "[NO_EVIDENCE]"}

Task:
Produce a compact seed summary for a later expansion stage.

Primary objective:
- Preserve as many distinct, supported, answer-relevant branches as possible.
- Do not drop a supported branch just because another branch looks more dominant.
- A single document or evidence sentence may support multiple answer relations. Preserve each relevant relation with its distinguishing time, version, location, role, or event-stage qualifier.
- Directly answer each supported reading of the query rather than only describing the document or topic.
- If any relevant answer branch is supported, include it even when other readings remain unresolved. Do not discard a supported partial answer merely because a complete answer is unavailable.

Answer wording and support:
- Keep answer-bearing names, dates, numbers, titles, and coordinated lists in the wording and order used by the supporting evidence, in both final_summary and any plan claims.
- Do not shorten a full date, drop a name component, round a number, abbreviate or paraphrase an answer phrase, reorder a list, or split a single list answer across sentences. Do not insert extra words inside an answer phrase.
- Keep the answer phrase attached to the relationship it supports. Preserve time, location, version, and uncertainty qualifiers; date historical rankings, roles, and statistics as given in the evidence.
- Ground the relationship as well as the answer phrase: do not infer an actor-role link from a cast list, an episode number from a season number, or a release date for one location from another location.

Important:
- Evidence lines include only:
  - source bucket: core / support / packed
  - DOC_ID
  - sentence text
- Ignore sentence-level indexing completely.
- A citation refers to DOC_ID only.
{plan_purpose}
Output requirements:
{plan_requirements}- final_summary must preserve all supported answer branches that are relevant to the query
- every sentence in final_summary must end with DOC_ID citation brackets
- use citation format like [5] or [5][1]
- do not invent citation ids
- each sentence must be supported by the evidence it cites
- do not collapse two different supported standards into one sentence
- if two answer branches are both supported, include both
- if one sentence gives the main answer and another gives an alternative standard / authority / interpretation, keep both
{decision_rule}
Formatting requirements:
- Return STRICT JSON only
- Top-level keys must be exactly:
  {output_keys}
- No markdown
- No comments
- No prose outside JSON

Return a JSON object using the required keys and the format illustrated above, with content grounded only in the actual evidence for the actual query.
""".strip()

    return system, user


def parse_summary_json(raw_text: str, use_plan: bool = True) -> Dict[str, Any]:
    if not isinstance(use_plan, bool):
        raise ValueError("use_plan must be a bool")
    raw_text = strip_qwen_thinking(raw_text).strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        obj = json.loads(raw_text)
    except Exception:
        if use_plan:
            m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if not m:
                raise ValueError(f"Invalid summary JSON:\n{raw_text}")
            block = m.group(0)
        else:
            # No-plan mode must not silently discard an explicit plan before JSON.
            block = raw_text
        block = re.sub(r",\s*([}\]])", r"\1", block)
        obj = json.loads(block)

    if not isinstance(obj, dict):
        raise ValueError("summary output is not a JSON object")

    if not use_plan:
        # Do not hide a generated plan by deleting it after inference.
        if set(obj) != {"final_summary"}:
            raise ValueError("USE_PLAN=0 requires only final_summary; unexpected fields: "
                             + ", ".join(sorted(set(obj) - {"final_summary"})))
        if not isinstance(obj["final_summary"], str) or not obj["final_summary"].strip():
            raise EmptyFinalSummaryError("final_summary is empty")
        return {
            "plan_style": "none", "plan": [],
            "final_summary": _norm_ws(obj["final_summary"]),
        }

    plan_style = _norm_ws(obj.get("plan_style", ""))
    final_summary = _norm_ws(obj.get("final_summary", ""))

    if plan_style not in {"direct", "parallel"}:
        plan_style = "direct"

    raw_plan = obj.get("plan", [])
    if not isinstance(raw_plan, list):
        raw_plan = []

    clean_plan = []
    for item in raw_plan:
        if not isinstance(item, dict):
            continue

        claim = _norm_ws(item.get("claim", ""))
        citations = item.get("citations", [])

        if not claim:
            continue
        if not isinstance(citations, list):
            citations = []

        clean_citations = []
        seen = set()
        for c in citations:
            cc = str(c).strip()
            if not cc:
                continue
            if cc in seen:
                continue
            seen.add(cc)
            clean_citations.append(cc)

        if not clean_citations:
            continue

        clean_plan.append({
            "claim": claim,
            "citations": clean_citations,
        })

    if not final_summary:
        raise EmptyFinalSummaryError("final_summary is empty")

    return {
        "plan_style": plan_style,
        # Post-generation truncation silently desynchronizes plan and summary.
        "plan": clean_plan,
        "final_summary": final_summary,
    }


# ============================================================
# Local backend helper (kept for compatibility)
# ============================================================
def load_local_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def generate_local_chat(
    system_prompt: str,
    user_prompt: str,
    tokenizer,
    model,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    import torch
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    if not text:
        raise ValueError("Empty local model output.")
    return text


# ============================================================
# File loading / path helpers
# ============================================================
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_query_struct_path(sample_idx: int, query_struct_dir: Optional[Path]):
    return query_struct_dir / f"query_{sample_idx}.json" if query_struct_dir is not None else ""


def resolve_topk_path(sample_idx: int, topk_dir: Path) -> Path:
    return topk_dir / f"topk_{sample_idx}.json"


def resolve_local_structure_path(sample_idx: int, local_structure_dir: Optional[Path]) -> Optional[Path]:
    if local_structure_dir is None:
        return None
    return local_structure_dir / f"local_structure_{sample_idx}.json"


def resolve_summary_out_path(sample_idx: int, summary_out_dir: Path) -> Path:
    return summary_out_dir / f"summary_{sample_idx}.json"


def load_current_context(
    query_struct_path: str,
    topk_path: str,
    local_structure_path: str = "",
    query_mode: str = "structured",
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load context with the same query isolation as summary_range.py.

    original: use topk.query; fall back to query_<idx>.json's plain query
    only when necessary. Never load or inspect query_struct, including a
    query_struct embedded in a topk result.
    structured: preserve the existing query_<idx>.json requirement.
    local_structure_path is retained only for CLI compatibility.
    """
    if query_mode not in {"structured", "original"}:
        raise ValueError("query_mode must be structured or original")

    topk_obj = load_topk_json(Path(topk_path).expanduser())
    raw_query = topk_obj.get("query", "")
    query = _norm_ws(raw_query) if isinstance(raw_query, str) else ""
    query_struct: Dict[str, Any] = {}

    # Original mode never even opens the query structure file when topk.query
    # exists; this prevents inadvertent structured-guidance leakage.
    if query_mode == "structured" or not query:
        if not query_struct_path:
            raise ValueError("query file is required in structured mode or when topk.query is absent")
        qstruct_obj = _read_json(str(Path(query_struct_path).expanduser()))
        if not isinstance(qstruct_obj, dict):
            raise ValueError(f"query json must be an object: {query_struct_path}")
        if not query:
            fallback = qstruct_obj.get("query", "")
            query = _norm_ws(fallback) if isinstance(fallback, str) else ""
        if query_mode == "structured":
            query_struct = _safe_dict(qstruct_obj.get("query_struct", {}))
            if not query_struct:
                raise ValueError("query_struct missing or invalid")

    if not query:
        raise ValueError("query missing in both topk and query json")
    return query, query_struct, topk_obj, None


def iter_sample_indices(start_idx: int, end_idx: int) -> List[int]:
    if end_idx < 0:
        raise ValueError("END_IDX must be >= 0 for deterministic batch build.")
    if end_idx <= start_idx:
        raise ValueError(f"Require END_IDX > START_IDX, got start={start_idx}, end={end_idx}")
    return list(range(start_idx, end_idx))


def make_custom_id(sample_idx: int) -> str:
    return f"summary-{sample_idx}"


def parse_custom_id(custom_id: str) -> int:
    m = CUST_RE.match(custom_id or "")
    if not m:
        raise ValueError(f"Bad custom_id: {custom_id}")
    return int(m.group(1))


def _infer_range_from_jsonl_path(jsonl_path: str) -> Tuple[int, int]:
    name = Path(jsonl_path).name
    m = _JSONL_RANGE_RE.search(name)
    if not m:
        raise ValueError(f"cannot infer start/end idx from jsonl filename: {jsonl_path}")
    return int(m.group(1)), int(m.group(2))


def _read_batches_tsv(tsv_path: Path) -> List[Dict[str, str]]:
    if not tsv_path.exists():
        raise FileNotFoundError(f"tsv not found: {tsv_path}")

    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty tsv: {tsv_path}")

    header = lines[0].strip().split("\t")
    rows: List[Dict[str, str]] = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        row = dict(zip(header, parts))
        rows.append(row)

    return rows


# ============================================================
# Build batch request from summary task
# ============================================================
def build_one_request(
    sample_idx: int,
    query: str,
    query_struct: Dict[str, Any],
    topk_obj: Dict[str, Any],
    local_obj: Optional[Dict[str, Any]],
    openai_model: str,
    max_output_tokens: int,
    query_mode: str = "structured",
    use_plan: bool = True,
    max_core: int = 5,
    max_support: int = 5,
    max_packed: int = 12,
) -> Dict[str, Any]:
    evidence_lines = build_evidence_lines(
        topk_obj=topk_obj,
        max_core=max_core,
        max_support=max_support,
        max_packed=max_packed,
    )
    system_prompt, user_prompt = build_summary_prompt(
        query=query,
        qstruct=query_struct,
        evidence_lines=evidence_lines,
        query_mode=query_mode,
        use_plan=bool(use_plan),
    )
    return {
        "custom_id": make_custom_id(sample_idx),
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


# ============================================================
# Submit / Status
# ============================================================
def cmd_build_split(args):
    query_struct_dir = Path(args.query_struct_dir).expanduser() if args.query_struct_dir else None
    topk_dir = Path(args.topk_dir).expanduser()
    local_structure_dir = Path(args.local_structure_dir).expanduser() if args.local_structure_dir else None
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    chunk_size = int(args.chunk_size)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

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
                query_struct_path = resolve_query_struct_path(sample_idx, query_struct_dir)
                topk_path = resolve_topk_path(sample_idx, topk_dir)
                local_structure_path = resolve_local_structure_path(sample_idx, local_structure_dir)

                if args.query_mode == "structured" and not query_struct_path.exists():
                    print(f"[WARN] missing query_struct: {query_struct_path}")
                    missing += 1
                    continue

                if not topk_path.exists():
                    print(f"[WARN] missing topk json: {topk_path}")
                    missing += 1
                    continue

                local_path_str = ""
                if local_structure_path and local_structure_path.exists():
                    local_path_str = str(local_structure_path)

                try:
                    query, query_struct, topk_obj, local_obj = load_current_context(
                        query_struct_path=str(query_struct_path),
                        topk_path=str(topk_path),
                        local_structure_path=local_path_str,
                        query_mode=args.query_mode,
                    )
                except Exception as e:
                    print(f"[WARN] skip invalid context for sample_idx={sample_idx}: {e}")
                    missing += 1
                    continue

                # Inference permits empty evidence and renders [NO_EVIDENCE].
                # Do not silently diverge by dropping this sample at build time.
                try:
                    req = build_one_request(
                        sample_idx=sample_idx,
                        query=query,
                        query_struct=query_struct,
                        topk_obj=topk_obj,
                        local_obj=local_obj,
                        openai_model=args.openai_model,
                        max_output_tokens=int(args.max_output_tokens),
                        query_mode=args.query_mode,
                        use_plan=bool(args.use_plan),
                        max_core=args.max_core,
                        max_support=args.max_support,
                        max_packed=args.max_packed,
                    )
                except Exception as e:
                    print(f"[WARN] skip request build failure for sample_idx={sample_idx}: {e}")
                    missing += 1
                    continue

                fout.write(json.dumps(req, ensure_ascii=False) + "\n")
                written += 1
                covered_samples += 1

        if written > 0:
            _ensure_nonempty_file(out_jsonl, "summary split jsonl")
            files.append(str(out_jsonl))
        else:
            if out_jsonl.exists():
                out_jsonl.unlink()

        total_requests += written
        total_samples += covered_samples
        total_missing += missing

    if total_requests == 0:
        raise RuntimeError(
            f"build_split wrote 0 requests. Check dirs/start/end. "
            f"query_struct_dir={query_struct_dir} topk_dir={topk_dir} start={start_idx} end={end_idx}"
        )

    payload = {
        "query_struct_dir": str(query_struct_dir) if query_struct_dir is not None else "",
        "topk_dir": str(topk_dir),
        "local_structure_dir": str(local_structure_dir) if local_structure_dir else "",
        "out_dir": str(out_dir),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "chunk_size": chunk_size,
        "num_files": len(files),
        "num_samples_covered": total_samples,
        "num_missing_inputs": total_missing,
        "num_requests": total_requests,
        "files": files,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or not {"jsonl", "batch_id"}.issubset(reader.fieldnames):
            raise ValueError(f"Malformed TSV header: {path}")
        return list(reader)


def _save_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _resolve_manifest(path_str: str) -> Path:
    path = Path(path_str).expanduser() if path_str else required_path(DEFAULT_JOBS_DIR, "JOBS_DIR or --tsv") / "latest_batches.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Batch TSV not found: {path}; submit_split first or set TSV_PATH")
    return path


def cmd_submit_split(args):
    """Build, safely submit, and record all Batch IDs in a stable TSV.

    Existing batches are only reused after verifying the original request bytes
    and configuration. Legacy metadata without checksums is never overwritten.
    """
    api_configuration("openai")
    if args.completion_window != "24h":
        raise ValueError("Batch completion_window must be 24h")
    if int(args.chunk_size) <= 0:
        raise ValueError("chunk_size must be > 0")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.job_name):
        raise ValueError("Unsafe JOB_NAME; use letters, digits, _, - or .")

    jobs = Path(args.out_dir).expanduser().resolve()
    jobs.mkdir(parents=True, exist_ok=True)
    qdir = Path(args.query_struct_dir).expanduser().resolve() if args.query_struct_dir else None
    topkdir = Path(args.topk_dir).expanduser().resolve()
    start, end, chunk = int(args.start_idx), int(args.end_idx), int(args.chunk_size)
    expected = end - start
    if start < 0 or expected <= 0:
        raise ValueError("Require 0 <= START_IDX < END_IDX (end exclusive)")
    manifest = jobs / f"{args.job_name}.batches.tsv"
    old_rows = {str(Path(r["jsonl"]).resolve()): r for r in _manifest_rows(manifest)}
    options = {
        "query_struct_dir": str(qdir) if qdir is not None else "", "topk_dir": str(topkdir),
        "query_mode": args.query_mode, "use_plan": str(int(args.use_plan)),
        "openai_model": args.openai_model, "max_output_tokens": str(args.max_output_tokens),
        "max_core": str(args.max_core), "max_support": str(args.max_support),
        "max_packed": str(args.max_packed),
    }

    # Build in staging so an existing, already submitted JSONL is NEVER rewritten.
    with tempfile.TemporaryDirectory(prefix=f".{args.job_name}.staging.", dir=jobs) as tmpdir:
        staging = Path(tmpdir)
        cmd_build_split(SimpleNamespace(**{**vars(args), "out_dir": str(staging)}))
        parts = []
        actual_total = 0
        for first in range(start, end, chunk):
            last = min(first + chunk, end)
            filename = f"{args.job_name}_{first:04d}_{last:04d}.jsonl"
            staged = staging / filename
            if not staged.is_file() or staged.stat().st_size == 0:
                raise RuntimeError(f"Empty/missing split: {filename}")
            count = sum(bool(line.strip()) for line in staged.read_text(encoding="utf-8").splitlines())
            if count != last - first:
                raise RuntimeError(f"Incomplete input: {filename} has {count}/{last-first} requests; no API submission")
            actual_total += count
            dest = jobs / filename
            meta_path = dest.with_suffix(".meta.json")
            row_old = old_rows.get(str(dest), {})
            old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else row_old
            fingerprint = _file_sha256(staged)
            if old.get("batch_id"):
                if not old.get("jsonl_sha256"):
                    raise RuntimeError(f"Legacy submitted metadata without checksum: {meta_path}. "
                                       "Recover the existing Batch ID; use a NEW JOB_NAME for new submissions.")
                if old["jsonl_sha256"] != fingerprint:
                    raise RuntimeError(f"Previously submitted requests differ for {filename}; use NEW JOB_NAME")
                for key, val in options.items():
                    if str(old.get(key, "")) != val:
                        raise RuntimeError(f"Already submitted {filename} with different {key}; use NEW JOB_NAME")
                if not dest.exists() or _file_sha256(dest) != fingerprint:
                    raise RuntimeError(f"Submitted JSONL missing/changed: {dest}")
            elif meta_path.exists() or row_old.get("batch_id"):
                raise RuntimeError(f"Existing ambiguous submission record for {filename}; refusing duplicate submission")
            parts.append((staged, dest, meta_path, first, last, fingerprint, old))
        if actual_total != expected:
            raise RuntimeError(f"Expected {expected} summary requests but built {actual_total}")

        completed_rows = []
        for staged, dest, meta_path, first, last, fingerprint, old in parts:
            if old.get("batch_id"):
                info = dict(old)
                print(f"[REUSE] {dest.name}: {info['batch_id']} (no new API request)")
            else:
                # Keep a permanent exact copy of the bytes being submitted.
                shutil.copy2(staged, dest)
                info = _submit_one_jsonl(dest, args.openai_model, args.completion_window, dest.stem)
                print(f"[SUBMITTED] {dest.name}: {info['batch_id']} ({info['status']})")
            info.update(options)
            info.update({"jsonl": str(dest), "meta": str(meta_path), "jsonl_sha256": fingerprint,
                         "start_idx": first, "end_idx": last})
            meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            completed_rows.append({k: info.get(k, "") for k in TSV_FIELDS})
            # Persist after every successful submission, not only at loop end.
            _save_manifest(manifest, completed_rows)
            _save_manifest(jobs / "latest_batches.tsv", completed_rows)

    print(f"[BATCH_TSV] {manifest}")
    print(f"[LATEST_TSV] {jobs / 'latest_batches.tsv'}")
    print(f"[SUMMARY] batches={len(completed_rows)} new={sum(not p[-1].get('batch_id') for p in parts)} reused={sum(bool(p[-1].get('batch_id')) for p in parts)}")


def cmd_status_tsv(args):
    args.tsv = str(_resolve_manifest(args.tsv))
    return cmd_status_from_tsv(args)


def cmd_consume_tsv(args):
    """Consume completed batches, using exactly the submitted request IDs/settings."""
    api_configuration("openai")
    tsv = _resolve_manifest(args.tsv)
    rows = _manifest_rows(tsv)
    if not rows:
        raise ValueError(f"Empty manifest: {tsv}")
    client = _get_client()
    jobs = Path(args.jobs_dir).expanduser()
    jobs.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.summary_out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    qdir = Path(args.query_struct_dir).expanduser().resolve() if args.query_struct_dir else None
    topkdir = Path(args.topk_dir).expanduser().resolve()
    counts = {"consumed": 0, "pending": 0, "errors": 0, "written": 0,
              "skipped_existing": 0, "skipped_empty": 0}
    failed_rows = []
    empty_ids = []

    for row in rows:
        bid = row.get("batch_id", "")
        jp = Path(row.get("jsonl", "")).expanduser()
        try:
            if not bid or not jp.is_file():
                raise ValueError(f"Missing Batch ID or request JSONL: {bid} / {jp}")
            if row.get("jsonl_sha256") and _file_sha256(jp) != row["jsonl_sha256"]:
                raise ValueError(f"Submitted request JSONL changed: {jp}")
            # Query structure is not an input in original mode.
            checked_dirs = (("topk_dir", topkdir),)
            if args.query_mode == "structured":
                checked_dirs += (("query_struct_dir", qdir),)
            for name, actual in checked_dirs:
                if row.get(name) and Path(row[name]).expanduser().resolve() != actual:
                    raise ValueError(f"TSV {name} differs from current {actual}")
            if row.get("use_plan") and row["use_plan"] != str(int(args.use_plan)):
                raise ValueError("USE_PLAN differs from submitted requests; pass matching USE_PLAN")
            if row.get("query_mode") and row["query_mode"] != args.query_mode:
                raise ValueError("QUERY_MODE differs from submitted requests")
            batch = client.batches.retrieve(bid)
            if batch.status != "completed":
                counts["pending"] += 1
                print(f"[SKIP] {bid}: {batch.status}")
                continue
            ofid = getattr(batch, "output_file_id", None)
            if not ofid:
                raise ValueError(f"Completed Batch has no output_file_id: {bid}")
            requests = [json.loads(line) for line in jp.read_text(encoding="utf-8").splitlines() if line.strip()]
            expected_ids = {parse_custom_id(req["custom_id"]) for req in requests}
            if len(expected_ids) != len(requests):
                raise ValueError(f"Duplicate request custom_id in {jp}")
            raw = client.files.content(ofid).text
            raw_path = jobs / f"{jp.stem}.{bid}.output.jsonl"
            raw_path.write_text(raw, encoding="utf-8")
            seen = {}
            for line in raw.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                idx = parse_custom_id(obj.get("custom_id", ""))
                if idx not in expected_ids or idx in seen:
                    raise ValueError(f"Unexpected or duplicated summary-{idx} in {bid}")
                if obj.get("error") or (obj.get("response") or {}).get("status_code", 200) >= 400:
                    raise ValueError(f"API error at summary-{idx}: {obj.get('error')}")
                seen[idx] = extract_output_text_from_batch_line(obj)
            missing = sorted(expected_ids - set(seen))
            if missing:
                raise ValueError(f"Missing {len(missing)} output IDs in {bid}: {missing[:15]}")

            # Validate all outputs first. Empty final_summary is an intentional skip,
            # not a failure of the other valid samples in the same Batch.
            decoded = {}
            empty_this_batch = []
            for idx, raw_text in seen.items():
                qpath = resolve_query_struct_path(idx, qdir)
                tpath = resolve_topk_path(idx, topkdir)
                if not tpath.exists():
                    raise FileNotFoundError(f"Missing source topk for summary-{idx}: {tpath}")
                if args.query_mode == "structured" and not qpath.exists():
                    raise FileNotFoundError(f"Missing source query structure for summary-{idx}: {qpath}")
                try:
                    decoded[idx] = process_one_summary_output(idx, qpath, tpath, None, raw_text,
                                                               use_plan=bool(args.use_plan))
                except EmptyFinalSummaryError:
                    empty_this_batch.append(idx)
                    print(f"[SKIP-EMPTY] summary-{idx}: final_summary is empty")
            for idx, obj in sorted(decoded.items()):
                out = resolve_summary_out_path(idx, out_dir)
                if out.is_file() and not args.overwrite:
                    counts["skipped_existing"] += 1
                    continue
                out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                counts["written"] += 1
            counts["consumed"] += 1
            counts["skipped_empty"] += len(empty_this_batch)
            empty_ids.extend(empty_this_batch)
            print(f"[CONSUMED] {bid}: {len(decoded)} valid, {len(empty_this_batch)} empty skipped")
        except Exception as exc:
            counts["errors"] += 1
            failed_rows.append(f"{bid}\t{exc}")
            print(f"[FAILED] {bid}: {exc}")
    failed_path = jobs / f"{tsv.stem}.consume.failed_ids.txt"
    failed_path.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
    empty_path = jobs / f"{tsv.stem}.consume.skipped_empty_ids.txt"
    empty_path.write_text("\n".join(f"summary-{i}" for i in sorted(set(empty_ids))) +
                          ("\n" if empty_ids else ""), encoding="utf-8")
    print(json.dumps({"tsv": str(tsv), **counts, "summary_out_dir": str(out_dir),
                      "failed_ids_out": str(failed_path),
                      "skipped_empty_ids_out": str(empty_path)}, ensure_ascii=False, indent=2))
    if counts["errors"]:
        raise RuntimeError(f"{counts['errors']} Batch(es) failed to consume; check {failed_path}")



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
    tsv_path = Path(args.tsv).expanduser()
    rows = _read_batches_tsv(tsv_path)

    results = []
    for row in rows:
        batch_id = row.get("batch_id", "").strip()
        if not batch_id:
            continue

        try:
            batch = client.batches.retrieve(batch_id)

            rc = getattr(batch, "request_counts", None)
            if rc is not None:
                try:
                    rc_total = rc.total
                    rc_completed = rc.completed
                    rc_failed = rc.failed
                except Exception:
                    rc_total = rc_completed = rc_failed = None
            else:
                rc_total = rc_completed = rc_failed = None

            results.append({
                "jsonl": row.get("jsonl", ""),
                "batch_id": batch.id,
                "status": batch.status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "total": rc_total,
                "completed": rc_completed,
                "failed": rc_failed,
            })
        except Exception as e:
            results.append({
                "jsonl": row.get("jsonl", ""),
                "batch_id": batch_id,
                "status": "ERROR",
                "error": str(e),
            })

    print(json.dumps({
        "tsv": str(tsv_path),
        "num_rows": len(rows),
        "results": results,
    }, ensure_ascii=False, indent=2))


# ============================================================
# Consume and write final summary_{idx}.json
# ============================================================
def _debug_dump_parse_failure(
    sample_idx: int,
    raw_text: str,
    parsed: Dict[str, Any],
    user_prompt: str,
):
    print(f"===== RAW MODEL OUTPUT (sample_idx={sample_idx}) =====")
    print(raw_text)
    print("=====================================================")
    print(f"===== PARSED MODEL JSON (sample_idx={sample_idx}) =====")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    print("======================================================")
    print(f"===== USER PROMPT (sample_idx={sample_idx}) =====")
    print(user_prompt)
    print("===============================================")


def process_one_summary_output(
    sample_idx: int,
    query_struct_path: Path,
    topk_path: Path,
    local_structure_path: Optional[Path],
    raw_text: str,
    use_plan: bool = True,
) -> Dict[str, Any]:
    # Both inference and Batch use the same parser, never inventing missing claims.
    parsed = parse_summary_json(raw_text, use_plan=bool(use_plan))
    return {
        "plan_style": parsed["plan_style"],
        "plan": parsed["plan"],
        "final_summary": parsed["final_summary"],
    }


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

    query_struct_dir = Path(args.query_struct_dir).expanduser() if args.query_struct_dir else None
    topk_dir = Path(args.topk_dir).expanduser()
    local_structure_dir = Path(args.local_structure_dir).expanduser() if args.local_structure_dir else None
    summary_out_dir = Path(args.summary_out_dir).expanduser()
    summary_out_dir.mkdir(parents=True, exist_ok=True)

    failed_out: Optional[Path] = Path(args.failed_ids_out).expanduser() if args.failed_ids_out else None
    if failed_out:
        failed_out.parent.mkdir(parents=True, exist_ok=True)

    start_i = int(args.start_idx)
    end_i = int(args.end_idx)

    text = client.files.content(ofid).text

    if raw_out_path:
        raw_out_path.write_text(text, encoding="utf-8")
        print(f"[saved raw batch output] {raw_out_path}")

    grouped: Dict[int, str] = {}
    failed = 0
    failed_rows: List[str] = []

    for line in text.splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj.get("custom_id", "")

        try:
            sample_idx = parse_custom_id(custom_id)

            if sample_idx < start_i:
                continue
            if end_i >= 0 and sample_idx >= end_i:
                continue

            raw_text = extract_output_text_from_batch_line(obj)
            grouped[sample_idx] = raw_text

        except Exception as e:
            failed += 1
            failed_rows.append(f"{custom_id}\t{e}")
            print(f"[FAILED-PARSE] {custom_id}: {e}")

    ok = 0
    missing_input = 0
    skipped_empty = 0

    target_indices = iter_sample_indices(start_i, end_i)

    for sample_idx in target_indices:
        try:
            query_struct_path = resolve_query_struct_path(sample_idx, query_struct_dir)
            topk_path = resolve_topk_path(sample_idx, topk_dir)
            local_structure_path = resolve_local_structure_path(sample_idx, local_structure_dir)
            out_path = resolve_summary_out_path(sample_idx, summary_out_dir)

            if args.query_mode == "structured" and not query_struct_path.exists():
                missing_input += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing query_struct: {query_struct_path}")
                print(f"[WARN] missing query_struct: {query_struct_path}")
                continue

            if not topk_path.exists():
                missing_input += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing topk json: {topk_path}")
                print(f"[WARN] missing topk json: {topk_path}")
                continue

            raw_text = grouped.get(sample_idx, "")
            if not raw_text:
                failed += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing batch model output")
                print(f"[FAILED-WRITE] sample-{sample_idx}: missing batch model output")
                continue

            out = process_one_summary_output(
                sample_idx=sample_idx,
                query_struct_path=query_struct_path,
                topk_path=topk_path,
                local_structure_path=local_structure_path,
                raw_text=raw_text,
                use_plan=bool(args.use_plan),
            )

            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] wrote {out_path}")
            ok += 1

        except EmptyFinalSummaryError:
            skipped_empty += 1
            print(f"[SKIP-EMPTY] summary-{sample_idx}: final_summary is empty")
        except Exception as e:
            failed += 1
            failed_rows.append(f"sample-{sample_idx}\t{e}")
            print(f"[FAILED-WRITE] sample-{sample_idx}: {e}")

    summary = {
        "batch_id": batch.id,
        "status": batch.status,
        "ok": ok,
        "failed": failed,
        "skipped_empty": skipped_empty,
        "missing_input": missing_input,
        "summary_out_dir": str(summary_out_dir),
        "start_idx": start_i,
        "end_idx": end_i,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failed_out:
        failed_out.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
        print(f"[saved failed ids] {failed_out}")


def cmd_consume_from_tsv(args):
    api_configuration("openai")

    client = _get_client()
    tsv_path = Path(args.tsv).expanduser()
    rows = _read_batches_tsv(tsv_path)

    query_struct_dir = Path(args.query_struct_dir).expanduser() if args.query_struct_dir else None
    topk_dir = Path(args.topk_dir).expanduser()
    local_structure_dir = Path(args.local_structure_dir).expanduser() if args.local_structure_dir else None
    summary_out_dir = Path(args.summary_out_dir).expanduser()
    summary_out_dir.mkdir(parents=True, exist_ok=True)

    jobs_dir = Path(args.jobs_dir).expanduser()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    consumed = 0
    skipped_not_completed = 0
    failed = 0
    skipped_empty = 0
    failed_rows: List[str] = []

    for row in rows:
        batch_id = row.get("batch_id", "").strip()
        jsonl_path = row.get("jsonl", "").strip()

        if not batch_id:
            failed += 1
            failed_rows.append(f"missing_batch_id\t{jsonl_path}")
            continue

        try:
            start_idx, end_idx = _infer_range_from_jsonl_path(jsonl_path)
        except Exception as e:
            failed += 1
            failed_rows.append(f"{batch_id}\tbad_jsonl_range\t{jsonl_path}\t{e}")
            continue

        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            if status != "completed":
                skipped_not_completed += 1
                print(f"[SKIP] {batch_id} status={status} jsonl={jsonl_path}")
                continue

            ofid = getattr(batch, "output_file_id", None)
            if not ofid:
                failed += 1
                failed_rows.append(f"{batch_id}\tmissing_output_file_id")
                continue

            raw_text = client.files.content(ofid).text

            job_stem = Path(jsonl_path).stem
            raw_out_jsonl = jobs_dir / f"{job_stem}.output.jsonl"
            failed_ids_out = jobs_dir / f"{job_stem}.failed_ids.txt"

            raw_out_jsonl.write_text(raw_text, encoding="utf-8")
            print(f"[saved raw batch output] {raw_out_jsonl}")

            grouped: Dict[int, str] = {}
            parse_failed = 0

            for line in raw_text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                custom_id = obj.get("custom_id", "")
                try:
                    sample_idx = parse_custom_id(custom_id)
                    if sample_idx < start_idx or sample_idx >= end_idx:
                        continue
                    raw_model_text = extract_output_text_from_batch_line(obj)
                    grouped[sample_idx] = raw_model_text
                except Exception as e:
                    parse_failed += 1
                    failed_rows.append(f"{batch_id}\t{custom_id}\tparse_error\t{e}")

            ok_this_batch = 0
            empty_this_batch = 0
            batch_fail_rows: List[str] = []

            for sample_idx in range(start_idx, end_idx):
                try:
                    query_struct_path = resolve_query_struct_path(sample_idx, query_struct_dir)
                    topk_path = resolve_topk_path(sample_idx, topk_dir)
                    local_structure_path = resolve_local_structure_path(sample_idx, local_structure_dir)
                    out_path = resolve_summary_out_path(sample_idx, summary_out_dir)

                    if args.query_mode == "structured" and not query_struct_path.exists():
                        batch_fail_rows.append(f"{batch_id}\tsample-{sample_idx}\tmissing_query_struct\t{query_struct_path}")
                        continue

                    if not topk_path.exists():
                        batch_fail_rows.append(f"{batch_id}\tsample-{sample_idx}\tmissing_topk_json\t{topk_path}")
                        continue

                    raw_model_text = grouped.get(sample_idx, "")
                    if not raw_model_text:
                        batch_fail_rows.append(f"{batch_id}\tsample-{sample_idx}\tmissing_batch_model_output")
                        continue

                    out = process_one_summary_output(
                        sample_idx=sample_idx,
                        query_struct_path=query_struct_path,
                        topk_path=topk_path,
                        local_structure_path=local_structure_path,
                        raw_text=raw_model_text,
                        use_plan=bool(args.use_plan),
                    )
                    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[OK] wrote {out_path}")
                    ok_this_batch += 1

                except EmptyFinalSummaryError:
                    empty_this_batch += 1
                    print(f"[SKIP-EMPTY] summary-{sample_idx}: final_summary is empty")
                except Exception as e:
                    batch_fail_rows.append(f"{batch_id}\tsample-{sample_idx}\twrite_error\t{e}")

            failed_ids_out.write_text(
                "\n".join(batch_fail_rows) + ("\n" if batch_fail_rows else ""),
                encoding="utf-8"
            )

            failed_rows.extend(batch_fail_rows)
            skipped_empty += empty_this_batch

            print(json.dumps({
                "batch_id": batch_id,
                "status": status,
                "jsonl": jsonl_path,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "ok_written": ok_this_batch,
                "skipped_empty": empty_this_batch,
                "parse_failed": parse_failed,
                "raw_out_jsonl": str(raw_out_jsonl),
                "failed_ids_out": str(failed_ids_out),
            }, ensure_ascii=False, indent=2))

            consumed += 1

        except Exception as e:
            failed += 1
            failed_rows.append(f"{batch_id}\tconsume_error\t{e}")
            print(f"[FAILED] batch_id={batch_id}: {e}")

    if args.failed_ids_out:
        final_failed = Path(args.failed_ids_out).expanduser()
        final_failed.parent.mkdir(parents=True, exist_ok=True)
        final_failed.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")
        print(f"[saved failed ids] {final_failed}")

    print(json.dumps({
        "tsv": str(tsv_path),
        "consumed": consumed,
        "skipped_not_completed": skipped_not_completed,
        "skipped_empty": skipped_empty,
        "failed": failed,
        "query_struct_dir": str(query_struct_dir) if query_struct_dir is not None else "",
        "topk_dir": str(topk_dir),
        "summary_out_dir": str(summary_out_dir),
    }, ensure_ascii=False, indent=2))


def cmd_consume_local(args):
    import glob

    query_struct_dir = Path(args.query_struct_dir).expanduser() if args.query_struct_dir else None
    topk_dir = Path(args.topk_dir).expanduser()
    local_structure_dir = Path(args.local_structure_dir).expanduser() if args.local_structure_dir else None
    jobs_dir = Path(args.jobs_dir).expanduser()
    pattern = args.pattern
    summary_out_dir = Path(args.summary_out_dir).expanduser()
    failed_out = Path(args.failed_ids_out).expanduser()

    start_i = int(args.start_idx)
    end_i = int(args.end_idx)

    paths = [Path(p) for p in glob.glob(str(jobs_dir / pattern))]
    if not paths:
        raise FileNotFoundError(f"no files match: {jobs_dir / pattern}")

    summary_out_dir.mkdir(parents=True, exist_ok=True)
    failed_out.parent.mkdir(parents=True, exist_ok=True)

    grouped: Dict[int, str] = {}
    failed_rows: List[str] = []
    parse_failed = 0

    for fp in sorted(paths):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")

            try:
                sample_idx = parse_custom_id(cid)
                if sample_idx < start_i:
                    continue
                if end_i >= 0 and sample_idx >= end_i:
                    continue
                raw = extract_output_text_from_batch_line(obj)
                grouped[sample_idx] = raw
            except Exception as e:
                parse_failed += 1
                failed_rows.append(f"{cid}\t{e}")

    ok = 0
    write_failed = 0
    skipped_empty = 0
    missing_input = 0

    for sample_idx in iter_sample_indices(start_i, end_i):
        try:
            query_struct_path = resolve_query_struct_path(sample_idx, query_struct_dir)
            topk_path = resolve_topk_path(sample_idx, topk_dir)
            local_structure_path = resolve_local_structure_path(sample_idx, local_structure_dir)
            out_path = resolve_summary_out_path(sample_idx, summary_out_dir)

            if args.query_mode == "structured" and not query_struct_path.exists():
                missing_input += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing query_struct: {query_struct_path}")
                continue

            if not topk_path.exists():
                missing_input += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing topk json: {topk_path}")
                continue

            if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
                print(f"[SKIP] existing output (use --overwrite): {out_path}")
                continue

            raw_text = grouped.get(sample_idx, "")
            if not raw_text:
                write_failed += 1
                failed_rows.append(f"sample-{sample_idx}\tmissing batch model output")
                continue

            out = process_one_summary_output(
                sample_idx=sample_idx,
                query_struct_path=query_struct_path,
                topk_path=topk_path,
                local_structure_path=local_structure_path,
                raw_text=raw_text,
                use_plan=bool(args.use_plan),
            )
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] wrote {out_path}")
            ok += 1

        except EmptyFinalSummaryError:
            skipped_empty += 1
            print(f"[SKIP-EMPTY] summary-{sample_idx}: final_summary is empty")
        except Exception as e:
            write_failed += 1
            failed_rows.append(f"sample-{sample_idx}\t{e}")

    failed_out.write_text("\n".join(failed_rows) + ("\n" if failed_rows else ""), encoding="utf-8")

    print(json.dumps({
        "jobs_dir": str(jobs_dir),
        "pattern": pattern,
        "summary_out_dir": str(summary_out_dir),
        "start_idx": start_i,
        "end_idx": end_i,
        "num_output_files": len(paths),
        "ok": ok,
        "parse_failed": parse_failed,
        "skipped_empty": skipped_empty,
        "write_failed": write_failed,
        "missing_input": missing_input,
        "failed_out": str(failed_out),
    }, indent=2, ensure_ascii=False))



# ============================================================
# Proxy direct Chat Completions (no native Batch billing).
# Preserves exact Summary S3 prompt, parsing, and output schema.
# ============================================================
def cmd_run_chat(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import Counter
    import threading

    start, end = int(args.start_idx), int(args.end_idx)
    indices = iter_sample_indices(start, end)  # end-exclusive, same as Batch S3
    if int(args.workers) < 1:
        raise ValueError("workers must be >= 1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.job_name):
        raise ValueError("Invalid job_name")
    if args.query_mode not in {"original", "structured"}:
        raise ValueError("query_mode must be original or structured")
    if not os.environ.get("PROXY_API_KEY"):
        raise EnvironmentError("PROXY_API_KEY is required")

    qdir = Path(args.query_struct_dir).expanduser().resolve() if args.query_struct_dir else None
    topkdir = Path(args.topk_dir).expanduser().resolve()
    jobsdir = Path(args.jobs_dir).expanduser().resolve()
    summarydir = Path(args.summary_out_dir).expanduser().resolve()
    if not topkdir.is_dir():
        raise FileNotFoundError(f"Missing TOPK_DIR: {topkdir}")
    if args.query_mode == "structured" and not qdir.is_dir():
        raise FileNotFoundError(f"Missing QUERY_STRUCT_DIR: {qdir}")
    rundir = jobsdir / args.job_name
    rawdir = rundir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    summarydir.mkdir(parents=True, exist_ok=True)

    settings = {
        "query_struct_dir": str(qdir) if qdir is not None else "",
        "topk_dir": str(topkdir),
        "summary_out_dir": str(summarydir),
        "model_requested": args.openai_model,
        "api_base_url": api_configuration("proxy")[1],
        "query_mode": args.query_mode,
        "use_plan": int(args.use_plan),
        "max_output_tokens": int(args.max_output_tokens),
        "max_core": int(args.max_core),
        "max_support": int(args.max_support),
        "max_packed": int(args.max_packed),
    }
    config_path = rundir / "run_chat.config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != settings:
            raise RuntimeError(f"Existing job has different S3 settings; use another JOB_NAME: {config_path}")
    else:
        _write_atomic_json_proxy(config_path, settings)

    pending = []
    existing_valid = 0
    for i in indices:
        final = resolve_summary_out_path(i, summarydir)
        if final.is_file() and final.stat().st_size > 0 and not args.overwrite:
            obj = json.loads(final.read_text(encoding="utf-8"))
            if not isinstance(obj, dict) or not isinstance(obj.get("final_summary"), str) or not obj["final_summary"].strip():
                raise ValueError(f"Existing summary_{i}.json is invalid: {final}")
            if args.use_plan and (not isinstance(obj.get("plan"), list) or obj.get("plan_style") not in {"direct", "parallel"}):
                raise ValueError(f"Existing summary_{i}.json has no valid plan: {final}")
            if not args.use_plan and (obj.get("plan_style") != "none" or obj.get("plan") != []):
                raise ValueError(f"Existing summary_{i}.json is not no-plan: {final}")
            existing_valid += 1
        else:
            pending.append(i)

    print(f"[CHAT RUN] requested={len(indices)} pending={len(pending)} existing_valid={existing_valid} "
          f"workers={args.workers} model={args.openai_model} query_mode={args.query_mode} "
          f"use_plan={args.use_plan} range=[{start},{end})", flush=True)
    print("[MODE] Proxy Chat Completions (NOT native Batch billing); Summary S3 prompt/parser unchanged", flush=True)

    local_state = threading.local()

    def worker(i):
        qpath = resolve_query_struct_path(i, qdir)
        tpath = resolve_topk_path(i, topkdir)
        if not tpath.is_file():
            raise FileNotFoundError(f"Missing topk_{i}.json: {tpath}")
        topk = load_topk_json(tpath)
        if "sample_idx" in topk and topk["sample_idx"] != i:
            raise ValueError(f"topk_{i}.json sample_idx mismatch")
        query, qstruct, topk, local_obj = load_current_context(
            query_struct_path=str(qpath), topk_path=str(tpath),
            query_mode=args.query_mode,
        )
        # This existing function is the exact one used by the S3 Batch path:
        # same evidence rendering, original-query prompt and plan instructions.
        request = build_one_request(
            sample_idx=i, query=query, query_struct=qstruct,
            topk_obj=topk, local_obj=local_obj,
            openai_model=args.openai_model,
            max_output_tokens=int(args.max_output_tokens),
            query_mode=args.query_mode, use_plan=bool(args.use_plan),
            max_core=int(args.max_core), max_support=int(args.max_support),
            max_packed=int(args.max_packed),
        )
        payload = request["body"]
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        rawpath = rawdir / f"summary-{i}.json"
        cached = None
        if rawpath.is_file() and not args.overwrite:
            saved = json.loads(rawpath.read_text(encoding="utf-8"))
            if saved.get("prompt_sha256") == digest and isinstance(saved.get("output_text"), str):
                try:
                    process_one_summary_output(i, qpath, tpath, None, saved["output_text"], bool(args.use_plan))
                except (ValueError, KeyError, TypeError):
                    pass  # Invalid/truncated API reply: regenerate rather than cache a broken label
                else:
                    cached = saved
        if cached is not None:
            text = cached["output_text"]
            model_returned = cached.get("returned_model") or "unknown"
        else:
            client = getattr(local_state, "client", None)
            if client is None:
                client = _get_proxy_chat_client()
                local_state.client = client
            text, model_returned, response = _proxy_chat_completion(
                client=client,
                system_prompt=payload["input"][0]["content"],
                user_prompt=payload["input"][1]["content"],
                model=payload["model"],
                max_output_tokens=payload["max_output_tokens"],
            )
            _write_atomic_json_proxy(rawpath, {
                "custom_id": request["custom_id"],
                "prompt_sha256": digest,
                "requested_model": args.openai_model,
                "returned_model": model_returned,
                "output_text": text,
                "response": response.model_dump() if hasattr(response, "model_dump") else None,
            })

        output = process_one_summary_output(
            sample_idx=i, query_struct_path=qpath, topk_path=tpath,
            local_structure_path=None, raw_text=text, use_plan=bool(args.use_plan),
        )
        final = resolve_summary_out_path(i, summarydir)
        _write_atomic_json_proxy(final, output)
        return i, final, model_returned

    from collections import Counter
    succeeded, failed = 0, {}
    model_distribution = Counter()
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        tasks = {executor.submit(worker, i): i for i in pending}
        for future in as_completed(tasks):
            i = tasks[future]
            try:
                _, path, version = future.result()
                succeeded += 1
                model_distribution[version] += 1
                print(f"[OK] summary-{i} model={version} -> {path}", flush=True)
            except Exception as exc:
                failed[i] = f"{type(exc).__name__}: {exc}"
                print(f"[FAILED] summary-{i}: {failed[i]}", flush=True)

    missing = [i for i in indices if not resolve_summary_out_path(i, summarydir).is_file()]
    summary = {
        "requested": len(indices),
        "succeeded_this_run": succeeded,
        "skipped_existing": existing_valid,
        "final_valid": len(indices) - len(missing),
        "final_missing": missing,
        "failed": failed,
        "returned_model_distribution_this_run": dict(model_distribution),
        "summary_out_dir": str(summarydir),
        "job_dir": str(rundir),
        "settings": settings,
    }
    _write_atomic_json_proxy(rundir / "run_chat.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if missing or failed:
        raise RuntimeError(f"Proxy Summary S3 incomplete: failed={len(failed)} missing={len(missing)}")

# ============================================================
# Optional single local/openai mode for parity
# ============================================================
def generate_openai_chat(system_prompt: str, user_prompt: str, model_name: str, max_output_tokens: int) -> str:
    client = _get_client()
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
    )
    text = getattr(resp, "output_text", "") or ""
    if not text.strip():
        raise ValueError("Empty OpenAI model output.")
    return text.strip()


def call_model_single(
    query: str,
    query_struct: Dict[str, Any],
    topk_obj: Dict[str, Any],
    local_obj: Optional[Dict[str, Any]] = None,
    backend: str = "openai",
    openai_model: str = "gpt-4o",
    tokenizer=None,
    local_model=None,
    max_output_tokens: int = 600,
    query_mode: str = "structured",
    use_plan: bool = True,
    max_core: int = 5,
    max_support: int = 5,
    max_packed: int = 12,
) -> Dict[str, Any]:
    evidence_lines = build_evidence_lines(topk_obj, max_core, max_support, max_packed)
    system_prompt, user_prompt = build_summary_prompt(
        query, query_struct, evidence_lines, query_mode=query_mode, use_plan=bool(use_plan)
    )
    if backend == "proxy":
        raw_text = generate_proxy_chat(system_prompt, user_prompt, openai_model, max_output_tokens)
    elif backend == "openai":
        raw_text = generate_openai_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=openai_model,
            max_output_tokens=max_output_tokens,
        )
    elif backend == "local":
        if tokenizer is None or local_model is None:
            raise ValueError("backend='local' requires tokenizer and local_model")
        raw_text = generate_local_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tokenizer=tokenizer,
            model=local_model,
            max_new_tokens=max_output_tokens,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return parse_summary_json(raw_text, use_plan=bool(use_plan))


def cmd_single(args):
    tokenizer = None
    local_model = None
    if args.backend == "local":
        tokenizer, local_model = load_local_model(args.model_name)

    query, query_struct, topk_obj, local_obj = load_current_context(
        query_struct_path=args.query_struct,
        topk_path=args.topk,
        local_structure_path=args.local_structure,
        query_mode=args.query_mode,
    )

    out = call_model_single(
        query=query,
        query_struct=query_struct,
        topk_obj=topk_obj,
        local_obj=local_obj,
        backend=args.backend,
        openai_model=args.openai_model,
        tokenizer=tokenizer,
        local_model=local_model,
        max_output_tokens=int(args.max_output_tokens),
        query_mode=args.query_mode,
        use_plan=bool(args.use_plan),
        max_core=args.max_core,
        max_support=args.max_support,
        max_packed=args.max_packed,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build_split = sub.add_parser("build_split")
    ap_build_split.add_argument("--query_struct_dir", type=str, default=DEFAULT_QUERY_STRUCT_DIR)
    ap_build_split.add_argument("--topk_dir", type=str, default=DEFAULT_TOPK_DIR)
    ap_build_split.add_argument("--local_structure_dir", type=str, default="")
    ap_build_split.add_argument("--out_dir", type=str, default=DEFAULT_JOBS_DIR)
    ap_build_split.add_argument("--start_idx", type=int, required=True)
    ap_build_split.add_argument("--end_idx", type=int, required=True)
    ap_build_split.add_argument("--chunk_size", type=int, default=50)
    ap_build_split.add_argument("--job_name", type=str, default="asqa_summary_batch")
    ap_build_split.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_build_split.add_argument("--max_output_tokens", type=int, default=600)
    ap_build_split.add_argument("--query_mode", choices=["structured", "original"], default=QUERY_MODE_DEFAULT)
    ap_build_split.add_argument("--use_plan", type=int, choices=[0, 1], default=1)
    ap_build_split.add_argument("--max_core", type=int, default=5)
    ap_build_split.add_argument("--max_support", type=int, default=5)
    ap_build_split.add_argument("--max_packed", type=int, default=12)
    ap_build_split.set_defaults(func=cmd_build_split)

    ap_submit_split = sub.add_parser("submit_split")
    ap_submit_split.add_argument("--query_struct_dir", type=str, default=DEFAULT_QUERY_STRUCT_DIR)
    ap_submit_split.add_argument("--topk_dir", type=str, default=DEFAULT_TOPK_DIR)
    ap_submit_split.add_argument("--local_structure_dir", type=str, default="")
    ap_submit_split.add_argument("--out_dir", type=str, default=DEFAULT_JOBS_DIR)
    ap_submit_split.add_argument("--start_idx", type=int, required=True)
    ap_submit_split.add_argument("--end_idx", type=int, required=True)
    ap_submit_split.add_argument("--chunk_size", type=int, default=50)
    ap_submit_split.add_argument("--job_name", type=str, default="asqa_summary_batch")
    ap_submit_split.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_submit_split.add_argument("--max_output_tokens", type=int, default=600)
    ap_submit_split.add_argument("--completion_window", type=str, default="24h")
    ap_submit_split.add_argument("--query_mode", choices=["structured", "original"], default=QUERY_MODE_DEFAULT)
    ap_submit_split.add_argument("--use_plan", type=int, choices=[0, 1], default=1)
    ap_submit_split.add_argument("--max_core", type=int, default=5)
    ap_submit_split.add_argument("--max_support", type=int, default=5)
    ap_submit_split.add_argument("--max_packed", type=int, default=12)
    ap_submit_split.set_defaults(func=cmd_submit_split)

    ap_status = sub.add_parser("status")
    ap_status.add_argument("--batch_id", type=str, required=True)
    ap_status.set_defaults(func=cmd_status)

    ap_status_tsv = sub.add_parser("status_from_tsv", aliases=["status_tsv"])
    ap_status_tsv.add_argument("--tsv", type=str, default="")
    ap_status_tsv.set_defaults(func=cmd_status_tsv)

    ap_consume = sub.add_parser("consume")
    ap_consume.add_argument("--batch_id", type=str, required=True)
    ap_consume.add_argument("--query_struct_dir", type=str, default=DEFAULT_QUERY_STRUCT_DIR)
    ap_consume.add_argument("--topk_dir", type=str, default=DEFAULT_TOPK_DIR)
    ap_consume.add_argument("--local_structure_dir", type=str, default="")
    ap_consume.add_argument("--summary_out_dir", type=str, default=DEFAULT_SUMMARY_DIR)
    ap_consume.add_argument("--raw_out_jsonl", type=str, default="")
    ap_consume.add_argument("--failed_ids_out", type=str, default="")
    ap_consume.add_argument("--start_idx", type=int, required=True)
    ap_consume.add_argument("--end_idx", type=int, required=True)
    ap_consume.add_argument("--query_mode", choices=["structured", "original"], default=QUERY_MODE_DEFAULT)
    ap_consume.add_argument("--use_plan", type=int, choices=[0, 1], default=1)
    ap_consume.add_argument("--max_core", type=int, default=5)
    ap_consume.add_argument("--max_support", type=int, default=5)
    ap_consume.add_argument("--max_packed", type=int, default=12)
    ap_consume.set_defaults(func=cmd_consume)

    ap_consume_tsv = sub.add_parser("consume_from_tsv", aliases=["consume_tsv"])
    ap_consume_tsv.add_argument("--tsv", type=str, default="")
    ap_consume_tsv.add_argument("--jobs_dir", type=str, default=DEFAULT_JOBS_DIR)
    ap_consume_tsv.add_argument("--query_struct_dir", type=str, default=DEFAULT_QUERY_STRUCT_DIR)
    ap_consume_tsv.add_argument("--topk_dir", type=str, default=DEFAULT_TOPK_DIR)
    ap_consume_tsv.add_argument("--local_structure_dir", type=str, default="")
    ap_consume_tsv.add_argument("--summary_out_dir", type=str, default=DEFAULT_SUMMARY_DIR)
    ap_consume_tsv.add_argument("--failed_ids_out", type=str, default="")
    ap_consume_tsv.add_argument("--query_mode", choices=["structured", "original"], default=QUERY_MODE_DEFAULT)
    ap_consume_tsv.add_argument("--use_plan", type=int, choices=[0, 1], default=1)
    ap_consume_tsv.add_argument("--max_core", type=int, default=5)
    ap_consume_tsv.add_argument("--max_support", type=int, default=5)
    ap_consume_tsv.add_argument("--max_packed", type=int, default=12)
    ap_consume_tsv.add_argument("--overwrite", action="store_true", help="Allow replacing existing summary JSON outputs")
    ap_consume_tsv.set_defaults(func=cmd_consume_tsv)

    ap_consume_local = sub.add_parser("consume_local")
    ap_consume_local.add_argument("--jobs_dir", type=str, required=True)
    ap_consume_local.add_argument("--pattern", type=str, default="batch_*.output.jsonl")
    ap_consume_local.add_argument("--query_struct_dir", type=str, default=DEFAULT_QUERY_STRUCT_DIR)
    ap_consume_local.add_argument("--topk_dir", type=str, default=DEFAULT_TOPK_DIR)
    ap_consume_local.add_argument("--local_structure_dir", type=str, default="")
    ap_consume_local.add_argument("--summary_out_dir", type=str, default=DEFAULT_SUMMARY_DIR)
    ap_consume_local.add_argument("--failed_ids_out", type=str, required=True)
    ap_consume_local.add_argument("--start_idx", type=int, required=True)
    ap_consume_local.add_argument("--end_idx", type=int, required=True)
    ap_consume_local.add_argument("--query_mode", choices=["structured", "original"], default=QUERY_MODE_DEFAULT)
    ap_consume_local.add_argument("--overwrite", action="store_true", help="replace existing summary files")
    ap_consume_local.add_argument("--use_plan", type=int, choices=[0, 1], default=1)
    ap_consume_local.add_argument("--max_core", type=int, default=5)
    ap_consume_local.add_argument("--max_support", type=int, default=5)
    ap_consume_local.add_argument("--max_packed", type=int, default=12)
    ap_consume_local.set_defaults(func=cmd_consume_local)

    ap_run_chat = sub.add_parser("run_chat")
    ap_run_chat.add_argument("--query_struct_dir", type=str, default=DEFAULT_QUERY_STRUCT_DIR)
    ap_run_chat.add_argument("--topk_dir", type=str, default=DEFAULT_TOPK_DIR)
    ap_run_chat.add_argument("--jobs_dir", type=str, default=DEFAULT_JOBS_DIR)
    ap_run_chat.add_argument("--summary_out_dir", type=str, default=DEFAULT_SUMMARY_DIR)
    ap_run_chat.add_argument("--start_idx", type=int, required=True)
    ap_run_chat.add_argument("--end_idx", type=int, required=True)
    ap_run_chat.add_argument("--job_name", type=str, default="asqa_train_summary_s3_proxy")
    ap_run_chat.add_argument("--workers", type=int, default=4)
    ap_run_chat.add_argument("--openai_model", type=str, default="gpt-4o")
    ap_run_chat.add_argument("--max_output_tokens", type=int, default=1200)
    ap_run_chat.add_argument("--query_mode", choices=["structured", "original"], default="original")
    ap_run_chat.add_argument("--use_plan", type=int, choices=[0, 1], default=1)
    ap_run_chat.add_argument("--max_core", type=int, default=5)
    ap_run_chat.add_argument("--max_support", type=int, default=5)
    ap_run_chat.add_argument("--max_packed", type=int, default=12)
    ap_run_chat.add_argument("--overwrite", action="store_true")
    ap_run_chat.set_defaults(func=cmd_run_chat)

    ap_single = sub.add_parser("single")
    ap_single.add_argument("--backend", type=str, default=BACKEND, choices=["openai", "local", "proxy"])
    ap_single.add_argument("--openai_model", type=str, default=OPENAI_MODEL)
    ap_single.add_argument("--model_name", type=str, default=LOCAL_MODEL_NAME)
    ap_single.add_argument("--max_output_tokens", type=int, default=MAX_OUTPUT_TOKENS)
    ap_single.add_argument("--query_struct", default="",
                           help="Required for structured mode or original-mode fallback when topk.query is missing")
    ap_single.add_argument("--topk", required=True)
    ap_single.add_argument("--local_structure", default="")
    ap_single.add_argument("--out", default="")
    ap_single.add_argument("--query_mode", choices=["structured", "original"], default=QUERY_MODE_DEFAULT)
    ap_single.add_argument("--use_plan", type=int, choices=[0, 1], default=1)
    ap_single.add_argument("--max_core", type=int, default=5)
    ap_single.add_argument("--max_support", type=int, default=5)
    ap_single.add_argument("--max_packed", type=int, default=12)
    ap_single.set_defaults(func=cmd_single)

    args = ap.parse_args()
    validate_optional_paths(args)
    if args.cmd in {"submit_split", "status", "status_from_tsv", "status_tsv", "consume", "consume_from_tsv", "consume_tsv"}:
        require_batch()
    configure_paths(args, ("topk_dir", "out_dir", "jobs_dir", "summary_out_dir", "topk"))
    if getattr(args, "query_mode", "original") == "structured":
        configure_paths(args, ("query_struct_dir", "query_struct"))
    elif getattr(args, "query_struct_dir", ""):
        args.query_struct_dir = required_path(args.query_struct_dir, "--query_struct_dir", "dir")
    if getattr(args, "backend", "") == "local":
        args.model_name = str(required_path(args.model_name, "--model_name / MODEL", "dir"))
    if hasattr(args, "tsv") and not args.tsv:
        args.tsv = os.environ.get("TSV_PATH") or os.environ.get("TSV", "") or str(required_path(DEFAULT_JOBS_DIR, "--tsv / TSV or JOBS_DIR") / "latest_batches.tsv")
    args.func(args)


if __name__ == "__main__":
    main()
