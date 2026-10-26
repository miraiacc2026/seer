#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, lazy_symbol, required_path, env_path, api_client, api_configuration
torch = LazyModule("torch")
AutoTokenizer = lazy_symbol("transformers", "AutoTokenizer")
AutoModelForCausalLM = lazy_symbol("transformers", "AutoModelForCausalLM")
OpenAI = lazy_symbol("openai", "OpenAI")

MODEL_DEFAULT = os.environ.get("MODEL", "")

BASE_TOKENIZER_DEFAULT = os.environ.get(
    "BASE_TOKENIZER",
    MODEL_DEFAULT,
)

DISABLE_QWEN_THINKING = os.environ.get("DISABLE_QWEN_THINKING", "1") == "1"

# Match data_gen_batch/rerank_topk_asqa.py evidence-packing fallback.
MIN_PACKED_SENTS = int(os.environ.get("MIN_PACKED_SENTS", "3"))


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


def load_tokenizer_safely():
    tokenizer = AutoTokenizer.from_pretrained(
        str(required_path(BASE_TOKENIZER_DEFAULT, "BASE_TOKENIZER / MODEL", "dir")),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def generate_openai_chat(system_prompt: str, user_prompt: str, model_name: str, max_output_tokens: int):
    if OpenAI is None:
        raise ImportError(
            "openai package is not available in this environment. "
            "Use BACKEND=local or install openai into the active Python env."
        )

    client = api_client("openai")

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
        raise ValueError("Empty output from OpenAI.")
    return text.strip()


# def apply_chat_template_qwen_safe(tokenizer, messages):
#     kwargs = dict(
#         messages=messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_tensors="pt",
#         return_dict=True,
#     )

#     if DISABLE_QWEN_THINKING:
#         try:
#             return tokenizer.apply_chat_template(
#                 **kwargs,
#                 enable_thinking=False,
#             )
#         except TypeError:
#             pass

#     return tokenizer.apply_chat_template(**kwargs)
def apply_chat_template_qwen_safe(tokenizer, messages):
    template_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    if DISABLE_QWEN_THINKING:
        try:
            return tokenizer.apply_chat_template(
                conversation=messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError as e:
            msg = str(e)
            if "enable_thinking" not in msg:
                try:
                    return tokenizer.apply_chat_template(
                        messages,
                        **template_kwargs,
                    )
                except TypeError:
                    raise e

    try:
        return tokenizer.apply_chat_template(
            conversation=messages,
            **template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
        )

# def generate_local_chat(
#     system_prompt: str,
#     user_prompt: str,
#     tokenizer,
#     model,
#     max_new_tokens: int,
# ):
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]

#     inputs = apply_chat_template_qwen_safe(tokenizer, messages)

#     model_device = getattr(model, "device", None)
#     if model_device is None:
#         model_device = next(model.parameters()).device

#     inputs = {k: v.to(model_device) for k, v in inputs.items()}

#     eos_token_id = tokenizer.eos_token_id
#     pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             num_beams=1,
#             eos_token_id=eos_token_id,
#             pad_token_id=pad_token_id,
#         )

#     gen = output_ids[0][inputs["input_ids"].shape[-1]:]
#     text = tokenizer.decode(gen, skip_special_tokens=True).strip()
#     text = strip_qwen_thinking(text)

#     if not text:
#         raise ValueError("Empty output from local model.")
#     return text
def generate_local_chat(
    system_prompt: str,
    user_prompt: str,
    tokenizer,
    model,
    max_new_tokens: int,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = apply_chat_template_qwen_safe(tokenizer, messages)

    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}

    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device

    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    gen = output_ids[0][prompt_len:]

    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    text = strip_qwen_thinking(text)

    if not text:
        raise ValueError("Empty output from local model.")
    return text


def generate_local_chat_batch(
    requests,
    tokenizer,
    model,
    max_new_tokens: int,
):
    """Generate several independent chat prompts in one model.generate call.

    Each request is a ``(system_prompt, user_prompt)`` pair.  The prompts are
    left-padded to a common width, so greedy decoding is identical to running
    each prompt separately while allowing the GPU to process them together.
    """
    if not requests:
        return []
    if len(requests) == 1:
        system_prompt, user_prompt = requests[0]
        return [generate_local_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
        )]

    encoded = []
    for system_prompt, user_prompt in requests:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        item = apply_chat_template_qwen_safe(tokenizer, messages)
        if isinstance(item, torch.Tensor):
            item = {"input_ids": item}
        normalized = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] == 1:
                value = value[0]
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            normalized[key] = value
        encoded.append(normalized)

    # tokenizer.pad handles left padding and creates a single attention mask.
    inputs = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )

    prompt_width = inputs["input_ids"].shape[-1]
    generated = output_ids[:, prompt_width:]
    texts = []
    for row in generated:
        text = tokenizer.decode(row, skip_special_tokens=True).strip()
        text = strip_qwen_thinking(text)
        if not text:
            raise ValueError("Empty output from local model in batched generation.")
        texts.append(text)
    return texts


# ============================================================
# Query-struct normalization helpers
# ============================================================
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


# ============================================================
# Prompt
# ============================================================
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


# ============================================================
# Robust parsing with salvage
# ============================================================
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


def rerank_candidates(
    query: str,
    qstruct: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    backend: str = "local",
    openai_model: str = "gpt-5.2",
    tokenizer=None,
    local_model=None,
    batch_size: int = 6,
    max_output_tokens: int = 1400,
    query_mode: str = "structured",
):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    labeled = []
    prompt_batches = []
    idx = 0
    while idx < len(candidate_pool):
        batch = candidate_pool[idx: idx + batch_size]
        prompt_batches.append((idx, batch, build_rerank_prompt(
            query, qstruct, batch, query_mode=query_mode
        )))
        idx += batch_size

    if prompt_batches and int(os.environ.get("SAMPLE_IDX", "-1")) == 0:
        system_prompt, user_prompt = prompt_batches[0][2]
        print("========== SYSTEM PROMPT ==========")
        print(system_prompt)
        print("========== USER PROMPT ==========")
        print(user_prompt)
        print("==================================")

    if backend == "local":
        if tokenizer is None or local_model is None:
            raise ValueError("backend='local' requires tokenizer and local_model.")
        # All prompt chunks for one sample share the same task and generation
        # length.  Running them as one batch removes the per-chunk Python/GPU
        # launch overhead while preserving one result per candidate chunk.
        # The default of 3 matches 18 candidates / 6 candidates per prompt.
        infer_batch = int(os.environ.get("RERANK_INFER_BATCH", "3"))
        if infer_batch <= 0:
            infer_batch = len(prompt_batches) or 1
        raw_outputs = []
        for start in range(0, len(prompt_batches), infer_batch):
            group = prompt_batches[start: start + infer_batch]
            raw_outputs.extend(generate_local_chat_batch(
                requests=[item[2] for item in group],
                tokenizer=tokenizer,
                model=local_model,
                max_new_tokens=max_output_tokens,
            ))
    elif backend == "openai":
        raw_outputs = [
            generate_openai_chat(system_prompt, user_prompt, openai_model, max_output_tokens)
            for _, _, (system_prompt, user_prompt) in prompt_batches
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    for (idx, batch, _), raw in zip(prompt_batches, raw_outputs):
        raw = strip_qwen_thinking(raw)
        try:
            parsed = parse_rerank_json(raw)
            results = parsed.get("results", [])
            if not isinstance(results, list):
                results = []
        except Exception as e:
            print(f"[WARN] parse_rerank_json failed at batch starting idx={idx}: {e}")
            try:
                debug_path = env_path("OUT_JSON").with_suffix(f".batch_{idx}.raw_error.txt")
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(raw, encoding="utf-8")
            except Exception:
                pass
            results = []

        if not results:
            salvaged = _salvage_partial_results(raw)
            if salvaged:
                print(f"[WARN] using salvaged partial results at batch idx={idx}: {len(salvaged)} items")
                results = salvaged
                try:
                    debug_path = env_path("OUT_JSON").with_suffix(f".batch_{idx}.raw_error.txt")
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    debug_path.write_text(raw, encoding="utf-8")
                except Exception:
                    pass

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


# def pack_final_evidence(
#     labeled_rows: List[Dict[str, Any]],
#     topk_core: int = 5,
#     topk_support: int = 8,
#     max_packed_sents: int = 14,
# ):
#     core_rows = [r for r in labeled_rows if r.get("bucket") == "core"]
#     support_rows = [r for r in labeled_rows if r.get("bucket") == "support"]
#     drop_rows = [r for r in labeled_rows if r.get("bucket") == "drop"]

#     core_rows.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)
#     support_rows.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)
#     drop_rows.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)

#     core_rows = core_rows[:topk_core]
#     support_rows = support_rows[:topk_support]

#     if len(core_rows) == 0 and len(support_rows) == 0 and labeled_rows:
#         def sort_key(x):
#             sc = x.get("score", {}) or {}
#             return (
#                 float(x.get("llm_score", 0.0)),
#                 float(sc.get("total", 0.0)),
#             )

#         fallback = sorted(labeled_rows, key=sort_key, reverse=True)[0]
#         fallback = dict(fallback)
#         fallback["bucket"] = "support"
#         fallback["llm_score"] = max(float(fallback.get("llm_score", 0.0)), 0.01)

#         old_reason = str(fallback.get("reason", "")).strip()
#         if old_reason:
#             fallback["reason"] = old_reason + " | fallback_keep_one"
#         else:
#             fallback["reason"] = "fallback_keep_one"

#         support_rows = [fallback]

#         fallback_key = (int(fallback["doc_id"]), int(fallback["sent_id"]))
#         drop_rows = [
#             r for r in drop_rows
#             if (int(r["doc_id"]), int(r["sent_id"])) != fallback_key
#         ]

#     packed = []
#     seen = set()

#     for group_name, rows in [("core", core_rows), ("support", support_rows)]:
#         for r in rows:
#             if len(packed) >= max_packed_sents:
#                 break
#             k = (int(r["doc_id"]), int(r["sent_id"]))
#             if k in seen:
#                 continue
#             seen.add(k)
#             packed.append(_make_slim_row_with_meta(r, source=group_name))

#     return core_rows, support_rows, drop_rows, packed

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


def main():
    import argparse
    argparse.ArgumentParser(description="Single-sample reranking. Required env: CANDIDATE_JSON, OUT_JSON, MODEL (local).").parse_args()
    sample_idx = int(os.environ.get("SAMPLE_IDX", "0"))
    candidate_path = env_path("CANDIDATE_JSON", "file")
    out_path = env_path("OUT_JSON")

    BACKEND = os.environ.get("BACKEND", "local").strip().lower()
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    QUERY_MODE = os.environ.get("QUERY_MODE", "structured").strip().lower()
    if QUERY_MODE not in {"structured", "original"}:
        raise ValueError("QUERY_MODE must be 'structured' or 'original'")
    RERANK_BATCH = int(os.environ.get("RERANK_BATCH", "6"))
    RERANK_INFER_BATCH = int(os.environ.get("RERANK_INFER_BATCH", "3"))
    MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1400"))

    TOPK_CORE = int(os.environ.get("TOPK_CORE", "5"))
    TOPK_SUPPORT = int(os.environ.get("TOPK_SUPPORT", "8"))
    MAX_PACKED_SENTS = int(os.environ.get("MAX_PACKED_SENTS", "14"))

    tokenizer = None
    local_model = None

    if BACKEND == "openai":
        if OpenAI is None:
            raise ImportError("openai package is not installed.")
        api_configuration("openai")

    elif BACKEND == "local":
        local_model_name = str(env_path("MODEL", "dir"))

        if not Path(local_model_name).expanduser().exists():
            raise FileNotFoundError(f"Local MODEL path not found: {local_model_name}")

        if not Path(BASE_TOKENIZER_DEFAULT).expanduser().exists():
            raise FileNotFoundError(f"BASE_TOKENIZER path not found: {BASE_TOKENIZER_DEFAULT}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        tokenizer = load_tokenizer_safely()
        print(f"[INFO] Loaded tokenizer from: {BASE_TOKENIZER_DEFAULT}")

        attn_impl = os.environ.get("ATTN_IMPL", "").strip()
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if device == "cuda" else None,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        local_model = AutoModelForCausalLM.from_pretrained(
            local_model_name,
            **model_kwargs,
        )
        local_model.eval()

        if getattr(local_model, "generation_config", None) is not None:
            local_model.generation_config.do_sample = False
            local_model.generation_config.temperature = None
            local_model.generation_config.top_p = None
            local_model.generation_config.top_k = None
            local_model.generation_config.pad_token_id = tokenizer.pad_token_id
            local_model.generation_config.eos_token_id = tokenizer.eos_token_id

        print(f"[INFO] Loaded local model from: {local_model_name}")
        print(f"[INFO] BACKEND={BACKEND}")
        print(f"[INFO] DISABLE_QWEN_THINKING={DISABLE_QWEN_THINKING}")
        print(f"[INFO] torch.cuda.is_available={torch.cuda.is_available()}")
        print(f"[INFO] torch.cuda.device_count={torch.cuda.device_count()}")

    else:
        raise ValueError(f"Unsupported BACKEND: {BACKEND}")

    obj = json.loads(candidate_path.read_text(encoding="utf-8"))

    if not isinstance(obj, dict):
        raise ValueError(f"candidate json must be object: {candidate_path}")
    if "query" not in obj:
        raise ValueError(f"candidate json missing key 'query': {candidate_path}")
    if "candidate_pool" not in obj:
        raise ValueError(f"candidate json missing key 'candidate_pool': {candidate_path}")
    if not isinstance(obj["candidate_pool"], list):
        raise ValueError(f"candidate_pool must be list: {candidate_path}")

    query = obj["query"]
    qstruct = normalize_query_struct(obj.get("query_struct", {}), query)

    max_input_candidates = int(os.environ.get("MAX_INPUT_CANDIDATES", "18"))
    candidate_pool = obj["candidate_pool"][:max_input_candidates]

    print(f"[INFO] sample_idx={sample_idx}")
    print(f"[INFO] candidate_path={candidate_path}")
    print(f"[INFO] out_path={out_path}")
    print(f"[INFO] query={query}")
    print(f"[INFO] num_candidates={len(candidate_pool)}")
    print(f"[INFO] has_query_struct={'query_struct' in obj}")
    print(f"[INFO] RERANK_BATCH={RERANK_BATCH}")
    print(f"[INFO] RERANK_INFER_BATCH={RERANK_INFER_BATCH}")
    print(f"[INFO] QUERY_MODE={QUERY_MODE}")
    print(f"[INFO] MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS}")

    labeled_rows = rerank_candidates(
        query=query,
        qstruct=qstruct,
        candidate_pool=candidate_pool,
        backend=BACKEND,
        openai_model=OPENAI_MODEL,
        tokenizer=tokenizer,
        local_model=local_model,
        batch_size=RERANK_BATCH,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        query_mode=QUERY_MODE,
    )

    core_rows, support_rows, drop_rows, packed = pack_final_evidence(
        labeled_rows=labeled_rows,
        topk_core=TOPK_CORE,
        topk_support=TOPK_SUPPORT,
        max_packed_sents=MAX_PACKED_SENTS,
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
            "backend": BACKEND,
            "openai_model": OPENAI_MODEL if BACKEND == "openai" else None,
            "local_model": os.environ.get("MODEL") if BACKEND == "local" else None,
            "base_tokenizer": BASE_TOKENIZER_DEFAULT if BACKEND == "local" else None,
            "disable_qwen_thinking": DISABLE_QWEN_THINKING if BACKEND == "local" else None,
            "num_candidates": len(candidate_pool),
            "topk_core": TOPK_CORE,
            "topk_support": TOPK_SUPPORT,
            "max_packed_sents": MAX_PACKED_SENTS,
            "rerank_batch": RERANK_BATCH,
            "rerank_infer_batch": RERANK_INFER_BATCH,
            "query_mode": QUERY_MODE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FINISHED] saved: {out_path.resolve()}")
    print(
        f"[STATS] candidates={len(candidate_pool)} "
        f"core={len(core_rows)} support={len(support_rows)} drop={len(drop_rows)} "
        f"packed={len(packed)}"
    )


if __name__ == "__main__":
    main()
