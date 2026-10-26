#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, lazy_symbol, required_path, validate_optional_paths
torch = LazyModule("torch")
AutoTokenizer = lazy_symbol("transformers", "AutoTokenizer")
AutoModelForCausalLM = lazy_symbol("transformers", "AutoModelForCausalLM")


# ============================================================
# Config
# ============================================================
LOCAL_MODEL_NAME = os.environ.get("MODEL", "")

BASE_TOKENIZER_NAME = os.environ.get(
    "BASE_TOKENIZER",
    LOCAL_MODEL_NAME,
)

MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "600"))

DISABLE_QWEN_THINKING = os.environ.get("DISABLE_QWEN_THINKING", "1") == "1"
ATTN_IMPL = os.environ.get("ATTN_IMPL", "eager").strip()


# ============================================================
# Qwen3 helpers
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


def _get_model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
# Local model helpers
# ============================================================
def load_local_model(model_name: str, tokenizer_name: str):
    model_name = str(required_path(model_name, "MODEL", "dir"))
    tokenizer_name = str(required_path(tokenizer_name, "BASE_TOKENIZER", "dir"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if device == "cuda" else None,
        "local_files_only": True,
        "trust_remote_code": True,
    }

    if ATTN_IMPL:
        model_kwargs["attn_implementation"] = ATTN_IMPL

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    print(f"[INFO] loaded tokenizer from: {tokenizer_name}")
    print(f"[INFO] loaded model from: {model_name}")
    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[INFO] torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"[INFO] DISABLE_QWEN_THINKING = {DISABLE_QWEN_THINKING}")
    print(f"[INFO] ATTN_IMPL = {ATTN_IMPL}")

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

    inputs = apply_chat_template_qwen_safe(tokenizer, messages)

    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}

    model_device = _get_model_input_device(model)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][prompt_len:]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    text = strip_qwen_thinking(text)

    if not text:
        raise ValueError("Empty local model output.")
    return text


# ============================================================
# Small normalization helpers
# ============================================================
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


# ============================================================
# Query-struct interpretation
# EXACTLY aligned with batch inference.py
# ============================================================
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


# ============================================================
# Topk loading
# EXACTLY aligned with batch inference.py
# ============================================================
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

    if "query" not in obj:
        raise ValueError(f"topk json missing key 'query': {topk_path}")

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


# ============================================================
# Prompt
# S3 prompt is copied verbatim from data_gen_batch/asqa_summary_batch.py.
# ============================================================
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


# ============================================================
# Parse summary output
# Conservative repairs only: preserve the decoded answer text and prompt.
# ============================================================
class EmptyFinalSummaryError(ValueError):
    """A valid model response with no summary, matching the S3 teacher parser."""


def _escape_unescaped_json_quotes(value: str) -> str:
    # Escaping quotes changes only JSON syntax; json.loads returns the same text.
    out = []
    backslashes = 0
    for char in value:
        if char == '"' and backslashes % 2 == 0:
            out.append('\\')
        out.append(char)
        if char == "\\":
            backslashes += 1
        else:
            backslashes = 0
    return "".join(out)


def _repair_summary_json(block: str) -> str:
    block = re.sub(r",\s*([}\]])", r"\1", block)

    # Qwen occasionally emits quotation marks inside claim/summary text without
    # JSON escaping. Find the field boundary from the following known key/end.
    claim_pattern = re.compile(
        r'("claim"\s*:\s*")(.+?)(?="\s*,?\s*"citations"\s*:)',
        flags=re.DOTALL,
    )
    block = claim_pattern.sub(
        lambda m: m.group(1) + _escape_unescaped_json_quotes(m.group(2)), block
    )
    summary_pattern = re.compile(
        r'("final_summary"\s*:\s*")(.+?)(?="\s*}\s*$)',
        flags=re.DOTALL,
    )
    block = summary_pattern.sub(
        lambda m: m.group(1) + _escape_unescaped_json_quotes(m.group(2)), block
    )

    # Citations are DOC_ID strings; repair a missing comma between adjacent IDs.
    def repair_citations(match):
        content = re.sub(r'(?<=")\s+(?=")', ', ', match.group(2))
        return match.group(1) + content + match.group(3)

    block = re.sub(
        r'("citations"\s*:\s*\[)([^\]]*)(\])',
        repair_citations,
        block,
        flags=re.DOTALL,
    )

    # Repair clearly missing separators between known object fields/plan items.
    for _ in range(3):
        previous = block
        block = re.sub(r",\s*([}\]])", r"\1", block)
        block = re.sub(r"(?<=\})\s*(?=\{)", ",\n", block)
        block = re.sub(
            r'(?<=[}\]"])\s*(?="(?:plan_style|plan|final_summary|claim|citations)"\s*:)',
            ",\n",
            block,
        )
        if block == previous:
            break
    return block


def parse_summary_json(raw_text: str, use_plan: bool = True) -> Dict[str, Any]:
    if not isinstance(use_plan, bool):
        raise ValueError("use_plan must be a bool")

    clean = strip_qwen_thinking(raw_text).strip()
    clean = re.sub(r"^\s*```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```\s*$", "", clean).strip()

    candidates = [clean]
    if use_plan:
        # Permit harmless surrounding text, but do not strip a generated plan
        # in no-plan mode.
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match and match.group(0) != clean:
            candidates.insert(0, match.group(0))

    obj = None
    last_error = None
    for candidate in candidates:
        for attempt in (candidate, _repair_summary_json(candidate)):
            try:
                obj = json.loads(attempt)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        if obj is not None:
            break

    if obj is None:
        raise ValueError(f"Invalid summary JSON ({last_error}):\n{clean}")

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
# IO helpers
# ============================================================
def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_single_sample(
    sample_idx: Optional[int],
    topk_path: Optional[str],
    query_path: Optional[str],
    topk_dir: Optional[str],
    query_dir: Optional[str],
) -> Tuple[int, Path, Path, Dict[str, Any], Dict[str, Any], str, Dict[str, Any]]:
    if sample_idx is not None:
        if not topk_dir or not query_dir:
            raise ValueError("When using --sample_idx, both --topk_dir and --query_dir are required.")
        tp = Path(topk_dir).expanduser() / f"topk_{sample_idx}.json"
        qp = Path(query_dir).expanduser() / f"query_{sample_idx}.json"
    else:
        if not topk_path or not query_path:
            raise ValueError("Provide either --sample_idx with dirs, or --topk_path and --query_path.")
        tp = Path(topk_path).expanduser()
        qp = Path(query_path).expanduser()

        m = re.search(r"topk_(\d+)\.json$", tp.name)
        sample_idx = int(m.group(1)) if m else -1

    if not tp.exists():
        raise FileNotFoundError(f"topk json not found: {tp}")
    if not qp.exists():
        raise FileNotFoundError(f"query json not found: {qp}")

    topk_obj = load_topk_json(tp)
    query_obj = _read_json(qp)

    query = _norm_ws(topk_obj.get("query", "") or query_obj.get("query", ""))
    qstruct = _safe_dict(query_obj.get("query_struct", {}))

    if not query:
        raise ValueError(f"query missing in both topk and query json: {tp} / {qp}")
    if not qstruct:
        raise ValueError(f"query_struct missing or invalid: {qp}")

    return sample_idx, tp, qp, topk_obj, query_obj, query, qstruct


def write_summary_json(
    sample_idx: int,
    raw_text: str,
    out_dir: Path,
) -> Path:
    parsed = parse_summary_json(raw_text)

    out = {
        "plan_style": parsed["plan_style"],
        "plan": parsed["plan"],
        "final_summary": parsed["final_summary"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{sample_idx}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ============================================================
# Single-sample pipeline
# This is the local single-item equivalent of batch's:
# build_one_request -> model generate -> parse_summary_json -> write_summary_json
# ============================================================
def run_single_local(
    sample_idx: int,
    query: str,
    qstruct: Dict[str, Any],
    topk_obj: Dict[str, Any],
    tokenizer,
    model,
    max_output_tokens: int,
    max_core: int,
    max_support: int,
    max_packed: int,
) -> Tuple[str, Dict[str, Any], List[str], str, str]:
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
    )

    raw_text = generate_local_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_output_tokens,
    )

    parsed = parse_summary_json(raw_text)

    return raw_text, parsed, evidence_lines, system_prompt, user_prompt


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # input modes
    ap.add_argument("--sample_idx", type=int, default=None, help="sample idx; with --topk_dir and --query_dir")
    ap.add_argument("--topk_dir", type=str, default="", help="dir containing topk_{idx}.json")
    ap.add_argument("--query_dir", type=str, default="", help="dir containing query_{idx}.json")

    ap.add_argument("--topk_path", type=str, default="", help="explicit single topk path")
    ap.add_argument("--query_path", type=str, default="", help="explicit single query path")

    # model
    ap.add_argument("--model_name", type=str, default=LOCAL_MODEL_NAME)
    ap.add_argument("--tokenizer_name", type=str, default=BASE_TOKENIZER_NAME)
    ap.add_argument("--max_output_tokens", type=int, default=MAX_OUTPUT_TOKENS)

    # evidence limits: EXACTLY same defaults as batch build_split
    ap.add_argument("--max_core", type=int, default=5)
    ap.add_argument("--max_support", type=int, default=5)
    ap.add_argument("--max_packed", type=int, default=12)

    # outputs
    ap.add_argument("--out_dir", type=str, default="", help="write summary_{idx}.json into this dir")
    ap.add_argument("--out_path", type=str, default="", help="explicit output json path")
    ap.add_argument("--save_raw_text", type=str, default="", help="optional path to save raw model text")
    ap.add_argument("--dump_prompt_dir", type=str, default="", help="optional dir to save system/user prompt for debug")
    ap.add_argument("--print_only", action="store_true", help="print parsed JSON only, do not write file")

    args = ap.parse_args()
    validate_optional_paths(args)

    args.model_name = str(required_path(args.model_name, "--model_name / MODEL", "dir"))
    args.tokenizer_name = str(required_path(args.tokenizer_name or args.model_name, "--tokenizer_name / BASE_TOKENIZER", "dir"))

    sample_idx, topk_path, query_path, topk_obj, query_obj, query, qstruct = load_single_sample(
        sample_idx=args.sample_idx,
        topk_path=args.topk_path or None,
        query_path=args.query_path or None,
        topk_dir=args.topk_dir or None,
        query_dir=args.query_dir or None,
    )

    tokenizer, model = load_local_model(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
    )

    raw_text, parsed, evidence_lines, system_prompt, user_prompt = run_single_local(
        sample_idx=sample_idx,
        query=query,
        qstruct=qstruct,
        topk_obj=topk_obj,
        tokenizer=tokenizer,
        model=model,
        max_output_tokens=int(args.max_output_tokens),
        max_core=int(args.max_core),
        max_support=int(args.max_support),
        max_packed=int(args.max_packed),
    )

    if args.save_raw_text:
        raw_path = Path(args.save_raw_text).expanduser()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_text, encoding="utf-8")
        print(f"[INFO] saved raw model text to: {raw_path}")

    if args.dump_prompt_dir:
        dump_dir = Path(args.dump_prompt_dir).expanduser()
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / f"system_{sample_idx}.txt").write_text(system_prompt, encoding="utf-8")
        (dump_dir / f"user_{sample_idx}.txt").write_text(user_prompt, encoding="utf-8")
        (dump_dir / f"evidence_{sample_idx}.txt").write_text("\n".join(evidence_lines), encoding="utf-8")
        print(f"[INFO] dumped prompt/evidence to: {dump_dir}")

    out_obj = {
        "plan_style": parsed["plan_style"],
        "plan": parsed["plan"],
        "final_summary": parsed["final_summary"],
    }

    if args.print_only or (not args.out_dir and not args.out_path):
        print(json.dumps(out_obj, ensure_ascii=False, indent=2))
        return

    if args.out_path:
        out_path = Path(args.out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote {out_path}")
        return

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"summary_{sample_idx}.json"
    final_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {final_path}")


if __name__ == "__main__":
    main()
