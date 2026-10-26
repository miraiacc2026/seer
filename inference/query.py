#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, lazy_symbol, required_path, api_client, api_configuration
torch = LazyModule("torch")
OpenAI = lazy_symbol("openai", "OpenAI")
AutoModelForCausalLM = lazy_symbol("transformers", "AutoModelForCausalLM")
AutoTokenizer = lazy_symbol("transformers", "AutoTokenizer")


DOC_RE = re.compile(r"(?m)Document\s*\[(\d+)\]\s*:\s*")
QUESTION_RE = re.compile(r"(?i)Question\s*:\s*(.*)", re.MULTILINE)
WORD_RE = re.compile(r"[a-z0-9]+", re.I)

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


def preprocess_legacy(record: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
    inp = record.get("input", "")
    out = record.get("output", "")

    inp = inp.strip() if isinstance(inp, str) else ""
    out = out.strip() if isinstance(out, str) else ""

    m_q = QUESTION_RE.search(inp)
    query = m_q.group(1).strip() if m_q else ""

    if m_q:
        inp = QUESTION_RE.sub("", inp, count=1).strip()

    documents: List[Dict[str, Any]] = []
    matches = list(DOC_RE.finditer(inp))

    if not matches:
        if inp:
            documents.append({"doc_id": 1, "text": inp})
        return query, documents, out

    for i, m in enumerate(matches):
        doc_id = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(inp)
        text = inp[start:end].strip()
        documents.append({"doc_id": doc_id, "text": text})

    return query, documents, out


def load_records(src: Path) -> List[Dict[str, Any]]:
    with src.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        records = obj["data"]
    elif isinstance(obj, dict) and isinstance(obj.get("examples"), list):
        records = obj["examples"]
    else:
        raise ValueError(f"Unsupported JSON root in {src}")

    if not all(isinstance(x, dict) for x in records):
        raise ValueError("All records must be JSON objects.")
    return records


def extract_query_from_record(record: Dict[str, Any]) -> str:
    for key in ["query", "question"]:
        q = record.get(key)
        if isinstance(q, str) and q.strip():
            return q.strip()

    try:
        q, _, _ = preprocess_legacy(record)
        if q.strip():
            return q.strip()
    except Exception:
        pass

    return ""


# def generate_llama(messages, tokenizer, model, max_new_tokens=650):
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_tensors="pt",
#         return_dict=True,
#     )
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             num_beams=1,
#             temperature=None,
#             top_p=None,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     gen = output_ids[0][inputs["input_ids"].shape[-1]:]
#     return tokenizer.decode(gen, skip_special_tokens=True).strip()

def generate_llama(messages, tokenizer, model, max_new_tokens=650):
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    except TypeError:
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
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()

    # Extra safety for Qwen thinking traces if template fallback ignored enable_thinking.
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    return text

def generate_openai_json(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5.2",
    max_output_tokens: int = 1400,
) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
    )
    text = getattr(resp, "output_text", "") or ""
    if not text.strip():
        raise ValueError("Empty model output")
    return text.strip()


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


def parse_query_once(
    query: str,
    tokenizer=None,
    model=None,
    client: OpenAI = None,
    backend: str = "local",
    openai_model: str = "gpt-5.2",
    max_new_tokens: int = 650,
):
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

    if backend == "openai":
        if client is None:
            raise ValueError("backend='openai' requires an OpenAI client.")
        raw = generate_openai_json(
            client=client,
            system_prompt=system,
            user_prompt=user,
            model=openai_model,
            max_output_tokens=max_new_tokens,
        )
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = generate_llama(messages, tokenizer, model, max_new_tokens=max_new_tokens)

    return safe_parse_json(raw)


def write_query_struct_json(out_path: Path, sample_idx: int, query: str, query_struct: dict) -> None:
    payload = {
        "sample_idx": sample_idx,
        "query": query,
        "query_struct": query_struct,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--src", type=str, default=os.environ.get("SRC_JSON"))
    ap.add_argument("--model_name", type=str, default=os.environ.get("MODEL", ""))
    ap.add_argument("--tokenizer_name", type=str, default=os.environ.get("BASE_TOKENIZER", ""))
    ap.add_argument("--backend", type=str, choices=["local", "openai"], default=os.environ.get("BACKEND", "local"))
    ap.add_argument("--openai_model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-5.2"))
    ap.add_argument("--max_output_tokens", type=int, default=int(os.environ.get("MAX_OUTPUT_TOKENS", "1400")))
    ap.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR"))
    args = ap.parse_args()

    src = required_path(args.src, "--src / SRC_JSON", "file")
    out_dir = required_path(args.out_dir, "--out_dir / OUT_DIR", "output").resolve()
    if args.backend == "local":
        args.model_name = str(required_path(args.model_name, "--model_name / MODEL", "dir"))
        args.tokenizer_name = str(required_path(args.tokenizer_name or args.model_name, "--tokenizer_name / BASE_TOKENIZER", "dir"))
    else:
        api_configuration("openai")
    out_path = out_dir / f"query_{args.sample_idx}.json"

    records = load_records(src)
    if args.sample_idx < 0 or args.sample_idx >= len(records):
        raise IndexError(f"sample_idx={args.sample_idx} out of range (len={len(records)})")

    record = records[args.sample_idx]
    query = extract_query_from_record(record)
    if not query:
        raise ValueError("Failed to extract query from record.")

    print(f"sample_idx={args.sample_idx}")
    print(f"backend={args.backend}")
    print(f"output={out_path}")

    tokenizer = None
    model = None
    client = None

    if args.backend == "openai":
        client = api_client("openai")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else None

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=True,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # model = AutoModelForCausalLM.from_pretrained(
        #     args.model_name,
        #     torch_dtype=dtype,
        #     device_map="auto" if device == "cuda" else None,
        #     local_files_only=True,
        # )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
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

    try:
        raw_struct = parse_query_once(
            query=query,
            tokenizer=tokenizer,
            model=model,
            client=client,
            backend=args.backend,
            openai_model=args.openai_model,
            max_new_tokens=args.max_output_tokens,
        )
        query_struct = post_filter_query_struct(raw_struct, query)
        print("parsed=ok")
    except Exception as e:
        print(f"parsed=fallback ({e})")
        query_struct = build_minimal_query_struct(query)

    write_query_struct_json(out_path, args.sample_idx, query, query_struct)
    print(f"saved={out_path.resolve()}")


if __name__ == "__main__":
    main()
