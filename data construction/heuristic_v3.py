#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a sentence candidate pool in three steps.

1. Prepare sentences with source IDs and optional neighboring context.
2. Score relevance using semantics, query matches and answer/context cues.
3. Select a bounded pool of core, parallel and remaining candidates.

These are heuristic selection groups, not final evidence-role labels.
build_candidate_pool retains the existing range-runner interface. JSON field
names are translated at export, so existing downstream scripts still work.
"""
import math
import os
import re
from typing import Any, Dict, List, Set

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule
pysbd = LazyModule("pysbd")

from qwen_relevance import load_qwen_reranker, score_sent_texts_with_qwen_reranker

CANDIDATE_MODES = ("full", "no_structure")
BUDGET_MODES = ("legacy", "fixed")


def build_candidate_pool(
    docs_obj: Dict[str, Any],
    q_obj: Dict[str, Any],
    reranker_bundle,
    max_candidates: int = 120,
    final_slots: int = 18,
    min_core: int = 5,
    min_parallel: int = 8,
    core_doc_cap: int = 3,
    batch_size: int = 64,
    max_input_passages: int = 8,
    sim_window: int = 1,
    survival_min: float = 0.18,
    noise_for_drop: float = 0.35,
    min_total_keep: float = 0.05,
    final_topk_passages: int = 5,
    candidate_mode: str = "full",
    budget_mode: str = "legacy",
):
    """Prepare -> score -> select. The legacy min_* names mean quotas, not minima.

    core_doc_cap is shared by core and parallel; drop has no document cap.
    full + legacy preserves the previous algorithm. no_structure disables all
    query-structure signals, including the entity-mismatch penalty.
    fixed uses the same unique-sentence budget for both modes; score filters
    become preferences when more sentences are needed to fill that budget.
    """
    if candidate_mode not in CANDIDATE_MODES:
        raise ValueError(f"candidate_mode must be one of {CANDIDATE_MODES}")
    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode must be one of {BUDGET_MODES}")
    query = clean_text(docs_obj.get("query", ""))
    # None means the entire structured signal is disabled, not an empty entity
    # list (which would still activate the old mismatch penalty).
    signals = query_signals(q_obj["query_struct"]) if candidate_mode == "full" else None
    documents = docs_obj.get("documents", [])
    if max_input_passages and max_input_passages > 0:
        documents = documents[:max_input_passages]
    fixed = budget_mode == "fixed"
    if fixed:
        if min(final_slots, min_core, min_parallel, max_candidates) < 0:
            raise ValueError("fixed budget requires non-negative sizes and quotas")
        if core_doc_cap is not None and core_doc_cap < 0:
            raise ValueError("fixed budget requires a non-negative document cap")
        if max_candidates < final_slots:
            raise ValueError("fixed budget requires CANDIDATES >= FINAL_SLOTS")
        if final_topk_passages < len(documents):
            raise ValueError(
                "fixed budget must retain all input passages so both modes use the same universe; "
                "use MAX_INPUT_PASSAGES=5 FINAL_TOPK_PASSAGES=5"
            )

    sentences = prepare_sentences(documents, sim_window)
    all_ranked = score_sentences(
        sentences, query, signals, reranker_bundle, batch_size,
        min_total_keep, survival_min, noise_for_drop, retain_filtered=fixed,
    )
    ranked = [row for row in all_ranked if passes_score_filters(
        row["scores"], min_total_keep, survival_min, noise_for_drop,
    )] if fixed else all_ranked
    passages = rank_passages(all_ranked if fixed else ranked, documents)
    selected_passages = passages[:final_topk_passages]
    selected_ids = {p["doc_id"] for p in selected_passages}
    eligible = [row for row in ranked if row["doc_id"] in selected_ids][:max_candidates]
    if fixed:
        candidates, budget = select_fixed_candidates(
            eligible, all_ranked, final_slots, min_core, min_parallel, core_doc_cap,
        )
    else:
        candidates = select_candidates(
            eligible, pool_size=final_slots, core_quota=min_core,
            parallel_quota=min_parallel, document_cap=core_doc_cap,
        )
    result = {
        "working_topk_passages": passages,
        "topk_passages": selected_passages,
        "candidate_pool": candidates,
    }
    if fixed:
        result["candidate_budget"] = budget
    return result


# 1. Prepare sentences. Preserve their original IDs even when noise is removed.
def prepare_sentences(documents, context_window):
    rows = []
    for document in documents:
        sentences = sent_split(clean_text(document.get("doc_text", "")))
        for sentence_id, text in enumerate(sentences):
            sentence = clean_text(text)
            if should_hard_drop(sentence):
                continue
            rows.append({
                "doc_id": int(document.get("doc_id", -1)),
                "sent_id": sentence_id,
                "sentence": sentence,
                "context": make_context_window(sentences, sentence_id, context_window) or sentence,
            })
    return rows


# 2. Score sentences. Model relevance uses the original query, as before.
def score_sentences(rows, query, signals, reranker, batch_size,
                    min_score, min_support, noisy_threshold, retain_filtered=False):
    if not rows:
        return []
    semantics = score_sent_texts_with_qwen_reranker(
        query=query,
        sent_texts=[row["context"] for row in rows],
        tokenizer=reranker["tokenizer"], model=reranker["model"],
        batch_size=batch_size,
        max_length=int(os.environ.get("RERANK_MAX_LENGTH", "1024")),
        instruction=os.environ.get(
            "RERANK_INSTRUCTION",
            "Given a web search query, retrieve relevant passages that answer the query",
        ),
        prefix_tokens=reranker["prefix_tokens"], suffix_tokens=reranker["suffix_tokens"],
        token_true_id=reranker["token_true_id"], token_false_id=reranker["token_false_id"],
        device=reranker["device"],
    )
    ranked = []
    for row, semantic in zip(rows, semantics):
        scores = sentence_scores(row["sentence"], query, signals, float(semantic))
        if not retain_filtered and not passes_score_filters(scores, min_score, min_support, noisy_threshold):
            continue
        ranked.append({**row, "scores": scores})
    return sorted(ranked, key=lambda row: row["scores"]["relevance"], reverse=True)


def passes_score_filters(scores, min_score, min_support, noisy_threshold):
    return not (
        scores["relevance"] < min_score
        or (scores["support"] < min_support and scores["noise"] >= noisy_threshold)
    )


def sentence_scores(sentence, query, signals, semantic):
    """One relevance score and two selection priorities; preserve arithmetic order."""
    if signals is None:
        entity = term = relation = scope = 0.0
        context_cues = context_cue_score(sentence, [], [], [])
        mismatch = 0.0
    else:
        entity = entity_match(sentence, signals["aliases"])
        term = term_match(sentence, signals["terms"])
        relation = term_match(sentence, signals["relations"])
        scope = scope_match(sentence, signals["scope"])
        context_cues = context_cue_score(
            sentence, signals["scope"], signals["context_terms"], signals["answer_terms"],
        )
        mismatch = entity_mismatch_penalty(sentence, entity, semantic, relation)
    relation_or_term = max(relation, 0.7 * term)
    agreement = (
        math.sqrt(max(0.0, entity * relation_or_term))
        if entity > 0 and relation_or_term > 0 else 0.0
    )
    answer_form = answer_form_score(sentence)
    answer_cues = answer_cue_score(sentence, query)
    noise = noise_penalty(sentence)

    relevance = (
        0.26 * semantic
        + 0.16 * entity
        + 0.20 * relation
        + 0.10 * agreement
        + 0.10 * scope
        + 0.08 * term
        + 0.04 * answer_cues
        + 0.02 * context_cues
        + 0.01 * answer_form
        - 0.05 * noise
        - 0.08 * mismatch
    )
    relevance = max(0.0, min(1.0, float(relevance)))

    # Core prioritizes relation/scope matches; parallel preserves answer or context cues.
    core = 0.50 * relevance + 0.22 * relation + 0.14 * scope + 0.10 * agreement + 0.04 * answer_form
    answer_priority = (
        0.46 * relevance + 0.16 * answer_cues + 0.16 * entity
        + 0.06 * answer_form + 0.08 * semantic + 0.08 * relation
    )
    context_priority = (
        0.46 * relevance + 0.28 * context_cues + 0.16 * scope
        + 0.06 * semantic + 0.04 * relation
    )
    parallel = max(answer_priority, context_priority)
    return {
        "entity": float(entity), "term": float(term), "relation": float(relation),
        "scope": float(scope), "semantic": float(semantic), "agreement": float(agreement),
        "answer_form": float(answer_form), "answer_cues": float(answer_cues),
        "context_cues": float(context_cues), "noise": float(noise),
        "entity_mismatch": float(mismatch), "relevance": float(relevance),
        "support": float(max(core, parallel)), "core": float(core), "parallel": float(parallel),
    }


# 3. Rank passages and select candidates under the existing budgets.
def rank_passages(ranked, documents):
    if not ranked:
        return []
    by_document = {}
    for row in ranked:
        by_document.setdefault(row["doc_id"], []).append(row)
    texts = {int(d["doc_id"]): clean_text(d.get("doc_text", "")) for d in documents}
    passages = []
    for document_id, rows in by_document.items():
        # rows already follow the global stable relevance ordering.
        best = rows[0]
        core = max(row["scores"]["core"] for row in rows)
        parallel = max(row["scores"]["parallel"] for row in rows)
        top_two = [row["scores"]["relevance"] for row in rows[:2]]
        mean_top_two = sum(top_two) / len(top_two)
        support_count = sum(row["scores"]["relevance"] >= 0.25 for row in rows)
        support_bonus = min(1.0, support_count / 3.0)
        score = 0.38 * core + 0.34 * parallel + 0.18 * mean_top_two + 0.10 * support_bonus
        passages.append({
            "doc_id": document_id, "final_doc_score": float(score),
            "max_core": core, "max_parallel": parallel,
            "mean_top2_total": float(mean_top_two), "support_count": support_count,
            "support_bonus": float(support_bonus),
            "top_sentence": {
                "doc_id": best["doc_id"], "sent_id": best["sent_id"],
                "sentence": clean_text(best["sentence"]),
                "score": export_scores(best["scores"]),
                "core_score": best["scores"]["core"],
                "parallel_score": best["scores"]["parallel"],
            },
            "doc_text": texts.get(document_id, ""),
        })
    keys = ("final_doc_score", "max_core", "max_parallel", "mean_top2_total", "support_bonus")
    return sorted(passages, key=lambda p: tuple(p[k] for k in keys), reverse=True)


def select_candidates(rows, pool_size, core_quota, parallel_quota, document_cap,
                      unique_budget=False):
    """Select core, parallel, then remaining rows. Legacy deduplicates text last."""
    selected, used, per_document = [], set(), {}
    seen_during_selection = set()
    for group, priority, quota, cap in (
        ("core", "core", core_quota, document_cap),
        ("parallel", "parallel", parallel_quota, document_cap),
        ("drop", "relevance", None, None),
    ):
        budget = max(0, pool_size - len(selected)) if quota is None else quota
        if unique_budget:
            budget = min(budget, max(0, pool_size - len(selected)))
        ordered = sorted(
            rows,
            key=lambda row: (row["scores"][priority], row["scores"]["relevance"]),
            reverse=True,
        )
        taken = 0
        for row in ordered:
            if taken >= budget:
                break
            document_id = row["doc_id"]
            text_key = _norm(clean_text(row["sentence"]))
            if unique_budget and (not text_key or text_key in seen_during_selection):
                continue
            key = f'{document_id}:{row["sent_id"]}:{_norm(row["sentence"])[:80]}'
            if key in used or (cap is not None and per_document.get(document_id, 0) >= cap):
                continue
            selected.append({**row, "candidate_type": group})
            seen_during_selection.add(text_key)
            used.add(key)
            per_document[document_id] = per_document.get(document_id, 0) + 1
            taken += 1

    candidates, seen_text = [], set()
    for row in selected:
        sentence = clean_text(row["sentence"])
        key = _norm(sentence)
        if not sentence or key in seen_text:
            continue
        seen_text.add(key)
        candidates.append(export_candidate(row, sentence))
        if len(candidates) >= pool_size:
            break
    return candidates


def select_fixed_candidates(eligible, all_ranked, pool_size, core_quota,
                            parallel_quota, document_cap):
    """Common budget rule for A1/A2: unique slots, then ranked refill if needed.

    Refill uses only the same cleaned input sentences and the active mode's
    scores. It never adds query-structure signals back to no_structure.
    """
    available = len({_norm(clean_text(row["sentence"])) for row in all_ranked
                     if clean_text(row["sentence"])})
    target = min(pool_size, available)
    candidates = select_candidates(
        eligible, target, core_quota, parallel_quota, document_cap, unique_budget=True,
    )
    initial_count = len(candidates)
    used = {_norm(row["sentence"]) for row in candidates}
    for row in all_ranked:
        if len(candidates) >= target:
            break
        sentence = clean_text(row["sentence"])
        key = _norm(sentence)
        if not key or key in used:
            continue
        candidates.append(export_candidate({**row, "candidate_type": "drop"}, sentence))
        used.add(key)
    if len(candidates) != target:
        raise RuntimeError(f"Could not fill candidate budget: target={target}, actual={len(candidates)}")
    return candidates, {
        "requested_count": pool_size,
        "available_unique_sentences": available,
        "target_count": target,
        "actual_count": len(candidates),
        "refill_count": len(candidates) - initial_count,
    }


# Only the export boundary retains the old diagnostic names.
_SCORE_NAMES = {
    "entity": "entity", "term": "term", "relation": "relation", "scope": "scope",
    "semantic": "sim", "agreement": "concurrence", "answer_form": "answer_shape",
    "answer_cues": "answer_unit", "context_cues": "clarification",
    "noise": "noise_penalty", "entity_mismatch": "entity_conflict_penalty",
    "relevance": "total", "support": "survival_score",
}


def export_scores(scores):
    return {old_name: scores[name] for name, old_name in _SCORE_NAMES.items()}


def export_candidate(row, sentence):
    scores = row["scores"]
    names = (
        ("total", "relevance"), ("sim", "semantic"), ("entity", "entity"),
        ("relation", "relation"), ("scope", "scope"), ("term", "term"),
        ("answer_unit", "answer_cues"), ("clarification", "context_cues"),
        ("core_score", "core"), ("parallel_score", "parallel"),
    )
    return {
        "doc_id": row["doc_id"], "sent_id": row["sent_id"],
        "candidate_type": row["candidate_type"], "sentence": sentence,
        "score": {output: scores[internal] for output, internal in names},
    }


# Query signals: only fields that actually affect the original scorer.
def query_signals(structure):
    entities = structure.get("anchor_entities") or structure.get("core_entities") or []
    aliases = structure.get("anchor_aliases") or structure.get("entity_aliases") or {}
    terms = structure.get("anchor_terms") or []
    attributes = [a for a in structure.get("attributes") or [] if isinstance(a, dict)]
    control = structure.get("retrieval_control") or {}

    relations = []
    for key in ("relation_terms", "relation_synonym", "relation_expansion", "relation_markers"):
        relations.extend(structure.get(key) or [])
    for attribute in attributes:
        relations.extend(attribute.get("bridge_triggers") or [])
        relations.extend(attribute.get("hypothesis_triggers") or [])
        phrase = attribute.get("attribute", "")
        if isinstance(phrase, str) and phrase.strip():
            relations.append(phrase)
    for key in ("relation_markers", "answerability_markers", "parallel_cues"):
        relations.extend(control.get(key) or [])
    scope = list(structure.get("scope_terms") or [])
    for attribute in attributes:
        scope.extend(attribute.get("time_or_scope") or [])

    return {
        "aliases": build_alias_map(entities, aliases),
        "terms": _dedup_terms(terms),
        "relations": _dedup_terms(relations) or _dedup_terms(terms),
        "scope": expand_scope_phrases(_dedup_terms(scope)),
        "context_terms": _dedup_terms(control.get("parallel_cues") or []),
        "answer_terms": _dedup_terms(control.get("answerability_markers") or []),
    }


# Text rules below preserve the original matching and normalization behavior.
_SEG = None
_BULLET_RE = re.compile(r"[•●▪◦·]+")
DOC_RE = re.compile(r"(?m)Document\s*\[(\d+)\]\s*:\s*")
QUESTION_RE = re.compile(r"(?i)Question\s*:\s*(.*)", re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")
_DIGIT_RE = re.compile(r"\d")
TITLE_CASE_SPAN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b")
QUOTED_TITLE_RE = re.compile(r'"[^"]+"|“[^”]+”|\'[^\']+\'')
NOISE_RE = re.compile(
    r"\b("
    r"vote|polls|nominees|survey|ranking|retrieved:|http|https|continue reading|"
    r"tagged|categories|cookie|newsletter|subscribe|sign up|advertisement|"
    r"read more|watch video|external links|all rights reserved|copyright|"
    r"share this|login|register"
    r")\b",
    re.IGNORECASE
)
COPULA_RE = re.compile(r"\b(is|was|were|are)\b", re.IGNORECASE)
APPOSITIVE_RE = re.compile(r",\s+(an?|the)\s+[^,]{1,60},")
COMPARE_RE = re.compile(r"\b(than|compared with|compared to|unlike|whereas|while|however|but)\b", re.IGNORECASE)
TRANSITION_RE = re.compile(r"\b(however|therefore|thus|meanwhile|later|then|in addition|as a result|alternatively|according to)\b", re.IGNORECASE)
ROLE_RE = re.compile(r"\b(actor|singer|writer|author|director|president|governor|mayor|founder|inventor|scientist|novel|film|album|song|series|company|organization|city|country)\b", re.IGNORECASE)


def preprocess(record):
    if not isinstance(record, dict):
        return "", [], ""

    query = ""
    for key in ["query", "question"]:
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            query = v.strip()
            break

    documents = []

    docs = record.get("docs")
    if not isinstance(docs, list):
        docs = record.get("passages")
    if not isinstance(docs, list):
        docs = []

    if docs:
        for i, d in enumerate(docs, start=1):
            if not isinstance(d, dict):
                continue
            text = (d.get("text") or d.get("content") or d.get("passage") or "").strip()
            if not text:
                continue
            documents.append({
                "doc_id": i,
                "text": text,
            })
        return query, documents, record.get("answer", "")

    inp = record.get("input", "").strip()
    out = record.get("output", "").strip()

    m_q = QUESTION_RE.search(inp)
    if m_q and not query:
        query = m_q.group(1).strip()

    if m_q:
        inp = QUESTION_RE.sub("", inp, count=1).strip()

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

def sent_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = _BULLET_RE.sub(". ", text)
    global _SEG
    if _SEG is None:
        _SEG = pysbd.Segmenter(language="en", clean=False)
    sents = _SEG.segment(text)
    return [s.strip() for s in sents if s and s.strip()]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def tokenize(text: str) -> List[str]:
    return [x.lower() for x in _WORD_RE.findall(text or "")]

def strip_leading_meta(s: str) -> str:
    s = (s or "").strip()
    head = s[:120]
    if "|" in head and not re.search(r"[.?!]", head):
        s = s.split("|")[-1].strip()
    return s

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\bTags:\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\|\s*[^.?!]{0,80}$", "", s)
    s = strip_leading_meta(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_digit(s: str) -> bool:
    return bool(_DIGIT_RE.search(s or ""))

def extract_years(s: str) -> Set[str]:
    return set(_YEAR_RE.findall(s or ""))

def is_pure_meta_noise(s: str) -> bool:
    s = clean_text(s)
    if not s:
        return True
    if NOISE_RE.search(s):
        return True
    if len(s) < 80 and s.isupper():
        return True
    if len(s) < 70 and s.endswith("?") and s.count(" ") <= 8:
        return True
    return False

def should_hard_drop(sentence: str) -> bool:
    s = clean_text(sentence)
    if not s:
        return True
    if is_pure_meta_noise(s):
        return True
    return False

def _destutter(w: str) -> str:
    if len(w) >= 2 and w[-1] == w[-2]:
        return w[:-1]
    return w

def stem(w: str) -> str:
    w = (w or "").lower()
    if len(w) <= 3:
        return w
    if len(w) > 5 and w.endswith("ing"):
        w = _destutter(w[:-3])
    elif len(w) > 4 and w.endswith("ed"):
        w = _destutter(w[:-2])
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w

def stem_set_from_text(text: str):
    return {stem(t) for t in tokenize(text) if t}

def _dedup_terms(terms):
    out, seen = [], set()
    for t in terms or []:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        k = _norm(tt)
        if k in seen:
            continue
        seen.add(k)
        out.append(tt)
    return out

def build_alias_map(anchor_entities, anchor_aliases):
    alias_map = {}
    for ent in anchor_entities or []:
        forms = set()
        if isinstance(ent, str) and ent.strip():
            forms.add(ent.strip())
        al = (anchor_aliases or {}).get(ent, [])
        if isinstance(al, list):
            for a in al:
                if isinstance(a, str) and a.strip():
                    forms.add(a.strip())
        alias_map[ent] = forms
    return alias_map

def _is_subseq(haystack_tokens, needle_tokens) -> bool:
    if not needle_tokens:
        return False
    n = len(needle_tokens)
    if n == 1:
        return needle_tokens[0] in set(haystack_tokens)
    for i in range(0, len(haystack_tokens) - n + 1):
        if haystack_tokens[i:i + n] == needle_tokens:
            return True
    return False

def _variant_weight(canonical: str, variant: str) -> float:
    can_toks = tokenize(canonical)
    var_toks = tokenize(variant)
    if not can_toks or not var_toks:
        return 0.0
    if variant.strip().lower() == canonical.strip().lower():
        return 1.0
    w = len(var_toks) / max(1, len(can_toks))
    return max(0.5, min(1.0, w))

def entity_match(sentence: str, alias_map) -> float:
    if not alias_map:
        return 0.0

    w_tokens = tokenize(sentence)
    if not w_tokens:
        return 0.0

    total = len(alias_map)
    acc = 0.0

    for ent, forms in alias_map.items():
        best = 0.0
        candidates = [ent] + list(forms or [])
        for v in candidates:
            if not isinstance(v, str) or not v.strip():
                continue
            v_tokens = tokenize(v)
            if not v_tokens:
                continue
            if _is_subseq(w_tokens, v_tokens):
                best = max(best, _variant_weight(ent, v))
                if best >= 1.0:
                    break
        acc += best

    return acc / max(1, total)

def term_match(sentence: str, terms) -> float:
    if not terms:
        return 0.0

    s = clean_text(sentence)
    if not s:
        return 0.0

    s_norm = _norm(s)
    s_stems = stem_set_from_text(s)

    hits = 0
    uniq = _dedup_terms(terms)
    for t in uniq:
        if " " in t:
            if _norm(t) in s_norm:
                hits += 1
        else:
            if stem(t) in s_stems:
                hits += 1

    if hits <= 0:
        return 0.0
    if hits == 1:
        return 0.6
    return 1.0

def expand_scope_phrases(scopes):
    out = []
    for s in scopes or []:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s:
            continue
        out.append(s)

        m = re.match(r"end of (.+)", s, flags=re.IGNORECASE)
        if m:
            x = m.group(1).strip()
            if x:
                out.extend([f"at the end of {x}", f"after {x}", f"following {x}", f"by the end of {x}"])

        m = re.match(r"after (.+)", s, flags=re.IGNORECASE)
        if m:
            x = m.group(1).strip()
            if x:
                out.extend([f"following {x}", f"post {x}"])

        m = re.match(r"before (.+)", s, flags=re.IGNORECASE)
        if m:
            x = m.group(1).strip()
            if x:
                out.extend([f"prior to {x}", f"preceding {x}"])

    seen = set()
    dedup = []
    for s in out:
        ns = _norm(s)
        if ns in seen:
            continue
        seen.add(ns)
        dedup.append(s)
    return dedup

def scope_match(sentence: str, scope_terms) -> float:
    if not scope_terms:
        return 0.0

    s = clean_text(sentence)
    if not s:
        return 0.0

    s_norm = _norm(s)
    s_stems = stem_set_from_text(s)
    s_years = extract_years(s)

    best = 0.0
    for p in scope_terms:
        if not isinstance(p, str) or not p.strip():
            continue

        p_norm = _norm(p)
        if p_norm in s_norm:
            best = max(best, 1.0)
            continue

        p_toks = tokenize(p)
        if not p_toks:
            continue

        p_stems = {stem(t) for t in p_toks}
        overlap = len(p_stems & s_stems) / max(1, len(p_stems))
        if overlap >= 0.8:
            best = max(best, 0.8)
        elif overlap >= 0.5:
            best = max(best, 0.6)

        p_years = extract_years(p)
        if p_years and s_years and (p_years & s_years):
            best = max(best, 0.7)

    return min(1.0, best)

def _cue_sum(weighted_matches):
    score = 0.0
    for weight, matched in weighted_matches:
        if matched:
            score += weight
    return min(1.0, score)


def answer_form_score(sentence):
    """Surface form of a possible answer: numbers, definitions and role words."""
    s = clean_text(sentence)
    if not s:
        return 0.0
    return _cue_sum([
        (0.20, has_digit(s)),
        (0.20, COPULA_RE.search(s)),
        (0.20, APPOSITIVE_RE.search(s)),
        (0.10, COMPARE_RE.search(s)),
        (0.15, extract_years(s)),
        (0.15, ROLE_RE.search(s)),
    ])


def answer_cue_score(sentence, query):
    """Potential answer content, with a small bonus for the question's answer type."""
    s, q = clean_text(sentence), clean_text(query).lower()
    if not s:
        return 0.0
    return _cue_sum([
        (0.20, extract_years(s)),
        (0.08, has_digit(s)),
        (0.18, APPOSITIVE_RE.search(s)),
        (0.08, COPULA_RE.search(s)),
        (0.18, TITLE_CASE_SPAN_RE.search(sentence)),
        (0.15, QUOTED_TITLE_RE.search(sentence)),
        (0.16, ROLE_RE.search(s)),
        (0.10, q.startswith(("who", "whose", "what is the name of", "what was the name of"))
         and TITLE_CASE_SPAN_RE.search(sentence)),
        (0.12, q.startswith(("when", "what year", "what date")) and extract_years(s)),
        (0.12, q.startswith("where") and re.search(r"\b(in|at|from|near)\s+[A-Z][a-z]+", sentence)),
        (0.10, q.startswith(("which", "what film", "what book", "what album", "what song"))
         and (QUOTED_TITLE_RE.search(sentence) or TITLE_CASE_SPAN_RE.search(sentence))),
    ])


def context_cue_score(sentence, scope_terms, context_terms, answer_terms):
    """Comparisons, transitions, scope qualifications and source/authority cues."""
    s = clean_text(sentence)
    if not s:
        return 0.0
    return _cue_sum([
        (0.25, COMPARE_RE.search(s)),
        (0.20, TRANSITION_RE.search(s)),
        (0.25, scope_match(s, scope_terms) > 0),
        (0.15, term_match(s, context_terms) > 0),
        (0.10, term_match(s, answer_terms) > 0),
        (0.20, re.search(r"\baccording to\b|\bunder\b|\bin official\b|\brecognized by\b", s, flags=re.I)),
    ])


def entity_mismatch_penalty(sentence: str, entity_sc: float, sim_sc: float, relation_sc: float) -> float:
    s = clean_text(sentence)
    if not s:
        return 0.0

    penalty = 0.0
    num_title_spans = len(TITLE_CASE_SPAN_RE.findall(sentence))

    if entity_sc <= 0.0 and sim_sc >= 0.55 and relation_sc >= 0.25 and num_title_spans >= 2:
        penalty += 0.25
    if entity_sc <= 0.0 and sim_sc >= 0.65 and num_title_spans >= 3:
        penalty += 0.20

    return min(0.5, penalty)

def noise_penalty(sentence: str) -> float:
    s = clean_text(sentence)
    if not s:
        return 1.0

    pen = 0.0
    if NOISE_RE.search(s):
        pen += 0.60
    if len(s) < 80 and s.isupper():
        pen += 0.30
    if len(s.split()) <= 3 and not has_digit(s):
        pen += 0.20
    return min(1.0, pen)

def make_context_window(sents: List[str], idx: int, window: int = 1) -> str:
    parts = []
    left = max(0, idx - window)
    right = min(len(sents), idx + window + 1)
    for j in range(left, right):
        sj = clean_text(sents[j])
        if sj:
            parts.append(sj)
    return " ".join(parts).strip()

if __name__ == "__main__":
    # Keep the original one-sample entry point; I/O lives in the range runner.
    from heuristic_range_v3 import main
    raise SystemExit(main(single=True))
