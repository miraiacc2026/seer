#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASQA teacher S3 V2 filtering, preserving original records and sample_idx.

Two commands:
  score-citations  Run the *original* ALCE eval.py NLI for each sample (GPU).
  filter           CPU-only staged filtering using that score cache.

Only NEW output files are written. Input JSON, gold data, summary, topk,
heuristic candidate pools, eval.py and its metadata are never modified.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import string
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import configure_paths
CITE = re.compile(r"\[(\d+)\]")
PUNCT = set(string.punctuation)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="." + path.name,
                                         suffix=".tmp", dir=path.parent, delete=False) as fh:
            temp = Path(fh.name)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def dump_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def load_rows(path: Path):
    root = read_json(path)
    if not isinstance(root, dict) or not isinstance(root.get("data"), list):
        raise ValueError("Expected ALCE {config, data} object")
    rows = root["data"]
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("sample_idx"), int):
            raise ValueError(f"Row at position {i} has no ORIGINAL integer sample_idx")
        idx = row["sample_idx"]
        if idx < 0 or idx in seen:
            raise ValueError(f"Negative/duplicate sample_idx={idx}")
        seen.add(idx)
        for field, typ in [("question", str), ("output", str), ("answer", str),
                           ("docs", list), ("qa_pairs", list)]:
            if not isinstance(row.get(field), typ):
                raise ValueError(f"sample_idx={idx} has bad {field}")
    return root, rows


def hash_of_sample(row):
    # Fingerprint protects mapping: DO NOT index citation scores by row position.
    record = {"sample_idx": row["sample_idx"], "sample_id": row.get("sample_id"),
              "question": row["question"], "output": row["output"], "docs": row["docs"]}
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_score_cache(path):
    cache = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                idx = int(rec["sample_idx"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Bad citation score at {path}:{line_no}: {exc}") from exc
            cache[idx] = rec
    return cache


def score_citations(args):
    """Use the exact AutoAIS score semantics of uploaded ALCE eval.py."""
    _, rows = load_rows(args.input_json)
    eval_py = args.eval_py.resolve()
    if not eval_py.is_file():
        raise FileNotFoundError(eval_py)
    eval_hash = hashlib.sha256(eval_py.read_bytes()).hexdigest()
    # Import eval.py lazily: this branch alone requires torch/transformers/GPU.
    sys.path.insert(0, str(eval_py.parent))
    spec = importlib.util.spec_from_file_location("alce_eval_for_teacher_filter", eval_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_autoais_loaded()
    nli_model = str(mod.AUTOAIS_MODEL)
    limit = args.limit if args.limit is not None else len(rows)
    selected = rows[:limit]
    saved = read_score_cache(args.scores_jsonl)
    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    computed = reused = 0
    with args.scores_jsonl.open("a", encoding="utf-8") as out:
        for row in selected:
            idx = row["sample_idx"]
            fingerprint = hash_of_sample(row)
            old = saved.get(idx)
            if old and old.get("source_sha256") == fingerprint and \
                    old.get("eval_sha256") == eval_hash and \
                    old.get("nli_model") == nli_model and \
                    old.get("at_most_citations") == args.at_most_citations and \
                    all(isinstance(old.get(k), (int, float)) for k in ("citation_rec", "citation_prec")):
                reused += 1
                continue
            # eval.py strips after first newline; the merged dataset should already
            # contain only one line. Fail instead of silently scoring different text.
            if "\n" in row["output"] or "\r" in row["output"]:
                raise ValueError(f"Multiline output sample_idx={idx}; will diverge from eval.py")
            normalized = row["output"].strip().replace("<|im_end|>", "")
            score_row = dict(row)
            score_row["output"] = normalized
            precision = recall = 0.0
            if normalized:
                recall, precision = mod.compute_autoais_one_item(
                    score_row, qampari=False, at_most_citations=args.at_most_citations)
            rec = {"sample_idx": idx, "sample_id": row.get("sample_id"),
                   "source_sha256": fingerprint, "eval_sha256": eval_hash,
                   "nli_model": nli_model, "at_most_citations": args.at_most_citations,
                   "citation_rec": float(recall), "citation_prec": float(precision)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()  # checkpoint every example; recover from interrupted Slurm jobs
            saved[idx] = rec
            computed += 1
            if computed % 25 == 0:
                print(f"[CITATIONS] new={computed} reused={reused} last_idx={idx}", flush=True)
    # Compact multiple checkpoint attempts into 1 deterministic line per index.
    clean = [saved[r["sample_idx"]] for r in rows if r["sample_idx"] in saved]
    atomic_write(args.scores_jsonl, "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in clean))
    print(f"[CITATIONS DONE] new={computed} reused={reused} "
          f"scored={len(clean)}/{len(rows)} -> {args.scores_jsonl}")
    if args.limit is None and len(clean) != len(rows):
        raise RuntimeError("Incomplete per-sample citation metrics")


def normalize_answer(text: str) -> str:
    """ALCE-like normalisation, but with word boundaries at matching time."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def has_gold(short_answers, context: str) -> bool:
    body = normalize_answer(context)
    for phrase in short_answers or []:
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        target = normalize_answer(phrase)
        if target and re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", body):
            return True
    return False


def has_gold_any(short_answers, contexts) -> bool:
    """Avoid synthetic matches across candidate/document boundaries."""
    return any(has_gold(short_answers, context) for context in contexts)


def split_sentences(text: str):
    """Try the same NLTK tokenizer as ALCE; fallback if NLTK resource unavailable."""
    try:
        from nltk import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except (ImportError, LookupError):
        # Fallback handles standard S3 'Sentence [doc]. Sentence [doc].' format.
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[\"'(]*[A-Z0-9])", text)
                if s.strip()]


def load_indexed(path: Path, idx: int):
    if not path.is_file():
        return None, "missing_file"
    try:
        obj = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(obj, dict):
        return None, "invalid_object"
    if "sample_idx" in obj and obj["sample_idx"] != idx:
        return None, "wrong_index"
    return obj, None


def text_of_packed(topk):
    return " ".join(str(x.get("sent_content", ""))
                    for x in (topk.get("packed_evidence") or []) if isinstance(x, dict))


def reason_stats(rec, counter):
    for reason in rec["all_flags"]:
        counter[reason] += 1


def harmonic(p, r):
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def citation_aggregate(rows, by_idx):
    """ALCE-style macro P/R + harmonic(mean P, mean R) and mean per-sample F1."""
    vals = [by_idx[row["sample_idx"]] for row in rows]
    if not vals:
        return {"n": 0, "citation_precision": None, "citation_recall": None,
                "citation_f1_from_macro_pr": None, "mean_sample_citation_f1": None}
    p = sum(v["citation_prec"] for v in vals) / len(vals)
    r = sum(v["citation_rec"] for v in vals) / len(vals)
    f = sum(v["citation_f1"] for v in vals) / len(vals)
    return {"n": len(vals), "citation_precision": round(p * 100, 4),
            "citation_recall": round(r * 100, 4),
            "citation_f1_from_macro_pr": round(harmonic(p, r) * 100, 4),
            "mean_sample_citation_f1": round(f * 100, 4)}


def filter_data(args):
    """V2: L1 empty only; L2 sample NLI citation F1; L3 severe omission; L4 hard word cap.

    Diagnostic warnings never become hard exclusions. All retained rows are the
    *original unchanged dictionaries*. NLI scores from unchanged eval.py are reused.
    """
    base, rows = load_rows(args.input_json)
    existing_scores = read_score_cache(args.scores_jsonl)
    selected_scores = {}
    invalid_scores = []
    for row in rows:
        idx = row["sample_idx"]
        score = existing_scores.get(idx)
        if (score is None or score.get("source_sha256") != hash_of_sample(row)
                or score.get("at_most_citations") != args.at_most_citations
                or any(not isinstance(score.get(key), (float, int))
                       or not math.isfinite(float(score[key]))
                       or not 0 <= float(score[key]) <= 1
                       for key in ("citation_prec", "citation_rec"))):
            invalid_scores.append(idx)
        else:
            p, r = float(score["citation_prec"]), float(score["citation_rec"])
            selected_scores[idx] = {"citation_prec": p, "citation_rec": r,
                                    "citation_f1": harmonic(p, r)}
    if invalid_scores:
        raise RuntimeError(
            f"Missing/stale/invalid citation scores for {len(invalid_scores)} samples "
            f"(first {invalid_scores[:15]}). Reuse asqa_teacher_citation_scores.sh "
            "to compute the remaining NLI scores. Refusing unsafe sample alignment."
        )
    if not 0 <= args.min_citation_f1 <= 1:
        raise ValueError("--min-citation-f1 must be in [0,1]")
    if not 0 <= args.max_omission_fraction <= 1:
        raise ValueError("--max-omission-fraction must be in [0,1]")
    if args.max_words <= 0:
        raise ValueError("--max-words must be positive")

    stages = ["structure", "citation", "omission", "length"]
    removed = collections.Counter()
    warnings_all = collections.Counter()
    hard_reasons = collections.Counter()
    kept = []
    diagnostics = []
    for row in rows:
        idx = row["sample_idx"]
        score = selected_scores[idx]
        flags = {stage: [] for stage in stages}
        warnings = []
        summary, err = load_indexed(args.summary_dir / f"summary_{idx}.json", idx)
        topk, topkerr = load_indexed(args.topk_dir / f"topk_{idx}.json", idx)
        pool, poolerr = load_indexed(args.pool_dir / f"candidate_pool_{idx}.json", idx)

        # L1: explicit user choice: ONLY obviously empty teacher output or
        # completely empty retrieved passage evidence gets hard-filtered.
        # Missing auxiliary files / plan mismatches / citations become WARNINGS.
        output = row["output"]
        if not output.strip():
            flags["structure"].append("empty_output")
        doc_texts = [str(d.get("text", "")) for d in row["docs"] if isinstance(d, dict)]
        if not any(s.strip() for s in doc_texts):
            flags["structure"].append("empty_all_docs")
        if 0 < sum(not d.strip() for d in doc_texts) < len(doc_texts):
            warnings.append("some_docs_empty")
        if err:
            warnings.append("summary_" + err)
        if topkerr:
            warnings.append("topk_" + topkerr)
        if poolerr:
            warnings.append("pool_" + poolerr)
        if summary is not None:
            if summary.get("final_summary") != output:
                warnings.append("summary_output_mismatch")
            plan = summary.get("plan")
            if not isinstance(plan, list):
                warnings.append("invalid_plan")
                plan_count = 0
            else:
                plan_count = len(plan)
                if not plan or any(not isinstance(c, dict)
                                   or not isinstance(c.get("claim"), str)
                                   or not c["claim"].strip()
                                   or not isinstance(c.get("citations"), list)
                                   or not c["citations"] for c in plan):
                    warnings.append("invalid_plan")
        else:
            plan_count = 0

        if topk is not None and str(topk.get("query", "")).strip() != row["question"].strip():
            warnings.append("topk_query_mismatch")
        if pool is not None and str(pool.get("query", "")).strip() != row["question"].strip():
            warnings.append("pool_query_mismatch")
        packed = (topk or {}).get("packed_evidence", [])
        if not isinstance(packed, list):
            warnings.append("invalid_packed")
            packed = []
        packed_texts = [str(p.get("sent_content", "")) for p in packed if isinstance(p, dict)]
        pool_rows = (pool or {}).get("candidate_pool", [])
        pool_rows = pool_rows if isinstance(pool_rows, list) else []
        pool_texts = [str(p.get("sentence", "")) for p in pool_rows if isinstance(p, dict)]
        packed_ids = set()
        for item in packed:
            if not isinstance(item, dict):
                warnings.append("invalid_packed_item")
                continue
            try:
                packed_ids.add(int(item["doc_id"]))
            except (ValueError, TypeError, KeyError):
                warnings.append("invalid_packed_doc_id")
        citations = [int(v) for v in CITE.findall(output)]
        if not citations:
            warnings.append("no_citations")
        if any(v < 1 or v > len(row["docs"]) for v in citations):
            warnings.append("citation_out_of_range")
        if any(v not in packed_ids for v in citations):
            warnings.append("citation_not_in_packed")
        if re.search(r"\[[^\]]+\]", CITE.sub("", output)):
            warnings.append("malformed_citation")
        # Text stats for reporting only. ALCE removes bracket citations before
        # splitting words; do not truncate/rewrite a single answer here.
        no_citations_output = CITE.sub("", output)
        words = len(no_citations_output.split())
        sents = split_sentences(output)
        cited_sentences = sum(bool(CITE.search(sent)) for sent in sents)
        if sents and any(not CITE.search(sent) for sent in sents):
            warnings.append("uncited_answer_sentence")
        if plan_count > cited_sentences:
            warnings.append("plan_claims_exceed_cited_sentences")
        if cited_sentences > plan_count + 2:
            warnings.append("cited_sentences_exceed_plan_plus_2")

        qa_pairs = row["qa_pairs"]
        per_pairs = []
        for j, pair in enumerate(qa_pairs):
            short = pair.get("short_answers", []) if isinstance(pair, dict) else []
            per_pairs.append({
                "qa_idx": j,
                "docs": has_gold_any(short, doc_texts),
                "pool": has_gold_any(short, pool_texts),
                "packed": has_gold_any(short, packed_texts),
                "output": has_gold(short, no_citations_output),
            })
        output_gold_not_pool = sum(q["output"] and not q["pool"] for q in per_pairs)
        if output_gold_not_pool:
            warnings.append("gold_in_output_but_not_pool")
        packed_hit = sum(q["packed"] for q in per_pairs)
        missing_packed = sum(q["packed"] and not q["output"] for q in per_pairs)
        omitted_frac = missing_packed / packed_hit if packed_hit else 0.0

        # L2: ONLY the per-sample harmonic citation F1 threshold.
        if score["citation_f1"] < args.min_citation_f1:
            flags["citation"].append("citation_f1_below_threshold")
        # L3: preserve the original >=2 missed, >=50% of packed gold rule.
        if topk is not None and missing_packed >= 2 and omitted_frac >= args.max_omission_fraction:
            flags["omission"].append("missed_at_least_half_packed_gold")
        if topk is None:
            warnings.append("omission_not_auditable_without_topk")
        # L4: delete rather than shorten. No plan-vs-sentence heuristic filtering.
        if words > args.max_words:
            flags["length"].append("too_long")

        active = next((stage for stage in stages if flags[stage]), None)
        rec = {
            "sample_idx": idx, "sample_id": row.get("sample_id"),
            "kept": active is None, "removed_at_stage": active,
            "stage_flags": flags,
            "warning_flags": sorted(set(warnings)),
            "citation_prec": score["citation_prec"],
            "citation_rec": score["citation_rec"],
            "citation_f1": score["citation_f1"],
            "word_count": words, "sentence_count": len(sents),
            "cited_sentence_count": cited_sentences, "plan_claim_count": plan_count,
            "qa_pair_count": len(per_pairs),
            "docs_gold_count": sum(p["docs"] for p in per_pairs),
            "pool_gold_count": sum(p["pool"] for p in per_pairs),
            "packed_gold_count": packed_hit,
            "output_gold_count": sum(p["output"] for p in per_pairs),
            "output_gold_not_pool": output_gold_not_pool,
            "missed_packed_gold_pairs": missing_packed,
            "missed_packed_fraction": omitted_frac,
            "qa_pair_coverage": per_pairs,
        }
        diagnostics.append(rec)
        warnings_all.update(rec["warning_flags"])
        for stage in stages:
            hard_reasons.update(flags[stage])
        if active:
            removed[active] += 1
        else:
            kept.append(row)  # exact original row, NO changes to keys or metadata

    premerged = []
    if args.premerge_skipped_txt.is_file():
        premerged = [int(t.strip()) for t in args.premerge_skipped_txt.read_text().splitlines() if t.strip()]
    all_skipped = sorted(set(premerged) | {r["sample_idx"] for r in diagnostics if not r["kept"]})
    survivors = list(rows)
    stages_report = []
    scores_by_stage = {"before_filter": citation_aggregate(rows, selected_scores)}
    for stage in stages:
        start_count = len(survivors)
        drops = {d["sample_idx"] for d in diagnostics if d["removed_at_stage"] == stage}
        survivors = [row for row in survivors if row["sample_idx"] not in drops]
        scores_by_stage["after_" + stage] = citation_aggregate(survivors, selected_scores)
        stages_report.append({"stage": stage, "input": start_count, "removed": len(drops),
                              "removed_fraction_of_stage": len(drops)/start_count if start_count else 0,
                              "remaining": len(survivors)})
    assert [x["sample_idx"] for x in survivors] == [x["sample_idx"] for x in kept]

    before_mean_words = sum(r["word_count"] for r in diagnostics) / len(diagnostics) if diagnostics else 0
    after_mean_words = sum(r["word_count"] for r in diagnostics if r["kept"]) / len(kept) if kept else 0
    difficult_before = sum(r["qa_pair_count"] >= 4 for r in diagnostics)
    difficult_after = sum(r["qa_pair_count"] >= 4 for r in diagnostics if r["kept"])
    report = {
        "filter_version": "v2-citation-f1-empty-only",
        "input": str(args.input_json), "citations": str(args.scores_jsonl),
        "input_examples": len(rows), "previously_missing_examples": len(premerged),
        "filter_parameters": {
            "min_citation_f1": args.min_citation_f1,
            "at_most_citations": args.at_most_citations,
            "max_omission_fraction": args.max_omission_fraction,
            "max_words": args.max_words,
            "structure_policy": "hard-drop only empty output or all document texts empty",
            "other_structure_and_output_gold_not_pool": "warnings only",
            "length_policy": "hard drop above max_words; no truncation",
        },
        "stages": stages_report,
        "citation_scores_percent": scores_by_stage,
        "removed_sample_idxs_by_stage": {
            stage: [r["sample_idx"] for r in diagnostics if r["removed_at_stage"] == stage]
            for stage in stages
        },
        "remaining": len(kept), "removed_in_filter": len(rows) - len(kept),
        "hard_reason_counts_including_overlap": dict(sorted(hard_reasons.items())),
        "warning_counts": dict(sorted(warnings_all.items())),
        "original_sample_indices_skipped_total": len(all_skipped),
        "mean_words_before": round(before_mean_words, 4),
        "mean_words_after": round(after_mean_words, 4),
        "percent_4plus_gold_before": round(100*difficult_before/len(rows), 3) if rows else 0,
        "percent_4plus_gold_after": round(100*difficult_after/len(kept), 3) if kept else 0,
        "alternative_80_words_would_remove_extra_kept": sum(r["kept"] and r["word_count"]>80 for r in diagnostics),
        "warning_below_min_kept": len(kept) < args.min_kept,
    }
    outputs = [args.out_json, args.audit_jsonl, args.report_json, args.skipped_txt]
    sources = [args.input_json, args.scores_jsonl, args.premerge_skipped_txt]
    if len(set(outputs)) != len(outputs) or set(outputs) & set(sources):
        raise ValueError("Unsafe output collision with source or another output")
    protected_dirs = {args.pool_dir.resolve(), args.topk_dir.resolve(), args.summary_dir.resolve()}
    if any(p.resolve().parent in protected_dirs for p in outputs):
        raise ValueError("Refusing output write in raw teacher data directories")
    if not kept:
        raise RuntimeError("All samples filtered; refusing to write empty JSON")
    atomic_write(args.out_json, dump_json({**base, "data": kept}))
    atomic_write(args.audit_jsonl, "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in diagnostics))
    atomic_write(args.report_json, dump_json(report))
    atomic_write(args.skipped_txt, "".join(f"{idx}\n" for idx in all_skipped))
    print(f"[FILTER V2] input={len(rows)} previous_missing={len(premerged)} f1_threshold={args.min_citation_f1:.2f}")
    for stage in stages_report:
        print(f"[STAGE] {stage['stage']:9} removed={stage['removed']:4} "
              f"remaining={stage['remaining']:4} (stage_drop={stage['removed_fraction_of_stage']:.1%})")
    for label in ("before_filter", "after_citation", "after_length"):
        metrics = scores_by_stage[label]
        print(f"[NLI {label}] N={metrics['n']} "
              f"P={metrics['citation_precision']} R={metrics['citation_recall']} "
              f"F1_macroPR={metrics['citation_f1_from_macro_pr']} "
              f"mean_sample_F1={metrics['mean_sample_citation_f1']}")
    print(f"[DONE] kept={len(kept)} skipped_total_original={len(all_skipped)} "
          f"words_mean={before_mean_words:.2f}->{after_mean_words:.2f} "
          f"4+QA={report['percent_4plus_gold_before']:.2f}%->{report['percent_4plus_gold_after']:.2f}%")
    if report["warning_below_min_kept"]:
        print(f"[WARN] Retained {len(kept)} < target {args.min_kept}; review, no auto-relaxation")
    print(f"[FILES] {args.out_json}\n        {args.audit_jsonl}\n        {args.report_json}\n        {args.skipped_txt}")



def propose_reranker_patches(args):
    """Produce auditable gold-based drop->support proposals, without mutating R3 files.

    Fallback-packed sentences keep their DROP labels (user requirement).
    Actual application to reranker SFT is a separate, explicit step.
    """
    gold = read_json(args.gold_json)
    rows = gold.get("data") if isinstance(gold, dict) else gold
    if not isinstance(rows, list):
        raise ValueError("Gold JSON must contain a list of ASQA Train records")
    patches = []
    skipped_topk = []
    skipped_pool = []
    fallback_protected = 0
    for pos, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Gold train position {pos} malformed")
        idx = row.get("sample_idx", pos)
        if idx != pos:
            raise ValueError("Gold indices must match original file positions")
        topk, err = load_indexed(args.topk_dir / f"topk_{idx}.json", idx)
        pool, perr = load_indexed(args.pool_dir / f"candidate_pool_{idx}.json", idx)
        if err:
            skipped_topk.append(idx)
            continue
        if perr:
            skipped_pool.append(idx)
            continue
        if str(topk.get("query", "")).strip() != str(row.get("question", "")).strip() or \
           str(pool.get("query", "")).strip() != str(row.get("question", "")).strip():
            raise ValueError(f"Mismatch in original query for reranker labels at {idx}")
        pairs = row.get("qa_pairs") or []
        fallback_keys = set()
        for evidence in topk.get("packed_evidence") or []:
            if isinstance(evidence, dict) and evidence.get("source") == "fallback":
                try:
                    fallback_keys.add((int(evidence["doc_id"]), int(evidence["sent_id"])))
                except (KeyError, ValueError, TypeError):
                    continue
        pool_by_key = {}
        for c in pool.get("candidate_pool") or []:
            if isinstance(c, dict):
                try:
                    pool_by_key[int(c["doc_id"]), int(c["sent_id"])] = str(c.get("sentence", ""))
                except (KeyError, ValueError, TypeError):
                    continue
        for dropped in topk.get("dropped") or []:
            if not isinstance(dropped, dict):
                continue
            try:
                key = (int(dropped["doc_id"]), int(dropped["sent_id"]))
            except (KeyError, ValueError, TypeError):
                continue
            if key in fallback_keys:
                fallback_protected += 1
                continue  # Preserve original DROP supervision for fallback-packed rows.
            sentence = str(dropped.get("sent_content", ""))
            if key not in pool_by_key or sentence.strip() != pool_by_key[key].strip():
                continue  # Cannot verify candidate identity; no speculative relabel.
            gold_pairs = [i for i, pair in enumerate(pairs)
                          if isinstance(pair, dict) and has_gold(pair.get("short_answers", []), sentence)]
            if not gold_pairs:
                continue
            patches.append({
                "sample_idx": idx, "sample_id": row.get("sample_id"),
                "doc_id": key[0], "sent_id": key[1], "sentence": sentence,
                "old_bucket": "drop", "proposed_bucket": "support",
                "gold_qa_pair_indices": gold_pairs,
                "match_rule": "normalized exact answer, word boundaries",
            })
    targets = [args.patches_jsonl, args.patch_report_json]
    if len(set(targets)) < 2 or any(p.resolve().parent in
              {args.topk_dir.resolve(), args.pool_dir.resolve()} for p in targets):
        raise ValueError("Unsafe reranker patch output path")
    atomic_write(args.patches_jsonl,
                 "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in patches))
    patch_report = {
        "gold_examples": len(rows), "patches": len(patches),
        "affected_samples": len(set(p["sample_idx"] for p in patches)),
        "protected_fallback_rows": fallback_protected,
        "missing_topk": skipped_topk,
        "missing_pool": skipped_pool,
        "note": "PROPOSALS ONLY: existing topk and teacher labels unchanged; apply explicitly to Reranker SFT later.",
    }
    atomic_write(args.patch_report_json, dump_json(patch_report))
    print(f"[RERANK PATCHES] candidates={len(patches)} affected_samples={patch_report['affected_samples']} "
          f"protected_fallback={fallback_protected}")
    print(f"[FILES] {args.patches_jsonl}\n        {args.patch_report_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    a = sp.add_parser("score-citations", help="Reuse ALCE AutoAIS, no source file modifications")
    a.add_argument("--input-json", type=str, default=None)
    a.add_argument("--eval-py", type=str, default=None)
    a.add_argument("--scores-jsonl", type=str, default=None)
    a.add_argument("--at-most-citations", type=int, default=3)
    a.add_argument("--limit", type=int, default=None, help="For debugging only; do NOT filter from partial scores")
    a.set_defaults(func=score_citations)
    b = sp.add_parser("filter", help="V2: L1 empty only; L2 citation F1; L3 omission; L4 hard word cap")
    b.add_argument("--input-json", type=str, default=None)
    b.add_argument("--scores-jsonl", type=str, default=None)
    b.add_argument("--pool-dir", type=str, default=None)
    b.add_argument("--topk-dir", type=str, default=None)
    b.add_argument("--summary-dir", type=str, default=None)
    b.add_argument("--premerge-skipped-txt", type=str, default=None)
    b.add_argument("--out-json", type=str, default=None, help="Output asqa_train_teacher_s3_filtered.json path")
    b.add_argument("--audit-jsonl", type=str, default=None, help="Output filter_diagnostics.jsonl path")
    b.add_argument("--report-json", type=str, default=None, help="Output filter_report.json path")
    b.add_argument("--skipped-txt", type=str, default=None, help="Output filtered_out_sample_idx.txt path")
    b.add_argument("--min-citation-f1", type=float, default=.70)
    b.add_argument("--at-most-citations", type=int, default=3)
    b.add_argument("--max-omission-fraction", type=float, default=.50)
    b.add_argument("--max-words", type=int, default=100)
    b.add_argument("--min-kept", type=int, default=3500)
    b.set_defaults(func=filter_data)
    c = sp.add_parser("propose-reranker-patches", help="Optional: export gold-based drop->support patch proposals only")
    c.add_argument("--gold-json", type=str, default=None)
    c.add_argument("--pool-dir", type=str, default=None)
    c.add_argument("--topk-dir", type=str, default=None)
    c.add_argument("--patches-jsonl", type=str, default=None, help="Output reranker_gold_support_patches.jsonl path")
    c.add_argument("--patch-report-json", type=str, default=None, help="Output reranker_patch_report.json path")
    c.set_defaults(func=propose_reranker_patches)
    args = ap.parse_args()
    configure_paths(args, ("input_json", "scores_jsonl", "pool_dir", "topk_dir", "summary_dir",
                          "premerge_skipped_txt", "out_json", "audit_jsonl", "report_json", "skipped_txt",
                          "eval_py", "gold_json", "patches_jsonl", "patch_report_json"))
    for attr in ("input_json", "scores_jsonl", "pool_dir", "topk_dir", "summary_dir",
                 "premerge_skipped_txt", "out_json", "audit_jsonl", "report_json", "skipped_txt", "eval_py", "gold_json", "patches_jsonl", "patch_report_json"):
        if hasattr(args, attr):
            setattr(args, attr, getattr(args, attr).expanduser().resolve())
    args.func(args)

if __name__ == "__main__":
    main()
