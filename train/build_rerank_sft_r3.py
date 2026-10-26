#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Qwen3-8B R3/A4 reranker SFT messages from archived GPT-4o labels.

NO gold QA answers, query_struct, scores, reasons, or synthetic bucket labels are used.
Does not load a model or touch the FSDP training pipeline.

Default policy for post-packing missing labels: labeled_only.
This removes *unlabeled candidates from both input and target*; it never
converts unknown labels to 'drop'. See audit.json before training.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import cli_path

DEFAULT_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_rerank_fsdp.py"

# Same definitions as the revised R3/A4 Student SFT system prompt.
# If the trainer Python is present, extract that constant from its AST
# so the data builder cannot silently diverge from the real training script.
FALLBACK_SYSTEM_PROMPT = """You are an evidence reranker for retrieval-augmented question answering.

Your task is to label the given candidate sentences, NOT to answer the question.
Use ONLY the original question and the candidate sentences. Do not use query
structuring fields, outside knowledge, or gold QA answers.

Prioritize BOTH factual precision and answer coverage:
- Keep evidence for each distinct valid answer branch (for example, a different
  date, entity, time period, geographic scope, version, or interpretation).
- Do not treat complementary facts or different valid answers as duplicates.
- Prefer retaining an additional grounded, useful fact over omitting a valid
  answer branch; do not retain irrelevant sentences just to increase coverage.

Assign exactly one bucket to each candidate:
- "core": directly supports a valid answer to the original question.
- "support": adds a distinct, relevant, verifiable fact, alternative answer,
  qualification, date, or scope that core evidence does not already cover.
  It does not have to independently answer the entire question.
- "drop": irrelevant, weak, fragmentary, context-dependent, near-duplicate,
  unsupported, or unsafe to cite.

Use each candidate's ORIGINAL doc_id and sent_id. Label every input candidate
exactly once. Do not fabricate candidate identifiers or evidence.

Return STRICT JSON only, with this exact output structure:
{"results":[{"doc_id":1,"sent_id":2,"bucket":"core"}]}
The only fields in each result are doc_id, sent_id, and bucket.
Do NOT generate scores, reasons, evidence text, or any other fields."""


def load_system_prompt(train_script: Path) -> str:
    if not train_script.is_file():
        raise FileNotFoundError(
            f"Trainer script not found: {train_script}\n"
            "Use --train-script to point to the UPDATED R3 trainer (never the old one)."
        )
    tree = ast.parse(train_script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_SYSTEM_PROMPT" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise ValueError("DEFAULT_SYSTEM_PROMPT must be a literal string")
            if value != FALLBACK_SYSTEM_PROMPT:
                raise ValueError(
                    "Trainer DEFAULT_SYSTEM_PROMPT is different from the approved R3/A4 "
                    "label-only prompt. Please use train_rerank_fsdp_r3_gold.py or review explicitly."
                )
            return value
    raise ValueError("DEFAULT_SYSTEM_PROMPT not found in the trainer script")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def norm(s: str) -> str:
    return " ".join(s.split())


def require_int(v: Any, field: str) -> int:
    if type(v) is not int:
        raise ValueError(f"{field} is not an integer: {v!r}")
    return v


def key_of(row: dict) -> Tuple[int, int]:
    return require_int(row.get("doc_id"), "doc_id"), require_int(row.get("sent_id"), "sent_id")


def user_prompt(query: str, candidates: List[dict]) -> str:
    # GPT-4o Teacher A4 input layout: CAND_ID / DOC_ID / SENT_ID / TEXT.
    # Teacher's retained coverage principles; NO request for score/reason,
    # because archived labels do not contain reliable score/reason supervision.
    chunks = []
    for n, item in enumerate(candidates, start=1):
        chunks.append(
            f"CAND_ID: {n}\n"
            f"DOC_ID: {item['doc_id']}\n"
            f"SENT_ID: {item['sent_id']}\n"
            f"TEXT: {item['sentence']}"
        )
    return (
        f"Original query:\n{query}\n\n"
        "Task:\n"
        "Assign one bucket (core, support, or drop) to EVERY candidate sentence.\n\n"
        "Rules:\n"
        "1. core: direct answer-bearing evidence, not mere lexical overlap.\n"
        "2. support: a distinct relevant factual contribution, including an alternative "
        "answer, date, qualification, time, or scope; it need not stand alone as the answer.\n"
        "3. drop: irrelevant, duplicate, fragmentary, unsupported, or unsafe evidence.\n"
        "4. Preserve ALL distinct supported answer branches and complementary facts. "
        "Do not mistake different valid answers for duplicates.\n"
        "5. Prefer useful grounded coverage over over-filtering; never keep irrelevant "
        "evidence simply to reach a quota.\n"
        "6. Use original DOC_ID/SENT_ID; output each candidate exactly once.\n\n"
        'Return STRICT JSON only: {"results":[{"doc_id":1,"sent_id":2,"bucket":"core"}]}\n'
        "No score, reason, or sentence text in the JSON response.\n\n"
        "Candidates:\n" + "\n\n".join(chunks)
    )


def build_one(sample_idx: int, root: Path, system: str, incomplete_policy: str):
    cp = read_json(root / "candidate_pool" / f"candidate_pool_{sample_idx}.json")
    rr = read_json(root / "rerank" / f"topk_{sample_idx}.json")

    if cp.get("sample_idx") != sample_idx or rr.get("sample_idx") != sample_idx:
        raise ValueError("sample_idx mismatch between archived files")
    if rr.get("rerank_meta", {}).get("query_mode") != "original":
        raise ValueError("Teacher rerank was not QUERY_MODE=original")
    query = cp.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Empty original query")
    if norm(query) != norm(str(rr.get("query", ""))):
        raise ValueError("Candidate/Rerank original query mismatch")
    pool = cp.get("candidate_pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError("Empty or invalid candidate_pool")
    # Teacher actually fed candidate_pool[:max_input_candidates] into R3.
    teacher_count = rr.get("rerank_meta", {}).get("num_candidates")
    if type(teacher_count) is not int or not 0 < teacher_count <= 18:
        raise ValueError(f"Unexpected rerank_meta.num_candidates: {teacher_count!r}")
    if len(pool) < teacher_count:
        raise ValueError("Archive candidate pool shorter than teacher's candidate count")
    candidates = pool[:teacher_count]
    if len(candidates) == 0:
        raise ValueError("No candidate sentences")

    pool_keys = []
    for c in candidates:
        k = key_of(c)
        sentence = c.get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"Candidate {k} has empty sentence")
        pool_keys.append(k)
    if len(set(pool_keys)) != len(pool_keys):
        raise ValueError("Duplicate candidate (doc_id, sent_id) in candidate pool")

    # Only archived labels from core_evidence/support_evidence/dropped are used.
    # 'packed_evidence' is NOT training supervision (may include fallback drops).
    labels: Dict[Tuple[int, int], str] = {}
    for field, bucket in (
        ("core_evidence", "core"),
        ("support_evidence", "support"),
        ("dropped", "drop"),
    ):
        entries = rr.get(field)
        if not isinstance(entries, list):
            raise ValueError(f"Invalid {field} list")
        for item in entries:
            k = key_of(item)
            if item.get("bucket") != bucket:
                raise ValueError(f"Unexpected bucket for {k} in {field}")
            if k in labels:
                raise ValueError(f"Duplicate/conflicting archived label for {k}")
            labels[k] = bucket

    pool_set = set(pool_keys)
    unexpected = set(labels) - pool_set
    if unexpected:
        raise ValueError(f"Archived labels not in original candidate pool: {sorted(unexpected)[:5]}")
    missing = pool_set - set(labels)
    if missing and incomplete_policy == "skip":
        return None, {"sample_idx": sample_idx, "reason": "incomplete_labels",
                      "candidate_count": len(candidates), "labeled_count": len(labels),
                      "missing": [list(x) for x in sorted(missing)]}
    if missing and incomplete_policy == "error":
        raise ValueError(f"Missing labels for {len(missing)} candidates: {sorted(missing)[:5]}")

    # Under labeled_only, remove UNKNOWN candidates from BOTH input and target.
    # Never invent 'drop' labels. Candidate order stays identical to the source.
    selected = [c for c in candidates if key_of(c) in labels]
    if not selected:
        raise ValueError("No archived labels for any candidate")
    result_rows = [
        {"doc_id": c["doc_id"], "sent_id": c["sent_id"], "bucket": labels[key_of(c)]}
        for c in selected
    ]
    target = json.dumps({"results": result_rows}, ensure_ascii=False, separators=(",", ":"))
    if len(result_rows) != len(selected):
        raise AssertionError("Input/target mismatch")
    example = {
        "dataset": "asqa_r3_a4",
        "sample_idx": sample_idx,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt(query.strip(), selected)},
            {"role": "assistant", "content": target},
        ],
    }
    audit = {
        "sample_idx": sample_idx,
        "candidate_count": len(candidates),
        "labeled_count": len(labels),
        "training_candidate_count": len(selected),
        "missing_count": len(missing),
        "missing": [list(x) for x in sorted(missing)],
        "bucket_counts": dict(Counter(labels[key_of(c)] for c in selected)),
    }
    return example, audit


def group_split(examples: List[dict], val_size: int, seed: int):
    if val_size == 0:
        return examples, []
    if val_size >= len(examples):
        raise ValueError(f"val_size={val_size} >= valid examples={len(examples)}")
    groups = defaultdict(list)
    for ex in examples:
        # Normalize the original query for leakage-resistant grouping.
        # User prompt always starts 'Original query:\n'.
        user = ex["messages"][1]["content"]
        q = user.split("\n\nTask:\n", 1)[0].split("Original query:\n", 1)[1]
        groups[norm(q).casefold()].append(ex)
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)
    picked, count = set(), 0
    for i, group in enumerate(group_list):
        proposed = count + len(group)
        if abs(proposed - val_size) <= abs(count - val_size):
            picked.add(i)
            count = proposed
        if count >= val_size:
            break
    val_ids = {ex["sample_idx"] for i in picked for ex in group_list[i]}
    train = [ex for ex in examples if ex["sample_idx"] not in val_ids]
    val = [ex for ex in examples if ex["sample_idx"] in val_ids]
    if not train or not val:
        raise ValueError("Train/val split empty")
    return train, val


def write_json(path: Path, payload: Any):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-root", type=cli_path, required=True)
    ap.add_argument("--output-dir", type=cli_path, required=True)
    ap.add_argument("--train-script", type=cli_path, default=DEFAULT_TRAIN_SCRIPT)
    ap.add_argument("--val-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-examples", type=int, default=-1,
                    help="Quick dry/smoke check; -1 reads all retained sample indices")
    ap.add_argument("--incomplete-policy", choices=["labeled_only", "skip", "error"],
                    default="labeled_only",
                    help="Handle post-topk candidates without preserved bucket labels")
    ap.add_argument("--no-write", action="store_true", help="Audit and print only")
    args = ap.parse_args()
    root = args.gold_root.expanduser()
    system = load_system_prompt(args.train_script.expanduser())
    filtered = read_json(root / "asqa_train_teacher_s3_filtered.json")
    rows = filtered.get("data") if isinstance(filtered, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Filtered dataset must have top-level {'data': [...]} schema")
    ids = [require_int(r.get("sample_idx"), "sample_idx") for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicated sample_idx in filtered source")
    if args.max_examples >= 0:
        ids = ids[:args.max_examples]
    if args.val_size < 0:
        raise ValueError("val-size must be nonnegative")

    examples, sample_audits, failures = [], [], []
    stats = Counter()
    for idx in ids:
        try:
            example, audit = build_one(idx, root, system, args.incomplete_policy)
            sample_audits.append(audit)
            if example is None:
                stats["skipped_incomplete"] += 1
                continue
            stats["built"] += 1
            stats["pool_candidates"] += audit["candidate_count"]
            stats["training_candidates"] += audit["training_candidate_count"]
            stats["missing_labels"] += audit["missing_count"]
            stats["partial_samples"] += bool(audit["missing_count"])
            stats.update({f"bucket_{k}": v for k, v in audit["bucket_counts"].items()})
            examples.append(example)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            failures.append({"sample_idx": idx, "error": f"{type(exc).__name__}: {exc}"})

    # Fail closed: no silent dropping for schema or index errors.
    if failures:
        print(f"[ERROR] {len(failures)} schema/index failures; first 10:")
        print(json.dumps(failures[:10], ensure_ascii=False, indent=2))
        raise RuntimeError("Fix archived data/assumptions first; no SFT files were written")
    if not examples:
        raise RuntimeError("No SFT examples were constructed")

    train, val = group_split(examples, args.val_size, args.seed)
    train = sorted(train, key=lambda ex: ex["sample_idx"])
    val = sorted(val, key=lambda ex: ex["sample_idx"])
    report = {
        "gold_root": str(root),
        "trainer_prompt_from": str(args.train_script),
        "incomplete_policy": args.incomplete_policy,
        "filtered_rows_considered": len(ids),
        "total_sft": len(examples),
        "train_count": len(train),
        "val_count": len(val),
        "stats": dict(stats),
        "missing_label_sample_indices": [a["sample_idx"] for a in sample_audits if a.get("missing_count", 0)],
        "skipped_incomplete_indices": [a["sample_idx"] for a in sample_audits if a.get("reason") == "incomplete_labels"],
        "notes": [
            "Labels are archived core_evidence/support_evidence/dropped only.",
            "Unlabeled candidates are removed from INPUT AND TARGET under labeled_only.",
            "No fabricated drop labels, scores, reasons, gold QA pairs, or query structuring fields.",
            "Validation is grouped by normalized original query; not ASQA Dev 948.",
            "Token length and response-only masking require separate checks before GPU training.",
        ],
    }
    print(json.dumps({k: v for k, v in report.items() if k not in (
        "missing_label_sample_indices", "skipped_incomplete_indices")}, ensure_ascii=False, indent=2))
    if args.no_write:
        return
    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "asqa_r3_rerank_train.json", {"examples": train})
    write_json(out / "asqa_r3_rerank_val.json", {"examples": val})
    write_json(out / "asqa_r3_rerank_all.json", {"examples": sorted(examples, key=lambda ex: ex["sample_idx"])})
    write_json(out / "asqa_r3_rerank_audit.json", report)
    with (out / "asqa_r3_rerank_sample_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in sample_audits:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[WRITTEN]", out)
    for p in sorted(out.glob("asqa_r3_rerank*")):
        print(" -", p.name, f"({p.stat().st_size:,} bytes)")
    print("\n[PREVIEW] first example")
    first = train[0] if train else val[0]
    print("sample_idx:", first["sample_idx"])
    print("user head:", first["messages"][1]["content"][:400])
    print("assistant head:", first["messages"][2]["content"][:300])


if __name__ == "__main__":
    main()
