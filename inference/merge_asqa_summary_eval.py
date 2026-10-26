#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge SEER A1-A5 summary_{idx}.json into ALCE eval.py input.

Target eval.py requires a top-level {"data": [...]} and, per sample,
question, output, answer (ROUGE), qa_pairs (STR-EM/QA), docs (citations).
Only `output` is replaced by summary.final_summary. The original documents,
QA references, questions, annotations and their ordering are preserved.

Default is a strict full-948 evaluation: missing/invalid predictions cause a
non-zero exit, a diagnostic report is written, and the eval JSON is NOT written.
Use --allow-partial only to diagnose or evaluate a non-comparable subset.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import cli_path

TOPK_VARIANT = {"A1": "A1", "A2": "A2", "A3": "A3", "A4": "A1", "A5": "A3"}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as f:
            tmp = Path(f.name)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def normalized(s: str) -> str:
    return " ".join(s.split())


def source_rows(obj):
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        cfg = obj.get("config")
        if cfg is None:
            cfg = {}
        return obj["data"], cfg
    if isinstance(obj, list):
        return obj, {"dataset": "ASQA", "split": "dev", "retrieval": "GTR top-5"}
    raise ValueError("Source must be a list or a JSON object containing a 'data' list")


def validate_reference(item, i: int):
    if not isinstance(item, dict):
        raise ValueError(f"source[{i}] must be a JSON object")
    if "sample_idx" in item and (type(item["sample_idx"]) is not int or item["sample_idx"] != i):
        raise ValueError(f"source[{i}].sample_idx does not match its row position")
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        raise ValueError(f"source[{i}].question missing/empty")
    if not isinstance(item.get("answer"), str):
        raise ValueError(f"source[{i}].answer missing (required for ROUGE)")
    docs = item.get("docs")
    if not isinstance(docs, list) or len(docs) != 5:
        raise ValueError(f"source[{i}].docs must be the original five GTR documents")
    for j, doc in enumerate(docs):
        if not isinstance(doc, dict) or not isinstance(doc.get("title"), str):
            raise ValueError(f"source[{i}].docs[{j}] missing title")
        body = doc.get("sent") if "sent" in doc else doc.get("text")
        if not isinstance(body, str):
            raise ValueError(f"source[{i}].docs[{j}] missing text/sent")
    pairs = item.get("qa_pairs")
    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError(f"source[{i}].qa_pairs must be a nonempty list")
    for j, pair in enumerate(pairs):
        if not isinstance(pair, dict) or not isinstance(pair.get("question"), str):
            raise ValueError(f"source[{i}].qa_pairs[{j}] missing question")
        answers = pair.get("short_answers")
        if not isinstance(answers, list) or not answers or not all(isinstance(a, str) for a in answers):
            raise ValueError(f"source[{i}].qa_pairs[{j}] missing short_answers")


def validate_prediction(pred, sample_idx: int, item, variant: str):
    if not isinstance(pred, dict):
        return None, "invalid_summary_object"
    if "sample_idx" in pred and (type(pred["sample_idx"]) is not int or pred["sample_idx"] != sample_idx):
        return None, "sample_idx_mismatch"
    if "query" in pred:
        if not isinstance(pred["query"], str) or normalized(pred["query"]) != normalized(item["question"]):
            return None, "question_mismatch"
    answer = pred.get("final_summary")
    if not isinstance(answer, str) or not answer.strip():
        return None, "empty_final_summary"
    answer = answer.strip()
    if "\n" in answer or "\r" in answer:
        # Uploaded ALCE eval.py silently keeps only the first line.
        return None, "multiline_final_summary"
    if "<|im_end|>" in answer:
        return None, "special_token_in_answer"
    for match in re.finditer(r"\[(\d+)", answer):
        citation_id = int(match.group(1))
        if citation_id < 1 or citation_id > len(item["docs"]):
            return None, "citation_out_of_range"
    # No-citation answers remain valid evaluation predictions and score accordingly.

    meta = pred.get("summary_meta")
    if isinstance(meta, dict):
        if meta.get("query_mode") not in (None, "original"):
            return None, "wrong_summary_query_mode"
        expected_plan = variant != "A4"
        if "use_plan" in meta and meta["use_plan"] is not expected_plan:
            return None, "wrong_use_plan"
        topk = meta.get("topk_path")
        if isinstance(topk, str) and topk:
            topk_path = Path(topk)
            if topk_path.name != f"topk_{sample_idx}.json" or topk_path.parent.name != f"topk_{TOPK_VARIANT[variant]}":
                return None, "wrong_topk_source"
    return answer, None


def check_annotations(item, i: int):
    """Report malformed optional annotations: eval.py directly indexes [0] and [1]."""
    annotations = item.get("annotations")
    if annotations is None:
        return False
    if (
        isinstance(annotations, list)
        and len(annotations) >= 2
        and all(isinstance(a, dict) and isinstance(a.get("long_answer"), str) for a in annotations[:2])
    ):
        return False
    # eval.py falls back to 'answer' only if annotations is absent/None.
    # Drop unusable OPTIONAL annotations in the emitted copy, never alter source.
    item.pop("annotations", None)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--variant", choices=tuple(TOPK_VARIANT), default="A1")
    parser.add_argument("--source", type=cli_path, required=True)
    parser.add_argument("--root", type=cli_path, required=True)
    parser.add_argument("--summary-dir", type=cli_path, default=None)
    parser.add_argument("--out", type=cli_path, default=None)
    parser.add_argument("--expected-count", type=int, default=948)
    parser.add_argument("--allow-partial", action="store_true", help="Write only valid rows; NOT comparable to full-dev metrics")
    args = parser.parse_args()

    variant = args.variant
    source = args.source.expanduser().resolve()
    root = args.root.expanduser().resolve()
    summary_dir = (args.summary_dir or root / f"summary_{variant}").expanduser().resolve()
    out = (args.out or root / f"asqa_{variant}_eval.json").expanduser().resolve()
    report_path = out.with_suffix(".merge_report.json")
    skipped_path = out.with_suffix(".skipped_indices.txt")

    if not source.is_file():
        parser.error(f"Source GTR top-5 JSON missing: {source}")
    if not summary_dir.is_dir():
        parser.error(f"Summary directory missing: {summary_dir}")
    protected = {source}
    if out in protected or report_path in protected or skipped_path in protected:
        parser.error("Output paths must not overwrite input JSON")
    if out.parent == summary_dir or report_path.parent == summary_dir or skipped_path.parent == summary_dir:
        parser.error("Place output files outside the summary source directory")

    rows, config = source_rows(read_json(source))
    if args.expected_count > 0 and len(rows) != args.expected_count:
        parser.error(f"Expected {args.expected_count} dev rows, source contains {len(rows)}")

    emitted = []
    skips = []
    reasons = collections.Counter()
    warnings = collections.Counter()
    first_example = {}

    for i, base in enumerate(rows):
        # Invalid reference data is a fatal error, not a failed prediction.
        validate_reference(base, i)
        prediction_path = summary_dir / f"summary_{i}.json"
        if not prediction_path.is_file():
            reason, prediction = "missing_summary", None
        else:
            try:
                prediction = read_json(prediction_path)
                reason = None
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                prediction = None
                reason = "unreadable_summary"
                first_example.setdefault(reason, {"sample_idx": i, "detail": repr(exc)})

        answer = None
        if reason is None:
            answer, reason = validate_prediction(prediction, i, base, variant)

        if reason is not None:
            skips.append(i)
            reasons[reason] += 1
            first_example.setdefault(reason, {"sample_idx": i})
            continue

        item = dict(base)  # preserve source question/docs/gold and original document ordering
        item["output"] = answer  # the ONLY model-prediction field consumed by eval.py
        if check_annotations(item, i):
            warnings["invalid_annotations_fell_back_to_answer"] += 1
        emitted.append(item)

    report = {
        "variant": variant,
        "source": str(source),
        "summary_dir": str(summary_dir),
        "out": str(out),
        "expected_topk": f"topk_{TOPK_VARIANT[variant]}",
        "requested": len(rows),
        "valid": len(emitted),
        "skipped": len(skips),
        "reasons": dict(sorted(reasons.items())),
        "examples": dict(sorted(first_example.items())),
        "warnings": dict(sorted(warnings.items())),
        "allow_partial": args.allow_partial,
        "full_dev_complete": len(emitted) == len(rows),
    }
    atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_text(skipped_path, "".join(f"{i}\n" for i in skips))
    print(f"[CHECK] {variant}: source={len(rows)} valid={len(emitted)} skipped={len(skips)}")
    print(f"[CHECK] reasons={report['reasons']}")
    print(f"[INFO] report: {report_path}")
    print(f"[INFO] skipped indices: {skipped_path}")

    if skips and not args.allow_partial:
        print("[FAIL] Strict full-dev mode: some summaries missing/invalid.", file=sys.stderr)
        print("[FAIL] Eval JSON NOT written/updated. Fix missing outputs or use --allow-partial for diagnostics.", file=sys.stderr)
        return 2
    if not emitted:
        print("[FAIL] No valid predictions; refusing empty eval JSON.", file=sys.stderr)
        return 3
    payload = {"config": config, "data": emitted}
    atomic_text(out, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[OK] Wrote ALCE eval.py input: {out} ({out.stat().st_size / 1024**2:.2f} MiB)")
    if skips:
        print("[WARNING] PARTIAL EVALUATION: scores are on the retained subset, NOT directly comparable to 948-sample results.")
    else:
        print("[PASS] Full 948-sample output ready for eval.py --f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
