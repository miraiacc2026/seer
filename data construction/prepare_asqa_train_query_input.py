#!/usr/bin/env python3
"""Unwrap a 4,353-row ASQA train export for asqa_query_batch.py.

The current query pipeline accepts a top-level list of records with `question`
and `docs`. This script keeps every record byte-semantically unchanged (including
QA gold and source document ordering). It saves the original wrapped JSON and
its `config` before optionally replacing the original file atomically.

Default: write a separate *_query_ready.json (safest).
Use --inplace to update the original path after making an immutable backup.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import cli_path

EXPECTED_ITEMS = 4353


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def atomic_json_write(dest: Path, obj) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def audit(records):
    if not isinstance(records, list):
        raise ValueError("root.data must be a list")
    if len(records) != EXPECTED_ITEMS:
        raise ValueError(f"Expected {EXPECTED_ITEMS} ASQA train records, got {len(records)}")

    docs_counts = Counter()
    qa_counts = Counter()
    errors = []
    qa_gold_missing = 0
    sample_ids = []
    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            errors.append(f"row {idx}: not a dictionary")
            continue
        question = row.get("question")
        docs = row.get("docs")
        qa_pairs = row.get("qa_pairs")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"row {idx}: missing/empty question")
        if not isinstance(docs, list):
            errors.append(f"row {idx}: docs not a list")
        else:
            docs_counts[len(docs)] += 1
            if len(docs) != 5:
                errors.append(f"row {idx}: expected 5 docs, got {len(docs)}")
            for j, doc in enumerate(docs):
                if not isinstance(doc, dict) or not isinstance(doc.get("text"), str) or not doc["text"].strip():
                    errors.append(f"row {idx}, doc {j}: missing/nontext document text")
        if not isinstance(qa_pairs, list):
            errors.append(f"row {idx}: qa_pairs not a list")
        else:
            qa_counts[len(qa_pairs)] += 1
            if not qa_pairs:
                qa_gold_missing += 1
        if "sample_id" in row:
            sample_ids.append(str(row["sample_id"]))
    if errors:
        for err in errors[:25]:
            print("[SCHEMA ERROR]", err)
        raise ValueError(f"Validation failed: {len(errors)} issue(s); original file unchanged")

    print("[AUDIT] records:", len(records))
    print("[AUDIT] documents per record:", dict(sorted(docs_counts.items())))
    print("[AUDIT] QA pairs per record:", dict(sorted(qa_counts.items())))
    print("[AUDIT] empty qa_pairs:", qa_gold_missing)
    print("[AUDIT] sample IDs present:", len(sample_ids))
    print("[AUDIT] duplicate sample IDs:", len(sample_ids) - len(set(sample_ids)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=cli_path, required=True)
    parser.add_argument("--inplace", action="store_true", help="Replace the original JSON after backing it up")
    parser.add_argument("--out", type=cli_path, help="Destination for separate query-ready JSON (default: *_query_ready.json)")
    args = parser.parse_args()
    src = args.src.expanduser().resolve()
    if args.inplace and args.out is not None:
        parser.error("Use --inplace OR --out, not both")
    if not src.is_file():
        raise FileNotFoundError(src)

    with src.open("r", encoding="utf-8") as f:
        wrapped = json.load(f)
    if not isinstance(wrapped, dict) or not isinstance(wrapped.get("data"), list):
        raise ValueError("Expected source root {'config': ..., 'data': [...]}; no files changed")
    data = wrapped["data"]
    audit(data)

    dest = src if args.inplace else (args.out.expanduser().resolve() if args.out else src.with_name(src.stem + "_query_ready.json"))
    if not args.inplace and dest == src:
        raise ValueError("Cannot write to original path without --inplace")
    if not args.inplace and dest.exists():
        raise FileExistsError(f"Destination exists: {dest}; refusing accidental overwrite")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = src.with_name(f"{src.name}.bak_{stamp}")
    config_file = src.with_name(src.stem + ".original_config.json")
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}; no files changed")
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as cf:
            if json.load(cf) != wrapped.get("config", {}):
                raise ValueError(f"Existing config differs from source: {config_file}; no files changed")

    original_hash = sha256(src)
    # Backup before writing anything else.
    shutil.copy2(src, backup)
    if sha256(backup) != original_hash:
        raise RuntimeError("Backup checksum mismatch; no conversion performed")
    if not config_file.exists():
        atomic_json_write(config_file, wrapped.get("config", {}))
    atomic_json_write(dest, data)

    with dest.open("r", encoding="utf-8") as f:
        reloaded = json.load(f)
    if reloaded != data:
        raise RuntimeError("Read-back mismatch; restore original from backup")
    print("[OK] Original backup:", backup)
    print("[OK] Original SHA256:", original_hash)
    print("[OK] Original config:", config_file)
    print("[OK] Query-ready JSON:", dest)
    print("[OK] Output type: list; records:", len(reloaded))
    print("[OK] First three sample IDs:", [x.get("sample_id") for x in reloaded[:3]])
    print("[OK] Original records / metadata fields preserved; no docs or QA pairs changed")
    if not args.inplace:
        print(f'[NEXT] SRC_JSON="{dest}" bash asqa_query_batch.sh validate_input')
    else:
        print("[NEXT] bash asqa_query_batch.sh validate_input")


if __name__ == "__main__":
    main()
