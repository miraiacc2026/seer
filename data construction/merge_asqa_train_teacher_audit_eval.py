#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge available ASQA Train gold + GPT-4o S3 summaries for BOTH audit and ALCE eval.

Missing/invalid Summary (or R3 TopK) files are skipped, NOT assigned an empty
prediction. Save ORIGINAL skipped sample indices, one integer per line, to TXT.
Kept records retain their original sample_idx, so the audit script joins them
correctly to candidate_pool_<idx>.json and topk_<idx>.json.

Only generated output JSON and skipped-index TXT are replaced on reruns. Source
train, candidate pools, R3 TopK and S3 summaries are never modified.
"""

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import cli_path



def read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def rows_and_config(data, config_path: Path):
    if isinstance(data, dict) and isinstance(data.get('data'), list):
        return data['data'], data.get('config')
    if not isinstance(data, list):
        raise ValueError('Train must be a top-level list or {config, data} object')
    if config_path.is_file():
        return data, read_json(config_path)
    return data, {'dataset': 'ASQA', 'split': 'train', 'retrieval': 'GTR Top-5'}


def normalize_question(text):
    return re.sub(r'\s+', ' ', text.strip())


def atomic_write(path: Path, contents: str):
    """Replace a GENERATED file atomically; do not change any source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as f:
            tmp = Path(f.name)
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=cli_path, required=True)
    ap.add_argument('--config', type=cli_path, required=True)
    ap.add_argument('--summary-dir', type=cli_path, required=True)
    ap.add_argument('--topk-dir', type=cli_path, required=True)
    ap.add_argument('--out', type=cli_path, required=True)
    ap.add_argument('--skipped-txt', type=cli_path, required=True)
    ap.add_argument('--expected-size', type=int, default=4353)
    ap.add_argument('--skip-topk-check', action='store_true',
                    help='Only if R3 TopK files are unavailable; alignment verification is weaker')
    args = ap.parse_args()

    train_path, config_path, summary_dir, topk_dir, out, skipped_txt = [
        p.expanduser().resolve() for p in (
            args.train, args.config, args.summary_dir, args.topk_dir,
            args.out, args.skipped_txt,
        )
    ]
    if len({out, skipped_txt}) != 2:
        raise ValueError('Output JSON and skipped TXT must use different paths')
    if {out, skipped_txt} & {train_path, config_path}:
        raise ValueError('Refusing to overwrite the train input or config')
    if out.parent == summary_dir or skipped_txt.parent == summary_dir:
        raise ValueError('Refusing to place generated results inside the Summary input directory')
    if out.parent == topk_dir or skipped_txt.parent == topk_dir:
        raise ValueError('Refusing to place generated results inside the TopK input directory')

    train, config = rows_and_config(read_json(train_path), config_path)
    if args.expected_size and len(train) != args.expected_size:
        raise ValueError(f'Expected {args.expected_size} train rows; got {len(train)}')
    if not isinstance(config, dict):
        raise ValueError('Expected config dictionary')
    if not summary_dir.is_dir():
        raise FileNotFoundError(f'S3 summary directory missing: {summary_dir}')
    if not args.skip_topk_check and not topk_dir.is_dir():
        raise FileNotFoundError(f'R3 TopK directory missing: {topk_dir}')

    merged = []
    skipped = []
    reasons = Counter()
    reason_examples = {}
    citations_used = Counter()
    sample_ids = set()

    def skip(idx, reason, detail=None):
        skipped.append(idx)
        reasons[reason] += 1
        reason_examples.setdefault(reason, (idx, str(detail) if detail is not None else ''))

    for i, src in enumerate(train):
        # Treat invalid GOLD as fatal: it is not a missing teacher prediction.
        if not isinstance(src, dict):
            raise ValueError(f'Train #{i}: not an object')
        q, docs, qa, gold = (src.get(k) for k in ('question', 'docs', 'qa_pairs', 'answer'))
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f'Train #{i}: invalid question')
        if not isinstance(docs, list) or len(docs) != 5 or any(
            not isinstance(d, dict) or not isinstance(d.get('title'), str)
            or not isinstance(d.get('text'), str) or not d['text'].strip()
            for d in docs
        ):
            raise ValueError(f'Train #{i}: expected exactly 5 valid GTR docs')
        if not isinstance(qa, list) or not qa or any(
            not isinstance(p, dict) or not isinstance(p.get('short_answers'), list)
            or not p['short_answers'] for p in qa
        ):
            raise ValueError(f'Train #{i}: missing/invalid gold qa_pairs')
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError(f'Train #{i}: invalid gold answer')
        if 'sample_id' in src:
            sid = str(src['sample_id'])
            if sid in sample_ids:
                raise ValueError(f'Train #{i}: duplicate sample_id {sid}')
            sample_ids.add(sid)
        if 'sample_idx' in src and src['sample_idx'] != i:
            raise ValueError(f'Train #{i}: sample_idx mismatch in train')

        summary_path = summary_dir / f'summary_{i}.json'
        if not summary_path.is_file():
            skip(i, 'missing_summary')
            continue
        try:
            pred = read_json(summary_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            skip(i, 'unreadable_summary', exc)
            continue
        if not isinstance(pred, dict):
            skip(i, 'invalid_summary_structure')
            continue
        if pred.get('sample_idx', i) != i:
            skip(i, 'summary_index_mismatch', pred.get('sample_idx'))
            continue
        final = pred.get('final_summary')
        if not isinstance(final, str) or not final.strip():
            skip(i, 'empty_summary')
            continue
        # The uploaded eval.py silently truncates the output at its first newline.
        if '\n' in final or '\r' in final:
            skip(i, 'multiline_summary')
            continue
        citations = [int(s) for s in re.findall(r'\[(\d+)\]', final)]
        if any(not 1 <= cid <= len(docs) for cid in citations):
            skip(i, 'invalid_citation')
            continue

        if not args.skip_topk_check:
            topk_path = topk_dir / f'topk_{i}.json'
            if not topk_path.is_file():
                skip(i, 'missing_topk')
                continue
            try:
                topk = read_json(topk_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                skip(i, 'unreadable_topk', exc)
                continue
            if not isinstance(topk, dict):
                skip(i, 'invalid_topk_structure')
                continue
            if topk.get('sample_idx', i) != i:
                skip(i, 'topk_index_mismatch', topk.get('sample_idx'))
                continue
            if normalize_question(str(topk.get('query', ''))) != normalize_question(q):
                skip(i, 'topk_query_mismatch')
                continue
            if (topk.get('rerank_meta') or {}).get('query_mode') != 'original':
                skip(i, 'wrong_rerank_query_mode')
                continue

        item = dict(src)  # unchanged gold, QA pairs and docs
        item['sample_idx'] = i  # CRITICAL: index is not re-numbered after skipping
        item['output'] = final
        anns = item.get('gold_annotations', item.get('annotations', []))
        if 'gold_annotations' not in item:
            item['gold_annotations'] = anns
        # eval.py indexes annotations[0] and annotations[1] if present.
        if not isinstance(item.get('annotations'), list) or len(item.get('annotations')) < 2:
            item.pop('annotations', None)
        merged.append(item)
        citations_used.update(citations)

    # Check that no missing-summary skip shifts the audit sample indices.
    assert len(merged) + len(skipped) == len(train)
    assert len({r['sample_idx'] for r in merged}) == len(merged)
    for item in merged:
        original = train[item['sample_idx']]
        for name in ('question', 'docs', 'qa_pairs', 'answer', 'sample_id', 'wikipages'):
            if original.get(name) != item.get(name):
                raise AssertionError(f'Unexpected gold change at #{item["sample_idx"]}: {name}')
        if not item['output']:
            raise AssertionError(f'Empty output at #{item["sample_idx"]}')

    # Always write the skip list, including if all summaries are missing.
    atomic_write(skipped_txt, ''.join(f'{i}\n' for i in skipped))
    if not merged:
        raise RuntimeError(f'No valid summaries; wrote skip list to {skipped_txt}')

    payload = {'config': config, 'data': merged}
    atomic_write(out, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

    print(f'[PASS] original_train={len(train)} kept={len(merged)} skipped={len(skipped)}')
    print(f'[PASS] original sample_idx preserved: first={merged[0]["sample_idx"]} last={merged[-1]["sample_idx"]}')
    print('[PASS] compatible with audit.py (original idx) and ALCE eval.py (root["data"])')
    print(f'[OK] merged JSON: {out} ({out.stat().st_size / 1024**2:.2f} MiB)')
    print(f'[OK] skipped TXT: {skipped_txt} (one original index per line)')
    print(f'[INFO] skip reasons: {dict(sorted(reasons.items()))}')
    for reason, (idx, detail) in sorted(reason_examples.items()):
        print(f'  [EXAMPLE] {reason}: idx={idx}' + (f' detail={detail}' if detail else ''))
    print(f'[INFO] citations seen: {dict(sorted(citations_used.items()))}')
    print('[NOTE] Metrics on this file apply only to KEPT samples; S3 is seed-summary, not final expansion.')
    print('[NOTE] Re-running updates ONLY the merged JSON and skipped TXT to include newly finished summaries.')


if __name__ == '__main__':
    main()
