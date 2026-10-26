#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect the FINAL filtered ASQA Train teacher bundle by ORIGINAL sample_idx.

Input stage directories and the output bundle must be explicitly supplied.
Only writes to a NEW bundle directory; never modifies source files or teacher metadata.
Claims are copied out from Summary S3's existing `plan` (not generated anew).
No Hugging Face / OpenAI / GPU dependency.
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import cli_path



def read_json(p):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def norm(s):
    return re.sub(r'\s+', ' ', s.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--filtered', type=cli_path, required=True)
    ap.add_argument('--dest', type=cli_path, required=True)
    ap.add_argument('--expected', type=int, default=3789,
                    help='0 disables expected sample count assertion')
    ap.add_argument('--dry-run', action='store_true', help='Check sources without writing')
    ap.add_argument('--allow-missing', action='store_true',
                    help='Copy valid stages even if some source stage files are missing')
    ap.add_argument('--query-dir', type=cli_path, required=True)
    ap.add_argument('--pool-dir', type=cli_path, required=True)
    ap.add_argument('--rerank-dir', type=cli_path, required=True)
    ap.add_argument('--summary-dir', type=cli_path, required=True)
    args = ap.parse_args()

    src_filtered = args.filtered.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    source_stages = {
        'query_struct': (args.query_dir.expanduser().resolve(), 'query'),
        'candidate_pool': (args.pool_dir.expanduser().resolve(), 'candidate_pool'),
        'rerank': (args.rerank_dir.expanduser().resolve(), 'topk'),
        'summary': (args.summary_dir.expanduser().resolve(), 'summary'),
    }
    if not src_filtered.is_file():
        raise FileNotFoundError(src_filtered)
    if dest == Path(dest.anchor) or dest == Path.cwd():
        raise ValueError('Destination must be a dedicated bundle directory')
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Refusing to mix/overwrite an existing bundle: {dest}')
    if dest == src_filtered.parent or dest in [p for p, _ in source_stages.values()]:
        raise ValueError('Destination must not point to source directories')

    root = read_json(src_filtered)
    if not isinstance(root, dict) or not isinstance(root.get('data'), list):
        raise ValueError('Filtered JSON must be ALCE {config,data}')
    rows = root['data']
    if args.expected and len(rows) != args.expected:
        raise ValueError(f'Expected {args.expected}, got {len(rows)}. Check source file.')

    indices = []
    sample_ids = set()
    for pos, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Invalid row at {pos}')
        idx = row.get('sample_idx')
        if type(idx) is not int or idx < 0:
            raise ValueError(f'Missing ORIGINAL integer sample_idx at row {pos}')
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'No question at {idx}')
        if not isinstance(row.get('output'), str) or not row['output'].strip():
            raise ValueError(f'No output at {idx}')
        indices.append(idx)
        sid = row.get('sample_id')
        if sid is not None:
            if sid in sample_ids:
                raise ValueError(f'Duplicate sample_id={sid}')
            sample_ids.add(sid)
    if indices != sorted(set(indices)):
        raise ValueError('Filtered indices must be unique and ascending; no renumbering')

    # First pass: preflight full alignment. No changes to source or destination.
    paths = {stage: {} for stage in source_stages}
    summaries = {}
    missing = {stage: [] for stage in source_stages}
    bad = []
    for row in rows:
        idx = row['sample_idx']
        for stage, (folder, prefix) in source_stages.items():
            path = folder / f'{prefix}_{idx}.json'
            if not path.is_file():
                missing[stage].append(idx)
                continue
            try:
                obj = read_json(path)
                if not isinstance(obj, dict):
                    raise ValueError('expected JSON object')
                if obj.get('sample_idx', idx) != idx:
                    raise ValueError(f'sample_idx={obj.get("sample_idx")}')
                query = obj.get('query')
                if isinstance(query, str) and norm(query) != norm(row['question']):
                    raise ValueError('original query differs from filtered record')
                if stage == 'summary':
                    if obj.get('final_summary') != row['output']:
                        raise ValueError('final_summary differs from filtered output')
                    if not isinstance(obj.get('plan'), list):
                        raise ValueError('S3 plan is not a list')
                    summaries[idx] = obj
                paths[stage][idx] = path
            except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as e:
                bad.append({'stage': stage, 'sample_idx': idx, 'file': str(path), 'reason': str(e)})

    print(f'[BUNDLE] selected={len(rows)} original_idx={indices[0]}..{indices[-1]}')
    for stage in source_stages:
        print(f'[CHECK] {stage:14} verified={len(paths[stage]):4} missing={len(missing[stage]):4}')
    print(f'[CHECK] invalid/mismatched={len(bad)}')
    if bad:
        print('[EXAMPLES]', bad[:5])
    if args.dry_run:
        print('[DRY-RUN] Nothing written.')
        return
    if bad:
        raise RuntimeError('Some source JSONs are malformed or mismatched; refusing partial/misaligned bundle')
    if any(missing.values()) and not args.allow_missing:
        print('[MISSING EXAMPLES]', {k: v[:12] for k, v in missing.items() if v})
        raise RuntimeError('Stage files are missing. Inspect paths or use --allow-missing intentionally.')

    dest.mkdir(parents=True, exist_ok=True)
    # Copy filtered data verbatim, including exact original query, QA Gold, docs and output.
    shutil.copy2(src_filtered, dest / 'asqa_train_teacher_s3_filtered.json')
    for stage in source_stages:
        outdir = dest / stage
        outdir.mkdir(exist_ok=True)
        for idx, src in paths[stage].items():
            shutil.copy2(src, outdir / src.name)

    # There is no separate Claim model here: faithfully extract *existing* S3 plan.
    claim_dir = dest / 'claims'
    claim_dir.mkdir(exist_ok=True)
    for row in rows:
        idx = row['sample_idx']
        summary = summaries.get(idx)
        if summary is None:
            continue
        claim = {
            'sample_idx': idx,
            'query': row['question'],
            'plan_style': summary.get('plan_style'),
            'plan': summary['plan'],
            'source': f'summary_{idx}.json#plan',
        }
        (claim_dir / f'claim_{idx}.json').write_text(
            json.dumps(claim, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    meta_dir = dest / 'metadata'
    meta_dir.mkdir(exist_ok=True)
    (meta_dir / 'kept_sample_idx.txt').write_text(
        ''.join(f'{i}\n' for i in indices), encoding='utf-8')
    for name in ('filter_report.json', 'filtered_out_sample_idx.txt'):
        src = src_filtered.parent / name
        if src.is_file():
            shutil.copy2(src, meta_dir / name)
    manifest = {
        'source_filtered_json': str(src_filtered),
        'destination': str(dest),
        'selected': len(indices),
        'sample_idx_policy': 'original indices; no renumbering',
        'stage_counts': {key: len(val) for key, val in paths.items()},
        'extracted_s3_plan_claims': len(summaries),
        'claims_provenance': 'S3 summary_N.json plan, not independently generated',
        'source_dirs': {key: str(val[0]) for key, val in source_stages.items()},
        'missing_by_stage': {k: v for k, v in missing.items()},
        'source_metadata_preserved': True,
    }
    (meta_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[DONE] Bundle: {dest}')
    print(f'[DONE] filtered={len(rows)} claims={len(summaries)}')
    print(f'[DONE] manifest={meta_dir / "manifest.json"}')


if __name__ == '__main__':
    main()
