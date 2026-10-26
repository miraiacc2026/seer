#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build ASQA Summary S3/A4 messages-only SFT data from archived GPT-4o outputs.

Inputs: archived rerank/topk_{sample_idx}.json (original query and R3 evidence)
Targets: archived summary/summary_{sample_idx}.json (unchanged plan_style, plan,
         final_summary). Selection: filtered 3,789 sample_idx, not ASQA Dev.

No generated claims, inferred citations, gold QA pairs, query structuring, or
post-generation plan truncation. Does not change FSDP/Trainer/launchers.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import cli_path


# Based on GPT-4o asqa_summary.py build_summary_prompt(query_mode='original',
# use_plan=True). Removed only obsolete seed/expansion terminology and retained
# S3 relation coverage, literal answer-phrase preservation, DOC_ID citation.
SYSTEM_PROMPT = """You are a high-recall attribution planner and final-answer writer for retrieval-augmented question answering.

Use ONLY the original question and supplied evidence. Do NOT invent facts or citations.
Produce the FINAL answer directly, not an intermediate seed summary.
Preserve all distinct supported, answer-relevant relations and alternative answers.
Do not collapse valid branches that differ by time, version, location, role,
event stage, authority, counting scheme, scope, interpretation, or standard.
Preserve exact answer-bearing names, dates, numbers, titles, ordered lists, and
qualifiers from the evidence; do not change the relationship they express.
A single source may support more than one answer relation. Retain every such
relevant relation in both the plan and the final answer.
Use DOC_ID citations ONLY, never SENT_IDs. Every sentence of the final answer
must be grounded and end with DOC_ID citation brackets such as [2] or [2][3].
Output STRICT JSON with exactly: plan_style, plan, final_summary.
plan_style is 'direct' for one answer relation and 'parallel' for distinct
answer relations, interpretations, scopes, or event stages.
plan is a list of {claim, citations} objects; citations are lists of DOC_ID strings.
final_summary is the complete cited answer. No markdown or extra text."""

EXAMPLE = {
    'plan_style': 'parallel',
    'plan': [
        {'claim': 'Construction on the Lark Museum began in March 2007.', 'citations': ['2']},
        {'claim': 'The museum opened on June 18, 2009.', 'citations': ['2']},
    ],
    'final_summary': 'Construction on the Lark Museum began in March 2007 [2]. The museum opened on June 18, 2009 [2].',
}


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def norm(value: str) -> str:
    return ' '.join(value.split())


def text_of_sentence(row: dict) -> str:
    # Same precedence as GPT-4o teacher build_evidence_lines().
    return norm(row.get('sent_content', '') or row.get('sentence', '') or row.get('text', ''))


def build_evidence_lines(topk: dict, max_core: int, max_support: int, max_packed: int) -> Tuple[List[str], Dict[str, int]]:
    """Faithful to S3 teacher formatting: [bucket] DOC_ID=X TEXT=sentence.

    Dedup is bucket-specific: a sentence shown as [core] may also appear
    under [packed], as in the original teacher prompt.
    """
    lines, used = [], set()
    counts = {}
    for field, bucket, cap in (
        ('core_evidence', 'core', max_core),
        ('support_evidence', 'support', max_support),
        ('packed_evidence', 'packed', max_packed),
    ):
        source = topk.get(field)
        if not isinstance(source, list):
            raise ValueError(f'{field} must be a list')
        count = 0
        for row in source[:cap]:
            if not isinstance(row, dict):
                continue
            doc_id = row.get('doc_id', -1)
            sent = text_of_sentence(row)
            if not sent:
                continue
            key = (doc_id, sent.lower(), bucket)
            if key in used:
                continue
            used.add(key)
            lines.append(f'[{bucket}] DOC_ID={doc_id} TEXT={sent}')
            count += 1
        counts[bucket] = count
    return lines, counts


def make_user_prompt(query: str, lines: List[str]) -> str:
    """Original question, S3 rules & formatted evidence, no Query Structure."""
    return f"""Formatting example (fictional; NOT evidence for the actual query):
Example query: When was the Lark Museum built?
Example evidence: [core] DOC_ID=2 TEXT=Construction on the Lark Museum began in March 2007, and the museum opened on June 18, 2009.
Example output:
{json.dumps(EXAMPLE, ensure_ascii=False, indent=2)}

The example shows that one source can support distinct answer relations. It does not prescribe the number of claims.

Original query:
{query}

Evidence:
{chr(10).join(lines) if lines else '[NO_EVIDENCE]'}

Task: Write a complete, compact, citation-grounded FINAL answer with an attribution plan.

Coverage and wording:
- Include every distinct supported answer branch. Do not choose just the dominant interpretation.
- Retain separate relations for different times, versions, locations, roles, scopes, or event stages.
- Preserve exact answer phrases, proper names, full dates, numbers, and coordinated lists as stated in the evidence.
- Do not invent actor-role links, episode numbers, dates, or unsupported inferences.
- A relevant partial answer is preferable to omitting an available supported branch.
- Keep each plan claim tied to the supporting DOC_ID(s). Repeated DOC_IDs across claims are allowed.
- The final_summary must express the supported claims and end every sentence with DOC_ID citations.

Output rules:
- plan_style: 'direct' for one relation, 'parallel' for multiple distinct relations.
- plan: a list of objects with 'claim' and 'citations' (nonempty DOC_ID string lists for cited claims).
- final_summary: the FINAL cited answer, not an intermediate seed summary.
- Use only DOC_IDs appearing in the evidence; never use SENT_IDs.
- Return strict JSON only, exactly these three top-level fields, without markdown or commentary."""


def audit_target(target: Any) -> dict:
    """Validate without rewriting or dropping teacher content.

    Empty plan is allowed and audited because the GPT-4o parser may save one.
    The trainer validator needs to accommodate it when revised next.
    """
    if not isinstance(target, dict) or set(target) != {'plan_style', 'plan', 'final_summary'}:
        raise ValueError('summary must contain exactly plan_style, plan, final_summary')
    if target['plan_style'] not in ('direct', 'parallel'):
        raise ValueError('invalid plan_style')
    if not isinstance(target['final_summary'], str) or not target['final_summary'].strip():
        raise ValueError('empty final_summary')
    plan = target['plan']
    if not isinstance(plan, list):
        raise ValueError('plan must be a list')
    for i, item in enumerate(plan):
        if not isinstance(item, dict) or set(item) != {'claim', 'citations'}:
            raise ValueError(f'plan[{i}] must contain exactly claim/citations')
        if not isinstance(item['claim'], str) or not item['claim'].strip():
            raise ValueError(f'plan[{i}] has empty claim')
        if not isinstance(item['citations'], list) or not item['citations']:
            raise ValueError(f'plan[{i}] missing citations')
        if not all(isinstance(x, str) and x.strip() for x in item['citations']):
            raise ValueError(f'plan[{i}] has invalid citation element')
    return {'plan_count': len(plan), 'empty_plan': not bool(plan), 'plan_style': target['plan_style'],
            'answer_words': len(target['final_summary'].split()),
            'non_digit_plan_citations': sum(not x.strip().isdigit() for p in plan for x in p['citations'])}


def build_one(idx: int, root: Path, max_core: int, max_support: int, max_packed: int) -> Tuple[dict, dict, str]:
    rr = read_json(root / 'rerank' / f'topk_{idx}.json')
    target = read_json(root / 'summary' / f'summary_{idx}.json')
    if not isinstance(rr, dict):
        raise ValueError('topk root must be object')
    if rr.get('sample_idx') != idx:
        raise ValueError('topk sample_idx mismatch')
    query = rr.get('query')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('missing original query in topk')
    meta = rr.get('rerank_meta') or {}
    if meta.get('query_mode') != 'original':
        raise ValueError(f'expected original query mode, got {meta.get("query_mode")!r}')
    lines, counts = build_evidence_lines(rr, max_core, max_support, max_packed)
    details = audit_target(target)
    # Diagnostics only: never repair or invent teacher citations.
    available_doc_ids = {str(r.split(' DOC_ID=', 1)[1].split(' TEXT=', 1)[0]).strip()
                         for r in lines if ' DOC_ID=' in r and ' TEXT=' in r}
    cited_plan = {c for p in target['plan'] for c in p['citations']}
    cited_final = set(re.findall(r'\[(\d+)\]', target['final_summary']))
    details['unavailable_plan_doc_ids'] = sorted(cited_plan - available_doc_ids)
    details['unavailable_final_doc_ids'] = sorted(cited_final - available_doc_ids)

    ex = {
        'dataset': 'asqa_s3_a4',
        'sample_idx': idx,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': make_user_prompt(query.strip(), lines)},
            {'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False, separators=(',', ':'))},
        ],
    }
    details.update({'sample_idx': idx, 'query': query.strip(), 'evidence_lines': len(lines),
                    'evidence_bucket_counts': counts, 'empty_evidence': not bool(lines)})
    return ex, details, query.strip()


def get_filtered_rows(root: Path) -> list:
    src = read_json(root / 'asqa_train_teacher_s3_filtered.json')
    if not isinstance(src, dict) or not isinstance(src.get('data'), list):
        raise ValueError('expected filtered JSON object with data list')
    return src['data']


def load_rerank_split_ids(folder: Path) -> Tuple[set, set]:
    result = []
    for name in ('asqa_r3_rerank_train.json', 'asqa_r3_rerank_val.json'):
        rows = read_json(folder / name)
        if not isinstance(rows, dict) or not isinstance(rows.get('examples'), list):
            raise ValueError(f'invalid rerank split: {name}')
        ids = [r.get('sample_idx') for r in rows['examples']]
        if not all(type(v) is int for v in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'invalid or duplicate sample_idx: {name}')
        result.append(set(ids))
    if result[0] & result[1]:
        raise ValueError('Rerank Train/Val sample_idx overlap')
    return result[0], result[1]


def split_by_original_query(examples: List[dict], queries: Dict[int, str], val_size: int, seed: int):
    if val_size < 0 or val_size >= len(examples):
        raise ValueError('val_size must be >=0 and smaller than dataset size')
    if val_size == 0:
        return examples, []
    groups = defaultdict(list)
    for ex in examples:
        idx = ex['sample_idx']
        groups[norm(queries[idx]).casefold()].append(idx)
    values = list(groups.values())
    random.Random(seed).shuffle(values)
    val_ids = set()
    for group in values:
        if abs(len(val_ids) + len(group) - val_size) <= abs(len(val_ids) - val_size):
            val_ids.update(group)
        if len(val_ids) >= val_size:
            break
    train = [x for x in examples if x['sample_idx'] not in val_ids]
    val = [x for x in examples if x['sample_idx'] in val_ids]
    if not train or not val:
        raise ValueError('empty train or val partition')
    return train, val


def split_examples(examples: List[dict], queries: Dict[int, str], split_mode: str,
                   rerank_split_dir: Path, val_size: int, seed: int):
    train_path = rerank_split_dir / 'asqa_r3_rerank_train.json'
    val_path = rerank_split_dir / 'asqa_r3_rerank_val.json'
    exists = [p.is_file() for p in (train_path, val_path)]
    if split_mode == 'rerank' or (split_mode == 'auto' and all(exists)):
        if not all(exists):
            raise FileNotFoundError('both rerank train/val files are required')
        train_ids, val_ids = load_rerank_split_ids(rerank_split_dir)
        source_ids = {x['sample_idx'] for x in examples}
        if train_ids | val_ids != source_ids:
            only_source = sorted(source_ids - (train_ids | val_ids))
            only_rerank = sorted((train_ids | val_ids) - source_ids)
            raise ValueError('rerank split sample set differs from Summary source: '
                             f'summary-only={only_source[:10]} rerank-only={only_rerank[:10]}. '
                             'Do not silently assign/unassign samples; use --split-mode query if intentional.')
        # Cross-split query leakage is still checked, even for reused splits.
        tr_queries = {norm(queries[i]).casefold() for i in train_ids}
        va_queries = {norm(queries[i]).casefold() for i in val_ids}
        if tr_queries & va_queries:
            raise ValueError('duplicate original queries appear in both rerank splits')
        return ([x for x in examples if x['sample_idx'] in train_ids],
                [x for x in examples if x['sample_idx'] in val_ids], 'rerank')
    if split_mode == 'auto' and any(exists):
        raise FileNotFoundError('only one rerank split file exists; fix before running')
    train, val = split_by_original_query(examples, queries, val_size, seed)
    return train, val, 'query'


def write_json(path: Path, obj: Any) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold-root', type=cli_path, required=True)
    ap.add_argument('--output-dir', type=cli_path, required=True)
    ap.add_argument('--rerank-split-dir', type=cli_path, required=True)
    ap.add_argument('--split-mode', choices=('auto', 'rerank', 'query'), default='auto',
                    help='auto: reuse rerank split when both files exist, otherwise group by query')
    ap.add_argument('--val-size', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-core', type=int, default=5)
    ap.add_argument('--max-support', type=int, default=8)
    ap.add_argument('--max-packed', type=int, default=14)
    ap.add_argument('--max-examples', type=int, default=-1)
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    if min(args.max_core, args.max_support, args.max_packed) < 0:
        ap.error('evidence maximums must be nonnegative')
    root = args.gold_root.expanduser()
    filtered = get_filtered_rows(root)
    ids = [x.get('sample_idx') if isinstance(x, dict) else None for x in filtered]
    if any(type(x) is not int for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('filtered data has invalid or duplicated sample_idx')
    if args.max_examples >= 0:
        ids = ids[:args.max_examples]
    original_questions = {x['sample_idx']: x.get('question') for x in filtered if isinstance(x, dict)}

    examples, audits, failures, queries = [], [], [], {}
    stats = Counter()
    for idx in ids:
        try:
            ex, detail, query = build_one(idx, root, args.max_core, args.max_support, args.max_packed)
            question = original_questions.get(idx)
            if not isinstance(question, str) or norm(question) != norm(query):
                raise ValueError('topk original query != filtered question')
            examples.append(ex)
            queries[idx] = query
            audits.append(detail)
            stats['total_built'] += 1
            stats['plan_items'] += detail['plan_count']
            stats['empty_plans'] += int(detail['empty_plan'])
            stats['empty_evidence'] += int(detail['empty_evidence'])
            stats['non_digit_plan_citations'] += detail['non_digit_plan_citations']
            stats['samples_with_unavailable_plan_citations'] += bool(detail['unavailable_plan_doc_ids'])
            stats['samples_with_unavailable_final_citations'] += bool(detail['unavailable_final_doc_ids'])
            stats['direct'] += detail['plan_style'] == 'direct'
            stats['parallel'] += detail['plan_style'] == 'parallel'
        except (OSError, ValueError, TypeError, KeyError) as e:
            failures.append({'sample_idx': idx, 'error': f'{type(e).__name__}: {e}'})

    if failures:
        print(f'[ERROR] {len(failures)} failed input/target checks. First 15:')
        print(json.dumps(failures[:15], ensure_ascii=False, indent=2))
        raise RuntimeError('No files were written. Fix data/schema mismatch first.')
    if not examples:
        raise ValueError('No examples')
    train, val, method = split_examples(examples, queries, args.split_mode,
                                       args.rerank_split_dir.expanduser(), args.val_size, args.seed)
    train.sort(key=lambda x: x['sample_idx'])
    val.sort(key=lambda x: x['sample_idx'])
    examples.sort(key=lambda x: x['sample_idx'])
    report = {
        'source': str(root), 'split_method': method, 'train_size': len(train),
        'val_size': len(val), 'total_sft': len(examples), 'requested_val_size': args.val_size,
        'seed': args.seed, 'max_core': args.max_core, 'max_support': args.max_support,
        'max_packed': args.max_packed, 'stats': dict(stats),
        'empty_plan_sample_indices': [x['sample_idx'] for x in audits if x['empty_plan']],
        'empty_evidence_sample_indices': [x['sample_idx'] for x in audits if x['empty_evidence']],
        'unavailable_plan_citation_sample_indices': [x['sample_idx'] for x in audits if x['unavailable_plan_doc_ids']],
        'unavailable_final_citation_sample_indices': [x['sample_idx'] for x in audits if x['unavailable_final_doc_ids']],
        'notes': [
            'Teacher target values are preserved; no plan truncation, claim generation, or invented citations.',
            'Original Query only; query_struct is not read or put in the User Prompt.',
            'Summary input matches the S3 teacher evidence rendering: core/support/packed at caps 5/8/14.',
            'The training Python system-prompt override must be updated to match this builder in the NEXT stage.',
            'The training Python validator must permit any preserved empty plans, if present.',
            'Token-length and response-only loss checks are not part of this builder.',
        ],
    }
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_indices')}, ensure_ascii=False, indent=2))
    if args.no_write:
        return
    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'asqa_s3_summary_train.json', {'examples': train})
    write_json(out / 'asqa_s3_summary_val.json', {'examples': val})
    write_json(out / 'asqa_s3_summary_all.json', {'examples': examples})
    write_json(out / 'asqa_s3_summary_audit.json', report)
    with (out / 'asqa_s3_summary_sample_audit.jsonl').open('w', encoding='utf-8') as f:
        for r in audits:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for p in sorted(out.glob('asqa_s3_summary*')):
        print(f'[OK] {p} ({p.stat().st_size:,} bytes)')
    print('[PREVIEW] sample_idx=', train[0]['sample_idx'])
    print('[PREVIEW] User:', train[0]['messages'][1]['content'][:250].replace('\n', ' | '))
    print('[PREVIEW] Assistant:', train[0]['messages'][2]['content'][:400])


if __name__ == '__main__':
    main()
