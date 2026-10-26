#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build 3,789 ASQA Query-Structure SFT messages from archived GPT-4o labels.

System and User prompts are an exact copy of the GPT-4o Teacher's
build_messages(query) (asqa query-structure generator, 2026-10-15).
Assistant target is the archived query_struct JSON, without regeneration.
Validation split is identical to both Rerank R3 and Summary S3 (if available).
Does not read gold QA answers, R3 evidence, Summary outputs or Query predictions.
"""

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import cli_path

TRAINER = Path(__file__).resolve().parent / "train_query_fsdp.py"

# START VERBATIM GPT-4o TEACHER PROMPT FUNCTION
def build_messages(query: str) -> List[Dict[str, str]]:
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


    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# END VERBATIM GPT-4o TEACHER PROMPT FUNCTION

REQUIRED = {
    'anchor_entities', 'anchor_aliases', 'anchor_terms', 'relation_terms',
    'relation_synonym', 'relation_expansion', 'relation_markers', 'scope_terms',
    'attributes', 'retrieval_control', 'retrieval_views',
}
VIEW_NAMES = {'literal', 'entity_expanded', 'semantic'}


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, payload: Any):
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')


def norm(text: str) -> str:
    return ' '.join(text.split()).casefold()


def valid_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def validate_teacher_target(obj: Any) -> None:
    """Matches the schema accepted by the existing train_query_fsdp.py."""
    if not isinstance(obj, dict) or not REQUIRED.issubset(obj):
        raise ValueError('missing required Query Structure keys')
    for key in ('anchor_entities', 'anchor_terms', 'relation_terms',
                'relation_synonym', 'relation_expansion', 'relation_markers',
                'scope_terms'):
        if not valid_str_list(obj.get(key)):
            raise ValueError(f'{key} must be list[str]')
    aliases = obj['anchor_aliases']
    if not isinstance(aliases, dict) or any(
        not isinstance(k, str) or not valid_str_list(v)
        for k, v in aliases.items()
    ):
        raise ValueError('invalid anchor_aliases')
    attrs = obj['attributes']
    if not isinstance(attrs, list):
        raise ValueError('attributes must be list')
    for i, item in enumerate(attrs):
        if not isinstance(item, dict) or not {
            'attribute', 'bridge_triggers', 'hypothesis_triggers', 'time_or_scope'
        }.issubset(item):
            raise ValueError(f'invalid attributes[{i}] keys')
        if not isinstance(item['attribute'], str) or any(
            not valid_str_list(item[key])
            for key in ('bridge_triggers', 'hypothesis_triggers', 'time_or_scope')
        ):
            raise ValueError(f'invalid attributes[{i}] content')
    control = obj['retrieval_control']
    if not isinstance(control, dict) or any(
        not valid_str_list(control.get(key))
        for key in ('relation_markers', 'answerability_markers', 'parallel_cues')
    ):
        raise ValueError('invalid retrieval_control')
    views = obj['retrieval_views']
    if not isinstance(views, list) or not views:
        raise ValueError('retrieval_views must be nonempty list')
    names = set()
    for i, item in enumerate(views):
        if not isinstance(item, dict):
            raise ValueError(f'bad retrieval_views[{i}]')
        name, text = item.get('view_name'), item.get('text')
        if not isinstance(name, str) or name not in VIEW_NAMES or not isinstance(text, str):
            raise ValueError(f'bad retrieval_views[{i}] name/text')
        names.add(name)
    if not VIEW_NAMES.issubset(names):
        raise ValueError('missing literal/entity_expanded/semantic view')


def verify_trainer_system(trainer_path: Path, teacher_system: str) -> None:
    if not trainer_path.is_file():
        raise FileNotFoundError(f'Query trainer absent: {trainer_path}')
    parsed = ast.parse(trainer_path.read_text(encoding='utf-8'))
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'DEFAULT_SYSTEM_PROMPT'
            for t in node.targets
        ):
            system = ast.literal_eval(node.value)
            if system != teacher_system:
                raise ValueError('Student DEFAULT_SYSTEM_PROMPT differs from Teacher prompt')
            print('[OK] Student trainer System Prompt == GPT-4o Teacher System Prompt')
            return
    raise ValueError('DEFAULT_SYSTEM_PROMPT not found in train_query_fsdp.py')


def build_one(idx: int, root: Path, filtered_question: str):
    file = root / 'query_struct' / f'query_{idx}.json'
    row = load_json(file)
    if not isinstance(row, dict) or row.get('sample_idx') != idx:
        raise ValueError(f'query_struct sample_idx mismatch: {file}')
    query = row.get('query')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('empty archived query')
    if norm(query) != norm(filtered_question):
        raise ValueError('archived query does not match filtered question')
    target = row.get('query_struct')
    validate_teacher_target(target)
    messages = build_messages(query)
    if len(messages) != 2 or [m['role'] for m in messages] != ['system','user']:
        raise ValueError('unexpected GPT-4o prompt message structure')
    assistant = json.dumps(target, ensure_ascii=False, separators=(',', ':'))
    example = {
        'dataset': 'asqa_query_s3',
        'sample_idx': idx,
        'messages': messages + [{'role': 'assistant', 'content': assistant}],
    }
    audit = {
        'sample_idx': idx,
        'anchor_entities': len(target['anchor_entities']),
        'relation_terms': len(target['relation_terms']),
        'attributes': len(target['attributes']),
        'retrieval_views': len(target['retrieval_views']),
        'assistant_chars': len(assistant),
        'query': query,
    }
    return example, audit


def read_split_ids(path: Path) -> set:
    obj = load_json(path)
    if not isinstance(obj, dict) or not isinstance(obj.get('examples'), list):
        raise ValueError(f'invalid split file: {path}')
    ids = [x.get('sample_idx') for x in obj['examples'] if isinstance(x, dict)]
    if len(ids) != len(obj['examples']) or any(type(i) is not int for i in ids):
        raise ValueError(f'invalid sample_idx in {path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate sample_idx in {path}')
    return set(ids)


def find_split(folder: Path, prefix: str):
    tr = folder / (prefix + '_train.json')
    va = folder / (prefix + '_val.json')
    present = (tr.is_file(), va.is_file())
    if any(present) and not all(present):
        raise FileNotFoundError(f'partial train/val split in {folder}: {prefix}')
    if not all(present):
        return None
    tr_ids, va_ids = read_split_ids(tr), read_split_ids(va)
    if tr_ids & va_ids:
        raise ValueError(f'overlapping split in {folder}: {prefix}')
    return tr_ids, va_ids


def make_group_split(ids: List[int], queries: dict, val_size: int, seed: int):
    if val_size <= 0 or val_size >= len(ids):
        raise ValueError('val-size must be >0 and < number of examples')
    groups = defaultdict(list)
    for idx in ids:
        groups[norm(queries[idx])].append(idx)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    validation = set()
    for group in shuffled:
        if abs(len(validation) + len(group) - val_size) <= abs(len(validation) - val_size):
            validation.update(group)
        if len(validation) >= val_size:
            break
    training = set(ids) - validation
    if not training or not validation:
        raise ValueError('empty train or val set')
    return training, validation


def choose_split(ids: List[int], queries: dict, folder: Path, mode: str,
                 val_size: int, seed: int):
    rerank = find_split(folder, 'asqa_r3_rerank')
    summary = find_split(folder / 'summary_s3', 'asqa_s3_summary')
    if rerank and summary and rerank != summary:
        raise ValueError('Rerank and Summary train/val splits differ; resolve first')
    options = {'rerank': rerank, 'summary': summary}
    if mode in options:
        selected = options[mode]
        if selected is None:
            raise FileNotFoundError(f'{mode} train/val JSON missing')
        method = mode
    elif mode == 'auto':
        selected = rerank or summary
        method = 'rerank' if rerank else ('summary' if summary else 'query')
    else:
        selected = None
        method = 'query'
    if selected is None:
        selected = make_group_split(ids, queries, val_size, seed)
    train_ids, val_ids = selected
    if train_ids | val_ids != set(ids):
        raise ValueError('split sample IDs differ from filtered data; use full dataset or --split-mode query')
    if {norm(queries[i]) for i in train_ids} & {norm(queries[i]) for i in val_ids}:
        raise ValueError('same original query appears in both train and val')
    return train_ids, val_ids, method


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold-root', type=cli_path, required=True)
    ap.add_argument('--output-dir', type=cli_path, required=True)
    ap.add_argument('--sft-root', type=cli_path, required=True)
    ap.add_argument('--train-script', type=cli_path, default=TRAINER)
    ap.add_argument('--split-mode', choices=['auto','rerank','summary','query'], default='auto')
    ap.add_argument('--val-size', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-examples', type=int, default=-1)
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    root = args.gold_root.expanduser()
    filtered = load_json(root / 'asqa_train_teacher_s3_filtered.json')
    if not isinstance(filtered, dict) or not isinstance(filtered.get('data'), list):
        raise ValueError('Gold master must have {"data": [...]}')
    ids, original_questions = [], {}
    for row in filtered['data']:
        if not isinstance(row, dict) or type(row.get('sample_idx')) is not int:
            raise ValueError('invalid sample_idx in filtered Gold master')
        idx, question = row['sample_idx'], row.get('question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'empty question for {idx}')
        if idx in original_questions:
            raise ValueError(f'duplicate sample_idx={idx}')
        ids.append(idx)
        original_questions[idx] = question
    if args.max_examples >= 0:
        ids = ids[:args.max_examples]
    if not ids:
        raise ValueError('no input samples')
    teacher_system = build_messages('sanity-check')[0]['content']
    verify_trainer_system(args.train_script.expanduser(), teacher_system)
    failures, examples, audits, queries = [], [], [], {}
    for idx in ids:
        try:
            ex, audit = build_one(idx, root, original_questions[idx])
            examples.append(ex)
            audits.append(audit)
            queries[idx] = audit['query']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({'sample_idx': idx, 'error': f'{type(exc).__name__}: {exc}'})
    if failures:
        print('[ERROR] Samples failed schema/index checks:', len(failures))
        print(json.dumps(failures[:15], ensure_ascii=False, indent=2))
        raise RuntimeError('Stopped: no SFT data written due to invalid samples')
    tr_ids, va_ids, method = choose_split(ids, queries, args.sft_root.expanduser(),
                                           args.split_mode, args.val_size, args.seed)
    train = sorted((e for e in examples if e['sample_idx'] in tr_ids), key=lambda e:e['sample_idx'])
    val = sorted((e for e in examples if e['sample_idx'] in va_ids), key=lambda e:e['sample_idx'])
    examples.sort(key=lambda e:e['sample_idx'])
    stats = {
        'total_built': len(examples),
        'total_anchor_entities': sum(a['anchor_entities'] for a in audits),
        'total_relation_terms': sum(a['relation_terms'] for a in audits),
        'total_attributes': sum(a['attributes'] for a in audits),
        'zero_anchor_entity_samples': sum(not a['anchor_entities'] for a in audits),
        'zero_relation_term_samples': sum(not a['relation_terms'] for a in audits),
    }
    report = {
        'source': str(root), 'split_method': method,
        'train_size': len(train), 'val_size': len(val), 'total_sft': len(examples),
        'seed': args.seed, 'train_script': str(args.train_script.expanduser()),
        'system_prompt_matches_teacher': True,
        'user_prompt_is_exact_teacher_build_messages': True,
        'stats': stats,
        'notes': [
            'Uses full GPT-4o Teacher Query Structuring system/user messages (including few-shot example).',
            'Uses only archived query_struct as supervision; no synthetic or filtered-away targets.',
            'Student trainer accepts messages directly, with no User Prompt reconstruction.',
            'Train/val reuses Rerank and Summary partitions when present; query grouped.',
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.no_write:
        return
    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'asqa_query_s3_train.json', {'examples': train})
    write_json(out / 'asqa_query_s3_val.json', {'examples': val})
    write_json(out / 'asqa_query_s3_all.json', {'examples': examples})
    write_json(out / 'asqa_query_s3_audit.json', report)
    with (out / 'asqa_query_s3_sample_audit.jsonl').open('w', encoding='utf-8') as f:
        for audit in audits:
            f.write(json.dumps(audit, ensure_ascii=False) + '\n')
    for p in sorted(out.glob('asqa_query_s3*')):
        print(f'[OK] {p} ({p.stat().st_size:,} bytes)')
    first = train[0]
    print('[PREVIEW] sample_idx:',first['sample_idx'])
    print('[PREVIEW] system:',first['messages'][0]['content'][:200])
    print('[PREVIEW] user head:',first['messages'][1]['content'][:250].replace('\n',' | '))
    print('[PREVIEW] assistant head:',first['messages'][2]['content'][:300])


if __name__ == '__main__':
    main()
