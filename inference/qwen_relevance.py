"""Qwen relevance scoring: one forward pass per batch, no text generation.

The prompts, tokenizer options, model options and yes/no scoring are unchanged
from the original heuristic. The candidate-selection algorithm lives in
heuristic_v3.py; this file only handles the model.
"""
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, lazy_symbol, no_grad, required_path
torch = LazyModule("torch")
AutoTokenizer = lazy_symbol("transformers", "AutoTokenizer")
AutoModelForCausalLM = lazy_symbol("transformers", "AutoModelForCausalLM")


def format_rerank_instruction(instruction: str, query: str, doc: str) -> str:
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )

def build_qwen_prefix_suffix(tokenizer):
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )

    suffix = (
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n"
        "</think>\n\n"
    )

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    return prefix_tokens, suffix_tokens

def process_qwen_rerank_inputs(
    tokenizer,
    pairs,
    instruction,
    max_length,
    prefix_tokens,
    suffix_tokens,
    device,
):
    formatted = [
        format_rerank_instruction(instruction=instruction, query=q, doc=d)
        for q, d in pairs
    ]

    enc = tokenizer(
        formatted,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )

    input_ids = []
    for ids in enc["input_ids"]:
        input_ids.append(prefix_tokens + ids + suffix_tokens)

    batch = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt",
    )

    batch = {k: v.to(device) for k, v in batch.items()}
    return batch

@no_grad
def compute_qwen_yes_probs(
    model,
    inputs,
    token_true_id,
    token_false_id,
):
    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :].float()

    true_logits = logits[:, token_true_id]
    false_logits = logits[:, token_false_id]

    pair_logits = torch.stack([false_logits, true_logits], dim=1)
    log_probs = torch.nn.functional.log_softmax(pair_logits, dim=1)
    probs_yes = log_probs[:, 1].exp()

    return probs_yes.detach().cpu().tolist()

@no_grad
def score_sent_texts_with_qwen_reranker(
    query: str,
    sent_texts: List[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    instruction: str,
    prefix_tokens,
    suffix_tokens,
    token_true_id: int,
    token_false_id: int,
    device: str,
) -> List[float]:
    pairs = [(query, t) for t in sent_texts]
    all_scores = []

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]

        batch_inputs = process_qwen_rerank_inputs(
            tokenizer=tokenizer,
            pairs=batch_pairs,
            instruction=instruction,
            max_length=max_length,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
        )

        batch_scores = compute_qwen_yes_probs(
            model=model,
            inputs=batch_inputs,
            token_true_id=token_true_id,
            token_false_id=token_false_id,
        )
        all_scores.extend(batch_scores)

    return all_scores

def load_qwen_reranker(model_path: str, device: str):
    model_path = str(required_path(model_path, "RERANK_MODEL", "dir"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    prefix_tokens, suffix_tokens = build_qwen_prefix_suffix(tokenizer)

    token_true_ids = tokenizer("yes", add_special_tokens=False).input_ids
    token_false_ids = tokenizer("no", add_special_tokens=False).input_ids

    if len(token_true_ids) != 1 or len(token_false_ids) != 1:
        raise ValueError(
            f"Expected single-token yes/no, got yes={token_true_ids}, no={token_false_ids}"
        )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "token_true_id": token_true_ids[0],
        "token_false_id": token_false_ids[0],
        "device": device,
    }
