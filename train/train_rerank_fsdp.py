#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import random
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seer_config import LazyModule, lazy_symbol, required_path
torch = LazyModule("torch")
dist = LazyModule("torch.distributed")
Dataset = lazy_symbol("datasets", "Dataset")
AutoConfig = lazy_symbol("transformers", "AutoConfig")
AutoTokenizer = lazy_symbol("transformers", "AutoTokenizer")
SFTConfig = lazy_symbol("trl", "SFTConfig")
SFTTrainer = lazy_symbol("trl", "SFTTrainer")

DEFAULT_SYSTEM_PROMPT = """You are an evidence reranker for retrieval-augmented question answering.

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

VALID_BUCKETS = {"core", "support", "drop"}


def is_main_process() -> bool:
    return os.environ.get("RANK") in (None, "0")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_id}: {e}")
    return rows


def list_data_files(data_path: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]

    if os.path.isdir(data_path):
        files = sorted(
            glob.glob(os.path.join(data_path, "*.json"))
            + glob.glob(os.path.join(data_path, "*.jsonl"))
        )
        if not files:
            raise ValueError(f"No .json or .jsonl files found in directory: {data_path}")
        return files

    raise ValueError(f"data_path does not exist: {data_path}")


def extract_examples_from_obj(obj: Any, source_name: str) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "examples" in obj and isinstance(obj["examples"], list):
            return obj["examples"]
        raise ValueError(f"{source_name}: JSON object must contain key 'examples'.")

    if isinstance(obj, list):
        return obj

    raise ValueError(f"{source_name}: unsupported JSON structure.")


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def validate_output_text_schema(output_text: str) -> bool:
    """Validate the compact, label-only SFT target derived from R3 topk JSON.

    Expected: {"results": [{"doc_id": 1, "sent_id": 2, "bucket": "core"}, ...]}
    Neither reason nor a meaningful score is preserved in archived teacher data.
    """
    try:
        obj = json.loads(output_text)
    except (TypeError, ValueError):
        return False

    if not isinstance(obj, dict) or set(obj) != {"results"}:
        return False

    results = obj["results"]
    if not isinstance(results, list) or not results:
        return False

    seen = set()
    for item in results:
        if not isinstance(item, dict) or set(item) != {"doc_id", "sent_id", "bucket"}:
            return False
        if type(item["doc_id"]) is not int or type(item["sent_id"]) is not int:
            return False
        if not isinstance(item["bucket"], str) or item["bucket"] not in VALID_BUCKETS:
            return False
        key = (item["doc_id"], item["sent_id"])
        if key in seen:
            return False
        seen.add(key)

    return True


def try_extract_messages(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    messages-only path, aligned to new gold_sft data.

    Expected:
    [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
    """
    messages = ex.get("messages", None)
    if not isinstance(messages, list) or len(messages) < 3:
        return None

    role_to_content: Dict[str, str] = {}
    for m in messages:
        if not isinstance(m, dict):
            return None
        role = m.get("role", None)
        content = m.get("content", None)
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        role_to_content[role] = content

    system = DEFAULT_SYSTEM_PROMPT  # R3 original-query / label-only SFT task; ignore stale systems.
    user = role_to_content.get("user", "").strip()
    assistant = role_to_content.get("assistant", "").strip()

    if not user or not assistant:
        return None

    if not validate_output_text_schema(assistant):
        return None

    return {
        "system": system if system else DEFAULT_SYSTEM_PROMPT,
        "user": user,
        "assistant": assistant,
    }


def normalize_one_example(ex: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, str]]:
    """
    Messages-only normalization.
    SFT JSON is constructed separately from archived candidate_pool/topk pairs.
    Do NOT read topk files here or fall back to legacy input/output fields.
    """
    msg_pack = try_extract_messages(ex)
    if msg_pack is None:
        return None

    ex_id = ex.get("sample_idx", fallback_id)
    dataset_name = ex.get("dataset", "rerank_topk")

    return {
        "id": f"{dataset_name}_{ex_id}",
        "system": msg_pack["system"],
        "user": msg_pack["user"],
        "assistant": msg_pack["assistant"],
    }


def load_training_examples(data_path: str, max_examples: int = -1) -> List[Dict[str, str]]:
    files = list_data_files(data_path)

    all_examples: List[Dict[str, str]] = []
    bad_count = 0

    for fp in files:
        ext = os.path.splitext(fp)[1].lower()

        if ext == ".json":
            obj = read_json(fp)
            raw_examples = extract_examples_from_obj(obj, fp)
        elif ext == ".jsonl":
            raw_examples = read_jsonl(fp)
        else:
            continue

        for i, ex in enumerate(raw_examples):
            if not isinstance(ex, dict):
                bad_count += 1
                continue

            normed = normalize_one_example(ex, fallback_id=f"{os.path.basename(fp)}_{i}")
            if normed is None:
                bad_count += 1
                continue

            all_examples.append(normed)

    if max_examples != -1:
        all_examples = all_examples[: max(0, max_examples)]

    if is_main_process():
        print(f"[Data] loaded valid examples: {len(all_examples)}")
        print(f"[Data] skipped invalid examples: {bad_count}")

    if not all_examples:
        raise ValueError(
            "No valid training examples found. "
            "This script is messages-only and expects the new rerank schema."
        )

    return all_examples


def has_chat_template(tok: AutoTokenizer) -> bool:
    return bool(getattr(tok, "chat_template", None))


def fallback_serialize(messages: List[Dict[str, str]]) -> str:
    out = []
    for m in messages:
        role = m["role"].upper()
        out.append(f"<|{role}|>\n{m['content']}\n")
    return "\n".join(out)


def apply_chat_template_no_thinking(tok: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    if not has_chat_template(tok):
        return fallback_serialize(messages)

    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def build_messages(example: Dict[str, str]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ]
    }


def format_dataset(ds: Dataset, tok: AutoTokenizer) -> Dataset:
    ds = ds.map(build_messages)

    def _apply(batch):
        texts = []
        for msgs in batch["messages"]:
            texts.append(apply_chat_template_no_thinking(tok, msgs))
        return {"text": texts}

    return ds.map(_apply, batched=True, remove_columns=ds.column_names)


def resolve_attn_impl(attn_impl: str) -> Optional[str]:
    raw = (attn_impl or "auto").strip().lower()
    if raw in ("", "auto", "none"):
        return None

    alias_map = {
        "fa2": "flash_attention_2",
        "flash-attn-2": "flash_attention_2",
        "flash_attn_2": "flash_attention_2",
    }
    raw = alias_map.get(raw, raw)

    allowed = {"eager", "sdpa", "flash_attention_2"}
    if raw not in allowed:
        raise ValueError(f"Unsupported attn_impl={attn_impl}. Choose from: auto/eager/sdpa/flash_attention_2")

    if raw == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as e:
            if is_main_process():
                print(f"[Warn] flash_attention_2 requested but flash_attn import failed: {e}")
                print("[Warn] Falling back to eager attention for stability.")
            return "eager"

    return raw


def resolve_fsdp_wrap_cls(model_id: str, override: str) -> str:
    raw = (override or "auto").strip()
    if raw and raw.lower() != "auto":
        return raw

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    archs = getattr(cfg, "architectures", None) or []

    if model_type == "qwen3" or any("Qwen3" in a for a in archs):
        return "Qwen3DecoderLayer"

    if model_type == "qwen2" or any("Qwen2" in a for a in archs):
        return "Qwen2DecoderLayer"

    if model_type == "llama" or any("Llama" in a for a in archs):
        return "LlamaDecoderLayer"

    if model_type == "mistral" or any("Mistral" in a for a in archs):
        return "MistralDecoderLayer"

    raise ValueError(
        f"Cannot auto-resolve FSDP wrap class for model_type={model_type}, architectures={archs}. "
        "Pass --fsdp_wrap_cls manually, e.g. --fsdp_wrap_cls Qwen3DecoderLayer."
    )


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True,
                   help="Path to a .json/.jsonl file OR a directory containing them.")
    p.add_argument("--model_id", type=str, required=True,
                   help="HF model id or local model path.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save checkpoints and final model.")
    p.add_argument("--tokenizer_id", type=str, default="",
                   help="Tokenizer path/id. If empty, fall back to model_id.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_examples", type=int, default=-1)

    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--logging_steps", type=int, default=10)

    p.add_argument("--packing", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--force_grad_ckpt", action="store_true")

    p.add_argument("--attn_impl", type=str, default="eager",
                   help="auto | eager | sdpa | flash_attention_2")
    p.add_argument("--fsdp_wrap_cls", type=str, default="auto",
                   help="auto | Qwen3DecoderLayer | Qwen2DecoderLayer | LlamaDecoderLayer | ...")

    p.add_argument("--save_strategy", choices=["no", "epoch", "steps"], default="no")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=1)

    return p.parse_args()


def main():
    args = get_args()
    for name in ("data_path", "model_id", "output_dir"):
        required_path(getattr(args, name), f"--{name}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU not available.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    if world_size > 1 and args.grad_ckpt and not args.force_grad_ckpt:
        if is_main_process():
            print("[Warn] Disabling HF gradient checkpointing under multi-GPU FSDP.")
            print("[Warn] Reason: this setup is unstable in your environment.")
        args.grad_ckpt = False

    attn_impl = resolve_attn_impl(args.attn_impl)
    fsdp_wrap_cls = resolve_fsdp_wrap_cls(args.model_id, args.fsdp_wrap_cls)

    if is_main_process():
        print(f"[FSDP] world_size={world_size} | local_rank={local_rank} | device={device}")
        print("[Task] rerank top-k evidence SFT for Qwen3-8B")
        print(f"[Config] max_len={args.max_len}")
        print(f"[Config] batch_size={args.batch_size}")
        print(f"[Config] grad_accum={args.grad_accum}")
        print(f"[Config] global_batch={args.batch_size * world_size * args.grad_accum}")
        print(f"[Config] lr={args.lr}")
        print(f"[Config] attn_impl={attn_impl or 'model_default'}")
        print(f"[Config] grad_ckpt={args.grad_ckpt}")
        print(f"[Config] packing={args.packing}")
        print(f"[Config] fsdp_wrap_cls={fsdp_wrap_cls}")
        print(f"[Config] save_strategy={args.save_strategy}")

    examples = load_training_examples(args.data_path, max_examples=args.max_examples)
    train_ds = Dataset.from_list(examples).shuffle(seed=args.seed)

    tok_source = args.tokenizer_id if args.tokenizer_id else args.model_id
    tok = AutoTokenizer.from_pretrained(tok_source, use_fast=True, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_ds = format_dataset(train_ds, tok)

    if is_main_process():
        print("-" * 100)
        print("[Preview] first sample head (1600 chars):")
        print(train_ds[0]["text"][:1600])
        print("-" * 100)

    report_to = "wandb" if os.environ.get("WANDB_PROJECT") else "none"

    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "trust_remote_code": True,
    }
    if attn_impl is not None:
        model_init_kwargs["attn_implementation"] = attn_impl

    fsdp_config = {
        "transformer_layer_cls_to_wrap": [fsdp_wrap_cls],
        "backward_prefetch": "backward_post",
        "forward_prefetch": False,
        "limit_all_gathers": True,
        "use_orig_params": True,
        "sync_module_states": True,
        "cpu_ram_efficient_loading": True,
        "activation_checkpointing": True,
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    }

    sft_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_len,
        dataset_text_field="text",
        bf16=True,
        logging_steps=args.logging_steps,
        logging_first_step=True,

        model_init_kwargs=model_init_kwargs,

        fsdp=["full_shard", "auto_wrap"],
        fsdp_config=fsdp_config,

        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        save_only_model=True,

        packing=args.packing,
        report_to=report_to,
        run_name=os.path.basename(args.output_dir.rstrip("/")),
    )

    if args.grad_ckpt:
        sft_kwargs["gradient_checkpointing"] = True
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": True}
    else:
        sft_kwargs["gradient_checkpointing"] = False

    sft_cfg = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=args.model_id,
        processing_class=tok,
        train_dataset=train_ds,
        args=sft_cfg,
    )

    if is_main_process():
        print(f"[Train] starting Qwen3 rerank SFT with FSDP... report_to={report_to}")

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        print(f"[Done] model/tokenizer saved to {args.output_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
