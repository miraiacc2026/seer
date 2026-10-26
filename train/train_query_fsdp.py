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
Dataset = lazy_symbol("datasets", "Dataset")
AutoConfig = lazy_symbol("transformers", "AutoConfig")
AutoTokenizer = lazy_symbol("transformers", "AutoTokenizer")
SFTConfig = lazy_symbol("trl", "SFTConfig")
SFTTrainer = lazy_symbol("trl", "SFTTrainer")

DEFAULT_SYSTEM_PROMPT = (
    "You are a query-understanding module for evidence retrieval in a RAG pipeline. "
    "Do not answer the question. Output compact JSON only. "
    "Prefer high-recall, query-grounded retrieval signals."
)

REQUIRED_TOP_KEYS = {
    "anchor_entities",
    "anchor_aliases",
    "anchor_terms",
    "relation_terms",
    "relation_synonym",
    "relation_expansion",
    "relation_markers",
    "scope_terms",
    "attributes",
    "retrieval_control",
    "retrieval_views",
}

REQUIRED_RETRIEVAL_CONTROL_KEYS = {
    "relation_markers",
    "answerability_markers",
    "parallel_cues",
}

VALID_VIEW_NAMES = {"literal", "entity_expanded", "semantic"}


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


def _is_list_of_str(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def _is_dict_str_to_list_str(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    for k, v in x.items():
        if not isinstance(k, str):
            return False
        if not _is_list_of_str(v):
            return False
    return True


def validate_output_text_schema(output_text: str) -> bool:
    try:
        obj = json.loads(output_text)
    except Exception:
        return False

    if not isinstance(obj, dict):
        return False

    if not REQUIRED_TOP_KEYS.issubset(set(obj.keys())):
        return False

    if not _is_list_of_str(obj.get("anchor_entities")):
        return False
    if not _is_dict_str_to_list_str(obj.get("anchor_aliases")):
        return False
    if not _is_list_of_str(obj.get("anchor_terms")):
        return False
    if not _is_list_of_str(obj.get("relation_terms")):
        return False
    if not _is_list_of_str(obj.get("relation_synonym")):
        return False
    if not _is_list_of_str(obj.get("relation_expansion")):
        return False
    if not _is_list_of_str(obj.get("relation_markers")):
        return False
    if not _is_list_of_str(obj.get("scope_terms")):
        return False

    attrs = obj.get("attributes")
    if not isinstance(attrs, list):
        return False
    for item in attrs:
        if not isinstance(item, dict):
            return False
        needed = {"attribute", "bridge_triggers", "hypothesis_triggers", "time_or_scope"}
        if not needed.issubset(set(item.keys())):
            return False
        if not isinstance(item["attribute"], str):
            return False
        if not _is_list_of_str(item["bridge_triggers"]):
            return False
        if not _is_list_of_str(item["hypothesis_triggers"]):
            return False
        if not _is_list_of_str(item["time_or_scope"]):
            return False

    rc = obj.get("retrieval_control")
    if not isinstance(rc, dict):
        return False
    if not REQUIRED_RETRIEVAL_CONTROL_KEYS.issubset(set(rc.keys())):
        return False
    if not _is_list_of_str(rc.get("relation_markers")):
        return False
    if not _is_list_of_str(rc.get("answerability_markers")):
        return False
    if not _is_list_of_str(rc.get("parallel_cues")):
        return False

    views = obj.get("retrieval_views")
    if not isinstance(views, list) or len(views) == 0:
        return False

    seen_view_names = set()
    for item in views:
        if not isinstance(item, dict):
            return False
        if "view_name" not in item or "text" not in item:
            return False
        if not isinstance(item["view_name"], str):
            return False
        if not isinstance(item["text"], str):
            return False
        if item["view_name"] not in VALID_VIEW_NAMES:
            return False
        seen_view_names.add(item["view_name"])

    if not VALID_VIEW_NAMES.issubset(seen_view_names):
        return False

    return True


def try_extract_messages(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
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

    system = role_to_content.get("system", "").strip()
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
    msg_pack = try_extract_messages(ex)
    if msg_pack is None:
        return None

    ex_id = ex.get("sample_idx", fallback_id)
    dataset_name = ex.get("dataset", "query_expansion")

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
            "This script is messages-only; every example must have valid messages."
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


def apply_chat_template_no_thinking(tok: AutoTokenizer, msgs: List[Dict[str, str]]) -> str:
    if not has_chat_template(tok):
        return fallback_serialize(msgs)

    try:
        return tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            msgs,
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


def infer_fsdp_wrap_cls(model_id: str) -> str:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()

    if "qwen3" in model_type:
        return "Qwen3DecoderLayer"
    if "qwen2" in model_type:
        return "Qwen2DecoderLayer"
    if "llama" in model_type:
        return "LlamaDecoderLayer"
    if "mistral" in model_type:
        return "MistralDecoderLayer"

    return "Qwen3DecoderLayer"


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_examples", type=int, default=-1)

    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--logging_steps", type=int, default=10)

    p.add_argument("--packing", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--use_flash_attn", action="store_true")

    p.add_argument("--save_strategy", choices=["no", "epoch", "steps"], default="epoch")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--fsdp_wrap_cls", type=str, default="auto")

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

    if is_main_process():
        print(f"[FSDP] world_size={world_size} | local_rank={local_rank} | device={device}")
        print("[Task] query expansion SFT for Qwen3-8B")

    examples = load_training_examples(args.data_path, max_examples=args.max_examples)
    train_ds = Dataset.from_list(examples).shuffle(seed=args.seed)

    tok = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_ds = format_dataset(train_ds, tok)

    if is_main_process():
        print("-" * 100)
        print("[Preview] first sample head:")
        print(train_ds[0]["text"][:1200])
        print("-" * 100)

    report_to = "wandb" if os.environ.get("WANDB_PROJECT") else "none"

    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "use_cache": False,
    }
    if args.use_flash_attn:
        model_init_kwargs["attn_implementation"] = "flash_attention_2"

    wrap_cls = infer_fsdp_wrap_cls(args.model_id) if args.fsdp_wrap_cls == "auto" else args.fsdp_wrap_cls

    if is_main_process():
        print(f"[FSDP] transformer_layer_cls_to_wrap={wrap_cls}")

    fsdp_config = {
        "transformer_layer_cls_to_wrap": [wrap_cls],
        "activation_checkpointing": bool(args.grad_ckpt),
        "backward_prefetch": "backward_pre",
        "limit_all_gathers": True,
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    }

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_len,
        dataset_text_field="text",
        bf16=True,
        logging_steps=args.logging_steps,

        model_init_kwargs=model_init_kwargs,

        fsdp=["full_shard", "auto_wrap"],
        fsdp_config=fsdp_config,

        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        save_only_model=True,

        packing=args.packing,
        report_to=report_to,
        run_name=os.path.basename(args.output_dir.rstrip("/")),
        remove_unused_columns=True,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=args.model_id,
        processing_class=tok,
        train_dataset=train_ds,
        args=sft_cfg,
    )

    if is_main_process():
        print(f"[Train] starting Qwen3 query expansion SFT with FSDP... report_to={report_to}")

    trainer.train()
    trainer.save_model(args.output_dir)

    if trainer.is_world_process_zero():
        tok.save_pretrained(args.output_dir)
        print(f"[Done] model/tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
