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


DEFAULT_SYSTEM_PROMPT = """You are a citation-faithful final-answer planner and writer for retrieval-augmented QA.

Given the ORIGINAL question and the supplied core, support, and packed evidence,
produce an attribution plan and the FINAL citation-grounded answer directly.
This stage directly produces the complete answer. Use ONLY the supplied
evidence; do not invent facts or use external knowledge, query-structure
fields, or gold QA answers.

Primary objective: preserve every distinct, supported, answer-relevant relation.
- Answer the question directly. Do not merely summarize the topic or document.
- Do not drop a supported answer branch because another seems more likely.
- A single evidence sentence or DOC_ID may support several distinct relations.
- Preserve alternative interpretations, dates, event stages, versions, locations,
  roles, authorities, counting conventions, scopes, and standards with their
  distinguishing qualifiers. Do not merge incompatible branches.
- Retain supported partial answers when other parts remain unresolved.
- Preserve evidence-grounded answer phrases: exact names, full dates, numbers,
  titles, coordinated lists, and their relations/qualifiers. Do not shorten,
  reorder, round, or loosely paraphrase answer-bearing expressions.
- Do not infer relationships absent from the evidence.

Return STRICT JSON with EXACTLY three top-level keys:
{
  "plan_style": "parallel",
  "plan": [{"claim": "a supported answer relation", "citations": ["2"]}],
  "final_summary": "A complete supported answer [2]."
}

Plan rules:
- "direct" means one supported answer relation (possibly a coordinated list).
- "parallel" means two or more distinct supported answer relations, scopes,
  interpretations, or event stages. Prefer "parallel" when multiple exist.
- Each claim must express an answer-bearing factual relation with necessary
  qualifiers, with non-empty citations listing supporting DOC_ID strings.
- Include every distinct supported relation in plan, even if claims cite the
  same DOC_ID. A coordinated list sharing one relation can remain one claim.
- Preserve the planned relations, phrases, and qualifiers in final_summary.

Citation rules:
- Cite DOC_IDs only, in forms like [5] or [5][1]. Never cite sentence IDs.
- Every sentence of final_summary must end with supporting DOC_ID citations.
- Every cited relation must be supported by its cited evidence; never invent IDs.
- Do not emit markdown, extra JSON keys, comments, or text outside JSON."""

VALID_PLAN_STYLES = {"direct", "parallel"}


def is_main_process() -> bool:
    return os.environ.get("RANK") in (None, "0")


def normalize_ws(text: str) -> str:
    return " ".join((text or "").strip().split())


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
            glob.glob(os.path.join(data_path, "*.json")) +
            glob.glob(os.path.join(data_path, "*.jsonl"))
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


def _is_list_of_citation_strings(x: Any) -> bool:
    # GPT-4o S3 parser preserves citations as non-empty strings.
    if not isinstance(x, list) or not x:
        return False
    return all(isinstance(item, str) and bool(item.strip()) for item in x)


def validate_summary_output_schema_light(output_text: str) -> bool:
    """Validate the ACTUAL S3 teacher output without truncating or fabricating plan items.

    Teacher parse_summary_json(use_plan=True) may legitimately return an empty
    plan after cleaning; it retains final_summary and plan_style in that case.
    """
    try:
        obj = json.loads(output_text)
    except (TypeError, ValueError):
        return False

    if not isinstance(obj, dict) or set(obj) != {"plan_style", "plan", "final_summary"}:
        return False
    if not isinstance(obj["plan_style"], str) or obj["plan_style"] not in VALID_PLAN_STYLES:
        return False
    if not _is_nonempty_str(obj["final_summary"]):
        return False

    plan = obj["plan"]
    if not isinstance(plan, list):
        return False
    # Check ALL claims, rather than stopping on the first acceptable item.
    # An empty plan is preserved as-is, not replaced with a synthetic claim.
    for item in plan:
        if not isinstance(item, dict) or set(item) != {"claim", "citations"}:
            return False
        if not _is_nonempty_str(item["claim"]):
            return False
        if not _is_list_of_citation_strings(item["citations"]):
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

    system = DEFAULT_SYSTEM_PROMPT  # Always use the current S3/A4 final-answer task.
    user = role_to_content.get("user", "").strip()
    assistant = role_to_content.get("assistant", "").strip()

    if not user or not assistant:
        return None

    if not validate_summary_output_schema_light(assistant):
        return None

    return {
        "system": system,
        "user": user,
        "assistant": assistant,
    }


def normalize_one_example(ex: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, str]]:
    if not isinstance(ex, dict):
        return None

    msg_pack = try_extract_messages(ex)
    if msg_pack is None:
        return None

    ex_id = ex.get("sample_idx", fallback_id)
    dataset_name = ex.get("dataset", "summary")

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

            normed = normalize_one_example(
                ex=ex,
                fallback_id=f"{os.path.basename(fp)}_{i}",
            )
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
            "This script expects the new summary gold_sft messages schema."
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
    """
    Qwen3 supports enable_thinking=False in its chat template.
    For non-Qwen tokenizers or older templates, fall back safely.
    """
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
            txt = apply_chat_template_no_thinking(tok, msgs)
            texts.append(txt)
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
        raise ValueError(
            f"Unsupported attn_impl={attn_impl}. "
            "Choose from: auto/eager/sdpa/flash_attention_2"
        )

    if raw == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as e:
            if is_main_process():
                print(f"[Warn] flash_attention_2 requested but flash_attn import failed: {e}")
                print("[Warn] Falling back to eager attention for stability.")
            return "eager"

    return raw


def infer_fsdp_wrap_cls(model_id: str, requested: str = "auto") -> str:
    requested = (requested or "auto").strip()

    if requested and requested.lower() != "auto":
        return requested

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "") or ""
    architectures = getattr(cfg, "architectures", None) or []

    model_type_l = model_type.lower()
    arch_text = " ".join(architectures).lower()

    if "qwen3" in model_type_l or "qwen3" in arch_text:
        return "Qwen3DecoderLayer"

    if "qwen2" in model_type_l or "qwen2" in arch_text:
        return "Qwen2DecoderLayer"

    if "llama" in model_type_l or "llama" in arch_text:
        return "LlamaDecoderLayer"

    if "mistral" in model_type_l or "mistral" in arch_text:
        return "MistralDecoderLayer"

    return "Qwen3DecoderLayer"


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to a .json/.jsonl file OR a directory containing them.",
    )
    p.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HF model id or local model path.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and final model.",
    )
    p.add_argument(
        "--tokenizer_id",
        type=str,
        default="",
        help="Tokenizer path/id. If empty, fall back to model_id.",
    )

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

    p.add_argument(
        "--attn_impl",
        type=str,
        default="eager",
        help="auto | eager | sdpa | flash_attention_2",
    )

    p.add_argument(
        "--fsdp_wrap_cls",
        type=str,
        default="auto",
        help="auto | Qwen3DecoderLayer | Qwen2DecoderLayer | LlamaDecoderLayer | ...",
    )

    p.add_argument(
        "--save_strategy",
        choices=["no", "epoch", "steps"],
        default="no",
        help="Use no for final-only save.",
    )
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
    fsdp_wrap_cls = infer_fsdp_wrap_cls(args.model_id, args.fsdp_wrap_cls)

    if is_main_process():
        print(f"[FSDP] world_size={world_size} | local_rank={local_rank} | device={device}")
        print("[Task] summary SFT for Qwen3-8B")
        print(f"[Config] model_id={args.model_id}")
        print(f"[Config] tokenizer_id={args.tokenizer_id or '<use model_id>'}")
        print(f"[Config] max_len={args.max_len}")
        print(f"[Config] batch_size={args.batch_size}")
        print(f"[Config] grad_accum={args.grad_accum}")
        print(f"[Config] global_batch={args.batch_size * world_size * args.grad_accum}")
        print(f"[Config] lr={args.lr}")
        print(f"[Config] attn_impl={attn_impl or 'model_default'}")
        print(f"[Config] grad_ckpt={args.grad_ckpt}")
        print(f"[Config] packing={args.packing}")
        print(f"[FSDP] transformer_layer_cls_to_wrap={fsdp_wrap_cls}")
        print(f"[Save] save_strategy={args.save_strategy} | final save always enabled")

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
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        packing=args.packing,
        report_to=report_to,
        run_name=os.path.basename(args.output_dir.rstrip("/")),
    )

    if args.save_strategy == "steps":
        sft_kwargs["save_steps"] = args.save_steps

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
        print(f"[Train] starting summary SFT with FSDP... report_to={report_to}")

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        print(f"[Done] model/tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
