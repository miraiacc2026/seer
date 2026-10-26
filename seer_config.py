"""Deployment configuration only; importing this module has no side effects."""
import importlib
import os
from functools import wraps
from pathlib import Path


def required_path(value, label, kind=None):
    """Reject unset paths before Path('') can silently select the current directory."""
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required; supply its CLI argument or environment variable")
    path = Path(value).expanduser()
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    if kind == "output" and path.exists() and not path.is_dir():
        raise ValueError(f"{label} must be an output directory: {path}")
    return path


def env_path(name, kind=None):
    return required_path(os.environ.get(name), name, kind)


def cli_path(value):
    return required_path(value, "path argument")


def configure_paths(args, names):
    """Resolve required CLI paths (or corresponding UPPER_CASE environment variables)."""
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is None or not str(value).strip():
                value = os.environ.get(name.upper())
            kind = "dir" if name in {"candidate_dir", "topk_dir", "query_struct_dir", "pool_dir", "summary_dir"} else None
            if name in {"src", "input_json", "gold_json", "eval_py", "topk", "query_struct", "input_jsonl"}:
                kind = "file"
            setattr(args, name, required_path(value, f"--{name} / {name.upper()}", kind))
    return args


def api_configuration(mode=None):
    """Select credentials by explicit mode or exact configured endpoint equality."""
    proxy = os.environ.get("PROXY_BASE_URL", "").strip().rstrip("/")
    official = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    selected = mode or os.environ.get("API_MODE", "").strip().lower()
    if selected and selected not in {"proxy", "openai"}:
        raise ValueError("API_MODE must be proxy or openai")
    effective = proxy if selected == "proxy" else (official or (proxy if not selected else ""))
    is_proxy = selected == "proxy" or bool(proxy and effective == proxy)
    if is_proxy and not proxy:
        raise ValueError("PROXY_BASE_URL is required for proxy API calls")
    key_name = "PROXY_API_KEY" if is_proxy else "OPENAI_API_KEY"
    key = os.environ.get(key_name, "").strip()
    if not key:
        raise EnvironmentError(f"{key_name} is required for {'proxy' if is_proxy else 'openai'} API calls")
    return key, effective or None


def api_client(mode=None):
    key, base_url = api_configuration(mode)
    from openai import OpenAI
    return OpenAI(
        api_key=key, base_url=base_url,
        timeout=float(os.environ.get("OPENAI_TIMEOUT", "120")),
        max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "2")),
    )


def validate_optional_paths(args):
    """An absent optional path is valid; a whitespace-only path never means cwd."""
    for name, value in vars(args).items():
        if value is not None and value != "" and isinstance(value, (str, Path)) and (
            name.endswith(("_dir", "_path", "_json", "_jsonl", "_out", "_txt"))
            or name in {"src", "out", "topk", "query_struct", "tsv", "local_structure"}
        ):
            required_path(value, f"--{name}")


def require_batch():
    if os.environ.get("API_SUPPORTS_BATCH", "0").strip() != "1":
        raise ValueError("Native Batch requires API_SUPPORTS_BATCH=1 for the configured endpoint; use run_chat otherwise")


class LazyModule:
    """Keep CLI help and configuration validation independent of model packages."""
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        return getattr(importlib.import_module(self.name), attr)


def lazy_symbol(module, name):
    class Symbol:
        def __getattr__(self, attr):
            return getattr(getattr(importlib.import_module(module), name), attr)

        def __call__(self, *args, **kwargs):
            return getattr(importlib.import_module(module), name)(*args, **kwargs)
    return Symbol()


def no_grad(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch
        with torch.no_grad():
            return function(*args, **kwargs)
    return wrapped
