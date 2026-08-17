"""Download a tokenizer from the HuggingFace Hub into the local tokenizer folder.

Some checkpoints ship a `tokenizer_config.json` written against a newer transformers
version than the one pinned by lmbench. When loading fails with a known compatibility
error, this script applies an in-place fix and retries. Currently handled:

- `extra_special_tokens` as a list -> dict (list members like `<|video|>` become
  `{"<name>_token": "<|<name>|>"}`). Required for some Gemma-family tokenizers under
  transformers < 5.

Usage:
    uv run python scripts/download_hf_tokenizer.py google/gemma-3-27b-it
    uv run python scripts/download_hf_tokenizer.py google/gemma-3-27b-it --dest ./tokenizer
"""

import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

TOKENIZER_FILE_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "chat_template.jinja",
]


def derive_local_name(model_id: str) -> str:
    """Derive a local tokenizer directory name from a HF model id.

    Examples:
        `google/gemma-3-27b-it` -> `gemma-3-27b-it`
        `meta-llama/Llama-3.1-70B-Instruct` -> `llama-3.1-70b-instruct`
    """
    last = model_id.split("/")[-1]
    sanitized = re.sub("[^a-zA-Z0-9]", "-", last)
    sanitized = re.sub("-+", "-", sanitized).strip("-")
    return sanitized.lower()


def patch_extra_special_tokens(config_path: Path) -> bool:
    """Convert `extra_special_tokens` from list to dict if needed.

    Returns True if the file was modified, False otherwise.
    """
    if not config_path.exists():
        return False
    cfg = json.loads(config_path.read_text())
    val = cfg.get("extra_special_tokens")
    if not isinstance(val, list):
        return False

    converted: dict[str, str] = {}
    for i, tok in enumerate(val):
        match = re.fullmatch(r"<\|(\w+)\|>", tok) if isinstance(tok, str) else None
        key = f"{match.group(1)}_token" if match else f"extra_token_{i}"
        converted[key] = tok
    cfg["extra_special_tokens"] = converted
    config_path.write_text(json.dumps(cfg, indent=2))
    return True


def try_load(path: Path) -> Exception | None:
    """Attempt to load the tokenizer. Returns the caught exception or None on success."""
    try:
        AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    except Exception as e:
        return e
    return None


def is_extra_special_tokens_list_error(err: Exception) -> bool:
    """Identify the `extra_special_tokens` list-vs-dict failure from transformers."""
    return isinstance(err, AttributeError) and "'list' object has no attribute 'keys'" in str(err)


def main() -> int:
    """Entry point: download, verify, and optionally patch a HuggingFace tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", help="HuggingFace model id, e.g. google/gemma-3-27b-it")
    parser.add_argument(
        "--dest",
        default="tokenizer",
        help="Destination directory for tokenizers (default: ./tokenizer)",
    )
    args = parser.parse_args()

    name = derive_local_name(args.model_id)
    dest = Path(args.dest) / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer for '{args.model_id}' -> {dest}")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(dest),
        allow_patterns=TOKENIZER_FILE_PATTERNS,
    )

    err = try_load(dest)
    if err is None:
        print(f"OK: tokenizer loads from {dest}")
        return 0

    if is_extra_special_tokens_list_error(err):
        print("Detected `extra_special_tokens` list-vs-dict incompatibility; patching...")
        if patch_extra_special_tokens(dest / "tokenizer_config.json"):
            err = try_load(dest)
            if err is None:
                print(f"OK: patched tokenizer loads from {dest}")
                return 0

    print(f"FAILED: {type(err).__name__}: {err}", file=sys.stderr)
    print(
        "No known patch covers this error. Inspect tokenizer_config.json manually or extend this script.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
