"""Standalone Qianfan-OCR extraction with the anaphora-aware prompt.

Codex 2026-05-09: minimal, throw-away script for the 1-doc A/B test.
- DOES NOT touch CachedExtractor / vlm_audit (those still serve the
  default canonical-only prompt and the shared vlm_cache/ directory).
- Writes to a separate cache directory so the legacy default cache
  is preserved and easy to diff against.

Cache layout matches load_vlm_cache's expected pattern::
    {cache_root}/{tag}/{doc_id}__{tag}__{prompt_hash}.json

Default cache_root = ``dockg_refiner/outputs/vlm_cache_anaphora_1doc``.
Default tag         = ``qianfan-llamacpp_seed0``.

Run record (one .run.log entry per doc) captures:
    max_new_tokens / n_ctx / prompt_variant / prompt_hash /
    _truncated / raw_output_chars / json_chars / wall_seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extractor.cached_extractor import _try_recover_truncated_json
from .extractor.llamacpp_backend import make_llamacpp_backend
from .extractor.schema import build_extraction_prompt_anaphora


def _list_image_paths(doc_dir: Path) -> List[str]:
    imgs = sorted(doc_dir.glob("image_p*.jpg"))
    return [str(p) for p in imgs]


def _extract_one(
    doc_dir: Path,
    backend,
    prompt: str,
    prompt_hash: str,
    backend_tag: str,
    cache_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    """Run the backend on one doc and write the parsed JSON to cache.

    Returns the run record (not the parsed JSON).
    """
    doc_id = doc_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{doc_id}__{backend_tag}__{prompt_hash}.json"

    if cache_path.exists() and not force:
        print(f"[skip] cache hit -> {cache_path.name}", flush=True)
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = {}
        return {
            "doc_id": doc_id,
            "cache_hit": True,
            "cache_path": str(cache_path),
            "_truncated": bool(cached.get("_truncated", False)),
            "json_chars": len(json.dumps(cached, ensure_ascii=False)),
        }

    image_paths = _list_image_paths(doc_dir)
    if not image_paths:
        raise FileNotFoundError(f"no images in {doc_dir}")

    t0 = time.time()
    raw = backend(image_paths, prompt)
    wall = time.time() - t0

    parsed: Dict[str, Any]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        recovered = _try_recover_truncated_json(raw)
        if recovered is not None:
            parsed = recovered
            parsed.setdefault("_truncated", True)
        else:
            parsed = {
                "_error": "json_decode",
                "_reason": str(exc),
                "_raw": raw,
            }

    cache_path.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "doc_id": doc_id,
        "cache_hit": False,
        "cache_path": str(cache_path),
        "_truncated": bool(parsed.get("_truncated", False)),
        "raw_output_chars": len(raw),
        "json_chars": len(json.dumps(parsed, ensure_ascii=False)),
        "wall_seconds": round(wall, 2),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anaphora-aware Qianfan extraction (1-doc A/B)"
    )
    p.add_argument("--dataset", required=True, type=Path,
                   help="Synthetic contract dataset root (contains doc_*).")
    p.add_argument("--doc_id", required=True, type=str,
                   help="Specific doc to extract (e.g. doc_000002).")
    p.add_argument("--cache_root", type=Path,
                   default=Path("dockg_refiner/outputs/vlm_cache_anaphora_1doc"),
                   help="Cache root; per-tag subdirs are created under it.")
    p.add_argument("--backend_tag", type=str, default="qianfan-llamacpp_seed0",
                   help="Tag used in cache filename and per-tag subdir.")
    p.add_argument("--seed", type=int, default=0)
    # Codex 2026-05-09 token budget for anaphora variant
    p.add_argument("--n_ctx", type=int, default=16384)
    p.add_argument("--max_new_tokens", type=int, default=12288)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n_gpu_layers", type=int, default=999)
    p.add_argument(
        "--model_path", type=str,
        default=r"C:\Users\sar\PycharmProjects\qianfan\Qianfan-OCR-q4_k_m.gguf",
    )
    p.add_argument(
        "--mmproj_path", type=str,
        default=r"C:\Users\sar\PycharmProjects\qianfan\qianfan-ocr-mmproj.gguf",
    )
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if cache file exists.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    doc_dir = args.dataset / args.doc_id
    if not doc_dir.exists():
        print(f"[FATAL] doc dir not found: {doc_dir}", file=sys.stderr)
        return 2

    prompt = build_extraction_prompt_anaphora()
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]

    cache_dir = args.cache_root / args.backend_tag

    print(f"[anaphora-extract] doc={args.doc_id} tag={args.backend_tag}", flush=True)
    print(f"  prompt_variant=anaphora  prompt_hash={prompt_hash}", flush=True)
    print(f"  n_ctx={args.n_ctx}  max_new_tokens={args.max_new_tokens}  "
          f"temp={args.temperature}  seed={args.seed}", flush=True)
    print(f"  cache_dir={cache_dir}", flush=True)

    backend = make_llamacpp_backend(
        model_path=args.model_path,
        mmproj_path=args.mmproj_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        verbose=False,
    )

    record = _extract_one(
        doc_dir=doc_dir,
        backend=backend,
        prompt=prompt,
        prompt_hash=prompt_hash,
        backend_tag=args.backend_tag,
        cache_dir=cache_dir,
        force=args.force,
    )

    run_meta = {
        "doc_id": args.doc_id,
        "prompt_variant": "anaphora",
        "prompt_hash": prompt_hash,
        "n_ctx": args.n_ctx,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "backend_tag": args.backend_tag,
        "cache_path": record["cache_path"],
        "_truncated": record["_truncated"],
        "json_chars": record["json_chars"],
        "raw_output_chars": record.get("raw_output_chars"),
        "wall_seconds": record.get("wall_seconds"),
        "cache_hit": record["cache_hit"],
    }
    log_path = cache_dir / f"{args.doc_id}__{args.backend_tag}.run.log"
    log_path.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== anaphora extraction summary ===", flush=True)
    for k, v in run_meta.items():
        print(f"  {k}: {v}")
    if record["_truncated"]:
        print("\n[WARN] _truncated=True — codex red line. Bump max_new_tokens.",
              file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
