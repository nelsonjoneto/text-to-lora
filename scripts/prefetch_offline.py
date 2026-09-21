#!/usr/bin/env python
"""Prefetch everything this repo touches so it can run with HF_HUB_OFFLINE=1.

    python scripts/prefetch_offline.py                 # default: mistral base only
    python scripts/prefetch_offline.py --bases all     # all three base models
    python scripts/prefetch_offline.py --verify        # check offline mode actually works
    python scripts/prefetch_offline.py --bases llama --skip-datasets   # top up one base later

Datasets are fetched through `load_dataset` with the exact kwargs the repo uses, not
via a bare repo download: the datasets cache is keyed by (path, name, split), so a
plain file download can still leave `load_dataset` wanting network at run time.
"""
import argparse, glob, os, sys, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

BASES = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma":   "google/gemma-2-2b-it",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
}
T2L_CHECKPOINTS = {  # base -> subdir in the SakanaAI/text-to-lora repo
    "mistral": "trained_t2l/mistral_7b_t2l",
    "gemma":   "trained_t2l/gemma_2b_t2l",
    "llama":   "trained_t2l/llama_8b_t2l",
}
EMB_MODEL = "Alibaba-NLP/gte-large-en-v1.5"

# Benchmarks, with the exact load kwargs used at eval time. piqa needs the Hub's
# parquet conversion because datasets>=4 refuses to run its loading script.
BENCHMARKS = [
    ("openai/gsm8k",            dict(name="main", split="test")),
    ("google/boolq",            dict(split="validation")),
    ("allenai/winogrande",      dict(name="winogrande_debiased", split="validation")),
    ("ybisk/piqa",              dict(split="validation", revision="refs/convert/parquet")),
    ("Rowan/hellaswag",         dict(split="validation")),
    ("allenai/ai2_arc",         dict(name="ARC-Easy", split="test")),
    ("allenai/ai2_arc",         dict(name="ARC-Challenge", split="test")),
    ("allenai/openbookqa",      dict(split="test")),
    ("openai/openai_humaneval", dict(split="test")),
    # mbpp has configs ['full','sanitized'] and NO default, so an unnamed config is
    # ambiguous offline. The repo actually reads MBPP via evalplus, not HF, but cache
    # both configs explicitly so nothing can surprise us later.
    ("google-research-datasets/mbpp", dict(name="full", split="test")),
    ("google-research-datasets/mbpp", dict(name="sanitized", split="test")),
]
# Also pulled by the training-time splits (train[:500]) the quick-eval path uses.
BENCHMARK_TRAIN_SPLITS = [
    ("Rowan/hellaswag", dict(split="train[:500]")),
    ("allenai/winogrande", dict(name="winogrande_debiased", split="train[:500]")),
    ("google/boolq", dict(split="train[:500]")),
    ("ybisk/piqa", dict(split="train[:500]", revision="refs/convert/parquet")),
    ("allenai/ai2_arc", dict(name="ARC-Easy", split="validation[:500]")),
    ("allenai/ai2_arc", dict(name="ARC-Challenge", split="validation[:500]")),
    ("allenai/openbookqa", dict(split="validation[:500]")),
]


def lol_datasets():
    """The ~500 Super-NaturalInstructions tasks, read from the repo's own metadata."""
    import yaml
    out = {}
    for f in glob.glob("tasks/*/metadata.yaml"):
        try:
            kw = (yaml.safe_load(open(f)) or {}).get("ds_kwargs") or {}
            path = kw.get("path")
            if path and "/" in path:
                out[(path, kw.get("name"), kw.get("split"))] = True
        except Exception:
            pass
    return sorted(out)


def fetch_dataset(path, name=None, split=None, revision=None, retries=1):
    """Fetch one dataset. The Hub rate-limits bulk pulls, so retry with backoff."""
    import datasets, random, time
    kw = {k: v for k, v in dict(name=name, split=split, revision=revision).items() if v}
    last = None
    for attempt in range(retries):
        try:
            datasets.load_dataset(path, **kw)
            return f"{path} {kw.get('name','')} {kw.get('split','')}".strip()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                # exponential backoff with jitter; 429s need real time, not a quick retry
                time.sleep(min(60, 2 ** attempt * 5) + random.uniform(0, 3))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["mistral"],
                    choices=list(BASES) + ["all"], help="which base models to fetch")
    ap.add_argument("--workers", type=int, default=3,
                    help="keep low: 8 tripped Hub rate limiting at ~330/510")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-datasets", action="store_true",
                    help="models and checkpoints only (for topping up a gated base later)")
    ap.add_argument("--verify", action="store_true",
                    help="set HF_HUB_OFFLINE=1 and confirm everything loads from cache")
    args = ap.parse_args()

    if not os.path.exists("pyproject.toml"):
        sys.exit("run this from the text-to-lora repo root")

    if args.verify:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("=== verifying with HF_HUB_OFFLINE=1 ===")
        bad = []
        for path, kw in BENCHMARKS:
            try:
                fetch_dataset(path, **kw)
            except Exception as e:
                bad.append(f"{path} {kw}: {type(e).__name__}")
        lol = lol_datasets()
        for path, name, split in lol:
            try:
                fetch_dataset(path, name, split)
            except Exception as e:
                bad.append(f"{path}: {type(e).__name__}")
        print(f"  benchmarks + {len(lol)} lol datasets checked")
        if bad:
            print(f"  {len(bad)} NOT available offline:")
            for b in bad[:15]:
                print("   ", b)
            sys.exit(1)
        print("  all datasets load offline")
        return

    bases = list(BASES) if "all" in args.bases else args.bases
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    if not args.skip_models:
        from huggingface_hub import snapshot_download
        print("=== models ===")
        for b in bases:
            print(f"  {BASES[b]} ...", flush=True)
            snapshot_download(BASES[b], ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5"])
        print(f"  {EMB_MODEL} ...", flush=True)
        snapshot_download(EMB_MODEL)

        print("=== T2L checkpoints ===")
        for b in bases:
            sub = T2L_CHECKPOINTS[b]
            print(f"  {sub} ...", flush=True)
            try:
                snapshot_download("SakanaAI/text-to-lora", local_dir=".",
                                  allow_patterns=[f"{sub}/*"])
            except Exception as e:
                print(f"    skipped ({type(e).__name__})")

    if args.skip_datasets:
        print("(skipping datasets)")
        return

    print("=== benchmark datasets ===")
    for path, kw in BENCHMARKS + BENCHMARK_TRAIN_SPLITS:
        try:
            print("  ", fetch_dataset(path, **kw), flush=True)
        except Exception as e:
            print(f"   FAILED {path} {kw}: {type(e).__name__}: {str(e)[:90]}")

    print("=== evalplus (HumanEval+/MBPP+) ===")
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        get_human_eval_plus(); get_mbpp_plus()
        print("   cached")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {str(e)[:90]}")

    lol = lol_datasets()
    print(f"=== Lots-of-LoRAs task datasets ({len(lol)}, ~0.7GB total) ===")
    done = 0
    errors = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_dataset, p, n, s, retries=args.retries): p for p, n, s in lol}
        for f in as_completed(futs):
            try:
                f.result(); done += 1
            except Exception as e:
                errors.setdefault(type(e).__name__, []).append((futs[f], str(e)[:120]))
            n_done = done + sum(len(v) for v in errors.values())
            if n_done % 50 == 0:
                print(f"   {n_done}/{len(lol)}  ok={done} failed={n_done-done}", flush=True)
    failed = sum(len(v) for v in errors.values())
    print(f"   done: {done} cached, {failed} failed")
    for kind, items in sorted(errors.items(), key=lambda kv: -len(kv[1])):
        print(f"   {len(items):4d}x {kind}")
        print(f"        e.g. {items[0][0]}: {items[0][1]}")
    if failed:
        print("   re-run to retry - already-cached datasets are skipped quickly")

    print()
    print("Now switch to offline with:")
    print("   export HF_HUB_OFFLINE=1")
    print("Verify with:")
    print("   python scripts/prefetch_offline.py --verify")


if __name__ == "__main__":
    main()
