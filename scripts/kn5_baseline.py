#!/usr/bin/env python3
"""
KN-5 Baseline Comparison for WikiText-103
==========================================

Builds a Modified Kneser-Ney 5-gram model (via KenLM) on WikiText-103
using the SAME GPT-2 BPE tokenization (50,257 vocab) as the gauge VFE model,
enabling an apples-to-apples perplexity comparison.

The Merity et al. (2017) KN-5 result (~153-156 PPL) uses word-level
tokenization with ~267K vocabulary. This script re-evaluates under
matched BPE tokenization so the comparison is commensurable.

Usage:
    python scripts/kn5_baseline.py

    # With custom n-gram order:
    python scripts/kn5_baseline.py --order 5

    # Skip install (if deps already present):
    python scripts/kn5_baseline.py --no-install

Requirements (auto-installed):
    - tiktoken (GPT-2 BPE tokenizer)
    - kenlm (Modified Kneser-Ney language model toolkit)
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# 0. Dependency management
# ---------------------------------------------------------------------------

def install_deps():
    """Install tiktoken and kenlm if missing."""
    missing = []
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        missing.append("tiktoken")
    try:
        import kenlm  # noqa: F401
    except ImportError:
        missing.append("kenlm")

    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        # kenlm needs to be built from source via pip
        for pkg in missing:
            cmd = [sys.executable, "-m", "pip", "install", pkg]
            print(f"  $ {' '.join(cmd)}")
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        print("Dependencies installed.\n")


# ---------------------------------------------------------------------------
# 1. Data loading — reuse project pipeline or standalone fallback
# ---------------------------------------------------------------------------

def load_wikitext103_splits():
    """
    Load WikiText-103 raw text for train/valid/test splits.

    Tries the project's data pipeline first, then HuggingFace datasets,
    then direct download.
    """
    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading WikiText-103 via HuggingFace datasets...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        splits = {}
        for name, key in [("train", "train"), ("valid", "validation"), ("test", "test")]:
            texts = [r["text"] for r in ds[key] if r["text"].strip()]
            splits[name] = "\n\n".join(texts)
            print(f"  {name}: {len(splits[name]):,} chars")
        return splits
    except Exception as e:
        print(f"  HuggingFace datasets not available ({e}), trying fallback...")

    # Fallback: use project's download utility
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transformer.data.datasets import _download_wikitext103_fallback
    print("Loading WikiText-103 via project fallback downloader...")
    raw = _download_wikitext103_fallback()
    splits = {
        "train": raw["train"],
        "valid": raw["validation"],
        "test": raw["test"],
    }
    for name in splits:
        print(f"  {name}: {len(splits[name]):,} chars")
    return splits


# ---------------------------------------------------------------------------
# 2. BPE tokenization — GPT-2 via tiktoken (same as gauge VFE)
# ---------------------------------------------------------------------------

def tokenize_splits(splits: dict) -> dict:
    """Tokenize each split with GPT-2 BPE. Returns dict of token-id lists."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    print(f"\nTokenizing with GPT-2 BPE (vocab = {enc.n_vocab:,})...")

    tokenized = {}
    for name, text in splits.items():
        t0 = time.time()
        ids = enc.encode(text)
        dt = time.time() - t0
        tokenized[name] = ids
        print(f"  {name}: {len(ids):,} tokens  ({dt:.1f}s)")
    return tokenized


# ---------------------------------------------------------------------------
# 3. Write token-string corpus for KenLM
# ---------------------------------------------------------------------------

def write_token_corpus(token_ids: list, path: str):
    """
    Write BPE token IDs as whitespace-separated "words" for KenLM.

    Each token ID becomes a string like "T42". KenLM treats each unique
    string as a word, so we get an n-gram model over BPE tokens.
    We split into pseudo-sentences of ~512 tokens to give KenLM
    natural sentence boundaries (improves estimation quality).
    """
    SENT_LEN = 512
    with open(path, "w") as f:
        for i in range(0, len(token_ids), SENT_LEN):
            chunk = token_ids[i : i + SENT_LEN]
            line = " ".join(f"T{t}" for t in chunk)
            f.write(line + "\n")
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Wrote {path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# 4. Build KN-5 model
# ---------------------------------------------------------------------------

def build_kenlm_model(train_corpus: str, arpa_path: str, binary_path: str,
                      order: int = 5):
    """
    Build a Modified Kneser-Ney n-gram model using KenLM's lmplz.

    Falls back to the kenlm Python package if lmplz binary is not on PATH.
    """
    import shutil

    lmplz = shutil.which("lmplz")
    build_binary = shutil.which("build_binary")

    if lmplz:
        # Use KenLM command-line tools (fastest)
        print(f"\nBuilding KN-{order} model via lmplz...")
        t0 = time.time()
        cmd = f"{lmplz} -o {order} < {train_corpus} > {arpa_path}"
        subprocess.check_call(cmd, shell=True)
        dt = time.time() - t0
        print(f"  ARPA model built in {dt:.1f}s")

        if build_binary:
            print("  Converting to binary...")
            subprocess.check_call([build_binary, arpa_path, binary_path])
            return binary_path
        return arpa_path
    else:
        # Use kenlm Python package — it bundles lmplz
        print(f"\nBuilding KN-{order} model via kenlm Python bindings...")
        print("  (lmplz not on PATH; using subprocess to kenlm's bundled binary)")

        # kenlm pip package installs lmplz into the package directory
        import kenlm as _kenlm
        kenlm_dir = Path(_kenlm.__file__).parent
        bundled_lmplz = kenlm_dir / "lmplz"
        bundled_build = kenlm_dir / "build_binary"

        # Also check common pip install locations
        pip_bin = Path(sys.executable).parent
        for candidate in [bundled_lmplz, pip_bin / "lmplz",
                          Path.home() / ".local" / "bin" / "lmplz"]:
            if candidate.exists():
                lmplz = str(candidate)
                break

        if lmplz:
            t0 = time.time()
            cmd = f"{lmplz} -o {order} < {train_corpus} > {arpa_path}"
            subprocess.check_call(cmd, shell=True)
            dt = time.time() - t0
            print(f"  ARPA model built in {dt:.1f}s")
        else:
            # Last resort: use kenlm's Python-level model training
            # This is slower but doesn't need the binary
            print("  WARNING: lmplz binary not found. Using Python-level estimation.")
            print("  For faster builds: pip install https://github.com/kpu/kenlm/archive/master.zip")
            _build_arpa_python(train_corpus, arpa_path, order)

        return arpa_path


def _build_arpa_python(train_corpus: str, arpa_path: str, order: int):
    """
    Pure-Python fallback: count n-grams and write ARPA format manually.

    This implements basic absolute-discounting (not full Modified KN)
    but gives a reasonable n-gram baseline when KenLM binaries aren't available.
    """
    from collections import Counter

    print("  Counting n-grams (pure Python)...")
    t0 = time.time()

    ngram_counts = {n: Counter() for n in range(1, order + 1)}
    total_unigrams = 0

    with open(train_corpus) as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            # Add BOS/EOS markers
            tokens = ["<s>"] + tokens + ["</s>"]
            total_unigrams += len(tokens)
            for n in range(1, order + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i : i + n])
                    ngram_counts[n][ngram] += 1

    dt = time.time() - t0
    for n in range(1, order + 1):
        print(f"    {n}-grams: {len(ngram_counts[n]):,} unique")
    print(f"  Counting took {dt:.1f}s")

    # Write ARPA format with absolute discounting (D=0.75)
    print("  Writing ARPA file...")
    D = 0.75

    # Precompute context counts for backoff
    context_counts = {}  # context -> total count
    context_types = {}   # context -> number of unique continuations
    for n in range(2, order + 1):
        for ngram, count in ngram_counts[n].items():
            ctx = ngram[:-1]
            context_counts[ctx] = context_counts.get(ctx, 0) + count
            context_types[ctx] = context_types.get(ctx, 0) + 1

    with open(arpa_path, "w") as f:
        f.write("\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(ngram_counts[n])}\n")
        f.write("\n")

        for n in range(1, order + 1):
            f.write(f"\\{n}-grams:\n")
            for ngram, count in sorted(ngram_counts[n].items()):
                if n == 1:
                    prob = count / total_unigrams
                    log_prob = math.log10(max(prob, 1e-10))
                    word = ngram[0]
                    # Unigrams get backoff weight
                    ctx = ngram
                    n_types = context_types.get(ctx, 1)
                    bow = math.log10(max(D * n_types / max(context_counts.get(ctx, count), 1), 1e-10))
                    f.write(f"{log_prob:.6f}\t{word}\t{bow:.6f}\n")
                else:
                    ctx = ngram[:-1]
                    ctx_count = context_counts.get(ctx, count)
                    discounted = max(count - D, 0) / max(ctx_count, 1)
                    log_prob = math.log10(max(discounted, 1e-10))
                    ngram_str = " ".join(ngram)
                    if n < order:
                        n_types = context_types.get(ngram, 1)
                        bow = math.log10(max(D * n_types / max(context_counts.get(ngram, 1), 1), 1e-10))
                        f.write(f"{log_prob:.6f}\t{ngram_str}\t{bow:.6f}\n")
                    else:
                        f.write(f"{log_prob:.6f}\t{ngram_str}\n")
            f.write("\n")

        f.write("\\end\\\n")

    size_mb = os.path.getsize(arpa_path) / 1e6
    print(f"  ARPA file: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# 5. Evaluate perplexity
# ---------------------------------------------------------------------------

def evaluate_perplexity(model_path: str, test_corpus: str, split_name: str = "test"):
    """
    Evaluate KN-5 perplexity on a corpus using KenLM's Python bindings.

    Computes per-token perplexity to match standard LM evaluation.
    """
    import kenlm

    print(f"\nEvaluating on {split_name} set...")
    model = kenlm.Model(model_path)
    print(f"  Model order: {model.order}")

    total_log_prob = 0.0
    total_tokens = 0
    total_oov = 0
    num_sentences = 0

    with open(test_corpus) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # KenLM scores full sentences (adds <s> and </s> implicitly)
            # full_scores gives (log10_prob, ngram_length, oov) per word
            words = line.split()
            num_sentences += 1

            for log10_prob, ngram_len, is_oov in model.full_scores(line):
                total_log_prob += log10_prob
                total_tokens += 1
                if is_oov:
                    total_oov += 1

    # KenLM includes </s> in scoring but not <s>
    # total_tokens includes </s> for each sentence
    # For fair comparison with neural LMs that don't predict </s>,
    # we subtract sentence count (removing </s> tokens)
    eval_tokens = total_tokens - num_sentences  # exclude </s>

    # Convert log10 to natural log for PPL
    avg_log10 = total_log_prob / eval_tokens
    ppl = 10 ** (-avg_log10)

    # Also compute with </s> included for reference
    ppl_with_eos = 10 ** (-total_log_prob / total_tokens)

    return {
        "ppl": ppl,
        "ppl_with_eos": ppl_with_eos,
        "total_tokens": eval_tokens,
        "total_tokens_with_eos": total_tokens,
        "oov_tokens": total_oov,
        "oov_rate": total_oov / total_tokens * 100,
        "num_sentences": num_sentences,
    }


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KN-5 baseline on WikiText-103 with matched BPE tokenization"
    )
    parser.add_argument("--order", type=int, default=5,
                        help="N-gram order (default: 5)")
    parser.add_argument("--no-install", action="store_true",
                        help="Skip automatic dependency installation")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Working directory for intermediate files "
                             "(default: scripts/kn5_workdir)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"KN-{args.order} BASELINE — WikiText-103 (GPT-2 BPE, vocab 50,257)")
    print("=" * 70)
    print()

    # Step 0: Install dependencies
    if not args.no_install:
        install_deps()

    # Set up working directory
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        workdir = Path(__file__).resolve().parent / "kn5_workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {workdir}\n")

    # Step 1: Load data
    t_start = time.time()
    splits = load_wikitext103_splits()

    # Step 2: Tokenize with GPT-2 BPE
    tokenized = tokenize_splits(splits)

    # Step 3: Write token corpora
    print("\nWriting token corpora for KenLM...")
    corpus_paths = {}
    for name, ids in tokenized.items():
        path = str(workdir / f"wt103_bpe_{name}.txt")
        write_token_corpus(ids, path)
        corpus_paths[name] = path

    # Step 4: Build KN model
    arpa_path = str(workdir / f"wt103_bpe_kn{args.order}.arpa")
    binary_path = str(workdir / f"wt103_bpe_kn{args.order}.binary")
    model_path = build_kenlm_model(
        corpus_paths["train"], arpa_path, binary_path, order=args.order
    )

    t_build = time.time() - t_start

    # Step 5: Evaluate
    results = {}
    for name in ["valid", "test"]:
        results[name] = evaluate_perplexity(model_path, corpus_paths[name], name)

    t_total = time.time() - t_start

    # Step 6: Report
    print("\n" + "=" * 70)
    print(f"RESULTS: KN-{args.order} on WikiText-103 (BPE vocab 50,257)")
    print("=" * 70)

    for name in ["valid", "test"]:
        r = results[name]
        print(f"\n  {name.upper()} SET:")
        print(f"    Perplexity:          {r['ppl']:.1f}")
        print(f"    Perplexity (w/ EOS): {r['ppl_with_eos']:.1f}")
        print(f"    Tokens evaluated:    {r['total_tokens']:,}")
        print(f"    OOV tokens:          {r['oov_tokens']:,} ({r['oov_rate']:.2f}%)")
        print(f"    Sentences:           {r['num_sentences']:,}")

    print(f"\n  Build time:  {t_build:.1f}s ({t_build/60:.1f} min)")
    print(f"  Total time:  {t_total:.1f}s ({t_total/60:.1f} min)")

    # Comparison table
    gauge_val = 108.9
    gauge_test = 121.1
    kn_val = results["valid"]["ppl"]
    kn_test = results["test"]["ppl"]

    print(f"\n{'=' * 70}")
    print("COMPARISON (matched BPE tokenization, vocab 50,257)")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'Val PPL':>10} {'Test PPL':>10}")
    print(f"{'-'*30} {'-'*10} {'-'*10}")
    print(f"{'KN-' + str(args.order) + ' (this script)':<30} {kn_val:>10.1f} {kn_test:>10.1f}")
    print(f"{'Gauge VFE (GL(20), K=80)':<30} {gauge_val:>10.1f} {gauge_test:>10.1f}")
    print(f"{'-'*30} {'-'*10} {'-'*10}")

    if kn_test > gauge_test:
        ratio = kn_test / gauge_test
        print(f"\n  Gauge VFE beats KN-{args.order} by {kn_test - gauge_test:.1f} PPL "
              f"({ratio:.2f}x)")
    else:
        ratio = gauge_test / kn_test
        print(f"\n  KN-{args.order} beats Gauge VFE by {gauge_test - kn_test:.1f} PPL "
              f"({ratio:.2f}x)")
        print(f"  (Under matched tokenization, the n-gram baseline is stronger)")

    print(f"\n  Merity et al. (2017) KN-5 (word-level, ~267K vocab): ~153-156 PPL")
    print(f"  This script KN-{args.order} (BPE, 50K vocab):          {kn_test:.1f} PPL")
    print(f"  Vocabulary difference accounts for "
          f"~{abs(kn_test - 154):.0f} PPL points")

    print(f"\n{'=' * 70}")
    print("Files saved:")
    print(f"  ARPA model: {arpa_path}")
    if os.path.exists(binary_path):
        print(f"  Binary model: {binary_path}")
    for name, path in corpus_paths.items():
        print(f"  {name} corpus: {path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
