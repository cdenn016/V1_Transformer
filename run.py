#!/usr/bin/env python3
"""
Click-to-run Pure VFE Transformer.

Usage:
    python run.py                          # Synthetic data, CPU, immediate
    python run.py --data wikitext2         # WikiText-2 (needs: pip install datasets transformers)
    python run.py --device cuda            # GPU
    python run.py --belief-dim 64 --n-heads 8  # Larger model

No nn.Module. No autograd. No loss.backward().
Just variational free energy descent on a gauge-covariant prior bank.
"""

import argparse
import time
import sys

import torch

from transformer.pure_vfe.config import PureVFEConfig
from transformer.pure_vfe.model import PureVFETransformer
from transformer.pure_vfe.gauge import monitor_omega_health


# ── Synthetic data (zero external deps) ─────────────────────────────────────

def make_synthetic_data(vocab_size, seq_len, n_seqs=500):
    """
    Generate synthetic token sequences with local bigram structure.
    Each token biases the next toward a nearby vocabulary region,
    giving the model something non-trivial to learn.
    """
    data = torch.zeros(n_seqs, seq_len + 1, dtype=torch.long)
    for i in range(n_seqs):
        tok = torch.randint(0, vocab_size, (1,)).item()
        data[i, 0] = tok
        for t in range(1, seq_len + 1):
            # Next token is drawn from a window around the current token
            window = max(vocab_size // 20, 5)
            lo = max(0, tok - window)
            hi = min(vocab_size, tok + window)
            tok = torch.randint(lo, hi, (1,)).item()
            data[i, t] = tok
    return data


# ── WikiText-2 loader ───────────────────────────────────────────────────────

def load_wikitext2(seq_len):
    """Load WikiText-2. Requires: pip install datasets transformers"""
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    text = "\n".join([x for x in ds["text"] if x.strip()])
    token_ids = tokenizer.encode(text)
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    n_seqs = len(token_ids) // (seq_len + 1)
    token_ids = token_ids[: n_seqs * (seq_len + 1)]
    return token_ids.reshape(n_seqs, seq_len + 1)


# ── Batch iterator ──────────────────────────────────────────────────────────

def batches(data, batch_size, device, shuffle=True):
    n = data.shape[0]
    if shuffle:
        data = data[torch.randperm(n)]
    for i in range(0, n - batch_size + 1, batch_size):
        chunk = data[i : i + batch_size].to(device)
        yield chunk[:, :-1], chunk[:, 1:]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Pure VFE Transformer — click to run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data",
        choices=["synthetic", "wikitext2"],
        default="synthetic",
        help="Data source (default: synthetic — no extra deps)",
    )
    p.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument("--belief-dim", type=int, default=16, help="Belief dimension K")
    p.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--n-esteps", type=int, default=6, help="E-step iterations (depth)")
    p.add_argument("--eta-E", type=float, default=0.1, help="E-step learning rate")
    p.add_argument("--eta-M", type=float, default=0.001, help="M-step learning rate")
    p.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--vocab-size", type=int, default=256, help="Vocab size (synthetic only)")
    p.add_argument("--save", type=str, default=None, help="Path to save best checkpoint")
    p.add_argument("--log-interval", type=int, default=5, help="Print every N steps")
    args = p.parse_args()

    # ── Resolve device ──
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Load data ──
    if args.data == "wikitext2":
        try:
            print("Loading WikiText-2...")
            data = load_wikitext2(args.seq_len)
            vocab_size = 50257  # GPT-2 vocab
            print(f"  {data.shape[0]} sequences, vocab={vocab_size}")
        except ImportError:
            print("ERROR: WikiText-2 requires:  pip install datasets transformers")
            print("       Or run with --data synthetic (default)")
            sys.exit(1)
    else:
        vocab_size = args.vocab_size
        data = make_synthetic_data(vocab_size, args.seq_len)
        print(f"Synthetic data: {data.shape[0]} seqs, vocab={vocab_size}, seq_len={args.seq_len}")

    # ── Build config ──
    head_dim = args.belief_dim // args.n_heads
    config = PureVFEConfig(
        vocab_size=vocab_size,
        belief_dim=args.belief_dim,
        n_heads=args.n_heads,
        head_dim=head_dim,
        n_esteps=args.n_esteps,
        eta_E=args.eta_E,
        eta_M=args.eta_M,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        device=device,
        use_cuda_kernels=(device == "cuda"),
    )

    # ── Build model ──
    model = PureVFETransformer(config)
    params = model.param_count()

    print()
    print("=" * 60)
    print("  Pure VFE Transformer")
    print("=" * 60)
    print(f"  belief_dim={config.belief_dim}  n_heads={config.n_heads}  head_dim={config.head_dim}")
    print(f"  n_esteps={config.n_esteps}  tau={config.tau:.2f}")
    print(f"  eta_E={config.eta_E}  eta_M={config.eta_M}")
    print(f"  device={device}  params={params['total']:,}")
    for k, v in params.items():
        if k != "total":
            print(f"    {k}: {v:,}")
    print("=" * 60)
    print()
    print(f"{'Epoch':>5} {'Step':>6} {'Loss':>8} {'PPL':>10} "
          f"{'VFE_0':>10} {'VFE_f':>10} {'s/step':>8}")
    print("-" * 60)

    # ── Train ──
    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for inputs, targets in batches(data, config.batch_size, device):
            t0 = time.time()
            logits, ce_loss, vfe_history = model.update(inputs, targets)
            dt = time.time() - t0

            epoch_loss += ce_loss
            epoch_steps += 1
            global_step += 1

            if global_step % args.log_interval == 0:
                ppl = torch.exp(torch.tensor(ce_loss)).item()
                vfe_0 = vfe_history[0] if vfe_history else 0.0
                vfe_f = vfe_history[-1] if vfe_history else 0.0
                print(
                    f"{epoch:5d} {global_step:6d} {ce_loss:8.3f} {ppl:10.1f} "
                    f"{vfe_0:10.1f} {vfe_f:10.1f} {dt:8.3f}"
                )

                # Gauge health check
                health = monitor_omega_health(model.prior_Omega[:100], "Omega")
                if health["Omega/cond_max"] > 100:
                    print(f"  [WARN] condition number high: {health['Omega/cond_max']:.1f}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"\n  Epoch {epoch} summary: loss={avg_loss:.3f}  ppl={avg_ppl:.1f}\n")

        if args.save and avg_loss < best_loss:
            best_loss = avg_loss
            model.save(args.save)
            print(f"  Saved best model -> {args.save}\n")

    print("Done.")
    return model


if __name__ == "__main__":
    main()
