"""
Training loop for the Pure VFE Transformer.

No optimizer. No loss.backward(). Just VFE descent.
"""

import time
import torch

from .config import PureVFEConfig
from .model import PureVFETransformer
from .gauge import monitor_omega_health


def load_wikitext2(seq_len=64, split='train'):
    """Load WikiText-2 dataset. Returns list of [seq_len] token id tensors."""
    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
    except ImportError:
        print("Install: pip install datasets transformers")
        raise

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    # Concatenate and tokenize
    text = '\n'.join([x for x in ds['text'] if x.strip()])
    token_ids = tokenizer.encode(text)
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # Chunk into sequences
    n_seqs = len(token_ids) // (seq_len + 1)
    token_ids = token_ids[:n_seqs * (seq_len + 1)]
    token_ids = token_ids.reshape(n_seqs, seq_len + 1)

    return token_ids  # Each row: [input_0, ..., input_{N-1}, target_{N-1}]


def make_batches(data, batch_size, device='cuda', shuffle=True):
    """Yield (input, target) batches."""
    n = data.shape[0]
    if shuffle:
        perm = torch.randperm(n)
        data = data[perm]

    for i in range(0, n - batch_size + 1, batch_size):
        batch = data[i:i + batch_size].to(device)
        inputs = batch[:, :-1]   # [B, N]
        targets = batch[:, 1:]   # [B, N]
        yield inputs, targets


def train(config=None, n_epochs=10, log_interval=10, save_path=None):
    """
    Train the Pure VFE Transformer on WikiText-2.

    Args:
        config: PureVFEConfig (uses defaults if None)
        n_epochs: number of training epochs
        log_interval: print every N steps
        save_path: path to save model checkpoints
    """
    if config is None:
        config = PureVFEConfig()

    print("=" * 60)
    print("Pure VFE Transformer — Training")
    print("=" * 60)
    print(f"  belief_dim={config.belief_dim}, n_heads={config.n_heads}, "
          f"head_dim={config.head_dim}")
    print(f"  n_esteps={config.n_esteps}, tau={config.tau:.2f}")
    print(f"  eta_E={config.eta_E}, eta_M={config.eta_M}")
    print(f"  max_seq_len={config.max_seq_len}, batch_size={config.batch_size}")
    print(f"  device={config.device}, cuda_kernels={config.use_cuda_kernels}")

    # Load data
    print("\nLoading WikiText-2...")
    data = load_wikitext2(seq_len=config.max_seq_len)
    print(f"  {data.shape[0]} sequences of length {config.max_seq_len}")

    # Create model
    model = PureVFETransformer(config)
    params = model.param_count()
    print(f"\nModel parameters: {params['total']:,}")
    for k, v in params.items():
        if k != 'total':
            print(f"  {k}: {v:,}")

    # Attempt to load CUDA kernels
    if config.use_cuda_kernels and config.device == 'cuda':
        from .cuda_ext import get_cuda_ext
        cuda_ext = get_cuda_ext()
        if cuda_ext:
            print("\n[CUDA kernels active]")
        else:
            print("\n[Falling back to PyTorch ops]")

    print("\n" + "-" * 60)
    print(f"{'Epoch':>5} {'Step':>6} {'Loss':>8} {'PPL':>10} "
          f"{'VFE_0':>10} {'VFE_f':>10} {'s/step':>8}")
    print("-" * 60)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for inputs, targets in make_batches(data, config.batch_size, config.device):
            t0 = time.time()

            logits, ce_loss, vfe_history, diag = model.update(inputs, targets)

            dt = time.time() - t0
            epoch_loss += ce_loss
            epoch_steps += 1
            global_step += 1

            if global_step % log_interval == 0:
                ppl = torch.exp(torch.tensor(ce_loss)).item()
                vfe_0 = vfe_history[0] if vfe_history else 0.0
                vfe_f = vfe_history[-1] if vfe_history else 0.0
                vfe_ratio = vfe_f / max(abs(vfe_0), 1e-8) if vfe_0 != 0 else 0.0
                print(f"{epoch:5d} {global_step:6d} {ce_loss:8.3f} {ppl:10.1f} "
                      f"{vfe_0:10.1f} {vfe_f:10.1f} {dt:8.3f}")

                # E-step gradient norms (final iteration)
                if diag['grad_norm_mu']:
                    print(f"  [GRAD] mu={diag['grad_norm_mu'][-1]:.2f} "
                          f"sigma={diag['grad_norm_sigma'][-1]:.2f} "
                          f"omega={diag['grad_norm_omega'][-1]:.2f}")
                if diag['nan_events'] > 0:
                    print(f"  [WARN] NaN recovery events: {diag['nan_events']}")

                # Monitor prior health
                with torch.no_grad():
                    # Sigma eigenvalues
                    sig_eigs = torch.linalg.eigvalsh(model.prior_Sigma[:100])
                    sig_min = sig_eigs[..., 0].min().item()
                    sig_max = sig_eigs[..., -1].max().item()

                    # Mu norms
                    mu_norms = model.prior_mu.norm(dim=-1)
                    mu_mean = mu_norms.mean().item()
                    mu_max = mu_norms.max().item()

                    # VFE convergence ratio
                    if vfe_ratio > 1.1:
                        print(f"  [WARN] VFE diverged: ratio={vfe_ratio:.3f}")
                    elif len(vfe_history) > 1:
                        print(f"  [INFO] VFE {vfe_0:.1f}→{vfe_f:.1f} "
                              f"({len(vfe_history)} steps, ratio={vfe_ratio:.3f})")

                    if sig_min < 0.05 or mu_max > 10.0:
                        print(f"  [WARN] Σ_min={sig_min:.4f} Σ_max={sig_max:.2f} "
                              f"μ_mean={mu_mean:.2f} μ_max={mu_max:.2f}")

                # Monitor gauge health
                health = monitor_omega_health(model.prior_Omega[:100], "prior_Omega")
                print(f"  [DIAG] Ω cond: mean={health['prior_Omega/cond_mean']:.1f} "
                      f"max={health['prior_Omega/cond_max']:.1f} "
                      f"det_range=[{health['prior_Omega/det_min']:.3f}, "
                      f"{health['prior_Omega/det_max']:.3f}]")
                if health['prior_Omega/cond_max'] > 100:
                    print(f"  [WARN] Ω condition number high: {health['prior_Omega/cond_max']:.1f}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"\n  Epoch {epoch} avg: loss={avg_loss:.3f} ppl={avg_ppl:.1f}")

        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            model.save(save_path)
            print(f"  Saved best model to {save_path}")

        print()

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Pure VFE Transformer')
    parser.add_argument('--belief-dim', type=int, default=32)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-esteps', type=int, default=12)
    parser.add_argument('--eta-E', type=float, default=0.1)
    parser.add_argument('--eta-M', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--no-cuda-kernels', action='store_true')
    args = parser.parse_args()

    config = PureVFEConfig(
        belief_dim=args.belief_dim,
        n_heads=args.n_heads,
        head_dim=args.belief_dim // args.n_heads,
        n_esteps=args.n_esteps,
        eta_E=args.eta_E,
        eta_M=args.eta_M,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        device=args.device,
        use_cuda_kernels=not args.no_cuda_kernels,
    )

    train(config, n_epochs=args.epochs, save_path=args.save)
