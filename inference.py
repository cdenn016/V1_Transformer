#!/usr/bin/env python3
"""
Model Inference & Qualitative Analysis
=======================================

Click-to-run inference for GaugeTransformerLM with belief-state
and KL-attention visualization.

Supports text generation, next-token probability inspection,
attention pattern extraction (beta/KL from the gauge-theoretic
attention mechanism), and belief state visualization (mu, sigma, phi).

Instructions:
    1. Set CHECKPOINT_PATH below to your best_model.pt
    2. Run this script (F5 in Spyder, or python inference.py)
    3. Type prompts interactively, or it will run example prompts first
"""

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

from pathlib import Path

CHECKPOINT_PATH = str(Path("checkpoints_publication") / "ffn_VFE_dynamic" / "best_model.pt")

# Set to None for auto-detect from checkpoint config
# Override: 'wikitext-2', 'wikitext-103', or 'wiki-ja'
DATASET = None

MAX_TOKENS = 50
TEMPERATURE = 0.8
NUM_SAMPLES = 3

# =============================================================================
# CODE - No need to edit below
# =============================================================================

import sys
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from transformer.utils.checkpoint import load_model, get_tokenizer


class GaugeTransformerInference:
    """Inference wrapper for GaugeTransformerLM.

    Provides generation, next-token probabilities, attention pattern
    extraction, and belief state inspection. Attention patterns include
    beta (KL-based weights) and pairwise KL divergences from the last layer.
    Belief states include mu (B, N, K), sigma, and phi (gauge parameters).
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None, dataset_name: Optional[str] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        print(f"Loading model from {checkpoint_path}...")
        # trusted=True for self-saved checkpoints (config dataclass + non-tensor objects).
        self.model, self.config = load_model(checkpoint_path, trusted=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Auto-detect dataset from config if not specified
        if dataset_name is None:
            dataset_name = self.config.get('dataset', 'wikitext-103')
        self.dataset_name = dataset_name
        self.is_japanese = (dataset_name == 'wiki-ja')

        print(f"Loading tokenizer ({dataset_name})...")
        self.tokenizer = get_tokenizer(self.config, dataset_name=dataset_name)
        if self.tokenizer is None:
            raise RuntimeError("Could not load tokenizer. Install tiktoken: pip install tiktoken")

        print(f"Model loaded: K={self.config['embed_dim']}, "
              f"{self.config['n_layers']} layers, "
              f"vocab={self.config['vocab_size']}")
        print(f"Device: {self.device}")

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text)
        return torch.tensor([token_ids], device=self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.squeeze().tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids)

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8,
                 top_k: Optional[int] = 40, top_p: Optional[float] = 0.9, num_samples: int = 1) -> List[str]:
        prompt_ids = self.encode(prompt)
        results = []
        for _ in range(num_samples):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    prompt_ids=prompt_ids.clone(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            results.append(self.decode(generated_ids[0]))
        return results

    def get_next_token_probs(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        token_ids = self.encode(text)
        with torch.no_grad():
            logits = self.model(token_ids)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)

        top_probs, top_ids = torch.topk(probs, min(top_k, probs.size(0)))
        results = []
        for prob, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
            try:
                token_str = self.decode(torch.tensor([tok_id]))
            except (KeyError, IndexError, RuntimeError):
                token_str = f"[{tok_id}]"
            results.append((token_str, prob))
        return results

    def get_attention_patterns(self, text: str) -> Dict[str, torch.Tensor]:
        """Extract attention beta, KL divergences, and mu from the last layer.

        Returns:
            Dict with 'beta' (n_heads, N, N), 'kl' (n_heads, N, N),
            'mu' (N, K), and 'tokens' (list of str).
        """
        token_ids = self.encode(text)
        with torch.no_grad():
            logits, attn_info = self.model.forward_with_attention(token_ids, targets=None)

        tokens = []
        for i in range(token_ids.shape[1]):
            try:
                tok_str = self.decode(token_ids[0, i:i+1])
                if len(tok_str) > 10:
                    tok_str = tok_str[:8] + ".."
            except (KeyError, IndexError, RuntimeError):
                tok_str = f"[{token_ids[0, i].item()}]"
            tokens.append(tok_str)

        return {
            'beta': attn_info['beta'][-1, 0].cpu(),
            'kl': attn_info['kl'][-1, 0].cpu(),
            'mu': attn_info['mu'][0].cpu(),
            'tokens': tokens,
        }

    def get_belief_states(self, text: str) -> Dict[str, torch.Tensor]:
        """Extract per-token belief states from a forward pass.

        Returns:
            Dict with 'mu' (N, K), 'sigma' (N, K) or (N, K, K) or None,
            'phi' (N, phi_dim), 'tokens' (list of str), 'logits' (N, vocab).
        """
        token_ids = self.encode(text)
        with torch.no_grad():
            logits, agent_states = self.model(token_ids, return_agents=True)

        tokens = []
        for i in range(token_ids.shape[1]):
            try:
                tok_str = self.decode(token_ids[0, i:i+1])
            except (KeyError, IndexError, RuntimeError):
                tok_str = f"[{token_ids[0, i].item()}]"
            tokens.append(tok_str)

        return {
            'mu': agent_states['mu'][0].cpu(),
            'sigma': agent_states['sigma'][0].cpu() if agent_states['sigma'] is not None else None,
            'phi': agent_states['phi'][0].cpu(),
            'tokens': tokens,
            'logits': logits[0].cpu(),
        }


def _setup_cjk_fonts(plt):
    """Configure matplotlib CJK font support if needed."""
    import matplotlib.font_manager as fm
    cjk_fonts = [
        'MS Gothic', 'Yu Gothic', 'Meiryo',
        'Noto Sans CJK JP', 'Noto Sans JP',
        'IPAGothic', 'IPAexGothic',
        'Hiragino Sans', 'Hiragino Kaku Gothic Pro',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in cjk_fonts:
        if font_name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False
            return


def visualize_attention(attn_data: Dict, save_path: Optional[str] = None, head_idx: int = 0,
                        figsize: Tuple[int, int] = (10, 8)):
    """Plot attention weights and KL divergences for a single head.

    Args:
        attn_data: Dict from get_attention_patterns() with 'beta', 'kl', 'tokens'.
        save_path: If provided, save figure to this path instead of displaying.
        head_idx: Which attention head to visualize.
        figsize: Figure dimensions.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    tokens = attn_data['tokens']
    if any(ord(c) > 0x2E80 for t in tokens for c in t):
        _setup_cjk_fonts(plt)

    beta = attn_data['beta']
    kl = attn_data['kl']
    n_heads = beta.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    im1 = ax1.imshow(beta[head_idx].numpy(), cmap='Blues', aspect='auto')
    ax1.set_title(f'Attention Weights (Head {head_idx})')
    ax1.set_xlabel('Key Position (j)')
    ax1.set_ylabel('Query Position (i)')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_yticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(tokens, fontsize=8)
    plt.colorbar(im1, ax=ax1, label='beta_ij')

    ax2 = axes[1]
    kl_plot = kl[head_idx].numpy()
    kl_plot = np.clip(kl_plot, 1e-6, None)
    im2 = ax2.imshow(np.log10(kl_plot + 1), cmap='Reds', aspect='auto')
    ax2.set_title(f'KL Divergences (Head {head_idx})')
    ax2.set_xlabel('Key Position (j)')
    ax2.set_ylabel('Query Position (i)')
    ax2.set_xticks(range(len(tokens)))
    ax2.set_yticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(tokens, fontsize=8)
    plt.colorbar(im2, ax=ax2, label='log10(KL + 1)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_beliefs(belief_data: Dict, save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 4)):
    """Plot belief means (mu) heatmap and per-token norm bar chart.

    Args:
        belief_data: Dict from get_belief_states() with 'mu' (N, K) and 'tokens'.
        save_path: If provided, save figure to this path instead of displaying.
        figsize: Figure dimensions.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        return

    tokens = belief_data['tokens']
    if any(ord(c) > 0x2E80 for t in tokens for c in t):
        _setup_cjk_fonts(plt)

    mu = belief_data['mu']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    im1 = ax1.imshow(mu.numpy().T, aspect='auto', cmap='RdBu_r')
    ax1.set_title('Belief Means (mu)')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Embedding Dimension')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im1, ax=ax1)

    ax2 = axes[1]
    norms = torch.norm(mu, dim=-1).numpy()
    ax2.bar(range(len(tokens)), norms)
    ax2.set_title('Belief Norms ||mu||')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('L2 Norm')
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved belief visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def interactive_mode(inference: GaugeTransformerInference):
    """Interactive text generation mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Commands:")
    print("  /quit       - Exit")
    print("  /temp N     - Set temperature (e.g., /temp 0.5)")
    print("  /tokens N   - Set max tokens (e.g., /tokens 100)")
    print("  /samples N  - Set num samples (e.g., /samples 5)")
    print("  /attention  - Show attention for last input")
    print("  /probs      - Show next-token probabilities")
    print("=" * 70)

    temperature = 0.8
    max_tokens = 50
    num_samples = 1
    last_input = None

    while True:
        try:
            user_input = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.startswith('/'):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == '/quit':
                print("Goodbye!")
                break
            elif cmd == '/temp' and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature")
            elif cmd == '/tokens' and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                except ValueError:
                    print("Invalid token count")
            elif cmd == '/samples' and len(parts) > 1:
                try:
                    num_samples = int(parts[1])
                    print(f"Num samples set to {num_samples}")
                except ValueError:
                    print("Invalid sample count")
            elif cmd == '/attention' and last_input:
                print("\nExtracting attention patterns...")
                attn_data = inference.get_attention_patterns(last_input)
                visualize_attention(attn_data)
            elif cmd == '/probs' and last_input:
                probs = inference.get_next_token_probs(last_input, top_k=10)
                print("\nNext token probabilities:")
                for tok, prob in probs:
                    bar = "=" * int(prob * 40)
                    print(f"  {prob:5.1%} {bar} '{tok}'")
            else:
                print("Unknown command or missing argument")
            continue

        last_input = user_input
        print(f"\nGenerating ({num_samples} sample(s), temp={temperature})...")

        samples = inference.generate(
            user_input,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )

        for i, sample in enumerate(samples, 1):
            prompt_end = len(user_input)
            generated = sample[prompt_end:] if len(sample) > prompt_end else ""
            if num_samples > 1:
                print(f"\n[{i}] {user_input}\033[92m{generated}\033[0m")
            else:
                print(f"\n{user_input}\033[92m{generated}\033[0m")


def main():
    # Check checkpoint
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        print("\nPlease edit CHECKPOINT_PATH at the top of this file.")
        return

    # Load model
    inference = GaugeTransformerInference(
        checkpoint_path=str(checkpoint_path),
        dataset_name=DATASET,
    )

    # Run example prompts first
    if inference.is_japanese:
        prompts = ["日本", "東京は", "歴史的に", "科学者たちは", "世界の"]
    else:
        prompts = ["The", "In the beginning", "Scientists have discovered",
                    "The meaning of life is", "Once upon a time, there was"]

    print("\n" + "=" * 70)
    print("GENERATION EXAMPLES")
    print(f"Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}, Samples: {NUM_SAMPLES}")
    print("=" * 70)

    for prompt in prompts:
        print(f"\n{'~' * 60}")
        print(f"PROMPT: \"{prompt}\"")
        print(f"{'~' * 60}")

        # Next-token probabilities
        top_tokens = inference.get_next_token_probs(prompt, top_k=5)
        print("\nNext token probabilities:")
        for tok, prob in top_tokens:
            bar = "=" * int(prob * 30)
            print(f"  {prob:5.1%} {bar} '{tok}'")

        # Generate samples
        print(f"\nGenerated ({NUM_SAMPLES} samples):")
        samples = inference.generate(
            prompt,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            num_samples=NUM_SAMPLES,
        )
        for i, sample in enumerate(samples, 1):
            generated = sample[len(prompt):] if len(sample) > len(prompt) else ""
            print(f"  [{i}] {prompt}\033[92m{generated}\033[0m")

    # Drop into interactive mode
    print("\n" + "=" * 70)
    print("Now entering interactive mode...")
    interactive_mode(inference)


if __name__ == '__main__':
    main()
