#!/usr/bin/env python3
"""
Simple text generation script with publication-quality attention visualization.

Usage:
    python generate.py path/to/best_model.pt
"""

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from transformer.utils.checkpoint import load_model, get_tokenizer


def plot_attention_publication(
    beta: torch.Tensor,
    kl: torch.Tensor,
    tokens: list,
    save_path: str = "attention_pattern.pdf",
    title: str = None,
):
    """
    Create publication-quality attention pattern visualization.

    Args:
        beta: (n_heads, seq_len, seq_len) attention weights
        kl: (n_heads, seq_len, seq_len) KL divergences
        tokens: List of token strings
        save_path: Output path (.pdf recommended for publications)
        title: Optional title for the figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Publication settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'text.usetex': False,  # Set True if LaTeX is available
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    n_heads = beta.shape[0]
    seq_len = beta.shape[1]

    # Truncate long tokens for display
    display_tokens = []
    for t in tokens:
        t = t.replace('\n', '\\n').replace('\t', '\\t')
        if len(t) > 8:
            t = t[:6] + '..'
        display_tokens.append(t)

    # Choose layout based on number of heads
    if n_heads <= 4:
        # Show all heads + mean + KL
        n_cols = min(n_heads + 2, 4)
        n_rows = (n_heads + 2 + n_cols - 1) // n_cols
        fig_width = 3.5 * n_cols
        fig_height = 3.2 * n_rows
    else:
        # Show subset: heads 0, mid, last + mean + KL
        n_cols = 4
        n_rows = 2
        fig_width = 14
        fig_height = 7
        selected_heads = [0, n_heads // 2, n_heads - 1]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Color maps
    attn_cmap = 'Blues'
    kl_cmap = 'Oranges'

    def plot_matrix(ax, matrix, tokens, cmap, title, vmin=None, vmax=None, show_cbar=True):
        """Helper to plot a single attention matrix."""
        im = ax.imshow(
            matrix,
            cmap=cmap,
            aspect='equal',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )

        # Axis labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        ax.set_xlabel('Key (j)', fontsize=10)
        ax.set_ylabel('Query (i)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='medium')

        # Colorbar
        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=8)

        # Grid for clarity
        ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tokens), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)

        return im

    plot_idx = 0

    # Plot individual heads
    if n_heads <= 4:
        heads_to_plot = range(n_heads)
    else:
        heads_to_plot = selected_heads

    for h in heads_to_plot:
        if plot_idx >= len(axes):
            break
        plot_matrix(
            axes[plot_idx],
            beta[h].numpy(),
            display_tokens,
            attn_cmap,
            f'Head {h}: Attention β',
        )
        plot_idx += 1

    # Mean attention across heads
    if plot_idx < len(axes):
        mean_beta = beta.mean(dim=0).numpy()
        plot_matrix(
            axes[plot_idx],
            mean_beta,
            display_tokens,
            attn_cmap,
            'Mean Attention β',
        )
        plot_idx += 1

    # KL divergence (mean across heads, log scale for visibility)
    if plot_idx < len(axes):
        mean_kl = kl.mean(dim=0).numpy()
        # Log transform for better visualization
        mean_kl_log = np.log10(mean_kl + 1e-6)
        plot_matrix(
            axes[plot_idx],
            mean_kl_log,
            display_tokens,
            kl_cmap,
            'Mean KL Divergence (log₁₀)',
        )
        plot_idx += 1

    # Hide unused axes
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
    print(f"Saved publication-quality figure to: {save_path}")

    # Also save PNG for quick preview
    png_path = save_path.rsplit('.', 1)[0] + '.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved preview to: {png_path}")

    plt.close(fig)


def get_attention_data(model, tokenizer, text, device):
    """Extract attention patterns from model."""
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], device=device)

    with torch.no_grad():
        logits, attn_info = model.forward_with_attention(input_ids)

    # Decode tokens for labels
    tokens = []
    for i in range(len(token_ids)):
        try:
            tok = tokenizer.decode([token_ids[i]])
        except Exception:
            tok = f"[{token_ids[i]}]"
        tokens.append(tok)

    return {
        'beta': attn_info['beta'][0].cpu(),  # (n_heads, N, N)
        'kl': attn_info['kl'][0].cpu(),      # (n_heads, N, N)
        'tokens': tokens,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate.py <checkpoint_path>")
        print("Example: python generate.py experiments/best_model.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, config = load_model(checkpoint_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"Model loaded ({device})")

    # Load tokenizer
    tokenizer = get_tokenizer(config)
    if tokenizer is None:
        print("Error: Could not load tokenizer. Install tiktoken: pip install tiktoken")
        sys.exit(1)

    # Settings
    max_tokens = 50
    temperature = 0.8
    last_input = None

    print("\n" + "=" * 60)
    print("GAUGE TRANSFORMER TEXT GENERATION")
    print("=" * 60)
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print("\nCommands:")
    print("  quit          - Exit")
    print("  temp X        - Set temperature (e.g., temp 0.5)")
    print("  tokens X      - Set max tokens (e.g., tokens 100)")
    print("  attention     - Save attention pattern for last input")
    print("=" * 60)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text:
            continue

        if text.lower() == 'quit':
            print("Goodbye!")
            break

        if text.lower().startswith('temp '):
            try:
                temperature = float(text.split()[1])
                print(f"Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("Usage: temp 0.8")
            continue

        if text.lower().startswith('tokens '):
            try:
                max_tokens = int(text.split()[1])
                print(f"Max tokens set to {max_tokens}")
            except (IndexError, ValueError):
                print("Usage: tokens 50")
            continue

        if text.lower() == 'attention':
            if last_input is None:
                print("No input yet. Enter some text first.")
                continue
            try:
                print(f"Extracting attention for: \"{last_input}\"")
                attn_data = get_attention_data(model, tokenizer, last_input, device)
                plot_attention_publication(
                    attn_data['beta'],
                    attn_data['kl'],
                    attn_data['tokens'],
                    save_path="attention_pattern.pdf",
                    title=f"Gauge Transformer Attention: \"{last_input[:30]}{'...' if len(last_input) > 30 else ''}\"",
                )
            except ImportError:
                print("matplotlib required. Install with: pip install matplotlib")
            except Exception as e:
                print(f"Error generating attention plot: {e}")
            continue

        # Store for attention visualization
        last_input = text

        # Encode input
        token_ids = tokenizer.encode(text)
        input_ids = torch.tensor([token_ids], device=device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
            )

        # Decode and print
        output_text = tokenizer.decode(output_ids[0].tolist())
        generated = output_text[len(text):]

        print(f"\nModel: {text}\033[92m{generated}\033[0m")


if __name__ == '__main__':
    main()
