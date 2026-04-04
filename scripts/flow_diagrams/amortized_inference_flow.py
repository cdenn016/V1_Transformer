"""
Amortized Inference Mode -- Architecture Flow Diagram

Publication-quality "Attention is All You Need" style flow diagram for the
amortized_inference mode of the Gauge-Transformer.

Click-to-run: edit config dict below, then execute.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ══════════════════════════════════════════════════════════════════════
# Config -- edit here, then press Run
# ══════════════════════════════════════════════════════════════════════
CONFIG = {
    'output_dir': 'figures/flow_diagrams',
    'filename': 'amortized_inference_flow',
    'formats': ['pdf', 'png'],
    'dpi': 300,
}

# ══════════════════════════════════════════════════════════════════════
# Okabe-Ito colorblind-safe palette
# ══════════════════════════════════════════════════════════════════════
C = {
    'embed':    '#56B4E9',
    'pos':      '#93D3F5',
    'attn':     '#E69F00',
    'vfe':      '#009E73',
    'vfe_bg':   '#B8E8D8',
    'add':      '#F0E442',
    'norm':     '#E0E0E0',
    'out':      '#D55E00',
    'grad':     '#CC79A7',
    'softmax':  '#F5F5F5',
    'border':   '#555555',
    'edge':     '#333333',
    'txt':      '#000000',
    'dim':      '#777777',
    'math':     '#555555',
    'detach':   '#CC0000',
}

# ══════════════════════════════════════════════════════════════════════
# Layout constants
# ══════════════════════════════════════════════════════════════════════
CX = 5.8        # center x of main flow column
W_MAIN = 3.4    # width of main-level boxes
W_SUB = 2.8     # width of sub-boxes inside block
W_ADD = 1.6     # width of Add boxes
H_BOX = 0.50    # standard box height
H_ATTN = 0.72   # attention box height
H_VFE = 1.50    # VFE box height
H_NORM = 0.36   # norm box height
H_ADD = 0.36    # add box height
SP = 0.32       # standard vertical spacing between components
SP_SM = 0.20    # small spacing


# ══════════════════════════════════════════════════════════════════════
# Drawing primitives
# ══════════════════════════════════════════════════════════════════════

def box(ax, cx, cy, w, h, label, fc, fs=9, fw='normal', tc='black',
        ec=C['edge'], lw=0.8, alpha=1.0, zorder=3):
    """Rounded rectangle with centered label."""
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.07', facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    if label:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=tc, zorder=zorder + 1)
    return cy - h / 2, cy + h / 2  # y_bot, y_top


def arrow(ax, x0, y0, x1, y1, c=C['edge'], lw=1.1, ls='-', zorder=2):
    """Simple straight arrow."""
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='->,head_width=0.12,head_length=0.07',
        color=c, linewidth=lw, linestyle=ls, zorder=zorder,
        mutation_scale=11)
    ax.add_patch(a)


def side_text(ax, cx, cy, text, side='right', offset=0.15, fs=7, c=C['dim'],
              style='italic', fw='normal'):
    """Annotation to left or right of a box."""
    ha = 'left' if side == 'right' else 'right'
    sign = 1 if side == 'right' else -1
    ax.text(cx + sign * offset, cy, text, ha=ha, va='center',
            fontsize=fs, color=c, style=style, fontweight=fw)


# ══════════════════════════════════════════════════════════════════════
# Main diagram
# ══════════════════════════════════════════════════════════════════════

def create_diagram():
    fig, ax = plt.subplots(figsize=(11.0, 15.5))
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 15.5)
    ax.set_aspect('equal')
    ax.axis('off')

    y = 0.55  # cursor moves upward

    # ─── 1. Input ─────────────────────────────────────────────
    ax.text(CX, y, 'Input Token IDs', ha='center', va='center',
            fontsize=10.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#AAAAAA', lw=0.6))
    side_text(ax, CX + W_MAIN / 2, y, r'$(B,\;N)$')

    y += 0.50
    arrow(ax, CX, y, CX, y + SP)
    y += SP

    # ─── 2. Gauge Token Embedding ─────────────────────────────
    _, y_top = box(ax, CX, y + H_BOX / 2, W_MAIN, H_BOX + 0.06,
                   'Gauge Token Embedding', C['embed'], fs=10, fw='bold')
    y_embed_center = y + H_BOX / 2
    side_text(ax, CX + W_MAIN / 2, y_embed_center,
              r'$\rightarrow\;\mu,\;\Sigma,\;\phi$   $(B, N, K)$')
    y = y_top

    y += 0.08
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ─── 3. Prior Storage ─────────────────────────────────────
    prior_h = 0.58
    prior_cy = y + prior_h / 2
    box(ax, CX, prior_cy, 3.0, prior_h, '', '#F6F6F6',
        ec='#BBBBBB', lw=0.5, alpha=0.9)
    ax.text(CX - 0.05, prior_cy + 0.11,
            r'$\mu_p = \mu_q.\mathrm{clone}()$', ha='center', va='center',
            fontsize=7.5, color=C['grad'], fontweight='bold', zorder=5)
    ax.text(CX + 1.1, prior_cy + 0.11,
            r'live $\nabla$', ha='left', va='center',
            fontsize=6.5, color=C['grad'], fontweight='bold', zorder=5)
    ax.text(CX - 0.05, prior_cy - 0.12,
            r'$\sigma_p = \sigma_q.\mathrm{clone}()$', ha='center', va='center',
            fontsize=7.5, color='#999999', zorder=5)
    ax.text(CX + 1.1, prior_cy - 0.12,
            r'detached $\oslash$', ha='left', va='center',
            fontsize=6.5, color=C['detach'], fontweight='bold', zorder=5)
    y = prior_cy + prior_h / 2

    y += 0.08
    arrow(ax, CX, y, CX, y + SP)
    y += SP

    # ─── 4. Positional Gauge Encoding ─────────────────────────
    pos_cy = y + H_BOX / 2
    _, y_top = box(ax, CX, pos_cy, W_MAIN, H_BOX, 'Positional Gauge Encoding',
                   C['pos'], fs=9.5, fw='bold')
    side_text(ax, CX - W_MAIN / 2, pos_cy,
              r'$\phi = \log(\exp(\phi_{\mathrm{tok}})\cdot\exp(\phi_{\mathrm{pos}}))$',
              side='left', offset=0.15, fs=6.8, c=C['math'])
    y = y_top

    y += 0.10
    arrow(ax, CX, y, CX, y + SP)
    y += SP

    # ════════════════════════════════════════════════════════════
    # 5. Transformer Block
    # ════════════════════════════════════════════════════════════
    y_block_entry = y  # where flow enters the block
    y_block_start = y - 0.18

    # ── 5a. LayerNorm 1 ──
    ln1_cy = y + H_NORM / 2
    _, y_top = box(ax, CX, ln1_cy, W_SUB, H_NORM, 'LayerNorm', C['norm'], fs=8)
    y_before_ln1 = y  # for residual start
    y = y_top

    y += 0.06
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ── 5b. KL-Divergence Multi-Head Attention ──
    attn_cy = y + H_ATTN / 2
    _, y_top = box(ax, CX, attn_cy, W_SUB, H_ATTN,
                   'KL-Divergence\nMulti-Head Attention',
                   C['attn'], fs=9.5, fw='bold')
    # Math annotations -- left side
    side_text(ax, CX - W_SUB / 2, attn_cy + 0.10,
              r'$\beta_{ij} = \mathrm{softmax}\left(\frac{-\mathrm{KL}(q_i \| \Omega_{ij} q_j)}{\kappa\sqrt{K}}\right)$',
              side='left', offset=0.15, fs=6.5, c=C['math'])
    side_text(ax, CX - W_SUB / 2, attn_cy - 0.14,
              r'$\Omega_{ij} = \exp(\phi_i \cdot G)\,\exp(-\phi_j \cdot G)$',
              side='left', offset=0.15, fs=6.5, c=C['math'])
    y = y_top

    y += 0.06
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ── 5c. Add (attention residual) ──
    add1_cy = y + H_ADD / 2
    _, y_top = box(ax, CX, add1_cy, W_ADD, H_ADD, 'Add', C['add'], fs=8, fw='bold')

    # Residual: from block_entry, bypass LN1+Attn, into Add1
    res_x = CX + W_SUB / 2 + 0.35
    ax.plot([CX + W_ADD / 2 + 0.02, res_x], [y_block_entry, y_block_entry],
            color=C['edge'], lw=0.8, zorder=1, clip_on=False)
    ax.plot([res_x, res_x], [y_block_entry, add1_cy],
            color=C['edge'], lw=0.8, zorder=1, clip_on=False)
    arrow(ax, res_x, add1_cy, CX + W_ADD / 2 + 0.02, add1_cy,
          lw=0.8)
    y = y_top

    y += 0.06
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ── 5d. LayerNorm 2 ──
    ln2_cy = y + H_NORM / 2
    y_before_ln2 = y  # for second residual
    _, y_top = box(ax, CX, ln2_cy, W_SUB, H_NORM, 'LayerNorm', C['norm'], fs=8)
    y = y_top

    y += 0.06
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ── 5e. VFE E-Step Loop ── (THE HEART)
    vfe_cy = y + H_VFE / 2

    # Background fill
    box(ax, CX, vfe_cy, W_SUB + 0.4, H_VFE, '', C['vfe_bg'],
        ec=C['vfe'], lw=1.4, alpha=0.35, zorder=2)

    # Title
    ax.text(CX, vfe_cy + H_VFE / 2 - 0.17, 'VFE E-Step',
            ha='center', va='top', fontsize=10.5, fontweight='bold',
            color=C['vfe'], zorder=5)
    ax.text(CX, vfe_cy + H_VFE / 2 - 0.38, 'Natural Gradient Descent',
            ha='center', va='top', fontsize=7.5, color='#006650', zorder=5)

    # Internal operations
    ops = [
        (r'Self-coupling:  $\alpha\,(\mu_q - \mu_p)\,/\,\sigma_p$',
         C['grad'], 'bold'),
        (r'Alignment:  $\sum_j \beta_{ij}\;\partial\mathrm{KL}_{ij}/\partial\mu_i$  + softmax coupling',
         C['txt'], 'normal'),
        (r'Update:  $\Delta\mu = -\eta\;\Sigma_q\;\nabla_\mu F$',
         C['txt'], 'normal'),
        (r'Softmax coupling $\equiv$ nonlinearity  (replaces GELU)',
         '#006650', 'bold'),
    ]
    op_y = vfe_cy + 0.12
    for text, color, fw in ops:
        ax.text(CX, op_y, text, ha='center', va='center',
                fontsize=6.5, color=color, fontweight=fw,
                style='italic' if fw == 'bold' and color == '#006650' else 'normal',
                zorder=5)
        op_y -= 0.22

    # Loop indicator (right side)
    loop_x = CX + (W_SUB + 0.4) / 2 + 0.15
    loop_y_bot = vfe_cy - H_VFE / 2 + 0.15
    loop_y_top = vfe_cy + H_VFE / 2 - 0.15
    loop_arrow = FancyArrowPatch(
        (loop_x, loop_y_bot), (loop_x, loop_y_top),
        arrowstyle='->,head_width=0.10,head_length=0.06',
        color=C['vfe'], linewidth=1.4,
        connectionstyle='arc3,rad=-0.45',
        zorder=5, mutation_scale=10)
    ax.add_patch(loop_arrow)
    ax.text(loop_x + 0.35, vfe_cy, r'$T$ iter.', ha='left', va='center',
            fontsize=7.5, color=C['vfe'], fontweight='bold', rotation=90, zorder=5)

    # mu_p LIVE arrow from left
    mu_arrow_y = vfe_cy + 0.30
    mu_arrow_x_end = CX - (W_SUB + 0.4) / 2 - 0.02
    mu_arrow_x_start = mu_arrow_x_end - 1.0
    arrow(ax, mu_arrow_x_start, mu_arrow_y, mu_arrow_x_end, mu_arrow_y,
          c=C['grad'], lw=1.6, ls='--')
    ax.text(mu_arrow_x_start - 0.08, mu_arrow_y,
            r'$\mu_p$ (live $\nabla$)', ha='right', va='center',
            fontsize=7.5, color=C['grad'], fontweight='bold')

    _, y_top = (vfe_cy - H_VFE / 2, vfe_cy + H_VFE / 2)
    y = y_top

    y += 0.06
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ── 5f. Add (VFE residual) ──
    add2_cy = y + H_ADD / 2
    _, y_top = box(ax, CX, add2_cy, W_ADD, H_ADD, 'Add', C['add'], fs=8, fw='bold')

    # Residual: from after Add1, bypass LN2+VFE, into Add2
    res_x2 = CX + W_SUB / 2 + 0.65
    y_res2_start = add1_cy + H_ADD / 2 + 0.02
    ax.plot([CX + W_ADD / 2 + 0.02, res_x2], [y_res2_start, y_res2_start],
            color=C['edge'], lw=0.8, zorder=1, clip_on=False)
    ax.plot([res_x2, res_x2], [y_res2_start, add2_cy],
            color=C['edge'], lw=0.8, zorder=1, clip_on=False)
    arrow(ax, res_x2, add2_cy, CX + W_ADD / 2 + 0.02, add2_cy, lw=0.8)
    y = y_top

    # ── Block boundary ──
    y_block_end = y + 0.18
    block_w = W_SUB + 1.8
    bx0 = CX - block_w / 2
    bh = y_block_end - y_block_start
    blk = FancyBboxPatch(
        (bx0, y_block_start), block_w, bh,
        boxstyle='round,pad=0.12', facecolor='none', edgecolor=C['border'],
        linewidth=1.0, linestyle=(0, (6, 3)), zorder=1)
    ax.add_patch(blk)
    ax.text(CX + block_w / 2 + 0.22, y_block_start + bh / 2,
            r'$\times\; L$', ha='left', va='center',
            fontsize=12, fontweight='bold', color=C['border'])

    y = y_block_end
    y += 0.08
    arrow(ax, CX, y, CX, y + SP)
    y += SP

    # ─── 6. Linear Output Projection ─────────────────────────
    out_cy = y + H_BOX / 2
    _, y_top = box(ax, CX, out_cy, W_MAIN, H_BOX,
                   'Linear Output Projection', C['out'],
                   fs=9.5, fw='bold', tc='white')
    side_text(ax, CX + W_MAIN / 2, out_cy,
              r'$W_{\mathrm{out}}\,\mu_q \;\to\; (B, N, V)$',
              fs=7, c=C['dim'])
    y = y_top

    y += 0.06
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ─── 7. Softmax ──────────────────────────────────────────
    sm_cy = y + H_BOX * 0.7 / 2
    _, y_top = box(ax, CX, sm_cy, 2.0, H_BOX * 0.7, 'Softmax',
                   C['softmax'], fs=9, ec='#AAAAAA')
    y = y_top

    y += 0.06
    arrow(ax, CX, y, CX, y + SP_SM)
    y += SP_SM

    # ─── Output ──────────────────────────────────────────────
    ax.text(CX, y + 0.06, 'Token Probabilities', ha='center', va='center',
            fontsize=10.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#AAAAAA', lw=0.6))

    # ════════════════════════════════════════════════════════════
    # Gradient flow annotation -- right margin
    # ════════════════════════════════════════════════════════════
    grad_x = CX + block_w / 2 + 0.85
    grad_y_bot = y_embed_center - 0.1
    grad_y_top = out_cy + 0.1

    arrow(ax, grad_x, grad_y_top, grad_x, grad_y_bot,
          c=C['grad'], lw=2.0, ls='--')
    ax.text(grad_x + 0.20, (grad_y_top + grad_y_bot) / 2,
            r'$\nabla\;$live  $(s\!=\!1)$', ha='left', va='center',
            fontsize=8, color=C['grad'], fontweight='bold', rotation=90)
    ax.text(grad_x, grad_y_top + 0.22,
            'Straight-through\ngradient to\nembeddings',
            ha='center', va='bottom', fontsize=6.5, color=C['grad'],
            style='italic')

    # ════════════════════════════════════════════════════════════
    # Title
    # ════════════════════════════════════════════════════════════
    ax.text(CX, 15.1, 'Amortized Inference Mode',
            ha='center', va='center', fontsize=15, fontweight='bold',
            color=C['txt'])
    ax.text(CX, 14.75,
            'amortized_inference = True,   implicit_em = False',
            ha='center', va='center', fontsize=8.5, color='#777777',
            family='monospace')

    # ════════════════════════════════════════════════════════════
    # Legend -- top left
    # ════════════════════════════════════════════════════════════
    lx, ly = 0.45, 14.35
    items = [
        (C['embed'],  'Gauge Embedding'),
        (C['attn'],   'KL-Divergence Attention'),
        (C['vfe_bg'], 'VFE E-Step (core)'),
        (C['out'],    'Output Projection'),
        (C['add'],    'Residual Addition'),
        (C['norm'],   'Layer Normalization'),
    ]
    for i, (fc, label) in enumerate(items):
        iy = ly - i * 0.26
        p = FancyBboxPatch((lx, iy - 0.07), 0.26, 0.14,
                           boxstyle='round,pad=0.02', facecolor=fc,
                           edgecolor='#888888', linewidth=0.4, zorder=5)
        ax.add_patch(p)
        ax.text(lx + 0.36, iy, label, ha='left', va='center',
                fontsize=7, color='#333333', zorder=5)

    # Dashed gradient entry
    iy = ly - len(items) * 0.26
    ax.plot([lx, lx + 0.26], [iy, iy],
            color=C['grad'], lw=1.5, ls='--', zorder=5)
    ax.text(lx + 0.36, iy, r'Live gradient  ($s = 1$)',
            ha='left', va='center', fontsize=7, color='#333333', zorder=5)

    # ════════════════════════════════════════════════════════════
    # Footer
    # ════════════════════════════════════════════════════════════
    ax.text(CX, 0.18,
            r'Belief tuple $(\mu,\;\Sigma,\;\phi)$ flows upward through all layers',
            ha='center', va='center', fontsize=7.5, color='#999999',
            style='italic')

    return fig


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'mathtext.fontset': 'dejavusans',
        'savefig.dpi': CONFIG['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'savefig.facecolor': 'white',
    })

    fig = create_diagram()

    out_dir = Path(CONFIG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in CONFIG['formats']:
        path = out_dir / f"{CONFIG['filename']}.{fmt}"
        fig.savefig(path, format=fmt, dpi=CONFIG['dpi'],
                    facecolor='white', edgecolor='none')
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == '__main__':
    main()
