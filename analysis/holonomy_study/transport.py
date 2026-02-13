"""
Transport Operator Extraction from Pretrained Transformers
==========================================================

Three methods for extracting the effective transport T_{ij} from token j
to token i through a pretrained transformer:

Method 0: Attention path defect (scalar proxy, trivially cheap)
Method 1: Attention-decomposed transport (one forward pass, approximate)
Method 2: Jacobian probing (exact, P*N forward passes)

The effective transport T_{ij} = dh_i^{(L)} / dh_j^{(0)} captures how
a perturbation at position j in the input propagates to position i at the
output, including all attention, FFN, layer norm, and residual effects.

CAUSAL NOTE: For causal (autoregressive) models like GPT-2, T[i,j] is
nonzero only when j <= i (earlier tokens influence later ones, not vice
versa). Curvature is measured via *path composition defect* on ordered
triples a < b < c rather than closed loops:

    D_{abc} = T[c,b] @ T[b,a] @ pinv(T[c,a])

For flat transport, D = I (path doesn't matter). Curvature means D != I.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class TransportResult:
    """Container for extracted transport operators."""
    # T[i, j] is a (d, d) matrix: effective transport from j to i
    # Full shape: (N, N, d, d) where N = sequence length, d = hidden dim
    # For causal models, T[i, j] is nonzero only when j <= i
    transport: torch.Tensor  # (N, N, d, d)
    method: str
    n_tokens: int
    d_model: int
    metadata: dict


# =========================================================================
# Method 0: Attention path defect (scalar proxy for causal models)
# =========================================================================

def attention_path_defect(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Scalar proxy for holonomy in causal models.

    For ordered triples a < b < c, compare:
        direct:   alpha[c, a]           (c attends directly to a)
        indirect: alpha[c, b] * alpha[b, a]  (c -> b -> a two-hop)

    Path defect = |indirect - direct| / (indirect + direct + eps)

    For flat attention flow, multi-hop composition should equal direct
    attention. Deviation indicates path-dependent information routing.

    Args:
        model: HuggingFace GPT2Model with output_attentions
        input_ids: (1, N) token IDs
        attention_mask: (1, N) attention mask

    Returns:
        dict with:
            'attentions': tuple of (1, H, N, N) per layer
            'defect_per_layer': (L, num_triples) defect values
            'triples': list of (a, b, c) ordered index triples
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attentions = outputs.attentions  # tuple of (1, H, N, N)
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure model.config.output_attentions = True "
            "or pass output_attentions=True. Got None from model forward pass."
        )
    N = input_ids.shape[1]

    # Average over heads
    attn_matrices = [a.squeeze(0).mean(dim=0) for a in attentions]

    # Ordered triples a < b < c (all causal edges are valid)
    triples = _sample_ordered_triples(N, max_triples=500)

    defect_per_layer = []
    for A in attn_matrices:
        defects = []
        for a, b, c in triples:
            direct = A[c, a]
            indirect = A[c, b] * A[b, a]
            denom = direct + indirect + 1e-12
            defects.append(abs(indirect - direct) / denom)
        defect_per_layer.append(torch.stack(defects))

    return {
        'attentions': attentions,
        'defect_per_layer': torch.stack(defect_per_layer),  # (L, T)
        'triples': triples,
    }


# Keep old name as alias for backward compat in experiment.py
attention_flow_asymmetry = attention_path_defect


# =========================================================================
# Method 1: Attention-decomposed transport (approximate)
# =========================================================================

def attention_decomposed_transport(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
) -> TransportResult:
    """
    Compute effective transport via attention weights and value projections.

    Per-layer transport from j to i:
        T_{ij}^{(l)} = sum_h alpha_{ij}^{(l,h)} * W_O^{(h)} @ W_V^{(h)}

    Plus identity on diagonal (residual connection).

    The full transport is the (i,j) block of the product of per-layer
    block matrices: T^{eff} = prod_l T^{(l)}.

    Ignores FFN nonlinearity and layer norm — fast but approximate.

    Args:
        model: HuggingFace GPT2Model
        input_ids: (1, N) token IDs
        layers: which layers to include (default: all)

    Returns:
        TransportResult with transport shape (N, N, d, d)
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attentions = outputs.attentions  # tuple of (1, H, N, N)
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure model.config.output_attentions = True."
        )
    N = input_ids.shape[1]

    # Extract weight matrices
    transformer_blocks = model.h if hasattr(model, 'h') else model.transformer.h
    n_layers = len(transformer_blocks)
    if layers is None:
        layers = list(range(n_layers))

    d_model = transformer_blocks[0].attn.c_proj.weight.shape[0]
    device = input_ids.device

    # Build composed transport as block matrix product
    # T_composed[i,j] is (d, d) — start with identity
    T_composed = torch.zeros(N, N, d_model, d_model, device=device)
    for i in range(N):
        T_composed[i, i] = torch.eye(d_model, device=device)

    for l in layers:
        block = transformer_blocks[l]
        attn = block.attn

        # GPT-2 uses Conv1D: weight is (d_in, d_out), so transpose
        # c_attn projects to [Q, K, V] concatenated
        W_qkv = attn.c_attn.weight  # (d_model, 3*d_model)
        d_head = d_model // attn.num_heads
        n_heads = attn.num_heads

        # Extract W_V: last d_model columns
        W_V = W_qkv[:, 2*d_model:3*d_model]  # (d_model, d_model)

        # W_O = c_proj
        W_O = attn.c_proj.weight  # (d_model, d_model)

        # alpha: (1, H, N, N) -> (H, N, N)
        alpha = attentions[l].squeeze(0)

        # Precompute per-head WOV matrices
        WOV = []  # list of (d_model, d_model)
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            W_V_h = W_V[:, sl]  # (d_model, d_head)
            W_O_h = W_O[:, sl]  # (d_model, d_head) for Conv1D
            WOV.append(W_O_h @ W_V_h.T)  # (d_model, d_model)

        # Build per-layer transport: T^{(l)}_{ij} for all i,j
        T_layer = torch.zeros(N, N, d_model, d_model, device=device)

        # Residual connection on diagonal
        for i in range(N):
            T_layer[i, i] = torch.eye(d_model, device=device)

        # Attention contribution (vectorized over heads)
        for h in range(n_heads):
            # alpha[h]: (N, N), WOV[h]: (d, d)
            # T_layer[i,j] += alpha[h,i,j] * WOV[h]
            # Vectorized: (N, N, 1, 1) * (d, d) -> (N, N, d, d)
            T_layer += alpha[h].unsqueeze(-1).unsqueeze(-1) * WOV[h]

        # Compose: T_new[i,j] = sum_k T_layer[i,k] @ T_old[k,j]
        T_new = torch.einsum('ikab,kjbc->ijac', T_layer, T_composed)
        T_composed = T_new

    return TransportResult(
        transport=T_composed,
        method='attention_decomposed',
        n_tokens=N,
        d_model=d_model,
        metadata={'layers': layers, 'n_layers': n_layers},
    )


def per_layer_holonomy(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_triples: int = 300,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Compute per-layer path defect in a single fused pass.

    Streams one layer at a time (never stores more than one (N,N,d,d)
    tensor), vectorizes the triple computation with torch.bmm.

    For each layer l and ordered triple (a, b, c):
        T_direct   = T_l[c, a]
        T_indirect = T_l[c, b] @ T_l[b, a]
        kappa = ||T_indirect_hat - T_direct_hat||_F   (unit-norm)

    Returns dict with:
        'kappa_mean':      float, mean defect across layers and triples
        'kappa_per_layer': list of float, mean defect per layer
        'kappa_all':       (n_layers, n_triples) array
        'triples':         list of (a,b,c)
        'n_layers':        int
        'n_tokens':        int
        'd_model':         int
    """
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure model.config.output_attentions = True."
        )

    N = input_ids.shape[1]
    transformer_blocks = model.h if hasattr(model, 'h') else model.transformer.h
    n_layers = len(transformer_blocks)
    d_model = transformer_blocks[0].attn.c_proj.weight.shape[0]
    device = input_ids.device

    # Sample triples once
    triples = _sample_ordered_triples(N, max_triples=max_triples, seed=seed)
    n_tri = len(triples)
    # Precompute index tensors for batched gathering
    idx_a = torch.tensor([t[0] for t in triples], device=device)
    idx_b = torch.tensor([t[1] for t in triples], device=device)
    idx_c = torch.tensor([t[2] for t in triples], device=device)

    kappa_all = np.zeros((n_layers, n_tri))

    for l in range(n_layers):
        block = transformer_blocks[l]
        attn = block.attn

        W_qkv = attn.c_attn.weight
        d_head = d_model // attn.num_heads
        n_heads = attn.num_heads
        W_V = W_qkv[:, 2*d_model:3*d_model]
        W_O = attn.c_proj.weight

        alpha = attentions[l].squeeze(0)  # (H, N, N)

        # Build per-layer transport: T[i,j] = I[i==j] + sum_h alpha[h,i,j] * WOV_h
        T = torch.zeros(N, N, d_model, d_model, device=device)
        for i in range(N):
            T[i, i] = torch.eye(d_model, device=device)

        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            WOV_h = W_O[:, sl] @ W_V[:, sl].T
            T += alpha[h].unsqueeze(-1).unsqueeze(-1) * WOV_h

        # Vectorized defect computation for all triples at once
        # Gather: T_ca[t] = T[c_t, a_t], etc.  shape: (n_tri, d, d)
        T_ca = T[idx_c, idx_a]       # (n_tri, d, d)
        T_ba = T[idx_b, idx_a]       # (n_tri, d, d)
        T_cb = T[idx_c, idx_b]       # (n_tri, d, d)

        # Batched matmul: T_indirect = T_cb @ T_ba
        T_indirect = torch.bmm(T_cb, T_ba)  # (n_tri, d, d)

        # Unit-norm directional defect (vectorized)
        norm_ca = torch.norm(T_ca.reshape(n_tri, -1), dim=1, keepdim=True)   # (n_tri, 1)
        norm_ind = torch.norm(T_indirect.reshape(n_tri, -1), dim=1, keepdim=True)

        # Avoid division by zero
        norm_ca = norm_ca.clamp(min=1e-30)
        norm_ind = norm_ind.clamp(min=1e-30)

        T_ca_hat = T_ca.reshape(n_tri, -1) / norm_ca       # (n_tri, d*d)
        T_ind_hat = T_indirect.reshape(n_tri, -1) / norm_ind

        kappas = torch.norm(T_ind_hat - T_ca_hat, dim=1)   # (n_tri,)
        kappa_all[l] = kappas.detach().cpu().numpy()

        # Free this layer's transport immediately
        del T, T_ca, T_ba, T_cb, T_indirect

    # Aggregate
    mean_per_triple = np.nanmean(kappa_all, axis=0)
    kappa_per_layer = [float(np.nanmean(kappa_all[l])) for l in range(n_layers)]

    return {
        'kappa_mean': float(np.nanmean(mean_per_triple)),
        'kappa_median': float(np.nanmedian(mean_per_triple)),
        'kappa_std': float(np.nanstd(mean_per_triple)),
        'kappa_max': float(np.nanmax(mean_per_triple)),
        'kappa_per_layer': kappa_per_layer,
        'kappa_all': kappa_all,
        'triples': triples,
        'n_layers': n_layers,
        'n_tokens': N,
        'd_model': d_model,
    }


# Keep old function for backward compat (returns list of TransportResult)
def per_layer_transports(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[TransportResult]:
    """Legacy: use per_layer_holonomy() instead for speed."""
    result = per_layer_holonomy(model, input_ids, attention_mask)
    # Can't reconstruct full transports from summary — raise helpful error
    raise NotImplementedError(
        "per_layer_transports() is deprecated. Use per_layer_holonomy() directly."
    )


# =========================================================================
# Method 2: Jacobian probing (exact)
# =========================================================================

def jacobian_transport(
    model,
    input_ids: torch.Tensor,
    embedding_layer: nn.Module,
    attention_mask: Optional[torch.Tensor] = None,
    n_probes: int = 50,
    epsilon: float = 1e-3,
    seed: int = 42,
) -> TransportResult:
    """
    Compute effective transport via perturbation probing.

    For each position j and probe direction e_k:
        1. Perturb input embedding at j by epsilon * e_k
        2. Forward pass
        3. Measure (output_perturbed - output_clean) / epsilon at all positions i
        4. This gives one column of T_{ij} per probe

    Reconstruct transport from P probes via least-squares.

    Args:
        model: HuggingFace GPT2Model
        input_ids: (1, N) token IDs
        embedding_layer: the embedding module (model.wte or model.transformer.wte)
        n_probes: number of random probe directions P
        epsilon: perturbation magnitude
        seed: random seed for probe directions

    Returns:
        TransportResult with transport shape (N, N, d, d)
    """
    device = input_ids.device
    N = input_ids.shape[1]

    # Get clean output
    with torch.no_grad():
        clean_embeds = embedding_layer(input_ids)  # (1, N, d)
        # Add position embeddings if GPT-2
        if hasattr(model, 'wpe') or (hasattr(model, 'transformer') and hasattr(model.transformer, 'wpe')):
            wpe = model.wpe if hasattr(model, 'wpe') else model.transformer.wpe
            pos_ids = torch.arange(N, device=device).unsqueeze(0)
            clean_embeds = clean_embeds + wpe(pos_ids)

        clean_output = _forward_from_embeds(model, clean_embeds, attention_mask)
        # clean_output: (1, N, d)

    d_model = clean_output.shape[-1]

    # Generate random probe directions
    rng = torch.Generator(device='cpu').manual_seed(seed)
    probes = torch.randn(n_probes, d_model, generator=rng, device=device)
    probes = probes / probes.norm(dim=-1, keepdim=True)  # unit vectors

    transport = torch.zeros(N, N, d_model, d_model, device=device)

    for j in range(N):
        # Collect responses for all probes at position j
        responses = []  # will be (P, N, d)

        for p in range(n_probes):
            perturbed_embeds = clean_embeds.clone()
            perturbed_embeds[0, j, :] += epsilon * probes[p]

            with torch.no_grad():
                perturbed_output = _forward_from_embeds(
                    model, perturbed_embeds, attention_mask
                )

            response = (perturbed_output[0] - clean_output[0]) / epsilon  # (N, d)
            responses.append(response)

        # responses: (P, N, d)
        responses = torch.stack(responses, dim=0)

        probes_pinv = torch.linalg.pinv(probes)  # (d, P)

        for i in range(N):
            transport[i, j] = (probes_pinv @ responses[:, i, :]).T

    return TransportResult(
        transport=transport,
        method='jacobian_probing',
        n_tokens=N,
        d_model=d_model,
        metadata={'n_probes': n_probes, 'epsilon': epsilon},
    )


# =========================================================================
# Helpers
# =========================================================================

def _forward_from_embeds(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run model forward pass from embeddings, return hidden states."""
    # Handle both raw GPT2Model and wrapped models
    if hasattr(model, 'transformer'):
        # GPT2LMHeadModel — use the inner transformer
        core = model.transformer
    else:
        core = model

    outputs = core(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    return outputs.last_hidden_state


def _sample_ordered_triples(
    N: int,
    max_triples: int = 500,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """
    Sample ordered token index triples (a < b < c) for causal holonomy.

    For causal models, all three causal edges T[c,b], T[b,a], T[c,a]
    are nonzero when a < b < c, making the path defect well-defined.
    """
    import itertools

    all_triples = list(itertools.combinations(range(N), 3))

    if len(all_triples) <= max_triples:
        return all_triples

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(all_triples), size=max_triples, replace=False)
    return [all_triples[i] for i in sorted(indices)]


# Keep old name for any callers
_sample_triangles = _sample_ordered_triples


def load_model(model_name: str = 'gpt2', device: str = 'cpu'):
    """
    Load a pretrained transformer model for transport extraction.

    Args:
        model_name: HuggingFace model name (default: 'gpt2')
        device: 'cpu' or 'cuda'

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import GPT2Model, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, attn_implementation='eager')
    model.config.output_attentions = True
    model = model.to(device)
    model.eval()

    return model, tokenizer
