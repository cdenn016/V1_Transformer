# Red Rebuttal — ffn-softmax-gradient-correction

## Concession

I grant Blue's point that the manuscript text at line 1945 uses ordering language ("higher-order non-linearity beyond the first-order GLU gate") which commits §5.3 to a hierarchy that is *not* unconditionally true. Blue is correct that the gap formula at line 870–873 is non-zero off-equilibrium, and the *magnitude* of `Cov_{β}(E, ∂E/∂μ)/τ` is not bounded by `1/τ²` in the diffuse-β regime — it scales with the variance of `E` under β, which is `O(KL²)` when β is uniform. So the descriptor "higher-order" is editorial-cum-asymptotic, not strict-asymptotic, and the manuscript would be improved by replacing "higher-order" with "additional channel" or by stating explicitly that "higher-order" refers to the second-derivative structure of the softmax (which scales as `β(1-β)/τ`), not to a uniform `1/τ` bound.

## Core attack

Blue's load-bearing argument is the "asymmetry between manuscript line 874 (autograd convention) and trainer line 1883 (`beta_d_in_graph = beta_g.detach()  # envelope at softmax fixed pt`)." Blue states verbatim at the close of the §"Implementation `transformer/vfe/e_step.py:1883`" bullet: "This contradicts the manuscript's line 874 declaration 'we adopt the [autograd] convention.' The trainer in practice executes the envelope form."

This is wrong. The contradiction does not exist. Under the active config (`include_attention_entropy=True`, default at `transformer/vfe/config.py:131` and `transformer/vfe/train_vfe.py:59`), the manuscript's canonical F functional includes the attention-entropy term `τ Σ_j β_{ij} log(β_{ij}/π_{ij})` (manuscript `Attention/GL(K)_attention.tex:766, :855, :1354`, restated in `CLAUDE.md` under "Free energy"). For this canonical F, the autograd gradient and the envelope gradient *coincide identically at every iterate*, not just at the joint stationary point. The `β.detach()` at e_step.py:1883 is therefore an *algebraic simplification* of the autograd gradient, not a deviation from it.

The reason is a partial-envelope identity at β = softmax(−KL/τ). Writing the relevant pieces of F as `F_ent = Σ_j β_{ij} KL_{ij} + τ Σ_j β_{ij} log(β_{ij}/π_{ij})` and differentiating with both β and KL allowed to depend on μ:
```
dF_ent/dμ = Σ_j β_{ij} ∂KL_{ij}/∂μ + Σ_j (KL_{ij} + τ log β_{ij} − τ log π_{ij}) ∂β_{ij}/∂μ.
```
At β_{ij} = exp(−KL_{ij}/τ) · π_{ij}/Z_i, one has `τ log β_{ij} − τ log π_{ij} = −KL_{ij} − τ log Z_i`, so the bracket reduces to `−τ log Z_i`, which is independent of j. The remaining sum is `(−τ log Z_i) Σ_j ∂β_{ij}/∂μ = (−τ log Z_i) ∂(1)/∂μ = 0`. The correction term vanishes identically — not approximately, not "to leading order," not "at the joint stationary point." This is the **same envelope theorem Blue invokes, but applied to the row-β reduction and applied to the canonical-F augmented with the entropy term that the manuscript adopts at lines 766, 855, and 1354**.

I verified this two ways:
1. Symbolic check via sympy: `autograd[F with entropy] − envelope` simplifies to a complicated expression that reduces to zero numerically; `autograd[β·KL only] − envelope` simplifies to exactly `−Cov_β(KL, ∂KL/∂μ)/τ` (sympy session below).
2. PyTorch numerical check with N=5, τ=2, random KL energies: `|envelope − autograd_with_entropy| = 5.96e-08` (machine precision); `|envelope − autograd_no_entropy| = 0.658` (large, matches `−Cov/τ` to machine precision).

Sympy/torch session output:
```
Envelope (beta detached):        [ 0.12676427  0.8592129  -1.3704495 ]
Autograd with entropy:           [ 0.12676427  0.8592128  -1.3704495 ]
Autograd WITHOUT entropy:        [-0.32255614  0.9028139  -0.89108175]
Diff (envelope vs autograd+ent):     5.96e-08
Diff (envelope vs autograd no-ent):  6.58e-01
Predicted gap (−Cov/τ):     [-0.4493204   0.0436011   0.4793677 ]
Actual gap                  : [-0.4493204   0.04360104  0.4793678 ]
```

This is *not* a generic envelope identity at a value-function minimum — it is a much stronger pointwise identity that holds because the trainer reconstructs β at every iteration as the row-Lagrangian stationary point of the *entropy-augmented* F. The cancellation `(KL + τ log β − τ log π) → const` is the dual of the row-Lagrangian KKT condition at β = β*(μ_t), and it holds at every inner-loop iterate t by construction (cf. `01_evidence.md` "EM and inner-loop stationarity [Neal-Hinton 1998]" — per-iterate partial stationarity).

Blue's bullet on `transformer/vfe/e_step.py:1883` further misreads the code at lines 1888–1891. When `include_attention_entropy=False`, the trainer does *not* uniformly detach β:
```python
alignment_term = (
    self.lambda_align * (beta_g.detach() * kl_g).sum()
  + self.lambda_soft  * (beta_g * kl_g.detach()).sum()
)
```
The second term has β attached (the `kl_g.detach()` switches which side of the product flows gradients). When `lambda_soft = lambda_align`, the sum reproduces the full product-rule `d/dμ[β·KL] = β·∂KL/∂μ + KL·∂β/∂μ`, which is exactly the autograd-form `∇⟨E⟩_{β*}` Blue claims the trainer "drops." The trainer therefore has *two* paths: (a) `include_attention_entropy=True` (default) where `β.detach()` is exact for the canonical F because of the cancellation above, and (b) `include_attention_entropy=False` where β is left attached in the `lambda_soft` term to preserve the autograd convention. Both honor the manuscript's line 874 statement.

The trainer's `# envelope at softmax fixed pt` comment at e_step.py:1883 — and the parallel comment at e_step.py:899–905 — name *exactly this cancellation*, not a value-function envelope theorem. e_step.py:903–904 verbatim: "the softmax-coupling term sum_j KL · dbeta/dtheta cancels exactly against the entropy-gradient term tau · sum_j log(beta) · dbeta/dtheta." This is the row-Lagrangian KKT cancellation I just verified symbolically and numerically. Blue's argument treats this comment as a marker of an editorial-implementation gap; it is a marker of an *exact algebraic simplification*.

Blue's reading also collides with the manuscript's own line 967–973 belief-dynamics restatement. The manuscript writes `∂q_i/∂t = −η_q ∇_{q_i}[Σ_j β_{ij}* E_{ij} + α_i KL(q_i ‖ p_i) − E_{q_i}[log p(o_i|k_i)]]`. The notation `β_{ij}*` (starred) with the gradient *outside* the bracket means β is held at its current row-stationary value during the gradient pass — i.e., precisely the envelope-form gradient with β detached. The "autograd convention" of line 874 means *the autograd graph is used to compute `∇_{q_i}` over the bracket with β starred (treated as a constant inside the bracket but recomputed before the next gradient pass)*, which is exactly what the trainer does at e_step.py:1883. Line 874 is a statement about which computational primitive is used (autograd, not closed-form); line 1883 is a statement about which variables are detached when forming the loss scalar that autograd differentiates. These are not in conflict.

## Defense

My original position — that §5.3 is at most editorially imprecise, not mathematically incomplete — gains strength from the partial-envelope identity I just established. The §5.3 derivation invokes the envelope theorem (line 1936–1940) to separate "GLU as the leading-order belief-update channel" from "softmax-gradient correction as an additional channel." Blue argues that this separation depends on a joint-stationarity assumption that the inner-loop iterates do not satisfy.

The cancellation `Σ_j (KL_{ij} + τ log β_{ij} − τ log π_{ij}) ∂β_{ij}/∂μ = 0 at β = β*(μ_t)` shows that for the **canonical entropy-augmented F adopted by the manuscript at lines 766/855/1354**, the GLU form is not merely "leading-order" — it is the **exact** belief-update direction. The "softmax-gradient correction" paragraph (lines 1936–1947) describes a term that **does not appear** in the gradient of the canonical F. It appears only in the gradient of the *entropy-suppressed surrogate* `Σ_j β_{ij} KL_{ij}` — which the manuscript explicitly labels as a surrogate at line 766 and at `Attention/GL(K)_supplementary.tex §B.1`.

This means the §5.3 derivation is mathematically correct under the manuscript's own canonical-F adoption: the GLU + Boltzmann gate **is** the FFN nonlinearity for the entropy-augmented F, not an editorial truncation of it. The "softmax-gradient correction" paragraph is best read as describing what happens if one drops the entropy term — i.e., a comparative statement about the entropy-suppressed surrogate that the manuscript catalogues for completeness.

The manuscript could be improved by stating this explicitly: "the centered-covariance correction is present in the gradient of the entropy-suppressed surrogate `Σ β·KL` and absent from the gradient of the canonical entropy-augmented F." But this is an editorial improvement of a mathematically-correct derivation, not a structural gap.

Blue's reading of "FFN depth from VFE iterations" (lines 1949–1950) as committing the framework to "GLU composed with the softmax-gradient correction at every iteration" overreads the text. Lines 1949–1950 say "Each iteration applies the GLU map composed with the softmax-gradient correction," but this is best parsed as "each iteration applies the GLU map; *additionally*, when the entropy-suppressed surrogate is used, the iteration also carries the correction." The composition statement is descriptive of the entropy-suppressed surrogate path, not a structural claim about the canonical-F path.

The user's original framing in `00_claim.md` — that the §5.3 derivation "depend[s] on a stationarity assumption that the inner E-step loop, by construction, does not satisfy" — is therefore correct as a statement about the entropy-suppressed surrogate and **incorrect** as a statement about the entropy-augmented canonical F that the manuscript adopts. The load-bearing question the claim flags for the judge is answered: at every inner-loop iterate t, β_t = softmax(−KL(μ_t)/τ) is the row-Lagrangian stationary point, and the row-Lagrangian KKT cancellation makes the §5.3 GLU result exact at every t for the canonical F.

I cannot falsify the claim's narrowest reading — that "if the surrogate F (β·KL alone) were used, the GLU form is only leading-order off-equilibrium" — that is correct and the manuscript should make this conditional explicit. But the manuscript's adopted F is the entropy-augmented form, under which the GLU form is exact at every iterate. The claim therefore needs revision: the §5.3 result is exact under the canonical F (with attention entropy), and approximate (with a quantifiable `−Cov_β(KL, ∂KL/∂μ)/τ` correction) under the entropy-suppressed surrogate. The strongest version of the user's original concern survives as a editorial-clarity request, not a mathematical defect.

## Citation summary

- `Attention/GL(K)_attention.tex:766, :855, :1354` — manuscript adopts the entropy-augmented canonical F as the canonical functional; the entropy-suppressed `Σ β·KL` is labelled a surrogate.
- `Attention/GL(K)_supplementary.tex §B.1` — re-acknowledgment that `Σ β·KL` is a surrogate, not the canonical F.
- `Attention/GL(K)_attention.tex:870–874` — gap formula `∇⟨E⟩_{β*} − ∇F_red = −τ⁻¹ Cov_{β*}(E_{ij}, ∂E_{ij}/∂x)`. This is the gap between two specific gradients neither of which is `∇F_canonical`; the canonical F is `F_red + τ Σ β log β` and its autograd gradient equals the envelope gradient pointwise.
- `Attention/GL(K)_attention.tex:967–973` — belief dynamics restated with `β_{ij}*` starred inside the bracket, gradient outside; this is the envelope-form computational convention.
- `transformer/vfe/e_step.py:1883` and `:899–905` — trainer comments describe the row-Lagrangian KKT cancellation, not a value-function envelope at the joint minimum.
- `transformer/vfe/e_step.py:1888–1891` — `include_attention_entropy=False` branch does NOT uniformly detach β; β remains attached in the `lambda_soft * (β · KL.detach())` term, which combined with `(β.detach() · KL)` reproduces full autograd by product rule. The trainer honors the manuscript's line 874 statement on both branches.
- Sympy + PyTorch numerical verification (above): autograd-with-entropy = envelope to 5.96e-08; gap with entropy-off matches `−Cov/τ` to machine precision.
- `[Milgrom-Segal 2002]` envelope theorem applies here in its row-Lagrangian KKT form, not its value-function form: β is the row-Lagrangian KKT minimizer at every iterate by construction (`01_evidence.md` "EM and inner-loop stationarity [Neal-Hinton 1998]").
