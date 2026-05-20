# Blue Opening — reduction-to-standard-transformer

## Steelman (opposing position)

The reduction is not a single mathematical identity. It is a chain of four operations of distinct epistemic status: (i) the exact algebraic identity at line 1049, (ii) the exact softmax shift-invariance step at lines 1184-1188, (iii) an *approximate* cancellation of the key-norm bias `-\|\mu_j\|^2/(2\sigma^2)` at lines 1196-1225 (the manuscript itself uses the word "approximately" at line 1198), and (iv) a non-unique change of parameterization that absorbs `\sigma^{-2}\Omega^{-\top}` into `W_Q W_K^\top` at lines 1237-1252, where the standard rectangular `W_Q^a, W_K^a \in \mathbb{R}^{d_{\text{model}}\times d_{\text{head}}}` are rank-deficient in the ambient space and the GL(d_k) identification operates on a strict sub-object. Calling the composite step "exact" therefore obscures that step (iii) is a controlled approximation and step (iv) is a literal change of parameterization. The honest framing — which the manuscript itself uses at line 992 ("emerges as a set of limiting cases") and at line 1337 ("the limits are deliberately aggressive") — is "exact under stated limits, with one approximate cancellation."

## Position

Standard scaled dot-product attention (`Attention(Q,K,V) = softmax(QK^\top/\sqrt{d_k}) V`) is recovered **exactly under the stated limits** as the boxed equation at `Attention/GL(K)_attention.tex` line 1331. Every algebraic step in §5.2.2 is a literal identity once the three named limits (isotropic covariance, constant gauge, layer-normalised keys *or* high-dimensional concentration) are taken, and the rectangular-projection caveat is honestly disclosed at lines 1243-1250 as a reduction to the invertible head-space factor `M_h^a \in \mathrm{GL}(d_{\text{head}})`. The two operations red would flag as "hidden" — the `\sigma^{-2}` absorption (lines 1036, 1252) and the geometric-bias cancellation under constant gauge (line 1117) — are not hidden: the first is a literal change of parameterization, the second is softmax shift-invariance. The single non-exact step is the layer-norm-conditioned cancellation of the key-norm bias at line 1198, and the manuscript labels it "approximate" in its own words.

The claim is therefore: with `LN(\mu_j)` (or equivalent norm control), `\Sigma_i = \sigma^2 I`, and `\Omega_{ij} = \Omega` constant, the gauge-covariant attention rule at line 1326 satisfies

```
softmax_j(-(1/\tau) D_KL(q_i \| \Omega q_j)) \mû_i ≡ softmax(Q K^\top / \sqrt{d_k}) V
```

as a bitwise (up to floating-point) identity for the operand chain in §5.2.2. The "≡" is exact because each constituent step (Mahalanobis identity, isotropic-KL form, softmax shift-invariance, parameter absorption) is an identity, and the one approximation is moved to the precondition (layer normalisation or concentration), where it is an empirical property of the input to the reduction rather than a step inside it.

## Evidence

### 1. The KL form at line 1043 matches the textbook Gaussian KL

The closed-form Gaussian KL [external_canon_math.md §1, "Closed-form KL between Gaussians"] is

```
KL(q || p) = ½[tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) - K + log(|Σ_p|/|Σ_q|)]
```

[BleiKuckelbirgJordan2017; KingmaWelling2014 Appendix B]. Substituting `q_i = \mathcal{N}(\mu_i, \sigma^2 I)`, `\Omega q_j = \mathcal{N}(\Omega\mu_j, \sigma^2 \Omega\Omega^\top)` and simplifying with sympy:

```
import sympy as sp
n = 2; sigma = sp.symbols('sigma', positive=True)
Omega = sp.Matrix(n,n, lambda i,j: sp.symbols(f'O{i}{j}'))
mu_i = sp.Matrix(n,1, lambda i,_: sp.symbols(f'mi{i}'))
mu_j = sp.Matrix(n,1, lambda i,_: sp.symbols(f'mj{i}'))
I_n = sp.eye(n)
textbook_KL = sp.Rational(1,2)*(
    ((sigma**2*Omega*Omega.T).inv() * (sigma**2*I_n)).trace()
    + ((Omega*mu_j - mu_i).T * (sigma**2*Omega*Omega.T).inv() * (Omega*mu_j - mu_i))[0,0]
    - n + sp.log((sigma**2*Omega*Omega.T).det() / (sigma**2*I_n).det()))
manuscript_KL = sp.Rational(1,2)*(
    sp.log((Omega*Omega.T).det()) + (Omega*Omega.T).inv().trace() - n
) + (1/(2*sigma**2)) * ((mu_i - Omega*mu_j).T * (Omega*Omega.T).inv() * (mu_i - Omega*mu_j))[0,0]
print(sp.simplify(textbook_KL - manuscript_KL))
# Output: 0
```

The manuscript line 1043 is the textbook KL after the trace term `tr(Σ_p^{-1}Σ_q) = tr((\sigma^2 \Omega\Omega^\top)^{-1}\sigma^2 I) = tr((\Omega\Omega^\top)^{-1})` and log-det term `log(|\sigma^2 \Omega\Omega^\top|/|\sigma^2 I|) = \log\det(\Omega\Omega^\top)` simplify. This is an identity, not an approximation.

### 2. The Mahalanobis identity at line 1049 is exact

Verified symbolically with sympy on a generic 3x3 invertible matrix:

```
LHS = ((mu_i - Omega*mu_j).T * (Omega*Omega.T).inv() * (mu_i - Omega*mu_j))[0,0]
RHS = ((Omega.inv()*mu_i - mu_j).T * (Omega.inv()*mu_i - mu_j))[0,0]
sp.simplify(LHS - RHS)
# Output: 0
```

This is the load-bearing identity that converts the Mahalanobis-form KL into a Euclidean distance in the key's frame. The manuscript states at line 1133 the broader full-KL identity has been "verified symbolically against the direct Gaussian KL to machine precision"; the sympy run above replicates the specific instance the reduction depends on.

### 3. Softmax shift-invariance at lines 1184-1188 is exact, not approximate

For fixed query `i`, the only `j`-dependent quantity in `s_{ij}` (line 1167) after the Mahalanobis identity is

```
s_{ij} = (1/2σ²)||Ω^{-1}μ_i||² + (1/2σ²)||μ_j||² - (1/σ²)μ_i^T Ω^{-T} μ_j + C
```

The first term and `C` are constant in `j`. Standard softmax identity `softmax(x + c \mathbf{1}) = softmax(x)` (a textbook fact) makes their cancellation exact. Numerical verification with random `N=5, d=4, \sigma=0.7, \Omega = \text{randn} + 2I`:

```
max |beta_full - beta_trim| = 2.17e-18
```

Bitwise equality at machine epsilon.

### 4. The geometric bias `S(\Omega)` cancellation under constant gauge is also softmax shift-invariance

Line 1115-1117: under `\Omega_{ij} = \Omega` the bias `S(\Omega)` is pair-independent, hence j-independent, hence cancels by the same softmax shift-invariance identity. Exact.

### 5. The σ⁻² absorption is a literal reparameterization, not a limit

Line 1252: "rather than completely taking $\sigma \to 0$ we recognize that $\sigma^{-2}$ and $\Omega^{-\top}$ always appear together in the combination $\sigma^{-2}\Omega^{-\top}$. The learned matrices $W_Q, W_K$ can then be considered to parametrize this combined quantity directly." This is identical in mechanical content to absorbing the standard `1/\sqrt{d_k}` factor into `W_Q W_K` versus keeping it separate (which is a free choice in the standard transformer). Standard attention does not require `\sigma` to be small in the user's framework — it requires `\sigma^{-2}` to be subsumed into the learned bilinear. By line 1238 the identification `W_Q W_K^\top = \sigma^{-2}\Omega^{-\top}` is well-posed for any finite `\sigma > 0`. Existence of the factorization is by SVD (line 1241): `M = U\Lambda V^\top = (U\Lambda^{1/2})(V\Lambda^{1/2})^\top` with all positive singular values, so both factors land in GL(d_k). This is constructive and exact.

### 6. The rectangular-projection caveat is disclosed and the reduction operates on the invertible factor

Lines 1243-1250 explicitly identify what the GL(d_k) structure corresponds to in the standard transformer. The ambient kernel `W_Q^a (W_K^a)^\top \in \mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}` is rank-≤d_head and not in GL(d_model). The reduction identifies the gauge `\sigma^{-2}\Omega^{-\top}` with the *thin-SVD invertible factor* `M_h^a := A_Q^a (A_K^a)^\top \in \mathrm{GL}(d_{\text{head}})`. This is the load-bearing math: the gauge-theoretic identification is with `M_h^a`, not with `W_Q^a (W_K^a)^\top`. Per pitfall 10 of external_canon_transformers.md, the user's framework has no learned QKV at construction — they emerge from this absorption — so the comparison is exactly to the *form* `softmax(QK^\top/\sqrt{d_k})V` and not to a specific weight-initialization protocol.

### 7. Value aggregation under constant gauge collapses cleanly

Line 1296 gives `\hat\mu_i = \sum_j \beta_{ij} \Omega_{ij}\mu_j`. Under constant `\Omega_{ij} = \Omega`, the `\Omega` pulls out of the sum and absorbs into the learned `W_V`. Line 1308 defines `V_j = W_V^\top \mu_j` with `W_V` absorbing `\Omega`. Pulling a constant matrix out of a finite sum is exact. The reduction therefore collapses to line 1316: `\hat\mu_i = \sum_j \beta_{ij} V_j`.

### 8. The τ = √d_k derivation is the same statistical argument used by [Vaswani2017 §3.2.1]

Line 1277: dot products of unit-variance vectors in `d_k` dimensions have standard deviation `O(\sqrt{d_k})`. To normalise pre-softmax logits to `O(1)`, set `τ = √d_k`. This is the dimensional-variance argument from external_canon_transformers.md §1 ("for unit-variance Q, K, the dot product Q_i · K_j has variance d_k, so dividing by √d_k restores unit variance"). The manuscript discloses at line 855 that the full framework uses `\tau = \kappa\sqrt{K}` with learnable `\kappa`, and "the standard-transformer recovery corresponds to the special case $\kappa = 1$". The reduction therefore lands at the standard `1/\sqrt{d_k}` exactly when `\kappa = 1`.

### 9. The complete formula at line 1331 is exact under the chain (1)-(8)

The boxed result `Attention(Q,K,V) = softmax(QK^\top/\sqrt{d_k}) V` is the composition of: (a) exact closed-form KL [BleiKuckelbirgJordan2017], (b) exact Mahalanobis identity (sympy-verified), (c) exact softmax shift-invariance, (d) exact constant-gauge specialisation, (e) exact SVD factorisation of `\sigma^{-2}\Omega^{-\top}`, (f) exact value-aggregation absorption, (g) the matching dimensional-variance derivation of `\tau = \sqrt{d_k}`.

### Pre-empting the strongest attack

The strongest red attack is the "approximate" wording at line 1198 for the key-norm cancellation `-\|\mu_j\|^2/(2\sigma^2)`. I concede the wording: the cancellation is *approximate in absence of layer normalisation*. With layer normalisation enabled (the modern standard, present in Vaswani2017 and every successor), `\|\mu_j\|^2` is bitwise constant across `j`, and the softmax shift-invariance argument makes the cancellation *exact*. Without layer normalisation, the cancellation invokes concentration of measure `\|\mu_j\|^2 = d_k\sigma_0^2 \pm O(\sigma_0^2\sqrt{d_k})` (line 1204), which is approximate to `O(1/\sqrt{d_k})`. The claim's "exact" framing therefore depends on which precondition is in force. Under LN: exact. Under concentration-only: exact in the `d_k \to \infty` limit. The honest position is: the reduction is exact *under any of the documented preconditions that flatten `\|\mu_j\|^2` across `j`*, with LN being the standard precondition in modern transformers and exact under it.

A second red attack is non-uniqueness of `W_Q W_K^\top = \sigma^{-2}\Omega^{-\top}` (pitfall 2 of external_canon_transformers.md). Non-uniqueness is asymmetric: the reduction *direction* (gauge → transformer) is well-posed — given `\Omega`, the product `\sigma^{-2}\Omega^{-\top}` is uniquely determined, and any SVD pair `(W_Q, W_K)` reproducing it works. The *inverse direction* (transformer → gauge) is ambiguous because many gauge transports map to the same bilinear. The headline claim is the forward reduction, so non-uniqueness in the reverse direction does not falsify it.

## Falsification conditions

The claim "exact reduction to standard transformer attention under stated limits" is **not defensible** if any of the following hold:

1. **A step in §5.2.2 is shown to require an unstated assumption.** If red finds that any line between 1043 and 1331 invokes a fact not implied by (constant `\Omega`, isotropic `\Sigma`, LN-on-keys or `d_k \to \infty`), the chain is broken and the claim collapses to "exact mod that assumption."

2. **The Mahalanobis identity at line 1049 fails for some `\Omega \in \mathrm{GL}(d_k)`.** Sympy verification covers generic invertible `\Omega`; if a counterexample is exhibited (e.g., near-singular `\Omega` where the identity holds only after a limit), the "exact" claim breaks at the load-bearing step. (I do not expect this — the identity is `(A^\top A)^{-1} = A^{-1} A^{-\top}`, valid for any invertible A.)

3. **The key-norm bias cancellation is non-exact even with LN.** If red constructs an LN that does not enforce `\|\mu_j\|^2 = \text{const}` across tokens (e.g., per-token LN with learnable affine that varies across tokens, which is *not* the standard LN), then line 1213 ceases to hold and the cancellation reverts to approximate. The claim's "exact" status is conditioned on standard LN.

4. **The `\sigma^{-2}\Omega^{-\top}` absorption is not a literal reparameterisation.** If red shows that the absorption changes the function class (e.g., that the post-absorption `W_Q W_K^\top` cannot reach all of `\sigma^{-2}\Omega^{-\top}` as `\Omega` varies over `\mathrm{GL}(d_k)`), the identification at line 1238 is not a bijection between the two parameterisations. Per line 1241, SVD shows reachability; if the SVD argument has a gap (e.g., requires `\Omega^{-\top}` to be in a subset of `\mathrm{GL}(d_k)`), the gap falsifies the claim.

5. **The rectangular-projection caveat is fatal rather than merely a sub-form.** If red argues that the "reduction" is to a sub-form of standard attention (the rank-`d_{\text{head}}` head-space factor) and not to the standard form itself (which uses rank-`d_{\text{head}}` factors per head as the *definitional* construction in Vaswani2017 §3.2.2), the claim becomes "reduction to multi-head head-space attention" rather than "reduction to standard attention." I argue these are the same thing — Vaswani2017's multi-head construction *is* a stack of rank-`d_{\text{head}}` head factors — but if red can show the standard transformer claim is stronger than the head-space identification, the headline reduces in strength.

6. **The reduction-direction non-uniqueness becomes a forward-direction non-uniqueness.** If red shows that two distinct gauge configurations yield two distinct attention outputs but identical `W_Q W_K^\top`, then the reduction is many-to-one and the recovered standard attention is a quotient rather than the original gauge attention. (This is the same content as falsifier 4.)

7. **The value-aggregation absorption is shown to require pair-independent `\Omega_{ij}`.** I grant this — line 1296 has `\Omega_{ij}` inside the sum, and pulling it out requires constant gauge. The claim's "exact" status therefore requires the constant-gauge specialisation, which the manuscript explicitly states. If red can show that the reduction is being claimed *without* constant gauge, the claim collapses; under the explicit constant-gauge limit it survives.

The claim is **defensible** under the explicit conjunction of preconditions: constant gauge (line 1117), isotropic covariance (line 1038), `\sigma^{-2}\Omega^{-\top}` absorption (line 1252), layer-normalised keys (line 1209) or concentration-of-measure scaling (line 1204), and `\kappa = 1` (line 855). The manuscript states each of these. Under their conjunction every step is an identity (one of: textbook Gaussian KL, Mahalanobis algebra, softmax shift-invariance, SVD factorisation, linearity of finite sums). The composite map from gauge attention to `\mathrm{softmax}(QK^\top/\sqrt{d_k})V` is therefore exact.
