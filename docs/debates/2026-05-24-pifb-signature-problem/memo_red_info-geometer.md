# Memo — debate-expert-info-geometer — red — opening — pifb-signature-problem

## Lens
Fisher–Rao metric, statistical content, decoupling of fiber geometry from base signature.

## Steelman of the opposing position
The sector split is coherent and conservative: it keeps the belief fiber a real positive-definite Fisher–Rao geometry (so KL$\ge0$, $\mathcal{F}\ge0$, Čencov-invariance all survive) and confines the indefinite signature to the connection sector, exactly so as not to corrupt the statistical content — a principled containment.

## My position (in service of red)
The containment is precisely the problem for the claim's *significance*. The section is emphatic and correct (`:2847`, `:2902`) that the Fisher–Rao metric on the fiber "remains positive semi-definite throughout" and that "the indefinite signature is a property of how gauge frames vary over the base manifold, not of the statistical geometry within a single fiber." Read literally, this means the Lorentzian signature is *disconnected from the information content of the beliefs*. The Fisher metric — the unique (Čencov) statistical metric, the thing that makes this an *information*-geometric framework — contributes only the positive-definite block $G^{\mathrm{Fisher}}_{\mu\nu}$ (`:2902`). The sign that makes the geometry Lorentzian comes entirely from $\mathrm{tr}(A_\mu A_\nu)$, a Lie-algebra trace form that is gauge-noninvariant (`:2829`, `:2904`) and carries no statistical information: it is a function of the *frame field's twist*, not of any $q_i$, $\mu_i$, or $\Sigma_i$.

This guts the "it from bit" thesis at exactly this section. The whole point of deriving spacetime from information would be that the signature reflects something about the beliefs/observations. Here it reflects an imaginary assignment to a frame component, with the real (statistical) sector explicitly excluded from carrying the sign. The manuscript even confirms the variational principle is *blind* to the choice that produces the signature: at `:2952`, "the KL functional between real Gaussian beliefs is invariant under that choice" (real vs imaginary $\phi_\tau$ of the same magnitude). So the free-energy functional — the framework's entire dynamical content — cannot see, and therefore cannot select, the signature. The signature is bolted onto a sector that the statistics do not touch.

"Structurally compatible with Lorentzian signature" is therefore true only in the weak sense that one can append an indefinite non-statistical bilinear form to a positive-definite statistical one. That is compatible in the way that adding $-dt^2$ to a Riemannian metric is "compatible with Lorentzian signature" — a relabeling, not a derivation, and explicitly not an *information*-geometric one.

## Evidence
- Čencov (1972) uniqueness theorem [Cencov1972; AmariNagaoka2000 Ch. 2]: the Fisher–Rao metric is the unique (up to scale) metric on a statistical manifold invariant under sufficient statistics. The signature here lives *outside* this unique statistical metric, on the connection's trace form — hence is not an information-geometric quantity.
- Manuscript `:2847`, `:2902`: signature lives on $\mathrm{tr}(A_\mu A_\nu)$, "not of the statistical geometry within a single fiber"; $G^{\mathrm{Fisher}}$ is positive semi-definite.
- Manuscript `:2952`: "the KL functional between real Gaussian beliefs is invariant under that choice" — the variational principle cannot distinguish the signature-producing assignment from its real counterpart. The dynamics are blind to the signature.

## Newly-discovered canon (for 01b_extended_evidence.md)
- Amari & Nagaoka (2000), *Methods of Information Geometry*, AMS, Ch. 2–3: the Fisher metric and its uniqueness; dual (e-/m-) connections are the natural connection structure *induced by the statistics*. A connection trace form decoupled from the Fisher metric carries no statistical-invariance guarantee.
- Čencov, N. N. (1972/1982), *Statistical Decision Rules and Optimal Inference*, AMS: invariance-under-sufficient-statistics uniqueness; a metric/signature not derived from the statistical structure is, by this theorem, not an information metric.

## Falsification conditions
My position is wrong if (a) the section exhibits a coupling whereby the belief statistics ($\mu_i,\Sigma_i$, the free energy) select or constrain the connection's imaginary assignment — it explicitly denies this at `:2952`; or (b) "information geometry $\to$ spacetime" is reinterpreted to allow the signature to come from a non-statistical sector, in which case the framework's distinctive "it from bit" claim is abandoned for this result.

## Confidence
HIGH — the decoupling is the manuscript's own stated design (`:2847`, `:2952`); the Čencov uniqueness theorem fixes what an information metric is.
