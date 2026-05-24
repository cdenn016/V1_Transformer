# Memo — Red / geometer / opening

## Steelman
The geometric obstructions are correctly stated and I grant the full concession floor. (1) For
constant g the adjoint action preserves tr(A_μ A_ν); the single-copy Haar average over a compact
G is well-defined but trivial. (2) For local g(c) the gauge-orbit average lives on Map(C, G), an
infinite-dimensional manifold of gauge-group-valued fields, so the average is a functional
integral, not a finite-dimensional Haar integral. (3) For non-compact SO(1,3) the Haar measure is
infinite even for constant g.

## Falsify (limited — I largely confirm rather than attack the geometry)
Statement (3) rests on the standard theorem that a locally compact group has finite Haar measure
iff it is compact [Folland, *A Course in Abstract Harmonic Analysis*; Knapp, *Lie Groups Beyond
an Introduction*]. SO(1,3) is non-compact (the boosts form a non-compact ℝ³ direction), so its
total Haar measure diverges and the normalization ∫_G dg = ∞ makes ⟨G_i⟩ ill-defined without a
regulator. The manuscript states exactly this. Correct.

My only geometric flag concerns completeness, not error. The manuscript treats the local-g case
as requiring "a gauge fixing or a regulator," which is right, but it does not note that even the
*compact* connected pieces (U(1), SU(2), SU(3)) inherit the infinite-dimensional obstruction once
g is promoted to a field g(c): the issue is not non-compactness of the structure group but the
infinite dimension of Map(C, G). Compactness of G rescues the *constant*-g average (where it is
trivial anyway, by cyclicity) but not the local-g average (where the content lives). The
manuscript's prose runs these together slightly — it pairs "infinite-dimensional functional
integral" with local g, then separately notes non-compact SO(1,3) has infinite Haar "even for
constant g." Both true, but a reader could come away thinking compactness of G is the operative
fix for local g. It is not. This is a minor expository gap inside an otherwise correct concession,
not a falsification of the claim.

The load-bearing geometric point for the debate: the consensus metric ⟨G_i⟩ is, in the regime
that carries the section's physical content (local g, where the Maurer-Cartan term is the source
of non-invariance), an object that does not exist as a finite quantity. The manuscript concedes
this honestly. That concession is correct — which means the circularity and SM-enumeration
attacks (philosophy-of-science, info-geometer, gauge-theorist memos) must carry the red case;
the geometry itself is sound and should be conceded.

## External primary-source citation (not the manuscript)
- Folland, G. B., *A Course in Abstract Harmonic Analysis* — Haar measure exists and is unique up
  to scale on a locally compact group; total mass is finite iff the group is compact.
- Knapp, A. W., *Lie Groups Beyond an Introduction* — non-compact semisimple groups (SO(1,3))
  have infinite Haar volume; boosts are the non-compact directions.

## Falsification condition
Wrong if SO(1,3) is compact (it is not) or if Map(C, G) is finite-dimensional (it is not, for any
non-trivial base C). Both standard. The manuscript's geometric statements stand.

## Newly-discovered canon
- Haar measure finiteness ⇔ compactness (standard; Folland, Knapp). Used to confirm the
  non-compact SO(1,3) obstruction at :2977.
