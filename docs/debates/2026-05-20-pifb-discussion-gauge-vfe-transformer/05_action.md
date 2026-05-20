# Action — pifb-discussion-gauge-vfe-transformer

**From verdict:** RED_WINS_NARROW

## Recommended action

Three edits to `Attention/Participatory_it_from_bit.tex`, all applied.

### Edit 1 — line 3190 grammar fix (mandatory)

Inserted "the framework's analog of" before "the feed-forward network" so the parameter-free qualifier is correctly routed to the framework object where the CLAUDE.md NO NEURAL NETWORKS hard constraint holds. Removes the grammatical conflation with Vaswani 2017 §3.3's two-layer learned MLP.

### Edit 2 — line 3190 W_Q/W_K/W_V parenthetical (optional, applied)

Added "(under the gauge-fixed reading developed at Section~\ref{sec:transformer_zero_dim})" parenthetical to "play the role of gauge transformations." Points the reader to the body machinery at sec:transformer_zero_dim where the carving / GL(d_k) bilinear construction is established.

### Edit 3 — line 3192 positional-encoding × RoPE clarification (mandatory)

Replaced the categorical "eliminating the need for separate positional encoding mechanisms" with a form that distinguishes the framework's full form (gauge frames intrinsic on a non-zero-dimensional base) from the zero-dimensional transformer implementation (RoPE realizing the per-token rotational frame). Cross-references `sec:transformer_zero_dim`, `sec:scope_limitations`, and cites `su2024roformer` for RoPE.

## Follow-up debates (if any)

None required. One narrower question is admissible but optional:

1. **Whether the alternative-reading citations (Tsai 2019, Ramsauer 2021, Millidge 2021) deserve one-sentence engagement** rather than just listing. Small, optional editorial improvement.

Not required by the present verdict.
