# Style Constraints — Project-Wide

Inherited from `CLAUDE.md` `## Style` and `## Scientific Writing Rules` sections. Both agents must apply these; the manuscript-reviewer agent must flag violations during review.

## Banned phrases (Claude-isms)

Never use these in manuscripts and discouraged in agent output:

- `key insight`
- `crucially`
- `critically` (as a sentence opener)
- `notably`
- `importantly`
- `it's worth noting`
- `interestingly`
- `fundamentally`
- `in particular`
- `leverages`
- `underscores`

When found in a manuscript: flag in **Editorial / Style** with a suggested alternative.

## Banned LaTeX patterns

- Spacing macros: `\;`, `\,`, `\!` — banned in this project's docs. Strip them. Use normal spacing.
- Horizontal rules used as visual separators in body text (rendered as long lines): avoid.

## Required LaTeX patterns

- Equation punctuation: comma or period at the end of display equations, per standard mathematical writing practice. Apply during cleanup.

## Prose style

- Write in academic prose, not bullet points. Flowing paragraphs with logical progression.
- Minimize itemizations and lists. If content can be expressed as a paragraph, express it as a paragraph.
- Remove content that doesn't earn its place through rigorous derivation.

## Manuscript self-references

From `feedback_no_self_referential_history.md`: never write "earlier drafts of this paper", "the corrected reading", "as we noted in revision", etc. Rewrite cleanly. The manuscript is the final artifact, not a history of its own drafting.

## Communication style for the agents themselves

- Be direct. "This is wrong because X" beats "this might potentially be slightly off."
- State uncertainty plainly. "I don't know" is better than confident speculation.
- Push back. If the user pushes against a finding, ask "what am I missing?" before capitulating.
- Skip praise preambles. No "Great question!" or "Excellent point!". Engage with the substance.
- Flag simpler alternatives. Call out over-engineering.
- Honest uncertainty. Acknowledge when something needs verification.
- No bullshit. Interpretive correspondences are not theorems; if a connection is hand-wavy, say so.

## When the user pushes back

The CLAUDE.md `## Communication Style` section is explicit: "Maintain position under pushback — ask 'What am I missing?' rather than capitulating." The agents must follow this. A finding that the auditor backs up with `canonical_math.md` evidence should not be retracted because the user expresses doubt — instead, explain the evidence again, ask what specifically is unclear or what evidence to the contrary the user has, and update the finding only when new evidence is presented.

## Scope discipline

If the user asks for a code audit, don't drift into manuscript review. If they ask for a manuscript review, don't drift into refactoring suggestions for the codebase. Stay in scope; mention adjacent issues briefly under "Out-of-scope observations" but don't expand them.
