"""
Irony / Literal / Control Sentence Pairs
=========================================

Curated dataset for testing the flat bundle conjecture.

Design principles:
    - Ironic and literal pairs share the same target sentence where possible,
      differing only in context. This isolates the effect of ironic meaning
      from surface-level word statistics.
    - Controls match approximate complexity/surprise without irony.
    - Each example is (context + target) as a single string for the model.

Labels:
    'ironic'  — verbal irony, sarcasm, or situational irony
    'literal' — same or similar surface form, literal meaning
    'control' — complex/surprising but non-ironic
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SentencePair:
    """A labeled sentence for holonomy measurement."""
    text: str
    label: str          # 'ironic', 'literal', or 'control'
    pair_id: int        # links ironic/literal pairs
    target: str = ''    # the shared target phrase (if applicable)


def load_irony_pairs() -> List[SentencePair]:
    """
    Load curated irony/literal/control dataset.

    Returns list of SentencePair with balanced labels.
    """
    pairs = []
    pid = 0

    # =====================================================================
    # PAIRED: same target sentence, different context
    # These are the strongest test — identical words, opposite meaning
    # =====================================================================

    _paired = [
        # (ironic_context, literal_context, shared_target)
        (
            "The basement was flooding and the roof was leaking.",
            "The sun was shining and the birds were singing.",
            "What a perfect day.",
        ),
        (
            "He crashed the car into a tree on his first driving lesson.",
            "He parallel parked flawlessly on his first attempt.",
            "Well, that went exactly as planned.",
        ),
        (
            "She burned dinner and set off every smoke alarm in the house.",
            "She prepared a five course meal that impressed all the guests.",
            "She really outdid herself this time.",
        ),
        (
            "The presentation crashed, the projector died, and he forgot his notes.",
            "The presentation was flawless and the client signed immediately.",
            "That could not have gone better.",
        ),
        (
            "It rained every single day of their beach vacation.",
            "They had sunshine and clear skies for their entire trip.",
            "The weather was just wonderful.",
        ),
        (
            "He showed up three hours late with no explanation.",
            "He arrived right on time with coffee for everyone.",
            "How thoughtful of him.",
        ),
        (
            "The team lost by forty points in the championship game.",
            "The team won the championship in a dominant performance.",
            "What an incredible performance.",
        ),
        (
            "The restaurant served cold food after a two hour wait.",
            "The restaurant served exquisite food with impeccable timing.",
            "The service was absolutely outstanding.",
        ),
        (
            "He left all the dishes in the sink again for the third day.",
            "He cleaned the entire apartment and did all the laundry.",
            "He is so helpful around the house.",
        ),
        (
            "The new software update deleted all her files.",
            "The new software update made everything faster and smoother.",
            "Great improvement.",
        ),
        (
            "He locked his keys in the car for the second time today.",
            "He solved the complex engineering problem in under an hour.",
            "Brilliant move.",
        ),
        (
            "The meeting lasted four hours and accomplished nothing.",
            "The meeting was concise and resolved every open issue.",
            "That was a productive use of everyone's time.",
        ),
        (
            "She spent a fortune on a dress she will never wear.",
            "She found the perfect dress at an incredible discount.",
            "What a smart purchase.",
        ),
        (
            "The plumber flooded the bathroom while fixing a small leak.",
            "The plumber fixed the issue quickly and charged less than quoted.",
            "He really knows what he is doing.",
        ),
        (
            "The diet lasted exactly one day before he ate an entire cake.",
            "The diet worked and he lost twenty pounds in three months.",
            "That took real commitment.",
        ),
        (
            "The surprise party was ruined when he walked in early.",
            "The surprise party went perfectly and she was completely shocked.",
            "Everything went according to plan.",
        ),
        (
            "She accidentally replied all with a complaint about her boss.",
            "She sent a thoughtful message that resolved the team conflict.",
            "That email was a stroke of genius.",
        ),
        (
            "He assembled the furniture backwards and had parts left over.",
            "He assembled the furniture in twenty minutes with no mistakes.",
            "He is quite the handyman.",
        ),
        (
            "The GPS led them into a dead end road in the middle of nowhere.",
            "The GPS found the fastest route and saved them an hour.",
            "Technology is so reliable.",
        ),
        (
            "His first painting looked like a toddler made it.",
            "His first painting was exhibited at the local gallery.",
            "He is a natural talent.",
        ),
    ]

    for ironic_ctx, literal_ctx, target in _paired:
        pairs.append(SentencePair(
            text=f"{ironic_ctx} {target}",
            label='ironic',
            pair_id=pid,
            target=target,
        ))
        pairs.append(SentencePair(
            text=f"{literal_ctx} {target}",
            label='literal',
            pair_id=pid,
            target=target,
        ))
        pid += 1

    # =====================================================================
    # UNPAIRED ironic (standalone sarcasm / verbal irony)
    # =====================================================================

    _ironic_standalone = [
        "Oh wonderful, another Monday morning meeting about meetings.",
        "I love how the train is late again, it makes the commute so exciting.",
        "Nothing says relaxation like spending your vacation answering work emails.",
        "Sure, let me just add that to my completely empty schedule.",
        "I am shocked, truly shocked, that the politician broke a promise.",
        "How nice of the airline to lose my luggage, I needed a new wardrobe anyway.",
        "Yes, because what every deadline needs is another committee review.",
        "Thank goodness for autocorrect, it always knows what I mean to say.",
        "Another parking ticket, just what I wanted for my birthday.",
        "I especially love when the printer jams right before an important meeting.",
    ]

    for text in _ironic_standalone:
        pairs.append(SentencePair(
            text=text, label='ironic', pair_id=pid, target=''
        ))
        pid += 1

    # =====================================================================
    # UNPAIRED literal (straightforward factual/descriptive)
    # =====================================================================

    _literal_standalone = [
        "The Monday morning meeting covered the quarterly budget projections.",
        "The train arrived at the station three minutes behind schedule.",
        "She spent part of her vacation reviewing documents from the office.",
        "His schedule for Tuesday was packed with back to back appointments.",
        "The senator reversed his position on the infrastructure bill.",
        "The airline reported that her suitcase had been sent to the wrong city.",
        "The committee requested an additional review before the deadline.",
        "The autocorrect feature changed several words in her message.",
        "She received a parking ticket on the morning of her birthday.",
        "The printer jammed fifteen minutes before the board presentation.",
    ]

    for text in _literal_standalone:
        pairs.append(SentencePair(
            text=text, label='literal', pair_id=pid, target=''
        ))
        pid += 1

    # =====================================================================
    # CONTROLS: complex, surprising, or unusual — but not ironic
    # These should match the perplexity range of ironic sentences
    # =====================================================================

    _controls = [
        "The octopus has three hearts and blue blood circulating through its body.",
        "Cleopatra lived closer in time to the moon landing than to the building of the pyramids.",
        "A group of flamingos is called a flamboyance.",
        "Honey never spoils and archaeologists have found edible honey in ancient tombs.",
        "The shortest war in history lasted thirty eight minutes between Britain and Zanzibar.",
        "Venus rotates so slowly that a day there is longer than a year.",
        "There are more possible chess games than atoms in the observable universe.",
        "Bananas are berries but strawberries are not.",
        "The inventor of the Pringles can is buried in one.",
        "Oxford University is older than the Aztec Empire.",
        "A jiffy is an actual unit of time equal to one hundredth of a second.",
        "The unicorn is the national animal of Scotland.",
        "More people are killed by vending machines than by sharks each year.",
        "The total weight of ants on earth roughly equals the total weight of humans.",
        "A bolt of lightning is five times hotter than the surface of the sun.",
        "Wombat droppings are cube shaped.",
        "The northernmost point of Brazil is closer to Canada than to southern Brazil.",
        "The entire internet weighs about the same as a strawberry.",
        "There is a species of jellyfish that is biologically immortal.",
        "The Empire State Building has its own zip code.",
    ]

    for text in _controls:
        pairs.append(SentencePair(
            text=text, label='control', pair_id=pid, target=''
        ))
        pid += 1

    return pairs


def get_paired_only(pairs: List[SentencePair]) -> List[SentencePair]:
    """Filter to only paired ironic/literal examples (strongest test)."""
    # Paired examples have both ironic and literal entries with same pair_id
    from collections import Counter
    id_counts = Counter(p.pair_id for p in pairs)
    paired_ids = {pid for pid, count in id_counts.items() if count >= 2}
    return [p for p in pairs if p.pair_id in paired_ids]


def by_label(pairs: List[SentencePair]) -> dict:
    """Group sentence pairs by label."""
    groups = {'ironic': [], 'literal': [], 'control': []}
    for p in pairs:
        groups[p.label].append(p)
    return groups
