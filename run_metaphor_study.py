#!/usr/bin/env python3
"""
Metaphor Holonomy Study: Partial Compositionality as Gauge Curvature
=====================================================================

Tests whether metaphorical language exhibits different gauge curvature
than literal language. Metaphors sit BETWEEN idioms and literal language
on the compositionality gradient:

    literal (fully compositional) > metaphor (partial) > idiom (frozen)

Usage:
    python run_metaphor_study.py
"""

from study_utils import StudyConfig, run_study, ROOT


def main():
    from analysis.holonomy_study.metaphor_datasets import load_metaphor_pairs, by_label, get_paired_only

    cfg = StudyConfig(
        phenomenon_label='metaphorical',
        phenomenon_short='met',
        load_pairs=load_metaphor_pairs,
        by_label=by_label,
        get_paired_only=get_paired_only,
        get_phrase=lambda sp: getattr(sp, 'metaphor', ''),
        colors={
            'metaphorical': '#ff7f0e',   # orange
            'literal':      '#2ca02c',   # green
            'control':      '#1f77b4',   # blue
        },
        output_dir=ROOT / 'results' / 'metaphor_study',
        result_filename='metaphor_results.json',
        dataset_name='metaphor_pairs',
        synthesis_lines=[
            '  Metaphor gauge structure analysis:',
            '',
            '  Compositionality gradient prediction:',
            '    literal  (full):    holonomy baseline, curvature baseline',
            '    metaphor (partial): intermediate effect',
            '    idiom    (none):    holonomy DOWN d=-0.34, curvature UP d=+0.41',
            '',
            '  Gauge-theoretic reading:',
            '    - Metaphors use PRODUCTIVE analogical mappings',
            '    - The connection A should be partially smooth',
            '    - The field strength F should be moderate',
            '    - If confirmed: curvature tracks compositionality gradient',
        ],
    )
    run_study(cfg)


if __name__ == '__main__':
    main()
