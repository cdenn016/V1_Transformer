#!/usr/bin/env python3
"""
Idiom Holonomy Study: Non-Compositionality as Gauge Curvature
================================================================

Tests whether non-compositional language (idioms) exhibits higher gauge
curvature than compositional language (literal usage of the same phrases).

Usage:
    python run_idiom_study.py
"""

from study_utils import StudyConfig, run_study, ROOT


def main():
    from analysis.holonomy_study.idiom_datasets import load_idiom_pairs, by_label, get_paired_only

    cfg = StudyConfig(
        phenomenon_label='idiomatic',
        phenomenon_short='idiom',
        load_pairs=load_idiom_pairs,
        by_label=by_label,
        get_paired_only=get_paired_only,
        get_phrase=lambda sp: getattr(sp, 'idiom', ''),
        colors={
            'idiomatic': '#d62728',   # red
            'literal':   '#2ca02c',   # green
            'control':   '#1f77b4',   # blue
        },
        output_dir=ROOT / 'results' / 'idiom_study',
        result_filename='idiom_results.json',
        dataset_name='idiom_pairs',
        synthesis_lines=[
            '  Idiom gauge structure analysis:',
            '',
            '  Non-compositionality prediction:',
            '    - Idioms: holonomy DOWN (smooth transport), curvature UP (non-additive)',
            '    - Literal: baseline',
            '',
            '  Gauge-theoretic reading:',
            '    - Idioms have FROZEN compositional structure',
            '    - The connection A is smooth (no path dependence)',
            '    - The field strength F is high (meaning is non-additive)',
        ],
    )
    run_study(cfg)


if __name__ == '__main__':
    main()
