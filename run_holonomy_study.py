#!/usr/bin/env python3
"""
Holonomy Study: Ironic Language as Gauge Curvature
====================================================

Measures gauge curvature in pretrained GPT-2 to test whether ironic language
exhibits different holonomy and curvature than literal language.

Usage:
    python run_holonomy_study.py
"""

from study_utils import StudyConfig, run_study, ROOT


def main():
    from analysis.holonomy_study.datasets import load_irony_pairs, by_label, get_paired_only

    cfg = StudyConfig(
        phenomenon_label='ironic',
        phenomenon_short='ironic',
        load_pairs=load_irony_pairs,
        by_label=by_label,
        get_paired_only=get_paired_only,
        get_phrase=lambda sp: getattr(sp, 'target_phrase', getattr(sp, 'ironic', '')),
        colors={
            'ironic':  '#d62728',   # red
            'literal': '#2ca02c',   # green
            'control': '#1f77b4',   # blue
        },
        output_dir=ROOT / 'results' / 'holonomy_study',
        result_filename='holonomy_results.json',
        dataset_name='irony_pairs',
        synthesis_lines=[
            '  Irony gauge structure analysis:',
            '',
            '  Double dissociation prediction:',
            '    - Holonomy (path defect) may decrease for ironic usage',
            '    - Curvature (superposition violation) may increase',
            '',
            '  Gauge-theoretic reading:',
            '    - Irony inverts surface meaning → non-trivial gauge transport',
            '    - The connection A should show different path structure',
            '    - The field strength F captures non-additive interaction',
        ],
    )
    run_study(cfg)


if __name__ == '__main__':
    main()
