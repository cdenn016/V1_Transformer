import numpy as np
from pathlib import Path
from transformer.vfe.semantic_clustering.plotting import plot_mu_clustering

def test_plot_writes_pdf_and_png(tmp_path):
    coords = np.random.default_rng(0).normal(size=(20, 2))
    labels = np.array([0]*10 + [1]*10)
    metrics = {"silhouette": 0.5, "n_clusters": 2}
    pdf, png = plot_mu_clustering(coords, labels, metrics, outdir=tmp_path)
    assert Path(pdf).exists() and Path(png).exists()


def test_select_annotation_skips_empty_and_spreads_across_clusters():
    from transformer.vfe.semantic_clustering.plotting import (
        _select_annotation_indices,
    )
    coords = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]], dtype=float
    )
    labels = np.array([0, 0, 1, 1])
    # index 0 is a byte-fallback that sanitized to '' upstream; it must be skipped
    strings = ["", "the", "of", "and"]
    sel = _select_annotation_indices(coords, labels, strings, max_annot=10)
    assert 0 not in sel               # empty label dropped
    assert set(sel) == {1, 2, 3}      # all informative points, both clusters
    # budget cap is honoured
    assert len(_select_annotation_indices(coords, labels, strings, max_annot=1)) == 1


def test_sanitize_label_strips_replacement_char():
    from transformer.vfe.semantic_clustering.pipeline import _sanitize_label
    assert _sanitize_label("�") == ""       # byte-fallback collapses to empty
    assert _sanitize_label("a�\x00b") == "ab"
    assert _sanitize_label(" the") == " the"      # real labels are untouched
