import numpy as np
from pathlib import Path
from transformer.vfe.semantic_clustering.plotting import plot_mu_clustering

def test_plot_writes_pdf_and_png(tmp_path):
    coords = np.random.default_rng(0).normal(size=(20, 2))
    labels = np.array([0]*10 + [1]*10)
    metrics = {"silhouette": 0.5, "n_clusters": 2}
    pdf, png = plot_mu_clustering(coords, labels, metrics, outdir=tmp_path)
    assert Path(pdf).exists() and Path(png).exists()
