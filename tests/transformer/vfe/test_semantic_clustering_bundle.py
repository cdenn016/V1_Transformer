import torch

from transformer.vfe.semantic_clustering.bundle import BeliefBundle


def test_bundle_holds_fields_and_n():
    n, K, n_gen = 5, 8, 4
    b = BeliefBundle(
        mu=torch.zeros(n, K),
        sigma=torch.ones(n, K),
        phi=torch.zeros(n, n_gen),
        token_ids=torch.arange(n),
        token_strings=None,
        generators=None,
        irrep_dims=[K],
        source="vocab",
        layer="final",
        diagonal=True,
    )
    assert b.n == n
    assert b.K == K
    assert b.diagonal is True
