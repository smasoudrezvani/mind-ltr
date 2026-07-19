import numpy as np

from mind_ltr.ranking.lambdas import lambdas_one_group
from mind_ltr.ranking.metrics import ndcg_at_k


def test_lambdas_match_hand_computation():
    """The 3-item toy from the lesson: A relevant, ranked 2nd behind B.

    Hand-derived values (sigma=1):
      pair (A,B): rho=0.62246, |dNDCG|=0.36907 -> 0.229731
      pair (A,C): rho=0.42556, |dNDCG|=0.13093 -> 0.055718
      lambda_A = +0.285449, lambda_B = -0.229731, lambda_C = -0.055718
    """
    labels = np.array([1.0, 0.0, 0.0])       # A, B, C
    scores = np.array([0.5, 1.0, 0.2])
    grad, hess = lambdas_one_group(labels, scores, sigma=1.0)

    np.testing.assert_allclose(grad, [0.285449, -0.229731, -0.055718], atol=1e-5)
    assert np.all(hess >= 0)
    # Forces balance: what pushes up equals what pushes down.
    assert abs(grad.sum()) < 1e-12
    # The top-of-list inversion (A vs B) dominates the forces.
    assert abs(grad[1]) > abs(grad[2])


def test_gradient_ascent_on_lambdas_fixes_the_ranking():
    """Treat scores themselves as parameters; ascending our lambdas must
    drive NDCG to 1.0. This proves the lambdas are valid ascent directions
    for ranking quality, independent of any tree model."""
    rng = np.random.default_rng(0)
    labels = (rng.random(30) < 0.25).astype(float)
    scores = rng.normal(size=30)

    before = ndcg_at_k(labels, scores, k=30)
    for _ in range(300):
        grad, _ = lambdas_one_group(labels, scores)
        scores = scores + 0.5 * grad
    after = ndcg_at_k(labels, scores, k=30)

    assert after > before
    assert after == 1.0
