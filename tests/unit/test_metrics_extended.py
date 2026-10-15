"""Extended retrieval metric tests."""

from src.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics():
    relevant = ["a", "b"]
    retrieved = ["c", "a", "b"]
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert mrr(relevant, retrieved) == 0.5
    assert ndcg_at_k(relevant, retrieved, 3) > 0
