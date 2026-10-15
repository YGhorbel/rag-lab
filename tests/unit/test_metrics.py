"""Lightweight metric helper tests."""

from src.evaluation.metrics import recall_at_k


def test_recall_at_k():
    relevant = ["a", "b"]
    retrieved = ["b", "c", "a"]
    assert recall_at_k(relevant, retrieved, k=2) == 0.5
    assert recall_at_k(relevant, retrieved, k=3) == 1.0
