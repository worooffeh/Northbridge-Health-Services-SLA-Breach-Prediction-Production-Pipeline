from src.models.evaluate import classification_metrics


def test_metrics_are_bounded():
    y = [0, 0, 1, 1]
    p = [0.1, 0.3, 0.7, 0.9]
    m = classification_metrics(y, p)
    assert 0 <= m["roc_auc"] <= 1
    assert 0 <= m["pr_auc"] <= 1
    assert m["lift_at_10pct"] >= 0
