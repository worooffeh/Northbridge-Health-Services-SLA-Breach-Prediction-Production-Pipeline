import pandas as pd
from src.monitoring.drift import numeric_psi, categorical_psi


def test_identical_series_have_near_zero_psi():
    n = pd.Series(range(100))
    c = pd.Series(["a", "b"] * 50)
    assert numeric_psi(n, n) < 1e-6
    assert categorical_psi(c, c) < 1e-6
