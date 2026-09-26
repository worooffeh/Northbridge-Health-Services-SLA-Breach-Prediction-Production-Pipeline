from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _psi_from_props(ref, cur):
    ref = np.clip(np.asarray(ref, dtype=float), EPS, None)
    cur = np.clip(np.asarray(cur, dtype=float), EPS, None)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    r = reference.dropna().astype(float)
    c = current.dropna().astype(float)
    if r.nunique() < 2 or len(r) == 0 or len(c) == 0:
        return 0.0
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    rp = pd.cut(r, edges, include_lowest=True).value_counts(sort=False).to_numpy()
    cp = pd.cut(c, edges, include_lowest=True).value_counts(sort=False).reindex(
        pd.cut(r, edges, include_lowest=True).cat.categories, fill_value=0
    ).to_numpy()
    return _psi_from_props(rp, cp)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    rp = reference.fillna("__missing__").astype(str).value_counts().reindex(cats + (["__missing__"] if reference.isna().any() or current.isna().any() else []), fill_value=0)
    cp = current.fillna("__missing__").astype(str).value_counts().reindex(rp.index, fill_value=0)
    return _psi_from_props(rp.to_numpy(), cp.to_numpy())


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric:
        if col in reference and col in current:
            rows.append({"feature": col, "type": "numeric", "psi": numeric_psi(reference[col], current[col])})
    for col in categorical:
        if col in reference and col in current:
            rows.append({"feature": col, "type": "categorical", "psi": categorical_psi(reference[col], current[col])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["status"] = pd.cut(out["psi"], [-np.inf, 0.10, 0.25, np.inf], labels=["stable", "watch", "critical"])
    return out.sort_values("psi", ascending=False)
