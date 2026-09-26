from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

NOTE_THEMES = {
    "investigation": r"investigat|review|root cause|reconcil|validat|verif|check",
    "external_party": r"insurer|client|patient|gp\b|practice|provider|hospital",
    "approval": r"authoris|approval|approved|confirm",
    "financial": r"credit|refund|payment|charge|invoice|amount",
    "document": r"report|summary|letter|certificate|record|document",
    "escalation": r"corrective|escalat|complaint|remedial|manager",
}


def add_outcome_teacher_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["response_hours"] = (out["FirstResponseAt"] - out["CreatedAt"]).dt.total_seconds() / 3600
    out["resolution_hours"] = (out["ResolvedAt"] - out["FirstResponseAt"]).dt.total_seconds() / 3600
    notes = out["ResolutionNotes"].fillna("").str.lower()
    theme_cols = []
    for name, pattern in NOTE_THEMES.items():
        col = f"note_{name}"
        out[col] = notes.str.contains(pattern, regex=True).astype(int)
        theme_cols.append(col)
    out["note_action_count"] = out[theme_cols].sum(axis=1)
    return out


def causal_historical_features(df: pd.DataFrame, alpha: float = 20.0) -> pd.DataFrame:
    """Create history features from cases resolved before each ticket's creation time."""
    data = add_outcome_teacher_columns(df).sort_values(["CreatedAt", "TicketID"]).copy()
    out = pd.DataFrame(index=data.index)

    resolved = data.loc[data["ResolvedAt"].notna()].sort_values("ResolvedAt").to_dict("records")
    ptr = 0
    burden_store = defaultdict(list)
    risk_store = defaultdict(lambda: [0, 0])
    global_breaches = 0
    global_cases = 0

    for current_time, batch in data.groupby("CreatedAt", sort=True):
        while ptr < len(resolved) and pd.Timestamp(resolved[ptr]["ResolvedAt"]) < pd.Timestamp(current_time):
            row = resolved[ptr]
            cc = (row["CategoryID"], row["Channel"])
            ac = (row["AssignedAgentID"], row["CategoryID"])
            if pd.notna(row.get("resolution_hours")) and row["resolution_hours"] >= 0:
                burden_store[cc].append(float(row["resolution_hours"]))
            for key in (('cc', cc), ('ac', ac)):
                risk_store[key][0] += int(row["SLABreached"])
                risk_store[key][1] += 1
            global_breaches += int(row["SLABreached"])
            global_cases += 1
            ptr += 1

        global_rate = global_breaches / global_cases if global_cases else data["SLABreached"].mean()
        for idx, row in batch.iterrows():
            cc = (row["CategoryID"], row["Channel"])
            ac = (row["AssignedAgentID"], row["CategoryID"])
            burdens = burden_store.get(cc, [])
            out.loc[idx, "hist_resolution_burden"] = np.median(burdens) if burdens else np.nan
            out.loc[idx, "hist_burden_support"] = len(burdens)
            for name, key in (("hist_cc_breach_rate", ('cc', cc)), ("hist_agent_category_breach_rate", ('ac', ac))):
                b, n = risk_store[key]
                out.loc[idx, name] = (b + alpha * global_rate) / (n + alpha) if n + alpha else global_rate
                out.loc[idx, name + "_support"] = n
    return out.reindex(df.index)


def fit_serving_lookups(df: pd.DataFrame, alpha: float = 20.0) -> dict:
    data = add_outcome_teacher_columns(df)
    complete = data.loc[data["ResolvedAt"].notna()].copy()
    global_rate = float(complete["SLABreached"].mean())

    burden = complete.groupby(["CategoryID", "Channel"])["resolution_hours"].agg(["median", "count"])
    cc = complete.groupby(["CategoryID", "Channel"])["SLABreached"].agg(["sum", "count"])
    ac = complete.groupby(["AssignedAgentID", "CategoryID"])["SLABreached"].agg(["sum", "count"])

    def risk_map(table):
        result = {}
        for key, row in table.iterrows():
            result["|".join(map(str, key if isinstance(key, tuple) else (key,)))] = {
                "rate": float((row["sum"] + alpha * global_rate) / (row["count"] + alpha)),
                "support": int(row["count"]),
            }
        return result

    burden_map = {}
    for key, row in burden.iterrows():
        burden_map["|".join(map(str, key))] = {"median": float(row["median"]), "support": int(row["count"])}

    return {
        "global_breach_rate": global_rate,
        "global_resolution_burden": float(complete["resolution_hours"].median()),
        "burden_category_channel": burden_map,
        "risk_category_channel": risk_map(cc),
        "risk_agent_category": risk_map(ac),
    }
