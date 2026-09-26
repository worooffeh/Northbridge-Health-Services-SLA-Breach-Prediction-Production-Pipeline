from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = {"Phone", "Email", "Form", "Portal"}


def add_base_features(df: pd.DataFrame, sla_due_known: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["created_hour"] = out["CreatedAt"].dt.hour
    out["created_day_of_week"] = out["CreatedAt"].dt.dayofweek
    out["is_weekend"] = (out["created_day_of_week"] >= 5).astype(int)
    out["is_after_hours"] = (
        (out["created_hour"] < 8) | (out["created_hour"] >= 18) | (out["is_weekend"] == 1)
    ).astype(int)
    out["priority_urgency"] = 5 - out["PriorityID"]
    out["hour_sin"] = np.sin(2 * np.pi * out["created_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["created_hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["created_day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["created_day_of_week"] / 7)

    if sla_due_known and "SLADueAt" in out:
        out["SLAWindowHours"] = (
            (out["SLADueAt"] - out["CreatedAt"]).dt.total_seconds() / 3600
        )
        out["due_hour"] = out["SLADueAt"].dt.hour
        out["due_day_of_week"] = out["SLADueAt"].dt.dayofweek
        out["due_on_weekend"] = (out["due_day_of_week"] >= 5).astype(int)
    return out


def add_description_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out.get("Description", pd.Series("", index=out.index)).fillna("").str.lower()
    out["description_word_count"] = text.str.split().str.len().astype(float)
    out["description_has_currency"] = text.str.contains(r"£|\binvoice\b|\bcharge\b|\bpayment\b", regex=True).astype(int)
    out["description_has_insurer"] = text.str.contains(r"insur|bupa|cigna|axa|aviva|wpa|vitality", regex=True).astype(int)
    out["description_has_document"] = text.str.contains(r"report|summary|letter|certificate|record|document", regex=True).astype(int)
    out["description_has_appointment"] = text.str.contains(r"appointment|reschedul|cancel|booking", regex=True).astype(int)
    return out
