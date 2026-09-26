from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

LEAKAGE_COLUMNS = [
    "FirstResponseAt",
    "ResolvedAt",
    "Status",
    "ResolutionNotes",
    "BreachStatus",
]


def audit_frame(df: pd.DataFrame, target: str = "SLABreached") -> dict:
    missing = df.isna().mean().sort_values(ascending=False)
    duplicate_tickets = int(df["TicketID"].duplicated().sum()) if "TicketID" in df else None
    out = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_ticket_ids": duplicate_tickets,
        "target_prevalence": float(df[target].mean()) if target in df else None,
        "missing_fraction": {k: float(v) for k, v in missing.items()},
        "leakage_columns_present": [c for c in LEAKAGE_COLUMNS if c in df.columns],
        "date_min": str(df["CreatedAt"].min()) if "CreatedAt" in df else None,
        "date_max": str(df["CreatedAt"].max()) if "CreatedAt" in df else None,
    }
    return out


def write_audit(audit: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
