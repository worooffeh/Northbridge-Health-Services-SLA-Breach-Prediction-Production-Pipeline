from __future__ import annotations

import numpy as np
import pandas as pd


def build_online_frame(payload: dict, lookups: dict) -> pd.DataFrame:
    row = dict(payload)
    created = pd.Timestamp(row["CreatedAt"])
    due = pd.Timestamp(row["SLADueAt"]) if row.get("SLADueAt") else pd.NaT
    row["created_hour"] = created.hour
    row["created_day_of_week"] = created.dayofweek
    row["is_weekend"] = int(created.dayofweek >= 5)
    row["is_after_hours"] = int(created.hour < 8 or created.hour >= 18 or created.dayofweek >= 5)
    row["priority_urgency"] = 5 - int(row["PriorityID"])
    row["hour_sin"] = np.sin(2 * np.pi * created.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * created.hour / 24)
    row["dow_sin"] = np.sin(2 * np.pi * created.dayofweek / 7)
    row["dow_cos"] = np.cos(2 * np.pi * created.dayofweek / 7)
    if pd.notna(due):
        row["SLAWindowHours"] = (due - created).total_seconds() / 3600
        row["due_hour"] = due.hour
        row["due_day_of_week"] = due.dayofweek
        row["due_on_weekend"] = int(due.dayofweek >= 5)

    row.setdefault("TeamDailyCapacity", float(row.get("DailyCapacity", 1.0)))
    for name in ["agent_open_backlog", "team_open_backlog", "agent_arrivals_1h", "agent_arrivals_4h", "team_arrivals_1h", "team_arrivals_4h"]:
        row.setdefault(name, 0.0)
    row["agent_backlog_per_capacity"] = row["agent_open_backlog"] / max(float(row.get("DailyCapacity", 1.0)), 1e-9)
    row["team_backlog_per_capacity"] = row["team_open_backlog"] / max(float(row.get("TeamDailyCapacity", 1.0)), 1e-9)

    cc_key = f'{row["CategoryID"]}|{row["Channel"]}'
    ac_key = f'{row.get("AssignedAgentID", "")}|{row["CategoryID"]}'
    burden = lookups.get("burden_category_channel", {}).get(cc_key, {})
    cc_risk = lookups.get("risk_category_channel", {}).get(cc_key, {})
    ac_risk = lookups.get("risk_agent_category", {}).get(ac_key, {})
    row["hist_resolution_burden"] = burden.get("median", lookups.get("global_resolution_burden"))
    row["hist_burden_support"] = burden.get("support", 0)
    row["hist_cc_breach_rate"] = cc_risk.get("rate", lookups.get("global_breach_rate"))
    row["hist_cc_breach_rate_support"] = cc_risk.get("support", 0)
    row["hist_agent_category_breach_rate"] = ac_risk.get("rate", lookups.get("global_breach_rate"))
    row["hist_agent_category_breach_rate_support"] = ac_risk.get("support", 0)

    text = str(row.get("Description", "")).lower()
    row["description_word_count"] = len(text.split())
    row["description_has_currency"] = int(any(t in text for t in ["£", "invoice", "charge", "payment"]))
    row["description_has_insurer"] = int(any(t in text for t in ["insur", "bupa", "cigna", "axa", "aviva", "wpa", "vitality"]))
    row["description_has_document"] = int(any(t in text for t in ["report", "summary", "letter", "certificate", "record", "document"]))
    row["description_has_appointment"] = int(any(t in text for t in ["appointment", "reschedul", "cancel", "booking"]))
    row["category_channel"] = f'{row["CategoryID"]}__{row["Channel"]}'
    row["priority_contract"] = f'{row["PriorityID"]}__{row["ContractTier"]}'
    if row.get("SLAWindowHours"):
        row["burden_to_sla"] = row["hist_resolution_burden"] / row["SLAWindowHours"]
    else:
        row["burden_to_sla"] = np.nan
    return pd.DataFrame([row])
