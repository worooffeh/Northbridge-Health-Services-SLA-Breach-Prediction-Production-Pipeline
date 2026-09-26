from __future__ import annotations

import pandas as pd
from .base import add_base_features, add_description_signals
from .pressure import add_pressure_features
from .historical import causal_historical_features


def build_offline_features(master: pd.DataFrame, agents: pd.DataFrame, sla_due_known: bool = True) -> pd.DataFrame:
    out = add_base_features(master, sla_due_known=sla_due_known)
    out = add_pressure_features(out, agents)
    hist = causal_historical_features(out)
    out = pd.concat([out, hist], axis=1)
    out = add_description_signals(out)
    out["category_channel"] = out["CategoryID"].astype(str) + "__" + out["Channel"].astype(str)
    out["priority_contract"] = out["PriorityID"].astype(str) + "__" + out["ContractTier"].astype(str)
    if "SLAWindowHours" in out:
        out["burden_to_sla"] = out["hist_resolution_burden"] / out["SLAWindowHours"].replace(0, pd.NA)
    else:
        out["burden_to_sla"] = pd.NA
    return out
