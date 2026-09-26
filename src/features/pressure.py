from __future__ import annotations

import heapq
import numpy as np
import pandas as pd


def open_backlog_at_creation(df: pd.DataFrame, group_col: str) -> pd.Series:
    result = pd.Series(0.0, index=df.index)
    infinity = pd.Timestamp.max.value
    for _, group in df.groupby(group_col, sort=False):
        group = group.sort_values(["CreatedAt", "TicketID"])
        resolution_heap: list[int] = []
        for created_time, batch in group.groupby("CreatedAt", sort=True):
            now = pd.Timestamp(created_time).value
            while resolution_heap and resolution_heap[0] <= now:
                heapq.heappop(resolution_heap)
            result.loc[batch.index] = len(resolution_heap)
            for idx in batch.index:
                resolved = df.loc[idx, "ResolvedAt"]
                finish = infinity if pd.isna(resolved) else pd.Timestamp(resolved).value
                heapq.heappush(resolution_heap, finish)
    return result


def _recent_counts(df: pd.DataFrame, group_col: str, hours: int) -> pd.Series:
    result = pd.Series(0.0, index=df.index)
    delta = hours * 3600 * 10**9
    for _, group in df.groupby(group_col, sort=False):
        idx = group.sort_values(["CreatedAt", "TicketID"]).index
        times = df.loc[idx, "CreatedAt"].astype("int64").to_numpy()
        previous = np.searchsorted(times, times, side="left")
        left = np.searchsorted(times, times - delta, side="left")
        result.loc[idx] = (previous - left).astype(float)
    return result


def add_pressure_features(df: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["agent_open_backlog"] = open_backlog_at_creation(out, "AssignedAgentID")
    out["team_open_backlog"] = open_backlog_at_creation(out, "TeamID")
    for h in (1, 4):
        out[f"agent_arrivals_{h}h"] = _recent_counts(out, "AssignedAgentID", h)
        out[f"team_arrivals_{h}h"] = _recent_counts(out, "TeamID", h)

    active = agents.loc[agents["IsActive"].eq(1)] if "IsActive" in agents else agents
    team_capacity = active.groupby("TeamID")["DailyCapacity"].sum()
    out["TeamDailyCapacity"] = out["TeamID"].map(team_capacity)
    out["agent_backlog_per_capacity"] = out["agent_open_backlog"] / out["DailyCapacity"].replace(0, np.nan)
    out["team_backlog_per_capacity"] = out["team_open_backlog"] / out["TeamDailyCapacity"].replace(0, np.nan)
    return out
