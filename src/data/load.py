from __future__ import annotations

from pathlib import Path
import pandas as pd

TIME_COLUMNS = ["CreatedAt", "FirstResponseAt", "ResolvedAt", "SLADueAt"]


def _read_csv(path: Path) -> pd.DataFrame:
    last = None
    for encoding in ("utf-8", "cp1252", "latin1"): # Try different encodings until one works
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:    # If the current encoding fails, try the next one. Raise the last exception if all fail
            last = exc
    raise last


def load_source_tables(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    tickets = _read_csv(data_dir / "Tickets.csv")
    agents = _read_csv(data_dir / "Agents.csv")
    clients = _read_csv(data_dir / "Clients.csv")

    for col in TIME_COLUMNS:
        if col in tickets:
            tickets[col] = pd.to_datetime(tickets[col], dayfirst=True, errors="coerce")

    for col in ("ContractStartDate", "ContractEndDate"):
        if col in clients:
            clients[col] = pd.to_datetime(clients[col], dayfirst=True, errors="coerce")

    return tickets, agents, clients


def build_ticket_master(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join Tickets -> Clients -> Agents into one model-ready table.

    Only the columns the kick-off doc names as model features (plus the
    identifiers needed to join and the target label) are carried through.
    """
    tickets = tables["Tickets"].copy()          # tables is a dict of DataFrames. tickets is a copy of the "Tickets" sheet.
    clients = tables["Clients"][["ClientID", "ContractTier", "SLACreditClause"]]  # clients is a filtered copy of the "Clients" sheet.
    agents = tables["Agents"][["AgentID", "Hub", "TeamID", "DailyCapacity"]]      # agents is a filtered copy of the "Agents" sheet.

    for col in ("CreatedAt", "FirstResponseAt", "ResolvedAt", "SLADueAt"):
        tickets[col] = pd.to_datetime(tickets[col], errors="coerce")

    master = tickets.merge(clients, on="ClientID", how="left")
    master = master.merge(
        agents, left_on="AssignedAgentID", right_on="AgentID", how="left"
    )
    master = master.drop(columns=["AgentID"])

    unmatched_clients = master["ContractTier"].isna().sum()
    unmatched_agents = master["Hub"].isna().sum()
    if unmatched_clients or unmatched_agents:
        raise ValueError(
            f"Join produced nulls: {unmatched_clients} tickets with no matching "
            f"client, {unmatched_agents} with no matching agent. Check for "
            "orphaned foreign keys in the source data before proceeding."
        )

    return master
