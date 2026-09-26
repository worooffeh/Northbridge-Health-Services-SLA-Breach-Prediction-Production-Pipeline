from __future__ import annotations

from pathlib import Path
import pandas as pd

TIME_COLUMNS = ["CreatedAt", "FirstResponseAt", "ResolvedAt", "SLADueAt"]


def _read_csv(path: Path) -> pd.DataFrame:
    last = None
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
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


def build_master_table(tickets: pd.DataFrame, agents: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    client_cols = ["ClientID", "ContractTier", "SLACreditClause"]
    master = (
        tickets.merge(
            agents,
            left_on="AssignedAgentID",
            right_on="AgentID",
            how="left",
            validate="many_to_one",
        )
        .merge(clients[client_cols], on="ClientID", how="left", validate="many_to_one")
        .sort_values(["CreatedAt", "TicketID"])
        .reset_index(drop=True)
    )
    return master
