from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    PriorityID: int = Field(ge=1, le=4)
    CategoryID: int = Field(ge=1, le=5)
    Channel: Literal["Phone", "Email", "Form", "Portal"]
    ContractTier: Literal["Enterprise", "Premium", "Standard"]
    TeamID: int
    AssignedAgentID: int | None = None
    SLACreditClause: int = Field(ge=0, le=1)
    DailyCapacity: float = Field(gt=0)
    TeamDailyCapacity: float | None = Field(default=None, gt=0)
    CreatedAt: datetime
    SLADueAt: datetime | None = None
    Description: str = ""
    agent_open_backlog: float = 0
    team_open_backlog: float = 0
    agent_arrivals_1h: float = 0
    agent_arrivals_4h: float = 0
    team_arrivals_1h: float = 0
    team_arrivals_4h: float = 0


class PredictionResponse(BaseModel):
    breach_probability: float
    predicted_breach: bool
    threshold: float
    model_name: str
    experiment: str
    model_version: str
