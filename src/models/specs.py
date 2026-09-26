BASE_CATEGORICAL = ["CategoryID", "Channel", "ContractTier", "TeamID"]
BASE_NUMERIC = [
    "priority_urgency", "SLACreditClause", "DailyCapacity", "created_hour",
    "created_day_of_week", "is_weekend", "is_after_hours",
]
TIME_NUMERIC = [
    "priority_urgency", "SLACreditClause", "DailyCapacity", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "is_weekend", "is_after_hours", "SLAWindowHours",
    "due_hour", "due_day_of_week", "due_on_weekend",
]
PRESSURE_NUMERIC = [
    "agent_open_backlog", "team_open_backlog", "TeamDailyCapacity",
    "agent_backlog_per_capacity", "team_backlog_per_capacity",
    "agent_arrivals_1h", "agent_arrivals_4h", "team_arrivals_1h", "team_arrivals_4h",
]
HISTORY_NUMERIC = [
    "hist_resolution_burden", "hist_burden_support", "hist_cc_breach_rate",
    "hist_cc_breach_rate_support", "hist_agent_category_breach_rate",
    "hist_agent_category_breach_rate_support", "burden_to_sla",
]
TEXT_NUMERIC = [
    "description_word_count", "description_has_currency", "description_has_insurer",
    "description_has_document", "description_has_appointment",
]

EXPERIMENTS = {
    "E1_baseline": {"categorical": BASE_CATEGORICAL, "numeric": BASE_NUMERIC},
    "E2_sla_time": {"categorical": BASE_CATEGORICAL, "numeric": TIME_NUMERIC},
    "E3_service_pressure": {"categorical": BASE_CATEGORICAL, "numeric": TIME_NUMERIC + PRESSURE_NUMERIC},
    "E4_historical_signal": {
        "categorical": BASE_CATEGORICAL,
        "numeric": TIME_NUMERIC + PRESSURE_NUMERIC + HISTORY_NUMERIC,
    },
    "E5_full_signal": {
        "categorical": BASE_CATEGORICAL + ["category_channel", "priority_contract"],
        "numeric": TIME_NUMERIC + PRESSURE_NUMERIC + HISTORY_NUMERIC + TEXT_NUMERIC,
    },
}
