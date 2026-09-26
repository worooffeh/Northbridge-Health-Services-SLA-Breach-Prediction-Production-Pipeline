# Architecture

## Principle

Every feature must carry a prediction-time contract: **when was this value actually knowable?**

```mermaid
flowchart LR
  Sources[Ticket / Agent / Client Sources] --> Audit[Schema + Leakage Audit]
  Audit --> PIT[Point-in-Time Feature Engineering]
  PIT --> Experiments[5 Experiments x 3 Classifiers]
  Experiments --> Champion[Champion Selection]
  Champion --> MLflow[MLflow Registry / Artifacts]
  Champion --> Explain[SHAP / Coefficient Explanation]
  Champion --> API[FastAPI Scoring]
  API --> Dash[Dash Operations Dashboard]
  API --> Outcomes[Observed Outcomes]
  Outcomes --> Monitor[Drift + Performance Monitoring]
  Monitor -->|gate passed| Retrain[Retraining Pipeline]
  Retrain --> Experiments
```

## Prediction points

- **T0**: ticket creation.
- **T1**: immediately after agent/team assignment — current production target in this repository.
- **T2**: first response.
- **T3**: in-flight re-scoring.

A field can be leakage at T0 but legal at T2. The timestamped process state, not the column name alone, determines legality.
