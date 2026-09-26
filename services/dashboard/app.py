from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "experiments" / "validation_results.csv"
DRIFT = ROOT / "reports" / "monitoring" / "drift_report.csv"
META = ROOT / "models" / "champion" / "metadata.json"

prefix = os.getenv("DASH_REQUESTS_PATHNAME_PREFIX", "/")
app = Dash(__name__, requests_pathname_prefix=prefix)
server = app.server

results = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
drift = pd.read_csv(DRIFT) if DRIFT.exists() else pd.DataFrame()
metadata = json.loads(META.read_text()) if META.exists() else {}

children = [
    html.H1("Northbridge SLA Breach Monitoring"),
    html.P("Champion model, experiment comparison and drift status."),
    html.Div([
        html.Strong("Champion: "), html.Span(f"{metadata.get('experiment', 'not trained')} / {metadata.get('model_name', '-')}")
    ]),
]

if not results.empty:
    fig = px.bar(results, x="experiment", y="roc_auc", color="model", barmode="group", title="Validation ROC-AUC")
    children += [dcc.Graph(figure=fig)]
if not drift.empty:
    dfig = px.bar(drift.head(20), x="psi", y="feature", color="status", orientation="h", title="Feature drift (PSI)")
    children += [dcc.Graph(figure=dfig), dash_table.DataTable(drift.to_dict("records"), page_size=10)]

app.layout = html.Div(children, style={"maxWidth": "1200px", "margin": "40px auto", "fontFamily": "Arial"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
