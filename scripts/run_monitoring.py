from pathlib import Path
from src.config import data_dir
from src.data.load import load_source_tables, build_master_table
from src.features.build import build_offline_features
from src.models.artifacts import load_champion
from src.monitoring.drift import drift_report


def main():
    tickets, agents, clients = load_source_tables(data_dir())
    master = build_master_table(tickets, agents, clients)
    features = build_offline_features(master, agents)
    bundle = load_champion("models/champion/model.joblib")
    fitted = bundle["fitted"]

    # Demonstration split: earlier 70% is reference, latest 30% is current.
    cut = int(len(features) * 0.70)
    reference = features.iloc[:cut]
    current = features.iloc[cut:]
    report = drift_report(reference, current, fitted.numeric, fitted.categorical)
    out = Path("reports/monitoring/drift_report.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(report.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
