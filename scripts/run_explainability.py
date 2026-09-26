from src.config import data_dir
from src.data.load import load_source_tables, build_master_table
from src.features.build import build_offline_features
from src.models.artifacts import load_champion
from src.explainability.shap_utils import global_explanation


def main():
    tickets, agents, clients = load_source_tables(data_dir())
    master = build_master_table(tickets, agents, clients)
    features = build_offline_features(master, agents)
    bundle = load_champion("models/champion/model.joblib")
    summary = global_explanation(bundle["fitted"], features.tail(500), "reports/explainability")
    print(summary)


if __name__ == "__main__":
    main()
