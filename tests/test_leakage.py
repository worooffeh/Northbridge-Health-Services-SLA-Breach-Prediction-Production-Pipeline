from src.data.audit import LEAKAGE_COLUMNS
from src.models.specs import EXPERIMENTS


def test_direct_leakage_columns_never_enter_model_specs():
    used = set()
    for spec in EXPERIMENTS.values():
        used.update(spec["categorical"])
        used.update(spec["numeric"])
    assert not used.intersection(LEAKAGE_COLUMNS)
    assert "SLABreached" not in used
