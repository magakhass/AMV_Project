"""
Unit tests for feature engineering — date features, temporal split, and the
encoder routing on a small synthetic frames
"""
import pandas as pd
from src.config import CFG
from src.features import (add_date_features, build_preprocessor, get_feature_columns, temporal_split)

def test_add_date_features():
    d = CFG["data"]["date_col"]
    df = pd.DataFrame({d: pd.to_datetime(["2022-11-09", "2025-06-02"])})
    out = add_date_features(df)
    assert out["sale_year"].tolist() == [2022, 2025]
    assert out["sale_month"].tolist() == [11, 6]


def test_temporal_split_is_ordered():
    d = CFG["data"]["date_col"]
    df = pd.DataFrame({d: pd.to_datetime([f"2020-01-{i:02d}" for i in range(1, 11)])})
    train, test = temporal_split(df)
    # every training date precedes every test date, and sizes follow the fraction
    assert train[d].max() <= test[d].min()
    assert len(test) == int(len(df) * CFG["split"]["test_fraction"])


def test_build_preprocessor_routes_by_cardinality():
    num_cols, cat_cols = get_feature_columns()
    # high-cardinality area (>50 distinct) should be target-encoded; the rest one-hot
    rows = 120
    data = {c: ["A"] * rows for c in cat_cols}
    data["area_name_en"] = [f"community_{i % 60}" for i in range(rows)]
    pre = build_preprocessor(pd.DataFrame(data))

    routing = {name: cols for name, _, cols in pre.transformers}
    assert "area_name_en" in routing["target"]
    assert "property_type_en" in routing["onehot"]
