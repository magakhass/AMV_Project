"""
Unit tests for the cleaning pipeline.
These run on tiny synthetic frames (no real dataset needed), to verify the core logic
"""
import pandas as pd
from src.config import CFG
from src.data import filter_sales
from src.preprocess import (clean_categoricals, drop_unused_columns, filter_by_year, parse_bedrooms, trim_outliers)


def test_filter_sales_keeps_only_sales():
    df = pd.DataFrame({"trans_group_en": ["Sales", "Mortgages", "Gifts", "sales "]})
    out = filter_sales(df)
    # both "Sales" and the padded/lower "sales " kept and the rest are dropped
    assert len(out) == 2
    assert out["trans_group_en"].str.strip().str.casefold().eq("sales").all()


def test_filter_by_year_drops_pre_cutoff():
    cutoff = CFG["cleaning"]["min_instance_year"]
    df = pd.DataFrame({CFG["data"]["date_col"]: pd.to_datetime(
        [f"{cutoff - 3}-01-01", f"{cutoff}-06-01", f"{cutoff + 5}-09-01"])})
    out = filter_by_year(df)
    assert (out[CFG["data"]["date_col"]].dt.year >= cutoff).all()
    assert len(out) == 2


def test_trim_outliers_respects_bounds():
    c = CFG["cleaning"]
    df = pd.DataFrame({
        CFG["data"]["target_col"]: [1, 60_000, 2_000_000, 5e8],
        "procedure_area": [5, 100, 200, 100_000],
    })
    out = trim_outliers(df)
    assert out[CFG["data"]["target_col"]].between(c["min_price"], c["max_price"]).all()
    assert out["procedure_area"].between(c["min_area"], c["max_area"]).all()
    assert len(out) == 2 # only the two middle rows pass both bounds


def test_parse_bedrooms_mapping():
    s = pd.Series(["1 B/R", "Studio", "2 B/R", "Office", "Single Room", None])
    out = parse_bedrooms(s)
    assert out.tolist()[:3] == [1, 0, 2]
    assert out[5] is pd.NA # missing is NA
    assert out[3] is pd.NA # "Office" is not a bedroom count
    assert out[4] == 1 # "Single Room" equals 1


def test_clean_categoricals_fills_and_strips():
    col = "property_usage_en" # one of the configured categoricals
    df = pd.DataFrame({col: [" Residential ", None, ""]})
    out = clean_categoricals(df)
    assert out[col].isna().sum() == 0
    assert out[col].tolist() == ["Residential", "Unknown", "Unknown"]


def test_drop_unused_columns():
    drop = CFG["data"]["drop_cols"]
    df = pd.DataFrame({**{c: [1] for c in drop}, "keep_me": [1]})
    out = drop_unused_columns(df)
    assert not any(c in out.columns for c in drop)
    assert "keep_me" in out.columns
