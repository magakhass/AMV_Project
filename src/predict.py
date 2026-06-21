"""
Value a property with the trained AVM.
Mirrors the pipeline's feature engineering so a raw record becomes the columns the model expects, then returns a price in AED.
Unknown/new categorical values are handled by the fitted encoders, so records with unseen communities or some other features still score.

Library:  from src.predict import value_property
CLI:      python -m src.predict path/to/records.json
          python -m src.predict # runs a worked example
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
from .config import CFG, resolve
from .features import add_date_features, get_feature_columns
from .preprocess import UNKNOWN, parse_bedrooms

def _load_model():
    # Load the tuned model if present, else the best baseline model
    models = resolve(CFG["paths"]["models"])
    for name in ("hgb_tuned.joblib", "best_model.joblib"):
        path = models / name
        if path.exists():
            return joblib.load(path), name
    raise FileNotFoundError(
        f"No model found in {models}. Run python -m src.train (and tune.py) first.")


def prepare(records) -> pd.DataFrame:
    # Turn raw property record into the models feature columns
    df = pd.DataFrame(records if isinstance(records, list) else [records])
    num_cols, cat_cols = get_feature_columns()
    d = CFG["data"]["date_col"]

    # sale date to sale_year / sale_month (default to today if blank)
    if d in df.columns:
        df[d] = pd.to_datetime(df[d], errors="coerce").fillna(pd.Timestamp.today())
    else:
        df[d] = pd.Timestamp.today()
    df = add_date_features(df)

    # bedrooms derived from rooms_en, same as preprocess
    df["bedrooms"] = parse_bedrooms(df["rooms_en"]) if "rooms_en" in df.columns else pd.NA

    # guarantee every feature column exists and is the right type
    for c in cat_cols:
        if c not in df.columns:
            df[c] = UNKNOWN
        df[c] = (df[c].astype("string").str.strip().replace("", pd.NA).fillna(UNKNOWN))
    for c in num_cols:
        if c not in df.columns:
            df[c] = float("nan")
    df[num_cols] = df[num_cols].astype("float64")

    return df[num_cols + cat_cols]


def value_property(records, model=None) -> list:
    # Return predicted price in AED for raw property record
    if model is None:
        model, _ = _load_model()
    return model.predict(prepare(records)).tolist()


def main():
    if len(sys.argv) > 1:
        records = json.loads(Path(sys.argv[1]).read_text())
    else:
        records = [{
            "property_type_en": "Unit", "property_sub_type_en": "Flat",
            "property_usage_en": "Residential", "reg_type_en": "Existing Properties",
            "procedure_name_en": "Sell", "rooms_en": "2 B/R",
            "area_name_en": "Business Bay", "nearest_metro_en": "Business Bay",
            "nearest_mall_en": "Dubai Mall", "nearest_landmark_en": "Burj Khalifa",
            "has_parking": 1, "procedure_area": 95.0, "instance_date": "2026-06-01",
        }]

    model, name = _load_model()
    for rec, price in zip(records if isinstance(records, list) else [records], value_property(records, model=model)):
        area = rec.get("area_name_en", "?")
        rooms = rec.get("rooms_en", "?")
        print(f"{rooms} in {area}: {price:,.0f} AED")
    print(f"(model: {name})")


if __name__ == "__main__":
    main()
