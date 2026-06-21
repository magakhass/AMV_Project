"""
After data.py, this script:
  - drops columns that are structurally empty for Sales or pure ETL metadata,
  - drops pre-2008 data,
  - trims price/area outliers to acceptable bounds (from config),
  - derives a numeric "bedrooms" count from the "rooms_en" text,
  - normalises categorical text and fills structural nulls with "Unknown".
No encoding/splitting the data - that is in preprocess.py

Run:  python -m src.preprocess
"""
from __future__ import annotations
import re
import pandas as pd
from .config import CFG, resolve

UNKNOWN = "Unknown"

# rooms_en mixes bedroom counts ("2 B/R"), Studio, and non-residential labels
# (like Office, Shop, Store, GYM, PENTHOUSE). Map only the ones with a clear count.
_BR_RE = re.compile(r"^\s*(\d+)\s*B/R", re.IGNORECASE)
_WORD_TO_BEDROOMS = {"studio": 0, "single room": 1}

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop Sales-irrelevant columns
    drop = [c for c in CFG["data"].get("drop_cols", []) if c in df.columns]
    out = df.drop(columns=drop)
    print(f"drop_unused_columns: removed {drop}")
    return out


def filter_by_year(df: pd.DataFrame) -> pd.DataFrame:
    # Drop pre-2008 based on EDA findings
    min_year = CFG["cleaning"].get("min_instance_year")
    if not min_year:
        return df
    d = CFG["data"]["date_col"]
    before = len(df)
    out = df[df[d].dt.year >= min_year].copy()
    print(f"filter_by_year: {before:,} -> {len(out):,} rows (year >= {min_year})")
    return out

def trim_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows whose price and area fall within acceptable bounds (config)
    c = CFG["cleaning"]
    target = CFG["data"]["target_col"]
    area = "procedure_area"
    before = len(df)
    keep = (
        df[target].between(c["min_price"], c["max_price"])
        & df[area].between(c["min_area"], c["max_area"])
    )
    out = df[keep].copy()
    print(f"trim_outliers: {before:,} -> {len(out):,} rows  "
          f"(price [{c['min_price']:,}-{c['max_price']:,}], "
          f"area [{c['min_area']}-{c['max_area']}])")
    return out


def parse_bedrooms(rooms: pd.Series) -> pd.Series:
    """
    Turn the  "rooms_en" text into a numeric bedroom count.
    "N B/R" = N, "Studio" = 0, "Single Room" = 1. Non-residential labels
    (Office, Shop, Store, GYM, PENTHOUSE) and blanks become "NA", since they aren't a bedroom count
    """
    def one(value):
        if not isinstance(value, str):
            return pd.NA
        word = value.strip().lower()
        if word in _WORD_TO_BEDROOMS:
            return _WORD_TO_BEDROOMS[word]
        match = _BR_RE.match(value)
        return int(match.group(1)) if match else pd.NA

    return rooms.map(one).astype("Int64")


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # A missing nearest-metro or sub-type is informative (land has none), so we label it "Unknown" rather than drop the row
    df = df.copy()
    for c in CFG["data"]["categorical_features"]:
        if c in df.columns:
            df[c] = (
                df[c].astype("string").str.strip()
                .replace("", pd.NA)
                .replace("أخرى", pd.NA) # from EDA
                .fillna(UNKNOWN)
            )
    return df


def cast_party_counts(df: pd.DataFrame) -> pd.DataFrame:
    # Cast the no_of_parties_role_* string columns to numeric
    df = df.copy()
    for c in [c for c in df.columns if c.startswith("no_of_parties_role_")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def preprocess(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(resolve(CFG["paths"]["interim"]))
    df = drop_unused_columns(df)
    df = trim_outliers(df)
    df["bedrooms"] = parse_bedrooms(df["rooms_en"])
    df = clean_categoricals(df)
    df = cast_party_counts(df)
    print(f"\ncleaned shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"bedrooms non-null: {df['bedrooms'].notna().sum():,} "
          f"({df['bedrooms'].notna().mean() * 100:.1f}%)")
    return df


if __name__ == "__main__":
    cleaned = preprocess()
    out = resolve(CFG["paths"]["cleaned"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out, index=False)
    print(f"\nsaved cleaned -> {out}")
