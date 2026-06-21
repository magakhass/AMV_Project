"""
This script only loads, filters to Sales transactions, drops redundant
columns, and casts types, but it doesnt remove outliers or build features
(that's preprocess.py / features.py)

Run to produce the interim file:
    python -m src.data
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import CFG, resolve


def load_csv(path: str | Path) -> pd.DataFrame:
    # Read a CSV, trying encodings common in UAE government exports (arabic can be encoded in different formats like Windows 1256)
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1256", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
    raise RuntimeError(f"Could not read {path}: {last_err}")


def filter_sales(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only Sales transactions (drop Mortgages and Gifts)
    col = CFG["data"]["sales_filter"]["column"]
    keep = CFG["data"]["sales_filter"]["keep"]
    if col not in df.columns:
        raise KeyError(
            f"Expected filter column '{col}' not found. "
            f"First columns seen: {list(df.columns)[:12]}"
        )
    before = len(df)
    mask = df[col].str.strip().str.casefold() == keep.casefold()
    out = df[mask].copy()
    print(f"filter_sales: {before:,} -> {len(out):,} rows  ({col} == '{keep}')")
    return out


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    # Drop Arabic duplicate columns (_ar) and system ID columns we don't model
    arabic = [c for c in df.columns if c.endswith("_ar")]
    ids = [c for c in CFG["data"]["id_cols"] if c in df.columns]
    to_drop = sorted(set(arabic) | set(ids))
    out = df.drop(columns=to_drop)
    print(f"drop_redundant: removed {len(to_drop)} cols "
          f"({len(arabic)} Arabic, {len(ids)} IDs)")
    return out


# The DLD dictionary claims DD-MM-YYYY, but the real export is ISO (YYYY-MM-DD) and a column can mix formats, so we try each one
DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y")


def parse_dates(s: pd.Series) -> pd.Series:
    # Parse a date column by trying several known formats and combining them. unknown formats get added to DATE_FORMATS
    raw_nonnull = s.notna().sum()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        out.loc[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    parsed = int(out.notna().sum())
    if parsed < raw_nonnull:
        missed = s[out.isna() & s.notna()]
        print(f"parse_dates: {parsed:,}/{raw_nonnull:,} parsed; "
              f"{len(missed):,} unmatched, e.g. {missed.head(5).tolist()}")
    else:
        print(f"parse_dates: all {parsed:,} dates parsed.")
    return out


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast known numeric and date columns from string to proper dtypes.
    Bad values become NaN/NaT (errors='coerce') rather than raising, so errors dont break the code
    """
    df = df.copy()
    numeric = (
        CFG["data"]["numeric_features"]
        + [CFG["data"]["target_col"], CFG["data"]["alt_target_col"]]
    )
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    date_col = CFG["data"]["date_col"]
    if date_col in df.columns:
        df[date_col] = parse_dates(df[date_col])
    return df


def profile(df: pd.DataFrame) -> None:
    # Summary to check
    target = CFG["data"]["target_col"]
    print(f"\nshape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    if target in df.columns:
        print(f"\n{target} summary:\n{df[target].describe().to_string()}")
    nulls = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(f"\nnull % (top 15):\n{nulls.head(15).to_string()}")


def load_sales_table(raw_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(raw_path) if raw_path else resolve(CFG["paths"]["raw"])
    df = load_csv(path)
    df = filter_sales(df)
    df = drop_redundant(df)
    df = cast_types(df)
    return df


if __name__ == "__main__":
    table = load_sales_table()
    profile(table)
    out = resolve(CFG["paths"]["interim"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"\nsaved interim -> {out}")