"""
Adds date features, selects the model columns, and splits temporally (train on
earlier transactions, test on later ones, so the evaluation reflects valuing
future sales rather than memorising the past)

Run:  python -m src.features
"""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from .config import CFG, resolve


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    # Get numeric year and month from the transaction date
    df = df.copy()
    d = CFG["data"]["date_col"]
    df["sale_year"] = df[d].dt.year
    df["sale_month"] = df[d].dt.month
    return df


def get_feature_columns() -> tuple[list[str], list[str]]:
    # Return (numeric_cols, categorical_cols) the model uses
    num = list(CFG["data"]["numeric_features"]) + ["sale_year", "sale_month"]
    cat = list(CFG["data"]["categorical_features"])
    return num, cat


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Split by date: earliest rows going to train, most recent to test
    d = CFG["data"]["date_col"]
    frac = CFG["split"]["test_fraction"]
    df = df.sort_values(d).reset_index(drop=True)
    cut = int(len(df) * (1 - frac))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"temporal_split: train {len(train):,} "
          f"({train[d].min().date()} -> {train[d].max().date()}), "
          f"test {len(test):,} "
          f"({test[d].min().date()} -> {test[d].max().date()})")
    return train, test


def build_preprocessor(X_train: pd.DataFrame, max_onehot: int = 50) -> ColumnTransformer:
    """
    Unfitted preprocessor: median-impute numerics (with a missing indicator),
    one-hot the low-cardinality categoricals, target-encode the high-cardinality ones
    We fit this only on the training data on the model pipelines to not leak the test values
    """
    num_cols, cat_cols = get_feature_columns()
    nunique = X_train[cat_cols].nunique()
    onehot_cols = nunique[nunique <= max_onehot].index.tolist()
    target_cols = nunique[nunique > max_onehot].index.tolist()
    print(f"build_preprocessor: one-hot {onehot_cols}")
    print(f"build_preprocessor: target-encode {target_cols}")

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                     sparse_output=False), onehot_cols),
            ("target", TargetEncoder(target_type="continuous"), target_cols),
        ],
        remainder="drop",
    )

def build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Load cleaned table ==> Date features ==> Temporal split ==> Save
    df = pd.read_parquet(resolve(CFG["paths"]["cleaned"]))
    df = add_date_features(df)

    num_cols, cat_cols = get_feature_columns()
    feat_cols = num_cols + cat_cols
    target = CFG["data"]["target_col"]

    # cast numerics to float
    df[num_cols] = df[num_cols].astype("float64")

    train, test = temporal_split(df)
    X_train, y_train = train[feat_cols], train[target]
    X_test, y_test = test[feat_cols], test[target]

    out = resolve(CFG["paths"]["processed"]).parent
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(out / "X_train.parquet", index=False)
    X_test.to_parquet(out / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(out / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(out / "y_test.parquet", index=False)
    print(f"\nsaved X/y train/test -> {out}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_features()
    build_preprocessor(X_train) # prints the one-hot vs target-encode routing
