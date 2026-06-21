"""
Run:  python -m src.train
"""
from __future__ import annotations
import joblib
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score)
from .config import CFG, resolve
from .models import make_models


def load_split():
    # Load the parquet files
    d = resolve(CFG["paths"]["processed"]).parent
    X_train = pd.read_parquet(d / "X_train.parquet")
    X_test = pd.read_parquet(d / "X_test.parquet")
    y_train = pd.read_parquet(d / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(d / "y_test.parquet").iloc[:, 0]
    print(f"loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test


def score(name: str, y_true, y_pred) -> dict:
    # Compute the regression metrics in AED
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{name}: RMSE={rmse:>13,.0f},  MAE={mae:>13,.0f}, R2={r2:6.3f}, MAPE={mape:7.1%}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def main():
    X_train, X_test, y_train, y_test = load_split()
    models = make_models(X_train)
    results = []
    preds = {"y_true": y_test.to_numpy()}
    best_name, best_rmse, best_model = None, None, None

    for name, model in models.items():
        print(f"\ntraining {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        m = score(name, y_test, y_pred)
        results.append(m)
        if best_rmse is None or m["rmse"] < best_rmse:
            best_name, best_rmse, best_model = name, m["rmse"], model

    # persist metrics (reports/), predictions (data/processed/), and best model (models/)
    reports = resolve(CFG["paths"]["figures"]).parent
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(reports / "metrics.csv", index=False)

    proc = resolve(CFG["paths"]["processed"]).parent
    pd.DataFrame(preds).to_parquet(proc / "test_predictions.parquet", index=False)

    models_dir = resolve(CFG["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "best_model.joblib")

    print(f"\nbest: {best_name}  (RMSE {best_rmse:,.0f})")
    print(f"saved metrics -> {reports / 'metrics.csv'}")
    print(f"saved predictions -> {proc / 'test_predictions.parquet'}")
    print(f"saved best model -> {models_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
