"""
Hyperparameter tuning on the HGB
The search uses TimeSeriesSplit so validation respects temporal order and stays fast.
The best configuration is then refit on the full training set and scored on the held-out test set

Run:  python -m src.tune
"""
from __future__ import annotations
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from .config import CFG, resolve
from .models import make_models
from .train import load_split, score

# For the CV and TimeSeriesSplit
TUNE_ROWS = 250_000 # most recent training rows to search over
N_CANDIDATES = 20 # random configurations to try
N_SPLITS = 3 # TimeSeriesSplit folds

PARAM_DIST = {
    "regressor__model__learning_rate": [0.03, 0.05, 0.08, 0.10, 0.15],
    "regressor__model__max_leaf_nodes": [31, 63, 127, 255],
    "regressor__model__min_samples_leaf": [20, 50, 100, 200],
    "regressor__model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "regressor__model__max_iter": [300, 500, 800],
}

def main():
    X_train, X_test, y_train, y_test = load_split()

    # Search on the most recent slice of train, kept in time order for TimeSeriesSplit
    X_tune, y_tune = X_train.tail(TUNE_ROWS), y_train.tail(TUNE_ROWS)
    print(f"tuning on the most recent {len(X_tune):,} training rows")

    hgb = make_models(X_train)["hgb"] # full train set
    search = RandomizedSearchCV(
        estimator=hgb,
        param_distributions=PARAM_DIST,
        n_iter=N_CANDIDATES,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_root_mean_squared_error",
        random_state=CFG["seed"],
        n_jobs=1, # HGB is already multi-threaded
        verbose=1,
    )
    search.fit(X_tune, y_tune)

    print("\nbest params:")
    for k, v in search.best_params_.items():
        print(f"  {k.split('__')[-1]:20s} {v}")
    print(f"best CV RMSE (AED): {-search.best_score_:,.0f}")

    # Refit on the whole set and then test
    best = clone(hgb).set_params(**search.best_params_)
    best.fit(X_train, y_train)
    print("\ntuned HGB on the held-out test set:")
    score("hgb*", y_test, best.predict(X_test))

    models_dir = resolve(CFG["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, models_dir / "hgb_tuned.joblib")
    pd.DataFrame(search.cv_results_).to_csv(
        resolve(CFG["paths"]["figures"]).parent / "tuning_results.csv", index=False)
    print(f"\nsaved tuned model -> {models_dir / 'hgb_tuned.joblib'}")


if __name__ == "__main__":
    main()
