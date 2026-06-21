"""
This script is only for model definitions: 
1. A linear baseline
2. Gradient boosting
3. MLP
Each is a full pipeline (preprocess -> scale -> model) wrapped so it trains on log1p(price) and predicts back in AED!

The preprocessor is cloned per model so each pipeline fits its own encoders on
the training fold (no shared state, and the target encoder never sees test)
"""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import CFG
from .features import build_preprocessor

SEED = CFG["seed"]

def _log_target(estimator) -> TransformedTargetRegressor:
    # Train on log1p(price); inverse-transform predictions back to AED
    return TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1
    )


def make_models(X_train) -> dict:
    # Build all model pipelines
    pre = build_preprocessor(X_train)

    ridge = Pipeline([
        ("pre", clone(pre)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    hgb = Pipeline([
        ("pre", clone(pre)),
        ("model", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            l2_regularization=1.0, early_stopping=True, random_state=SEED)),
    ])

    # The neural net is the slow one
    mlp = Pipeline([
        ("pre", clone(pre)),
        ("scale", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512,
            early_stopping=True, n_iter_no_change=5, max_iter=40,
            random_state=SEED)),
    ])

    return {
        "ridge": _log_target(ridge),
        "hgb": _log_target(hgb),
        "mlp": _log_target(mlp),
    }
