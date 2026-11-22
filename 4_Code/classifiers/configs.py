"""Hyperparameter configurations of the tuning run. Trial 0 of every classifier is the configuration fixed before tuning;
trials 1..N_TRIALS-1 are drawn at random, without repetition, from the grids below with a fixed
seed, so the list is the same on every machine. The number of trials is the same for all nine classifiers."""
import random

MODELS = ["catboost", "lightgbm", "xgboost", "deepmlp", "ft_transformer", "vchnet", "swingnnet", "fmtunet", "timesenclip"]
PATCH = ["vchnet", "swingnnet", "fmtunet", "timesenclip"]
N_TRIALS = 30
K_FOLDS = 3
DEFAULT = {
    "catboost": dict(iterations=180, depth=6, learning_rate=0.08, l2_leaf_reg=3, class_weight="balanced"),
    "lightgbm": dict(n_estimators=180, num_leaves=31, learning_rate=0.08, min_child_samples=20, colsample_bytree=0.8, subsample=1.0, reg_lambda=0, class_weight="balanced"),
    "xgboost": dict(n_estimators=180, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1, class_weight="none"),
    "deepmlp": dict(hidden=[96, 48], alpha=1e-4, learning_rate_init=1e-3, batch_size="auto"),
    "ft_transformer": dict(d=32, n_layers=2, ff=64, dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=20, batch=256, class_weight="none"),
    "vchnet": dict(d=48, dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=10, batch=64, class_weight="none"),
    "swingnnet": dict(d=48, lr=1e-3, weight_decay=1e-4, epochs=10, batch=64, class_weight="none"),
    "fmtunet": dict(d=32, dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=10, batch=64, class_weight="none"),
    "timesenclip": dict(d=48, lr=1e-3, weight_decay=1e-4, epochs=10, batch=64, class_weight="none"),
}
_LR = [3e-4, 1e-3, 3e-3]; _WD = [1e-5, 1e-4, 1e-3]; _CW = ["none", "balanced"]
GRID = {
    "catboost": dict(iterations=[180, 300, 500, 800], depth=[4, 6, 8], learning_rate=[0.03, 0.05, 0.08, 0.12], l2_leaf_reg=[1, 3, 10], class_weight=["balanced", "sqrt", "none"]),
    "lightgbm": dict(n_estimators=[180, 300, 500, 800], num_leaves=[15, 31, 63], learning_rate=[0.03, 0.05, 0.08, 0.12], min_child_samples=[10, 20, 40],
                     colsample_bytree=[0.6, 0.8, 1.0], subsample=[0.6, 0.8, 1.0], reg_lambda=[0, 1, 10], class_weight=["balanced", "none"]),
    "xgboost": dict(n_estimators=[180, 300, 500, 800], max_depth=[4, 6, 8], learning_rate=[0.03, 0.05, 0.08, 0.12], subsample=[0.6, 0.8, 1.0],
                    colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 3, 5], reg_lambda=[1, 3, 10], class_weight=_CW),
    "deepmlp": dict(hidden=[[96, 48], [128, 64], [256, 128], [128, 64, 32]], alpha=[1e-5, 1e-4, 1e-3, 1e-2], learning_rate_init=_LR, batch_size=["auto", 64, 128]),
    "ft_transformer": dict(d=[32, 64], n_layers=[2, 3], ff=[64, 128], dropout=[0.0, 0.1, 0.2], lr=_LR, weight_decay=_WD, epochs=[20, 40], batch=[128, 256], class_weight=_CW),
    "vchnet": dict(d=[32, 48, 64], dropout=[0.0, 0.1, 0.2], lr=_LR, weight_decay=_WD, epochs=[10, 20], batch=[64, 128], class_weight=_CW),
    "swingnnet": dict(d=[32, 48, 64], lr=_LR, weight_decay=_WD, epochs=[10, 20], batch=[64, 128], class_weight=_CW),
    "fmtunet": dict(d=[32, 48], dropout=[0.0, 0.1, 0.2], lr=_LR, weight_decay=_WD, epochs=[10, 20], batch=[64, 128], class_weight=_CW),
    "timesenclip": dict(d=[32, 48, 96], lr=_LR, weight_decay=_WD, epochs=[10, 20, 40], batch=[64, 128], class_weight=_CW),
}


def trials(model):
    """The N_TRIALS configurations of a classifier (list of dicts); trial 0 is the configuration fixed before tuning."""
    assert all(DEFAULT[model][k] in v for k, v in GRID[model].items()) and set(DEFAULT[model]) == set(GRID[model]), model
    rng = random.Random(1000 + MODELS.index(model)); out = [dict(DEFAULT[model])]; seen = {repr(sorted(DEFAULT[model].items(), key=str))}
    while len(out) < N_TRIALS:
        hp = {k: rng.choice(v) for k, v in GRID[model].items()}; key = repr(sorted(hp.items(), key=str))
        if key not in seen: seen.add(key); out.append(hp)
    return out
