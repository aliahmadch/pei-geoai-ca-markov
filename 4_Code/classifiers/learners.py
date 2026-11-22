"""Fit and predict for the nine classifiers with a configuration (dict of settings, see configs.py). Class codes are 1-5
(forest, agricultural land, wetland, settlement, otherland); the networks use a fixed seed."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import nets

SEED = 42
N_CLASS = 5
TREES = ("catboost", "lightgbm", "xgboost")
NETS = ("ft_transformer", "vchnet", "swingnnet", "fmtunet", "timesenclip")


def make_tree(name, hp, threads):
    cw = hp.get("class_weight", "none")
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=hp["iterations"], depth=hp["depth"], learning_rate=hp["learning_rate"], l2_leaf_reg=hp["l2_leaf_reg"],
                                  loss_function="MultiClass", random_seed=SEED, verbose=False, allow_writing_files=False, thread_count=threads,
                                  auto_class_weights={"balanced": "Balanced", "sqrt": "SqrtBalanced", "none": None}[cw])
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=hp["n_estimators"], num_leaves=hp["num_leaves"], learning_rate=hp["learning_rate"], min_child_samples=hp["min_child_samples"],
                              colsample_bytree=hp["colsample_bytree"], subsample=hp["subsample"], subsample_freq=1 if hp["subsample"] < 1 else 0, reg_lambda=hp["reg_lambda"],
                              random_state=SEED, class_weight="balanced" if cw == "balanced" else None, verbosity=-1, n_jobs=threads)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["learning_rate"], subsample=hp["subsample"],
                             colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"], reg_lambda=hp["reg_lambda"], objective="multi:softprob",
                             num_class=N_CLASS, tree_method="hist", random_state=SEED, n_jobs=threads, verbosity=0)
    raise KeyError(name)


def make_mlp(hp, cls=MLPClassifier):
    return cls(hidden_layer_sizes=tuple(hp["hidden"]), activation="relu", alpha=hp["alpha"], learning_rate_init=hp["learning_rate_init"], batch_size=hp["batch_size"],
               max_iter=250, random_state=SEED, early_stopping=True, n_iter_no_change=15)


def fit_net(name, hp, Xtr, ytr, seed=SEED, threads=1, Xte=None, yte=None, log=None):
    """AdamW, cross-entropy, batches from a random permutation per epoch, fixed number of epochs. With log, loss and accuracy on
    the training and test pixels are recorded after every epoch (this does not change the trained network)."""
    torch.set_num_threads(threads); torch.manual_seed(seed); np.random.seed(seed)
    patch = Xtr.ndim == 4
    if patch: mean = Xtr.mean(axis=(0, 2, 3), keepdims=True); std = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    else: mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    A = torch.from_numpy(((Xtr - mean) / std).astype(np.float32)); ya = torch.from_numpy(ytr.astype(np.int64) - 1)
    model = nets.make_net(name, hp)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    w = None
    if hp.get("class_weight", "none") == "balanced":
        cnt = np.bincount(ytr.astype(np.int64) - 1, minlength=N_CLASS); w = torch.tensor(len(ytr) / (N_CLASS * np.maximum(cnt, 1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w); lf_eval = nn.CrossEntropyLoss()
    if log is not None:
        B = torch.from_numpy(((Xte - mean) / std).astype(np.float32)); yb = torch.from_numpy(yte.astype(np.int64) - 1)

        def ev(Xv, Yv):
            model.eval(); L = 0.0; C = 0
            with torch.no_grad():
                for i in range(0, len(Xv), 512):
                    o = model(Xv[i:i + 512]); L += lf_eval(o, Yv[i:i + 512]).item() * len(o); C += (o.argmax(1) == Yv[i:i + 512]).sum().item()
            model.train(); return L / len(Xv), C / len(Xv)
    model.train()
    for ep in range(1, hp["epochs"] + 1):
        perm = torch.randperm(len(A))
        for i in range(0, len(A), hp["batch"]):
            idx = perm[i:i + hp["batch"]]; opt.zero_grad(); lf(model(A[idx]), ya[idx]).backward(); opt.step()
        if log is not None:
            trl, tra = ev(A, ya); tel, tea = ev(B, yb)
            log.append(dict(model=name, epoch=ep, train_loss=trl, train_acc=tra, test_loss=tel, test_acc=tea, n_train=len(A), n_test=len(B)))
    return {"name": name, "kind": "net", "hp": dict(hp), "state": {k: v.clone() for k, v in model.state_dict().items()}, "mean": np.asarray(mean, np.float32), "std": np.asarray(std, np.float32)}


def fit(name, hp, Xtr, ytr, seed=SEED, threads=1):
    y0 = ytr.astype(np.int64) - 1
    if name in TREES:
        m = make_tree(name, hp, threads)
        if name == "xgboost" and hp.get("class_weight", "none") == "balanced": m.fit(Xtr, y0, sample_weight=compute_sample_weight("balanced", y0))
        else: m.fit(Xtr, y0)
        return {"name": name, "kind": "tree", "hp": dict(hp), "model": m}
    if name == "deepmlp":
        sc = StandardScaler(); m = make_mlp(hp); m.fit(sc.fit_transform(Xtr), y0)
        return {"name": name, "kind": "mlp", "hp": dict(hp), "model": m, "scaler": sc}
    return fit_net(name, hp, Xtr, ytr, seed, threads)


@torch.no_grad()
def predict(pack, X, threads=1):
    """Class codes 1-5."""
    if pack["kind"] == "tree": return np.asarray(pack["model"].predict(X)).reshape(-1).astype(np.int64) + 1
    if pack["kind"] == "mlp": return np.asarray(pack["model"].predict(pack["scaler"].transform(X))).astype(np.int64) + 1
    torch.set_num_threads(threads)
    model = nets.make_net(pack["name"], pack["hp"]); model.load_state_dict(pack["state"]); model.eval()
    Xs = torch.from_numpy(((X - pack["mean"]) / pack["std"]).astype(np.float32)); out = []
    for i in range(0, len(Xs), 512): out.append(model(Xs[i:i + 512]).argmax(1).numpy())
    return np.concatenate(out).astype(np.int64) + 1
