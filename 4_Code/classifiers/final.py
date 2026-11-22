"""Final models of one classifier, all five years, one CPU thread.
  python final.py <model> tuned      selected configuration, main partition (seed 42): fit on all training blocks, classify the
                                     test blocks; learning curves are logged while training; models and predictions are saved
  python final.py <model> default    the same with the configuration fixed before tuning (trial 0), for reference
  python final.py <model> alt        selected configuration, alternative partition (seed 1): wall-clock time of fit plus
                                     classification of the common test pixels, and the predictions
Outputs: results/final/<part>/ (predictions, curves, timing) and results/models/ (tuned models, portable formats)."""
import os, sys, json, time
model, part = sys.argv[1], sys.argv[2]
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[v] = "1"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA = os.environ.get("PEI_DATA", os.path.join(HERE, "..", "..", "data"))   # data/tables and data/patches of the data set
from common import spatial_mask, feature_matrix
import configs, learners
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except Exception: pass
QUICK = os.environ.get("PEI5_QUICK") == "1"
R = os.path.join(HERE, "results_quick" if QUICK else "results"); O = os.path.join(R, "final", part); MD = os.path.join(R, "models")
os.makedirs(O, exist_ok=True); os.makedirs(MD, exist_ok=True)
done = os.path.join(O, "%s.done" % model)
if os.path.exists(done): print("exists", done); sys.exit(0)
for f in os.listdir(O):                                    # restart of an unfinished run: drop its partial files
    if f.startswith(model + "_") or f.startswith("curves_%s" % model) or f.startswith("timing_%s" % model): os.remove(os.path.join(O, f))
if part == "default": hp = dict(configs.DEFAULT[model]); trial = 0
else:
    s = json.load(open(R + "/selected.json"))["selected"][model]; hp = s["hp"]; trial = s["trial"]
run = dict(hp)
if QUICK:
    for k, v in (("epochs", 1), ("iterations", 20), ("n_estimators", 20)):
        if k in run: run[k] = v
L1 = np.array([0, 1, 2, 2, 3, 4, 5]); patch = model in configs.PATCH; PSEED = 1 if part == "alt" else 42
years = [2021] if QUICK else [2021, 2022, 2023, 2024, 2025]


class LoggedMLP(MLPClassifier):
    """Records the accuracy of every epoch; the fit itself is that of MLPClassifier."""
    def _update_no_improvement_count(self, early_stopping, X, y, *args):
        super()._update_no_improvement_count(early_stopping, X, y, *args)
        rec = {}
        for nm, (Xs, ys) in dict(self._sets, val=(X, np.asarray(y))).items():
            p = self._forward_pass_fast(Xs).argmax(1); rec[nm] = float((p == ys).mean()); rec["n_" + nm] = len(ys)
        self._log.append(rec)


def n_trees(run): return run.get("iterations", run.get("n_estimators"))


def staged(m, X, it):
    if model == "catboost": return m.predict_proba(X, ntree_end=it)
    if model == "lightgbm": return m.predict_proba(X, num_iteration=it)
    return m.predict_proba(X, iteration_range=(0, it))


def logloss(P, y0): return float(-np.log(np.clip(P[np.arange(len(y0)), y0], 1e-15, 1)).mean())


def save_model(pack, year):
    b = os.path.join(MD, "%d_%s" % (year, model))
    if model == "catboost": pack["model"].save_model(b + ".cbm")
    elif model == "lightgbm": pack["model"].booster_.save_model(b + ".txt")
    elif model == "xgboost": pack["model"].save_model(b + ".json")
    elif model == "deepmlp":
        m, sc = pack["model"], pack["scaler"]
        np.savez(b + ".npz", mean=sc.mean_, scale=sc.scale_, classes=m.classes_, n_layers=len(m.coefs_), **{"W%d" % i: w for i, w in enumerate(m.coefs_)}, **{"b%d" % i: w for i, w in enumerate(m.intercepts_)})
        joblib.dump({"model": m, "scaler": sc}, b + ".joblib")
    else: torch.save({"name": model, "hp": pack["hp"], "state": pack["state"], "mean": pack["mean"], "std": pack["std"]}, b + ".pt")


timing = []
for year in years:
    df = pd.read_csv(f"{DATA}/tables/train_{year}.csv"); X, y6 = feature_matrix(df); y = L1[y6]; tr = spatial_mask(df, seed=PSEED)
    z = np.load(f"{DATA}/patches/patches_{year}.npz"); P, ok = z["P"], z["ok"]; te = ~tr; common = te & ok
    fit_rows = tr & ok if patch else tr; F = P if patch else X; t0 = time.time()
    if part == "alt":
        t = time.perf_counter(); pack = learners.fit(model, run, F[fit_rows], y[fit_rows], seed=PSEED, threads=1); p = learners.predict(pack, F[common], threads=1); secs = time.perf_counter() - t
        timing.append(dict(seed=PSEED, year=year, model=model, trial=trial, secs=secs, oa=float((p == y[common]).mean()), n_train=int(fit_rows.sum()), n_test=int(common.sum())))
        np.savez_compressed(f"{O}/{model}_{year}.npz", idx=np.flatnonzero(common), y=y[common].astype(np.uint8), block=df.block_id.to_numpy()[common], pred=p.astype(np.uint8))
        print(year, model, part, "OA %.4f" % timing[-1]["oa"], "%.1f s" % secs, flush=True); continue
    ev_rows = common if patch else te                       # patch networks are scored where a 9 x 9 window exists
    if model in learners.TREES:
        pack = learners.fit(model, run, F[fit_rows], y[fit_rows], threads=1); p = learners.predict(pack, F[ev_rows]); n = n_trees(run); rows = []
        for it in sorted(set(list(range(1, 10)) + list(range(10, n + 1, 5)) + [n])):
            for nm, msk in (("train", fit_rows), ("test", ev_rows)):
                Pr = staged(pack["model"], F[msk], it)
                rows.append(dict(year=year, model=model, iteration=it, set=nm, n=int(msk.sum()), acc=float((Pr.argmax(1) + 1 == y[msk]).mean()), loss=logloss(Pr, y[msk] - 1)))
        assert abs(rows[-1]["acc"] - (p == y[ev_rows]).mean()) < 1e-12
        cur = pd.DataFrame(rows)
    elif model == "deepmlp":
        sc = StandardScaler(); Xs = sc.fit_transform(F[fit_rows]); Xt = sc.transform(F[ev_rows]); y0 = y - 1
        try:
            m = learners.make_mlp(run, LoggedMLP); m._sets = dict(all=(Xs, y0[fit_rows]), test=(Xt, y0[ev_rows])); m._log = []
            m.fit(Xs, y0[fit_rows])
            cur = pd.DataFrame(m._log); cur.insert(0, "epoch", np.arange(1, len(cur) + 1)); cur.insert(0, "model", model); cur.insert(0, "year", year)
            assert np.allclose(cur.val, m.validation_scores_)
            cur["train_loss"] = m.loss_curve_; cur["fit"] = (cur["all"] * cur.n_all - cur["val"] * cur.n_val) / (cur.n_all - cur.n_val)
            cur["best_epoch"] = int(np.argmax(m.validation_scores_)) + 1
            del m._sets, m._log; m.__class__ = MLPClassifier
        except Exception as e:                              # other scikit-learn internals: fit without the per-epoch test curve
            print("logging hook not usable (%r); plain fit" % (e,), flush=True)
            m = learners.make_mlp(run); m.fit(Xs, y0[fit_rows])
            cur = pd.DataFrame(dict(year=year, model=model, epoch=np.arange(1, len(m.loss_curve_) + 1), val=m.validation_scores_, train_loss=m.loss_curve_, best_epoch=int(np.argmax(m.validation_scores_)) + 1))
        pack = {"name": model, "kind": "mlp", "hp": dict(run), "model": m, "scaler": sc}; p = learners.predict(pack, F[ev_rows]); cur["saved_test"] = float((p == y[ev_rows]).mean())
    else:
        log = []; pack = learners.fit_net(model, run, F[fit_rows], y[fit_rows], seed=PSEED, threads=1, Xte=F[ev_rows], yte=y[ev_rows], log=log); p = learners.predict(pack, F[ev_rows])
        assert abs(log[-1]["test_acc"] - (p == y[ev_rows]).mean()) < 1e-12
        cur = pd.DataFrame(log); cur.insert(0, "year", year)
    full = np.zeros(int(te.sum()), np.uint8); full[(ok[te] if patch else np.ones(int(te.sum()), bool))] = p
    np.savez_compressed(f"{O}/{model}_{year}.npz", idx=np.flatnonzero(te), ok=ok[te], y=y[te].astype(np.uint8), block=df.block_id.to_numpy()[te], pred=full)
    cur.to_csv(f"{O}/curves_{model}.csv", mode="a", header=not os.path.exists(f"{O}/curves_{model}.csv"), index=False)
    if part == "tuned": save_model(pack, year)
    print(year, model, part, "trial", trial, "test OA %.4f" % (p == y[ev_rows]).mean(), "| common %.4f" % (full[ok[te]] == y[common]).mean(), "| %.0f s" % (time.time() - t0), flush=True)
if timing: pd.DataFrame(timing).to_csv(f"{O}/timing_{model}.csv", index=False)
import sklearn, platform
cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][:1]
json.dump(dict(model=model, part=part, trial=trial, hp=hp, quick=QUICK, cpu=cpu, host=platform.node(), versions=dict(sklearn=sklearn.__version__, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__)),
          open(done, "w"), indent=1)
print("DONE", model, part)
