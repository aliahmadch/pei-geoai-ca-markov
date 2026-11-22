"""One trial of the hyperparameter search for one classifier.
For every year, the training blocks of the main partition (seed 42) are split into K_FOLDS groups of whole 3 km blocks
(GroupKFold); the classifier is fitted on K-1 groups and scored on the remaining one. The 20% test blocks are dropped
right after loading and take no part in the search.
  python tune.py <model> <trial> [--threads N]        -> results/tune/<model>_<trial>.json
Environment variable PEI5_QUICK=1: smoke test (2021 only, tiny models), written to results_quick/."""
import os, sys, json, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("trial", type=int); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[v] = str(a.threads)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DATA = os.environ.get("PEI_DATA", os.path.join(HERE, "..", "..", "data"))   # data/tables and data/patches of the data set
from common import spatial_mask, feature_matrix
import configs, learners
QUICK = os.environ.get("PEI5_QUICK") == "1"
OUT = os.path.join(HERE, "results_quick" if QUICK else "results", "tune"); os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "%s_%03d.json" % (a.model, a.trial))
if os.path.exists(out): print("exists", out); sys.exit(0)
hp = configs.trials(a.model)[a.trial]; run = dict(hp)
if QUICK:
    for k, v in (("epochs", 1), ("iterations", 20), ("n_estimators", 20)):
        if k in run: run[k] = v
L1 = np.array([0, 1, 2, 2, 3, 4, 5]); patch = a.model in configs.PATCH; rows = []; t00 = time.time()
for year in ([2021] if QUICK else [2021, 2022, 2023, 2024, 2025]):
    df = pd.read_csv(f"{DATA}/tables/train_{year}.csv"); X, y6 = feature_matrix(df); tr = spatial_mask(df, seed=42)
    ok = np.load(f"{DATA}/patches/patches_{year}.npz")["ok"]
    keep = tr & ok if patch else tr                        # training blocks only; the test blocks are not used below
    y = L1[y6][keep]; groups = df.block_id.to_numpy()[keep]; okk = ok[keep]
    F = np.load(f"{DATA}/patches/patches_{year}.npz")["P"][keep] if patch else X[keep]
    del df, X, y6
    for k, (i_fit, i_val) in enumerate(GroupKFold(n_splits=configs.K_FOLDS).split(F, y, groups)):
        t0 = time.time(); pack = learners.fit(a.model, run, F[i_fit], y[i_fit], threads=a.threads); p = learners.predict(pack, F[i_val], threads=a.threads)
        yv = y[i_val]; c = okk[i_val]
        rows.append(dict(year=year, fold=k, n_fit=int(len(i_fit)), n_val=int(len(i_val)), blocks_val=int(len(set(groups[i_val]))), oa=float((p == yv).mean()),
                         oa_common=float((p[c] == yv[c]).mean()), n_common=int(c.sum()), f1=float(f1_score(yv, p, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0)),
                         bacc=float(balanced_accuracy_score(yv, p)), secs=round(time.time() - t0, 2)))
        print(a.model, a.trial, year, k, "OA %.4f" % rows[-1]["oa"], "%.0f s" % rows[-1]["secs"], flush=True)
import sklearn, torch
res = dict(model=a.model, trial=a.trial, hp=hp, quick=QUICK, threads=a.threads, folds=rows, mean_oa=float(np.mean([r["oa"] for r in rows])),
           mean_oa_common=float(np.mean([r["oa_common"] for r in rows])), secs=round(time.time() - t00, 1), versions=dict(sklearn=sklearn.__version__, torch=torch.__version__, numpy=np.__version__))
json.dump(res, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
print("DONE", out, "mean CV OA %.4f" % res["mean_oa"], "%.0f s" % res["secs"])
