"""Select one configuration per classifier: highest mean cross-validated OA over the 15 year-folds (ties: lower trial number).
Writes results/selected.json and results/tuning_all_trials.csv. Classifiers are also ranked by their cross-validated OA on
the pixels common to all classifiers (those with a full 9 x 9 window)."""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import configs
R = os.path.join(HERE, "results_quick" if os.environ.get("PEI5_QUICK") == "1" else "results")
rows = []
for f in sorted(glob.glob(R + "/tune/*.json")):
    d = json.load(open(f)); oa = [r["oa"] for r in d["folds"]]
    rows.append(dict(model=d["model"], trial=d["trial"], cv_oa=float(np.mean(oa)), cv_oa_sd=float(np.std(oa)), cv_oa_min=float(np.min(oa)), cv_oa_common=d["mean_oa_common"],
                     cv_f1=float(np.mean([r["f1"] for r in d["folds"]])), n_folds=len(oa), secs=d["secs"], hp=json.dumps(d["hp"], sort_keys=True)))
T = pd.DataFrame(rows).sort_values(["model", "trial"]); T.to_csv(R + "/tuning_all_trials.csv", index=False)
sel = {}; missing = {}
for m in configs.MODELS:
    t = T[T.model == m]; missing[m] = sorted(set(range(configs.N_TRIALS)) - set(t.trial))
    if not len(t): continue
    b = t.sort_values(["cv_oa", "trial"], ascending=[False, True]).iloc[0]; d0 = t[t.trial == 0]
    sel[m] = dict(trial=int(b.trial), hp=json.loads(b.hp), cv_oa=float(b.cv_oa), cv_oa_sd=float(b.cv_oa_sd), cv_oa_common=float(b.cv_oa_common), cv_f1=float(b.cv_f1),
                  default_cv_oa=float(d0.cv_oa.iloc[0]) if len(d0) else None, trials_done=int(len(t)), trials_missing=missing[m])
rank = sorted(sel, key=lambda m: -sel[m]["cv_oa_common"])
json.dump(dict(selected=sel, rank_by_cv_oa_common=rank, n_trials=configs.N_TRIALS, k_folds=configs.K_FOLDS), open(R + "/selected.json", "w"), indent=1)
for m in rank:
    s = sel[m]; print("%-15s trial %2d  CV OA %.4f (default %s)  common %.4f  trials %d/%d  %s" % (m, s["trial"], s["cv_oa"], "%.4f" % s["default_cv_oa"] if s["default_cv_oa"] is not None else "n/a",
                                                                                              s["cv_oa_common"], s["trials_done"], configs.N_TRIALS, json.dumps(s["hp"])))
if any(missing.values()): print("MISSING TRIALS:", {m: v for m, v in missing.items() if v})
