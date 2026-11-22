"""ANN-CA-Markov land-change model for the five-class annual maps (forest, agricultural land, wetland, settlement,
otherland) and the 2025 hindcast.

  * State maps: annual XGBoost maps after the 3 x 3 majority filter, land pixels only (maj_land.npy).
  * Transition potential: multilayer perceptron (one hidden layer of 32 units, early stopping) predicting the class at t1
    from the one-hot class at t0, the class composition of the 8 neighbours at t0 and five standardised drivers
    (elevation, slope, log population, log distance to roads, log distance to waterways); 300,000 training pixels drawn
    at random from 80% of the 3 km blocks, validation on 200,000 pixels of the remaining 20% of blocks.
  * Quantity: Markov chain, n_ij = N_i(t) * P_ij, with P estimated over the calibration interval (same length as the step).
  * Allocation: for each source class, transitions in descending order of quantity; the pixels with the highest potential
    for the target class among those not yet converted are converted.

Run in the working folder prepared by prepare_inputs.py (or set PEI_WORK to it):
    python ca_model.py hindcast
Outputs in res/: ca_metrics_hindcast.csv, Markov counts and probabilities, class areas, the simulated 2025 map and the
random-allocation map (land vectors, .npy) and the fitted transition-potential model (transition_potential_mlp.joblib/.npz)."""
import os, sys, json, time
import numpy as np, pandas as pd, cv2, joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score

os.chdir(os.environ.get('PEI_WORK', os.getcwd()))
W = './'
LAND = np.load(W + 'land_mask.npy')
H, WD = LAND.shape
NAMES = ['Forest', 'Agricultural land', 'Wetland', 'Settlement', 'Otherland']
K = len(NAMES)
YEARS = [2021, 2022, 2023, 2024, 2025]
FEATURES = ['class_t0_' + n for n in NAMES] + ['neighbours_' + n for n in NAMES] + ['elevation', 'slope', 'log_population', 'log_dist_roads', 'log_dist_waterways']
rng = np.random.default_rng(42)
BLOCK = np.load(W + 'block_land.npy')            # 3 km block of each land pixel (int32)


def load_state(kind='maj'):
    return np.load(W + {'maj': 'maj_land.npy', 'raw': 'raw_land.npy'}[kind])


def drivers():
    """Standardised drivers of the land pixels (n x 5, float16)."""
    return np.load(W + 'drivers_std_f16.npy', mmap_mode='r')


def neighbourhood(state_land):
    """Counts (0-8) of each class among the 8 neighbours, for land pixels."""
    full = np.zeros((H, WD), np.uint8); full[LAND] = state_land
    out = np.zeros((len(state_land), K), np.uint8)
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    for c in range(1, K + 1):
        m = (full == c).astype(np.uint8)
        s = cv2.filter2D(m, -1, k.astype(np.float32), borderType=cv2.BORDER_CONSTANT)  # uint8 output, exact for counts <= 8
        out[:, c - 1] = s[LAND]
        del m, s
    return out


def features(state, nb, drv, idx):
    oh = np.eye(K, dtype=np.float32)[state[idx] - 1]
    return np.hstack([oh, nb[idx].astype(np.float32) / 8.0, np.asarray(drv[idx], dtype=np.float32)])


def markov(s0, s1):
    ct = np.bincount((s0.astype(np.int64) - 1) * K + (s1 - 1), minlength=K * K).reshape(K, K)
    return ct, ct / ct.sum(1, keepdims=True)


def fit_potential(s0, s1, nb, drv, n_train=300_000, val_n=200_000):
    blocks = np.arange(BLOCK.max() + 1); rng.shuffle(blocks)
    test_blocks = set(blocks[:int(round(0.2 * len(blocks)))].tolist())
    tb = np.zeros(BLOCK.max() + 1, bool); tb[list(test_blocks)] = True; is_val = tb[BLOCK]
    cand = np.flatnonzero(~is_val)
    tr_idx = rng.choice(cand, n_train, replace=False)
    va_idx = rng.choice(np.flatnonzero(is_val), val_n, replace=False)
    Xtr = features(s0, nb, drv, tr_idx); ytr = s1[tr_idx]
    sc = StandardScaler().fit(Xtr)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, early_stopping=True, n_iter_no_change=15,
                        learning_rate_init=1e-3, alpha=1e-4, random_state=42)
    mlp.fit(sc.transform(Xtr), ytr)
    Xva = features(s0, nb, drv, va_idx); yva = s1[va_idx]
    pr = mlp.predict(sc.transform(Xva))
    pers = s0[va_idx]
    ch = yva != pers
    val = dict(n_train=int(len(tr_idx)), n_val=int(len(va_idx)), val_OA=accuracy_score(yva, pr), val_kappa=cohen_kappa_score(yva, pr),
               persistence_OA=accuracy_score(yva, pers), persistence_kappa=cohen_kappa_score(yva, pers),
               val_change_share=float(ch.mean()), predicted_change_share=float((pr != pers).mean()),
               change_recall=float(((pr != pers) & ch).sum() / max(ch.sum(), 1)),
               change_precision=float(((pr != pers) & ch).sum() / max((pr != pers).sum(), 1)),
               changed_pixel_dest_acc=float((pr[ch] == yva[ch]).mean()) if ch.any() else np.nan)
    return mlp, sc, val


def save_potential_model(mlp, sc, val, path):
    """The fitted transition-potential model: scikit-learn objects (.joblib) and plain arrays (.npz)."""
    joblib.dump(dict(mlp=mlp, scaler=sc, features=FEATURES, classes=[int(c) for c in mlp.classes_], class_names=NAMES, validation=val), path + '.joblib')
    np.savez(path + '.npz', mean=sc.mean_, scale=sc.scale_, classes=mlp.classes_, W0=mlp.coefs_[0], b0=mlp.intercepts_[0],
             W1=mlp.coefs_[1], b1=mlp.intercepts_[1], features=np.array(FEATURES))


def potentials(mlp, sc, state, nb, drv, chunk=500_000):
    out = np.zeros((len(state), K), np.uint8)
    for a in range(0, len(state), chunk):
        idx = np.arange(a, min(a + chunk, len(state)))
        p = mlp.predict_proba(sc.transform(features(state, nb, drv, idx)))
        full = np.zeros((len(idx), K), np.float32); full[:, mlp.classes_ - 1] = p
        out[idx] = np.rint(full * 255).astype(np.uint8)
    return out


def allocate(state, pot, P):
    new = state.copy(); N = np.bincount(state, minlength=K + 1)[1:]
    quotas = np.rint(N[:, None] * P).astype(np.int64)
    taken = np.zeros(len(state), bool)
    for i in range(K):
        src = np.flatnonzero(state == i + 1)
        order = [j for j in np.argsort(-quotas[i]) if j != i and quotas[i, j] > 0]
        for j in order:
            avail = src[~taken[src]]
            n = min(quotas[i, j], len(avail))
            if n == 0: continue
            score = pot[avail, j].astype(np.float32) + rng.random(len(avail), dtype=np.float32) * 1e-3
            pick = avail[np.argpartition(-score, n - 1)[:n]]
            new[pick] = j + 1; taken[pick] = True
    return new, quotas


def compare(ref0, ref1, sim1):
    """Three-map comparison (Pontius) and agreement statistics."""
    obs = ref1 != ref0; simc = sim1 != ref0
    hits = (obs & simc & (sim1 == ref1)).sum(); wrong = (obs & simc & (sim1 != ref1)).sum()
    misses = (obs & ~simc).sum(); fa = (~obs & simc).sum()
    fom = hits / max(hits + wrong + misses + fa, 1)
    n = len(ref1)
    a_ref = np.bincount(ref1, minlength=K + 1)[1:] / n; a_sim = np.bincount(sim1, minlength=K + 1)[1:] / n
    qd = 0.5 * np.abs(a_ref - a_sim).sum()
    td = (sim1 != ref1).mean()
    return dict(OA=1 - td, kappa=cohen_kappa_score(ref1[::50], sim1[::50]), FoM=fom, hits=int(hits), wrong_hits=int(wrong),
                misses=int(misses), false_alarms=int(fa), obs_change_share=float(obs.mean()), sim_change_share=float(simc.mean()),
                quantity_disagreement=qd, allocation_disagreement=td - qd)


def ref_points(year, sim_map):
    """Accuracy of a map at the independent spatially blocked test pixels of `year`."""
    idx = -np.ones(LAND.shape, np.int64); idx[LAND] = np.arange(LAND.sum())
    t = pd.read_csv(W + f'test_{year}.csv')
    ii = idx[((779250 - t.y) // 10).astype(int).values, ((t.x - 291810) // 10).astype(int).values]
    ok = ii >= 0
    return accuracy_score(t['class'].values[ok], sim_map[ii[ok]])


def run(kind, t0, t1, t_start, horizon_label, ref_year=None, tag=''):
    os.makedirs(W + 'res', exist_ok=True)
    S = load_state(kind); yi = {y: k for k, y in enumerate(YEARS)}
    s0, s1, sst = S[yi[t0]], S[yi[t1]], S[yi[t_start]]
    drv = drivers()
    t = time.time(); nb0 = neighbourhood(s0)
    mlp, sc, val = fit_potential(s0, s1, nb0, drv); del nb0
    save_potential_model(mlp, sc, val, W + 'res/transition_potential_mlp')
    print('transition potential: validation OA %.4f, persistence %.4f (%.0f s)' % (val['val_OA'], val['persistence_OA'], time.time() - t), flush=True)
    ct, P = markov(s0, s1)
    nbs = neighbourhood(sst); pot = potentials(mlp, sc, sst, nbs, drv); del nbs
    sim, quotas = allocate(sst, pot, P)
    res = dict(tag=tag, maps=kind, calib=f'{t0}-{t1}', start=t_start, horizon=horizon_label, **{f'mlp_{k}': v for k, v in val.items()})
    if ref_year is not None:
        ref = S[yi[ref_year]]
        res.update({f'sim_{k}': v for k, v in compare(sst, ref, sim).items()})
        res.update({f'persist_{k}': v for k, v in compare(sst, ref, sst.copy()).items()})
        # null model: the same Markov quantities, allocated at random within each source class
        rnd, _ = allocate(sst, rng.integers(0, 256, size=(len(sst), K), dtype=np.uint8), P)
        np.save(W + f'res/rand_{tag}.npy', rnd)
        res.update({f'randalloc_{k}': v for k, v in compare(sst, ref, rnd).items()})
        res['refpts_acc_sim'] = ref_points(ref_year, sim); res['refpts_acc_persist'] = ref_points(ref_year, sst)
        res['refpts_acc_mapped'] = ref_points(ref_year, ref); res['refpts_acc_randalloc'] = ref_points(ref_year, rnd)
    a0 = np.bincount(sst, minlength=K + 1)[1:] * 0.01; a1 = np.bincount(sim, minlength=K + 1)[1:] * 0.01
    areas = pd.DataFrame({'class': NAMES, f'{t_start}_ha': a0, f'{horizon_label}_ha': a1, 'change_ha': a1 - a0})
    np.save(W + f'res/sim_{tag}.npy', sim)
    pd.DataFrame(ct, index=NAMES, columns=NAMES).to_csv(W + f'res/calib_counts_{tag}.csv')
    pd.DataFrame(P, index=NAMES, columns=NAMES).to_csv(W + f'res/calib_P_{tag}.csv')
    areas.to_csv(W + f'res/areas_{tag}.csv', index=False)
    res['secs'] = time.time() - t
    return res, areas


JOBS = {'hindcast': dict(kind='maj', t0=2021, t1=2023, t_start=2023, horizon_label='2025sim', ref_year=2025, tag='hindcast_2021_2023_to_2025')}

if __name__ == '__main__':
    for name in sys.argv[1:]:
        r, a = run(**JOBS[name])
        print(json.dumps(r, indent=1, default=float)); print(a.round(1).to_string(), flush=True)
        pd.DataFrame([r]).to_csv(W + f'res/ca_metrics_{name}.csv', index=False)
