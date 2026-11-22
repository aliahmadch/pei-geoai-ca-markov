"""Conditional 2029 scenario with the ANN-CA-Markov model of ca_model.py (run ca_model.py hindcast first).

  * Temporally consistent land: pixels with the same class in 2021 and 2022 and in 2024 and 2025.
  * 2025 baseline: the 2025 class where the 2024 and 2025 maps agree, otherwise the pixel's most frequent class in
    2021-2025 (ties: the class mapped most recently).
  * Quantity: Markov transition probabilities from the 2021 -> 2025 transitions of temporally consistent pixels, one
    four-year step from the baseline.
  * Transition potential: the MLP of the hindcast (2021 -> 2023, seed 42), evaluated at the baseline.
  * Allocation: rank-based, as in the hindcast.
Also: accuracy of the 2025 map and of the baseline at the independent 2025 test pixels (all, and without test pixels at
the location of a training pixel of 2021-2024), and the spatial pattern of the projected and the observed change in 2 km
cells (share of change in the 5% of cells with the most change; correlation of the changed area across cells).

    python ca_scenario_2029.py        (in the working folder, or with PEI_WORK set to it)
Outputs in res/: baseline_2025_consistent.npy, sim_scenario_2029.npy (land vectors), calib_counts_scenario_2029.csv,
calib_P_scenario_2029.csv, areas_scenario_2029.csv, scenario_2029_metrics.json."""
import json, time, numpy as np, pandas as pd
import ca_model as C

K, NAMES, YEARS = C.K, C.NAMES, C.YEARS; t = time.time()
S = C.load_state('maj'); m = {y: np.asarray(S[k]) for k, y in enumerate(YEARS)}; del S
cons = (m[2021] == m[2022]) & (m[2024] == m[2025]); late = m[2024] == m[2025]
# ---- temporally consistent 2025 baseline
cnt = np.zeros((len(m[2025]), K), np.uint8)
for y in YEARS: cnt[np.arange(len(cnt)), m[y] - 1] += 1
mx = cnt.max(1); mode = np.zeros(len(mx), np.uint8)
for y in YEARS:                                   # the last year whose class is among the most frequent wins
    sel = cnt[np.arange(len(cnt)), m[y] - 1] == mx; mode[sel] = m[y][sel]
del cnt
base = np.where(late, m[2025], mode).astype(np.uint8); np.save(C.W + 'res/baseline_2025_consistent.npy', base)
# ---- Markov quantities from the transitions of temporally consistent pixels
ct, P = C.markov(m[2021][cons], m[2025][cons])
# ---- transition potentials of the hindcast MLP (fresh generator, seed 42, first draw is the fit: as in ca_model.py hindcast)
C.rng = np.random.default_rng(42)
drv = C.drivers(); nb0 = C.neighbourhood(m[2021])
mlp, sc, val = C.fit_potential(m[2021], m[2023], nb0, drv); del nb0
try:
    hc = pd.read_csv(C.W + 'res/ca_metrics_hindcast.csv').iloc[0]
    assert abs(val['val_OA'] - hc.mlp_val_OA) < 1e-12 and abs(val['persistence_OA'] - hc.mlp_persistence_OA) < 1e-12, 'not the hindcast MLP'
except FileNotFoundError:
    print('res/ca_metrics_hindcast.csv not found: the identity with the hindcast MLP was not checked')
nbs = C.neighbourhood(base); pot = C.potentials(mlp, sc, base, nbs, drv); del nbs
C.rng = np.random.default_rng(42)                                    # tie-breaking noise of the allocation
sim, quotas = C.allocate(base, pot, P); del pot
np.save(C.W + 'res/sim_scenario_2029.npy', sim)
a_map = np.bincount(m[2025], minlength=K + 1)[1:] * 0.01; a0 = np.bincount(base, minlength=K + 1)[1:] * 0.01; a1 = np.bincount(sim, minlength=K + 1)[1:] * 0.01
areas = pd.DataFrame({'class': NAMES, '2025_map_ha': a_map, '2025_baseline_ha': a0, '2029_ha': a1, 'change_ha': a1 - a0})
areas.to_csv(C.W + 'res/areas_scenario_2029.csv', index=False)
pd.DataFrame(ct, index=NAMES, columns=NAMES).to_csv(C.W + 'res/calib_counts_scenario_2029.csv')
pd.DataFrame(P, index=NAMES, columns=NAMES).to_csv(C.W + 'res/calib_P_scenario_2029.csv')
# ---- 2025 map and baseline at the independent 2025 test pixels
idx = -np.ones(C.LAND.shape, np.int64); idx[C.LAND] = np.arange(C.LAND.sum())
te = pd.read_csv(C.W + 'test_2025.csv'); ii = idx[((779250 - te.y) // 10).astype(int).values, ((te.x - 291810) // 10).astype(int).values]
assert (ii >= 0).all(); yref = te['class'].values
train_xy = set()                                  # locations of the training pixels of 2021-2024
for yr in range(2021, 2025):
    tr = pd.read_csv(C.W + f'train_{yr}.csv'); train_xy |= set(zip(tr.x.round(1), tr.y.round(1)))
ov = np.array([(round(a, 1), round(b, 1)) in train_xy for a, b in zip(te.x, te.y)])
acc = lambda mp, s: float((mp[ii[s]] == yref[s]).mean())
allt = np.ones(len(te), bool)
# ---- spatial pattern of the projected change and of the change observed on consistent land, 2021-2025 (2 km cells)
rr, cc = np.nonzero(C.LAND); nx2 = (C.WD + 199) // 200; cell2 = (rr // 200) * nx2 + cc // 200; del rr, cc
nc = cell2.max() + 1; landc = np.bincount(cell2, minlength=nc) > 0; k5 = int(np.ceil(0.05 * landc.sum()))


def conc(mask):
    c = np.bincount(cell2[mask], minlength=nc)[landc].astype(float)
    return float(np.sort(c)[::-1][:k5].sum() / max(c.sum(), 1)), c


s0, s1 = m[2021], m[2025]
pat = dict(n_cells_2km=int(landc.sum()), top_cells=k5)
for lab, a, b in (('all', None, None), ('agri_to_forest', 2, 1), ('forest_to_agri', 1, 2), ('otherland_to_agri', 5, 2), ('agri_to_otherland', 2, 5), ('to_settlement', None, 4)):
    if lab == 'all': sm, ob = sim != base, cons & (s0 != s1)
    elif a is None: sm, ob = (sim == b) & (base != b), cons & (s1 == b) & (s0 != b)
    else: sm, ob = (base == a) & (sim == b), cons & (s0 == a) & (s1 == b)
    (cs, csm), (co, com) = conc(sm), conc(ob)
    pat[lab] = dict(projected_ha=float(sm.sum() / 100), observed_ha=float(ob.sum() / 100), top5pct_cells_share_projected=cs,
                    top5pct_cells_share_observed=co, r_cells=float(np.corrcoef(csm, com)[0, 1]))
out = dict(consistent_land_share=float(cons.mean()), late_consistent_share=float(late.mean()),
           changed_ha=float((sim != base).sum() * 0.01), changed_share=float((sim != base).mean()),
           transition_potential_validation={k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in val.items()},
           test_pixels_2025=int(len(te)), test_pixels_2025_at_training_location=int(ov.sum()),
           accuracy_2025_map=acc(m[2025], allt), accuracy_2025_baseline=acc(base, allt),
           accuracy_2025_map_without_training_locations=acc(m[2025], ~ov), accuracy_2025_baseline_without_training_locations=acc(base, ~ov),
           pattern_2km_cells=pat)
json.dump(out, open(C.W + 'res/scenario_2029_metrics.json', 'w'), indent=1)
print(areas.round(1).to_string(index=False)); print(json.dumps(out, indent=1, default=float)); print('%.0f s' % (time.time() - t))
