"""Load a saved classifier and classify the blocked test pixels of its year.

    python predict_example.py --root <data set folder> --year 2025 --classifier xgboost

Classifiers: catboost, lightgbm, xgboost, deepmlp, ft_transformer (13 features per pixel) and vchnet, swingnnet, fmtunet,
timesenclip (13 x 9 x 9 windows; only pixels with has_9x9_window = 1). Prints the overall accuracy against class5."""
import argparse, os, sys
import numpy as np, pandas as pd

BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI', 'BSI', 'elev', 'slope', 'gsw_occ']
PATCH = ('vchnet', 'swingnnet', 'fmtunet', 'timesenclip')


def predict(base, clf, X):
    """base: path without extension, e.g. models/2025_xgboost; X: n x 13 features (or n x 13 x 9 x 9 windows). Returns classes 1-5."""
    if clf == 'catboost':
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(base + '.cbm'); return np.asarray(m.predict(X)).reshape(-1).astype(int) + 1
    if clf == 'lightgbm':
        import lightgbm
        return lightgbm.Booster(model_file=base + '.txt').predict(X).argmax(1) + 1
    if clf == 'xgboost':
        import xgboost
        b = xgboost.Booster(); b.load_model(base + '.json'); return b.inplace_predict(X).argmax(1) + 1
    if clf == 'deepmlp':                  # plain arrays (.npz); the same model as a scikit-learn object is in the .joblib file
        z = np.load(base + '.npz'); h = (X - z['mean']) / z['scale']; n = int(z['n_layers'])
        for i in range(n):
            h = h @ z['W%d' % i] + z['b%d' % i]
            if i < n - 1: h = np.maximum(h, 0)
        return z['classes'][h.argmax(1)] + 1
    import torch                          # networks: the architecture is defined in nets.py (this folder)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nets
    pk = torch.load(base + '.pt', map_location='cpu', weights_only=False)   # trusted file of this data set
    net = nets.make_net(pk['name'], pk['hp']); net.load_state_dict(pk['state']); net.eval()
    Xs = torch.from_numpy(((X - pk['mean']) / pk['std']).astype(np.float32))
    with torch.no_grad(): return np.concatenate([net(Xs[i:i + 512]).argmax(1).numpy() for i in range(0, len(Xs), 512)]) + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--year', type=int, required=True); ap.add_argument('--classifier', required=True)
    a = ap.parse_args()
    t = pd.read_csv(os.path.join(a.root, 'data', 'tables', f'train_{a.year}.csv'))
    test = (t.partition_seed42 == 'test').to_numpy()
    if a.classifier in PATCH:
        test = test & (t.has_9x9_window.to_numpy() == 1)
        X = np.load(os.path.join(a.root, 'data', 'patches', f'patches_{a.year}.npz'))['P'][test]
    else:
        X = np.nan_to_num(t.loc[test, BANDS].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p = predict(os.path.join(a.root, 'models', f'{a.year}_{a.classifier}'), a.classifier, X)
    print(a.year, a.classifier, 'test pixels', int(test.sum()), 'OA %.4f' % (p == t.class5.to_numpy()[test]).mean())


if __name__ == '__main__':
    main()
