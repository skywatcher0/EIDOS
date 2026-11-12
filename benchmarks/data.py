"""Discovery-only preprocessing; exact experimental samples use portable CSV."""
import zlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from eidos.protocol.strict import five_role_split_by_env
from eidos._utils.data import quantize_y
from .common import ROOT

def task_seed(name, partition=0):
    return 42 + int(zlib.crc32(name.encode('utf-8')) & 0x7FFFFFFF) + 7919 * partition

def env_from_first_feature(values, q=5):
    series = pd.Series(np.asarray(values, dtype=float))
    env = pd.qcut(series.rank(method='average'), q=q, labels=False, duplicates='drop')
    arr = np.asarray(env, dtype=int)
    if np.unique(arr).size < 3:
        arr = np.minimum(np.floor(series.rank(method='average', pct=True).to_numpy() * q).astype(int), q-1)
    return arr.astype(int)

def prepare(name, track='pmlb', partition=0):
    if track == 'pmlb':
        path = ROOT / 'data/pmlb/samples' / (name + '.csv.gz')
    else:
        path = ROOT / 'data/openml/raw' / (name + '.csv.gz')
    table = pd.read_csv(path, float_precision='round_trip')
    row_ids = np.arange(len(table))
    seed = task_seed(name, partition)
    if track == 'openml' and len(table) > 300:
        take = np.sort(np.random.default_rng(seed).choice(len(table), size=300, replace=False))
        row_ids = row_ids[take]
        table = table.iloc[take].reset_index(drop=True)
    y = table.pop('target').astype(float)
    x = table.astype(float)
    env = env_from_first_feature(x.iloc[:, 0].to_numpy(float))
    roles = five_role_split_by_env(env, seed=seed)
    disc = roles == 'disc'
    if track == 'openml':
        x = x[[c for c in x.columns if np.std(x.loc[disc,c].to_numpy(float)) > 1e-12]]
    if x.shape[1] > 8:
        corr = []
        for col in x.columns:
            xv = x.loc[disc, col].to_numpy(float)
            val = 0.0 if np.std(xv) < 1e-12 else abs(float(np.corrcoef(xv, y.loc[disc].to_numpy(float))[0,1]))
            corr.append((val if np.isfinite(val) else 0.0, str(col)))
        x = x[[c for _, c in sorted(corr, reverse=True)[:8]]]
    scaler = StandardScaler().fit(x.loc[disc].to_numpy(float))
    out = pd.DataFrame(scaler.transform(x.to_numpy(float)), columns=[f'x{i}' for i in range(x.shape[1])])
    yc = float(np.mean(y.to_numpy(float)[disc])) if track == 'pmlb' else float(y[disc].mean())
    ys = float(np.std(y.to_numpy(float)[disc])) if track == 'pmlb' else float(y[disc].std(ddof=0))
    if ys <= 1e-12:
        raise ValueError('Near-constant discovery target')
    out['y_tilde'] = quantize_y((y.to_numpy(float) - yc) / ys, delta_y=0.001)
    out['env_id'] = env
    out['split'] = roles
    out['sample_id'] = np.arange(len(out))
    meta = {'dataset':name, 'partition':partition, 'seed':seed, 'source_row_ids':row_ids.tolist(),
            'row_id_scope':'bundled experimental sample' if track=='pmlb' else 'cleaned full OpenML table',
            'selected_original_features':list(x.columns),
            'feature_affine':{f'x{i}':[float(scaler.mean_[i]),float(scaler.scale_[i])] for i in range(x.shape[1])},
            'target_affine':[yc,ys]}
    return out, meta
