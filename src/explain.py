"""SHAP otomatis untuk tiga model wakil: DPformer + DL baseline terbaik + ML terbaik.

Wakil dipilih OTOMATIS dari peringkat rata-rata di dalam kelompok (analysis.pick_representatives)
setelah evaluasi final. Pemilihan ini hanya menentukan model mana yang DIJELASKAN; tidak ada
tuning ulang. Atribusi bersifat post-hoc/deskriptif:
  * background = baris TRAIN saja, sampel yang dijelaskan = baris TEST (tidak ikut melatih apa pun);
  * semua nilai dikonversi ke satuan yang sama: kontribusi terhadap RETURN h-hari (bukan harga),
    sehingga model ML (keluaran return) dan DL (keluaran z-score return) sebanding;
  * kontribusi diagregasi per VARIABEL (Antam + 7 makro): fitur turunan (lag/rolling) dan langkah
    waktu jendela DL dijumlahkan (sifat aditif Shapley terjaga).

Penjelas:
  ML : TreeExplainer (LightGBM, XGBoost, RF, ExtraTrees, CatBoost, GBM); LinearExplainer (ElasticNet);
       PermutationExplainer sebagai cadangan (AdaBoost, HistGB, SVR, atau bila TreeExplainer gagal).
  DL : shap.GradientExplainer (expected gradients); bila gagal pada arsitektur/versi tertentu,
       implementasi expected-gradients sendiri (identik secara prinsip) dipakai.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .config import LGB_PARAMS
from .pipeline import (EXO_COLS, EXO_LEVEL_COLS, build_lgb_features, build_dl_explain_arrays,
                       multihorizon_split, train_dl_h, train_lightgbm_h, train_ml_h)

GROUP_ORDER = ['emas'] + EXO_LEVEL_COLS
GROUP_LABEL = {'emas': 'Antam (return, lag & rolling)', 'dxy': 'DXY', 'sp500': 'S&P 500',
               'usd_idr': 'USD/IDR', 'ihsg': 'IHSG', 'us10yr': 'US10Y', 'ffr': 'FFR', 'bi_rate': 'BI-Rate'}
TREE_MODELS = ('lightgbm', 'xgboost', 'rf', 'extra_trees', 'catboost', 'gbm')


def group_of(feature: str) -> str:
    """Nama fitur -> variabel induk ('emas' atau salah satu dari 7 indikator makro)."""
    for c in EXO_LEVEL_COLS:
        if feature in (f'{c}_ret', f'{c}_diff'):
            return c
    return 'emas'


def _even_idx(n: int, k: int) -> np.ndarray:
    return np.arange(n) if n <= k else np.linspace(0, n - 1, k).astype(int)


def _pack(values_by_group: np.ndarray, groups: list[str], base: float, pred: np.ndarray, X_by_group: np.ndarray,
          dates, method: str, extra: dict | None = None) -> dict:
    """Paket hasil seragam. values/base/pred dalam satuan RETURN h-hari (pecahan, bukan persen)."""
    comp = base + values_by_group.sum(axis=1)
    denom = float(np.mean(np.abs(pred - base))) or 1e-12
    out = {'values': np.asarray(values_by_group, float), 'groups': list(groups), 'base': float(base),
           'pred': np.asarray(pred, float), 'X': np.asarray(X_by_group, float), 'dates': pd.DatetimeIndex(dates),
           'method': method,
           'completeness_err': float(np.mean(np.abs(comp - pred)) / denom)}
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------
def explain_ml(p: pd.Series, exo: pd.DataFrame, model_name: str, h: int, rows_explain: list[int],
               n_background: int = 50, seed: int = 42, stored_pred: pd.DataFrame | None = None,
               max_evals_perm: int | None = None) -> dict:
    """Latih ulang model ML (deterministik; konfigurasi identik run utama), verifikasi prediksinya sama
    dengan yang tersimpan, lalu hitung SHAP pada baris `rows_explain` (posisi origin TEST)."""
    import shap

    F = build_lgb_features(p, exo)
    feats = list(F.columns)
    dates = p.index
    tr_dates, _, te_pos = multihorizon_split(p, h)
    if model_name == 'lightgbm':
        model, _, rows, _, y_pred = train_lightgbm_h(p, F, h, LGB_PARAMS)
        predict = lambda X: model.predict(np.asarray(X))
    else:
        model, _, rows, _, y_pred, _ = train_ml_h(p, F, h, model_name)
        predict = lambda X: model.predict(np.asarray(X))
    repro = None
    if stored_pred is not None:
        # stored_pred: DataFrame berindeks tanggal target dengan kolom y_pred (analysis.read_pred)
        mine = pd.Series(np.asarray(y_pred).ravel(), index=[dates[i + h] for i in rows])
        j = stored_pred['y_pred'].reindex(mine.index)
        repro = float(np.nanmax(np.abs(j.values - mine.values)) / max(1.0, float(np.nanmax(np.abs(j.values)))))

    tr_ok = [d for d in tr_dates if d in F.index]
    X_tr = F.loc[tr_ok, feats]
    X_bg = X_tr.iloc[_even_idx(len(X_tr), n_background)]
    ex_dates = [dates[i] for i in rows_explain if dates[i] in F.index]
    X_ex = F.loc[ex_dates, feats]

    method = None
    sv = None
    base = None
    if model_name in TREE_MODELS:
        try:
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X_ex.values)
            sv = sv[0] if isinstance(sv, list) else sv
            ev = np.asarray(expl.expected_value).ravel()
            base = float(ev[0])
            method = 'TreeExplainer'
        except Exception as e:                                           # pragma: no cover
            print(f'  [SHAP] TreeExplainer gagal ({type(e).__name__}); beralih ke PermutationExplainer')
            sv = None
    if sv is None and model_name == 'elasticnet':
        expl = shap.LinearExplainer(model, X_bg.values)
        sv = expl.shap_values(X_ex.values)
        base = float(np.mean(predict(X_bg.values)))
        method = 'LinearExplainer'
    if sv is None:
        masker = shap.maskers.Independent(X_bg.values, max_samples=len(X_bg))
        expl = shap.Explainer(predict, masker, algorithm='permutation')
        ex = expl(X_ex.values, max_evals=max_evals_perm or (10 * len(feats) + 1), silent=True)
        sv = np.asarray(ex.values)
        base = float(np.mean(ex.base_values))
        method = 'PermutationExplainer'
    sv = np.asarray(sv, float)
    pred = np.asarray(predict(X_ex.values), float).ravel()

    groups = GROUP_ORDER
    gv = np.zeros((len(X_ex), len(groups)))
    gx = np.zeros_like(gv)
    for k, g in enumerate(groups):
        cols = [i for i, f in enumerate(feats) if group_of(f) == g]
        gv[:, k] = sv[:, cols].sum(axis=1)
        prim = 'lag_0' if g == 'emas' else [f for f in feats if group_of(f) == g][0]
        gx[:, k] = X_ex[prim].values
    out = _pack(gv, groups, base, pred, gx, ex_dates, method,
                {'h': h, 'feature_values': X_ex, 'feature_shap': pd.DataFrame(sv, index=X_ex.index, columns=feats),
                 'repro_maxrel_diff': repro, 'n_background': len(X_bg)})
    return out


# ---------------------------------------------------------------------------
# DL
# ---------------------------------------------------------------------------
def expected_gradients(model, X_bg: np.ndarray, X: np.ndarray, nsamples: int = 200, seed: int = 0) -> np.ndarray:
    """Expected gradients (Erion et al. 2021) = penjelas yang dipakai shap.GradientExplainer:
    phi = E_{x'~bg, a~U(0,1)} [ (x - x') * df(x' + a (x - x'))/dx ]. Return (n, T, C)."""
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    bg = np.asarray(X_bg, np.float32)
    phi = np.zeros(X.shape, np.float64)
    for _ in range(nsamples):
        ref = bg[rng.integers(0, len(bg), size=len(X))]
        a = rng.uniform(size=(len(X), 1, 1)).astype(np.float32)
        pt = tf.convert_to_tensor(ref + a * (X - ref))
        with tf.GradientTape() as tape:
            tape.watch(pt)
            out = model(pt, training=False)
        g = tape.gradient(out, pt).numpy()
        phi += (X - ref) * g
    return phi / nsamples


def _dl_attribution(model, X_bg: np.ndarray, X: np.ndarray, nsamples: int = 200, seed: int = 0):
    """(phi[n,T,C], metode). Utama: shap.GradientExplainer; cadangan: expected_gradients sendiri."""
    try:
        import shap
        np.random.seed(seed)
        expl = shap.GradientExplainer(model, X_bg)
        sv = expl.shap_values(X, nsamples=nsamples)
        sv = sv[0] if isinstance(sv, list) else sv
        sv = np.asarray(sv)
        if sv.ndim == 4 and sv.shape[-1] == 1:
            sv = sv[..., 0]
        if sv.shape != X.shape or not np.isfinite(sv).all():
            raise ValueError(f'keluaran GradientExplainer tidak valid: {sv.shape}')
        return sv.astype(np.float64), 'shap.GradientExplainer'
    except Exception as e:
        print(f'  [SHAP] GradientExplainer gagal ({type(e).__name__}: {str(e)[:80]}); memakai expected_gradients sendiri')
        return expected_gradients(model, X_bg, X, nsamples=nsamples, seed=seed), 'expected_gradients (sendiri)'


def build_dl_model(model_name: str, units: int, n_feat: int, seq_len: int, builders):
    return builders[model_name](seq_length=seq_len, units_1=units, n_feat=n_feat)


def get_dl_model(p: pd.Series, exo: pd.DataFrame | None, model_name: str, h: int, weights_path: str,
                 builders: dict, seq_len: int, units: int, batch: int, lr: float, wd: float,
                 epochs: int, patience: int, seed: int = 42, stored_mae: float | None = None):
    """Muat bobot tersimpan; bila tak ada, latih ulang (seed & konfigurasi run utama) dan simpan.
    Return (model, info). info['sumber'] in {'bobot_tersimpan','dilatih_ulang'}; untuk model yang dilatih
    ulang dicatat MAE-nya dan selisih terhadap MAE run utama (bila diberikan)."""
    n_feat = 1 if exo is None else 1 + len(EXO_COLS)
    if os.path.exists(weights_path):
        m = build_dl_model(model_name, units, n_feat, seq_len, builders)
        m.load_weights(weights_path)
        return m, {'sumber': 'bobot_tersimpan', 'weights': weights_path}
    model, metrics, *_ = train_dl_h(p, exo, h, builders[model_name], seq_len, units, epochs=epochs,
                                    patience=patience, batch_size=batch, learning_rate=lr, weight_decay=wd,
                                    seed=seed, eval_split='test')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_weights(weights_path)
    info = {'sumber': 'dilatih_ulang', 'weights': weights_path, 'MAE_retrain': float(metrics['MAE'])}
    if stored_mae is not None:
        info['MAE_run_utama'] = float(stored_mae)
        info['selisih_rel'] = float(abs(metrics['MAE'] - stored_mae) / stored_mae)
    return model, info


def explain_dl(p: pd.Series, exo: pd.DataFrame, model, h: int, seq_len: int, n_background: int = 50,
               n_explain: int = 100, nsamples: int = 200, seed: int = 42) -> dict:
    """SHAP DL pada jendela input (background TRAIN, sampel TEST). Keluaran model = z-score return;
    dikonversi ke satuan return dengan simpangan baku/rata-rata target TRAIN."""
    tr_dates, _, te_pos = multihorizon_split(p, h)
    yh = (p.shift(-h) / p - 1).loc[tr_dates].values
    mu_t, sd_t = float(np.mean(yh)), float(np.std(yh))            # StandardScaler memakai std populasi (ddof=0)
    X_bg, X_ex, rows_ex = build_dl_explain_arrays(p, exo, h, seq_len, max_background=n_background,
                                                  max_samples=n_explain, seed=seed)
    phi, method = _dl_attribution(model, X_bg, X_ex, nsamples=nsamples, seed=seed)
    f_bg = np.asarray(model.predict(X_bg, verbose=0)).ravel()
    f_ex = np.asarray(model.predict(X_ex, verbose=0)).ravel()
    base_z = float(f_bg.mean())
    chan = phi.sum(axis=1)                                           # (n, C): jumlah atas langkah waktu
    cols = ['emas'] + EXO_LEVEL_COLS
    assert chan.shape[1] == len(cols), (chan.shape, cols)
    values = chan * sd_t                                             # satuan return
    pred = f_ex * sd_t + mu_t
    base = base_z * sd_t + mu_t
    xval = X_ex.mean(axis=1)                                         # nilai kanal (z-score) rata-rata jendela
    return _pack(values, cols, base, pred, xval, [p.index[i] for i in rows_ex], method,
                 {'h': h, 'target_std_train': sd_t, 'target_mean_train': mu_t, 'n_background': len(X_bg),
                  'phi_time': phi * sd_t})


# ---------------------------------------------------------------------------
# ringkasan & penyimpanan
# ---------------------------------------------------------------------------
def importance_table(res: dict) -> pd.DataFrame:
    """Rata-rata |SHAP| per variabel (poin persen return h-hari) + porsi (%)."""
    v = np.abs(res['values']).mean(axis=0) * 100
    t = pd.DataFrame({'variabel': res['groups'], 'label': [GROUP_LABEL.get(g, g) for g in res['groups']],
                      'mean_abs_shap_pp': v})
    t['porsi_pct'] = 100 * t.mean_abs_shap_pp / t.mean_abs_shap_pp.sum()
    t['arah_rata2_pp'] = res['values'].mean(axis=0) * 100
    return t.sort_values('mean_abs_shap_pp', ascending=False).reset_index(drop=True)


def pick_waterfall_sample(res: dict) -> int:
    """Sampel yang dijelaskan dengan simpangan prediksi terbesar dari nilai dasar (deterministik)."""
    return int(np.argmax(np.abs(res['pred'] - res['base'])))


def save_result(res: dict, outdir: str, tag: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    t = importance_table(res)
    t.to_csv(os.path.join(outdir, f'shap_{tag}_importance.csv'), index=False)
    np.savez_compressed(os.path.join(outdir, f'shap_{tag}_values.npz'),
                        values=res['values'], X=res['X'], pred=res['pred'], base=res['base'],
                        groups=np.array(res['groups']), dates=res['dates'].values.astype('datetime64[ns]'))
    meta = {'tag': tag, 'method': res['method'], 'completeness_err_rel': res['completeness_err'],
            'n_explained': int(len(res['pred'])), 'n_background': int(res.get('n_background', 0)),
            'base_return_pct': res['base'] * 100,
            'top3': t.head(3)[['variabel', 'porsi_pct']].round(2).to_dict('records')}
    for k in ('repro_maxrel_diff', 'target_std_train'):
        if k in res and res[k] is not None:
            meta[k] = res[k]
    with open(os.path.join(outdir, f'shap_{tag}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return meta
