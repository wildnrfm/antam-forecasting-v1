"""Pelari satu-run bersama untuk notebook 05, 06, 07, 13, 14, 15, 16.

Satu definisi cara melatih/menyimpan sebuah run (ML atau DL) supaya eksperimen utama dan
eksperimen tambahan (ablation, skenario B, walk-forward, seed) tidak bisa berbeda diam-diam.

Konvensi berkas (per run, di `runs_dir`):
    <label>_<model>_h<h>.json          metrik + metadata (protocol, rev, data_hash, ...)
    <label>_<model>_h<h>_pred.csv      tanggal (target), y_true, y_pred   (harga)
    <label>_<model>_h<h>_history.json  kurva loss DL
    <label>_<model>_h<h>_fimp.csv      feature importance ML (bila ada)
`label` = 'univariat' / 'multivariat' pada eksperimen utama; nama skenario/kelompok pada eksperimen
tambahan (mis. 'B_global', 'multivariat_default').
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from .config import (BASE_LR, BASE_WD, DL_BATCH, DL_UNITS, EPOCHS, LGB_PARAMS, PATIENCE, PROTOCOL, SEQ_LEN)
from .pipeline import (EXO_COLS, SEED, build_lgb_features, epoch_info, free_memory, run_is_current,
                       train_dl_h, train_lightgbm_h, train_ml_h)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data(root: str = '..'):
    """(df, p, exo): data_gabungan.csv, harga Antam, dan 7 fitur makro stasioner (sudah lag-1)."""
    df = pd.read_csv(os.path.join(root, 'data', 'processed', 'data_gabungan.csv'), index_col=0, parse_dates=True)
    return df, df['harga_jual'], df[EXO_COLS]


def data_fingerprint(df: pd.DataFrame) -> str:
    """Sidik jari isi data yang dipakai model (harga + 7 fitur makro). Dicatat di tiap run baru."""
    arr = np.ascontiguousarray(np.round(df[['harga_jual'] + EXO_COLS].to_numpy(float), 10))
    h = hashlib.sha1()
    h.update(arr.tobytes())
    h.update(f'{df.index[0]}|{df.index[-1]}|{len(df)}'.encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# berkas
# ---------------------------------------------------------------------------
def run_paths(runs_dir: str, label: str, model: str, h: int) -> dict:
    base = os.path.join(runs_dir, f'{label}_{model}_h{h}')
    return {'json': base + '.json', 'pred': base + '_pred.csv', 'hist': base + '_history.json',
            'fimp': base + '_fimp.csv'}


def pred_frame(p: pd.Series, rows, h: int, y_true, y_pred) -> pd.DataFrame:
    dates = p.index
    return pd.DataFrame({'tanggal': [dates[i + h] for i in rows],
                         'y_true': np.asarray(y_true).ravel().tolist(),
                         'y_pred': np.asarray(y_pred).ravel().tolist()})


def save_run(runs_dir: str, label: str, model: str, h: int, res: dict, pred_df: pd.DataFrame,
             hist: dict | None = None, fimp: dict | None = None) -> None:
    P = run_paths(runs_dir, label, model, h)
    pred_df.to_csv(P['pred'], index=False)                    # prediksi dulu, JSON terakhir = penanda "selesai"
    if hist is not None:
        with open(P['hist'], 'w', encoding='utf-8') as f:
            json.dump(hist, f)
    if fimp is not None:
        pd.Series(fimp, name='importance').rename_axis('feature').to_csv(P['fimp'])
    with open(P['json'], 'w', encoding='utf-8') as f:
        json.dump(res, f)


def read_runs(runs_dir: str, labels, models, horizons, rev: str | None = None,
              data_hash: str | None = None) -> pd.DataFrame:
    """Kumpulkan semua run VALID (protokol/rev/sidik jari cocok) menjadi satu tabel."""
    rows = []
    for lb in labels:
        for m in models:
            for h in horizons:
                P = run_paths(runs_dir, lb, m, h)
                if run_is_current(P['json'], rev, data_hash) and os.path.exists(P['pred']):
                    with open(P['json'], encoding='utf-8') as f:
                        rows.append(json.load(f))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# satu run
# ---------------------------------------------------------------------------
def run_ml(p: pd.Series, exo_use: pd.DataFrame | None, label: str, model: str, h: int, split=None,
           exo_cols=None, F: pd.DataFrame | None = None, rev: str | None = None,
           data_hash: str | None = None, extra: dict | None = None):
    """Satu run ML. exo_use=None -> univariat. F (opsional) = tabel fitur yang sudah dibangun.
    Return (res, pred_df, None, feat_imp)."""
    t0 = time.time()
    if F is None:
        F = build_lgb_features(p, exo_use, exo_cols)
    feat_imp = None
    if model == 'lightgbm':
        mdl, metrics, rows, y_true, y_pred = train_lightgbm_h(p, F, h, LGB_PARAMS, split=split)
        feat_imp = dict(zip(mdl.feature_name(), [float(v) for v in mdl.feature_importance('gain')]))
    else:
        _, metrics, rows, y_true, y_pred, feat_imp = train_ml_h(p, F, h, model, split=split)
    res = {'setup': label, 'model': model, 'horizon': h, **metrics, 'test_n': len(rows),
           'waktu_detik': round(time.time() - t0, 1), 'protocol': PROTOCOL}
    if rev is not None:
        res['rev'] = rev
    if data_hash is not None:
        res['data_hash'] = data_hash
    if extra:
        res.update(extra)
    return res, pred_frame(p, rows, h, y_true, y_pred), None, feat_imp


def run_dl(p: pd.Series, exo_use: pd.DataFrame | None, label: str, model: str, h: int,
           lr: float = BASE_LR, wd: float = BASE_WD, seed: int = SEED, split=None, exo_cols=None,
           weights_path: str | None = None, data_hash: str | None = None, extra: dict | None = None):
    """Satu run DL (konfigurasi Table 3 jurnal acuan; lihat config.py). exo_use=None -> univariat.
    weights_path: simpan bobot terbaik (restore_best_weights) untuk SHAP tanpa melatih ulang.
    Return (res, pred_df, hist_dict, None)."""
    from .models_dl import BUILDERS                      # impor lambat: TensorFlow
    t0 = time.time()
    mdl, metrics, rows, y_true, y_pred, hist = train_dl_h(
        p, exo_use, h, BUILDERS[model], SEQ_LEN, DL_UNITS[model], epochs=EPOCHS, patience=PATIENCE,
        batch_size=DL_BATCH[model], learning_rate=lr, weight_decay=wd, seed=seed, split=split,
        exo_cols=exo_cols)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        mdl.save_weights(weights_path)
    dt = time.time() - t0
    free_memory()
    res = {'setup': label, 'model': model, 'horizon': h, **metrics, 'test_n': len(rows),
           'waktu_detik': round(dt, 1), 'lr_used': lr, 'wd_used': wd, 'seed': seed,
           'protocol': PROTOCOL, **epoch_info(hist)}
    if data_hash is not None:
        res['data_hash'] = data_hash
    if extra:
        res.update(extra)
    return res, pred_frame(p, rows, h, y_true, y_pred), hist, None


# ---------------------------------------------------------------------------
# kisi kombinasi
# ---------------------------------------------------------------------------
def run_grid(tasks, run_fn, runs_dir: str, rev: str | None = None, data_hash: str | None = None,
             force: bool = False) -> dict:
    """Jalankan semua (label, model, h) yang belum punya hasil VALID.

    run_fn(label, model, h) -> (res, pred_df, hist, fimp). Hasil valid dilewati (bisa dilanjutkan
    kapan saja); hasil basi (protokol/rev/data berbeda) ditimpa. Kegagalan satu run tidak
    menghentikan sisanya, tetapi dicatat -> pemanggil WAJIB memeriksa cnt['gagal'] == 0."""
    os.makedirs(runs_dir, exist_ok=True)
    cnt = {'baru': 0, 'sudah_ada': 0, 'basi_ditimpa': 0, 'gagal': 0, 'daftar_gagal': []}
    for label, model, h in tasks:
        P = run_paths(runs_dir, label, model, h)
        valid = run_is_current(P['json'], rev, data_hash) and os.path.exists(P['pred'])
        if valid and not force:
            cnt['sudah_ada'] += 1
            continue
        if os.path.exists(P['json']) and not valid:
            cnt['basi_ditimpa'] += 1
        print(f'RUN  {label}/{model}/h{h}', end=' ... ', flush=True)
        try:
            res, pred_df, hist, fimp = run_fn(label, model, h)
        except Exception as e:                                    # noqa: BLE001
            print(f'GAGAL: {type(e).__name__}: {e}')
            cnt['gagal'] += 1
            cnt['daftar_gagal'].append((label, model, h, f'{type(e).__name__}: {e}'))
            free_memory()
            continue
        save_run(runs_dir, label, model, h, res, pred_df, hist, fimp)
        cnt['baru'] += 1
        ep = f", {res['n_epochs']} epoch, terbaik={res['best_epoch']}" if 'n_epochs' in res else ''
        print(f"OK MAE={res['MAE']:,.0f} ({res['waktu_detik']}s{ep})")
    print(f"\nSelesai. Baru: {cnt['baru']} (menimpa hasil basi: {cnt['basi_ditimpa']}) | "
          f"sudah ada & valid: {cnt['sudah_ada']} | gagal: {cnt['gagal']}")
    return cnt
