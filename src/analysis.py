"""Fungsi analisis pasca-evaluasi (tidak melatih model apa pun).

Dipakai notebook 09-16: membaca prediksi tersimpan (`*_pred.csv`), menghitung metrik
bebas-skala, uji, stabilitas sub-periode, rezim volatilitas, arah, dan diagnosis loss curve.

Semua pemilihan model di sini bersifat DESKRIPTIF setelah evaluasi final; tidak dipakai
untuk tuning ulang.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import stats as S
from .pipeline import multihorizon_split, dl_target_baselines


# ---------------------------------------------------------------------------
# data & prediksi
# ---------------------------------------------------------------------------
def load_data(root: str = '..') -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(os.path.join(root, 'data', 'processed', 'data_gabungan.csv'),
                     index_col=0, parse_dates=True)
    return df, df['harga_jual']


def drift_frame(p: pd.Series, h: int, split=None) -> pd.DataFrame:
    """Prediksi Drift (rata-rata return h-hari pada TRAIN, konstan) pada baris uji.
    Indeks = tanggal target (i+h). Kolom: origin, p0, y_true, y_pred, r_true, r_pred, mu."""
    tr_dates, _, te_pos = split if split is not None else multihorizon_split(p, h)
    mu = float(np.mean((p.shift(-h) / p - 1).loc[tr_dates].values))
    te_pos = np.asarray(list(te_pos), int)
    p0 = p.iloc[te_pos].values
    out = pd.DataFrame({
        'origin': p.index[te_pos], 'p0': p0, 'y_true': p.iloc[te_pos + h].values,
        'y_pred': p0 * (1.0 + mu)}, index=p.index[te_pos + h])
    out.index.name = 'tanggal'
    out['r_true'] = out.y_true / out.p0 - 1.0
    out['r_pred'] = out.y_pred / out.p0 - 1.0
    out['mu'] = mu
    return out


def read_pred(results_dir: str, setup: str, model: str, h: int, runs_sub: str = 'runs') -> pd.DataFrame:
    f = os.path.join(results_dir, runs_sub, f'{setup}_{model}_h{h}_pred.csv')
    x = pd.read_csv(f, parse_dates=['tanggal']).set_index('tanggal')
    return x


def attach(pred: pd.DataFrame, dfr: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Gabungkan prediksi model dengan kerangka Drift (origin, p0) berdasarkan tanggal target.
    Memverifikasi y_true identik. Mengembalikan kolom: origin, p0, y_true, y_pred, r_true, r_pred."""
    j = dfr[['origin', 'p0', 'y_true']].join(pred[['y_true', 'y_pred']], how='inner', rsuffix='_m')
    if len(j) != len(pred):
        raise ValueError(f'baris tidak sejajar dengan Drift: {len(j)} vs {len(pred)}')
    if not np.allclose(j.y_true.values, j.y_true_m.values, rtol=0, atol=tol * max(1.0, np.abs(j.y_true.values).max())):
        raise ValueError('y_true model berbeda dari y_true Drift')
    out = j[['origin', 'p0', 'y_true', 'y_pred']].copy()
    out['r_true'] = out.y_true / out.p0 - 1.0
    out['r_pred'] = out.y_pred / out.p0 - 1.0
    return out


def vol_causal(p: pd.Series, window: int = 20) -> pd.Series:
    """Volatilitas realisasi (simpangan baku return harian, `window` hari) di hari origin.
    Hanya memakai informasi sampai hari origin -> aman untuk pengelompokan rezim."""
    return p.pct_change().rolling(window).std()


# ---------------------------------------------------------------------------
# ranking kelompok & wakil SHAP
# ---------------------------------------------------------------------------
def group_ranking(tabel: pd.DataFrame, models: list[str], by: str = 'MAE') -> pd.DataFrame:
    """Peringkat rata-rata DALAM kelompok (mis. 10 DL baseline) di seluruh (setup, horizon).
    Dasar pemilihan 'wakil terbaik' otomatis. Pemutus seri: rata-rata % perbaikan vs Drift
    (pct_vs_drift, lebih besar lebih baik), lalu nama model (deterministik)."""
    t = tabel[tabel.model.isin(models)].copy()
    t['rank_grup'] = t.groupby(['setup', 'horizon'])[by].rank(method='min')
    g = (t.groupby('model')
         .agg(mean_rank=('rank_grup', 'mean'), wins=('rank_grup', lambda x: int((x == 1).sum())),
              mean_MAE=('MAE', 'mean'),
              mean_pct_vs_drift=('pct_vs_drift', 'mean') if 'pct_vs_drift' in t else ('MAE', 'mean'))
         .reset_index())
    g = g.sort_values(['mean_rank', 'mean_pct_vs_drift', 'model'],
                      ascending=[True, False, True]).reset_index(drop=True)
    return g


def pick_representatives(tabel: pd.DataFrame, dl_models: list[str], ml_models: list[str],
                         always: str = 'dpformer') -> dict:
    """Tiga model SHAP: DPformer (tetap), terbaik dari DL baseline (tanpa DPformer), terbaik dari ML."""
    dl_base = [m for m in dl_models if m != always]
    g_dl, g_ml = group_ranking(tabel, dl_base), group_ranking(tabel, ml_models)
    return {'dpformer': always, 'dl': g_dl.iloc[0].model, 'ml': g_ml.iloc[0].model,
            'tabel_dl': g_dl, 'tabel_ml': g_ml}


# ---------------------------------------------------------------------------
# metrik bebas skala & stabilitas
# ---------------------------------------------------------------------------
def scale_free_metrics(dfm: pd.DataFrame, dfr: pd.DataFrame) -> dict:
    """Metrik di ruang return untuk satu model, terhadap Drift (dfr = drift_frame)."""
    a, f, fd = dfm.r_true.values, dfm.r_pred.values, dfr.r_pred.values
    return {
        'RMSE_ret': float(np.sqrt(np.mean((a - f) ** 2))),
        'MAE_ret': float(np.mean(np.abs(a - f))),
        'R2oos_vs_drift': S.oos_r2(a, f, fd),
        'std_ratio': S.std_ratio(f, a),
        'corr_ret': float(np.corrcoef(a, f)[0, 1]) if np.std(f) > 0 else np.nan,
    }


def block_table(dfm: pd.DataFrame, dfr: pd.DataFrame, n_blocks: int = 4) -> pd.DataFrame:
    """Rasio MSE-return model/Drift per blok waktu uji sama panjang (<1 = lebih baik dari Drift)."""
    n = len(dfm)
    blk = np.minimum((np.arange(n) * n_blocks) // n, n_blocks - 1)
    rows = []
    for b in range(n_blocks):
        k = blk == b
        a, f, fd = dfm.r_true.values[k], dfm.r_pred.values[k], dfr.r_pred.values[k]
        rows.append({'blok': b + 1, 'mulai': dfm.index[k][0], 'akhir': dfm.index[k][-1], 'n': int(k.sum()),
                     'rel_mse': float(np.mean((a - f) ** 2) / np.mean((a - fd) ** 2))})
    return pd.DataFrame(rows)


def regime_table(dfm: pd.DataFrame, dfr: pd.DataFrame, vol: pd.Series, thresholds, min_n: int = 20) -> pd.DataFrame:
    """Rasio MSE-return model/Drift per rezim volatilitas (0 tenang, 1 sedang, 2 tinggi) di hari origin.
    thresholds = (q33, q67) dihitung dari data TRAIN saja (pemanggil)."""
    v = vol.reindex(pd.DatetimeIndex(dfm.origin)).values
    reg = np.where(v < thresholds[0], 0, np.where(v < thresholds[1], 1, 2))
    rows = []
    for g in range(3):
        k = reg == g
        if k.sum() >= min_n:
            a, f, fd = dfm.r_true.values[k], dfm.r_pred.values[k], dfr.r_pred.values[k]
            rows.append({'rezim': g, 'n': int(k.sum()),
                         'rel_mse': float(np.mean((a - f) ** 2) / np.mean((a - fd) ** 2))})
    return pd.DataFrame(rows)


def vol_thresholds(p: pd.Series, h: int, window: int = 20, split=None):
    tr_dates, _, _ = split if split is not None else multihorizon_split(p, h)
    v = vol_causal(p, window).loc[tr_dates].dropna().values
    return tuple(np.quantile(v, [1 / 3, 2 / 3]))


def direction_row(dfm: pd.DataFrame, h: int) -> dict:
    """Akurasi arah + uji Pesaran-Timmermann pada sampel tak-tumpang-tindih (stride h)."""
    idx = np.arange(0, len(dfm), max(h, 1))
    r = S.pesaran_timmermann(dfm.r_true.values[idx], dfm.r_pred.values[idx])
    hit_all = float(((dfm.r_true.values > 0) == (dfm.r_pred.values > 0)).mean())
    return {'n_pt': r['n'], 'hit_pt': r['hit'], 'base_up': r['base_up'], 'pred_up': r['pred_up'],
            'p_one_pt': r['p_one'], 'hit_all': hit_all}


# ---------------------------------------------------------------------------
# diagnosis loss curve
# ---------------------------------------------------------------------------
def diagnosis_loss(history: dict, val_null: float, train_null: float = 1.0,
                   tol_skill: float = 0.02, overfit_rise: float = 0.05,
                   overfit_train_drop: float = 0.03) -> dict:
    """Diagnosis loss curve berbasis 'skill' terhadap model nol (memprediksi rata-rata train).

    skill = 1 - loss / loss_model_nol. Target DL sudah z-score pada train, sehingga loss
    model nol di train = 1 dan di validasi = val_null (pergeseran rezim membuat val_null != 1).

    Dua atribut TERPISAH:
      kelas (seberapa jauh epoch terbaik mengalahkan model nol di validasi)
        BELAJAR       : skill val terbaik >= +tol_skill
        TANPA_SINYAL  : |skill val terbaik| < tol_skill  (model hanya menemukan rata-rata;
                        wajar untuk return emas yang hampir efisien)
        DI_BAWAH_NOL  : skill val terbaik < -tol_skill   (bahkan epoch terbaik lebih buruk dari
                        model nol; rata-rata train tidak cocok dengan validasi / model tak
                        cukup terlatih)
      overfit (bool): setelah epoch terbaik, loss validasi naik > overfit_rise * val_null
                        DAN loss train masih turun > overfit_train_drop * train_null.
                        Bobot yang dipakai untuk uji = epoch terbaik (restore_best_weights),
                        jadi overfit di sini menjelaskan bentuk kurva, bukan model akhir.
    """
    loss = np.asarray(history.get('loss', []), float)
    val = np.asarray(history.get('val_loss', []), float)
    if len(loss) == 0 or len(val) == 0:
        return {'kelas': 'NO_HISTORY', 'overfit': False}
    best = int(np.argmin(val))
    n = len(val)
    s_val_best = 1.0 - val[best] / val_null
    s_val_final = 1.0 - val[-1] / val_null
    ib = min(best, len(loss) - 1)
    s_tr_best = 1.0 - loss[ib] / train_null
    s_tr_final = 1.0 - loss[-1] / train_null
    rise = (val[-1] - val[best]) / val_null
    drop = (loss[ib] - loss[-1]) / train_null
    overfit = bool(rise > overfit_rise and drop > overfit_train_drop)
    if s_val_best >= tol_skill:
        kelas = 'BELAJAR'
    elif s_val_best > -tol_skill:
        kelas = 'TANPA_SINYAL'
    else:
        kelas = 'DI_BAWAH_NOL'
    return {'kelas': kelas, 'overfit': overfit, 'best_epoch': best + 1, 'n_epochs': n,
            'skill_val_best': float(s_val_best), 'skill_val_final': float(s_val_final),
            'skill_train_best': float(s_tr_best), 'skill_train_final': float(s_tr_final),
            'val_min': float(val[best]), 'val_final': float(val[-1]),
            'train_final': float(loss[-1]), 'val_null': float(val_null),
            'naik_val_setelah_best': float(rise), 'turun_train_setelah_best': float(drop),
            'stopped_early': bool(n < 100)}


def loss_diagnosis_table(runs_dir: str, p: pd.Series, setups, models, horizons, seq_len: int = 10) -> pd.DataFrame:
    """Diagnosis seluruh history DL yang ada di runs_dir (file *_history.json)."""
    rows = []
    cache = {}
    for s in setups:
        for m in models:
            for h in horizons:
                f = os.path.join(runs_dir, f'{s}_{m}_h{h}_history.json')
                if not os.path.exists(f):
                    continue
                with open(f, encoding='utf-8') as fh:
                    hist = json.load(fh)
                if h not in cache:
                    cache[h] = dl_target_baselines(p, h, seq_len=seq_len)
                nb = cache[h]
                rows.append({'setup': s, 'model': m, 'horizon': h, **diagnosis_loss(hist, nb['val_mse_null'], nb['train_mse_null'])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# jendela peristiwa (dipilih OBJEKTIF dari data, bukan dipilih agar model tampak bagus)
# ---------------------------------------------------------------------------
def choose_event_windows(p: pd.Series, test_index: pd.DatetimeIndex, win: int = 120) -> list[dict]:
    """Tiga jendela peristiwa pada periode uji, dipilih OBJEKTIF dari harga (bukan dari kinerja model)
    dan tidak saling tumpang-tindih:
      (1) Reli terbesar     : kenaikan harga terbesar dalam `win` hari,
      (2) Volatilitas tinggi: rata-rata |return| harian tertinggi dalam `win` hari,
      (3) Sideways          : |perubahan harga| terkecil dalam `win` hari.
    Urutan jendela mengikuti waktu."""
    s = p.reindex(test_index).dropna()
    n = len(s)
    if n < 4 * win:
        win = max(20, n // 5)
    v = s.values
    chg = np.full(n, np.nan)
    chg[win:] = v[win:] / v[:-win] - 1.0                      # perubahan selama jendela yang berakhir di pos
    av = s.pct_change().abs().rolling(win).mean().values       # mean |return| harian, jendela berakhir di pos
    taken: list[tuple[int, int]] = []

    def free(pos: int) -> bool:
        st = pos - win
        return all(pos < a or st > b for a, b in taken)

    def pick(score: np.ndarray, largest: bool) -> int | None:
        order = np.argsort(-score if largest else score)
        for pos in order:
            if np.isfinite(score[pos]) and pos >= win and free(int(pos)):
                return int(pos)
        return None

    out = []
    for nama, score, largest in (('Reli terbesar', chg, True), ('Volatilitas tertinggi', av, True),
                                 ('Sideways', np.abs(chg), False)):
        pos = pick(score, largest)
        if pos is None:
            continue
        taken.append((pos - win, pos))
        out.append({'nama': nama, 'mulai': s.index[pos - win], 'akhir': s.index[pos]})
    return sorted(out, key=lambda e: e['mulai'])


# ---------------------------------------------------------------------------
# model wakil (SHAP, ablation, skenario B, walk-forward)
# ---------------------------------------------------------------------------
def load_wakil(results_dir: str) -> dict:
    """Baca results/wakil_model.json (ditulis notebook 09): {'dpformer': .., 'dl': .., 'ml': ..}."""
    f = os.path.join(results_dir, 'wakil_model.json')
    if not os.path.exists(f):
        raise FileNotFoundError('wakil_model.json belum ada: jalankan notebook 09 terlebih dahulu.')
    with open(f, encoding='utf-8') as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# tabel uji statistik & stabilitas (dipakai notebook 09, 10, 12, 15, 16)
# ---------------------------------------------------------------------------
def _frames(p: pd.Series, horizons, split_by_h: dict | None = None) -> dict:
    return {h: drift_frame(p, h, None if split_by_h is None else split_by_h[h]) for h in horizons}


def _load_attached(results_dir, setup, model, h, fr, runs_sub='runs'):
    f = os.path.join(results_dir, runs_sub, f'{setup}_{model}_h{h}_pred.csv')
    if not os.path.exists(f):
        return None
    return attach(read_pred(results_dir, setup, model, h, runs_sub), fr[h])


def metrics_return_table(results_dir, p, setups, models, horizons, runs_sub='runs', split_by_h=None) -> pd.DataFrame:
    """Metrik bebas skala (ruang return) + arah, satu baris per (setup, model, horizon)."""
    fr = _frames(p, horizons, split_by_h)
    rows = []
    for s in setups:
        for m in models:
            for h in horizons:
                d = _load_attached(results_dir, s, m, h, fr, runs_sub)
                if d is None:
                    continue
                rows.append({'setup': s, 'model': m, 'horizon': h, **scale_free_metrics(d, fr[h]), **direction_row(d, h)})
    return pd.DataFrame(rows)


def dm_vs_drift_table(results_dir, p, setups, models, horizons, runs_sub='runs', kernel='rectangular',
                      split_by_h=None) -> pd.DataFrame:
    """Uji Diebold-Mariano model vs Drift. Utama: loss kuadrat RUANG RETURN, statistik HLN (t).
    Pembanding: ruang harga. Koreksi BH: (i) seluruh uji dalam satu setup, (ii) per (setup, horizon)."""
    fr = _frames(p, horizons, split_by_h)
    rows = []
    for s in setups:
        for m in models:
            for h in horizons:
                d = _load_attached(results_dir, s, m, h, fr, runs_sub)
                if d is None:
                    continue
                a = S.dm_compare(d.y_true.values, d.y_pred.values, fr[h].y_pred.values, d.p0.values, h, 'return', 'sq', kernel)
                b = S.dm_compare(d.y_true.values, d.y_pred.values, fr[h].y_pred.values, d.p0.values, h, 'price', 'sq', kernel)
                rows.append({'setup': s, 'model': m, 'horizon': h, 'n': a['n'], 'rel_mse_ret': a['rel_loss'],
                             'stat_hln': a['stat_hln'], 'p_one_hln': a['p_one_hln'], 'p_two_hln': a['p_two_hln'],
                             'p_one_normal': a['p_one'], 'lrv_fallback': a['lrv_fallback'],
                             'rel_mse_price': b['rel_loss'], 'stat_hln_price': b['stat_hln'], 'p_one_hln_price': b['p_one_hln']})
    t = pd.DataFrame(rows)
    if len(t):
        t['q_bh_setup'] = np.nan
        t['q_bh_setup_h'] = np.nan
        for _, g in t.groupby('setup'):
            t.loc[g.index, 'q_bh_setup'] = S.bh_fdr(g.p_one_hln.values, 0.05)[0]
        for _, g in t.groupby(['setup', 'horizon']):
            t.loc[g.index, 'q_bh_setup_h'] = S.bh_fdr(g.p_one_hln.values, 0.05)[0]
        t['lebih_baik_dari_drift'] = t.rel_mse_ret < 1
    return t


def dm_multi_vs_uni_table(results_dir, p, models, horizons, runs_sub='runs', kernel='rectangular',
                          split_by_h=None, a='multivariat', b='univariat', runs_sub_b=None) -> pd.DataFrame:
    """DM: multivariat (A) vs univariat (B) untuk tiap model & horizon (stat < 0 = A lebih baik).
    a, b = label run; runs_sub_b = sub-folder run B bila berbeda dari A (mis. skenario B vs multivariat lag-1)."""
    fr = _frames(p, horizons, split_by_h)
    rows = []
    for m in models:
        for h in horizons:
            da = _load_attached(results_dir, a, m, h, fr, runs_sub)
            db = _load_attached(results_dir, b, m, h, fr, runs_sub_b or runs_sub)
            if da is None or db is None:
                continue
            r = S.dm_compare(da.y_true.values, da.y_pred.values, db.y_pred.values, da.p0.values, h, 'return', 'sq', kernel)
            rows.append({'model': m, 'horizon': h, 'n': r['n'], 'rel_mse_ret': r['rel_loss'], 'stat_hln': r['stat_hln'],
                         'p_one_hln': r['p_one_hln'], 'p_two_hln': r['p_two_hln']})
    t = pd.DataFrame(rows)
    if len(t):
        t['q_bh'] = S.bh_fdr(t.p_one_hln.values, 0.05)[0]
        t['q_bh_h'] = np.nan
        for _, g in t.groupby('horizon'):
            t.loc[g.index, 'q_bh_h'] = S.bh_fdr(g.p_one_hln.values, 0.05)[0]
    return t


def mcs_tables(results_dir, p, setups, models, horizons, alpha=0.10, B=2000, seed=0, runs_sub='runs',
               split_by_h=None, include_drift=True):
    """MCS (loss kuadrat return) per (setup, horizon); Drift ikut sebagai kandidat. Return (incl, pvals) DataFrame
    berindeks nama model, kolom 'Univ. h=5', dst."""
    fr = _frames(p, horizons, split_by_h)
    incl, pv = {}, {}
    for s in setups:
        for h in horizons:
            names, L = [], []
            for m in models:
                d = _load_attached(results_dir, s, m, h, fr, runs_sub)
                if d is None:
                    continue
                names.append(m)
                L.append(S.loss_series(d.y_true.values, d.y_pred.values, d.p0.values, 'return', 'sq'))
            if len(names) < 2:
                continue
            if include_drift:
                names.append('drift')
                L.append(S.loss_series(fr[h].y_true.values, fr[h].y_pred.values, fr[h].p0.values, 'return', 'sq'))
            # panjang seragam (semua model dievaluasi pada baris uji yang sama)
            res = S.mcs(np.column_stack(L), names, alpha=alpha, B=B, block=max(h, 5), seed=seed)
            key = f'{"Univ" if s == "univariat" else "Multi" if s == "multivariat" else s}. h={h}'
            incl[key] = pd.Series({n: (n in res['included']) for n in names})
            pv[key] = pd.Series(res['pvalues'])
    return pd.DataFrame(incl), pd.DataFrame(pv)


def block_regime_tables(results_dir, p, setups, models, horizons, n_blocks=4, runs_sub='runs', split_by_h=None):
    """Rasio MSE-return model/Drift per blok waktu uji dan per rezim volatilitas (batas dari TRAIN saja)."""
    fr = _frames(p, horizons, split_by_h)
    vol = vol_causal(p, 20)
    brow, rrow = [], []
    for s in setups:
        for h in horizons:
            thr = vol_thresholds(p, h, split=None if split_by_h is None else split_by_h[h])
            for m in models:
                d = _load_attached(results_dir, s, m, h, fr, runs_sub)
                if d is None:
                    continue
                b = block_table(d, fr[h], n_blocks)
                b['setup'], b['model'], b['horizon'] = s, m, h
                brow.append(b)
                r = regime_table(d, fr[h], vol, thr)
                r['setup'], r['model'], r['horizon'] = s, m, h
                rrow.append(r)
    return (pd.concat(brow, ignore_index=True) if brow else pd.DataFrame(),
            pd.concat(rrow, ignore_index=True) if rrow else pd.DataFrame())


def summarize_dm(t: pd.DataFrame, p_col: str = 'p_one_hln', q_cols=('q_bh_setup', 'q_bh_setup_h')) -> dict:
    """Ringkasan angka utama tabel DM (untuk teks naskah)."""
    out = {'n_uji': int(len(t)), 'n_p<0.05': int((t[p_col] < 0.05).sum()), 'n_p<0.10': int((t[p_col] < 0.10).sum()),
           'n_lebih_baik_dari_drift (rel_mse<1)': int((t.rel_mse_ret < 1).sum()) if 'rel_mse_ret' in t else None}
    for q in q_cols:
        if q in t:
            out[f'n_{q}<0.05'] = int((t[q] < 0.05).sum())
    return out


# ---------------------------------------------------------------------------
# perbandingan varian (ablation fitur, skenario B, MOSSA vs default) dan walk-forward
# ---------------------------------------------------------------------------
def variant_table(results_dir, p, variants: dict, models, horizons, ref: str | None = None,
                  kernel: str = 'rectangular', split_by_h=None) -> pd.DataFrame:
    """Bandingkan beberapa VARIAN run terhadap Drift (dan terhadap varian acuan `ref`).

    variants = {nama: (runs_sub, label)} -> berkas `<results_dir>/<runs_sub>/<label>_<model>_h<h>_pred.csv`.
    Kolom: grup, model, horizon, n, MAE, pct_vs_drift (MAE harga; positif = lebih baik dari Drift),
    rel_mse_ret & p_one_hln (DM vs Drift, ruang return), dan bila ref diberikan:
    rel_mse_vs_ref, stat_vs_ref, p_one_vs_ref (stat < 0 / rel < 1 = varian ini lebih baik daripada ref)."""
    fr = _frames(p, horizons, split_by_h)
    cache = {}
    rows = []
    for g, (sub, lb) in variants.items():
        for m in models:
            for h in horizons:
                d = _load_attached(results_dir, lb, m, h, fr, sub)
                cache[(g, m, h)] = d
    for g in variants:
        for m in models:
            for h in horizons:
                d = cache[(g, m, h)]
                if d is None:
                    continue
                dr = fr[h]
                a = S.dm_compare(d.y_true.values, d.y_pred.values, dr.y_pred.values, d.p0.values, h, 'return', 'sq', kernel)
                mae = float(np.mean(np.abs(d.y_true.values - d.y_pred.values)))
                mae_d = float(np.mean(np.abs(dr.y_true.values - dr.y_pred.values)))
                row = {'grup': g, 'model': m, 'horizon': h, 'n': a['n'], 'MAE': mae,
                       'pct_vs_drift': 100.0 * (mae_d - mae) / mae_d, 'rel_mse_ret': a['rel_loss'],
                       'p_one_hln': a['p_one_hln']}
                if ref is not None and g != ref and cache.get((ref, m, h)) is not None:
                    dref = cache[(ref, m, h)]
                    b = S.dm_compare(d.y_true.values, d.y_pred.values, dref.y_pred.values, d.p0.values, h, 'return', 'sq', kernel)
                    row.update({'rel_mse_vs_ref': b['rel_loss'], 'stat_vs_ref': b['stat_hln'], 'p_one_vs_ref': b['p_one_hln']})
                rows.append(row)
    return pd.DataFrame(rows)


def wf_tables(results_dir, p, setups, models, horizons, folds_by_h: dict, runs_fmt: str = 'walkforward/f{k}/runs',
              kernel: str = 'rectangular'):
    """Walk-forward (rolling-origin, expanding window). folds_by_h[h] = pipeline.walkforward_splits(p, h, ...).

    Return (per_fold, pooled):
      per_fold: setup, model, horizon, blok(=nomor fold), mulai, akhir, n, rel_mse (MSE return model / Drift), p_one_hln
      pooled  : setup, model, horizon, n, rel (rasio MSE gabungan), stat_hln, p_one_hln, n_fold_lebih_baik, n_fold,
                q_bh (per setup), q_bh_h (per setup x horizon)
    Uji DM gabungan memakai deret selisih-loss seluruh fold berurutan (blok uji tak tumpang-tindih).
    Kombinasi yang tidak lengkap di semua fold dilewati."""
    pf, po = [], []
    for s in setups:
        for m in models:
            for h in horizons:
                parts = []
                ok = True
                for fd in folds_by_h[h]:
                    k = fd['fold']
                    f = os.path.join(results_dir, runs_fmt.format(k=k), f'{s}_{m}_h{h}_pred.csv')
                    if not os.path.exists(f):
                        ok = False
                        break
                    fr = drift_frame(p, h, fd['split'])
                    d = attach(pd.read_csv(f, parse_dates=['tanggal']).set_index('tanggal'), fr)
                    parts.append((fd, d, fr))
                if not ok or not parts:
                    continue
                better = 0
                for fd, d, fr in parts:
                    r = S.dm_compare(d.y_true.values, d.y_pred.values, fr.y_pred.values, d.p0.values, h, 'return', 'sq', kernel)
                    better += int(r['rel_loss'] < 1)
                    pf.append({'setup': s, 'model': m, 'horizon': h, 'blok': fd['fold'], 'mulai': fd['start'], 'akhir': fd['end'],
                               'n': r['n'], 'rel_mse': r['rel_loss'], 'p_one_hln': r['p_one_hln']})
                yt = np.concatenate([d.y_true.values for _, d, _ in parts])
                yp = np.concatenate([d.y_pred.values for _, d, _ in parts])
                yd = np.concatenate([fr.y_pred.values for _, _, fr in parts])
                p0 = np.concatenate([d.p0.values for _, d, _ in parts])
                r = S.dm_compare(yt, yp, yd, p0, h, 'return', 'sq', kernel)
                po.append({'setup': s, 'model': m, 'horizon': h, 'n': r['n'], 'rel': r['rel_loss'], 'stat_hln': r['stat_hln'],
                           'p_one_hln': r['p_one_hln'], 'n_fold_lebih_baik': better, 'n_fold': len(parts)})
    per_fold, pooled = pd.DataFrame(pf), pd.DataFrame(po)
    if len(pooled):
        pooled['q_bh'] = np.nan
        pooled['q_bh_h'] = np.nan
        for _, g in pooled.groupby('setup'):
            pooled.loc[g.index, 'q_bh'] = S.bh_fdr(g.p_one_hln.values, 0.05)[0]
        for _, g in pooled.groupby(['setup', 'horizon']):
            pooled.loc[g.index, 'q_bh_h'] = S.bh_fdr(g.p_one_hln.values, 0.05)[0]
    return per_fold, pooled
