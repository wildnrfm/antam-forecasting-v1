"""Analisis sekunder untuk naskah (notebook 17-19). TIDAK mengubah protokol utama.

Isi
---
* pembanding Drift ADAPTIF (uji stres, kausal)         : bench_frame, level_shift_table
* selang kepercayaan + efek terkecil yang terdeteksi    : ci_stats, single_ci_table, ci_summary
* kombinasi bobot-sama                                  : combo_table
* tangga waktu informasi                                : make_exo_lead, ladder_table, fimp_share_table, shap_rung
* figur                                                 : fig_forest, fig_ci_summary, fig_combo_ladder,
                                                          fig_combo_heat, fig_timing_ladder, fig_timing_shap

Prinsip
-------
1. SEKUNDER / EKSPLORATIF. Dijalankan setelah hasil utama dilihat; tidak dipakai memilih model atau
   hiperparameter; seluruh hasil dilaporkan, termasuk yang tidak mendukung cerita. Bukan uji konfirmatori.
2. Statistik memakai kode yang sama dengan notebook 10 (`stats.dm_test`: kernel persegi sampai lag h-1,
   koreksi HLN, loss kuadrat di ruang return). Selang kepercayaan = selang-t dari statistik DM-HLN yang sama.
3. Pembanding adaptif memakai HANYA return h-hari yang labelnya sudah diketahui pada origin (j <= i-h): kausal.
4. Modul ini tidak melatih apa pun kecuali `shap_rung` (melatih ulang satu model ML deterministik untuk SHAP,
   persis seperti notebook 11).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as _st

from . import analysis as A
from . import plots as P
from . import stats as S
from .config import ALL_MODELS, DL_MODELS, ML_MODELS, QUICK
from .pipeline import EXO_B_SAMEDAY, multihorizon_split

# ---------------------------------------------------------------------------
# konfigurasi (ditetapkan di sini, bukan dari hasil)
# ---------------------------------------------------------------------------
ROLL_WINDOW = 250                                    # ~1 tahun hari kerja; konvensional, tidak dituning
BENCH = ('drift', 'drift_exp', 'drift_roll')
BENCH_COL = {'drift': 'r_drift', 'drift_exp': 'r_exp', 'drift_roll': 'r_roll'}
RECIPES = {'EW_semua21': list(ALL_MODELS), 'EW_ML10': list(ML_MODELS), 'EW_DL11': list(DL_MODELS)}
CI_LEVEL = 0.95
EPS_EXCLUDE = (0.05, 0.10)                           # "perbaikan >= 5% / 10% tersingkir" bila batas bawah selang > 1-eps
MDE_ALPHA, MDE_POWER = 0.05, 0.80                    # uji satu-sisi 5%, daya 80%

RUNG_ORDER = ['lag1', 'hari_t', 'plasebo_t1']
RUNG_SPEC = {'lag1': ('multivariat', 'runs'),                       # protokol utama (indikator makro lag-1)
             'hari_t': ('skenario_b', 'skenario_b/runs'),           # notebook 15
             'plasebo_t1': ('plasebo_t1', 'tangga_waktu/runs')}     # notebook 19 (run baru)
PLACEBO_LABEL, PLACEBO_DIR = 'plasebo_t1', 'tangga_waktu'
SHAP_MODELS = ('lightgbm', 'rf', 'extra_trees')      # model pohon deterministik (XGBoost tidak: n_jobs=-1 tidak reprodusibel)
SHAP_H = (20,) if QUICK else (1, 20)
SHAP_N_BG = 20 if QUICK else 50
SHAP_N_EX = 20 if QUICK else 100

# ---------------------------------------------------------------------------
# teks figur (id/en), mengikuti bahasa yang diatur lewat plots.set_lang
# ---------------------------------------------------------------------------
_T2 = {
    'id': {
        'rel': 'Rasio MSE return (model / pembanding)', 'uni': 'Univariat', 'multi': 'Multivariat',
        'ci_typ': 'selang 95% tipikal (median batas)', 'med': 'median', 'dot': 'model (setup)', 'best': 'terbaik (pasca-seleksi)',
        'mde': 'Efek terkecil yang terdeteksi\n(daya 80%, uji satu-sisi 5%), %',
        'b_drift': 'Drift (rata-rata train)', 'b_exp': 'Drift ekspansif (kausal)', 'b_roll': 'Drift bergulir 250 hari (kausal)',
        'oracle': 'konstan = rata-rata return uji (tidak layak)', 'median_single': 'median model tunggal',
        'rel_drift': 'MSE return relatif terhadap Drift (<1: lebih baik)',
        'rung_lag1': 'lag-1\n(protokol\nutama)', 'rung_hari_t': 'penutupan\nhari t\n(skenario B)',
        'rung_plasebo_t1': 'penutupan\nhari t+1\n(plasebo)', 'median_model': 'median 10 model ML',
        'share': 'porsi |SHAP| (%)', 'ci_note': 'n/h = jumlah jendela independen efektif',
        'clip_note': 'batang selang yang melewati batas sumbu terpotong', 'sig': '* p<0,05   † lolos BH (FDR 5%)', 'ew_note': 'resep ditetapkan sebelum dihitung; 3 resep x 5 horizon x 2 setup = 30 uji',
    },
    'en': {
        'rel': 'Return-MSE ratio (model / benchmark)', 'uni': 'Univariate', 'multi': 'Multivariate',
        'ci_typ': 'typical 95% interval (median bounds)', 'med': 'median', 'dot': 'model (setup)', 'best': 'best (post-selection)',
        'mde': 'Minimum detectable effect\n(80% power, one-sided 5%), %',
        'b_drift': 'Drift (train mean)', 'b_exp': 'Expanding drift (causal)', 'b_roll': 'Rolling 250-day drift (causal)',
        'oracle': 'constant = test-mean return (infeasible)', 'median_single': 'median single model',
        'rel_drift': 'Return MSE relative to Drift (<1: better)',
        'rung_lag1': 'lag-1\n(main\nprotocol)', 'rung_hari_t': 'close of\nday t\n(scenario B)',
        'rung_plasebo_t1': 'close of\nday t+1\n(placebo)', 'median_model': 'median of 10 ML models',
        'share': 'share of |SHAP| (%)', 'ci_note': 'n/h = effective number of independent windows',
        'clip_note': 'interval bars beyond the axis limits are clipped', 'sig': '* p<0.05   † passes BH (FDR 5%)', 'ew_note': 'recipes fixed before computing; 3 recipes x 5 horizons x 2 setups = 30 tests',
    },
}


def _t(key: str) -> str:
    lang = P._lang if P._lang in _T2 else 'id'
    return _T2[lang].get(key, _T2['id'].get(key, key))


def bench_label(k: str) -> str:
    return _t({'drift': 'b_drift', 'drift_exp': 'b_exp', 'drift_roll': 'b_roll'}[k])


# ---------------------------------------------------------------------------
# pembanding Drift (tetap + dua versi adaptif kausal)
# ---------------------------------------------------------------------------
def bench_frame(p: pd.Series, h: int, split=None, roll_window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Kerangka evaluasi pada baris uji. Indeks = tanggal target (i+h).

    Kolom: origin, p0, y_true, r_true, serta tiga pembanding konstan-per-baris di ruang return:
      r_drift : Drift resmi (rata-rata return h-hari pada TRAIN; satu nilai untuk semua baris uji)
      r_exp   : rata-rata SELURUH return h-hari yang labelnya sudah diketahui pada origin i (j <= i-h)
      r_roll  : rata-rata `roll_window` return h-hari terakhir yang labelnya sudah diketahui pada origin i
    r_exp dan r_roll KAUSAL (tidak memakai label yang belum terselesaikan pada origin)."""
    dfr = A.drift_frame(p, h, split)
    te = np.asarray(list(split[2] if split is not None else multihorizon_split(p, h)[2]), int)
    pv = p.to_numpy(float)
    Rh = pv[h:] / pv[:-h] - 1.0                              # Rh[j] = R_h(origin j), j = 0..n-h-1
    cs = np.concatenate([[0.0], np.cumsum(Rh)])              # cs[k] = jumlah Rh[0..k-1]
    k_end = te - h                                           # label terakhir yang sudah terselesaikan di origin te
    if (k_end < 0).any() or (k_end > len(Rh) - 1).any():
        raise ValueError('indeks label di luar jangkauan')
    exp_ = cs[k_end + 1] / (k_end + 1)
    lo = np.maximum(0, k_end - roll_window + 1)
    roll = (cs[k_end + 1] - cs[lo]) / (k_end + 1 - lo)
    out = dfr[['origin', 'p0', 'y_true', 'r_true']].copy()
    out['r_drift'] = dfr['r_pred'].to_numpy(float)
    out['r_exp'] = exp_
    out['r_roll'] = roll
    return out


def _check_truth(a: pd.DataFrame, b: pd.DataFrame, what: str = '') -> None:
    if len(a) != len(b) or not np.allclose(a.r_true.to_numpy(float), b.r_true.to_numpy(float), rtol=0, atol=1e-9):
        raise ValueError(f'baris/return aktual tidak sejajar dengan kerangka pembanding {what}')


def check_bench_causal(p: pd.Series, horizons, n_origin: int = 25, roll_window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Pagar kausalitas Drift adaptif. Untuk origin uji terpilih, hitung ulang r_exp dan r_roll HANYA dari harga sampai
    origin (deret dipotong di origin; harga sesudahnya dibuang, jadi tidak mungkin terpakai) dan bandingkan dengan nilai
    di `bench_frame`. KONTROL POSITIF: versi bocor (satu label ekstra, yang ujung harganya jatuh SESUDAH origin) harus
    berbeda, supaya pagar ini terbukti peka. Melempar AssertionError bila ada selisih atau kontrol tidak peka."""
    rows = []
    for h in horizons:
        b = bench_frame(p, h, roll_window=roll_window)
        pos = p.index.get_indexer(b.origin)
        pv = p.to_numpy(float)
        rfull = pv[h:] / pv[:-h] - 1.0                      # rfull[j] = return h-hari dengan origin j (butuh harga j+h)
        sel = np.unique(np.linspace(0, len(b) - 1, min(n_origin, len(b))).astype(int))
        worst, ctrl = 0.0, 0.0
        for k in sel:
            i = int(pos[k])
            pt = pv[:i + 1]                                  # hanya harga sampai origin (inklusif)
            rt = pt[h:] / pt[:-h] - 1.0                      # label yang sudah terselesaikan di origin: j = 0..i-h
            worst = max(worst, abs(float(rt.mean()) - float(b.r_exp.iloc[k])),
                        abs(float(rt[-roll_window:].mean()) - float(b.r_roll.iloc[k])))
            leaky = float(rfull[:i - h + 2].mean())          # j = 0..i-h+1: label terakhir butuh harga i+1 (masa depan)
            ctrl = max(ctrl, abs(leaky - float(rt.mean())))
        rows.append({'horizon': h, 'n_origin_dicek': int(len(sel)), 'selisih_maks': worst, 'selisih_kontrol_bocor': ctrl})
    out = pd.DataFrame(rows)
    if (out.selisih_maks > 1e-12).any():
        raise AssertionError(f'Drift adaptif tidak kausal: selisih maksimum {out.selisih_maks.max():.2e}')
    if (out.selisih_kontrol_bocor <= 1e-12).any():
        raise AssertionError('kontrol positif tidak peka: versi bocor tidak berbeda dari versi kausal')
    return out


def level_shift_table(p: pd.Series, horizons) -> pd.DataFrame:
    """Seberapa jauh rata-rata train (Drift) tertinggal dari rezim uji, dan berapa MSE yang bisa dihemat
    oleh konstanta saja. rel_* = MSE-return pembanding / MSE-return Drift (<1: lebih baik dari Drift).
    `rel_oracle_konstan` memakai rata-rata return UJI (tidak layak dipakai; hanya batas bawah untuk konstanta)."""
    rows = []
    for h in horizons:
        b = bench_frame(p, h)
        y = b.r_true.to_numpy(float)
        mse = lambda r: float(np.mean((y - np.asarray(r, float)) ** 2))
        m0 = mse(b.r_drift.to_numpy())
        orc = mse(np.full(len(y), y.mean()))
        rows.append({'horizon': h, 'n': len(b), 'mu_train_pct': 100 * float(b.r_drift.iloc[0]),
                     'mean_uji_pct': 100 * float(y.mean()), 'rel_drift_exp': mse(b.r_exp.to_numpy()) / m0,
                     'rel_drift_roll': mse(b.r_roll.to_numpy()) / m0, 'rel_oracle_konstan': orc / m0,
                     'maks_hemat_konstan_pct': 100 * (1 - orc / m0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# selang kepercayaan dan efek terkecil yang terdeteksi
# ---------------------------------------------------------------------------
def ci_stats(r_true, r_a, r_b, h: int, level: float = CI_LEVEL, kernel: str = 'rectangular') -> dict:
    """Rasio MSE-return A/B (A = model, B = pembanding) + selang kepercayaan + MDE.

    Semuanya berasal dari deret selisih loss d_t = L_A,t - L_B,t yang sama dengan uji DM (stats.dm_test):
      se       = sqrt(lrv/n) / hln                      (galat baku statistik DM-HLN)
      selang   = 1 + (dbar +- t_{n-1, (1+level)/2} * se) / mean(L_B)
      MDE      = (z_{1-alpha} + z_{daya}) * se / mean(L_B)   (perbaikan relatif MSE terkecil yang terdeteksi)
    n_eff = n/h adalah jumlah jendela h-hari yang tidak saling tumpang-tindih (kasar)."""
    r_true, r_a, r_b = (np.asarray(x, float) for x in (r_true, r_a, r_b))
    La, Lb = (r_true - r_a) ** 2, (r_true - r_b) ** 2
    ok = np.isfinite(La) & np.isfinite(Lb)
    La, Lb = La[ok], Lb[ok]
    t = S.dm_test(La - Lb, h, kernel)
    n, mb = t['n'], float(Lb.mean()) if len(Lb) else np.nan
    nan = np.nan
    out = {'n': n, 'n_eff': n / h, 'rel_mse': float(La.mean() / mb) if (len(Lb) and mb > 0) else nan,
           'ci_lo': nan, 'ci_hi': nan, 'se_rel': nan, 'mde_rel': nan,
           'stat_hln': t['stat_hln'], 'p_one_hln': t['p_one_hln'], 'p_two_hln': t['p_two_hln']}
    if not np.isfinite(t['lrv']) or not (mb > 0):
        return out
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    se = np.sqrt(max(t['lrv'], 0.0) / n) / hln
    tq = _st.t.ppf(0.5 + level / 2, n - 1)
    out.update(ci_lo=1 + (t['dbar'] - tq * se) / mb, ci_hi=1 + (t['dbar'] + tq * se) / mb, se_rel=se / mb,
               mde_rel=(_st.norm.ppf(1 - MDE_ALPHA) + _st.norm.ppf(MDE_POWER)) * se / mb)
    return out


def single_ci_table(res_dir: str, p: pd.Series, setups, models, horizons, runs_sub: str = 'runs',
                    benches=BENCH) -> pd.DataFrame:
    """Satu baris per (setup, model, horizon, pembanding): rasio MSE-return, selang 95%, MDE, p DM-HLN satu-sisi.
    q_bh = Benjamini-Hochberg dalam keluarga (setup, pembanding) = semua model x horizon."""
    fr = A._frames(p, horizons)
    bf = {h: bench_frame(p, h) for h in horizons}
    rows = []
    for s in setups:
        for m in models:
            for h in horizons:
                d = A._load_attached(res_dir, s, m, h, fr, runs_sub)
                if d is None:
                    continue
                b = bf[h].reindex(d.index)
                _check_truth(d, b, f'({s}, {m}, h={h})')
                for k in benches:
                    rows.append({'setup': s, 'model': m, 'horizon': h, 'bench': k,
                                 **ci_stats(d.r_true.values, d.r_pred.values, b[BENCH_COL[k]].values, h)})
    t = pd.DataFrame(rows)
    if len(t):
        t['q_bh'] = np.nan
        for _, g in t.groupby(['setup', 'bench']):
            t.loc[g.index, 'q_bh'] = S.bh_fdr(g.p_one_hln.values, 0.05)[0]
    return t


def ci_summary(t: pd.DataFrame, eps=EPS_EXCLUDE) -> pd.DataFrame:
    """Ringkasan per (pembanding, horizon) dari `single_ci_table`.
    n_tersingkir_X = jumlah kombinasi yang batas bawah selang 95%-nya > 1-X (perbaikan >= X% tersingkir)."""
    rows = []
    for (b, h), g in t.groupby(['bench', 'horizon']):
        r = {'bench': b, 'horizon': h, 'n_komb': len(g), 'n_eff_median': float(g.n_eff.median()),
             'rel_median': float(g.rel_mse.median()), 'rel_min_pascaseleksi': float(g.rel_mse.min()),
             'ci_lo_median': float(g.ci_lo.median()), 'ci_hi_median': float(g.ci_hi.median()),
             'mde_median_pct': float(100 * g.mde_rel.median()),
             'n_rel<1': int((g.rel_mse < 1).sum()), 'n_p<0.05': int((g.p_one_hln < 0.05).sum()),
             'n_q<0.05': int((g.q_bh < 0.05).sum())}
        for e in eps:
            r[f'n_tersingkir_{int(round(e * 100))}pct'] = int((g.ci_lo > 1 - e).sum())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# kombinasi bobot-sama
# ---------------------------------------------------------------------------
def combo_table(res_dir: str, p: pd.Series, setups, horizons, recipes: dict | None = None, runs_sub: str = 'runs',
                benches=BENCH) -> pd.DataFrame:
    """Rata-rata sederhana prediksi RETURN beberapa model (bobot sama; tanpa memilih model dari hasil uji).

    Satu baris per (setup, horizon, resep). Kolom penting:
      rel_<b>, p_<b>, q_<b>, ci_lo_<b>, ci_hi_<b> : terhadap pembanding b (drift / drift_exp / drift_roll); q = BH atas
                                                    seluruh uji (setup x horizon x resep) per pembanding
      mse_mean_member_vs_drift  : rata-rata MSE anggota / MSE Drift. MSE kombinasi SELALU <= angka ini (konveksitas),
                                  jadi 'kombinasi lebih baik dari anggota tipikal' bukan temuan empiris.
      rel_oracle_vs_drift       : MSE konstanta = rata-rata return UJI / MSE Drift (tidak layak; batas konstanta)
      rel_combo_vs_oracle       : MSE kombinasi / MSE konstanta-oracle; > 1 berarti kombinasi kalah dari konstanta itu
      std_ratio, pred_up        : simpangan baku prediksi / aktual; porsi hari prediksi naik"""
    recipes = RECIPES if recipes is None else recipes
    all_models = list(dict.fromkeys(m for ms in recipes.values() for m in ms))
    fr = A._frames(p, horizons)
    bf = {h: bench_frame(p, h) for h in horizons}
    rows = []
    for s in setups:
        for h in horizons:
            F = {}
            for m in all_models:
                d = A._load_attached(res_dir, s, m, h, fr, runs_sub)
                if d is not None:
                    F[m] = d
            if not F:
                continue
            ref = next(iter(F.values()))
            idx = ref.index
            for m, d in F.items():
                if not d.index.equals(idx):
                    raise ValueError(f'baris uji {m} ({s}, h={h}) tidak sama dengan model lain')
            b = bf[h].reindex(idx)
            _check_truth(ref, b, f'({s}, h={h})')
            y = ref.r_true.to_numpy(float)
            Rm = pd.DataFrame({m: d.r_pred.to_numpy(float) for m, d in F.items()}, index=idx)
            mse = lambda r: float(np.mean((y - np.asarray(r, float)) ** 2))
            m0, m_or = mse(b.r_drift.to_numpy()), mse(np.full(len(y), y.mean()))
            for name, members in recipes.items():
                ms = [m for m in members if m in Rm]
                if len(ms) < 2:
                    continue
                rc = Rm[ms].mean(axis=1).to_numpy()
                row = {'setup': s, 'horizon': h, 'resep': name, 'k': len(ms), 'k_nominal': len(members), 'n': len(y),
                       'mse_mean_member_vs_drift': float(np.mean([mse(Rm[m].to_numpy()) for m in ms])) / m0,
                       'rel_oracle_vs_drift': m_or / m0, 'rel_combo_vs_oracle': mse(rc) / m_or,
                       'std_ratio': float(S.std_ratio(rc, y)), 'pred_up': float(np.mean(rc > 0)),
                       'mean_pred_pct': 100 * float(rc.mean()), 'mean_true_pct': 100 * float(y.mean())}
                for k in benches:
                    st = ci_stats(y, rc, b[BENCH_COL[k]].to_numpy(), h)
                    row.update({f'rel_{k}': st['rel_mse'], f'p_{k}': st['p_one_hln'],
                                f'ci_lo_{k}': st['ci_lo'], f'ci_hi_{k}': st['ci_hi']})
                rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        for k in benches:
            t[f'q_{k}'] = S.bh_fdr(t[f'p_{k}'].values, 0.05)[0]
    return t


# ---------------------------------------------------------------------------
# tangga waktu informasi
# ---------------------------------------------------------------------------
def make_exo_lead(exo: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Geser kolom pasar (EXO_B_SAMEDAY) `lead` hari kerja ke depan; FFR dan BI-Rate tetap lag-1.
    lead=1  -> baris t memuat penutupan hari t (identik dengan pipeline.make_exo_scenario_b; skenario B)
    lead=2  -> baris t memuat penutupan hari t+1: informasi yang BELUM ada saat peramalan (plasebo, tidak layak).
    Baris terakhir `lead` bernilai NaN dan gugur otomatis dari fitur (build_lgb_features memakai dropna)."""
    out = exo.copy()
    for c in EXO_B_SAMEDAY:
        out[c] = exo[c].shift(-lead)
    return out


def ladder_table(res_dir: str, p: pd.Series, models, horizons, rungs=RUNG_ORDER) -> pd.DataFrame:
    """Setiap anak tangga dibandingkan dengan Drift, Drift bergulir, dan anak tangga pertama (lag-1), pada baris uji
    yang SAMA (irisan tanggal target; plasebo h=1 kehilangan satu baris terakhir). rel < 1 = lebih baik.
    q_* = BH per anak tangga atas semua model x horizon."""
    fr = A._frames(p, horizons)
    bf = {h: bench_frame(p, h) for h in horizons}
    rows = []
    for m in models:
        for h in horizons:
            D = {}
            for r in rungs:
                lb, sub = RUNG_SPEC[r]
                d = A._load_attached(res_dir, lb, m, h, fr, sub)
                if d is not None:
                    D[r] = d
            if len(D) < len(rungs):
                continue
            idx = D[rungs[0]].index
            for r in rungs[1:]:
                idx = idx.intersection(D[r].index)
            b = bf[h].reindex(idx)
            base = D[rungs[0]].reindex(idx)
            for r in rungs:
                a = D[r].reindex(idx)
                _check_truth(a, b, f'({r}, {m}, h={h})')
                y, ra = a.r_true.to_numpy(float), a.r_pred.to_numpy(float)
                sd, sr = ci_stats(y, ra, b.r_drift.to_numpy(), h), ci_stats(y, ra, b.r_roll.to_numpy(), h)
                row = {'model': m, 'horizon': h, 'rung': r, 'n': sd['n'],
                       'rel_drift': sd['rel_mse'], 'p_drift': sd['p_one_hln'],
                       'rel_roll': sr['rel_mse'], 'p_roll': sr['p_one_hln'],
                       'rel_lag1': np.nan, 'p_lag1': np.nan, 'ci_lo_lag1': np.nan, 'ci_hi_lag1': np.nan}
                if r != rungs[0]:
                    sl = ci_stats(y, ra, base.r_pred.to_numpy(float), h)
                    row.update(rel_lag1=sl['rel_mse'], p_lag1=sl['p_one_hln'], ci_lo_lag1=sl['ci_lo'], ci_hi_lag1=sl['ci_hi'])
                rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        for col, qcol in (('p_lag1', 'q_lag1'), ('p_drift', 'q_drift'), ('p_roll', 'q_roll')):
            t[qcol] = np.nan
            for _, g in t.groupby('rung'):
                t.loc[g.index, qcol] = S.bh_fdr(g[col].values, 0.05)[0]
    return t


def fimp_share_table(res_dir: str, models, horizons, rungs=RUNG_ORDER) -> pd.DataFrame:
    """Porsi importance (%) per variabel dari berkas *_fimp.csv yang SUDAH ada (tanpa melatih ulang).
    Skala impurity/gain/|koefisien| berbeda antar model, jadi hanya PORSI per model yang dibandingkan.
    Indikatif: impurity-based cenderung bias; gunakan SHAP sebagai angka utama."""
    from .explain import GROUP_ORDER, group_of
    rows = []
    for r in rungs:
        lb, sub = RUNG_SPEC[r]
        for m in models:
            for h in horizons:
                f = os.path.join(res_dir, sub, f'{lb}_{m}_h{h}_fimp.csv')
                if not os.path.exists(f):
                    continue
                x = pd.read_csv(f)
                s = x.groupby(x.feature.map(group_of)).importance.sum()
                tot = float(s.sum())
                if tot <= 0:
                    continue
                for g in GROUP_ORDER:
                    rows.append({'rung': r, 'model': m, 'horizon': h, 'variabel': g, 'porsi_pct': 100 * float(s.get(g, 0.0)) / tot})
    return pd.DataFrame(rows)


def shap_rung(p: pd.Series, exo_rung: pd.DataFrame, model: str, h: int, n_background: int = SHAP_N_BG,
              n_explain: int = SHAP_N_EX, stored_pred: pd.DataFrame | None = None):
    """SHAP satu model ML pada satu anak tangga (melatih ulang model deterministik, seperti notebook 11).
    Baris yang dijelaskan = origin uji yang tersebar merata dan SAMA untuk semua anak tangga (origin terakhir dibuang
    karena plasebo t+1 tidak punya nilainya). Return (tabel porsi per variabel, info)."""
    from . import explain as E
    te = list(multihorizon_split(p, h)[2])[:-1]
    idx = np.unique(np.linspace(0, len(te) - 1, min(n_explain, len(te))).astype(int))
    res = E.explain_ml(p, exo_rung, model, h, [te[i] for i in idx], n_background, 42, stored_pred=stored_pred)
    tab = E.importance_table(res)
    tab['model'], tab['horizon'] = model, h
    info = {'method': res['method'], 'repro_maxrel_diff': res['repro_maxrel_diff'],
            'completeness_err': res['completeness_err'], 'n_explained': int(len(res['pred']))}
    return tab, info


# ---------------------------------------------------------------------------
# figur
# ---------------------------------------------------------------------------
_C_UNI, _C_MULTI = '#4c78a8', '#f58518'


def fig_forest(t: pd.DataFrame, models, horizons, bench: str = 'drift'):
    """Forest plot: rasio MSE-return model/pembanding dengan selang 95%, satu panel per horizon, dua setup per model."""
    g0 = t[t.bench == bench]
    n = len(models)
    n_ml = len([m for m in models if m in P.ML_NAMES])
    fig, axs = plt.subplots(1, len(horizons), figsize=(2.55 * len(horizons) + 1.5, 0.255 * n + 1.9), sharey=True, squeeze=False)
    axs = axs[0]
    y = np.arange(n)
    off = {'univariat': -0.18, 'multivariat': 0.18}
    col = {'univariat': _C_UNI, 'multivariat': _C_MULTI}
    for ax, h in zip(axs, horizons):
        g = g0[g0.horizon == h]
        for s in ('univariat', 'multivariat'):
            gs = g[g.setup == s].set_index('model').reindex(models)
            ok = gs.rel_mse.notna().to_numpy()
            if not ok.any():
                continue
            yy = y[ok] + off[s]
            ax.hlines(yy, gs.ci_lo.to_numpy()[ok], gs.ci_hi.to_numpy()[ok], color=col[s], lw=1.2)
            ax.plot(gs.rel_mse.to_numpy()[ok], yy, 'o', ms=3, color=col[s])
        ax.axvline(1.0, color='k', lw=0.8, ls='--')
        ax.axhline(n_ml - 0.5, color='#999999', lw=0.6)
        if len(g):
            lo, hi = np.nanpercentile(g.ci_lo, 3), np.nanpercentile(g.ci_hi, 97)
            ax.set_xlim(max(min(lo, 0.97) - 0.02, 0.2), min(max(hi, 1.03) + 0.02, 3.0))
        ax.set_title(f'h={h}\nn/h≈{g.n_eff.median():.0f}' if len(g) else f'h={h}', fontsize=9)
        ax.grid(axis='y', alpha=0.0)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([P.label(m) for m in models])
    axs[0].set_ylim(n - 0.4, -0.7)
    for lab in axs[0].get_yticklabels():
        if lab.get_text() == 'DPformer':
            lab.set_fontweight('bold')
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.12 + 0.5 / max(n, 1), wspace=0.08)
    fig.text(0.55, 0.062, _t('rel') + f'  [{bench_label(bench)}]', ha='center', fontsize=8.5)
    h1 = plt.Line2D([], [], color=_C_UNI, marker='o', ms=3.5, lw=1.2, label=_t('uni'))
    h2 = plt.Line2D([], [], color=_C_MULTI, marker='o', ms=3.5, lw=1.2, label=_t('multi'))
    fig.legend(handles=[h1, h2], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.55, 0.0), fontsize=8)
    fig.text(0.99, 0.004, _t('clip_note'), ha='right', fontsize=6.5, color='#666666')
    return fig


def fig_ci_summary(t: pd.DataFrame, horizons, bench: str = 'drift'):
    """(a) sebaran rasio MSE model/pembanding dengan selang 95% tipikal; (b) efek terkecil yang terdeteksi (MDE)."""
    g0 = t[t.bench == bench]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={'width_ratios': [1.55, 1]})
    x = np.arange(len(horizons))
    rng = np.random.default_rng(0)
    for i, h in enumerate(horizons):
        g = g0[g0.horizon == h]
        if g.empty:
            continue
        lo, hi, med = float(g.ci_lo.median()), float(g.ci_hi.median()), float(g.rel_mse.median())
        a1.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color='#bbbbbb', alpha=0.55, lw=0))
        a1.plot([i - 0.3, i + 0.3], [med, med], color='k', lw=1.7)
        a1.plot(i + rng.uniform(-0.22, 0.22, len(g)), g.rel_mse.to_numpy(), '.', color=_C_UNI, ms=4, alpha=0.55)
        a1.plot(i, float(g.rel_mse.min()), 'v', color='#d62728', ms=6)
    a1.axhline(1.0, color='k', lw=0.8, ls='--')
    a1.set_xticks(x)
    a1.set_xticklabels([f'h={h}' for h in horizons])
    a1.set_ylabel(_t('rel') + f'\n[{bench_label(bench)}]', fontsize=8.5)
    a1.set_title('(a)', loc='left', fontsize=10, fontweight='bold')
    hs = [plt.Rectangle((0, 0), 1, 1, color='#bbbbbb', alpha=0.55), plt.Line2D([], [], color='k', lw=1.7),
          plt.Line2D([], [], color=_C_UNI, marker='.', ls='', ms=6), plt.Line2D([], [], color='#d62728', marker='v', ls='', ms=6)]
    a1.legend(hs, [_t('ci_typ'), _t('med'), _t('dot'), _t('best')], frameon=False, fontsize=7.5, loc='best')
    mde = [100 * float(g0[g0.horizon == h].mde_rel.median()) for h in horizons]
    a2.bar(x, mde, color='#8c8c8c', width=0.6)
    for i, v in enumerate(mde):
        a2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    a2.set_xticks(x)
    a2.set_xticklabels([f'h={h}' for h in horizons])
    a2.set_ylabel(_t('mde'), fontsize=8.5)
    a2.set_title('(b)', loc='left', fontsize=10, fontweight='bold')
    a2.set_ylim(0, max(mde) * 1.18 if len(mde) else 1)
    fig.tight_layout()
    return fig


def fig_combo_ladder(C: pd.DataFrame, LS: pd.DataFrame, single_med: pd.DataFrame, horizons,
                     setups=('univariat', 'multivariat')):
    """MSE-return relatif terhadap Drift resmi: kombinasi vs konstanta adaptif (kausal) vs konstanta-oracle."""
    fig, axs = plt.subplots(1, len(setups), figsize=(5.5 * len(setups), 4.1), sharey=True, squeeze=False)
    axs = axs[0]
    ls = LS.set_index('horizon').reindex(horizons)
    x = np.arange(len(horizons))
    for ax, s in zip(axs, setups):
        ax.axhline(1.0, color='k', lw=1.1, label=_t('b_drift'))
        ax.plot(x, ls.rel_drift_exp.to_numpy(), '-s', color='#444444', ms=3.5, lw=1.2, label=_t('b_exp'))
        ax.plot(x, ls.rel_drift_roll.to_numpy(), '-^', color='#888888', ms=3.5, lw=1.2, label=_t('b_roll'))
        ax.plot(x, ls.rel_oracle_konstan.to_numpy(), ':', color='#a0a0a0', lw=2.0, label=_t('oracle'))
        for name, c in (('EW_ML10', '#2a9d8f'), ('EW_DL11', '#e76f51'), ('EW_semua21', '#6a4c93')):
            g = C[(C.setup == s) & (C.resep == name)].set_index('horizon').reindex(horizons)
            ax.plot(x, g.rel_drift.to_numpy(), '-o', color=c, ms=4, lw=1.7, label=name.replace('_', '-'))
        sm = single_med[single_med.setup == s].set_index('horizon').reindex(horizons)
        ax.plot(x, sm.rel_mse.to_numpy(), '--', color=_C_UNI, lw=1.0, label=_t('median_single'))
        ax.set_xticks(x)
        ax.set_xticklabels([f'h={h}' for h in horizons])
        ax.set_title(_t('uni') if s == 'univariat' else _t('multi') if s == 'multivariat' else s, fontsize=10)
    axs[0].set_ylabel(_t('rel_drift'), fontsize=8.5)
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc='lower center', ncol=4, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.27, top=0.92, wspace=0.06)
    return fig


def fig_combo_heat(C: pd.DataFrame, horizons, benches=BENCH, setups=('univariat', 'multivariat'),
                   recipes=('EW_semua21', 'EW_ML10', 'EW_DL11'), lim: float = 0.15):
    """Peta rasio MSE-return kombinasi/pembanding (<1 = kombinasi lebih baik), tiga pembanding berdampingan."""
    rows = [(s, r) for s in setups for r in recipes]
    fig, axs = plt.subplots(1, len(benches), figsize=(4.5 * len(benches), 0.42 * len(rows) + 1.9), sharey=True, squeeze=False)
    axs = axs[0]
    im = None
    for ax, b in zip(axs, benches):
        mat = np.full((len(rows), len(horizons)), np.nan)
        pm, qm = mat.copy(), mat.copy()
        for i, (s, r) in enumerate(rows):
            g = C[(C.setup == s) & (C.resep == r)].set_index('horizon').reindex(horizons)
            mat[i], pm[i], qm[i] = g[f'rel_{b}'].to_numpy(), g[f'p_{b}'].to_numpy(), g[f'q_{b}'].to_numpy()
        im = ax.imshow(mat, cmap='RdBu_r', norm=TwoSlopeNorm(1.0, 1 - lim, 1 + lim), aspect='auto')
        for i in range(len(rows)):
            for j in range(len(horizons)):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                mark = '†' if (np.isfinite(qm[i, j]) and qm[i, j] < 0.05) else '*' if (np.isfinite(pm[i, j]) and pm[i, j] < 0.05) else ''
                ax.text(j, i, f'{v:.3f}{mark}', ha='center', va='center', fontsize=7)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f'h={h}' for h in horizons])
        ax.set_title(bench_label(b), fontsize=9)
        ax.grid(False)
    short = {'univariat': 'Univ', 'multivariat': 'Multi'}
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([f"{short.get(s, s)} · {r.replace('_', '-')}" for s, r in rows], fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.90, top=0.88, bottom=0.17, wspace=0.05)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax).set_label(_t('rel'), fontsize=7.5)
    fig.text(0.5, 0.012, _t('sig') + '   |   ' + _t('ew_note'), ha='center', fontsize=7.5)
    return fig


def fig_timing_ladder(L: pd.DataFrame, models, horizons, rungs=RUNG_ORDER):
    """Rasio MSE-return model/Drift di tiap anak tangga waktu informasi; satu panel per horizon, garis per model."""
    # sumbu-y BERSAMA: bentuk "V" hanya pada h=1 dan datarnya h>=5 terlihat apa adanya (tidak diperbesar per panel)
    fig, axs = plt.subplots(1, len(horizons), figsize=(2.6 * len(horizons) + 1.2, 4.1), squeeze=False, sharey=True)
    axs = axs[0]
    x = np.arange(len(rungs))
    for ax, h in zip(axs, horizons):
        g = L[L.horizon == h]
        for m in models:
            gm = g[g.model == m].set_index('rung').reindex(rungs)
            if gm.rel_drift.notna().sum() < 2:
                continue
            ax.plot(x, gm.rel_drift.to_numpy(), '-o', ms=3, lw=0.9, color=P.model_color(m), alpha=0.85)
        med = g.groupby('rung').rel_drift.median().reindex(rungs)
        ax.plot(x, med.to_numpy(), '-D', color='k', lw=2.0, ms=5, zorder=5)
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels([_t(f'rung_{r}') for r in rungs], fontsize=6.8)
        ax.set_title(f'h={h}', fontsize=9)
        ax.set_xlim(-0.35, len(rungs) - 0.65)
    axs[0].set_ylabel(_t('rel_drift'), fontsize=8.5)
    hs = [plt.Line2D([], [], color=P.model_color(m), marker='o', ms=3.5, lw=1.0) for m in models]
    hs.append(plt.Line2D([], [], color='k', marker='D', ms=5, lw=2.0))
    fig.legend(hs, [P.label(m) for m in models] + [_t('median_model')], loc='lower center', ncol=6, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.26, top=0.90, wspace=0.28, left=0.07, right=0.99)
    return fig


def fig_timing_shap(SH: pd.DataFrame, horizons, rungs=RUNG_ORDER):
    """Porsi |SHAP| (%) per variabel (rata-rata model pohon) untuk tiap (horizon, anak tangga): batang bertumpuk."""
    from .explain import GROUP_LABEL, GROUP_ORDER
    cols = {'emas': '#9e9e9e', 'dxy': '#d62728', 'us10yr': '#e377c2', 'sp500': '#1f77b4', 'usd_idr': '#ff7f0e',
            'ihsg': '#2ca02c', 'ffr': '#c5b0d5', 'bi_rate': '#aec7e8'}
    labs = [(h, r) for h in horizons for r in rungs]
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(labs) + 1.7))
    left = np.zeros(len(labs))
    yy = np.arange(len(labs))
    for g in GROUP_ORDER:
        vals = []
        for h, r in labs:
            s = SH[(SH.horizon == h) & (SH.rung == r) & (SH.variabel == g)]
            vals.append(float(s.porsi_pct.mean()) if len(s) else 0.0)
        vals = np.array(vals)
        ax.barh(yy, vals, left=left, color=cols.get(g, '#777777'), edgecolor='white', lw=0.5,
                label=GROUP_LABEL.get(g, g).split(' (')[0])
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(left[i] + v / 2, yy[i], f'{v:.0f}', ha='center', va='center', fontsize=7, color='white')
        left += vals
    ax.set_yticks(yy)
    ax.set_yticklabels([f"h={h} · {_t(f'rung_{r}').replace(chr(10), ' ')}" for h, r in labs], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(_t('share'))
    ax.set_xlim(0, 100)
    ax.legend(ncol=4, frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    ax.grid(axis='y', alpha=0.0)
    fig.tight_layout()
    return fig
