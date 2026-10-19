"""Figur untuk skripsi/naskah (gaya mengikuti Bao et al. 2026 + figur diagnostik tambahan).

Setiap fungsi menerima data (DataFrame/array) dan mengembalikan matplotlib Figure; penyimpanan
lewat save_fig(). Label figur dua bahasa: set_lang('id') (default) atau set_lang('en') untuk naskah
jurnal (env TA_LANG juga dihormati lewat src.config.LANG).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

try:
    from .config import LANG as _LANG0
except Exception:                                         # pragma: no cover
    _LANG0 = 'id'

# ---------------------------------------------------------------------------
# bahasa & label
# ---------------------------------------------------------------------------
_T = {
    'id': {
        'actual': 'Aktual', 'pred': 'Prediksi', 'horizon': 'Horizon (hari kerja)', 'model': 'Model',
        'price': 'Harga Antam (Rp/gram)', 'price_m': 'Harga Antam (juta Rp/gram, skala log)',
        'train': 'Latih', 'val': 'Validasi', 'test': 'Uji', 'vol': 'Volatilitas 60 hari (%/hari)',
        'drift': 'Drift', 'date': 'Tanggal', 'epoch': 'Epoch', 'loss': 'Loss (MSE, z-score target)',
        'univ': 'Univariat', 'multi': 'Multivariat', 'rmse_k': 'RMSE (ribu Rp)', 'mae': 'MAE (Rp)',
        'mape': 'MAPE (%)', 'better': 'lebih baik', 'worse': 'lebih buruk', 'corr': 'Korelasi',
        'std': 'Simpangan baku (ternormalisasi)', 'rel_mse': 'MSE return relatif terhadap Drift',
        'rel_mae': 'MAE relatif terhadap Drift', 'blok': 'Blok waktu uji', 'regime': 'Rezim volatilitas',
        'calm': 'tenang', 'mid': 'sedang', 'high': 'tinggi', 'improve': 'Perbaikan terhadap model lain (%)',
        'zero_share': 'Porsi hari dengan return harian = 0', 'year': 'Tahun',
        'null_val': 'model nol (rata-rata train) - validasi', 'null_tr': 'model nol - train',
        'best_ep': 'epoch terbaik', 'trainc': 'train', 'valc': 'validasi',
        'iter': 'Iterasi MOSSA', 'fit': 'Fitness (MAE validasi, Rp)', 'best_so_far': 'terbaik sejauh ini',
        'rolling': 'MSE return bergulir relatif terhadap Drift (jendela {w} hari)',
        'sig': 'p<0,05 (uji DM, HLN)', 'sig_bh': 'lolos BH (FDR 5%)', 'inmcs': 'Masuk MCS 90%',
        'lag_ret': 'Return Antam hari t vs variabel makro', 'lag_pred': 'Korelasi dengan return Antam t+1',
        'used': 'dipakai model (lag-1)', 'sameday': 'penutupan hari t (skenario B)',
        'shap_mean': 'Rata-rata |nilai SHAP| (pentingnya fitur)', 'shap_val': 'Kontribusi Shapley (dampak pada output model)',
        'feat_val': 'Nilai fitur', 'low': 'Rendah', 'high_': 'Tinggi', 'base': 'nilai dasar', 'final': 'prediksi',
        'fold': 'Fold', 'seed': 'Seed', 'events': 'Jendela peristiwa',
        'mae_k': 'MAE (ribu Rp)', 'ev_rally': 'Reli terbesar', 'ev_vol': 'Volatilitas tertinggi',
        'ev_flat': 'Sideways', 'overfit': 'overfit setelah epoch terbaik',
        'cls_BELAJAR': 'Belajar (lebih baik dari model nol)', 'cls_TANPA_SINYAL': 'Tanpa sinyal (≈ model nol)',
        'cls_DI_BAWAH_NOL': 'Di bawah model nol',
        'overlap': 'tumpang-tindih h-1 lag', 'acf_note': 'pita ±1,96/√n tidak berlaku pada lag < h',
        'err_ret': 'galat return (%)',
        'in_set': 'di dalam himpunan MCS', 'out_set': 'tersingkir dari MCS',
        'bh_fam': 'keluarga uji BH: 105 uji per setup',
    },
    'en': {
        'actual': 'Actual', 'pred': 'Prediction', 'horizon': 'Horizon (business days)', 'model': 'Model',
        'price': 'Antam price (IDR/gram)', 'price_m': 'Antam price (million IDR/gram, log scale)',
        'train': 'Train', 'val': 'Validation', 'test': 'Test', 'vol': '60-day volatility (%/day)',
        'drift': 'Drift', 'date': 'Date', 'epoch': 'Epoch', 'loss': 'Loss (MSE, z-scored target)',
        'univ': 'Univariate', 'multi': 'Multivariate', 'rmse_k': 'RMSE (thousand IDR)', 'mae': 'MAE (IDR)',
        'mape': 'MAPE (%)', 'better': 'better', 'worse': 'worse', 'corr': 'Correlation',
        'std': 'Standard deviation (normalised)', 'rel_mse': 'Return MSE relative to Drift',
        'rel_mae': 'MAE relative to Drift', 'blok': 'Test sub-period', 'regime': 'Volatility regime',
        'calm': 'calm', 'mid': 'medium', 'high': 'high', 'improve': 'Improvement over other models (%)',
        'zero_share': 'Share of days with zero daily return', 'year': 'Year',
        'null_val': 'null model (train mean) - validation', 'null_tr': 'null model - train',
        'best_ep': 'best epoch', 'trainc': 'train', 'valc': 'validation',
        'iter': 'MOSSA iteration', 'fit': 'Fitness (validation MAE, IDR)', 'best_so_far': 'best so far',
        'rolling': 'Rolling return MSE relative to Drift ({w}-day window)',
        'sig': 'p<0.05 (DM test, HLN)', 'sig_bh': 'passes BH (FDR 5%)', 'inmcs': 'In 90% MCS',
        'lag_ret': 'Antam return on day t vs macro variables', 'lag_pred': 'Correlation with Antam return t+1',
        'used': 'used by models (lag-1)', 'sameday': 'close of day t (scenario B)',
        'shap_mean': 'Mean |SHAP value| (feature importance)', 'shap_val': 'Shapley contribution (impact on model output)',
        'feat_val': 'Feature value', 'low': 'Low', 'high_': 'High', 'base': 'base value', 'final': 'prediction',
        'fold': 'Fold', 'seed': 'Seed', 'events': 'Event windows',
        'mae_k': 'MAE (thousand IDR)', 'ev_rally': 'Largest rally', 'ev_vol': 'Highest volatility',
        'ev_flat': 'Sideways', 'overfit': 'overfits after best epoch',
        'cls_BELAJAR': 'Learns (beats null model)', 'cls_TANPA_SINYAL': 'No signal (≈ null model)',
        'cls_DI_BAWAH_NOL': 'Below null model',
        'overlap': 'overlap (h-1 lags)', 'acf_note': '±1.96/√n band invalid for lags < h',
        'err_ret': 'return error (%)',
        'in_set': 'inside the MCS', 'out_set': 'eliminated from the MCS',
        'bh_fam': 'BH family: 105 tests per setup',
    },
}
_lang = _LANG0 if _LANG0 in _T else 'id'


def set_lang(lang: str) -> None:
    global _lang
    _lang = lang if lang in _T else 'id'


def tr(key: str, **kw) -> str:
    s = _T[_lang].get(key, _T['id'].get(key, key))
    return s.format(**kw) if kw else s


MODEL_LABEL = {
    'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'rf': 'Random Forest', 'catboost': 'CatBoost',
    'gbm': 'GBM', 'svr': 'SVR', 'extra_trees': 'ExtraTrees', 'hist_gbm': 'HistGBM',
    'adaboost': 'AdaBoost', 'elasticnet': 'ElasticNet', 'lstm': 'LSTM', 'gru': 'GRU',
    'bilstm': 'BiLSTM', 'bigru': 'BiGRU', 'transformer': 'Transformer', 'dlinear': 'DLinear',
    'nlinear': 'NLinear', 'patchtst': 'PatchTST', 'lstm_att': 'LSTM-Attn', 'gru_att': 'GRU-Attn',
    'dpformer': 'DPformer', 'drift': 'Drift', 'DRIFT': 'Drift',
}
ML_NAMES = ['lightgbm', 'xgboost', 'rf', 'catboost', 'gbm', 'svr', 'extra_trees', 'hist_gbm', 'adaboost', 'elasticnet']
DL_NAMES = ['lstm', 'gru', 'bilstm', 'bigru', 'transformer', 'dlinear', 'nlinear', 'patchtst', 'lstm_att', 'gru_att']


def label(m: str) -> str:
    return MODEL_LABEL.get(m, m)


def model_color(m: str) -> str:
    if m in ('dpformer',):
        return '#c2185b'
    if m in ('drift', 'DRIFT'):
        return '#111111'
    if m in ML_NAMES:
        return plt.get_cmap('GnBu')(0.40 + 0.55 * ML_NAMES.index(m) / (len(ML_NAMES) - 1))
    if m in DL_NAMES:
        return plt.get_cmap('YlOrRd')(0.35 + 0.60 * DL_NAMES.index(m) / (len(DL_NAMES) - 1))
    return '#777777'


C_TRAIN, C_VAL, C_TEST = '#4c78a8', '#f2c14e', '#e45756'
C_ACT = '#d4a017'


def set_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 200, 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.alpha': 0.25, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'figure.constrained_layout.use': False,
    })


set_style()


def save_fig(fig, name: str, figdir: str, formats=('png', 'pdf'), dpi: int = 200) -> list[str]:
    os.makedirs(figdir, exist_ok=True)
    out = []
    for ext in formats:
        pth = os.path.join(figdir, f'{name}.{ext}')
        fig.savefig(pth, dpi=dpi, bbox_inches='tight')
        out.append(pth)
    return out


def _panel(ax, letter: str) -> None:
    ax.text(-0.02, 1.04, f'({letter})', transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')


# ---------------------------------------------------------------------------
# 0. diagram kerangka (setara Fig. 1-2 jurnal acuan; digambar ulang dari kode proyek ini)
# ---------------------------------------------------------------------------
def _fbox(ax, x, y, w, h, text, fc='#ffffff', ec='#555555', fs=8.0, bold=False, tc='#111111', lw=1.0, ha='center', pad=0.012):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={pad}', fc=fc, ec=ec, lw=lw))
    ax.text(x + (w / 2 if ha == 'center' else 0.012), y + h / 2, text, ha=ha, va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=tc, linespacing=1.25)


def _farrow(ax, p1, p2, color='#444444', lw=1.3, rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=11, lw=lw, color=color,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def fig_framework(info: dict):
    """Kerangka penelitian (setara Fig. 1 jurnal acuan), digambar dari konfigurasi proyek.
    info: n (hari kerja), start, end (str), horizons (list), n_ml, n_dl."""
    import textwrap
    id_ = _lang == 'id'
    hz = ', '.join(str(h) for h in info['horizons'])
    cols = [
        ('1', 'Data' if id_ else 'Data', '#dbe7f5', [
            'Harga emas Antam (jual, Rp/gram)' if id_ else 'Antam gold price (IDR/gram)',
            '7 indikator makro: DXY, S&P 500, USD/IDR, IHSG, US10Y, FFR, BI-Rate' if id_ else '7 macro indicators: DXY, S&P 500, USD/IDR, IHSG, US10Y, FFR, BI-Rate',
            f"{info['n']} hari kerja, {info['start']} s.d. {info['end']}" if id_ else f"{info['n']} business days, {info['start']} to {info['end']}"]),
        ('2', 'Praproses' if id_ else 'Preprocessing', '#e3f1df', [
            'Hari tanpa pengumuman: forward-fill' if id_ else 'Days without a quote: forward-fill',
            'Makro dibuat stasioner (log-return / selisih)' if id_ else 'Macro series made stationary (log-return / difference)',
            'Makro di-lag 1 hari (utama); skenario B: hari-t (sensitivitas)' if id_ else 'Macro lagged 1 day (main); scenario B: same-day (sensitivity)',
            'FFR bulanan dipakai mulai bulan berikutnya' if id_ else 'Monthly FFR used from the following month']),
        ('3', 'Split dan fitur' if id_ else 'Split and features', '#fdf0d5', [
            'Kronologis 70/10/20 + embargo h hari' if id_ else 'Chronological 70/10/20 + h-day embargo',
            'Scaler/z-score hanya fit pada train' if id_ else 'Scalers fitted on train only',
            'ML: 15 (uni) / 22 (multi) fitur tabular' if id_ else 'ML: 15 (uni) / 22 (multi) tabular features',
            'DL: jendela 10 hari x 1 / 8 kanal' if id_ else 'DL: 10-day window x 1 / 8 channels']),
        ('4', 'Model' if id_ else 'Models', '#f6dde8', [
            f'Direct multi-horizon: h = {hz}' if id_ else f'Direct multi-horizon: h = {hz}',
            f"{info['n_ml']} model ML" if id_ else f"{info['n_ml']} ML models",
            f"{info['n_dl']} DL baseline" if id_ else f"{info['n_dl']} DL baselines",
            'DPformer + MOSSA: (lr, wd) dicari dengan fitness = MAE validasi ber-embargo' if id_ else 'DPformer + MOSSA: (lr, wd) searched with embargoed validation MAE',
            'Benchmark: Drift (rata-rata return train)' if id_ else 'Benchmark: Drift (train mean return)']),
        ('5', 'Evaluasi' if id_ else 'Evaluation', '#e6e0f3', [
            'Ruang harga: MAE, RMSE, sMAPE, MAPE' if id_ else 'Price space: MAE, RMSE, sMAPE, MAPE',
            'Uji: Diebold-Mariano (HLN), BH-FDR, MCS, Pesaran-Timmermann' if id_ else 'Tests: Diebold-Mariano (HLN), BH-FDR, MCS, Pesaran-Timmermann',
            'Ketahanan: seed, ablation, skenario B, walk-forward, rezim' if id_ else 'Robustness: seeds, ablation, scenario B, walk-forward, regimes',
            'SHAP pada 3 model terpilih otomatis' if id_ else 'SHAP on 3 automatically selected models']),
    ]
    fig, ax = plt.subplots(figsize=(14.5, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    n = len(cols)
    gap = 0.028
    w = (1 - gap * (n - 1)) / n
    for i, (num, title, color, lines) in enumerate(cols):
        x = i * (w + gap)
        _fbox(ax, x, 0.84, w, 0.11, f'{num}. {title}', fc=color, ec='#444444', fs=9.5, bold=True)
        ax.add_patch(plt.Rectangle((x, 0.10), w, 0.70, fc='#ffffff', ec='#444444', lw=1.0))
        y = 0.775
        for ln in lines:
            wrapped = textwrap.fill('- ' + ln, width=int(w * 112), subsequent_indent='  ')
            nl = wrapped.count('\n') + 1
            ax.text(x + 0.010, y, wrapped, ha='left', va='top', fontsize=8.0, linespacing=1.25)
            y -= 0.032 * nl + 0.035
        if i < n - 1:
            _farrow(ax, (x + w + 0.002, 0.895), (x + w + gap - 0.002, 0.895))
    ax.text(0.5, 0.0, ('Seluruh pemilihan model/hiperparameter hanya memakai validasi; uji disentuh sekali untuk evaluasi akhir.' if id_
                       else 'All model/hyperparameter selection uses validation only; the test set is touched once for final evaluation.'),
            ha='center', va='bottom', fontsize=8.5, style='italic', color='#333333')
    return fig


def fig_dpformer_arch(n_feat: int = 8, seq_len: int = 10, d_model: int = 128, ff: int = 256, heads: int = 4):
    """Diagram arsitektur DPformer SEPERTI DIIMPLEMENTASIKAN di src/models_dl.py::build_dpformer
    (adaptasi dari Bao et al., 2026): dekomposisi harga -> jalur trend (patch + encoder) dan jalur seasonal
    (conv1d + encoder) -> fusi theta yang dipelajari. MOSSA berada di luar jaringan (mencari lr, wd)."""
    id_ = _lang == 'id'
    fig, ax = plt.subplots(figsize=(14.5, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ct, cs, cf, ci = '#f6dde8', '#dbe7f5', '#fdf0d5', '#eeeeee'
    fs = 7.8
    # input & dekomposisi
    _fbox(ax, 0.005, 0.40, 0.095, 0.20, (f'Input\n{seq_len} hari x {n_feat} kanal\n(return emas + makro)' if id_
                                          else f'Input\n{seq_len} days x {n_feat} channels\n(gold return + macro)'), fc=ci, fs=fs)
    _fbox(ax, 0.125, 0.40, 0.105, 0.20, ('Dekomposisi harga\n(rata-rata bergerak\nWs = 10)' if id_ else 'Price decomposition\n(moving average\nWs = 10)'), fc=ci, fs=fs)
    _farrow(ax, (0.100, 0.50), (0.125, 0.50))
    # dua jalur
    xs0 = 0.265
    bw, bh, gx = 0.097, 0.17, 0.010
    trend = [('Trend + makro\n(kanal makro\ndisinkronkan)' if id_ else 'Trend + macro\n(macro channels\nsynchronised)'),
             ('Patch\npanjang 2, stride 1' if id_ else 'Patch\nlength 2, stride 1'),
             (f'Embedding {d_model}\n+ PE sinusoidal' if id_ else f'Embedding {d_model}\n+ sinusoidal PE'),
             (f'Encoder x1\n{heads} head, FF {ff}' if id_ else f'Encoder x1\n{heads} heads, FF {ff}'),
             ('Flatten\nDense 256 (ReLU)\nDense 1' if id_ else 'Flatten\nDense 256 (ReLU)\nDense 1')]
    seas = [('Seasonal + makro\n(kanal makro\ndisinkronkan)' if id_ else 'Seasonal + macro\n(macro channels\nsynchronised)'),
            ('Conv1D\nkernel 5, 128 filter' if id_ else 'Conv1D\nkernel 5, 128 filters'),
            ('+ PE sinusoidal' if id_ else '+ sinusoidal PE'),
            (f'Encoder x1\n{heads} head, FF {ff}' if id_ else f'Encoder x1\n{heads} heads, FF {ff}'),
            ('Flatten\nDense 256 (ReLU)\nDense 1' if id_ else 'Flatten\nDense 256 (ReLU)\nDense 1')]
    for lane, (items, y0, col) in enumerate(((trend, 0.62, ct), (seas, 0.20, cs))):
        for k, txt in enumerate(items):
            x = xs0 + k * (bw + gx)
            _fbox(ax, x, y0, bw, bh, txt, fc=col, fs=fs)
            if k < len(items) - 1:
                _farrow(ax, (x + bw, y0 + bh / 2), (x + bw + gx, y0 + bh / 2))
        yl = y0 + bh / 2
        _farrow(ax, (0.230, 0.50), (xs0, yl), rad=-0.18 if lane == 0 else 0.18)
    # fusi
    xf = xs0 + 5 * (bw + gx) + 0.004
    _fbox(ax, xf, 0.40, 0.125, 0.20, ('Fusi\ntheta = sigmoid(Dense)\ny = theta*trend\n+ (1-theta)*seasonal' if id_
                                      else 'Fusion\ntheta = sigmoid(Dense)\ny = theta*trend\n+ (1-theta)*seasonal'), fc=cf, fs=fs)
    for y0 in (0.62, 0.20):
        xe = xs0 + 4 * (bw + gx) + bw
        _farrow(ax, (xe, y0 + bh / 2), (xf, 0.50), rad=0.18 if y0 > 0.5 else -0.18)
    _farrow(ax, (xf + 0.125, 0.50), (xf + 0.148, 0.50))
    ax.text(xf + 0.152, 0.50, ('return\nh-hari' if id_ else 'h-day\nreturn'), ha='left', va='center', fontsize=fs + 0.4, fontweight='bold')
    # MOSSA
    _fbox(ax, 0.125, 0.015, 0.80, 0.12, ('MOSSA (di luar jaringan): mencari (learning rate, weight decay) dalam [1e-4, 1e-2] x [0, 1e-2]; '
                                         'populasi 20, hingga 100 iterasi; fitness = MAE validasi ber-embargo (uji tidak dilihat)' if id_
                                         else 'MOSSA (outside the network): searches (learning rate, weight decay) in [1e-4, 1e-2] x [0, 1e-2]; '
                                         'population 20, up to 100 iterations; fitness = embargoed validation MAE (test never seen)'),
          fc='#ffffff', ec='#888888', fs=7.4, lw=0.9)
    ax.text(0.005, 0.96, ('Arsitektur DPformer seperti diimplementasikan di proyek ini (adaptasi dari Bao et al., 2026)' if id_
                          else 'DPformer architecture as implemented in this project (adapted from Bao et al., 2026)'),
            ha='left', va='top', fontsize=10.5, fontweight='bold')
    return fig


# ---------------------------------------------------------------------------
# 1. data / EDA
# ---------------------------------------------------------------------------
def fig_price_regimes(p: pd.Series, bounds: dict, window: int = 60):
    """Harga + volatilitas dengan pembagian latih/validasi/uji.
    bounds = {'val_start': ts, 'test_start': ts}."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    t0, tv, tt, t1 = p.index[0], bounds['val_start'], bounds['test_start'], p.index[-1]
    ax[0].plot(p.index, p.values / 1e6, color='#222', lw=0.9)
    ax[0].set_yscale('log')
    lo_, hi_ = float(np.nanmin(p.values)) / 1e6, float(np.nanmax(p.values)) / 1e6
    ticks = [t for t in (0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4) if lo_ * 0.97 <= t <= hi_ * 1.03]
    ax[0].set_yticks(ticks)
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax[0].yaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylabel(tr('price_m'))
    vol = p.pct_change().rolling(window).std() * 100
    ax[1].plot(vol.index, vol.values, color='#555', lw=0.8)
    for a in ax:
        a.axvspan(t0, tv, color=C_TRAIN, alpha=.12, lw=0)
        a.axvspan(tv, tt, color=C_VAL, alpha=.22, lw=0)
        a.axvspan(tt, t1, color=C_TEST, alpha=.12, lw=0)
    for (a_, b_) in ((t0, tv), (tv, tt), (tt, t1)):
        m = p.pct_change().loc[a_:b_].std() * 100
        ax[1].hlines(m, a_, b_, color='k', lw=1.6, ls='--')
        ax[1].text(a_ + (b_ - a_) / 2, m * 1.08, f'{m:.2f}%', ha='center', fontsize=7.5)
    ax[1].set_ylabel(tr('vol'))
    ax[1].set_xlabel(tr('date'))
    handles = [Patch(color=C_TRAIN, alpha=.3, label=tr('train')), Patch(color=C_VAL, alpha=.4, label=tr('val')),
               Patch(color=C_TEST, alpha=.3, label=tr('test'))]
    ax[0].legend(handles=handles, loc='upper left', ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def fig_zero_share(p: pd.Series):
    r = p.pct_change().dropna()
    s = (r == 0).groupby(r.index.year).mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(s.index, s.values, color='#7a8aa0')
    ax.set_ylabel(tr('zero_share') + ' (%)')
    ax.set_xlabel(tr('year'))
    ax.set_xticks(s.index)
    ax.tick_params(axis='x', rotation=60)
    fig.tight_layout()
    return fig


def fig_leadlag(cc_contemp: pd.DataFrame, cc_pred: pd.DataFrame):
    """cc_contemp: index = etiket baris (lag), kolom = variabel; korelasi dengan return Antam hari t.
    cc_pred: index = ['used','sameday'], kolom = variabel; korelasi dengan return Antam t+1."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.35, 1]})
    a = ax[0]
    im = a.imshow(cc_contemp.values, cmap='RdBu_r', vmin=-0.35, vmax=0.35, aspect='auto')
    a.set_xticks(range(cc_contemp.shape[1]))
    a.set_xticklabels(cc_contemp.columns, rotation=30, ha='right')
    a.set_yticks(range(cc_contemp.shape[0]))
    a.set_yticklabels(cc_contemp.index)
    for i in range(cc_contemp.shape[0]):
        for j in range(cc_contemp.shape[1]):
            v = cc_contemp.values[i, j]
            a.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=7.5,
                   color='white' if abs(v) > 0.2 else 'black')
    a.grid(False)
    a.set_title(tr('lag_ret'))
    _panel(a, 'a')
    b = ax[1]
    x = np.arange(cc_pred.shape[1])
    w = 0.38
    b.bar(x - w / 2, cc_pred.loc['used'].values, w, color='#9aa5b1', label=tr('used'))
    b.bar(x + w / 2, cc_pred.loc['sameday'].values, w, color='#c2185b', label=tr('sameday'))
    b.axhline(0, color='k', lw=.6)
    b.set_xticks(x)
    b.set_xticklabels(cc_pred.columns, rotation=30, ha='right')
    b.set_title(tr('lag_pred'))
    b.legend(frameon=False)
    _panel(b, 'b')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. hasil gaya jurnal (Fig. 4-7 Bao et al.)
# ---------------------------------------------------------------------------
def _order_models(tabel: pd.DataFrame, metric: str = 'MAE') -> list[str]:
    s = tabel.groupby('model')[metric].mean().sort_values()
    return list(s.index)


def fig_results_composite(tabel: pd.DataFrame, curves: dict, drift_row: pd.DataFrame, setup: str,
                          horizons: list[int], curve_h: int, order_by: str = 'MAE'):
    """Gaya Fig. 4 jurnal: (a) kurva prediksi vs aktual, (b) heatmap RMSE, (c) MAE bertumpuk, (d) MAPE.
    tabel: kolom setup, model, horizon, MAE, RMSE, MAPE (tabel_hasil_final, satu setup).
    curves: {'Aktual': Series, nama_model: Series, ...} indeks tanggal target, pada horizon curve_h.
    drift_row: DataFrame (horizon, MAE, RMSE, MAPE) benchmark Drift."""
    t = tabel[tabel.setup == setup]
    models = _order_models(t, order_by)
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    # (a)
    a = fig.add_subplot(gs[0, 0])
    for k, s in curves.items():
        if k == 'Aktual':
            a.plot(s.index, s.values / 1e6, color=C_ACT, lw=2.0, label=tr('actual'))
        elif k in ('drift', 'DRIFT'):
            a.plot(s.index, s.values / 1e6, color='k', lw=1.0, ls='--', label='Drift')
        else:
            a.plot(s.index, s.values / 1e6, color=model_color(k), lw=1.0 if k != 'dpformer' else 1.6, label=label(k))
    a.set_ylabel(tr('price').replace('Rp/gram', 'juta Rp/gram') if _lang == 'id' else 'Antam price (million IDR/gram)')
    a.set_title(f'h = {curve_h}  |  {tr("univ") if setup == "univariat" else tr("multi")}')
    a.legend(ncol=3, frameon=False, loc='upper left')
    a.tick_params(axis='x', rotation=20)
    _panel(a, 'a')
    # (b) heatmap RMSE: teks = RMSE ribu Rp, warna = % terhadap Drift
    b = fig.add_subplot(gs[0, 1])
    piv = t.pivot(index='model', columns='horizon', values='RMSE').reindex(index=models, columns=horizons)
    dr = drift_row.set_index('horizon')['RMSE'].reindex(horizons)
    rel = (piv.div(dr, axis=1) - 1) * 100
    lim = float(np.nanmax(np.abs(rel.values)))
    lim = max(5.0, min(lim, 30.0))
    im = b.imshow(rel.values, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    b.set_xticks(range(len(horizons)))
    b.set_xticklabels([f'h={h}' for h in horizons])
    b.set_yticks(range(len(models)))
    b.set_yticklabels([label(m) for m in models])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                b.text(j, i, f'{v / 1e3:.1f}', ha='center', va='center', fontsize=6.8)
    b.grid(False)
    cb = fig.colorbar(im, ax=b, fraction=0.04, pad=0.02)
    cb.set_label(('RMSE vs Drift (%)  [merah = lebih buruk]' if _lang == 'id' else 'RMSE vs Drift (%)  [red = worse]'), fontsize=7.5)
    b.set_title(tr('rmse_k'))
    _panel(b, 'b')
    # (c) MAE bertumpuk
    c = fig.add_subplot(gs[1, 0])
    pm = t.pivot(index='model', columns='horizon', values='MAE').reindex(index=models, columns=horizons)
    order = pm.sum(axis=1).sort_values().index
    left = np.zeros(len(order))
    cols = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, len(horizons)))
    for k, h in enumerate(horizons):
        vals = pm.loc[order, h].values
        c.barh(range(len(order)), vals, left=left, color=cols[k], label=f'h={h}', height=0.75)
        left += vals
    c.set_yticks(range(len(order)))
    c.set_yticklabels([label(m) for m in order])
    c.invert_yaxis()
    c.set_xlabel(tr('mae') + (', ditumpuk antar-horizon' if _lang == 'id' else ', stacked over horizons'))
    c.legend(ncol=len(horizons), frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    _panel(c, 'c')
    # (d) MAPE
    d = fig.add_subplot(gs[1, 1])
    pp = t.pivot(index='model', columns='horizon', values='MAPE').reindex(index=models, columns=horizons)
    order2 = pp.mean(axis=1).sort_values().index
    x = np.arange(len(order2))
    for k, h in enumerate(horizons):
        d.plot(x, pp.loc[order2, h].values, marker='o', ms=3, lw=1.2, color=cols[k], label=f'h={h}')
        d.axhline(float(drift_row.set_index('horizon').loc[h, 'MAPE']), color=cols[k], lw=.7, ls=':')
    d.set_xticks(x)
    d.set_xticklabels([label(m) for m in order2], rotation=70, ha='right')
    d.set_ylabel(tr('mape'))
    d.legend(ncol=len(horizons), frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
             title=('garis titik-titik = Drift' if _lang == 'id' else 'dotted = Drift'), title_fontsize=7)
    _panel(d, 'd')
    return fig


def fig_improvement(tabel: pd.DataFrame, setup: str, horizons: list[int], ref: str = 'dpformer'):
    """Gaya Fig. 5/Tabel 6 jurnal: (a) MAE per horizon (batang berkelompok); (b) perbaikan % `ref`
    terhadap tiap model lain (positif = ref lebih baik)."""
    t = tabel[tabel.setup == setup]
    models = _order_models(t)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={'width_ratios': [1.35, 1]})
    a = ax[0]
    n = len(models)
    w = 0.8 / n
    pm = t.pivot(index='model', columns='horizon', values='MAE').reindex(index=models, columns=horizons)
    for i, m in enumerate(models):
        a.bar(np.arange(len(horizons)) + i * w - 0.4 + w / 2, pm.loc[m].values / 1e3, width=w, color=model_color(m),
              edgecolor='k' if m == ref else 'none', lw=0.8, label=label(m))
    a.set_xticks(range(len(horizons)))
    a.set_xticklabels([f'h={h}' for h in horizons])
    a.set_ylabel(tr('mae_k'))
    a.legend(ncol=4, frameon=False, fontsize=6.5, loc='upper left')
    _panel(a, 'a')
    b = ax[1]
    others = [m for m in models if m != ref]
    imp = pd.DataFrame({h: [(1 - pm.loc[ref, h] / pm.loc[m, h]) * 100 for m in others] for h in horizons}, index=others)
    lim = max(10.0, float(np.nanmax(np.abs(imp.values))))
    im = b.imshow(imp.values, cmap='RdBu', norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    b.set_xticks(range(len(horizons)))
    b.set_xticklabels([f'h={h}' for h in horizons])
    b.set_yticks(range(len(others)))
    b.set_yticklabels([label(m) for m in others])
    for i in range(imp.shape[0]):
        for j in range(imp.shape[1]):
            b.text(j, i, f'{imp.values[i, j]:.1f}', ha='center', va='center', fontsize=7)
    b.grid(False)
    cb = fig.colorbar(im, ax=b, fraction=0.05, pad=0.02)
    cb.set_label((f'{label(ref)} lebih baik (+) / lebih buruk (-)' if _lang == 'id' else f'{label(ref)} better (+) / worse (-)'), fontsize=7.5)
    b.set_title(tr('improve') + ' - MAE')
    _panel(b, 'b')
    fig.tight_layout()
    return fig, imp


def taylor_diagram(ax, std_ratio: dict, corr: dict, colors: dict, rmax: float | None = None):
    """Diagram Taylor di ruang return. std_ratio = std(prediksi)/std(aktual), corr = korelasi dengan aktual."""
    names = list(std_ratio)
    rmax = rmax or max(1.5, 1.15 * max(std_ratio.values()))
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rlim(0, rmax)
    rc = np.array([-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 0.95, 0.99, 1.0])
    ax.set_thetagrids(np.degrees(np.arccos(rc)), [f'{c:g}' for c in rc], fontsize=7)
    # lingkaran RMS (pusat di titik acuan (1,0))
    th = np.linspace(0, np.pi, 200)
    for rms in (0.5, 1.0, 1.5):
        x = 1 + rms * np.cos(th)
        y = rms * np.sin(th)
        rr = np.hypot(x, y)
        tt = np.arctan2(y, x)
        keep = (rr <= rmax) & (y >= 0)
        ax.plot(tt[keep], rr[keep], color='#2a9d8f', lw=.7, ls='--')
    ax.plot([0], [1], marker='*', ms=13, color=C_ACT, mec='k', mew=.5, zorder=5)
    for m in names:
        c = np.clip(corr[m], -1, 1)
        ax.plot([np.arccos(c)], [std_ratio[m]], 'o', ms=6.5, color=colors[m], mec='k', mew=.4, zorder=4)
    ax.set_ylabel(tr('std'), labelpad=25, fontsize=8)
    ax.grid(alpha=.3)


def fig_panel_h(preds: dict, drift: pd.DataFrame, tabel: pd.DataFrame, setup: str, h: int, ref: str = 'dpformer'):
    """Gaya Fig. 6 jurnal untuk satu horizon:
    (a) scatter aktual vs prediksi `ref` (level harga), (b) idem untuk return, (c) diagram Taylor (return),
    (d) MAE terurut, (e) MAPE terurut. preds = {model: DataFrame(attach(...))}."""
    t = tabel[(tabel.setup == setup) & (tabel.horizon == h)].set_index('model')
    fig = plt.figure(figsize=(14, 9.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.34, width_ratios=[1, 1, 1.05])
    d = preds[ref]
    a = fig.add_subplot(gs[0, 0])
    a.scatter(d.y_true / 1e6, d.y_pred / 1e6, s=6, alpha=.45, color=model_color(ref))
    lo, hi = min(d.y_true.min(), d.y_pred.min()) / 1e6, max(d.y_true.max(), d.y_pred.max()) / 1e6
    a.plot([lo, hi], [lo, hi], 'k--', lw=.8)
    a.set_xlabel(tr('actual') + ' (juta Rp)' if _lang == 'id' else 'Actual (million IDR)')
    a.set_ylabel(tr('pred') + (' (juta Rp)' if _lang == 'id' else ' (million IDR)'))
    a.set_title(f'{label(ref)}  h={h}  [{("level harga" if _lang == "id" else "price level")}]')
    _panel(a, 'a')
    a2 = fig.add_subplot(gs[0, 1])
    a2.scatter(d.r_true * 100, d.r_pred * 100, s=6, alpha=.45, color=model_color(ref))
    lo, hi = min(d.r_true.min(), d.r_pred.min()) * 100, max(d.r_true.max(), d.r_pred.max()) * 100
    a2.plot([lo, hi], [lo, hi], 'k--', lw=.8)
    a2.axhline(0, color='gray', lw=.5)
    a2.axvline(0, color='gray', lw=.5)
    a2.set_xlabel(('Return aktual %d hari (%%)' if _lang == 'id' else 'Actual %d-day return (%%)') % h)
    a2.set_ylabel(('Return prediksi (%)' if _lang == 'id' else 'Predicted return (%)'))
    a2.set_title(f'{label(ref)}  h={h}  [return]')
    _sd = float(np.std(d.r_pred.values))
    _cc = float(np.corrcoef(d.r_true.values, d.r_pred.values)[0, 1]) if _sd > 0 else float('nan')
    a2.text(0.03, 0.97, f'σ(prediksi)/σ(aktual) = {_sd / float(np.std(d.r_true.values)):.2f}\nρ = {_cc:+.2f}'
            if _lang == 'id' else f'σ(pred)/σ(actual) = {_sd / float(np.std(d.r_true.values)):.2f}\nρ = {_cc:+.2f}',
            transform=a2.transAxes, va='top', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbbbbb', alpha=.9))
    _panel(a2, 'b')
    b = fig.add_subplot(gs[0, 2], projection='polar')
    sr, cr, cl = {}, {}, {}
    for m, x in preds.items():
        s = np.std(x.r_pred.values)
        sr[m] = s / np.std(x.r_true.values)
        cr[m] = np.corrcoef(x.r_true.values, x.r_pred.values)[0, 1] if s > 0 else 0.0
        cl[m] = model_color(m)
    taylor_diagram(b, sr, cr, cl)
    b.set_title('Taylor (return)', pad=14)
    _panel(b, 'c')
    h_ = [Line2D([0], [0], marker='o', color='w', markerfacecolor=model_color(m), mec='k', mew=.4, ms=6, label=label(m)) for m in preds]
    b.legend(handles=h_, loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False, fontsize=6.3, ncol=1)
    # (c) MAE
    c = fig.add_subplot(gs[1, 0:2])
    mae = t['MAE'].sort_values()
    c.barh(range(len(mae)), mae.values / 1e3, color=[model_color(m) for m in mae.index])
    c.set_yticks(range(len(mae)))
    c.set_yticklabels([label(m) for m in mae.index])
    c.invert_yaxis()
    if 'drift_MAE' in t.columns:
        c.axvline(float(t['drift_MAE'].iloc[0]) / 1e3, color='k', ls='--', lw=1, label='Drift')
        c.legend(frameon=False)
    c.set_xlabel(tr('mae_k'))
    _panel(c, 'd')
    e = fig.add_subplot(gs[1, 2])
    mp = t['MAPE'].sort_values()
    e.barh(range(len(mp)), mp.values, color=[model_color(m) for m in mp.index])
    e.set_yticks(range(len(mp)))
    e.set_yticklabels([label(m) for m in mp.index], fontsize=7)
    e.invert_yaxis()
    e.set_xlabel(tr('mape'))
    _panel(e, 'e')
    return fig


_EV_KEY = {'Reli terbesar': 'ev_rally', 'Volatilitas tertinggi': 'ev_vol', 'Sideways': 'ev_flat'}


def fig_events(p: pd.Series, test_index, curves_by_event: list, events: list, title: str = ''):
    """Gaya Fig. 7 jurnal: harga + jendela peristiwa (atas) dan zoom prediksi (bawah).
    curves_by_event[i] = {'Aktual': Series, model: Series, ...} untuk events[i]."""
    n = len(events)
    fig = plt.figure(figsize=(5.4 * n + 1.0, 7.2))
    gs = GridSpec(2, n, figure=fig, height_ratios=[1, 1.15], hspace=0.46, wspace=0.16)
    top = fig.add_subplot(gs[0, :])
    s = p.reindex(test_index).dropna()
    top.plot(s.index, s.values / 1e6, color=C_ACT, lw=1.4)
    cols = ['#e45756', '#4c78a8', '#59a14f', '#b07aa1']
    ymax = top.get_ylim()[1]
    for k, ev in enumerate(events):
        top.axvspan(ev['mulai'], ev['akhir'], color=cols[k % 4], alpha=.18)
        mid = ev['mulai'] + (ev['akhir'] - ev['mulai']) / 2
        top.text(mid, ymax, f'({chr(97 + k)})', ha='center', va='top', fontsize=10, fontweight='bold', color=cols[k % 4])
    top.set_ylabel(('Harga Antam (juta Rp/gram)' if _lang == 'id' else 'Antam price (million IDR/gram)'))
    top.set_title(title or tr('events'))
    for k, (ev, cv) in enumerate(zip(events, curves_by_event)):
        ax = fig.add_subplot(gs[1, k])
        for m, ser in cv.items():
            if m == 'Aktual':
                ax.plot(ser.index, ser.values / 1e6, color=C_ACT, lw=2.0, label=tr('actual'))
            elif m in ('drift', 'DRIFT'):
                ax.plot(ser.index, ser.values / 1e6, color='k', lw=1.0, ls='--', label='Drift')
            else:
                ax.plot(ser.index, ser.values / 1e6, color=model_color(m), lw=1.2, label=label(m))
        nm = tr(_EV_KEY[ev['nama']]) if ev['nama'] in _EV_KEY else ev['nama']
        ax.set_title(f"({chr(97 + k)}) {nm}\n{ev['mulai']:%Y-%m-%d} - {ev['akhir']:%Y-%m-%d}", color=cols[k % 4], fontsize=9)
        ax.tick_params(axis='x', rotation=25)
        if k == 0:
            first_ax = ax
            ax.set_ylabel(('juta Rp/gram' if _lang == 'id' else 'million IDR/gram'))
    hh, ll = first_ax.get_legend_handles_labels()
    fig.legend(hh, ll, loc='lower center', ncol=len(ll), frameon=False, bbox_to_anchor=(0.5, -0.045))
    return fig


# ---------------------------------------------------------------------------
# 3. diagnostik pelatihan
# ---------------------------------------------------------------------------
_DIAG_COLOR = {'BELAJAR': '#1e8449', 'TANPA_SINYAL': '#6c757d', 'DI_BAWAH_NOL': '#d68910'}


def _cls_short(kelas: str) -> str:
    return tr('cls_' + kelas).split(' (')[0] if kelas in _DIAG_COLOR else kelas


def fig_loss_grid(histories: dict, diag: pd.DataFrame, nulls: dict, setup: str, h: int, models: list[str]):
    """Loss curve semua model pada satu (setup, h) + garis acuan 'model nol'.
    histories[model] = dict history; diag = loss_diagnosis_table; nulls = dl_target_baselines.
    Skala sumbu-y tetap [0, 1.6] agar antarpanel sebanding (epoch awal yang sangat tinggi terpotong)."""
    n = len(models)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.75 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for k, m in enumerate(models):
        ax = axes.ravel()[k]
        ax.set_visible(True)
        hist = histories.get(m)
        if hist is None:
            ax.text(.5, .5, 'n/a', ha='center')
            continue
        ep = np.arange(1, len(hist['loss']) + 1)
        ax.plot(ep, hist['loss'], color='#4c78a8', lw=1.2, label=tr('trainc'))
        ax.plot(ep, hist['val_loss'], color='#e45756', lw=1.2, label=tr('valc'))
        ax.axhline(nulls['val_mse_null'], color='#e45756', ls=':', lw=1.0, label=tr('null_val'))
        ax.axhline(nulls['train_mse_null'], color='#4c78a8', ls=':', lw=1.0, label=tr('null_tr'))
        be = int(np.argmin(hist['val_loss'])) + 1
        ax.axvline(be, color='k', ls='--', lw=.7, label=tr('best_ep'))
        row = diag[(diag.setup == setup) & (diag.model == m) & (diag.horizon == h)]
        kelas = row.iloc[0].kelas if len(row) else ''
        ovf = bool(row.iloc[0].overfit) if len(row) and 'overfit' in row.columns else False
        ttl = f'{label(m)}: {_cls_short(kelas)}' + (' + overfit' if ovf else '')
        ax.set_title(ttl, fontsize=8, color=_DIAG_COLOR.get(kelas, 'k'))
        ax.set_xlabel(tr('epoch'), fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_ylim(0, 1.6)
        if k % ncol == 0:
            ax.set_ylabel(tr('loss'), fontsize=7)
    hs, ls = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{tr("univ") if setup == "univariat" else tr("multi")}  |  h = {h}', y=1.0, fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


def fig_loss_classes(diag: pd.DataFrame, horizons: list[int], models: list[str]):
    """Ringkasan diagnosis: warna = kelas sinyal (epoch terbaik vs model nol), angka = epoch terbaik,
    bingkai merah = overfit setelah epoch terbaik (bobot epoch terbaik tetap yang dipakai)."""
    from matplotlib.patches import Rectangle
    order = ['BELAJAR', 'TANPA_SINYAL', 'DI_BAWAH_NOL']
    kelas_id = {k: i for i, k in enumerate(order)}
    cmap = plt.matplotlib.colors.ListedColormap([_DIAG_COLOR[k] for k in order])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for k, s in enumerate(('univariat', 'multivariat')):
        mat = np.full((len(models), len(horizons)), np.nan)
        for i, m in enumerate(models):
            for j, h in enumerate(horizons):
                r = diag[(diag.setup == s) & (diag.model == m) & (diag.horizon == h)]
                if len(r):
                    mat[i, j] = kelas_id.get(r.iloc[0].kelas, np.nan)
        ax[k].imshow(mat, cmap=cmap, vmin=-0.5, vmax=2.5, aspect='auto')
        ax[k].set_xticks(range(len(horizons)))
        ax[k].set_xticklabels([f'h={h}' for h in horizons])
        ax[k].set_yticks(range(len(models)))
        ax[k].set_yticklabels([label(m) for m in models])
        ax[k].set_title(tr('univ') if s == 'univariat' else tr('multi'))
        ax[k].grid(False)
        for i in range(len(models)):
            for j in range(len(horizons)):
                r = diag[(diag.setup == s) & (diag.model == models[i]) & (diag.horizon == horizons[j])]
                if len(r):
                    ax[k].text(j, i, str(int(r.iloc[0].best_epoch)), ha='center', va='center', fontsize=7.5, color='white')
                    if bool(r.iloc[0].overfit):
                        ax[k].add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='#c0392b', lw=2.4))
    leg = [Patch(color=_DIAG_COLOR[c], label=tr('cls_' + c)) for c in order]
    leg.append(Patch(fill=False, ec='#c0392b', lw=2, label=tr('overfit')))
    fig.legend(handles=leg, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_mossa(mossa: dict, base_lr: float = 1e-3, base_wd: float = 1e-4):
    """Konvergensi & lanskap pencarian MOSSA. mossa = isi mossa_dpformer.json."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={'width_ratios': [1, 1, 1]})
    colors = {'univariat': '#4c78a8', 'multivariat': '#e45756'}
    for s in ('univariat', 'multivariat'):
        if s not in mossa:
            continue
        h = pd.DataFrame(mossa[s]['history'])
        ax[0].plot(h['iter'], h['best_fit'], marker='o', ms=3, color=colors[s], label=tr('univ') if s == 'univariat' else tr('multi'))
    ax[0].set_xlabel(tr('iter'))
    ax[0].set_ylabel(tr('fit') + ' - ' + tr('best_so_far'))
    ax[0].legend(frameon=False)
    _panel(ax[0], 'a')
    for k, s in enumerate(('univariat', 'multivariat')):
        if s not in mossa or not mossa[s].get('trace'):
            continue
        tr_ = pd.DataFrame(mossa[s]['trace'])
        sc = ax[1 + k].scatter(tr_['lr'], tr_['wd'], c=tr_['fit'], s=14, cmap='viridis_r')
        ax[1 + k].set_xscale('log')
        ax[1 + k].plot([mossa[s]['lr']], [mossa[s]['wd']], '*', ms=15, color='#c2185b', mec='k', label='MOSSA')
        ax[1 + k].plot([base_lr], [base_wd], 's', ms=8, color='w', mec='k', label=('default jurnal' if _lang == 'id' else 'journal default'))
        ax[1 + k].set_xlabel('learning rate')
        ax[1 + k].set_ylabel('weight decay')
        ax[1 + k].set_title(tr('univ') if s == 'univariat' else tr('multi'))
        fig.colorbar(sc, ax=ax[1 + k], fraction=0.046, pad=0.02).set_label(tr('fit'), fontsize=7)
        ax[1 + k].legend(frameon=True, framealpha=.92, fontsize=7, loc='lower left')
        _panel(ax[1 + k], 'bc'[k])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. uji statistik & stabilitas
# ---------------------------------------------------------------------------
def fig_dm_heatmap(res: pd.DataFrame, models: list[str], horizons: list[int], p_col: str = 'p_one_hln',
                   q_col: str = 'q_bh', title: str = '', setups=('univariat', 'multivariat'), setup_titles=None,
                   lim: float = 0.25, bh_note: str | None = None):
    """Rasio loss relatif model/benchmark per (setup) x model x horizon.
    res: kolom setup, model, horizon, rel, p_col, q_col. Warna jenuh pada 1 +- lim (default 25%).
    setups: nilai kolom `setup` yang dibuat panel (default dua setup utama); setup_titles: judul panel."""
    n = len(models)
    fig, ax = plt.subplots(1, len(setups), figsize=(5.75 * len(setups), 0.30 * n + 1.9), sharey=True, squeeze=False)
    ax = ax[0]
    im = None
    for k, s in enumerate(setups):
        t = res[res.setup == s]
        mat = t.pivot(index='model', columns='horizon', values='rel').reindex(index=models, columns=horizons)
        pm = t.pivot(index='model', columns='horizon', values=p_col).reindex(index=models, columns=horizons)
        qm = t.pivot(index='model', columns='horizon', values=q_col).reindex(index=models, columns=horizons)
        im = ax[k].imshow(mat.values, cmap='RdBu_r', norm=TwoSlopeNorm(1.0, 1 - lim, 1 + lim), aspect='auto')
        ax[k].set_xticks(range(len(horizons)))
        ax[k].set_xticklabels([f'h={h}' for h in horizons])
        ax[k].set_yticks(range(n))
        ax[k].set_yticklabels([label(m) for m in models])
        for i in range(n):
            for j in range(len(horizons)):
                v = mat.values[i, j]
                if not np.isfinite(v):
                    continue
                mark = ''
                if np.isfinite(qm.values[i, j]) and qm.values[i, j] < 0.05:
                    mark = '†'
                elif np.isfinite(pm.values[i, j]) and pm.values[i, j] < 0.05:
                    mark = '*'
                ax[k].text(j, i, f'{v:.3f}{mark}', ha='center', va='center', fontsize=7)
        _ttl = (setup_titles or {}).get(s) or (tr('univ') if s == 'univariat' else tr('multi') if s == 'multivariat' else s)
        ax[k].set_title(_ttl)
        ax[k].grid(False)
    fig.subplots_adjust(left=0.13 if len(setups) == 2 else 0.20, right=0.90, top=0.93 if not title else 0.90,
                        bottom=0.09, wspace=0.05)
    cax = fig.add_axes([0.92, 0.22, 0.016, 0.56])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(tr('rel_mse') + ('  [<1: model lebih baik]' if _lang == 'id' else '  [<1: model better]'), fontsize=7.5)
    fig.text(0.5, 0.005, f"* {tr('sig')}    † {tr('sig_bh')} ({bh_note if bh_note is not None else tr('bh_fam')})", ha='center', fontsize=8)
    if title:
        fig.suptitle(title, y=0.98)
    return fig


def fig_mcs(incl: pd.DataFrame, pvals: pd.DataFrame, models: list[str] | None = None):
    """incl: DataFrame bool (model x kolom), pvals: p-MCS. Menampilkan keanggotaan MCS 90%
    (hijau = di dalam himpunan; angka = p-MCS). Baris diurutkan menurut rata-rata p-MCS."""
    if models is None:
        models = list(pvals.mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(0.95 * incl.shape[1] + 3.0, 0.31 * len(models) + 1.7))
    mat = incl.reindex(models).astype(float).values
    ax.imshow(mat, cmap=plt.matplotlib.colors.ListedColormap(['#eceff1', '#2a9d8f']), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(incl.shape[1]))
    ax.set_xticklabels(incl.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([label(m) for m in models])
    pm = pvals.reindex(models)
    for i in range(len(models)):
        for j in range(incl.shape[1]):
            ax.text(j, i, f'{pm.values[i, j]:.2f}', ha='center', va='center', fontsize=6.8,
                    color='white' if mat[i, j] else '#444')
    ax.grid(False)
    ax.set_title(tr('inmcs') + ' (' + ('angka = p-MCS' if _lang == 'id' else 'number = MCS p-value') + ')')
    ax.legend(handles=[Patch(fc='#2a9d8f', label=tr('in_set')), Patch(fc='#eceff1', ec='#999999', label=tr('out_set'))],
              loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def fig_block_heatmap(tbl: pd.DataFrame, models: list[str], blocks: list[int], setups=('univariat', 'multivariat'),
                      h: int = 20, col_key: str = 'blok', lim: float = 0.35):
    """tbl: kolom setup, model, horizon, blok, rel_mse. Heatmap model x blok untuk satu horizon.
    col_key='fold' dipakai untuk walk-forward (kolom `blok` berisi nomor fold)."""
    fig, ax = plt.subplots(1, len(setups), figsize=(5.6 * len(setups), 0.34 * len(models) + 1.6), sharey=True)
    ax = np.atleast_1d(ax)
    for k, s in enumerate(setups):
        t = tbl[(tbl.setup == s) & (tbl.horizon == h)]
        mat = t.pivot(index='model', columns='blok', values='rel_mse').reindex(index=models, columns=blocks)
        im = ax[k].imshow(mat.values, cmap='RdBu_r', norm=TwoSlopeNorm(1.0, 1 - lim, 1 + lim), aspect='auto')
        ax[k].set_xticks(range(len(blocks)))
        ax[k].set_xticklabels([f'{tr(col_key)} {b}' for b in blocks], rotation=20)
        ax[k].set_yticks(range(len(models)))
        ax[k].set_yticklabels([label(m) for m in models])
        for i in range(len(models)):
            for j in range(len(blocks)):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax[k].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
        ax[k].set_title((tr('univ') if s == 'univariat' else tr('multi') if s == 'multivariat' else s) + f'  h={h}')
        ax[k].grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(tr('rel_mse'), fontsize=7.5)
    return fig


def fig_regime_bars(tbl: pd.DataFrame, setups=('univariat', 'multivariat'), horizons=(5, 20, 30)):
    """Median rasio MSE relatif (atas semua model) per rezim volatilitas, per horizon.
    Batang dimulai dari garis 1,0 (= sama dengan Drift): ke atas = lebih buruk dari Drift."""
    fig, ax = plt.subplots(1, len(setups), figsize=(5.4 * len(setups), 3.7), sharey=True)
    ax = np.atleast_1d(ax)
    names = [tr('calm'), tr('mid'), tr('high')]
    w = 0.8 / len(horizons)
    cols = ['#4c78a8', '#f58518', '#54a24b', '#b279a2']
    allv = []
    for k, s in enumerate(setups):
        for j, h in enumerate(horizons):
            t = tbl[(tbl.setup == s) & (tbl.horizon == h)].groupby('rezim').rel_mse.median().reindex([0, 1, 2])
            allv += list(t.values)
            ax[k].bar(np.arange(3) + j * w - 0.4 + w / 2, t.values - 1.0, bottom=1.0, width=w, color=cols[j % 4], label=f'h={h}')
        ax[k].axhline(1.0, color='k', lw=.9)
        ax[k].set_xticks(range(3))
        ax[k].set_xticklabels(names)
        ax[k].set_xlabel(tr('regime'))
        ax[k].set_title(tr('univ') if s == 'univariat' else tr('multi'))
    v = np.array([x for x in allv if np.isfinite(x)])
    pad = max(0.03, 0.15 * float(v.max() - v.min()))
    ax[0].set_ylim(min(v.min(), 1.0) - pad, max(v.max(), 1.0) + pad)
    ax[0].set_ylabel(tr('rel_mse') + (' (median 21 model)' if _lang == 'id' else ' (median of 21 models)'))
    ax[0].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    return fig


def fig_gain_vs_horizon(tabel: pd.DataFrame, horizons: list[int], highlight: list[str] | None = None,
                        setups=('univariat', 'multivariat'), setup_titles: dict | None = None):
    """Perbaikan MAE terhadap Drift (%, positif = lebih baik dari Drift) per horizon.
    Garis tipis abu-abu = 21 model, berwarna = model sorotan, hitam tebal = median seluruh model.
    tabel: kolom setup, model, horizon, pct_vs_drift."""
    highlight = highlight or []
    fig, ax = plt.subplots(1, len(setups), figsize=(5.6 * len(setups), 3.9), sharey=True, squeeze=False)
    ax = ax[0]
    x = np.arange(len(horizons))
    for k, s in enumerate(setups):
        t = tabel[tabel.setup == s].pivot(index='model', columns='horizon', values='pct_vs_drift').reindex(columns=horizons)
        for m in t.index:
            if m not in highlight:
                ax[k].plot(x, t.loc[m].values, color='#b8b8b8', lw=.8, zorder=1)
        for m in highlight:
            if m in t.index:
                ax[k].plot(x, t.loc[m].values, marker='o', ms=3.5, lw=1.8, color=model_color(m), label=label(m), zorder=3)
        ax[k].plot(x, t.median(axis=0).values, color='k', lw=2.4, label=('median seluruh model' if _lang == 'id' else 'median of all models'), zorder=4)
        ax[k].axhline(0, color='k', lw=.8, ls='--', zorder=2)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([f'h={h}' for h in horizons])
        ax[k].set_title((setup_titles or {}).get(s) or (tr('univ') if s == 'univariat' else tr('multi') if s == 'multivariat' else s))
        _panel(ax[k], 'abc'[k % 3])
    ax[0].set_ylabel(('MAE lebih baik dari Drift (%)' if _lang == 'id' else 'MAE improvement over Drift (%)'))
    hs, ls = ax[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center', ncol=len(ls), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_ablation(tbl: pd.DataFrame, models: list[str], horizons: list[int], groups: list[str], group_names: dict | None = None):
    """Ablation kelompok fitur: rasio MSE-return terhadap Drift (<1 = lebih baik dari Drift), satu panel per model.
    tbl: kolom model, horizon, grup, rel_mse_ret. Batang dimulai dari garis 1,0."""
    fig, ax = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.7), sharey=True, squeeze=False)
    ax = ax[0]
    cols = plt.get_cmap('viridis')(np.linspace(0.12, 0.88, len(groups)))
    w = 0.8 / len(groups)
    vals = []
    for k, m in enumerate(models):
        for i, g in enumerate(groups):
            t = tbl[(tbl.model == m) & (tbl.grup == g)].set_index('horizon').rel_mse_ret.reindex(horizons)
            vals += [v for v in t.values if np.isfinite(v)]
            ax[k].bar(np.arange(len(horizons)) + i * w - 0.4 + w / 2, t.values - 1.0, bottom=1.0, width=w, color=cols[i],
                      label=(group_names or {}).get(g, g))
        ax[k].axhline(1.0, color='k', lw=.9)
        ax[k].set_xticks(range(len(horizons)))
        ax[k].set_xticklabels([f'h={h}' for h in horizons])
        ax[k].set_title(label(m))
        _panel(ax[k], 'abcdef'[k % 6])
    v = np.array(vals)
    pad = max(0.01, 0.15 * float(v.max() - v.min())) if len(v) else 0.02
    ax[0].set_ylim(min(v.min(), 1.0) - pad, max(v.max(), 1.0) + pad)
    ax[0].set_ylabel(tr('rel_mse'))
    hs, ls = ax[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center', ncol=len(ls), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_ml_importance(imp: pd.DataFrame, title: str = ''):
    """Porsi pentingnya fitur per VARIABEL untuk model ML (importance bawaan model, dinormalkan per model).
    imp: baris = model, kolom = variabel (emas, dxy, ...), nilai = porsi (jumlah 1). Skala importance bawaan
    (gain / impurity / |koefisien|) berbeda antarfamili model: bandingkan POLA, bukan angka mutlak."""
    from .explain import GROUP_LABEL
    order = list(imp.sum(axis=0).sort_values(ascending=False).index)
    imp = imp[order]
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(imp) + 1.8))
    left = np.zeros(len(imp))
    for i, g in enumerate(order):
        v = imp[g].values * 100
        ax.barh(range(len(imp)), v, left=left, color=_PIE_COLORS[i % len(_PIE_COLORS)], label=GROUP_LABEL.get(g, g), height=0.72)
        left += v
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([label(m) for m in imp.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(('porsi importance bawaan model (%)' if _lang == 'id' else 'share of built-in importance (%)'))
    if title:
        ax.set_title(title)
    ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16), fontsize=7)
    fig.tight_layout()
    return fig


def fig_rolling_rel(series: dict, vol: pd.Series | None = None, window: int = 120):
    """series = {nama: DataFrame(attach)}; Drift diambil dari kolom 'r_drift' yang disisipkan pemanggil."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for m, d in series.items():
        e_m = (d.r_true - d.r_pred) ** 2
        e_d = (d.r_true - d.r_drift) ** 2
        rel = e_m.rolling(window).mean() / e_d.rolling(window).mean()
        ax.plot(d.index, rel.values, color=model_color(m), lw=1.3, label=label(m), zorder=3)
    ax.axhline(1.0, color='k', lw=.8, ls='--', zorder=2)
    ax.set_ylabel(tr('rolling', w=window))
    ax.legend(frameon=False, ncol=3, loc='upper right')
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    if vol is not None:
        a2 = ax.twinx()
        v = vol.reindex(next(iter(series.values())).index, method='ffill')
        a2.fill_between(v.index, 0, v.values * 100, color='gray', alpha=.17, lw=0)
        a2.set_ylim(0, float(np.nanmax(v.values * 100)) * 1.15)
        a2.set_ylabel(tr('vol').replace('60', '20'))
        a2.grid(False)
        a2.set_zorder(1)
    fig.tight_layout()
    return fig


def fig_resid_diag(series: dict, nlags: int = 40, h: int | None = None):
    """Diagnostik galat return model terpilih: histogram, QQ, ACF. series = {nama: DataFrame(attach)}.
    Galat peramalan h-langkah tumpang-tindih -> ACF wajar tinggi sampai lag h-1 (MA(h-1) menurut
    konstruksi); pita +-1,96/sqrt(n) hanya berlaku untuk lag >= h. Area lag < h diarsir."""
    from scipy import stats as st
    n = len(series)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.9 * n), squeeze=False)
    for i, (m, d) in enumerate(series.items()):
        e = (d.r_true - d.r_pred).values
        axes[i, 0].hist(e * 100, bins=50, color=model_color(m), alpha=.85)
        axes[i, 0].set_xlabel(tr('err_ret'))
        axes[i, 0].set_title(label(m))
        (osm, osr), (slope, icpt, _r) = st.probplot(e, dist='norm', fit=True)
        axes[i, 1].plot(osm, osr * 100, '.', ms=3, color=model_color(m))
        axes[i, 1].plot(osm, (slope * osm + icpt) * 100, color='k', lw=1)
        axes[i, 1].set_xlabel(('kuantil normal' if _lang == 'id' else 'normal quantiles'))
        axes[i, 1].set_ylabel(tr('err_ret'))
        axes[i, 1].set_title('QQ')
        x = e - e.mean()
        ac = [1.0] + [float(np.dot(x[k:], x[:-k]) / np.dot(x, x)) for k in range(1, nlags + 1)]
        axes[i, 2].bar(range(nlags + 1), ac, color='#555', width=.8)
        ci = 1.96 / np.sqrt(len(e))
        axes[i, 2].axhline(ci, color='r', ls=':', lw=.8)
        axes[i, 2].axhline(-ci, color='r', ls=':', lw=.8)
        if h:
            axes[i, 2].axvspan(-0.5, h - 0.5, color='#f2c14e', alpha=.18, lw=0, label=tr('overlap'))
            if i == 0:
                axes[i, 2].legend(frameon=False, fontsize=7)
        axes[i, 2].set_title('ACF')
        axes[i, 2].set_xlabel('lag')
    fig.tight_layout()
    return fig


def fig_seed_box(df: pd.DataFrame, horizons: list[int], models: list[str], setup: str):
    """df: kolom model, horizon, seed, MAE (termasuk seed utama 42). Box + titik per seed, dinormalkan
    ke rata-rata per (model, horizon)."""
    fig, ax = plt.subplots(1, len(horizons), figsize=(2.9 * len(horizons), 3.6), sharey=True)
    ax = np.atleast_1d(ax)
    for k, h in enumerate(horizons):
        data, pos = [], []
        for i, m in enumerate(models):
            v = df[(df.model == m) & (df.horizon == h)].MAE.values
            if len(v):
                data.append(v / v.mean())
                pos.append(i)
        ax[k].boxplot(data, positions=pos, widths=.55, showfliers=False)
        for p_, v in zip(pos, data):
            ax[k].scatter(np.full(len(v), p_) + np.random.default_rng(0).uniform(-.12, .12, len(v)), v, s=9, color='#c2185b', zorder=3)
        ax[k].set_xticks(range(len(models)))
        ax[k].set_xticklabels([label(m) for m in models], rotation=60, ha='right')
        ax[k].set_title(f'h={h}')
        ax[k].axhline(1.0, color='k', lw=.6, ls=':')
    ax[0].set_ylabel('MAE / ' + ('rata-rata seed' if _lang == 'id' else 'seed mean'))
    fig.suptitle(tr('univ') if setup == 'univariat' else tr('multi'), y=1.02)
    fig.tight_layout()
    return fig


def fig_bars_by_horizon(df: pd.DataFrame, value: str, horizons: list[int], groups: list[str], group_col: str,
                        ylabel: str, title: str = '', colors: dict | None = None, ref: float | None = 0.0):
    """Batang berkelompok sederhana: sumbu-x horizon, batang = nilai `value` per grup (mis. skenario/ablation)."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        t = df[df[group_col] == g].set_index('horizon')[value].reindex(horizons)
        ax.bar(np.arange(len(horizons)) + i * w - 0.4 + w / 2, t.values, width=w, label=str(g),
               color=(colors or {}).get(g))
    if ref is not None:
        ax.axhline(ref, color='k', lw=.8)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f'h={h}' for h in horizons])
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=min(4, len(groups)))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. SHAP (gaya Fig. 8-9 jurnal): bar + beeswarm + pie, waterfall, perbandingan antarmodel
# ---------------------------------------------------------------------------
_SHAP_GROUP_LABEL = {'emas': 'Antam (return, lag & rolling)' if True else '', 'dxy': 'DXY', 'sp500': 'S&P 500',
                     'usd_idr': 'USD/IDR', 'ihsg': 'IHSG', 'us10yr': 'US10Y', 'ffr': 'FFR', 'bi_rate': 'BI-Rate'}
_PIE_COLORS = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2', '#ff9da6', '#9d755d']


def _glabel(g: str) -> str:
    if g == 'emas':
        return 'Antam (return, lag & rolling)' if _lang == 'id' else 'Antam (return, lags & rolling)'
    return _SHAP_GROUP_LABEL.get(g, g)


def _beeswarm_offsets(x: np.ndarray, width: float = 0.38, nbins: int = 25) -> np.ndarray:
    """Jitter vertikal berbasis kepadatan lokal (mirip beeswarm SHAP), deterministik."""
    if len(x) == 0:
        return x
    rng = np.random.default_rng(0)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-15:
        return rng.uniform(-width, width, len(x))
    bins = np.minimum(((x - lo) / (hi - lo) * nbins).astype(int), nbins - 1)
    dens = np.bincount(bins, minlength=nbins).astype(float)
    dens = dens / dens.max()
    return rng.uniform(-1, 1, len(x)) * width * dens[bins]


def fig_shap_summary(res: dict, title: str = ''):
    """(a) bar rata-rata |SHAP| per variabel, (b) beeswarm (warna = nilai variabel: biru rendah, merah tinggi),
    (c) pie porsi pentingnya. res = keluaran src.explain (values dalam satuan return; ditampilkan dalam poin persen)."""
    V = res['values'] * 100
    X = res['X']
    groups = list(res['groups'])
    imp = np.abs(V).mean(axis=0)
    order = list(np.argsort(-imp))
    fig = plt.figure(figsize=(15, 4.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.55, 0.95], wspace=0.55)
    a = fig.add_subplot(gs[0, 0])
    yy = np.arange(len(order))
    a.barh(yy, imp[order], color='#4c78a8')
    a.set_yticks(yy)
    a.set_yticklabels([_glabel(groups[i]) for i in order])
    a.invert_yaxis()
    for y_, i in zip(yy, order):
        a.text(imp[i], y_, f' {imp[i]:.3f}', va='center', fontsize=7.5)
    a.set_xlabel(('rata-rata |SHAP| (poin persen return)' if _lang == 'id' else 'mean |SHAP| (percentage points of return)'))
    a.set_xlim(0, imp.max() * 1.22)
    _panel(a, 'a')
    b = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap('coolwarm')
    sc = None
    for r, i in enumerate(order):
        x = V[:, i]
        xv = X[:, i]
        pr = pd.Series(xv).rank(pct=True).values if np.std(xv) > 0 else np.full(len(xv), 0.5)
        sc = b.scatter(x, r + _beeswarm_offsets(x), c=pr, cmap=cmap, vmin=0, vmax=1, s=11, alpha=.85, linewidths=0)
    b.axvline(0, color='k', lw=.6)
    b.set_yticks(yy)
    b.set_yticklabels([_glabel(groups[i]) for i in order])
    b.invert_yaxis()
    b.set_xlabel(('nilai SHAP (poin persen return %d-hari)' if _lang == 'id' else 'SHAP value (percentage points of %d-day return)') % res.get('h', 0) if res.get('h') else tr('shap_val'))
    cb = fig.colorbar(sc, ax=b, fraction=0.04, pad=0.02)
    cb.set_ticks([0, 1])
    cb.set_ticklabels([tr('low'), tr('high_')])
    cb.set_label(tr('feat_val'), fontsize=8)
    _panel(b, 'b')
    c = fig.add_subplot(gs[0, 2])
    share = imp[order] / imp.sum() * 100
    cols = [_PIE_COLORS[k % len(_PIE_COLORS)] for k in range(len(order))]
    wedges, _t = c.pie(share, colors=cols, startangle=90, counterclock=False,
                       wedgeprops=dict(edgecolor='white', linewidth=1.0))
    for w, sh in zip(wedges, share):
        if sh >= 4:
            ang = np.deg2rad((w.theta1 + w.theta2) / 2)
            c.text(0.62 * np.cos(ang), 0.62 * np.sin(ang), f'{sh:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    c.legend(wedges, [f'{_glabel(groups[i])} ({s_:.1f}%)' for i, s_ in zip(order, share)], loc='center left',
             bbox_to_anchor=(0.92, 0.5), frameon=False, fontsize=7)
    c.set_title(('porsi pentingnya' if _lang == 'id' else 'importance share'), fontsize=9)
    _panel(c, 'c')
    if title:
        fig.suptitle(title, y=1.02, fontsize=11)
    return fig


def fig_shap_waterfall(res: dict, idx: int, title: str = '', actual: float | None = None):
    """Gaya Fig. 9 jurnal: satu sampel uji. Dasar = rata-rata keluaran pada background (TRAIN);
    batang = kontribusi tiap variabel (merah menaikkan, biru menurunkan prediksi return); atas = prediksi."""
    v = res['values'][idx] * 100
    groups = list(res['groups'])
    base = res['base'] * 100
    pred = res['pred'][idx] * 100
    order = list(np.argsort(np.abs(v)))               # kecil dulu (dasar) -> besar (atas)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    cum = base
    ypos = np.arange(len(order))
    for y_, i in zip(ypos, order):
        left = cum
        cum = cum + v[i]
        ax.barh(y_, v[i], left=left, color='#d62728' if v[i] >= 0 else '#1f77b4', height=0.62)
        ax.text(max(left, cum) + 0.01 * (abs(pred - base) + 0.05), y_, f'{v[i]:+.3f}', va='center', fontsize=7.5)
    ax.axvline(base, color='gray', lw=.8, ls=':')
    ax.axvline(pred, color='k', lw=.8, ls='--')
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{_glabel(groups[i])}  [{res['X'][idx, i]:+.2g}]" for i in order], fontsize=8)
    lo, hi = min(base, pred, cum), max(base, pred)
    pad = 0.18 * max(1e-3, abs(v).sum())
    ax.set_xlim(min(lo, base + np.minimum.accumulate(np.r_[0, v[order]]).min()) - pad * 0.3, max(hi, base + np.maximum.accumulate(np.r_[0, v[order]]).max()) + pad)
    ax.set_xlabel(('prediksi return %d-hari (%%)' if _lang == 'id' else 'predicted %d-day return (%%)') % res['h'] if res.get('h') else '%')
    ttl = f"{tr('base')} = {base:+.3f}%   ->   {tr('final')} = {pred:+.3f}%"
    if actual is not None:
        ttl += f"   |   {tr('actual').lower()} = {actual * 100:+.2f}%"
    ax.set_title((title + '\n' if title else '') + f"{res['dates'][idx]:%Y-%m-%d}   " + ttl, fontsize=9)
    fig.tight_layout()
    return fig


def fig_shap_compare(tables: dict, labels: dict | None = None):
    """Porsi pentingnya (%) per variabel untuk beberapa model (batang berkelompok) - perbandingan lintas keluarga."""
    names = list(tables)
    groups = list(tables[names[0]].variabel)
    order = list(pd.concat([t.set_index('variabel').porsi_pct.reindex(groups) for t in tables.values()], axis=1).mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    w = 0.8 / len(names)
    cols = ['#c2185b', '#f58518', '#4c78a8', '#54a24b']
    for k, n in enumerate(names):
        s = tables[n].set_index('variabel').porsi_pct.reindex(order)
        ax.bar(np.arange(len(order)) + k * w - 0.4 + w / 2, s.values, width=w, color=cols[k % 4],
               label=(labels or {}).get(n, label(n)))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_glabel(g).replace(' (return, lag & rolling)', '').replace(' (return, lags & rolling)', '') for g in order], rotation=25, ha='right')
    ax.set_ylabel(('porsi |SHAP| (%)' if _lang == 'id' else 'share of |SHAP| (%)'))
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
