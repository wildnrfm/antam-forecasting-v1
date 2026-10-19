"""Uji statistik untuk membandingkan forecast (dipakai notebook 10, 12, 15, 16).

Isi:
- loss_series       : deret loss per hari (ruang return = bebas skala, atau ruang harga)
- dm_test           : Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold (HLN)
- bh_fdr            : Benjamini-Hochberg (FDR)
- mcs               : Model Confidence Set (Hansen, Lunde & Nason, 2011), bootstrap blok
- pesaran_timmermann: uji akurasi arah (Pesaran & Timmermann, 1992)
- oos_r2            : R^2 out-of-sample terhadap benchmark (1 - MSE_model / MSE_benchmark)

Konvensi tanda: statistik DM < 0 berarti model A LEBIH BAIK daripada model B
(loss A - loss B negatif). p_one = P(model A lebih baik) satu-sisi.

Catatan metodologis (penting untuk laporan):
* Forecast h-langkah dengan origin harian saling tumpang-tindih, sehingga selisih loss
  berautokorelasi sampai lag h-1. Varians jangka panjang dihitung dengan kernel
  PERSEGI (rectangular) sampai lag h-1 = bentuk asli DM (1995) yang dipakai HLN (1997).
  Kernel Bartlett (Newey-West) menyusutkan autokovarians sehingga varians cenderung
  diremehkan untuk proses MA(h-1) -> uji terlalu mudah menolak H0. Bartlett tetap
  tersedia sebagai pembanding (kernel='bartlett').
* Loss di ruang HARGA (rupiah^2) tidak stasioner skala: harga emas naik ~3x di periode
  uji, sehingga selisih loss didominasi 2025-2026. Loss di ruang RETURN
  (R_h = harga(i+h)/harga(i) - 1) bebas skala; ini yang dijadikan acuan utama
  (sama dengan target latih model). Ruang harga disediakan sebagai pembanding.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def to_returns(y_true, y_pred, p0):
    """(harga aktual, harga prediksi, harga di origin) -> (R_h aktual, R_h prediksi)."""
    y_true, y_pred, p0 = (np.asarray(x, float) for x in (y_true, y_pred, p0))
    return y_true / p0 - 1.0, y_pred / p0 - 1.0


def loss_series(y_true, y_pred, p0=None, space: str = 'return', kind: str = 'sq'):
    """Deret loss per observasi.

    space: 'return' (bebas skala; butuh p0) | 'price' (rupiah)
    kind : 'sq' (kuadrat) | 'abs' (absolut)
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if space == 'return':
        if p0 is None:
            raise ValueError("space='return' membutuhkan p0 (harga di origin)")
        a, f = to_returns(y_true, y_pred, p0)
        e = a - f
    elif space == 'price':
        e = y_true - y_pred
    else:
        raise ValueError(f'space tidak dikenal: {space!r}')
    if kind == 'sq':
        return e ** 2
    if kind == 'abs':
        return np.abs(e)
    raise ValueError(f'kind tidak dikenal: {kind!r}')


# ---------------------------------------------------------------------------
# Diebold-Mariano (+ HLN)
# ---------------------------------------------------------------------------
def _lrv(d: np.ndarray, lags: int, kernel: str) -> float:
    n = len(d)
    u = d - d.mean()
    lrv = float(np.dot(u, u) / n)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0) if kernel == 'bartlett' else 1.0
        lrv += 2.0 * w * float(np.dot(u[k:], u[:-k]) / n)
    return lrv


def dm_test(d, h: int, kernel: str = 'rectangular') -> dict:
    """Uji DM atas deret selisih loss d_t = L_A,t - L_B,t (H1 satu-sisi: A lebih baik).

    Mengembalikan stat (DM mentah), stat_hln (setelah koreksi sampel kecil HLN),
    p_one / p_two (normal) dan p_one_hln / p_two_hln (t dengan n-1 df; DIREKOMENDASIKAN),
    plus dbar, lrv, n, dan flag lrv_fallback bila varians jangka panjang <= 0
    (kernel persegi dapat bernilai negatif pada sampel hingga) sehingga dipakai Bartlett.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {'n': n, 'dbar': np.nan, 'lrv': np.nan, 'stat': np.nan, 'stat_hln': np.nan,
           'p_one': np.nan, 'p_two': np.nan, 'p_one_hln': np.nan, 'p_two_hln': np.nan,
           'lrv_fallback': False}
    if n < max(10, 2 * h):
        return out
    lags = min(max(h - 1, 0), n - 1)
    lrv = _lrv(d, lags, kernel)
    fb = False
    if lrv <= 1e-18 and kernel != 'bartlett':
        lrv = _lrv(d, lags, 'bartlett')
        fb = True
    dbar = float(d.mean())
    if lrv <= 1e-18:                       # d konstan (mis. dua forecast identik)
        out.update(dbar=dbar, lrv=lrv, stat=0.0, stat_hln=0.0, p_one=0.5, p_two=1.0,
                   p_one_hln=0.5, p_two_hln=1.0, lrv_fallback=fb)
        return out
    stat = dbar / np.sqrt(lrv / n)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    sh = stat * hln
    out.update(
        dbar=dbar, lrv=lrv, stat=float(stat), stat_hln=float(sh), lrv_fallback=fb,
        p_one=float(_st.norm.cdf(stat)), p_two=float(2 * _st.norm.sf(abs(stat))),
        p_one_hln=float(_st.t.cdf(sh, df=n - 1)), p_two_hln=float(2 * _st.t.sf(abs(sh), df=n - 1)))
    return out


def dm_compare(y_true, pred_a, pred_b, p0, h, space='return', kind='sq', kernel='rectangular') -> dict:
    """Bandingkan dua forecast harga: A vs B (A lebih baik bila stat < 0)."""
    la = loss_series(y_true, pred_a, p0, space, kind)
    lb = loss_series(y_true, pred_b, p0, space, kind)
    r = dm_test(la - lb, h, kernel)
    r['rel_loss'] = float(la.mean() / lb.mean()) if lb.mean() > 0 else np.nan
    return r


# ---------------------------------------------------------------------------
# multiple testing
# ---------------------------------------------------------------------------
def bh_fdr(pvals, alpha: float = 0.05):
    """Benjamini-Hochberg. Mengembalikan (q-value, ditolak). NaN dibiarkan NaN/False."""
    p = np.asarray(pvals, float)
    q = np.full(p.shape, np.nan)
    rej = np.zeros(p.shape, bool)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q, rej
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1.0)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    tmp = np.empty(m)
    tmp[order] = qv
    q[ok] = tmp
    rej[ok] = tmp < alpha
    return q, rej


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------
def _block_indices(T: int, block: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Indeks bootstrap blok bergerak (circular), bentuk (B, T)."""
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    return idx.reshape(B, -1)[:, :T]


def mcs(losses, names=None, alpha: float = 0.10, B: int = 2000, block: int | None = None,
        seed: int = 0, statistic: str = 'Tmax') -> dict:
    """Model Confidence Set (Hansen, Lunde & Nason, 2011).

    losses: array (T, M) loss per waktu per model (makin kecil makin baik).
    block : panjang blok bootstrap (default max(h, 10) dipilih pemanggil).
    statistic: 'Tmax' (default) atau 'TR' (range).
    Mengembalikan dict: included (nama model di MCS 1-alpha), pvalues (p-MCS tiap model),
    order (urutan eliminasi, model terburuk dulu).
    """
    L = np.asarray(losses, float)
    T, M = L.shape
    names = list(names) if names is not None else [f'm{i}' for i in range(M)]
    if len(names) != M:
        raise ValueError('names harus sepanjang jumlah kolom losses')
    block = int(block or max(10, int(round(T ** (1 / 3)))))
    rng = np.random.default_rng(seed)
    idx = _block_indices(T, block, B, rng)                       # (B, T)
    Lb_mean = np.empty((B, M))                                   # (B, M) rata-rata loss bootstrap
    for c0 in range(0, B, 250):                                  # per potongan: hemat memori (B*T*M bisa ratusan MB)
        Lb_mean[c0:c0 + 250] = L[idx[c0:c0 + 250]].mean(axis=1)
    L_mean = L.mean(axis=0)

    alive = list(range(M))
    elim, pvals_step = [], []
    while len(alive) > 1:                       # eliminasi sampai tersisa satu model (konvensi MCS)
        a = np.array(alive)
        m = len(a)
        # d_ij rata-rata (m x m) dan versi bootstrap (B x m x m)
        dbar = L_mean[a][:, None] - L_mean[a][None, :]
        dbar_i = dbar.sum(axis=1) / m                            # d_i.
        Lbm = Lb_mean[:, a]                                      # (B, m)
        db_b = Lbm[:, :, None] - Lbm[:, None, :]                 # (B, m, m)
        db_i_b = db_b.sum(axis=2) / m                            # (B, m)
        var_i = ((db_i_b - dbar_i[None, :]) ** 2).mean(axis=0)   # varians bootstrap d_i.
        var_i = np.maximum(var_i, 1e-30)
        if statistic == 'Tmax':
            t_i = dbar_i / np.sqrt(var_i)
            tb_i = (db_i_b - dbar_i[None, :]) / np.sqrt(var_i)[None, :]
            T_obs = t_i.max()
            T_boot = tb_i.max(axis=1)
            worst_local = int(np.argmax(t_i))
        elif statistic == 'TR':
            vij = np.maximum(((db_b - dbar[None, :, :]) ** 2).mean(axis=0), 1e-30)
            tij = np.abs(dbar) / np.sqrt(vij)
            np.fill_diagonal(tij, 0.0)
            T_obs = tij.max()
            tb = np.abs((db_b - dbar[None, :, :]) / np.sqrt(vij)[None, :, :])
            idiag = np.arange(m)
            tb[:, idiag, idiag] = 0.0
            T_boot = tb.reshape(B, -1).max(axis=1)
            worst_local = int(np.argmax(dbar_i))                 # model dengan rata-rata loss terburuk
        else:
            raise ValueError(f'statistic tidak dikenal: {statistic!r}')
        pvals_step.append(float((T_boot >= T_obs).mean()))
        elim.append(alive.pop(worst_local))
    # p-MCS: maksimum kumulatif dari langkah pertama (monoton); model terakhir = 1
    pm = np.maximum.accumulate(pvals_step) if pvals_step else np.array([])
    pmcs = np.ones(M)
    for k, mi in enumerate(elim):
        pmcs[mi] = pm[k]
    inc = [names[i] for i in range(M) if pmcs[i] >= alpha]
    order = [names[i] for i in elim]
    return {'included': inc, 'pvalues': dict(zip(names, pmcs.tolist())),
            'order': order, 'alpha': alpha, 'block': block, 'B': B, 'best': names[alive[0]] if alive else None}


# ---------------------------------------------------------------------------
# arah
# ---------------------------------------------------------------------------
def pesaran_timmermann(actual, predicted) -> dict:
    """Uji Pesaran-Timmermann (1992): apakah tanda prediksi berasosiasi dengan tanda aktual.

    actual, predicted: return (bukan harga). Prediksi yang tandanya konstan -> NaN
    (uji tidak terdefinisi). Untuk h > 1 gunakan sampel tak-tumpang-tindih (stride h).
    """
    a = (np.asarray(actual, float) > 0).astype(float)
    f = (np.asarray(predicted, float) > 0).astype(float)
    n = len(a)
    hit = float((a == f).mean())
    py, px = a.mean(), f.mean()
    base = {'n': n, 'hit': hit, 'base_up': float(py), 'pred_up': float(px)}
    if px in (0.0, 1.0) or py in (0.0, 1.0) or n < 10:
        return {**base, 'p_star': np.nan, 'stat': np.nan, 'p_one': np.nan}
    ps = py * px + (1 - py) * (1 - px)
    v_hit = ps * (1 - ps) / n
    v_ps = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
            + 4 * px * py * (1 - px) * (1 - py) / n ** 2)
    den = v_hit - v_ps
    if den <= 0:
        return {**base, 'p_star': float(ps), 'stat': np.nan, 'p_one': np.nan}
    z = (hit - ps) / np.sqrt(den)
    return {**base, 'p_star': float(ps), 'stat': float(z), 'p_one': float(_st.norm.sf(z))}


# ---------------------------------------------------------------------------
# skala bebas
# ---------------------------------------------------------------------------
def oos_r2(actual_ret, model_ret, bench_ret) -> float:
    """R^2 out-of-sample (Campbell & Thompson, 2008) = 1 - MSE_model / MSE_benchmark,
    di ruang return. > 0 berarti model lebih baik daripada benchmark."""
    a, m, b = (np.asarray(x, float) for x in (actual_ret, model_ret, bench_ret))
    mse_b = np.mean((a - b) ** 2)
    return float(1.0 - np.mean((a - m) ** 2) / mse_b) if mse_b > 0 else np.nan


def std_ratio(pred_ret, actual_ret) -> float:
    """Rasio simpangan baku prediksi / aktual (di ruang return). ~0 = prediksi hampir konstan."""
    s = np.std(np.asarray(actual_ret, float))
    return float(np.std(np.asarray(pred_ret, float)) / s) if s > 0 else np.nan
