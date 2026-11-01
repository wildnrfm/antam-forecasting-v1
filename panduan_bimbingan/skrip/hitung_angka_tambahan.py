"""Tiga perhitungan kecil yang dipakai panduan_bimbingan/PENJELASAN.md tetapi belum dihasilkan
notebook. HANYA MEMBACA; tidak menulis apa pun. Hanya butuh numpy + pandas.

Skrip pembantu dokumentasi (rilis 1.0.2) — BUKAN bagian eksperimen dan tidak dipanggil notebook.

Pakai (dari akar repo):
    python panduan_bimbingan/skrip/hitung_angka_tambahan.py
atau dengan jalur sendiri:
    python panduan_bimbingan/skrip/hitung_angka_tambahan.py <outputs/results> <data_gabungan.csv>

Isi:
  1. Kestabilan peringkat 6 model DL antar-seed (dari multiseed_semua.csv / multiseed_ringkasan.csv).
  2. Koreksi Benjamini–Hochberg pada keluarga seluruh 210 uji DM vs Drift (p_one_hln), plus pemeriksaan
     bahwa implementasi BH di sini mereproduksi kolom q_bh_setup (105 uji) dan q_bh_setup_h (21 uji).
  3. Arah harga pada setiap jendela walk-forward fold × horizon (dari walkforward_per_fold.csv).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # keluaran berisi karakter non-ASCII (mis. →); aman juga bila dialihkan ke berkas di Windows
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
R = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'outputs/results'
DG = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'data/processed/data_gabungan.csv'
pd.set_option('display.width', 200)


def bh(pv):
    """q-value Benjamini–Hochberg (step-up) untuk satu keluarga uji."""
    pv = np.asarray(pv, dtype=float)
    m = len(pv)
    order = np.argsort(pv)
    q = np.empty(m)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        prev = min(prev, pv[i] * m / (k + 1))
        q[i] = prev
    return q


# ------------------------------------------------------------------ 1. seed
print('=' * 8, '1. Kestabilan antar-seed (6 model DL × 2 setup × 5 horizon × 5 seed)')
ms = pd.read_csv(R / 'multiseed_ringkasan.csv')
al = pd.read_csv(R / 'multiseed_semua.csv')
rasio = []
for (s, h), g in ms.groupby(['setup', 'horizon']):
    rasio.append((s, h, (g.MAE_mean.max() - g.MAE_mean.min()) / g.MAE_std.mean()))
rasio = pd.DataFrame(rasio, columns=['setup', 'horizon', 'rentang_arsitektur_per_SD_seed'])
print(f"rentang MAE antar-arsitektur ÷ SD antar-seed: min {rasio.iloc[:, 2].min():.1f}, median {rasio.iloc[:, 2].median():.1f}, maks {rasio.iloc[:, 2].max():.1f}")

rows = []
for (s, h), g in al.groupby(['setup', 'horizon']):
    P = g.pivot(index='model', columns='seed', values='MAE')
    rk = P.rank(axis=0)  # 1 = MAE terkecil; korelasi Pearson pada peringkat = Spearman
    rho = [np.corrcoef(rk[a], rk[b])[0, 1] for a, b in itertools.combinations(P.columns, 2)]
    rows.append((s, h, P.idxmin(axis=0).nunique() == 1, float(np.mean(rho))))
st = pd.DataFrame(rows, columns=['setup', 'horizon', 'model_terbaik_sama_di_semua_seed', 'rho_rata2_antar_pasangan_seed'])
print(st.round(3).to_string(index=False))
print(f"sel dengan model terbaik yang sama di kelima seed: {int(st.model_terbaik_sama_di_semua_seed.sum())} dari {len(st)}")
print(f"korelasi peringkat rata-rata per sel: min {st.iloc[:, 3].min():.2f}, median {st.iloc[:, 3].median():.2f}, maks {st.iloc[:, 3].max():.2f}")

# ------------------------------------------------------------------ 2. BH keluarga
print('\n' + '=' * 8, '2. BH pada berbagai keluarga uji (DM vs Drift, 210 uji)')
dm = pd.read_csv(R / 'dm_vs_drift_detail.csv')
q210 = bh(dm.p_one_hln.values)
print(f'keluarga seluruh 210 uji   : q terendah {q210.min():.4f} | jumlah q < 0,05 = {int((q210 < .05).sum())}')
q105 = pd.Series(index=dm.index, dtype=float)
q21 = pd.Series(index=dm.index, dtype=float)
for _, g in dm.groupby('setup'):
    q105[g.index] = bh(g.p_one_hln.values)
for _, g in dm.groupby(['setup', 'horizon']):
    q21[g.index] = bh(g.p_one_hln.values)
print(f'keluarga per setup (105)   : q terendah {q105.min():.4f} | jumlah q < 0,05 = {int((q105 < .05).sum())}'
      f'  (selisih maks vs kolom q_bh_setup: {float((q105 - dm.q_bh_setup).abs().max()):.1e})')
print(f'keluarga setup×horizon (21): q terendah {q21.min():.4f} | jumlah q < 0,05 = {int((q21 < .05).sum())}'
      f'  (selisih maks vs kolom q_bh_setup_h: {float((q21 - dm.q_bh_setup_h).abs().max()):.1e})')

# ------------------------------------------------------------------ 3. arah harga walk-forward
print('\n' + '=' * 8, '3. Arah harga per jendela walk-forward (fold × horizon), origin pertama → origin terakhir')
p = pd.read_csv(DG, parse_dates=['tanggal']).set_index('tanggal').harga_jual
wf = pd.read_csv(R / 'walkforward_per_fold.csv', parse_dates=['mulai', 'akhir'])
win = wf.groupby(['horizon', 'blok']).agg(mulai=('mulai', 'first'), akhir=('akhir', 'first'), n=('n', 'first')).reset_index()
win['perubahan_harga_%'] = [round(100 * (p.loc[r.akhir] / p.loc[r.mulai] - 1), 1) for r in win.itertuples()]
print(win.to_string(index=False))
naik = int((win['perubahan_harga_%'] > 0).sum())
print(f"harga naik pada {naik} dari {len(win)} jendela; rentang {win['perubahan_harga_%'].min()}% s.d. {win['perubahan_harga_%'].max()}%")
