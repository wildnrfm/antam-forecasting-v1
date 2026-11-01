"""Audit berkas mentah harga Antam: hari kerja tanpa pengumuman (forward-fill), komposisi hari
return-nol, dan jam pengumuman. HANYA MEMBACA; tidak menulis apa pun.

Skrip pembantu dokumentasi (rilis 1.0.2) — BUKAN bagian eksperimen dan tidak dipanggil notebook.
Angka yang dihasilkannya dikutip di panduan_bimbingan/PENJELASAN.md (Bab 03, 04, 11).

Pakai (dari akar repo):
    python panduan_bimbingan/skrip/audit_data_mentah.py
atau dengan jalur sendiri:
    python panduan_bimbingan/skrip/audit_data_mentah.py <Antam_historical_gold_prices.csv> <outputs/results>

Definisi (mengikuti tahap pembersihan data repo):
  * harga harian = pengumuman TERAKHIR pada tanggal kalender itu;
  * hari kerja   = Senin–Jumat dari pengumuman pertama sampai terakhir;
  * hari kerja tanpa pengumuman diisi forward-fill (harga hari kerja sebelumnya).
Definisi return-nol di sini sedikit berbeda dari `rezim_periode_split.csv` pada digit kedua
desimal (mis. 22,46% vs 22,47%) karena cara baris pertama dan batas periode dihitung.
"""
import sys
from pathlib import Path

import pandas as pd

try:  # aman bila keluaran dialihkan ke berkas di Windows
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'data/raw/antam_historical_gold_prices_download_asli/Antam_historical_gold_prices.csv'
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'outputs/results'

raw = pd.read_csv(RAW, parse_dates=['Date']).sort_values('Date')
raw['day'] = raw.Date.dt.normalize()
raw['hm'] = raw.Date.dt.hour + raw.Date.dt.minute / 60
print('pengumuman mentah (baris):', len(raw), '| pertama', raw.Date.min().date(), '| terakhir', raw.Date.max().date())

harian = raw.groupby('day')['Gold Price'].last()
bd = pd.bdate_range(harian.index.min(), harian.index.max())
ann = bd.isin(harian.index)
print('hari kerja (Sen–Jum) pertama..terakhir:', len(bd), '| tanpa pengumuman (di-forward-fill):', int((~ann).sum()),
      f'({100 * (~ann).mean():.2f}%)')
print('tanggal pengumuman yang jatuh di akhir pekan (tidak membentuk baris sendiri):', int((~harian.index.isin(bd)).sum()))

pp = harian.reindex(harian.index.union(bd)).ffill().reindex(bd)
ret = pp.pct_change()
df = pd.DataFrame({'zero': ret == 0, 'ann': ann}, index=bd)

ts = pd.read_csv(RES / 'tabel_split.csv')
tr_end = pd.Timestamp(ts[ts.h == 1].iloc[0]['latih_s.d.'])
tr = df.loc[:tr_end].iloc[1:]  # baris pertama tidak punya return
nz = int(tr.zero.sum())
print(f'\nperiode latih (h=1) s.d. {tr_end.date()}: {len(tr)} hari kerja bertanda return, return nol = {nz} ({100 * nz / len(tr):.2f}%)')
print(f'  dari hari return-nol: ada pengumuman (harga sama)  = {int((tr.zero & tr.ann).sum())} ({100 * (tr.zero & tr.ann).sum() / nz:.1f}%)')
print(f'  dari hari return-nol: tanpa pengumuman (ffill)     = {int((tr.zero & ~tr.ann).sum())} ({100 * (tr.zero & ~tr.ann).sum() / nz:.1f}%)')
print(f'  porsi return-nol di antara hari ber-pengumuman saja = {100 * (tr.zero & tr.ann).sum() / tr.ann.sum():.2f}%')

# jam pengumuman terakhir per hari kerja (jam 00:00 = jam tidak tercatat)
last = raw.groupby('day').tail(1)
last = last[last.Date.dt.weekday < 5]
tod = last.hm[last.hm > 0]
q = lambda x: f'{int(x):02d}:{int(round((x - int(x)) * 60)):02d}'
print(f'\njam pengumuman terakhir per hari kerja: tercatat pada {len(tod)} dari {len(last)} hari ({100 * len(tod) / len(last):.0f}%)')
print('  median', q(tod.median()), '| P5', q(tod.quantile(.05)), '| P95', q(tod.quantile(.95)), '| P99', q(tod.quantile(.99)))
print(f'  sebelum 07:00: {100 * (tod < 7).mean():.1f}% | sesudah 17:00: {100 * (tod > 17).mean():.1f}%')
print('\nCatatan: zona waktu jam di berkas mentah diasumsikan WIB; belum diverifikasi dengan catatan resmi Antam.')
