# Peramalan Harga Emas Antam — 21 Model dengan Protokol Bebas-Kebocoran

Eksperimen peramalan *direct multi-horizon* atas harga jual emas Antam (Rp/gram) untuk
horizon **1, 5, 10, 20, dan 30 hari kerja**, membandingkan **21 model** (10 machine
learning + 11 deep learning, termasuk DPformer) terhadap benchmark Drift, dengan
penekanan pada **kejujuran metode**: setiap pagar anti-*lookahead* diuji secara otomatis,
dan setiap klaim statistik dilaporkan lengkap dengan daya ujinya (MDE).

| | |
|---|---|
| **Versi repo** | `1.0.1` (lihat [`VERSION`](VERSION)) |
| **Freeze hasil** | `1.0.0` · 2026-10-19 |
| **Protokol** | `v9-21model-dpformer-bao-aligned-honest-mossa-shap-leakage-safe-embargo` |
| **Revisi ML** | `ml-r2` |
| **Sidik jari data** | `7aea8db80d44` |
| **Python** | 3.13.0 |
| **Integritas** | [`MANIFEST.sha256`](MANIFEST.sha256) (repo) · [`outputs/results/MANIFEST.json`](outputs/results/MANIFEST.json) (freeze hasil) |

---

## 1. Data

| | |
|---|---|
| Sumber utama | Harga emas Antam (dataset tidak resmi, `data/raw/antam_historical_gold_prices_download_asli/`) |
| Makro | Yahoo Finance (DXY, S&P 500, USD/IDR, IHSG) · FRED (US 10Y, FFR) · BIS (BI-Rate) |
| Periode | **2010-01-05 → 2026-08-03** |
| Bentuk | `data/processed/data_gabungan.csv` — **4.325 baris × 16 variabel** (di luar kolom `tanggal`), tanpa *missing* |
| Kalender | Hari kerja emas; hari libur tanpa pengumuman diisi **forward-fill** (bukan interpolasi, karena interpolasi memakai harga pengumuman berikutnya = *lookahead*) |

**Tujuh variabel makro** sudah di-*lag* 1 hari kerja sejak tahap penggabungan, sehingga
pada origin peramalan `i` seluruh fitur makro berasal dari hari `≤ i`. FFR bulanan mulai
dipakai pada bulan berikutnya (*as-of*).

Fitur model memakai **versi stasioner** dari level makro (log-return untuk indeks/kurs,
selisih untuk suku bunga/yield), bukan level mentahnya — level seperti S&P 500 dan FFR
*trending* dan berada di luar rentang latih pada periode uji, sehingga model hanya akan
meng-ekstrapolasi.

---

## 2. Metode

### Target

```
R_h(i) = p(i+h) / p(i) − 1        ← return kumulatif h hari, diprediksi langsung (direct)
p̂(i+h) = p(i) · (1 + r̂)           ← evaluasi dilakukan di ruang LEVEL HARGA
```

### Dua setup fitur

| Setup | Isi |
|---|---|
| **Univariat** | `lag_0` (return hari-i) + 8 lag + 6 statistik rolling → **15 fitur** |
| **Multivariat** | di atas + **7 fitur makro stasioner** → **22 fitur** |
| **DL (11 model)** | jendela `SEQ_LEN = 10` hari multi-channel, z-score yang di-*fit* **hanya** pada data latih |

### Split kronologis 70/10/20 + embargo

Pembagian murni kronologis (tanpa pengacakan), horizon-aware, identik untuk seluruh model:

| h | Latih | Validasi | Validasi (embargo) | Uji | Uji mulai |
|---:|---:|---:|---:|---:|---|
| 1 | 3028 | 432 | 431 | 864 | 2023-04-11 |
| 5 | 3024 | 432 | 427 | 864 | 2023-04-05 |
| 10 | 3021 | 431 | 421 | 863 | 2023-03-30 |
| 20 | 3014 | 430 | 410 | 861 | 2023-03-20 |
| 30 | 3007 | 429 | 399 | 859 | 2023-03-08 |

**Embargo** = baris validasi yang target `i+h`-nya jatuh di periode uji **dibuang dari
early stopping**. Tanpa ini, *early stopping* pada run final bisa diam-diam membaca
harga periode uji.

### 21 model

- **10 ML:** LightGBM · XGBoost · RandomForest · CatBoost · GBM · SVR · ExtraTrees ·
  HistGB · AdaBoost · ElasticNet
- **11 DL:** LSTM · GRU · BiLSTM · BiGRU · Transformer · DLinear · NLinear · PatchTST ·
  LSTM+Attention · GRU+Attention · **DPformer**

Konfigurasi 10 baseline DL mengikuti **Bao et al. (2026), Table 3** sedekat mungkin
(formula *as-published*, tidak dituning ulang). DPformer dituning dengan **MOSSA**
(populasi 20 / 100 iterasi, fitness **hanya** dari baris validasi ber-embargo).

> **Catatan kesimetrian tuning:** hanya DPformer yang dituning. Ablation
> `outputs/results/ablation_mossa_vs_default.csv` menunjukkan MOSSA **tidak** memberi
> perbaikan signifikan atas default (**0 dari 10 uji lolos BH-FDR**), sehingga
> kesimetrian tuning ini tidak memengaruhi kesimpulan.

### Benchmark Drift

`r̂ = μ_train` — rata-rata return pada periode latih. Selain itu tersedia `drift_exp`
(ekspansif) dan `drift_roll` (bergulir 250 hari) sebagai pembanding **causal-adaptif**.

---

## 3. Struktur repository

```
antam_forecasting/
├── data/
│   ├── raw/                    dataset Antam + 8 sumber makro + cache fallback API
│   └── processed/              8 file bersih + glitch_dibuang.csv + data_gabungan.csv
├── src/                        MESIN — seluruh logika ada di sini (lihat §5)
├── notebooks/                  00–19, orkestrasi (lihat §4)
├── outputs/
│   ├── figures/                69 PNG + 69 PDF
│   └── results/                48 CSV top-level + subdirektori (runs, walkforward, dst.)
├── panduan_bimbingan/          RINGKASAN.md + PENJELASAN.md (Bahasa Indonesia)
├── README.md · CHANGELOG.md · VERSION
├── requirements.txt · requirements-full.txt
└── MANIFEST.sha256
```

**Prinsip arsitektur:** notebook **tidak melatih model sendiri**. Semua pelatihan
melewati `src/runners.py`; analisis dan figur melewati modul `src/` (kecuali NB 00–01
yang khusus mengunduh/membersihkan data), sehingga eksperimen utama, ablation, skenario B,
walk-forward, dan uji seed tidak mungkin berbeda diam-diam.

---

## 4. Peta notebook (jalankan berurutan 00 → 19)

| Tahap | NB | Isi |
|---|---|---|
| **A. Data** | `00_unduh_indikator_makro` | Unduh 7 indikator makro + *fallback* ke cache bila API mati |
| | `01_pembersihan_data` | Bersihkan 8 deret → hari kerja, forward-fill |
| | `02_penggabungan_data` | Gabung + **lag-1** + transformasi stasioner → `data_gabungan.csv` |
| **B. Desain** | `03_eda_dan_stasioneritas` | Uji ADF, lead-lag, porsi return nol, profil per tahun |
| | `04_feature_engineering` | Dua jalur fitur, split 70/10/20, profil rezim pasar |
| **C. Model** | `05_dpformer` | 2 setup × 5 horizon + **tuning MOSSA** |
| | `06_dl_models` | 10 baseline DL |
| | `07_ml_models` | 10 model ML |
| **D. Integritas** | `08_validasi_integritas` | **7 pagar anti-bocoran** (mutation testing) |
| **E. Hasil** | `09_hasil_dan_ranking` | Peringkat MAE, metrik bebas-skala, pemilihan model wakil |
| | `10_uji_statistik` | DM-HLN, BH-FDR, MCS 90%, Pesaran–Timmermann |
| | `11_interpretasi_shap` | SHAP untuk 3 model wakil |
| | `12_analisis_tambahan` | Diagnosis loss curve, stabilitas blok/rezim |
| **F. Robustness** | `13_uji_robustness_seed` | 5 seed (42–46) untuk 6 model DL |
| | `14_ablation_study` | 4 kelompok fitur + MOSSA vs default |
| | `15_skenario_b` | Origin malam hari: informasi penutupan pasar hari-*t* |
| | `16_walk_forward` | Rolling-origin **6 fold**, 2020-08 → 2026-07 |
| **G. Sekunder** | `17_selang_kepercayaan_vs_drift` | Selang kepercayaan + **MDE** |
| | `18_kombinasi_dan_drift_adaptif` | Kombinasi bobot-sama + Drift causal-adaptif |
| | `19_tangga_waktu_informasi` | Tangga `lag-1` → `hari-t` → `plasebo t+1` |

---

## 5. Peta modul `src/`

| Modul | Peran |
|---|---|
| `config.py` | Seluruh konstanta eksperimen (horizon, unit, batch, MOSSA, `PROTOCOL`) |
| `pipeline.py` | Fitur tabular, split horizon-aware, target DL, scaler, deteksi run basi |
| `models_dl.py` | 11 arsitektur DL termasuk DPformer |
| `runners.py` | **Satu-satunya** jalur pelatihan/penyimpanan run + `data_fingerprint` |
| `checks.py` | 7 pagar integritas — merusak data uji lalu memastikan hasil tidak berubah |
| `analysis.py` | Metrik pasca-evaluasi, stabilitas, diagnosis loss curve |
| `stats.py` | DM-HLN, BH-FDR, MCS, Pesaran–Timmermann, R² OOS |
| `explain.py` | SHAP otomatis untuk 3 model wakil |
| `ekstra.py` | Analisis sekunder (notebook 17–19): MDE, kombinasi, tangga waktu |
| `plots.py` | Seluruh figur (label dua bahasa, `TA_LANG=id` / `en`) |

---

## 6. Cara menjalankan

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt          # dependensi langsung
# atau, untuk reproduksi persis:
python -m pip install -r requirements-full.txt     # 155 paket terkunci

# 2. jalankan notebook berurutan
jupyter notebook
# 00 → 01 → 02 → ... → 19
```

> Sebagian besar notebook **tidak mengulang pekerjaan yang sudah ada** — `src/runners.py`
> memeriksa `protocol`, `rev`, dan `data_hash` pada tiap run; run yang sudah basi saja yang
> dihitung ulang.

### Mode cepat (smoke test)

```powershell
$env:TA_QUICK = "1"    # horizon [5,20], EPOCHS=2, unit kecil, output ke outputs_quick/
```

### Variabel lingkungan

| Var | Default | Fungsi |
|---|---|---|
| `TA_QUICK` | `0` | Mode cepat (smoke test) |
| `TA_LANG` | `id` | Bahasa label figur: `id` atau `en` |
| `TA_SEED` | `42` | Seed acak |
| `TA_THREADS` | — | Batas jumlah thread |

### Verifikasi integritas

```powershell
# PowerShell: laporkan file yang hash-nya tidak cocok dengan MANIFEST.sha256
$ok = 0; $bad = @()
Get-Content MANIFEST.sha256 | Where-Object { $_ -match '^[0-9a-f]{64}\s{2}' } | ForEach-Object {
    if ($_ -match '^([0-9a-f]{64})\s{2}(.+)$') {
        $hash = (Get-FileHash $Matches[2] -Algorithm SHA256).Hash.ToLower()
        if ($hash -eq $Matches[1]) { $ok++ } else { $bad += $Matches[2] }
    }
}
"COCOK: $ok | TIDAK COCOK: $($bad.Count)"
$bad
```

Di Git Bash/Linux: `sha256sum -c MANIFEST.sha256`.

---

## 7. Hasil utama

### Model terbaik per horizon (MAE ruang harga, `pct_vs_drift` = % lebih baik dari Drift)

| Horizon | Univariat | % vs Drift | Multivariat | % vs Drift |
|---:|---|---:|---|---:|
| 1 | PatchTST | 0,59 | LightGBM | 0,43 |
| 5 | SVR | 1,27 | Transformer | 1,97 |
| 10 | AdaBoost | 3,78 | AdaBoost | 3,87 |
| 20 | AdaBoost | 6,77 | AdaBoost | 4,76 |
| 30 | RandomForest | 3,21 | PatchTST | 2,43 |

Peringkat rata-rata keseluruhan dipimpin **AdaBoost** (`mean_rank` 2,8, menang di 4
sel/10), diikuti RandomForest (6,1) dan ElasticNet (6,3).

### Temuan yang perlu dilaporkan apa adanya

| Uji | Hasil |
|---|---|
| DM vs Drift (210 uji) | 115/210 `rel_mse < 1`; **16** dengan `p < 0,05`; **1** lolos BH-FDR dengan keluarga per setup×horizon → *univariat transformer, h=1, `rel_mse` 0,9950, `q` 0,041*. Dengan keluarga seluruh 210 uji, **0** yang lolos |
| MCS 90% | Hampir seluruh model **dan** Drift masuk *confidence set* pada hampir semua horizon |
| Multivariat vs univariat (105 uji) | **0 lolos BH-FDR** dengan keluarga 105 uji → 7 variabel makro tidak menambah informasi di luar return emas sendiri. Dengan keluarga per horizon, **2** lolos (GBM & HistGBM h=1, `q` 0,0138) |
| Walk-forward (6 fold, 2020–2026) | Tidak ada model yang signifikan mengalahkan Drift |
| 5 seed × 6 model DL (60 kombinasi) | CV MAE **0,034% – 1,30%** (median 0,29%) → variasi seed jauh lebih kecil daripada selisih antar-model |
| MOSSA vs default | **0/10 uji lolos BH** → tuning tidak signifikan |
| **Skenario B (penutupan hari-*t*), h=1** | 10 model ML: **median `rel_mse` 0,9267** (**7,3%** lebih baik dari baseline multivariat lag-1), **9/10 lolos BH**. Varian 12 model (tambah DPformer & GRU+Att): median 0,9267, **11/12 lolos** |
| Skenario B, h ≥ 5 | 10 model ML: tidak ada yang lolos BH (median ≈ 1,00). Varian 12 model: DPformer lolos di h=20 (`q` 0,0149) |
| **Plasebo t+1** (tidak layak) | median ≈ 1,01, **0 lolos BH** → bukti tidak ada artefak |

**Interpretasi:** temuan positif yang paling konsisten adalah **informasi penutupan
pasar hari-*t* membantu di h=1** (9 dari 10 model ML), dan plasebo t+1 gagal sehingga
temuan itu bukan artefak. Pada varian 12 model ada satu pengecualian (DPformer, h=20,
`q` 0,0149); karena tidak muncul di 10 model ML dan tidak konsisten lintas horizon,
temuan utama tetap h=1. Untuk h ≥ 5, dan untuk fitur makro secara umum, tidak ada
perbedaan yang dapat dibuktikan secara konsisten.

---

## 8. Stempel reprodusibilitas

Setiap `outputs/results/runs/*.json` memuat tiga stempel berikut. Bila salah satunya
berubah, run dianggap **basi** dan dihitung ulang otomatis.

```
protocol  = v9-21model-dpformer-bao-aligned-honest-mossa-shap-leakage-safe-embargo
rev       = ml-r2
data_hash = 7aea8db80d44          ← data_fingerprint(data_gabungan.csv)
```

Saat freeze `v1.0.0`: **210/210 run CURRENT, 0 basi.**

---

## 9. Batasan yang diketahui

Dilaporkan di sini agar tidak ditemukan pertama kali saat *review*:

1. **Satu aset, satu negara.** Seluruh kesimpulan berlaku untuk harga emas Antam
   (Rp/gram), bukan untuk emas secara umum.
2. **Harga postingan administratif, bukan harga transaksi.** 22,5% hari pada periode
   latih tidak bergerak (*return* nol) — deret ini punya mikrostruktur berbeda dari
   deret harga pasar.
3. **Tidak memakai harga emas dunia (XAU/USD).** Tujuh fitur makro adalah DXY, S&P 500,
   USD/IDR, IHSG, US 10Y, FFR, BI-Rate — tidak ada XAU/USD/LBMA/COMEX.
4. **Daya uji terbatas pada horizon panjang.** `n_eff` median di h=30 hanya **28,6**
   observasi independen; MDE terhadap Drift adaptif mencapai **48–70%**. Absennya
   perbedaan pada horizon panjang **bukan** bukti ketiadaan efek.
5. **Periode uji adalah reli besar** (harga 1,04 → 3,17 juta Rp; volatilitas 0,97%/hari vs
   0,65% di validasi). Walk-forward mencakup 2020–2026, tetapi 2010–2020 hanya data latih.
6. **Sumber data emas tidak resmi** (dataset komunitas). Perlu validasi silang terhadap
   situs resmi Antam.
7. **⚠️ Ukuran file bobot.** `outputs/results/models/bilstm_multivariat_h20.weights.h5`
   berukuran **96,9 MB** — hanya 3,1 MB di bawah limit keras GitHub (100 MB menolak
   total). Bila file ini membesar saat *re-run* dan push gagal: `git rm --cached` file
   tersebut, atau alihkan ke Git LFS.

---

## 10. Sumber data & status

- Harga emas Antam: dataset tidak resmi berlisensi MIT —
  [`data/raw/antam_historical_gold_prices_download_asli/README.md`](data/raw/antam_historical_gold_prices_download_asli/README.md)
- Indikator makro: Yahoo Finance, FRED, BIS
- Acuan metodologis: Bao et al. (2026), *Expert Systems with Applications*
- Dokumen pendamping (Bahasa Indonesia):
  [`panduan_bimbingan/RINGKASAN.md`](panduan_bimbingan/RINGKASAN.md) ·
  [`panduan_bimbingan/PENJELASAN.md`](panduan_bimbingan/PENJELASAN.md)
- **Belum ada lisensi** untuk kode repository ini (sengaja dilewati pada v1.0.0).

---

<sub>Repo v1.0.1 · hasil dibekukan pada v1.0.0 · Protokol `v9` · data_hash `7aea8db80d44` · 2026-10-21</sub>
