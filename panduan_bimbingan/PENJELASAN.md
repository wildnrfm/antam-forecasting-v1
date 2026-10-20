# PENJELASAN — Dari Awal Sampai Akhir

**Panduan bimbingan untuk menjelaskan proyek ini di depan dosen.**

Proyek: peramalan harga jual emas Antam · 21 model · 210 eksperimen
Versi: **1.0.0** · Protokol `v9-21model-dpformer-bao-aligned-honest-mossa-shap-leakage-safe-embargo`
Sidik jari data: `7aea8db80d44` · Python 3.13.0

> **Ringkasan satu halaman:** [`RINGKASAN.md`](RINGKASAN.md)
> **Rujukan teknis:** [`../README.md`](../README.md)

---

## Cara memakai dokumen ini

| Kalau waktu… | Baca |
|---|---|
| **5 menit** | [`RINGKASAN.md`](RINGKASAN.md) — cukup untuk membuka pembicaraan |
| **30 menit** | Bab **01** → **05** → **09** → **12** (masalah, desain, uji statistik, kesimpulan) |
| **2 jam / mendalam** | Seluruh dokumen urut, ditambah Lampiran **C** (FAQ pertanyaan dosen) |

Setiap bab memakai kerangka seragam:

- **❓ Pertanyaan yang dijawab** — inti bab dalam satu kalimat
- **💡 Penjelasan** — kenapa langkah ini diambil, apa alternatifnya, kenapa alternatifnya ditolak
- **📁 Bukti di repo** — file, tabel, dan figur yang bisa langsung ditunjukkan

![Kerangka penelitian](../outputs/figures/fig_kerangka_penelitian.png)

---

## Daftar isi

| Bab | Judul |
|---|---|
| 01 | Latar belakang dan pertanyaan penelitian |
| 02 | Data dan sumber |
| 03 | Pembersihan dan penggabungan |
| 04 | EDA dan stasioneritas |
| 05 | Desain eksperimen: target, fitur, pembagian data |
| 06 | Model yang dipakai |
| 07 | Tujuh pagar integritas |
| 08 | Hasil dasar dan peringkat |
| 09 | Uji statistik — dan cara menyajikan hasil negatif |
| 10 | Interpretasi (SHAP) |
| 11 | Ketahanan: seed, ablation, walk-forward, plasebo |
| 12 | Kesimpulan, kontribusi, dan batasan |
| A | Peta file repo |
| B | Glosarium |
| C | FAQ — kemungkinan pertanyaan dosen |
| D | Cara menjalankan ulang |

---

# 01 — Latar belakang dan pertanyaan penelitian

### ❓ Pertanyaan yang dijawab

*Apa sebenarnya yang diprediksi, dan mengapa itu bukan sekadar "menebak harga emas"?*

### 💡 Penjelasan

Emas dalam Rupiah punya dua pembentuk harga: harga emas dalam dolar dunia, dan nilai
tukar USD/IDR. Keduanya digerakkan oleh faktor yang sama-sama bisa diamati — kebijakan
suku bunga AS, imbal hasil obligasi, sentimen pasar global. Jadi secara intuisi,
**seharusnya** harga emas Antam bisa ditebak dari data makro lampau.

Pertanyaan penelitiannya bukan "apakah bisa ditebak", melainkan **seberapa yakin kita
bisa mengklaim demikian**. Dua hal membuat pertanyaan itu lebih sulit dari kelihatannya:

1. **Harga Antam adalah harga postingan administratif.** Bukan harga transaksi yang
   bergerak setiap detik. Pada hari tanpa pengumuman, harganya tetap — 22,47% hari pada
   periode latih tidak bergerak sama sekali. Deret ini punya mikrostruktur yang berbeda
   dari deret harga pasar biasa.
2. **Menguji banyak model menimbulkan masalah sendiri.** Kalau 21 model diuji, beberapa
   akan "menang" secara kebetulan. Tanpa koreksi uji berganda (BH-FDR) dan tanpa benchmark
   yang jujur, kita akan menemukan sesuatu yang tidak ada.

Empat pertanyaan yang kemudian dijawab berurutan:

| # | Pertanyaan | Dijawab di |
|---|---|---|
| 1 | Bisa diprediksi tidak, dan seberapa yakin? | Bab 09 |
| 2 | Apakah fitur makro membantu dibanding harga sendiri saja? | Bab 09 |
| 3 | Apakah hasilnya bergantung pada keberuntungan seed / model / periode? | Bab 11 |
| 4 | Kalau negatif semua, apa kontribusinya? | Bab 12 |

### 📁 Bukti di repo

```
src/config.py                              PROTOCOL, ML_REV, HORIZONS, SETUPS
notebooks/03_eda_dan_stasioneritas.ipynb   pengantar konsep & statistik deskriptif
outputs/results/statistik_deskriptif.csv
```

---

# 02 — Data dan sumber

### ❓ Pertanyaan yang dijawab

*Data apa yang masuk, sepanjang apa, dan mengapa variabel makro harus di-*lag *dari awal?*

### 💡 Penjelasan

**Deret utama:** harga jual emas Antam per gram, 4.325 hari kerja dari 2010-01-05 sampai
2026-08-03, 16 kolom, tanpa *missing* dan tanpa duplikat. Kalender emas punya celah
(hari libur nasional), tetapi celah terpanjang hanya **3 hari**.

**Tujuh variabel makro** — bukan level, melainkan **transformasi stasioner**:

| Kelompok | Variabel | Transformasi | Kenapa |
|---|---|---|---|
| Log-return | DXY · S&P 500 · USD/IDR · IHSG | `log(x_t / x_{t-1})` | levelnya *trending* dan keluar dari rentang latih |
| Selisih poin | US 10Y · FFR · BI-Rate | `x_t − x_{t-1}` (poin %) | tingkat suku bunga naik/turun, bukan bergerak bebas |

**Kenapa di-lag sejak tahap penggabungan, bukan saat pemodelan?** Pada origin peramalan
`i`, kita hanya boleh memakai informasi yang sudah diketahui hari `i`. Data makro
dirilis pada atau setelah hari `i`, jadi kalau dipakai apa adanya, model akan membaca
masa depan. Menunda lag-nya ke tahap fitur rawan terlupa; menaruhnya di tahap penggabungan
membuat seluruh eksperimen otomatis aman.

Pembacaan level-vs-stasioner juga penting: S&P 500 punya rentang 1.022–7.609 (x7) dan
FFR 0,05–5,33% (x100). Model yang dilatih pada 2010–2021 **tidak pernah melihat** level
seperti itu, jadi ia hanya bisa mengekstrapolasi.

> **Yang sengaja tidak dipakai:** harga emas dunia (XAU/USD), LBMA, COMEX. Ini batasan
> yang harus diakui lebih dulu — lihat Bab 12 dan Lampiran C.

### 📁 Bukti di repo

```
data/raw/                        8 sumber mentah + cache fallback API
data/processed/data_gabungan.csv tabel master: 4.325 × 16, fp 7aea8db80d44
src/pipeline.py                  EXO_LEVEL_COLS, EXO_LOGRET_COLS, EXO_DIFF_COLS, EXO_CLIP=5.0
notebooks/00_unduh_indikator_makro.ipynb
notebooks/02_penggabungan_data.ipynb
```

---

# 03 — Pembersihan dan penggabungan

### ❓ Pertanyaan yang dijawab

*Bagaimana mengisi hari libur tanpa diam-diam menyuntikkan informasi masa depan?*

### 💡 Penjelasan

Ketika harga tidak diumumkan, nilainya harus diisi. Ada dua cara:

| Cara | Cara kerja | Masalah |
|---|---|---|
| **Interpolasi** | rata-rata harga sebelum dan sesudahnya | ❌ Memakai harga **berikutnya** = *lookahead* |
| **Forward-fill** | bawa terus harga hari sebelumnya | ✅ Hanya memakai masa lalu |

Forward-fill dipilih karena konsisten dengan kenyataan operasional: kalau Antam tidak
mengumumkan hari itu, yang berlaku tetap pengumuman terakhir. Keputusan ini juga membuat
**porsi return nol** menjadi terlihat apa adanya — dan angka itu sendiri justru menjadi
temuan (lihat Bab 04).

Untuk FFR yang hanya dirilis bulanan, dipakai nilai **as-of**: pengumuman bulan M hanya
mulai dipakai pada bulan berikutnya. Memakainya pada bulan yang sama berarti membaca
keputusan sebelum diumumkan.

**Kenapa hal-hal kecil ini dipisah jadi bab sendiri?** Karena *lookahead* tidak
menyebabkan error — ia menyebabkan **hasil yang terlalu bagus**. Tidak ada yang memberi
peringatan. Satu-satunya cara menangkapnya adalah pagar di Bab 07.

### 📁 Bukti di repo

```
notebooks/01_pembersihan_data.ipynb     bersihkan 8 deret, forward-fill
notebooks/02_penggabungan_data.ipynb    gabung + lag-1 + transformasi stasioner
data/processed/data_gabungan.csv
```

---

# 04 — EDA dan stasioneritas

### ❓ Pertanyaan yang dijawab

*Apakah deret ini bisa dipakai langsung, dan informasi apa yang sebenarnya menggerakkannya?*

### 💡 Penjelasan

**Stasioneritas (Augmented Dickey–Fuller).** H0: deret memiliki *unit root*. Hasilnya
jelas dan menjadi dasar seluruh transformasi:

| Deret | ADF | p-value | Kesimpulan |
|---|---:|---:|---|
| Harga (level) | 1,4424 | 0,9973 | **Tidak** stasioner — gagal tolak H0 |
| Return harian | −22,7899 | ≈ 0 | Stasioner |
| Harga (diff-1) | −8,2807 | ≈ 0 | Stasioner |

Karena itu targetnya **return**, bukan level, dan fiturnya pun versi stasioner.

**Harga yang tidak bergerak.** Sebagian besar hari menghasilkan return tepat nol, dan
porsinya turun tajam seiring waktu:

| Periode | Porsi return nol | SD harian | Rentang harga |
|---|---:|---:|---|
| Latih (2010–2021) | **22,47%** | 0,69% | 0,39 – 1,06 jt |
| Validasi (2021–2023) | 12,79% | 0,65% | 0,91 – 1,06 jt |
| Uji (2023–2026) | 7,20% | **0,97%** | **1,04 – 3,17 jt** |

Puncaknya tahun 2014: **40,6%** hari tidak bergerak. Ini bukan cacat data — itu
memang sifat harga postingan. Tetapi ia berarti model harus belajar meramal
**kapan harga bergerak**, bukan hanya seberapa jauh.

**Regim yang berubah.** Periode uji adalah reli besar (harga 1,04 → 3,17 juta, volatilitas
0,97%/hari vs 0,65% di validasi). Artinya latih dan uji berada di rezim berbeda — ini
salah satu alasan hasil di Bab 09 sebagian negatif.

**Waktu informasi (lead–lag).** Dihitung apakah korelasi fitur makro terbesar terhadap
return emas ada pada X(t−2), X(t−1), atau X(t). Untuk **semua** variabel, korelasi
terkuat muncul di **X(t−1)** — misalnya DXY −0,288 dan US 10Y −0,233, jauh lebih besar
daripada X(t) (−0,062 dan −0,017). Ini konfirmasi empiris bahwa lag-1 yang dipakai
adalah tempat sinyalnya berada.

### 📁 Bukti di repo

![Korelasi variabel](../outputs/figures/eda_korelasi_variabel.png)

![Lead-lag waktu informasi](../outputs/figures/fig_leadlag.png)

![Porsi return nol](../outputs/figures/fig_porsi_return_nol.png)

![Distribusi return](../outputs/figures/eda_distribusi_return.png)

```
outputs/results/adf_hasil.csv                 3 baris: level, return, diff-1
outputs/results/porsi_return_nol_per_tahun.csv 2010–2026
outputs/results/rezim_periode_split.csv        latih / validasi / uji
outputs/results/leadlag_waktu_informasi.csv    korelasi pada X(t−2), X(t−1), X(t)
notebooks/03_eda_dan_stasioneritas.ipynb
```

---

# 05 — Desain eksperimen: target, fitur, pembagian data

### ❓ Pertanyaan yang dijawab

*Apa yang diprediksi, dengan fitur apa, dan bagaimana data dibagi agar tidak bocor?*

### 💡 Penjelasan

**Target — prediksi langsung (*direct multi-horizon*), bukan rekursif.**

```
R_h(i) = p(i+h) / p(i) − 1        ← return kumulatif h hari, diprediksi langsung
p̂(i+h) = p(i) · (1 + r̂)           ← evaluasi dilakukan di ruang LEVEL HARGA
```

*Rekursif* (memprediksi 1 hari, lalu hasilnya dipakai lagi) menumpuk galat di horizon
panjang. *Direct* melatih model khusus untuk setiap horizon. Evaluasi dilakukan di
ruang harga, bukan return, karena yang ditanyakan pengguna adalah "berapa rupiah",
dan satu persen galat di harga mahal berbeda maknanya di harga murah.

**Dua setup fitur, jumlahnya berbeda:**

| Setup | Isi | Jumlah |
|---|---|---:|
| **Univariat** | `lag_0` (return hari-i) + beberapa lag + `roll_mean` + `roll_std` | **15** |
| **Multivariat** | di atas + 7 fitur eksogen stasioner | **22** |

`lag_0` ikut serta karena pada origin `i`, return hari `i` **sudah diketahui** — jendela
rolling juga berakhir di hari-`i`, bukan `i−1`. Ini disamakan dengan jendela DL.

Untuk model deep learning, inputnya jendela `SEQ_LEN = 10` hari multi-channel, dan
z-score di-*fit* **hanya** pada data latih. Kanal eksogen di-*clip* ke ±5 z-score
(`EXO_CLIP`) agar *outlier* tidak mendominasi.

**Pembagian data kronologis + embargo.**

| h | Latih | Validasi | Validasi (embargo) | Uji | Uji mulai |
|---:|---:|---:|---:|---:|---|
| 1 | 3028 | 432 | 431 | 864 | 2023-04-11 |
| 5 | 3024 | 432 | 427 | 864 | 2023-04-05 |
| 10 | 3021 | 431 | 421 | 863 | 2023-03-30 |
| 20 | 3014 | 430 | 410 | 861 | 2023-03-20 |
| 30 | 3007 | 429 | 399 | 859 | 2023-03-08 |

**Apa itu embargo dan mengapa perlu?** Bayangkan origin validasi pada hari terakhir
periode validasi, dengan `h = 5`. Targetnya `p(i+5)` — dan `i+5` jatuh **di dalam
periode uji**. Kalau baris itu dipakai untuk *early stopping*, maka saat menentukan
model terbaik kita sudah membaca harga periode uji. Embargo membuang baris-baris itu:

```python
# src/pipeline.py
va_ok = [d for d in va_dates if d in F.index and pos_of[d] + h < te_pos[0]]
assert va_ok and all(pos_of[d] + h < te_pos[0] for d in va_ok), \
    'baris validasi untuk early stopping menyentuh periode test'
```

Pada h=30, **30 dari 429** baris validasi dibuang (429 → 399) — persis sebanyak `h`.
Pembuangannya meningkat seiring horizon, persis seperti yang diharapkan.

### 📁 Bukti di repo

```
src/pipeline.py          build_lgb_features (docstring: "Jumlah fitur: 15 / 22")
                          multihorizon_split, baris embargo + assert
outputs/results/tabel_split.csv
notebooks/04_feature_engineering.ipynb
```

---

# 06 — Model yang dipakai

### ❓ Pertanyaan yang dijawab

*Mengapa 21 model, apa itu DPformer, dan apakah tuning-nya adil?*

### 💡 Penjelasan

**10 machine learning:**

| | | | | |
|---|---|---|---|---|
| LightGBM | XGBoost | RandomForest | CatBoost | GBM |
| SVR | ExtraTrees | HistGB | AdaBoost | ElasticNet |

**11 deep learning:** LSTM · GRU · BiLSTM · BiGRU · Transformer · DLinear · NLinear ·
PatchTST · LSTM+Attention · GRU+Attention · **DPformer**

Banyak model dipakai bukan untuk "mencari yang paling banyak menang", melainkan untuk
**uji ketahanan**: kalau kesimpulannya hanya berlaku untuk satu arsitektur, ia lemah.
Kalau kesimpulan yang sama muncul dari 21 arsitektur berbeda, ia kuat.

**DPformer** adalah model yang dituning lebih serius. Konfigurasi **MOSSA**
(populasi **20**, **100** iterasi, fitness dihitung **hanya** dari baris validasi
ber-embargo, toleransi konvergensi 0,001, patience 20) menghasilkan `lr ≈ 0,0009` dan
`weight_decay ≈ 0,0075` — di sekitar default `Adam(lr=1e-3)` yang dipakai semua model DL.

**Apakah tuning hanya untuk DPformer itu tidak adil?** Pertanyaan yang tepat, dan
jawabannya diukur, bukan diandaikan: ablation MOSSA vs default pada 10 sel
(setup × horizon) menunjukkan **0 dari 10 lolos BH-FDR** (nilai `q` terendah 0,0713).
Artinya tuning yang ekstra itu **tidak memberi keuntungan nyata**, jadi kesimetrian
tuning bukan faktor yang memengaruhi kesimpulan. Ini harus disebut kalau ditanya —
lebih baik kita yang mengangkatnya lebih dulu.

**Kenapa mengacu ke Bao et al. (2026)?** Konfigurasi 10 baseline DL mengikuti
*Table 3* paper tersebut sedekat mungkin (rumus *as-published*, tidak dituning ulang),
supaya ada acuan eksternal yang bisa diperiksa reviewer.

### 📁 Bukti di repo

![Arsitektur DPformer](../outputs/figures/fig_arsitektur_dpformer.png)

![Konvergensi MOSSA](../outputs/figures/fig_mossa_konvergensi.png)

```
src/config.py                            MODELS_ML, DL_MODELS, MOSSA_* (pop 20, iter 100)
src/models_dl.py                         11 arsitektur DL
notebooks/05_dpformer.ipynb              2 setup × 5 horizon + tuning MOSSA
notebooks/06_dl_models.ipynb             10 baseline DL
notebooks/07_ml_models.ipynb             10 model ML
outputs/results/ablation_mossa_vs_default.csv   0/10 lolos BH
```

---

# 07 — Tujuh pagar integritas

### ❓ Pertanyaan yang dijawab

*Bagaimana membuktikan bahwa angka-angka di Bab 08–09 tidak dihasilkan oleh kebocoran
informasi?*

### 💡 Penjelasan

Ini bagian pembeda utama proyek. Kebocoran data (*lookahead*) **tidak menghasilkan
error** — ia menghasilkan **hasil yang terlalu bagus**, dan tidak ada yang memberi
peringatan. Satu-satunya cara menangkapnya adalah merusak data secara sengaja lalu
memastikan hasilnya tidak berubah.

Tujuh pagar, sesuai daftar kanonik di notebook 08:

| # | Pagar | Jenis |
|---:|---|---|
| 1 | Universum model = 21 (10 ML + 11 DL) | konsistensi |
| 2 | Data bersih: indeks, NaN, konsistensi fitur stasioner dengan level ber-lag, tanpa lookahead eksogen | konsistensi |
| 3 | Fitur latih/validasi **tidak berubah** ketika data uji dirusak | **mutation test** |
| 4 | Fitness DPformer (MOSSA) tidak bisa melihat uji; *early stopping* run final (ML & DL) tidak melihat harga periode uji | **mutation test** |
| 5 | Struktur walk-forward (embargo), skenario B (tanpa lookahead), SVR (ekuivarian terhadap skala target) | **mutation test** |
| 6 | Seluruh hasil utama memakai protokol/revisi/data yang sama dan **sejajar dengan data saat ini** | konsistensi |
| 7 | Diagnosis kurva loss seluruh 110 run DL: *skill* terhadap model nol + penanda overfit | diagnosis |

**Yang paling menarik dari pagar 3 dan 4 — angkanya:**

```
mae_val_asli            : 18904,92
mae_val_test_dirusak    : 18904,92      ← TIDAK BERUBAH, digit sama persis
mae_test_asli           : 63137,27
mae_test_test_dirusak   : 192339,70      ← justru BERUBAH drastis
```

Data uji dirusak habis-habisan: **fitness yang dipakai untuk memilih model tidak
bergerak sedikit pun**, sementara metrik uji berubah 3×. Artinya metrik uji memang
tidak pernah dipakai untuk memilih apa pun — pagar tersebut **sensitif** (ia bisa
mendeteksi kebocoran), bukan sekadar lolos kosong. Begitu juga untuk LightGBM:
iterasi *early stopping* tetap **8 → 8**.

> **Catatan jujur dari notebook itu sendiri:** *overfit/underfit tidak bisa dijamin nol
> hanya dari kode. Yang bisa dibuat ketat: mencegah kebocoran, early stopping hanya
> dari validasi, dan mengukur gejalanya secara jujur tanpa memilih model berdasarkan uji.*

### 📁 Bukti di repo

```
notebooks/08_validasi_integritas.ipynb   daftar 7 pagar + seluruh output [OK]
src/checks.py                            9 fungsi cek_* yang merealisasikan 7 poin
```

---

# 08 — Hasil dasar dan peringkat

### ❓ Pertanyaan yang dijawab

*Siapa model terbaik, dan seberapa banyak ia mengalahkan Drift?*

### 💡 Penjelasan

**Benchmark Drift** = `r̂ = μ_train`, rata-rata return pada periode latih. Ini benchmark
yang sangat menipu: ia tidak melihat apa pun kecuali rata-rata masa lalu, tetapi pada
deret yang hampir efisien, **susah dikalahkan**. Kalau model tidak bisa mengalahkannya,
model itu memang belum berguna.

**Model terbaik per horizon** (MAE di ruang harga; `% vs Drift` = persentase lebih baik):

| Horizon | Univariat | % vs Drift | Multivariat | % vs Drift |
|---:|---|---:|---|---:|
| 1 | PatchTST | 0,59 | LightGBM | 0,43 |
| 5 | SVR | 1,27 | Transformer | 1,97 |
| 10 | AdaBoost | 3,78 | AdaBoost | 3,87 |
| 20 | AdaBoost | 6,77 | AdaBoost | 4,76 |
| 30 | RandomForest | 3,21 | PatchTST | 2,43 |

**Peringkat rata-rata keseluruhan** (dihitung di seluruh 10 sel: 2 setup × 5 horizon):

| Model | `mean_rank` | Menang di… |
|---|---:|---|
| **AdaBoost** | **2,8** | 4 dari 10 sel |
| RandomForest | 6,1 | 1 |
| ElasticNet | 6,3 | 0 |

Dua hal menarik yang layak disebut lebih dulu daripada ditanya:

1. **Pemenangnya bukan model yang canggih.** AdaBoost — boosting klasik dari 1999 —
   mengungguli Transformer, PatchTST, dan DPformer. Ini konsisten dengan temuan di
   Bab 09: kalau sinyalnya tipis, model besar justru lebih mudah menghafal kebisingan.
2. **Selisihnya kecil.** Bahkan pemenang hanya 0,43–6,77% lebih baik dari Drift.
   Angka ini **belum** diuji statistik — itu pekerjaan Bab 09, dan hasilnya jauh lebih
   suram daripada tabel ini menunjukkan.

### 📁 Bukti di repo

![Hasil univariat](../outputs/figures/fig_hasil_univariat.png)

![Hasil multivariat](../outputs/figures/fig_hasil_multivariat.png)

![Perbaikan terhadap horizon](../outputs/figures/fig_perbaikan_vs_horizon.png)

```
outputs/results/model_terbaik_per_horizon.csv   tabel di atas
outputs/results/ringkasan_model.csv             mean_rank, wins
outputs/results/tabel_hasil_final.csv           210 baris lengkap
notebooks/09_hasil_dan_ranking.ipynb
```

---

# 09 — Uji statistik — dan cara menyajikan hasil negatif

> **Bab ini paling mungkin ditanya.** Kalau hanya sempat satu bab, baca yang ini.

### ❓ Pertanyaan yang dijawab

*Angka-angka Bab 08 itu nyata atau kebetulan — dan kalau ternyata hampir semua tidak
signifikan, apa kontribusinya?*

### 💡 Penjelasan

**Empat lapis uji, berurutan dari longgar ke ketat.**

**Lapis 1 — Diebold–Mariano dengan koreksi HLN.** Menguji apakah galat model signifikan
lebih kecil daripada Drift. 210 uji (21 model × 2 setup × 5 horizon).

**Lapis 2 — Koreksi Benjamini–Hochberg (BH-FDR).** Dengan 210 uji, pada taraf 5%
diharapkan **sekitar 10 temuan palsu** hanya karena kebetulan. Tanpa koreksi, "16 model
mengalahkan Drift" itu angka yang menyesatkan.

**Lapis 3 — Model Confidence Set (MCS) 90%.** Bukan memilih pemenang, tetapi
menanyakan: *set mana yang masih layak disebut pemenang?* Kalau setnya berisi semua
orang, tidak ada pemenang.

**Lapis 4 — Selang kepercayaan + MDE.** Berapa besar perbedaan yang **bisa** kita
deteksi dengan daya uji yang tersedia? Ini yang mengubah makna "tidak signifikan".

---

**Hasilnya, apa adanya:**

| Uji | n | Lolos |
|---|---:|---|
| DM vs Drift — `rel_mse < 1` | 210 | **115** |
| DM vs Drift — `p < 0,05` | 210 | 16 |
| DM vs Drift — **lolos BH-FDR** | 210 | **1** |
| Multivariat vs univariat — `rel_mse < 1` | 105 | 40 |
| Multivariat vs univariat — `p < 0,05` | 105 | 5 |
| Multivariat vs univariat — **lolos BH-FDR** | 105 | **0** |

> **Catatan soal "keluarga" (famili) uji.** Koreksi BH bergantung pada keluarga uji yang
> dikoreksi. Hasil di atas memakai dua keluarga: **1 dari 210** dihitung dengan keluarga
> per setup×horizon (`q_bh_setup_h`), sedangkan **0 dari 105** memakai keluarga seluruh
> 105 uji makro (`q_bh`). Kalau makro dikoreksi per horizon, ada **2** yang lolos
> (GBM dan HistGBM di h=1, `q_bh_h` = 0,0138) — tetapi keduanya jatuh begitu keluarga
> diperluas ke seluruh 105 uji. Pilihan keluarga ini harus disebut apa adanya, bukan
> disembunyikan: ia satu-satunya hal yang mengubah "0" menjadi "2".

**Satu-satunya yang bertahan:**

```
setup       model          horizon   rel_mse   p_one_hln   q_bh
univariat   transformer         1     0,9950     0,00195   0,0410
```

Sisanya tenggelam. MCS mengatakan hal yang sama dengan cara berbeda: **209 dari 220**
sel masuk *confidence set*, dan **Drift sendiri masuk di seluruh 10 sel** (2 setup ×
5 horizon) — artinya Drift tidak pernah bisa dikesampingkan.

---

**Lalu, daya ujinya bagaimana?**

Ini bagian yang paling sering disalahpahami. Median **observasi independen** per
horizon (karena data harian saling berkorelasi, `n` efektif jauh lebih kecil dari jumlah
baris):

| Horizon | Baris uji | `n_eff` median | MDE vs Drift | MDE vs Drift **bergulir** |
|---:|---:|---:|---:|---:|
| 1 | 864 | 864,0 | 3,42% | 4,16% |
| 5 | 864 | 172,8 | 4,48% | 4,29% |
| 10 | 863 | 86,3 | 5,09% | 4,83% |
| 20 | 861 | 43,1 | 6,55% | **48,12%** |
| 30 | 859 | **28,6** | 5,79% | **70,42%** |

*MDE = Minimum Detectable Effect — perbedaan terkecil yang masih bisa dideteksi.*

Artinya: pada h=30 terhadap Drift bergulir, kita baru bisa mendeteksi perbedaan
**sebesar 70%**. Kalaupun ada model yang lebih baik 10%, **kita tidak punya cukup data
untuk membuktikannya**. Karena itu:

> **"Tidak signifikan" ≠ "tidak ada efek." Untuk horizon panjang, itu berarti
> "daya ujinya tidak cukup."**

---

**Cara menyajikannya di depan dosen — tiga kalimat yang sudah teruji:**

1. *"Kami tidak menyembunyikan hasil negatifnya. Dari 210 uji, hanya 1 yang lolos
   koreksi uji berganda, dan angka itu harus dibaca apa adanya."*
2. *"Tetapi kami juga mengukur daya ujinya, dan di horizon 30 hari MDE kami 70% —
   jadi klaim 'tidak ada efek' pun tidak bisa kami buktikan. Jawabannya adalah
   'belum terbukti', bukan 'tidak ada'."*
3. *"Yang positif justru muncul di tempat yang tidak kami duga: informasi penutupan
   pasar hari-t membantu tepat di h=1, dan plasebo t+1 gagal — jadi itu bukan
   artefak. Itu ada di bab 11."*

### 📁 Bukti di repo

![DM vs Drift](../outputs/figures/fig_dm_vs_drift.png)

![Multivariat vs univariat](../outputs/figures/fig_dm_multi_vs_uni.png)

![Model Confidence Set](../outputs/figures/fig_mcs.png)

![Selang kepercayaan dan MDE](../outputs/figures/fig_ci_ringkasan.png)

```
outputs/results/dm_vs_drift_detail.csv                210 baris, kolom q_bh_setup_h
outputs/results/dm_multivariat_vs_univariat_detail.csv 105 baris, kolom q_bh
outputs/results/mcs_keanggotaan.csv                   22 entri × 10 sel = 209 True
outputs/results/ci_ringkasan_per_horizon.csv          n_eff, MDE per horizon
notebooks/10_uji_statistik.ipynb
notebooks/17_selang_kepercayaan_vs_drift.ipynb
```

---

# 10 — Interpretasi (SHAP)

### ❓ Pertanyaan yang dijawab

*Model yang sudah dilatih, sebenarnya memakai apa untuk memutuskan?*

### 💡 Penjelasan

SHAP dijalankan pada **tiga model wakil** di setup multivariat h=20: **DPformer**,
**GRU+Attention** (wakil DL), dan **AdaBoost** (wakil ML). Untuk model DL dipakai
`GradientExplainer`, untuk ML `PermutationExplainer`, dengan 50 sampel latar belakang
dan 100 sampel yang dijelaskan.

Temuannya **berbeda jauh antara model DL dan model ML**, dan perbedaan itu justru
mengkonfirmasi Bab 08:

| Variabel | DPformer | GRU-Attn | **AdaBoost** |
|---|---:|---:|---:|
| US 10Y | **22,4%** | **20,0%** | 1,2% |
| DXY | 17,9% | 13,6% | 1,2% |
| S&P 500 | 14,9% | 12,1% | 16,2% |
| IHSG | 14,0% | 15,9% | 3,0% |
| **Emas (lag & rolling)** | 11,6% | 10,5% | **75,8%** |
| BI-Rate | 8,4% | 9,3% | 0,0% |
| USD/IDR | 7,8% | 11,9% | 2,4% |
| FFR | 3,2% | 6,6% | 0,2% |

**Cara membaca:** model DL **mengandalkan makro global** (US 10Y + DXY + S&P 500
≈ 55%), sedangkan AdaBoost **hampir sepenuhnya mengandalkan harga emasnya sendiri**
(75,8%). Padahal AdaBoost itu model yang peringkatnya justru lebih baik.

Ini konsisten dengan temuan Bab 09 bahwa fitur makro tidak menambah informasi secara
signifikan: model yang paling bergantung pada makro justru **bukan** yang terbaik.

### 📁 Bukti di repo

![SHAP perbandingan tiga model wakil](../outputs/figures/fig_shap_perbandingan.png)

```
outputs/results/shap/shap_ringkasan.json                 wakil, metode, completeness_err
outputs/results/shap/shap_porsi_per_model.csv            tabel di atas
outputs/results/shap/shap_*_importance.csv               detail per model
notebooks/11_interpretasi_shap.ipynb
```

---

# 11 — Ketahanan: seed, ablation, walk-forward, plasebo

### ❓ Pertanyaan yang dijawab

*Apakah temuan di bab-bab sebelumnya bertahan ketika berubah periode, seed, atau
konfigurasi — dan bisakah dibuktikan bukan artefak?*

### 💡 Penjelasan

Empat serangan berbeda terhadap hasil yang sama.

**1. Seed acak (notebook 13).** 6 model DL × 2 setup × 5 horizon × 5 seed (42–46)
= **60 kombinasi**. Hasil: **CV MAE 0,034% – 1,30%**, median **0,29%**. Peringkat
jauh lebih stabil daripada selisih antar-model — jadi peringkat di Bab 08 bukan
keberuntungan seed.

**2. Ablasi fitur (notebook 14).** Menjawab "fitur mana yang benar-benar dipakai" —
termasuk ablation MOSSA vs default yang disebut di Bab 06 (**0/10 lolos BH**).

**3. Walk-forward (notebook 16).** Daripada mengandalkan satu periode uji, data
digulirkan: **6 fold × 260 hari**, mencakup 2020-08 s.d. 2026-07, diuji pada h=5 dan
h=20. Hasil: **0 model lolos BH-FDR** (`q_bh` terendah 0,794; pada famili per horizon
`q_bh_h` terendah 0,397). Ini replikasi yang jujur dari Bab 09 di luar periode uji utama.

**4. Skenario B dan plasebo — ini temuan positifnya.**

*Skenario B* memakai informasi penutupan pasar **hari-t**, yaitu informasi yang sudah
benar-benar tersedia saat ramalan diambil. Bandingkan dengan *lag-1* (baseline) dan
*plasebo t+1* (tanggal diputar mundur — seharusnya **tidak** punya kekuatan apa pun).

| Anak tangga, h=1 | Median rasio vs lag-1 | Lebih baik | Lolos BH |
|---|---:|---:|---:|
| **Hari-t** (skenario B) | **0,9266** | 10 dari 10 | **9 dari 10** |
| **Plasebo t+1** (harusnya gagal) | 1,0086 | 2 dari 10 | **0 dari 10** |

Dan dari tabel skenario B yang lebih luas (n = 12): median `rel_mse` **0,9267**,
**11 dari 12 lolos BH**.

Dua kecocokan ini yang membuat temuannya kuat: **perbaikan nyata muncul di hari-t, dan
hilang total saat tanggalnya diputar mundur.** Untuk h ≥ 5, semuanya menghilang
(rasio ≈ 1,00, 0 lolos BH) — jadi manfaatnya memang hanya di horizon 1 hari.

> **Inilah kalimat pembelaan yang paling kuat:** bukan "model kami lebih baik", melainkan
> *"kami menemukan satu hal yang benar-benar terukur, kami uji dengan plasebo, dan
> plasebonya gagal — sedangkan untuk sisanya kami laporkan bahwa kami tidak punya
> cukup daya uji untuk mengklaim apa pun."*

### 📁 Bukti di repo

![Walk-forward gabungan](../outputs/figures/fig_walkforward_gabungan.png)

![Skenario B vs Drift](../outputs/figures/fig_skenario_b_dm.png)

![Tangga waktu informasi + plasebo](../outputs/figures/fig_tangga_waktu.png)

```
outputs/results/multiseed_ringkasan.csv            60 baris, kolom CV_pct
outputs/results/ablation_ringkasan.csv              ablation fitur
outputs/results/ablation_mossa_vs_default.csv       0/10 lolos BH
outputs/results/walkforward_gabungan.csv            6 fold, 0 lolos BH
outputs/results/tangga_waktu_ringkasan.csv          hari-t / lag1 / plasebo t+1
outputs/results/skenario_b_vs_multivariat_lag1.csv  11/12 lolos BH di h=1
notebooks/13, 14, 15, 16, 19
```

---

# 12 — Kesimpulan, kontribusi, dan batasan

### ❓ Pertanyaan yang dijawab

*Jadi, apa kontribusi proyek ini?*

### 💡 Penjelasan

**Satu paragraf yang bisa dibaca apa adanya:**

> Pada horizon 1 hari, terdapat sinyal nyata dan terukur dari informasi penutupan pasar
> hari-t: perbaikan median 7,3% terhadap baseline, signifikan pada 9 dari 10 model setelah
> koreksi BH, dan hilang total pada kontrol plasebo. Pada horizon 5–30 hari, dan untuk
> seluruh tujuh fitur makro global, tidak ada perbedaan yang dapat dibuktikan secara
> statistik terhadap benchmark Drift — tetapi pada h=30 daya uji mencapai MDE 70%,
> sehingga ketiadaan bukti di sana tidak boleh dibaca sebagai bukti ketiadaan.

**Tiga kontribusi metodologis** yang tidak bergantung pada hasil:

1. **Tujuh pagar integritas dengan mutation testing.** Bukti numerik bahwa *early
   stopping* tidak bisa melihat data uji (`mae_val` identik digit demi digit saat data
   uji dirusak, sementara `mae_test` berubah 3×). Ini bisa diterapkan ke proyek lain.
2. **Pelaporan MDE bersama setiap hasil negatif.** Tanpa ini, "tidak signifikan" terbaca
   sebagai "tidak ada", padahal dua hal itu berbeda.
3. **Konsistensi lapis:** setiap klaim statistik melewati DM → BH-FDR → MCS →
   selang kepercayaan, sehingga tidak ada klaim yang berdiri di atas satu uji saja.

**Tiga batasan yang harus disebut lebih dulu** (jangan menunggu ditanya):

| # | Batasan | Angka pasti |
|---|---|---|
| 1 | **Daya uji terbatas di horizon panjang** | `n_eff` median h=30 = **28,63**; MDE vs Drift bergulir h=30 = **70,42%** |
| 2 | **Bukan harga pasar** | harga postingan administratif; **22,47%** hari di periode latih return nol (puncak 2014: 40,6%) |
| 3 | **Tidak ada XAU/USD** | fitur hanya DXY, S&P 500, USD/IDR, IHSG, US 10Y, FFR, BI-Rate |
| 4 | **Satu aset, satu negara** | kesimpulan berlaku untuk emas Antam (Rp/gram) saja |
| 5 | **Periode uji adalah satu rezim reli** | 2023–2026, harga 1,04 → 3,17 juta; 2010–2020 hanya data latih |
| 6 | **Sumber data emas tidak resmi** | dataset komunitas — perlu validasi silang ke situs resmi |

### 📁 Bukti di repo

```
../CHANGELOG.md            perubahan dan stempel versi
../README.md bagian 9      batasan yang diketahui (termasuk risiko ukuran file .h5)
outputs/results/MANIFEST.json  ringkasan mesin-baca seluruh hasil
```

---

# Lampiran A — Peta file repo

Kalau ditanya *"ini ada di mana?"*, jawabannya:

| Yang ditanya | Lokasi |
|---|---|
| Konfigurasi semua eksperimen | `src/config.py` |
| Logika fitur, split, embargo | `src/pipeline.py` |
| Arsitektur 11 model DL | `src/models_dl.py` |
| **Satu-satunya** jalur pelatihan & penyimpanan hasil | `src/runners.py` |
| 7 pagar integritas | `src/checks.py` → `notebooks/08_*.ipynb` |
| Uji statistik (DM, BH, MCS, PT) | `src/stats.py` → `notebooks/10_*.ipynb` |
| SHAP | `src/explain.py` → `notebooks/11_*.ipynb` |
| Semua figur (69 PNG + 69 PDF) | `outputs/figures/` |
| Semua tabel hasil (48 CSV) | `outputs/results/` |
| 210 run + prediksi + kurva loss | `outputs/results/runs/` |
| Bobot model `.weights.h5` | `outputs/results/models/` |
| Data master | `data/processed/data_gabungan.csv` |
| Notebook urut 00–19 | `notebooks/` |

**Alur jalannya satu eksperimen:**
`notebook → src/runners.py → src/pipeline.py (fitur/split) → src/models_*.py → simpan JSON/CSV`
— notebook tidak pernah menghitung sendiri, sehingga eksperimen utama, ablation, dan
walk-forward tidak mungkin berbeda diam-diam.

---

# Lampiran B — Glosarium

| Istilah | Arti |
|---|---|
| **Direct multi-horizon** | Setiap horizon (1/5/10/20/30) punya model khusus, bukan diramalkan berurutan. Menghindari penumpukan galat. |
| **Embargo** | Baris validasi yang target `i+h`-nya jatuh di periode uji **dibuang** dari *early stopping*. |
| **Drift (benchmark)** | `r̂ = μ_train` — memprediksi return rata-rata periode latih untuk semua hari. Sangat sulit dikalahkan. |
| **Drift bergulir** | Drift yang diperbarui tiap 250 hari — versi adaptif, lebih menantang. |
| **Diebold–Mariano (DM)** | Uji apakah galat model signifikan lebih kecil daripada pembanding. |
| **HLN** | Koreksi Harvey–Newbold–Leybourne: memperbaiki DM yang bias ke atas pada data ber-*overlap*. |
| **BH-FDR** | Koreksi *Benjamini–Hochberg* untuk uji berganda. Tanpa ini, 210 uji pada taraf 5% menghasilkan ≈10 temuan palsu. |
| **MCS** | *Model Confidence Set* — set model yang masih layak disebut pemenang. Kalau berisi semua orang → tidak ada pemenang. |
| **MDE** | *Minimum Detectable Effect* — perbedaan terkecil yang masih bisa dideteksi dengan daya uji yang tersedia. |
| **`n_eff`** | Observasi independen efektif; jauh lebih kecil dari jumlah baris karena data saling berkorelasi. |
| **`rel_mse`** | MSE model ÷ MSE pembanding. **< 1 = lebih baik.** |
| **Plasebo t+1** | Kontrol: tanggal input diputar mundur sehari. Seharusnya kehilangan kekuatan — kalau tidak, ada artefak. |
| **Skenario B** | Ramalan memakai penutupan pasar **hari-t** (informasi yang sudah pasti tersedia). |
| **Stasioneritas / ADF** | Uji apakah deret punya *unit root*. Deret non-stasioner wajib ditransformasi sebelum dipakai. |
| **Mutation testing** | Merusak data uji sengaja, lalu memastikan hasil latih **tidak berubah** — bukti tidak ada kebocoran. |
| **Overfit / model nol** | Model nol = selalu memprediksi rata-rata target latih. *Skill* diukur relatif terhadapnya. |

---

# Lampiran C — FAQ: kemungkinan pertanyaan dosen

> **Ini bagian yang paling bernilai saat bimbingan.** Baca sekali sebelum pertemuan.

### Q1 — *"Kalau hasilnya hampir semua tidak signifikan, apa kontribusinya?"*

Jawab tiga langkah: (a) kami **tidak menyembunyikannya** — 1 dari 210 lolos BH, angkanya
segitu; (b) kami **mengukur daya ujinya**, dan MDE h=30 vs Drift bergulir adalah 70% —
jadi "tidak ada efek" pun tidak bisa kami buktikan, jawaban yang jujur adalah **"belum
terbukti"**; (c) ada **satu temuan positif yang teruji plasebo** — informasi hari-t di
h=1, 9/10 lolos BH, dan plasebo t+1 gagal total.

### Q2 — *"Kenapa XAU/USD (harga emas dunia) tidak dipakai?"*

Kejujuran dulu: **memang tidak dipakai**, dan itu keterbatasan yang kami catat di
README. Tujuh fitur makro hanya DXY, S&P 500, USD/IDR, IHSG, US 10Y, FFR, BI-Rate.
DXY dipakai sebagai *proxy* dolar. Kalau diminta, ini adalah tambahan paling bernilai
untuk iterasi berikutnya.

### Q3 — *"Kenapa hanya DPformer yang dituning? Tidak adil?"*

Pertanyaan bagus, dan **sudah kami ukur, bukan diandaikan**: ablation MOSSA vs default
pada 10 sel menunjukkan **0/10 lolos BH-FDR** (`q` terendah 0,0713). Jadi tuning ekstra
itu tidak memberi keuntungan nyata, dan kesimetrian tuning bukan faktor yang
mempengaruhi kesimpulan.

### Q4 — *"Kenapa perlu 21 model? Bukannya cuma bikin ribet?"*

Justru sebaliknya: banyak model dipakai sebagai **uji ketahanan**. Kalau kesimpulan hanya
berlaku untuk satu arsitektur, ia lemah. Kalau kesimpulan yang sama muncul dari 21
arsitektur — termasuk ElasticNet yang sederhana sekaligus Transformer yang kompleks —
ia kuat. Selain itu, tanpa 21 model, koreksi uji berganda (BH-FDR) tidak punya makna.

### Q5 — *"Kenapa evaluasinya di ruang harga, bukan return?"*

Karena yang ditanyakan pengguna adalah "berapa rupiah harganya". Satu persen galat di
harga 400 ribu berbeda maknanya dengan 1 persen di harga 3 juta. Metrik return tetap
dihitung (`metrik_return.csv`), tetapi utamanya di harga.

### Q6 — *"Kalau datanya harga administratif, apa gunanya?"*

Ini justru yang membuat menarik: **22,47% hari di periode latih tidak bergerak**, jadi
model harus belajar **kapan harga bergerak**, bukan hanya seberapa jauh — dan itu
berbeda dari deret harga pasar biasa. Kami laporkan ini sebagai temuan, bukan menyembunyikannya.

### Q7 — *"Apa itu embargo, dan kenapa repot-repot?"*

Lihat Bab 05. Ringkasnya: baris validasi terakhir punya target `i+h` yang jatuh di
periode uji. Kalau tetap dipakai untuk *early stopping*, saat memilih model terbaik kita
sudah membaca harga uji. Pada h=30, 30 dari 429 baris dibuang. Tanpa ini, hasilnya akan
**terlalu bagus dan tidak ada yang menyadarinya**.

### Q8 — *"Kenapa multivariat tidak lebih baik dari univariat?"*

Dari 105 uji, 40 punya rasio < 1 dan 5 mencapai `p < 0,05`, tetapi **0 lolos BH-FDR**
(keluarga = seluruh 105 uji). Yang paling mendekati adalah GBM (0,915) dan HistGBM (0,910)
di h=1 — `q` = 0,069, tetap tidak lolos. SHAP memberi petunjuk: model DL bergantung 55%
pada makro global, padahal **model terbaik (AdaBoost) 75,8%-nya justru dari harga emas
sendiri**. Ada kemungkinan makro global sudah terwakili oleh DXY/return emas sendiri.

### Q9 — *"Ini kan pakai AI, sepenuhnya otomatis?"*

Seluruh pelatihan/analisis lewat `src/runners.py`, **satu jalur** untuk eksperimen utama,
ablation, walk-forward, dan uji seed. Notebook tidak menghitung sendiri. Ditambah 7 pagar
integritas di Bab 07 — termasuk *mutation testing* yang bisa ditunjukkan angkanya.

### Q10 — *"Bisa dipakai untuk trading?"*

Belum, dan itu disimpulkan sendiri: pada h=5–30 tidak ada model yang signifikan
mengalahkan Drift, dan di MCS **Drift masuk di seluruh 10 sel**. Kalau dipaksa, satu-satunya
yang layak diuji lanjut adalah **h=1 dengan informasi penutupan hari-t** — asumsi jam
kerja dan biaya transaksi harus dimasukkan dulu.

### Q11 — *"Kenapa walk-forward hanya diuji h=5 dan h=20?"*

6 fold × 260 hari sudah menghabiskan data (2020-08 s.d. 2026-07). h=5 dan h=20 dipilih
sebagai perwakilan horizon pendek dan panjang. Ini keterbatasan yang kami catat, bukan
keputusan yang disembunyikan.

### Q12 — *"Apa bedanya Drift biasa dan Drift bergulir?"*

Drift biasa = rata-rata **seluruh** periode latih, tetap selamanya. Drift bergulir =
diperbarui tiap 250 hari mengikuti data terkini, sehingga **adaptif**. Karena itu MDE
terhadap Drift bergulir jauh lebih besar (70% di h=30) — ia adalah pembanding yang jauh
lebih menantang.

### Q13 — *"Kenapa semua model DL pakai Adam lr=1e-3?"*

Keseragaman supaya perbandingan adil dan hasilnya bisa direproduksi. Yang dikombinasikan
adalah `lr` dan `weight_decay` (grid 3×4). Patokannya Bao et al. (2026), *Table 3*.

### Q14 — *"Bagaimana kalau mau menambah variabel / mengulang?"*

Jalannya di Lampiran D. Yang harus dijaga: tiga stempel di setiap run JSON —

```
protocol  = v9-21model-dpformer-bao-aligned-honest-mossa-shap-leakage-safe-embargo
rev       = ml-r2
data_hash = 7aea8db80d44
```

Kalau salah satunya berubah, run dianggap **basi** dan otomatis dihitung ulang — jadi
tidak mungkin mencampur hasil lama dengan data baru.

---

# Lampiran D — Cara menjalankan ulang

Detail teknis lengkap ada di [`../README.md`](../README.md) §6. Ringkasnya:

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt        # 17 paket langsung terkunci

# 2. jalankan notebook berurutan 00 → 19
jupyter notebook
```

```powershell
# 3. mode cepat (smoke test, beberapa menit)
$env:TA_QUICK = "1"      # horizon [5,20], 2 epoch, output ke outputs_quick/

# 4. variabel lain
$env:TA_LANG = "id"      # bahasa label figur: id | en
$env:TA_SEED = "42"
$env:TA_THREADS = "8"
```

Sebagian besar notebook **tidak mengulang pekerjaan yang sudah ada** — `src/runners.py`
memeriksa `protocol`, `rev`, dan `data_hash` pada tiap run, dan hanya run basi yang
dihitung ulang.

**Stempel saat freeze `v1.0.0`:** 210/210 run CURRENT, 0 basi · 20/20 notebook
tereksekusi, 0 error output · 69 PNG + 69 PDF.

---

<sub>Untuk bimbingan · `panduan_bimbingan/` · Versi 1.0.0 · Protokol `v9` · data_hash `7aea8db80d44`
<br>File ini <strong>belum di-commit</strong> ke git (lihat `git status`).</sub>
