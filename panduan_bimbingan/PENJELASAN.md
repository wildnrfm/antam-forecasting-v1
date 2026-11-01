# PENJELASAN — Dari Awal Sampai Akhir

**Catatan belajar pribadi: memahami proyek ini dari awal sampai akhir sebelum bimbingan.**
Saat bimbingan, yang ditunjukkan ke dosen adalah [`ringkasan_eksperimen.xlsx`](ringkasan_eksperimen.xlsx)
(lembar `Jawaban`), dan yang diucapkan mengikuti [`NASKAH_BIMBINGAN.md`](NASKAH_BIMBINGAN.md).

Proyek: peramalan harga jual emas Antam · 21 model · 210 eksperimen
Versi dokumen: **1.0.2** (hasil dibekukan pada 1.0.0) · Protokol `v9-21model-dpformer-bao-aligned-honest-mossa-shap-leakage-safe-embargo`
Sidik jari data: `7aea8db80d44` · Python 3.13.0

> **Rujukan teknis:** [`../README.md`](../README.md)

---

## Cara memakai dokumen ini

| Kalau waktu… | Baca |
|---|---|
| **5 menit** | Bab **12** — satu paragraf kesimpulan, tiga kontribusi, dan tujuh batasan yang harus disebut lebih dulu |
| **30 menit** | Bab **01** → **05** → **09** → **12** (masalah, desain, uji statistik, kesimpulan) |
| **2 jam / mendalam** | Seluruh dokumen urut, ditambah Lampiran **C** (FAQ pertanyaan dosen) |
| **Butuh angka cepat** | [`ringkasan_eksperimen.xlsx`](ringkasan_eksperimen.xlsx) — buka lembar `Jawaban` (hasil singkat, tabel per horizon, dan tanya-jawab untuk "gimana hasilnya?"); lembar lain memuat tabel dan grafik per topik, semuanya dihitung dari CSV hasil |
| **Mau latihan bicara** | [`NASKAH_BIMBINGAN.md`](NASKAH_BIMBINGAN.md) — naskah lisan untuk bimbingan: alur ±3–4 menit, 13 jawaban singkat untuk pertanyaan yang mungkin muncul, dan kalimat yang sebaiknya dihindari |

Setiap bab memakai kerangka seragam:

- **❓ Pertanyaan yang dijawab** — inti bab dalam satu kalimat
- **💡 Penjelasan** — kenapa langkah ini diambil, apa alternatifnya, kenapa alternatifnya ditolak
- **📁 Bukti di repo** — file, tabel, dan figur yang bisa langsung ditunjukkan

![Kerangka penelitian](../outputs/figures/fig_kerangka_penelitian.png)

*Versi lebar yang lebih terbaca — dengan pembeda tiga jenis langkah: **A** = mengikuti
Bao et al. (2026), **B** = adaptasi, **C** = kontribusi penelitian ini — ada di
[`../diagram/alur_penelitian_lebar.png`](../diagram/alur_penelitian_lebar.png)
(sumber teks: `../diagram/alur_penelitian_mermaid.txt`).*

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
**mungkin** harga emas Antam bisa ditebak dari data makro lampau.

Pertanyaan penelitiannya bukan "apakah bisa ditebak", melainkan **seberapa yakin kita
bisa mengklaim demikian**. Dua hal membuat pertanyaan itu lebih sulit dari kelihatannya:

1. **Harga Antam adalah harga postingan administratif.** Bukan harga transaksi yang
   bergerak setiap detik. Harganya tetap sampai ada pengumuman baru — 22,47% hari pada
   periode latih tidak bergerak sama sekali (return nol). Deret ini punya mikrostruktur
   yang berbeda dari deret harga pasar biasa.
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
2026-08-03, 16 variabel (tabel master 4.325 × 17 kolom termasuk tanggal), tanpa *missing*
dan tanpa duplikat. Kalender berisi **seluruh hari kerja Senin–Jumat** (865 hari untuk tiap
hari dalam pekan), sehingga selisih tanggal antar-baris hanya **1 hari** (3.459 kali) atau
**3 hari** (865 kali; Jumat → Senin) — hari libur nasional tidak menghasilkan celah, karena
hari kerja tanpa pengumuman harga Antam (**198** dari 4.326 hari kerja di berkas mentah,
4,6%) diisi *forward-fill* (Bab 03).

**Tujuh variabel makro** — bukan level, melainkan **transformasi stasioner**:

| Kelompok | Variabel | Transformasi | Kenapa |
|---|---|---|---|
| Log-return | DXY · S&P 500 · USD/IDR · IHSG | `log(x_t / x_{t-1})` | levelnya *trending* dan keluar dari rentang latih |
| Selisih poin | US 10Y · FFR · BI-Rate | `x_t − x_{t-1}` (poin %) | tingkat suku bunga naik/turun, bukan bergerak bebas |

**Kenapa di-lag sejak tahap penggabungan, bukan saat pemodelan?** Ramalan pada origin `i`
dibuat setelah harga Antam hari `i` diumumkan (pagi hari). Pada saat itu penutupan pasar
hari `i` — IHSG sore hari, pasar AS dini hari berikutnya — belum ada, jadi kalau nilai
hari `i` dipakai apa adanya, model akan membaca masa depan. Karena itu seluruh fitur
makro digeser 1 hari kerja: baris `i` memuat penutupan hari `i−1`. Konvensi ini
**konservatif** (tidak memakai penutupan hari `i` sama sekali), dan Skenario B (Bab 11)
menguji apa yang berubah bila penutupan hari `i` sudah diketahui. Menunda lag ke tahap
fitur rawan terlupa; menaruhnya di tahap penggabungan membuat seluruh eksperimen
otomatis aman.

Pembacaan level-vs-stasioner juga penting: pada periode uji, **87,3%** baris level
S&P 500 (rentang seluruh data 1.022–7.609, ×7) dan **100%** baris level FFR (0,05–5,33%,
×100) berada di luar rentang periode latih; setelah transformasi stasioner hanya 0,1% dan
0% (`eksogen_level_vs_stasioner.csv`). Model yang dilatih pada level akan banyak
mengekstrapolasi.

> **Yang sengaja tidak dipakai:** harga emas dunia (XAU/USD), LBMA, COMEX. Ini batasan
> yang harus diakui lebih dulu — lihat Bab 12 dan Lampiran C.

### 📁 Bukti di repo

```
data/raw/                        Antam + 8 berkas makro (7 terpakai; id_tbill tidak dipakai) + cache fallback API
data/processed/data_gabungan.csv tabel master: 4.325 × 17 (tanggal + 16 variabel), fp 7aea8db80d44
outputs/results/eksogen_level_vs_stasioner.csv   % baris uji di luar rentang latih (level vs stasioner)
src/pipeline.py                  EXO_LEVEL_COLS, EXO_LOGRET_COLS, EXO_DIFF_COLS, EXO_CLIP=5.0
notebooks/00_unduh_indikator_makro.ipynb
notebooks/02_penggabungan_data.ipynb
```

---

# 03 — Pembersihan dan penggabungan

### ❓ Pertanyaan yang dijawab

*Bagaimana mengisi hari kerja tanpa pengumuman harga (termasuk hari libur nasional) tanpa diam-diam menyuntikkan informasi masa depan?*

### 💡 Penjelasan

Ketika harga tidak diumumkan, nilainya harus diisi. Ada dua cara:

| Cara | Cara kerja | Masalah |
|---|---|---|
| **Interpolasi** | rata-rata harga sebelum dan sesudahnya | ❌ Memakai harga **berikutnya** = *lookahead* |
| **Forward-fill** | bawa terus harga hari sebelumnya | ✅ Hanya memakai masa lalu |

Forward-fill dipilih karena konsisten dengan kenyataan operasional: kalau Antam tidak
mengumumkan hari itu, yang berlaku tetap pengumuman terakhir. Ini terjadi pada **198 dari
4.326 hari kerja** (4,6%) di berkas mentah. Konsekuensinya, sebagian hari return-nol di
Bab 04 berasal dari pengisian ini (±25% di periode latih), bukan dari pengumuman yang
harganya sama (±75%) — dan angka itu sendiri justru menjadi temuan (lihat Bab 04).

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
panduan_bimbingan/skrip/audit_data_mentah.py   hitungan 198 hari tanpa pengumuman dan komposisi return nol
                                               (audit berkas mentah; belum ada notebook yang menghasilkannya)
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

**Harga yang tidak bergerak.** Sebagian hari menghasilkan return tepat nol (22,47% di
periode latih), dan porsinya turun tajam seiring waktu:

| Periode | Porsi return nol | SD harian | Rentang harga |
|---|---:|---:|---|
| Latih (2010–2021) | **22,47%** | 0,69% | 0,39 – 1,06 jt |
| Validasi (2021–2023) | 12,79% | 0,65% | 0,91 – 1,06 jt |
| Uji (2023–2026) | 7,20% | **0,97%** | **1,04 – 3,17 jt** |

Puncaknya tahun 2014: **40,6%** hari tidak bergerak. Sebagian besar ini memang sifat
harga postingan: ±75% hari return-nol di periode latih adalah hari **dengan** pengumuman
yang harganya sama dengan sebelumnya. Tetapi ±25% adalah hari kerja **tanpa** pengumuman
yang diisi *forward-fill* (Bab 03) — bukan cacat data, namun juga bukan semuanya "harga
yang tidak berubah" (audit: `panduan_bimbingan/skrip/audit_data_mentah.py`). Intinya, model
harus belajar meramal **kapan harga bergerak**, bukan hanya seberapa jauh.

**Regim yang berubah.** Periode uji didominasi satu reli besar yang disusul koreksi (harga
1,04 juta → puncak 3,17 juta pada 2026-01-29, lalu turun 17,6% sampai akhir data;
volatilitas 0,97%/hari vs 0,65% di validasi), dan rata-rata return harian periode uji
≈ 3,7× rata-rata periode latih. Artinya latih dan uji berada di rezim berbeda — ini
**mungkin** salah satu alasan hasil di Bab 09 sebagian negatif; pengaruhnya pada
perbandingan dengan Drift dibahas di Bab 08 (butir 3).

**Waktu informasi (lead–lag).** Dihitung korelasi return emas hari *t* dengan penutupan
setiap variabel makro pada hari *t−2*, *t−1*, dan *t* (`leadlag_waktu_informasi.csv`).
Hanya **3 dari 7** variabel (DXY, USD/IDR, US 10Y) yang korelasi mutlaknya terbesar pada
*t−1* — DXY −0,288 dan US 10Y −0,233, dibanding −0,062 dan −0,017 pada *t*; keempat
variabel lain (S&P 500, IHSG, FFR, BI-Rate) puncaknya di *t−2*, dengan |r| ≤ 0,084.

Cara membacanya untuk desain: protokol utama meramal return *t+1* dari penutupan hari
*t−1* (pada tabel ini sama dengan baris *t−2*: DXY +0,094, US 10Y −0,022 — hampir tanpa
korelasi), sedangkan sinyal terkuat ada pada penutupan hari *t* (baris *t−1*: DXY −0,288,
US 10Y −0,233). Itulah yang diuji Skenario B (Bab 11). Jadi tabel ini **bukan** konfirmasi
bahwa lag-1 menempatkan sinyal pada tempatnya, melainkan **motivasi** Skenario B — dan
karena skenario itu dirancang setelah melihat korelasi pada seluruh sampel, temuannya
bersifat eksploratif.

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
panjang. *Direct* melatih model khusus untuk setiap horizon. Metrik titik (MAE, RMSE,
MAPE) dihitung di ruang harga, karena yang ditanyakan pengguna adalah "berapa rupiah",
dan satu persen galat di harga mahal berbeda maknanya di harga murah. Uji
Diebold–Mariano (Bab 09) memakai galat kuadrat **return** (`rel_mse_ret`); versi di ruang
harga (`rel_mse_price`) ada di berkas yang sama sebagai pembanding.

**Dua setup fitur, jumlahnya berbeda:**

| Setup | Isi | Jumlah |
|---|---|---:|
| **Univariat** | `lag_0` (return hari-i) + 8 lag (1, 2, 3, 5, 7, 14, 21, 30) + `roll_mean` dan `roll_std` (jendela 7, 14, 30) | **15** |
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

**DPformer** adalah model yang dituning lebih serius. **MOSSA** (populasi **20**,
maksimum **100** iterasi; fitness = MAE **hanya** dari baris validasi ber-embargo; satu
pencarian per setup pada h=20 dengan 10 epoch per evaluasi, hasilnya dipakai untuk semua
horizon) berhenti dini bila `best_fit` tidak membaik ≥ 0,1% selama 20 iterasi (*patience* 20)
— aktualnya di iterasi **22** (univariat) dan **28** (multivariat). Hasilnya `lr` =
7,2×10⁻⁴ / 9,3×10⁻⁴ dan `weight_decay` = 8,4×10⁻³ / 7,5×10⁻³ (univariat / multivariat):
`lr` dekat default `AdamW` 1e-3 yang dipakai semua baseline DL, tetapi `weight_decay`
≈ **75–84×** defaultnya (1e-4). Implementasinya *single-objective*; jurnal acuan menyebut
MOSSA multi-objektif tanpa pseudocode (lihat notebook 05).

**Apakah tuning hanya untuk DPformer itu tidak adil?** Pertanyaan yang tepat, dan
sebagian jawabannya diukur: ablation MOSSA vs default pada 10 sel (setup × horizon)
menunjukkan **0 dari 10 lolos BH-FDR** (nilai `q` terendah 0,0713), dan MOSSA lebih baik
di 4 dari 10 sel menurut MAE. Artinya manfaat tuning ekstra itu **tidak terbukti** — itu
bukan bukti bahwa tuning tidak berpengaruh. Argumen yang lebih kuat: ketidaksimetrian ini
hanya menguntungkan DPformer, dan DPformer tetap tidak mengungguli Drift secara konsisten
(tak satu pun uji DPformer vs Drift lolos BH), sehingga kecil kemungkinannya mengubah
kesimpulan utama. Ini harus disebut kalau ditanya — lebih baik kita yang mengangkatnya
lebih dulu.

**Kenapa mengacu ke Bao et al. (2026)?** Konfigurasi 10 baseline DL mengikuti
*Table 3* paper tersebut sedekat mungkin (rumus *as-published*, tidak dituning ulang),
supaya ada acuan eksternal yang bisa diperiksa reviewer.

### 📁 Bukti di repo

![Arsitektur DPformer](../outputs/figures/fig_arsitektur_dpformer.png)

![Konvergensi MOSSA](../outputs/figures/fig_mossa_konvergensi.png)

```
src/config.py                            ML_MODELS, DL_MODELS, MOSSA_* (pop 20, iter 100)
src/models_dl.py                         11 arsitektur DL
notebooks/05_dpformer.ipynb              2 setup × 5 horizon + tuning MOSSA
notebooks/06_dl_models.ipynb             10 baseline DL
notebooks/07_ml_models.ipynb             10 model ML
outputs/results/ablation_mossa_vs_default.csv   0/10 lolos BH
```

---

# 07 — Tujuh pagar integritas

### ❓ Pertanyaan yang dijawab

*Bagaimana memeriksa bahwa angka-angka di Bab 08–09 tidak dihasilkan oleh kebocoran
informasi?*

### 💡 Penjelasan

Ini bagian pembeda utama proyek. Kebocoran data (*lookahead*) **tidak menghasilkan
error** — ia menghasilkan **hasil yang terlalu bagus**, dan tidak ada yang memberi
peringatan. Salah satu cara paling andal menangkapnya adalah merusak data secara
sengaja lalu memastikan hasilnya tidak berubah.

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
bergerak sedikit pun**, sementara metrik uji berubah 3×. Artinya, pada jalur yang
diuji (pemilihan fitness MOSSA dan *early stopping*), metrik uji tidak dipakai —
pagar tersebut **sensitif** (ia bisa mendeteksi kebocoran), bukan sekadar lolos kosong.
Begitu juga untuk LightGBM: iterasi *early stopping* tetap **8 → 8**.

> **Catatan jujur dari notebook itu sendiri:** *overfit/underfit tidak bisa dijamin nol
> hanya dari kode. Yang bisa dibuat ketat: mencegah kebocoran, early stopping hanya
> dari validasi, dan mengukur gejalanya secara jujur tanpa memilih model berdasarkan uji.*

**Batas pagar ini:** ia menutup jalur kebocoran yang diuji dan **tidak membuktikan**
bahwa tidak ada jalur lain. Keputusan desain yang diambil setelah melihat data seluruh
sampel, misalnya, tidak tertangkap — Skenario B dirancang setelah melihat korelasi
lead–lag seluruh sampel, sehingga temuannya eksploratif (Bab 04 dan 11).

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

**Benchmark Drift** = `r̂ = μ_train`, rata-rata return h-hari pada periode latih (tetap
untuk seluruh periode uji). Benchmark ini sangat sederhana: ia tidak melihat apa pun
kecuali rata-rata masa lalu, tetapi pada deret dengan sinyal tipis ia **susah dikalahkan**.
Kalau model tidak bisa mengalahkannya, model itu belum terbukti berguna. Dua pembanding yang
lebih adaptif — Drift ekspansif dan Drift bergulir 250 hari — dipakai di Bab 09 (MDE) dan
Bab 08 butir 3.

**Model terbaik per horizon** (MAE di ruang harga; `% vs Drift` = persentase lebih baik):

| Horizon | Univariat | % vs Drift | Multivariat | % vs Drift |
|---:|---|---:|---|---:|
| 1 | PatchTST | 0,59 | LightGBM | 0,43 |
| 5 | SVR | 1,27 | Transformer | 1,97 |
| 10 | AdaBoost | 3,78 | AdaBoost | 3,87 |
| 20 | AdaBoost | 6,77 | AdaBoost | 4,76 |
| 30 | RandomForest | 3,21 | PatchTST | 2,43 |

**Peringkat rata-rata keseluruhan** (berdasarkan MAE, di seluruh 10 sel: 2 setup × 5 horizon):

| Model | `mean_rank` | Menang di… |
|---|---:|---|
| **AdaBoost** | **2,8** | 4 dari 10 sel |
| RandomForest | 6,1 | 1 |
| ElasticNet | 6,3 | 0 |

Dua hal menarik yang layak disebut lebih dulu daripada ditanya:

1. **Pemenangnya bukan model yang canggih.** AdaBoost — boosting klasik — punya
   peringkat rata-rata 2,8, sedangkan Transformer 9,8, PatchTST 8,8, dan DPformer 10,0.
   Ini sejalan dengan dugaan bahwa kalau sinyalnya tipis, model besar lebih mudah
   menghafal kebisingan — tetapi dugaan itu belum diuji langsung di sini.
2. **Selisihnya kecil.** Bahkan pemenang hanya 0,43–6,77% lebih baik dari Drift.
   Angka ini **belum** diuji statistik — itu pekerjaan Bab 09, dan hasilnya jauh lebih
   suram daripada tabel ini menunjukkan. Secara keseluruhan MAE model lebih baik dari
   Drift pada 108 dari 210 run (51,4%).
3. **Sebagian selisih itu kemungkinan pergeseran level, bukan kemampuan meramal.** Rata-rata
   return harian periode uji (reli) ≈ 3,7× rata-rata periode latih, padahal Drift memakai
   rata-rata periode latih yang tetap. Sebagai pembanding, konstanta *oracle* (rata-rata
   periode uji — tidak layak dipakai) pun menurunkan MSE Drift 0,65–15,9% pada h=1–30. Dan
   terhadap Drift ekspansif atau bergulir, jumlah kombinasi (dari 42) dengan `rel_mse < 1`
   turun drastis: pada h=10, 32 → 19 → 3.

### 📁 Bukti di repo

![Hasil univariat](../outputs/figures/fig_hasil_univariat.png)

![Hasil multivariat](../outputs/figures/fig_hasil_multivariat.png)

![Perbaikan terhadap horizon](../outputs/figures/fig_perbaikan_vs_horizon.png)

```
outputs/results/model_terbaik_per_horizon.csv   tabel di atas
outputs/results/ringkasan_model.csv             mean_rank, wins
outputs/results/tabel_hasil_final.csv           210 baris lengkap
outputs/results/pergeseran_level_pembanding.csv Drift vs konstanta oracle (pergeseran level)
outputs/results/ci_ringkasan_per_horizon.csv    jumlah rel_mse < 1 vs Drift tetap / ekspansif / bergulir
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

**Lapis 1 — Diebold–Mariano dengan koreksi HLN.** Menguji apakah galat kuadrat return model
signifikan lebih kecil daripada Drift (satu sisi). 210 uji (21 model × 2 setup × 5 horizon).

**Lapis 2 — Koreksi Benjamini–Hochberg (BH-FDR).** Dengan 210 uji, pada taraf 5% kita
mengharapkan sekitar 10,5 uji (210 × 5%) bernilai `p < 0,05` hanya karena kebetulan,
*seandainya tidak ada efek sama sekali*. Tanpa koreksi, "16 uji mengalahkan Drift pada
`p < 0,05`" tidak jauh dari angka yang diharapkan dari kebetulan — jadi angka itu
menyesatkan kalau dibaca sebagai 16 model yang terbukti lebih baik.

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
| DM vs Drift — **lolos BH-FDR** (keluarga 21 uji: per setup×horizon) | 210 | **1** |
| DM vs Drift — lolos BH-FDR (keluarga 105 uji atau seluruh 210 uji) | 210 | 0 |
| Multivariat vs univariat — `rel_mse < 1` | 105 | 40 |
| Multivariat vs univariat — `p < 0,05` | 105 | 5 |
| Multivariat vs univariat — **lolos BH-FDR** (keluarga seluruh 105 uji) | 105 | **0** |
| Multivariat vs univariat — lolos BH-FDR (keluarga per horizon, 21 uji) | 105 | 2 |

> **Catatan soal "keluarga" (famili) uji.** Koreksi BH bergantung pada keluarga uji yang
> dikoreksi, dan pilihan keluarga mengubah hasilnya. **DM vs Drift:** dengan keluarga per
> setup×horizon (21 uji; `q_bh_setup_h`) **1** lolos; dengan keluarga per setup (105 uji;
> `q_bh_setup`, `q` terendah 0,2017) dan keluarga seluruh 210 uji (`q` terendah 0,2750;
> hitungan di `panduan_bimbingan/skrip/hitung_angka_tambahan.py`) **0** yang lolos.
> **Multivariat vs univariat:** dengan keluarga seluruh 105 uji (`q_bh`) **0** lolos;
> dengan keluarga per horizon (21 uji; `q_bh_h`) **2** lolos (GBM dan HistGBM di h=1,
> `q_bh_h` = 0,0138) — keduanya jatuh begitu keluarga diperluas. Jadi satu-satunya temuan
> DM vs Drift bergantung pada keluarga terkecil, dan dua temuan makro bergantung pada
> keluarga per horizon. Pilihan keluarga ini harus disebut apa adanya, bukan disembunyikan.

**Satu-satunya yang bertahan (keluarga 21 uji):**

```
setup       model          horizon   rel_mse   p_one_hln   q_bh_setup_h
univariat   transformer         1     0,9950     0,00195   0,0410
```

Efeknya kecil: MSE return hanya ≈ 0,5% lebih rendah daripada Drift (`rel_mse` 0,9950).
Sisanya tenggelam. MCS mengatakan hal yang sama dengan cara berbeda: **209 dari 220**
entri (22 model dan benchmark × 10 kombinasi setup×horizon) masuk *confidence set*, dan
**Drift sendiri masuk di seluruh 10 kombinasi** (2 setup × 5 horizon) — artinya Drift
tidak pernah bisa dikesampingkan.

---

**Lalu, daya ujinya bagaimana?**

Ini bagian yang paling sering disalahpahami. Median **observasi independen** per
horizon (`n_eff` = `n` ÷ h: return h-hari yang saling tumpang-tindih tidak independen,
jadi `n` efektif jauh lebih kecil dari jumlah baris), dan MDE terhadap tiga pembanding:

| Horizon | Baris uji | `n_eff` median | MDE vs Drift (tetap) | MDE vs Drift ekspansif | MDE vs Drift **bergulir** |
|---:|---:|---:|---:|---:|---:|
| 1 | 864 | 864,0 | 3,42% | 3,43% | 4,16% |
| 5 | 864 | 172,8 | 4,48% | 4,29% | 12,03% |
| 10 | 863 | 86,3 | 5,09% | 4,83% | 21,26% |
| 20 | 861 | 43,1 | 6,55% | 9,77% | **48,12%** |
| 30 | 859 | **28,6** | 5,79% | 9,06% | **70,42%** |

*MDE = Minimum Detectable Effect — penurunan MSE relatif terkecil terhadap pembanding yang
masih bisa dideteksi (uji satu sisi 5%, daya 80%; median dari 42 kombinasi per horizon;
`ci_ringkasan_per_horizon.csv`). MDE besar berarti **daya uji rendah** — selisih galat
terhadap Drift bergulir (jendela 250 hari) lebih berisik — bukan berarti pembandingnya
"lebih sulit dikalahkan".*

Artinya: terhadap Drift **tetap** (benchmark utama) MDE hanya 3,4–6,6% di semua horizon,
tetapi terhadap Drift **bergulir** pada h=30 kita baru bisa mendeteksi perbedaan
**sebesar 70%**. Kalaupun ada model yang lebih baik 10% terhadap pembanding bergulir itu,
**kita tidak punya cukup data untuk membuktikannya**. Karena itu:

> **"Tidak signifikan" ≠ "tidak ada efek." Untuk horizon panjang, itu berarti
> "daya ujinya tidak cukup."**

---

**Cara menyajikannya di depan dosen — tiga kalimat yang bisa dipakai:**

1. *"Kami tidak menyembunyikan hasil negatifnya. Dari 210 uji, hanya 1 yang lolos
   koreksi uji berganda — dan hanya pada keluarga uji terkecil; pada keluarga yang lebih
   luas tidak ada. Angka itu harus dibaca apa adanya."*
2. *"Kami juga mengukur daya ujinya: terhadap Drift tetap MDE kami 3–7% di semua horizon,
   tetapi terhadap Drift bergulir di horizon 30 hari MDE-nya 70% — jadi di sana klaim
   'tidak ada efek' pun tidak bisa kami buktikan. Jawabannya adalah 'belum terbukti',
   bukan 'tidak ada'."*
3. *"Ada satu pola yang tidak kami duga: ketika penutupan pasar hari-t dimasukkan, 10 dari
   10 model ML lebih baik di h=1, dan plasebo t+1 tidak menolong. Kami menandainya
   eksploratif, karena skenarionya dirancang setelah melihat korelasi lead–lag dan jam
   pengumuman Antam perlu diverifikasi. Itu ada di bab 11."*

### 📁 Bukti di repo

![DM vs Drift](../outputs/figures/fig_dm_vs_drift.png)

![Multivariat vs univariat](../outputs/figures/fig_dm_multi_vs_uni.png)

![Model Confidence Set](../outputs/figures/fig_mcs.png)

![Selang kepercayaan dan MDE](../outputs/figures/fig_ci_ringkasan.png)

```
outputs/results/dm_vs_drift_detail.csv                210 baris, kolom q_bh_setup (105) dan q_bh_setup_h (21)
outputs/results/dm_multivariat_vs_univariat_detail.csv 105 baris, kolom q_bh (105) dan q_bh_h (21)
outputs/results/mcs_keanggotaan.csv                   22 entri × 10 kombinasi = 220; 209 True
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
`GradientExplainer` — metode pendekatan, galat kelengkapan (`completeness_err`) ≈ 0,10 —
sedangkan untuk AdaBoost `PermutationExplainer`, yang eksak (galat ≈ 10⁻¹⁵). Latar
belakang 50 sampel dan 100 sampel yang dijelaskan. Hanya satu setup dan satu horizon yang
dijelaskan, jadi tabel berikut tidak boleh digeneralisasi ke horizon lain.

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

**Cara membaca:** model DL **memberi bobot besar pada makro global** (DXY + S&P 500 +
US 10Y + FFR = **58%** pada DPformer dan **52%** pada GRU-Attn), sedangkan AdaBoost
**hampir sepenuhnya mengandalkan harga emasnya sendiri** (75,8%; kelompok makro global tadi
hanya 19%). Padahal AdaBoost itu model yang peringkat rata-ratanya lebih baik.

Ini sejalan dengan temuan Bab 09 bahwa fitur makro (lag-1) tidak terbukti menambah
informasi: model yang paling bergantung pada makro bukan yang terbaik. Tetapi hati-hati:
SHAP mengukur **ketergantungan model**, bukan **manfaat akurasi** — bobot besar pada suatu
variabel tidak berarti variabel itu memperbaiki ramalan, dan dua jenis penjelas (gradien
vs permutasi) membatasi perbandingan langsung antar-kolom.

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

**1. Seed acak (notebook 13).** 6 dari 11 model DL (DLinear, NLinear, GRU, GRU+Attention,
LSTM+Attention, DPformer; LSTM, BiLSTM, BiGRU, Transformer, dan PatchTST tidak diulang) ×
2 setup × 5 horizon × 5 seed (42–46) = **60 kombinasi** (300 run). Hasil: **CV MAE
0,034% – 1,30%**, median **0,29%**, dan rentang MAE antar-arsitektur dalam satu sel =
**5–28×** simpangan baku antar-seed. Jadi variasi seed kecil dibanding selisih
antar-arsitektur (di antara 6 model itu). Tetapi **urutan**
arsitektur belum stabil: model terbaik sama di kelima seed hanya pada **1 dari 10** sel
setup×horizon, dan korelasi peringkat (Spearman) antar-seed 0,65–0,93 (median 0,82) —
jadi peringkat yang tipis di Bab 08 jangan dibaca terlalu kuat. (Perhitungan peringkat ini
dari `multiseed_semua.csv`, dengan `panduan_bimbingan/skrip/hitung_angka_tambahan.py`;
belum ada di notebook 13.)

**2. Ablasi (notebook 14).** Dua bagian. (a) *Kelompok fitur*: univariat (A) vs makro
global (B: DXY, S&P 500, US 10Y, FFR) vs makro domestik (C: USD/IDR, IHSG, BI-Rate) vs
seluruh makro (D), pada tiga model wakil (DPformer, GRU+Attention, AdaBoost) di h=5 dan
h=20 — 18 uji terhadap kelompok A, **0 signifikan** (`q` terendah 0,78). (b) *MOSSA vs
default* yang disebut di Bab 06 (**0/10 lolos BH**).

**3. Walk-forward (notebook 16).** Daripada mengandalkan satu periode uji, origin
digulirkan: **6 fold × 260 hari** (origin ≈ 2020-07/08 s.d. 2026-07), diuji pada h=5 dan
h=20 dengan 12 model (memakai `lr`/`weight_decay` default, bukan hasil MOSSA). Hasil: dari
48 uji (12 model × 2 setup × 2 horizon), 18 punya `rel < 1`, **0** dengan `p < 0,05`, dan
**0 lolos BH-FDR** (`q_bh` terendah 0,794; pada famili per horizon `q_bh_h` terendah
0,397). Ini cek konsistensi atas Bab 09 pada periode yang lebih panjang, bukan replikasi
yang independen: fold 4–6 seluruhnya dan fold 3 sebagian berada di dalam periode uji
utama. Arah pasarnya pun tidak seimbang: harga naik pada 11 dari 12 jendela fold×horizon
(+1% s.d. +40%; satu jendela turun 7,9%).

**4. Skenario B dan plasebo — satu pola yang konsisten, tetapi eksploratif.**

**(a) Apa yang diuji.** *Skenario B* memakai penutupan pasar **hari-t** (DXY, S&P 500,
USD/IDR, IHSG, US 10Y; FFR dan BI-Rate tetap lag-1) untuk meramal return t+1 — ramalan
dibuat **malam hari**, setelah pasar hari-t tutup dan **sebelum** harga Antam hari t+1
diumumkan. Itu layak hanya bila pengumuman Antam t+1 memang jatuh setelah penutupan pasar
AS hari-t. Berkas mentah mencatat jam untuk 82% hari kerja (median 08:18; P5–P95
07:50–09:02; zona waktu diasumsikan WIB), yang mendukung asumsi itu — tetapi **belum
diverifikasi dengan catatan resmi Antam**. (Teks notebook 15 dan 19 menyebut dataset tidak
mencatat jam; hitungan di atas dari `panduan_bimbingan/skrip/audit_data_mentah.py`.)

**(b) Pembanding.** *Lag-1* (baseline protokol utama) dan *plasebo t+1*: penutupan pasar
hari **t+1**, yang terbit **setelah** harga target dan mustahil tersedia saat ramalan —
pada h=1 seharusnya **tidak** menolong. (Pada h ≥ 2 plasebo memuat pergerakan di dalam
jendela target, sehingga bukan ujian kegagalan.)

| Anak tangga, h=1 | Median rasio vs lag-1 | Lebih baik | Lolos BH |
|---|---:|---:|---:|
| **Hari-t** (skenario B) | **0,9266** | 10 dari 10 | **9 dari 10** |
| **Plasebo t+1** (harusnya tidak menolong) | 1,0086 | 2 dari 10 | **0 dari 10** |

Varian yang lebih luas (12 model, notebook 15; `skenario_b_vs_multivariat_lag1.csv`): median
`rel_mse` **0,9267**, **11 dari 12 lolos BH** (keluarga 60 uji). Median 0,9266 vs 0,9267
berbeda karena sumbernya: notebook 19 membandingkan pada irisan baris (n = 863), notebook 15
memakai seluruh baris uji (n = 864).

**(c) Pembacaan.** Perbaikan muncul pada hari-t di h=1 dan **tidak** muncul pada plasebo;
untuk h ≥ 5, semuanya hilang (rasio ≈ 1,00; 0 lolos BH pada 10 model ML). Ada satu
pengecualian pada varian 12 model: DPformer h=20 (`rel` 0,9714; `q` 0,0149 pada keluarga
60 uji dan 0,0297 per horizon) — tidak muncul pada 10 model ML, jadi tidak dijadikan
temuan utama.

**(d) Mengapa ini eksploratif.** Skenario B dirancang **setelah** melihat korelasi lead–lag
pada seluruh sampel (Bab 04; notebook 03). Hanya satu ekspektasi (E2: plasebo tidak
menolong) yang ditetapkan sebelum hasilnya dilihat (notebook 19; E1 dan E3 mengulang hasil
notebook 15). Plasebo yang tidak menolong hanya **mengurangi kemungkinan** artefak
penyelarasan waktu; ia tidak membuktikan ketiadaannya. Dan kelayakannya bergantung pada
jam pengumuman Antam (a).

> **Kalimat yang aman dipakai:** *"Kami menemukan satu pola terukur — informasi penutupan
> hari-t memperbaiki ramalan 1 hari — yang tidak muncul pada plasebo. Kami menandainya
> eksploratif karena skenarionya dirancang setelah melihat data; untuk sisanya kami
> laporkan bahwa daya uji tidak cukup untuk mengklaim apa pun."*

### 📁 Bukti di repo

![Walk-forward gabungan](../outputs/figures/fig_walkforward_gabungan.png)

![Skenario B vs Drift](../outputs/figures/fig_skenario_b_dm.png)

![Tangga waktu informasi + plasebo](../outputs/figures/fig_tangga_waktu.png)

```
outputs/results/multiseed_ringkasan.csv            60 baris, kolom CV_pct
outputs/results/multiseed_semua.csv                300 run (dasar hitungan peringkat antar-seed)
outputs/results/ablation_ringkasan.csv              ablasi kelompok fitur (18 uji vs kelompok A)
outputs/results/ablation_mossa_vs_default.csv       0/10 lolos BH
outputs/results/walkforward_gabungan.csv            48 uji, 0 lolos BH
outputs/results/tangga_waktu_ringkasan.csv          hari-t / lag1 / plasebo t+1
outputs/results/tangga_waktu_harapan.csv            ekspektasi E1–E3 dan hasilnya
outputs/results/skenario_b_vs_multivariat_lag1.csv  11/12 lolos BH di h=1
notebooks/13, 14, 15, 16, 19
panduan_bimbingan/skrip/                            angka tambahan: peringkat antar-seed, BH keluarga 210,
                                                    arah harga walk-forward, audit data mentah
```

---

# 12 — Kesimpulan, kontribusi, dan batasan

### ❓ Pertanyaan yang dijawab

*Jadi, apa kontribusi proyek ini?*

### 💡 Penjelasan

**Satu paragraf yang bisa dibaca apa adanya:**

> Pada horizon 1 hari, ada pola yang konsisten dari informasi penutupan pasar hari-t
> (eksploratif): perbaikan median ≈ 7,3% terhadap baseline lag-1, lebih baik pada 10 dari
> 10 model dan signifikan pada 9 dari 10 setelah koreksi BH, dan tidak muncul pada kontrol
> plasebo. Pada horizon 5–30 hari, dan untuk tujuh fitur makro lag-1, tidak ada perbedaan
> yang dapat dibuktikan secara statistik terhadap benchmark Drift — dan daya ujinya
> terbatas (MDE terhadap Drift bergulir mencapai 70% pada h=30), sehingga ketiadaan bukti
> di sana tidak boleh dibaca sebagai bukti ketiadaan. Jawaban yang jujur adalah
> "belum terbukti", bukan "tidak ada".

**Tiga kontribusi metodologis** yang tidak bergantung pada hasil:

1. **Tujuh pagar integritas dengan mutation testing.** Bukti numerik bahwa *early
   stopping* tidak bisa melihat data uji (`mae_val` identik digit demi digit saat data
   uji dirusak, sementara `mae_test` berubah 3×). Ini bisa diterapkan ke proyek lain.
2. **Pelaporan MDE bersama hasil negatif.** Tanpa ini, "tidak signifikan" terbaca
   sebagai "tidak ada", padahal dua hal itu berbeda.
3. **Pelaporan berlapis:** klaim statistik utama dilaporkan bersama DM → BH-FDR → MCS →
   selang kepercayaan, dan sensitivitas terhadap pilihan keluarga uji ditampilkan,
   sehingga tidak ada klaim yang berdiri di atas satu uji saja.

**Tujuh batasan yang harus disebut lebih dulu** (jangan menunggu ditanya):

| # | Batasan | Angka pasti |
|---|---|---|
| 1 | **Daya uji terbatas di horizon panjang** | `n_eff` median h=30 = **28,63**; MDE vs Drift tetap h=30 = **5,79%**, tetapi vs Drift bergulir = **70,42%** |
| 2 | **Bukan harga pasar** | harga postingan administratif; **22,47%** hari di periode latih return nol (puncak 2014: 40,6%), ±25% di antaranya hasil *forward-fill* |
| 3 | **Tidak ada XAU/USD** | fitur hanya DXY, S&P 500, USD/IDR, IHSG, US 10Y, FFR, BI-Rate |
| 4 | **Satu aset, satu negara** | kesimpulan berlaku untuk emas Antam (Rp/gram) saja |
| 5 | **Periode uji didominasi satu reli lalu koreksi** | 2023–2026, harga 1,04 → puncak 3,17 juta (2026-01-29) → −17,6% di akhir data; walk-forward: harga naik di 11 dari 12 jendela |
| 6 | **Sumber data emas tidak resmi** | dataset komunitas — perlu validasi silang ke situs resmi |
| 7 | **Temuan Skenario B eksploratif** | dirancang setelah melihat korelasi lead–lag seluruh sampel; kelayakan bergantung pada jam pengumuman Antam yang belum diverifikasi dengan catatan resmi |

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
| Ringkasan angka + grafik (Excel) | `panduan_bimbingan/ringkasan_eksperimen.xlsx` |
| Naskah lisan untuk bimbingan | `panduan_bimbingan/NASKAH_BIMBINGAN.md` |
| Diagram alur penelitian (A/B/C) | `diagram/` |
| Skrip pembantu angka dokumen (bukan eksperimen) | `panduan_bimbingan/skrip/` |

**Alur jalannya satu eksperimen:**
`notebook → src/runners.py → src/pipeline.py (fitur/split, model ML) / src/models_dl.py (model DL) → simpan JSON/CSV`
— notebook tidak pernah menghitung sendiri, sehingga eksperimen utama, ablation, dan
walk-forward tidak mungkin berbeda diam-diam.

---

# Lampiran B — Glosarium

| Istilah | Arti |
|---|---|
| **Direct multi-horizon** | Setiap horizon (1/5/10/20/30) punya model khusus, bukan diramalkan berurutan. Menghindari penumpukan galat. |
| **Embargo** | Baris validasi yang target `i+h`-nya jatuh di periode uji **dibuang** dari *early stopping*. |
| **Drift (benchmark)** | `r̂ = μ_train` — memprediksi return h-hari rata-rata periode latih untuk semua hari uji. Sederhana, tetapi sulit dikalahkan ketika sinyalnya tipis. |
| **Drift ekspansif** | Drift adaptif: rata-rata semua return h-hari yang labelnya sudah diketahui pada origin; diperbarui di setiap origin. |
| **Drift bergulir** | Drift adaptif: rata-rata 250 return h-hari terakhir yang sudah diketahui; diperbarui di setiap origin. Estimasinya berisik, sehingga MDE terhadapnya besar (daya uji rendah). |
| **Diebold–Mariano (DM)** | Uji apakah galat model signifikan lebih kecil daripada pembanding. |
| **HLN** | Koreksi sampel kecil Harvey–Leybourne–Newbold pada statistik DM (dibandingkan dengan distribusi *t*); dipakai bersama penaksir varians jangka panjang untuk galat h-hari yang tumpang-tindih. |
| **BH-FDR** | Koreksi *Benjamini–Hochberg* untuk uji berganda. Tanpa ini, 210 uji pada taraf 5% diharapkan menghasilkan ≈10,5 uji `p < 0,05` hanya karena kebetulan bila tidak ada efek sama sekali. |
| **MCS** | *Model Confidence Set* — set model yang masih layak disebut pemenang. Kalau berisi semua orang → tidak ada pemenang. |
| **MDE** | *Minimum Detectable Effect* — perbedaan terkecil yang masih bisa dideteksi dengan daya uji yang tersedia. |
| **`n_eff`** | Observasi independen efektif; jauh lebih kecil dari jumlah baris karena data saling berkorelasi. |
| **`rel_mse`** | MSE galat return model ÷ MSE pembanding (`rel_mse_ret`; versi ruang harga: `rel_mse_price`). **< 1 = lebih baik.** |
| **Plasebo t+1** | Kontrol: input memakai penutupan pasar hari **t+1**, yang terbit setelah harga target dan mustahil tersedia saat ramalan. Pada h=1 seharusnya tidak menolong — kalau menolong, ada artefak penyelarasan waktu. |
| **Skenario B** | Ramalan malam hari yang memakai penutupan pasar **hari-t** (layak bila harga Antam t+1 diumumkan setelah penutupan pasar AS hari-t — belum diverifikasi). **Eksploratif.** |
| **Stasioneritas / ADF** | Uji apakah deret punya *unit root*. Deret non-stasioner wajib ditransformasi sebelum dipakai. |
| **Mutation testing** | Merusak data uji sengaja, lalu memastikan hasil latih **tidak berubah** — bukti bahwa pelatihan tidak membaca data uji (menutup jalur yang diuji, bukan semua jalur kebocoran). |
| **Overfit / model nol** | Model nol = selalu memprediksi rata-rata target latih. *Skill* diukur relatif terhadapnya. |

---

# Lampiran C — FAQ: kemungkinan pertanyaan dosen

> **Ini bagian yang paling bernilai saat bimbingan.** Baca sekali sebelum pertemuan.

### Q1 — *"Kalau hasilnya hampir semua tidak signifikan, apa kontribusinya?"*

Jawab tiga langkah: (a) kami **tidak menyembunyikannya** — 1 dari 210 lolos BH, dan hanya
pada keluarga uji terkecil; (b) kami **mengukur daya ujinya**, dan MDE h=30 vs Drift
bergulir adalah 70% — jadi "tidak ada efek" pun tidak bisa kami buktikan, jawaban yang
jujur adalah **"belum terbukti"**; (c) ada **satu pola eksploratif yang diuji dengan
plasebo** — informasi hari-t di h=1, 9/10 lolos BH, dan plasebo t+1 tidak menolong;
tandai sebagai eksploratif karena dirancang setelah melihat data.

### Q2 — *"Kenapa XAU/USD (harga emas dunia) tidak dipakai?"*

Kejujuran dulu: **memang tidak dipakai**, dan itu keterbatasan yang kami catat di
README. Tujuh fitur makro hanya DXY, S&P 500, USD/IDR, IHSG, US 10Y, FFR, BI-Rate.
DXY dipakai sebagai *proxy* dolar. Kalau diminta, ini adalah tambahan paling bernilai
untuk iterasi berikutnya.

### Q3 — *"Kenapa hanya DPformer yang dituning? Tidak adil?"*

Pertanyaan bagus, dan **sebagian sudah kami ukur**: ablation MOSSA vs default pada 10 sel
menunjukkan **0/10 lolos BH-FDR** (`q` terendah 0,0713), dan MOSSA lebih baik di 4 dari
10 sel menurut MAE — jadi manfaat tuning **tidak terbukti** (bukan terbukti nol;
`weight_decay` hasil MOSSA ≈ 75–84× default). Ketidaksimetrian ini hanya menguntungkan
DPformer, dan DPformer tetap tidak mengungguli Drift secara konsisten, sehingga kecil
kemungkinannya mengubah kesimpulan utama.

### Q4 — *"Kenapa perlu 21 model? Bukannya cuma bikin ribet?"*

Justru sebaliknya: banyak model dipakai sebagai **uji ketahanan**. Kalau kesimpulan hanya
berlaku untuk satu arsitektur, ia lemah. Kalau kesimpulan yang sama muncul dari 21
arsitektur — termasuk ElasticNet yang sederhana sekaligus Transformer yang kompleks —
ia kuat. Konsekuensinya, karena banyak model diuji, koreksi uji berganda (BH-FDR) menjadi
**keharusan**, bukan pilihan.

### Q5 — *"Kenapa evaluasinya di ruang harga, bukan return?"*

Sebagian. **Metrik titik** (MAE, RMSE, MAPE) dihitung di ruang harga, karena yang
ditanyakan pengguna adalah "berapa rupiah harganya" — satu persen galat di harga 400 ribu
berbeda maknanya dengan 1 persen di harga 3 juta. Tetapi **uji Diebold–Mariano** memakai
galat kuadrat **return** (`rel_mse_ret`): galat di ruang harga sama dengan galat return
dikali harga origin, sehingga hasil di ruang harga memberi bobot lebih besar pada periode
ketika harga tinggi; versi ruang harga (`rel_mse_price`) ada di berkas yang sama sebagai
pembanding. Metrik return tambahan ada di `metrik_return.csv`.

### Q6 — *"Kalau datanya harga administratif, apa gunanya?"*

Ini justru yang membuat menarik: **22,47% hari di periode latih tidak bergerak** (±75%
di antaranya hari dengan pengumuman berharga sama, ±25% hari tanpa pengumuman yang diisi
*forward-fill*), jadi model harus belajar **kapan harga bergerak**, bukan hanya seberapa
jauh — dan itu berbeda dari deret harga pasar biasa. Kami laporkan ini sebagai temuan,
bukan menyembunyikannya.

### Q7 — *"Apa itu embargo, dan kenapa repot-repot?"*

Lihat Bab 05. Ringkasnya: baris validasi terakhir punya target `i+h` yang jatuh di
periode uji. Kalau tetap dipakai untuk *early stopping*, saat memilih model terbaik kita
sudah membaca harga uji. Pada h=30, 30 dari 429 baris dibuang. Tanpa ini, hasilnya akan
**terlalu bagus dan tidak ada yang menyadarinya**.

### Q8 — *"Kenapa multivariat tidak lebih baik dari univariat?"*

Dari 105 uji, 40 punya rasio < 1 (median 1,003) dan 5 mencapai `p < 0,05`, tetapi **0 lolos
BH-FDR** (keluarga = seluruh 105 uji; dengan keluarga per horizon, 2 lolos — GBM dan HistGBM
di h=1).
Yang paling mendekati adalah GBM (0,915) dan HistGBM (0,910) di h=1 — `q` = 0,069, tidak
lolos. Jadi yang bisa kami katakan adalah "tidak terbukti", bukan "tidak membantu". SHAP
(h=20, tiga model wakil) hanya memberi gambaran ketergantungan: model DL memberi bobot
52–58% pada kelompok makro global (DXY, S&P 500, US 10Y, FFR), sedangkan AdaBoost — peringkat
rata-rata terbaik — **75,8%-nya dari harga emas sendiri**. Dugaan bahwa informasi makro sudah
terwakili oleh return emas sendiri (termasuk lewat DXY) **belum kami uji**; SHAP tidak
membuktikannya.

### Q9 — *"Ini kan pakai AI, sepenuhnya otomatis?"*

Jawab dua bagian, dan **isi bagian pertama sesuai kenyataan dan aturan kampus/jurnal**.

**(1) Peran AI — diisi penulis.** Susunan yang jujur (ganti isian `[ISI: …]`):
*"Saya memakai [ISI: nama alat AI] untuk [ISI: bagian pekerjaan — mis. membantu menulis
kode, menyusun draf dokumentasi, memeriksa angka]. Keputusan desain eksperimen (target,
split, embargo, benchmark, pilihan model) [ISI: siapa yang memutuskan dan bagaimana kamu
memeriksanya]. Hasil saya jalankan dan periksa sendiri dengan [ISI: cara — mis. menjalankan
ulang notebook, mencocokkan angka dengan CSV]. Penggunaan ini [ISI: dilaporkan di bagian …
/ sesuai kebijakan …]."* Jangan menyatakan hal yang tidak benar tentang siapa yang
mengerjakan apa.

**(2) "Sepenuhnya otomatis?" — tidak, dan ini yang bisa ditunjukkan.** Seluruh
pelatihan/analisis lewat `src/runners.py`, **satu jalur** untuk eksperimen utama,
ablation, walk-forward, dan uji seed — notebook tidak menghitung sendiri, jadi hasil tidak
bisa berbeda diam-diam antar-notebook. Ditambah 7 pagar integritas di Bab 07, termasuk
*mutation testing* yang bisa ditunjukkan angkanya. Otomatisasi ini mengurangi galat teknis,
tetapi **tidak menggantikan** keputusan penelitian dan pemeriksaan hasil oleh penulis.

### Q10 — *"Bisa dipakai untuk trading?"*

Belum, dan itu kesimpulan penelitian ini sendiri: pada h=5–30 tidak ada model yang
signifikan mengalahkan Drift (setelah koreksi BH), dan di MCS **Drift masuk di seluruh 10
kombinasi**; di h=1 pun hanya satu uji yang lolos pada keluarga terkecil, dengan efek ≈0,5%.
Pola yang paling layak diuji lanjut adalah **h=1 dengan informasi penutupan hari-t** —
tetapi itu eksploratif, jam pengumuman Antam harus diverifikasi, dan biaya transaksi serta
selisih harga jual–beli kembali (*buyback*) belum dimasukkan. Ini bukan rekomendasi trading.

### Q11 — *"Kenapa walk-forward hanya diuji h=5 dan h=20?"*

6 fold × 260 hari memakai sekitar enam tahun data terakhir (origin ≈ 2020-07/08 s.d.
2026-07). h=5 dan h=20 dipilih sebagai perwakilan horizon pendek dan panjang, dengan alasan
[ISI: alasan yang benar-benar kamu pakai — mis. biaya komputasi]. Ini keterbatasan yang
kami catat, bukan keputusan yang disembunyikan.

### Q12 — *"Apa bedanya Drift biasa dan Drift bergulir?"*

Drift biasa (benchmark utama) = rata-rata return h-hari **seluruh** periode latih, tetap
untuk seluruh periode uji. Drift ekspansif = rata-rata semua return h-hari yang sudah
diketahui pada origin (bertambah seiring waktu). Drift bergulir = rata-rata **250** return
h-hari terakhir yang sudah diketahui, diperbarui di setiap origin, sehingga **adaptif**.
MDE terhadap Drift bergulir jauh lebih besar (48% di h=20, 70% di h=30) karena estimasinya
sendiri berisik — artinya **daya uji lebih rendah**, bukan bahwa ia "lebih sulit
dikalahkan". Sebaliknya, banyak model yang tampak lebih baik dari Drift tetap **tidak**
lebih baik dari Drift adaptif: pada h=10, `rel_mse < 1` pada 32 dari 42 kombinasi vs Drift
tetap, 19/42 vs ekspansif, 3/42 vs bergulir — sebagian keunggulan atas Drift tetap
kemungkinan pergeseran level (Bab 08, butir 3).

### Q13 — *"Kenapa semua model DL pakai AdamW lr=1e-3?"*

Keseragaman supaya perbandingan adil dan hasilnya bisa direproduksi: sepuluh baseline DL
memakai AdamW dengan `lr` 1e-3 dan `weight_decay` 1e-4, tanpa grid atau tuning, mengikuti
Bao et al. (2026), *Table 3* (patokan eksternal yang bisa diperiksa). Hanya DPformer yang
dituning (MOSSA, Bab 06) — dan manfaat tuning itu tidak terbukti (0/10 lolos BH).

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
$env:TA_THREADS = "2"    # batas thread CPU (default 2); atur sebelum kernel notebook dimulai
```

Seed acak **tidak** diatur lewat variabel lingkungan: `SEED = 42` ada di `src/pipeline.py`
dan seed tambahan (43–46) di `SEEDS_EXTRA` pada `src/config.py`.

Sebagian besar notebook **tidak mengulang pekerjaan yang sudah ada** — `src/runners.py`
memeriksa `protocol`, `rev`, dan `data_hash` pada tiap run, dan hanya run basi yang
dihitung ulang.

**Stempel saat freeze `v1.0.0`:** 210/210 run CURRENT, 0 basi · 20/20 notebook
tereksekusi, 0 error output · 69 PNG + 69 PDF.

---

<sub>Catatan belajar · `panduan_bimbingan/` · Dokumen v1.0.2 (hasil dibekukan pada v1.0.0) · Protokol `v9` · data_hash `7aea8db80d44` · 2026-10-31</sub>
