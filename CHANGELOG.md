# Changelog

Semua perubahan signifikan proyek ini didokumentasikan di sini.

Format mengikuti [Keep a Changelog](https://keepachangelog.com/id/id/1.1.0/);
penomoran versi mengikuti [Semantic Versioning](https://semver.org/lang/id/).

---

## [1.0.2] — 2026-10-31

**Rilis dokumentasi.** Tidak ada perubahan pada `data/`, `notebooks/`, `outputs/`, atau
`src/`; seluruh hasil eksperimen tetap pada freeze **v1.0.0** (`data_hash`
`7aea8db80d44`). Semua angka pada dokumen dicocokkan ulang dengan CSV hasil dan log notebook.

### Diperbaiki

- **Plasebo t+1** dideskripsikan sesuai kode: inputnya penutupan pasar hari *t+1* (terbit
  setelah harga target; tak layak), bukan "tanggal diputar mundur"; "bukti tidak ada artefak"
  diganti "mengurangi kemungkinan artefak penyelarasan waktu"
- **Pagar integritas / mutation testing** (PENJELASAN bab 07 dan glosarium): "bukti tidak ada
  kebocoran" diganti "pelatihan tidak membaca data uji" (jalur yang diuji); ditambah paragraf
  batas pagar (keputusan desain setelah melihat seluruh sampel tidak tertangkap)
- **Tabel MDE** (PENJELASAN bab 09): kolom "Drift bergulir" pada h=5 dan h=10 berisi nilai
  Drift ekspansif (4,29% / 4,83%); nilai yang benar 12,03% / 21,26%. Kolom Drift ekspansif
  ditambahkan, dan MDE terhadap Drift tetap (benchmark utama; 3,4–6,6%) ditampilkan di samping
  48–70% milik Drift bergulir
- **Pembacaan lead–lag** (PENJELASAN bab 04) diluruskan: korelasi terkuat ada pada penutupan
  hari-*t* (yang diuji Skenario B), bukan pada nilai lag-1 yang dipakai desain utama; "semua
  variabel" menjadi 3 dari 7
- **Keluarga BH**: satu-satunya temuan DM vs Drift hanya lolos pada keluarga per setup×horizon
  (21 uji); pada keluarga 105 dan 210 uji tidak lolos
- **Seed**: klaim "peringkat stabil" dan "ratusan kali" diganti angka terukur (rentang MAE
  antar-arsitektur 5–28× SD seed; model terbaik sama di 5 seed hanya 1 dari 10 sel; korelasi
  peringkat antar-seed median 0,82)
- **MOSSA**: berhenti dini di iterasi 22/28 (bukan 100); satu pencarian per setup pada h=20;
  `weight_decay` hasil MOSSA ≈ 75–84× default; implementasi *single-objective*; optimizer AdamW
  (bukan Adam); klaim grid 3×4 dihapus (tidak ada di kode)
- **SHAP**: "≈55% makro global" diganti 58% (DPformer) dan 52% (GRU-Attn) untuk kelompok DXY,
  S&P 500, US 10Y, FFR; `PermutationExplainer` hanya untuk AdaBoost; galat kelengkapan
  GradientExplainer ≈ 0,10 dicatat
- **Walk-forward**: origin ≈ 2020-07/08 → 2026-07; fold 4–6 seluruhnya dan fold 3 sebagian
  berada di periode uji utama (bukan di luar); arah harga dilaporkan per jendela fold×horizon
  (naik pada 11 dari 12 jendela, satu turun 7,9%)
- Variabel lingkungan `TA_SEED` dihapus dari dokumentasi (tidak ada di kode); `TA_THREADS`
  default 2 dan dibaca oleh notebook
- Sumber data: 7 deret makro terpakai + `id_tbill` (diunduh, tidak dipakai); hanya harga Antam
  dan BI-Rate yang manual
- Porsi return nol 22,5% → 22,47%; "reli besar" → reli lalu koreksi (−17,6% dari puncak
  2026-01-29); "sebagian besar hari" return nol → 22,47%
- Glosarium: HLN = Harvey–Leybourne–Newbold; definisi Drift bergulir/ekspansif, plasebo, Skenario B
- "Tiga batasan" → tujuh batasan; ditambah "Skenario B eksploratif"
- Nomor versi di README dan PENJELASAN diselaraskan; kalimat "belum di-commit"
  dihapus; klaim `src/models_*.py` → `src/models_dl.py`
- Peran dokumen dijelaskan: PENJELASAN adalah catatan belajar pribadi (tidak ditunjukkan ke
  dosen); yang ditunjukkan saat bimbingan adalah `ringkasan_eksperimen.xlsx` (lembar `Jawaban`),
  dan yang diucapkan mengikuti `NASKAH_BIMBINGAN.md`. Excel tidak lagi menunjuk ke PENJELASAN
- PENJELASAN Q8: median rasio multivariat vs univariat (1,003) dicantumkan

### Ditambahkan

- `diagram/alur_penelitian_lebar.{mmd,png,pdf}` — render lebar 3×3 dari diagram alur;
  `alur_penelitian_mermaid.txt` dikoreksi (label forward-fill dan glitch kurs, MOSSA, keluarga
  BH, MDE, plasebo, SHAP, dan kesimpulan disamakan dengan hasil/kode)
- `panduan_bimbingan/ringkasan_eksperimen.xlsx` dibangun ulang: 12 lembar, 18 grafik Excel asli,
  9 gambar figur, seluruh angka dihitung dari CSV hasil. Tampilannya polos: judul, tabel
  berbingkai tipis, tanpa kartu angka besar dan tanpa banner berwarna. Lembar depan `Jawaban`
  dibuat untuk bimbingan: hasil dalam tiga kalimat, MAE dan MAPE per horizon terhadap Drift,
  delapan angka penting, dua grafik, batasan, enam tanya-jawab dengan tautan ke lembar rujukan,
  dan tiga pertanyaan untuk dosen
- `panduan_bimbingan/NASKAH_BIMBINGAN.md` — naskah lisan untuk menjawab "gimana hasilnya?":
  alur ±3–4 menit dengan penunjuk bagian lembar `Jawaban` yang dibuka, 13 jawaban singkat untuk
  pertanyaan yang mungkin muncul (dengan rujukan ke bab dan FAQ di PENJELASAN), daftar kalimat
  yang sebaiknya dihindari, versi satu menit, dan hal yang masih harus diisi penulis sendiri.
  Angkanya sama dengan PENJELASAN dan lembar `Jawaban`
- FAQ Q9 (penggunaan AI) ditulis ulang sebagai templat jawaban yang harus diisi penulis
- Catatan "eksploratif" untuk Skenario B di README dan PENJELASAN; catatan
  pergeseran level (rata-rata return uji ≈ 3,7× rata-rata latih) di README dan PENJELASAN
- `panduan_bimbingan/skrip/audit_data_mentah.py` dan `hitung_angka_tambahan.py` — dua skrip
  pembantu (hanya membaca; bukan bagian eksperimen) untuk angka yang dipakai dokumen tetapi
  belum dihasilkan notebook: hari kerja tanpa pengumuman (198) dan komposisi return nol
  (±75% pengumuman berharga sama / ±25% *forward-fill*), jam pengumuman, kestabilan peringkat
  antar-seed, BH pada keluarga 210 uji, dan arah harga per jendela walk-forward
- `MANIFEST.sha256` dibuat ulang untuk v1.0.2 (2570 berkas): hash dokumen yang berubah
  diperbarui, delapan berkas ditambahkan (`diagram/` dan `ringkasan_eksperimen.xlsx`
  sebelumnya belum tercatat di MANIFEST v1.0.1; `NASKAH_BIMBINGAN.md` baru), dan satu berkas
  dihapus (`RINGKASAN.md`)

### Dihapus

- `panduan_bimbingan/RINGKASAN.md`. Isinya sudah ada di PENJELASAN (bab 01, 09, 11, dan 12)
  dan di lembar `Jawaban` pada `ringkasan_eksperimen.xlsx`; dokumen belajar sekarang satu,
  yaitu PENJELASAN

### Catatan

- Median Skenario B h=1 ≈ 0,927 dilaporkan dengan sumbernya: 0,9266 (notebook 19, n=863,
  irisan baris dengan plasebo) dan 0,9267 (notebook 15, n=864).
- Teks notebook 15 dan 19 menyatakan dataset tidak mencatat jam publikasi; berkas mentah
  Antam mencatat jam untuk 82% hari kerja (median 08:18, zona waktu diasumsikan WIB;
  `audit_data_mentah.py`). Notebook tidak diubah karena hasil dibekukan; jam tersebut tetap
  perlu diverifikasi dengan catatan resmi sebelum Skenario B dijadikan klaim.

---
## [1.0.1] — 2026-10-21

**Rilis dokumentasi.** Tidak ada perubahan pada `data/`, `notebooks/`, `outputs/`, atau
`src/`; seluruh hasil eksperimen tetap pada freeze **v1.0.0** (`data_hash`
`7aea8db80d44`).

### Diperbaiki

- Judul README yang terpotong: "Bebas-Ke" → "Bebas-Kebocoran"
- Klaim Skenario B diselaraskan dengan naskah: headline **10 model ML** (median
  `rel_mse` 0,9267; 9/10 lolos BH), varian 12 model dicatat sebagai catatan kaki
  (11/12 lolos di h=1; DPformer lolos di h=20)
- Catatan keluarga uji BH ditambahkan: 1 vs 0 untuk DM vs Drift, dan 0 vs 2 untuk
  perbandingan multivariat vs univariat
- Typo "resli" → "resmi"
- Jumlah dependensi langsung: 20 → **17**
- Pohon repository, prinsip arsitektur, dan perintah verifikasi `MANIFEST.sha256`
  diperbarui
- `panduan_bimbingan/` ditambahkan ke repository

---

## [1.0.0] — 2026-10-19

**Versi pembekuan (freeze) pertama.** Seluruh hasil eksperimen dikunci dan diverifikasi
sebelum ditulis ke sini.

### Stempel versi

| | |
|---|---|
| Protokol | `v9-21model-dpformer-bao-aligned-honest-mossa-shap-leakage-safe-embargo` |
| Revisi ML | `ml-r2` |
| Sidik jari data | `7aea8db80d44` |
| Python | 3.13.0 |
| Integritas | `MANIFEST.sha256` · `outputs/results/MANIFEST.json` |

### Dibekukan

- **210/210 run utama CURRENT, 0 basi** (21 model × 2 setup × 5 horizon)
- **110** loss-curve DL (`*_history.json`) · **80** feature importance (`*_fimp.csv`)
- **69 PNG + 69 PDF** figur · **48 CSV** hasil top-level
- **20/20 notebook** ter-eksekusi penuh, 0 error output
- Data: `data_gabungan.csv` — 4.325 baris, 2010-01-05 → 2026-08-03

### Ditambahkan

- `README.md` — dokumentasi lengkap: protokol, peta notebook & modul, cara reproduksi,
  hasil utama, dan daftar batasan yang diketahui
- `requirements.txt` — 17 dependensi langsung, terkunci dari versi terpasang
- `requirements-full.txt` — 155 entri `pip freeze --all` untuk reproduksi persis
- `MANIFEST.sha256` — checksum seluruh isi repository
- `outputs/results/MANIFEST.json` — ringkasan mesin-baca tentang hasil yang dibekukan
- `.gitignore`
- Inisialisasi version control + tag `v1.0.0`

### Protokol v9 (yang dikunci pada versi ini)

1. Early stopping DL & boosting di **semua** mode memakai baris validasi **ber-embargo**
2. Harga emas hari kerja tanpa pengumuman = **forward-fill**, bukan interpolasi
3. FFR bulanan dipakai mulai **bulan berikutnya** (*as-of*)
4. Fitur ML memuat return hari-i (`lag_0`)
5. Seleksi hiperparameter (MOSSA) **hanya** dari baris validasi

### Revisi historis

- **`ml-r2`** — SVR memakai target terstandardisasi (epsilon dalam satuan simpangan baku
  target). Versi sebelumnya memakai `epsilon=0.02` pada target mentah sehingga pada h=1
  seluruh target jatuh di dalam tabung epsilon dan SVR hanya menghasilkan konstanta.
  Revisi ini **tidak** mengubah `PROTOCOL`, sehingga hasil DL & MOSSA tetap sah.

### Tidak diubah

Tidak ada file hasil yang dimodifikasi pada pembekuan ini — seluruh tahap 1.0.0
bersifat **penambahan** (lihat `MANIFEST.json` → `verifikasi`).
