# Changelog

Semua perubahan signifikan proyek ini didokumentasikan di sini.

Format mengikuti [Keep a Changelog](https://keepachangelog.com/id/id/1.1.0/);
penomoran versi mengikuti [Semantic Versioning](https://semver.org/lang/id/).

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
- `requirements.txt` — 20 dependensi langsung, terkunci dari versi terpasang
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
