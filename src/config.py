"""Konfigurasi eksperimen 21-model Antam + DPformer.

Protokol utama:
- 5 horizon: 1, 5, 10, 20, 30 hari.
- split kronologis 70/10/20 atas posisi yang memiliki target h.
- 11 DL: 10 baseline Bao et al. + DPformer.
- 10 ML: 5 model lama + 5 model tambahan.
- Drift tetap tersedia sebagai benchmark diagnostik, tetapi tidak dihitung sebagai
  salah satu dari 21 model utama.
- Semua preprocessing/scaling yang belajar parameter hanya fit pada train.
- Protokol v9 (final): (1) early stopping DL & boosting di SEMUA mode memakai baris validasi
  ber-embargo (target h-harinya i+h masih di dalam periode validasi); (2) harga emas
  hari kerja tanpa pengumuman = forward-fill (bukan interpolasi); (3) FFR bulanan
  dipakai mulai bulan berikutnya (as-of); (4) fitur ML memuat return hari-i (lag_0).
"""
import os

QUICK = os.environ.get("TA_QUICK") == "1"

HORIZONS = [5, 20] if QUICK else [1, 5, 10, 20, 30]
SETUPS = ["univariat", "multivariat"]

# Bao et al. (2026), Table 3 sedekat mungkin.
SEQ_LEN = 10
EPOCHS = 2 if QUICK else 100
PATIENCE = 10
BASE_LR = 1e-3
BASE_WD = 1e-4

DL_BATCH = {
    "lstm": 1024, "gru": 128, "bilstm": 1024, "bigru": 128,
    "lstm_att": 64, "gru_att": 128, "transformer": 64,
    "dlinear": 128, "nlinear": 128, "patchtst": 256, "dpformer": 256,
}
DL_UNITS = {
    "lstm": 1024, "gru": 512, "bilstm": 1024, "bigru": 512,
    "lstm_att": 64, "gru_att": 256, "transformer": 256,
    "dlinear": 16, "nlinear": 16, "patchtst": 128, "dpformer": 128,
}
if QUICK:
    DL_UNITS = {k: min(v, 32) for k, v in DL_UNITS.items()}

# DPformer: budget FULL mengikuti paper (population 20 / 100 iterations).
# QUICK memakai budget kecil hanya untuk smoke test. Fitness selalu validation-only.
MOSSA_EPOCHS = 1 if QUICK else 10
MOSSA_H = 20
MOSSA_POP = 3 if QUICK else 20
MOSSA_ITER = 1 if QUICK else 100
# Penghentian dini MOSSA (OPSIONAL; adaptasi komputasi). None = jalankan penuh MOSSA_ITER
# sesuai jurnal. Contoh: MOSSA_PATIENCE = 20 -> berhenti bila best_fit tidak membaik
# >= MOSSA_MIN_REL_DELTA (relatif) selama 20 iterasi berturut-turut; MOSSA_ITER tetap
# batas maksimum. Bila dipakai, tulis di laporan sebagai kriteria konvergensi.
MOSSA_PATIENCE = 20
MOSSA_MIN_REL_DELTA = 1e-3
MOSSA_PRODUCERS = 0.20
MOSSA_WARNERS = 0.10
LR_RANGE = (1e-4, 1e-2)
WD_RANGE = (0.0, 1e-2)

SEEDS_EXTRA = [43] if QUICK else [43, 44, 45, 46]
MODELS_MULTISEED = [
    "dpformer", "gru", "gru_att", "lstm_att", "dlinear", "nlinear",
]

# 10 ML = 5 existing + 5 additional.
ML_MODELS = [
    "lightgbm", "xgboost", "rf", "catboost", "gbm",
    "svr", "extra_trees", "hist_gbm", "adaboost", "elasticnet",
]
DL_MODELS = [
    "lstm", "gru", "bilstm", "bigru", "transformer",
    "dlinear", "nlinear", "patchtst", "lstm_att", "gru_att", "dpformer",
]
ALL_MODELS = ML_MODELS + DL_MODELS
assert len(ML_MODELS) == 10
assert len(DL_MODELS) == 11
assert len(ALL_MODELS) == 21

# --- Revisi konfigurasi ML (tanpa mengubah PROTOCOL, jadi hasil DL & MOSSA tetap sah) ------------
# 'ml-r2': SVR memakai target terstandardisasi (epsilon dalam satuan simpangan baku target);
# versi lama epsilon=0.02 pada target mentah sehingga untuk h=1 (std return ~0,7%) seluruh
# target jatuh di dalam tabung epsilon. Run ML tanpa rev yang sama dijalankan ulang otomatis
# oleh notebook 07 (total ML sekitar 6 menit).
ML_REV = "ml-r2"

# Hiperparameter LightGBM (tetap, tidak dituning pada test): dipakai notebook 07, 14-16 dan SHAP.
LGB_PARAMS = {"learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 5}

# --- Analisis statistik (notebook 10, 12) ---------------------------------------------------------
DM_KERNEL = "rectangular"        # kernel persegi sampai lag h-1 + koreksi HLN (lihat src/stats.py)
MCS_ALPHA = 0.10                 # model confidence set 90%
MCS_B = 200 if QUICK else 5000   # replikasi bootstrap blok
N_BLOCKS = 4                     # sub-periode test (blok sama panjang) untuk uji stabilitas

# --- SHAP otomatis: DPformer + wakil terbaik 10 DL + wakil terbaik 10 ML (notebook 11) ---------
SHAP_SETUP = "multivariat"
SHAP_H = 20
SHAP_N_BACKGROUND = 20 if QUICK else 50
SHAP_N_EXPLAIN = 20 if QUICK else 100
SHAP_NSAMPLES = 20 if QUICK else 200     # sampel expected-gradients per jendela (DL)

# --- Ablation fitur (notebook 14): A dan D memakai hasil utama (univariat / multivariat) -----------
ABLATION_GROUPS = {
    "A_univariat": [],
    "B_global": ["dxy_ret", "sp500_ret", "us10yr_diff", "ffr_diff"],
    "C_domestik": ["usd_idr_ret", "ihsg_ret", "bi_rate_diff"],
    "D_penuh": ["dxy_ret", "sp500_ret", "usd_idr_ret", "ihsg_ret",
                "us10yr_diff", "ffr_diff", "bi_rate_diff"],
}
# Model yang diablasi = tiga model wakil otomatis (DPformer, DL baseline terbaik, ML terbaik; lihat
# results/wakil_model.json yang ditulis notebook 09), sama dengan yang dijelaskan SHAP (notebook 11).
ABL_HORIZONS = [20] if QUICK else [5, 20]

# --- Skenario B (notebook 15): informasi penutupan pasar hari-t (origin malam hari) -----------
SCEN_B_DIRNAME = "skenario_b"
SCEN_B_DL = True      # sertakan DL wakil (DPformer + DL terbaik) di skenario B selain 10 ML

# --- Walk-forward / rolling-origin (notebook 16) -----------------------------------------------
WF_N_FOLDS = 2 if QUICK else 6
WF_FOLD_LEN = 120 if QUICK else 260          # ~1 tahun hari kerja per blok uji
WF_HORIZONS = [20] if QUICK else [5, 20]
WF_ML_MODELS = ML_MODELS if not QUICK else ["lightgbm", "rf", "elasticnet"]
# DL pada walk-forward: DPformer + DL baseline wakil (otomatis) dengan lr/wd DEFAULT (1e-3 / 1e-4),
# bukan hasil MOSSA: MOSSA dituning pada validasi split utama (2021-2023) yang berada di dalam
# jendela uji beberapa fold -> memakainya akan membocorkan informasi ke fold itu.

# --- Bahasa label figur: 'id' (default) atau 'en' (untuk naskah jurnal). Env TA_LANG. ---------
LANG = os.environ.get("TA_LANG", "id")

# Batas diagnosa; bukan angka untuk memilih model berdasarkan test.
TRAIN_VAL_GAP_WARN = 0.25
VAL_TRAIN_RATIO_WARN = 3.0

PROTOCOL = "v9-21model-dpformer-bao-aligned-honest-mossa-shap-leakage-safe-embargo" + ("-QUICK" if QUICK else "")
OUT = os.path.join("..", "outputs_quick" if QUICK else "outputs")
