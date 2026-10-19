"""Pipeline eksperimen Tugas Akhir: direct multi-horizon forecasting.
Target : R_h(i) = p(i+h)/p(i) - 1 (return kumulatif h hari).
Univariat: fitur = return 1-hari emas (lag + rolling utk LightGBM,
           window utk DL).
Multivariat: + 7 variabel makro eksogen dalam bentuk STASIONER (log-return utk
           indeks/kurs, selisih utk suku bunga/yield; lihat make_exo_features),
           sudah di-lag 1 hari di data_gabungan.csv -> TIDAK ada lookahead.
Evaluasi di ruang level harga: p_hat(i+h) = p(i) * (1 + r_hat).
Split kronologis 70/10/20 identik utk semua model.
Tuning hyperparameter (MOSSA) HANYA boleh memakai baris validasi
(eval_split='val'); baris test tidak disentuh.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Level mentah indikator makro (kolom di data_gabungan.csv, sudah di-lag 1 hari).
EXO_LEVEL_COLS = ['dxy', 'sp500', 'usd_idr', 'ihsg', 'us10yr', 'ffr', 'bi_rate']
# Fitur eksogen yang DIPAKAI MODEL = versi stasioner dari level di atas.
# Level (mis. S&P500 / FFR) trending dan di periode test berada di luar rentang
# train -> model mengekstrapolasi. Log-return / selisih menghilangkan masalah itu.
EXO_LOGRET_COLS = ['dxy', 'sp500', 'usd_idr', 'ihsg']   # -> log(x_t / x_{t-1})
EXO_DIFF_COLS = ['us10yr', 'ffr', 'bi_rate']            # -> x_t - x_{t-1} (poin %)
EXO_COLS = ([f'{c}_ret' for c in EXO_LOGRET_COLS] +
            [f'{c}_diff' for c in EXO_DIFF_COLS])
EXO_CLIP = 5.0   # z-score kanal eksogen di-clip ke [-5, 5] (hanya input DL)
LAGS = (1, 2, 3, 5, 7, 14, 21, 30)
WINDOWS = (7, 14, 30)
SEED = 42
# Versi protokol eksperimen (didefinisikan di config.py). Dicatat di tiap file
# hasil; hasil dengan versi berbeda dianggap basi (stale): dijalankan ulang dan
# tidak ikut ranking.
try:
    from .config import PROTOCOL
except ImportError:                      # dipakai sebagai skrip biasa
    from config import PROTOCOL


# Skenario B ("origin malam hari"): deret PASAR harian yang penutupannya hari-t sudah diketahui
# saat pasar AS tutup (malam hari t, sebelum harga Antam hari t+1 diumumkan). FFR (bulanan)
# dan BI-Rate TETAP lag-1 karena waktu publikasinya tidak setegas penutupan pasar.
EXO_B_SAMEDAY = ['dxy_ret', 'sp500_ret', 'usd_idr_ret', 'ihsg_ret', 'us10yr_diff']


def make_exo_scenario_b(exo: pd.DataFrame) -> pd.DataFrame:
    """exo (lag-1, seperti data_gabungan.csv) -> eksogen skenario B.

    Baris t memuat nilai pasar hari t (= baris t+1 pada exo yang sudah di-lag) untuk
    EXO_B_SAMEDAY; kolom lain tetap. Baris terakhir (nilai hari-t belum ada) bernilai NaN
    pada kolom yang digeser -> tidak pernah menjadi origin (target h>=1 tidak tersedia).
    """
    out = exo.copy()
    for c in EXO_B_SAMEDAY:
        out[c] = exo[c].shift(-1)
    return out


def walkforward_splits(p: pd.Series, h: int, n_folds: int = 6, fold_len: int = 260,
                       val_frac: float = 0.10):
    """Split rolling-origin (expanding window) untuk uji ketahanan.

    Blok uji berurutan sepanjang n_folds*fold_len origin TERAKHIR (blok terakhir berakhir
    di origin terakhir yang punya target). Untuk fold k dengan origin uji pertama s:
      - baris yang boleh dipakai belajar = origin i dengan i+h <= s-1 (label sudah diketahui
        sebelum s) -> embargo h hari otomatis terpenuhi;
      - val = val_frac terakhir dari baris itu (untuk early stopping), train = sisanya;
      - uji = origin s .. s+fold_len-1.
    Return: list dict {fold, split=(tr_dates, va_dates, te_pos), start, end}.
    """
    n = len(p)
    last_origin = n - h - 1
    dates = p.index
    folds = []
    for k in range(n_folds):
        s = last_origin + 1 - (n_folds - k) * fold_len
        m = s - h                                   # baris belajar: posisi 0..m-1
        if m < 200:
            raise ValueError(f'fold {k}: data latih terlalu sedikit (m={m}); kurangi n_folds/fold_len')
        n_va = max(1, int(m * val_frac))
        tr_dates = dates[:m - n_va]
        va_dates = dates[m - n_va:m]
        te_pos = list(range(s, s + fold_len))
        folds.append({'fold': k + 1, 'split': (tr_dates, va_dates, te_pos),
                      'start': dates[s], 'end': dates[s + fold_len - 1]})
    return folds


def make_exo_features(levels: pd.DataFrame) -> pd.DataFrame:
    """Level makro (UNSHIFTED) -> 7 fitur stasioner (log-return / selisih).
    Baris pertama (tak punya pendahulu) diisi 0. Pemanggil bertugas men-shift
    1 hari kerja sesudahnya (diff(shift(x)) == shift(diff(x)) -> tanpa lookahead).
    """
    out = pd.DataFrame(index=levels.index)
    for c in EXO_LOGRET_COLS:
        out[f'{c}_ret'] = np.log(levels[c]).diff()
    for c in EXO_DIFF_COLS:
        out[f'{c}_diff'] = levels[c].diff()
    return out[EXO_COLS].fillna(0.0)


def run_is_current(jf: str, rev: str | None = None, data_hash: str | None = None) -> bool:
    """True bila file hasil run ada DAN dibuat dengan PROTOCOL yang sama.

    rev (opsional): revisi konfigurasi model. Dipakai untuk model yang konfigurasinya
    diperbaiki tanpa mengubah protokol (mis. ML_REV di config.py). Run lama yang tidak
    mencatat rev yang sama dianggap basi dan dijalankan ulang otomatis.
    data_hash (opsional): sidik jari data_gabungan.csv (runners.data_fingerprint). Run yang MENCATAT
    sidik jari berbeda dianggap basi (data berubah); run lama tanpa catatan sidik jari diterima
    (nilainya diverifikasi terhadap data di notebook 08).
    """
    import json, os
    if not os.path.exists(jf):
        return False
    try:
        with open(jf, encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('protocol') != PROTOCOL:
            return False
        if rev is not None and meta.get('rev') != rev:
            return False
        if data_hash is not None and meta.get('data_hash') not in (None, data_hash):
            return False
        return True
    except Exception:
        return False


def free_memory():
    """Lepas memori TensorFlow/Keras antar-run (clear_session + gc). Tanpa ini RAM
    terus naik selama ratusan run berurutan dalam satu kernel notebook."""
    import gc
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def epoch_info(hist_dict: dict) -> dict:
    vl = hist_dict.get('val_loss') or hist_dict.get('loss')
    return {'best_epoch': int(np.argmin(vl)) + 1, 'n_epochs': len(vl)}


def save_mossa(path: str, results: dict, meta: dict):
    """results: {setup: (lr, wd, history[, trace])}. Menyimpan juga penanda
    bahwa fitness = validasi + versi protokol (dicek load_mossa)."""
    import json
    payload = {'_meta': {**meta, 'fitness_split': 'val', 'protocol': PROTOCOL}}
    for k, v in results.items():
        payload[k] = {'lr': v[0], 'wd': v[1], 'history': v[2],
                      'trace': v[3] if len(v) > 3 else []}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)


def load_mossa(path: str):
    """{setup: (lr, wd, history)} atau None jika file tak ada / dibuat dengan
    protokol lama (mis. fitness berbasis test) -> pemanggil harus cari ulang."""
    import json, os
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        saved = json.load(f)
    meta = saved.get('_meta', {})
    if meta.get('protocol') != PROTOCOL or meta.get('fitness_split') != 'val':
        print(f'MOSSA: {os.path.basename(path)} dibuat dgn protokol lama '
              f'(bukan fitness validasi) -> DIABAIKAN, pencarian diulang.')
        return None
    return {k: (v['lr'], v['wd'], v['history']) for k, v in saved.items()
            if k != '_meta'}


def evaluate_all(y_true, y_pred):
    y_true = np.asarray(y_true).flatten().astype(float)
    y_pred = np.asarray(y_pred).flatten().astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-9, denom)
    smape = float(100 * np.mean(np.abs(y_true - y_pred) / denom))
    denom2 = np.where(y_true == 0, 1e-9, np.abs(y_true))
    mape = float(100 * np.mean(np.abs(y_true - y_pred) / denom2))
    return {'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape}


def multihorizon_split(p: pd.Series, h: int):
    """Split 70/10/20 kronologis murni.
    Dari n total data, posisi valid sebagai input adalah 0..(n-h-1)
    karena target posisi i adalah p[i+h] (harus tersedia).
    Rasio: train 70%, val 10%, test 20% dari valid_n.
    """
    n = len(p)
    valid_n = n - h          # jumlah posisi yang bisa dipakai sebagai input
    n_test  = max(1, int(valid_n * 0.20))
    n_val   = max(1, int(valid_n * 0.10))
    n_tr    = valid_n - n_test - n_val

    dates    = p.index
    tr_dates = dates[:n_tr]
    va_dates = dates[n_tr:n_tr + n_val]
    te_pos   = list(range(n_tr + n_val, n_tr + n_val + n_test))
    return tr_dates, va_dates, te_pos


def build_lgb_features(p: pd.Series, exo: pd.DataFrame | None = None, exo_cols=None):
    """Fitur tabular untuk model ML: return hari origin (lag_0), lag return, rolling
    return + eksogen (jika ada). Di-align berdasarkan TANGGAL.

    Himpunan informasi SAMA dengan window DL: return hari-i (origin) diketahui saat
    prediksi, jadi ikut sebagai fitur (lag_0) dan jendela rolling berakhir di hari-i
    (bukan i-1). Eksogen sudah lag-1 -> aman. Jumlah fitur: 15 (univariat) / 22
    (multivariat)."""
    r = p.pct_change()
    F = pd.DataFrame(index=p.index)
    F['lag_0'] = r
    for lag in LAGS:
        F[f'lag_{lag}'] = r.shift(lag)
    for w in WINDOWS:
        F[f'roll_mean_{w}'] = r.rolling(w).mean()
        F[f'roll_std_{w}']  = r.rolling(w).std()
    if exo is not None:
        for c in (EXO_COLS if exo_cols is None else list(exo_cols)):
            F[c] = exo[c]
    return F.dropna()


def train_lightgbm_h(p: pd.Series, F: pd.DataFrame, h: int,
                     params: dict | None = None,
                     num_boost_round: int = 1000,
                     early_stopping_rounds: int = 50,
                     split=None):
    """split: (tr_dates, va_dates, te_pos) kustom (mis. dari walkforward_splits);
    None = split 70/10/20 utama."""
    import lightgbm as lgb
    tr_dates, va_dates, te_pos = split if split is not None else multihorizon_split(p, h)
    dates  = p.index
    pos_of = {d: i for i, d in enumerate(dates)}
    feats  = list(F.columns)

    tr_ok = [d for d in tr_dates if d in F.index]
    # embargo: hanya baris validasi yang target h-harinya (i+h) masih di dalam periode
    # validasi yang dipakai untuk early stopping -> harga periode test tidak ikut
    # menentukan model yang terpilih (juga pada run FINAL).
    va_ok = [d for d in va_dates if d in F.index and pos_of[d] + h < te_pos[0]]
    assert va_ok and all(pos_of[d] + h < te_pos[0] for d in va_ok), \
        'baris validasi untuk early stopping menyentuh periode test'
    te_ok = [i for i in te_pos if dates[i] in F.index]

    X_tr = F.loc[tr_ok][feats]
    X_va = F.loc[va_ok][feats]
    X_te = F.loc[[dates[i] for i in te_ok]][feats]
    y_tr = (p.iloc[[pos_of[d] + h for d in tr_ok]].values /
            p.loc[tr_ok].values - 1).astype(float)
    y_va = (p.iloc[[pos_of[d] + h for d in va_ok]].values /
            p.loc[va_ok].values - 1).astype(float)

    base = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1,
            'seed': SEED}
    if params:
        base.update(params)
    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))
    model = lgb.train(
        base,
        lgb.Dataset(X_tr, label=y_tr),
        valid_sets=[lgb.Dataset(X_va, label=y_va,
                                reference=lgb.Dataset(X_tr, label=y_tr))],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )
    r_hat   = model.predict(X_te)
    metrics = evaluate_h(p, te_ok, r_hat, h)
    y_true  = p.iloc[[i + h for i in te_ok]].values
    y_pred  = p.iloc[te_ok].values * (1 + np.asarray(r_hat).flatten())
    return model, metrics, te_ok, y_true, y_pred


def train_ml_h(p: pd.Series, F: pd.DataFrame, h: int, model_name: str,
               params: dict | None = None, split=None):
    """Trainer 10 ML.

    Prinsip anti-leakage:
    - split kronologis berasal dari multihorizon_split();
    - X_train/X_val/X_test tidak dicampur;
    - model yang membutuhkan scaling memakai StandardScaler yang FIT hanya
      pada train;
    - validation dipakai hanya untuk early stopping model boosting yang
      mendukungnya; test baru disentuh pada prediksi final.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.compose import TransformedTargetRegressor

    tr_dates, va_dates, te_pos = split if split is not None else multihorizon_split(p, h)
    dates = p.index
    pos_of = {d: i for i, d in enumerate(dates)}
    feats = list(F.columns)

    tr_ok = [d for d in tr_dates if d in F.index]
    # embargo: hanya baris validasi yang target h-harinya (i+h) masih di dalam periode
    # validasi yang dipakai untuk early stopping -> harga periode test tidak ikut
    # menentukan model yang terpilih (juga pada run FINAL).
    va_ok = [d for d in va_dates if d in F.index and pos_of[d] + h < te_pos[0]]
    assert va_ok and all(pos_of[d] + h < te_pos[0] for d in va_ok), \
        'baris validasi untuk early stopping menyentuh periode test'
    te_ok = [i for i in te_pos if dates[i] in F.index]

    X_tr = F.loc[tr_ok, feats].values
    X_va = F.loc[va_ok, feats].values
    X_te = F.loc[[dates[i] for i in te_ok], feats].values
    y_tr = (p.iloc[[pos_of[d] + h for d in tr_ok]].values /
            p.loc[tr_ok].values - 1).astype(float)
    y_va = (p.iloc[[pos_of[d] + h for d in va_ok]].values /
            p.loc[va_ok].values - 1).astype(float)

    p_ = dict(params or {})

    if model_name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=1000, learning_rate=0.02, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, random_state=SEED,
            verbosity=0, **p_)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    elif model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            random_state=SEED, n_jobs=-1, **p_)
        model.fit(X_tr, y_tr)

    elif model_name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.02, depth=6,
            l2_leaf_reg=5.0, early_stopping_rounds=50,
            random_seed=SEED, verbose=0, allow_writing_files=False, **p_)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    elif model_name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            min_samples_leaf=3, subsample=0.8, random_state=SEED, **p_)
        model.fit(X_tr, y_tr)

    elif model_name == "svr":
        # SVR sensitif terhadap skala FITUR dan TARGET -> keduanya distandardisasi, scaler
        # hanya fit pada train. Revisi ML_REV 'ml-r2': epsilon (lebar tabung) dinyatakan dalam
        # satuan simpangan baku target. Versi lama memakai epsilon=0.02 pada target mentah;
        # untuk h=1 simpangan baku return hanya ~0,7% sehingga seluruh target jatuh DI DALAM
        # tabung dan SVR hanya menghasilkan konstanta. Default sklearn (C=1, epsilon=0.1)
        # dipakai apa adanya (tidak dituning).
        model = TransformedTargetRegressor(
            regressor=Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(C=1.0, gamma="scale", epsilon=0.1, kernel="rbf", **p_)),
            ]),
            transformer=StandardScaler())
        model.fit(X_tr, y_tr)

    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=500, max_features=0.8, min_samples_leaf=2,
            random_state=SEED, n_jobs=-1, **p_)
        model.fit(X_tr, y_tr)

    elif model_name == "hist_gbm":
        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=15, l2_regularization=1.0,
            early_stopping=False, random_state=SEED, **p_)
        model.fit(X_tr, y_tr)

    elif model_name == "adaboost":
        base = None
        model = AdaBoostRegressor(
            n_estimators=500, learning_rate=0.03, loss="square",
            random_state=SEED, **p_)
        model.fit(X_tr, y_tr)

    elif model_name == "elasticnet":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.15,
                                 max_iter=10000, random_state=SEED, **p_)),
        ])
        model.fit(X_tr, y_tr)

    else:
        raise ValueError(f"model_name tidak dikenal: {model_name!r}")

    r_hat = model.predict(X_te)
    metrics = evaluate_h(p, te_ok, r_hat, h)
    y_true = p.iloc[[i + h for i in te_ok]].values
    y_pred = p.iloc[te_ok].values * (1 + np.asarray(r_hat).flatten())

    # Feature importance hanya jika model memang menyediakannya.
    feat_imp = None
    if hasattr(model, "feature_importances_"):
        feat_imp = dict(zip(feats, model.feature_importances_.tolist()))
    elif hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            feat_imp = dict(zip(feats, np.abs(np.asarray(inner.coef_)).ravel().tolist()))
    elif hasattr(model, "coef_"):
        feat_imp = dict(zip(feats, np.abs(np.asarray(model.coef_)).ravel().tolist()))

    return model, metrics, te_ok, y_true, y_pred, feat_imp

def _check_eval_split(eval_split: str):
    if eval_split not in ('test', 'val'):
        raise ValueError(f"eval_split harus 'test' atau 'val', bukan {eval_split!r}")


def train_dl_h(p: pd.Series, exo: pd.DataFrame | None, h: int,
               build_fn, seq_len: int, units_1: int, epochs: int = 100,
               patience: int = 15, batch_size: int = 32,
               learning_rate: float = 1e-3, weight_decay: float = 1e-4,
               seed: int = SEED, eval_split: str = 'test', split=None, exo_cols=None):
    """Training DL direct-h. Input window multi-channel (return + eksogen),
    target R_h. Scaling z-score (StandardScaler) fit hanya pada train --
    mengikuti jurnal acuan Bao et al. (2026) section 3.2.2. Kanal eksogen
    (sudah stasioner) di-clip ke [-EXO_CLIP, EXO_CLIP] setelah z-score.
    Optimizer AdamW dengan weight_decay -- mengikuti jurnal acuan:
    "the AdamW optimizer is used for all models, with a base learning rate
    of 0.001 and a weight decay coefficient of 0.0001" (unified default,
    dipakai 10 arsitektur baseline). Untuk DPformer, (learning_rate,
    weight_decay) di sini diisi hasil tuning MOSSA (lihat notebook 05),
    menggantikan base default itu -- persis seperti jurnal.

    eval_split:
      'test' (default) : metrik dihitung pada baris TEST (evaluasi final).
      'val'            : metrik dihitung pada baris VALIDASI; baris test tidak
                         disentuh sama sekali. Hanya baris validasi yang
                         target h-hari-nya (i+h) masih di dalam periode
                         validasi yang dipakai (embargo h hari), baik untuk
                         metrik MAUPUN untuk early stopping. Ini yang dipanggil
                         fitness MOSSA; src/checks.py membuktikan tidak ada
                         kebocoran dari harga periode test.
      Pada KEDUA mode, early stopping hanya memakai baris validasi ber-embargo
      (i+h < awal periode test), jadi run FINAL pun tidak melihat harga periode test.
          seed: seed Python/NumPy/TensorFlow (default 42 = run utama; seed lain utk
          uji robustness di notebook 13).
    split: (tr_dates, va_dates, te_pos) kustom (mis. walkforward_splits); None = 70/10/20.
    exo_cols: subset/urutan kolom eksogen (default EXO_COLS) untuk ablation fitur.
    Return: model, metrics, rows_ok, y_true, y_pred, hist_dict
    """
    import tensorflow as tf
    _check_eval_split(eval_split)
    tf.keras.utils.set_random_seed(seed)

    tr_dates, va_dates, te_pos = split if split is not None else multihorizon_split(p, h)
    dates  = p.index
    pos_of = {d: i for i, d in enumerate(dates)}
    xcols  = list(EXO_COLS if exo_cols is None else exo_cols)

    r1     = p.pct_change().fillna(0.0)
    yh_all = (p.shift(-h) / p - 1)

    tr_pos = [pos_of[d] for d in tr_dates]
    va_pos = [pos_of[d] for d in va_dates]
    n_feat = 1 if exo is None else 1 + len(xcols)

    # scaler z-score per channel, fit train saja
    cols = ['emas_ret'] + xcols if exo is not None else ['emas_ret']
    raw  = pd.DataFrame({'emas_ret': r1})
    if exo is not None:
        raw = pd.concat([raw, exo[xcols]], axis=1)
    raw = raw.reindex(dates).fillna(0.0)

    X_in_all = np.zeros((len(p), n_feat))
    for j, c in enumerate(cols):
        scaler = StandardScaler().fit(
            raw[c].iloc[tr_pos].values.reshape(-1, 1))
        z = scaler.transform(raw[c].values.reshape(-1, 1)).flatten()
        if c != 'emas_ret':
            z = np.clip(z, -EXO_CLIP, EXO_CLIP)
        X_in_all[:, j] = z

    yh_tr  = yh_all.loc[tr_dates].values
    sc_tg  = StandardScaler().fit(yh_tr.reshape(-1, 1))

    def build_X(rows):
        X, y = [], []
        for i in rows:
            if i - seq_len < 0:
                continue
            X.append(X_in_all[i - seq_len + 1:i + 1])
            y.append(sc_tg.transform([[yh_all.iloc[i]]])[0, 0])
        return (np.array(X).reshape(-1, seq_len, n_feat),
                np.array(y).reshape(-1, 1))

    if eval_split == 'val':
        ev_pos = [i for i in va_pos if i + h < te_pos[0]]
        assert ev_pos and max(ev_pos) + h < te_pos[0], 'baris evaluasi val menyentuh periode test'
    else:
        ev_pos = te_pos

    X_tr, y_tr = build_X(tr_pos)
    # Early stopping / ReduceLROnPlateau memakai baris validasi ber-embargo di KEDUA mode
    # (val dan test): target h-harinya (i+h) harus masih di dalam periode validasi, jadi
    # tidak ada harga periode test yang ikut menentukan bobot yang terpilih.
    va_fit_pos = [i for i in va_pos if i + h < te_pos[0]]
    assert va_fit_pos and max(va_fit_pos) + h < te_pos[0], \
        'baris validasi untuk early stopping menyentuh periode test'
    X_va, y_va = build_X(va_fit_pos)
    X_ev, _    = build_X(ev_pos)
    rows_ok = [i for i in ev_pos if i - seq_len >= 0]

    model = build_fn(seq_length=seq_len, units_1=units_1, n_feat=n_feat)
    model.compile(optimizer=keras_optimizer(learning_rate, weight_decay), loss='mse', metrics=['mae'])
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=max(3, patience // 3),
            min_lr=1e-6, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, min_delta=1e-5,
            restore_best_weights=True),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
                     batch_size=batch_size, callbacks=callbacks, verbose=0)

    # Simpan training history untuk visualisasi loss curve
    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in hist.history.items()}

    r_hat_sc = model.predict(X_ev, verbose=0)
    r_hat    = sc_tg.inverse_transform(r_hat_sc).flatten()
    metrics  = evaluate_h(p, rows_ok, r_hat, h)
    y_true   = p.iloc[[i + h for i in rows_ok]].values
    y_pred   = p.iloc[rows_ok].values * (1 + np.asarray(r_hat).flatten())
    return model, metrics, rows_ok, y_true, y_pred, hist_dict


def dl_target_baselines(p: pd.Series, h: int, seq_len: int = 10, split=None) -> dict:
    """MSE 'model nol' (selalu memprediksi rata-rata target TRAIN) pada skala z-score target DL.

    Rujukan diagnosis loss curve: loss yang tidak lebih rendah dari angka ini berarti model
    belum belajar apa pun di luar rata-rata. Di train nilainya 1 (z-score fit pada train);
    di validasi (baris ber-embargo) nilainya var + mean^2 target validasi dalam satuan train,
    sehingga mencerminkan pergeseran rezim train -> validasi.
    Konstruksi baris identik dengan train_dl_h (filter i - seq_len >= 0, embargo validasi)."""
    tr_dates, va_dates, te_pos = split if split is not None else multihorizon_split(p, h)
    pos_of = {d: i for i, d in enumerate(p.index)}
    yh = (p.shift(-h) / p - 1)
    sc = StandardScaler().fit(yh.loc[tr_dates].values.reshape(-1, 1))
    tr_pos = [pos_of[d] for d in tr_dates if pos_of[d] - seq_len >= 0]
    va_pos = [pos_of[d] for d in va_dates if pos_of[d] + h < te_pos[0] and pos_of[d] - seq_len >= 0]
    z_tr = sc.transform(yh.iloc[tr_pos].values.reshape(-1, 1)).ravel()
    z_va = sc.transform(yh.iloc[va_pos].values.reshape(-1, 1)).ravel()
    return {'train_mse_null': float(np.mean(z_tr ** 2)), 'val_mse_null': float(np.mean(z_va ** 2)),
            'val_mean_z': float(z_va.mean()), 'val_std_z': float(z_va.std()),
            'n_train': int(len(z_tr)), 'n_val': int(len(z_va)), 'target_std_train': float(sc.scale_[0])}


def build_dl_explain_arrays(p: pd.Series, exo: pd.DataFrame | None, h: int, seq_len: int,
                            max_background: int = 50, max_samples: int = 100, seed: int = SEED,
                            split=None, exo_cols=None):
    """Prepare leakage-safe scaled input windows for post-hoc DL attribution.

    Background windows are sampled only from the training positions; explained
    windows are sampled only from test positions. Scaling is fit on training
    positions, exactly as in ``train_dl_h``. The helper returns model-input
    tensors, not target values. This supports SHAP GradientExplainer without
    touching the test set during training/tuning.
    """
    from sklearn.preprocessing import StandardScaler

    tr_dates, _, te_pos = split if split is not None else multihorizon_split(p, h)
    dates = p.index
    pos_of = {d: i for i, d in enumerate(dates)}
    tr_pos = [pos_of[d] for d in tr_dates]
    xcols = list(EXO_COLS if exo_cols is None else exo_cols)
    n_feat = 1 if exo is None else 1 + len(xcols)

    r1 = p.pct_change().fillna(0.0)
    raw = pd.DataFrame({'emas_ret': r1}, index=dates)
    if exo is not None:
        raw = pd.concat([raw, exo[xcols].reindex(dates)], axis=1)
    raw = raw.fillna(0.0)
    cols = ['emas_ret'] + xcols if exo is not None else ['emas_ret']

    X_all = np.zeros((len(p), n_feat), dtype=np.float32)
    for j, c in enumerate(cols):
        scaler = StandardScaler().fit(raw[c].iloc[tr_pos].values.reshape(-1, 1))
        z = scaler.transform(raw[c].values.reshape(-1, 1)).flatten()
        if c != 'emas_ret':
            z = np.clip(z, -EXO_CLIP, EXO_CLIP)
        X_all[:, j] = z.astype(np.float32)

    def build(rows):
        samples, kept = [], []
        for i in rows:
            if i - seq_len < 0:
                continue
            samples.append(X_all[i - seq_len + 1:i + 1])
            kept.append(i)
        return np.asarray(samples, dtype=np.float32).reshape(-1, seq_len, n_feat), kept

    X_bg, _ = build(tr_pos)
    X_explain, rows_explain = build(te_pos)
    rng = np.random.default_rng(seed)
    if len(X_bg) > max_background:
        idx = np.linspace(0, len(X_bg) - 1, max_background).astype(int)
        X_bg = X_bg[idx]
    if len(X_explain) > max_samples:
        idx = np.linspace(0, len(X_explain) - 1, max_samples).astype(int)
        X_explain = X_explain[idx]
        rows_explain = [rows_explain[i] for i in idx]
    return X_bg, X_explain, rows_explain


def keras_optimizer(lr: float = 1e-3, weight_decay: float = 1e-4):
    from tensorflow.keras.optimizers import AdamW
    return AdamW(learning_rate=lr, weight_decay=weight_decay)


def evaluate_h(p: pd.Series, te_pos, r_hat, h: int) -> dict:
    te_pos = list(te_pos)
    r_hat  = np.asarray(r_hat).flatten()
    y_true = p.iloc[[i + h for i in te_pos]].values
    y_pred = p.iloc[te_pos].values * (1 + r_hat)
    return evaluate_all(y_true, y_pred)



def drift_h(p: pd.Series, h: int, split=None):
    """Baseline DRIFT: prediksi return h-hari = rata-rata return h-hari pada
    data TRAIN (konstan utk semua hari test). Nol parameter yang dilatih,
    memakai split yang sama persis dengan model lain. Ini pembanding yang
    dipakai sebagai pembanding sederhana yang mempertahankan tren rata-rata
    return historis pada data latih. Drift bersifat diagnostik dan tidak ikut
    dalam 21 model utama.
    Return: metrics, rows, y_true, y_pred, mu
    """
    tr_dates, _, te_pos = split if split is not None else multihorizon_split(p, h)
    yh = (p.shift(-h) / p - 1).loc[tr_dates].values
    mu = float(np.mean(yh))
    te_pos = list(te_pos)
    y_true = p.iloc[[i + h for i in te_pos]].values
    y_pred = p.iloc[te_pos].values * (1.0 + mu)
    return evaluate_all(y_true, y_pred), te_pos, y_true, y_pred, mu
