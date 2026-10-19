"""Pagar integritas eksperimen.

Fokus:
1. Fitness DPformer tidak boleh berubah ketika data test dirusak.
2. Fitur tabular train/validation tidak boleh berubah ketika data test dirusak.
3. History DL diberi diagnosis overfit/underfit sebagai PERINGATAN, bukan
   alasan memilih hasil test.
4. Early stopping run FINAL (mode test) tidak boleh melihat harga periode test:
   model terpilih tidak berubah ketika harga periode test dirusak (embargo validasi).
5. Walk-forward: tiap fold tidak boleh memakai label yang jatuh pada/ sesudah origin uji pertama.
6. Skenario B (eksogen hari-t): fitur di origin t tidak boleh bergantung pada pasar hari t+1 dst.
7. SVR: prediksi harus ekuivarian terhadap skala target (mencegah epsilon salah skala).
"""
import numpy as np
import pandas as pd

from .pipeline import multihorizon_split, free_memory


def cek_fitness_tanpa_kebocoran(
    p, exo, h, trainer, build_fn, seq_len, units, batch_size
):
    _, _, te_pos = multihorizon_split(p, h)
    te0 = te_pos[0]

    p_bad = p.copy()
    p_bad.iloc[te0:] = p_bad.iloc[te0:] * 3.0

    exo_bad = None
    if exo is not None:
        exo_bad = exo.copy()
        exo_bad.iloc[te0:] = exo_bad.iloc[te0:] * 7.0 + 0.5

    def run(pp, ee, split):
        out = trainer(
            pp, ee, h, build_fn, seq_len, units,
            epochs=2, patience=2, batch_size=batch_size,
            learning_rate=1e-3, weight_decay=1e-4,
            seed=42, eval_split=split,
        )
        mae = float(out[1]["MAE"])
        free_memory()
        return mae

    v1 = run(p, exo, "val")
    v2 = run(p, exo, "val")
    v_bad = run(p_bad, exo_bad, "val")
    t1 = run(p, exo, "test")
    t_bad = run(p_bad, exo_bad, "test")

    noise = abs(v1 - v2)
    # Keras GPU/CPU determinism bisa tidak sempurna. Toleransi relatif terhadap
    # run ulang tetap kecil; perubahan besar akibat test mutation adalah leakage.
    assert abs(v1 - v_bad) <= max(10 * noise, 1e-4 * max(abs(v1), 1.0)), (
        f"KEBOCORAN: fitness validation berubah {v1:.6f} -> {v_bad:.6f}"
    )
    assert abs(t1 - t_bad) > 1e-3 * max(abs(t1), 1.0), (
        "Guard tidak sensitif: metrik test tidak berubah setelah test dirusak."
    )
    return {
        "mae_val_asli": v1,
        "mae_val_test_dirusak": v_bad,
        "noise_run_ulang": noise,
        "mae_test_asli": t1,
        "mae_test_test_dirusak": t_bad,
    }


def cek_fiturs_test_tidak_mengubah_train_val(p, exo, h, feature_builder):
    tr, va, te = multihorizon_split(p, h)
    te0 = te[0]
    p_bad = p.copy()
    p_bad.iloc[te0:] *= 3.0
    exo_bad = exo.copy() if exo is not None else None
    if exo_bad is not None:
        exo_bad.iloc[te0:] = exo_bad.iloc[te0:] * 7.0 + 0.5

    F1 = feature_builder(p, exo)
    F2 = feature_builder(p_bad, exo_bad)
    common = F1.index.intersection(F2.index)
    trainval_dates = list(tr) + list(va)
    common_tv = common.intersection(pd.DatetimeIndex(trainval_dates))
    a = F1.loc[common_tv].to_numpy()
    b = F2.loc[common_tv].to_numpy()
    ok = np.allclose(a, b, equal_nan=True)
    assert ok, "KEBOCORAN: fitur train/validation berubah setelah data test dirusak."
    return {"trainval_rows_checked": len(common_tv), "status": "OK"}


def _rusak_test(p, exo, h):
    """Salinan p/exo dengan periode test dirusak (cara yang sama dgn guard lain)."""
    te0 = multihorizon_split(p, h)[2][0]
    p_bad = p.copy()
    p_bad.iloc[te0:] = p_bad.iloc[te0:] * 3.0
    exo_bad = None
    if exo is not None:
        exo_bad = exo.copy()
        exo_bad.iloc[te0:] = exo_bad.iloc[te0:] * 7.0 + 0.5
    return p_bad, exo_bad


def cek_validasi_final_tanpa_label_test_ml(p, exo, h, feature_builder, lgb_params=None):
    """Mode FINAL (test), LightGBM: iterasi early stopping dan prediksi pada baris
    validasi yang sama tidak boleh berubah ketika harga periode test dirusak.
    Menutup celah lama: label validasi (i+h) bisa jatuh di periode test."""
    from .pipeline import train_lightgbm_h
    p_bad, exo_bad = _rusak_test(p, exo, h)
    F1, F2 = feature_builder(p, exo), feature_builder(p_bad, exo_bad)
    m1, met1, *_ = train_lightgbm_h(p, F1, h, lgb_params)
    m2, met2, *_ = train_lightgbm_h(p_bad, F2, h, lgb_params)
    tr, va, te = multihorizon_split(p, h)
    va_rows = [d for d in va if d in F1.index and d in F2.index]
    X = F1.loc[va_rows][list(F1.columns)]
    selisih = float(np.max(np.abs(m1.predict(X) - m2.predict(X))))
    # RMSE validasi ikut dibandingkan: tetap sensitif walau early stopping berhenti di
    # iterasi awal (model hampir konstan), karena label validasi yang tercemar harga
    # periode test akan mengubah RMSE secara drastis.
    rmse1 = float(m1.best_score["valid_0"]["rmse"])
    rmse2 = float(m2.best_score["valid_0"]["rmse"])
    assert (m1.best_iteration == m2.best_iteration and selisih <= 1e-12
            and abs(rmse1 - rmse2) <= 1e-12), (
        f"KEBOCORAN: model final berubah ketika test dirusak "
        f"(best_iter {m1.best_iteration} -> {m2.best_iteration}, "
        f"RMSE validasi {rmse1:.6g} -> {rmse2:.6g}, selisih prediksi {selisih:.3e})"
    )
    assert abs(met1["MAE"] - met2["MAE"]) > 1e-3 * max(abs(met1["MAE"]), 1.0), (
        "Guard tidak sensitif: metrik test tidak berubah setelah test dirusak."
    )
    return {"best_iter_asli": m1.best_iteration, "best_iter_test_dirusak": m2.best_iteration,
            "rmse_val_asli": rmse1, "rmse_val_test_dirusak": rmse2,
            "selisih_prediksi_val": selisih, "mae_test_asli": met1["MAE"],
            "mae_test_test_dirusak": met2["MAE"]}


def cek_validasi_final_tanpa_label_test_dl(
    p, exo, h, trainer, build_fn, seq_len, units, batch_size
):
    """Mode FINAL (test), DL: val_loss per epoch (dasar early stopping/ReduceLROnPlateau)
    tidak boleh berubah ketika harga periode test dirusak."""
    p_bad, exo_bad = _rusak_test(p, exo, h)

    def run(pp, ee):
        out = trainer(
            pp, ee, h, build_fn, seq_len, units,
            epochs=2, patience=2, batch_size=batch_size,
            learning_rate=1e-3, weight_decay=1e-4,
            seed=42, eval_split="test",
        )
        val_loss = np.asarray(out[5]["val_loss"], dtype=float)
        mae = float(out[1]["MAE"])
        free_memory()
        return val_loss, mae

    v1, m1 = run(p, exo)
    v2, _ = run(p, exo)
    vb, mb = run(p_bad, exo_bad)
    noise = float(np.max(np.abs(v1 - v2)))
    selisih = float(np.max(np.abs(v1 - vb)))
    assert len(v1) == len(vb) and selisih <= max(10 * noise, 1e-4 * max(float(np.max(np.abs(v1))), 1.0)), (
        f"KEBOCORAN: val_loss run final berubah ketika test dirusak (selisih {selisih:.3e})"
    )
    assert abs(m1 - mb) > 1e-3 * max(abs(m1), 1.0), (
        "Guard tidak sensitif: metrik test tidak berubah setelah test dirusak."
    )
    return {"val_loss_epoch_terakhir_asli": float(v1[-1]),
            "val_loss_epoch_terakhir_test_dirusak": float(vb[-1]),
            "noise_run_ulang": noise, "mae_test_asli": m1, "mae_test_test_dirusak": mb}


def diagnosis_history(history, gap_warn=0.25, ratio_warn=3.0):
    if not history or "loss" not in history or "val_loss" not in history:
        return {"status": "NO_HISTORY"}

    loss = np.asarray(history["loss"], dtype=float)
    val = np.asarray(history.get("val_loss", []), dtype=float)
    if len(loss) == 0 or len(val) == 0:
        return {"status": "NO_VALIDATION_HISTORY"}

    best = int(np.argmin(val))
    train_best = float(loss[min(best, len(loss)-1)])
    val_best = float(val[best])
    final_train = float(loss[-1])
    final_val = float(val[-1])
    gap = abs(val_best - train_best) / max(abs(val_best), 1e-8)
    ratio = val_best / max(train_best, 1e-8)

    if best < len(val) - 1 and final_val > val_best * 1.10:
        status = "OVERFIT_WARNING"
    elif ratio > ratio_warn or gap > gap_warn:
        status = "GENERALIZATION_GAP_WARNING"
    elif best >= len(val) - 2 and final_train > 1e-3 and final_val > 1e-3:
        status = "UNDERFIT_POSSIBLE"
    else:
        status = "OK"

    return {
        "status": status,
        "best_epoch": best + 1,
        "train_loss_at_best": train_best,
        "val_loss_best": val_best,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "val_train_ratio_at_best": ratio,
        "relative_gap_at_best": gap,
    }


# ---------------------------------------------------------------------------
# Pagar tambahan (versi final)
# ---------------------------------------------------------------------------
def cek_walkforward_embargo(p, h, n_folds, fold_len):
    """Struktur split walk-forward: blok uji berurutan tanpa celah/tumpang-tindih, blok terakhir
    berakhir di origin terakhir yang punya target, dan SEMUA label train+val (i+h) jatuh
    sebelum origin uji pertama (tidak ada label yang menyentuh periode uji)."""
    from .pipeline import walkforward_splits
    folds = walkforward_splits(p, h, n_folds, fold_len)
    pos_of = {d: i for i, d in enumerate(p.index)}
    last_origin = len(p) - h - 1
    prev_end = None
    for f in folds:
        tr, va, te = f['split']
        s = te[0]
        assert te == list(range(s, s + fold_len)), 'blok uji tidak kontigu'
        if prev_end is not None:
            assert s == prev_end + 1, 'blok uji tumpang-tindih / berjarak'
        prev_end = te[-1]
        for d in list(tr) + list(va):
            assert pos_of[d] + h < s, f'fold {f["fold"]}: label train/val menyentuh periode uji'
        assert len(va) >= 1 and pos_of[tr[-1]] < pos_of[va[0]], 'train/val tidak kronologis'
    assert prev_end == last_origin, 'blok terakhir tidak berakhir di origin terakhir'
    return {'folds': len(folds), 'status': 'OK'}


def cek_walkforward_tanpa_label_test_ml(p, exo, h, feature_builder, fold_idx=0, n_folds=3,
                                        fold_len=120, lgb_params=None):
    """Satu fold walk-forward, LightGBM: iterasi early stopping & RMSE validasi tidak boleh
    berubah ketika harga dari origin uji pertama dan sesudahnya dirusak."""
    from .pipeline import train_lightgbm_h, walkforward_splits
    folds = walkforward_splits(p, h, n_folds, fold_len)
    f = folds[fold_idx]
    s = f['split'][2][0]
    p_bad = p.copy()
    p_bad.iloc[s:] = p_bad.iloc[s:] * 3.0
    exo_bad = None
    if exo is not None:
        exo_bad = exo.copy()
        exo_bad.iloc[s:] = exo_bad.iloc[s:] * 7.0 + 0.5
    F1, F2 = feature_builder(p, exo), feature_builder(p_bad, exo_bad)
    m1, met1, *_ = train_lightgbm_h(p, F1, h, lgb_params, split=f['split'])
    m2, met2, *_ = train_lightgbm_h(p_bad, F2, h, lgb_params, split=f['split'])
    rmse1 = float(m1.best_score['valid_0']['rmse'])
    rmse2 = float(m2.best_score['valid_0']['rmse'])
    assert m1.best_iteration == m2.best_iteration and abs(rmse1 - rmse2) <= 1e-12, (
        f'KEBOCORAN walk-forward: model berubah saat harga uji dirusak '
        f'(best_iter {m1.best_iteration}->{m2.best_iteration}, RMSE val {rmse1:.6g}->{rmse2:.6g})')
    assert abs(met1['MAE'] - met2['MAE']) > 1e-3 * max(abs(met1['MAE']), 1.0), \
        'Guard tidak sensitif: metrik uji tidak berubah setelah harga uji dirusak.'
    return {'best_iter': m1.best_iteration, 'rmse_val': rmse1, 'status': 'OK'}


def cek_skenario_b_tanpa_lookahead(levels):
    """Skenario B: eksogen di origin t memuat pasar sampai hari t. Merusak level pasar pada
    hari-hari SESUDAH t0 tidak boleh mengubah fitur di origin <= t0 (dan harus mengubah origin t0+1,
    sebagai bukti uji ini sensitif). `levels` = DataFrame level mentah (EXO_LEVEL_COLS) pada
    kalender hari kerja emas, BELUM di-lag."""
    from .pipeline import make_exo_features, make_exo_scenario_b, EXO_B_SAMEDAY
    lag = make_exo_features(levels).shift(1)
    b = make_exo_scenario_b(lag)
    t0 = len(levels) // 2
    bad = levels.copy()
    bad.iloc[t0 + 1:] = bad.iloc[t0 + 1:] * 3.0 + 1.0
    b_bad = make_exo_scenario_b(make_exo_features(bad).shift(1))
    a = b.iloc[:t0 + 1][EXO_B_SAMEDAY].to_numpy()
    c = b_bad.iloc[:t0 + 1][EXO_B_SAMEDAY].to_numpy()
    assert np.allclose(a, c, equal_nan=True), 'LOOKAHEAD: fitur skenario B berubah oleh pasar hari-hari berikutnya'
    nxt = np.abs(b.iloc[t0 + 1][EXO_B_SAMEDAY].to_numpy(float) - b_bad.iloc[t0 + 1][EXO_B_SAMEDAY].to_numpy(float))
    assert np.nanmax(nxt) > 0, 'Guard tidak sensitif: perusakan data tidak mengubah fitur origin berikutnya'
    # kolom non-pasar (FFR, BI-Rate) harus tetap lag-1: identik dengan skenario utama
    others = [c for c in b.columns if c not in EXO_B_SAMEDAY]
    assert b[others].equals(lag[others]), 'FFR/BI-Rate harus tetap lag-1 pada skenario B'
    return {'origin_diperiksa': t0 + 1, 'status': 'OK'}


def cek_svr_skala_target(p, exo, h, feature_builder, faktor=10.0):
    """SVR harus ekuivarian terhadap skala target: melatih pada target * faktor menghasilkan
    prediksi * faktor (epsilon dalam satuan simpangan baku target). Versi lama (epsilon tetap
    pada target mentah) gagal uji ini."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    tr, va, te = multihorizon_split(p, h)
    F = feature_builder(p, exo)
    pos_of = {d: i for i, d in enumerate(p.index)}
    tr_ok = [d for d in tr if d in F.index]
    te_ok = [p.index[i] for i in te if p.index[i] in F.index]
    y = np.array([p.iloc[pos_of[d] + h] / p.loc[d] - 1 for d in tr_ok])

    def fit_pred(scale):
        m = TransformedTargetRegressor(
            regressor=Pipeline([('scaler', StandardScaler()), ('model', SVR(C=1.0, gamma='scale', epsilon=0.1))]),
            transformer=StandardScaler())
        m.fit(F.loc[tr_ok].values, y * scale)
        return m.predict(F.loc[te_ok].values)

    a, b = fit_pred(1.0), fit_pred(faktor)
    assert np.allclose(a * faktor, b, rtol=1e-6, atol=1e-9), 'SVR tidak ekuivarian terhadap skala target'
    return {'status': 'OK', 'std_prediksi': float(np.std(a))}


def cek_prediksi_sejajar_data(runs_dir, p, labels, models, horizons, split_fn=None, runs_valid_only=True):
    """Setiap *_pred.csv tersimpan harus (a) bertanggal = tanggal target (i+h) baris uji split utama,
    (b) memuat y_true == harga pada data SAAT INI (hasil & data konsisten: tidak ada hasil lama dari
    data yang berbeda), (c) y_pred berhingga, (d) jumlah baris == jumlah origin uji.
    Return DataFrame ringkasan; AssertionError bila ada yang tidak cocok."""
    import os
    split_fn = split_fn or multihorizon_split
    rows, salah = [], []
    for lb in labels:
        for m in models:
            for h in horizons:
                f = os.path.join(runs_dir, f'{lb}_{m}_h{h}_pred.csv')
                if not os.path.exists(f):
                    continue
                d = pd.read_csv(f, parse_dates=['tanggal'])
                _, _, te_pos = split_fn(p, h)
                tgt = p.index[np.asarray(te_pos, int) + h]
                ok_tgl = len(d) == len(tgt) and bool((pd.DatetimeIndex(d.tanggal) == tgt).all())
                ytrue = p.reindex(pd.DatetimeIndex(d.tanggal)).to_numpy(float)
                selisih = float(np.nanmax(np.abs(d.y_true.to_numpy(float) - ytrue))) if len(d) else np.nan
                ok_y = bool(np.isfinite(selisih) and selisih <= 1e-6 * max(1.0, float(np.nanmax(np.abs(ytrue)))))
                ok_pred = bool(np.isfinite(d.y_pred.to_numpy(float)).all())
                rows.append({'setup': lb, 'model': m, 'horizon': h, 'n': len(d), 'tanggal_ok': ok_tgl,
                             'y_true_ok': ok_y, 'y_pred_finite': ok_pred, 'maks_selisih_ytrue': selisih})
                if not (ok_tgl and ok_y and ok_pred):
                    salah.append(f'{lb}/{m}/h{h}')
    out = pd.DataFrame(rows)
    assert not salah, f'prediksi tidak sejajar dengan data/split saat ini: {salah[:10]} (total {len(salah)})'
    return out
