"""Arsitektur model deep learning -- proyek Bao-aligned (2026).
Semua model memetakan window (seq_length, n_feat) -> 1 output (R_h).
Input window berakhir di posisi i (informasi masa lalu), target R_h(i).

Implementasi 11 arsitektur Deep Learning utama.
Konfigurasi batch size, hidden/embedding dimension, jumlah head, dan jumlah
encoder diatur di src/config.py berdasarkan Table 3 Bao et al. (2026)
sejauh informasi numeriknya tersedia. DPformer memakai tuning MOSSA pada
validation set; konfigurasi dan batas tuning dijelaskan di config.py dan README.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm(seq_length: int, units_1: int = 64, n_feat: int = 1):
    inp = keras.Input(shape=(seq_length, n_feat))
    x = layers.LSTM(units_1)(inp)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def build_gru(seq_length: int, units_1: int = 128, n_feat: int = 1):
    inp = keras.Input(shape=(seq_length, n_feat))
    x = layers.GRU(units_1)(inp)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def build_bilstm(seq_length: int, units_1: int = 128, n_feat: int = 1):
    inp = keras.Input(shape=(seq_length, n_feat))
    x = layers.Bidirectional(layers.LSTM(units_1))(inp)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def build_bigru(seq_length: int, units_1: int = 128, n_feat: int = 1):
    inp = keras.Input(shape=(seq_length, n_feat))
    x = layers.Bidirectional(layers.GRU(units_1))(inp)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def _transformer_encoder_block(x, num_heads=1, d_model=256, ff_dim=256):
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(1, d_model // num_heads))(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, att]))
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))


def build_transformer(seq_length: int, units_1: int = 256, n_feat: int = 1,
                      encoder_layers: int = 1):
    """Transformer baseline following the paper's explicit architecture text.

    Bao et al. states in the abstract/methodology that the Transformer module is
    *solely encoder-based*. Table 3 reports embedding size 256, one encoder head,
    one encoder layer and feedforward dimension 256. The published table also
    contains decoder columns; because this conflicts with the explicit encoder-only
    description, this project does not invent a decoder path and documents the
    discrepancy in the README.
    """
    d_model = units_1
    inp = keras.Input(shape=(seq_length, n_feat))
    x = layers.Dense(d_model)(inp)
    for _ in range(encoder_layers):
        x = _transformer_encoder_block(x, num_heads=1, d_model=d_model, ff_dim=256)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def build_dlinear(seq_length: int, units_1: int = 16, n_feat: int = 1):
    """DLinear: dekomposisi moving-average (trend) + sisaan (seasonal),
    dua jalur linear terpisah (Zeng et al., 2023)."""
    kernel = min(seq_length, max(3, seq_length // 2))
    inp = keras.Input(shape=(seq_length, n_feat))
    trend = layers.AveragePooling1D(pool_size=kernel, strides=1,
                                    padding='same')(inp)
    seasonal = layers.Subtract()([inp, trend])
    t = layers.Flatten()(trend)
    s = layers.Flatten()(seasonal)
    out = layers.Dense(1)(t) + layers.Dense(1)(s)
    return keras.Model(inp, out)


def build_nlinear(seq_length: int, units_1: int = 16, n_feat: int = 1):
    """NLinear: normalisasi dengan nilai terakhir window per channel, linear,
    lalu kembalikan nilai terakhir channel target (Zeng et al., 2023)."""
    inp = keras.Input(shape=(seq_length, n_feat))
    last = layers.Lambda(lambda z: z[:, -1:, :])(inp)
    x = layers.Subtract()([inp, last])
    x = layers.Flatten()(x)
    out_delta = layers.Dense(1)(x)
    last_target = layers.Lambda(lambda z: z[:, -1, 0:1])(inp)
    out = layers.Add()([out_delta, last_target])
    return keras.Model(inp, out)


def _attn_block(x, n_heads: int, d_model: int, ff_dim: int | None = None):
    """Satu blok encoder ringkas: self-attention -> Add&Norm -> feedforward ->
    Add&Norm. Dipakai berulang oleh PatchTST dan DPformer
    supaya jumlah 'Encoder Layers' bisa diatur sesuai Table 3 paper."""
    att = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d_model // n_heads))(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, att]))
    ff = layers.Dense(ff_dim or d_model, activation='relu')(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))
    return x


def build_patchtst(seq_length: int, units_1: int = 128, n_feat: int = 1,
                   n_heads: int = 4, encoder_layers: int = 3, patch_len: int = 2,
                   stride: int = 2):
    """PatchTST baseline using the Table 3 settings.

    Table 3 reports batch 256, hidden dimension 128, 4 heads, feedforward
    dimension 128 and 3 encoder layers. The table additionally shows a second
    PatchTST value of 16 under the label ``Hidden Dimension``; the published
    label is ambiguous, so this project does not reinterpret 16 as model width.
    Patch length is not specified numerically in the methodology, so patch_len=2
    and non-overlapping stride=2 are retained as explicit implementation choices.
    """
    d_model = units_1
    inp = keras.Input(shape=(seq_length, n_feat))
    patches = layers.Lambda(
        lambda z: tf.signal.frame(z, frame_length=patch_len, frame_step=stride, axis=1)
    )(inp)
    if seq_length < patch_len:
        raise ValueError(f"seq_length={seq_length} harus >= patch_len={patch_len}")
    n_patches = (seq_length - patch_len) // stride + 1
    x = layers.Reshape((n_patches, patch_len * n_feat))(patches)
    x = layers.Dense(d_model)(x)
    for _ in range(encoder_layers):
        x = _attn_block(x, n_heads, d_model, ff_dim=128)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def build_lstm_att(seq_length: int, units_1: int = 64, n_feat: int = 1):
    """LSTM + Bahdanau additive attention."""
    inp = keras.Input(shape=(seq_length, n_feat))
    h = layers.LSTM(units_1, return_sequences=True)(inp)
    score = layers.Dense(1, use_bias=False)(h)
    score = layers.Flatten()(score)
    alpha = layers.Activation('softmax')(score)
    alpha = layers.Reshape((seq_length, 1))(alpha)
    context = layers.Multiply()([h, alpha])
    context = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(context)
    out = layers.Dense(1)(context)
    return keras.Model(inp, out)


def build_gru_att(seq_length: int, units_1: int = 256, n_feat: int = 1):
    """GRU + Bahdanau additive attention."""
    inp = keras.Input(shape=(seq_length, n_feat))
    h = layers.GRU(units_1, return_sequences=True)(inp)
    score = layers.Dense(1, use_bias=False)(h)
    score = layers.Flatten()(score)
    alpha = layers.Activation('softmax')(score)
    alpha = layers.Reshape((seq_length, 1))(alpha)
    context = layers.Multiply()([h, alpha])
    context = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(context)
    out = layers.Dense(1)(context)
    return keras.Model(inp, out)


# ---------------------------------------------------------------------------
# DPformer (adaptasi dari Bao et al., 2026, Expert Systems With Applications)
# -- dekomposisi trend-seasonal (ala DLinear) lalu dua jalur paralel: trend
# lewat patch + self-attention, seasonal lewat conv1d + self-attention,
# digabung via bobot theta yang dilatih (Dense+sigmoid).
#
# DPformer menggunakan MOSSA untuk mencari (learning_rate, weight_decay).
# Fitness dihitung pada validation set saja; test set tidak dipakai untuk tuning.
# Arsitektur & kernel decomposisi memakai Ws=10 seperti paper, dengan batas seq_length untuk robustness.
# Interpretasi post-hoc DPformer menggunakan SHAP GradientExplainer
# pada notebook interpretasi; pendekatan ini dicatat sebagai adaptasi.
# ---------------------------------------------------------------------------

def _sinusoidal_pe(n_positions: int, d_model: int):
    """Positional encoding sinusoidal standar (Vaswani et al., 2017), dipakai
    persis pola yang dideskripsikan di paper DPformer (Eq. 8/18)."""
    pos = np.arange(n_positions, dtype=np.float32)[:, None]
    i = np.arange(d_model, dtype=np.float32)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
    angles = pos * angle_rates
    pe = np.zeros((n_positions, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def _add_pe(x, n_positions: int, d_model: int):
    """Tambah positional encoding konstan ke tensor simbolik. Dibungkus
    layers.Lambda -- op TF mentah tidak boleh langsung menerima KerasTensor
    simbolik pada Keras 3 / TF>=2.16."""
    pe = _sinusoidal_pe(n_positions, d_model)
    return layers.Lambda(lambda z: z + tf.constant(pe, dtype=z.dtype))(x)


def build_dpformer(seq_length: int, units_1: int = 128, n_feat: int = 1,
                   n_heads: int = 4, encoder_layers: int = 1,
                   ff_dim: int | None = None):
    """DPformer. units_1 = embedding size (paper Table 3: 128), ff_dim =
    Feedforward Dimension (paper: 256), n_heads = Encoder Heads (paper: 4),
    encoder_layers = Encoder Layers (paper: 1)."""
    d_model = units_1
    ff_dim = ff_dim or 256
    inp = keras.Input(shape=(seq_length, n_feat))

    # 1) Dekomposisi hanya pada harga emas (channel 0), sesuai Eq. (2)-(4).
    #    Bao et al. menetapkan Ws=10. Macro indicators kemudian disinkronkan
    #    dengan komponen trend/seasonal pada kedua pathway.
    kernel = min(10, seq_length)
    price = layers.Lambda(lambda z: z[:, :, 0:1])(inp)
    trend_price = layers.AveragePooling1D(pool_size=kernel, strides=1, padding='same')(price)
    seasonal_price = layers.Subtract()([price, trend_price])
    if n_feat > 1:
        macro = layers.Lambda(lambda z: z[:, :, 1:])(inp)
        trend = layers.Concatenate(axis=-1)([trend_price, macro])
        seasonal = layers.Concatenate(axis=-1)([seasonal_price, macro])
    else:
        trend, seasonal = trend_price, seasonal_price

    # 2) Jalur trend: patch -> embedding -> PE -> N blok encoder.
    patch_len, stride = 2, 1
    if seq_length < patch_len:
        raise ValueError(f"seq_length={seq_length} harus >= patch_len={patch_len}")
    n_patches = (seq_length - patch_len) // stride + 1
    t_patches = layers.Lambda(
        lambda z: tf.signal.frame(z, frame_length=patch_len, frame_step=stride, axis=1)
    )(trend)
    xt = layers.Reshape((n_patches, patch_len * n_feat))(t_patches)
    xt = layers.Dense(d_model)(xt)
    xt = _add_pe(xt, n_patches, d_model)
    for _ in range(encoder_layers):
        xt = _attn_block_ff(xt, n_heads, d_model, ff_dim)
    xt = layers.Flatten()(xt)
    xt = layers.Dense(256, activation='relu')(xt)
    trend_out = layers.Dense(1, name='trend_out')(xt)

    # 3) Jalur seasonal: conv1d (Ks=5 di paper) -> PE -> N blok encoder.
    conv_k = min(5, seq_length)
    xs = layers.Conv1D(d_model, kernel_size=conv_k, padding='same')(seasonal)
    xs = _add_pe(xs, seq_length, d_model)
    for _ in range(encoder_layers):
        xs = _attn_block_ff(xs, n_heads, d_model, ff_dim)
    xs = layers.Flatten()(xs)
    xs = layers.Dense(256, activation='relu')(xs)
    seasonal_out = layers.Dense(1, name='seasonal_out')(xs)

    # 4) Fusion: P = theta*trend + (1-theta)*seasonal, theta dipelajari
    #    (paper tidak menjelaskan cara theta ditentukan).
    concat = layers.Concatenate()([trend_out, seasonal_out])
    theta = layers.Dense(1, activation='sigmoid', name='theta')(concat)
    inv_theta = layers.Lambda(lambda z: 1.0 - z)(theta)
    out = layers.Add()([
        layers.Multiply()([theta, trend_out]),
        layers.Multiply()([inv_theta, seasonal_out]),
    ])
    return keras.Model(inp, out)


def _attn_block_ff(x, n_heads: int, d_model: int, ff_dim: int):
    """Sama seperti _attn_block, tapi feedforward pakai dimensi terpisah
    (ff_dim) dari d_model -- sesuai Table 3 DPformer ('Feedforward Dimension'
    beda dari 'Embedding Size')."""
    att = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d_model // n_heads))(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, att]))
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))
    return x


def mossa_tune(fitness_fn, n_pop: int = 8, n_iter: int = 10,
               lr_range=(1e-4, 1e-2), wd_range=(0.0, 1e-2), seed: int = 42,
               trace: list | None = None, verbose: bool = True,
               checkpoint_path: str | None = None, ckpt_meta: dict | None = None,
               resume: bool = True, patience_iter: int | None = None,
               min_rel_delta: float = 1e-3):
    """Sparrow Search Algorithm (SSA) versi SEDERHANA & SINGLE-OBJECTIVE untuk
    tuning (learning_rate, weight_decay) -- mengikuti struktur populasi paper
    section 2.4 (~20% discoverer, ~10% scout/early-warner, sisanya joiner),
    tetapi BUKAN implementasi multi-objective penuh: fitness hanya satu angka.
    Dalam laporan sebut sebagai 'SSA/MOSSA disederhanakan'.

    fitness_fn(lr, wd) -> float, MAKIN KECIL MAKIN BAIK. Pemanggil WAJIB
    memakai metrik VALIDASI (train_dl_h/train_hybrid_h(..., eval_split='val'))
    -- fungsi ini tidak tahu data apa yang dipakai fitness_fn.

    Perbaikan dari versi sebelumnya (yang membuat hasil menempel di batas ruang
    cari: lr=1e-4 / lr=wd=1e-2):
      * pencarian dilakukan di ruang ternormalisasi z in [-1, 1]^2 berpusat di
        0; lr dipetakan di skala LOG (10^..), wd linear -> aturan SSA
        ('menyusut ke pusat', langkah N(0,1)) bekerja pada skala yang benar,
        bukan pada satuan mentah lr/wd.
      * peringkat (bukan indeks populasi) dipakai untuk memilah joiner kuat/lemah.
      * populasi awal Latin-hypercube (sebaran merata walau n_pop kecil).
      * langkah scout memakai selisih fitness ternormalisasi (bebas skala MAE).
      * kandidat yang tidak berubah tidak dievaluasi ulang.
      * fitness NaN/inf diperlakukan sebagai sangat buruk.

    CATATAN KOMPUTASI: mode FULL memakai population=20 dan iterasi=100 sesuai
    paper, sedangkan mode QUICK mengecilkan budget hanya untuk smoke test.
    Epoch fitness diatur oleh pemanggil sebagai penyesuaian biaya komputasi.

    trace: list opsional; tiap evaluasi ditambahkan sebagai dict
           {'lr','wd','fit'} (untuk transparansi / dicek di notebook).
    checkpoint_path: bila diisi, seluruh keadaan pencarian (populasi, fitness,
           history, trace, keadaan RNG) ditulis ke file JSON ini setelah populasi
           awal dan SETIAP iterasi (atomik: tulis .tmp lalu os.replace). Jika
           proses mati / laptop macet, panggil ulang dengan checkpoint_path yang
           sama: pencarian lanjut dari iterasi terakhir yang selesai dan
           menghasilkan urutan yang sama dengan run tanpa gangguan. Checkpoint
           hanya dipakai bila konfigurasi (n_pop, n_iter, rentang, seed,
           ckpt_meta) identik; selain itu diabaikan dan pencarian mulai dari awal.
    ckpt_meta: dict identitas protokol (protocol, fitness_split, setup, h, epochs)
           yang ikut dibandingkan saat resume.
    resume: False = abaikan checkpoint yang ada (mulai dari awal, lalu menimpa).
           Checkpoint dengan konfigurasi/protokol BEDA tidak ditimpa begitu saja: file
           lama diamankan sebagai <checkpoint>.beda-konfigurasi.bak.
    patience_iter: None (default) = jalankan sampai n_iter, sesuai jurnal. Bila diisi
           (mis. 20), pencarian berhenti dini jika best_fit belum membaik sedikitnya
           min_rel_delta (relatif; default 0.1%) selama patience_iter iterasi berturut-
           turut; n_iter tetap menjadi batas maksimum. Aturan ini TIDAK memengaruhi
           lintasan pencarian, sehingga checkpoint yang dibuat tanpa aturan boleh
           dilanjutkan dengan aturan ini (dan sebaliknya). Catat di laporan sebagai
           adaptasi komputasi (kriteria konvergensi).
    Error memori (MemoryError / ResourceExhaustedError) SENGAJA tidak ditelan
    menjadi 'fitness buruk': pencarian berhenti dan bisa dilanjutkan dari
    checkpoint setelah memori dibebaskan.
    Return: best_lr, best_wd, history
    """
    import json, os
    rng = np.random.default_rng(seed)
    n_disc = max(1, int(round(0.2 * n_pop)))
    n_scout = max(1, int(round(0.1 * n_pop)))
    lo_lr, hi_lr = np.log10(lr_range[0]), np.log10(lr_range[1])

    def decode(z):
        lr = 10 ** (lo_lr + (z[0] + 1.0) / 2.0 * (hi_lr - lo_lr))
        wd = wd_range[0] + (z[1] + 1.0) / 2.0 * (wd_range[1] - wd_range[0])
        return float(lr), float(wd)

    def evaluate(z):
        lr, wd = decode(z)
        try:
            f = float(fitness_fn(lr, wd))
        except (MemoryError, tf.errors.ResourceExhaustedError):
            raise                                   # kehabisan memori: bukan "fitness buruk"
        except Exception as e:                      # training meledak -> sangat buruk
            if verbose:
                print(f'    (evaluasi gagal lr={lr:.2e} wd={wd:.2e}: {type(e).__name__})')
            f = float('inf')
        if not np.isfinite(f):
            f = 1e18
        if trace is not None:
            trace.append({'lr': lr, 'wd': wd, 'fit': f})
        return f

    def snapshot(it):
        bi = int(fit.argmin())
        lr_b, wd_b = decode(z[bi])
        return {'iter': it, 'best_fit': float(fit[bi]), 'best_lr': lr_b, 'best_wd': wd_b}

    cfg = {'n_pop': n_pop, 'n_iter': n_iter, 'lr_range': list(lr_range),
           'wd_range': list(wd_range), 'seed': seed, **(ckpt_meta or {})}

    def save_ckpt(it):
        if not checkpoint_path:
            return
        d = os.path.dirname(checkpoint_path)
        if d:
            os.makedirs(d, exist_ok=True)
        state = {'cfg': cfg, 'it': it, 'z': z.tolist(), 'fit': fit.tolist(),
                 'history': history, 'trace': list(trace) if trace is not None else [],
                 'rng': rng.bit_generator.state}
        tmp = checkpoint_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(state, fh)
        os.replace(tmp, checkpoint_path)          # atomik: file lama tak pernah setengah-tertulis

    ck = None
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, encoding='utf-8') as fh:
                saved = json.load(fh)
            if saved.get('cfg') == cfg:
                ck = {'it': int(saved['it']), 'z': np.array(saved['z'], dtype=float),
                      'fit': np.array(saved['fit'], dtype=float),
                      'history': list(saved['history']), 'trace': list(saved['trace']),
                      'rng': saved['rng']}
                if ck['z'].shape != (n_pop, 2) or ck['fit'].shape != (n_pop,):
                    raise ValueError('bentuk checkpoint tidak cocok')
            else:
                try:
                    os.replace(checkpoint_path, checkpoint_path + '.beda-konfigurasi.bak')
                except OSError:
                    pass
                if verbose:
                    print('    (checkpoint MOSSA diabaikan: konfigurasi/protokol berbeda -> mulai dari awal; '
                          'file lama diamankan sebagai .beda-konfigurasi.bak)')
        except Exception as e:                      # file rusak / terpotong
            ck = None
            if verbose:
                print(f'    (checkpoint MOSSA tidak terbaca: {type(e).__name__} -> mulai dari awal)')

    if ck is not None:
        z, fit, history = ck['z'], ck['fit'], ck['history']
        rng.bit_generator.state = ck['rng']
        if trace is not None:
            trace[:] = ck['trace']
        start_it = ck['it'] + 1
        if verbose:
            print(f'    MOSSA: lanjut dari checkpoint (iterasi {ck["it"]}/{n_iter} sudah selesai)')
    else:
        # populasi awal: Latin hypercube di [-1, 1]^2
        u = np.stack([(rng.permutation(n_pop) + rng.uniform(size=n_pop)) / n_pop
                      for _ in range(2)], axis=1)
        z = 2.0 * u - 1.0
        fit = np.array([evaluate(zi) for zi in z])

        history = [snapshot(0)]
        if verbose:
            h0 = history[0]
            print(f'    MOSSA iter  0/{n_iter}: best_fit={h0["best_fit"]:.2f} '
                  f'lr={h0["best_lr"]:.2e} wd={h0["best_wd"]:.2e}')
        save_ckpt(0)
        start_it = 1

    # pelacak perbaikan berarti (dari history -> benar juga saat resume)
    ref_best, last_gain_it = history[0]['best_fit'], 0
    for rec in history[1:]:
        if rec['best_fit'] < ref_best * (1.0 - min_rel_delta):
            ref_best, last_gain_it = rec['best_fit'], rec['iter']

    for it in range(start_it, n_iter + 1):
        if patience_iter and (it - 1) - last_gain_it >= patience_iter:
            if verbose:
                print(f'    MOSSA berhenti dini di iterasi {it - 1}/{n_iter}: best_fit tidak membaik '
                      f'>= {100 * min_rel_delta:g}% selama {patience_iter} iterasi berturut-turut')
            break
        order = np.argsort(fit)                     # terbaik -> terburuk
        rank = np.empty(n_pop, dtype=int)
        rank[order] = np.arange(n_pop)
        best, worst = z[order[0]].copy(), z[order[-1]].copy()
        span = max(fit[order[-1]] - fit[order[0]], 1e-12)

        new = z.copy()
        for i in order[:n_disc]:                    # discoverer: eksplorasi global
            if rng.uniform() < 0.8:                 # aman -> menyusut ke pusat ruang
                new[i] = z[i] * np.exp(-it / (rng.uniform(0.1, 1.0) * n_iter))
            else:                                   # ada predator -> lompatan acak
                new[i] = z[i] + rng.normal(0, 1, size=2) * 0.15

        for i in order[n_disc:n_pop - n_scout]:     # joiner
            if rank[i] > n_pop / 2:                 # joiner lemah: terbang mencari makan
                new[i] = 0.3 * rng.normal(0, 1, size=2) * np.exp(
                    (worst - z[i]) / ((rank[i] + 1) ** 2))
            else:                                   # joiner kuat: mengikuti discoverer terbaik
                new[i] = best + np.abs(z[i] - best) * rng.normal(0, 1, size=2)

        for i in rng.choice(n_pop, size=n_scout, replace=False):   # scout (early-warner)
            if fit[i] > fit[order[0]]:              # di tepi kawanan -> merapat ke terbaik
                new[i] = best + rng.normal(0, 1, size=2) * np.abs(z[i] - best)
            else:                                   # yang terbaik waspada -> menjauh dari terburuk
                gap = (fit[order[-1]] - fit[i]) / span          # 0 (= terburuk) .. 1 (= terbaik)
                step = rng.uniform(-1, 1) * np.abs(z[i] - worst) / (gap + 0.25)
                new[i] = z[i] + np.clip(step, -0.5, 0.5)

        new = np.clip(new, -1.0, 1.0)
        changed = np.any(np.abs(new - z) > 1e-12, axis=1)
        new_fit = fit.copy()
        for i in np.where(changed)[0]:
            new_fit[i] = evaluate(new[i])
        improved = new_fit < fit
        z[improved] = new[improved]
        fit[improved] = new_fit[improved]

        history.append(snapshot(it))
        if verbose:
            hi_ = history[-1]
            print(f'    MOSSA iter {it:2d}/{n_iter}: best_fit={hi_["best_fit"]:.2f} '
                  f'lr={hi_["best_lr"]:.2e} wd={hi_["best_wd"]:.2e}')
        if history[-1]['best_fit'] < ref_best * (1.0 - min_rel_delta):
            ref_best, last_gain_it = history[-1]['best_fit'], it
        save_ckpt(it)

    bi = int(fit.argmin())
    best_lr, best_wd = decode(z[bi])
    return best_lr, best_wd, history


BUILDERS = {
    'lstm':        build_lstm,
    'gru':         build_gru,
    'bilstm':      build_bilstm,
    'bigru':       build_bigru,
    'transformer': build_transformer,
    'dlinear':     build_dlinear,
    'nlinear':     build_nlinear,
    'patchtst':    build_patchtst,
    'lstm_att':    build_lstm_att,
    'gru_att':     build_gru_att,
    'dpformer':    build_dpformer,
}

