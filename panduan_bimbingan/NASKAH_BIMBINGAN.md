# Naskah bimbingan: menjawab "gimana hasilnya?"

Naskah ini untuk dibaca lalu diucapkan dengan kata-kata sendiri, tidak perlu dihafal. Kalimat dalam tanda kutip hanya contoh. Angkanya sama dengan lembar `Jawaban` di [`ringkasan_eksperimen.xlsx`](ringkasan_eksperimen.xlsx), yang ditunjukkan ke dosen, dan dengan [`PENJELASAN.md`](PENJELASAN.md), catatan belajar untuk dibaca sebelum bimbingan.

Waktu: sekitar 3 sampai 4 menit untuk alur utama, lalu tanya jawab.

## Sebelum masuk

- Buka `ringkasan_eksperimen.xlsx`. File langsung terbuka di lembar `Jawaban`.
- Siapkan [`../diagram/alur_penelitian_lebar.png`](../diagram/alur_penelitian_lebar.png) kalau Ibu bertanya soal alur penelitian.
- `PENJELASAN.md` dibaca sebelum bimbingan (bab 01, 05, 09, 12 dan Lampiran C), tidak perlu dibuka di depan dosen.
- Isi dulu bagian `[ISI: ...]` di `PENJELASAN.md` untuk Q9 (penggunaan AI) dan Q11 (alasan walk-forward). Hanya penulis yang tahu jawabannya.

## 1. Pembuka (sekitar 20 detik)

Kalau Ibu bertanya "gimana hasilnya?", jawab langsung:

> "Hasilnya begini, Bu. Belum ada model yang terbukti mengalahkan Drift setelah koreksi uji berganda. Tapi ada satu pola yang konsisten, yaitu memakai penutupan pasar hari-t untuk ramalan 1 hari, dan itu masih eksploratif."

## 2. Alur penjelasan

Tiap langkah ada bagian di lembar `Jawaban` yang bisa ditunjuk.

**a. Apa yang diuji** (20 detik). Layar: bagian atas lembar.

> "Saya menguji apakah harga jual emas Antam bisa diramal 1, 5, 10, 20, dan 30 hari kerja ke depan. Ada 21 model, masing-masing dengan dan tanpa 7 variabel makro, jadi 210 eksperimen. Semuanya dibandingkan dengan Drift, yaitu prediksi rata-rata return periode latih yang nilainya tetap. Data dibagi urut waktu 70/10/20 dengan embargo, supaya data uji tidak ikut terbaca waktu pelatihan."

**b. Angka terbaik** (30 detik). Layar: tabel "Hasil per horizon".

> "Kalau dilihat dari angkanya, model terbaik di tiap horizon sedikit lebih baik dari Drift. Di h=1, PatchTST salah rata-rata Rp 13.170 per gram, Drift Rp 13.247, jadi hampir sama. Selisih paling besar di h=20, AdaBoost sekitar 6,8% lebih kecil. Tapi model terbaik ini saya pilih dari 42 kombinasi setelah melihat hasil uji, jadi selisihnya kemungkinan terlalu optimis."

**c. Kenapa belum terbukti** (40 detik). Layar: tabel "Angka penting", empat baris pertama.

> "Dari 210 eksperimen, 115 lebih baik dari Drift dan 16 punya p di bawah 0,05. Tapi dengan 210 uji, kalau murni kebetulan saja sudah wajar ada sekitar 10 yang lolos. Setelah koreksi uji berganda, tidak ada yang lolos di keluarga 105 maupun 210 uji. Hanya satu yang lolos di keluarga paling kecil, Transformer univariat h=1, dan MSE-nya cuma 0,5% lebih rendah. Di Model Confidence Set, Drift juga ikut masuk di semua 10 kombinasi."

**d. Makro dan model terbaik** (25 detik). Layar: bagian "Kalau ditanya".

> "Variabel makro juga belum terbukti menambah informasi. Dari 105 perbandingan multivariat lawan univariat, 40 punya rasio di bawah 1 dengan median 1,003, dan tidak ada yang lolos koreksi. Model dengan rata-rata peringkat terbaik malah AdaBoost, yang sederhana, sedangkan DPformer rata-rata peringkatnya 10."

**e. Satu pola yang bertahan** (40 detik). Layar: grafik kanan.

> "Satu hal yang konsisten: kalau penutupan pasar hari-t dipakai, MSE di h=1 turun 7,3%, dan 9 dari 10 model lolos koreksi. Waktu saya coba plasebo, yaitu penutupan hari t+1 yang sebenarnya belum ada saat ramalan dibuat, hasilnya tidak membantu. Tapi saya tandai eksploratif, karena skenarionya saya buat setelah melihat datanya, dan jam pengumuman harga Antam belum saya verifikasi dengan catatan resmi."

**f. Batasan** (30 detik). Layar: bagian "Batasan".

> "Jadi kesimpulan saya belum terbukti, dan itu belum tentu berarti tidak ada. Daya ujinya terbatas: di h=30 observasi independennya hanya sekitar 28,6. Datanya juga harga postingan Antam, di data latih 22,47% harinya harga tidak berubah. Tidak ada XAU/USD di fitur, dan periode ujinya didominasi satu reli panjang."

**g. Minta arahan** (20 detik). Layar: bagian "Yang mau saya tanyakan ke Ibu".

> "Ada tiga hal yang mau saya tanyakan. Pertama, kontribusinya saya bingkai sebagai audit pembanding dan waktu informasi, bukan model baru. Apakah Ibu setuju? Kedua, saya perlu memverifikasi jam publikasi harga Antam dengan catatan resmi, lalu mengonfirmasi skenario B di data setelah 3 Agustus 2026. Ketiga, naskah v2 saya arahkan ke Jurnal RESTI (SINTA 2). Saya mau minta pendapat Ibu, termasuk urutan penulis dan format afiliasi yang biasa Ibu pakai."

## 3. Kalau ditanya

Jawaban singkat. Rinciannya ada di `PENJELASAN.md` (nomor Q merujuk ke Lampiran C).

**Terus kontribusinya apa, kalau hampir semua tidak signifikan?** (Q1, bab 12)

> "Ada tiga, Bu. Hasil negatifnya saya laporkan apa adanya: hanya 1 dari 210 yang lolos koreksi, dan itu di keluarga uji terkecil. Daya ujinya saya ukur juga, jadi 'tidak signifikan' tidak dibaca sebagai 'tidak ada'. Lalu ada satu pola eksploratif yang diuji dengan plasebo. Selain itu pagar integritas dengan mutation testing bisa dipakai di proyek lain."

**Kenapa XAU/USD tidak dipakai?** (Q2)

> "Memang tidak saya pakai, dan itu saya catat sebagai keterbatasan. Fitur makronya DXY, S&P 500, USD/IDR, IHSG, US 10Y, FFR, dan BI-Rate, dengan DXY sebagai wakil dolar. Kalau Ibu mau, itu tambahan yang paling berharga untuk iterasi berikutnya."

**Kenapa hanya DPformer yang dituning? Tidak adil?** (Q3)

> "Sebagian sudah saya ukur. MOSSA lawan konfigurasi default di 10 sel: tidak ada yang lolos koreksi, jadi manfaat tuning tidak terbukti. Ketidakseimbangan ini hanya menguntungkan DPformer, dan DPformer tetap tidak mengungguli Drift secara konsisten, jadi kecil kemungkinan mengubah kesimpulan."

**Kenapa pembandingnya Drift?** (bab 08, Q12)

> "Drift itu prediksi rata-rata return periode latih yang tetap. Sederhana, tapi di deret yang sinyalnya tipis susah dikalahkan, jadi kalau model tidak bisa mengalahkannya, model itu belum terbukti berguna. Saya juga uji terhadap Drift ekspansif dan Drift bergulir 250 hari, dan keunggulan modelnya menyusut. Di h=10, dari 42 kombinasi yang lebih baik dari Drift tetap ada 32, dari Drift ekspansif 19, dan dari Drift bergulir tinggal 3."

**Apakah ada kebocoran data?** (bab 05 dan 07, Q7)

> "Saya jaga dengan tiga cara. Pembagian data urut waktu dengan embargo sebesar horizon, supaya baris validasi tidak menyentuh periode uji. Semua fitur makro digeser satu hari kerja. Dan ada tujuh pagar integritas, termasuk mutation testing: data uji sengaja dirusak, hasil validasi tetap sama sampai digit terakhir, sedangkan metrik uji berubah sekitar tiga kali. Tapi pagar ini hanya menutup jalur yang diuji. Keputusan desain yang diambil setelah melihat data, seperti skenario B, tidak tertangkap, makanya saya tandai eksploratif."

**Kenapa multivariat tidak lebih baik dari univariat?** (Q8)

> "Dari 105 perbandingan, 40 punya rasio di bawah 1 dan 5 mencapai p di bawah 0,05, tapi tidak ada yang lolos koreksi. Yang paling mendekati GBM dan HistGBM di h=1, rasionya sekitar 0,91, tapi itu pun hanya lolos kalau keluarga ujinya dipersempit per horizon. Jadi yang bisa saya katakan 'tidak terbukti', bukan 'tidak membantu'. Ada dugaan informasi makro sudah terwakili return emas sendiri, tapi itu belum saya uji."

**Kenapa horizon panjang tidak signifikan?** (bab 09)

> "Karena daya ujinya rendah. Return h hari saling tumpang tindih, jadi di h=30 observasi independen efektifnya hanya sekitar 28,6. Terhadap Drift tetap, selisih terkecil yang bisa saya deteksi 3,4 sampai 6,6%. Terhadap Drift bergulir 48% di h=20 dan 70% di h=30. Jadi di sana yang bisa saya katakan 'belum terbukti', bukan 'tidak ada efek'."

**Datanya dari mana, dan itu harga apa?** (bab 01 dan 02, Q6)

> "Harga jual emas Antam per gram, 4.325 hari kerja dari 5 Januari 2010 sampai 3 Agustus 2026. Ini harga postingan administratif, jadi tetap sampai ada pengumuman baru. Di data latih 22,47% harinya return-nya nol. Sumber datanya dataset komunitas yang belum saya validasi silang ke situs resmi, dan itu saya catat sebagai keterbatasan."

**Kenapa metrik di ruang harga, tapi uji DM di return?** (Q5)

> "MAE dan MAPE saya hitung di ruang harga karena yang ingin diketahui pengguna ramalan ini adalah 'berapa rupiah harganya'. Uji Diebold-Mariano memakai galat kuadrat return, supaya periode saat harga tinggi tidak otomatis berbobot lebih besar. Versi ruang harganya ada di berkas yang sama sebagai pembanding."

**Seberapa stabil terhadap seed?** (bab 11)

> "Variasi seed kecil: CV MAE 0,034 sampai 1,30%, mediannya 0,29%, dan selisih antar arsitektur 5 sampai 28 kali simpangan baku seed. Tapi model terbaiknya sama di kelima seed hanya di 1 dari 10 sel, jadi peringkat yang tipis jangan dibaca terlalu kuat."

**Bisa dipakai untuk trading?** (Q10)

> "Belum, dan itu kesimpulan penelitian ini sendiri. Di h=5 sampai 30 tidak ada model yang signifikan mengalahkan Drift, dan di h=1 hanya satu uji yang lolos dengan efek sekitar 0,5%. Pola hari-t menarik, tapi masih eksploratif, jam pengumuman Antam belum diverifikasi, dan biaya transaksi serta selisih harga jual dan beli kembali belum dihitung. Jadi ini bukan rekomendasi trading."

**Rencana publikasinya?**

> "Naskah v2 saya arahkan ke Jurnal RESTI (SINTA 2). Hasil review tidak bisa saya pastikan. Pola hari-t perlu dikonfirmasi di data baru dulu sebelum diklaim sebagai hasil. Kalau Ibu menyarankan jurnal lain, saya terbuka."

**Ini pakai AI, ya?** (Q9)

Jawab sesuai kenyataan, dengan susunan jawaban di `PENJELASAN.md` Lampiran C, Q9 (isi bagian `[ISI: ...]` dulu). Jangan menyatakan hal yang tidak benar tentang siapa mengerjakan apa. Lembar `Jawaban` dan naskah ini juga disusun dengan bantuan AI dari hasil CSV, jadi kalau ditanya, sebut apa adanya. Yang bisa ditunjukkan: seluruh pelatihan dan analisis lewat satu jalur kode (`src/runners.py`), ada tujuh pagar integritas, dan semua angka bisa dicocokkan ke CSV di `outputs/results/`.

## 4. Kalimat yang sebaiknya dihindari

| Hindari | Ganti dengan |
|---|---|
| "Model saya mengalahkan Drift" | "Secara angka sedikit lebih baik, tapi belum terbukti setelah koreksi uji berganda" |
| "Variabel makro membantu" atau "tidak membantu" | "Belum terbukti membantu" |
| "Skenario hari-t terbukti" atau "bisa dipakai trading" | "Pola eksploratif, perlu data baru dan verifikasi jam pengumuman Antam" |
| "Di horizon panjang tidak ada efek" | "Tidak terdeteksi, daya ujinya rendah" |
| "MOSSA tidak berguna" | "Manfaatnya tidak terbukti (0 dari 10 sel lolos)" |
| "Pasti masuk jurnal ..." atau menyebut Q1 | "Target pertama RESTI (SINTA 2), hasil review tidak bisa dipastikan" |

## 5. Kalau waktunya cuma 1 menit

> "Hasilnya, belum ada model yang terbukti mengalahkan Drift setelah koreksi uji berganda. Dari 210 eksperimen, 115 lebih baik secara angka, tapi tidak ada yang lolos di keluarga 105 dan 210 uji. Model terbaik di tiap horizon hanya sedikit lebih baik, dan itu dipilih setelah melihat hasil uji. Satu pola yang konsisten: penutupan pasar hari-t menurunkan MSE h=1 sebesar 7,3%, plasebo tidak membantu, tapi masih eksploratif. Kesimpulannya belum terbukti, dan belum tentu berarti tidak ada, karena daya ujinya terbatas. Saya mau minta arahan Ibu soal bingkai kontribusi dan verifikasi jam pengumuman Antam."

## Yang masih perlu diisi atau dicek sendiri

- Bagian `[ISI: ...]` di `PENJELASAN.md`, Q9 dan Q11.
- Label A/B/C pada diagram alur, dicocokkan dengan teks lengkap Bao et al. (2026).
- Jam pengumuman harga Antam. Naskah ini menulisnya "belum diverifikasi"; ubah kalau sudah dicek ke catatan resmi.
- Urutan penulis dan format afiliasi untuk naskah RESTI, ditanyakan ke Bu Yuni.
