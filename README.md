# MPL Indonesia Season 18 Champion Predictor

Project data science untuk memperkirakan peluang juara MPL Indonesia Season 18. Fondasi
project memisahkan data historis asli, data hasil normalisasi, feature table, model, dan
dashboard agar hasil eksperimen dapat direproduksi.

## Status saat ini

- Dataset historis Season 1-17 tersedia di `data/mpl-season*/` dan dianggap sebagai raw
  source yang tidak diubah oleh pipeline.
- Season 1-3 tidak diwajibkan mempunyai `player_season_stats` karena merupakan era
  non-franchise.
- Structural audit, semantic audit, normalisasi, pemetaan identitas, canonical dataset,
  laporan kualitas, dan EDA telah disiapkan.
- Definisi serta cutoff historis prediksi pramusim dan mingguan sudah ditetapkan.
- Snapshot feature table, fitur performa tim, Elo, strength of schedule, roster lagged,
  dan baseline probabilitas sudah tersedia.
- Model probabilitas kemenangan match, walk-forward backtest pramusim/mingguan, dan
  evaluasi kalibrasi sudah tersedia.
- Model final terkalibrasi sudah dilatih ulang pada Season 4-17.
- Tim, roster bertanggal, jadwal, dan hasil resmi Season 18 sudah diintegrasikan dengan
  snapshot 31 Agustus 2026.
- Simulasi Monte Carlo regular season dan playoff Season 18 sudah tersedia.
- Rekonstruksi pramusim serta pembaruan Week 1-3 sampai 31 Agustus 2026 sudah tersedia
  dengan cutoff leakage-safe untuk bundle deployment saat ini.
- Explainability global/lokal, dashboard interaktif, dan otomasi pipeline lokal sudah
  tersedia tanpa GitHub Actions.
- Inti simulasi sudah season-agnostic. Format playoff disimpan sebagai bracket deklaratif,
  divalidasi sebelum simulasi, dan tidak boleh diwariskan ke season baru tanpa konfirmasi.

## Model yang digunakan

Project memakai beberapa model dengan fungsi yang berbeda. Model produksi tidak diganti
otomatis hanya karena satu challenger unggul pada satu metrik.

| Model | Fungsi | Status |
| --- | --- | --- |
| Logistic Regression + Platt calibration | Probabilitas kemenangan pertandingan | **Model produksi** |
| Random Forest raw/calibrated | Pembanding hubungan nonlinear | Challenger |
| XGBoost CPU raw/calibrated | Pembanding gradient-boosted trees | Challenger |
| Snapshot Logistic Regression + temperature calibration | Prediksi juara langsung per snapshot | Benchmark |
| Elo strength | Baseline kekuatan dan ranking tim | Baseline |
| Uniform probability | Baseline peluang sama untuk semua tim | Baseline |

Sistem produksi saat ini memakai `match_logistic_calibrated`. Random Forest dan XGBoost
hanya digunakan pada walk-forward backtest dan tab `Perbandingan model`; keduanya belum
digunakan untuk menghasilkan prediksi juara Season 18. Project juga belum menggunakan
ensemble. Detail konfigurasi model berada di `config/model_config.json`.

## Metode yang digunakan

| Tahap | Metode | Kegunaan |
| --- | --- | --- |
| Persiapan data | Audit struktural/semantik, normalisasi, dan canonical identity | Menyatukan tim, organisasi, franchise slot, dan pemain lintas season |
| Feature engineering | 15 fitur selisih Team A-Team B | Mewakili Elo, win rate, game differential, form, strength of schedule, histori, dan rest days |
| Missing value | Median imputation dan missing indicator | Menangani fitur historis yang tidak tersedia tanpa menganggapnya nol |
| Scaling | StandardScaler untuk model logistic | Menyamakan skala fitur sebelum fitting |
| Symmetry | Training augmentation dan rata-rata prediksi forward/reverse | Mencegah urutan Team A-Team B memengaruhi probabilitas |
| Kalibrasi match | Platt calibration past-only | Membuat probabilitas kemenangan lebih sesuai frekuensi aktual |
| Validasi | Expanding-window walk-forward per season | Menguji model hanya menggunakan season yang sudah terjadi |
| Pembelajaran live | Regularized online logit-temperature learning | Belajar dari residual prediksi setelah hasil pertandingan tersedia |
| State tim | Elo, form, rekor match/game, dan strength of schedule | Memperbarui kekuatan tim untuk pertandingan berikutnya |
| Prediksi juara | Monte Carlo 20.000 iterasi | Menyimulasikan regular season, klasemen, playoff, final, dan juara |
| Playoff | Bracket deklaratif enam tim dan konversi BO3/BO5/BO7 | Mengikuti struktur eliminasi tanpa hard-code khusus satu season |

Metrik utama untuk memilih model adalah log loss dan Brier score karena sistem membutuhkan
probabilitas yang baik untuk simulasi. Accuracy, ROC-AUC, ECE, dan kestabilan antar-season
tetap dipakai sebagai guard tambahan. Semua update live mengikuti urutan `predict -> result
-> update`, sehingga hasil pertandingan tidak dapat memengaruhi prediksi pre-match-nya
sendiri.

## Menjalankan project

Project menggunakan Python, virtual environment bawaan Python, dan `pip`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --editable .
```

Sesudah environment terpasang dan aktif:
```bash
make audit
make semantic-audit
make normalize
make canonicalize
make analysis
make modeling
make models
make season18
make validate-season-config
make verify
make local-pipeline OBSERVED_AT=2026-08-31
make test
make lint
make dashboard
```

## Struktur direktori

```text
.
├── data/
│   ├── mpl-season1/ ... mpl-season17/  # CSV historis asli
│   ├── season18/                         # snapshot live tim, roster, jadwal, dan hasil
│   ├── interim/normalized/              # tujuh tabel Parquet hasil normalisasi
│   └── processed/
│       ├── canonical/                   # tabel canonical tim, pemain, dan pertandingan
│       ├── features/                    # snapshot dan pre-match feature table
│       └── predictions/                 # baseline, snapshot S18, dan explainability
├── artifacts/                           # model, encoder, dan metadata training
├── reports/
│   ├── semantic_audit.json              # temuan audit dan coverage per musim
│   ├── identity_mapping_summary.json     # ringkasan mapping tim dan pemain
│   ├── dataset_quality_report.json       # kualitas tabel dan kesiapan fitur
│   ├── eda_summary.json                  # statistik deskriptif berorientasi model
│   ├── prediction_policy_summary.json    # aturan dan ringkasan cutoff prediksi
│   ├── prediction_windows.csv            # snapshot historis pramusim/mingguan
│   ├── feature_engineering_report.json   # validasi cutoff dan coverage fitur
│   ├── baseline_report.json              # evaluasi baseline uniform dan Elo
│   ├── match_feature_report.json         # kualitas fitur pre-match
│   ├── model_evaluation_report.json      # ranking, probabilitas, dan kalibrasi
│   ├── final_model_selection.json        # keputusan dan cakupan training final
│   ├── season18_data_report.json         # sumber, cutoff, dan validasi data live
│   ├── season18_simulation_report.json   # asumsi dan probabilitas simulasi
│   ├── season18_prediction_updates.json  # cutoff pramusim dan mingguan S18
│   ├── explainability_report.json        # metode dan batas interpretasi model
│   └── figures/                         # visualisasi hasil analisis
├── docs/OPERATIONS.md                   # panduan training dan update lokal
├── scripts/run_local_pipeline.sh        # otomasi lokal train/update/verify/all
├── src/mpl_predictor/
│   ├── analysis/                        # kualitas data, EDA, dan timing prediksi
│   ├── data/                            # discovery, contract, dan audit data
│   ├── features/                        # Elo, performa, SoS, roster, dan snapshot
│   ├── models/                          # baseline dan model prediksi
│   ├── cli.py
│   ├── config.py
│   └── dashboard.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

Isi `data/interim`, `data/processed`, `artifacts`, dan `reports/figures` umumnya dihasilkan
ulang oleh pipeline dan tidak disimpan di Git. Lima file Parquet minimum yang dibaca
dashboard dikecualikan agar deployment Streamlit mempunyai snapshot prediksi siap baca.

## Kebijakan identitas

- `team_id` mewakili nama tim pada satu musim.
- `organization_id` menghubungkan organisasi atau brand lintas musim.
- `franchise_slot_id` hanya digunakan mulai Season 4 dan tetap sama ketika slot berganti
  brand atau pemilik.
- Aturan tim tersimpan di `config/team_identity_rules.csv` beserta dasar dan sumbernya.
- Alias pemain otomatis hanya mengabaikan kapitalisasi, spasi, dan tanda baca. Perubahan
  ejaan lain harus tercatat eksplisit di `config/player_alias_overrides.csv`.
- Nickname pendek atau ambigu ditandai `identity_review_required` dan dapat dikeluarkan
  dari fitur pemain sampai diverifikasi.

## Keputusan kualitas fitur

- `matches`, target juara, serta identity tim/franchise siap menjadi fondasi model.
- Score game pada `matches` menjadi sumber utama karena hasil per-game pada `games`
  belum lengkap di semua musim.
- `game duration` dan `drafts` hanya digunakan untuk eksperimen dengan kontrol coverage.
- `player_season_stats` tidak digunakan sebagai fitur inti karena hanya mencakup pemain
  terpilih, definisi snapshot berbeda, dan sebagian diterbitkan setelah musim selesai.
- Roster identity/continuity baru boleh menjadi fitur historis as-of setelah tersedia
  `announced_at` atau `valid_from`/`valid_to`.

## Definisi prediksi

Prediksi menghasilkan probabilitas juara untuk seluruh tim aktif dan jumlah probabilitas
per snapshot harus sama dengan 1.

- **Pramusim:** dijalankan setelah peserta dan roster resmi tersedia, tetapi sebelum
  pertandingan pertama. Backtest memakai cutoff satu hari sebelum regular season dimulai
  dan hanya menggunakan hasil musim sebelumnya.
- **Mingguan:** dijalankan setelah seluruh match regular season pada minggu tersebut
  selesai. Fitur hanya boleh memakai pertandingan dengan `week <= completed_week` dan
  completion date tidak melewati cutoff.
- Pembaruan playoff nantinya dibuat event-driven per pertandingan/bracket, bukan bagian
  dari prediksi mingguan.
- Backtest utama memakai expanding-window walk-forward, direkomendasikan mulai target
  Season 8 agar training awal memiliki empat musim franchise (Season 4-7).

Aturan lengkap tersimpan di `config/prediction_policy.json`; cutoff historis Season 4-17
tersimpan di `reports/prediction_windows.csv`.

## Snapshot feature table dan baseline

`build-features` menghasilkan satu baris untuk setiap tim pada setiap snapshot. Saat ini
terdapat 1.091 team-snapshot dari 129 snapshot historis dengan 42 kolom fitur yang
didefinisikan dan 36 kolom aktif.

Fitur yang tersedia meliputi:

- rating dan ranking Elo yang dibawa lintas musim melalui `franchise_slot_id`;
- performa regular season saat ini dan performa satu/tiga musim sebelumnya;
- form tiga dan lima pertandingan terakhir;
- strength of schedule berdasarkan Elo lawan dan win rate lawan;
- ukuran, role coverage, continuity, dan pengalaman roster musim sebelumnya;
- fitur current-roster as-of yang otomatis aktif jika `valid_from` atau `announced_at`
  tersedia.

Canonical roster historis Season 4-17 belum mempunyai tanggal efektif. Karena itu enam
fitur current-roster pada backtest tetap nonaktif dan bernilai missing, bukan dianggap nol.
Roster live Season 18 sudah mempunyai `valid_from=2026-08-31`, berdasarkan tanggal pertama
halaman resmi diverifikasi. Tanggal ini sengaja tidak dimundurkan ke awal musim.

Baseline terdiri dari probabilitas uniform dan probabilitas `elo_strength`. Evaluasi
chronological Season 8-17 mencakup 93 snapshot. Dibanding uniform, Elo memperbaiki
multiclass log loss sekitar 21,94% dan Brier score sekitar 15,15%. Nilai ini hanya baseline,
bukan prediksi final Season 18.

## Model dan walk-forward backtest

Model final tetap memakai logistic regression pada 15 fitur selisih team A–team B.
Random Forest dan XGBoost CPU ditambahkan sebagai challenger nonlinear dalam backtest,
bukan langsung menggantikan model final. Semua keluarga model dilatih dengan orientasi
pertandingan terbalik dan probabilitas forward/reverse dirata-ratakan agar prediksi simetris.
Feature table mencakup 992 match Season 4-17.

Backtest menggunakan outer fold per musim. Untuk target Season S:

- model hanya dilatih dari Season `< S`;
- kalibrasi hanya memakai prediksi out-of-fold dari musim sebelum S;
- model champion pramusim dan mingguan dipisah;
- model mingguan menerima `completed_week`, tetapi tidak menerima hasil setelah cutoff;
- probabilitas champion dinormalisasi agar berjumlah 1 pada setiap snapshot.

Pada 736 match evaluasi Season 8-17, model match terkalibrasi mencapai log loss 0,6325,
Brier score 0,2205, ROC-AUC 0,7007, accuracy 66,17%, dan ECE 0,0416. Kalibrasi Platt
memperbaiki log loss model match sekitar 0,61%.

Perbandingan challenger dengan konfigurasi konservatif menghasilkan:

| Keluarga | Varian terbaik | Log loss | Brier | Accuracy | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | raw symmetric | 0,631574 | 0,220240 | 65,08% | 0,053477 |
| Logistic Regression | Platt calibrated | 0,632518 | 0,220529 | 66,17% | 0,041553 |
| XGBoost | Platt calibrated | 0,637397 | 0,223210 | 63,99% | 0,040545 |

Random Forest unggul sangat tipis pada log loss dan Brier, tetapi accuracy serta ECE-nya
lebih buruk. Logistic Regression karena itu tetap menjadi model produksi sampai challenger
menunjukkan peningkatan yang lebih konsisten antar-season. Hasil lengkap tersedia pada
`reports/model_evaluation_report.json`; pemilihan model tidak dilakukan otomatis.

Temperature calibration memperbaiki log loss snapshot logistic dari 2,2610 menjadi
1,7323 atau sekitar 23,39%. Namun Elo tetap lebih baik dengan log loss 1,6816, mean rank
juara 2,01, top-1 49,46%, dan top-3 86,02%. Karena itu Elo dipertahankan sebagai benchmark
ranking langsung, sedangkan sistem final memakai model match terkalibrasi dan simulasi
struktur kompetisi untuk menghasilkan probabilitas juara.

## Model final dan simulasi Season 18

`train-final` melatih logistic regression pada seluruh 992 pertandingan Season 4-17.
Kalibrator Platt menggunakan 1.728 observasi prediksi out-of-fold Season 6-17, bukan
prediksi in-sample. Hasil S18 tidak mengubah koefisien logistic utama, tetapi hasil live
yang sudah tersedia melatih koreksi confidence online yang kecil dan teregularisasi.

Untuk setiap pertandingan S18, pipeline menyimpan probabilitas model dasar, menghitung
prediksi adaptif, lalu baru membaca hasil aktual. Residual `aktual - probabilitas` digunakan
untuk memperbarui confidence scale dan hasilnya juga memperbarui Elo, form, rekor, serta
strength of schedule. Karena urutannya predict-then-update, sebuah hasil tidak dapat
memengaruhi prediksi pertandingan itu sendiri. Setiap baris menyimpan jumlah hasil terdahulu,
error signal, scale sebelum/sesudah update, dan status update untuk audit.

Lapisan online juga diuji ulang pada 736 prediksi walk-forward Season 8-17 dengan state
di-reset pada awal setiap season. Accuracy tetap 66,17%, sedangkan Brier score turun dari
0,220528 menjadi 0,220427 dan log loss turun dari 0,632517 menjadi 0,632215. Perbaikannya
kecil, sehingga lapisan ini hanya mengoreksi confidence dan tidak menggantikan model utama.

Snapshot data live 31 Agustus 2026 berisi 9 tim, 59 pemain, 20 staf, dan 72 jadwal regular
season. Sebanyak 24 hasil sampai Week 3 dikunci sebagai hasil aktual; 48 pertandingan
tersisa disimulasikan. Simulasi default menjalankan 20.000 iterasi dengan random seed tetap,
top enam regular season, lalu bracket delapan seri yang mengikuti struktur Season 15-17.
Status format S18 saat ini `historical_assumption`; config wajib ditinjau lagi setelah
aturan playoff resmi S18 tersedia.

Probabilitas pertandingan tersisa dibekukan pada state data 31 Agustus 2026. Simulasi
memperbarui klasemen pada setiap iterasi, tetapi belum memperbarui ulang fitur Elo/form di
dalam iterasi. Roster S18 sudah terintegrasi secara temporal, namun belum menjadi kolom
fitur model match final versi 1.

Bracket playoff memakai konfigurasi deklaratif. Probabilitas seri referensi BO3 diubah
menjadi estimasi peluang per game, kemudian dihitung kembali sesuai BO5 atau BO7. Dengan
demikian panjang seri sekarang memengaruhi probabilitas juara, tetapi konversi tersebut
tetap merupakan asumsi model dan bukan probabilitas game yang dilatih secara terpisah.

## Prinsip pengembangan

1. Missing value tidak disamakan dengan angka nol.
2. Season 4-17 menjadi data utama era franchise.
3. Season 1-3 hanya digunakan untuk konteks historis atau eksperimen dengan bobot khusus.
4. Semua fitur mempunyai cutoff waktu untuk mencegah data leakage.
5. Validasi dilakukan berdasarkan urutan musim, bukan random split.
6. Model sederhana dan terjelaskan menjadi baseline sebelum model kompleks.
