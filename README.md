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
mpl-predictor audit
mpl-predictor semantic-audit
mpl-predictor normalize
mpl-predictor canonicalize
mpl-predictor quality-report
mpl-predictor eda
mpl-predictor prediction-policy
mpl-predictor build-features
mpl-predictor baseline
mpl-predictor build-match-features
mpl-predictor backtest
mpl-predictor sync-season18 --observed-at 2026-08-31
mpl-predictor train-final
mpl-predictor update-season18-predictions
mpl-predictor explain-season18
python -m pytest
ruff check .
streamlit run src/mpl_predictor/dashboard.py
```

Perintah yang sama tersedia melalui `Makefile`:

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

## Prediksi pramusim, mingguan, dan explainability

Rekonstruksi pramusim memakai cutoff 13 Agustus 2026, daftar peserta/jadwal S18, seluruh
hasil sampai S17, dan nol hasil S18. Karena roster pertama kali terverifikasi pada 31
Agustus, rekonstruksi tidak menggunakan roster tersebut. Ini adalah rekonstruksi as-of,
bukan bukti bahwa file prediksi benar-benar diterbitkan sebelum musim dimulai.

Snapshot yang tersedia:

- `S18_PRE`: 0 hasil tersedia, 72 regular-season match disimulasikan;
- `S18_W01`: 8 hasil tersedia dan 64 match disimulasikan;
- `S18_W02`: 16 hasil tersedia dan 56 match disimulasikan;
- `S18_W03`: 24 hasil tersedia dan 48 match disimulasikan.

Snapshot sumber resmi terbaru disimpan di `data/season18/snapshots/2026-08-31`. Snapshot
retrospektif 21 Agustus tetap disimpan untuk audit prediksi parsial Week 2. Roster baru
tersedia pada snapshot 31 Agustus karena tanggal verifikasi pertamanya adalah 31 Agustus.

Semua snapshot memakai 20.000 iterasi dan probabilitas juaranya berjumlah 1. Explainability
global menggunakan besaran koefisien logistic terstandardisasi. Explainability match
menampilkan kontribusi terhadap raw forward logit sebelum side-symmetry, kalibrasi Platt,
dan Monte Carlo; kontribusi tersebut tidak boleh dibaca sebagai hubungan sebab-akibat.

Tab Pertandingan mengevaluasi favorit pre-match terhadap pemenang aktual dengan status
`Benar`, `Salah`, atau `Belum dimainkan`. Hasil yang sudah selesai dimasukkan setelah
prediksi pre-match untuk melatih kalibrasi online serta memperbarui Elo dan form pertandingan
berikutnya. Dashboard membandingkan probabilitas dasar dan adaptif, jumlah hasil terdahulu
yang sudah dipelajari, serta confidence scale terbaru. Koefisien model final tidak dilatih
ulang setiap week; training ulang penuh tetap menjadi proses terpisah dan harus lolos
walk-forward backtest.

Dashboard dijalankan dengan `make dashboard`. Panduan update, training ulang, verifikasi,
dan contoh penjadwalan lokal tersedia di `docs/OPERATIONS.md`.
Tab `Perbandingan model` menampilkan hasil walk-forward Logistic Regression, Random Forest,
dan XGBoost langsung dari laporan evaluasi yang telah dibuat saat training lokal.

Untuk season setelah S18, gunakan template `config/season_template.json` dan ikuti
`docs/ADDING_A_SEASON.md`. Command lama yang mengandung `season18` tetap dipertahankan agar
deployment saat ini tidak rusak; command generik baru adalah `validate-season-config`,
`simulate-season`, dan `update-season-predictions`.

Untuk Streamlit Community Cloud, gunakan main file path `streamlit_app.py`. Entry point ini
menambahkan folder `src` ke import path sebelum memuat dashboard, sehingga package
`mpl_predictor` dapat ditemukan tanpa instalasi editable.

## Prinsip pengembangan

1. Raw CSV tidak diedit langsung.
2. Missing value tidak disamakan dengan angka nol.
3. Season 4-17 menjadi data utama era franchise.
4. Season 1-3 hanya digunakan untuk konteks historis atau eksperimen dengan bobot khusus.
5. Semua fitur mempunyai cutoff waktu untuk mencegah data leakage.
6. Validasi dilakukan berdasarkan urutan musim, bukan random split.
7. Model sederhana dan terjelaskan menjadi baseline sebelum model kompleks.
