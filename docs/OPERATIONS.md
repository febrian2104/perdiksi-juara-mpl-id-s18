# Operasional lokal MPL S18 Predictor

Project tidak memakai GitHub Actions. Audit, training, pembaruan data, prediksi, test, dan
validasi dokumentasi dijalankan di komputer lokal melalui `Makefile` atau satu script.

Inti simulasi mendukung season lain melalui konfigurasi, sedangkan sinkronisasi situs dan
dashboard deployment saat ini tetap khusus Season 18. Panduan menambahkan season tersedia
di `docs/ADDING_A_SEASON.md`.

## Persiapan satu kali

```bash
make setup
```

## Alur yang tersedia

Training ulang seluruh data historis dan model final:

```bash
./scripts/run_local_pipeline.sh train
```

Tahap `backtest` membandingkan Logistic Regression, Random Forest, dan XGBoost dengan fold
season serta kalibrasi past-only yang sama. XGBoost memakai paket `xgboost-cpu` agar instalasi
lokal dan Streamlit tidak menarik runtime GPU yang tidak digunakan. Laporan menyimpan semua
varian raw/calibrated, ranking challenger terkalibrasi, dan varian terbaik setiap keluarga.
Model produksi tidak berubah otomatis dari hasil perbandingan tersebut.

Mengambil data S18 terbaru, membuat ulang seluruh snapshot pramusim/mingguan, dan
memperbarui explainability:

```bash
./scripts/run_local_pipeline.sh update 2026-08-31
```

Untuk membuat snapshot retrospektif pada akhir tanggal tertentu tanpa memundurkan tanggal
roster, jalankan:

```bash
make snapshot-season18 AS_OF=2026-08-31
make update-predictions AS_OF=2026-08-31
make explain-season18
```

Snapshot 31 Agustus memakai 24 hasil sampai Week 3 dan menyisakan 48 pertandingan. Roster
yang pertama diverifikasi 31 Agustus tersedia pada snapshot ini. Snapshot retrospektif
21 Agustus tetap tersedia di `data/season18/snapshots/2026-08-21` untuk audit cutoff
parsial Week 2.

Menjalankan lint, seluruh test, dan pemeriksaan file dokumentasi:

```bash
./scripts/run_local_pipeline.sh verify
```

Menjalankan semuanya secara berurutan:

```bash
./scripts/run_local_pipeline.sh all 2026-08-31
```

Memvalidasi format regular season dan playoff aktif:

```bash
make validate-season-config
# atau
mpl-predictor validate-season-config --config config/simulation_config.json
```

Menjalankan season lain setelah config dan tabel live-nya siap:

```bash
mpl-predictor simulate-season \
  --config config/simulation_season19.json \
  --season-dir data/season19

mpl-predictor update-season-predictions \
  --config config/simulation_season19.json \
  --season-dir data/season19
```

Jika tanggal tidak diberikan, mode `update` menggunakan tanggal lokal saat command
dijalankan. Roster mempertahankan tanggal `valid_from` paling awal yang pernah diverifikasi;
anggota yang hilang dari halaman resmi ditutup dengan `valid_to` pada tanggal observasi.

## Jadwal pembaruan yang disarankan

Jalankan mode `update` setelah seluruh pertandingan pada satu week selesai. Script hanya
membuat snapshot mingguan baru jika semua pertandingan week tersebut sudah mempunyai hasil.
Contoh cron lokal berikut sengaja tidak dipasang otomatis:

```cron
30 23 * * 0 cd "/path/to/perdiksi juara mpl" && ./scripts/run_local_pipeline.sh update >> /tmp/mpl-s18-update.log 2>&1
```

Sesuaikan path, hari, dan jam dengan jadwal MPL. Periksa exit code serta log sebelum
menganggap pembaruan berhasil.

## Dashboard

```bash
make dashboard
```

Dashboard membaca output Parquet terbaru dan tidak melakukan training atau download data
ketika halaman dibuka. Jika output belum lengkap, dashboard menampilkan command pembaruan
yang perlu dijalankan. Tab `Perbandingan model` membaca
`reports/model_evaluation_report.json`; jalankan `make backtest` setelah mengubah kandidat
atau hyperparameter model.

Untuk deployment Streamlit Community Cloud, isi **Main file path** dengan:

```text
streamlit_app.py
```

`streamlit_app.py` tetap menjadi entry point yang direkomendasikan. Sebagai fallback,
`src/mpl_predictor/dashboard.py` juga menambahkan folder `src` ke import path sehingga
deployment lama tidak gagal ketika belum mengganti konfigurasi main file.

## Pembelajaran dari kesalahan prediksi

Setiap pembaruan mingguan menjalankan evaluasi secara kronologis. Pipeline menghitung
probabilitas pre-match terlebih dahulu, kemudian hasil aktual memperbarui dua bagian:

- state tim: Elo, rekor match/game, form, dan strength of schedule;
- kalibrasi online: residual `aktual - probabilitas` memperbarui confidence scale
  teregularisasi yang digunakan pertandingan berikutnya.

Probabilitas model dasar tetap disimpan agar efek adaptasi dapat dibandingkan secara
prequential. Parameter utama logistic regression tidak di-fit ulang dari sedikit hasil live;
training ulang penuh tetap memakai `make train-final`, data historis yang sudah dicanonical,
dan validasi walk-forward. Konfigurasi adaptasi berada di bagian `online_learning` pada
`config/feature_config.json`. Command `make backtest` turut memvalidasi lapisan online pada
fold historis dan menyimpan hasilnya di bagian `online_learning_backtest` dalam
`reports/model_evaluation_report.json`.

## Keluaran utama

- `data/season18/`: tim, riwayat roster bertanggal, jadwal, dan hasil resmi.
- `artifacts/final_match_model.joblib`: model final lokal.
- `data/processed/predictions/season18_snapshot_predictions.parquet`: riwayat pramusim dan
  mingguan.
- `data/processed/predictions/season18_snapshot_match_probabilities.parquet`: probabilitas
  dasar/adaptif per match, hasil aktual, residual prediksi, confidence scale, status akurasi
  pre-match, dan status pembaruan state.
- `data/processed/predictions/season18_*explanations.parquet`: explainability dashboard.
- `reports/season18_prediction_updates.json`: cutoff dan leakage guard tiap snapshot.
- `reports/explainability_report.json`: metode serta batas interpretasi explainability.

Folder `artifacts/` dan mayoritas `data/processed/` adalah output yang dapat dibuat ulang.
Lima file `season18_*` yang digunakan langsung oleh dashboard dikecualikan dari
`.gitignore`; file tersebut perlu di-commit setiap kali snapshot deployment diperbarui.

## Checklist deployment Streamlit

Sebelum push, pastikan lima file dashboard terlihat sebagai tracked/modified atau untracked:

```bash
git status --short data/processed/predictions
```

File yang wajib ikut ke repository:

- `season18_snapshot_predictions.parquet`
- `season18_snapshot_match_probabilities.parquet`
- `season18_global_feature_importance.parquet`
- `season18_match_explanations.parquet`
- `season18_team_explanations.parquet`

Setelah file, `.gitignore`, dan source code di-commit serta di-push, reboot aplikasi
Streamlit Cloud. Dashboard tidak membutuhkan model artifact untuk menampilkan snapshot
yang sudah dihitung.
