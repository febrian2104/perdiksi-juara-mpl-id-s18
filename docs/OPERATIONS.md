# Operasional lokal MPL S18 Predictor

Project tidak memakai GitHub Actions. Audit, training, pembaruan data, prediksi, test, dan
validasi dokumentasi dijalankan di komputer lokal melalui `Makefile` atau satu script.

## Persiapan satu kali

```bash
make setup
```

## Alur yang tersedia

Training ulang seluruh data historis dan model final:

```bash
./scripts/run_local_pipeline.sh train
```

Mengambil data S18 terbaru, membuat ulang seluruh snapshot pramusim/mingguan, dan
memperbarui explainability:

```bash
./scripts/run_local_pipeline.sh update 2026-08-31
```

Untuk membuat snapshot retrospektif pada akhir tanggal tertentu tanpa memundurkan tanggal
roster, jalankan:

```bash
make snapshot-season18 AS_OF=2026-08-21
make update-predictions AS_OF=2026-08-21
make explain-season18
```

Snapshot 21 Agustus memakai cutoff akhir hari WIB: 8 hasil Week 1 dan 2 hasil pada hari
pertama Week 2. Roster yang baru pertama diverifikasi 31 Agustus tidak dimasukkan.

Menjalankan lint, seluruh test, dan pemeriksaan file dokumentasi:

```bash
./scripts/run_local_pipeline.sh verify
```

Menjalankan semuanya secara berurutan:

```bash
./scripts/run_local_pipeline.sh all 2026-08-31
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
yang perlu dijalankan.

Untuk deployment Streamlit Community Cloud, isi **Main file path** dengan:

```text
streamlit_app.py
```

`streamlit_app.py` tetap menjadi entry point yang direkomendasikan. Sebagai fallback,
`src/mpl_predictor/dashboard.py` juga menambahkan folder `src` ke import path sehingga
deployment lama tidak gagal ketika belum mengganti konfigurasi main file.

## Keluaran utama

- `data/season18/`: tim, riwayat roster bertanggal, jadwal, dan hasil resmi.
- `artifacts/final_match_model.joblib`: model final lokal.
- `data/processed/predictions/season18_snapshot_predictions.parquet`: riwayat pramusim dan
  mingguan.
- `data/processed/predictions/season18_snapshot_match_probabilities.parquet`: probabilitas
  match per snapshot.
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
