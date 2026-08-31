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
- Feature engineering, model, simulasi, dan prediksi Season 18 merupakan tahap
  pengembangan berikutnya.

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
make test
make lint
make dashboard
```

## Struktur direktori

```text
.
├── data/
│   ├── mpl-season1/ ... mpl-season17/  # CSV historis asli
│   ├── interim/normalized/              # tujuh tabel Parquet hasil normalisasi
│   └── processed/canonical/             # tabel canonical tim, pemain, dan pertandingan
├── artifacts/                           # model, encoder, dan metadata training
├── reports/
│   ├── semantic_audit.json              # temuan audit dan coverage per musim
│   ├── identity_mapping_summary.json     # ringkasan mapping tim dan pemain
│   ├── dataset_quality_report.json       # kualitas tabel dan kesiapan fitur
│   ├── eda_summary.json                  # statistik deskriptif berorientasi model
│   ├── prediction_policy_summary.json    # aturan dan ringkasan cutoff prediksi
│   ├── prediction_windows.csv            # snapshot historis pramusim/mingguan
│   └── figures/                         # visualisasi hasil analisis
├── src/mpl_predictor/
│   ├── analysis/                        # kualitas data, EDA, dan timing prediksi
│   ├── data/                            # discovery, contract, dan audit data
│   ├── cli.py
│   ├── config.py
│   └── dashboard.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

Isi `data/interim`, `data/processed`, `artifacts`, dan `reports/figures` dihasilkan ulang
oleh pipeline dan tidak disimpan di Git, kecuali file `.gitkeep`.

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

## Prinsip pengembangan

1. Raw CSV tidak diedit langsung.
2. Missing value tidak disamakan dengan angka nol.
3. Season 4-17 menjadi data utama era franchise.
4. Season 1-3 hanya digunakan untuk konteks historis atau eksperimen dengan bobot khusus.
5. Semua fitur mempunyai cutoff waktu untuk mencegah data leakage.
6. Validasi dilakukan berdasarkan urutan musim, bukan random split.
7. Model sederhana dan terjelaskan menjadi baseline sebelum model kompleks.

## Urutan pengembangan

1. Setup project dengan `venv`, `pip`, dan `requirements.txt`. **Selesai.**
2. Structural data audit. **Selesai.**
3. Semantic data audit. **Selesai.**
4. Normalisasi nilai dan referensi tim dalam musim. **Selesai.**
5. Pemetaan identitas organisasi/franchise. **Selesai.**
6. Pemetaan identitas pemain lintas musim. **Selesai.**
7. Pembuatan canonical dataset. **Selesai.**
8. Laporan kualitas dataset dan kesiapan fitur. **Selesai.**
9. Exploratory Data Analysis. **Selesai.**
10. Definisi prediksi pramusim/mingguan dan cutoff historis. **Selesai.**
11. Pembuatan snapshot feature table berdasarkan cutoff.
12. Feature engineering performa tim, Elo, dan strength of schedule.
13. Feature engineering roster/pemain setelah data waktu efektif tersedia.
14. Baseline probabilitas dan Elo.
15. Model probabilitas kemenangan pertandingan.
16. Walk-forward backtesting pramusim dan per minggu.
17. Evaluasi ranking, probabilitas, dan kalibrasi.
18. Pemilihan serta training model final.
19. Simulasi regular season dan playoff untuk probabilitas juara.
20. Integrasi tim, roster bertanggal, jadwal, dan hasil Season 18.
21. Prediksi pramusim Season 18.
22. Pembaruan prediksi mingguan Season 18.
23. Explainability dan dashboard hasil prediksi.
24. Otomatisasi test, training, pembaruan prediksi, dan dokumentasi.
