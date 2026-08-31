# MPL Indonesia Season 18 Champion Predictor

Project data science untuk memperkirakan peluang juara MPL Indonesia Season 18. Fondasi
project memisahkan data historis asli, data hasil normalisasi, feature table, model, dan
dashboard agar hasil eksperimen dapat direproduksi.

## Status saat ini

- Dataset historis Season 1-17 tersedia di `data/mpl-season*/` dan dianggap sebagai raw
  source yang tidak diubah oleh pipeline.
- Season 1-3 tidak diwajibkan mempunyai `player_season_stats` karena merupakan era
  non-franchise.
- Structural audit, semantic audit, dan pipeline normalisasi telah disiapkan.
- Feature engineering, model, dan prediksi Season 18 merupakan tahap pengembangan
  berikutnya.

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
python -m pytest
ruff check .
streamlit run src/mpl_predictor/dashboard.py
```

Perintah yang sama tersedia melalui `Makefile`:

```bash
make audit
make semantic-audit
make normalize
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
│   └── processed/                       # feature table siap model
├── artifacts/                           # model, encoder, dan metadata training
├── reports/
│   ├── semantic_audit.json              # temuan audit dan coverage per musim
│   └── figures/                         # visualisasi hasil analisis
├── src/mpl_predictor/
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

## Prinsip pengembangan

1. Raw CSV tidak diedit langsung.
2. Missing value tidak disamakan dengan angka nol.
3. Season 4-17 menjadi data utama era franchise.
4. Season 1-3 hanya digunakan untuk konteks historis atau eksperimen dengan bobot khusus.
5. Semua fitur mempunyai cutoff waktu untuk mencegah data leakage.
6. Validasi dilakukan berdasarkan urutan musim, bukan random split.
7. Model sederhana dan terjelaskan menjadi baseline sebelum model kompleks.

## Urutan pengembangan

1. Structural dan semantic data audit. **Selesai.**
2. Normalisasi nilai dan referensi tim dalam musim. **Selesai.**
3. Pemetaan identitas organisasi/franchise dan pemain lintas musim.
4. EDA dan laporan kualitas data.
5. Pembuatan snapshot berdasarkan prediction cutoff.
6. Feature engineering performa tim dan roster.
7. Baseline Elo dan model pertandingan.
8. Walk-forward backtesting.
9. Kalibrasi probabilitas dan pemilihan model.
10. Simulasi regular season dan playoff.
11. Integrasi tim serta roster Season 18.
12. Dashboard prediksi pramusim dan mingguan.
13. Otomatisasi test, training, dan pembaruan prediksi.
