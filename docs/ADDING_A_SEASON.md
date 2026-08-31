# Menambahkan MPL Indonesia Season Baru

Fondasi simulasi tidak lagi mengunci nomor Season 18. Nomor season, jumlah tim playoff,
urutan tie-break, sumber peserta setiap pertandingan bracket, serta BO3/BO5/BO7 dibaca
dari konfigurasi. Sinkronisasi halaman resmi dan dashboard tetap membutuhkan penyesuaian
per season karena struktur situs, peserta, dan aturan dapat berubah.

## Prinsip pengaman

Format season sebelumnya tidak boleh otomatis dianggap berlaku. Config baru ditolak jika
`format_confirmation.status` masih `unconfirmed`, nomor season pada config tidak sama dengan
tabel live, ada seed yang tidak masuk tepat satu kali, ada referensi match ke depan, atau
championship match bukan pertandingan terakhir.

Status yang diterima:

- `official`: format sudah didukung sumber resmi season target;
- `historical_assumption`: format belum diumumkan lengkap dan asumsi berbasis season lama
  dipakai secara eksplisit. Status ini harus dijelaskan di `basis` dan ditampilkan sebagai
  keterbatasan hasil.

## Urutan onboarding

1. Salin `config/season_template.json` menjadi, misalnya,
   `config/simulation_season19.json`.
2. Isi `season`, `random_seed`, regular-season format, jumlah tim playoff, dan
   `ranking_order` berdasarkan aturan target.
3. Isi `format_confirmation` beserta tanggal observasi dan URL/catatan sumber. Jangan ubah
   status dari `unconfirmed` sebelum sumber atau asumsi benar-benar ditinjau.
4. Definisikan `playoffs.bracket` dalam urutan eksekusi. Peserta hanya boleh berasal dari
   `seed`, `winner`, atau `loser` pertandingan yang sudah didefinisikan sebelumnya.
5. Siapkan `data/seasonN/teams.csv`, `rosters.csv`, dan `schedule_results.csv` dengan schema
   yang sama seperti folder Season 18. Semua tabel harus memiliki satu nilai `season` yang
   sama dengan config.
6. Pastikan canonical historical dataset dan model final sudah dilatih sampai season
   terakhir sebelum target. Contoh: prediksi S19 idealnya memakai histori final sampai S18.
7. Validasi config, jalankan simulasi kecil, kemudian jalankan pembaruan penuh.

```bash
mpl-predictor validate-season-config --config config/simulation_season19.json

mpl-predictor simulate-season \
  --config config/simulation_season19.json \
  --season-dir data/season19 \
  --iterations 500

mpl-predictor update-season-predictions \
  --config config/simulation_season19.json \
  --season-dir data/season19
```

Output generik memakai prefix nomor season, misalnya `season19_simulation.parquet` dan
`season19_prediction_updates.json`.

## Format bracket deklaratif

Contoh satu pertandingan:

```json
{
  "match_id": "play_in_1",
  "round": "play_ins",
  "best_of": 5,
  "team_a": {"source": "seed", "seed": 3},
  "team_b": {"source": "seed", "seed": 6}
}
```

Pertandingan berikutnya dapat memakai hasil sebelumnya:

```json
{
  "match_id": "upper_semifinal_1",
  "round": "upper_semifinal",
  "best_of": 5,
  "team_a": {"source": "seed", "seed": 1},
  "team_b": {"source": "winner", "match_id": "play_in_1"}
}
```

Setiap seed harus muncul tepat sekali sebagai pintu masuk. Setelah itu, perpindahan tim
wajib memakai referensi `winner` atau `loser`. Panjang seri digunakan dalam probabilitas:
peluang seri referensi BO3 dikonversi menjadi peluang per game, lalu dihitung ulang untuk
BO5 atau BO7.

## Bagian yang tetap perlu dibuat per season

- adapter pengambilan halaman jadwal dan roster resmi;
- pemetaan organisasi, franchise slot, serta alias pemain baru;
- konfirmasi aturan tie-break dan bracket;
- pemilihan season aktif pada dashboard dan berkas deployment;
- audit perubahan schema atau jumlah pertandingan.

Pemisahan ini disengaja: mesin simulasi dapat digunakan ulang, tetapi perubahan aturan dan
sumber data tidak boleh lolos tanpa pemeriksaan manusia.
