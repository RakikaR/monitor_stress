Refactor code menjadi package modular untuk pengumpulan
dataset landmark wajah yang akan dipakai melatih model Temporal
Convolutional Network (TCN) untuk deteksi tingkat stres.

## Struktur Project

```
stress_monitor/
├── main.py                  # entry point 
├── config/settings.py       # semua parameter & "magic numbers"
├── core/
│   ├── capture.py           # loop kamera (orkestrasi)
│   ├── session.py           # state sesi & kontrol label oleh operator
│   ├── dataset_writer.py    # penulis CSV 
│   └── postprocessing.py    # imputasi + sliding window per sesi
├── utils/
│   ├── model_downloader.py  # unduh model MediaPipe
│   ├── normalization.py     # normalisasi koordinat (pure function)
│   └── expression.py        # klasifikasi ekspresi (HANYA overlay visual)
├── data/                     # CSV & .npy hasil akan tersimpan di sini
└── models/                   # file .task MediaPipe akan tersimpan di sini
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python main.py
```

Saat program berjalan:
1. Masukkan ID partisipan dan ID sesi saat diminta.
2. Jendela kamera akan terbuka. Tekan `1`/`2`/`3` kapan saja untuk
   mengubah label stres aktif sesuai protokol penelitianmu.
3. Tekan `ESC` untuk berhenti merekam.
4. Program otomatis menjalankan imputasi dan membentuk tensor TCN
   (`data/X_tcn_data.npy`, `data/y_tcn_labels.npy`).

## Yang Masih Perlu Putuskan 

- **Protokol stres terinduksi**: tugas/stimulus konkret apa yang akan
  memicu tiap level stres saat rekaman .
- **Validasi label** : kuesioner standar
  seperti PSS setelah sesi, untuk mengonfirmasi label operator cocok
  dengan kondisi subjektif partisipan.
- **Jumlah partisipan & durasi rekaman** per kelas — agar dataset
  cukup besar dan seimbang (`y_tensor` jangan didominasi satu label).
- Arsitektur TCN itu sendiri belum dibangun di sini — modul ini
  berhenti di titik menghasilkan `X_tcn_data.npy` / `y_tcn_labels.npy`
  yang siap dipakai sebagai input training.
